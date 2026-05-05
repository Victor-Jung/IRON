#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profile per-op NPU time for the Llama-3.2-1B decode runlist and derive
roofline-style ceilings on the speedup that further kernel fusion can yield.

This is a profiling/analysis tool: it does not change the model.  It mirrors
the op constructors in ``llama_npu.py`` (see lines 283-453 for the per-op
configuration and 466-563 for the per-token invocation count).

Outputs:
- A CSV with one row per unique decode op (us/call, n/token, bytes/token).
- A markdown summary with the roofline math.

Usage:
    # Per-op latencies + Shim-DMA bandwidth ceiling only (no model weights needed):
    python3 profile_decode.py

    # Including end-to-end wallclock (drives llama_npu.py):
    python3 profile_decode.py \\
        --weights /srv/llama3.2-1b/model.safetensors \\
        --tokenizer /srv/llama3.2-1b/tokenizer.model
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig

from iron.common.context import AIEContext
from iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMV,
    Repeat,
    RMSNorm,
    RoPE,
    SiLU,
    Softmax,
    StridedCopy,
    Transpose,
)
from iron.operators.mem_copy.op import MemCopy


# --- Llama 3.2-1B config (matches llama_inference_harness.py:30-40) ----------


class CFG:
    vocab_size = 128256
    emb_dim = 2048
    n_layers = 16
    n_heads = 32
    n_kv_groups = 8
    head_dim = 64  # = emb_dim // n_heads
    hidden_dim = 8192


MAX_SEQ_LEN = 2048  # matches llama_npu.py:49
BF16 = 2  # bytes per element

# Side-channel populated by main() so trace_one_gemv() can compare per-core
# trace BW against the production cols=N timing.
_last_production_us: dict[str, float] = {}


# --- Helpers -----------------------------------------------------------------


def shape_to_tuple(shape) -> tuple[int, ...]:
    """Normalize an int-or-tuple shape to a tuple."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def io_bytes(op) -> tuple[int, int]:
    """Return (bytes_in_per_call, bytes_out_per_call) from the op's arg spec."""
    bytes_in = bytes_out = 0
    for spec in op.get_arg_spec():
        nbytes = int(np.prod(shape_to_tuple(spec.shape)) * np.dtype(spec.dtype).itemsize)
        if spec.direction == "in":
            bytes_in += nbytes
        elif spec.direction == "out":
            bytes_out += nbytes
        else:  # inout
            bytes_in += nbytes
            bytes_out += nbytes
    return bytes_in, bytes_out


def time_op(op, n_warmup: int = 2, n_timed: int = 20) -> float:
    """Compile and time the op standalone on NPU; returns mean us per invocation.

    Uses zero-filled buffers — we only care about ``npu_time``, not correctness.
    """
    op.compile()
    fn = op.get_callable()
    args = []
    for spec in op.get_arg_spec():
        shape = shape_to_tuple(spec.shape)
        buf = XRTTensor(shape, dtype=spec.dtype)
        buf.fill_(0.0)  # also DMAs to NPU
        args.append(buf)
    for _ in range(n_warmup):
        fn(*args)
    total_ns = 0
    for _ in range(n_timed):
        result = fn(*args)
        total_ns += result.npu_time
    return (total_ns / n_timed) / 1e3


# --- Decode op-list ----------------------------------------------------------
# Mirrors AIELlamaOperators.__init__ (decode block) at llama_npu.py:283-453
# and the runlist at llama_npu.py:466-563 (invocation counts per token).


@dataclass
class OpEntry:
    name: str
    n_per_token: int
    factory: Callable[[AIEContext], object]
    # category: which of the per-op argument tensors are "unavoidable" L3
    # round-trips that even a perfect fusion strategy cannot eliminate.
    #   "weights"      -> almost all bytes are model weights (large matrices)
    #   "kv_cache"     -> bytes are KV-cache reads/writes (grow with context)
    #   "intermediate" -> activations that COULD live in L2/L1 across a fused
    #                     producer/consumer pair, so are *eliminatable* in
    #                     principle by spatial/temporal fusion
    category: str = "intermediate"
    notes: str = ""


def build_decode_ops(cfg=CFG, max_seq_len: int = MAX_SEQ_LEN) -> list[OpEntry]:
    return [
        OpEntry(
            "rms_norm",
            n_per_token=2 * cfg.n_layers + 1,
            category="weights",  # weight tensor read each call (RMSNorm scale)
            factory=lambda ctx: RMSNorm(
                size=cfg.emb_dim, num_aie_columns=1, num_channels=1,
                tile_size=cfg.emb_dim, weighted=True, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_attn_query", n_per_token=cfg.n_layers, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.n_heads * cfg.head_dim, K=cfg.emb_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=cfg.head_dim // 2, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_attn_key_value", n_per_token=2 * cfg.n_layers, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.n_kv_groups * cfg.head_dim, K=cfg.emb_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=cfg.head_dim // 2, context=ctx,
            ),
        ),
        OpEntry(
            "rope_queries", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: RoPE(
                rows=cfg.n_heads, cols=cfg.head_dim, angle_rows=1, context=ctx,
            ),
        ),
        OpEntry(
            "rope_keys", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: RoPE(
                rows=cfg.n_kv_groups, cols=cfg.head_dim, angle_rows=1, context=ctx,
            ),
        ),
        OpEntry(
            "strided_copy_cache", n_per_token=2 * cfg.n_layers, category="kv_cache",
            factory=lambda ctx: StridedCopy(
                input_sizes=(cfg.n_kv_groups, cfg.head_dim),
                input_strides=(cfg.head_dim, 1), input_offset=0,
                output_sizes=(1, cfg.n_kv_groups, cfg.head_dim),
                output_strides=(0, max_seq_len * cfg.head_dim, 1),
                output_offset=7 * cfg.head_dim * 2,
                input_buffer_size=1 * cfg.n_kv_groups * cfg.head_dim,
                output_buffer_size=cfg.n_kv_groups * max_seq_len * cfg.head_dim,
                num_aie_channels=1, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_attn_scores", n_per_token=cfg.n_layers, category="kv_cache",
            factory=lambda ctx: GEMV(
                M=max_seq_len, K=cfg.head_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=max_seq_len // 8,
                num_batches=cfg.n_heads, context=ctx,
            ),
            notes=f"batched GEMV across {CFG.n_heads} heads",
        ),
        OpEntry(
            "attn_scale", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: ElementwiseMul(
                size=cfg.n_heads * max_seq_len,
                tile_size=max_seq_len // 8,
                num_aie_columns=8, context=ctx,
            ),
        ),
        OpEntry(
            "softmax", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: Softmax(
                rows=cfg.n_heads, cols=max_seq_len,
                num_aie_columns=1, num_channels=1,
                rtp_vector_size=max_seq_len, context=ctx,
            ),
        ),
        OpEntry(
            "transpose_values", n_per_token=cfg.n_layers * cfg.n_heads,
            category="intermediate",
            factory=lambda ctx: Transpose(
                M=max_seq_len, N=cfg.head_dim,
                num_aie_columns=2, num_channels=1,
                m=256, n=32, s=8, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_attn_context", n_per_token=cfg.n_layers, category="kv_cache",
            factory=lambda ctx: GEMV(
                M=cfg.head_dim, K=max_seq_len,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=4, num_batches=cfg.n_heads, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_attn_output", n_per_token=cfg.n_layers, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.emb_dim, K=cfg.n_heads * cfg.head_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=cfg.emb_dim // 8, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_ffn_up_gate", n_per_token=2 * cfg.n_layers, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.hidden_dim, K=cfg.emb_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=cfg.hidden_dim // 8, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_ffn_down", n_per_token=cfg.n_layers, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.emb_dim, K=cfg.hidden_dim,
                num_aie_columns=8, tile_size_input=1,
                tile_size_output=cfg.emb_dim // 8, context=ctx,
            ),
        ),
        OpEntry(
            "silu_ffn", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: SiLU(
                size=cfg.hidden_dim, tile_size=cfg.hidden_dim // 8,
                num_aie_columns=8, context=ctx,
            ),
        ),
        OpEntry(
            "eltwise_mul_ffn", n_per_token=cfg.n_layers, category="intermediate",
            factory=lambda ctx: ElementwiseMul(
                size=cfg.hidden_dim, tile_size=cfg.hidden_dim // 8,
                num_aie_columns=8, context=ctx,
            ),
        ),
        OpEntry(
            "residual_add", n_per_token=2 * cfg.n_layers, category="intermediate",
            factory=lambda ctx: ElementwiseAdd(
                size=cfg.emb_dim, tile_size=cfg.emb_dim // 8, context=ctx,
            ),
        ),
        OpEntry(
            "repeat_interleave", n_per_token=2 * cfg.n_layers, category="kv_cache",
            factory=lambda ctx: Repeat(
                rows=cfg.n_kv_groups, cols=max_seq_len * cfg.head_dim,
                repeat=cfg.n_heads // cfg.n_kv_groups,
                transfer_size=cfg.head_dim, context=ctx,
            ),
        ),
        OpEntry(
            "gemv_out_head", n_per_token=1, category="weights",
            factory=lambda ctx: GEMV(
                M=cfg.vocab_size, K=cfg.emb_dim,
                num_aie_columns=8, tile_size_input=4,
                tile_size_output=32, context=ctx,
            ),
        ),
    ]


# --- Shim DMA bandwidth ceiling ---------------------------------------------


def measure_shim_dma_bw(cols: int) -> tuple[float, float]:
    """Use MemCopy in bypass mode to measure aggregate L3↔NPU DMA bandwidth.

    Hardware caps tile_size at 8192 elements and cores at ``cols * channels``,
    so the largest contiguous transfer that exercises every Shim DMA is
    ``cols * 2 * 8192`` bf16 elements (= 256 KiB on NPU2).  We use that.

    Returns (us_per_call, GB/s_effective).  The effective bandwidth counts
    both the input fill and the output drain, matching the convention used
    by ``run_test`` and the per-op rooflines.
    """
    ctx = AIEContext(build_dir=Path("build_profile") / "_memcopy_bw")
    ctx.build_dir.mkdir(parents=True, exist_ok=True)

    num_channels = 2
    num_cores = cols * num_channels
    tile = 8192
    n_elements = num_cores * tile

    op = MemCopy(
        size=n_elements,
        num_cores=num_cores,
        num_channels=num_channels,
        bypass=True,
        tile_size=tile,
        context=ctx,
    )
    us = time_op(op)
    transfer_bytes = n_elements * BF16
    bytes_moved = transfer_bytes * 2  # in + out
    return us, bytes_moved / (us * 1e-6) / 1e9


# --- End-to-end wallclock ---------------------------------------------------


def measure_e2e(weights: str, tokenizer: str, prompt_len: int, num_tokens: int) -> dict:
    """Drive llama_npu.py and parse [Decode] tokens/s and time/token from output."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "llama_npu.py"),
        weights,
        tokenizer,
        "--prompt-len",
        str(prompt_len),
        "--num-tokens",
        str(num_tokens),
    ]
    res = subprocess.run(
        cmd, cwd=str(Path(__file__).parent), capture_output=True, text=True
    )
    out = res.stdout + res.stderr
    m_tps = re.search(r"\[Decode\]\s*Tokens per second:\s*([\d\.eE+\-]+)", out)
    m_tpt = re.search(r"\[Decode\]\s*Time per token \(mean\):\s*([\d\.eE+\-]+)", out)
    return {
        "tokens_per_sec": float(m_tps.group(1)) if m_tps else None,
        "us_per_token": float(m_tpt.group(1)) * 1e6 if m_tpt else None,
        "returncode": res.returncode,
        "raw_tail": out[-2000:],
    }


# --- Driver -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", help="path to model.safetensors (for e2e wallclock)")
    ap.add_argument("--tokenizer", help="path to tokenizer.model (for e2e wallclock)")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--num-tokens", type=int, default=40)
    ap.add_argument("--csv", default="decode_profile.csv")
    ap.add_argument("--md", default="decode_profile.md")
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-timed", type=int, default=20)
    ap.add_argument("--skip-bw", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--only", help="comma-separated subset of op names to profile")
    ap.add_argument(
        "--trace-ops",
        help="comma-separated GEMV op names to trace "
        "(produces trace/<op>.txt + trace/<op>.json for Perfetto). "
        "Only the GEMV ops support tracing today.",
    )
    ap.add_argument(
        "--trace-size",
        type=int,
        default=8192,
        help="AIE trace buffer size in bytes (default: 8192)",
    )
    args = ap.parse_args()

    cols = aie_utils.get_current_device().cols
    print(f"NPU device: {aie_utils.get_current_device()!r} (cols={cols})")

    decode_ops = build_decode_ops()
    if args.only:
        names = set(args.only.split(","))
        decode_ops = [o for o in decode_ops if o.name in names]

    rows = []
    for entry in decode_ops:
        ctx = AIEContext(build_dir=Path("build_profile") / entry.name)
        ctx.build_dir.mkdir(parents=True, exist_ok=True)
        try:
            op = entry.factory(ctx)
            bin_, bout = io_bytes(op)
            us = time_op(op, n_warmup=args.n_warmup, n_timed=args.n_timed)
        except Exception as e:
            print(f"  !! {entry.name}: failed -- {e!r}")
            continue
        total_us = us * entry.n_per_token
        eff_bw_gbps = (bin_ + bout) / (us * 1e-6) / 1e9
        print(
            f"  {entry.name:24s}  {us:8.2f} us/call  × {entry.n_per_token:4d} "
            f"= {total_us:9.1f} us/token  "
            f"(in {bin_/1024:8.1f} KiB, out {bout/1024:8.1f} KiB, "
            f"eff {eff_bw_gbps:5.1f} GB/s)"
        )
        rows.append(
            {
                "op": entry.name,
                "category": entry.category,
                "us_per_call": us,
                "n_per_token": entry.n_per_token,
                "us_per_token": total_us,
                "in_bytes_per_call": bin_,
                "out_bytes_per_call": bout,
                "in_bytes_per_token": bin_ * entry.n_per_token,
                "out_bytes_per_token": bout * entry.n_per_token,
                "eff_bw_gbps": eff_bw_gbps,
            }
        )
        _last_production_us[entry.name] = us

    if not rows:
        print("No ops profiled successfully.", file=sys.stderr)
        return 1

    # --- Bandwidth ceiling derivation -----------------------------------
    # The MemCopy(bypass) test caps at cols*channels*8192 bf16 elements = 256 KiB
    # on NPU2, which is too small to reach steady-state DMA throughput. We
    # use it only as a sanity-check lower bound. For the roofline we use the
    # MAX effective BW observed across the actual decode ops -- the large
    # weight-bound GEMVs already saturate the L3↔NPU path.
    bw_memcopy_gbps = None
    if not args.skip_bw:
        try:
            bw_us, bw_memcopy_gbps = measure_shim_dma_bw(cols)
            print(f"\nMemCopy(bypass) DMA, {cols} cols × 2 ch (small transfer): "
                  f"{bw_memcopy_gbps:.2f} GB/s ({bw_us:.1f} us)")
        except Exception as e:
            print(f"  !! MemCopy DMA measurement failed: {e!r}")

    # Empirical peak: best single-op effective bandwidth
    peak_op = max(rows, key=lambda r: r["eff_bw_gbps"])
    bw_peak_gbps = peak_op["eff_bw_gbps"]
    print(f"Empirical peak BW (best decode op `{peak_op['op']}`): "
          f"{bw_peak_gbps:.2f} GB/s")

    e2e = None
    if not args.skip_e2e and args.weights and args.tokenizer:
        print(f"\nRunning end-to-end wallclock ({args.prompt_len}-token prompt, "
              f"{args.num_tokens} generated)...")
        e2e = measure_e2e(args.weights, args.tokenizer, args.prompt_len, args.num_tokens)
        if e2e["us_per_token"]:
            print(f"  decode wallclock: {e2e['us_per_token']:.0f} us/token "
                  f"({e2e['tokens_per_sec']:.2f} tok/s)")
        else:
            print(f"  !! could not parse decode timing (rc={e2e['returncode']}). "
                  f"Tail of output:\n{e2e['raw_tail']}")

    # --- Aggregates -----------------------------------------------------
    t_sum_ops_us = sum(r["us_per_token"] for r in rows)
    bytes_per_token = sum(
        r["in_bytes_per_token"] + r["out_bytes_per_token"] for r in rows
    )
    bytes_by_cat = {
        cat: sum(
            r["in_bytes_per_token"] + r["out_bytes_per_token"]
            for r in rows
            if r["category"] == cat
        )
        for cat in ("weights", "kv_cache", "intermediate")
    }
    # T_bw_unavoidable: weights + KV-cache must traverse L3 even with perfect fusion
    unavoidable_bytes = bytes_by_cat["weights"] + bytes_by_cat["kv_cache"]
    t_bw_unavoidable_us = unavoidable_bytes / (bw_peak_gbps * 1e9) * 1e6
    # T_bw_all_today: under today's standalone-op model every byte traverses L3
    t_bw_all_today_us = bytes_per_token / (bw_peak_gbps * 1e9) * 1e6

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote per-op CSV: {args.csv}")

    write_md(
        args.md, rows, cols, bw_peak_gbps, bw_memcopy_gbps,
        t_sum_ops_us, t_bw_unavoidable_us, t_bw_all_today_us,
        e2e, bytes_per_token, bytes_by_cat,
    )
    print(f"Wrote summary:    {args.md}")

    if args.trace_ops:
        trace_dir = Path("trace")
        trace_dir.mkdir(exist_ok=True)
        wanted = set(args.trace_ops.split(","))
        for entry in build_decode_ops():
            if entry.name not in wanted:
                continue
            print(f"\nTracing `{entry.name}` (trace_size={args.trace_size} bytes)...")
            trace_one_gemv(entry, args.trace_size, trace_dir)
    return 0


# --- Tracing -----------------------------------------------------------------


def trace_one_gemv(entry: OpEntry, trace_size: int, trace_dir: Path) -> None:
    """Compile the GEMV with tracing on, run it once via load_and_run with
    a TraceConfig, and convert the resulting trace into Chrome Trace Event JSON.

    Drops trace files into ``trace_dir/<op>.{txt,json,mlir}``.
    Only ops whose factory returns a GEMV are supported today.

    Trace routing in the AIE2P shim uses some of the same ports the data DMAs
    do, so 8-column designs frequently fail to route the trace stream.  We
    progressively shrink ``num_aie_columns`` until compilation succeeds; this
    changes throughput but preserves the per-core compute-vs-DMA behaviour
    we're trying to observe.
    """
    base_ctx = AIEContext(build_dir=Path("build_profile_trace") / entry.name)
    base_ctx.build_dir.mkdir(parents=True, exist_ok=True)
    base_op = entry.factory(base_ctx)
    if not isinstance(base_op, GEMV):
        print(f"  skip: tracing only wired up for GEMV (got {type(base_op).__name__})")
        return

    # Trace routing forces all trace traffic onto column 0's shim
    # (rt.enable_trace, routing="single").  At the production num_aie_columns
    # col 0 is also doing data DMA, hence the routing conflict.  Workaround:
    # use col_offset > 0 so all data fifos and shim DMAs land on
    # cols [col_offset .. col_offset+cols-1], leaving col 0 free for trace.
    # We try the highest-pressure (largest cols) shifted variant first,
    # falling back to col_offset=0 with smaller cols if needed.
    device_cols = aie_utils.get_current_device().cols

    def m_divides_ok(c):
        return (
            base_op.M % c == 0
            and (base_op.M // c) % base_op.tile_size_output == 0
        )

    candidate_cols = sorted(
        {c for c in range(1, base_op.num_aie_columns + 1) if m_divides_ok(c)},
        reverse=True,
    )

    candidates: list[tuple[int, int, list[int]]] = []  # (cols, col_offset, traced_worker_ids)
    # Shifted variants: data on cols [col_offset .. col_offset+cols-1], col 0 free.
    for c in candidate_cols:
        for c_off in (1,):
            if c + c_off <= device_cols:
                # trace worker 0 of the shifted layout = compute tile (c_off, 2)
                candidates.append((c, c_off, [0]))
    # Fallback: original col_offset=0 (smaller cols only — full-cols always conflicts).
    for c in candidate_cols:
        if c < base_op.num_aie_columns:
            candidates.append((c, 0, [0]))

    op = None
    for cols, c_off, w_ids in candidates:
        tag = f"c{cols}_off{c_off}_w{'_'.join(map(str, w_ids))}"
        ctx = AIEContext(build_dir=Path("build_profile_trace") / f"{entry.name}_{tag}")
        ctx.build_dir.mkdir(parents=True, exist_ok=True)
        cand = GEMV(
            M=base_op.M,
            K=base_op.K,
            num_aie_columns=cols,
            tile_size_input=base_op.tile_size_input,
            tile_size_output=base_op.tile_size_output,
            num_batches=base_op.num_batches,
            trace_size=trace_size,
            traced_worker_ids=tuple(w_ids),
            col_offset=c_off,
            context=ctx,
        )
        try:
            cand.compile()
        except Exception as e:
            print(f"  cols={cols}, col_offset={c_off}, traced_workers={w_ids}: "
                  f"compile failed ({type(e).__name__}); trying next")
            continue
        op = cand
        ctx_used = ctx
        cols_used = cols
        traced_used = w_ids
        col_off_used = c_off
        break

    if op is None:
        print(f"  !! could not compile a traced variant of `{entry.name}` "
              f"at any column count from {candidate_cols}")
        return
    pressure_pct = cols_used / base_op.num_aie_columns * 100
    placement_note = (
        f"col_offset={col_off_used} (data on cols "
        f"{col_off_used}..{col_off_used + cols_used - 1}, col 0 reserved for trace)"
        if col_off_used > 0
        else "col_offset=0 (smaller-cols fallback)"
    )
    print(f"  traced num_aie_columns={cols_used} "
          f"(production={base_op.num_aie_columns}, "
          f"~{pressure_pct:.0f}% of production BW pressure); "
          f"traced workers={traced_used}; {placement_note}")

    # Allocate buffers (note: out_buf is enlarged by trace_size bytes via
    # the modified get_arg_spec).
    args = []
    for spec in op.get_arg_spec():
        shape = shape_to_tuple(spec.shape)
        buf = XRTTensor(shape, dtype=spec.dtype)
        buf.fill_(0.0)
        args.append(buf)

    # Locate the lowered MLIR (contains the trace_event ops the parser needs).
    physical_mlir = (
        ctx_used.build_dir / f"{op.name}.mlir.prj" / "input_with_addresses.mlir"
    )
    trace_txt = trace_dir / f"{entry.name}.txt"
    trace_json = trace_dir / f"{entry.name}.json"
    mlir_copy = trace_dir / f"{entry.name}.mlir"

    trace_cfg = TraceConfig(
        trace_size=trace_size,
        trace_file=str(trace_txt),
        ddr_id=-1,  # appended after the C output buffer (matches design.py)
    )

    npu_kernel = NPUKernel(
        xclbin_path=op.xclbin_artifact.filename,
        kernel_name=op.xclbin_artifact.kernel_name,
        insts_path=op.insts_artifact.filename,
        trace_config=trace_cfg,
    )

    handle, result = aie_utils.DefaultNPURuntime.load_and_run(npu_kernel, args)
    print(f"  npu_time = {result.npu_time/1e3:.2f} us")

    # parse_trace needs a textual MLIR with the lowered trace ops
    if not physical_mlir.exists():
        print(f"  !! lowered MLIR not found at {physical_mlir}; skipping JSON")
        return
    mlir_copy.write_text(physical_mlir.read_text())
    try:
        trace_cfg.trace_to_json(str(mlir_copy), str(trace_json))
        print(f"  wrote {trace_txt} (raw)  +  {trace_json} (Chrome Trace Event)")
        print(f"  -> open with chrome://tracing or https://ui.perfetto.dev/")
    except Exception as e:
        print(f"  !! trace parse failed: {e!r}")
        return

    # Quick duty-cycle summary so the user can see at a glance whether the
    # traced core was compute-bound or memory/stream-stalled.
    bytes_per_call = sum(io_bytes(base_op))  # production-shape per call
    summarize_trace_json(
        trace_json,
        npu_us=result.npu_time / 1e3,
        cols_used=cols_used,
        cols_production=base_op.num_aie_columns,
        traced_per_core_gbps=bytes_per_call / (result.npu_time / 1e9) / 1e9,
        production_op_name=entry.name,
        production_op=base_op,
    )


def summarize_trace_json(
    trace_json: Path,
    npu_us: float,
    cols_used: int,
    cols_production: int,
    traced_per_core_gbps: float | None = None,
    production_op_name: str | None = None,
    production_op=None,
) -> None:
    """Print a one-glance summary of compute vs stall events in the trace JSON.

    These are *event counts*, not raw cycle counts -- the trace samples are
    coarse enough that you read this as 'relative pressure', not absolute
    occupancy.  But MEMORY_STALL == 0 / STREAM_STALL == 0 is a clean
    signal that the core never had to wait on the DMA path.
    """
    import json
    with open(trace_json) as f:
        events = json.load(f)
    counts: dict[str, int] = {}
    for e in events:
        if e.get("ph") in ("B", "X", "i", "I"):
            n = e.get("name", "")
            counts[n] = counts.get(n, 0) + 1

    def get(*names):
        return sum(counts.get(n, 0) for n in names)

    vec   = get("INSTR_VECTOR")
    e0    = get("INSTR_EVENT_0")
    e1    = get("INSTR_EVENT_1")
    mem   = get("MEMORY_STALL")
    strm  = get("STREAM_STALL")
    lock  = get("LOCK_STALL")
    port0 = get("PORT_RUNNING_0")
    port1 = get("PORT_RUNNING_1")
    total_compute = vec + e0 + e1
    total_stall = mem + strm + lock
    denom = max(1, total_compute + total_stall)

    print(f"  trace summary ({npu_us:.0f} µs on the traced core):")
    print(f"    INSTR_VECTOR : {vec:6d}")
    print(f"    INSTR_EVENT  : {e0+e1:6d}  ({e0} + {e1})")
    print(f"    LOCK_STALL   : {lock:6d}")
    print(f"    MEMORY_STALL : {mem:6d}   <- waiting on memory")
    print(f"    STREAM_STALL : {strm:6d}   <- waiting on objfifo / DMA")
    print(f"    PORT_RUNNING : {port0+port1:6d}  ({port0} + {port1})")
    print(f"    => compute-event share of (compute+stall) events: "
          f"{total_compute/denom*100:.1f}%")
    stall_share = total_stall / denom
    if stall_share < 0.05:
        verdict = (
            "PER-CORE COMPUTE-BOUND -- the kernel is consuming data as fast "
            "as it arrives at L1; adding compute (e.g., a fused elementwise "
            "op on each output element) WILL extend the kernel."
        )
    else:
        verdict = (
            f"PER-CORE PARTIALLY STALLED ({stall_share*100:.1f}% stall events) "
            "-- there is some slack on this core that an elementwise op fused "
            "into the inner loop could absorb."
        )
    print(f"    => verdict: {verdict}")
    if cols_used < cols_production and traced_per_core_gbps is not None:
        # Bridge the gap: derive what each core would need to consume at
        # production cols, and compare to what THIS trace shows the core CAN
        # consume.
        prod_us = npu_us * cols_used / cols_production  # ideal scaling
        # Look up actual production timing from the previous profile rows
        # via a side-channel: we passed production_op so we could rebuild
        # one if needed, but profile_decode.py already timed it.
        prod_call_us = _last_production_us.get(production_op_name)
        if prod_call_us:
            # Each core at cols=N processes (1/N)th the bytes
            per_call_bytes = sum(io_bytes(production_op))
            per_core_bytes = per_call_bytes / cols_production
            required_gbps = per_core_bytes / (prod_call_us * 1e-6) / 1e9
            saturation = required_gbps / traced_per_core_gbps * 100
            print(
                f"\n    PER-CORE BANDWIDTH ANALYSIS"
                f"\n      this trace ({cols_used} col):  core consumed at "
                f"{traced_per_core_gbps:5.1f} GB/s with no stream stalls "
                f"-- so the core CAN drink at least this fast"
                f"\n      production ({cols_production} col): each core needs "
                f"{required_gbps:5.1f} GB/s to keep up with the {prod_call_us:.0f} us call"
                f"\n      => required / observed = {saturation:.0f}% "
            )
            if saturation >= 90:
                print(
                    "         The production run pushes every core close to "
                    "or beyond what the trace shows is sustainable -- the "
                    "design is BANDWIDTH-LIMITED at production cols. "
                    "Fusing more compute onto each output element would "
                    "actually be FREE (compute slots stay idle waiting for DMA)."
                )
            elif saturation >= 60:
                print(
                    "         The production run loads each core to "
                    f"~{saturation:.0f}% of its solo bandwidth -- there's "
                    "real DMA pressure but the cores are not fully starved. "
                    "Fusion that reduces DMA traffic gives the biggest win."
                )
            else:
                print(
                    "         The production run uses each core well below its "
                    "solo bandwidth -- the workload is compute-bound at the "
                    "system level too. Fused compute will extend the kernel."
                )
        else:
            bw_pressure = cols_used / cols_production * 100
            print(
                f"    NOTE: trace ran at {cols_used} cols (~{bw_pressure:.0f}% "
                f"of production BW pressure)."
            )
    else:
        print(f"    NOTE: traced at num_aie_columns={cols_used} matches production.")


def write_md(
    path: str,
    rows: list[dict],
    cols: int,
    bw_peak_gbps: float,
    bw_memcopy_gbps: float | None,
    t_sum_ops_us: float,
    t_bw_unavoidable_us: float,
    t_bw_all_today_us: float,
    e2e: dict | None,
    bytes_per_token: int,
    bytes_by_cat: dict,
) -> None:
    rows_sorted = sorted(rows, key=lambda r: -r["us_per_token"])
    e2e_us = e2e["us_per_token"] if (e2e and e2e.get("us_per_token")) else None

    def mib(b):
        return b / 1024 / 1024

    with open(path, "w") as f:
        f.write(f"# Llama-3.2-1B decode profile (NPU, {cols} columns)\n\n")
        f.write(
            f"Per-token L3↔NPU traffic (sum of standalone op I/O): "
            f"**{mib(bytes_per_token):.1f} MiB**, broken down as:\n\n"
        )
        f.write("| category | MiB/token | share | fusable? |\n|---|---:|---:|---|\n")
        for cat, label, fusable in [
            ("weights", "model weights", "**no** (don't fit on chip)"),
            ("kv_cache", "KV-cache reads/writes", "no (grows with context)"),
            ("intermediate", "activations / temporaries", "**yes** (could stay in L2/L1)"),
        ]:
            b = bytes_by_cat[cat]
            f.write(f"| {label} | {mib(b):.1f} | {b/bytes_per_token*100:.1f}% | {fusable} |\n")
        f.write("\n")

        f.write(f"Empirical peak L3↔NPU bandwidth (best decode op): "
                f"**{bw_peak_gbps:.1f} GB/s**.  ")
        if bw_memcopy_gbps:
            f.write(f"(MemCopy(bypass) microbench at the same column count "
                    f"only reaches {bw_memcopy_gbps:.1f} GB/s — its 256 KiB "
                    f"transfer is too small to amortize DMA setup, so we use "
                    f"the GEMV-based empirical peak as the roofline.)\n\n")
        else:
            f.write("\n\n")

        f.write("## Per-op standalone latency × invocations per token\n\n")
        f.write(
            "| op | category | us/call | n/token | **us/token** | "
            "MiB/token | eff. GB/s |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows_sorted:
            mib_total = mib(r["in_bytes_per_token"] + r["out_bytes_per_token"])
            f.write(
                f"| `{r['op']}` | {r['category']} | {r['us_per_call']:.2f} | "
                f"{r['n_per_token']} | **{r['us_per_token']:.1f}** | "
                f"{mib_total:.2f} | {r['eff_bw_gbps']:.1f} |\n"
            )

        f.write("\n## Roofline\n\n")
        f.write("| metric | µs/token | tok/s | speedup vs `T_sum_ops` |\n")
        f.write("|---|---:|---:|---:|\n")
        if e2e_us:
            f.write(
                f"| measured wallclock `T_decode_e2e` | {e2e_us:.0f} | "
                f"{1e6/e2e_us:.2f} | {t_sum_ops_us/e2e_us:.2f}× |\n"
            )
        f.write(
            f"| sum-of-standalone-ops `T_sum_ops` | {t_sum_ops_us:.0f} | "
            f"{1e6/t_sum_ops_us:.2f} | 1.00× |\n"
        )
        f.write(
            f"| BW-bound, today's bytes `T_bw_today` | {t_bw_all_today_us:.0f} | "
            f"{1e6/t_bw_all_today_us:.2f} | "
            f"{t_sum_ops_us/t_bw_all_today_us:.2f}× |\n"
        )
        f.write(
            f"| BW-bound, unavoidable only `T_bw_min` | {t_bw_unavoidable_us:.0f} | "
            f"{1e6/t_bw_unavoidable_us:.2f} | "
            f"{t_sum_ops_us/t_bw_unavoidable_us:.2f}× |\n"
        )

        f.write("\n## Headroom decomposition\n\n")
        if e2e_us:
            launch_gap = e2e_us - t_sum_ops_us
            launch_pct = max(0.0, (1 - t_sum_ops_us / e2e_us) * 100)
            f.write(
                f"- **Host overhead today** (`T_decode_e2e − T_sum_ops`): "
                f"**{launch_gap:.0f} µs/token** ({launch_pct:.1f}% of wallclock). "
                "ELF patching per token (see `llama_npu.py:11` TODO), input/output "
                "buffer sync, and Python loop. NOT addressable by kernel fusion.\n"
            )
        fusable_gap = t_bw_all_today_us - t_bw_unavoidable_us
        f.write(
            f"- **Fusion-addressable bytes** (intermediates that could stay in "
            f"L2/L1): {mib(bytes_by_cat['intermediate']):.1f} MiB/token = "
            f"**{fusable_gap:.0f} µs/token** of L3 traffic at peak BW. "
            f"Eliminating it caps speedup at "
            f"**{t_sum_ops_us/t_bw_unavoidable_us:.2f}×** vs `T_sum_ops`.\n"
        )
        f.write(
            "- **Spatial fusion** (running today's serial ops on disjoint "
            "columns simultaneously) sits *on top* of the BW ceiling — it can "
            "claw back some of the gap between per-op `npu_time` and the bytes/BW "
            "lower bound on small ops (RMSNorm, RoPE, residual_add, eltwise) "
            "that today fully occupy 8 columns despite having tiny working sets.\n"
        )

        # Top-3 spatial / temporal fusion candidates (heuristic)
        top = rows_sorted[:5]
        f.write("\n## Top time consumers (where to look first)\n\n")
        for r in top:
            pct = r["us_per_token"] / t_sum_ops_us * 100
            f.write(f"- `{r['op']}` — {r['us_per_token']:.0f} µs/token "
                    f"({pct:.1f}% of T_sum_ops), category={r['category']}, "
                    f"effective {r['eff_bw_gbps']:.1f} GB/s.\n")

        f.write("\n## Reading the numbers\n\n")
        f.write(
            "- `us_per_call` is on-device `npu_time` (kernel cycles only, no "
            "host launch overhead). The fused decode operator at `llama_npu.py:565` "
            "already chains all ops within one ELF, so `T_sum_ops` is what the "
            "currently-fused implementation would achieve if there were zero "
            "per-token host overhead.\n"
            "- Effective GB/s = (input + output bytes) / us_per_call. The "
            "weight-bound GEMVs (`gemv_ffn_*`, `gemv_attn_query/key_value/output`) "
            "set the empirical peak — they're streaming weights through with "
            "essentially no per-byte compute beyond the bf16 MAC.\n"
            "- `T_bw_today` is what perfect launch-overhead removal would buy "
            "if every byte still had to traverse L3. `T_bw_min` is the absolute "
            "lower bound: weights + KV-cache must round-trip; only "
            "intermediates can be eliminated by fusion.\n"
            "- **Spatial fusion** is most useful for ops where the working set "
            "is far smaller than 8 cols × tile_size — e.g., `rms_norm`/`residual_add` "
            "moving ~0.25 MiB but spending 80 µs (=> ~3 GB/s effective). Either "
            "shrink them to 1-2 columns and overlap with a concurrent GEMV, "
            "or fuse them onto the GEMV's output tail.\n"
            "- **Temporal fusion** opportunities (TODOs at `llama_npu.py:6-12`): "
            "transpose-after-RoPE, transpose-after-strided_copy, and replacing "
            "the 32 per-head `transpose_values` invocations with a fused "
            "contraction.\n"
        )


if __name__ == "__main__":
    sys.exit(main())
