#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Trace a single GEMV configuration on the NPU and emit Chrome Trace Event JSON.

Useful for sweeping num_aie_columns to investigate memory-vs-compute behavior.

LOCK_STALL interpretation:
    A and C ObjectFifos are double-buffered (depth=2 in design.py), so the
    kernel can process tile N while DMA fills tile N+1.  A LOCK_STALL means
    the kernel finished its current tile *before* the DMA delivered the
    next one -- i.e., the design is bandwidth-limited at that point.

    LOCK_STALL ≈ 0  -> compute-bound (DMA hides under compute perfectly)
    LOCK_STALL > ~5% of INSTR_VECTOR -> memory-bound

To free col 0 for the trace stream (rt.enable_trace forces routing="single"
to col 0's shim), we shift data placement via col_offset; the data lives on
cols [col_offset .. col_offset+cols-1] and col 0 is reserved for trace.

Examples:
    # 7-col GEMV, M=14336 (= 7 * 2048), K=2048, col 0 reserved for trace.
    python3 trace_gemv.py --M 14336 --K 2048 --cols 7 --col-offset 1

    # Sweep cols on the production gemv_ffn_up_gate (M=8192, K=2048)
    for c in 1 2 4; do
      python3 trace_gemv.py --M 8192 --K 2048 --cols $c --col-offset 1 \\
                            --tile-size-output $((8192 / c)) \\
                            --tag ffn_up_gate_c$c
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig

from iron.common.context import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.gemv.op import GEMV
from iron.operators.gemv.reference import generate_golden_reference


def _patch_trace_parser_unbound_cycles_bug():
    """Workaround upstream bug in mlir_aie/utils/trace/parse.py:
    `cycles` is referenced before assignment when a stream's first command
    is `Repeat` (no preceding `Single`/`Multiple` to bind `cycles`).  We
    inject the initial bindings into convert_commands_to_json's __globals__
    via a wrapper that pre-assigns sane defaults before delegating.
    """
    from aie.utils.trace import parse as _parse

    if getattr(_parse, "_iron_cycles_patched", False):
        return
    original = _parse.convert_commands_to_json

    def patched(trace_events, commands, pid_events, events_module):
        # Re-implement the outer loop with cycles/multiple_list/event
        # initialized; delegate per-stream to the upstream function via a
        # tiny shim that pre-pends a no-op "Single" command if the first is
        # a Repeat (cheapest fix that keeps upstream semantics).
        for tt, byte_stream_dict in enumerate(commands):
            for loc, cmds in list(byte_stream_dict.items()):
                if cmds and "Repeat" in cmds[0].get("type", ""):
                    # Inject a synthetic zero-length Single so cycles=0,
                    # event=None get bound without producing visible events.
                    byte_stream_dict[loc] = [
                        {"type": "Single", "event": "0", "cycles": "0"}
                    ] + cmds
        return original(trace_events, commands, pid_events, events_module)

    _parse.convert_commands_to_json = patched
    # parse_trace imported convert_commands_to_json at module scope; rebind there too.
    _parse.parse_trace.__globals__["convert_commands_to_json"] = patched
    _parse._iron_cycles_patched = True


_patch_trace_parser_unbound_cycles_bug()


def shape_to_tuple(s):
    return s if isinstance(s, tuple) else (int(s),)


def _make_args(op: GEMV, golden) -> list:
    """Build a fresh args list (XRTTensors) and load the golden A and B into NPU."""
    M, K = op.M, op.K
    nb = op.num_batches
    args = []
    for i, spec in enumerate(op.get_arg_spec()):
        buf = XRTTensor(shape_to_tuple(spec.shape), dtype=spec.dtype)
        if i == 0:  # A
            tv = buf.torch_view()
            if nb > 1:
                tv = tv.reshape(nb, M, K)
                for b in range(nb):
                    tv[b] = golden["A"]
            else:
                tv.reshape(M, K)[:] = golden["A"]
        elif i == 1:  # B
            tv = buf.torch_view()
            if nb > 1:
                tv.reshape(nb, K)[:] = golden["B"]
            else:
                tv.reshape(K)[:] = golden["B"]
        else:  # C: zero (will be overwritten)
            buf.fill_(0.0)
        buf.to("npu")
        args.append(buf)
    return args


def time_and_trace(
    op: GEMV,
    trace_size: int,
    build_dir: Path,
    output_dir: Path,
    tag: str,
    n_warmup: int = 3,
    rel_tol: float = 0.04,
    abs_tol: float = 1e-3,
) -> dict:
    """Compile, warm up, run with TraceConfig, verify correctness, dump trace JSON."""
    op.compile()

    # Golden reference (real bf16 inputs, expected C from torch matmul).
    golden = generate_golden_reference(M=op.M, K=op.K)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_txt = output_dir / f"{tag}.txt"
    trace_json = output_dir / f"{tag}.json"
    mlir_copy = output_dir / f"{tag}.mlir"

    cfg = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt), ddr_id=-1)
    npu_kernel = NPUKernel(
        xclbin_path=op.xclbin_artifact.filename,
        kernel_name=op.xclbin_artifact.kernel_name,
        insts_path=op.insts_artifact.filename,
        trace_config=cfg,
    )

    # Warmup runs (same compiled kernel & TraceConfig, so they go through the
    # same prepare/extract path -- traces from these get overwritten by the
    # final run).  Each iteration gets a fresh args list because
    # prepare_args_for_trace replaces args[-1] in place.
    for _ in range(n_warmup):
        warmup_args = _make_args(op, golden)
        aie_utils.DefaultNPURuntime.load_and_run(npu_kernel, warmup_args)

    # Final timed + traced run; this is the trace we save and verify.
    args = _make_args(op, golden)
    handle, result = aie_utils.DefaultNPURuntime.load_and_run(npu_kernel, args)

    # Functional correctness: after extract_trace_from_args (run inside
    # load_and_run when ddr_id=-1), args[-1] is a numpy bf16 array of the
    # original C shape -- the actual NPU output.
    nb = op.num_batches
    actual = args[-1]  # numpy array, dtype bfloat16
    if nb > 1:
        actual = actual.reshape(nb, op.M)[0]  # all batches identical
    expected = golden["C"]  # torch.bfloat16, shape (M,)
    errors = verify_buffer(
        torch.from_numpy(actual.view(np.uint16)).view(torch.bfloat16),
        "C",
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_error_rate=0.001,  # bf16 dot-product accumulation can have a few outliers
    )
    correctness = "PASS" if not errors else f"FAIL ({len(errors)} mismatches)"

    physical_mlir = build_dir / f"{op.name}.mlir.prj" / "input_with_addresses.mlir"
    if not physical_mlir.exists():
        raise RuntimeError(f"lowered MLIR not found at {physical_mlir}")
    mlir_copy.write_text(physical_mlir.read_text())
    parse_error = None
    try:
        cfg.trace_to_json(str(mlir_copy), str(trace_json))
    except Exception as e:
        parse_error = e
        # Write a minimal placeholder JSON so the rest of the pipeline doesn't
        # NPE on a missing file; the raw .txt is still the source of truth.
        trace_json.write_text("[]")

    # Integrate duration per event name from B/E pairs (counts are misleading
    # because LOCK_STALL events last thousands of cycles each while INSTR_VECTOR
    # events are 1 cycle each).  Also keep counts for reference.
    with open(trace_json) as f:
        events = json.load(f)
    counts: dict[str, int] = {}
    durations: dict[str, int] = {}  # cycles
    open_starts: dict[tuple, int] = {}  # (name, pid, tid) -> ts of last B
    span_min = None
    span_max = None
    for e in events:
        ph = e.get("ph")
        name = e.get("name", "")
        ts = e.get("ts")
        if ph not in ("B", "E", "X"):
            continue
        if ts is not None:
            span_min = ts if span_min is None else min(span_min, ts)
            span_max = ts if span_max is None else max(span_max, ts)
        key = (name, e.get("pid"), e.get("tid"))
        if ph == "B":
            counts[name] = counts.get(name, 0) + 1
            open_starts[key] = ts
        elif ph == "E":
            start = open_starts.pop(key, None)
            if start is not None and ts is not None:
                durations[name] = durations.get(name, 0) + (ts - start)
        elif ph == "X":
            counts[name] = counts.get(name, 0) + 1
            durations[name] = durations.get(name, 0) + int(e.get("dur", 0))

    trace_span = (span_max - span_min) if (span_min is not None and span_max is not None) else 0

    return {
        "npu_us": result.npu_time / 1e3,
        "trace_txt": trace_txt,
        "trace_json": trace_json,
        "trace_mlir": mlir_copy,
        "counts": counts,
        "durations_cycles": durations,
        "trace_span_cycles": trace_span,
        "correctness": correctness,
        "n_warmup": n_warmup,
        "parse_error": parse_error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--M", type=int, required=True, help="rows of the matrix")
    ap.add_argument("--K", type=int, required=True,
                    help="cols of the matrix (must be multiple of kernel_vector_size=64)")
    ap.add_argument("--cols", type=int, required=True,
                    help="num_aie_columns of GEMV data; M must divide cols")
    ap.add_argument("--col-offset", type=int, default=1,
                    help="shift data placement to cols [col_offset..col_offset+cols-1]; "
                         "col 0 is reserved for trace.  Default 1.")
    ap.add_argument("--tile-size-input", type=int, default=4)
    ap.add_argument("--tile-size-output", type=int, default=None,
                    help="default = min(M//cols, 1024) clamped to fit L1 (~64 KB) "
                         "with depth=2 double-buffering")
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--traced-worker", type=int, default=0,
                    help="which worker (0..cols-1) to trace; default 0 = the core "
                         "at compute tile (col_offset, 2)")
    ap.add_argument("--trace-size", type=int, default=8192,
                    help="trace buffer bytes; default 8192")
    ap.add_argument("--n-warmup", type=int, default=3,
                    help="warmup iterations before the timed/traced run "
                         "(smooths cold-start outliers in npu_time); default 3")
    ap.add_argument("--rel-tol", type=float, default=0.04,
                    help="relative tolerance for correctness check; default 0.04")
    ap.add_argument("--abs-tol", type=float, default=1e-3,
                    help="absolute tolerance for correctness check; default 1e-3")
    ap.add_argument("--build-dir", default=None,
                    help="default = build_trace_gemv/<tag>/")
    ap.add_argument("--output-dir", default="trace",
                    help="where to write {tag}.{txt,json,mlir}")
    ap.add_argument("--tag", default=None,
                    help="filename stem; default = M{M}_K{K}_c{cols}_off{col_offset}")
    args = ap.parse_args()

    device = aie_utils.get_current_device()
    print(f"NPU: {device!r} (cols={device.cols})")

    # Validate
    if args.cols + args.col_offset > device.cols:
        print(f"ERROR: cols ({args.cols}) + col_offset ({args.col_offset}) "
              f"> device.cols ({device.cols})", file=sys.stderr)
        return 2
    if args.M % args.cols != 0:
        print(f"ERROR: M ({args.M}) is not divisible by cols ({args.cols})",
              file=sys.stderr)
        return 2
    if args.tile_size_output is not None:
        tso = args.tile_size_output
    else:
        # Pick the largest divisor of (M/cols) that is <= 1024.
        per_core_rows = args.M // args.cols
        tso = min(per_core_rows, 1024)
        while per_core_rows % tso != 0:
            tso -= 1
    if (args.M // args.cols) % tso != 0:
        print(f"ERROR: (M/cols)={args.M // args.cols} is not divisible by "
              f"tile_size_output={tso}", file=sys.stderr)
        return 2
    if args.K % 64 != 0:
        print(f"ERROR: K ({args.K}) must be a multiple of 64 "
              f"(kernel_vector_size)", file=sys.stderr)
        return 2
    if args.traced_worker >= args.cols:
        print(f"ERROR: traced_worker ({args.traced_worker}) >= cols ({args.cols})",
              file=sys.stderr)
        return 2

    tag = args.tag or f"M{args.M}_K{args.K}_c{args.cols}_off{args.col_offset}"
    build_dir = Path(args.build_dir or f"build_trace_gemv/{tag}")
    build_dir.mkdir(parents=True, exist_ok=True)

    ctx = AIEContext(build_dir=build_dir)
    op = GEMV(
        M=args.M,
        K=args.K,
        num_aie_columns=args.cols,
        tile_size_input=args.tile_size_input,
        tile_size_output=tso,
        num_batches=args.num_batches,
        trace_size=args.trace_size,
        traced_worker_ids=(args.traced_worker,),
        col_offset=args.col_offset,
        context=ctx,
    )

    print(f"GEMV M={args.M} K={args.K} cols={args.cols} col_offset={args.col_offset} "
          f"tile_size_output={tso} num_batches={args.num_batches} "
          f"traced_worker={args.traced_worker}")
    print(f"  data on cols [{args.col_offset}..{args.col_offset + args.cols - 1}]"
          f", trace stream lands on col 0's shim")

    info = time_and_trace(
        op, args.trace_size, build_dir, Path(args.output_dir), tag,
        n_warmup=args.n_warmup, rel_tol=args.rel_tol, abs_tol=args.abs_tol,
    )

    bytes_per_call = sum(
        int(np.prod(shape_to_tuple(s.shape)) * np.dtype(s.dtype).itemsize)
        for s in op.get_arg_spec()
    )
    eff_bw_per_core = bytes_per_call / args.cols / (info["npu_us"] * 1e-6) / 1e9

    c = info["counts"]
    d = info["durations_cycles"]
    span = max(1, info["trace_span_cycles"])

    def cyc(name): return d.get(name, 0)
    def cnt(name): return c.get(name, 0)

    vec_cyc  = cyc("INSTR_VECTOR")
    e0_cyc   = cyc("INSTR_EVENT_0") + cyc("INSTR_EVENT_1")
    lock_cyc = cyc("LOCK_STALL")
    mem_cyc  = cyc("MEMORY_STALL")
    strm_cyc = cyc("STREAM_STALL")
    port_cyc = cyc("PORT_RUNNING_0") + cyc("PORT_RUNNING_1")

    compute_cyc = vec_cyc + e0_cyc
    stall_cyc = lock_cyc + mem_cyc + strm_cyc

    print(f"\nResults ({info['npu_us']:.1f} us, after {info['n_warmup']} warmup runs):")
    print(f"  correctness: {info['correctness']}")
    if info.get("parse_error") is not None:
        print(f"  WARNING: trace JSON parse failed: {info['parse_error']!r}")
        print(f"           raw trace at {info['trace_txt']} is still valid; "
              "duration stats below are based on whatever the parser managed "
              "to emit.")
    print(f"  per-core effective BW: {eff_bw_per_core:.1f} GB/s")
    print(f"  trace span: {span} cycles "
          f"(~{span/info['npu_us']:.1f} cycles/us)")
    print(
        f"  {'event':14s}  {'count':>7s}  {'cycles':>10s}  {'% of span':>10s}"
    )
    for name in (
        "INSTR_VECTOR", "INSTR_EVENT_0", "INSTR_EVENT_1",
        "LOCK_STALL", "MEMORY_STALL", "STREAM_STALL",
        "PORT_RUNNING_0", "PORT_RUNNING_1",
    ):
        if cnt(name) == 0 and cyc(name) == 0:
            continue
        print(
            f"  {name:14s}  {cnt(name):>7d}  {cyc(name):>10d}  "
            f"{cyc(name)/span*100:>9.1f}%"
        )

    # The actual memory-vs-compute verdict, by cycle time:
    lock_pct = lock_cyc / span * 100
    compute_pct = compute_cyc / span * 100
    print(
        f"\n  compute (INSTR_VECTOR + INSTR_EVENT) : {compute_pct:5.1f}% of trace cycles"
    )
    print(
        f"  LOCK_STALL (waiting for L1 data)     : {lock_pct:5.1f}% of trace cycles "
        "<-- the actual memory-bound signal"
    )
    if lock_pct < 5:
        verdict = (f"COMPUTE-BOUND ({lock_pct:.1f}% LOCK_STALL): the depth=2 "
                   "double-buffer hides DMA latency well under compute.")
    elif lock_pct < 25:
        verdict = (f"MILDLY MEMORY-BOUND ({lock_pct:.1f}% LOCK_STALL): the "
                   "double-buffer can't fully hide DMA latency. Adding compute "
                   "to the inner loop is partially free until ~"
                   f"{lock_pct:.0f}% extra arithmetic.")
    else:
        verdict = (f"MEMORY-BOUND ({lock_pct:.1f}% LOCK_STALL): the kernel "
                   "spends a large fraction of cycles waiting for the next A "
                   "tile to arrive in L1. Reducing L3 traffic per output "
                   "(weight quantization, smaller tiles, fewer cores per shim) "
                   "is the lever; adding more compute is essentially free here.")
    print(f"  verdict      : {verdict}")
    print(f"\nTrace files:")
    print(f"  raw:    {info['trace_txt']}")
    print(f"  JSON:   {info['trace_json']}  (open in Perfetto)")
    print(f"  MLIR:   {info['trace_mlir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
