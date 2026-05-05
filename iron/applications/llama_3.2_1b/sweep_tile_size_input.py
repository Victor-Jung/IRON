#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep --tile-size-input (m_input) for a fixed GEMV shape and report:
- per-call NPU time
- per-core effective bandwidth
- LOCK_STALL % of trace cycles  (the memory-bound signal)
- INSTR_VECTOR cycles
- correctness (PASS/FAIL/SKIP)

Larger m_input means each shim DMA delivers a bigger (m_input, K) tile and
the kernel runs longer between FIFO acquire/release pairs.  If the GEMV's
effective bandwidth is capped by per-tile DMA setup overhead, BW should
climb (and LOCK_STALL should drop) as m_input grows.  If BW stays flat,
the cap is structural to the layout (memtile, FIFO synchronization, etc.),
not the tile granularity.

L1 budget for A double-buffer is depth(2) * m_input * K * 2 bytes.  With
K=2048, m_input >= 8 starts pressuring the ~64 KiB L1 (alongside B and C
buffers).  Out-of-budget configs will fail to compile -- we skip them.

Compose with --no-compute to compare the structural ceiling vs the real
GEMV (the no-compute kernel keeps the same DMA pattern with zero compute):
    python3 sweep_tile_size_input.py ... --no-compute

Usage example:
    python3 sweep_tile_size_input.py \\
        --M 14336 --K 2048 --cols 7 --col-offset 1 \\
        --m-inputs 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--col-offset", type=int, default=1)
    ap.add_argument("--m-inputs", default="1,2,4,8,16",
                    help="comma-separated list of tile_size_input values to sweep")
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--trace-size", type=int, default=8192)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--no-compute", action="store_true",
                    help="link mv_nocompute.cc -- isolates DMA path from kernel compute")
    ap.add_argument("--out-prefix", default="sweep",
                    help="trace files land in trace/<out_prefix>_ti<N>.{txt,json,mlir}")
    ap.add_argument("--keep-build", action="store_true",
                    help="don't wipe build_trace_gemv between configs (faster reruns)")
    args = ap.parse_args()

    m_inputs = [int(x) for x in args.m_inputs.split(",") if x.strip()]
    trace_gemv_py = str(Path(__file__).parent / "trace_gemv.py")

    rows: list[dict] = []
    for ti in m_inputs:
        tag = f"{args.out_prefix}_ti{ti}"
        build_dir = Path(f"build_trace_gemv/{tag}")
        if not args.keep_build and build_dir.exists():
            shutil.rmtree(build_dir)

        cmd = [
            sys.executable, trace_gemv_py,
            "--M", str(args.M),
            "--K", str(args.K),
            "--cols", str(args.cols),
            "--col-offset", str(args.col_offset),
            "--tile-size-input", str(ti),
            "--num-batches", str(args.num_batches),
            "--trace-size", str(args.trace_size),
            "--n-warmup", str(args.n_warmup),
            "--tag", tag,
        ]
        if args.no_compute:
            cmd.append("--no-compute")

        print(f"\n=== tile_size_input={ti} ===")
        print("  $ " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = result.stdout

        if result.returncode != 0:
            tail = (result.stderr or out)[-500:]
            print(f"  FAILED (exit {result.returncode}). Last 500 chars of stderr:")
            print("  " + "\n  ".join(tail.splitlines()))
            rows.append({"ti": ti, "status": f"compile/run failed (exit {result.returncode})"})
            continue

        # Pull the lines we care about out of trace_gemv.py's output
        row: dict = {"ti": ti}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Results ("):
                # 'Results (1234.5 us, after 3 warmup runs):'
                row["us"] = float(line.split("(")[1].split(" us")[0])
            elif line.startswith("correctness:"):
                row["correctness"] = line.split(":", 1)[1].strip()
            elif "per-core effective BW:" in line:
                row["gbps"] = float(line.split(":", 1)[1].strip().split()[0])
            elif line.startswith("INSTR_VECTOR") and len(line.split()) >= 4 and line.split()[1].isdigit():
                # 'INSTR_VECTOR  <count>  <cycles>  <pct>%'
                parts = line.split()
                row["vec_cyc"] = int(parts[2])
            elif line.startswith("LOCK_STALL") and len(line.split()) >= 4 and line.split()[1].isdigit():
                parts = line.split()
                row["lock_cyc"] = int(parts[2])
                row["lock_pct"] = parts[3].rstrip("%")
            elif "trace span:" in line:
                row["span_cyc"] = int(line.split("trace span:")[1].split()[0])
        rows.append(row)
        # Show the per-config full output too so the user can scroll
        for line in out.splitlines():
            print("  " + line)

    # Final summary table
    print("\n\n=== SUMMARY ===")
    print(f"GEMV M={args.M} K={args.K} cols={args.cols} col_offset={args.col_offset} "
          f"num_batches={args.num_batches} no_compute={args.no_compute}")
    print(
        f"\n{'m_input':>7s}  {'us/call':>9s}  {'GB/s/core':>9s}  "
        f"{'agg GB/s':>9s}  {'span_cyc':>9s}  {'vec_cyc':>9s}  "
        f"{'lock_cyc':>9s}  {'lock %':>7s}  correctness"
    )
    for r in rows:
        if "us" not in r:
            print(f"{r['ti']:>7d}  {'-':>9s}  {'-':>9s}  {'-':>9s}  "
                  f"{'-':>9s}  {'-':>9s}  {'-':>9s}  {'-':>7s}  "
                  f"{r.get('status', 'unknown')}")
            continue
        agg = r["gbps"] * args.cols
        print(
            f"{r['ti']:>7d}  {r['us']:>9.1f}  {r['gbps']:>9.2f}  "
            f"{agg:>9.2f}  {r.get('span_cyc', 0):>9d}  "
            f"{r.get('vec_cyc', 0):>9d}  {r.get('lock_cyc', 0):>9d}  "
            f"{r.get('lock_pct', '?'):>7s}  {r.get('correctness', '?')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
