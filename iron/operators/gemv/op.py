# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class GEMV(MLIROperator):
    """AIE-accelerated General Matrix-Vector/Vector-Matrix Multiplication layer"""

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    num_batches: int = 1
    kernel_vector_size: int = field(default=64, repr=False)
    # Tracing: when >0, configure the design to emit AIE trace packets to a
    # buffer appended after the C output (ddr_id=-1 in rt.enable_trace), and
    # grow the C arg spec by trace_size bytes so the host buffer is large
    # enough.  Default 0 keeps the op fully backward-compatible.
    trace_size: int = field(default=0, repr=False)
    traced_worker_ids: tuple = field(default=(), repr=False)
    # Shift worker + shim placement off column 0 so col 0 is reserved for
    # the trace stream (rt.enable_trace(routing="single") forces all trace
    # traffic onto col 0's shim, which collides with col 0's data DMAs).
    # Default 0 keeps the production behavior.
    col_offset: int = field(default=0, repr=False)
    # When True, link mv_nocompute.cc instead of mv.cc -- same kernel
    # symbol names and call signatures, but the bodies are no-ops.  The
    # FIFO acquire/release / DMA pattern is unchanged, so this measures
    # the *structural* DMA bandwidth ceiling of the GEMV data layout.
    # Output is meaningless under this flag (correctness check should be
    # skipped by the caller).
    compute_disabled: bool = field(default=False, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
        "num_batches": "batch",
        "trace_size": "trace",
    }

    def __post_init__(self):
        if self.tile_size_output is None:
            self.tile_size_output = self.tile_size_input

        if not (
            self.tile_size_output % self.tile_size_input == 0
            and self.tile_size_output >= self.tile_size_input
        ):
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if not (
            self.K >= self.kernel_vector_size and self.K % self.kernel_vector_size == 0
        ):
            raise ValueError("K must be multiple of kernel_vector_size")

        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)

        # When compute is disabled, use a different .o name so artifact
        # caching doesn't collide with the real GEMV kernel.
        suffix = "_nocompute" if self.compute_disabled else ""
        kwargs = {
            "verbose": mlir_verbose,
            "kernel_object": f"gemv_{self.K}k_{self.kernel_vector_size}vs{suffix}.o",
        }
        if self.trace_size > 0:
            kwargs["trace_size"] = self.trace_size
            if self.traced_worker_ids:
                kwargs["traced_worker_ids"] = list(self.traced_worker_ids)
        if self.col_offset > 0:
            kwargs["col_offset"] = self.col_offset
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.tile_size_input,
                    self.tile_size_output,
                    self.num_batches,
                ),
                kwargs,
            ),
        )

    def get_kernel_artifacts(self):
        suffix = "_nocompute" if self.compute_disabled else ""
        source_file = "mv_nocompute.cc" if self.compute_disabled else "mv.cc"
        return [
            KernelObjectArtifact(
                f"gemv_{self.K}k_{self.kernel_vector_size}vs{suffix}.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / source_file
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.K}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        return [
            AIERuntimeArgSpec("in", batch_dim + (self.M, self.K)),  # matrix
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector
            AIERuntimeArgSpec("out", batch_dim + (self.M,)),  # output
        ]
        # NOTE: when trace_size > 0 the C buffer is grown at runtime by
        # load_and_run/prepare_args_for_trace (ddr_id=-1 path); we do NOT
        # enlarge it here, otherwise the trace bytes land inside the
        # prefix region that gets discarded.
