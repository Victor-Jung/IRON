// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Drop-in no-compute replacement for mv.cc: same exported symbols and call
// signatures, but the bodies do nothing.  When linked into the GEMV design,
// the FIFO acquire/release pattern (and therefore the L3->shim->L1 DMA path)
// remains identical, while per-call vector compute drops to zero.  This
// isolates DMA / FIFO-synchronization throughput from kernel compute, which
// is useful for measuring the structural BW ceiling of the data movement
// layout used by GEMV (small (m_input, K) tiles, ObjectFifo back-pressure,
// etc.).

#define NOCPP
#include <stdint.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

extern "C" {

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out)
{
    (void)m;
    (void)row_offset;
    (void)a_in;
    (void)b_in;
    (void)c_out;
}

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out)
{
    (void)m;
    (void)row_offset;
    (void)a_in;
    (void)b_in;
    (void)c_out;
}

}  // extern "C"
