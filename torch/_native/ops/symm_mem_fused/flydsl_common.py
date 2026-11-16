# mypy: allow-untyped-defs
"""Pieces of the FlyDSL fused all-gather + matmul (flydsl_ag) state that the other fused ops share.

Imported on the first admitted call only (flydsl_impl._kernels): it pulls in flydsl and the vendored GEMM. The parts
that need neither are in _common.
"""

from __future__ import annotations

import flydsl.compiler as flyc

import torch
from torch._inductor.kernel.vendored_templates.flydsl.kernels.gemm_gfx950 import (
    GEMM_DTYPE_BF16,
    infer_has_k_tail,
    make_gemm_param_and_validate,
)


# flydsl_kernels.fused_hti_kernel's tile config.
CFG = dict(
    tile_m=256,
    tile_n=256,
    tile_k=64,
    stages=2,
    m_waves=2,
    n_waves=4,
    group_m=4,
    use_half_tile_interleaved=True,
)
CHUNK_ROWS = 256  # rows per flagged chunk of the gathered A: one tile row
SLOT = 32  # int32 stride between flags, so each flag has its own 128-byte line
STAGE_BLOCK = 4096


def control_block(device, ms):
    """The fused kernels' int32 control block: [epoch, timeout reporters, timeout ms (2^31 - 1 for none), unused]."""
    return torch.tensor(
        [0, 0, ms if ms > 0 else 2**31 - 1, 0], dtype=torch.int32, device=device
    )


def order_streams(st):
    """Order this call on state `st` (current device) after the state's previous call: a state's epoch and staging
    assume in-order calls (one state on two unordered streams hangs the AG); different states are
    independent. Under CUDA graph capture nothing is waited on: torch.cuda.graph synchronizes before capture, and a
    capturing stream may not wait on uncaptured work, so within one capture keep a state's calls on one stream. Nor
    is a replay ordered against eager calls: don't replay a graph while eager calls (or another replay) on the same
    state may run."""
    cur = torch.cuda.current_stream()
    if (
        st.stream is not None
        and st.stream != cur
        and not torch.cuda.is_current_stream_capturing()
    ):
        cur.wait_stream(st.stream)
    st.stream = cur


def arg(t):
    return flyc.from_torch_tensor(t).mark_layout_dynamic()


def gemm_param(m, n, k, b_t, group_m=CFG["group_m"]):
    prm = make_gemm_param_and_validate(
        m,
        n,
        k,
        {
            "dtype_id": GEMM_DTYPE_BF16,
            **CFG,
            "group_m": group_m,
            "a_is_transposed": False,
            "b_is_transposed": b_t,
            "has_bias": False,
            "has_k_tail": infer_has_k_tail(k, CFG["tile_k"], CFG["stages"], True),
        },
    )
    if prm is None:
        raise RuntimeError(f"FlyDSL GEMM: no valid config for M {m}, N {n}, K {k}")
    return prm
