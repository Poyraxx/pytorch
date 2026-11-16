# mypy: allow-untyped-defs
"""FlyDSL fused matmul + reduce-scatter (symm_mem::fused_matmul_reduce_scatter) on gfx950.

One state per (K, N, TP, B layout, rows, group, device), running one kernel (flydsl_kernels.rsmm_gfx950): every
block computes one tile of this rank's partial into its symmetric staging buffer in reduce-scatter order and signals
the chunk's owner; blocks [0, R) then sum this rank's chunk of every source's staging into the output. One call, all on
the current stream: bump the device epoch, the symmetric-memory barrier (every peer is done reading this rank's
staging from the previous call), the kernel. The staging buffer is M x N bf16 per state.

A state is built at its first call: a collective symmetric-memory rendezvous and a FlyDSL compile (which also runs the
kernel once, with no reducers and every counter add into local scratch), so every rank must make its calls in the
same order, and the first call of a state must not be under CUDA graph capture (later calls may be). A band counter
still short after _common.wait_ms() fails the launch with a [FATAL] line instead of hanging; the barrier times out the
same way from a state's second call on. States are built without a lock: make a state's first call from one thread.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import triton
import triton.language as tl

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch._native.instrumentation import instrument_triton_kernel
from torch.distributed.distributed_c10d import _resolve_process_group

from . import flydsl_kernels as K
from ._common import aligned, BAND_ROWS, check_group_agrees, rs_supported, wait_ms
from .flydsl_common import arg, control_block, gemm_param, order_streams


# (K, N, TP) -> reducer blocks, the best measured at M 65536 (Llama 3.1 70B); else R_DEFAULT.
R = {
    (1024, 8192, 8): 28,
    (3584, 8192, 8): 16,
    (1280, 8192, 8): 28,
    (7168, 8192, 8): 16,
    (2048, 8192, 4): 16,
    (7168, 8192, 4): 8,
    (2560, 8192, 4): 16,
    (14336, 8192, 4): 16,
    (4096, 8192, 2): 16,
    (14336, 8192, 2): 8,
    (5120, 8192, 2): 16,
    (28672, 8192, 2): 8,
}
R_DEFAULT = {8: 28, 4: 12, 2: 8}
_states: dict = {}


@instrument_triton_kernel("symm_mem::fused_matmul_reduce_scatter")
@triton.jit
def _bump(ep):
    tl.store(ep, tl.load(ep) + 1)


class _State:
    def __init__(self, m, k, n, b_t, r, br, group, a0, b):
        rank, world = group.rank(), group.size()
        self.m, self.k, self.n, self.r, self.rank, self.world = m, k, n, r, rank, world
        self.group, self.ms = group, wait_ms()
        self.lm, self.br = m // world, br
        self.nb, self.rpb = self.lm // br, br // 256
        tiles, g = (m // 256) * (n // 256), self.rpb * (n // 256)
        if self.lm % br or br % 256 or n % 256 or tiles % (8 * g):
            raise RuntimeError(
                f"fused_matmul_reduce_scatter (FlyDSL): M {m}, N {n} don't fit "
                f"{br}-row bands"
            )
        if br * n % (512 * 8 * 2):
            raise RuntimeError(
                f"fused_matmul_reduce_scatter (FlyDSL): N {n} doesn't give whole "
                f"reducer items per {br}-row band"
            )
        check_group_agrees(
            "fused_matmul_reduce_scatter",
            group,
            f"fused_matmul_reduce_scatter M {m} K {k} N {n} B transposed {b_t}",
        )
        self.buf = symm_mem.empty(
            m * n + self.nb * 32 * 2,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.stage = self.buf[: m * n].view(m, n)
        self.cnt = self.buf[m * n :].view(torch.int32)
        self.cnt.zero_()  # peers add only after this call's barrier, which this rank reaches after the zeroing
        self.cnt_off = m * n * 2
        self.ep = control_block("cuda", self.ms)
        self.bar_ms = 0
        self.prm = gemm_param(m, n, k, b_t)
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once with no reducers and every peer entry on this rank's own staging, counter
        # offset 0: counter adds land on staging words every call rewrites.
        table = torch.full(
            (world,), self.stage.data_ptr(), dtype=torch.int64, device="cuda"
        )
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.rsmm_gfx950,
            arg(self.stage),
            arg(a0),
            arg(b),
            table.data_ptr(),
            0,
            self.ep.data_ptr(),
            self.stage.data_ptr(),
            rank,
            world,
            0,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a, b):
        # Everything that can raise comes before _bump: the epoch it advances is never rolled back, and every wait
        # target is a multiple of it, so a call that bumps and then raises leaves this state out of step for good.
        out = torch.empty(self.lm, self.n, dtype=torch.bfloat16, device="cuda")
        _bump[(1,)](self.ep, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = self.ms
        self.gemm(
            self.stage,
            a,
            b,
            self.hdl.buffer_ptrs_dev,
            self.cnt_off,
            self.ep.data_ptr(),
            out.data_ptr(),
            self.rank,
            self.world,
            self.r,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return out


def run(A, B, reduce_op, scatter_dim, group_name):
    group = _resolve_process_group(group_name)
    world = group.size()
    k, n, b_t = rs_supported(A, B, reduce_op, scatter_dim, world)
    A, B = aligned(A), aligned(B)
    m = A.shape[0]
    skey = (k, n, world, b_t, m // world, group_name, A.device.index)
    with torch.cuda.device(A.device):
        st = _states.get(skey)
        # A group destroyed and re-created under the same name: the state's handles belong to the old one.
        if st is None or st.group is not group:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_matmul_reduce_scatter (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows "
                    f"{m // world}) is under CUDA graph capture; it compiles and rendezvouses, so make one eager "
                    "call first"
                )
            st = _State(
                m,
                k,
                n,
                b_t,
                R.get((k, n, world), R_DEFAULT[world]),
                BAND_ROWS[world],
                group,
                A,
                B,
            )
            _states[skey] = st
        order_streams(st)
        out = st(A, B)
    st.calls += 1
    return out
