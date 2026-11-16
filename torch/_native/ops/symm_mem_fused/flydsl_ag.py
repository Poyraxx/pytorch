# mypy: allow-untyped-defs
"""FlyDSL fused all-gather + matmul (symm_mem::fused_all_gather_matmul) on gfx950.

One state per (K, N, TP, B layout, rows per rank, group, device), running one kernel (flydsl_kernels.agmm_gfx950):
blocks [0, KC) pull the peers' shards into the gathered A chunk by chunk, then every block computes its tile, peer
tiles waiting on their chunk flag. One call, all on the current stream: bump the device epoch, stage the shard into
half (epoch & 1) of the symmetric buffer and into this rank's rows of the gathered A, flush (one L2 write-back per
XCD, which the peers' reads of the staged half need), the symmetric-memory barrier (every rank's staging is in, and
every peer is done reading this half from two calls ago), then the fused kernel.

A state is built at its first call: a collective symmetric-memory rendezvous and a FlyDSL compile (which also runs the
kernel once, with no copy blocks and the flags far past any target), so every rank must make its calls in the same
order, and the first call of a state must not be under CUDA graph capture (later calls may be: all state lives on the
device). The gathered A is written on every call, return_A or not. A chunk flag still missing after
_common.wait_ms() fails the launch with a [FATAL] line instead of hanging; the barrier times out the same way
from a state's second call on. States are built without a lock: make a state's first call from one thread.
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
from ._common import ag_supported, aligned, check_group_agrees, wait_ms
from .flydsl_common import (
    arg,
    CHUNK_ROWS,
    control_block,
    gemm_param,
    order_streams,
    SLOT,
    STAGE_BLOCK,
)


# Workgroups go round-robin over the XCDs (8 per XCD on gfx950), so every XCD's L2 gets a flush; a CU mask that left
# an XCD out would leave its staged lines unflushed.
FLUSH_PROGRAMS = 64
# (K, N / TP, TP) -> copy blocks, the best measured at M 65536 (Llama 3.1 70B); else KC_DEFAULT.
KC = {
    (8192, 7168, 8): 28,
    (8192, 1280, 8): 28,
    (8192, 3584, 8): 28,
    (8192, 1024, 8): 28,
    (8192, 14336, 4): 12,
    (8192, 2560, 4): 15,
    (8192, 7168, 4): 12,
    (8192, 2048, 4): 15,
    (8192, 28672, 2): 6,
    (8192, 5120, 2): 8,
    (8192, 14336, 2): 6,
    (8192, 4096, 2): 8,
}
KC_DEFAULT = {8: 28, 4: 12, 2: 8}
_states: dict = {}


@instrument_triton_kernel("symm_mem::fused_all_gather_matmul")
@triton.jit
def _bump(ep):
    tl.store(ep, tl.load(ep) + 1)


@instrument_triton_kernel("symm_mem::fused_all_gather_matmul")
@triton.jit
def _stage3(src, mine, own, ep, n, BUFS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask)
    # BUFS is a power of two; & keeps the half in range once the int32 epoch wraps (% would go negative), and the
    # kernel's copy blocks pick their source half the same way.
    tl.store(mine + (tl.load(ep) & (BUFS - 1)) * n + offs, v, mask=mask)
    tl.store(own + offs, v, mask=mask)


@instrument_triton_kernel("symm_mem::fused_all_gather_matmul")
@triton.jit
def _flush(sink):
    # Launched after _stage3: a system-scope release writes back its XCD's whole L2, which the peers' xGMI reads of
    # the staged half need (dirty lines are not written back at the kernel boundary).
    tl.atomic_add(sink, 0, sem="release", scope="sys")


def _group_m(tpr):
    # The tile order starts XCD x on order positions [x * GROUP_M, (x + 1) * GROUP_M): this rank's own rows, which
    # wait for no chunk, only if 8 * GROUP_M <= tpr (tile rows per rank); else whole XCDs start on peer rows and stall
    # on the copy. Smaller groups reuse B less, so take the largest that fits.
    return next((g for g in (4, 2, 1) if 8 * g <= tpr), 1)


class _State:
    def __init__(self, m, k, n, b_t, kc, group, a0, b):
        rank, world = group.rank(), group.size()
        self.m, self.k, self.n, self.kc, self.rank, self.world, self.group = (
            m,
            k,
            n,
            kc,
            rank,
            world,
            group,
        )
        self.ms = wait_ms()
        lm = m // world
        self.shard, self.nch = lm * k // 2, lm // CHUNK_ROWS
        self.shard_b, self.ch_b = lm * k * 2, CHUNK_ROWS * k * 2
        if lm % CHUNK_ROWS or kc % (world - 1) or self.ch_b % (512 * 16 * 8):
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): {lm} rows per rank, K {k} and "
                f"{kc} copy blocks don't fit the chunking"
            )
        if (m // 256) % 32 or (m // 256) * (n // 256) % 8:
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): M {m}, N {n} don't fit the "
                "XCD-interleaved tile order"
            )
        check_group_agrees(
            "fused_all_gather_matmul",
            group,
            f"fused_all_gather_matmul M {m} K {k} N {n} B transposed {b_t}",
        )
        self.buf = symm_mem.empty(
            2 * self.shard, dtype=torch.int32, device=torch.device("cuda")
        )
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.flags = torch.zeros(
            world * self.nch * SLOT, dtype=torch.int32, device="cuda"
        )
        self.ep = control_block("cuda", self.ms)
        self.bar_ms = 0
        self.sink = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.mult = kc // (world - 1)
        self.prm = gemm_param(m, n, k, b_t, _group_m(self.nch))
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once: with kc 0 (no copy blocks) on flags far past any target.
        never = torch.full_like(self.flags, 1 << 30)
        c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.agmm_gfx950,
            arg(c),
            arg(a0),
            arg(b),
            never.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            rank,
            world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            a0.data_ptr(),
            0,
            self.shard_b,
            self.ch_b,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a_shard, b):
        # Everything that can raise comes before _bump: the epoch it advances is never rolled back, and every wait
        # target is a multiple of it, so a call that bumps and then raises leaves this state out of step for good.
        gat = torch.empty(self.m, self.k, dtype=torch.bfloat16, device="cuda")
        c = torch.empty(self.m, self.n, dtype=torch.bfloat16, device="cuda")
        src = a_shard.view(torch.int32)
        own = gat.view(torch.int32).view(-1)[self.rank * self.shard :]
        _bump[(1,)](self.ep, num_warps=1)
        _stage3[(triton.cdiv(self.shard, STAGE_BLOCK),)](
            src,
            self.buf,
            own,
            self.ep,
            self.shard,
            BUFS=2,
            BLOCK=STAGE_BLOCK,
        )
        _flush[(FLUSH_PROGRAMS,)](self.sink, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = self.ms
        self.gemm(
            c,
            gat,
            b,
            self.flags.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            self.rank,
            self.world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            gat.data_ptr(),
            self.kc,
            self.shard_b,
            self.ch_b,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return gat, c


def run(A_shard, Bs, gather_dim, group_name, return_A=True):
    group = _resolve_process_group(group_name)
    world = group.size()
    k, n, b_t = ag_supported(A_shard, Bs, gather_dim, world)
    A_shard, Bs = aligned(A_shard), [aligned(Bs[0])]
    lm = A_shard.shape[0]
    skey = (k, n, world, b_t, lm, group_name, A_shard.device.index)
    with torch.cuda.device(A_shard.device):
        st = _states.get(skey)
        # A group destroyed and re-created under the same name: the state's handles belong to the old one.
        if st is None or st.group is not group:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_all_gather_matmul (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows {lm}) is "
                    "under CUDA graph capture; it compiles and rendezvouses, so make one eager call first"
                )
            a0 = torch.empty(lm * world, k, dtype=torch.bfloat16, device=A_shard.device)
            st = _State(
                lm * world,
                k,
                n,
                b_t,
                KC.get((k, n, world), KC_DEFAULT[world]),
                group,
                a0,
                Bs[0],
            )
            _states[skey] = st
        order_streams(st)
        gat, c = st(A_shard, Bs[0])
    st.calls += 1
    return (gat if return_A else None), [c]
