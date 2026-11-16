# mypy: allow-untyped-defs
"""The parts of the FlyDSL symm_mem overrides that need neither FlyDSL nor triton: the predicates, operand alignment,
the timeout setting and the ranks' agreement check before a state's rendezvous.

Imports nothing but torch, so the routing check (flydsl_impl) stays cheap and works without the DSL imported, and
tests run without a GPU. The predicates read only shapes, strides, dtypes, devices and the group size, all the same
on every rank of a group; the group size is an argument (flydsl_impl passes the group's, tests pass any).
"""

from __future__ import annotations

import collections
import os

import torch


WORLDS = (2, 4, 8)
# All-gather + matmul: above this N per rank the stock path is faster (its copy hides behind its own matmul).
AG_MAX_N = 20480
# Rows per reduce-scatter chunk band: the reducers sum a band once all its tiles from every source are in.
BAND_ROWS = {8: 256, 4: 512, 2: 1024}
TIMEOUT_ENV = "TORCH_SYMM_MEM_FLYDSL_TIMEOUT_MS"
_AGREE = (
    "TORCH_SYMM_MEM_FLYDSL, TORCH_DISABLE_NATIVE_JIT, the FlyDSL and triton installs"
)
_states_built: collections.Counter = collections.Counter()


def b_transposed(B):
    """True for W.t() (column-major), False for row-major, None for anything else."""
    if B.stride() == (1, B.shape[0]):
        return True
    if B.stride() == (B.shape[1], 1):
        return False
    return None


def _operands_ok(A, B):
    """bf16, 2D, on one device, A contiguous, K matching. Pointers are not read: they can differ between ranks, and
    every rank of the group must take the same path (the ops align their operands instead, see aligned())."""
    return (
        A.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and B.device == A.device
        and A.dim() == 2
        and B.dim() == 2
        and A.is_contiguous()
        and B.shape[0] == A.shape[1]
    )


def aligned(t):
    """t, or a copy with the same strides at a 16-byte aligned address (the kernels' vector loads)."""
    return t if t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.preserve_format)


def ag_supported(A_shard, Bs, gather_dim, world):
    """(K, N, B transposed) if the fused all-gather + matmul kernel takes this call, else None."""
    if gather_dim != 0 or len(Bs) != 1 or world not in WORLDS:
        return None
    B = Bs[0]
    if not _operands_ok(A_shard, B):
        return None
    b_t = b_transposed(B)
    (lm, k), n = A_shard.shape, B.shape[1]
    if b_t is None or lm <= 0 or n <= 0 or k <= 0:
        return None
    tm = (
        lm * world // 256
    )  # tile rows: the XCD-interleaved order needs whole groups of 32 and tiles in 8s
    if lm % 256 or n % 256 or k % 128 or tm % 32 or tm * (n // 256) % 8:
        return None
    if n > AG_MAX_N:
        return None
    # The gathered A and the output are addressed per tile; B's buffer offsets (one descriptor over K x N) and
    # _stage3's int32 offsets into the shard are still 32-bit.
    if k * n * 2 >= 2**32 or lm * k * 2 >= 2**32:
        return None
    return k, n, b_t


def rs_supported(A, B, reduce_op, scatter_dim, world):
    """(K, N, B transposed) if the fused matmul + reduce-scatter kernel takes this call, else None."""
    if reduce_op != "sum" or scatter_dim != 0 or world not in WORLDS:
        return None
    if not _operands_ok(A, B):
        return None
    b_t = b_transposed(B)
    (m, k), n = A.shape, B.shape[1]
    if b_t is None or m <= 0 or n <= 0 or k <= 0:
        return None
    br = BAND_ROWS[world]
    g = (
        br // 256 * (n // 256)
    )  # tiles per chunk band: the XCD-interleaved order takes them in groups of 8 g
    if (
        m % world
        or (m // world) % br
        or n % 256
        or k % 128
        or (m // 256) * (n // 256) % (8 * g)
    ):
        return None
    if br * n % (512 * 8 * 2):  # whole reducer items per band
        return None
    # A and the staging are addressed per tile; B's buffer offsets (one descriptor over K x N) and the TP 8 reducers'
    # (one descriptor per chunk, M / TP x N) are still 32-bit.
    if k * n * 2 >= 2**32 or (m // world) * n * 2 >= 2**32:
        return None
    if world == 8:
        # The TP 8 kernel's XCD order: XCD x takes 4 tile rows x (world * g / 32) tile columns of each band step, which
        # covers every tile only if these divide (N % 1024); otherwise tiles repeat, some chunks never get their counter
        # adds, and the owners' reducers spin forever.
        rows = world * (br // 256)
        if rows % 4 or 8 % (rows // 4) or world * g % 32:
            return None
    return k, n, b_t


def wait_ms() -> int:
    """TORCH_SYMM_MEM_FLYDSL_TIMEOUT_MS (default 60000; 0 = wait forever), read once per state before its rendezvous:
    how long a fused kernel's tile / reducer waits for a peer before it prints a [FATAL] line and traps, and the
    symmetric-memory barrier's timeout from the state's second call on (the first call's peers may still be
    compiling). The trap fails the launch and torch raises at the next sync; some HIP runtimes abort the process
    instead."""
    raw = os.environ.get(TIMEOUT_ENV, "60000")
    try:
        ms = int(raw)
    except ValueError:
        ms = -1
    if not 0 <= ms < 2**31:
        raise ValueError(
            f"{TIMEOUT_ENV}={raw!r}: expected milliseconds in [0, 2^31), 0 for no timeout"
        )
    return ms


def check_group_agrees(op, group, what):
    """Before a state's rendezvous: wait (the group store's timeout) until every rank of `group` is building the same
    state. A rank that took the stock path never arrives, and the rendezvous would otherwise leave this one waiting
    with no hint why."""
    from torch.distributed.distributed_c10d import _get_process_group_store

    store = _get_process_group_store(group)
    n = _states_built[group.group_name]
    _states_built[group.group_name] += 1
    keys = [f"symm_mem_flydsl/{n}/{r}" for r in range(group.size())]
    store.set(keys[group.rank()], what)
    try:
        store.wait(keys)
    except RuntimeError as e:  # DistStoreError on timeout
        raise RuntimeError(
            f"{op} (FlyDSL): not every rank of group {group.group_name} reached the FlyDSL path for {what}. "
            f"The overrides must be registered alike on every rank ({_AGREE}, the registry's off switches)"
        ) from e
    seen = {store.get(k).decode() for k in keys}
    if seen != {what}:
        raise RuntimeError(
            f"{op} (FlyDSL): the ranks of group {group.group_name} are building different states: {sorted(seen)}"
        )
