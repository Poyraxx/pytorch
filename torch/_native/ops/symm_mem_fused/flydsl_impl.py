"""FlyDSL override for symm_mem's fused all-gather + matmul (AsyncTP) on gfx950.

Opt-in: nothing is registered unless TORCH_SYMM_MEM_FLYDSL=1 is set before `import torch`. A call goes to FlyDSL
when its predicate admits it (bf16, 2D, one B, gather dim 0, the kernel's tile multiples, TP 2/4/8); anything else
falls through to the op's own implementation.

The predicates (_common.py, which imports nothing but torch) read only shapes, strides, dtypes, devices and the group
size, so every rank of a group takes the same path as long as every rank has the same overrides registered: the
FlyDSL kernels rendezvous on their first call at each shape, and a rank that fell through instead would leave the
others waiting. symm_mem._test_mode() turns the overrides off, as it does the stock kernels.

The op is in flydsl_ag.py; its kernel (flydsl_kernels.py) is built on torch's vendored FlyDSL GEMM
(torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py).
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import collections
import importlib
import os

import torch

from ... import flydsl_utils as fu, triton_utils as tu
from . import _common


_ENV = "TORCH_SYMM_MEM_FLYDSL"
_SUPPORTED_ARCHES = ("gfx950",)

# Calls per (op, path), path "flydsl"; read by tests.
_calls: collections.Counter = collections.Counter()


def _kernels(name: str):
    # Imported on the first admitted call: the kernel modules pull in flydsl and triton, which must stay out of
    # `import torch` and out of the routing check.
    return importlib.import_module(f".{name}", __package__)


def _admissible(t: torch.Tensor) -> bool:
    import torch.distributed._symmetric_memory as symm_mem

    # symm_mem._test_mode() sends the symm_mem ops to their fallbacks; the stock kernels check it first too.
    return (
        not symm_mem._is_test_mode
        and t.device.type == "cuda"
        and fu._is_supported_arch(t.device.index, _SUPPORTED_ARCHES)
    )


def _world(group_name) -> int:
    from torch.distributed.distributed_c10d import _resolve_process_group

    return _resolve_process_group(group_name).size()


def _ag_cond(A_shard, Bs, gather_dim, group_name, *, return_A=True) -> bool:
    if not _admissible(A_shard):
        return False
    return _common.ag_supported(A_shard, Bs, gather_dim, _world(group_name)) is not None


def _ag_impl(A_shard, Bs, gather_dim, group_name, *, return_A=True):
    _calls["ag", "flydsl"] += 1
    return _kernels("flydsl_ag").run(
        A_shard, Bs, gather_dim, group_name, return_A=return_A
    )


# The symm_mem ops exist only once `_DEFINING_MODULE` has run; test_override_declarations.py resolves these symbols.
# Builds without distributed have no such module, so they declare nothing.
_NAMESPACE = "symm_mem" if torch.distributed.is_available() else None
_DEFINING_MODULE = "torch.distributed._symmetric_memory"

# (op_symbol, cond, impl) for every override this module installs.
_OVERRIDES = (("fused_all_gather_matmul", _ag_cond, _ag_impl),)


def register_flydsl_symm_mem_overrides() -> None:
    # Bail out before importing the defining module whenever fu.register_op_override would drop the registration
    # anyway. Don't gate on torch.cuda.is_available() here: it calls cuInit and poisons fork.
    if (
        os.environ.get(_ENV) != "1"
        or _NAMESPACE is None
        or not fu.runtime_available()
        or not tu.runtime_available()
        or fu.check_native_jit_disabled()
        or not fu._version_is_ok()
    ):
        return

    importlib.import_module(_DEFINING_MODULE)

    for op_symbol, cond, impl in _OVERRIDES:
        fu.register_op_override(_NAMESPACE, op_symbol, "CUDA", cond=cond, impl=impl)
