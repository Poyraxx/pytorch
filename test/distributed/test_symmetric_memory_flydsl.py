# Owner(s): ["oncall: distributed"]
"""FlyDSL overrides of symm_mem::fused_all_gather_matmul / fused_matmul_reduce_scatter (torch/_native/ops/symm_mem_fused).

TestRegistration needs no GPU and no FlyDSL: the opt-in gate, the predicates at every TP, operand alignment, the
timeout setting and the ranks' agreement check. FlyAsyncTPTest follows AsyncTPTest (test_symmetric_memory.py) at
shapes the predicates admit, checking each call against PyTorch's reference (_fused_*_fallback), including repeated
calls on one state and CUDA graph replay, plus calls the predicates reject, symm_mem's test mode and the registry's
off switch. It needs FlyDSL and gfx950; world size = visible GPUs (2, 4 or 8).

  python test/distributed/test_symmetric_memory_flydsl.py
"""

import os


os.environ.setdefault("TORCH_SYMM_MEM_FLYDSL", "1")

import collections
import subprocess
import sys
import types
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch._native import flydsl_utils as fu, triton_utils as tu
from torch._native.ops.symm_mem_fused import _common, flydsl_impl
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    isRocmArchAnyOf,
    MI350_ARCH,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


requires_flydsl = skip_but_pass_in_sandcastle_if(
    not (fu.runtime_available() and fu._version_is_ok()),
    "FlyDSL is not installed",
)
requires_symm_mem = skip_but_pass_in_sandcastle_if(
    torch.version.hip is None or not PLATFORM_SUPPORTS_SYMM_MEM,
    "ROCm symmetric memory not available",
)

BF = torch.bfloat16
# (M, K, N, B layout), admitted at TP 2 / 4 / 8
AG_SHAPES = {
    "k384_n768_col": (8192, 384, 768, "col"),
    "k1152_n1280_row": (16384, 1152, 1280, "row"),
}
RS_SHAPES = {
    "row": {
        8: (2048, 128, 1024, "row"),
        4: (4096, 128, 256, "row"),
        2: (8192, 128, 256, "row"),
    },
    "col": {
        8: (4096, 640, 3072, "col"),
        4: (8192, 640, 768, "col"),
        2: (8192, 640, 768, "col"),
    },
}


def _tol(ref):
    # The op's own path multiplies per chunk, the reference in one mm: fp32 results differ in accumulation order.
    return {"rtol": 1e-5, "atol": 1e-3} if ref.dtype == torch.float32 else {}


_PROBE = (
    "import torch; "
    "print(torch.backends.python_native.get_dsl_operations('flydsl', lib_symbol='symm_mem'), "
    "torch.cuda.is_initialized())"
)


def _meta(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _meta_weight(k, n, lay):
    return _meta(n, k).t() if lay == "col" else _meta(k, n)


class TestRegistration(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @unittest.skipIf(not dist.is_available(), "torch.distributed is not available")
    def test_gate(self):
        """Registers only with TORCH_SYMM_MEM_FLYDSL=1, FlyDSL and triton installed and native JIT allowed."""
        registered = []

        def gate(env, triton=True, jit_disabled=False):
            registered.clear()
            with (
                mock.patch.dict(os.environ),
                mock.patch.object(fu, "runtime_available", return_value=True),
                mock.patch.object(fu, "_version_is_ok", return_value=True),
                mock.patch.object(
                    fu, "check_native_jit_disabled", return_value=jit_disabled
                ),
                mock.patch.object(tu, "runtime_available", return_value=triton),
                mock.patch.object(
                    fu,
                    "register_op_override",
                    side_effect=lambda ns, op, key, **kw: registered.append(
                        (ns, op, key)
                    ),
                ),
            ):
                os.environ.pop("TORCH_SYMM_MEM_FLYDSL", None)
                if env is not None:
                    os.environ["TORCH_SYMM_MEM_FLYDSL"] = env
                flydsl_impl.register_flydsl_symm_mem_overrides()
            return list(registered)

        self.assertEqual(gate(None), [])
        self.assertEqual(gate("0"), [])
        self.assertEqual(gate("1", triton=False), [])
        self.assertEqual(gate("1", jit_disabled=True), [])
        self.assertEqual(
            gate("1"),
            [
                ("symm_mem", "fused_all_gather_matmul", "CUDA"),
                ("symm_mem", "fused_matmul_reduce_scatter", "CUDA"),
            ],
        )

    @requires_flydsl
    def test_opt_in_registers_without_initializing_cuda(self):
        env = dict(os.environ, TORCH_SYMM_MEM_FLYDSL="1")
        out = subprocess.check_output(
            [sys.executable, "-c", _PROBE], env=env, text=True
        )
        self.assertEqual(
            out.strip().splitlines()[-1],
            "['fused_all_gather_matmul', 'fused_matmul_reduce_scatter'] False",
        )

    def test_predicate(self):
        """What the kernel takes, at every TP; the group size is the predicate's argument, so no GPU or group."""
        ok = _common.ag_supported
        m, k, n, lay = AG_SHAPES["k384_n768_col"]
        for w in (2, 4, 8):
            lm = m // w
            A, B = _meta(lm, k), _meta_weight(k, n, lay)
            self.assertEqual(ok(A, [B], 0, w), (k, n, True), f"TP {w}")
            self.assertEqual(
                ok(A, [_meta_weight(k, n, "row")], 0, w), (k, n, False), f"TP {w}"
            )
            for what, args in (
                ("fp32", (_meta(lm, k, dtype=torch.float32), [B], 0)),
                ("gather dim 1", (A, [B], 1)),
                ("two Bs", (A, [B, B], 0)),
                ("K % 128", (_meta(lm, k + 64), [_meta_weight(k + 64, n, lay)], 0)),
                ("M 0", (_meta(0, k), [B], 0)),
                ("N 0", (A, [_meta_weight(k, 0, "col")], 0)),
                ("K 0", (_meta(lm, 0), [_meta_weight(0, n, "row")], 0)),
                ("B on another device", (A, [torch.empty(k, n, dtype=BF)], 0)),
                ("A not contiguous", (_meta(k, lm).t(), [B], 0)),
            ):
                self.assertIsNone(ok(*args, w), f"TP {w}: {what}")
            self.assertIsNone(ok(A, [B], 0, 3), "TP 3")
            for cols, admitted in (
                (_common.AG_MAX_N, True),
                (_common.AG_MAX_N + 256, False),
            ):
                self.assertEqual(
                    ok(A, [_meta_weight(k, cols, lay)], 0, w) is not None,
                    admitted,
                    f"TP {w}: N {cols}",
                )
            # 32-bit bounds: the gathered A and the output are addressed per tile, so 4 GiB of output is fine; B
            # (one descriptor over K x N) and the shard (_stage3's int32 offsets) are rejected at 4 GiB.
            rows = 2**32 // (2 * 16384)
            self.assertIsNotNone(
                ok(_meta(rows // w, 128), [_meta_weight(128, 16384, "col")], 0, w),
                f"TP {w}: 4 GiB output",
            )
            for kk, admitted in ((2**17 - 128, True), (2**17, False)):
                self.assertEqual(
                    ok(_meta(8192 // w, kk), [_meta_weight(kk, 16384, "col")], 0, w)
                    is not None,
                    admitted,
                    f"TP {w}: B {kk} x 16384",
                )
            for rows, admitted in ((2**23, True), (2**24, False)):
                self.assertEqual(
                    ok(_meta(rows, 128), [_meta_weight(128, n, "col")], 0, w)
                    is not None,
                    admitted,
                    f"TP {w}: shard {rows} x 128",
                )

    def test_rs_predicate(self):
        ok = _common.rs_supported
        for w in (2, 4, 8):
            m, k, n, lay = RS_SHAPES["col"][w]
            A, B = _meta(m, k), _meta_weight(k, n, lay)
            self.assertEqual(ok(A, B, "sum", 0, w), (k, n, True), f"TP {w}")
            m, k, n, lay = RS_SHAPES["row"][w]
            self.assertEqual(
                ok(_meta(m, k), _meta_weight(k, n, lay), "sum", 0, w),
                (k, n, False),
                f"TP {w}",
            )
            m, k, n, lay = RS_SHAPES["col"][w]
            for what, args in (
                ("fp32", (_meta(m, k, dtype=torch.float32), B, "sum", 0)),
                ("scatter dim 1", (A, B, "sum", 1)),
                ("avg", (A, B, "avg", 0)),
                ("K % 128", (_meta(m, k + 64), _meta_weight(k + 64, n, lay), "sum", 0)),
                ("M 0", (_meta(0, k), B, "sum", 0)),
                ("N 0", (A, _meta_weight(k, 0, "col"), "sum", 0)),
                ("K 0", (_meta(m, 0), _meta_weight(0, n, "row"), "sum", 0)),
                ("B on another device", (A, torch.empty(k, n, dtype=BF), "sum", 0)),
                ("A not contiguous", (_meta(k, m).t(), B, "sum", 0)),
            ):
                self.assertIsNone(ok(*args, w), f"TP {w}: {what}")
            self.assertIsNone(ok(A, B, "sum", 0, 3), "TP 3")
            # 32-bit bounds: A and the staging are addressed per tile; B and the output chunk (M / TP x N) are
            # rejected at 4 GiB.
            rows, B = 2**32 // (2 * 65536), _meta_weight(128, 65536, "col")
            self.assertIsNotNone(
                ok(_meta(rows, 128), B, "sum", 0, w), f"TP {w}: 4 GiB output"
            )
            self.assertIsNone(
                ok(_meta(rows * w, 128), B, "sum", 0, w), f"TP {w}: 4 GiB chunk"
            )
            A = _meta(8192, 32768)
            for cols, admitted in ((65536 - 1024, True), (65536, False)):
                self.assertEqual(
                    ok(A, _meta_weight(32768, cols, "col"), "sum", 0, w) is not None,
                    admitted,
                    f"TP {w}: B 32768 x {cols}",
                )

    def test_aligned(self):
        """Misaligned operands are copied to an aligned address with the same strides; aligned ones are not."""
        a = torch.rand(32 * 32 + 1, dtype=BF)[1:].view(32, 32)
        w = torch.rand(64 * 32 + 1, dtype=BF)[1:].view(64, 32).t()
        for t in (a, w):
            self.assertNotEqual(t.data_ptr() % 16, 0)
            c = _common.aligned(t)
            self.assertEqual(c.data_ptr() % 16, 0)
            self.assertEqual(c.stride(), t.stride())
            self.assertEqual(c, t, atol=0, rtol=0)
        t = torch.rand(32, 32, dtype=BF)
        self.assertIs(_common.aligned(t), t)

    def test_timeout_env(self):
        env = _common.TIMEOUT_ENV
        for raw, ms in ((None, 60000), ("0", 0), ("1500", 1500)):
            with mock.patch.dict(os.environ):
                os.environ.pop(env, None)
                if raw is not None:
                    os.environ[env] = raw
                self.assertEqual(_common.wait_ms(), ms)
        for raw in ("abc", "-1", str(2**31)):
            with mock.patch.dict(os.environ, {env: raw}):
                with self.assertRaisesRegex(ValueError, env):
                    _common.wait_ms()

    @unittest.skipIf(not dist.is_available(), "torch.distributed is not available")
    def test_group_agreement(self):
        """A peer that never builds the state, or builds a different one, raises instead of hanging the rendezvous."""
        store = dist.HashStore()
        store.set_timeout(timedelta(milliseconds=200))
        group = types.SimpleNamespace(
            group_name="flydsl_test", rank=lambda: 0, size=lambda: 2
        )
        with (
            mock.patch.object(_common, "_states_built", collections.Counter()),
            mock.patch(
                "torch.distributed.distributed_c10d._get_process_group_store",
                return_value=store,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "TORCH_SYMM_MEM_FLYDSL"):
                _common.check_group_agrees("op", group, "state a")
            store.set("symm_mem_flydsl/1/1", "state b")
            with self.assertRaisesRegex(RuntimeError, "different states"):
                _common.check_group_agrees("op", group, "state a")
            store.set("symm_mem_flydsl/2/1", "state a")
            _common.check_group_agrees("op", group, "state a")


def weight(k, n, lay):
    """B [K, N]: W.t() (strides (1, K)) or row-major, as TorchTitan passes it."""
    return (
        torch.rand(n, k, dtype=BF, device="cuda").t()
        if lay == "col"
        else torch.rand(k, n, dtype=BF, device="cuda")
    )


@requires_symm_mem
@requires_flydsl
@instantiate_parametrized_tests
class FlyAsyncTPTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        if not isRocmArchAnyOf(MI350_ARCH):
            raise unittest.SkipTest(f"the FlyDSL kernels run on {MI350_ARCH} only")
        if self.world_size not in _common.WORLDS:
            raise unittest.SkipTest(f"the FlyDSL kernels take TP {_common.WORLDS}")
        if (
            "fused_all_gather_matmul"
            not in torch.backends.python_native.get_dsl_operations(
                "flydsl", lib_symbol="symm_mem"
            )
        ):
            raise unittest.SkipTest(
                "the FlyDSL symm_mem overrides are not registered (TORCH_SYMM_MEM_FLYDSL, TORCH_DISABLE_NATIVE_JIT)"
            )
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("warn")
        torch.utils.deterministic.fill_uninitialized_memory = True

    @staticmethod
    def _calls(op):
        return flydsl_impl._calls[op, "flydsl"]

    def _routed(self, op, before, flydsl):
        msg = (
            "the call did not go to FlyDSL"
            if flydsl
            else "FlyDSL took a call its predicate should reject"
        )
        self.assertEqual(self._calls(op), before + int(flydsl), msg)

    def _ag(self, A_shard, B, return_A, flydsl=True, gather_dim=0):
        before = self._calls("ag")
        out = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard,
            [B],
            gather_dim=gather_dim,
            group_name=dist.group.WORLD.group_name,
            return_A=return_A,
        )
        self._routed("ag", before, flydsl)
        return out

    def _check_ag(self, A_shard, B, ag, mm, return_A, gather_dim=0):
        from torch.distributed._symmetric_memory import (
            _fused_all_gather_matmul_fallback,
        )

        ag0, mm0 = _fused_all_gather_matmul_fallback(
            A_shard,
            [B],
            gather_dim=gather_dim,
            group_name=dist.group.WORLD.group_name,
            return_A=True,
        )
        if return_A:
            torch.testing.assert_close(ag, ag0, rtol=0, atol=0)
            self.assertEqual(ag.stride(), ag0.stride())
        else:
            self.assertIsNone(ag)
        torch.testing.assert_close(mm[0], mm0[0], **_tol(mm0[0]))
        self.assertEqual(mm[0].stride(), mm0[0].stride())

    def _rs(self, A, B, flydsl=True, scatter_dim=0):
        before = self._calls("rs")
        out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "sum", scatter_dim, dist.group.WORLD.group_name
        )
        self._routed("rs", before, flydsl)
        return out

    def _check_rs(self, A, B, out, scatter_dim=0):
        from torch.distributed._symmetric_memory import (
            _fused_matmul_reduce_scatter_fallback,
        )

        out0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "sum", scatter_dim=scatter_dim, group_name=dist.group.WORLD.group_name
        )
        torch.testing.assert_close(out, out0, **_tol(out0))
        self.assertEqual(out.stride(), out0.stride())

    def _inputs(self, shape):
        m, k, n, lay = AG_SHAPES[shape]
        return (
            torch.rand(m // self.world_size, k, dtype=BF, device="cuda"),
            weight(k, n, lay),
        )

    def _rs_inputs(self, layout):
        m, k, n, lay = RS_SHAPES[layout][self.world_size]
        return torch.rand(m, k, dtype=BF, device="cuda"), weight(k, n, lay)

    @skip_if_lt_x_gpu(2)
    @parametrize("shape", list(AG_SHAPES))
    @parametrize("return_A", [True, False])
    def test_fused_all_gather_matmul(self, shape: str, return_A: bool) -> None:
        self._init_process()
        A_shard, B = self._inputs(shape)
        ag, mm = self._ag(A_shard, B, return_A)
        self._check_ag(A_shard, B, ag, mm, return_A)

    @skip_if_lt_x_gpu(2)
    @parametrize("layout", list(RS_SHAPES))
    def test_fused_matmul_reduce_scatter(self, layout: str) -> None:
        self._init_process()
        A, B = self._rs_inputs(layout)
        self._check_rs(A, B, self._rs(A, B))

    @skip_if_lt_x_gpu(2)
    def test_repeated_calls(self) -> None:
        """Calls on one state reuse its staging (AG: call 3 writes call 1's half) and accumulate its epoch, flags
        and counters."""
        self._init_process()
        A_shard, B = self._inputs("k1152_n1280_row")
        A, Br = self._rs_inputs("row")
        for i in range(4):
            A_shard.copy_(torch.rand_like(A_shard))
            ag, mm = self._ag(A_shard, B, i % 2 == 0)
            self._check_ag(A_shard, B, ag, mm, i % 2 == 0)
            A.copy_(torch.rand_like(A))
            self._check_rs(A, Br, self._rs(A, Br))

    @skip_if_lt_x_gpu(2)
    def test_graph_capture(self) -> None:
        """One eager call per op builds the states; then both ops in one CUDA graph, replayed with new inputs."""
        self._init_process()
        A_shard, Bg = self._inputs("k384_n768_col")
        A, Br = self._rs_inputs("col")
        self._ag(A_shard, Bg, True)
        self._rs(A, Br)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ag, mm = self._ag(A_shard, Bg, True)
            out = self._rs(A, Br)
        for _ in range(3):
            for t in (A_shard, Bg, A, Br):
                t.copy_(torch.rand_like(t))
            graph.replay()
            torch.cuda.synchronize()
            self._check_ag(A_shard, Bg, ag, mm, True)
            self._check_rs(A, Br, out)

    @skip_if_lt_x_gpu(2)
    def test_misaligned_operands(self) -> None:
        """A at an 8-byte offset is admitted (alignment can differ between ranks) and copied before the kernel."""
        self._init_process()

        def offset(t):
            buf = torch.empty(t.numel() + 4, dtype=t.dtype, device=t.device)
            moved = buf[4:].view_as(t)
            moved.copy_(t)
            self.assertNotEqual(moved.data_ptr() % 16, 0)
            return moved

        A_shard, B = self._inputs("k384_n768_col")
        A_shard = offset(A_shard)
        ag, mm = self._ag(A_shard, B, True)
        self._check_ag(A_shard, B, ag, mm, True)
        A, B = self._rs_inputs("col")
        A = offset(A)
        self._check_rs(A, B, self._rs(A, B))

    @skip_if_lt_x_gpu(2)
    def test_rejected_calls_fall_through(self) -> None:
        """fp32, a K the tiles don't divide, an N per rank above AG_MAX_N, and gather / scatter dim 1 go to the op's own
        implementation."""
        self._init_process()
        m, k, n, lay = AG_SHAPES["k384_n768_col"]
        for A_shard, B in (
            (
                torch.rand(m // self.world_size, k, device="cuda"),
                torch.rand(k, n, device="cuda"),
            ),
            (
                torch.rand(m // self.world_size, k + 64, dtype=BF, device="cuda"),
                weight(k + 64, n, lay),
            ),
            (
                torch.rand(m // self.world_size, k, dtype=BF, device="cuda"),
                weight(k, _common.AG_MAX_N + 256, lay),
            ),
        ):
            ag, mm = self._ag(A_shard, B, True, flydsl=False)
            self._check_ag(A_shard, B, ag, mm, True)
        A_shard, B = torch.rand(4, 512, k, dtype=BF, device="cuda"), weight(k, n, lay)
        ag, mm = self._ag(A_shard, B, True, flydsl=False, gather_dim=1)
        self._check_ag(A_shard, B, ag, mm, True, gather_dim=1)
        m, k, n, lay = RS_SHAPES["col"][self.world_size]
        A, B = torch.rand(m, k, device="cuda"), torch.rand(k, n, device="cuda")
        self._check_rs(A, B, self._rs(A, B, flydsl=False))
        A, B = self._rs_inputs("col")
        self._check_rs(A, B, self._rs(A, B, flydsl=False, scatter_dim=1), scatter_dim=1)

    @skip_if_lt_x_gpu(2)
    def test_test_mode(self) -> None:
        """Under symm_mem._test_mode() an admitted call takes the op's fallback, as the stock kernel does."""
        self._init_process()
        import torch.distributed._symmetric_memory as symm_mem

        A_shard, B = self._inputs("k384_n768_col")
        A, Br = self._rs_inputs("col")
        with symm_mem._test_mode():
            ag, mm = self._ag(A_shard, B, True, flydsl=False)
            out = self._rs(A, Br, flydsl=False)
        self._check_ag(A_shard, B, ag, mm, True)
        self._check_rs(A, Br, out)

    @skip_if_lt_x_gpu(2)
    def test_deregistered(self) -> None:
        """With the FlyDSL overrides deregistered an admitted call takes the op's own path; re-enabling restores them."""
        self._init_process()
        A_shard, B = self._inputs("k384_n768_col")
        fu.deregister_op_overrides()
        try:
            ag, mm = self._ag(A_shard, B, True, flydsl=False)
            self._check_ag(A_shard, B, ag, mm, True)
        finally:
            torch._native.registry.reenable_op_overrides(enable_dsl_names="flydsl")
        ag, mm = self._ag(A_shard, B, True)
        self._check_ag(A_shard, B, ag, mm, True)


if __name__ == "__main__":
    run_tests()
