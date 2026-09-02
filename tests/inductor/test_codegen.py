# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import regex as re
import sympy
import torch
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_code,
)
from torch._inductor.virtualized import V
from torch.testing import FileCheck

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.compute_ops import (
    SymbolKind,
    _per_core_symbolic_dim_info,
    _symbolic_split_info,
    _tensor_has_symbolic_split,
)
from torch_spyre._inductor.codegen.superdsc import (
    _align_matmul_dim_labels,
    _align_pool_dim_labels,
    _matmul_role_shapes,
    _resolve_sdsc_size,
    compile_op_spec,
    parse_op_spec,
)
from torch_spyre._inductor.constants import MATMUL_OPERANDS_INFO_KEY
from torch_spyre._inductor.core_mapping import derive_operation_mapping
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.lowering import lower_bmm, lower_mm, lower_scaled_mm
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.op_spec_validation import validate_op_specs
from torch_spyre._inductor.spyre_kernel import (
    _matmul_destination_shapes,
    simplify_op_spec,
)
from torch_spyre._inductor.work_division import (
    _collect_symbol_metadata,
    _effective_size,
    _valid_divisor_basis,
    adjust_it_space_for_sticks,
)


def _total_core_split(source: str) -> int:
    """Return the product of split factors in the emitted ``iteration_space``.

    Co-optimization spreads the work division across several ``c`` dims rather
    than loading all cores onto ``c0``, so the total core usage is the product
    of the per-dim split factors (e.g. ``c0:(256, 2), c1:(128, 4), c2:(512, 4)``
    uses ``2 * 4 * 4 == 32`` cores).
    """
    match = re.search(r"iteration_space=\{([^}]*)\}", source)
    assert match, "no iteration_space found in emitted source"
    factors = [
        int(f) for f in re.findall(r"sympify\('\d+'\),\s*(\d+)\)", match.group(1))
    ]
    assert factors, f"no split factors found in {match.group(1)!r}"
    product = 1
    for f in factors:
        product *= f
    return product


class TestSpyreConfig(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_config_default(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        comp_fn = torch.compile(fn)
        out, source_codes = run_and_get_code(comp_fn, x)
        # print("test_config_default")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").run(source_codes[0])
        # Co-optimization spreads the split across dims; the product of the
        # per-dim split factors must add up to the configured core count.
        self.assertEqual(_total_core_split(source_codes[0]), config.sencores)

    @config.patch({"sencores": 64})
    def test_config_too_many_sencores(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        with self.assertRaisesRegex(
            InductorError,
            "Unsupported: Spyre backend does not support: invalid SENCORES value 64",
        ):
            comp_fn = torch.compile(fn)
            comp_fn(x)

    @config.patch({"sencores": 16})
    def test_sencores_16(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")
        cfn = torch.compile(fn)
        out, source_codes = run_and_get_code(cfn, x)
        # print("test_sencores 16")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").run(source_codes[0])
        # Co-optimization spreads the split across dims; the product of the
        # per-dim split factors must add up to the configured core count.
        self.assertEqual(_total_core_split(source_codes[0]), config.sencores)

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_pointwise_split(self):
        """Symbolic batch dim must split by ``granularity`` not ``max_size`` (#2287).

        ``[s, 128]`` fp16 with ``s in [64, 1024]`` (granularity = 64). The planner picks the largest
        divisor of granularity ≤ SENCORES = 32, so the batch dim absorbs all
        32 cores and the static stick dim gets split 1.
        """
        fn = torch.add
        x = torch.randn((1024, 128), dtype=torch.float16)
        y = torch.randn_like(x)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        torch._dynamo.mark_dynamic(y, 0, min=64, max=1024)
        # dynamic=True not needed: mark_dynamic already makes dim 0 symbolic.
        comp_fn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"), y.to("spyre"))
        # Iteration space embeds (size_expr, split). The symbolic batch dim's
        # split must equal SENCORES=32; the static stick dim's split must be 1.
        FileCheck().check("sdsc_fused_add").run(source_codes[0])
        # Co-optimization spreads the split across dims; the product of the
        # per-dim split factors must add up to the configured core count.
        self.assertEqual(_total_core_split(source_codes[0]), config.sencores)

    # ------------------------------------------------------------------
    # Unit tests for the symbolic-shape sidecar in work_division.py
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_v(lower=None, upper=None, optimization_hint=None):
        """
        Mock V whose ShapeEnv reports the given lower / upper bounds.
        """
        shape_env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=lower, upper=upper)
        )
        sizevars = SimpleNamespace(shape_env=shape_env)
        if optimization_hint is not None:
            sizevars.optimization_hint = lambda _e: optimization_hint
        return SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))

    def test_collect_symbol_metadata_opt_in(self):
        """
        User-marked dynamic dim (finite max) enters the metadata dict.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch(
                "torch_spyre._inductor.pass_utils.V",
                self._mock_v(lower=sympy.Integer(2), upper=sympy.Integer(512)),
            ):
                result = _collect_symbol_metadata({s0: s0})
        # max comes straight from the ShapeEnv upper bound;
        # granularity is the smallest divisor of 512 with d >= 4 and
        # 512/d <= 32, which is 16.
        self.assertEqual(result, {s0: (512, 16)})

    def test_collect_symbol_metadata_auto_dynamic_skipped(self):
        """
        Dynamo-promoted symbols (no finite max) are skipped, not assigned.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with patch(
            "torch_spyre._inductor.pass_utils.V",
            self._mock_v(
                lower=sympy.Integer(2), upper=sympy.oo, optimization_hint=1024
            ),
        ):
            self.assertEqual(_collect_symbol_metadata({s0: s0}), {})

    def test_dispatch_helpers_symbolic_vs_concrete(self):
        """
        ``_effective_size`` and ``_valid_divisor_basis`` dispatch on ``v in meta``.
        """
        s0 = sympy.Symbol("s0")
        it_space = {s0: sympy.Integer(128)}
        meta = {s0: (512, 16)}
        # In meta: use the (max, granularity) tuple.
        self.assertEqual(_effective_size(s0, it_space, meta), 512)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta), 16)
        # Not in meta: fall through to concretize_expr(it_space[v]).
        self.assertEqual(_effective_size(s0, it_space, meta={}), 128)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta={}), 128)

    def test_symbolic_stick_dim_raises_unsupported(self):
        """
        A symbolic dim that lands on a tensor's stick coord is rejected.
        This is a follow up work.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        # Minimal TensorDep stand-in: the function only reads
        # device_coords[-1], dep.name, and layout.device_layout.elems_per_stick().
        fake_td = SimpleNamespace(
            dep=SimpleNamespace(name="fake_buf"),
            layout=SimpleNamespace(
                device_layout=SimpleNamespace(elems_per_stick=lambda: 64)
            ),
            device_coords=[s0],
        )
        with self.assertRaises(Unsupported) as cm:
            adjust_it_space_for_sticks(
                {s0: sympy.Integer(128)}, [fake_td], {s0: (512, 64)}
            )
        self.assertIn("symbolic stick dim", str(cm.exception))

    def test_inplace_op_run_call_deduplicates_args(self):
        """An inplace op (x *= 2) must not pass the same tensor twice to .run().

        With symbolic args, the MLIR bundle emits one input_arg param per unique
        tensor.  Passing arg0_1 twice would cause a "Number of inputs mismatches"
        error at launch time.  This test verifies the generated .run() call
        contains no duplicate tensor arguments.
        """

        def fn(x):
            x *= 2
            return x

        x = torch.randn((4, 128), dtype=torch.float16, device="spyre")
        cfn = torch.compile(fn)
        _, source_codes = run_and_get_code(cfn, x)
        code = source_codes[0]
        # Find the .run(...) call for the fused kernel
        run_lines = [ln.strip() for ln in code.splitlines() if ".run(" in ln]
        self.assertTrue(run_lines, "No .run(...) call found in generated code")
        for line in run_lines:
            # Extract the argument list between the outermost parentheses
            args_str = line[line.index("(") + 1 : line.rindex(")")]
            args = [a.strip() for a in args_str.split(",")]
            self.assertEqual(
                len(args),
                len(set(args)),
                f"Duplicate args in .run() call: {line}",
            )


class TestResolveSdscSize(InductorTestCase):
    """Unit tests for superdsc._resolve_sdsc_size."""

    def test_concrete_sympy_integer(self):
        self.assertEqual(_resolve_sdsc_size(sympy.Integer(256), {}), 256)

    def test_concrete_python_int(self):
        self.assertEqual(_resolve_sdsc_size(128, {}), 128)

    def test_symbolic_in_bounds_returns_max(self):
        # bounds carries (max, granularity); index [0] is max.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        self.assertEqual(_resolve_sdsc_size(s0, {"s0": (1024, 64)}), 1024)

    def test_symbolic_not_in_bounds_uses_guarding_hint(self):
        # Symbol absent from bounds → _concretize_for_sdsc → guarding_hint_or_throw.
        # The SDSC/DeepTools boundary is correctness-critical, so it must resolve
        # the *true* concrete size (guarding_hint_or_throw), not an optimization
        # heuristic that could silently emit a fallback (e.g. sys.maxsize).
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        sizevars = SimpleNamespace(guarding_hint_or_throw=lambda _: 128)
        mock_v = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch("torch_spyre._inductor.codegen.superdsc.V", mock_v):
            self.assertEqual(_resolve_sdsc_size(s0, {}), 128)

    def test_symbolic_not_in_bounds_raises_on_unbacked(self):
        # An unbacked symbol at the SDSC boundary must fail loudly rather than
        # produce a bogus concrete size, so guarding_hint_or_throw's raise
        # propagates out of _concretize_for_sdsc.
        s0 = sympy.Symbol("s0", integer=True, positive=True)

        def _raise(_e):
            raise RuntimeError("unbacked symbol at SDSC boundary")

        sizevars = SimpleNamespace(guarding_hint_or_throw=_raise)
        mock_v = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch("torch_spyre._inductor.codegen.superdsc.V", mock_v):
            with self.assertRaises(RuntimeError):
                _resolve_sdsc_size(s0, {})


class TestAlignPoolDimLabels(InductorTestCase):
    """Unit tests for superdsc._align_pool_dim_labels.

    Survival of each pool dim role is derived from the node's live NCHW output
    ranges [N, C, H_out, W_out]; statically size-1 output dims are dropped from
    the emitted (NHWC) label list.  Window dims always survive (kH>1, kW>1 is
    guaranteed by the lowering delegation guard).  These cases mirror the shapes
    exercised by the hardware-only test_avg_pool2d_base.
    """

    def test_all_dims_present(self):
        # N=2, C=3, H_out=W_out=8 -> every role survives.
        labels = _align_pool_dim_labels((2, 3, 8, 8), 6)
        self.assertEqual(labels, ["mb", "i", "j", "out", "ki", "kj"])

    def test_batch_dropped(self):
        # N=1 -> "mb" filtered out; iteration space rank 5.
        labels = _align_pool_dim_labels((1, 3, 8, 8), 5)
        self.assertEqual(labels, ["i", "j", "out", "ki", "kj"])

    def test_channel_dropped(self):
        # C=1 -> "out" filtered out; iteration space rank 5.
        labels = _align_pool_dim_labels((2, 1, 8, 8), 5)
        self.assertEqual(labels, ["mb", "i", "j", "ki", "kj"])

    def test_batch_and_channel_dropped(self):
        # N=1 and C=1 -> both "mb" and "out" filtered out; rank 4.
        labels = _align_pool_dim_labels((1, 1, 8, 8), 4)
        self.assertEqual(labels, ["i", "j", "ki", "kj"])

    def test_symbolic_dims_never_dropped(self):
        # A symbolic (dynamic) output dim must not be treated as size-1.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        labels = _align_pool_dim_labels((s0, 3, 8, 8), 6)
        self.assertEqual(labels, ["mb", "i", "j", "out", "ki", "kj"])

    def test_rank_mismatch_raises(self):
        # Label count disagreeing with the reported iteration-space rank is a
        # loud error rather than silent wrong-code.
        with self.assertRaises(ValueError):
            _align_pool_dim_labels((2, 3, 8, 8), 5)

    def test_missing_ranges_raises(self):
        with self.assertRaises(ValueError):
            _align_pool_dim_labels(None, 6)

    def test_wrong_rank_ranges_raises(self):
        with self.assertRaises(ValueError):
            _align_pool_dim_labels((2, 3, 8), 5)


class TestSymbolKindDimension(InductorTestCase):
    """Unit tests for the dimension variant added to compute_ops.SymbolKind."""

    def test_factory_sets_all_fields(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.kind, "dimension")
        self.assertEqual(sk.granularity, 64)
        self.assertEqual(sk.max_value, 1024)
        self.assertEqual(sk.pytorch_sym, "s0")

    def test_is_dimension_true(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertTrue(sk.is_dimension)

    def test_address_fields_are_sentinels(self):
        # Address-specific fields must not be set by the dimension factory so
        # they don't collide with kernel/pool symbol-table entries.
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.arg_index, -1)
        self.assertEqual(sk.base_sym_idx, -1)
        self.assertEqual(sk.offset, 0)

    def test_kernel_is_not_dimension(self):
        self.assertFalse(SymbolKind.kernel(arg_index=0).is_dimension)

    def test_pool_is_not_dimension(self):
        self.assertFalse(SymbolKind.pool().is_dimension)


class TestPerCoreSymbolicDimInfo(InductorTestCase):
    """Unit tests for compute_ops._per_core_symbolic_dim_info."""

    def test_no_symbolic_dims_returns_empty(self):
        self.assertEqual(_per_core_symbolic_dim_info({}, {}), {})

    def test_single_dim_no_split(self):
        # work_slices == 1 means undivided: maxSize_/granularity_ pass through.
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 1}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 1024, "granularity_": 64}},
        )

    def test_single_dim_split_across_cores(self):
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 256, "granularity_": 16}},
        )

    def test_granularity_floors_at_one(self):
        # granularity // wk_slices would floor to 0; result must clamp to 1
        # so the runtime never sees a zero batch-size granularity.
        symbolic_dims = {"c0": ("s0", 1, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        result = _per_core_symbolic_dim_info(symbolic_dims, work_slices)
        self.assertEqual(result["c0"], {"maxSize_": 256, "granularity_": 1})

    def test_multiple_symbolic_dims_independent(self):
        symbolic_dims = {
            "c0": ("s0", 64, 1024),
            "c1": ("s1", 32, 512),
        }
        work_slices = {
            sympy.Symbol("c0"): 4,
            sympy.Symbol("c1"): 2,
        }
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {
                "c0": {"maxSize_": 256, "granularity_": 16},
                "c1": {"maxSize_": 256, "granularity_": 16},
            },
        )


class TestSdscJsonSymbolicDimSmoke(InductorTestCase):
    """Smoke test: a symbolic iteration-space dim survives end-to-end through
    compile_op_spec (parse_op_spec + generate_sdsc) into the emitted SDSC
    JSON's dimToSymbolMapping_ / symbolicDimInfo_ fields.

    Fixture uses a [512, 256] fp16 stick-layout tensor with the row dim
    made symbolic. Because _resolve_sdsc_size resolves a symbolic dim to
    its declared max (512), every downstream computation (padding,
    stick-dim detection, core slicing) runs identically to the equivalent
    concrete case -- only the symbolic_dims side-channel asserted on here
    differs.
    """

    _DEVICE_SIZE = [4, 512, 64]
    _HBM_BASE = 0x400000000

    def _make_symbolic_op_spec(self) -> OpSpec:
        c_row, c_col = sympy.Symbol("c_row"), sympy.Symbol("c_col")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        coords = [c_col // 64, c_row, sympy.Mod(c_col, 64)]

        def _tensor_arg(is_input, arg_index, hbm_base):
            return TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=list(self._DEVICE_SIZE),
                device_coordinates=coords,
                allocation={"hbm": hbm_base},
            )

        iteration_space = {
            c_row: (s0, 1),
            c_col: (sympy.Integer(256), 1),
        }
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            args=[
                _tensor_arg(True, 0, self._HBM_BASE),
                _tensor_arg(True, 1, self._HBM_BASE + 0x1000),
                _tensor_arg(False, 2, self._HBM_BASE + 0x100000000),
            ],
            op_info={},
            symbolic_dim_bounds={"s0": (512, 64)},  # (max, granularity)
        )

    def test_symbolic_dim_fields_in_sdsc_json(self):
        op_spec = self._make_symbolic_op_spec()
        sdsc_json, _, _, _ = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])

        top = next(iter(sdsc_json.values()))
        dsc = next(iter(top["dscs_"][0].values()))

        # "s0" is registered as dim-symbol id -1 and bound to the SDSC "mb"
        # dim (c_row maps to the first non-output dim label for a 2-dim op).
        self.assertEqual(dsc["dimToSymbolMapping_"], {"mb": [-1]})

        for stage in ("ss_", "el_"):
            sym_info = dsc["dataStageParam_"]["0"][stage]["symbolicDimInfo_"]
            self.assertEqual(sym_info, {"mb": {"maxSize_": 512, "granularity_": 64}})


class TestMatmulLoweringMetadata(InductorTestCase):
    @staticmethod
    def _fake_tensor(name, shape, dtype, stride=None):
        def get_stride():
            if stride is None:
                raise NotImplementedError
            return stride

        return SimpleNamespace(
            name=name,
            shape=list(shape),
            realize=lambda: None,
            make_loader=lambda: lambda _indices: None,
            get_size=lambda: list(shape),
            get_stride=get_stride,
            get_dtype=lambda: dtype,
            get_device=lambda: torch.device("cpu"),
            get_name=lambda: name,
        )

    def _captured_reduction(self, lowering_fn, lhs, rhs):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(realize=lambda: None)

        with patch(
            "torch_spyre._inductor.lowering.SpyreReduction.create",
            side_effect=create,
        ):
            # Bypass Inductor's registration wrappers: they validate that the
            # mocked return is real IR, while this test intentionally captures
            # only the arguments passed to SpyreReduction.create.  lower_bmm
            # has two registration decorators, so unwrap to the original
            # implementation rather than peeling a single layer.
            while hasattr(lowering_fn, "__wrapped__"):
                lowering_fn = lowering_fn.__wrapped__
            lowering_fn(lhs, rhs)
        return captured

    def test_all_matmul_lowerings_preserve_ordered_logical_operands(self):
        symbolic_batch = sympy.Symbol("s0", integer=True, positive=True)
        cases = (
            (
                lower_mm,
                self._fake_tensor(
                    "lhs_mm",
                    (symbolic_batch, 4, 128),
                    torch.float16,
                    stride=(512, 128, 1),
                ),
                self._fake_tensor("rhs_mm", (128, 64), torch.float16),
                (symbolic_batch, 4, 64),
            ),
            (
                lower_bmm,
                self._fake_tensor(
                    "lhs_bmm", (2, 4, 128), torch.float16, stride=(512, 128, 1)
                ),
                self._fake_tensor(
                    "rhs_bmm", (2, 128, 64), torch.float16, stride=(8192, 64, 1)
                ),
                (2, 4, 64),
            ),
            (
                lower_scaled_mm,
                self._fake_tensor("lhs_fp8", (4, 128), torch.float8_e4m3fn),
                self._fake_tensor("rhs_fp8", (128, 64), torch.float8_e4m3fn),
                (4, 64),
            ),
        )

        for lowering_fn, lhs, rhs, output_shape in cases:
            with self.subTest(lowering=lowering_fn.__name__):
                captured = self._captured_reduction(lowering_fn, lhs, rhs)
                metadata = captured["op_info"][MATMUL_OPERANDS_INFO_KEY]
                self.assertEqual(
                    metadata["shapes"],
                    (tuple(lhs.shape), tuple(rhs.shape), output_shape),
                )
                self.assertEqual(
                    metadata["batch_dim_owners"],
                    (
                        (True,) * max(0, len(lhs.shape) - 2),
                        (True,) * max(0, len(rhs.shape) - 2),
                    ),
                )

    def test_lowering_rejects_inconsistent_matmul_contracts(self):
        invalid_cases = (
            (
                self._fake_tensor("lhs", (4, 64, 64, 128), torch.float16),
                self._fake_tensor("rhs", (1, 4, 128, 64), torch.float16),
            ),
            (
                self._fake_tensor("lhs", (2, 4, 128), torch.float16),
                self._fake_tensor("rhs", (2, 64, 32), torch.float16),
            ),
        )

        for lhs, rhs in invalid_cases:
            with (
                self.subTest(lhs=lhs.shape, rhs=rhs.shape),
                self.assertRaisesRegex(Unsupported, "invalid matmul contract"),
            ):
                self._captured_reduction(lower_bmm, lhs, rhs)

    def test_lowering_records_zero_stride_batch_ownership(self):
        lhs = self._fake_tensor(
            "lhs",
            (4, 67, 256),
            torch.float16,
            stride=(67 * 256, 256, 1),
        )
        rhs = self._fake_tensor(
            "rhs",
            (4, 256, 128),
            torch.float16,
            stride=(0, 128, 1),
        )

        captured = self._captured_reduction(lower_bmm, lhs, rhs)

        self.assertEqual(
            captured["op_info"][MATMUL_OPERANDS_INFO_KEY]["batch_dim_owners"],
            ((True,), (False,)),
        )

    def test_destination_shape_accounts_for_coarse_tiling(self):
        full_shape = (1, 32, 64, 64)
        ir_node = SimpleNamespace(
            loop_info=SimpleNamespace(
                loop_count=(2, 4),
                loop_tiled_dims=((1,), (1, 2)),
            )
        )

        self.assertEqual(
            _matmul_destination_shapes(full_shape, ir_node),
            ((1, 4, 16, 64),),
        )
        self.assertEqual(
            _matmul_destination_shapes(full_shape, ir_node, is_mutation_alias=True),
            (full_shape, (1, 4, 16, 64)),
        )


class TestMatmulTensorRoleLabels(InductorTestCase):
    def _make_matmul_spec(
        self,
        operand_shapes,
        observed_dim_orders,
        *,
        fp8_kernel=False,
        batch_dim_owners=None,
    ):
        """Build a host-only OpSpec with selected live physical dimensions."""
        output_shape = operand_shapes[-1]
        if batch_dim_owners is None:
            batch_dim_owners = tuple(
                (True,) * max(0, len(shape) - 2) for shape in operand_shapes[:2]
            )
        labels = ["ki", "kj", "y", "x", "mb", "out", "in"][-(len(output_shape) + 1) :]
        output_roles = tuple(labels[:-1])
        batch_roles = output_roles[:-2]
        m_role, n_role = output_roles[-2:]
        k_role = labels[-1]

        def roles_for(shape, tail):
            batch_rank = len(shape) - 2
            return batch_roles[-batch_rank:] + tail if batch_rank else tail

        argument_roles = (
            roles_for(operand_shapes[0], (m_role, k_role)),
            roles_for(operand_shapes[1], (k_role, n_role)),
            output_roles,
        )
        iteration_role_sizes = tuple(zip(output_roles, output_shape)) + (
            (k_role, operand_shapes[0][-1]),
        )
        raw_symbols = {
            role: sympy.Symbol(f"d{idx}")
            for idx, (role, size) in enumerate(iteration_role_sizes)
            if int(size) != 1
        }
        iteration_space = {
            raw_symbols[role]: (sympy.Integer(size), 1)
            for role, size in iteration_role_sizes
            if int(size) != 1
        }

        args = []
        semantic_sticks = (k_role, n_role, n_role)
        for idx, (shape, roles, observed_order, stick_role) in enumerate(
            zip(
                operand_shapes,
                argument_roles,
                observed_dim_orders,
                semantic_sticks,
            )
        ):
            role_sizes = dict(zip(roles, shape))
            coordinates = [raw_symbols[role] for role in reversed(observed_order)]
            coordinates.append(
                raw_symbols[stick_role]
                if int(role_sizes[stick_role]) != 1
                else sympy.S.Zero
            )
            device_size = [
                max(1, int(role_sizes[role])) for role in reversed(observed_order)
            ] + [128 if fp8_kernel and idx == 1 else 64]
            args.append(
                TensorArg(
                    is_input=idx < 2,
                    arg_index=idx,
                    device_dtype=(
                        DataFormats.SEN143_FP8
                        if fp8_kernel and idx < 2
                        else DataFormats.SEN169_FP16
                    ),
                    device_size=device_size,
                    device_coordinates=coordinates,
                    allocation={"hbm": idx},
                    element_arrangement=(
                        ElementArrangement.QFP8WT
                        if fp8_kernel and idx == 1
                        else ElementArrangement.STANDARD
                    ),
                )
            )

        return OpSpec(
            op="batchmatmulfp8" if fp8_kernel else "batchmatmul",
            is_reduction=True,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            args=args,
            op_info={},
            matmul_operand_shapes=tuple(
                tuple(sympy.Integer(dim) for dim in shape) for shape in operand_shapes
            ),
            matmul_operand_batch_dim_owners=batch_dim_owners,
        )

    def _factorize_physical_axis(self, spec, symbol, first_factor):
        extent = int(spec.iteration_space[symbol][0])
        self.assertEqual(extent % first_factor, 0)
        second_factor = extent // first_factor
        replacements = 0
        for arg in spec.args:
            for dim, (size, coordinate) in enumerate(
                zip(arg.device_size, arg.device_coordinates)
            ):
                if coordinate != symbol:
                    continue
                self.assertEqual(int(size), extent)
                arg.device_size[dim : dim + 1] = [first_factor, second_factor]
                arg.device_coordinates[dim : dim + 1] = [
                    sympy.Mod(symbol, first_factor),
                    sympy.floor(symbol / first_factor),
                ]
                replacements += 1
                break
        return replacements

    @staticmethod
    def _matmul_layout_orders(sdsc_spec):
        return [
            [str(dim) for dim in sdsc_spec.layouts[arg.layout]["dim_order"]]
            for arg in sdsc_spec.args
        ]

    @staticmethod
    def _matmul_stick_orders(sdsc_spec):
        return [
            [str(dim) for dim in sdsc_spec.layouts[arg.layout]["stick_dim_order"]]
            for arg in sdsc_spec.args
        ]

    def test_role_reconstruction_rejects_inconsistent_shapes(self):
        invalid_shapes = (
            (
                (4, 64, 64, 128),
                (1, 4, 128, 64),
                (1, 4, 64, 64),
            ),
            (
                (1, 64, 32, 128),
                (4096, 4096),
                (1, 64, 4096),
            ),
        )

        for shapes in invalid_shapes:
            with self.subTest(shapes=shapes):
                symbolic_shapes = tuple(
                    tuple(sympy.Integer(dim) for dim in shape) for shape in shapes
                )
                with self.assertRaisesRegex(
                    ValueError, "invalid matmul operand/output contract"
                ):
                    _matmul_role_shapes(symbolic_shapes)

    def test_batch_ownership_rank_is_validated(self):
        malformed = self._make_matmul_spec(
            ((4, 67, 256), (4, 256, 128), (4, 67, 128)),
            (("x", "in", "mb"), ("in", "out"), ("x", "out", "mb")),
            batch_dim_owners=((True,), ()),
        )

        with self.assertRaisesRegex(
            ValueError, "invalid matmul operand batch-dimension ownership"
        ):
            parse_op_spec(malformed)

    def test_parse_rejects_incomplete_matmul_semantics(self):
        cases = (
            (None, ((True,), (False,))),
            (
                (
                    (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
                    (sympy.Integer(1), sympy.Integer(4096), sympy.Integer(4096)),
                    (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
                ),
                None,
            ),
            (None, None),
        )
        for shapes, owners in cases:
            with self.subTest(shapes=shapes, owners=owners):
                malformed = self._make_matmul_spec(
                    (
                        (1, 1, 4096),
                        (1, 4096, 4096),
                        (1, 1, 4096),
                    ),
                    (("in",), ("in", "out"), ("out",)),
                    batch_dim_owners=((True,), (False,)),
                )
                malformed.matmul_operand_shapes = shapes
                malformed.matmul_operand_batch_dim_owners = owners
                with self.assertRaisesRegex(
                    ValueError,
                    "requires authoritative matmul_operand_shapes and "
                    "matmul_operand_batch_dim_owners",
                ):
                    parse_op_spec(malformed)

    def test_alignment_uses_logical_roles_not_post_tiling_extents(self):
        shapes = (
            (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
            (sympy.Integer(1), sympy.Integer(4096), sympy.Integer(4096)),
            (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
        )
        iteration_space = {
            sympy.Symbol("d0"): (sympy.Integer(1), 1),
            sympy.Symbol("d1"): (sympy.Integer(1), 1),
        }

        self.assertEqual(
            _align_matmul_dim_labels(shapes, iteration_space),
            ["out", "in"],
        )

    def test_alignment_rejects_factorized_logical_role(self):
        shapes = (
            (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
            (sympy.Integer(1), sympy.Integer(4096), sympy.Integer(4096)),
            (sympy.Integer(1), sympy.Integer(1), sympy.Integer(4096)),
        )
        factorized_k_space = {
            sympy.Symbol("c0"): (sympy.Integer(4096), 32),
            sympy.Symbol("c1"): (sympy.Integer(128), 1),
            sympy.Symbol("z0"): (sympy.Integer(32), 1),
        }

        with self.assertRaisesRegex(
            ValueError,
            "must contain exactly the ordered non-unit logical roles",
        ):
            _align_matmul_dim_labels(shapes, factorized_k_space)

    def test_prefill_simplification_restores_squeezed_batch_role(self):
        """Preserve the physical slot of the squeezed batch role in QK."""
        d0, d1, d2, d3 = sympy.symbols("d0 d1 d2 d3")
        args = [
            TensorArg(
                is_input=True,
                arg_index=18,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[64, 64, 2, 4, 64],
                device_coordinates=[
                    d1,
                    sympy.S.Zero,
                    sympy.floor(d3 / 64),
                    d0,
                    sympy.Mod(d3, 64),
                ],
                allocation={"hbm": 18},
            ),
            TensorArg(
                is_input=True,
                arg_index=17,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 128, 64],
                device_coordinates=[
                    d0,
                    d3,
                    sympy.Mod(d2, 64),
                ],
                allocation={"hbm": 17},
            ),
            TensorArg(
                is_input=False,
                arg_index=-1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 64, 64],
                device_coordinates=[
                    d0,
                    d1,
                    sympy.Mod(d2, 64),
                ],
                allocation={"lx": 0},
            ),
        ]
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                d0: (sympy.Integer(4), 1),
                d1: (sympy.Integer(64), 32),
                d2: (sympy.Integer(64), 1),
                d3: (sympy.Integer(128), 1),
            },
            args=args,
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (1, 32, 64, 128))),
                tuple(map(sympy.Integer, (1, 32, 128, 64))),
                tuple(map(sympy.Integer, (1, 32, 64, 64))),
            ),
            matmul_operand_batch_dim_owners=((True, True), (True, True)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")

        self.assertEqual(list(spec.iteration_space), [d0, d1, d2, d3])
        y_role = sympy.Symbol("y")
        self.assertTrue(
            all(
                coordinate.free_symbols <= {d0, d1, d2, d3, y_role}
                for arg in spec.args
                for coordinate in arg.device_coordinates
            )
        )
        self.assertTrue(
            all(
                any(
                    y_role in coordinate.free_symbols
                    for coordinate in arg.device_coordinates
                )
                for arg in spec.args
            ),
            "the physical placeholder shared by lhs/rhs/output must retain the "
            "canonical unit batch role",
        )
        self.assertEqual(
            _align_matmul_dim_labels(spec.matmul_operand_shapes, spec.iteration_space),
            ["x", "mb", "out", "in"],
        )

        sdsc_spec, _ = parse_op_spec(spec)
        self.assertEqual(
            self._matmul_layout_orders(sdsc_spec),
            [
                ["x", "in", "y", "mb"],
                ["in", "x", "y", "out"],
                ["mb", "x", "y", "out"],
            ],
        )
        self.assertEqual(
            self._matmul_stick_orders(sdsc_spec),
            [["in"], ["out"], ["out"]],
        )
        self.assertEqual(
            [
                {str(dim): gap for dim, gap in arg.backGap.items()}
                for arg in sdsc_spec.args
            ],
            [{"y": 63}, {}, {}],
        )

    def test_decode_simplification_restores_unique_squeezed_batch_role(self):
        """Decode B=1/M=1 maps the physical gap only to incidence-proven B."""
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        args = [
            TensorArg(
                is_input=True,
                arg_index=0,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[64, 4, 128],
                device_coordinates=[sympy.S.Zero, c0, c2],
                allocation={"hbm": 0},
            ),
            TensorArg(
                is_input=True,
                arg_index=1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[1, 4, 128, 64],
                device_coordinates=[sympy.S.Zero, c0, c2, c1],
                allocation={"hbm": 1},
            ),
            TensorArg(
                is_input=False,
                arg_index=2,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[1, 4, 64],
                device_coordinates=[sympy.S.Zero, c0, c1],
                allocation={"hbm": 2},
            ),
        ]
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                c0: (sympy.Integer(4), 4),
                c1: (sympy.Integer(64), 1),
                c2: (sympy.Integer(128), 2),
            },
            args=args,
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (1, 32, 1, 128))),
                tuple(map(sympy.Integer, (1, 32, 128, 64))),
                tuple(map(sympy.Integer, (1, 32, 1, 64))),
            ),
            matmul_operand_batch_dim_owners=((True, True), (True, True)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")

        y_role = sympy.Symbol("y")
        mb_role = sympy.Symbol("mb")
        self.assertEqual(list(spec.iteration_space), [c0, c1, c2])
        self.assertTrue(
            all(
                any(
                    y_role in coordinate.free_symbols
                    for coordinate in arg.device_coordinates
                )
                for arg in spec.args
            )
        )
        self.assertTrue(
            all(
                mb_role not in coordinate.free_symbols
                for arg in spec.args
                for coordinate in arg.device_coordinates
            ),
            "the all-operand physical gap must map to B, not the lhs/output-only M",
        )

        sdsc_spec, _ = parse_op_spec(spec)
        self.assertEqual(
            self._matmul_layout_orders(sdsc_spec),
            [
                ["x", "y", "in", "mb"],
                ["in", "x", "y", "out"],
                ["x", "y", "out", "mb"],
            ],
        )
        self.assertEqual(
            self._matmul_stick_orders(sdsc_spec),
            [["in"], ["out"], ["out"]],
        )
        self.assertEqual(
            [
                {str(dim): gap for dim, gap in arg.backGap.items()}
                for arg in sdsc_spec.args
            ],
            [{"y": 63}, {}, {}],
        )

    def test_simplification_rejects_ambiguous_squeezed_batch_role(self):
        """Two all-operand unit batch roles cannot be assigned positionally."""
        m, n, k = sympy.symbols("m n k")
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                m: (sympy.Integer(4), 1),
                n: (sympy.Integer(64), 1),
                k: (sympy.Integer(128), 1),
            },
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=0,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 4, 128],
                    device_coordinates=[sympy.S.Zero, m, k],
                    allocation={"hbm": 0},
                ),
                TensorArg(
                    is_input=True,
                    arg_index=1,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 128, 64],
                    device_coordinates=[sympy.S.Zero, k, n],
                    allocation={"hbm": 1},
                ),
                TensorArg(
                    is_input=False,
                    arg_index=2,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 4, 64],
                    device_coordinates=[sympy.S.Zero, m, n],
                    allocation={"hbm": 2},
                ),
            ],
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (1, 1, 4, 128))),
                tuple(map(sympy.Integer, (1, 1, 128, 64))),
                tuple(map(sympy.Integer, (1, 1, 4, 64))),
            ),
            matmul_operand_batch_dim_owners=((True, True), (True, True)),
        )

        with (
            V.set_graph_handler(SimpleNamespace()),
            self.assertRaisesRegex(
                Unsupported,
                r"cannot map synthetic unit axis .* candidates are \['x', 'y'\]",
            ),
        ):
            simplify_op_spec(spec)

    def test_simplification_maps_unit_m_only_to_owning_operands(self):
        """Pre-broadcast provenance restores M on lhs/output, never rhs."""
        x, n, k = sympy.symbols("x n k")
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                x: (sympy.Integer(2), 1),
                n: (sympy.Integer(64), 1),
                k: (sympy.Integer(128), 1),
            },
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=0,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 64, 128],
                    device_coordinates=[x, sympy.S.Zero, k],
                    allocation={"hbm": 0},
                ),
                TensorArg(
                    is_input=True,
                    arg_index=1,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 128, 64],
                    device_coordinates=[x, k, n],
                    allocation={"hbm": 1},
                ),
                TensorArg(
                    is_input=False,
                    arg_index=2,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 64, 64],
                    device_coordinates=[x, sympy.S.Zero, n],
                    allocation={"hbm": 2},
                ),
            ],
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (2, 1, 128))),
                tuple(map(sympy.Integer, (2, 128, 64))),
                tuple(map(sympy.Integer, (2, 1, 64))),
            ),
            matmul_operand_batch_dim_owners=((True,), (True,)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")

        m_role = sympy.Symbol("mb")
        self.assertEqual(
            [
                any(
                    m_role in coordinate.free_symbols
                    for coordinate in arg.device_coordinates
                )
                for arg in spec.args
            ],
            [True, False, True],
        )
        sdsc_spec, _ = parse_op_spec(spec)
        self.assertEqual(
            self._matmul_layout_orders(sdsc_spec),
            [["mb", "x", "in"], ["in", "x", "out"], ["mb", "x", "out"]],
        )

    def test_simplification_drops_padding_only_synthetic_axis(self):
        """A physical gap with no static-unit logical role remains padding."""
        x, m, n, k = sympy.symbols("x m n k")

        def tensor(is_input, index, size, coordinates):
            return TensorArg(
                is_input=is_input,
                arg_index=index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=size,
                device_coordinates=coordinates,
                allocation={"hbm": index},
            )

        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                x: (sympy.Integer(2), 1),
                m: (sympy.Integer(4), 1),
                n: (sympy.Integer(64), 1),
                k: (sympy.Integer(128), 1),
            },
            args=[
                tensor(True, 0, [64, 2, 4, 128], [sympy.S.Zero, x, m, k]),
                tensor(True, 1, [1, 2, 128, 64], [sympy.S.Zero, x, k, n]),
                tensor(False, 2, [1, 2, 4, 64], [sympy.S.Zero, x, m, n]),
            ],
            op_info={},
            matmul_operand_shapes=tuple(
                tuple(map(sympy.Integer, shape))
                for shape in ((2, 4, 128), (2, 128, 64), (2, 4, 64))
            ),
            matmul_operand_batch_dim_owners=((True,), (True,)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")
        self.assertEqual(list(spec.iteration_space), [x, m, n, k])
        self.assertFalse(
            any(
                str(symbol).startswith("z")
                for arg in spec.args
                for coordinate in arg.device_coordinates
                for symbol in coordinate.free_symbols
            )
        )

    def test_simplification_rejects_gap_from_nonowning_operand(self):
        """A rhs-only gap cannot be relabeled as lhs/output-only unit M."""
        x, n, k = sympy.symbols("x n k")
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                x: (sympy.Integer(2), 1),
                n: (sympy.Integer(64), 1),
                k: (sympy.Integer(128), 1),
            },
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=0,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 128],
                    device_coordinates=[x, k],
                    allocation={"hbm": 0},
                ),
                TensorArg(
                    is_input=True,
                    arg_index=1,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 64, 128, 64],
                    device_coordinates=[x, sympy.S.Zero, k, n],
                    allocation={"hbm": 1},
                ),
                TensorArg(
                    is_input=False,
                    arg_index=2,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 64],
                    device_coordinates=[x, n],
                    allocation={"hbm": 2},
                ),
            ],
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (2, 1, 128))),
                tuple(map(sympy.Integer, (2, 128, 64))),
                tuple(map(sympy.Integer, (2, 1, 64))),
            ),
            matmul_operand_batch_dim_owners=((True,), (True,)),
        )

        with (
            V.set_graph_handler(SimpleNamespace()),
            self.assertRaisesRegex(
                Unsupported,
                r"from tensor\(s\) \[1\].*candidates are \[\]",
            ),
        ):
            simplify_op_spec(spec)

    def test_simplification_validates_full_matmul_contract(self):
        malformed = self._make_matmul_spec(
            ((4, 128), (128, 64), (4, 64)),
            (("in", "mb"), ("in", "out"), ("out", "mb")),
        )
        malformed.matmul_operand_shapes = 42

        with self.assertRaisesRegex(
            Unsupported, "invalid authoritative matmul metadata"
        ):
            simplify_op_spec(malformed)

    def test_fp8_simplification_restores_unit_k_by_provenance(self):
        x, m, n = sympy.symbols("x m n")
        spec = OpSpec(
            op="batchmatmulfp8",
            is_reduction=True,
            iteration_space={
                x: (sympy.Integer(2), 1),
                m: (sympy.Integer(4), 1),
                n: (sympy.Integer(64), 1),
            },
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=0,
                    device_dtype=DataFormats.SEN143_FP8,
                    device_size=[2, 4, 128],
                    device_coordinates=[x, m, sympy.S.Zero],
                    allocation={"hbm": 0},
                ),
                TensorArg(
                    is_input=True,
                    arg_index=1,
                    device_dtype=DataFormats.SEN143_FP8,
                    device_size=[2, 128, 128],
                    device_coordinates=[x, sympy.S.Zero, n],
                    allocation={"hbm": 1},
                    element_arrangement=ElementArrangement.QFP8WT,
                ),
                TensorArg(
                    is_input=False,
                    arg_index=2,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[2, 4, 64],
                    device_coordinates=[x, m, n],
                    allocation={"hbm": 2},
                ),
            ],
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (2, 4, 1))),
                tuple(map(sympy.Integer, (2, 1, 64))),
                tuple(map(sympy.Integer, (2, 4, 64))),
            ),
            matmul_operand_batch_dim_owners=((True,), (True,)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        k_role = sympy.Symbol("in")
        self.assertEqual(
            [
                any(
                    k_role in coordinate.free_symbols
                    for coordinate in arg.device_coordinates
                )
                for arg in spec.args
            ],
            [True, True, False],
        )
        sdsc_spec, _ = parse_op_spec(spec)
        self.assertEqual(
            self._matmul_stick_orders(sdsc_spec),
            [["in"], ["in", "out"], ["out"]],
        )

    def test_simplification_refines_factorized_m_with_shared_rhs(self):
        spec = self._make_matmul_spec(
            ((8, 64), (64, 64), (8, 64)),
            (("mb",), ("in",), ("mb",)),
        )
        (m, _n, _k) = spec.iteration_space
        self.assertEqual(self._factorize_physical_axis(spec, m, 4), 2)

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")
        self.assertEqual(
            tuple(tuple(map(int, shape)) for shape in spec.matmul_operand_shapes),
            ((4, 2, 64), (64, 64), (4, 2, 64)),
        )
        self.assertEqual(spec.matmul_operand_batch_dim_owners, ((True,), ()))
        self.assertEqual(
            _align_matmul_dim_labels(spec.matmul_operand_shapes, spec.iteration_space),
            ["x", "mb", "out", "in"],
        )

    def test_simplification_refines_factorized_batch_axis(self):
        spec = self._make_matmul_spec(
            ((16, 8, 64), (16, 64, 64), (16, 8, 64)),
            (
                ("x", "mb"),
                ("x", "in"),
                ("x", "mb"),
            ),
        )
        (batch, _m, _n, _k) = spec.iteration_space
        self.assertEqual(self._factorize_physical_axis(spec, batch, 8), 3)

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")
        self.assertEqual(
            tuple(tuple(map(int, shape)) for shape in spec.matmul_operand_shapes),
            (
                (8, 2, 8, 64),
                (8, 2, 64, 64),
                (8, 2, 8, 64),
            ),
        )
        self.assertEqual(
            spec.matmul_operand_batch_dim_owners,
            ((True, True), (True, True)),
        )
        self.assertEqual(
            _align_matmul_dim_labels(spec.matmul_operand_shapes, spec.iteration_space),
            ["y", "x", "mb", "out", "in"],
        )

    def test_simplification_promotes_factorized_m_for_batched_rhs(self):
        spec = self._make_matmul_spec(
            ((2, 12, 128), (2, 128, 64), (2, 12, 64)),
            (
                ("x", "mb"),
                ("x", "in"),
                ("x", "mb"),
            ),
        )
        (_batch, m, _n, _k) = spec.iteration_space
        self.assertEqual(self._factorize_physical_axis(spec, m, 4), 2)

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")
        self.assertEqual(
            tuple(tuple(map(int, shape)) for shape in spec.matmul_operand_shapes),
            (
                (2, 4, 3, 128),
                (2, 4, 128, 64),
                (2, 4, 3, 64),
            ),
        )
        self.assertEqual(
            spec.matmul_operand_batch_dim_owners,
            ((True, True), (True, False)),
        )
        self.assertEqual(
            _align_matmul_dim_labels(spec.matmul_operand_shapes, spec.iteration_space),
            ["y", "x", "mb", "out", "in"],
        )

    def test_factorized_m_relabels_leading_static_unit_batch_role(self):
        spec = self._make_matmul_spec(
            ((1, 8, 64), (1, 64, 64), (1, 8, 64)),
            (("mb",), ("in",), ("mb",)),
            batch_dim_owners=((True,), (True,)),
        )
        (m, _n, _k) = spec.iteration_space
        self.assertEqual(self._factorize_physical_axis(spec, m, 4), 2)
        for index, arg in enumerate(spec.args):
            arg.device_size.insert(0, 64 if index == 0 else 1)
            arg.device_coordinates.insert(0, sympy.S.Zero)

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        validate_op_specs([spec], stage="after_simplification")
        self.assertEqual(
            tuple(tuple(map(int, shape)) for shape in spec.matmul_operand_shapes),
            (
                (1, 4, 2, 64),
                (1, 4, 64, 64),
                (1, 4, 2, 64),
            ),
        )
        self.assertEqual(
            spec.matmul_operand_batch_dim_owners,
            ((True, True), (True, False)),
        )
        y_role = sympy.Symbol("y")
        x_role = sympy.Symbol("x")
        self.assertTrue(
            all(
                any(
                    y_role in coordinate.free_symbols
                    for coordinate in arg.device_coordinates
                )
                for arg in spec.args
            )
        )
        self.assertTrue(
            all(
                x_role not in coordinate.free_symbols
                for arg in spec.args
                for coordinate in arg.device_coordinates
            )
        )

    def test_simplification_preserves_non_unit_factorization(self):
        """A normalization-created radix digit remains visible and rejected."""
        n, k = sympy.symbols("n k")
        spec = OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                n: (sympy.Integer(4096), 32),
                k: (sympy.Integer(4096), 1),
            },
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=0,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 4096],
                    device_coordinates=[sympy.S.Zero, k],
                    allocation={"hbm": 0},
                ),
                TensorArg(
                    is_input=True,
                    arg_index=1,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 2, 128, 4096],
                    device_coordinates=[
                        sympy.floor(k / 128),
                        sympy.S.Zero,
                        sympy.Mod(k, 128),
                        n,
                    ],
                    allocation={"hbm": 1},
                ),
                TensorArg(
                    is_input=False,
                    arg_index=2,
                    device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 4096],
                    device_coordinates=[sympy.S.Zero, n],
                    allocation={"hbm": 2},
                ),
            ],
            op_info={},
            matmul_operand_shapes=(
                tuple(map(sympy.Integer, (1, 1, 4096))),
                tuple(map(sympy.Integer, (1, 4096, 4096))),
                tuple(map(sympy.Integer, (1, 1, 4096))),
            ),
            matmul_operand_batch_dim_owners=((True,), (True,)),
        )

        with V.set_graph_handler(SimpleNamespace()):
            simplify_op_spec(spec)

        self.assertEqual(
            [int(size) for size, _split in spec.iteration_space.values()],
            [4096, 128, 32],
        )
        with self.assertRaisesRegex(
            ValueError, "must contain exactly the ordered non-unit logical roles"
        ):
            _align_matmul_dim_labels(spec.matmul_operand_shapes, spec.iteration_space)

    def test_decode_projection_keeps_equal_n_k_roles_positional(self):
        """Equal N/K extents cannot make the lhs acquire the output-N role."""
        spec = self._make_matmul_spec(
            (
                (1, 1, 4096),
                (1, 4096, 4096),
                (1, 1, 4096),
            ),
            (("in",), ("in", "out"), ("out",)),
            batch_dim_owners=((True,), (False,)),
        )

        sdsc_spec, _ = parse_op_spec(spec)

        self.assertEqual(
            [str(dim) for dim in sdsc_spec.iteration_space],
            ["x", "mb", "out", "in"],
        )
        self.assertEqual(
            self._matmul_layout_orders(sdsc_spec),
            [
                ["in", "mb", "x"],
                ["in", "out"],
                ["out", "mb", "x"],
            ],
        )
        self.assertNotIn(
            "out",
            self._matmul_layout_orders(sdsc_spec)[0],
            "projection lhs incorrectly acquired the equal-sized N role",
        )

    def test_role_matrix(self):
        """Logical roles survive rank, broadcast, unit-axis, and FP8 variants."""
        cases = (
            (
                "rank2",
                ((4, 128), (128, 64), (4, 64)),
                (("in", "mb"), ("in", "out"), ("out", "mb")),
                [["in", "mb"], ["in", "out"], ["out", "mb"]],
                None,
                False,
            ),
            (
                "rank3_shared_rhs",
                ((2, 4, 128), (128, 64), (2, 4, 64)),
                (("x", "in", "mb"), ("in", "out"), ("x", "out", "mb")),
                [["x", "in", "mb"], ["in", "out"], ["x", "out", "mb"]],
                None,
                False,
            ),
            (
                "rank3_batched",
                ((2, 4, 128), (2, 128, 64), (2, 4, 64)),
                (
                    ("x", "in", "mb"),
                    ("x", "out", "in"),
                    ("x", "out", "mb"),
                ),
                [
                    ["x", "in", "mb"],
                    ["x", "out", "in"],
                    ["x", "out", "mb"],
                ],
                None,
                False,
            ),
            (
                "rank3_broadcast_rhs",
                ((4, 67, 256), (4, 256, 128), (4, 67, 128)),
                (("x", "in", "mb"), ("in", "out"), ("x", "out", "mb")),
                [["x", "in", "mb"], ["in", "out"], ["x", "out", "mb"]],
                ((True,), (False,)),
                False,
            ),
            (
                "unit_m_and_n",
                ((1, 1, 128), (1, 128, 1), (1, 1, 1)),
                (("in",), ("in",), ()),
                [
                    ["in", "mb", "x"],
                    ["in", "out", "x"],
                    ["out", "mb", "x"],
                ],
                None,
                False,
            ),
            (
                "unit_k",
                ((32, 4, 1), (32, 1, 64), (32, 4, 64)),
                (("x", "mb"), ("x", "out"), ("x", "out", "mb")),
                [
                    ["x", "mb", "in"],
                    ["x", "out", "in"],
                    ["x", "out", "mb"],
                ],
                None,
                False,
            ),
            (
                "unit_k_qfp8wt",
                ((4, 1), (1, 64), (4, 64)),
                (("mb",), ("out",), ("mb", "out")),
                [["mb", "in"], ["out", "in"], ["mb", "out"]],
                None,
                True,
            ),
        )
        for name, shapes, observed, expected, owners, fp8 in cases:
            with self.subTest(name=name):
                sdsc_spec, _ = parse_op_spec(
                    self._make_matmul_spec(
                        shapes,
                        observed,
                        batch_dim_owners=owners,
                        fp8_kernel=fp8,
                    )
                )
                expected_dims = [
                    "ki",
                    "kj",
                    "y",
                    "x",
                    "mb",
                    "out",
                    "in",
                ][-(len(shapes[-1]) + 1) :]
                self.assertEqual(
                    [str(dim) for dim in sdsc_spec.iteration_space], expected_dims
                )
                self.assertEqual(
                    [arg.layout for arg in sdsc_spec.args],
                    ["INPUT", "KERNEL", "OUTPUT"],
                )
                self.assertEqual(self._matmul_layout_orders(sdsc_spec), expected)
                expected_sticks = (
                    [["in"], ["in", "out"], ["out"]]
                    if fp8
                    else [["in"], ["out"], ["out"]]
                )
                self.assertEqual(self._matmul_stick_orders(sdsc_spec), expected_sticks)


class TestSymbolKindKernelDerivedSymbolic(InductorTestCase):
    """Unit tests for the kernel_derived_symbolic variant of SymbolKind, added
    for per-core symbolic start addresses.

    This is the SDSC-JSON marker only: is_derived_symbolic gates the SDSC
    per-core registration. is_derived stays False so the existing bundle
    kernel_derived addi branch does not match it (the real per-core arith
    formula is a separate later PR).
    """

    def test_factory_sets_all_fields(self):
        sk = SymbolKind.kernel_derived_symbolic(
            arg_index=2,
            core_idx=5,
            split_count=8,
            base_sym_idx=3,
            pytorch_sym="s0",
        )
        self.assertEqual(sk.kind, "kernel_derived_symbolic")
        self.assertEqual(sk.arg_index, 2)
        self.assertEqual(sk.core_idx, 5)
        self.assertEqual(sk.split_count, 8)
        self.assertEqual(sk.base_sym_idx, 3)
        self.assertEqual(sk.pytorch_sym, "s0")

    def test_no_stride_stored_on_marker(self):
        # This PR only tags the symbolic split; no per-element stride is
        # computed here (that is the bundle-arm follow-up), so offset stays at
        # its default sentinel.
        sk = SymbolKind.kernel_derived_symbolic(
            arg_index=0,
            core_idx=1,
            split_count=4,
            base_sym_idx=0,
            pytorch_sym="s0",
        )
        self.assertEqual(sk.offset, 0)

    def test_is_derived_symbolic_true(self):
        sk = SymbolKind.kernel_derived_symbolic(
            arg_index=0,
            core_idx=1,
            split_count=4,
            base_sym_idx=0,
            pytorch_sym="s0",
        )
        self.assertTrue(sk.is_derived_symbolic)

    def test_is_derived_strict_check(self):
        # is_derived must be False so the existing bundle.py arith.addi branch
        # (which matches kernel_derived) does not pick up this variant.
        sk = SymbolKind.kernel_derived_symbolic(
            arg_index=0,
            core_idx=1,
            split_count=4,
            base_sym_idx=0,
            pytorch_sym="s0",
        )
        self.assertFalse(sk.is_derived)
        self.assertFalse(sk.is_pool)
        self.assertFalse(sk.is_dimension)

    def test_kernel_derived_is_distinguishable(self):
        # kernel_derived and kernel_derived_symbolic share base_sym_idx
        # semantics but must be distinguishable by the codegen.
        concrete = SymbolKind.kernel_derived(base_sym_idx=0, offset=128, arg_index=0)
        self.assertFalse(concrete.is_derived_symbolic)
        self.assertTrue(concrete.is_derived)


class TestSymbolicSplitPredicates(InductorTestCase):
    """Unit tests for _symbolic_split_info and _tensor_has_symbolic_split.

    The predicates decide whether a tensor's core split lands on a symbolic
    dim, which is what gates per-core symbolic address emission in
    generate_sdsc. Lightweight stubs avoid the full TensorArg / SDSCSpec
    construction path.
    """

    _SYMBOLIC_DIMS_MB = {"mb": ("s0", 64, 1024)}
    _MB_SYM = sympy.Symbol("mb")
    _OUT_SYM = sympy.Symbol("out")

    @staticmethod
    def _stub_tensor(arg_index, scales, strides):
        return SimpleNamespace(
            arg_index=arg_index,
            scales=scales,
            strides=strides,
        )

    def test_symbolic_split_returns_info(self):
        # mb is symbolic and split across 8 cores; the tensor uses it.
        tensor = self._stub_tensor(
            arg_index=0,
            scales={self._MB_SYM: 1, self._OUT_SYM: 1},
            strides={self._MB_SYM: 256, self._OUT_SYM: 1},
        )
        work_slices = {self._MB_SYM: 8, self._OUT_SYM: 1}
        info = _symbolic_split_info(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        self.assertEqual(info, ("mb", 8, "s0"))
        self.assertTrue(
            _tensor_has_symbolic_split(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        )

    def test_symbolic_dim_not_split_returns_none(self):
        # mb is symbolic but work_slices == 1, so it is not actually split.
        tensor = self._stub_tensor(
            arg_index=0,
            scales={self._MB_SYM: 1, self._OUT_SYM: 1},
            strides={self._MB_SYM: 256, self._OUT_SYM: 1},
        )
        work_slices = {self._MB_SYM: 1, self._OUT_SYM: 8}
        self.assertIsNone(
            _symbolic_split_info(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        )
        self.assertFalse(
            _tensor_has_symbolic_split(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        )

    def test_no_symbolic_dims_returns_none(self):
        tensor = self._stub_tensor(
            arg_index=0,
            scales={self._MB_SYM: 1, self._OUT_SYM: 1},
            strides={self._MB_SYM: 256, self._OUT_SYM: 1},
        )
        work_slices = {self._MB_SYM: 8, self._OUT_SYM: 1}
        self.assertIsNone(_symbolic_split_info(tensor, work_slices, {}))

    def test_pool_tensor_skipped(self):
        # Pool tensors have arg_index < 0 and no kernel base to derive from, so
        # the predicate skips them even when a symbolic dim is split.
        tensor = self._stub_tensor(
            arg_index=-1,
            scales={self._MB_SYM: 1, self._OUT_SYM: 1},
            strides={self._MB_SYM: 256, self._OUT_SYM: 1},
        )
        work_slices = {self._MB_SYM: 8, self._OUT_SYM: 1}
        self.assertIsNone(
            _symbolic_split_info(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        )

    def test_reduced_or_broadcast_dim_skipped(self):
        # scales <= 0 means the tensor reduces along or broadcasts against this
        # dim, so its per-core address does not depend on the symbolic value.
        tensor = self._stub_tensor(
            arg_index=0,
            scales={self._MB_SYM: -1, self._OUT_SYM: 1},
            strides={self._MB_SYM: 0, self._OUT_SYM: 1},
        )
        work_slices = {self._MB_SYM: 8, self._OUT_SYM: 1}
        self.assertIsNone(
            _symbolic_split_info(tensor, work_slices, self._SYMBOLIC_DIMS_MB)
        )


class TestGenerateSdscSymbolicPerCoreAddresses(InductorTestCase):
    """End-to-end at the SDSC-JSON layer: a symbolic-batch add with a per-core
    split emits kernel_derived_symbolic per-core addresses.

    Fixture mirrors TestSdscJsonSymbolicDimSmoke but with work_slices that
    actually split the symbolic dim across 8 cores (s0 max=512, granularity=64).
    This asserts on the SDSC JSON only. The real per-core arith formula and its
    symbolDefinitions_ content are a separate later PR, so symbolDefinitions_
    stays empty here.
    """

    _DEVICE_SIZE = [4, 512, 64]
    _HBM_BASE = 0x400000000
    _NUM_CORES = 8

    def _make_symbolic_op_spec(self) -> OpSpec:
        c_row, c_col = sympy.Symbol("c_row"), sympy.Symbol("c_col")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        coords = [c_col // 64, c_row, sympy.Mod(c_col, 64)]

        def _tensor_arg(is_input, arg_index, hbm_base):
            return TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=list(self._DEVICE_SIZE),
                device_coordinates=coords,
                allocation={"hbm": hbm_base},
            )

        iteration_space = {
            c_row: (s0, self._NUM_CORES),
            c_col: (sympy.Integer(256), 1),
        }
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            args=[
                _tensor_arg(True, 0, self._HBM_BASE),
                _tensor_arg(True, 1, self._HBM_BASE + 0x1000),
                _tensor_arg(False, 2, self._HBM_BASE + 0x100000000),
            ],
            op_info={},
            symbolic_dim_bounds={"s0": (512, 64)},
        )

    def test_per_core_symbolic_addresses_emitted(self):
        op_spec = self._make_symbolic_op_spec()
        sdsc_json, _, _, symbol_kinds = compile_op_spec(
            idx=0, op_spec=op_spec, symbols=[]
        )

        top = next(iter(sdsc_json.values()))
        dsc = next(iter(top["dscs_"][0].values()))

        # The dim symbol s0 -> mb is still bound the same way; this change
        # layers per-core symbols on top of the dim mapping.
        self.assertEqual(dsc["dimToSymbolMapping_"], {"mb": [-1]})

        # Every HBM tensor's allocate node carries the symbolic-address flag and
        # a full per-core data_ map whose c>0 values are symbol id strings.
        hbm_allocate_nodes = [
            node
            for node in dsc["scheduleTree_"]
            if node.get("nodeType_") == "allocate" and node.get("component_") == "hbm"
        ]
        self.assertEqual(len(hbm_allocate_nodes), 3)

        per_core_symbol_ids: list[int] = []
        for node in hbm_allocate_nodes:
            self.assertEqual(node["isStartAddrSymbolic_"], 1)
            data = node["startAddressCoreCorelet_"]["data_"]
            self.assertEqual(len(data), self._NUM_CORES)
            for c in range(self._NUM_CORES):
                key = f"[{c}, 0, 0]"
                self.assertIn(key, data)
                value = int(data[key])
                # c>0 must be a negative symbol id: a positive value would mean
                # the per-core symbolic path did not fire for that core.
                if c > 0:
                    self.assertLess(value, 0)
                    per_core_symbol_ids.append(value)

        # The dim-symbol id (-1) must not collide with any per-core address id.
        self.assertNotIn(-1, per_core_symbol_ids)

        # symbol_kinds carries the dim kinds followed by the address kinds.
        # Exactly 3 HBM tensors * (NUM_CORES - 1) symbolic per-core addresses.
        symbolic_kinds = [sk for sk in symbol_kinds if sk.is_derived_symbolic]
        self.assertEqual(len(symbolic_kinds), 3 * (self._NUM_CORES - 1))
        for sk in symbolic_kinds:
            self.assertEqual(sk.split_count, self._NUM_CORES)
            self.assertEqual(sk.pytorch_sym, "s0")
            self.assertGreaterEqual(sk.core_idx, 1)

        # This PR only tags the symbolic split; no per-element stride is stored
        # on the marker (that is the bundle-arm follow-up), so offset stays at
        # its default sentinel for every symbolic per-core symbol.
        for sk in symbolic_kinds:
            self.assertEqual(sk.offset, 0)

        # SDSC-only change: the real per-core arith formula is a later PR, so
        # symbolDefinitions_ stays empty.
        self.assertEqual(top["symbolDefinitions_"], {})

    def test_dim_symbol_ids_lower_magnitude_than_address_ids(self):
        # Dim symbols must occupy the smallest-magnitude (closest to zero) ids
        # in the SDSC's local range, before any address symbol. A future change
        # that inverts this would silently shift bundle.mlir operand positions.
        op_spec = self._make_symbolic_op_spec()
        _, _, _, symbol_kinds = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])
        first_address = next(
            i for i, sk in enumerate(symbol_kinds) if not sk.is_dimension
        )
        # All dimension kinds come before any address kind.
        for sk in symbol_kinds[:first_address]:
            self.assertTrue(sk.is_dimension)
        for sk in symbol_kinds[first_address:]:
            self.assertFalse(sk.is_dimension)
        # And at least one symbolic per-core address was registered.
        self.assertTrue(
            any(sk.is_derived_symbolic for sk in symbol_kinds[first_address:])
        )
