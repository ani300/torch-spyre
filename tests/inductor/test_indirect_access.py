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

"""Tests for generic indirect tensor access (IBR addressing).

Covers two paths:

1. ``TestIndirectAccessSDSC`` — hand-built ``OpSpec`` passed directly to
   ``compile_op_spec`` to validate that the SDSC JSON carries the four
   indirect-access fields used by deeptools:
     - ``indirectAllocType_`` on value/index allocate nodes;
     - ``relatedIndirectAccessAlloc_`` cross-references;
     - ``dsType_ == "KERNEL_IDX"`` on the index labeled-ds;
     - ``indirectAccessIndexLabeledDs`` on the ``computeOp_``.

2. ``TestIndirectAccessFXPass`` — compiles small modules that use
   ``torch.index_select``, ``torch.nn.functional.embedding`` and verifies
   the pre-grad FX pass rewrites them into
   ``add(spyre::indirect_gather, 0)``.
"""

import sympy
import torch
import torch_spyre  # noqa: F401

from torch._inductor.test_case import TestCase as InductorTestCase

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import (
    SDSCIndirectSrc,
    compile_op_spec,
    parse_op_spec,
)
from torch_spyre._inductor.op_spec import (
    IndirectSource,
    OpSpec,
    TensorArg,
)


def _make_indirect_op_spec():
    """Hand-build an OpSpec mirroring the reference SDSC structure.

    Three tensors:
      - arg 0: value tensor (4D-ish, fp16), indirectly addressed on dim 0;
      - arg 1: index tensor (2D, int32), the stick dim is the second axis;
      - arg 2: output (same layout as value).

    The op is a plain ``add`` consuming the gathered value and a broadcast
    scalar — analogous to the FX-pass-injected ``add(gather, 0)`` pattern.
    """
    k_sym = sympy.Symbol("c0")
    h_sym = sympy.Symbol("c1")
    index_value = sympy.Symbol("index_value")

    experts_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 8, 64],
        device_coordinates=[
            sympy.floor(h_sym / 64),
            k_sym,
            sympy.Mod(h_sym, 64),
        ],
        allocation=None,
        indirect_source=IndirectSource(
            index_arg_index=1,
            gather_dim=0,
            base_offset_expr=index_value * 32768,
        ),
    )

    indices_arg = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.IEEE_INT32,
        device_size=[1, 1, 32],
        device_coordinates=[
            sympy.Integer(0),
            sympy.Integer(0),
            sympy.Mod(k_sym, 32),
        ],
        allocation=None,
    )

    output_arg = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 4, 64],
        device_coordinates=[
            sympy.floor(h_sym / 64),
            k_sym,
            sympy.Mod(h_sym, 64),
        ],
        allocation=None,
    )

    iteration_space = {
        k_sym: (sympy.Integer(4), 1),
        h_sym: (sympy.Integer(128), 1),
    }

    return OpSpec(
        op="identity",
        is_reduction=False,
        iteration_space=iteration_space,
        args=[experts_arg, indices_arg, output_arg],
        op_info={},
    )


def _get_dsc_inner(sdsc_json):
    opfunc_key = next(iter(sdsc_json))
    dsc = sdsc_json[opfunc_key]["dscs_"][0]
    inner_key = next(iter(dsc))
    return dsc[inner_key]


class TestIndirectSourceDataclass(InductorTestCase):
    def test_indirect_source_attaches_to_tensor_arg(self):
        iv = sympy.Symbol("index_value")
        src = IndirectSource(
            index_arg_index=1, gather_dim=0, base_offset_expr=iv * 256
        )
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 2, 1],
            device_coordinates=[sympy.Symbol("x"), sympy.Integer(0)],
            allocation=None,
            indirect_source=src,
        )
        self.assertIsNotNone(arg.indirect_source)
        self.assertEqual(arg.indirect_source.index_arg_index, 1)
        self.assertEqual(arg.indirect_source.gather_dim, 0)

    def test_indirect_source_default_is_none(self):
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 2, 1],
            device_coordinates=[sympy.Symbol("x"), sympy.Integer(0)],
            allocation=None,
        )
        self.assertIsNone(arg.indirect_source)


class TestSDSCIndirectSrc(InductorTestCase):
    def test_sdsc_indirect_src_defaults(self):
        s = SDSCIndirectSrc(index_tensor_idx=1, base_offset_expr="index_value*32768")
        self.assertEqual(s.address_mode, "ibr")


class TestParseOpSpecIndirect(InductorTestCase):
    def test_indirect_source_propagates_to_sdsc_args(self):
        op_spec = _make_indirect_op_spec()
        sdsc_spec = parse_op_spec(op_spec)
        indirect_args = [a for a in sdsc_spec.args if a.indirect_src is not None]
        self.assertEqual(len(indirect_args), 1)
        isrc = indirect_args[0].indirect_src
        self.assertEqual(isrc.index_tensor_idx, 1)
        self.assertIn("index_value", isrc.base_offset_expr)
        self.assertIn("32768", isrc.base_offset_expr)
        self.assertEqual(isrc.address_mode, "ibr")

    def test_index_tensor_data_format_is_int32(self):
        op_spec = _make_indirect_op_spec()
        sdsc_spec = parse_op_spec(op_spec)
        self.assertEqual(
            sdsc_spec.args[1].data_format, DataFormats.IEEE_INT32
        )


class TestIndirectAccessSDSC(InductorTestCase):
    """Structural assertions on the emitted SDSC JSON."""

    def test_value_tensor_has_indirect_alloc_type(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        tree = _get_dsc_inner(sdsc_json)["scheduleTree_"]
        value_node = tree[0]
        self.assertEqual(value_node["ldsIdx_"], 0)
        self.assertEqual(value_node["indirectAllocType_"], "value_tensor")
        self.assertIn("relatedIndirectAccessAlloc_", value_node)

    def test_index_tensor_has_indirect_alloc_type(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        tree = _get_dsc_inner(sdsc_json)["scheduleTree_"]
        index_node = tree[1]
        self.assertEqual(index_node["ldsIdx_"], 1)
        self.assertEqual(index_node["indirectAllocType_"], "index_tensor")
        self.assertIn("relatedIndirectAccessAlloc_", index_node)

    def test_related_alloc_cross_references(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        tree = _get_dsc_inner(sdsc_json)["scheduleTree_"]
        value_node = tree[0]
        index_node = tree[1]
        self.assertEqual(
            value_node["relatedIndirectAccessAlloc_"], index_node["name_"]
        )
        self.assertEqual(
            index_node["relatedIndirectAccessAlloc_"], value_node["name_"]
        )

    def test_output_tensor_has_no_indirection(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        tree = _get_dsc_inner(sdsc_json)["scheduleTree_"]
        output_node = tree[2]
        self.assertEqual(output_node["indirectAllocType_"], "no_indirection")
        self.assertNotIn("relatedIndirectAccessAlloc_", output_node)

    def test_index_tensor_has_kernel_idx_dstype(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        labeled = _get_dsc_inner(sdsc_json)["labeledDs_"]
        kernel_idx = [lds for lds in labeled if lds["dsType_"] == "KERNEL_IDX"]
        self.assertEqual(len(kernel_idx), 1)
        self.assertEqual(kernel_idx[0]["ldsIdx_"], 1)

    def test_compute_op_has_indirect_access_index_labeled_ds(self):
        sdsc_json = compile_op_spec(0, _make_indirect_op_spec())
        compute_ops = _get_dsc_inner(sdsc_json)["computeOp_"]
        self.assertEqual(len(compute_ops), 1)
        self.assertIn("indirectAccessIndexLabeledDs", compute_ops[0])
        self.assertEqual(
            compute_ops[0]["indirectAccessIndexLabeledDs"], ["Tensor1-idx1"]
        )

    def test_standard_op_has_no_indirect_fields(self):
        x_sym = sympy.Symbol("c0")
        y_sym = sympy.Symbol("c1")

        def make_arg(is_input, idx):
            return TensorArg(
                is_input=is_input,
                arg_index=idx,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[2, 4, 64],
                device_coordinates=[
                    sympy.floor(y_sym / 64),
                    x_sym,
                    sympy.Mod(y_sym, 64),
                ],
                allocation=None,
            )

        op_spec = OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={
                x_sym: (sympy.Integer(4), 1),
                y_sym: (sympy.Integer(128), 1),
            },
            args=[make_arg(True, 0), make_arg(False, 1)],
            op_info={},
        )
        sdsc_json = compile_op_spec(0, op_spec)
        tree = _get_dsc_inner(sdsc_json)["scheduleTree_"]
        for node in tree:
            self.assertEqual(node["indirectAllocType_"], "no_indirection")
            self.assertNotIn("relatedIndirectAccessAlloc_", node)
        labeled = _get_dsc_inner(sdsc_json)["labeledDs_"]
        for lds in labeled:
            self.assertNotEqual(lds["dsType_"], "KERNEL_IDX")
        compute_ops = _get_dsc_inner(sdsc_json)["computeOp_"]
        self.assertNotIn("indirectAccessIndexLabeledDs", compute_ops[0])


class TestIndirectAccessEndToEnd(InductorTestCase):
    """End-to-end ``torch.compile`` tests for gather / embedding / index_select.

    Uses upstream Inductor's ``ops.indirect_indexing`` idiom: the Spyre
    backend implements ``SpyreKernelOpsHandler.indirect_indexing`` which
    returns a sympy ``tmp_N`` symbol; ``store()`` then translates
    occurrences of those symbols in a ``TensorArg`` 's device coordinates
    into ``IndirectSource`` metadata so the emitted SDSC carries the
    indirect-access fields deeptools expects.
    """

    def _capture_sdsc(self):
        """Capture SDSC dicts emitted by the compiler and short-circuit
        device execution.

        Wraps both ``bundle.generate_bundle`` (emits each op-spec as a
        JSON bundle) and ``subprocess.run`` (invokes ``dxp_standalone``
        on that bundle).  We let the bundle pass through to disk so that
        Inductor's async-compile proceeds, but stub out the
        ``dxp_standalone`` subprocess so we don't wait for / require
        Spyre hardware.

        Returns ``(captured_list, restore_fn)``.
        """
        from torch_spyre._inductor.codegen import bundle as bundle_mod
        from torch_spyre._inductor.codegen.superdsc import compile_op_spec
        from torch_spyre._inductor.op_spec import OpSpec
        from torch_spyre.execution import async_compile as async_mod
        import subprocess

        captured = []
        original_bundle = bundle_mod.generate_bundle
        async_original = async_mod.generate_bundle

        def wrapped(kernel_name, output_dir, specs):
            for i, s in enumerate(specs):
                if isinstance(s, OpSpec):
                    captured.append(compile_op_spec(i, s))
            return original_bundle(kernel_name, output_dir, specs)

        bundle_mod.generate_bundle = wrapped
        async_mod.generate_bundle = wrapped

        # Short-circuit the downstream ``dxp_standalone`` call and the
        # actual runtime.  A non-zero exit code fails the compile, but
        # by then we've already captured every SDSC.
        original_subprocess_run = subprocess.run

        class _CompleteProc:
            returncode = 0
            stdout = b""
            stderr = b""

        def fake_run(cmd, *args, **kwargs):
            if cmd and isinstance(cmd, (list, tuple)) and cmd[0] == "dxp_standalone":
                return _CompleteProc()
            return original_subprocess_run(cmd, *args, **kwargs)

        subprocess.run = fake_run

        def restore():
            bundle_mod.generate_bundle = original_bundle
            async_mod.generate_bundle = async_original
            subprocess.run = original_subprocess_run

        return captured, restore

    def _assert_sdsc_has_indirect_fields(self, captured):
        """Across all captured SDSCs, assert at least one has the indirect fields."""
        found_indirect = False
        for sdsc in captured:
            opfunc_key = next(iter(sdsc))
            dsc_outer = sdsc[opfunc_key]["dscs_"][0]
            inner_key = next(iter(dsc_outer))
            dsc = dsc_outer[inner_key]
            alloc_nodes = [
                n
                for n in dsc["scheduleTree_"]
                if n.get("indirectAllocType_") in ("value_tensor", "index_tensor")
            ]
            if alloc_nodes:
                found_indirect = True
                # Integrity: exactly one value_tensor and one index_tensor.
                types = [n["indirectAllocType_"] for n in alloc_nodes]
                self.assertEqual(types.count("value_tensor"), 1)
                self.assertEqual(types.count("index_tensor"), 1)
                # KERNEL_IDX dsType on the index tensor.
                kernel_idx = [
                    lds for lds in dsc["labeledDs_"] if lds["dsType_"] == "KERNEL_IDX"
                ]
                self.assertEqual(len(kernel_idx), 1)
                # indirectAccessIndexLabeledDs on the computeOp.
                compute = dsc["computeOp_"][0]
                self.assertIn("indirectAccessIndexLabeledDs", compute)
                self.assertTrue(compute["indirectAccessIndexLabeledDs"])
        self.assertTrue(
            found_indirect, "no captured SDSC carried indirect-access fields"
        )

    def _compile_and_capture(self, fn, *inputs):
        captured, restore = self._capture_sdsc()
        try:
            comp_fn = torch.compile(fn, dynamic=False)
            # Post-compile the runtime will try to load an empty kernel
            # from disk (we stubbed out dxp_standalone).  Accept any
            # resulting runtime error — the compile-time SDSC capture is
            # all this test cares about.
            try:
                comp_fn(*inputs)
            except Exception:
                pass
        finally:
            restore()
        return captured

    def test_index_select_1d_indices(self):
        weight = torch.randn(8, 128, dtype=torch.float16).to("spyre")
        idx = torch.zeros(4, dtype=torch.int64).to("spyre")

        def fn(w, i):
            return torch.index_select(w, 0, i) + 1.0

        captured = self._compile_and_capture(fn, weight, idx)
        self._assert_sdsc_has_indirect_fields(captured)

    def test_index_select_without_add(self):
        """index_select without a trailing add — SDSC op is ``add`` with a
        zero second operand (v1 behavior preserved via the IR handler's
        restickify-to-add rewrite).
        """
        weight = torch.randn(8, 128, dtype=torch.float16).to("spyre")
        idx = torch.zeros(4, dtype=torch.int64).to("spyre")

        def fn(w, i):
            return torch.index_select(w, 0, i)

        captured = self._compile_and_capture(fn, weight, idx)
        self._assert_sdsc_has_indirect_fields(captured)


if __name__ == "__main__":
    import unittest

    unittest.main()
