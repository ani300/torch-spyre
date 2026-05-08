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

    def test_index_tensor_data_format_is_senuint32(self):
        """The index tensor's data format is rewritten to SENUINT32.

        The senulator's L3_LDIMU handler asserts
        ``srcStick.getType() == DataFormats::SENUINT32``
        (``deeptools/senulator/memoryElement.cpp:768``) and the
        reference SDSC from the paged-attention flow also uses
        SENUINT32 for its KERNEL_IDX lds.  IEEE_INT32 is silently
        mis-decoded and the IBR ends up holding garbage.
        """
        op_spec = _make_indirect_op_spec()
        sdsc_spec = parse_op_spec(op_spec)
        self.assertEqual(
            sdsc_spec.args[1].data_format, DataFormats.SENUINT32
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


def _gather_idx_convert(
    raw_indices: torch.Tensor,
    value_base_addr: int,
    value_shape: list[int],
    layout_dim_order: list[str],
    dim_ids: list[str],
    stick_dims: list[str],
    stick_sizes: list[int],
    gather_dim_id: str,
    is_sen1p5: bool = False,
) -> torch.Tensor:
    """Port of ``Dsm::fillGiiAttributes`` + ``ConvertData_gather_idx``.

    See ``deeptools/dsm/host_node_senops.cpp:2816`` (attribute fill) and
    ``deeptools/util/sen_data_convert.cpp:3022`` (per-index transform).
    Converts plain integer indices into the absolute device addresses
    the IBR expects.

    This is a quick/dirty host-side transform used only to validate the
    end-to-end flow; the proper fix is to emit a ``ConvertData_gather_idx``
    node in the bundle itself so the runtime handles the conversion.
    """
    raw = raw_indices.cpu().to(torch.int64).tolist()

    # fillGiiAttributes:
    #   dimIdToElems[dim] = shape[i]
    #   dimIdToShape[dim] = shape[i] (/ stickSize for stick dims)
    dim_id_to_elems = {d: value_shape[i] for i, d in enumerate(dim_ids)}
    dim_id_to_shape = dict(dim_id_to_elems)
    for sd, ss in zip(stick_dims, stick_sizes):
        dim_id_to_shape[sd] = dim_id_to_shape[sd] // ss

    base_addr = value_base_addr
    if is_sen1p5:
        assert base_addr % 2 == 0
        base_addr //= 2

    # Compute skip_addr_ / idx_prev_cum_size_ for the (single) gather dim.
    # ``layout_dim_order`` is inner → outer.  skip_addr accumulates the
    # stick-unit sizes of all dims INNER to the gather dim.
    idx_prev_cum_sizes: list[int] = []
    skip_addrs: list[int] = []
    prev_cum = 1
    idx_prev_cum_sizes.append(prev_cum)
    skip = 1
    for entry in layout_dim_order:
        if entry == gather_dim_id:
            prev_cum *= dim_id_to_elems[entry]
            break
        skip *= dim_id_to_shape[entry]
    if is_sen1p5:
        assert skip % 2 == 0
        skip //= 2
    skip_addrs.append(skip)

    # ConvertData_gather_idx (single-dim case):
    #   addr[j] = base_addr + idx_val[j] * skip_addr[0]
    out = [base_addr + int(v) * skip_addrs[0] for v in raw]
    return torch.tensor(out, dtype=torch.int64)


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
        """Across all captured SDSCs, assert at least one has the indirect fields.

        Beyond the indirect-access fields (``indirectAllocType_``,
        ``KERNEL_IDX`` dsType, ``indirectAccessIndexLabeledDs``), asserts
        that ``computeOp_`` has the two-operand shape deeptools' ``add``
        opcode requires: two ``inputLabeledDs`` entries (the indirect
        value + the broadcast-zero constant injected by
        ``indirect_add_zero_pass``) and ``opFuncName == "add"``.
        """
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
                # computeOp_ shape: two inputLabeledDs (value + broadcast
                # zero), indirectAccessIndexLabeledDs present, opFuncName
                # == "add".
                compute = dsc["computeOp_"][0]
                self.assertEqual(compute["opFuncName"], "add")
                self.assertEqual(
                    len(compute["inputLabeledDs"]),
                    2,
                    "indirect-access SDSC must have 2 inputLabeledDs "
                    "entries (value + broadcast-zero)",
                )
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
        # Shapes chosen so the scheduler uses all 32 cores.  Smaller
        # inputs produce a degenerate fold distribution that deeptools
        # rejects downstream.
        weight = torch.randn(1024, 2048, dtype=torch.float16).to("spyre")
        idx = torch.zeros(512, dtype=torch.int64).to("spyre")

        def fn(w, i):
            return torch.index_select(w, 0, i) + 1.0

        captured = self._compile_and_capture(fn, weight, idx)
        self._assert_sdsc_has_indirect_fields(captured)

    def test_index_select_without_add(self):
        """index_select without a user-written trailing add.

        ``indirect_add_zero_pass`` appends ``+ 0.0`` at the FX level so
        the emitted SDSC has the two-input shape deeptools requires.

        Indices are pre-transformed host-side by the python port of
        ``Dsm::fillGiiAttributes`` + ``ConvertData_gather_idx`` so the
        IBR receives absolute device addresses.  This is a temporary
        test fixture; the proper fix is to emit a
        ``ConvertData_gather_idx`` node in the bundle itself.
        """
        weight = torch.randn(1024, 2048, dtype=torch.float16).to("spyre")
        # Weight tensor at arg_index=1 → segment 1 at 0x400000000.
        value_base_addr = 0x400000000
        # For a 2-D fp16 [1024, 2048] tensor with row-major layout and the
        # innermost dim (out) sticked at 64 elems: layoutDimOrder is
        # inner→outer = [out, mb].  Gather dim = mb (row).
        #
        #   dim_id_to_shape = {out: 32 sticks, mb: 1024 rows}
        #   skip_addr for mb = dim_id_to_shape[out] = 32 (sticks per row)
        #   addr[i] = base_addr + idx[i] * 32
        raw_idx = torch.zeros(512, dtype=torch.int64)
        idx_addrs = _gather_idx_convert(
            raw_indices=raw_idx,
            value_base_addr=value_base_addr,
            value_shape=[2048, 1024],  # matches dim_ids order = [out, mb]
            layout_dim_order=["out", "mb"],
            dim_ids=["out", "mb"],
            stick_dims=["out"],
            stick_sizes=[64],
            gather_dim_id="mb",
            is_sen1p5=False,
        )
        idx = idx_addrs.to("spyre")

        def fn(w, i):
            return torch.index_select(w, 0, i)

        captured = self._compile_and_capture(fn, weight, idx)
        self._assert_sdsc_has_indirect_fields(captured)

    def test_index_select_sdsc_matches_baseline_shape(self):
        """The indirect-access SDSC mirrors the plain-add baseline's
        ``computeOp_`` operand shape (2 inputs, 1 output, sfp exUnit).
        """
        weight = torch.randn(1024, 2048, dtype=torch.float16).to("spyre")
        idx = torch.zeros(512, dtype=torch.int64).to("spyre")

        def fn(w, i):
            return torch.index_select(w, 0, i)

        captured = self._compile_and_capture(fn, weight, idx)
        indirect_dscs = []
        for sdsc in captured:
            key = next(iter(sdsc))
            dsc_outer = sdsc[key]["dscs_"][0]
            inner_key = next(iter(dsc_outer))
            dsc = dsc_outer[inner_key]
            if any(
                n.get("indirectAllocType_") == "value_tensor"
                for n in dsc["scheduleTree_"]
            ):
                indirect_dscs.append(dsc)
        self.assertEqual(
            len(indirect_dscs),
            1,
            "expected exactly one indirect-access DSC",
        )
        compute = indirect_dscs[0]["computeOp_"][0]
        self.assertEqual(compute["exUnit"], "sfp")
        self.assertEqual(compute["opFuncName"], "add")
        self.assertEqual(len(compute["inputLabeledDs"]), 2)
        self.assertEqual(len(compute["outputLabeledDs"]), 1)
        self.assertEqual(len(compute["indirectAccessIndexLabeledDs"]), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
