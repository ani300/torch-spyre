# Copyright 2026 The Torch-Spyre Authors.
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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch_spyre._inductor.insert_restickify as spyre_restickify
import torch_spyre._inductor.lowering as spyre_lowering
from torch_spyre._inductor.constants import MATMUL_OPERANDS_INFO_KEY
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.insert_restickify import InputEdgeSwapHandler
from torch_spyre._inductor.lowering import (
    _matmul_batch_dim_owners,
    _matmul_operands_info,
)
from torch_spyre._inductor.optimize_restickify import (
    INF,
    FixedInOutNode,
)


class _FakeOperand:
    def __init__(self, stride, data=None):
        self._stride = stride
        self.data = data

    def get_stride(self):
        return self._stride


class TestMatmulOperandMetadata(unittest.TestCase):
    def test_batch_ownership_uses_stride_not_logical_extent(self):
        batched = _FakeOperand((8192, 64, 1))
        shared = _FakeOperand((0, 64, 1))

        self.assertEqual(_matmul_batch_dim_owners(batched, (32, 128, 64)), (True,))
        self.assertEqual(_matmul_batch_dim_owners(shared, (32, 128, 64)), (False,))

    def test_partial_batch_broadcast_keeps_other_owned_axis(self):
        operand = _FakeOperand((8192, 0, 64, 1))
        self.assertEqual(
            _matmul_batch_dim_owners(operand, (2, 32, 128, 64)),
            (True, False),
        )

    def test_unit_inserted_axis_uses_underlying_storage_rank(self):
        class FakeBaseView:
            def __init__(self, base_shape):
                self.data = SimpleNamespace(get_size=lambda: base_shape)

        shared = _FakeOperand((8192, 64, 1), data=FakeBaseView((128, 64)))
        reshaped = _FakeOperand((8192, 64, 1), data=FakeBaseView((4, 2048)))
        with patch.object(spyre_lowering.ir, "BaseView", FakeBaseView):
            self.assertEqual(_matmul_batch_dim_owners(shared, (1, 128, 64)), (False,))
            self.assertEqual(_matmul_batch_dim_owners(reshaped, (1, 128, 64)), (True,))

    def test_inserted_axis_can_be_derived_when_stride_is_unavailable(self):
        class FakeBaseView:
            def __init__(self, base_shape):
                self.data = SimpleNamespace(get_size=lambda: base_shape)

        shared = _FakeOperand(None, data=FakeBaseView((128, 64)))
        with patch.object(spyre_lowering.ir, "BaseView", FakeBaseView):
            self.assertEqual(_matmul_batch_dim_owners(shared, (1, 128, 64)), (False,))

    def test_unavailable_stride_fails_instead_of_assuming_owned(self):
        with self.assertRaisesRegex(Unsupported, "unresolved batch axes"):
            _matmul_batch_dim_owners(_FakeOperand(None), (4, 128, 64))

    def test_mismatched_stride_rank_fails_instead_of_assuming_owned(self):
        with self.assertRaisesRegex(Unsupported, "stride rank"):
            _matmul_batch_dim_owners(_FakeOperand((128, 1)), (4, 128, 64))

    def test_symbolic_nonzero_stride_is_owned(self):
        unknown = sympy.Symbol("unknown_stride", integer=True)
        self.assertEqual(
            _matmul_batch_dim_owners(_FakeOperand((unknown, 64, 1)), (4, 128, 64)),
            (True,),
        )

    def test_info_keeps_ownership_positional(self):
        lhs = _FakeOperand((512, 128, 1))
        rhs = _FakeOperand((0, 64, 1))
        info = _matmul_operands_info(lhs, rhs, (8, 4, 128), (8, 128, 64), (8, 4, 64))[
            MATMUL_OPERANDS_INFO_KEY
        ]

        self.assertEqual(info["batch_dim_owners"], ((True,), (False,)))


class _FakeEdge:
    def __init__(self, name, costs):
        self.dep = SimpleNamespace(name=name)
        self._costs = costs
        self._in_layouts = tuple(costs)

    def cost(self, source, target):
        return self._costs[source].get(target, INF)


class TestDuplicateOperandCosts(unittest.TestCase):
    def test_fixed_input_cost_accounts_for_every_same_name_operand(self):
        source = object()
        lhs_required = object()
        rhs_required = object()
        output = object()
        lhs = _FakeEdge("shared", {source: {lhs_required: 3.0}})
        rhs = _FakeEdge("shared", {source: {rhs_required: 5.0}})
        node = FixedInOutNode(
            [lhs, rhs],
            required_out_stl=output,
            required_in_stls=[lhs_required, rhs_required],
        )

        self.assertEqual(node.min_input_cost("shared", source, output), 8.0)
        # The optimizer walks semantic edge_costs (not deduplicated ReadWrites),
        # so an exact-alias producer layout is supplied once per operand role.
        self.assertEqual(node.cost([source, source], output), 8.0)

    def test_fixed_input_cost_rejects_if_any_same_name_role_is_infeasible(self):
        source = object()
        lhs_required = object()
        rhs_required = object()
        output = object()
        lhs = _FakeEdge("shared", {source: {lhs_required: 3.0}})
        rhs = _FakeEdge("shared", {source: {rhs_required: INF}})
        node = FixedInOutNode(
            [lhs, rhs],
            required_out_stl=output,
            required_in_stls=[lhs_required, rhs_required],
        )

        self.assertEqual(node.min_input_cost("shared", source, output), INF)

    def test_fixed_input_cost_rejects_unexplained_cardinality_mismatch(self):
        source = object()
        required = object()
        output = object()
        node = FixedInOutNode(
            [
                _FakeEdge("first", {source: {required: 1.0}}),
                _FakeEdge("second", {source: {required: 1.0}}),
            ],
            required_out_stl=output,
            required_in_stls=[required, required],
        )

        with self.assertRaisesRegex(ValueError, "cardinality"):
            node.cost([source], output)


class TestInputEdgeSwapHandler(unittest.TestCase):
    class _Loads:
        def __init__(self):
            self.names = []

        def load(self, name, index):
            self.names.append(name)
            return name, index

    def test_single_target_redirects_all_occurrences(self):
        """When only one unique target exists for a (name, index) pair, the
        upstream handler (#4176) redirects every occurrence to that target —
        not just the one whose occurrence number matches the swap entry."""
        index = sympy.Symbol("k")
        loads = self._Loads()
        handler = InputEdgeSwapHandler(
            loads,
            [("shared", index, 1, "resticked_rhs")],
        )

        self.assertEqual(handler.load("shared", index)[0], "resticked_rhs")
        self.assertEqual(handler.load("shared", index)[0], "resticked_rhs")
        self.assertEqual(loads.names, ["resticked_rhs", "resticked_rhs"])

    def test_same_name_distinct_indices_select_distinct_targets(self):
        row, column = sympy.symbols("row column")
        loads = self._Loads()
        handler = InputEdgeSwapHandler(
            loads,
            [
                ("shared", row, 0, "resticked_lhs"),
                ("shared", column, 0, "resticked_rhs"),
            ],
        )

        self.assertEqual(handler.load("shared", row)[0], "resticked_lhs")
        self.assertEqual(handler.load("shared", column)[0], "resticked_rhs")

    def test_equivalent_index_from_fresh_trace_matches_canonical_dependency(self):
        output, reduction = sympy.symbols("d0 d1")
        live_output, live_reduction = sympy.symbols("q0 q1")
        loads = self._Loads()
        handler = InputEdgeSwapHandler(
            loads,
            [("activation", reduction, 0, "resticked_activation")],
            index_replacements={
                live_output: output,
                live_reduction: reduction,
            },
        )

        self.assertEqual(
            handler.load("activation", live_reduction)[0],
            "resticked_activation",
        )

    def test_fresh_trace_keeps_distinct_same_name_edges_positional(self):
        row, column = sympy.symbols("d0 d1")
        live_row, live_column = sympy.symbols("q0 q1")
        loads = self._Loads()
        handler = InputEdgeSwapHandler(
            loads,
            [
                ("shared", row, 0, "resticked_lhs"),
                ("shared", column, 0, "resticked_rhs"),
            ],
            index_replacements={live_row: row, live_column: column},
        )

        self.assertEqual(handler.load("shared", live_row)[0], "resticked_lhs")
        self.assertEqual(handler.load("shared", live_column)[0], "resticked_rhs")

    def test_insertion_retrace_redirects_exact_alias_edges_to_single_target(self):
        """When two loads read the same buffer with the same index, ReadWrites
        deduplicates them into one MemoryDep.  The upstream handler (#4176)
        redirects all occurrences of a (name, index) pair with one unique
        target to that target, so both loads end up reading the restickified
        buffer and get_read_writes collapses them into one dep."""
        graph = GraphLowering(fx.symbolic_trace(lambda: None))
        device = torch.device("cpu")
        dtype = torch.float32

        with V.set_graph_handler(graph):

            def shared_twice(index):
                i, j = index
                offset = 8 * i + j
                lhs = V.ops.load("shared", offset)
                rhs = V.ops.load("shared", offset)
                return V.ops.add(lhs, rhs)

            pointwise = Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=shared_twice,
                ranges=[8, 8],
            )
            consumer = ComputedBuffer(
                name="consumer",
                layout=FixedLayout(device, dtype, [8, 8], [8, 1]),
                data=pointwise.data.data,
            )
            consumer.operation_name = "consumer"
            consumer.origins = OrderedSet()
            graph.name_to_buffer[consumer.get_name()] = consumer

            original_reads = list(consumer.get_read_writes().reads)
            self.assertEqual(len(original_reads), 1)
            self.assertEqual(original_reads[0].name, "shared")

            restick_pointwise = Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=lambda index: V.ops.load("shared", 8 * index[0] + index[1]),
                ranges=[8, 8],
            )
            resticked = ComputedBuffer(
                name="resticked_rhs",
                layout=FixedLayout(device, dtype, [8, 8], [8, 1]),
                data=restick_pointwise.data.data,
            )
            resticked.operation_name = "resticked_rhs"
            resticked.origins = OrderedSet()
            operations = [consumer]

            def create_restickify(restick_arg_info, op):
                self.assertIs(op, consumer)
                self.assertEqual(restick_arg_info["occurrence"], 1)
                graph.name_to_buffer[resticked.get_name()] = resticked
                operations.append(resticked)
                return "shared", resticked

            restick_plan = [
                {
                    "arg_name": "shared",
                    "dep_index": original_reads[0].index,
                    "occurrence": 1,
                    "target_layout": object(),
                }
            ]
            with patch.object(
                spyre_restickify,
                "_create_restickify_node",
                side_effect=create_restickify,
            ):
                spyre_restickify.insert_restickify_on_node_inputs(
                    consumer, restick_plan, operations
                )

            self.assertIs(operations[0], resticked)
            reconstructed = operations[1]
            self.assertIsNot(reconstructed, consumer)

            reconstructed_reads = list(reconstructed.get_read_writes().reads)
            self.assertEqual(
                [read.name for read in reconstructed_reads],
                ["resticked_rhs"],
            )

            _, body, _ = reconstructed.get_default_sizes_body()
            generated_loads = [
                entry.buffer_name for entry in body.memory_usage[MemoryUsageType.LOAD]
            ]
            self.assertEqual(generated_loads, ["resticked_rhs", "resticked_rhs"])


if __name__ == "__main__":
    unittest.main()
