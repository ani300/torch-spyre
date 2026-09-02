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

# This file contains inductor passes that are only needed as temp fixes

from math import prod

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from .logging_utils import get_inductor_logger
from .pass_utils import copy_fx_custom_meta

aten = torch.ops.aten

logger = get_inductor_logger("work_division")

_RESHAPE_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)


def _shapes_statically_equal(lhs, rhs) -> bool:
    """Whether two shape sequences are provably equal without adding guards."""
    return len(lhs) == len(rhs) and statically_known_true(
        sym_eq(tuple(lhs), tuple(rhs))
    )


def _node_shape(node: torch.fx.Node) -> tuple | None:
    """Return an FX node's fake/meta shape, if one is available."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    shape = getattr(val, "shape", None)
    return tuple(shape) if shape is not None else None


mm_to_bmm_pass = PatternMatcherPass(pass_name="unflatten_mm_to_bmm")
shared_rhs_bmm_pass = PatternMatcherPass(pass_name="unexpand_shared_rhs_bmm")
bmm_unflatten_pass = PatternMatcherPass(pass_name="unflatten_bmm_batch_dims")


@register_graph_pattern(
    CallFunction(aten.mm.default, Arg(), Arg()),
    pass_dict=mm_to_bmm_pass,
)
def _unflatten_mm_to_bmm(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Convert view(3D→2D) → mm(2D, 2D) → view(2D→3D) into bmm(3D, unsqueeze(2D)).

    When torch.matmul is called with a batched input and a 2D weight, the
    decomposition flattens the batch dimensions:
      1. view(input, [B*M, K])
      2. mm(flattened, weight) -> [B*M, N]
      3. view(mm_result, [B, M, N])

    The Spyre backend handles bmm better. This pass converts the pattern into
    ``spyre.batched_matmul(input_3d, weight_2d)``.  Keeping the shared weight
    rank-2 preserves that it has no batch ownership; the Spyre lowering has a
    dedicated 3D-by-2D case.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs, rhs = mat1_node, mat2_node

    # LHS must be a reshape that flattens a higher-dim tensor to 2D
    if not (
        isinstance(lhs, torch.fx.Node)
        and lhs.op == "call_function"
        and lhs.target in _RESHAPE_OPS
    ):
        return
    lhs_input = lhs.args[0]
    if not (isinstance(lhs_input, torch.fx.Node) and "val" in lhs_input.meta):
        return
    lhs_orig_shape = list(lhs_input.meta["val"].shape)
    # This rewrite models exactly one batch axis: [B, M, K] @ [K, N].
    # A rank-4 (or higher) input flattened to mm has multiple logical batch/
    # feature axes.  Feeding that tensor directly to ``batched_matmul`` while
    # expanding the weight from ``shape[:-2]`` changes the matmul contract and
    # can make a small view appear to cover the weight's full M-by-K domain.
    # Higher-rank matmuls must remain mm here; ``bmm_unflatten_pass`` handles
    # the distinct case where both operands were flattened into aten.bmm.
    if len(lhs_orig_shape) != 3:
        return
    if "val" not in lhs.meta:
        return
    lhs_flat_shape = list(lhs.meta["val"].shape)
    if len(lhs_flat_shape) != 2:
        return

    # RHS must be a plain 2D tensor (not a reshaped one)
    if not (isinstance(rhs, torch.fx.Node) and "val" in rhs.meta):
        return
    rhs_shape = list(rhs.meta["val"].shape)
    if len(rhs_shape) != 2:
        return

    batch, rows, contraction = lhs_orig_shape
    rhs_contraction, columns = rhs_shape
    if not statically_known_true(sym_eq(contraction, rhs_contraction)):
        return
    if not _shapes_statically_equal(lhs_flat_shape, (batch * rows, contraction)):
        return

    # The mm result must feed into exactly one view that restores batch dims
    mm_users = list(node.users.keys())
    if len(mm_users) != 1:
        return
    output_view = mm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return
    if "val" not in output_view.meta:
        return
    output_shape = list(output_view.meta["val"].shape)
    if not _shapes_statically_equal(output_shape, (batch, rows, columns)):
        return

    # Keep the shared weight rank-2.  Materializing an expanded batch axis
    # gives a stride-zero tensor a non-unit logical batch role, which is not a
    # real device coordinate and must not participate in work ownership.
    with graph.inserting_before(node):
        bmm_node = graph.call_function(
            torch.ops.spyre.batched_matmul.default,
            args=(lhs_input, rhs),
        )
        bmm_node.meta["val"] = output_view.meta["val"]
        copy_fx_custom_meta(node, bmm_node)

    # Replace all uses of mm and output view with the bmm
    node.replace_all_uses_with(bmm_node)
    output_view.replace_all_uses_with(bmm_node)

    # Clean up dead nodes
    graph.erase_node(output_view)
    graph.erase_node(node)
    if not lhs.users:
        graph.erase_node(lhs)


def _node_stride(node: torch.fx.Node) -> tuple | None:
    """Return an FX node's fake/meta strides, if they are available."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    stride = getattr(val, "stride", None)
    return tuple(stride()) if callable(stride) else None


def _is_reshape_derived_matrix(node: torch.fx.Node) -> bool:
    """Whether a matrix view reaches storage through a reshape-like operation."""
    while isinstance(node, torch.fx.Node) and node.op == "call_function":
        if node.target in _RESHAPE_OPS:
            return True
        if node.target not in (
            aten.permute.default,
            aten.transpose.int,
            aten.t.default,
        ):
            break
        node = node.args[0]
    return False


@register_graph_pattern(
    CallFunction(aten.bmm.default, Arg(), Arg()),
    pass_dict=shared_rhs_bmm_pass,
)
def _unexpand_shared_rhs_bmm(
    match: Match, lhs_node: torch.fx.Node, rhs_node: torch.fx.Node
) -> None:
    """Recover a rank-2 RHS that matmul broadcast solely for ``aten.bmm``.

    PyTorch's matmul decomposition turns ``[B, M, K] @ [K, N]`` into a BMM
    whose RHS is ``expand(unsqueeze(rhs, 0), [B, K, N])``.  Keeping that
    synthetic axis through lowering loses the authoritative fact that the
    weight is shared, especially for B=1 where its stride need not be zero.
    Rewrite only the exact structural and shape/stride-preserving form to the
    backend's native 3D-by-2D matmul contract.
    """
    node = match.nodes[-1]
    graph = node.graph

    expanded = None
    unsqueezed = rhs_node
    if (
        isinstance(rhs_node, torch.fx.Node)
        and rhs_node.op == "call_function"
        and rhs_node.target == aten.expand.default
    ):
        expanded = rhs_node
        unsqueezed = rhs_node.args[0]

    if not (
        isinstance(unsqueezed, torch.fx.Node)
        and unsqueezed.op == "call_function"
        and unsqueezed.target == aten.unsqueeze.default
        and len(unsqueezed.args) >= 2
    ):
        return

    rhs_matrix = unsqueezed.args[0]
    if not isinstance(rhs_matrix, torch.fx.Node):
        return
    rhs_matrix_shape = _node_shape(rhs_matrix)
    lhs_shape = _node_shape(lhs_node)
    unsqueezed_shape = _node_shape(unsqueezed)
    rhs_shape = _node_shape(rhs_node)
    output_shape = _node_shape(node)
    if any(
        shape is None
        for shape in (
            lhs_shape,
            rhs_matrix_shape,
            unsqueezed_shape,
            rhs_shape,
            output_shape,
        )
    ):
        return
    assert lhs_shape is not None
    assert rhs_matrix_shape is not None
    assert unsqueezed_shape is not None
    assert rhs_shape is not None
    assert output_shape is not None

    if len(lhs_shape) != 3 or len(rhs_matrix_shape) != 2:
        return
    try:
        unsqueeze_dim = int(unsqueezed.args[1])
    except (TypeError, ValueError):
        return
    if unsqueeze_dim < 0:
        unsqueeze_dim += len(rhs_matrix_shape) + 1
    if unsqueeze_dim != 0 or _is_reshape_derived_matrix(rhs_matrix):
        return

    batch, rows, contraction = lhs_shape
    rhs_contraction, columns = rhs_matrix_shape
    if not statically_known_true(sym_eq(contraction, rhs_contraction)):
        return
    if not _shapes_statically_equal(unsqueezed_shape, (1, rhs_contraction, columns)):
        return
    if not _shapes_statically_equal(rhs_shape, (batch, rhs_contraction, columns)):
        return
    if not _shapes_statically_equal(output_shape, (batch, rows, columns)):
        return
    if expanded is None and not statically_known_true(sym_eq(batch, 1)):
        return

    # Validate address semantics as well as logical shapes.  Both view ops must
    # preserve the matrix strides, and an expanded non-unit batch must have no
    # storage advance along its synthetic axis.
    matrix_stride = _node_stride(rhs_matrix)
    unsqueezed_stride = _node_stride(unsqueezed)
    rhs_stride = _node_stride(rhs_node)
    if (
        matrix_stride is None
        or unsqueezed_stride is None
        or rhs_stride is None
        or not _shapes_statically_equal(unsqueezed_stride[1:], matrix_stride)
        or not _shapes_statically_equal(rhs_stride[1:], matrix_stride)
    ):
        return
    if not statically_known_true(sym_eq(batch, 1)) and not statically_known_true(
        sym_eq(rhs_stride[0], 0)
    ):
        return

    with graph.inserting_before(node):
        matmul_node = graph.call_function(
            torch.ops.spyre.batched_matmul.default,
            args=(lhs_node, rhs_matrix),
        )
        matmul_node.meta["val"] = node.meta["val"]
        copy_fx_custom_meta(node, matmul_node)

    node.replace_all_uses_with(matmul_node)
    graph.erase_node(node)
    if expanded is not None and not expanded.users:
        graph.erase_node(expanded)
    if not unsqueezed.users:
        graph.erase_node(unsqueezed)


def _is_batch_collapsing_reshape(node: torch.fx.Node) -> bool:
    """Check if a node is a reshape that collapses batch dims into a single dim."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function":
        return False
    if node.target not in _RESHAPE_OPS:
        return False
    # The reshape output should be 3D (batch_product, M, K)
    output_shape = node.args[1]
    if not isinstance(output_shape, (list, tuple)) or len(output_shape) != 3:
        return False
    # The input should be higher dimensional
    input_node = node.args[0]
    if isinstance(input_node, torch.fx.Node) and "val" in input_node.meta:
        input_ndim = input_node.meta["val"].dim()
        return input_ndim > 3
    return False


@register_graph_pattern(
    CallFunction(aten.bmm.default, Arg(), Arg()),
    pass_dict=bmm_unflatten_pass,
)
def _unflatten_bmm_batch_dims(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Undo the matmul decomposition's flattening of batch dimensions into 3D bmm.

    The matmul decomposition in torch/_decomp/decompositions.py converts N-D
    matmuls (e.g. 4D SDPA attention) into 3D by:
      1. expand(input, [B, H, M, K]) -> reshape([B*H, M, K])
      2. expand(input, [B, H, K, N]) -> reshape([B*H, K, N])
      3. bmm(reshaped1, reshaped2) -> [B*H, M, N]
      4. view(bmm_result, [B, H, M, N]) -> back to original dims

    This pass removes the reshape/view wrapper so the bmm operates on the
    original higher-dimensional tensors, which the Spyre backend can handle
    natively via its 4D batch matmul lowering.

    This is needed as the flattened views are not supported by the current
    backend. When KTIR is implemented this pass can be dropped.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs_reshape, rhs_reshape = mat1_node, mat2_node

    # Both inputs must be reshape/view that collapse batch dims to 3D
    if not _is_batch_collapsing_reshape(lhs_reshape):
        return
    if not _is_batch_collapsing_reshape(rhs_reshape):
        return

    # The bmm result must feed into exactly one view that restores the batch dims
    bmm_users = list(node.users.keys())
    if len(bmm_users) != 1:
        return
    output_view = bmm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return

    # Get the original (pre-reshape) tensors
    lhs_orig = lhs_reshape.args[0]  # the expand or original tensor
    rhs_orig = rhs_reshape.args[0]

    # Prove the entire reshape sandwich before removing it.  Equal element
    # counts are not enough: a reshape can collapse different logical batch
    # prefixes, interchange M/K/N, or restore the result in a different order.
    # Reusing those operands directly would then make lower_bmm index one
    # producer with another operand's matrix domain.
    lhs_orig_shape = _node_shape(lhs_orig)
    rhs_orig_shape = _node_shape(rhs_orig)
    lhs_flat_shape = _node_shape(lhs_reshape)
    rhs_flat_shape = _node_shape(rhs_reshape)
    bmm_shape = _node_shape(node)
    output_shape = _node_shape(output_view)
    if (
        lhs_orig_shape is None
        or rhs_orig_shape is None
        or lhs_flat_shape is None
        or rhs_flat_shape is None
        or bmm_shape is None
        or output_shape is None
    ):
        return

    # lower_bmm currently has a native contract for exactly two batch axes.
    # Leave other ranks as the original, semantically valid flattened bmm.
    if len(lhs_orig_shape) != 4 or len(rhs_orig_shape) != 4:
        return

    lhs_batch = lhs_orig_shape[:-2]
    rhs_batch = rhs_orig_shape[:-2]
    lhs_rows, lhs_contraction = lhs_orig_shape[-2:]
    rhs_contraction, rhs_columns = rhs_orig_shape[-2:]
    flat_batch = prod(lhs_batch)

    if not _shapes_statically_equal(lhs_batch, rhs_batch):
        return
    if not statically_known_true(sym_eq(lhs_contraction, rhs_contraction)):
        return
    if not _shapes_statically_equal(
        lhs_flat_shape, (flat_batch, lhs_rows, lhs_contraction)
    ):
        return
    if not _shapes_statically_equal(
        rhs_flat_shape, (flat_batch, rhs_contraction, rhs_columns)
    ):
        return
    if not _shapes_statically_equal(bmm_shape, (flat_batch, lhs_rows, rhs_columns)):
        return
    if not _shapes_statically_equal(output_shape, (*lhs_batch, lhs_rows, rhs_columns)):
        return

    # Replace the 3D bmm with a spyre.batched_matmul that accepts N-D inputs.
    # Using aten.bmm.default with >3D args would crash FakeTensorUpdater.
    with graph.inserting_before(node):
        matmul_node = graph.call_function(
            torch.ops.spyre.batched_matmul.default,
            args=(lhs_orig, rhs_orig),
        )
        matmul_node.meta["val"] = output_view.meta["val"]
        copy_fx_custom_meta(node, matmul_node)

    # Replace all uses of the output view with the new matmul
    output_view.replace_all_uses_with(matmul_node)
    node.replace_all_uses_with(matmul_node)
    graph.erase_node(output_view)
    graph.erase_node(node)

    # Clean up dead reshape nodes
    for reshape_node in (lhs_reshape, rhs_reshape):
        if not reshape_node.users:
            expand_node = reshape_node.args[0]
            graph.erase_node(reshape_node)
            # Also remove the expand if it's now unused
            if (
                isinstance(expand_node, torch.fx.Node)
                and expand_node.op == "call_function"
                and expand_node.target == aten.expand.default
                and not expand_node.users
            ):
                graph.erase_node(expand_node)


def decompose_addmm(graph: torch.fx.Graph) -> None:
    """Decompose ``aten.addmm.default`` into ``add(scaled_input, alpha*mm)``.

    Inductor's post-grad pattern matcher re-fuses ``add(input, mm(a, b))`` back
    into ``aten.addmm.default`` after AOTAutograd, defeating the upstream
    decomposition. With no Spyre lowering for ``addmm``, the op then falls
    back to ``extern_kernels.addmm`` which produces an ``ExternKernelOut``
    without a ``FixedTiledLayout`` and breaks subsequent Spyre passes.

    This pass undoes the re-fusion at FX time so the resulting ``mm``,
    ``mul`` and ``add`` nodes flow through the existing Spyre lowerings.
    Any ``alpha`` / ``beta`` scalars become ``aten.mul.Scalar`` nodes whose
    scalar constants are later materialized into ``spyre.constant`` tensors by
    the LoopLevel IR multi-ops pass (``split_multi_ops``).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.addmm.default:
            continue
        input_node, mat1, mat2 = node.args[0], node.args[1], node.args[2]
        beta = node.kwargs.get("beta", 1)
        alpha = node.kwargs.get("alpha", 1)

        out_meta = node.meta.get("val", None)

        with graph.inserting_before(node):
            mm_node = graph.call_function(aten.mm.default, args=(mat1, mat2))
            if out_meta is not None:
                mm_node.meta["val"] = torch.empty_like(out_meta, device="meta")
            copy_fx_custom_meta(node, mm_node)

            scaled_mm = mm_node
            if alpha != 1:
                scaled_mm = graph.call_function(aten.mul.Scalar, args=(mm_node, alpha))
                if out_meta is not None:
                    scaled_mm.meta["val"] = torch.empty_like(out_meta, device="meta")
                copy_fx_custom_meta(node, scaled_mm)

            if beta == 0:
                replacement = scaled_mm
            else:
                scaled_input = input_node
                if beta != 1:
                    scaled_input = graph.call_function(
                        aten.mul.Scalar, args=(input_node, beta)
                    )
                    in_meta = (
                        input_node.meta.get("val", None)
                        if isinstance(input_node, torch.fx.Node)
                        else None
                    )
                    if in_meta is not None:
                        scaled_input.meta["val"] = torch.empty_like(
                            in_meta, device="meta"
                        )
                    copy_fx_custom_meta(node, scaled_input)

                replacement = graph.call_function(
                    aten.add.Tensor, args=(scaled_input, scaled_mm)
                )
                if out_meta is not None:
                    replacement.meta["val"] = torch.empty_like(out_meta, device="meta")
                copy_fx_custom_meta(node, replacement)

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)

    graph.lint()
