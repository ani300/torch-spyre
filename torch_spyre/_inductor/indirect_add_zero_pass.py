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

"""FX pass that appends ``+ 0.0`` to indirect-indexing ops with no
consuming ``aten.add`` user.

Deeptools' SDSC ``add`` opcode requires two ``inputLabeledDs`` entries.
The Spyre ``ops.indirect_indexing`` lowering produces a single-input
compute op (just the indirectly-loaded value) unless a subsequent
``aten.add`` supplies the second operand.  This pass inserts a trivial
``+ 0.0`` after each un-consumed index op so that:

  * the post-grad ``convert_constant_with_graph_node`` pass promotes the
    ``0.0`` to a real ``spyre.constant.default`` tensor;
  * upstream Inductor's ``aten.add`` lowering fuses the gather and add
    into one pointwise, producing a two-input SDSC compute op
    (indirect value + broadcast zero) plus the ``indirectAccessIndexLabeledDs``.

Nodes that already feed into ``aten.add.Tensor`` / ``aten.add.Scalar``
are left untouched; the existing add supplies the second operand.
"""

from __future__ import annotations

import torch
import torch.fx


aten = torch.ops.aten


_INDIRECT_TARGETS = (
    aten.index_select.default,
    aten.index.Tensor,
    aten.embedding.default,
)

_ADD_TARGETS = (aten.add.Tensor, aten.add.Scalar)


def _has_add_user(node: torch.fx.Node) -> bool:
    for user in node.users:
        if user.op == "call_function" and user.target in _ADD_TARGETS:
            return True
    return False


def _should_rewrite(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.target not in _INDIRECT_TARGETS:
        return False
    if not node.users:
        # Dead node; skip.
        return False
    return not _has_add_user(node)


def _inject_add_zero(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    const_before: torch.fx.Node,
) -> None:
    """Insert ``aten.add.Tensor(node, spyre.constant(0, dtype, spyre))``
    after ``node`` and redirect users.

    The zero constant is placed **before** ``const_before`` (typically
    the first compute node of the graph) so that at scheduling time the
    extern ``SpyreConstantFallback`` appears *before* the gather op in
    the operations list. This prevents Spyre's bundle-level fusion pass
    (`spyre_fuse_nodes` in ``fusion.py``) from breaking the bundle
    between the gather and the add: the extern gets its own leading
    bundle, and the gather + add are free to fuse together downstream
    and emit a single SDSC with both inputs.

    We use ``spyre.constant.default`` directly instead of a Python
    scalar so that the constant appears as a distinct FX node whose
    insertion position we control (``convert_constant_with_graph_node``
    would otherwise place the promoted constant right before its
    consuming ``aten.add.Tensor``, interleaving it between the gather
    and the add).
    """
    val = node.meta.get("val")
    dtype = val.dtype if val is not None else torch.float16

    # Insert the constant node at the top of the graph (before any
    # compute).  The spyre.constant custom op takes (value, dtype, device)
    # and produces a size-1 tensor.
    with graph.inserting_before(const_before):
        zero_node = graph.call_function(
            torch.ops.spyre.constant.default,
            args=(0.0, dtype, torch.device("spyre")),
        )
    zero_node.meta["val"] = torch.zeros([1], dtype=dtype)

    # Now splice the add immediately after ``node``, redirecting users.
    original_users = list(node.users)
    with graph.inserting_after(node):
        add_node = graph.call_function(aten.add.Tensor, args=(node, zero_node))
    add_node.meta["val"] = val
    for user in original_users:
        user.replace_input_with(node, add_node)


def _first_non_placeholder(graph: torch.fx.Graph) -> torch.fx.Node | None:
    for n in graph.nodes:
        if n.op != "placeholder":
            return n
    return None


def indirect_add_zero_pass(graph: torch.fx.Graph) -> None:
    """Rewrite un-consumed indirect-indexing ops into ``add(op, const_zero)``.

    The injected ``spyre.constant`` zero is placed at the top of the
    graph so that at scheduling time the resulting ``SpyreConstantFallback``
    node precedes the gather — see ``_inject_add_zero`` for why ordering
    matters for Spyre's bundle fusion.
    """
    const_before = _first_non_placeholder(graph)
    if const_before is None:
        return
    changed = False
    for node in list(graph.nodes):
        if _should_rewrite(node):
            _inject_add_zero(graph, node, const_before)
            changed = True
    if changed:
        graph.lint()
