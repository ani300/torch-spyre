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

"""Minimal COMPILE-TIME repro for the invoke_subgraph + split_multi_ops
graph-identity assertion.

A nested_compile_region block with a *fused multi-op* pointwise body
(relu(h*3+1) → mul,add,relu in one loop body, so split_multi_ops fires and
reaches its FX-node insertion), on Spyre-device tensors, called N times so
the region lowers to an invoke_subgraph HOP. Fails during torch.compile with:

    torch._inductor.exc.InductorError: AssertionError:
        Node to insert before is not in graph.
    (torch_spyre/_inductor/split_multi_ops.py:548 -> fx/graph.py:1651)

Root cause: the subgraph ComputedBuffer (repeated_subgraph0_buf0) has an
`origins` set that spans TWO fx.Graphs — the parent's invoke_subgraph /
get_attr nodes AND the subgraph-local `mul` node. split_multi_ops picks
`next(iter(op.origins))`, which returns the parent's invoke_subgraph node
(not in the subgraph's gl.graph), so gl.graph.inserting_before(orig_node)
asserts. The fix selects the origin whose `.graph is gl.graph`.

Runs on fake tensors during compile; codegen reaches the pass without device
execution, so this does NOT contend for the VFIO card.
"""
import traceback

import torch
from torch import nn
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  installs the inductor passes
from torch_spyre.constants import DEVICE_NAME


class Block(nn.Module):
    def forward(self, h):
        # Fused multi-op pointwise body: mul -> add -> relu in one loop body,
        # forcing split_multi_ops to materialize an intermediate and hit the
        # FX-node insertion at split_multi_ops.py:548.
        return torch.relu(h * 3.0 + 1.0)


def region(block):
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)
    return nested_compile_region(wrapper)


def main():
    blocks = [region(Block()) for _ in range(3)]

    def outer(h):
        for b in blocks:
            h = b(h)
        return h

    h = torch.randn(2, 64, dtype=torch.float16, device=DEVICE_NAME)
    compiled = torch.compile(outer, dynamic=False, fullgraph=True)
    try:
        out = compiled(h)
        print("COMPILE OK, out.shape =", tuple(out.shape))
    except Exception:
        print(">>> EXCEPTION during compile:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
