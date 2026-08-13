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

"""COMPILE-TIME repro for the invoke_subgraph subgraph-input STL MISALIGNMENT
(Blocker 3).

Root cause (device-proven, /tmp/wf_fscil.log): the Spyre layout pass converts
graph inputs with

    for name, real_input in zip(graph.graph_input_names, V.get_real_inputs())
    (propagate_layouts.py:1583)

When the pass runs on a *subgraph* GraphLowering (lazily, during the parent's
codegen), ``V.graph`` is correctly the subgraph, but ``V.real_inputs`` is NEVER
re-scoped -- it still holds the PARENT graph's real inputs. So the zip pairs the
subgraph's Nth graph-input with the PARENT's Nth real-input, which is a
different tensor. Its device layout (STL) is then stamped onto the wrong buffer.

In the Granite whole-forward this stamped selected_freqs' STL
([64,2,2,1,1,64]) onto the fused-QKV weight ([4096,2048]) -> the weight's
matmul could not restickify to carry y_var=d1 -> abort:

    Unsupported: batchmatmul: cannot restickify any input layout of y to
        carry y_var=d1   (propagate_layouts.py:804)

This repro provokes the SAME misalignment on the host with fake tensors (no
VFIO card). The region body holds a Linear whose WEIGHT becomes a subgraph
graph-input; the parent-scoped real-input at that position is a differently
shaped tensor, so the subgraph input receives a foreign STL and the block's
matmul aborts at compile time.

Runs on fake tensors during compile; codegen reaches the layout pass without
device execution, so this does NOT contend for the VFIO card. Contrast with
repro_invoke_subgraph_split.py, which covers the split_multi_ops assert
(Blockers 1/2) with a pointwise-only body.
"""

import traceback

import torch
from torch import nn
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  installs the inductor passes
from torch_spyre.constants import DEVICE_NAME


class Block(nn.Module):
    """Region body with a lifted WEIGHT input (the Linear) whose subgraph
    graph-input position collides with a differently shaped parent input."""

    def __init__(self, d_in, d_out):
        super().__init__()
        # Non-square projection so the weight's [d_out, d_in] shape is
        # distinct from the activation's shape -- a foreign STL stamped onto
        # it will not be restickify-compatible with its matmul.
        self.proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h):
        z = self.proj(h)  # matmul: h @ proj.weight.T, weight is a subgraph input
        return torch.relu(z * 3.0 + 1.0)


def region(block):
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


def main():
    b, s, d_in, d_out = 1, 64, 2048, 4096
    # Several distinctly shaped parent inputs so the subgraph's positional
    # inputs (activation + lifted weight) mis-zip against foreign tensors.
    blk = region(Block(d_in, d_out).to(dtype=torch.float16, device=DEVICE_NAME))

    def outer(h, w_back):
        z = blk(h)  # [b, s, d_out]
        # feed back through a distinctly shaped weight so the parent real-input
        # list has entries whose shape != the subgraph weight's shape
        return torch.matmul(z, w_back)  # [b, s, d_in]

    h = torch.randn(b, s, d_in, dtype=torch.float16, device=DEVICE_NAME)
    w_back = torch.randn(d_out, d_in, dtype=torch.float16, device=DEVICE_NAME)
    compiled = torch.compile(outer, dynamic=False, fullgraph=True)
    try:
        out = compiled(h, w_back)
        print("COMPILE OK, out.shape =", tuple(out.shape))
    except Exception:
        print(">>> EXCEPTION during compile:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
