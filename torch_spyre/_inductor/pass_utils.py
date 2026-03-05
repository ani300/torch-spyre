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

from typing import NamedTuple

from sympy import Expr, Symbol

import sympy
from torch._inductor.ir import FixedLayout
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout, compute_view_layout
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def compute_permute_stl(
    dims: list, starting_stl: SpyreTensorLayout
) -> SpyreTensorLayout:
    ndims = len(dims)

    inv_perm = [0] * ndims
    for new_pos, old_pos in enumerate(dims):
        inv_perm[old_pos] = new_pos

    new_dim_map = [inv_perm[dim] for dim in starting_stl.dim_map]

    return SpyreTensorLayout(
        starting_stl.device_size, new_dim_map, starting_stl.device_dtype
    )


def compute_transpose_stl(
    dim0: int, dim1: int, starting_stl: SpyreTensorLayout
) -> SpyreTensorLayout:
    dim_map = starting_stl.dim_map
    for idx, dim in enumerate(dim_map):
        if dim == dim0:
            dim_map[idx] = dim1
        elif dim == dim1:
            dim_map[idx] = dim0
    return SpyreTensorLayout(
        starting_stl.device_size, dim_map, starting_stl.device_dtype
    )


# This partial_view_info dict contains the history of the view ops originating
# from a realized buffer for a specific compilation
# This is used to compute the device layouts for op layout propagation and also for
# OpSpec generation later on.


# The list of view op infos will contain a dictionary with info for each view.
# For example, for a permute view op:
# {
#   "type": "permute",
#   "dims": [3, 1, 2, 0],
#   "new_layout": TensorBox.get_layout()
# }
# With this information, we can compute the new STL from a starting STL
def propagate_view_stl(
    view_op_list: list, starting_stl: SpyreTensorLayout
) -> SpyreTensorLayout:
    new_stl = starting_stl
    for view_op_info in view_op_list:
        if view_op_info["type"] == "permute":
            new_stl = compute_permute_stl(view_op_info["dims"], new_stl)
        elif view_op_info["type"] == "transpose":
            new_stl = compute_transpose_stl(
                view_op_info["dim0"], view_op_info["dim1"], new_stl
            )
        elif view_op_info["type"] == "view":
            old_sizes = tuple([int(s) for s in view_op_info["old_sizes"]])
            new_sizes = tuple([int(s) for s in view_op_info["new_sizes"]])
            new_stl = compute_view_layout(old_sizes, new_sizes, new_stl)
        elif view_op_info["type"] == "squeeze":
            old_sizes = tuple([int(s) for s in view_op_info["old_sizes"]])
            new_sizes = tuple([int(s) for s in view_op_info["new_sizes"]])
            new_stl = compute_view_layout(old_sizes, new_sizes, new_stl)
        elif view_op_info["type"] == "unsqueeze":
            old_sizes = tuple([int(s) for s in view_op_info["old_sizes"]])
            new_sizes = tuple([int(s) for s in view_op_info["new_sizes"]])
            new_stl = compute_view_layout(old_sizes, new_sizes, new_stl)
        else:
            raise Unsupported("This view op is not supported in stickification yet")
    return new_stl


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")

            # TODO: Add check that the index matches the final
            # layout in the views to ensure it's the right tree
            if (
                getattr(V.graph, "partial_view_info", None)
                and arg.name in V.graph.partial_view_info
            ):
                # Apply all the views in order to obtain the final STL for the FTL
                new_stl = propagate_view_stl(
                    V.graph.partial_view_info[arg.name], layout.device_layout
                )
                layout.device_layout = new_stl
                print("Updated layout")
            print(f"Final layout {layout.device_layout}")

            res.append(SchedNodeArg(arg, layout))
    return res


def wildcard_symbol(dim) -> Symbol:
    return sympy.Symbol(f"*_{dim}")


def is_wildcard(s: Symbol) -> bool:
    return s.name.startswith("*_")


def map_dims_to_vars(layout: FixedLayout, index: Expr) -> dict[int, Symbol]:
    """
    Construct a mapping from the dimensions of layout
    to the free variables of index that correspond to them.
    Dimensions of size 1 are mapped to a wild_card_symbol of `*`

    This works by reversing the algorithm used by torch._inductor.ir. _fixed_indexer to build index.
    """
    result = {}
    for sym in index.free_symbols:
        stride_val = sympy_subs(index, {sym: 1}) - sympy_subs(index, {sym: 0})
        if stride_val in layout.stride:
            idx = layout.stride.index(stride_val)
            result[idx] = sym

    for d in range(len(layout.size)):
        if d not in result:
            assert layout.size[d] == 1, "non-trivial dim missing from index expression"
            result[d] = wildcard_symbol(d)

    return result
