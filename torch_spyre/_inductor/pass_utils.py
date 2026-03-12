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

from typing import NamedTuple, Sequence

from sympy import Expr, Symbol

import sympy
from torch._inductor.ir import FixedLayout
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.views import compute_coordinates, compute_device_coordinates

from .ir import FixedTiledLayout


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout
    dev_coords: Sequence[sympy.Expr]


def propagate_view_stl(
    stl: SpyreTensorLayout,
    host_size: list[int],
    host_stride: list[int],
    new_size: list[int],
    new_stride: list[int],
) -> SpyreTensorLayout:
    """Compute a new SpyreTensorLayout from concrete host sizes and strides.

    This is used for eager-mode view operations (permute, transpose, etc.)
    where the new sizes and strides are known directly.
    """
    fixed_device_size, fixed_dim_map, fixed_it_device_dim_map = _compute_device_layout(
        host_size,
        host_stride,
        stl.device_size,
        stl.dim_map,
        new_size,
        new_stride,
    )
    print(fixed_device_size, fixed_dim_map, fixed_it_device_dim_map)
    return SpyreTensorLayout(
        fixed_device_size,
        [dm[0] for dm in fixed_it_device_dim_map],
        stl.device_dtype,
    )


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(layout.size, layout.stride, dep.ranges, dep.index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_device_coordinates(
        layout.size,
        layout.stride,
        layout.device_layout.device_size,
        layout.device_layout.dim_map,
        dep.ranges,
        dep.index,
    )


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            buffer_layout = buf.get_layout()

            if not isinstance(buffer_layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")

            host_coords = host_coordinates(buffer_layout, arg)
            dev_coords = device_coordinates(buffer_layout, arg)
            print(f"host_coords: {host_coords}; dev_coords: {dev_coords}")
            print(f"Buffer layout {buffer_layout.device_layout}")

            res.append(SchedNodeArg(arg, buffer_layout, dev_coords))
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
