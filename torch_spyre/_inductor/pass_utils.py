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
from torch._inductor.dependencies import index_vars_no_squeeze
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout

from .ir import FixedTiledLayout


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def _compute_device_layout(
    size: list[int],
    stride: list[int],
    device_size: list[int],
    dim_map: list[int],
    ranges: list[int],
    steps: list[int],
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Core algorithm for computing a new device layout from indexing info.

    Given the host tensor's size/stride, its device layout (device_size/dim_map),
    and the iteration ranges and per-variable steps of an indexing expression,
    compute (fixed_device_size, fixed_dim_map) for the viewed tensor.

    Args:
        size: Host tensor sizes.
        stride: Host tensor strides.
        device_size: Device dimension sizes from SpyreTensorLayout.
        dim_map: Device-to-host dimension mapping from SpyreTensorLayout.
        ranges: Iteration range for each variable.
        steps: Stride (coefficient) for each variable in the index expression.

    Returns:
        (fixed_device_size, fixed_dim_map) for the new SpyreTensorLayout.
    """
    # Stage 1: compute split
    # split[i] is the stride of device dim i w.r.t. host dim dim_map[i]
    s = [1] * len(size)
    split = [0] * len(dim_map)
    for i in range(len(dim_map) - 1, -1, -1):
        j = dim_map[i]
        split[i] = s[j]
        s[j] *= device_size[i]

    # Stage 2: compute it_host_dim_map
    # For each var, find which host dimensions it indexes into
    num_vars = len(ranges)
    it_host_dim_map: list[list[tuple[int, int, int]]] = [
        [] for _ in range(num_vars)  # type: ignore[list-item]
    ]
    for i in range(num_vars):
        step = steps[i]
        if step == 0 or ranges[i] == 1:
            continue  # var does not occur in indexer or has range 1
        it_host_dim_map[i] = [()]  # expect at least one match
        limit = step * ranges[i]
        max_stride_below = 0
        for j in range(len(size)):
            if size[j] == 1:
                continue
            sj = stride[j]
            if sj > step and sj < limit:
                it_host_dim_map[i].append((j, 1, sj // step))
            elif sj <= step and sj > max_stride_below:
                max_stride_below = sj
                it_host_dim_map[i][0] = (j, step // max_stride_below, 1)

    # Stage 3: compute it_device_dim_map
    # For each var's host dim mappings, find corresponding device dimensions
    it_device_dim_map: list[list[tuple[int, int, int]]] = [[] for _ in range(num_vars)]
    for i in range(num_vars):
        for entry in it_host_dim_map[i]:
            if not entry:
                continue
            j, num, den = entry
            for k in range(len(dim_map)):
                if j != dim_map[k]:
                    continue  # device dim k does not map to host dim j
                if (
                    ranges[i] * num // den > split[k]
                    and num // den < split[k] * device_size[k]
                ):
                    if num // den // split[k] > 0:
                        it_device_dim_map[i].append((k, num // den // split[k], 1))
                    else:
                        it_device_dim_map[i].append((k, 1, split[k] * den // num))

    # Stage 4: fix device layout
    # Collect indexing contributions per device dim, sort by stride (desc),
    # then split device dims with multiple contributors
    
    # order indexing terms for each coordinate in decreasing stride order
    terms: list[list[tuple[int, int, int | None, int | None]]] = [[] for _ in range(len(device_size))]
    for i in range(len(device_size)):
        if device_size[i] == 1:
            terms[i].append((1, i, None, None))
            continue
        for k in range(len(it_device_dim_map)):
            for j, num, den in it_device_dim_map[k]:
                if j != i:
                    continue
                terms[i].append((num, j, den, k))
        terms[i].sort()
        terms[i].reverse()

    # split device dimensions with multiple indexing terms into
    fixed_device_size: list[int] = []
    fixed_dim_map: list[int] = []
    fixed_it_device_dim_map = []
    for i in range(len(terms)):
        current = 1
        for num, j, den, k in terms[i]:
            fixed_device_size.append(device_size[i] // num // current)
            current *= device_size[i] // num
            fixed_dim_map.append(dim_map[i])
            fixed_it_device_dim_map.append((dim_map[k], den))

    return fixed_device_size, fixed_dim_map, fixed_it_device_dim_map


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


def propagate_view_ftl(
    buffer_layout: FixedTiledLayout,
    var_ranges: dict[sympy.Expr, sympy.Expr],
    index: sympy.Expr,
) -> FixedTiledLayout:
    """Compute a new FixedTiledLayout by analyzing a sympy index expression."""
    stl = buffer_layout.device_layout
    size = [int(s) for s in buffer_layout.size]
    stride = [int(s) for s in buffer_layout.stride]

    vars_list = sorted(index.free_symbols, key=lambda s: s.name)
    ranges = [int(var_ranges[v]) for v in vars_list]
    steps = [int(index.subs(v, 1) - index.subs(v, 0)) for v in vars_list]
    print(
        f"index: {index}, var_ranges: {var_ranges}, vars_list: {vars_list}, host size: {size}, host stride: {stride}, ranges: {ranges}, steps: {steps}"
    )

    new_stl = propagate_view_stl(
        stl,
        size,
        stride,
        ranges,
        steps,
    )

    return FixedTiledLayout(
        buffer_layout.device,
        buffer_layout.dtype,
        buffer_layout.size,
        buffer_layout.stride,
        new_stl,
    )


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            buffer_layout = buf.get_layout()

            if not isinstance(buffer_layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")

            _, var_ranges = index_vars_no_squeeze(*n._sizes, prefix="c")

            print(f"Stickify {n.get_name()} {arg} {var_ranges}")
            arg_layout = propagate_view_ftl(buffer_layout, var_ranges, arg.index)
            print(f"Buffer layout {buffer_layout.device_layout}")
            print(f"Arg layout {arg_layout.device_layout}")

            res.append(SchedNodeArg(arg, arg_layout))
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
