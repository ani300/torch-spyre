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

from typing import Any, Callable, Optional

import sympy
from torch import Tensor
from torch._inductor import ir
from torch._inductor.codegen.common import DeferredLine
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
    codegen_reinterpret_view_helper,
)
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator

from torch_spyre._C import compute_view_layout, SpyreTensorLayout
from .pass_utils import propagate_view_stl
from .stickify import FixedTiledLayout
from .errors import Unsupported


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from torch_spyre._inductor.runtime import ConstantArg, TensorArg, OpSpec, UnimplementedOp
                from torch_spyre._inductor.runtime.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline("from torch_spyre._C import as_strided_with_layout")
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        out = (
            f"{name} = spyre_empty_with_layout("
            f"{codegen_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"{layout.dtype}, "
            f"{layout.device_layout!r})"
        )

        return out

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        # Get the innermost buffer's layout info to help reinterpret view.
        # Consider a chain of (ReinterpretView <- TensorBox| StorageBox)... <- buffer
        # If we only use x.data to determine the reinterpret, we may get wrong layout.
        # For example:
        # x = ReinterpretView(
        #       Storage(
        #         ReinterpretView(
        #           storage(
        #             Buffer(name='buf0', layout=(size=(2, 5, 10), ...)
        #           ),
        #           layout=(10, 10),
        #         ),
        #       ),
        #       layout=(10, 10),
        #     )
        # In this case, x.data.layout == x.layout is (10, 10), the reinterpret view will return buf0,
        # but buf0 need to be viewed from (2, 5, 10) to (10, 10).
        # So we need to dig into the chain to find the innermost buffer's layout.
        d_size, d_stride, d_offset, d_dtype, d_stl, collapsible = (
            spyre_codegen_reinterpret_view_helper(data)
        )

        def apply_spyre_reinterpret(
            name, tgt_size, tgt_stride, tgt_offset, tgt_stl, cast_dtype, base_dtype
        ):
            s = self.codegen_python_shape_tuple(tgt_size)
            st = self.codegen_python_shape_tuple(tgt_stride)
            off = self.codegen_sizevar(tgt_offset)
            expr = (
                f"as_strided_with_layout("
                f"{name}, {s}, {st}, {off}, "
                f"{tgt_stl!r})"
            )
            if cast_dtype is not None and cast_dtype != base_dtype:
                return f"aten.view.dtype({expr}, {cast_dtype})"
            return expr

        name = data.get_name()
        collapsed = collapsible and offset == d_offset
        if collapsed:
            same_layout = size == d_size and stride == d_stride
            base_dtype = d_dtype
        else:
            same_layout = (
                size == data.layout.size
                and stride == data.layout.stride
                and offset == data.layout.offset
            )
            base_dtype = data.dtype

        if same_layout:
            if dtype is not None and dtype != base_dtype:
                return f"aten.view.dtype({name}, {dtype})"
            return f"{name}"
        
        # print(d_size, d_stride, size, stride)
        stl = propagate_view_stl(d_stl, [int(s) for s in d_size], [int(s) for s in d_stride], size, stride)

        return apply_spyre_reinterpret(name, size, stride, offset, stl, dtype, base_dtype)


def spyre_codegen_reinterpret_view_helper(data):
    """
    Collapse a chain of ReinterpretView <- StorageBox
    <- ReinterpretView <- StorageBox.... <- buffer wrappers if every layer
    has the same offset as the innermost (base) buffer.

    Returns:
        (size, stride, offset, dtype, spyre_tensor_layout, collapsible: bool)
    """
    if isinstance(data, ir.Buffer):
        lay = data.get_layout()
        assert isinstance(lay, FixedTiledLayout), "The base buffer doesn't have a FixedTiledLayout"
        stl = lay.device_layout
        return lay.size, lay.stride, lay.offset, lay.dtype, stl, True

    layouts: list[Any] = []
    cur = data
    while isinstance(cur, (ir.TensorBox, ir.StorageBox, ir.ReinterpretView)):
        lay = cur.get_layout()
        if lay is None:
            return None, None, None, None, None, False
        layouts.append(lay)
        cur = cur.data  # unwrap

    if not isinstance(cur, ir.Buffer):
        return None, None, None, None, None, False

    # All wrapper offsets must match base offset to be collapsible
    for lay in layouts:
        if lay.offset != cur.get_layout().offset:
            return None, None, None, None, None, False

    base_lay = cur.get_layout()
    base_stl = None
    if isinstance(cur, ir.InputBuffer):
        # On the cases where we are compiling a single data op, the lowering/stickification pass never happens
        # Get the STL from the tensor input
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if name != cur.get_name():
                continue
            if isinstance(real_input, Tensor):
                base_stl = real_input.device_tensor_layout()
                if base_stl is None:
                    # All spyre tensors are created with device layouts.
                    # Therefore we expect all graph inputs to have them.
                    raise Unsupported(
                        f"missing device_tensor_layout on graph input {name}"
                    )
        assert base_stl is not None, "Input not found"
    else:
        assert isinstance(base_lay, FixedTiledLayout), "The base buffer doesn't have a FixedTiledLayout"
        base_stl = base_lay.device_layout
    print("DEBUG!", base_lay, cur, cur.get_name(), base_stl)
    return base_lay.size, base_lay.stride, base_lay.offset, base_lay.dtype, base_stl, True


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x
