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

from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import ValueWithLineMap
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator

from .errors import Unsupported
from .ir import FixedTiledLayout
from .constants import SEGMENT_SIZE


def _inject_pool_alloc_dealloc(wrapper_str: str, pool_alloc_code: str) -> str:
    """Inject the HBM ``_pool`` allocation/free into every function scope that
    uses it.

    A compiled Spyre graph can emit MORE than one function that references the
    shared ``_pool`` byte region -- the top-level ``Runner.call`` body plus each
    reused ``invoke_subgraph`` body (a nested_compile_region decoder block is
    lowered to one such function and called once per layer). Every one of them
    passes ``_pool`` as the first argument to its ``sdsc_*.run(...)`` kernel
    calls, so every one of them must allocate ``_pool`` in its OWN local scope
    and free it before its OWN ``return``.

    The previous implementation scanned the whole wrapper text for the FIRST
    ``.run(`` and the LAST ``return (`` and injected once. With subgraph
    inlining the subgraph function is emitted textually before ``Runner.call``,
    so the single allocation landed in the subgraph and ``Runner.call``
    referenced an undefined ``_pool`` -> UnboundLocalError at runtime. This
    walks each function body independently instead.
    """
    lines = wrapper_str.split("\n")

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    # A function body starts at a `def ` line and runs until the next line at
    # the same-or-lower indentation that is itself a `def`/`class` (a sibling
    # or the enclosing scope closing). Collect [start, end) spans for every
    # `def`, innermost first is unnecessary -- spans don't overlap for our
    # emitted code (methods live inside a class but the class body has no
    # `.run(`/`_pool` of its own).
    spans: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            def_indent = _indent(line)
            end = len(lines)
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if not nxt.strip():
                    continue
                stripped = nxt.lstrip()
                if _indent(nxt) <= def_indent and (
                    stripped.startswith("def ") or stripped.startswith("class ")
                ):
                    end = j
                    break
            spans.append((i, end))

    # Process spans from last to first so earlier spans' indices stay valid as
    # we insert lines into later ones.
    for start, end in sorted(spans, reverse=True):
        body = lines[start:end]
        uses_pool = any(".run(_pool" in ln for ln in body)
        already_allocs = any(ln.strip().startswith("_pool = ") for ln in body)
        if not uses_pool or already_allocs:
            continue

        # Free `_pool` before this scope's `return (` (search from the end of
        # the span so we hit this function's return, not a nested one).
        for k in range(end - 1, start - 1, -1):
            if lines[k].strip().startswith("return ("):
                lines.insert(k, " " * _indent(lines[k]) + "del _pool")
                break

        # Allocate `_pool` before this scope's first kernel call.
        for k in range(start, end):
            if ".run(" in lines[k]:
                lines.insert(k, " " * _indent(lines[k]) + pool_alloc_code)
                break

    return "\n".join(lines)


class _SpyreWrapperCodegenMixin(PythonWrapperCodegen):
    """Spyre wrapper-codegen behavior shared by the top-level graph wrapper and
    the invoke_subgraph wrapper.

    These overrides are about *how a Spyre buffer/op is emitted* (device-layout
    allocation, HBM-pool reuse/free, constant-tensor fallback). They are
    graph-role-agnostic, so both the parent graph and a nested_compile_region
    subgraph need them. Role-specific behavior (header emission, benchmark
    harness, launcher naming) is NOT here — it stays on the concrete wrapper
    classes, so a subgraph keeps the stock ``write_header = pass`` and does not
    double-emit imports.
    """

    def _patch_sizevars(self) -> None:
        # Spyre device layout is not fully visible to Inductor, so its loop
        # simplification would be wrong (see noop_simplify_loops_impl). Applied
        # per GraphLowering, so the subgraph's own sizevars is patched too.
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    def generate(self, is_inference):
        """Override to add pool allocation/deallocation around kernel calls."""
        result_tuple = super().generate(is_inference)
        wrapper_value_with_linemap, kernel_decls = result_tuple

        hbm_pool_size = getattr(V.graph, "hbm_pool_size", 0)
        if hbm_pool_size > 0:
            wrapper_str = str(wrapper_value_with_linemap.value)
            wrapper_str = _inject_pool_alloc_dealloc(
                wrapper_str, self.allocate_hbm_pool()
            )
            wrapper_value_with_linemap = ValueWithLineMap(
                value=wrapper_str, line_map=wrapper_value_with_linemap.line_map
            )

        return (wrapper_value_with_linemap, kernel_decls)

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

    def generate_const_tensor_fallback(self, node):
        value = node.constant_args[0]
        dtype = node.layout.dtype
        device = node.layout.device
        self.writeline(
            f'{node.get_name()} = spyre_constant_tensor({value}, torch.device("{device}"), {dtype})'
        )

    def _is_hbm_pool_buffer(self, buffer: BufferLike) -> bool:
        layout = buffer.get_layout()
        return isinstance(layout, FixedTiledLayout) and "hbm_pool" in layout.allocation

    def codegen_free_buffer(self, buffer: BufferLike) -> None:
        if not self._is_hbm_pool_buffer(buffer):
            super().codegen_free_buffer(buffer)

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        new_offset = new.get_layout().offset or 0
        old_offset = old.get_layout().offset or 0
        if (
            old.get_size() == new.get_size()
            and old.get_stride() == new.get_stride()
            and old_offset == new_offset
        ):
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        new_stl = new.get_layout().device_layout
        # reinterpret_tensor_with_layout's offset arg is added to old_name's
        # *current* runtime storage_offset (see spyre_views.cpp), not treated
        # as an absolute target. Passing old's offset directly double-counts
        # it; the delta to new's offset is what actually shifts to the right
        # place.
        offset_increment = new_offset - old_offset
        # Static-shape paths only: a symbolic offset would render a bare
        # sympy expression into the generated source below.
        if getattr(offset_increment, "free_symbols", None):
            raise Unsupported(
                f"symbolic storage_offset not supported in buffer-reuse codegen: "
                f"{offset_increment!r}"
            )
        reinterpret_view = f"reinterpret_tensor_with_layout({old_name}, {new.get_size()}, {new.get_stride()}, {offset_increment}, {new_stl!r})"
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def allocate_hbm_pool(self):
        """Allocate the intermediate pool."""
        # _pool is an opaque, flat byte-addressed region: individual buffers
        # are placed into it at byte offsets (see hbm_pool_planning.py's
        # layout.allocation["hbm_pool"] = offset) and it is passed to kernels
        # as-is, never sliced/reshaped in Python.  A 1D uint8 tensor of
        # pool_size_bytes elements is the natural, unambiguous
        # representation -- no need to route it through a multi-dim device
        # layout, and no need to divide by 128 (each uint8 element is
        # already one byte, not one stick).
        pool_size_bytes = getattr(V.graph, "hbm_pool_size", SEGMENT_SIZE)
        return (
            f"_pool = spyre_empty_with_layout("
            f"({pool_size_bytes},), (1,), "
            f"torch.uint8, SpyreTensorLayout(device_size=[{pool_size_bytes}], "
            f"stride_map=[1], device_dtype=DataFormats.SENINT8))"
        )


class SpyrePythonWrapperCodegen(_SpyreWrapperCodegenMixin, PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        self._patch_sizevars()

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
            return SpyreSubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from sympy import sympify
                from torch_spyre._inductor.op_spec import TensorArg, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
                from torch_spyre.execution.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor as reinterpret_tensor"
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor_with_layout"
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")


class SpyreSubgraphPythonWrapperCodegen(
    _SpyreWrapperCodegenMixin, SubgraphPythonWrapperCodegen
):
    """Spyre wrapper for an invoke_subgraph body (e.g. a nested_compile_region
    decoder block reused across layers).

    Inherits the stock subgraph plumbing (launcher naming, empty ``write_header``
    so imports are not re-emitted, input/output signature handling) from
    ``SubgraphPythonWrapperCodegen`` and layers the Spyre buffer/op codegen on
    top via the mixin. Without this, subgraph codegen fell back to the stock
    wrapper and hit ``AttributeError`` the moment it emitted a Spyre buffer
    (e.g. a SpyreConstantFallback materialized by split_multi_ops).
    """

    def __init__(
        self,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        super().__init__(subgraph_name, parent_wrapper, partition_signatures)
        self._patch_sizevars()


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
