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


import dataclasses
from typing import Any, Optional, Sequence

from sympy import Symbol, Expr
from torch_spyre._C import DataFormats
import torch


@dataclasses.dataclass
class IndirectSource:
    """Describes data-dependent (indirect) addressing for a tensor argument.

    When present on a TensorArg, the tensor's base address along
    ``gather_dim`` is computed at runtime by reading values from another
    tensor argument (the index tensor) rather than from a static affine
    expression. This models the ``A[X[i]]`` pattern needed for
    gather/index_select/embedding workloads.

    Attributes:
        index_arg_index: Position of the index tensor in the OpSpec's
            ``args`` list. The index tensor supplies the runtime-computed
            indices.
        gather_dim: Which dimension of *this* tensor is indirectly addressed.
        base_offset_expr: A sympy expression that maps each index value to a
            byte offset. The free variable ``index_value`` represents the
            value read from the index tensor at runtime.
    """

    index_arg_index: int
    gather_dim: int
    base_offset_expr: Expr


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        device_dtype: The device dtype of the tensor elements.
        device_size: The device size (as per SpyreTensorLayout) of the Tensor
        device_coordinates: The sympy Exprs that describe how elements in the Tensor are accessed.
                Free variables in device_coordinates refer to entries in the OpSpec's iteration_space.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
        indirect_source: If present, this tensor uses data-dependent
            addressing. The tensor's base address along the gather dimension
            is computed at runtime from values of another tensor argument
            (the index tensor).
    """

    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    indirect_source: Optional[IndirectSource] = None


@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation. The values are tuples of (range, core_division).
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
    """

    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]


@dataclasses.dataclass
class UnimplementedOp:
    op: str


def spyre_constant_tensor(const_val, device, dtype=torch.float16):
    return torch.tensor([const_val], dtype=dtype).to(device)
