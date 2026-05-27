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

"""Spyre-specific decompositions and PrivateUse1 dispatch-key kernels.

There is exactly one public entry point: ``register_spyre_decompositions``.
It records a decomposition for ``torch.compile`` (consumed via Inductor's
``get_decomp_fn``); for aten ops, a PrivateUse1 kernel reaching the same
implementation is auto-installed at runtime init so eager-mode dispatch
reaches it too. Eager-only registration is intentionally not exposed.

The Spyre decomposition table built by ``get_spyre_decomp_table`` is
independent from PyTorch's global ``torch._inductor.decomposition.decompositions``
registry; Spyre never mutates the global table.
"""

import math
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch._decomp as decomp

from .constants import DEVICE_NAME
from .errors import Unsupported


# Spyre-specific decompositions, populated by ``@register_spyre_decompositions``.
spyre_decompositions: dict = {}

# Inductor default decompositions to drop on Spyre. They produce code the
# backend cannot lower today; falling through to the CPU fallback is preferable
# until those issues are fixed.
spyre_decompositions_to_exclude = [
    torch.ops.aten.triu,
    torch.ops.aten.tril,
]

OpOrOps = Union[
    torch._ops.OperatorBase, Sequence[torch._ops.OperatorBase]
]

# Module-level Library handles, kept alive for the lifetime of the process.
# ``torch.library.Library`` uses ``weakref.finalize`` to call ``m.reset()`` on
# GC, which would silently unregister every kernel from the C++ dispatcher.
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False


def register_spyre_decompositions(ops: OpOrOps):
    """Register a Spyre-specific decomposition for one or more operators.

    The function is added to the Spyre decomposition table; Inductor reads it
    via ``get_decomp_fn`` during ``torch.compile`` / ``make_fx``. For aten ops,
    ``_register_spyre_dispatchkey_kernels_permanently`` additionally installs a
    PrivateUse1 kernel pointing at the same function at runtime init, so
    eager-mode dispatch reaches it too. This is required for
    ``CompositeImplicitAutograd`` ops (``rms_norm``, ``layer_norm``, ...); it
    is harmless for the rest.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


def get_spyre_decomp_table() -> dict[Any, Callable[..., Any]]:
    """Return the decomposition table Inductor sees when compiling for Spyre.

    Builds a fresh dict on each call from ``select_decomp_table()`` (Inductor's
    default, itself cached upstream) plus Spyre additions and exclusions.
    Independent from ``torch._inductor.decomposition.decompositions`` — Spyre
    never mutates the global registry.
    """
    from torch._inductor.decomposition import select_decomp_table
    from torch._ops import OpOverload, OpOverloadPacket
    from torch_spyre.ops.fallbacks import fallback_ops

    table = dict(select_decomp_table())

    def _drop(op):
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                table.pop(getattr(op, overload_name), None)
        elif isinstance(op, OpOverload):
            table.pop(op, None)

    for op in spyre_decompositions_to_exclude:
        _drop(op)
    for op in fallback_ops:
        _drop(op)
    table.update(spyre_decompositions)
    return table


class _OPWrapper:
    """PrivateUse1 kernel that lazily ``torch.compile``-s a Spyre decomposition.

    The first eager call compiles the decomposition (with ``dynamic=False``);
    subsequent eager calls reuse the compiled entry point. When invoked from
    inside an active ``torch.compile`` context, the wrapped function is called
    directly — re-entering ``torch.compile`` would be wrong.
    """

    def __init__(self, fn):
        self._fn = fn
        self._compiled_fn = None

    def __call__(self, *args, **kwargs):
        from torch.utils import _pytree as pytree

        leaves = pytree.tree_leaves(args) + pytree.tree_leaves(kwargs)
        if any(
            isinstance(x, torch.Tensor)
            and getattr(x.device, "type", None) != DEVICE_NAME
            for x in leaves
        ):
            devs = [x.device if isinstance(x, torch.Tensor) else None for x in leaves]
            raise RuntimeError(
                f"Spyre decomposition function called with inputs on a different "
                f"device! Args devices: {devs=}"
            )
        if torch.compiler.is_compiling():
            return self._fn(*args, **kwargs)
        if self._compiled_fn is None:
            self._compiled_fn = torch.compile(self._fn, dynamic=False)
        return self._compiled_fn(*args, **kwargs)


def _register_spyre_dispatchkey_kernels_permanently():
    """Install PrivateUse1 / AutogradPrivateUse1 kernels for every aten op
    that has a Spyre decomposition and no pre-existing PrivateUse1 kernel.

    Idempotent; called from ``_SpyreImpl._lazy_init`` after eager ops and
    custom ops have been imported, so the existing-kernel check sees the final
    set of registered backends.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")
    has_pu1 = torch._C._dispatch_has_kernel_for_dispatch_key

    for op, fn in spyre_decompositions.items():
        if op.namespace != "aten" or has_pu1(op._name, "PrivateUse1"):
            continue
        # Autograd key: fall through so PrivateUse1 is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel)
        # PrivateUse1 key: dispatch into a lazy-compile wrapper.
        _spyre_lib.impl(op._name, _OPWrapper(fn))

    _dispatchkey_kernels_registered = True


###############################################################################
##                       Spyre decompositions                                ##
###############################################################################


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# Implement ones via identity broadcast: create a size-1 tensor (ones_scalar), expand to
# target size, then clone (identity) to materialize. Clone op with identity is merged.
@register_spyre_decompositions([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    scalar = torch.ops.spyre.ones_scalar(device, dtype=dtype)
    return scalar.reshape(()) if not size else scalar.expand(size).clone()


@register_spyre_decompositions([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    dev = device if device is not None else self.device
    dt = dtype if dtype is not None else self.dtype
    scalar = torch.ops.spyre.ones_scalar(dev, dtype=dt)
    return scalar.reshape(()) if not size else scalar.expand(size).clone()


# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decompositions([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


@register_spyre_decompositions([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decompositions([torch.ops.aten.addmm.default, torch.ops.aten.addmm.out])
def addmm_decomp(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: Union[int, float] = 1,
    alpha: Union[int, float] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Decompose addmm into basic operations: out = beta * input + alpha * (mat1 @ mat2)
    """
    # Compute matrix multiplication using matmul to handle batched tensors
    mm_result = mat1 @ mat2

    # Apply alpha scaling if needed
    if alpha != 1:
        mm_result = alpha * mm_result

    # Apply beta scaling and add input if needed
    if beta == 0:
        result = mm_result
    elif beta == 1:
        result = input + mm_result
    else:
        result = beta * input + mm_result

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


###############################################################################
##                    Spyre decompositions for aten ops                      ##
###############################################################################
# For aten ops, ``register_spyre_decompositions`` automatically installs a
# PrivateUse1 dispatch kernel as well (essential for CIA ops like rms_norm,
# layer_norm; harmless for the rest).
@register_spyre_decompositions([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    mean = torch.mean(input * input, dim=-1, keepdim=True)
    rsqrt_inp = torch.rsqrt(mean + eps)
    output = input * rsqrt_inp
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decompositions([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    # F.layer_norm treats weight=None as identity and bias=None as zero;
    # spyre.layernormnorm doesn't handle missing args, so substitute defaults.
    if weight is None:
        weight = input.new_ones(normalized_shape)
    if bias is None:
        bias = input.new_zeros(normalized_shape)
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decompositions([torch.ops.aten.topk])
def spyre_topk(
    input: torch.Tensor,
    k: int,
    dim: Optional[int] = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k > 4:
        raise Unsupported("Topk is not supported for this config")
    return torch.ops.spyre.topkvalue(input, k, dim), torch.ops.spyre.topkindex(
        input, k, dim
    )


@register_spyre_decompositions([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decompositions([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    return torch.ops.spyre.softplus(input, beta, threshold)


@register_spyre_decompositions([torch.ops.aten.linear.default])
def spyre_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    weight = weight.transpose(-1, -2)
    while weight.dim() < input.dim():
        weight = torch.unsqueeze(weight, 0)
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


@register_spyre_decompositions(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default]
)
def spyre__sdpa_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = query.size(0)
    num_heads = query.size(1)
    num_kvheads = key.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)

    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(query.shape[-1])
    scaling_factor = math.sqrt(scaling_factor)

    query = query * scaling_factor
    key = key * scaling_factor

    expansion = num_heads // num_kvheads
    if expansion != 1:
        key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
    key_t = key.transpose(-2, -1)

    attn = torch.matmul(query, key_t)

    if is_causal:
        assert attn_bias is None
        attn_bias = torch.full_like(attn, float("-inf"))
        attn_bias = attn_bias.triu(diagonal=1)

    if attn_bias is not None:
        attn = attn + attn_bias

    # TODO (aviros): Switch to _safe_softmax
    attn = torch.softmax(attn, -1)

    if dropout_p > 0.0:
        # TODO(aviros): Implement
        raise Unsupported("Attention dropout not implemented for Spyre")

    # Unused for now
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    # B, H, S, E
    out = torch.matmul(attn, value)

    # B, S, H, E
    # Do not remove contiguous here.
    # This is needed to maintain the API promise from SDPA (attn needs to have same size+stride as q)
    out = out.transpose(1, 2).clone(memory_format=torch.contiguous_format)

    # Returns (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
    return (
        out.transpose(1, 2),
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


## TODO(imaihal): Need to fix scalar tensor shape mismatch during Spyre-to-CPU transfer.
## See: https://github.com/torch-spyre/torch-spyre/issues/1172
## This will be enabled after solving this.
# @register_spyre_decompositions([torch.ops.aten.max.default])
# def spyre_max_default_decomp(input):
#    """
#    Decompose torch.max(input) with conditional CPU fallback for int64.
#
#    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
#    a CPU fallback registered in fallbacks.py.
#    For other dtypes (float16, float32, etc.), use amax.
#    """
#    if input.dtype == torch.int64:
#        # Use custom op with CPU fallback to avoid recursive decomposition
#        # Returns a scalar (0D) tensor
#        return torch.ops.spyre.max_default_int64_fallback(input)
#    else:
#        # Use amax for supported dtypes (can run on Spyre)
#        # Returns a scalar (0D) tensor
#        return torch.ops.aten.amax(input)


@register_spyre_decompositions([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_dim_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), decompose into amax and argmax operations.

    Returns a named tuple (values, indices) as expected by torch.max.

    # TODO (imaihal): Decomposed into torch.topk with k=1 to obtain both values and indices,
    #  or implement argmax in the backend compiler to get indices
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        return torch.ops.spyre.max_dim_int64_fallback(input, dim, keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decompositions([torch.ops.aten.min.dim])
def spyre_min_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.min(input, dim) with conditional CPU fallback for int64.

    Mirrors spyre_max_dim_decomp: int64 inputs go through a CPU-fallback custom
    op; other dtypes are decomposed into amin (Spyre-native) and argmin (CPU
    fallback). Returns a named tuple (values, indices) as expected by torch.min.
    """
    if input.dtype == torch.int64:
        return torch.ops.spyre.min_dim_int64_fallback(input, dim, keepdim)
    else:
        values = torch.ops.aten.amin(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input, dim=dim, keepdim=keepdim)
        return torch.return_types.min((values, indices))


@register_spyre_decompositions([torch.ops.aten.cat.default])
def decompose_cat(
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    orig_decomp = torch._inductor.decomposition.cat(tensors, dim)
    if orig_decomp == NotImplemented:
        expanded_size = 0
        for t in tensors:
            expanded_size += t.size(dim)
        output_size = list(tensors[0].size())
        output_size[dim] = expanded_size
        output = tensors[0].new_empty(output_size)
        offset = 0
        for input in tensors:
            output = torch.ops.spyre.overwrite_f(
                input=input, output=output, dims=[dim], offsets=[offset]
            )
            offset += input.size(dim)
        return output
    else:
        return orig_decomp


@register_spyre_decompositions([torch.ops.aten.constant_pad_nd.default])
def pad_decomp(
    input: torch.Tensor,
    pad: list[int],
    value: float = 0,
) -> torch.Tensor:
    # pad is in reverse dim order: (left_last, right_last, left_2nd_last, right_2nd_last, ...)
    n_dims_padded = len(pad) // 2

    # Negative pad values (cropping) require reading from a non-zero storage
    # offset or a sub-stick position, neither of which the SFP supports.
    if any(p < 0 for p in pad):
        raise Unsupported(
            f"constant_pad_nd: negative padding (cropping) is not supported on "
            f"Spyre (pad={pad})"
        )

    # Left-padding on the last (stick) dimension shifts the output start address
    # by `left` elements. The hardware can only express this in whole sticks, so
    # `left` must be a multiple of the stick size (64 fp16 elements).
    # Sub-stick left-padding on the last dimension is tracked in:
    # https://github.com/torch-spyre/torch-spyre/issues/1464
    last_dim_left = pad[0]
    if last_dim_left > 0:
        elems_per_stick = 128 // input.element_size()
        if last_dim_left % elems_per_stick != 0:
            raise Unsupported(
                f"constant_pad_nd: sub-stick left-padding on the last dimension is "
                f"not supported on Spyre (pad={pad}, left={last_dim_left}, "
                f"stick_size={elems_per_stick})"
            )

    # Build the padded output shape and collect which dimensions need padding.
    scalar = torch.ops.spyre.full([1], value, input.device, dtype=input.dtype)
    output_size = list(input.size())
    dims: list[int] = []
    offsets: list[int] = []
    for i in range(n_dims_padded - 1, -1, -1):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        if left + right == 0:
            continue
        dim = input.dim() - 1 - i
        output_size[dim] += left + right
        dims.append(dim)
        offsets.append(left)

    if not dims:
        return input

    output = scalar.expand(output_size).clone()
    output = torch.ops.spyre.overwrite_f(
        input=input, output=output, dims=dims, offsets=offsets
    )
    return output


@register_spyre_decompositions([torch.ops.aten.bitwise_not])
def bitwise_not(input: torch.Tensor) -> torch.Tensor:
    if input.dtype is torch.bool:
        return torch.logical_not(input)
    else:
        neg_one = torch.ops.aten.full_like(input, -1)
        return torch.ops.aten.bitwise_xor(input, neg_one)


@register_spyre_decompositions([torch.ops.aten.bitwise_and])
def bitwise_and(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.dtype is torch.bool and input2.dtype is torch.bool:
        return torch.ops.aten.logical_and(input1, input2)
    else:
        return torch.ops.aten.bitwise_not(
            torch.ops.aten.bitwise_or(
                torch.ops.aten.bitwise_not(input1), torch.ops.aten.bitwise_not(input2)
            )
        )


@register_spyre_decompositions([torch.ops.aten.sub.Tensor])
def sub_with_alpha(
    self: torch.Tensor, other: torch.Tensor, *, alpha: float = 1
) -> torch.Tensor:
    """
    Decompose torch.sub(a, b, alpha=alpha) into separate mul and sub operations.

    The Spyre backend does not have a single operation for a - alpha * b.
    When alpha != 1, we decompose into: a - (alpha * b)
    This ensures the operations are not fused by Inductor's optimization passes.
    """
    if alpha == 1:
        # Simple subtraction without alpha - use default behavior
        return NotImplemented
    else:
        # Decompose: sub(a, b, alpha) = sub(a, mul(b, alpha))
        scaled_other = torch.mul(other, alpha)
        return torch.sub(self, scaled_other)
