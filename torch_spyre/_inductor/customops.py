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

from typing import Optional, Sequence
import torch
import torch._dynamo
from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp
from torch_spyre.ops.eager import compile_once
from torch_spyre.ops.fallbacks import warn_fallback

from .errors import Unsupported

aten = torch.ops.aten


@torch.library.custom_op("spyre::softplus", mutates_args=(), device_types="spyre")
def softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    pass


@softplus.register_fake
def _(input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::layer_norm", mutates_args=())
def layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.native_layer_norm(x, normalized_shape, weight, bias, eps)[0].clone()


@layer_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::exx2", mutates_args=(), device_types="spyre")
def exx2(x: torch.Tensor, exx2Scale: float, useZeroMean: bool) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@exx2.register_fake
def _(x: torch.Tensor, exx2Scale: float, useZeroMean: bool):
    return x.new_empty(x.size()[:-1])


@torch.library.custom_op("spyre::layernormscale", mutates_args=(), device_types="spyre")
def layernormscale(x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@layernormscale.register_fake
def _(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::layernormnorm", mutates_args=(), device_types="spyre")
def layernormnorm(  # type: ignore[empty-body]
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    pass


@layernormnorm.register_fake
def _(
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::rms_norm", mutates_args=())
def rms_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.compile(torch.ops.spyre.rms_norm)(x, normalized_shape, weight, eps)


@rms_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::topkvalue", mutates_args=(), device_types="spyre")
def topkvalue(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkvalue.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size)


@torch.library.custom_op("spyre::topkindex", mutates_args=(), device_types="spyre")
def topkindex(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkindex.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size, dtype=torch.int64)


@torch.library.custom_op("spyre::gelu", mutates_args=(), device_types="spyre")
def gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    pass


@gelu.register_fake
def _(input: torch.Tensor, approximate: str = "none"):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::silu", mutates_args=(), device_types="spyre")
def silu(
    input: torch.Tensor,
) -> torch.Tensor:
    return aten.div.Tensor(input, 1 + aten.exp.default(-input))


@silu.register_fake
def _(input: torch.Tensor):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::clamp", mutates_args=(), device_types="spyre")
def clamp(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> torch.Tensor:
    pass


@clamp.register_fake
def _(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::empty", mutates_args=(), device_types="spyre")
def spyre_empty(
    size: Sequence[int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Eager-mode simulation: allocate on CPU and move to the Spyre device.
    # This is not a compute fallback — on hardware the compiled kernel receives
    # a device allocation from SpyreAllocator with no host-side initialisation.
    tmp = torch.empty(size, dtype=dtype, device="cpu")
    return tmp.to(device)


@spyre_empty.register_fake
def _(
    size: Sequence[int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    return torch.empty(size, dtype=dtype, device="spyre")


@torch.library.custom_op("spyre::logical_not", mutates_args=(), device_types="spyre")
def logical_not(input: torch.Tensor) -> torch.Tensor:
    pass


@logical_not.register_fake
def _(input: torch.Tensor):
    return input.new_empty(input.size())


@torch.library.custom_op(
    "spyre::copy_from_d2d", mutates_args=("dst",), device_types="spyre"
)
@compile_once("spyre.copy_from_d2d")
def copy_from_d2d(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_off: int,
    dst_off: int,
    compiled,
) -> None:
    # src_off/dst_off are the src/dst storage_offsets, passed as explicit ints
    # because a sliced tensor's offset is invisible to the compiled kernel
    # otherwise: a graph input's storage_offset is dropped by Inductor (its
    # FixedLayout.offset is 0 and SpyreTensorLayout has no offset field), so the
    # kernel binds the storage BASE pointer and reads from element 0. The
    # lowering (lower_spyre_from_d2d) consumes these ints to re-introduce the
    # offsets in-graph via a ReinterpretView, putting them into the coordinate
    # that superdsc bakes into the SDSC binary.
    #
    # specialize_int=True is required on top of that: dynamo's TENSOR_MATCH
    # guard keys on dtype/device/size/stride but NOT storage_offset, and its
    # default auto-dynamic promotes the offset int to a symbol after the second
    # distinct value — a symbolic offset cannot be baked as a constant into the
    # coordinate. specialize_int installs int-equality guards so each distinct
    # offset triggers a fresh trace and a fresh SDSC binary with the offset
    # baked as a constant. This mirrors the spyre.overwrite fix above (PR
    # #2084). Patch is call-scoped to leave process-wide dynamo behavior alone.
    # Note: one compiled binary per unique (input shape, offsets) tuple;
    # dynamo's cache_size_limit is bumped to 1024 in torch_spyre/__init__.py.
    with torch._dynamo.config.patch(specialize_int=True):
        return compiled(src, dst, src_off, dst_off)


@copy_from_d2d.register_fake
def _(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_off: int,
    dst_off: int,
) -> None:
    pass


# Copy src into dst in-place (the mutating primitive).
# No @compile_once needed: the body calls aten::copy_ which dispatches
# through spyre__copy_from → copy_from_d2d / copy_tensor — no cycle back
# to spyre::copy_ itself.
@torch.library.custom_op("spyre::copy_", mutates_args=("dst",), device_types="spyre")
def copy_(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dst.copy_(src)
    return dst


@copy_.register_fake
def _(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return dst


@torch.library.register_kernel("spyre::copy_", ["cpu"])
def copy__cpu(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dst.copy_(src)
    return dst


# Functional (out-of-place) wrapper for spyre::copy_.
# Returns a new tensor equal to dst with src written into it.
# The graph sees a pure value-producing node; Inductor's reinplacer will
# rewrite copy_f → copy_ (in-place) wherever it is safe to do so.
@torch.library.custom_op("spyre::copy_f", mutates_args=(), device_types="spyre")
def copy_f(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    result = dst.clone()
    torch.ops.spyre.copy_(src, result)
    return result


@copy_f.register_fake
def _(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return dst.clone()


inplaceable_ops[torch.ops.spyre.copy_f.default] = InplaceableOp(
    torch.ops.spyre.copy_.default, 1
)


# Copy input into output starting at offsets along dimensions dims and
# return the updated output.
@torch.library.custom_op(
    "spyre::overwrite", mutates_args=("output",), device_types="spyre"
)
@compile_once("spyre.overwrite")
def overwrite(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
    compiled,
) -> None:
    # specialize_int=True installs int-equality guards on the int-list
    # args so each unique (dims, offsets) triggers a fresh trace and a
    # fresh SDSC binary; without this dynamo's default specialize_int=
    # False reuses one baked binary across all values and scatters all
    # writes to the first call's offset (see test_overwrite.py).
    # Patch is call-scoped to leave process-wide dynamo behavior alone.
    # Note: this gives one compiled binary per unique (input shape, dims,
    # offsets) tuple. dynamo's cache_size_limit is bumped to 1024 in
    # torch_spyre/__init__.py — long-running workloads that scatter into
    # many distinct slots can blow past that. Symbolic offsets (one
    # binary, any value) are tracked in issues #220 / #1371-3.
    with torch._dynamo.config.patch(specialize_int=True):
        return compiled(input, output, dims, offsets)


@overwrite.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    return None


@torch.library.register_kernel("spyre::overwrite", ["cpu"])
def overwrite_cpu(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    sliced_t = output
    for i, dim in enumerate(dims):
        sliced_t = torch.narrow(sliced_t, dim, offsets[i], input.size(dim))
    sliced_t.copy_(input)


@torch.library.custom_op("spyre::overwrite_f", mutates_args=(), device_types="spyre")
def overwrite_f(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    result = output.clone()
    torch.ops.spyre.overwrite(input, result, dims, offsets)
    return result


@overwrite_f.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    return output.clone()


inplaceable_ops[torch.ops.spyre.overwrite_f.default] = InplaceableOp(
    torch.ops.spyre.overwrite.default, 1
)


@torch.library.custom_op("spyre::restickify", mutates_args=(), device_types="spyre")
def restickify(  # type: ignore[empty-body]
    x: torch.Tensor,
) -> torch.Tensor:
    pass


@torch.library.custom_op("spyre::max_dim_int64_fallback", mutates_args=())
def max_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.max(input, dim) when input is int64.

    The Spyre device kernel (registered in fallbacks.py via
    register_kernel(op, ["spyre"])) handles spyre-tensor inputs. PT 2.12
    routes calls with non-spyre tensor inputs (e.g. compare_with_cpu test
    paths) to this CompositeExplicitAutograd body, so it must compute the
    real result rather than raise.
    """
    return torch.max(input, dim=dim, keepdim=keepdim)


@max_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.max(input, dim, keepdim).
    """
    # Compute output shape based on dim and keepdim
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    # Return tuple of (values, indices) with the computed shape
    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


@torch.library.custom_op("spyre::unfold", mutates_args=(), device_types="spyre")
def spyre_unfold(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[int]] = None,
    stride: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Im2col unfold operation via torch.nn.functional.unfold.
    Converts (N, C, H, W) input to (N, C*K_h*K_w, L) where L = H_out * W_out.
    Uses CPU fallback for the unfold operation.
    """

    dilation = dilation or (1, 1)
    padding = padding or (0, 0)
    stride = stride or (1, 1)

    warn_fallback("torch.ops.spyre.unfold")
    # Move to CPU, perform unfold, move back to Spyre
    input_cpu = input.to("cpu")
    result_cpu = torch.nn.functional.unfold(
        input_cpu,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    return result_cpu.to(input.device)


@spyre_unfold.register_fake
def _(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[int]] = None,
    stride: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    dilation = dilation or (1, 1)
    padding = padding or (0, 0)
    stride = stride or (1, 1)

    N, C, H_in, W_in = input.shape
    K_h, K_w = kernel_size
    dil_h, dil_w = dilation
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    return input.new_empty((N, C * K_h * K_w, H_out * W_out))


@torch.library.custom_op(
    "spyre::reshape_via_cpu", mutates_args=(), device_types="spyre"
)
def spyre_reshape_via_cpu(
    input: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    Reshape operation that executes on CPU to avoid stick-alignment issues.

    When reshaping produces a shape with innermost dimension that doesn't align
    with stick boundaries (64 elements for fp16), the Inductor coordinate
    computation fails. This op moves to CPU, reshapes, then moves back to Spyre.

    This is similar to unfold, which also uses CPU fallback for correct layouts.
    """
    warn_fallback("torch.ops.spyre.reshape_via_cpu")
    input_cpu = input.to("cpu")
    result_cpu = input_cpu.reshape(shape)
    return result_cpu.to(input.device)


@spyre_reshape_via_cpu.register_fake
def _(
    input: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    return input.new_empty(shape)


@torch.library.custom_op("spyre::min_dim_int64_fallback", mutates_args=())
def min_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.min(input, dim) when input is int64. See
    max_dim_int64_fallback for the rationale on the body computing the
    real result instead of raising.
    """
    return torch.min(input, dim=dim, keepdim=keepdim)


@min_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.min(input, dim, keepdim).
    """
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


@torch.library.custom_op("spyre::max_default_int64_fallback", mutates_args=())
def max_default_int64_fallback(input: torch.Tensor) -> torch.Tensor:
    """
    CPU fallback for torch.max(input) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a 1D tensor with shape [1] containing the maximum value.
    """
    # This should never be called directly; the fallback in fallbacks.py handles it
    raise RuntimeError(
        "spyre::max_default_int64_fallback should be handled by CPU fallback registration"
    )


@max_default_int64_fallback.register_fake
def _(input: torch.Tensor):
    """
    Fake implementation for shape inference.
    Returns a scalar (0D) tensor matching the input dtype.
    """
    return input.new_empty([])


@torch.library.custom_op("spyre::batched_matmul", mutates_args=(), device_types="spyre")
def batched_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@batched_matmul.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output_shape = list(x.shape[:-1]) + [y.shape[-1]]
    return x.new_empty(output_shape)


@torch.library.custom_op("spyre::constant", mutates_args=(), device_types="spyre")
def spyre_constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    # This custom operator marks scalar constant in the FX graph.
    # Returning the scalar constant to avoid change in the operator schema which
    # consume the scalar constant as input.
    # This node will have a special handling at lowering to convert the scalar
    # constant to tensor.
    return fill_value


@spyre_constant.register_fake
def _constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    return fill_value


@torch.library.custom_op("spyre::to_dtype_cpu", mutates_args=(), device_types="spyre")
def to_dtype_cpu(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    warn_fallback(f"conversion from {input.dtype} to {dtype}")
    return input.cpu().to(dtype=dtype).to(input.device)


@to_dtype_cpu.register_fake
def _(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)


@torch.library.custom_op("spyre::qfp8ch", mutates_args=(), device_types="spyre")
def qfp8ch(input: torch.Tensor) -> torch.Tensor:
    """
    Channel-wise FP8 format conversion (pointwise, optimized for matmul).

    Converts input tensor to FP8 E4M3 format with channel-wise semantics.
    This operation ONLY performs format conversion - scaling must be done separately.

    Args:
        input: Input tensor (FP16/BF16/FP32) to convert to FP8
               Should already be scaled and clamped

    Returns:
        FP8 E4M3 tensor (same shape as input)

    Maps to: deeptools Qfp8ch operation
    """
    pass


@qfp8ch.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op(
    "spyre::quantize_fp8_with_scale", mutates_args=(), device_types="spyre"
)
def quantize_fp8_with_scale(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Quantize FP16 tensor to FP8 using pre-computed scale.

    Performs four steps:
    1. Compute inverse scale: inv_scale = 1 / scale (reciprocal, POINTWISE on sfp unit)
    2. Scale the input: x_scaled = x * inv_scale (POINTWISE)
    3. Clamp to FP8 E4M3 range: x_clamped = clamp(x_scaled, -448, 448) (POINTWISE)
    4. Convert to FP8 format: x_fp8 = qfp8ch(x_clamped) (POINTWISE format conversion)

    Args:
        input: Input tensor (FP16) to quantize, shape [batch, seq, hidden]
        scale: Quantization scale (FP16), shape [batch, seq, 1]

    Returns:
        FP8 E4M3 tensor (same shape as input)

    Example:
        >>> x = torch.randn(2, 4, 8, dtype=torch.float16, device='spyre')
        >>> x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)

    Note:
        - Uses reciprocal operation (hardware sfp unit) for 1/scale computation
    """
    pass


@quantize_fp8_with_scale.register_fake
def _(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op(
    "spyre::dequantize_fp8_with_scale", mutates_args=(), device_types="spyre"
)
def dequantize_fp8_with_scale(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # type: ignore[empty-body]
    """
    Dequantize FP8 tensor to FP16 using pre-computed scale.
    Performs two steps:
    1. Convert FP8 to FP16: x_fp16 = fp8todl16(x) (dtype conversion)
    2. Scale the output: x_scaled = x_fp16 * scale (POINTWISE)
    Args:
        input: Input tensor (FP8) to dequantize, shape [batch, seq, hidden]
        scale: Dequantization scale (FP16), shape [batch, seq, 1]
    Returns:
        FP16 tensor (same shape as input)
    Example:
        >>> @torch.compile(backend='inductor')
        >>> def dequant(x_fp8, scale):
        >>>     return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
    Note:
        - MUST use torch.compile(backend='inductor') - does not work in eager mode
        - Uses fp8todl16 operation for FP8→FP16 conversion
        - Scale must be FP16, NOT FP32
    """
    pass


@dequantize_fp8_with_scale.register_fake
def _(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Output is FP16 with same shape as input
    return torch.empty(input.size(), dtype=torch.float16, device=input.device)


@torch.library.custom_op("spyre::scaled_mm", mutates_args=(), device_types="spyre")
def scaled_mm(
    mat1: torch.Tensor, mat2: torch.Tensor, out_dtype: torch.dtype = None
) -> torch.Tensor:  # type: ignore[empty-body]
    """
    Raw FP8 matrix multiplication, with no scaling or bias applied.

    Scaling (scale_a, scale_b) and bias are intentionally NOT applied here -
    they're applied afterward at the decomposition level by scaled_mm_decomp,
    mirroring how dequantize_fp8_with_scale keeps its FP8->FP16 conversion
    separate from the subsequent scale multiply.
    """
    pass


@scaled_mm.register_fake
def _(
    mat1: torch.Tensor, mat2: torch.Tensor, out_dtype: torch.dtype = None
) -> torch.Tensor:
    output_shape = [mat1.shape[0], mat2.shape[-1]]
    return mat1.new_empty(output_shape, dtype=out_dtype or torch.float16)


@torch.library.custom_op(
    "spyre::quantize_weight_fp8_with_scale", mutates_args=(), device_types="spyre"
)
def quantize_weight_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    pass


@quantize_weight_fp8_with_scale.register_fake
def _(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op("spyre::qfp8wt", mutates_args=(), device_types="spyre")
def qfp8wt(input: torch.Tensor) -> torch.Tensor:
    pass


@qfp8wt.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op("spyre::causal_mask", mutates_args=(), device_types="spyre")
def causal_mask(
    seqlen_q: int,
    seqlen_kv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a causal additive mask on CPU and transfer to the target device.

    Shape: [1, 1, seqlen_q, seqlen_kv] in natural orientation: query i attends
    to keys 0..i.  Entries are 0.0 (keep) or -inf (masked).  The kept diagonal
    guarantees no fully-masked row (no 0/0 NaN denominator).

    Built entirely on CPU (tril + masked_fill_) so those in-place ops are
    opaque to torch.compile — assert_functional_graph is satisfied and the
    compiled graph only sees the resulting device tensor.  No device_types
    restriction is set because there are no tensor arguments to dispatch on;
    the device is an explicit parameter.
    """
    # Causal boolean lower-triangular pattern: True = attend, False = mask
    causal_cpu = torch.tril(
        torch.ones(seqlen_q, seqlen_kv, dtype=torch.bool, device="cpu")
    )
    mask_cpu = torch.zeros(1, 1, seqlen_q, seqlen_kv, dtype=dtype, device="cpu")
    mask_cpu.masked_fill_(~causal_cpu, float("-inf"))
    return mask_cpu.to(device=device)


@causal_mask.register_fake
def _(
    seqlen_q: int,
    seqlen_kv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(1, 1, seqlen_q, seqlen_kv, dtype=dtype, device=device)


@torch.library.custom_op(
    "spyre::sliding_window_attention", mutates_args=(), device_types="spyre"
)
def sliding_window_attention(  # type: ignore[empty-body]
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: Optional[float] = None,
    cache_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """
    Sliding-window attention entry point.

    query: [B, Hq, Lq, D]; key/value: [B, Hkv, Lk, D] (GQA expansion handled
    internally when Hq != Hkv). window_size is a compile-time constant (not a
    data-dependent tensor), matching flash-attention's window_size_left
    convention. Query row r (KV-cache coordinate max(0, Lk - Lq) + i) may
    attend to key position c iff 0 <= r - c < window_size (causal) or
    abs(r - c) < window_size (bidirectional, not yet implemented --
    is_causal=False raises Unsupported).

    ``cache_seqlen`` is how many tokens the cache has seen -- HF's
    ``cumulative_length``, vLLM's ``seq_lens`` -- as distinct from ``Lk``,
    the rows it allocates. Defaults to ``Lk``. All three orderings are
    accepted; ``SlidingWindowPlan``'s class docstring describes what each
    means for placement. It need NOT be a multiple of 64 (only ``Lk`` must
    be): it is a token count, not a memory offset.

    **Allocation contract:** ``Lk >= round_up_to_64(window_size + Lq - 1)``
    for the ``Lq`` of *this* call -- HF's ``get_mask_sizes`` formula rounded
    to a stick. Decode (``Lq=1``) therefore needs exactly ``window_size``,
    matching ``StaticSlidingWindowLayer``; a prefill chunk needs the extra
    ``Lq - 1`` because its earliest row still reaches ``window_size`` columns
    back. Prefill long sequences in chunks sized to the allocation.
    ``rejection_reason`` names the required row count when ``Lk`` is too
    narrow.

    **Zero-fill the allocation.** A block's buffer may overshoot the written
    prefix; causal masking discards those scores, but the multiply still
    happens, so the rows must hold finite values -- an additive ``-inf``
    cannot rescue a ``NaN``.

    **This op only reads.** Keeping a rolled cache in time order is the
    caller's job and is not demonstrated to work on Spyre; see
    ``spyre::kv_window``.

    MUST be called under torch.compile(backend="inductor") on the spyre
    device — see spyre_sliding_window_attention in decompositions.py for the
    real (windowed, block-skipping) Inductor lowering. This eager body is
    intentionally empty, matching spyre::exx2 / spyre::layernormscale.
    """
    pass


@sliding_window_attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: Optional[float] = None,
    cache_seqlen: Optional[int] = None,
) -> torch.Tensor:
    return query.new_empty(query.size())


@torch.library.custom_op(
    "spyre::window_band_mask", mutates_args=(), device_types="spyre"
)
def window_band_mask(
    read_start: int,
    q_block: int,
    buffer_width: int,
    q_row_origin: int,
    window_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Additive band over one Q block's KV window.

    Shape: [1, 1, q_block, buffer_width]; 0.0 = keep, -inf = masked. Both
    leading axes are size 1 and broadcast over batch and heads, which keeps
    this at q_block x buffer_width elements rather than one copy per head.

    Query row ``q_row_origin + i`` (an absolute KV-cache coordinate: row 0 of
    this block sits at ``seqlen_kv - seqlen_q + q_block*n``) may attend column
    ``read_start + j`` iff, with ``delta`` the difference of those two absolute
    coordinates:
      - causal:        0 <= delta < window_size
      - bidirectional: abs(delta) < window_size

    The band removes what the buffer over-covers: the stagger between rows
    inside the block, and the columns pulled in when a ragged first/last
    window is shifted to stay inside the cache.

    Takes ``read_start`` and ``q_row_origin`` -- an origin and an extent --
    rather than a block index and the sequence lengths, so the op knows nothing
    about how the caller blocks the query. ``read_start`` here is logical
    (``SlidingWindowPlan.read_start_logical``), not buffer-relative -- see
    the caller in decompositions.py.

    No ``cache_seqlen`` term, deliberately: the rows this op is given satisfy
    ``row < cache_seqlen`` by construction, so the causal ``delta >= 0``
    already forces ``column < cache_seqlen`` and a separate term could never
    disagree. It would become load-bearing only for a bidirectional window,
    where ``abs(delta) < window_size`` admits ``column > row`` -- that mode
    raises Unsupported today and must revisit this mask.

    Built entirely on CPU so the in-place ops stay opaque to torch.compile,
    matching spyre.causal_mask's rationale.
    """
    row = torch.arange(q_block, device="cpu") + q_row_origin
    column = torch.arange(buffer_width, device="cpu") + read_start
    delta = row.unsqueeze(-1) - column.unsqueeze(0)
    if is_causal:
        allowed = (delta >= 0) & (delta < window_size)
    else:
        allowed = delta.abs() < window_size
    mask_cpu = torch.zeros(allowed.shape, dtype=dtype, device="cpu")
    mask_cpu.masked_fill_(~allowed, float("-inf"))
    return mask_cpu.unsqueeze(0).unsqueeze(0).to(device=device)


@window_band_mask.register_fake
def _(
    read_start: int,
    q_block: int,
    buffer_width: int,
    q_row_origin: int,
    window_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(1, 1, q_block, buffer_width, dtype=dtype, device=device)


@torch.library.custom_op("spyre::kv_window", mutates_args=(), device_types="spyre")
def kv_window(  # type: ignore[empty-body]
    key: torch.Tensor,
    value: torch.Tensor,
    read_start: int,
    buffer_width: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read one Q block's slice of the KV cache.

    key/value: [B, Hkv, Lkv, E]. Returns (k_win, v_win) covering cache rows
    [read_start, read_start + buffer_width). k_win is [B, Hq, E, buffer_width],
    already **transposed** -- that is the layout the scores matmul wants, and
    doing it on the slice costs nothing.

    **One block per call.** The caller loops over Q blocks and reuses the same
    window buffer, so the extra memory is buffer_width rows for any query
    length rather than one buffer per block. That reuse is granted by the
    memory planner rather than expressed by the graph -- the blocks are
    independent of each other -- and it is granted today: the planned pool
    grows by one window per unit of buffer_width, not by one per block.

    ``read_start`` comes from SlidingWindowPlan. Passing it in, rather than a
    block index, keeps this op ignorant of how the query is blocked. It is
    bounded against the *allocation* by ``check_window_read``, never against
    the cache's logical position -- a still-filling cache's read routinely
    extends past the written prefix, and causal masking excludes those
    columns (see ``SlidingWindowPlan``).

    **Precondition for a compact buffer** (``buffer_origin != 0``): rows must
    stay contiguous and time-ordered, oldest at row 0, newest last. No
    ring/modulo indexing. Nothing here checks it -- this op only ever sees
    offsets, never tokens.

    **Maintaining that order is out of scope and unproven on this backend.**
    HF uses ``keys.roll(-1, dims=-2)`` plus an in-place overwrite; on Spyre
    only ``aten.copy_`` is known to lower. ``aten::roll`` has no lowering, no
    Spyre decomposition and no CPU fallback, so it would have to survive
    PyTorch's generic decomposition into narrow/cat -- and the ``cat`` route
    is suspect here (see ``TestKVWindowOp``). A caller wiring up a rolled
    cache must verify the write path independently.

    GQA expansion to num_heads happens here, applied to the window slice rather
    than to the full-length cache (see the decomposition for why).

    The mask that goes with this window is a separate op, spyre::window_band_mask
    -- reading the cache and building a band have no arguments in common.

    MUST be called under torch.compile(backend="inductor") on the spyre
    device — this eager body is intentionally empty, matching
    spyre::sliding_window_attention. The real logic is the
    register_spyre_decomposition lowering in decompositions.py.
    """
    pass


@kv_window.register_fake
def _(
    key: torch.Tensor,
    value: torch.Tensor,
    read_start: int,
    buffer_width: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .sliding_window_plan import check_window_read

    reason = check_window_read(
        read_start=read_start,
        buffer_width=buffer_width,
        cache_capacity=key.size(2),
        num_heads=num_heads,
        num_kv_heads=key.size(1),
        key_shape=tuple(key.shape),
        value_shape=tuple(value.shape),
    )
    if reason is not None:
        raise Unsupported(f"kv_window: {reason}")

    batch, _, _, head_dim = key.shape
    k_win = key.new_empty((batch, num_heads, head_dim, buffer_width))
    v_win = value.new_empty((batch, num_heads, buffer_width, head_dim))
    return k_win, v_win


@torch.library.custom_op(
    "spyre::stagger_to_standard_ea", mutates_args=(), device_types="spyre"
)
def stagger_to_standard_ea(x: torch.Tensor) -> torch.Tensor:
    """Restore standard Element Arrangement (EA) from staggered EA.

    The fp32todl16 conversion reorders elements when converting fp32 to fp16,
    producing staggered EA in the output.  This op applies mm(x, P.t()) to
    permute the last dimension back to standard EA so downstream ops see a
    normal fp16 tensor.

    The permutation matrix P encodes the hardware fp32→fp16 stagger pattern.
    Each fp16 stick (64 elements) staggers independently:
        stick_base = (phys // 64) * 64
        local_phys = phys % 64
        k, w = local_phys // 8, local_phys % 8
        local_logical = k*4 + w            if w < 4
                      = k*4 + (w-4) + 32   otherwise   (32 = fp16_stick // 2)
        logical = stick_base + local_logical

    ``half`` is always 32 (half of one fp16 stick), independent of N.
    N = x.shape[-1] must be a multiple of 64.

    P is built on CPU and transferred to the device.  It is determined solely
    by N, so it is constant for a given tensor width and can be reused across
    ops (lt, eq, etc.) that produce the same stagger pattern.

    Input:  fp16 tensor with staggered EA, any shape [..., N]
    Output: fp16 tensor with standard EA, same shape [..., N]
    """
    n = x.shape[-1]
    device = x.device

    # Build N×N permutation matrix on CPU, transfer to device.
    # Each fp16 stick (64 elements) staggers independently with half=32.
    FP16_STICK = 64
    half = FP16_STICK // 2  # 32 — fixed, independent of n
    P = torch.zeros(n, n, dtype=torch.float16, device="cpu")
    for phys_j in range(n):
        stick_base = (phys_j // FP16_STICK) * FP16_STICK
        local_phys = phys_j % FP16_STICK
        k, w = local_phys // 8, local_phys % 8
        local_logical = k * 4 + w if w < 4 else k * 4 + (w - 4) + half
        P[stick_base + local_logical, phys_j] = 1.0
    P = P.to(device=device)

    # Flatten all dims except the last into a single batch dim for mm,
    # then restore the original shape.  This works for any rank >= 1:
    # the stagger pattern only affects the last (stick) dimension, so
    # flattening leading dims produces the correct row ordering for mm.
    orig_shape = list(x.shape)
    m = x.numel() // n
    result = torch.mm(x.reshape(m, n), P.t()).reshape(orig_shape)
    return result


@stagger_to_standard_ea.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("spyre::prod_dim_int", mutates_args=(), device_types="spyre")
def prod_dim_int(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    pass


@prod_dim_int.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    if dim < 0:
        dim += input.ndim
    out_shape = list(input.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape = out_shape[:dim] + out_shape[dim + 1 :]
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)
