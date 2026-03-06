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

import math

import torch
import torch_spyre.ops.fallbacks  # noqa: F401


def maybe_wrap_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        return dim + ndims
    return dim


@torch.library.register_kernel("aten::mm", ["spyre"])  # type:ignore
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])  # type:ignore
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])  # type:ignore
def spyre__fill_scalar(
    self: torch.Tensor, other: int | float | bool | complex
) -> torch.Tensor:
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self


@torch.library.register_kernel("aten::normal_", ["spyre"])  # type:ignore
def spyre__normal_(self, mean=0.0, std=1.0, *, generator=None):
    # "normal_" generates a random tensor, thus copying
    # "self" back from SPYRE to CPU is not needed.
    # cpu_tmp = self.to("cpu")

    # Create a new tensor on cpu itself to avoid unnecessary data copy.
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)
    cpu_tmp.normal_(mean, std, generator=generator)
    self.copy_(cpu_tmp)
    return self


@torch.library.register_kernel("aten::zero_", ["spyre"])  # type:ignore
def spyre__zero_(self: torch.Tensor) -> torch.Tensor:
    """Zero out the tensor in-place."""
    # Create zeros on CPU
    tmp = torch.zeros(self.size(), dtype=self.dtype, device="cpu")
    # Copy to device
    self.copy_(tmp)
    # TODO: Can we zero out tensors in-place without copy
    return self


@torch.library.register_kernel("aten::silu.out", ["spyre"])  # type:ignore
def spyre__silu_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_silu = torch.compile(torch.ops.aten.silu.out, dynamic=False)
    return compiled_silu(self, out=out)


@torch.library.register_kernel("aten::mish.out", ["spyre"])  # type:ignore
def spyre__mish_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_mish = torch.compile(torch.ops.aten.mish.out, dynamic=False)
    return compiled_mish(self, out=out)


@torch.library.register_kernel("aten::uniform_", "spyre")  # type:ignore
def spyre__uniform_(self, from_=0.0, to=1.0, generator=None):
    # Create a new tensor on cpu
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)

    # Fill the CPU tensor with uniform random values
    cpu_tmp.uniform_(from_, to, generator=generator)

    # Copy the CPU tensor back to the spyre device
    self.copy_(cpu_tmp)

    return self


@torch.library.register_kernel(
    "aten::_scaled_dot_product_fused_attention_overrideable", ["spyre"]
)  # type:ignore
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
    def _sdpa_overrideable(
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
        max_seqlen_q = query.size(2)
        max_seqlen_kv = key.size(2)

        query = query.clone(memory_format=torch.contiguous_format)
        key = key.clone(memory_format=torch.contiguous_format)
        value = value.clone(memory_format=torch.contiguous_format)

        scaling_factor = scale
        if scaling_factor is None:
            scaling_factor = 1.0 / math.sqrt(query.shape[-1])
        scaling_factor = math.sqrt(scaling_factor)

        scaling_factor = torch.full_like(query, scaling_factor)

        query = query * scaling_factor
        key = key * scaling_factor

        key_t = key.transpose(-2, -1).clone(memory_format=torch.contiguous_format)

        attn = torch.matmul(query, key_t)

        if is_causal:
            assert attn_bias is None
            attn_bias = torch.full_like(attn, float("-inf"))
            attn_bias = attn_bias.triu(diagonal=1)

        if attn_bias is not None:
            attn.add_(attn_bias)

        # TODO (aviros): Switch to _safe_softmax
        attn = torch.softmax(attn, -1)

        if dropout_p > 0.0:
            # TODO(aviros): Implement
            pass

        # Unused for now
        logsumexp = torch.empty(
            (batch_size, num_heads, max_seqlen_q), dtype=torch.float16, device="spyre"
        )
        philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
        philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

        # B, H, S, E
        out = torch.matmul(attn, value)

        # B, S, H, E
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

    compiled_sdpa = torch.compile(_sdpa_overrideable, dynamic=False)
    return compiled_sdpa(
        query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
    )


# INSERT_CODEGEN_HERE
