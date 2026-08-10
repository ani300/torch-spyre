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

"""End-to-end correctness of spyre::sliding_window_attention.

Compared against the definition: full SDPA over the whole cache behind a band
mask. Unsupported shapes raise rather than falling back, so numbers coming out
at all prove the windowed path ran; which shapes are refused is settled in
test_kv_window.py without a device.

This does NOT establish that the spyre_hints produce device loops — untiled
code returns the right answer with one large intermediate.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py -v
"""

import unittest

import torch
import torch._dynamo
import torch.nn.functional as F

from utils_inductor import cached_randn, compare_with_cpu


def _band_mask(
    seqlen_q: int, seqlen_kv: int, window_size: int, dtype=torch.float16
) -> torch.Tensor:
    """Full [1, 1, Lq, Lkv] causal sliding-window mask -- the definition."""
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).unsqueeze(-1)
    k_pos = torch.arange(seqlen_kv).unsqueeze(0)
    delta = q_pos - k_pos
    allowed = (delta >= 0) & (delta < window_size)
    mask = torch.zeros(seqlen_q, seqlen_kv, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _attention(q, k, v, window_size):
    """Dispatch: the op on spyre, the masked reference on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(q, k, v, window_size, True)
    mask = _band_mask(q.size(2), k.size(2), window_size)
    return F.scaled_dot_product_attention(
        q, k, v, mask, enable_gqa=q.size(1) != k.size(1)
    )


def _inputs(batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim=64):
    query = cached_randn(
        (batch, heads, seqlen_q, head_dim), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (batch, kvheads, seqlen_kv, head_dim), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (batch, kvheads, seqlen_kv, head_dim), differentiation=3, dtype=torch.float16
    )
    return query, key, value


class TestSlidingWindowAttention(unittest.TestCase):
    """Shapes the op supports, against the masked reference."""

    def setUp(self):
        torch._dynamo.reset()

    def test_prefill_mha(self):
        # 4 blocks of 64, a 128-row window each.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_mha_wider_window(self):
        # W=128 -> a 192-row window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_prefill_gqa(self):
        # 8 query heads from 2 kv heads; the expand is inside the op.
        query, key, value = _inputs(1, 8, 2, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_batch(self):
        query, key, value = _inputs(2, 4, 4, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_head_dim_128(self):
        # Two sticks per row where 64 is one; the placement is in rows.
        query, key, value = _inputs(1, 8, 8, 256, 256, head_dim=128)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_long(self):
        # 32 blocks — a long unrolled loop rather than a handful.
        query, key, value = _inputs(1, 8, 8, 2048, 2048)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode(self):
        # One block reading exactly W rows: 64 of 4096.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode_gqa(self):
        query, key, value = _inputs(1, 8, 2, 1, 512)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_decode_long_cache(self):
        query, key, value = _inputs(1, 8, 8, 1, 8192)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_chunked_prefill(self):
        # Lq < Lkv: prefill continuing a warm cache.
        query, key, value = _inputs(1, 8, 8, 128, 512)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_query_length_not_a_multiple_of_the_block(self):
        # Lq=100 padded to 128 at the front. Back-padding would shift every
        # real row 28 positions and this would catch it.
        query, key, value = _inputs(1, 8, 8, 100, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode_window_not_a_multiple_of_the_stick(self):
        # The only decode case where the band add is emitted: W=64/128 mask
        # nothing and skip it.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_window_not_a_multiple_of_the_stick(self):
        # W=100: buffer rounds up to a stick, band masks by the true window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_window_covering_the_whole_cache(self):
        # buffer_width == seqlen_kv: degenerate, not a separate code path.
        query, key, value = _inputs(1, 8, 8, 128, 128)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_explicit_cache_seqlen_matches_the_default(self):
        # cache_seqlen defaults to the cache's allocated rows, so passing that
        # same number explicitly must not move a single window.
        query, key, value = _inputs(1, 8, 8, 128, 512)

        def attention(q, k, v, window_size):
            if q.device.type == "spyre":
                return torch.ops.spyre.sliding_window_attention(
                    q, k, v, window_size, True, None, k.size(2)
                )
            mask = _band_mask(q.size(2), k.size(2), window_size)
            return F.scaled_dot_product_attention(q, k, v, mask)

        compare_with_cpu(attention, query, key, value, 64, run_eager=False)

    def test_ragged_query_and_window_together(self):
        # An off-by-one in the pad arithmetic can survive either alone.
        query, key, value = _inputs(1, 8, 2, 100, 512)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)


if __name__ == "__main__":
    unittest.main()
