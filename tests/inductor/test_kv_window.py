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

"""The window placement, the algorithm it implies, and the op that reads it.

Three levels, in the order a failure is worth reading:

  1. PLACEMENT -- pure integer arithmetic, no device. Where each Q block reads
     from and how wide its buffer is. Everything else rests on one invariant:
     every query row's window lies inside the buffer its block reads.

  2. ALGORITHM -- on CPU: does attending against the compact buffer compute
     the same thing as masking the full cache? Two claims, kept apart because
     they say different things. The set of unmasked columns is *identical*
     (combinatorial, no tolerance) -- that is what licenses the approach. The
     output values agree -- that only says the arithmetic was not fumbled.

  3. THE OP -- spyre::kv_window on device, against slices built
     independently here.

Bit-exactness is not available at level 2 and must not be asserted: the
windowed softmax reduces buffer_width terms where the reference reduces
seqlen_kv, and sum blocks its reduction differently at different widths.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_kv_window.py -v
"""

import math

import pytest
import torch

from torch_spyre._inductor.sliding_window_plan import (
    STICK,
    SlidingWindowPlan,
    check_cache_geometry,
    check_window_read,
    plan_sliding_window,
    query_blocking,
    rejection_reason,
)
from utils_inductor import cached_randn, compare_with_cpu

# (seqlen_q, seqlen_kv, window): prefill, a window wider than one block, a warm
# cache, and decode against a long one.
SHAPES = [
    (512, 512, 64),
    (512, 512, 256),
    (256, 256, 64),
    (128, 512, 64),
    (1, 4096, 64),
    (512, 512, 100),  # window not a whole number of sticks
]
IDS = [f"q{q}_kv{kv}_w{w}" for q, kv, w in SHAPES]

BATCH, HEADS, HEAD_DIM = 1, 8, 64


def _plan(seqlen_q, seqlen_kv, window, q_block=None):
    q_block = query_blocking(seqlen_q)[0] if q_block is None else q_block
    plan = plan_sliding_window(seqlen_q, seqlen_kv, window, q_block=q_block)
    assert plan is not None, f"unsupported: {seqlen_q}/{seqlen_kv}/{window}"
    return plan


# --------------------------------------------------------------- 1. placement


class TestPlacement:
    """Where each block reads, and how wide its buffer is."""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_every_row_window_lies_inside_its_buffer(self, seqlen_q, seqlen_kv, window):
        # The invariant everything downstream rests on. A row whose window
        # escaped its buffer would attend to columns that were never read.
        plan = _plan(seqlen_q, seqlen_kv, window)
        for qi in range(plan.num_q_blocks):
            start = plan.read_start(qi)
            q_start, q_end = plan.block_q_range(qi)
            for q_index in range(q_start, q_end):
                low, high = plan.row_window(q_index)
                assert start <= low and high <= start + plan.buffer_width, (
                    f"block {qi} row {q_index}: window [{low},{high}) escapes "
                    f"buffer [{start},{start + plan.buffer_width})"
                )

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_reads_are_stick_aligned_and_in_bounds(self, seqlen_q, seqlen_kv, window):
        # Unaligned would force a restickify; out of bounds would fault.
        plan = _plan(seqlen_q, seqlen_kv, window)
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        for start in starts:
            assert start % STICK == 0
            assert 0 <= start and start + plan.buffer_width <= seqlen_kv
        assert starts == sorted(starts), "a window may not move backwards"

    def test_prefill_width_is_window_plus_one_block(self):
        # Rows inside a block have staggered windows spanning W + q_block - 1
        # columns, so the buffer is W + q_block. Assuming W is the mistake this
        # arithmetic exists to prevent.
        assert _plan(512, 512, 64).buffer_width == 64 + STICK
        assert _plan(512, 512, 256).buffer_width == 256 + STICK
        # A window that is not a whole number of sticks is fine -- the buffer
        # rounds up to one. 100 + 64 = 164 -> 192.
        assert _plan(512, 512, 100).buffer_width == 192

    def test_decode_width_is_exactly_the_window(self):
        # A single row has no stagger, so decode needs no extra block.
        plan = _plan(1, 4096, 64)
        assert plan.buffer_width == 64
        assert plan.num_q_blocks == 1

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_the_buffer_is_narrower_than_the_cache(self, seqlen_q, seqlen_kv, window):
        # The point of the exercise. Decode is the extreme: 64 rows of 4096.
        assert _plan(seqlen_q, seqlen_kv, window).buffer_width < seqlen_kv

    def test_blocks_tile_the_query_exactly(self):
        plan = _plan(512, 512, 64)
        covered = [
            i for qi in range(plan.num_q_blocks) for i in range(*plan.block_q_range(qi))
        ]
        assert covered == list(range(512))

    def test_row_window_is_the_causal_band(self):
        # The definition: row i at cache coordinate c attends [c-W+1, c].
        plan = _plan(512, 512, 64)
        assert plan.row_window(100) == (37, 101)
        assert plan.row_window(0) == (0, 1)  # clamped at the cache start

    def test_a_window_covering_the_cache_still_plans(self):
        # buffer_width == seqlen_kv is degenerate, not invalid: the read start
        # clamps to 0, every block reads the whole cache, and the band masks
        # it. It saves nothing there but it is not wrong, so it stays on the
        # one code path rather than being refused. Safe only because seqlen_kv
        # is required to be stick aligned -- otherwise ceil_stick could push
        # the buffer past the cache and the clamp negative.
        plan = _plan(128, 128, 128)
        assert plan.buffer_width == 128
        assert all(plan.read_start(n) == 0 for n in range(plan.num_q_blocks))

    def test_decode_masks_nothing_but_prefill_does(self):
        # What lets the body skip the band add at decode -- and only with a
        # stick-aligned window. An unaligned one floors the buffer origin below
        # the row's window start, so the band masks the difference and the add
        # is emitted after all. The body's skip is conditional on this, so both
        # directions are pinned.
        assert _plan(1, 4096, 64).block_is_fully_attended(0)
        assert _plan(1, 4096, 128).block_is_fully_attended(0)
        assert not _plan(1, 4096, 100).block_is_fully_attended(0)
        assert not _plan(1, 4096, 63).block_is_fully_attended(0)
        prefill = _plan(256, 256, 64)
        assert not any(
            prefill.block_is_fully_attended(n) for n in range(prefill.num_q_blocks)
        )


class TestRolledBuffer:
    """cache_capacity < seqlen_kv: a compact buffer that has filled and is
    sliding forward. Only reachable once check_cache_geometry allows the
    logical position past the physical allocation.
    """

    def test_blocks_read_different_offsets_not_the_collapsed_bound(self):
        # A coordinate-system bug shipped here: read_start's clamp compared a
        # logical window_origin (thousands) against a physical bound (tens to
        # hundreds), so min() always returned the bound and every block read
        # the same, mostly-wrong slice. Every SHAPES case above has
        # cache_capacity == seqlen_kv, where the bug is invisible -- physical
        # and logical coincide -- so this needs its own non-degenerate case.
        plan = plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=1024)
        assert plan is not None
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        assert len(set(starts)) > 1, f"collapsed to one value: {starts}"
        assert starts == sorted(starts), "a window may not move backwards"
        assert all(0 <= s and s + plan.buffer_width <= 1024 for s in starts)

    def test_read_start_logical_is_read_start_plus_buffer_origin(self):
        plan = plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=1024)
        for qi in range(plan.num_q_blocks):
            assert plan.read_start(qi) + plan.buffer_origin == plan.read_start_logical(
                qi
            )

    def test_every_row_window_lies_inside_the_logical_buffer(self):
        # The same invariant TestPlacement checks against read_start, checked
        # here against read_start_logical -- the one window_band_mask's
        # row/column delta actually needs to hold, since q_row_origin is
        # logical too.
        plan = plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=1024)
        for qi in range(plan.num_q_blocks):
            start = plan.read_start_logical(qi)
            q_start, q_end = plan.block_q_range(qi)
            for q_index in range(q_start, q_end):
                low, high = plan.row_window(q_index)
                assert start <= low and high <= start + plan.buffer_width

    def test_buffer_origin_is_zero_until_the_cache_outgrows_its_allocation(self):
        assert plan_sliding_window(512, 512, 64, q_block=64).buffer_origin == 0
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert plan.buffer_origin == 5120 - 64

    def test_a_buffer_sized_to_exactly_one_window_always_reads_from_zero(self):
        # capacity == buffer_width: the buffer holds nothing but the live
        # window, so there is nowhere to shift to regardless of how far the
        # logical position has grown.
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert all(plan.read_start(qi) == 0 for qi in range(plan.num_q_blocks))

    def test_a_buffer_too_small_for_a_multiblock_prefill_is_refused_up_front(self):
        # cache_capacity=192 clears the "narrower than one window" rejection
        # (buffer_width=128) but is still too small to serve the earliest
        # block of this 8-block prefill: its window reaches further back than
        # a 192-row rolled buffer holds. Previously this was only caught
        # later as a negative read_start via check_window_read's existing
        # bound -- loud, not silent, but from a less specific error site.
        # rejection_reason now names it directly.
        reason = rejection_reason(512, 5120, 64, True, 64, cache_capacity=192)
        assert reason is not None and "does not reach far enough back" in reason
        assert (
            plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=192) is None
        )

    def test_a_single_block_plan_is_unaffected_by_the_reach_check(self):
        # Decode and single-chunk prefill only ever have one block, so the
        # earliest-block check is checking the only block -- it must not
        # reject a shape check_cache_geometry and buffer_width already
        # accepted.
        assert rejection_reason(1, 5120, 64, True, 1, cache_capacity=64) is None
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert plan is not None


class TestStillFilling:
    """cache_seqlen < cache_capacity: a cache that has not filled its
    allocation yet, left-aligned (buffer_origin == 0, already correct without
    any change -- see TestRolledBuffer's sibling test for the outgrown case).

    The obvious design here adds a mask for the unwritten tail
    [cache_seqlen, cache_capacity) and a matching block_is_fully_attended
    condition. Neither is needed: rejection_reason already requires
    cache_seqlen stick-aligned, and every block's read is stick-aligned
    arithmetic built from a buffer_width sized to the *widest* block. Per-
    block required width is monotonic non-decreasing in block index (proven
    by test_per_block_width_is_monotonic below), so the last block is always
    that widest block, and its own width reaches cache_seqlen with zero
    slack once it -- already stick-aligned -- is ceil_stick'd. No block's
    read ever reaches the tail; the existing causal band is the only mask
    needed.
    """

    def test_a_still_filling_cache_plans_like_any_other(self):
        assert (
            plan_sliding_window(1, 128, 64, q_block=1, cache_capacity=4096) is not None
        )
        assert (
            plan_sliding_window(256, 256, 64, q_block=64, cache_capacity=4096)
            is not None
        )

    def test_buffer_origin_is_zero_while_still_filling(self):
        plan = plan_sliding_window(1, 128, 64, q_block=1, cache_capacity=4096)
        assert plan.buffer_origin == 0

    def test_per_block_width_is_monotonic(self):
        # The property the no-overshoot proof rests on: a later block never
        # needs a narrower buffer than an earlier one, so sizing buffer_width
        # to the max is the same as sizing it to the LAST block.
        for seqlen_q in (128, 256, 384, 512):
            for window in (63, 64, 100, 128, 150, 192, 201, 300):
                widths = []
                for qi in range(-(-seqlen_q // STICK)):
                    q_start = qi * STICK
                    q_end = min(seqlen_q, q_start + STICK)
                    first_coord, last_coord = q_start, q_end - 1
                    window_origin = max(
                        0, ((first_coord - window + 1) // STICK) * STICK
                    )
                    widths.append(last_coord - max(0, window_origin) + 1)
                assert widths == sorted(widths), (seqlen_q, window, widths)

    @pytest.mark.parametrize("seqlen_q", [64, 128, 192, 256, 320, 384, 448, 512])
    @pytest.mark.parametrize("window", [64, 100, 128, 150, 192, 63, 201, 300, 33])
    def test_no_block_ever_reads_past_cache_seqlen(self, seqlen_q, window):
        # The swept proof the class docstring and check_cache_geometry's
        # docstring both point to: for every stick-aligned cache_seqlen a
        # still-filling shape can reach, no block's physical read extends
        # into the allocated-but-unwritten tail.
        capacity = 4096
        for cache_seqlen in (seqlen_q + k * STICK for k in range(0, 10)):
            if cache_seqlen >= capacity:
                continue
            plan = plan_sliding_window(
                seqlen_q, cache_seqlen, window, q_block=STICK, cache_capacity=capacity
            )
            if plan is None:
                continue
            for qi in range(plan.num_q_blocks):
                stop = plan.read_start_logical(qi) + plan.buffer_width
                assert stop <= cache_seqlen, (
                    f"block {qi} reads to {stop}, past cache_seqlen={cache_seqlen}"
                )

    def test_no_block_ever_reads_past_cache_seqlen_at_decode(self):
        capacity = 4096
        for cache_seqlen in range(STICK, capacity, STICK):
            for window in (64, 100, 128, 150, 192, 63, 201, 300, 1):
                plan = plan_sliding_window(
                    1, cache_seqlen, window, q_block=1, cache_capacity=capacity
                )
                if plan is None:
                    continue
                stop = plan.read_start_logical(0) + plan.buffer_width
                assert stop <= cache_seqlen


class TestQueryPadding:
    """A query that does not divide into blocks is padded, not refused."""

    @pytest.mark.parametrize(
        "seqlen_q,expected",
        [(1, (1, 1)), (64, (64, 64)), (100, (64, 128)), (257, (64, 320))],
    )
    def test_block_size_and_padded_length(self, seqlen_q, expected):
        # Decode takes a single-row block and no padding: padding one row up to
        # a full block would be 64x the work for one row of output.
        assert query_blocking(seqlen_q) == expected

    @pytest.mark.parametrize("seqlen_q", [33, 100, 200, 257])
    def test_front_padding_preserves_every_real_coordinate(self, seqlen_q):
        # The claim the whole scheme rests on. Row i sits at cache coordinate
        # seqlen_kv - seqlen_q + i; once seqlen_q is the PADDED length the two
        # shifts cancel, so every real row keeps the coordinate its window is
        # derived from. Back-padding would move all of them.
        seqlen_kv, window = 512, 64
        _, padded = query_blocking(seqlen_q)
        pad = padded - seqlen_q
        plan = _plan(padded, seqlen_kv, window)
        for i in range(seqlen_q):
            coord = seqlen_kv - seqlen_q + i
            assert plan.row_window(pad + i) == (
                max(0, coord - window + 1),
                min(seqlen_kv, coord + 1),
            )

    @pytest.mark.parametrize("seqlen_q", [33, 100, 200, 257])
    def test_no_padded_row_is_fully_masked(self, seqlen_q):
        # A row with an empty window would make softmax produce nan, which
        # would poison the real rows through no fault of theirs.
        plan = _plan(query_blocking(seqlen_q)[1], 512, 64)
        for j in range(plan.seqlen_q):
            low, high = plan.row_window(j)
            assert low < high


class TestRejection:
    """Shapes the placement declines, and the reason it gives for each."""

    REJECTED = [
        ((256, 256, 64, False, 64), "causal"),
        ((256, 256, 0, True, 64), "window_size=0 must be positive"),
        ((256, 257, 64, True, 64), "pad the KV cache"),
        ((100, 256, 64, True, 64), "whole number of 64-row blocks"),
        ((512, 256, 64, True, 64), "exceeds seqlen_kv"),
    ]

    @pytest.mark.parametrize("args,expected", REJECTED)
    def test_each_reason_names_its_constraint(self, args, expected):
        # This message reaches a user in a compile-time warning, so
        # "unsupported shape" would not have been good enough.
        reason = rejection_reason(*args)
        assert reason is not None, f"{args} should be rejected"
        assert expected in reason, f"{args}: {reason}"

    def test_the_plan_agrees_with_the_reason(self):
        # rejection_reason is what plan_sliding_window decides with, so a shape
        # with a reason must not produce a plan. q_block=None means
        # query_blocking always returns a usable block size now, so a shape
        # with a reason is rejected on that reason rather than on the block.
        for args, _ in self.REJECTED:
            assert plan_sliding_window(*args[:4], q_block=args[4]) is None, args

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_supported_shapes_give_no_reason(self, seqlen_q, seqlen_kv, window):
        assert (
            rejection_reason(
                seqlen_q, seqlen_kv, window, True, query_blocking(seqlen_q)[0]
            )
            is None
        )

    def test_cache_capacity_is_optional_and_backward_compatible(self):
        # Omitted: every caller before this parameter existed sees no change.
        without = rejection_reason(512, 5120, 64, True, 64)
        assert without is None

    def test_cache_capacity_must_be_stick_aligned(self):
        # read_start's physical conversion needs buffer_origin
        # (seqlen_kv - cache_capacity) stick aligned; seqlen_kv already is.
        reason = rejection_reason(512, 5120, 64, True, 64, cache_capacity=1000)
        assert reason is not None and "cache_capacity=1000" in reason

    def test_cache_capacity_must_fit_at_least_one_buffer(self):
        # buffer_width for this shape is window(64) + q_block(64) = 128.
        reason = rejection_reason(512, 5120, 64, True, 64, cache_capacity=64)
        assert reason is not None and "narrower than the 128-row buffer" in reason

    def test_a_sufficient_cache_capacity_gives_no_reason(self):
        assert rejection_reason(512, 5120, 64, True, 64, cache_capacity=1024) is None

    def test_all_three_orderings_of_seqlen_and_capacity_are_accepted(self):
        # Reading only cares which side of cache_capacity the cache is on:
        # exactly full (equal), rolled and outgrown its allocation
        # (seqlen > capacity), and still filling it (seqlen < capacity, left-
        # aligned -- see TestStillFilling for why no masking change was
        # needed to support this last one).
        assert check_cache_geometry(256, 256) is None
        assert check_cache_geometry(5000, 64) is None
        assert check_cache_geometry(100, 4096) is None
        assert "must be positive" in check_cache_geometry(0, 256)
        assert "must be positive" in check_cache_geometry(256, 0)

    def test_a_hand_built_read_is_validated(self):
        # The op takes its placement as plain ints, so a caller that does not
        # go through the plan can walk off the end of the cache.
        shape = (BATCH, HEADS, 256, HEAD_DIM)
        ok = dict(
            read_start=0,
            buffer_width=128,
            cache_capacity=256,
            num_heads=8,
            num_kv_heads=8,
            key_shape=shape,
            value_shape=shape,
        )
        assert check_window_read(**ok) is None
        assert check_window_read(**{**ok, "read_start": 128}) is None  # last legal
        assert "runs past the cache" in check_window_read(**{**ok, "read_start": 129})
        assert check_window_read(**{**ok, "read_start": -1}) is not None
        assert "whole multiple" in check_window_read(**{**ok, "num_kv_heads": 3})
        # MLA-style: V's head_dim differs from K's -- the fake kernel's shape
        # would otherwise silently come from the wrong tensor.
        mismatched = (BATCH, HEADS, 256, HEAD_DIM * 2)
        assert "does not match" in check_window_read(
            **{**ok, "value_shape": mismatched}
        )

    def test_a_hand_built_read_into_the_unwritten_tail_is_rejected(self):
        # cache_seqlen=64 < cache_capacity=256: only rows [0, 64) are written.
        # Before cache_seqlen was threaded into kv_window, a caller bypassing
        # the plan could read [0, 128) here -- inside cache_capacity, so the
        # existing bound passed it -- and silently read 64 uninitialized rows.
        shape = (BATCH, HEADS, 256, HEAD_DIM)
        ok = dict(
            read_start=0,
            buffer_width=128,
            cache_capacity=256,
            num_heads=8,
            num_kv_heads=8,
            key_shape=shape,
            value_shape=shape,
        )
        reason = check_window_read(**ok, cache_seqlen=64)
        assert reason is not None and "unwritten tail" in reason
        assert check_window_read(**ok, cache_seqlen=128) is None  # reaches exactly
        assert check_window_read(**ok, cache_seqlen=256) is None  # exactly full
        assert check_window_read(**ok, cache_seqlen=5000) is None  # rolled, past cap
        assert check_window_read(**ok) is None  # omitted: unchanged


# --------------------------------------------------------------- 2. algorithm


def _expand_kv(tensor, expansion):
    if expansion == 1:
        return tensor
    return tensor.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)


def _reference_allowed(seqlen_q, seqlen_kv, window):
    """Full [seqlen_q, seqlen_kv] boolean band -- the definition of SWA."""
    q_coord = torch.arange(seqlen_q) + (seqlen_kv - seqlen_q)
    delta = q_coord.unsqueeze(-1) - torch.arange(seqlen_kv).unsqueeze(0)
    return (delta >= 0) & (delta < window)


def _block_allowed(plan: SlidingWindowPlan, qi: int) -> torch.Tensor:
    """[q_len, buffer_width] boolean band inside block qi's buffer."""
    start = plan.read_start(qi)
    cols = torch.arange(start, start + plan.buffer_width)
    q_start, q_end = plan.block_q_range(qi)
    rows = []
    for q_index in range(q_start, q_end):
        low, high = plan.row_window(q_index)
        rows.append((cols >= low) & (cols < high))
    return torch.stack(rows)


def _additive(allowed, dtype):
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype),
        torch.full((), float("-inf"), dtype=dtype),
    )


def _windowed_attention(query, key, value, plan, scale):
    """Per block, attend only against the compact window."""
    expansion = query.size(1) // key.size(1)
    out_blocks = []
    for qi in range(plan.num_q_blocks):
        q_start, q_end = plan.block_q_range(qi)
        start = plan.read_start(qi)
        stop = start + plan.buffer_width
        scores = (
            torch.matmul(
                query[:, :, q_start:q_end, :],
                _expand_kv(key[:, :, start:stop, :], expansion).transpose(-1, -2),
            )
            * scale
        )
        scores = scores + _additive(_block_allowed(plan, qi), query.dtype)
        out_blocks.append(
            torch.matmul(
                torch.softmax(scores, dim=-1),
                _expand_kv(value[:, :, start:stop, :], expansion),
            )
        )
    return torch.cat(out_blocks, dim=2)


def _masked_attention(query, key, value, window, scale):
    """The definition, computed literally."""
    expansion = query.size(1) // key.size(1)
    scores = torch.matmul(query, _expand_kv(key, expansion).transpose(-1, -2)) * scale
    scores = scores + _additive(
        _reference_allowed(query.size(2), key.size(2), window), query.dtype
    )
    return torch.matmul(torch.softmax(scores, dim=-1), _expand_kv(value, expansion))


def _cpu_inputs(seqlen_q, seqlen_kv, dtype=torch.float64, kvheads=HEADS):
    generator = torch.Generator().manual_seed(0)

    def make(shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    return (
        make((BATCH, HEADS, seqlen_q, HEAD_DIM)),
        make((BATCH, kvheads, seqlen_kv, HEAD_DIM)),
        make((BATCH, kvheads, seqlen_kv, HEAD_DIM)),
    )


class TestAlgorithm:
    """Does the compact window compute the same attention as the full mask?"""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_unmasked_column_sets_are_identical(self, seqlen_q, seqlen_kv, window):
        # Claim 1, exact. Combinatorial, so no tolerance is involved, and this
        # is what licenses the whole approach.
        plan = _plan(seqlen_q, seqlen_kv, window)
        reference = _reference_allowed(seqlen_q, seqlen_kv, window)
        for qi in range(plan.num_q_blocks):
            q_start, q_end = plan.block_q_range(qi)
            start = plan.read_start(qi)
            block = _block_allowed(plan, qi)
            for i, q_index in enumerate(range(q_start, q_end)):
                gathered = (block[i].nonzero().flatten() + start).tolist()
                assert gathered == reference[q_index].nonzero().flatten().tolist()
            # A fully masked row would make softmax produce nan.
            assert block.any(dim=-1).all()

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_output_matches_the_masked_reference(self, seqlen_q, seqlen_kv, window):
        # Claim 2, roundoff. float64, so this is about the arithmetic rather
        # than about float16.
        plan = _plan(seqlen_q, seqlen_kv, window)
        query, key, value = _cpu_inputs(seqlen_q, seqlen_kv)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        actual = _windowed_attention(query, key, value, plan, scale)
        expected = _masked_attention(query, key, value, window, scale)
        assert torch.isfinite(actual).all()
        assert (actual - expected).abs().max().item() < 1e-12

    def test_gqa_matches_in_float16(self):
        # The dtype the device runs in, the GQA expand, and the scale split
        # across query and key the way the decomposition does it -- at the
        # tolerance the device tests use.
        plan = _plan(256, 256, 64)
        query, key, value = _cpu_inputs(256, 256, dtype=torch.float16, kvheads=2)
        split = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))
        actual = _windowed_attention(query * split, key * split, value, plan, 1.0)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            _additive(_reference_allowed(256, 256, 64), torch.float16),
            enable_gqa=True,
        )
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_windowing_actually_changes_the_answer(self):
        # Guards the agreement tests above: if the band were a no-op they would
        # pass while proving nothing.
        plan = _plan(256, 256, 64)
        query, key, value = _cpu_inputs(256, 256)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        windowed = _windowed_attention(query, key, value, plan, scale)
        causal = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale
        )
        assert (windowed - causal).abs().max().item() > 1e-3


# ------------------------------------------------------------------ 3. the op


def _device_cache(differentiation, seqlen_kv=256, kvheads=HEADS):
    return cached_randn(
        (BATCH, kvheads, seqlen_kv, HEAD_DIM),
        differentiation=differentiation,
        dtype=torch.float16,
    )


def _call_op(cache, plan, block_index, which, value_cache=None):
    """which: 0 -> k_win, 1 -> v_win, 2 -> band.

    value_cache defaults to cache: most callers only inspect k_win or the
    band, where key and value being the same tensor is harmless. A test that
    inspects v_win must pass a value_cache that actually differs from cache,
    or a v_win sourced from the wrong tensor would pass unnoticed.

    The window and the band are separate ops -- they share no arguments -- so
    2 dispatches to the mask rather than indexing a third return value.
    """
    value_cache = cache if value_cache is None else value_cache
    q_start, _ = plan.block_q_range(block_index)
    if which == 2:
        return torch.ops.spyre.window_band_mask(
            plan.read_start(block_index),
            plan.q_block,
            plan.buffer_width,
            plan.q_kv_offset + q_start,
            plan.window_size,
            plan.is_causal,
            cache.dtype,
            cache.device,
        )
    return torch.ops.spyre.kv_window(
        cache,
        value_cache,
        plan.read_start(block_index),
        plan.buffer_width,
        HEADS,
    )[which]


def _reference_window(cache, plan, block_index, transpose=False):
    start = plan.read_start(block_index)
    window = cache[:, :, start : start + plan.buffer_width, :]
    if transpose:
        window = window.transpose(-1, -2)
    return _expand_kv(window, HEADS // cache.size(1))


def _reference_band(plan, block_index):
    band = _additive(_block_allowed(plan, block_index), torch.float16)
    return band.unsqueeze(0).unsqueeze(0)


class TestKVWindowOp:
    """The op on device, against slices built here.

    GQA is checked one block at a time, which is what the body consumes.
    Cat-ing several expanded windows together zeroes the leading slots on
    device -- a backend defect unrelated to this op, and nothing on this path
    does it.
    """

    def test_window_and_band(self):
        # All three outputs of one call, compared as a tuple so a failure names
        # which of them was wrong. k and v hold different values (differentiation
        # 1 vs 2), so a v_win sourced from the wrong tensor cannot pass by luck.
        plan = _plan(256, 256, 64)

        def fn(k, v):
            blocks = range(plan.num_q_blocks)
            if k.device.type == "spyre":
                return tuple(
                    torch.cat(
                        [_call_op(k, plan, n, which, value_cache=v) for n in blocks],
                        dim=1,
                    )
                    for which in (0, 1, 2)
                )
            return (
                torch.cat(
                    [_reference_window(k, plan, n, transpose=True) for n in blocks],
                    dim=1,
                ),
                torch.cat([_reference_window(v, plan, n) for n in blocks], dim=1),
                torch.cat([_reference_band(plan, n) for n in blocks], dim=1),
            )

        compare_with_cpu(fn, _device_cache(1), _device_cache(2), run_eager=False)

    def test_gqa_key_window_for_one_block(self):
        # Block 2 is the first whose read start is not 0, so a dropped window
        # shift shows up here rather than passing by accident.
        plan = _plan(256, 256, 64)
        assert plan.read_start(0) == 0 and plan.read_start(2) > 0

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 2, 0)
            return _reference_window(k, plan, 2, transpose=True)

        compare_with_cpu(
            fn,
            _device_cache(1, kvheads=2),
            _device_cache(2, kvheads=2),
            run_eager=False,
        )

    def test_decode_window(self):
        # buffer_width == window: 64 rows out of a 4096-row cache.
        plan = _plan(1, 4096, 64, q_block=1)
        cache = _device_cache(3, seqlen_kv=4096)

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 0, 0)
            return _reference_window(k, plan, 0, transpose=True)

        compare_with_cpu(fn, cache, cache, run_eager=False)

    def test_decode_band_is_all_zeros(self):
        # What the body relies on to skip the band add entirely.
        plan = _plan(1, 4096, 64, q_block=1)
        cache = _device_cache(3, seqlen_kv=4096)
        assert torch.equal(
            _reference_band(plan, 0), torch.zeros((1, 1, 1, 64), dtype=torch.float16)
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 0, 2)
            return _reference_band(plan, 0)

        compare_with_cpu(fn, cache, cache, run_eager=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
