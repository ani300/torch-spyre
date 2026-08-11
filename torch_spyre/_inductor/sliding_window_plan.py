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

"""Where each Q block reads its slice of the KV cache from.

Pure integer arithmetic — no torch — so the placement is testable without
hardware. The invariant everything rests on: every query row's window lies
inside the buffer its block reads.
"""

from dataclasses import dataclass

# Elements per 128-byte stick at fp16. At float32 a stick holds 32 and the
# alignment reasoning here does not carry.
STICK = 64


def _floor_stick(value: int) -> int:
    """Round down to a stick boundary, for negative values too."""
    return (value // STICK) * STICK


def _ceil_stick(value: int) -> int:
    """Round up to a stick boundary."""
    return -(-value // STICK) * STICK


@dataclass(frozen=True)
class SlidingWindowPlan:
    """Where each Q block reads, in absolute cache coordinates: row ``i`` sits
    at ``q_kv_offset + i``, which is what lets prefill and decode share one
    formula.

    Two lengths, deliberately separate. ``seqlen_kv`` is the cache's logical
    position (HF's ``cumulative_length``, vLLM's ``seq_lens``) -- what a
    coordinate *means*, so ``row_window`` clamps against it.
    ``cache_capacity`` is ``key.size(2)``, the rows physically allocated --
    what a *read* may not run past, so ``read_start`` clamps against it.
    Either may be larger:

    - ``seqlen_kv <= cache_capacity`` -- still filling. ``buffer_origin`` 0,
      left-aligned; ``[seqlen_kv, cache_capacity)`` is allocated but
      unwritten. A read routinely lands past ``seqlen_kv`` (which need not be
      stick-aligned) and that is safe: every query row's own coordinate is
      ``< seqlen_kv``, so causal masking's ``column <= row`` already excludes
      those columns. They must still hold *finite* values -- an additive
      ``-inf`` cannot rescue a ``NaN``.
    - ``seqlen_kv > cache_capacity`` -- a compact buffer that has filled and
      is sliding forward, ``buffer_origin > 0``. Assumes rows stay contiguous
      and time-ordered, oldest at the front; unchecked here, see ``kv_window``.
    - Equal -- the full-length, exactly-full cache.
    """

    seqlen_q: int
    seqlen_kv: int
    window_size: int
    q_block: int
    num_q_blocks: int
    buffer_width: int
    q_kv_offset: int
    is_causal: bool
    cache_capacity: int

    def block_q_range(self, qi: int) -> tuple[int, int]:
        """Half-open query-row range of Q block ``qi``."""
        q_start = qi * self.q_block
        return q_start, min(self.seqlen_q, q_start + self.q_block)

    def row_window(self, q_index: int) -> tuple[int, int]:
        """Half-open cache range query row ``q_index`` may attend to."""
        coord = self.q_kv_offset + q_index
        lo = max(0, coord - self.window_size + 1)
        hi = min(self.seqlen_kv, coord + 1)
        return lo, hi

    @property
    def buffer_origin(self) -> int:
        """Logical position of the buffer's physical row 0.

        0 while the cache fits inside its own allocation, so physical and
        logical coordinates coincide. Once it has outgrown the allocation this
        is where its oldest resident row sits.
        """
        return max(0, self.seqlen_kv - self.cache_capacity)

    def block_is_fully_attended(self, qi: int) -> bool:
        """True when block ``qi``'s band masks nothing, so the add can be skipped.

        Decode qualifies only for a stick-aligned window: otherwise read_start
        floors the origin below the row's window start (W=100, kv=4096: buffer
        from 3968, row reaches 3996) and the band masks the difference.

        Compares against ``read_start_logical`` because ``row_window`` speaks
        logical coordinates and both sides must agree on the space.
        """
        start = self.read_start_logical(qi)
        stop = start + self.buffer_width
        q_start, q_end = self.block_q_range(qi)
        return all(
            self.row_window(q_index)[0] <= start and self.row_window(q_index)[1] >= stop
            for q_index in range(q_start, q_end)
        )

    def read_start(self, qi: int) -> int:
        """Buffer-relative offset block ``qi`` reads from -- what a physical
        slice of the cache tensor needs (``kv_window``'s argument).

        The clamps are the point: the ragged first and last blocks *shift*
        rather than shrink, so every block's buffer is one shape and one
        allocation serves them all. The band removes what the shift drags in.

        Floors in PHYSICAL coordinates -- convert to buffer-relative first,
        round down to a stick after. The other order only lands on a stick
        boundary when ``buffer_origin`` is itself a stick multiple, which an
        arbitrary rolled-buffer ``seqlen_kv`` does not give; it is identical
        whenever ``buffer_origin`` is one.

        When ``cache_capacity == buffer_width`` the buffer holds exactly one
        window, so this is 0 for every block: there is nowhere to shift to.
        """
        q_start, _ = self.block_q_range(qi)
        first_coord = self.q_kv_offset + q_start
        window_start_logical = max(0, first_coord - self.window_size + 1)
        physical_origin = _floor_stick(window_start_logical - self.buffer_origin)
        return min(physical_origin, self.cache_capacity - self.buffer_width)

    def read_start_logical(self, qi: int) -> int:
        """``read_start`` converted back to a logical coordinate.

        ``window_band_mask`` needs this, not ``read_start``: its row side is
        logical, so the column side must be too for ``delta = row - column``
        to mean anything. A no-op whenever ``buffer_origin`` is 0.
        """
        return self.read_start(qi) + self.buffer_origin


def check_window_read(
    read_start: int,
    buffer_width: int,
    cache_capacity: int,
    num_heads: int,
    num_kv_heads: int,
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
) -> str | None:
    """Why this read is invalid, or None.

    kv_window takes its placement as plain ints, so a caller bypassing the plan
    can walk off the cache. Whether a *shape* is windowable is
    ``rejection_reason``'s question. Strings not exceptions, to keep this
    module free of torch and the backend's error classes.

    Bounds the read against the *allocation*, deliberately not against
    ``seqlen_kv``: a still-filling cache's buffer routinely extends past its
    written prefix, and causal masking already excludes those columns (see
    ``SlidingWindowPlan``), so a logical bound here would reject most valid
    plans.
    """
    if read_start < 0:
        return f"read_start={read_start} is negative"
    if buffer_width <= 0:
        return f"buffer_width={buffer_width} must be positive"
    if read_start + buffer_width > cache_capacity:
        return (
            f"window [{read_start}, {read_start + buffer_width}) runs past the "
            f"cache (cache_capacity={cache_capacity})"
        )
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return (
            f"num_heads={num_heads} is not a whole multiple of "
            f"num_kv_heads={num_kv_heads}"
        )
    if key_shape != value_shape:
        return f"key.shape={key_shape} does not match value.shape={value_shape}"
    return None


def _required_width(
    seqlen_q: int,
    window_size: int,
    q_block: int,
    q_kv_offset: int,
    buffer_origin: int = 0,
) -> int:
    """Widest span any block must cover, rounded up to a stick.

    Rows within a block have *staggered* windows, so a block spans
    ``W + q_len - 1`` columns rather than ``W`` — hence ``W + q_block`` for
    prefill and exactly ``W`` for decode, where one row has no stagger.

    No ``seqlen_kv``: ``last_coord`` is already bounded by the cache. A
    bidirectional window would reach ``last_coord + W - 1`` and need clamping.

    ``buffer_origin`` converts to the same buffer-relative space ``read_start``
    floors in, so the width is measured from where the floor actually lands.
    """
    widest = 0
    for qi in range(-(-seqlen_q // q_block)):
        q_start = qi * q_block
        q_end = min(seqlen_q, q_start + q_block)
        first_coord = q_kv_offset + q_start
        last_coord = q_kv_offset + q_end - 1
        window_start_logical = max(0, first_coord - window_size + 1)
        physical_start = _floor_stick(window_start_logical - buffer_origin)
        physical_end = (last_coord - buffer_origin) + 1
        widest = max(widest, physical_end - physical_start)
    return _ceil_stick(widest)


def query_blocking(seqlen_q: int) -> tuple[int, int]:
    """Block size and padded query length — one decision, so returned together.

    Decode takes a single-row block and no padding: padding one row to a full
    block would be 64x the work for one row of output. The caller pads at the
    FRONT (see spyre_sliding_window_attention).
    """
    if seqlen_q == 1:
        return 1, 1
    return STICK, _ceil_stick(seqlen_q)


def rejection_reason(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    is_causal: bool,
    q_block: int,
    cache_capacity: int | None = None,
) -> str | None:
    """Why this shape cannot be planned, or None if it can.

    Single source of truth: ``plan_sliding_window`` decides with it and the op
    raises with it, so the message names what actually failed.

    ``cache_capacity`` defaults to ``seqlen_kv``. When given it enables two
    further checks: the capacity must fit at least one buffer, and it must
    reach far enough back for the *earliest* block of a multi-block plan --
    which a capacity clearing the first check can still fail.

    Only the physical allocation needs stick alignment. ``seqlen_kv`` is a
    token count, not a memory offset (see ``read_start`` on flooring in
    physical coordinates), and an arbitrary position is handled by the causal
    band rather than by refusing the shape.
    """
    if not is_causal:
        return "bidirectional windows are not implemented, only causal ones"
    if seqlen_q <= 0 or seqlen_kv <= 0 or q_block <= 0:
        return (
            f"degenerate lengths (seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}, "
            f"q_block={q_block})"
        )
    if seqlen_q % q_block != 0:
        # Unreachable from the op, which pads first; reachable by a caller
        # choosing its own q_block. The body asserts equal blocks.
        return f"seqlen_q={seqlen_q} is not a whole number of {q_block}-row blocks"
    if window_size <= 0:
        return f"window_size={window_size} must be positive"
    if seqlen_kv - seqlen_q < 0:
        return f"seqlen_q={seqlen_q} exceeds seqlen_kv={seqlen_kv}"

    effective_capacity = seqlen_kv if cache_capacity is None else cache_capacity
    if effective_capacity % STICK != 0:
        return (
            f"cache_capacity={effective_capacity} must be a multiple of "
            f"{STICK}; pad the cache allocation to a stick boundary"
        )

    if cache_capacity is not None:
        q_kv_offset = seqlen_kv - seqlen_q
        buffer_origin = max(0, seqlen_kv - cache_capacity)
        buffer_width = _required_width(
            seqlen_q, window_size, q_block, q_kv_offset, buffer_origin
        )
        if cache_capacity < buffer_width:
            return (
                f"cache_capacity={cache_capacity} is narrower than the "
                f"{buffer_width}-row buffer this window needs"
            )
        # buffer_origin is fixed by the cache's total span while later blocks'
        # windows start later, so the EARLIEST block is the one that can reach
        # further back than the buffer holds -- and the unfloored window start
        # is monotonic in qi, so checking qi=0 is sufficient. Compared
        # unfloored to match read_start's physical-space floor: a floored
        # comparison could pass while the floor still lands negative.
        earliest_window_start = max(0, q_kv_offset - window_size + 1)
        if earliest_window_start < buffer_origin:
            return (
                f"cache_capacity={cache_capacity} does not reach far enough back "
                f"for this {seqlen_q}-row query: the earliest block needs logical "
                f"column {earliest_window_start}, but the buffer's oldest "
                f"resident row is {buffer_origin}"
            )

    return None


def plan_sliding_window(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    is_causal: bool = True,
    q_block: int = STICK,
    cache_capacity: int | None = None,
) -> SlidingWindowPlan | None:
    """The placement, or None for a shape ``rejection_reason`` declines.

    ``cache_capacity`` is the rows the cache physically allocates, as distinct
    from ``seqlen_kv``, its logical position. Defaults to ``seqlen_kv``.
    """
    if rejection_reason(
        seqlen_q, seqlen_kv, window_size, is_causal, q_block, cache_capacity
    ):
        return None
    if cache_capacity is None:
        cache_capacity = seqlen_kv

    q_kv_offset = seqlen_kv - seqlen_q
    buffer_origin = max(0, seqlen_kv - cache_capacity)
    buffer_width = _required_width(
        seqlen_q, window_size, q_block, q_kv_offset, buffer_origin
    )

    return SlidingWindowPlan(
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        window_size=window_size,
        q_block=q_block,
        num_q_blocks=-(-seqlen_q // q_block),
        buffer_width=buffer_width,
        q_kv_offset=q_kv_offset,
        is_causal=is_causal,
        cache_capacity=cache_capacity,
    )
