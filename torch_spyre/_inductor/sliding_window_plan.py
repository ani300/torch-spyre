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
    """Coordinates are absolute cache positions: row ``i`` sits at
    ``q_kv_offset + i``, matching hf-adapters' convention. That is what makes
    prefill and decode share one formula.

    ``seqlen_kv`` and ``cache_capacity`` used to be the same field doing two
    jobs; they now say which job is which regardless of whether they agree.
    ``seqlen_kv`` is the cache's logical position -- what a coordinate means,
    so ``row_window`` clamps against it. ``cache_capacity`` is the rows
    physically allocated -- what a *read* may not run past, so ``read_start``
    clamps against it. ``check_cache_geometry`` allows either to be larger:
    ``seqlen_kv < cache_capacity`` is a cache still filling its allocation
    (left-aligned, ``buffer_origin`` 0); ``seqlen_kv > cache_capacity`` is a
    compact buffer that has filled and is now sliding forward (rolled,
    ``buffer_origin > 0``); equal is the full-length, exactly-full cache
    every caller used before either parameter existed.
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

        0 whenever ``seqlen_kv <= cache_capacity`` -- the cache fits inside
        its own allocation and physical and logical coordinates coincide.
        Once ``seqlen_kv`` exceeds
        ``cache_capacity`` -- a rolled buffer that has filled and is sliding
        forward -- this is where its oldest resident row sits, on the
        assumption the buffer holds exactly the most recent
        ``cache_capacity`` logical positions (see ``check_cache_geometry``
        and the "row order" precondition on ``kv_window``).
        """
        return max(0, self.seqlen_kv - self.cache_capacity)

    def block_is_fully_attended(self, qi: int) -> bool:
        """True when block ``qi``'s band masks nothing, so the add can be skipped.

        Decode qualifies only for a stick-aligned window: otherwise read_start
        floors the origin below the row's window start (W=100, kv=4096: buffer
        from 3968, row reaches 3996) and the band masks the difference.

        Compares against ``read_start_logical``, not ``read_start``: this is
        weighing the buffer's physical extent against ``row_window``, which
        speaks logical coordinates, so both sides of the comparison need to
        agree on which space they're in.
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

        ``window_origin`` is computed in logical coordinates (it comes from
        ``q_kv_offset``, an absolute cache position); subtracting
        ``buffer_origin`` converts it to a buffer-relative index before the
        clamp, which is itself expressed in capacity terms. Skipping that
        conversion silently returns the same answer for every block once
        ``seqlen_kv`` and ``cache_capacity`` diverge, because a logical
        coordinate (thousands) clamped against a small physical bound just
        returns the bound -- this was wrong for two releases running before
        being caught by a non-degenerate test case.

        When ``cache_capacity == buffer_width`` (the buffer holds exactly one
        window), ``buffer_origin`` tracks ``window_origin`` exactly, so this
        collapses to ``min(0, 0) == 0`` for every block: there is nowhere to
        shift to, correctly, because the buffer only ever holds "now".
        """
        q_start, _ = self.block_q_range(qi)
        first_coord = self.q_kv_offset + q_start
        window_origin = max(0, _floor_stick(first_coord - self.window_size + 1))
        physical_origin = window_origin - self.buffer_origin
        return min(physical_origin, self.cache_capacity - self.buffer_width)

    def read_start_logical(self, qi: int) -> int:
        """``read_start``, converted back to a logical coordinate.

        ``window_band_mask`` needs this, not ``read_start``: its row side
        (``q_row_origin``) is logical, so the column side must be too for
        ``delta = row - column`` to mean anything. A no-op whenever
        ``buffer_origin`` is 0 -- true for a still-filling or exactly-full
        cache, false for a rolled buffer that has outgrown its allocation.
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
    cache_seqlen: int | None = None,
) -> str | None:
    """Why this read is invalid, or None.

    kv_window takes its placement as plain ints, so a caller bypassing the plan
    can walk off the cache. Whether a *shape* is windowable is
    ``rejection_reason``'s question. Strings not exceptions, to keep this
    module free of torch and the backend's error classes.

    ``cache_capacity`` is what both call sites have always passed here --
    ``key.size(2)``, the rows physically allocated -- so the bound below was
    already a capacity bound in practice; this only makes the parameter say
    so.

    ``cache_seqlen``, when given, is the logical position -- what
    ``check_cache_geometry`` validated at the op boundary -- and lets this
    function tell "runs past the allocation" (the ``cache_capacity`` bound
    above) apart from "runs into the allocation's unfilled tail" (below): a
    still-filling cache (``cache_seqlen < cache_capacity``) leaves
    ``[cache_seqlen, cache_capacity)`` unwritten, and a caller building its
    own ``read_start``/``buffer_width`` rather than going through the plan
    could otherwise read into it undetected -- the plan itself never does,
    by construction (see ``check_cache_geometry``'s docstring). Optional and
    defaulting to None so a caller with no logical position to give --
    exactly the "equal" ordering, before ``cache_seqlen`` existed -- sees no
    new rejection.
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
    if cache_seqlen is not None and read_start + buffer_width > cache_seqlen:
        # Unreachable once cache_seqlen >= cache_capacity: the bound above
        # already caught read_start + buffer_width > cache_capacity <=
        # cache_seqlen. Only fires for the still-filling case.
        return (
            f"window [{read_start}, {read_start + buffer_width}) runs into "
            f"the cache's unwritten tail (cache_seqlen={cache_seqlen} of "
            f"cache_capacity={cache_capacity} rows written)"
        )
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return (
            f"num_heads={num_heads} is not a whole multiple of "
            f"num_kv_heads={num_kv_heads}"
        )
    if key_shape != value_shape:
        return f"key.shape={key_shape} does not match value.shape={value_shape}"
    return None


def check_cache_geometry(cache_seqlen: int, cache_capacity: int) -> str | None:
    """Why this (logical length, physical capacity) pair cannot be read, or None.

    ``cache_seqlen`` is the true KV sequence position -- how many tokens the
    cache has seen, HF's ``cumulative_length`` and vLLM's ``seq_lens``.
    ``cache_capacity`` is ``key.size(2)``, the rows actually allocated.

    Reading only cares which side of ``cache_capacity`` the cache is on, not
    the write-side transition of crossing it -- HF's three states
    (still-filling / becoming-full / full) collapse to two here, since
    "becoming full" is a mid-update detail a reader never observes; it only
    ever sees the state an update left behind.

    ``cache_seqlen <= cache_capacity``: left-aligned, buffer row ``j`` holds
    logical position ``j`` for ``j < cache_seqlen``; ``[cache_seqlen,
    cache_capacity)`` is allocated but unwritten. No block's read ever
    reaches that tail: ``rejection_reason`` requires ``cache_seqlen``
    stick-aligned, and every block's read is built from stick-aligned
    arithmetic that (by induction on ``buffer_width`` being sized to the
    *widest* block, and per-block width being monotonic non-decreasing, so
    the last block sets it exactly) lands the last, furthest-reaching block's
    read exactly on ``cache_seqlen`` with no slack -- see
    ``TestStillFilling.test_no_block_ever_reads_past_cache_seqlen`` for the
    swept proof. No masking beyond the existing causal band is needed.

    ``cache_seqlen > cache_capacity``: a compact buffer that has filled and
    is now sliding forward, on the assumption (not checked here -- see
    ``kv_window``'s docstring) that its rows stay contiguous and
    time-ordered, oldest dropped from the front, so the buffer holds exactly
    the most recent ``cache_capacity`` positions.
    """
    if cache_seqlen <= 0:
        return f"cache_seqlen={cache_seqlen} must be positive"
    if cache_capacity <= 0:
        return f"cache_capacity={cache_capacity} must be positive"
    return None


def _required_width(
    seqlen_q: int, window_size: int, q_block: int, q_kv_offset: int
) -> int:
    """Widest span any block must cover, rounded up to a stick.

    Rows within a block have *staggered* windows, so a block spans
    ``W + q_len - 1`` columns rather than ``W`` — hence ``W + q_block`` for
    prefill and exactly ``W`` for decode, where one row has no stagger.

    No ``seqlen_kv``: ``last_coord`` is already bounded by the cache. A
    bidirectional window would reach ``last_coord + W - 1`` and need clamping.
    """
    widest = 0
    for qi in range(-(-seqlen_q // q_block)):
        q_start = qi * q_block
        q_end = min(seqlen_q, q_start + q_block)
        first_coord = q_kv_offset + q_start
        last_coord = q_kv_offset + q_end - 1
        window_origin = max(0, _floor_stick(first_coord - window_size + 1))
        widest = max(widest, last_coord - window_origin + 1)
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

    ``cache_capacity`` is optional and, when omitted, changes nothing -- every
    caller before this parameter existed still sees exactly the checks it saw
    before. When given, it enables three more: ``read_start``'s buffer-relative
    arithmetic needs ``cache_capacity`` stick-aligned the same way it always
    needed ``seqlen_kv`` aligned; needs at least one buffer's worth of room to
    place a read in; and needs enough room for the *first* block specifically,
    since a capacity that clears the previous check can still be too narrow
    for a multi-block plan whose earliest block reaches further back than a
    smaller-but-nonzero amount of slack covers.
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
    if seqlen_kv % STICK != 0:
        # seqlen_kv also fixes buffer_origin's alignment (seqlen_kv -
        # cache_capacity, both stick multiples once cache_capacity is
        # checked below), which read_start's physical conversion depends on.
        # For the default cache_capacity == seqlen_kv case this is also the
        # original reason: the last block's read must land exactly on
        # cache_capacity, itself a stick multiple, so seqlen_kv must be one.
        return (
            f"seqlen_kv={seqlen_kv} must be a multiple of {STICK}; pad the KV "
            "cache to a stick boundary"
        )
    if seqlen_kv - seqlen_q < 0:
        return f"seqlen_q={seqlen_q} exceeds seqlen_kv={seqlen_kv}"
    if cache_capacity is not None:
        if cache_capacity % STICK != 0:
            return (
                f"cache_capacity={cache_capacity} must be a multiple of {STICK}; "
                "pad the cache allocation to a stick boundary"
            )
        q_kv_offset = seqlen_kv - seqlen_q
        buffer_width = _required_width(seqlen_q, window_size, q_block, q_kv_offset)
        if cache_capacity < buffer_width:
            return (
                f"cache_capacity={cache_capacity} is narrower than the "
                f"{buffer_width}-row buffer this window needs"
            )
        # The buffer_width check above only rules out a capacity too small for
        # any single block. A multi-block plan's EARLIEST block can still
        # reach further back than the buffer's oldest resident row, because
        # buffer_origin is fixed by the cache's total span while later
        # blocks' windows start later. window_origin is monotonic in qi
        # (first_coord grows with q_start, floor_stick preserves order), so
        # the earliest block (qi=0) is the only one that can fail this and
        # checking it is sufficient. Uncaught, this surfaces later as a
        # negative read_start -- not silent, but from a less specific error
        # site than here.
        buffer_origin = max(0, seqlen_kv - cache_capacity)
        earliest_window_origin = max(0, _floor_stick(q_kv_offset - window_size + 1))
        if earliest_window_origin < buffer_origin:
            return (
                f"cache_capacity={cache_capacity} does not reach far enough back "
                f"for this {seqlen_q}-row query: the earliest block needs logical "
                f"column {earliest_window_origin}, but the buffer's oldest "
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

    ``cache_capacity`` is the rows the cache physically allocates, as
    distinct from ``seqlen_kv``, its logical position. Defaults to
    ``seqlen_kv`` -- a full-length cache that is exactly full, the only
    geometry callers passed before this parameter existed -- so every
    existing caller is unaffected.
    """
    if rejection_reason(
        seqlen_q, seqlen_kv, window_size, is_causal, q_block, cache_capacity
    ):
        return None
    if cache_capacity is None:
        cache_capacity = seqlen_kv

    q_kv_offset = seqlen_kv - seqlen_q
    buffer_width = _required_width(seqlen_q, window_size, q_block, q_kv_offset)

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
