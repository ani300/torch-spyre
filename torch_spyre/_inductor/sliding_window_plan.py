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
    ``seqlen_kv`` is the cache's logical position -- HF's
    ``cumulative_length``, vLLM's ``seq_lens``, what a coordinate *means* --
    so ``row_window`` clamps against it. ``cache_capacity`` is
    ``key.size(2)``, the rows physically allocated -- what a *read* may not
    run past -- so ``read_start`` clamps against it.

    Either may be larger. Reading only cares which side of
    ``cache_capacity`` the cache is on, not the write-side transition of
    crossing it: HF's three states (still-filling / becoming-full / full)
    collapse to two here, since "becoming full" is a mid-update detail a
    reader never observes -- it only ever sees the state an update left
    behind.

    ``seqlen_kv <= cache_capacity`` -- still filling its allocation.
    Left-aligned, ``buffer_origin`` 0: buffer row ``j`` holds logical
    position ``j`` for ``j < seqlen_kv``, and ``[seqlen_kv,
    cache_capacity)`` is allocated but unwritten. ``seqlen_kv`` is not
    required to be stick-aligned (only ``cache_capacity`` is, in
    ``rejection_reason``), so a block's physical read routinely lands past
    ``seqlen_kv``. Safe, and the reason no tail mask is needed: every query
    row's own coordinate is ``< seqlen_kv`` by construction
    (``q_kv_offset = seqlen_kv - seqlen_q``), so causal masking's
    ``column <= row`` already forces ``column < seqlen_kv`` -- no column
    past the written prefix can ever be causally attended. The precondition
    that does remain: the unwritten tail must hold *finite* values
    (zero-filled; see ``spyre::sliding_window_attention``'s docstring),
    since an additive ``-inf`` cannot rescue a ``NaN`` even when the causal
    term correctly assigns that ``-inf``.

    ``seqlen_kv > cache_capacity`` -- a compact buffer that has filled and
    is now sliding forward (rolled, ``buffer_origin > 0``), on the
    assumption (unchecked -- see ``kv_window``'s docstring) that its rows
    stay contiguous and time-ordered, oldest dropped from the front, so it
    holds exactly the most recent ``cache_capacity`` positions.

    Equal is the full-length, exactly-full cache every caller used before
    either parameter existed.
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
        Once ``seqlen_kv`` exceeds ``cache_capacity`` -- a rolled buffer
        that has filled and is sliding forward -- this is where its oldest
        resident row sits, on the assumption the buffer holds exactly the
        most recent ``cache_capacity`` logical positions (see this class's
        docstring and the "row order" precondition on ``kv_window``).
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

        Floors in PHYSICAL coordinates: convert the logical window start to
        a buffer-relative index first (subtract ``buffer_origin``), *then*
        round down to a stick. Flooring in logical coordinates first and
        subtracting after -- the previous order -- only lands on a
        stick-aligned physical offset when ``buffer_origin`` is itself a
        stick multiple, which held for every geometry ``rejection_reason``
        used to accept (both sides individually stick-aligned) but not for
        an arbitrary rolled-buffer ``seqlen_kv``: e.g. ``buffer_origin=55``
        would floor to physical offset ``9`` -- not a stick boundary, and
        wrong for a HW read regardless. Flooring after conversion is
        correct for both cases and identical to the old formula whenever
        ``buffer_origin`` is a stick multiple (floor distributes over
        subtracting an exact multiple of the modulus): every previously
        accepted shape reads from exactly the same offset as before.

        When ``cache_capacity == buffer_width`` (the buffer holds exactly one
        window), ``buffer_origin`` tracks the window's logical start exactly,
        so the unfloored physical offset is already 0 (or floors to 0) for
        every block: there is nowhere to shift to, correctly, because the
        buffer only ever holds "now".
        """
        q_start, _ = self.block_q_range(qi)
        first_coord = self.q_kv_offset + q_start
        window_start_logical = max(0, first_coord - self.window_size + 1)
        physical_origin = _floor_stick(window_start_logical - self.buffer_origin)
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

    Bounds the read against the *allocation*, deliberately not against the
    cache's logical position. A still-filling cache's buffer routinely
    extends past its written prefix -- ``seqlen_kv`` need not be
    stick-aligned, so ``read_start + buffer_width`` overshooting it is the
    normal case, not an error -- and those columns are excluded by causal
    masking anyway (see ``SlidingWindowPlan``'s class docstring). A bound
    on the logical position here would reject most valid plans.
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

    ``buffer_origin`` converts to the same buffer-relative space
    ``read_start`` floors in (see its docstring) -- the width must be
    measured from where the floor actually lands, physically, not from an
    unconverted logical floor. Defaults to 0, matching every caller before a
    rolled buffer existed, where logical and physical coincide.
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

    ``cache_capacity`` is optional and, when omitted, changes nothing -- every
    caller before this parameter existed still sees exactly the checks it saw
    before (``effective_capacity`` below is ``seqlen_kv`` in that case, so the
    alignment check is the same one that branch always saw). When given, it
    enables two more: needs at least one buffer's worth of room to place a
    read in; and needs enough room for the *first* block specifically, since a
    capacity that clears the previous check can still be too narrow for a
    multi-block plan whose earliest block reaches further back than a
    smaller-but-nonzero amount of slack covers.

    ``seqlen_kv`` itself is NOT required to be stick-aligned: it is the
    cache's logical position, a token count that increments by 1 (HF's
    ``cumulative_length``), not a memory offset. Only the physical
    allocation -- ``effective_capacity`` -- needs alignment, since that is
    what fixes ``read_start``'s stick boundary (see ``read_start``'s
    docstring on flooring in physical, not logical, coordinates). A
    still-filling or rolled cache at an arbitrary ``seqlen_kv`` is handled
    by the causal band, which already excludes every column past the written
    prefix (see this module's ``SlidingWindowPlan`` docstring), not by
    refusing anything that doesn't land on a stick.
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
        # The buffer_width check above only rules out a capacity too small for
        # any single block. A multi-block plan's EARLIEST block can still
        # reach further back than the buffer's oldest resident row, because
        # buffer_origin is fixed by the cache's total span while later
        # blocks' windows start later. The unfloored window start is
        # monotonic in qi (first_coord grows with q_start), so the earliest
        # block (qi=0) is the only one that can fail this and checking it is
        # sufficient. Uncaught, this surfaces later as a negative read_start
        # -- not silent, but from a less specific error site than here.
        # Compared unfloored, matching read_start's physical-space floor: a
        # floored comparison here could pass while the floor still lands
        # negative (flooring only ever moves a value down).
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
