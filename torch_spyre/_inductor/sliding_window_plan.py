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
    """

    seqlen_q: int
    seqlen_kv: int
    window_size: int
    q_block: int
    num_q_blocks: int
    buffer_width: int
    q_kv_offset: int
    is_causal: bool

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

    def block_is_fully_attended(self, qi: int) -> bool:
        """True when block ``qi``'s band masks nothing, so the add can be skipped.

        Decode qualifies only for a stick-aligned window: otherwise read_start
        floors the origin below the row's window start (W=100, kv=4096: buffer
        from 3968, row reaches 3996) and the band masks the difference.
        """
        start = self.read_start(qi)
        stop = start + self.buffer_width
        q_start, q_end = self.block_q_range(qi)
        return all(
            self.row_window(q_index)[0] <= start and self.row_window(q_index)[1] >= stop
            for q_index in range(q_start, q_end)
        )

    def read_start(self, qi: int) -> int:
        """Cache offset block ``qi`` reads from.

        The clamps are the point: the ragged first and last blocks *shift*
        rather than shrink, so every block's buffer is one shape and one
        allocation serves them all. The band removes what the shift drags in.
        """
        q_start, _ = self.block_q_range(qi)
        first_coord = self.q_kv_offset + q_start
        window_origin = max(0, _floor_stick(first_coord - self.window_size + 1))
        return min(window_origin, self.seqlen_kv - self.buffer_width)


def check_window_read(
    read_start: int,
    buffer_width: int,
    seqlen_kv: int,
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
    """
    if read_start < 0:
        return f"read_start={read_start} is negative"
    if buffer_width <= 0:
        return f"buffer_width={buffer_width} must be positive"
    if read_start + buffer_width > seqlen_kv:
        return (
            f"window [{read_start}, {read_start + buffer_width}) runs past the "
            f"cache (seqlen_kv={seqlen_kv})"
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
) -> str | None:
    """Why this shape cannot be planned, or None if it can.

    Single source of truth: ``plan_sliding_window`` decides with it and the op
    raises with it, so the message names what actually failed.
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
        # The last row attends up to seqlen_kv-1, so the last block's read must
        # satisfy start + buffer_width == seqlen_kv exactly. Both are stick
        # multiples, so that has no solution unless seqlen_kv is one. Reading
        # past the logical end into padding lanes is the only escape, and
        # compiled attention is independently wrong there.
        return (
            f"seqlen_kv={seqlen_kv} must be a multiple of {STICK}; pad the KV "
            "cache to a stick boundary"
        )
    if seqlen_kv - seqlen_q < 0:
        return f"seqlen_q={seqlen_q} exceeds seqlen_kv={seqlen_kv}"

    return None


def plan_sliding_window(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    is_causal: bool = True,
    q_block: int = STICK,
) -> SlidingWindowPlan | None:
    """The placement, or None for a shape ``rejection_reason`` declines."""
    if rejection_reason(seqlen_q, seqlen_kv, window_size, is_causal, q_block):
        return None

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
    )
