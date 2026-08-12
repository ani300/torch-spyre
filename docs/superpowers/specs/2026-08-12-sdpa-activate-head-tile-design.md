# SDPA `num_heads` tile activation — design

**Date:** 2026-08-12
**Branch:** `flash-attn-kv-loop` (PR #3672 family)
**File touched:** `torch_spyre/_inductor/decompositions.py`
(`spyre__sdpa_overrideable`)
**Predecessor handoff:** `docs/superpowers/plans/2026-08-11-sdpa-lq-tile-next-steps.md`
(Avenue #3)

## Problem

The flash-attention SDPA decomp already wraps its KV loop in a
`spyre_hint(tiles={"num_heads": max(1, num_heads // 4)})` scope
(`decompositions.py:506`), but every in-graph `spyre_hint(named_dims=…)`
seed names the head axis `"_h"` — a placeholder chosen precisely because it
does **not** match `"num_heads"`. `assign_dim_hints` drops any tile whose
named dim never propagated to a real buffer, so the head tile is a silent
no-op today: only the `max_seqlen_q` axis actually tiles.

The head tile was never disabled elsewhere in the pipeline — it was never
turned *on*. The placeholder seed name is the off-switch. The handoff plan's
Avenue #3 records the on-switch: "Renaming `_h`→`num_heads` in the seed lists
should light up those tiles."

Goal: activate the `num_heads` tile so SDPA tiles over the head axis in
addition to the query-length axis, and prove it stays numerically correct
on-device.

## Scope

**In scope**

- Rename the head-axis seed `"_h"` → `"num_heads"` at every code seed site in
  `spyre__sdpa_overrideable` (9 sites — see Design below).
- Keep the `num_heads // 4` tile size unchanged.
- Update the two explanatory comments that describe `"_h"` as an inactive
  placeholder so they match the new reality.
- Verify the existing H=8 prefill suite stays green (numeric proof).

**Explicitly out of scope (deferred, unchanged)**

- The `batch_size` tile: `"_b"` stays a placeholder. Head only.
- Causal / additive-mask prefill (`coarse_tile.py:144 hint_id appears in more
  than one group`) — handoff Avenue #1.
- `kv_tail` partial last block — handoff Avenue #2.
- Removing the `if True:` scaffolding — handoff Avenue #4.
- A dedicated distinct-head-count test — the H=8 suite already engages the
  tile (see Testing).

## Design

### The rename

The head axis is seeded `"_h"` at these 9 code sites in
`spyre__sdpa_overrideable`. Each is the second entry in a `named_dims=[…]`
list (physical stride order `[batch, head, seq, dim]`):

| Line | Seed context |
|---|---|
| 453 | M-seed accumulator (`[_b, _h, max_seqlen_q, head_dim]`) |
| 463 | M reduce (`[_b, _h, max_seqlen_q]`) |
| 472 | L-seed / running-sum accumulator (`[_b, _h, max_seqlen_q]`) |
| 502 | `q_scaled` producer (`[_b, _h, max_seqlen_q, head_dim]`) |
| 528 | per-block `v_blk.contiguous()` (`[_b, _h, blk_len, head_dim]`) |
| 542 | per-block `keys_T.contiguous()` (`[_b, _h, blk_len, head_dim]`) |
| 559 | scores after QK^T (`[_b, _h, max_seqlen_q, blk_len]`) |
| 601 | attn·V output (`[_b, _h, max_seqlen_q, head_dim]`) |
| 619 | final finalize-block seed (`[_b, _h, max_seqlen_q, head_dim]`) |

Change only the `"_h"` entry to `"num_heads"` at each. Leave `"_b"` alone.
The head axis is not a stick-aligned dimension (heads are not 64-element
chunks), so `num_heads // 4` is a plain loop-split count, not a byte-aligned
block — no stick rounding applies.

The exact seed-site line numbers must be re-confirmed at implementation time
against the current file (the KV-block work shifted lines); the anchor is the
`"_h"` string inside a `named_dims` list, not the line number.

### Comment updates

Two comments describe `"_h"` as a deliberate placeholder:

- `decompositions.py:497-501` — "The batch and head dims are deliberately
  named `_b`/`_h` … so only the max_seqlen_q tile activates for now. Renaming
  these … is all it takes to light up those tiles in a follow-up." After this
  change, the head half is done: reword to say the head dim is now
  `"num_heads"` (tile active) and only `"_b"` (batch) remains a placeholder.
- `decompositions.py:423` — a general explanatory comment giving an example
  seed `["_b","_h","max_seqlen_q",…]`. Update the head entry to `"num_heads"`
  so the illustrative example matches the code the reader will see.

### Why this is the whole change

The tile *scope* (`spyre_hint(tiles={"num_heads": …})`) already exists and
already computes the split. The only thing gating activation is whether a
real buffer carries the `"num_heads"` named dim for `assign_dim_hints` to
bind the tile to. Renaming the seeds supplies exactly that. This mirrors how
the `max_seqlen_q` tile was activated (commits `b8e5707`→`9673a57`).

## Testing

No new test. The four passing SDPA prefill cases all have `H=8`:

- `mha_prefill`, `gqa_prefill` — H=8, existing.
- `mha_prefill_8k`, `gqa_prefill_8k` — H=8, added this branch.

With `num_heads // 4 = 2`, the head tile splits H=8 into 4 tiles of 2 heads
on every one of these — so they exercise the newly-activated tile directly,
not incidentally. Each is checked numerically against the CPU
`scaled_dot_product_attention` reference by the framework's
`compare_with_cpu`. If all four stay green, head tiling is proven correct at
the shapes we ship. If any regress, that is the head tile breaking numerics
and must be diagnosed (systematic-debugging), not masked into `expect_fail`.

Verification is on-device on this Spyre-capable machine (`import torch` gives
device access; the test harness imports in the correct order).

## Risks

- **Sequenced ahead of causal/mask/kv_tail.** The handoff plan ordered
  Avenue #3 *after* #1 and #2 "so failures are attributable," and those remain
  unfixed. Mitigation: head-only keeps one variable moving, and the deferred
  cases fail with a *known, distinct* signature (`coarse_tile.py:144 hint_id
  appears in more than one group`), so a new head-tile failure mode is
  distinguishable from the pre-existing deferred one.
- **A second active tile dim may surface new codegen paths** (plan's words).
  If the rename does not compile or numerics drift, this is a genuine
  debugging session, not a guaranteed one-liner — invoke
  `superpowers:systematic-debugging`, do not mask.
- **Clean fallback.** The placeholder `"_h"` state is the known-good baseline
  just shipped (HEAD `14218a4`). If activation cannot be made green, reverting
  the rename restores it exactly.

## Ground rules

- Every commit `git commit -s -S` (DCO + GPG); verify `echo test | gpg
  --clearsign` first. Never commit unsigned. Don't commit until the targeted
  tests are green.
- `import regex` not `import re`; line length 88; `pre-commit run --files`
  before committing.
- `main` must track `upstream/main` (missing `flex::`/ABI symbols otherwise) —
  see memory `torch-spyre-main-tracks-upstream`.

## Related memories

`torch-spyre-sdpa-lq-tile-accumulator` (named-dim seeding activates a tile;
copy_f in-place accumulators give wrong numerics — the functional-SSA
recurrence must be preserved), `torch-spyre-sdpa-causal-alignment`,
`torch-spyre-main-tracks-upstream`.
