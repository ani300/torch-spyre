# SDPA head-tile blocker — team readout

**Branch:** flash-attn-kv-loop (PR #3672 family) · **Status:** blocked, decision
needed · **Refs:** #3674, #3432 · **Compiled:** 2026-08-12

---

## The short version

Head-dim tiling in the flash-attention SDPA decomp is blocked on a
work-division (WSR) naming limitation, **not** on the SDPA decomp itself.

The head tile was never *disabled* — it was never turned *on*. Activating it
(rename the head-axis seed `"_h"` → `"num_heads"` at 9 sites) is textually
correct and compiles cleanly, but the 4-case prefill suite fails numerically:

- **Head tile 0 is correct.**
- **Every later head tile is ~85% wrong** (26.8% aggregate mismatch, up to 1.24
  absolute) — whenever there are ≥2 KV blocks, which all four required cases
  have by construction.

Root cause: the only in-graph naming primitive, `spyre_hint(named_dims=…)`,
names an op's **output** only. It never gives the upstream K/V read a
buffer-level head-dim binding, so the tiler cannot compute each head tile's
offset into K/V. **There is no in-graph primitive that can.** PR #3674 is
numerically exact precisely because its K/V chunks are *graph inputs* — the one
place buffer-level naming exists today.

---

## What we tried — the rename is correct, the result is wrong

The KV loop already sits inside a `spyre_hint(tiles={"num_heads": num_heads //
4})` scope. But every in-graph seed named the head axis `"_h"` — a placeholder
that deliberately doesn't match the scope key, so `assign_dim_hints` dropped the
tile as a silent no-op. Renaming the 9 seed sites to `"num_heads"` is the whole
activation, and it lit the tile up as intended.

On device, the four prefill cases (`mha/gqa_prefill` + 8k variants) then failed
`compare_with_cpu`. The boundary is clean:

| KV blocks | Result |
|---|---|
| `num_kv_blocks == 1` | **All heads match CPU** (~1e-2). The Python KV loop runs once; no online-softmax carry crosses an iteration. Tile is fine in isolation. |
| `num_kv_blocks ≥ 2` | Head tile 0 correct; every later tile wrong by up to ~1.24 abs. Reproducible; **not** the transient `0x7b1b` RAS fault. |

The tell is *which* heads are wrong: tile 0 (offset 0) is right, all later tiles
are off. That is a per-tile **offset** failure, not arithmetic — the tiler reads
the wrong head slice of K/V for tiles 1…N.

---

## Root cause — in-graph naming is output-only

Our decomp slices K/V *inside* the graph (`k_blk = key[…, start:end, :]`) and
names the slice's *consumer* with `spyre_hint(named_dims=…)`. Tracing the
propagation in `torch_spyre/_inductor/wsr/propagate_named_dims.py`:

- **Input path (lines 406–408):** graph inputs get `_dim_prop_info` attached to
  the *input TensorBox itself* — buffer-level, so **every read** of that buffer
  inherits the head name.
- **Hint path (lines 416–456):** an in-graph `spyre_hint` op is seeded from its
  *own output* layout, sets `_dim_prop_info` on the op, then `continue`s.
- **The gap:** that `continue` skips `_compute_named_dims` (lines 473–474) — the
  only code that looks *backward* at an op's inputs. So the read of the
  underlying full `key` buffer never receives a head mapping.
- **Silently:** because the backward path never runs, the `_untracked` warning
  that would have flagged the missing mapping never fires. Invisible until the
  numbers come out wrong.

**Why Lq tiling works but head tiling doesn't:** the K/V block slice varies
along the *sequence* axis — each block still contains *all* heads. Query-length
tiling splits the query (a properly named buffer) and never needs an offset into
K/V. Head tiling splits the head axis, which lives on the un-named K/V read — so
tiles past 0 read the wrong heads.

**The structural wall:** there is no primitive today to name an *intermediate*
buffer's dims wholesale — only graph inputs get that. And naming the full `key`
instead doesn't survive the slice: `_consume_names` matches names to axes by
size product, and slicing shrinks the Lk axis so the binding is dropped.

---

## Why we can't just copy #3674

PR #3674 gets head tiling numerically **exact** (0.00% vs eager-CPU, H:4 + Lq:2
at 8k) by slicing K/V *outside* the traced graph and passing each dense chunk in
as a *named graph input*. That works because inputs are exactly the buffers that
receive whole-buffer naming.

`spyre__sdpa_overrideable` is a *decomposition*: its signature is fixed upstream
to `(query, key, value, …)` full tensors, and its entire body is traced into one
graph. Its K/V slices are intermediate buffers — never inputs. So #3674's exact
mechanism is unavailable to us as-is; the correctness came from the graph
boundary, and a decomp has no such boundary to hoist the chunking to.

**Also inherited from #3674:** layout solving fails past ~6 unrolled KV chunks
(forcing `kv_block = 2048`, ~4 chunks, at 8k), and a unit-size head tile
(`h_tiles == H`) crashes in read-copy insertion. Our `num_heads // 4` gives tile
*groups* of ≥2 heads, so the unit-tile crash doesn't bite — but the chunk
ceiling caps us at ~4 KV blocks.

---

## The decision — four ways forward

The rename is a no-op without a way to give the K/V read a real head binding.
The first is the general fix; the rest trade generality for smaller blast radius.

### A. New WSR naming primitive — *recommended (large, reusable)*

Add a production mechanism to seed an intermediate `ComputedBuffer`'s
`_dim_prop_info` at buffer level — mirroring the graph-input loop — so a dense
in-graph K/V clone carries head names that every read inherits. Touches the WSR
pass and needs a targeted unit test, but it's the general fix: it also unblocks
batch tiling and any future decomp with this shape.

### B. Fold heads into the batch axis — *medium, localized*

Reshape `[B, H, …] → [B·H, …]` so head tiling becomes tiling of a combined
leading axis, sidestepping the K/V head binding entirely. Stays inside
`decompositions.py`, but must verify the combined axis gets a real binding (not
the same placeholder silent-fail) and that GQA expansion still lines up.

### C. HOP / nested-compile boundary — *large, structural risk*

Restructure SDPA so a Python wrapper pre-chunks K/V and the chunks enter an inner
compiled region as inputs — faithfully reproducing #3674's caller pattern. Most
faithful port, but requires a higher-order-op or nested-`compile` structure
around a fixed-signature decomp. Highest structural uncertainty.

### D. Spike first, then commit — *small, de-risks*

Prove which approach actually compiles and passes on-device against the
1-block-vs-2-block repro before writing the plan. Slower to a plan, lowest risk
of designing the wrong thing. The repro is a ~1s-compile harness, so a spike is
cheap.

---

## Where things stand right now

- The `"_h"` → `"num_heads"` rename is applied in the working tree,
  **uncommitted**. Reverting it restores the known-good baseline (`HEAD
  14218a4`) exactly.
- Nothing has been committed or pushed for head tiling. The prior KV-block work
  on this branch is unaffected.
- The blocker is a plan-scope collision, not a coding mistake: the approved plan
  was `decompositions.py`-rename-only, and the real fix is in the WSR layer.

**Open question for the team:** do we invest in the general WSR primitive (A),
or take the localized batch-fold bet (B)?
