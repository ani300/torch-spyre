# SDPA `max_seqlen_q` tile — next-steps handoff

**Branch:** `flash-attn-kv-loop` (PR #3672 family). **Head at handoff:** `a5c7f4e`
(pushed to `origin`).
**File:** `torch_spyre/_inductor/decompositions.py`, function
`spyre__sdpa_overrideable` (~lines 381-640).

## Where we are

The flash-attention SDPA decomp coarse-tiles the query-length axis
(`max_seqlen_q`) into 64-row tiles, driven by in-graph `spyre_hint(named_dims=…)`
seeds + a `spyre_hint(tiles={"max_seqlen_q": …})` scope. Two things are solved and
green:

1. **Lq tile fires and is numerically correct for plain prefill.** Functional-SSA
   online-softmax recurrence (no `copy_f`), final divide folded into the last KV
   block so it inherits the tile.
2. **Transposed-input layout, OOM-safe.** SDPA callers pass q/k/v as
   `transpose(1,2)` views. The seeds bind to physical stride order, so a
   transposed input mapped the tile onto the wrong axis. Fixed by
   `query.contiguous()` (bounded) + a **per-block** `v_blk.contiguous()`
   (`[B,H,64,D]`, cheap) instead of the OOM-prone whole-tensor
   `key/value.contiguous()`. See commit `a5c7f4e`.

**Passing:** `test_sdpa_mha_prefill`, `test_sdpa_gqa_prefill` (2 passed, 2 xfailed
= `mha_decode`/`gqa_decode`).

**Still failing** (all in `tests/inductor/test_inductor_ops.py::TestOps`):
`test_sdpa_mha_prefill_causal`, `test_sdpa_gqa_prefill_causal`,
`test_sdpa_mha_prefill_kv_tail`, `test_sdpa_mha_prefill_kv_tail_causal`,
`test_sdpa_mha_prefill_mask`.

## Key mechanics to carry forward

- **Seeds bind to physical STRIDE order, not logical dim order.** The pass
  (`wsr/propagate_named_dims.py:420-466`) zips `named_dims` positionally to
  `op_out_coords(op)`; `op_out_coords` (`pass_utils.py:363`) derives coords from
  the op's frozen physical **strides** via `host_coordinates`.
- **Pointwise ops PRESERVE input stride order.** `pick_loop_order` (stock
  `torch/_inductor/scheduler.py:3619`) orders output dims by the read's stride
  magnitude → a mul on a transposed input freezes size-logical / stride-transposed.
  Correct *names* are not enough; strides must actually be contiguous.
- **`.contiguous()` is reliable ONLY because torch-spyre overrides `aten.clone`**
  (`lowering.py:996-1024`) to `freeze_layout_with_stride_order`. Stock Inductor's
  clone ignores `memory_format`. Any future "just add .contiguous()" fix depends
  on this override.
- **Matmul outputs inherit no names** — each `torch.matmul` needs its own
  `named_dims` seed or it becomes an untiled restickify boundary.
- **Lq-tile numerics are unverified upstream too:** the `#3674`
  `test_flash_tile_Lq` runs with `correctness=False`. There is no known-good
  numeric reference for Lq output-dim tiling other than our now-passing prefill.

## Avenues for next session (roughly ordered by value/effort)

### 1. Causal + mask variants — `hint_id appears in more than one group`
**Tests:** `*_prefill_causal`, `*_prefill_mask`.
**Symptom:** `coarse_tile: hint_id=N appears in both group 0 and group 1`. The
`is_causal`/`attn_bias` path adds `scores = scores + causal_mask[..., start:end]`
(decompositions.py:557-561) — an extra op that splits the single batch/head/Lq
hint scope across two loop nests, breaking coarse_tile's one-hint-scope-per-loop-
group invariant.
**Where to look:** `torch_spyre/_inductor/wsr/coarse_tile.py` (the group-assignment
that raises), and how the mask-add op lands relative to the tiled scores. The mask
add is currently *inside* the tile scope but *outside* the seeded producers.
**Candidate approaches:**
   - Seed the mask-add output too (`named_dims=["_b","_h","max_seqlen_q","blk_len"]`)
     so it joins the same group instead of forming a second one.
   - Precompute the full additive mask once and slice per block (already partly
     done for causal at line 471) — check whether the *slice* op is the one
     landing in the second group.
   - Investigate whether coarse_tile can tolerate a hint_id spanning groups
     (backend change, larger).
**Note:** `causal_mask` is top-left aligned (matches `spyre::causal_mask` tril);
see memory `torch-spyre-sdpa-causal-alignment` — don't reintroduce a bottom-right
assumption.

### 2. `kv_tail` — partial last KV block (Skv=130)
**Tests:** `*_prefill_kv_tail`, `*_prefill_kv_tail_causal`.
**Symptom (from an earlier run):** `FileNotFoundError … sdsc_23.json` in
`codegen/bundle.py:453` — a codegen/bundle failure, likely downstream of a
malformed last-block plan when `max_seqlen_kv` is not a multiple of
`kv_block_size=64` (130 → blocks of 64,64,2). The `blk_len` for the tail block is
2, and `_consume_names: no prefix of ['blk_len'] multiplies to 2` warnings appear.
**Where to look:** the loop at decompositions.py:501-518 already computes
`end = min(start + kv_block_size, max_seqlen_kv)` so `k_blk`/`v_blk` shrink
correctly, but the per-block `blk_len` named-dim size is declared from the FIRST
block (64) and the tail block (2) mismatches. Check `_named_dims` declare-once
semantics (propagate_named_dims.py:454 `setdefault`) — `blk_len` is pinned to 64,
so the size-2 tail under-tiles or mis-strides.
**Candidate approaches:**
   - Pad the last KV block to `kv_block_size` (mask the padding to -inf in scores)
     so every block is size 64 and `blk_len` is uniform. Cleanest; matches how
     many flash kernels handle ragged KV.
   - Give the tail block a distinct named dim (`blk_len_tail`) so it declares its
     own size — messier, more scopes.
   - Investigate the `sdsc_*.json` bundle failure directly (may be a symptom, not
     the cause).

### 3. Activate the `batch_size` / `num_heads` tiles
Currently the batch/head seeds are placeholders `"_b"`/`"_h"` that deliberately do
NOT match the `tiles={"batch_size": …}` / `{"num_heads": …}` scopes
(decompositions.py:439, 488, 520, 550, 592, 608), so only `max_seqlen_q` tiles.
Renaming `_b`→`batch_size` and `_h`→`num_heads` in the seed lists should light up
those tiles. **Low effort, but verify numerics** — this multiplies the active tile
dims and may surface new codegen paths. Do it AFTER 1 & 2 so failures are
attributable.

### 4. Remove dead scaffolding (`if True:`)
decompositions.py:496 has an `if True:` where a `work_div` scope used to be
(removed because `work_division_hint: dim d2 size=1 not evenly divisible by
split=8`). It's a faithful-green placeholder. Either delete it (re-indent the loop)
or, if reintroducing work-division, guard the split against reduced/size-1 axes.
Cosmetic + a latent decision point.

### 5. Push a proper numeric reference for Lq tiling upstream
`#3674`'s `test_flash_tile_Lq` asserts only loop structure (`correctness=False`).
Our prefill tests are now the only numeric proof. Consider contributing a
`correctness=True` Lq case to `tests/inductor/test_coarse_tile_e2e.py` so the
tiled-matmul-output path has a guard independent of the full SDPA op.

## Reproduction / debugging aids

- Probe scripts in `/tmp/probe_*.py` (may not survive across sessions): notably
  `probe_seed_layout.py` (dumps each seeded op's size/stride/coords — the tool
  that cracked the transposed-layout bug), `probe_tile_fires.py` (assign_dim_hints
  dump), `probe_named_sizes.py` (`_named_dims` contents).
- `assign_dim_hints` INFO dump: set `update_log_level("assign_dim_hints","INFO")`
  + `torch._inductor.config.compile_threads = 1`. Look for
  `['max_seqlen_q'] range=128 split_count=2 -> 64 per tile loop_var=d2`.
- Run the passing baseline: `pytest "tests/inductor/test_inductor_ops.py::TestOps"
  -k "(sdpa_mha_prefill or sdpa_gqa_prefill) and not causal and not kv_tail and not
  mask" -v`.

## Related memories

`torch-spyre-sdpa-lq-tile-accumulator` (full history + the OOM-safe revision),
`torch-spyre-sdpa-causal-alignment`, `torch-spyre-sdpa-torch-full`,
`torch-spyre-main-tracks-upstream`.

## Ground rules (project)

- Every commit `git commit -s -S` (DCO + GPG). Verify `echo test | gpg
  --clearsign` first. Never commit unsigned. Don't commit until the targeted
  tests are green (unless an explicit WIP checkpoint is authorized).
- `import regex` not `import re`; line length 88; run `pre-commit run --files …`
  before pushing.
- `main` must track `upstream/main`, not the `ani300` fork (missing `flex::`
  symbols otherwise) — see `torch-spyre-main-tracks-upstream`.
