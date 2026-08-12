# SDPA `num_heads` Tile Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the `num_heads` tile in the flash-attention SDPA decomp so it tiles over the head axis in addition to the query-length axis, proven numerically correct on-device.

**Architecture:** Rename the head-axis named-dim seed `"_h"` → `"num_heads"` at the 9 seed sites in `spyre__sdpa_overrideable`, so the already-present `spyre_hint(tiles={"num_heads": num_heads // 4})` scope has a propagated named dim to bind to (today `"_h"` deliberately does not match, making the tile a silent no-op). Batch (`"_b"`) stays a placeholder. No test changes — the four existing H=8 prefill cases engage the tile at `num_heads // 4 = 2` and verify numerics against the CPU reference.

**Tech Stack:** PyTorch Inductor out-of-tree backend (torch-spyre), pytest, `spyre_hint` named-dim/tile seeding, `ParameterizedTestMeta` test framework.

## Global Constraints

- Line length: 88 chars (ruff).
- `import regex` (as `re` when needed), never `import re`.
- Every commit signed: `git commit -s -S` (DCO + GPG). Verify `echo test | gpg --clearsign` succeeds first. Never commit unsigned.
- Do not commit until the targeted tests are green.
- Run `pre-commit run --files <changed files>` before committing.
- `main` must track `upstream/main` (ABI/`flex::` symbols otherwise) — see memory `torch-spyre-main-tracks-upstream`.
- Verification runs on THIS machine: `import torch` gives Spyre device access. Import order matters — the test harness handles it.
- The `_C.so` extension is already built and current (it exports `register_kernel_provenance`; this branch verified it green at HEAD `14218a4`). No rebuild needed unless an `ImportError` for a C++ symbol appears — then `make setup` and diagnose before proceeding.
- **Anchor edits on the `"_h"` string inside a `named_dims` list, NOT on line numbers.** Line numbers below are current as of HEAD `14218a4` but are advisory; re-grep to confirm before editing.
- **Head only.** Do NOT touch `"_b"` (batch). Do NOT touch the deferred causal/mask/kv_tail paths, the `if True:` scaffolding, or `expect_fail`.

---

### Task 1: Confirm green baseline before the rename

The known-good baseline is HEAD `14218a4` (placeholder `"_h"`, head tile inactive). Confirm the four prefill cases are green **before** the rename so any later red is attributable to the head tile, not to drift or the build.

**Files:**
- None modified. Verification only.

**Interfaces:**
- Consumes: the current built `_C.so`.
- Produces: a known-green baseline (`mha_prefill`, `gqa_prefill`, `mha_prefill_8k`, `gqa_prefill_8k` pass) recorded for comparison after Task 2.

- [ ] **Step 1: Confirm the 9 seed sites are still `"_h"` and locate them**

Run:
```bash
cd /mnt/home/spyre/torch-spyre
grep -n '"_h"' torch_spyre/_inductor/decompositions.py
```
Expected: 11 hits — 9 inside `named_dims=[…]` lists (the code seed sites) and 2 inside comments (a `["_b","_h",…]` example and a "deliberately named `_b`/`_h`" note). If the count differs, STOP: the file moved under you; re-establish the seed sites before editing.

- [ ] **Step 2: Run the four prefill cases (the head-tile guard suite)**

Run (each 8k case compiles ~46s+; ~6-8 min total, single process):
```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and (mha_prefill or gqa_prefill) and not causal and not kv_tail and not mask" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -15
```
Expected: `4 passed` (`mha_prefill`, `gqa_prefill`, `mha_prefill_8k`, `gqa_prefill_8k`). If any fails, STOP and diagnose the baseline (build or environment) before touching the decomp — do not start Task 2 against a red baseline.

- [ ] **Step 3: No commit** (verification only).

---

### Task 2: Rename the head-axis seed `"_h"` → `"num_heads"`

**Files:**
- Modify: `torch_spyre/_inductor/decompositions.py` (inside `spyre__sdpa_overrideable`) — 9 code seed sites + 2 comments.

**Interfaces:**
- Consumes: `num_heads` (bound at `decompositions.py:402` as `query.size(1)`), the existing `spyre_hint(tiles={"num_heads": max(1, num_heads // 4)})` scope (line ~506, unchanged), the green baseline from Task 1.
- Produces: an active `num_heads` tile — `assign_dim_hints` now binds the `num_heads` tile to buffers carrying the propagated `"num_heads"` named dim.

- [ ] **Step 1: Rename `"_h"` → `"num_heads"` at each of the 9 code seed sites**

Each site is the **second** entry in a `named_dims=[…]` list (physical stride order `[batch, head, seq, dim]`). Change only that entry; leave `"_b"` as the first entry untouched. The 9 sites (advisory line numbers at HEAD `14218a4`):

- Line ~453: `named_dims=["_b", "_h", "max_seqlen_q", "head_dim"]` → `["_b", "num_heads", "max_seqlen_q", "head_dim"]`
- Line ~463: `named_dims=["_b", "_h", "max_seqlen_q"]` → `["_b", "num_heads", "max_seqlen_q"]`
- Line ~472: `named_dims=["_b", "_h", "max_seqlen_q"]` → `["_b", "num_heads", "max_seqlen_q"]`
- Line ~502: `named_dims=["_b", "_h", "max_seqlen_q", "head_dim"]` → `["_b", "num_heads", "max_seqlen_q", "head_dim"]`
- Line ~528: `named_dims=["_b", "_h", "blk_len", "head_dim"]` → `["_b", "num_heads", "blk_len", "head_dim"]`
- Line ~542: `named_dims=["_b", "_h", "blk_len", "head_dim"]` → `["_b", "num_heads", "blk_len", "head_dim"]`
- Line ~559: `named_dims=["_b", "_h", "max_seqlen_q", "blk_len"]` → `["_b", "num_heads", "max_seqlen_q", "blk_len"]`
- Line ~601: `named_dims=["_b", "_h", "max_seqlen_q", "head_dim"]` → `["_b", "num_heads", "max_seqlen_q", "head_dim"]`
- Line ~619: the multi-line `named_dims=[` list whose second element is `"_h"` on its own line → change that line to `"num_heads",`

**Do NOT** use a blind `sed 's/"_h"/"num_heads"/g'` — that would also rewrite the two comment occurrences (handled deliberately in Step 2 with different wording) and the `"_h"` example on line ~423. Edit the 9 `named_dims` list entries specifically.

After editing, verify no code seed site still says `"_h"`:
```bash
cd /mnt/home/spyre/torch-spyre
grep -n '"_h"' torch_spyre/_inductor/decompositions.py
```
Expected: only the 2 comment lines remain (the `["_b","_h",…]` example ~423 and the "deliberately named" note ~497) — those are updated in Step 2. Zero occurrences inside any `named_dims=[…]`.

- [ ] **Step 2: Update the two comments so they match the new reality**

Comment A — the "deliberately named placeholder" block (~lines 497-501). It currently reads (approximately):
```python
    # The batch and head dims are deliberately named "_b"/"_h" -- placeholder
    # names that do NOT match the tiles={"batch_size": ...} / {"num_heads": ...}
    # contexts below, so only the max_seqlen_q tile activates for now. Renaming
    # these to "batch_size"/"num_heads" is all it takes to light up those tiles
    # in a follow-up.
```
Replace with:
```python
    # The head dim is named "num_heads" so it matches the
    # tiles={"num_heads": ...} scope below and the head tile activates. The
    # batch dim is still the placeholder "_b" -- it does NOT match
    # tiles={"batch_size": ...}, so the batch tile stays inactive for now.
    # Renaming "_b" -> "batch_size" is all it takes to light up the batch tile
    # in a follow-up.
```
(All lines ≤ 88 chars, 4-space indent.)

Comment B — the illustrative example (~line 423), currently:
```python
    # is [B, H, S, D]. Seeding ["_b","_h","max_seqlen_q",...] on such a buffer
```
Replace `"_h"` with `"num_heads"` so the example matches the code:
```python
    # is [B, H, S, D]. Seeding ["_b","num_heads","max_seqlen_q",...] on such a buffer
```
Check this line is still ≤ 88 chars after the edit; if it now exceeds 88, wrap the comment across two lines rather than truncating meaning.

- [ ] **Step 3: Run the head-tile guard suite on-device**

Run:
```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and (mha_prefill or gqa_prefill) and not causal and not kv_tail and not mask" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -20
```
Expected: `4 passed`. With `num_heads // 4 = 2`, the head tile splits H=8 into 4 tiles of 2 heads on every case, and `compare_with_cpu` asserts the Spyre output still matches the CPU `scaled_dot_product_attention` reference within tolerance. This is the numeric proof that the activated head tile is correct.

- [ ] **Step 4: If a case fails, diagnose (do not mask)**

If any case fails after the rename:
- Capture the real error (not the DEBUG spew), e.g.:
  ```bash
  cd /mnt/home/spyre/torch-spyre
  python3 -m pytest "tests/inductor/test_inductor_ops.py::TestOps::test_sdpa_mha_prefill" \
    -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -40
  ```
- A **compile/codegen error** (restickify, `hint_id`, bundle `sdsc_*.json`, `_consume_names`, a work-division split error) is the second active tile dim surfacing a new codegen path — invoke `superpowers:systematic-debugging`. Distinguish it from the pre-existing deferred signature `coarse_tile.py:144 hint_id appears in more than one group`, which belongs to the causal/mask path, NOT this non-causal suite.
- A **numeric mismatch** (`compare_with_cpu` tolerance) means the head tile broke the online-softmax recurrence — check that the functional-SSA accumulator carry (`new_max`/`correction`) is unaffected by the head split; see memory `torch-spyre-sdpa-lq-tile-accumulator` (copy_f in-place accumulators give wrong numerics — the recurrence must stay functional).
- A transient hardware RAS fault (`0x7b1b ComputeHardwareError`) at result readback is NOT a code defect — re-run the single case in isolation to confirm it passes (this branch saw one such transient at 8k under sustained load). Do not diagnose it as a head-tile bug.
- Do NOT add any case to `expect_fail` to make the suite pass; the cases exist to prove correctness.

- [ ] **Step 5: Lint the changed file**

Run:
```bash
cd /mnt/home/spyre/torch-spyre
pre-commit run --files torch_spyre/_inductor/decompositions.py
```
Expected: all hooks pass (ruff line-length, regex-import, mypy, etc.). Fix any reported issues and re-run.

- [ ] **Step 6: Commit**

```bash
cd /mnt/home/spyre/torch-spyre
git add torch_spyre/_inductor/decompositions.py
git commit -s -S -m "feat(sdpa): activate num_heads tile in flash-attention decomp

The num_heads tile scope (tiles={\"num_heads\": num_heads // 4}) was already
present but its head-axis named-dim seeds were the placeholder \"_h\", which
does not match the scope, so assign_dim_hints dropped the tile (silent
no-op). Renaming the 9 seed sites \"_h\" -> \"num_heads\" gives the tile a
propagated named dim to bind to, so SDPA now tiles over the head axis
(H=8 -> 4 tiles of 2) in addition to max_seqlen_q. Batch (_b) stays a
placeholder. Verified: the four H=8 prefill cases (mha/gqa prefill + 8k)
stay green against the CPU reference via compare_with_cpu.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Regression sweep — confirm no status drift

Confirm the head-tile activation did not shift any other SDPA case's status: the deferred causal/mask/kv_tail cases must fail exactly as before (same `coarse_tile.py:144` signature) — no new failures, no case silently fixed or newly broken.

**Files:**
- None modified. Verification only.

**Interfaces:**
- Consumes: the committed change from Task 2.
- Produces: a recorded status of the whole `test_sdpa` param family, and the explicit net delta of this task.

- [ ] **Step 1: Run the deferred (non-8k) cases**

Run the causal/mask/kv_tail cases directly (avoids the long 8k compiles and the fail-fast that would skip them; ~2 min):
```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and (causal or mask or kv_tail) and not 8k" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -20
```
Expected: the same pre-Task-2 status — `mha_prefill_causal`, `gqa_prefill_causal`, `mha_prefill_kv_tail_causal` FAIL with `InductorError` at `coarse_tile.py:144 hint_id appears in more than one group` (deferred, out of scope), and the non-causal deferred siblings pass. Confirm the failure **signature** is unchanged; a NEW signature on these cases means the head tile perturbed the causal path and must be investigated.

- [ ] **Step 2: Record the outcome**

Note in the final report the net delta: which cases pass (the four prefill cases now with the head tile active), and that the deferred cases fail unchanged. Do not alter `expect_fail`.

- [ ] **Step 3: No commit** (verification only).

---

## Self-Review

**Spec coverage:**
- Rename `"_h"` → `"num_heads"` at the 9 seed sites (`decompositions.py`) → Task 2 Step 1. ✓
- Keep `num_heads // 4` tile size (scope untouched) → Task 2 does not touch the scope. ✓
- Update the two placeholder comments → Task 2 Step 2. ✓
- Leave `"_b"` / batch untouched → Global Constraints + Task 2 Step 1 (edit only the second entry). ✓
- No new test; H=8 suite is the guard → Task 1, Task 2 Step 3. ✓
- On-device numeric verification + baseline-stays-green → Tasks 1, 2 (Step 3). ✓
- Deferred causal/mask/kv_tail untouched + status unchanged → Task 3. ✓
- Risk: sequenced ahead of causal/mask, distinguishable signature → Task 2 Step 4 + Task 3 Step 1 (signature check). ✓
- Clean fallback (revert restores known-good) → implicit; the rename is the only code change. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step gives the literal before/after string; every run step gives a literal command and expected result. The one advisory ("line numbers are advisory, anchor on the `"_h"` string") is a deliberate safeguard against line drift, with a concrete grep to re-establish sites — not a placeholder. ✓

**Type consistency:** `num_heads` is an `int` (`query.size(1)`), consumed by the unchanged `max(1, num_heads // 4)` tile expression. The seed name `"num_heads"` (string) is what `assign_dim_hints` matches against the `tiles={"num_heads": …}` key — same literal string, used consistently across all 9 sites and the tile scope. Generated test names are unchanged (no test edits). ✓
