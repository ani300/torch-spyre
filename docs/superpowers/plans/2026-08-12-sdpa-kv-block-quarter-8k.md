# SDPA KV-block ¼-length + 8k prefill tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Size the flash-attention SDPA KV block at ~¼ of the KV length (so block count stays ~4 instead of 128 at 8k) and add 8k prefill tests that prove numeric correctness on-device.

**Architecture:** One-line change to `kv_block_size` in the SDPA decomp (`spyre__sdpa_overrideable`), from fixed `64` to `max(64, round-up-to-stick(ceil(Skv/4)))`. Two new parametrized test cases (`mha_prefill_8k`, `gqa_prefill_8k`) matching the verified `B=1,H=8,Lq=512,Lk=8192,D=128` config, checked against the CPU reference by the existing `compare_with_cpu` framework.

**Tech Stack:** PyTorch Inductor out-of-tree backend (torch-spyre), pytest, `ParameterizedTestMeta` test framework.

## Global Constraints

- Line length: 88 chars (ruff).
- `import regex` (as `re` when needed), never `import re`.
- Every commit signed: `git commit -s -S` (DCO + GPG). Verify `echo test | gpg --clearsign` succeeds first. Never commit unsigned.
- Do not commit until the targeted tests are green.
- Run `pre-commit run --files <changed files>` before committing.
- `main` must track `upstream/main` (ABI/`flex::` symbols otherwise) — see memory `torch-spyre-main-tracks-upstream`.
- Verification runs on THIS machine: `import torch` gives Spyre device access (`torch.spyre.is_available()` is `True`). Import order matters — `import torch` before `import torch_spyre`, or the test harness which does this for you.

---

### Task 0: Confirm green baseline on the rebuilt extension

The merge with main pulled in `register_kernel_provenance` (commit `36af10b`); the pre-existing `_C.so` was stale and both prefill tests failed with an `ImportError` at codegen time. The extension has been rebuilt (`make setup`, `_C.so` now exports the symbol). This task confirms the baseline is genuinely green **before** touching the decomp, so any later red is attributable to our change and not the build.

**Files:**
- None modified. Verification only.

**Interfaces:**
- Consumes: rebuilt `torch_spyre/_C.so` (exports `register_kernel_provenance`).
- Produces: a known-green baseline — the two prefill param cases `test_sdpa_mha_prefill`, `test_sdpa_gqa_prefill` pass.

- [ ] **Step 1: Confirm the rebuilt extension exports the new symbol**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
python3 -c "import torch; from torch_spyre import _C; print('sym:', hasattr(_C,'register_kernel_provenance'))"
```

Expected: `sym: True`

- [ ] **Step 2: Run the SDPA prefill baseline (non-causal, non-tail, non-mask)**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and (mha_prefill or gqa_prefill) and not causal and not kv_tail and not mask and not 8k" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -15
```

Expected: `2 passed` (the `mha_prefill` and `gqa_prefill` cases). If either fails with anything other than a pre-known deferred cause, STOP and diagnose the build before proceeding — do not start Task 1 against a red baseline.

- [ ] **Step 3: No commit** (verification only).

---

### Task 1: Change `kv_block_size` to stick-rounded ¼ of KV length

**Files:**
- Modify: `torch_spyre/_inductor/decompositions.py:441` (inside `spyre__sdpa_overrideable`)

**Interfaces:**
- Consumes: `max_seqlen_kv` (already bound at `decompositions.py:405` as `key.size(2)`).
- Produces: `kv_block_size` (int) used unchanged by `num_kv_blocks` (line 443), the `for blk` loop (line 503), and `end = min(start + kv_block_size, max_seqlen_kv)` (line 505). No other line changes.

- [ ] **Step 1: Verify the current baseline still green (guard)**

This is covered by Task 0 Step 2. Do not proceed to Step 2 unless that baseline is green.

- [ ] **Step 2: Replace the hardcoded block size**

In `torch_spyre/_inductor/decompositions.py`, replace exactly:

```python
    kv_block_size = 64
```

with:

```python
    # Fixed block size of 64 gave num_blocks = ceil(Skv/64), which climbs
    # with sequence length -- 8k KV = 128 blocks, and block counts > 6 do
    # not yet compile (PR #3672). Size the block at ~1/4 of the KV length
    # instead, so the block *count* stays ~4 regardless of Skv (8k -> 2048,
    # 4 blocks). Round up to a 64-element fp16 stick so every block stays
    # stick-aligned; floor at 64 so short KV keeps the original small-block
    # behavior (128 -> 64, unchanged).
    kv_block_size = max(64, ((max_seqlen_kv + 3) // 4 + 63) // 64 * 64)
```

(All lines ≤ 88 chars. Keep the 4-space indent — this is inside the function body.)

- [ ] **Step 3: Re-run the baseline to prove the change is a no-op for small KV**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and (mha_prefill or gqa_prefill) and not causal and not kv_tail and not mask and not 8k" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -15
```

Expected: `2 passed`. For KV=128 the formula yields `max(64, ceil(128/4)=32 -> 64) = 64`, identical to the old fixed value, so these must stay green. If they regress, the formula or an unrelated build issue is at fault — STOP and diagnose.

- [ ] **Step 4: Lint the changed file**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
pre-commit run --files torch_spyre/_inductor/decompositions.py
```

Expected: all hooks pass (ruff line-length, regex-import, etc.). Fix any reported issues and re-run.

- [ ] **Step 5: Commit**

```bash
cd /mnt/home/spyre/torch-spyre
git add torch_spyre/_inductor/decompositions.py
git commit -s -S -m "feat(sdpa): size KV block at ~1/4 of KV length, stick-aligned

Fixed block size of 64 made num_blocks grow with sequence length (8k KV =
128 blocks); block counts > 6 do not yet compile (PR #3672). Sizing the
block at ceil(Skv/4) rounded up to a 64-element fp16 stick (floored at 64)
keeps the block count near 4 (8k -> 2048, 4 blocks) and leaves short-KV
tests on their exact current path (128 -> 64, unchanged).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add 8k MHA + GQA prefill test cases

**Files:**
- Modify: `tests/inductor/test_inductor_ops.py` — the `("test_sdpa", "test_sdpa_cpu")` `param_sets` dict. Insert the two new cases immediately after the `"gqa_prefill_causal"` case (ends ~line 3155) and before `"mha_prefill_kv_tail"` (~line 3156), grouping them with the other prefill cases.

**Interfaces:**
- Consumes: `cached_randn` (from `tests/inductor/utils_inductor.py`, already imported in the test module); the block-size change from Task 1 (required for these to compile with ~4 blocks instead of 128).
- Produces: two new parametrized tests, `test_sdpa_mha_prefill_8k` and `test_sdpa_gqa_prefill_8k` (names generated by `ParameterizedTestMeta` from the param_set keys), each verified against `test_sdpa_cpu` via the framework's `compare_with_cpu`.
- Param tuple order (from existing cases): `(q, k, v, attn_mask, is_causal, enable_gqa)`.

- [ ] **Step 1: Add the two param_set entries (this IS the test)**

In `tests/inductor/test_inductor_ops.py`, inside the `("test_sdpa", "test_sdpa_cpu")` `param_sets` dict, insert after the `"gqa_prefill_causal"` entry:

```python
                "mha_prefill_8k": (
                    cached_randn(
                        (1, 512, 8, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (1, 8192, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (1, 8192, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    False,
                    False,
                ),
                "gqa_prefill_8k": (
                    cached_randn(
                        (1, 512, 8, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (1, 8192, 2, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (1, 8192, 2, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    False,
                    True,
                ),
```

Do NOT add these keys to the `"expect_fail"` list — they are expected to pass.

- [ ] **Step 2: Run the two 8k cases on-device**

Run (each compiles ~46s+, so target them; single process):

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa and 8k" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -25
```

Expected: `2 passed` — `test_sdpa_mha_prefill_8k`, `test_sdpa_gqa_prefill_8k`. The `compare_with_cpu` framework asserts the Spyre output matches the CPU `scaled_dot_product_attention` reference within tolerance.

- [ ] **Step 3: If a case fails, diagnose (do not mask)**

If either 8k case fails:
- Capture the real error (not the DEBUG spew):

  ```bash
  cd /mnt/home/spyre/torch-spyre
  python3 -m pytest "tests/inductor/test_inductor_ops.py::TestOps::test_sdpa_mha_prefill_8k" \
    -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -40
  ```

- A `hint_id appears in more than one group` / restickify / bundle (`sdsc_*.json`) error is a codegen path issue at scale — invoke `superpowers:systematic-debugging`. Do NOT add the case to `expect_fail` to make the suite pass; the case exists to prove correctness.
- A numeric mismatch (`compare_with_cpu` tolerance) means the ¼-block online-softmax recurrence is wrong at 4 blocks — check the `correction`/`new_max` carry across the (now 4) blocks.

- [ ] **Step 4: Lint the changed file**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
pre-commit run --files tests/inductor/test_inductor_ops.py
```

Expected: all hooks pass. Fix and re-run if needed.

- [ ] **Step 5: Commit**

```bash
cd /mnt/home/spyre/torch-spyre
git add tests/inductor/test_inductor_ops.py
git commit -s -S -m "test(sdpa): add 8k mha/gqa prefill cases for 1/4 KV block

Adds mha_prefill_8k and gqa_prefill_8k (B=1,H=8,Lq=512,Lk=8192,D=128),
matching the verified 4x2048 config from PR #3674. With the stick-rounded
1/4 KV block these compile to 4 blocks and are checked against the CPU
scaled_dot_product_attention reference via compare_with_cpu.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Full regression sweep of the SDPA suite

Confirm the change did not shift any other SDPA case's status (the deferred causal/mask/kv_tail failures should remain exactly as before — no new failures, no accidental fixes claimed).

**Files:**
- None modified. Verification only.

**Interfaces:**
- Consumes: the committed changes from Tasks 1 and 2.
- Produces: a recorded status of the whole `test_sdpa` param family.

- [ ] **Step 1: Run the entire SDPA param family**

Run:

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest "tests/inductor/test_inductor_ops.py" \
  -k "test_sdpa" \
  -q -p no:cacheprovider 2>&1 | grep -vE "DEBUG|__trace" | tail -30
```

Expected: `mha_prefill`, `gqa_prefill`, `mha_prefill_8k`, `gqa_prefill_8k` pass. The deferred cases (`mha_prefill_causal`, `gqa_prefill_causal`, `mha_prefill_mask`, `mha_prefill_kv_tail`, `mha_prefill_kv_tail_causal`) fail or xfail exactly as they did before Task 1 — no NEW failures introduced, and no previously-failing case silently changed status.

- [ ] **Step 2: Record the outcome**

Note in the final report which cases pass and which remain deferred-failing, so the delta from this effort is explicit. Do not alter `expect_fail` to make deferred cases green.

- [ ] **Step 3: No commit** (verification only).

---

## Self-Review

**Spec coverage:**
- Rebuild prerequisite → Task 0. ✓
- Block-size formula change (`decompositions.py:441`) → Task 1. ✓
- 8k mha + gqa test cases → Task 2. ✓
- On-device verification + baseline-stays-green → Tasks 0, 1 (Step 3), 3. ✓
- Deferred scope (causal/mask/kv_tail/batch-head-tiles/`if True:`) untouched → not implemented, and Task 3 confirms their status is unchanged. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step has literal code; every run step has a literal command and expected result. ✓

**Type consistency:** `kv_block_size` is an `int` in Task 1 and consumed as an int by the existing loop arithmetic. The param tuple order `(q, k, v, attn_mask, is_causal, enable_gqa)` in Task 2 matches the existing cases' order verified in the spec. Generated test names `test_sdpa_mha_prefill_8k` / `test_sdpa_gqa_prefill_8k` are used consistently in Tasks 2 and 3. ✓
