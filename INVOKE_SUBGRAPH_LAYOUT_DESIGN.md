# Design sketch — propagate Spyre layouts through `InvokeSubgraph`

Status: DESIGN ONLY (no code changed). Follows systematic-debugging Phase 1/2,
device-verified root cause in `INVOKE_SUBGRAPH_SPLIT_DIAGNOSIS.md` (Blocker 3).

## The failure, one line

`batchmatmul: cannot restickify any input layout of y to carry y_var=d1`
(`propagate_layouts.py:804`) raised while lowering the **subgraph**, because the
subgraph's graph-input buffers carry **no Spyre device layout** — so the block's
attention batchmatmul has no stick-compatible candidate for its `y` operand.

## Why the subgraph inputs have no layout (mechanism, verified)

1. `InvokeSubgraph.create` (pytorch `ir.py:10219-10227`) lowers the region body
   on **fake** operands: `make_subgraph(example_inputs=fake_operands)` then
   `subgraph.graph.run(*fake_operands)`. The subgraph's `graph_inputs`
   InputBuffers are therefore built from fakes and hold plain `FixedLayout`.
2. Each operand is stride-aligned to its subgraph input by
   `constrain_to_fake_tensor` (`ir.py:10203-10217`), so **operand[i] ↔
   subgraph.graph_input_names[i] are positionally aligned** — but only strides
   are matched, not Spyre STLs.
3. The subgraph is scheduled + lowered *lazily during the parent's codegen*:
   `codegen_subgraph_common` (pytorch `wrapper.py:4174-4193`) calls
   `subgraph.graph.codegen()` under `V.set_graph_handler(subgraph.graph)`, which
   runs `subgraph._update_scheduler()` → torch-spyre's
   `propagate_spyre_tensor_layouts(subgraph.graph)`.
4. That pass's input-conversion block (`propagate_layouts.py:1582-1607`) does
   `stl = real_input.device_tensor_layout()`. For a subgraph the "real inputs"
   are fakes (or unset), and `device_tensor_layout()` **returns `None` for a
   FakeTensor** (`_monkey_patch.py:65-66`) → the block `continue`s → the input's
   `.layouts` stays empty. Starvation.
5. In the **parent** pass, the `InvokeSubgraph` node itself is an `ExternKernel`
   so it hits the `unhandled node type` warning (`propagate_layouts.py:1864-65`)
   and does nothing — the operands feeding it already have correct STLs from the
   parent's own propagation, but nothing carries them across the boundary.

Decisive control (Phase 2): the identical matmul in the **eager** path
(`test_e2e_token_compare_spyre[...granite-3.3-2b-instruct]`, blocks compiled
individually, no HOP) **passes 5/5**. Only the HOP boundary differs.

## Ordering guarantee (why the fix has the data it needs)

`GraphLowering.codegen()` runs `_update_scheduler()` (→ Spyre passes) **before**
`scheduler.codegen()` (pytorch `graph.py:2603-2611`). The subgraph is only
codegen'd *inside* the parent's `scheduler.codegen()`. Therefore, when we visit
the parent's `InvokeSubgraph` node during the parent pass, the parent has
already resolved `.layouts` on every operand, and the subgraph's own pass has
**not yet run**. We can seed the subgraph inputs from the parent side and the
subgraph pass will see them.

## Fix — forward direction (fixes the abort)

Handle `InvokeSubgraph` in `propagate_spyre_tensor_layouts`'s parent walk
instead of warning-and-skipping. When visiting the node:

- For each `(operand, input_name)` in
  `zip(op.inputs, op.subgraph.graph.graph_input_names)`:
  - Read the operand's resolved STL from the parent buffer
    (`operand`'s `.layouts[0]`, via the same `_get_prop_args`/`get_buffer`
    machinery the pass already uses for reads).
  - Stamp it onto the subgraph's graph-input TensorBox:
    `subgraph.graph.graph_inputs[input_name].layouts = [stl]`, and set the
    underlying `InputBuffer.layout` to the matching `FixedTiledLayout`
    (mirroring exactly what lines 1592-1607 do for a normal graph input, minus
    the fake-tensor gate — we already hold the STL, no `device_tensor_layout()`
    round-trip needed).
- Skip operands that are `ShapeAsConstantBuffer` / non-Spyre (no STL), matching
  the existing `stl is None → continue` policy.

Then make the subgraph's input-conversion block **respect a pre-seeded layout**:
in `propagate_layouts.py:1582-1607`, if `graph.graph_inputs[name].layouts` is
already populated (parent seeded it), use that instead of the
`device_tensor_layout()` (which is `None` for the fake) — i.e. the parent seed
takes precedence over the fake-tensor probe. This is the minimal, local change;
the rest of the subgraph pass is unchanged and now finds stick-compatible
candidates for the matmul `y`.

Linkage is safe because operand↔input positional alignment is guaranteed by
`constrain_to_fake_tensor` (point 2 above). No parent-pointer walking needed —
`op.subgraph.graph` is directly on the `InvokeSubgraph` IR node.

## Fix — reverse direction (correctness, not the current abort)

The parent's `MultiOutput` nodes for the subgraph results currently get
`generic_layout` (`propagate_layouts.py:1838-1840`). For the pointwise repro
that was harmless; for Granite the downstream parent ops (final `norm`,
`lm_head`) read those outputs and a generic layout can force an avoidable
restickify or a wrong candidate. After the subgraph pass has run, its
`graph_outputs` carry real STLs. Propagate them back: when the parent visits the
`InvokeSubgraph`/its `MultiOutput`s, copy `subgraph.graph.graph_outputs[i]`'s STL
onto `outputs[i].layouts` instead of `generic_layout`.

Caveat / ordering wrinkle: the subgraph pass runs *lazily at codegen*, i.e.
**after** the parent's layout pass has already visited the `MultiOutput`s. So
the reverse direction cannot simply read subgraph outputs during the parent
pass — they aren't computed yet. Options, in preference order:
  (a) Two-visit: parent pass records the `InvokeSubgraph`; a
      `_pre_fusion_custom_pass` (like `propagate_mutation_layouts`,
      `propagate_layouts.py:1872`) fixes up the `MultiOutput` STLs after the
      subgraph has been lowered. Mirrors the existing deferred-mutation pattern.
  (b) Eagerly lower the subgraph during the parent pass (force
      `subgraph.graph.codegen()` earlier) — heavier, changes ordering, risk of
      double-lowering; **not preferred**.
The abort is fixed by the forward direction alone; the reverse direction is a
follow-up for restickify-count / correctness on the parent tail and can land
separately if the forward fix already produces matching tokens.

## `work_division` — same gap, same shape

`work_division.py` emits the identical `unhandled node type InvokeSubgraph`
warning. Work-division metadata must likewise flow parent→subgraph (the
subgraph's ops need the parent's core/tile assignment context). Same seeding
approach on the `InvokeSubgraph` node. Confirm whether the current abort is
reached before or after work_division — if layout is fixed first and the graph
then aborts in work_division, this is the next domino. Investigate on-device
after the layout forward-fix, don't pre-fix blind.

## Test / verification plan

1. Extend `repro_invoke_subgraph_split.py` (host, fake tensors, no card) with a
   **matmul** inside the region (not just pointwise), reproducing `y_var`
   starvation at compile time → currently aborts, must `COMPILE OK` after fix.
2. Re-run the device gate
   `whole_forward_e2e.py::test_whole_forward_token_compare_spyre[...granite-3.3-2b-instruct]`
   → prefill + 4 decode top-1 tokens match HF ref. Any mismatch → STOP-AND-REPORT.
3. Regression: eager `test_e2e_token_compare_spyre` stays 5/5; split-path tests
   (`test_dedup_constants`, `test_provenance*`, `test_propagate_named_dims`,
   `test_log_passes`) stay green; `pre-commit` clean.

## Scope

- Forward seed in `propagate_spyre_tensor_layouts` (parent `InvokeSubgraph`
  branch + subgraph input-block pre-seed check): fixes the abort. Small, local.
- Reverse `MultiOutput` fixup via a deferred pass: correctness follow-up.
- `work_division` seeding: next blocker if it fires; verify on-device first.
- Depends on / composes with Fixes 1 & 2 (already applied) — those are
  prerequisites; this is the third and last known InvokeSubgraph blocker for the
  Granite whole-forward.
