# Stream w2-ring-assembly: root cause of the overlay ring-assembly workaround

## Cause in plain terms

The bug is not in ring assembly, and luma.gl's buffer reuse is not wrong. The overlay read padding rows that its input never wrote, and the noder then used them.

1. `GPUPolygonOverlay`'s `combine-positions` kernel (`gpu-polygon-overlay.ts`, "combine-positions") gives every combined row a vertex. Some rows belong to no ring:
   - the one closing slot that an **empty ring** gets;
   - the **tail rows** after the last ring when the last ring is empty.
   For these rows it copied `positions[ringOffsets[last]]`. No ring owns that row of the operand.
2. A fixed-capacity overlay result pads its offsets flat and writes only the vertices it produced. The dissolve's result in building tessellation (`ringOffsets 0,10,10,10…`, 32 ring slots) is an example. Rows 10 and up of its `positions` are never written in that frame.
3. In simple-enclosure mode, the clip overlay uses that result as `left`. Its 31 empty rings and its tail rows each copied row 10. That row is a recipe transient, so it holds whatever the physical buffer last held.
4. `GPUSnapRoundedNoding` reads **every** row of `lines.positions` in two places, including rows on no line:
   - the `bounds` reduction, which sets the lattice frame;
   - `snap-vertices`, where any non-finite row sets `state[0]` and so `overflow`.
   A stale NaN or out-of-range row therefore sets the clip's noding overflow, then its boundary overflow, then its `overflow`, so `complete` = 0. A stale finite row would silently widen the lattice instead.
5. The throwaway `sourceIds` view changed nothing in the kernels. It added a `polygon-groups` node and a transient, which changed the compiler's lifetime-based buffer assignment. Row 10 then happened to hold a harmless value.

Ring assembly itself reads nothing stale (audit below).

## Evidence

- **Reproduced.** I removed the workaround (the `polygons: {...output.geometry, sourceIds: transient('assembly-source-ids', …)}` block and its 4-line comment).
  - `urban-ecology-advanced.spec.ts:233` and `:359` both failed with `expected [1, 0] to deeply equal [1, 1]`.
- **Diagnostic spec** (temporary, deleted). It published every status of both tessellation overlays.
  - Building 1's clip read `overflow 1`, `boundaryOverflow 1`, `boundaryCount 7 = boundaryRequired 7` against a capacity of 256, and `candidateOverflow 0`.
  - So the flag was not a capacity flag: it came from noding `overflow`. Its non-capacity sources are `state[0]` (a non-finite or out-of-range vertex) and uncertain pairs.
  - The dissolve's `rings` were `0,10,10,…`, which confirms 31 empty trailing rings that read row 10.
  - After the fix, without the workaround: `complete 1,1`, and both overflows are 0.
- **Which physical buffers alias** could not be shown. `@luma.gl/graph` publishes only the totals; the logical-to-physical map is `@internal`, and `describeGPUCommandGraph` says so (allocation stays unset).
  - The poisoning spec below proves the mechanism without depending on the allocator.

## Fix

- `modules/analysis/src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay.ts`, `combine-positions`:
  - A row of a non-empty ring copies as before: its vertices, then the ring's first vertex for the closing slot and any row after it.
  - A row of an empty ring, and a tail row whose ring is empty, now gets `getPaddingVertex()`. That is the first stored vertex of the left operand, else of the right operand, else `(0, 0)` when both operands are empty (no segment exists then).
  - A padding row now duplicates an existing vertex. It adds no hot pixel and does not move the bounds, which is the frame the docs describe: "extent of the finite vertices".
  - The kernel uses if/else, not `select`, so it never reads the other operand at a foreign row.
  - A comment states the rule.
- The throwaway-view workaround is removed. Ring assembly now gets `polygons: output.geometry`.
- `modules/analysis/src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts`, `hash` kernel: `counters` was declared `read_write` but is only read (the quantizer reads `counters[4]`). It is now declared `read`, with a comment.
  - This is the only declared-access mismatch I found in ring assembly. It was harmless: an over-declared write only adds hazards.

## Spec that fails before and passes after

New: `modules/analysis/test/gpu-spatial-analysis/polygon-overlay/polygon-overlay-padding-rows.spec.ts` (headless, 3 tests). Each test puts empty rings and tail rows in caller-owned input; nothing promises what those rows hold.

1. **Dissolve, NaN padding.** Expects `overflow 0`, `boundaryOverflow 0`, and a result equal to the unpadded run.
   - Before the fix: `boundaryOverflow` 1.
2. **Dissolve, padding at 3e6.** Expects a result equal to the unpadded run; the far value would coarsen the lattice and move the 0.3/0.9 corners.
   - Before the fix: positions differed.
3. **Intersection, both operands padded with NaN and with -3e6.** Expects a result equal to the unpadded run.
   - Before the fix: `overflow 1`.

All three failed before the fix and pass after it. The two urban-ecology tests are the integration evidence (fail before, pass after, with no workaround).

## Verified

Headless:
- overlay, float32-lattice, snap-rounding, padding-rows, polygon-buffer and buffer-property: 6 files, 223 passed.
- coverage-topology (2), make-valid float32-edges and property, geometry-topology, ring-assembly (3), polygonization diagnostics: 9 files, 56 passed.
- urban-ecology-advanced, urban-ecology, urban-ecology-outputs, global-coordinates, cell-set-outline, isoband rings and 2 parity files: 8 files, 110 passed.
- kepler `catalog-gpu`, lenses `global-coordinates` and `urban-ecology`, overlay A/B: 3 files passed, 34 tests; 1 file skipped (6 tests, the A/B spec skips by design).

Node:
- overlay, face-labelling-executed, polygon-buffer, coverage-topology, make-valid parity, forwarded-capacity, ring-assembly (3), urban-ecology, urban-ecology-limits: 11 files, 114 passed.
- s3 scenario and pack specs, kernel-audit geometry and urban-ecology: 305 passed, 1 failed. The failure is `s3-geometry-pack` GPUShapeGenerator `latitude2`; it predates this change and is listed in w2-geometry Noticed 2.

Checks:
- `node scripts/typecheck-status.mjs 'polygon-overlay|ring-assembly'`: 0 matching errors.
- `npx biome check --write` on my 3 files: clean.
- `check-upstream-first`: no violations.

Not run: full suites, `yarn typecheck-tests`, `check-contributor-contract`.

## Other places checked

**Ring assembly: every kernel's reads and writes.** Every read is bounded by a count or by a row that an earlier node in the same encode writes:
- `quantize-init`, `quantize-extent` and `hash` are guarded by the valid rows. `ringOrigin` reads endpoint row 0, which is used only when a valid row exists.
- `match` writes `next`, `alt` and `flags` for every row before its guard.
- `claim` is initialised in `hash`; `resolve` reads `flags` written by `match`.
- The jumps, `leaders` and `ring-starts` work on all rows.
- `ring-count` and `offsets` read `ringStarts` only below the ring total.
- `ring-stats` publishes zeros at or past `ringCount`, and `ring-orientation`, `ring-groups` and `ring-shells` are bounded by `ringCount`.
- The polygon kernels read `polygonStarts` only below the polygon count.

So callers without `sourceIds` have no special case. `GPUIsobandRings`, `GPUCoverageDissolve`, `GPUPolygonize` and `GPUCellSetOutline` were all checked.

**Overlay after noding.** `piece-canonical` through `emit-boundary` and the `boundary-status` kernel are bounded by the noded count or by group counts.

**The same pattern elsewhere.** Not fixed; not mine:
1. `gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts:330-344` (`bounds`, `itemCount: vertexCount`) and `:401-438` (`snap-vertices`).
   - Both read every `lines.positions` row, including rows on no line (`lineOfVertex` is NONE). A non-finite row sets `overflow` and a far finite row coarsens the lattice.
   - This is the systemic fix. The owner should skip rows outside `[lineOffsets[0], lineOffsets[lineCount])` in both kernels, or document that every row must be a finite vertex in the frame.
   - The overlay no longer depends on it. `GPUPolygonize` with `snapRounding` passes caller lines straight through, so a caller with padded line buffers is exposed.
2. `gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts:255-281` (`ring-closure-positions`): the same shape as the old `combine-positions`.
   - An empty ring's closing slot and the tail rows read `positions[ringStart]` = `positions[ringOffsets[last]]`, a row that no ring owns.
   - It feeds `GPUPolygonize`, and with `snapRounding` the noder from item 1. Probable, not proven: I did not reproduce it.
3. `gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction.ts:93-97` and `close-rings` (around line 150).
   - The transient `closed-positions` has `positions.length + ringCount` rows. Only the rows of real rings are written: an already-closed ring adds no row, and tail rows past the input's last offset are never written.
   - `GPUMakeValid` with `snapRounding: {gridSize: 1}` then passes them on to item 2 and item 1. On a reused transient this can set `overflow` or change the lattice frame. Not reproduced.
4. `GPULineNoding` (`gpu-line-noding.ts`) is bounded by `pieceVertexCount`: not affected.

## Upstream

Nothing goes to luma.gl. The compiler reused buffers whose declared lifetimes do not overlap, which is correct, so I added no row to `dev-docs/upstream-workarounds.md`.

## Roadmap

In STD-08's "Remaining", "the transient-reuse defect behind the overlay ring-assembly workaround" can be removed.

If wanted, add a geometry-topology item: "snap-rounded noding reads `lines.positions` rows outside every line (bounds, snap state); `GPUMakeValid` closure and `GPUPolygonPrecisionReduction` closed positions leave rows that no ring owns unwritten or stale."

## Could not establish

- Which logical transient shared the physical buffer of the dissolve's `positions` (the mapping is `@internal` in luma.gl).
- Whether items 2 and 3 above fail today on any lens or example input.

## Follow-up (coordinator request, 2026-11-11)

### 1. Noder bounded at the source

File: `gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts`.

- New `isLineRow(row)`: true when `lineOffsets[0] <= row < min(lineOffsets[lineCount], rows)`.
- New `lineOffsets` binding on both kernels that read all rows:
  - the `bounds` reduction (`readItem`): a row outside every line is not valid, so it cannot move the lattice frame;
  - `snap-vertices`: such a row stays NaN and no longer sets `state[0]`, so it is no longer invalid input.
- Rows inside a line's range keep the old behaviour, and later kernels already work per line.
- No fixture changed.

Spec: `modules/analysis/test/gpu-spatial-analysis/geometry-topology/snap-rounded-noding-padding-rows.spec.ts`.

- Each case runs once without padding, then with padding rows of NaN, 3e6 and -3e6, and expects the identical result: status, lattice, pieces or rings, and positions.
- Two of its four callers test this item:
  - `GPUSnapRoundedNoding` directly: 2 leading and 3 trailing rows;
  - `GPUPolygonize` with `snapRounding`: 3 trailing rows.
- Fails before, passes after: I made `isLineRow` always true and both tests failed with `overflow [1]` against `[0]`, then restored it.
  - (A first attempt at that toggle was invalid WGSL, because `||` and `&&` need parentheses. I redid it with parentheses.)

### 2. Make-valid and precision reduction

Both reproduced, through the same spec file:

- `GPUMakeValid` with `snapRounding`: 2 empty holes and 3 trailing rows.
  - It still failed with the noder bounded. An empty ring's closure row is a one-row line, so it is inside a line's range, and `ring-closure-positions` copied `positions[ringOffsets[last]]` into it.
- `GPUPolygonPrecisionReduction`: same input.
  - Its `close-rings` writes only the rows of real rings into the transient `closed-positions`. Make-valid's closure then read the unwritten row of that transient for each empty ring.

Fix, in `gpu-make-valid.ts` `ring-closure-positions`:

- A ring with vertices copies as before.
- An empty ring's row, and tail rows after an empty last ring, get the first stored vertex (`positions[ringOffsets[0]]` when any ring has vertices), else `(0, 0)`.
- `GPUPolygonPrecisionReduction` needed no change of its own. Make-valid now reads only rows that ring offsets own, and precision reduction writes all of those.
- Both cases fail before (`overflow [1]`) and pass after.
- I did not separate how much of the precision-reduction failure came from the poisoned caller rows and how much from the unwritten transient. Either way, the read was of a row no ring owns.

### 3. Sweep: kernels that scan a whole positions or segments view

Fixed and proven:

- `polygon-coverage-topology/coverage-topology-kernels.ts` `createCoverageSnapRangeNodes` (`snap-range`, about line 113).
  - It counted every finite row far from the snap origin into `uncertainCount`, padding included.
  - It now takes `ringOffsets` and skips rows outside `[ringOffsets[0], ringOffsets[last])`.
  - Callers updated: `gpu-coverage-dissolve.ts` (snap-range call) and `gpu-coverage-simplification.ts` (snap-range call).
  - Spec: `modules/analysis/test/gpu-spatial-analysis/polygon-coverage-topology/coverage-padding-rows.spec.ts` (dissolve, 3 trailing rows of NaN or ±3e6, snap tolerance 0.001). Fails before (`uncertainCount` differed) and passes after.

Listed, not fixed (no padding defect proven):

- `polygon-coverage-topology/coverage-topology-kernels.ts:45` `getSnapOrigin()` reads row 0.
  - That row is padding only when `ringOffsets[0] > 0` (leading padding).
  - The dissolve and simplification vertex tables (`gpu-coverage-dissolve.ts` vertex-table kernel, `gpu-coverage-simplification.ts:446`) and `spatial-weights/gpu-contiguity-weights.ts:424` would then quantise against a padding origin.
  - Not reproduced; GeoArrow inputs start at 0.
- `polygon-buffer/gpu-buffer-surface.ts:747` `compact-flags`: `flag = index < end` does not test `index >= begin`, so rows before `offsets[0]` are kept and scattered.
  - Leading padding only. The offsets there are produced internally from 0, so not reachable from a caller as far as I can see.
- Checked and bounded:
  - `gpu-coverage-dissolve.ts` and `gpu-coverage-simplification.ts` per-vertex kernels: an `inRing` guard, then `vertexPolygon != INVALID`.
  - `gpu-coverage-validity.ts:377` and `:548`: `segmentValid` is `Ring(row) != NONE`.
  - `gpu-geometry-validity.ts`: per feature, per ring, or per capacity of its own pairs.
  - `gpu-line-noding.ts`: bounded by `pieceVertexCount`.
  - `gpu-polygonize.ts`: no row scan of its own.

### Verified (follow-up)

Headless:
- New padding specs: noder 4 of 4 and coverage 1 of 1.
- snap-rounded-noding, make-valid float32-edges and property, geometry-topology, geometry-refinement advanced and topology-refinement: 7 files, 90 passed.
- overlay (4 files, including padding-rows), polygon-buffer (2), coverage-topology (2): 8 files, 233 passed.
- urban-ecology-advanced, urban-ecology, urban-ecology-outputs, global-coordinates, kepler catalog-gpu, lenses urban-ecology: 6 files, 113 passed.
- After the coverage change: coverage padding, coverage-topology (2) and coverage-simplification: 4 files, 21 passed.

Node:
- coverage-simplification, coverage-topology, kernel-audit geometry, coverage-urban-ecology and its siblings, s3 geometry scenarios, cells pack, lines pack: 9 files, 248 passed.
- The geometry-topology and refinement node specs (executed WGSL included), with overlay, buffer, coverage and urban-ecology node specs: 12 files, 152 passed. This batch ran before the coverage snap-range change; the coverage node specs were rerun after it in the batch above.

Checks:
- `typecheck-status` for 'coverage|snap-rounded|make-valid|polygon-overlay|ring-assembly': 0 errors.
- biome: clean.
- `check-upstream-first`: no violations.
