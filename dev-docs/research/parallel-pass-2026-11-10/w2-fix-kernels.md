# w2-fix-kernels

Source: `dev-docs/research/parallel-pass-2026-11-10/review-kernels.md`. All five findings were reproduced on the headless GPU and fixed. No git commands were run except read-only ones.

## 1. Done

### Finding 1: `GPULineSplit` `flag-non-finite` read padding rows. Reproduced; fixed.

- **Spec (new):** `modules/analysis/test/gpu-spatial-analysis/line-split/line-split-padding-rows.spec.ts`, 4 tests.
  - It runs `GPULineSplit` directly, `GPULineNoding`, `GPUNetworkNoding`, and `GPUPolygonize` without snap rounding.
  - Each runs with 2 leading and 3 trailing padding rows filled with NaN, 3e6 or -3e6, and must match an unpadded run.
  - What is compared: all status words, pieces, positions, source ids and the CSR. Edge indices are rebased by the leading count, and the unspecified last-vertex edges are skipped.
- **Before the fix:** 4 of 4 failed. With NaN padding, `overflow` was 1. `GPUNetworkNoding` also had `pieces.status.overflow` 1, and polygonize had `nodedOverflow` 1. Nothing else differed.
- **After the fix:** 4 of 4 pass.
- **Change in `gpu-line-split.ts`:** `flag-non-finite` now binds `lineOffsets` and checks only rows in `[lineOffsets[0], min(lineOffsets[last], rows))`. A comment states the rule.
- **Other kernels in the file, read for the same pattern:**
  - `event-slots` and `self-contacts` call `lineOfVertex` only on segment ids taken from intersection pairs.
  - `line-splits`, `pieces` and `write-vertices` read only line ranges.
  - `self-contacts-clear` writes every row but reads none.
  - None of them read padding rows. No other change was needed.
- **Coverage gap:** `GPULineLoopRemoval` and `GPUOffsetCurveComponents` inherit the fix through `GPULineSplit` and `GPULineNoding`. They have no padding spec of their own.

### Finding 2: `GPUMakeValid` ring closure copied rows before `ringOffsets[0]`. Reproduced; fixed.

- **Spec (new):** `modules/analysis/test/gpu-spatial-analysis/geometry-topology/make-valid-leading-rows.spec.ts`, 2 tests (with and without snap rounding). Ring offsets are `[2, 6]`, there are 3 trailing rows, and the padding is NaN, 3e6 or -3e6.
- **Before:** with both fixes temporarily reverted, the test without snap rounding failed (`overflow` 1, `nodedOverflow` 1).
  - With only the make-valid fix applied, it passed.
  - With only the line-split fix applied, it also passed.
  - So either fix closes the symptom on its own, and both are kept.
- **Change in `gpu-make-valid.ts` (`ring-closure-positions`):** the redirect is now `if (index < ringStart || source >= ringEnd) { source = ringStart; }`. A leading row now repeats the first vertex of ring 0, the same padding rule as for empty rings. The comment was updated.

### Finding 3: `GPULineClipByPolygon` emitted a phantom segment from leading padding. Reproduced; real; fixed.

- **The reviewer's open question, answered:** yes, the phantom segment is output.
  - `sub-counts` gives it one sub-piece, and `sub-pieces` writes it with line 0.
  - Containment then classifies it by its midpoint, since no intersection pairs exist for it.
  - `classify`, `run-flags` and `write-vertices` then emit it, joined onto line 0's run when the endpoints chain.
- **Spec (new):** `modules/analysis/test/gpu-spatial-analysis/line-clip/line-clip-padding-rows.spec.ts`, 2 tests (inside and outside modes).
  - The padding is NaN, ±3e6, or 1.25. At 1.25 the phantom segment's midpoint lies inside the unit square.
- **Before the fix:**
  - Inside mode with 1.25 padding returned an extra piece `[(1.25, 1.25), (-1, 0.5), …]`: count 3 instead of 2.
  - Outside mode with 3e6 padding prepended `(3e6, 3e6)` to line 0's first piece.
- **After the fix:** 2 of 2 pass.
- **Change in `segment-ranges`:** `valid` now also requires `index >= lineOffsets[line]`. Trailing rows were already excluded by the `lineEnd` test.

### Finding 4: south point-cloud rows misassigned just below row boundaries. Reproduced; fixed.

- **Spec (new):** `modules/analysis/test/gpu-pointcloud/gpu-pointcloud-row-boundaries.spec.ts`, 4 tests on a 1-column by 1000-row grid with origin 0 and cell size 1. Each runs north, south and default.
  1. **Height (cell mode):** points at every boundary `k` and at the float32 just above and just below it, plus points off both edges. The cell index is read through terrain `i + 1` and checked against a binary64 reference: inside when `0 <= o < H`, south row `H - 1 - floor(o)`.
  2. **Height (bilinear and IDW):** the `INVALID_POINT` status must equal "outside" under the half-open rule.
  3. **Height (cell mode), random:** 200,000 uniform random float32 y values over `[-1, H + 1)`; every cell must match the reference.
  4. **Surfaces:** each row holds its lower-boundary point as the minimum and its just-below-next-boundary point as the maximum. Surface and terrain must equal the arranged expectation. This also checks that the grid-index key agrees with the cell source.
- **Before the fix:** 4 of 4 failed in south and default; north passed.
  - Test 1 had 490 mismatches. Examples: y = 0.99999994 gave row 998 instead of 999, and y = -1e-6 or -2^-20 counted as inside, in row 999.
  - Test 2: the points below the origin were not flagged invalid.
  - Test 3: 3 of 200,000 random points were misassigned.
  - Test 4: the surface and terrain arrays differed.
- **After the fix:** 4 of 4 pass.
- **Change in `getPointCloudCellSource` (`gpu-pointcloud-surfaces.ts`):** in the south branch, `inside` and `cell` are now computed from `southOffset`, the same half-open test as north. The cell is `northCell = clamp(floor(southOffset))` and `row = H - 1 - northCell.y` in u32.
  - `coordinate` (`H - offset.y`) is kept only as the continuous row coordinate for bilinear and IDW.
  - Surfaces, Height in every mode, Ground and SMRF all call this one function.
- **Change in `pointcloud-grid-index.ts`:**
  - In raster mode, `coordinate` is now the origin offset in both directions, and `inside` is the shared half-open test.
  - For south, the key row is `H - 1 - u32(floor(y))`.
  - The returned `coordinate` and `inside` are consumed only by `GPUPointCloudNeighborhoods`, which is not raster mode, so nothing else changes.
- **Docs:** `docs/api/pointcloud-analysis.md` (the row-order bullet) now states the half-open rule for both directions.
- **Which points change:** only points whose float32 `H - o` rounded across an integer, plus points just below the origin (now outside). Every other point keeps its cell, as test 3 shows.

### Finding 5: walking-cost parameters packed for the other model. Reproduced; fixed without a layout change.

- **Why no layout change:** the packed layout already tells the models apart. Slots 8 to 12 are all zero for Tobler, and the packer validates the GRASS threshold in slot 12 to be negative. So no model-identifier word was added.
  - `GPU_RASTER_WALKING_COST_PARAMETER_LENGTH` is unchanged.
  - The lens runner and the website run already pack with the matching model; neither needed an edit.
- **Changes in `gpu-raster-walking-cost.ts`:**
  - A WGSL `parametersMatchModel()` is compiled per `model.kind`:
    - GRASS: slot 12 `< 0` and slots 8 to 12 finite.
    - Tobler: slots 8 to 12 all `== 0`.
  - The relaxation round relaxes nothing on a mismatch. Sources keep cost 0, every other cell stays `+Infinity`, and `converged` is 1.
  - `output.status` gains an optional `invalidCount` (the existing `GPUInvalidStatusPort` name). It is written by a new single-invocation `check-parameters` node, added only when the view is given.
  - The view is validated for length; the node budget counts it.
  - The CPU never reads it back.
  - The TSDoc of the class, the props, the status and the packer states the contract.
- **Docs:** `docs/api/gpu-raster/expansion.md` (the walking-cost georeference section) gained a paragraph. The generated parameters block was not touched; `doc-parameters.node.spec.ts` passes.
- **Spec:** `modules/analysis/test/geospatial-expansion/raster-walking-cost-parameters.spec.ts` (edited).
  - New test "reports parameters packed for the other model". It runs a GRASS kernel with Tobler parameters and the reverse, and expects `invalidCount` 1 and costs `[0 at the source, Infinity elsewhere]`. It then repacks for the compiled model in the same compiled graph and expects `invalidCount` 0 and the float64 reference costs.
  - The existing regression test now also asserts `invalidCount` 0 on matched parameters.
- **Before the fix:** the new test failed ("grass kernel, tobler parameters: expected [0] to deeply equal [1]"). **After:** 4 of 4 pass, and the 74dddcf5 bit-regression is unchanged.

## 2. Verified

- **Headless:** `npx vitest run --project headless` on the following, 35 files, 437 tests passed, 0 failed:
  - directories `line-split`, `line-clip`, `geometry-topology`, `geometry-refinement` (which holds the loop-removal, offset-curve and precision-reduction specs), `gpu-network/network-noding`, `gpu-pointcloud`;
  - files `raster-walking-cost-parameters.spec.ts` and `raster-terrain-expansion.spec.ts`.
- **Headless:** lenses `kernel-records.spec.ts` plus the `segment-intersection` directory: 4 files, 20 passed.
- **Node:** `npx vitest run --project node` on the same directories plus the following, 27 files, 650 passed:
  - `raster-terrain-expansion.node`, `segment-intersection`, `analysis-operator-catalog.node`;
  - the kernel audits: `kernel-audit.node`, `-geometry`, `-network`, `-expansion`, `-pointcloud-and-datum`, `-raster`.
- **Node:** `doc-parameters.node.spec.ts` (1 passed), lenses `expansion-spatial.node` and `expansion-network-dataframe.node` (2 files, 29 passed).
- **Source typecheck:** `npx tsc -p tsconfig.check-src.json --noEmit` was run once and showed no errors in the touched files.
- **Checks:** `npx biome check --write` on all touched files; `yarn check-contributor-contract` reports no new findings; `node scripts/check-upstream-first.mjs` reports no violations.
- **Not run:**
  - the test-project typecheck (`yarn typecheck-tests` is banned; the specs compile under vitest);
  - the full suites;
  - the website build.

## 3. Not done

- There is no dedicated padding spec for `GPULineLoopRemoval` or `GPUOffsetCurveComponents`. Both are covered through the shared `GPULineSplit` and `GPULineNoding` fix, and their existing specs pass.
- The polygon side of `GPULineClipByPolygon` with `ringOffsets[0] > 0` was not examined. It belongs to the point-in-polygon join and the segment intersection, which are outside my files.

## 4. Website and examples work list

None required:

- The walking-cost layout and length are unchanged.
- `invalidCount` is optional.
- `website/src/gpu/runs/entries/raster-bands-indices/walking-cost.ts` already packs with `getModel(current.model)` and rebuilds when the model changes.

Optional: that run could wire `status.invalidCount` to show the check.

## 5. Roadmap

For any row naming these review findings (line-split padding, make-valid leading rows, line-clip leading padding, south row rounding, walking-cost model mismatch): delete the row. What remains, if a row is kept:

- "Padding specs for `GPULineLoopRemoval` and `GPUOffsetCurveComponents`, and for the polygon operand of line clip with `ringOffsets[0] > 0`."

## 6. Noticed

- `getSnapOrigin` reads row 0 (`coverage-topology-kernels.ts:45`), as in review item 5. It is outside my scope and untouched.
- The north raster branch of the grid index and the cell source were already correct. The south change makes the two directions share one membership test, so a point's cell no longer depends on the direction beyond the row mirror.
