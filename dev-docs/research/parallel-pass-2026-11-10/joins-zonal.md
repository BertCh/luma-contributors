# joins-zonal report (wave 1, 2026-11-10)

Paths are relative to `modules/analysis/src/gpu-spatial-analysis/` unless they start with `modules/`, `docs/` or `scripts/`.

## 1. Done

**Item 1, GPU-04: a BVH leaf overflow now reports and the anti join withholds rows.**
- Where it overflows: `GPUBVH` keeps only the first `leafCapacity` source rows (or the first rows in sorted order) when the feature count is above `leafCapacity`. This happens when a caller passes a `leafCapacity` that is too small, or a `GPUSpatialJoinPrepared` built with one. The affected probe rows cannot be identified, because a dropped feature can match any row. So every probe row is marked, which is the conservative choice.
- `spatial-join/spatial-join-passes.ts`:
  - `getCandidateTruncationNodes` takes an optional `bvhOverflow`. When it is nonzero, every row gets `candidateTruncated = 1` and `candidateOverflow = 1`.
  - `getSpatialJoinProbeNodes` passes `bvhOverflow` through.
  - New `getLeafOverflowStatusNodes`, `@internal`, for traversals that have no candidate table.
- `spatial-join/spatial-join-candidates.ts` passes `bvhOverflow` through.
- `spatial-join/gpu-spatial-predicate-join.ts` wires the join's own or prepared BVH overflow. As a result, `candidateTruncatedCount` equals the left row count after a BVH overflow, and the anti join withholds every row.
- `spatial-join/gpu-point-in-polygon-join.ts` wires the BVH overflow the same way. Docs updated.
- **Result change, stated plainly:** with `how: 'anti'` and a BVH leaf overflow, `output.unmatched` used to list rows whose only matches were among the dropped right features. It is now empty, `candidateTruncated` is all ones, and `candidateTruncatedCount` equals the left row count.
  - The spec `modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-spatial-join-bvh-overflow.spec.ts` ("anti join withholds every row after a BVH leaf overflow", both the built and the prepared BVH) checks this against a CPU reference join (`joinWithOracle`).
  - I confirmed it fails on the old behaviour: with the BVH term disabled, the old output was `[1, 2, 3]` against a reference of `[2]` ("built row 1: expected [ 2 ] to include 1"). PIP, inner-join and nearest specs also failed. I then restored the code.
  - For PIP, the inner join and the nearest join, the new flags only add information: the rows they return do not change.

**Item 2, GPU-04: nearest-join flags in every mode.**
- In `spatial-join/gpu-nearest-feature-join.ts`, `output.candidateTruncated` and `status.candidateOverflow` are now accepted in neighbors mode and in great-circle nearest-feature mode. Their only cause there is a BVH leaf overflow, published by `getLeafOverflowStatusNodes` from the traversal's BVH.
- Planar nearest-feature mode also folds in the BVH overflow.
- The constructor rule that rejected the flags outside planar nearest-feature mode is removed. The length check moved to the constructor: one row per query, for any query kind.
- The spec covers planar, great-circle, neighbors and great-circle neighbors, each against a reference nearest list, at `leafCapacity` 4 (exact, no flags) and 2 (flags set; every wrong row is flagged).

**Item 3, STD-04: join consumers pass the flags through.** All new fields are optional.
- `zonal-statistics/gpu-zonal-statistics.ts`: new `output.candidateTruncated` (polygon features only, one packed `uint32` per point) and `output.status.candidateOverflow` (filled with 0 for `feature-rows`).
- `spatial-join/gpu-buffer-selection.ts`: new `output.candidateTruncated` (per point) and `output.status.candidateOverflow`.
- `line-density/gpu-line-length-per-polygon.ts`: new `output.status.candidateOverflow` (pair capacity or BVH; it leaves out the crossing bound). The output is per polygon, so there is no per-row flag.
- Specs force each overflow:
  - Zonal and buffer selection: candidate capacity 1, then leaf capacity 2. Expected counts, mask and flags are checked against the references.
  - Line length: `pairCapacity: 1` gives `candidateOverflow = 1`, and a crossing-only overflow gives 0, extending the existing spec.
- Rejection cases: messages start with the contributor ID and name the prop and the required length.

**Item 4, STD-04: `GPURegionStatistics` named status ports.**
- `region-statistics/region-statistics-types.ts` adds the new exported type `GPURegionStatisticsStatusPort` (exported from `region-statistics/index.ts`). `output.status` now also takes:
  - optional `overflow`: flag bits `selectionTruncated | regionTruncated`;
  - optional `candidateOverflow`: bit `candidatesTruncated`, not folded into `overflow`.
- The summary header is unchanged. Every word and flag bit is documented in the layout constants and in `docs/api/gpu-spatial-analysis/operators-statistics.md`.
- `region-statistics/gpu-region-statistics.ts` adds a one-thread status node that reads the flags word. The summary node already binds 8 buffers, so the new outputs could not go there.
- `region-statistics-readback.ts` is untouched. The new spec `modules/analysis/test/gpu-spatial-analysis/region-statistics/gpu-region-statistics-status.spec.ts` decodes the summary with `decodeGPURegionStatistics` and checks the ports against the flags for:
  - a lasso over capacity;
  - a selection over capacity;
  - grid candidate truncation.

**Item 5, STD-03:** the two bare asserts are replaced with `validateRequirement`:
- `spatial-join/gpu-spatial-join-prepared.ts` (build nodes must be unconditional);
- `spatial-join/gpu-spatial-predicate-join.ts` (robust point-in-polygon passes must be compute passes).

**Docs:**
- `docs/api/gpu-spatial-analysis/operators-joins.md`: anti join, candidate overflow, buffer selection.
- `docs/api/gpu-spatial-analysis/operators-statistics.md`: region summary words, zonal ports.
- `docs/api/gpu-spatial-analysis/operators-geometry.md`: line length.

## 2. Verified

- `npx vitest run --project headless` on `gpu-spatial-join-bvh-overflow.spec.ts` (new, 8 tests), `gpu-spatial-join-candidate-truncation.spec.ts`, `gpu-region-statistics-status.spec.ts` (new, 2), `gpu-line-length-per-polygon.spec.ts`, `gpu-zonal-statistics.spec.ts` and `gpu-buffer-selection.spec.ts`: 6 files, 68 passed, 0 failed (final run).
- `npx vitest run --project headless` on `gpu-spatial-predicate-join.spec.ts`, `gpu-point-in-polygon-join.spec.ts`, `gpu-nearest-feature-join.spec.ts`, `gpu-nearest-feature-join-neighbors.spec.ts`, `gpu-nearest-feature-join-geodesic.spec.ts` and `gpu-buffer-selection.spec.ts`: 6 files, 79 passed.
- `npx vitest run --project headless` on `gpu-spatial-predicate-join-prepared.spec.ts`, `spatial-join-sort.spec.ts` and `gpu-zonal-statistics.spec.ts`: 3 files, 41 passed.
- `npx vitest run --project headless` on `gpu-region-statistics.spec.ts` and `gpu-region-statistics-grid.spec.ts` (with the new status spec): 3 files, 29 passed.
- `npx vitest run --project node`: 6 files, 61 passed:
  - `region-statistics-readback.node.spec.ts`
  - `gpu-region-statistics.node.spec.ts`
  - `gpu-region-statistics-grid.node.spec.ts`
  - `zonal-statistics.node.spec.ts`
  - `gpu-buffer-selection.node.spec.ts`
  - `spatial-join.node.spec.ts`
- `npx vitest run --project node`: 6 files, 65 passed:
  - `secondary-spatial-context.node.spec.ts`
  - `lon-lat-polygon-status.node.spec.ts`
  - `gpu-line-length-per-polygon-lon-lat-status.node.spec.ts`
  - `gpu-region-statistics-lon-lat-status.node.spec.ts`
  - `zonal-statistics-core-limits.node.spec.ts`
  - `nearest-scratch-chunking.node.spec.ts`
- Old-behaviour check: I disabled the BVH term temporarily and ran the new spec. 4 of 5 tests failed as expected, then I restored the code (byte copy) and reran green.
- `node scripts/typecheck-status.mjs 'gpu-spatial-analysis/(spatial-join|zonal-statistics|line-density|region-statistics)/'`: 0 matching errors, cycle finished at 6:36:39 PM, after my last source edit.
  - The watcher appears to cover `src` only, so the new and edited specs are **not typechecked**.
- `yarn check-contributor-contract --all`: no `bare-assert` left in `spatial-join`, and "no new findings".
- `node scripts/check-upstream-first.mjs`: no violations.
- `npx biome check` on all touched files: clean.
- Not run: the kernel-audit spec (no limits changed), parity specs, and `yarn test` (forbidden).

## 3. Not done

- Nothing in items 1 to 5 is left undone.
- Precision limit: after a BVH overflow every probe row is flagged. A row is not narrowed by whether it overlaps a dropped feature's box, which would need a per-row scan of the dropped features.

## 4. Wave 2 requests

- No renames.
- Operator catalog (`ontology/spatial-analysis-operators.ts`, shared): `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`, `GPUBufferSelection`, `GPUZonalStatistics`, `GPULineLengthPerPolygon` and `GPURegionStatistics` have no override, so their catalog `status` is `[]`. They now publish `overflow` and `candidateOverflow` (and `uncertainCount` for PIP and zonal). Adding status overrides is optional; I did not touch the catalog.
- `GPUNearestFeatureWeights` (`spatial-join/gpu-nearest-feature-weights.ts`) still rejects `status.candidateOverflow`. It uses the neighbors traversal, so it could now publish a BVH overflow the same way, in a follow-up.

## 5. Roadmap

- GPU-04 (section 2.16): delete the row. A BVH overflow now marks every probe row, the anti join withholds them, and the nearest-join flags work in every mode.
- STD-04: in "Remaining", delete these three clauses:
  - "zonal statistics, buffer selection and line-length-per-polygon do not pass the join flags through";
  - "a BVH overflow in the joins still marks no rows";
  - "region statistics packs status in header words".

  Remaining text: "per-row flags have per-family bit meanings; terrain, raster and most statistics classes use ad hoc status".
- STD-03: the two `spatial-join` bare asserts are gone; subtract 2 from the count.

## 6. Noticed

- `GPUSpatialJoinCandidates` already folds the BVH overflow into `output.status.overflow`. It has no `candidateTruncated` output, so a caller of the bare candidate stage still cannot tell which rows were affected.
- `GPURegionStatistics` folds the grid candidate flag into `selectedRows.status.overflow` (publish node `overflowSources`), unlike the contract rule that `candidateOverflow` is not folded into `overflow`. I left it as is, because changing it would change existing results.
- `getSpatialJoinProbeNodes` returns early for zero points, so `candidateOverflow` is not written in that case. This predates my change.
