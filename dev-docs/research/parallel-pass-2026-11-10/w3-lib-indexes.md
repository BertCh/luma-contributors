# w3-lib-indexes report

## 1. Done
Item 1, `GPUNeighborSearch` cross-join queries outside the bounds (fixed, correct result, not a status port):
- Cause: query validity used the target rule (finite and inside `bounds`). The cell lookups and cell bounds already tolerate a query outside the lattice, so only the gate and the query-order index were wrong.
- `neighbor-search/neighbor-search-kernels.ts`: new WGSL `isCrossJoinQueryInBounds`, `isCrossJoinQueryValid`, `isCrossJoinQueryTimeInBounds`; planar = finite only, great-circle keeps the bounds rule. `getCellColumn`/`getCellRow` clamp explicitly (no reliance on float-to-u32 saturation).
- `neighbor-search/gpu-neighbor-search.ts`: cross joins use them in the query-valid pass, the query grid positions (unit position clamped to [0,1] so the grid index does not skip the query; time clamped at 0) and the `invalidCount` pass. Self joins unchanged (their queries are targets). Prop/class docs and `neighbor-search-parameters.ts` `bounds` doc updated; `docs/api/gpu-spatial-analysis/operators-weights.md` (3 passages).
- Why exact: clamping to the lattice box is a projection; for a target t in the box, |t - q'| <= |t - q|, so the 3x3 / ring neighbourhood of the clamped cell holds every target within the radius, and all gap bounds are measured from the true position (conservative). Distances use the true position. BVH path was already exact.
- Reproduced first: with the fix switched off, the new spec fails (`knn 1 offsets`: all outside rows empty, invalidCount 4 not 3).
- Specs: new `test/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search-outside-queries.spec.ts` (grid and bvh; kNN k=1,7, kNN with radius, radius 6 and 40; queries outside every side and corner, 1e6 away, on the edge, a tie at equal distance; invalidCount stays 0; non-finite queries and out-of-bounds targets still counted). New time-axis case in `point-neighbor-grid-unification.spec.ts` (queries outside bounds and outside timeBounds, bucket count default and 16, radius and kNN). Oracle `neighbor-search-oracle.ts` and the brute-force reference in that spec now treat a cross-join query as valid when finite. Comment edit in `gpu-neighbor-search.spec.ts`.
- Fixture evidence: `test/oracles/geometry-neighbor-search-clustered.py` now treats a cross-join query as valid when finite; regenerated `fixtures/geometry-neighbor-search-clustered.json` with /usr/bin/python3 (scipy 1.13.1, numpy 2.0.2; the old file recorded scipy 1.16.3). Compared old and new: `clustered` and `collinear` scenes identical, `cross` differs only in `knn` (the query at (-1, 4) is now searched).

Item 2, island attachment indexed nearest (`spatial-weights/gpu-island-attachment.ts`):
- New prop `nearestSearch: 'auto' | 'scan' | 'grid'` (validated at construction, message `<id> nearestSearch must be ...`), exported constant `GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT = 4096` (same cut-over as local kriging and the segment BVH), exported type; `spatial-weights/index.ts` exports updated.
- Grid path: islands are the query rows (via `queryMask`) of a `GPUNeighborSearch` kNN cross join (k = 16, self included) over all rows, parameters from `createNeighborSearchDerivedParameterNodes` (extent of finite rows). The `nearest` kernel then picks among the candidates with the scan's own comparison (now a shared WGSL function `isCloserDifference`), ties to the lowest row, self skipped. The list proves the answer when it holds fewer than 16 entries (complete) or its farthest entry exceeds the chosen distance by more than 1e-5 relative; otherwise (empty list from an unusable lattice, overflow, near-ties wider than the list) that island runs the full scan in the same kernel. So the result equals the scan, not just usually.
- Tie rule: nearest finite other representative by the scan's scaled comparison; equal distance goes to the lowest row. Same on both paths.
- Spec: new `test/gpu-spatial-analysis/spatial-weights/island-attachment-index.spec.ts` (placed outside `spatial-regression/**`): 6,000 shuffled rows on an integer lattice (ties and duplicates, compared to a CPU reference, scan and auto too), real-valued with ulp-close steps, 80 coincident/tied representatives, NaN/Infinity representatives and islands, coordinates +-1.5e38 (squared distances overflow), no island and one island, 60,000 rows, constant check.
- Mutation check: disabling the rescan makes the +-1.5e38 case fail, so that path is exercised. Disabling it does not fail the tie case: the grid's (d^2, id) order already keeps the lowest ids of a tie.
- Timing (not asserted): 60,000 shuffled rows, 25 percent islands, Apple GPU, second run, readback included: grid 7 ms, scan 52 to 53 ms (first run 782 ms vs 94 ms, pipeline compile).
- Docs: `docs/api/gpu-spatial-analysis/coverage-statistics.md` paragraph. Kernel audit: two new cases `maximum-nearest-search-scan|grid` in `test/analysis-core/kernel-audit/cases-coverage-statistics.ts` (the existing `maximum-reciprocal-*` cases now audit the grid path through `auto`). No operator-catalog or lens descriptor change needed (props are not listed there).

## 2. Verified
- `npx vitest run --project headless` on `neighbor-search/` (7 files passed, 1 skipped; 36 tests passed), on `point-neighbor-grid-unification.spec.ts` (17/17), `island-attachment-index.spec.ts` + `spatial-regression/island-attachment-merge.spec.ts` (12/12).
- `--project node`: `neighbor-search/` + `spatial-weights-libpysal.node.spec.ts` (30/30); `kernel-audit/` + `coverage-statistics.node.spec.ts` (33/33).
- `node scripts/typecheck-status.mjs 'spatial-weights|neighbor-search'`: 0 errors. Scratch tsconfig over the 5 touched spec/oracle/audit files: no errors.
- `node scripts/check-upstream-first.mjs`: no violations. `yarn check-contributor-contract --all`: no finding for island attachment or neighbor-search. `npx biome check --write` on my files: clean.
- Wider consumer regression, headless (spatial-interpolation, space-time-tests, emerging-hot-spots, spatial-weights, spatial-clustering, outlier-detection, value-cube composition): 69 files passed, 1 skipped; 1,424 tests passed, 4 skipped. `--project node` `short-distance-warnings.node.spec.ts`: 12/12.
- Not run (forbidden): `yarn test`, build, verify-package-exports.

## 3. Not done
- Great-circle cross-join queries outside `bounds` are still empty and counted by `invalidCount` (the longitude wrap is defined by the bounds); documented.
- The island-attachment specs in `spatial-regression/**` were not edited (other session). The kernel-audit `maximum-reciprocal-*` cases now take the grid path.
- Rows limit of island attachment stays 65,535.

## 4. Work list for others
- None in `website/**` or `examples/**` need a change: grep found no use of `nearestSearch`, and cross-join users there that pass queries outside bounds now simply get neighbours.
- `website`/`examples` consumers of `GPUIslandAttachment` above 4,096 rows now use the grid path through `auto` (same result).

## 5. Roadmap (GPU-05)
Remove from "Remaining": the island-attachment nearest search and `GPUNeighborSearch` cross-join queries outside its bounds. Add to Done: island attachment `nearestSearch` (`'auto'` grid above 4,096 rows, exact rescan fallback; 7 ms vs 52 ms at 60,000 rows); planar cross-join queries are no longer clipped by `bounds`. Still remaining: MGWR local fits; dense default of cross-variogram and marked statistics; kepler snapping callers; great-circle cross-join queries outside bounds stay excluded.

## 6. Noticed
- `/usr/bin/python3 -I` cannot import numpy (user site is ignored); the generator docstring says `python3 -I`. Run it without `-I`.
- The `rows <= 65535` cap of island attachment and `GPUNeighborSearch` k-list memory (rows x 16 x 4 bytes x 2) are the only scaling limits of the grid path.
