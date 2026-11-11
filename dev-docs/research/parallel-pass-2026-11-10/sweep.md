# Report: sweep

## 1. Done
1. STD-03 (bare assert, files outside other streams): 5 asserts converted, 4 files, all `assert` imports removed.
   - `gpu-spatial-analysis/geographic-distribution/grouped-sum-reduction.ts`: `validateCapacity` for groupCount (1..256) and output count (1..4).
   - `permutation-inference/gpu-local-permutation-replay.ts`: `validateRequirement` (replay needs a lowered test).
   - `permutation-inference/gpu-local-permutation-test.ts`: unreachable `assert(false)` is now a plain Error naming the unsupported statistic.
   - `study-window/gpu-study-window-preparation.ts`: `validateRequirement` for the nested-CPU-gate invariant; the redundant `assert(!node.condition)` removed.
   - Specs: new `test/gpu-spatial-analysis/geographic-distribution/grouped-sum-reduction.node.spec.ts` (4 rejections); `computation-pack/local-permutation-replay.node.spec.ts` rejection now checks the message prefix.
   - No `dataframe-analysis` bare asserts exist outside pair-statistics.
2. Registry: both `unregistered-contributor` findings gone. `GPUSnapRoundedNoding` is `@internal` and not exported from any entry point, so a public catalog operator fails four ontology specs. It is therefore recorded in `ontology/all-entry-points/operator-annotations.ts` (limitation text from the class). `GPUNestedInterpolationValidation` is now an explicit key in `spatial-analysis-operators.ts` (was a spread); annotation added with tier reference and limits rows 256, foldCount 16, innerFoldCount 8, candidates 16, neighborCount 32.
3. DOC-01: `missing-observations.compute.ts` (two strings: reconstruction and hole fill; the contributors always publish the current iterate, the scene hides it), `docs/api/gpu-network/network-movement-coverage.md` (NW05: 64 rows is the limit only for backup coverage and p-dispersion; capacitated p-median is bounded by 2^20 candidates, i.e. 20 rows at 2 facilities, 10 at 4, 5 at 16), `docs/api/variable-standardization.md` (each column must be a single chunk, multi-chunk rejected; removed chunk-boundary claims), `docs/api/gpu-spatial-analysis/operators-validity.md` (leafCapacity default, overflow behaviour, `candidateOverflow`).
4. WEB-HIW-07: `docs/api/gpu-spatial-analysis/concepts.md` now cites `describeGPUCommandGraph` from `@luma-contributors/analysis/computation-record`; `recipes.md` states the `{contributors, outputs, intermediates, status}` result.
5. UF-06: `docs/api/computation-views.md:32` uses `gpu-spatial-analysis.local-moran`.
6. UF-09: three-line SPDX header (form of `spatial-analysis-explorer/vite.config.ts`) added to the three files.
7. Report only, see below.

## 2. Verified
- `npx vitest run --project node` on grouped-sum, permutation-inference, geographic-distribution, local-permutation-replay, pair-statistics-study-window, analysis-core/contracts: 17 files, 120 tests pass.
- `.../ontology` directory: 32 of 32 pass (an earlier failure was my first, reverted, public-operator attempt).
- `yarn check-contributor-contract --all`: unregistered-contributor 0, bare-assert 48 to 37, no new findings.
- `npx biome check --write` on touched src, spec and example files: clean.
- `doc-snippets.node.spec.ts`: pass. `doc-parameters.node.spec.ts`: 1 FAIL, `GPURasterWalkingCost` is undocumented. Not caused by my edits (the stale list is empty); raster-analysis stream territory.
- `node scripts/typecheck-status.mjs 'grouped-sum|permutation-inference|study-window|ontology/'`: 0 matching errors (cycle finished 6:30:26 PM). The new spec file was not necessarily in that cycle; `typecheck-tests` was not run.

## 3. Not done
- Item 7 is report-only (below). Bare asserts in other-stream directories left alone (concave-hull, geometry-edit, etc.).

## 4. Wave 2 requests
- Item 7 (STD-13). Stale text is in `modules/analysis/src/gpu-spatial-analysis/ontology/kepler-analysis-catalog.ts` (not `modules/kepler`), the `dorling-cartogram` capability `limitation`, ~line 48: "at most 1024 rows and 256 iterations". Current (`gpu-dorling-cartogram.ts`): rows at most 65,536 (`MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS`, exact all-pairs relaxation, row-tiled), `maximumIterations` at most 256 (default 100), neighbor grid side at most 1024 (so 1024 is the grid side, not rows). Change "1024 rows" to "65,536 rows". `modules/kepler/src/analysis/dorling-cartogram.ts` already uses 65536 and 256.
- If `GPUSnapRoundedNoding` is exported later, add it to capability `segment-and-coverage-topology` and the operator table (I removed that attempt) and drop the comment in operator-annotations.

## 5. Roadmap
- STD-03: remove the permutation-inference, study-window and geographic-distribution files from the remaining list.
- STD-13: "`kepler-analysis-catalog.ts` still states the old Dorling limits" remains until the one-word edit above; `GPUSnapRoundedNoding` registered as internal only.
- DOC-01 items for the four named files: delete. UF-06, UF-09, WEB-HIW-07 (the two items named): delete.

## 6. Noticed
- `GPURasterWalkingCost` has a packer but no docs section (doc-parameters spec fails).
- `gpu-nested-interpolation-validation.ts:89` uses a bare `throw new Error(`${id}: shape mismatch`)`, no prop or limit named.
- The repo has a stale worktree copy under `.claude/worktrees/agent-*/docs/...` that grep for `operators-validity.md` finds.
