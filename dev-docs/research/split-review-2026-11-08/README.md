# Split review: luma `e0ed33791` vs luma-contributors `edf9ca10`

> Status on 2026-11-11, when this report was added to the repository. The report is kept as its
> session wrote it on 2026-11-08, and everything in it was checked against `edf9ca10` only. Known to
> have changed since, and nothing else was re-checked:
>
> - Finding 2: records for `sparse-resistance-connectivity-recipe` and
>   `raster-resistance-connectivity-recipe` exist again in `docs/lens-records`.
> - Finding 6: 140 of the excluded kepler specs were re-enabled on 2026-11-10; 131 remain excluded
>   (roadmap row KE-TESTS).
> - Finding 15: the broken relative links in `docs/api` (ten, not nine) were fixed on 2026-11-11.
> - The `@luma.gl` packages moved from `10.0.0-alpha.10` to `10.0.0-alpha.12`.
>
> The findings that are still open are tracked from roadmap section 15.

Session luma-contributors-86, run 2026-11-08. Read-only. Nothing in the checkout was edited.

## Scope

- **Old:** luma.gl `backup/map-contributors-pre-split-2026-11-07` (`e0ed33791`): `modules/graph/src` seeds (992 files, from `scripts/split/closure.json`), `modules/graph/test`, `modules/graph-lenses`, `modules/kepler`, and the four contributor example directories.
- **New:** luma-contributors `edf9ca10` (`origin/upstream-first`), extracted with `git archive`.
- **Checked only against `edf9ca10`.** Nothing here was re-checked against the current working tree, which other sessions have changed since (`website/`, the lens JSON, 10 `modules/analysis/src` files).
- **Method:** files were matched by basename plus how many of the old file's non-import lines appear in the new one; exported names were diffed with a regex. Four read-only reviewer agents then classified every gap and diffed every file that scored below 0.9. "Verified" below means I re-checked the claim against the extracted files myself. "Reviewer-read" means only an agent's reading supports it.
- **Line references** like "line 43" are line numbers in `dev-docs/upstream-workarounds.md`.

## Gate run (2026-11-08, 08:17–08:42; stale)

`check-upstream-first` passed with no violations. `typecheck` (src) passed. `verify-package-exports` verified 17 entries. `lint` exited 0 with 63 warnings. `test-node`: 716 files, 7,907 tests passed, 375 skipped. Headless, 4 shards: 692 files, 9,142 tests passed, 799 skipped, 0 failed. The headless shards ran on top of uncommitted peer edits to 10 `modules/analysis/src` files (timestamped 08:21). The skips were not broken down, and I did not confirm whether headless used real WebGPU or a software adapter. `typecheck-tests` was not run (1,741 known errors).

## Coverage summary

| Area | Gap | Classification |
|---|---|---|
| Analysis source | 33 of 992 seeds with no same-basename file | 27 documented drops, 5 replaced, 1 undocumented loss (`analysis-core/linear-dispatch.ts`, `getLinearDispatchCount`: an unused helper) |
| Analysis exported symbols | 126 of 4,696 missing | about 117 documented, about 8 replaced, 1 undocumented loss |
| Kepler source symbols | 0 of 857 missing | complete |
| Lenses exported symbols | 72 of 1,349 missing | all belong to dropped packs or the recording downgrade |
| Analysis tests | 623 old files with no same-basename file | about 577 test luma-owned code, about 36 documented, 5 renamed, 5 undocumented losses |
| Examples | 6 of 2,881 files missing | 2 documented (explorer tile-LOD), 4 undocumented |

## Findings: undocumented drops

1. **The gpu-graph lens pack is gone; only SCC is recorded.** **Verified.**
   - `modules/graph-lenses/src/computation-lenses/scenarios/gpu-graph/` held 48 files and 14 scenarios. There is no such directory in the new repo.
   - The register (`upstream-workarounds.md` line 32; `upstream-first-pending.md` line 14) mentions only SCC.
   - Alpha.10 does declare `GPUGraphAlgorithms{PageRank,CoreNumber,LabelPropagation,ConnectedComponents,BreadthFirstSearch}` (`node_modules/@luma.gl/graph/dist`), and `examples/deck/spatial-analysis-showcase/src/scenes/flows/airline-network-graph.ts` already uses those classes. So the pack is probably restorable; that it runs on alpha.10 is untested.
   - `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts:37-51` filters "unregistered scenario" failures for dropped scenarios (reviewer-read).
2. **Two resistance lens scenarios are gone and recorded nowhere.** `gpu-spatial-analysis.sparse-resistance-connectivity-recipe` and `raster-resistance-connectivity-recipe`, together with their `.example.ts` files. **Reviewer-read.**
3. **The register's lens counts are off.** The dataframe query lens count is 9, not 8, because `data-frame-group-by-query` shares a file with `grouped-aggregation-query`. The gpu-core count of 25 is correct. **Reviewer-read.**
4. **Four showcase files are gone and recorded nowhere.** `examples/deck/spatial-analysis-showcase/src/scenes/networks/street-centrality.{compute,scene}.ts`, `scenes/raster/habitat-connectivity-circuit.ts` and `scripts/check-habitat-circuit-browser.mjs`. **Verified missing; that nothing records them is reviewer-read.**
   - `habitat-connectivity.{compute,scene}.ts` lost its circuit layer.
   - The scene registry is a glob, so nothing dangles.
   - The sinusoidal-projection scene lost its `wrapLongitude` branch-cut step, and `PROBE_COUNT` went from 8 to 4. This follows from upstream: the published `gpu-sinusoidal-projection.d.ts` has no `wrapLongitude`.
5. **Five kepler specs are gone and appear in no exclusion list or dev-doc.** **Verified absent from `test/kepler-compute-example-specs.json`.**
   - The specs are `modules/kepler/test/{h3-source.node,host-geometry,kepler-cluster.node,point-label-source.node,resident-heatmap}.spec.ts`.
   - Four import parked `examples/deck/kepler-compute/src/*` files.
   - `kepler-cluster.node` imports only `src/cluster/*`, `@kepler.gl/deckgl-layers` and `supercluster`, so it should be restorable as is (imports read, not run).
6. **The 271-spec kepler exclusion is too broad.** **Reviewer-read, static analysis only.**
   - Six excluded specs need only `node:` modules or example files: `acceptance-report`, `analysis4-policy-panels`, `globe-split-matrix`, `host-panel-accessibility`, `resident-performance` and one other. `resident-table-column-tools` additionally needs react and `@kepler.gl/components`.
   - Up to 130 excluded specs never reach `gpu-layers`, `@deck.gl/*` or `engine/arrow`.
7. **Undocumented test losses.** **Reviewer-read.**
   - `modules/graph/test/arrow/arrow-analytics-normalization.{node.spec,spec}.ts` held 7 tests of code that is now local (`modules/analysis/src/analysis-core/graph-support/`). Only a thin kepler spec still covers it.
   - `gpu-dataframe/pair-statistics/variogram-reference.node.spec.ts` (the scikit-gstat check). Its oracle remains but is no longer checked.
   - `gpu-dataframe/column-profile/parity-wp6-column-fixtures.parity.node.spec.ts` and `gpu-dataframe/parity-wp6-round8-fixtures.parity.node.spec.ts`. Their fixtures remain.
   - `utils/benchmark-settings.node.spec.ts`. The helper is still used.
   - Two bundle-purity tests ("keeps math.gl and proj4 out of …").

## Findings: behaviour changes

8. **EC03 is a smaller reimplementation, not an equal one.** **Verified.**
   - Old: `modules/graph/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance.ts:28-41`. New: `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity.ts:24-25, 74-75`.
   - Limits and defaults:

     | | Old | New |
     |---|---|---|
     | Max vertices | 16,777,216 | 1,048,576 |
     | Max edges | about 536M | 2,097,152 |
     | Default `maximumIterations` | 256 | 1024 |
     | Default `tolerance` | 1e-5 | 1e-4 |

   - Reviewer-read: component discovery moved from exact union-find to bounded label propagation and can report "undetermined" (`reachable == 2`). The props, return shape and status layout all changed.
   - `upstream-first-pending.md:47` and `docs/api/gpu-spatial-analysis/landscape-ecology.md:114` say `addRasterResistanceConnectivityRecipe` was dropped. The register entry for EC03 (line 55) does not mention the lower limits, and `CHANGELOG.md` calls the recipe "restored".
   - The upstream golden test became `sparse-resistance-connectivity.spec.ts:70`: tolerance 1e-6 (was 1e-5), 64 iterations (was 32), and the iteration-count assertion was removed.
9. **`GPUNetworkAnalyticsColumns` lost its `pageRank`, `coreNumber` and `communities` props.** **Verified** (old file lines 87-116; the new file has one mention, in its doc comment).
   - The docs are accurate: `docs/api/gpu-network/README.md:37-40` tells callers to use the luma.gl algorithms directly.
   - The register's line 24 reads as if nothing was lost.
   - Six of 7 tests in `gpu-network-analytics-columns.spec.ts` were removed with it.
   - `examples/.../networks-analysis.md:44` may still advertise the props; not checked.
10. **Lens records have less in them.** **Reviewer-read**, comparing all 410 records in `docs/lens-records` (old 462).
    - Source text went from 405 records to 0. No bindings or loop annotations remain.
    - Formulas were removed from 81 records and reduced in 102.
    - Iteration key frames shrank: edge-bundling 13→9, network-reachability 26→17, cost-distance 15→9, cost-distance-path 30→24, terrain-flow 18→12, network-structure 7→6.
    - The UI degrades gracefully: `kernel-source.ts` returns `status: 'unavailable'`.
11. **Crossfilter `kind: 'keys'` clauses on an external dimension now throw.** The lenses' `gpu-crossfilter-selection-backend.ts` lost `writeMask`, `setExternalActive` and `invalidate`. The register records the crossfilter limit but not the throw. **Reviewer-read.**
12. **Validation got weaker, and the register doesn't say so.** **Reviewer-read.**
    - `raster-analysis/raster-band-utils.ts:266` `validateRasterBand` lost the validity-view checks, the texture checks and the `noDataValue` check, and its length check went from `===` to `<`.
    - `pointcloud-kernel-nodes.ts` lost its disjoint-binding and chunk-view checks.
    - `gpu-terrain-rgb-decode.ts` now uses a bare size check with no offset alignment.
    - `shared/bin-index-wgsl.ts` lost its overflow-safe branch, which only matters for extents near ±3e38.
13. **Weakened tests.** **Reviewer-read.**
    - The four `gpu-network/computation-pack/*-recording` specs became `*-composition` and now only assert that the node graph is non-empty.
    - `projection-frame.node.spec.ts` lost 4 tests and `gpu-project.spec.ts` lost 1. These follow from the documented `targetFrame` limit, but the lost tests aren't named.

## Findings: docs and register

14. **`upstream-workarounds.md` errors.**
    - Line 43 says `offsetsWithinInput` and `implementation` are "accepted and ignored". Neither exists in `sorted-segment-sums.ts`, and the new `reduction` option isn't mentioned. **Verified.**
    - Line 41 says no analysis contributor accepts a recorder, but `computation-record/{gpu-transfer-mask,gpu-snapshot-tap,describe-command-graph}.ts` take `recorder?`. That depends on how "contributor" is meant; the wording should be clarified. **Reviewer-read.**
    - Line 53 says `queryPositions`. The published `GPUPairwisePointInPolygon` uses `points` and `polygonPositions`. **Reviewer-read.**
    - Line 50 cites `gpu-table-inspector.ts:539`; it is line 540. **Reviewer-read.**
15. **Nine broken relative links in `docs/api`** (one `../` too many after the pages moved up a level). Examples are `docs/api/gpu-network/alternative-routes.md:49`, `docs/api/gpu-network/network-movement-coverage.md` (6 links) and `docs/api/gpu-spatial-analysis/statistical-expansion.md`. **Reviewer-read.**
16. **The showcase in-app reference silently drops 7 pages.** `examples/deck/spatial-analysis-showcase/src/docs/reference-index.ts:16-22` globs `docs/api/gpu-graph*.md`, which doesn't exist, and `buildIndex` skips the missing pages. **Reviewer-read.**
17. **`docs/api/gpu-spatial-analysis/spatial-frames.md:133`** still says `GPUProjection` accepts `targetFrame`. **Reviewer-read.**
18. **`modules/lenses/src/computation-lenses/data/README.md`** still lists the deleted `g-nl-rail-graph.json` (line 17) and `k-core.json` (line 35). `s3-chicago-area-roads.json` isn't in `THIRD-PARTY-DATA.md`, though it wasn't in the old README either. **Reviewer-read.**

## Suspicious, unconfirmed

- The `repository.url` in all three package manifests is `visgl/luma-contributors`, but the remote is `BertCh/luma-contributors`.
- `examples/deck/kepler-compute/package.json` pins `@luma-contributors/analysis` at `10.0.0-alpha.10`; the package is at `0.1.0-alpha.0`. The example is parked.
- `analysis-core/network-topology.ts:244` builds a stand-in for a published `GPUGraphAlgorithmsTopology` with `as unknown as`. This is unchanged from the old code.
- `masked-weak-components.ts` re-implements the published connected-components algorithm.
- `kepler/src/session/resident-dataset.ts:347` reads Arrow's private `_nullCount`.
- `website/lens-staging/client-modules/ladder-lens.ts:100-101` imports `@luma.gl/graph-lenses/*`. That staging folder isn't built.

## Checked and fine

- No `.skip`, `.only` or `.todo` in either tree. `skipIf` counts in the contributor test directories are 28 old and 28 new.
- Every spec matches a vitest project glob.
- The only oracle fixtures dropped are the 8 documented ones.
- No `export *` collisions.
- `dev-docs/README.md` links resolve.
- `analysis` is publishable; `lenses` and `kepler` are private.
- Every file with an SPDX license line also has an `SPDX-FileCopyrightText` line.
- All 70 showcase dataset directories are named in `THIRD-PARTY-DATA.md`.

## Not covered

- GPU numerical equivalence beyond the existing tests, and a browser smoke test.
- Whether the 130 candidate kepler specs or the gpu-graph lenses actually run.
- The `website/` tree, beyond SPDX headers and links.
- Python and data scripts.
- Content diffs of the docs pages that were carried over.
- Anything in the working tree after `edf9ca10`.
