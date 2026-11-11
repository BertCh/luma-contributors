# w2-core-surface report (STD-09, breaking half)

## 1. Done

**Item 1: rename the core `validatePackedView` to `validateSinglePackedView`.**
- The published `validatePackedView` (`@luma.gl/graph/gpu-core`, `graph-data-view-utils.d.ts`, tagged `@internal` upstream) takes a `GraphDataView` and checks format, stride, row length and 4-byte offset. The local wrapper also accepts a `GraphVectorView`, rejects a chunked vector with `"<name> must be one packed view"`, then runs the published check. The new name says that: one (unchunked) packed view. The library review (`code-core-dataframe.md:212`) proposed the same name.
- How it was done: a TypeScript-AST script classified all 409 files that use the identifier by their import statement. 380 files bind the local function (every one of them imports it straight from `analysis-core/kernels/gpu-contributor-utils`, with no aliases and no barrel re-export), so the script replaced the whole identifier in those files only (2025 occurrences; the only non-code hit was a test title). The definition was edited by hand, its JSDoc now says how it differs, and no alias is left. 38 files were then reformatted by biome to wrap the longer calls; a hunk-by-hunk check showed every reformatted hunk contains the renamed call.
- 28 files import the PUBLISHED one from `@luma.gl/graph/gpu-core` and were left alone: modules/analysis/src: gpu-terrain/point-horizon/point-horizon-march.ts, raster-analysis/raster-band-utils.ts, raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid.ts, raster-analysis/flow-texture/gpu-line-integral-convolution.ts, raster-analysis/raster-sampling/raster-sampling-utils.ts, raster-analysis/flow-texture/gpu-streamlines.ts, raster-analysis/isolines/isolines-utils.ts, raster-analysis/particle-advection/gpu-particle-advection.ts, analysis-core/spherical-cell-area.ts, gpu-network/network-accessibility/gpu-network-accessibility.ts, dataframe-analysis/time-window-filter/gpu-time-window-filter.ts, gpu-spatial-analysis/spatial-autocorrelation/gpu-hot-spot-analysis.ts, gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts, pointcloud-analysis/pointcloud-kernel-nodes.ts; modules/kepler/src: analysis/{recipe-bindings,point-weights,point-idw,dorling-cartogram,point-multivariate-local-geary,point-kriging,point-dbscan,weighted-quantile-lisa,point-local-moran,ordinary-least-squares,polygon-weights,point-bivariate-local-moran}.ts, aggregation/kepler-group-by.ts; examples/deck/kepler-compute/src/resident-calendar-time.ts. The definition file imports the published one as `validatePackedGraphView`.
- Docs: `dev-docs/upstream-first-foundation.md` (2 mentions), `dev-docs/upstream-workarounds.md` (wrapper row: now names the wrapper and records the 2026-11-11 rename). The contract document's helper list never named this helper, and no entry barrel exports it (it is not in `analysis-core/index.ts`), so the contract text is unchanged. Exporting it from `analysis-core` would be an additive decision for later (see Noticed). No `docs/api` page named it.

**Item 2: internal-export findings 13 to 0** (the checker reported 13 findings today, for 12 symbols, not 15).
- Exported only to analysis code and tests (tests import them by source path), so they were removed from the barrels:
  - `geospatial-analysis`: `getGPUProjectionSpatialContext`, `getGPUProjectionFrameOrigin` (removed `export * from './projection-spatial-context'`; it exports nothing else).
  - `pointcloud-analysis`: `createPointCloudGridIndex`, `validatePointCloudSearchGrid` (`export *` replaced by explicit `getPointCloudRowDirection` + type `PointCloudGrid`); `getPointCloudCellSource`, `validatePointCloudGrid` (`export *` of `gpu-pointcloud-surfaces` replaced by its 8 values and 3 types); `getPlaneMomentsSource`, `getPlaneSolveSource` (explicit line removed).
  - `raster-analysis`: `getRasterMedialAxisNodes`, `getRasterWatershedForestNodes` (their `export *` lines removed; each module exports only that function).
  - `analysis-core` and `gpu-spatial-analysis`: `GPUStudyWindowSlabViews` (removed from `analysis-core/contracts/index.ts`; it is the type of the `@internal` `slabs` field).
- Public in practice, so the `@internal` tag was removed and the type documented: `SpatialSortCurve` (`gpu-spatial-analysis/spatial-join/spatial-join-passes.ts`). It is the type of the public `spatialSortCurve` prop of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` and `GPUBufferSelection`, `docs/api/gpu-spatial-analysis/operators-joins.md:538` names it, and the website inventory lists it as the prop type.
- Export-list comparison: a TS-checker script listed every exported name, its kind and its declaring file for all 12 entry barrels before and after. The diff is exactly the 13 removed (barrel, name) pairs above: analysis-core -1, geospatial-analysis -2, gpu-spatial-analysis -1, pointcloud-analysis -7, raster-analysis -2. No other line changed in any barrel. A second pass using `ts.getJSDocTags` finds no `@internal` declaration exported from any barrel (it was checked against a known positive). Script: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/w2core/list-exports.cjs`; lists: `.../w2core/exports-before`, `.../w2core/exports-after2`, `.../w2core/exports-after3`.

**Item 3: domain barrels no longer re-export the contract.**
- `gpu-spatial-analysis/index.ts`: removed `export * from '../analysis-core/contracts/index'` and `GPUParameterBuffer`, `GPUParameterBufferProps`, `GPUParameterFormat`. It still exports `getGPUGridIndexMaximumCellCount` (analysis-core does not export it) and `../analysis-core/ring-assembly/index` (`GPUSegmentRingAssembly` is a contributor, not contract).
- `gpu-network/index.ts`, `gpu-terrain/index.ts`: removed `GPUCompactOutput`, `GPUUint32Rows`, `GPUParameterBuffer`, `GPUParameterBufferProps`, `GPUParameterFormat`. gpu-terrain keeps `GPURasterCellSizeMode` and `GPURasterRowDirection`, which analysis-core does not export.
- Removed names: 173 from gpu-spatial-analysis, 5 from gpu-network, 5 from gpu-terrain. All 183 are exported by `analysis-core` with the same declaring file, checked line by line. List: `.../w2core/item3-removed.txt`.
- `analysis-core/index.ts`: the comment now says it is the only entry point that exports the contract.
- Consumers repointed with an AST script that splits each import or export declaration and moves only the names that left: 191 files (lenses/src 170, kepler/src 20, analysis/test 2: kernel-audit `cases-statistics.ts` and `cases-geometry.ts`). analysis/src had no barrel imports of these names. Four lens dynamic imports were fixed by hand: `inverse-distance-weighting`, `kriging`, `local-frame-projection` and `spatial-scan-statistic` `.lens.ts` now import `createGPULocalFrame` from `@luma-contributors/analysis/analysis-core`. Duplicate imports were merged in `kepler/src/analysis/recipe-bindings.ts` and `lenses/.../s1-regression-model.ts`. All 25 namespace imports (`import * as spatial ...`) were checked: none uses a moved name.
- Operator ontology: `GPUParameterBuffer` (a catalogued `helper`) now declares `package: '@luma-contributors/analysis/analysis-core'` (`gpu-spatial-analysis/ontology/spatial-analysis-operators.ts`). Specs updated to that meaning: `spatial-analysis-package-ownership`, `spatial-analysis-ontology` (public set = spatial barrel + `GPUTerrainPeakProminence` + `GPUParameterBuffer`), `spatial-analysis-queries`, `analysis-operator-catalog` (the package set now includes analysis-core). The operator count stays 261.
- Docs moved to analysis-core: `docs/api/gpu-spatial-analysis/README.md`, `gpu-network/README.md`, `gpu-terrain/README.md`, `gpu-dataframe-analysis.md`, `gpu-raster/README.md`, `gpu-raster/operations-analysis.md` (`GPUParameterBuffer`), and `gpu-spatial-analysis/spatial-frames.md` (`createGPULocalFrame`, `createGPUSpatialFrameFromCRS`).
- Resolution: `modules/analysis/package.json` already has `./analysis-core` (types, import, require), and `verify-package-exports.mjs` iterates every non-wildcard key, so it covers analysis-core. Lenses (177 files), kepler (35) and the website (23) already imported `@luma-contributors/analysis/analysis-core`, through the same tsconfig path and vitest resolution as the other entries.

**Item 4**: not started, as the brief asks; the roadmap text is in section 5.

## 2. Verified
- `node scripts/typecheck-status.mjs 'modules/'`: 1 error, `raster-analysis/cost-distance/gpu-cost-distance.ts(21,3)`, which still reports the old name at line 21. The file on disk has `validateSinglePackedView` there. The shared watcher stopped seeing this file: it was rewritten at 00:27:04 and a same-content rewrite produced no "File change detected". A focused TS-API check of the file shows 0 diagnostics, and the fresh `typecheck-tests` run below does not report it. Every other cycle is clean.
- `yarn check-contributor-contract --all`: internal-export 0, luma-reexport 0; no new findings. `node scripts/check-upstream-first.mjs`: no violations.
- `npx vitest run --project node modules/analysis/test` (whole analysis node project): 650 files, 7010 passed, 1 failed, 1 skipped. The failure is `gpu-point-pattern-indices.node.spec.ts > legacy rectangle WGSL stays byte-identical`, a WGSL hash changed by another session's `isFiniteFloat` to `isFiniteValue` edit in that file. My edits there touch only the validator name.
- Specific node specs, all passing after the fixes: analysis-core (layering, vocabulary, kernel-audit*, doc-snippets: 163 snippets, runtime-boundaries, no-bare-directory-imports, upstream-first-guard), ontology (4 specs), gpu-terrain barrel, contributor-validators, gpu-project.
- `npx vitest run --project node modules/kepler/test modules/lenses/test`: 279 files, 2405 passed, 7 failed, 385 skipped. None of the 7 failures is mine:
  - lens-twins: stale twins for linear-referencing and rectangle-clip. Twins come from record files; the example region excludes imports.
  - record-structure: 4 gpu-network records whose node ids no longer match.
  - s3-geometry-pack (`latitude2` WGSL symbol) and statistics-regions-pack (`getSsd`): WGSL text changed elsewhere.
- Headless, one spec per entry point, 12 files, all passing:
  - analysis-core: rts05-compositing
  - computation-record: computation-record-capture
  - dataframe-analysis: space-time-hot-spots-recipe
  - geospatial-analysis: gpu-point-buffer
  - gpu-spatial-analysis: polygon-buffer
  - gpu-terrain: terrain-peak-prominence
  - pointcloud-analysis: m3c2-golden and pointcloud-grid-index
  - raster-analysis: gpu-terrain-flow-field and parity-wp5
  - testing: lenses global-coordinates
  - gpu-network: kepler host-dbscan, which loads kepler/src/analysis/catalog-gpu-network and so the gpu-network barrel
  - The root entry was not changed and was not run separately.
- `NODE_OPTIONS=--max-old-space-size=8192 node scripts/typecheck-tests.mjs` (once): 4 errors, none mine:
  - 2 in `gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve.ts:593` (missing `ringOffsets`): the other agent's ring-assembly work.
  - `test/analysis-core/kernel-audit/cases-geometry.ts:4040` (`lineCount` not in the status port type): a status-port change from another stream.
  - `test/gpu-spatial-analysis/trajectory-interpolation/trajectory-twin-heading-signed-zero.spec.ts:165` (`overflow` vs `candidateOverflow`).
- `npx biome check` on every file I touched: clean, apart from 2 existing `noExplicitAny` warnings in `gpu-network/network-coarsening/gpu-network-coarsening.ts`.
- Not run: `yarn verify-package-exports` (needs `yarn build`), headless runs of the 380 renamed files (they get the same identifier from the same module; covered by the typecheck), and the website or examples.

## 3. Not done
- Item 4 (fold `gpu-spatial-analysis/shared` and the raster iteration gate into core; move `regression-row-tiles.ts` and `line-merge/list-ranking.ts`): out of scope, left for later.
- `validateSinglePackedView` is not exported from `analysis-core` (it never was). Third-party contributors still have no named packed-view validator. Decide whether to add it (additive).

## 4. Website and examples work list
- Item 1: none. examples/deck/kepler-compute/src/resident-calendar-time.ts uses the published luma.gl function and stays as it is.
- Item 2: none. Only `SpatialSortCurve` appears (website inventory), and it is still exported.
- Item 3: in every file and line below, move the listed names out of the `@luma-contributors/analysis/gpu-spatial-analysis|gpu-network|gpu-terrain` import into an import from `@luma-contributors/analysis/analysis-core`. Keep the other names where they are; "(whole statement)" means only the specifier changes. Then regenerate `website/src/catalog/generated/inventory.json`, where GPUParameterBuffer's `importPath` becomes `@luma-contributors/analysis/analysis-core`, because the ontology now declares that package.
- `website/src/gpu/runs/entries/geometry-processing/local-frame-projection.ts:8`: move `createGPULocalFrame` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/recipes-trajectories/space-time-hot-spots-recipe.ts:9`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/recipes-trajectories/hot-spot-analysis-recipe.ts:6`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/recipes-trajectories/spatial-regression-recipe.ts:8`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/recipes-trajectories/rate-cluster-map-recipe.ts:8`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/geometry-construction-overlay/parameter-buffer.ts:10`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/space-time-models/space-time-cross-correlation.ts:11`: move `type GPUMultipleTestingMethod` to `@luma-contributors/analysis/analysis-core`
- `website/src/gpu/runs/entries/space-time-models/_lib/gtwr-data.ts:9`: move `GPUTemporalContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `website/src/components/meta/pages/anatomy/kernel-density.example.ts:6`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/components/meta/pages/scheduling/compile-once.example.ts:6`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `website/src/components/meta/pages/workflows/drainage-network.example.ts:7`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/density-mode.ts:20`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-explorer/spatial-analysis-data.ts:5`: move `createGPULocalFrame, type GPULocalFrame` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-explorer/modes/recipes-kit.ts:22`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-explorer/modes/simplification-mode.ts:23`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/point-pattern-mode.ts:56`: move `GPU_STUDY_WINDOW_SUMMARY_LENGTH` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/dot-density-mode.ts:31`: move `type GPULocalFrame` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/rates-mode.ts:26`: move `type GPUInferenceAlternative` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/recipes-scenes-cells.ts:16`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:24`: move `type GPULocalFrame` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/regression-result-readouts.ts:12`: move `GPURegressionModelResult` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-explorer/modes/clusters-mode.ts:14`: move `type GPULocalFrame, type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-explorer/modes/cells-mode.ts:47`: move `type GPULocalFrame` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-shared/resources.ts:7`: move `GPUParameterBuffer, type GPUParameterFormat` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-shared/hot-spots-analysis.ts:21`: move `type GPUInferenceAlternative, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/cells/b8-quadbin-layer.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.compute.ts:6`: move `type GPUInferenceAlternative, type GPUInferenceMetadata` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/cells/cell-pyramid.scene.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/cells/spherical-observation-support.compute.ts:9`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/cells/h3-aggregation.compute.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/hot-spots.compute.ts:7`: move `createGPULocalFrame, GPU_WGS84_SEMI_MAJOR_AXIS_METERS, GPU_EARTH_MEAN_RADIUS_METERS, type GPUInferenceAlternative, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-hotspot-stats.ts:6`: move `type GPUSpatialContext, type GPUInferenceAlternative, type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-weights-kit.ts:5`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-space.ts:5`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-geography.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation.compute.ts:16`: move `type GPUInferenceAlternative, type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/time/emerging-hot-spots.scene.ts:5`: move `GPUInferenceMetadata` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/time/b13-categories.ts:5`: move `GPUInferenceMetadata, GPUMultipleTestingMethod` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/time/emerging-hot-spots.compute.ts:14`: move `type GPUInferenceMetadata, type GPUMultipleTestingMethod` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/time/quake-data.ts:11`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/networks/transit-data.ts:5`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/flight-corridors.compute.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/flight-corridors.compute.ts:13`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/us-shipping-density.ts:6`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/us-shipping.tracks.ts:6`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/migration-stopovers.compute.ts:7`: move `type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/harbor-continuous-encounters.compute.ts:8`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/movement/b12-tracks.ts:5`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/storm-cell-tracks.compute.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/ocean-drifters-data.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/storm-outage-exposure.compute.ts:23`: move `GPU_EARTH_AUTHALIC_RADIUS_METERS, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/pacific-boundary-logic.compute.ts:13`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/satellite-tracks.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/satellite-ground-tracks.compute.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/hurricane-landfall.compute.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/satellite-swath.compute.ts:7`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/hurricane-families.compute.ts:9`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/earth/spherical-boundaries.compute.ts:13`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/b14a-colorize.ts:14`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/b14b-terrain.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/b14a-grid.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/viewshed.compute.ts:8`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/cpu-dem.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/landforms-paint.ts:14`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/b14a-session.ts:28`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/summits-graphs.ts:18`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/relief-visualization-paint.ts:17`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/joins/route-footpoints.compute.ts:15`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-density.compute.ts:7`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/point-patterns-jitter.ts:6`: move `type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-clusters-outliers.compute.ts:7`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/earthquake-pair-structure.compute.ts:14`: move `GPU_STUDY_WINDOW_FLAG_EMPTY, GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/point-patterns.compute.ts:23`: move `GPU_STUDY_WINDOW_SUMMARY_LENGTH, type GPUPolygonGeometryFields` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/point-patterns-neighbours.ts:5`: move `type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-clusters.compute.ts:7`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/geometry/b3-common.ts:5`: move `GPU_PLANAR_SPATIAL_CONTEXT, type GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/geometry/b3-line-views-far.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/points/street-density.compute.ts:7`: move `type GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/geometry/wildfire-data.ts:6`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/geometry/polygon-measures-global.ts:6`: move `getGPUAuthalicSphereRadius` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/wind-flow.compute.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/geometry/polygon-measures.compute.ts:20`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/grid-alignment-frame.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/b16-common.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS, GPU_WGS84_FLATTENING` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/wildfire-terrain.compute.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/wildfire-raster-layer.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/regression/b6-polygons.ts:15`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/grid-alignment-projection.ts:13`: move `GPU_WGS84_FLATTENING, GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/grid-alignment.compute.ts:17`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/flows/flight-bundling-routes.ts:12`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/flows/bixi-tides.compute.ts:11`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/flows/flight-matrix.compute.ts:16`: move `GPUParameterBuffer` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/scenes/flows/bixi-bundles-stats.ts:6`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/engine/analysis-frame.ts:6`: move `createGPULocalFrame, GPU_EARTH_MEAN_RADIUS_METERS, GPU_WGS84_SEMI_MAJOR_AXIS_METERS, GPU_WGS84_FLATTENING` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/engine/analysis-frame.ts:12`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/engine/layer-wgsl.ts:5`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core`
- `examples/deck/spatial-analysis-showcase/src/engine/web-mercator.ts:5`: move `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/cartography/reference-geometry.ts:16`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/engine/projection.ts:5`: move `GPUSpatialContext` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/cartography/anchors.ts:11`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)
- `examples/deck/spatial-analysis-showcase/src/engine/relief.ts:6`: move `GPU_EARTH_MEAN_RADIUS_METERS` to `@luma-contributors/analysis/analysis-core` (whole statement: change the specifier)

## 5. Roadmap
STD-09 row, replace "Remaining (breaking): ..." with:
"Done 2026-11-11 (breaking): the domain entries no longer re-export the contract. 183 names were removed from `gpu-spatial-analysis`, `gpu-network` and `gpu-terrain`; import them from `analysis-core`. The `@internal` builders are no longer exported from any barrel (internal-export 0). The core `validatePackedView` is renamed `validateSinglePackedView`. Remaining, later: fold `gpu-spatial-analysis/shared` and the raster iteration gate (`cost-distance/raster-relaxation.ts`, used by line simplification) into core; move `regression-row-tiles.ts` and `line-merge/list-ranking.ts` to core when a second family needs them; decide whether `analysis-core` exports `validateSinglePackedView` and `getGPUGridIndexMaximumCellCount`. Website and examples imports move in the rebuild pass (list in the w2-core-surface report)."
Section 12.4 row "Rename of the local `GPUKMeans` and `validatePackedView`": drop `validatePackedView` (done).

## 6. Noticed
- The shared typecheck watcher no longer watches `raster-analysis/cost-distance/gpu-cost-distance.ts` (stale error since 00:27). Restart it before relying on it for that file.
- Failures from other streams: the `gpu-point-pattern-indices` legacy WGSL hash (an `isFiniteValue` rename inside WGSL breaks the "byte-identical" pin); lens twins and 4 gpu-network record-structure entries (LN-REC2); `s3-geometry-pack` `latitude2` and `statistics-regions-pack` `getSsd` formula bindings; 4 typecheck-tests errors (section 2).
- `docs/api/gpu-spatial-analysis/spatial-frames.md` imports were not caught by doc-snippets (that page's blocks may not be checked); I fixed them by hand.
- `modules/analysis/package.json` has no `./testing` export: it is private by design, but `testing/index.ts` is an entry barrel that the checker scans.
