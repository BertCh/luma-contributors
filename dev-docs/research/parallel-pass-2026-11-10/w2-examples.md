# w2-examples report

## 1. Done

Typecheck command (from `.github/workflows/test.yml`, the only place the examples are checked; the
root `yarn typecheck` and the shared watcher do not include them):
`NODE_OPTIONS=--max-old-space-size=8192 npx tsc --noEmit -p examples/deck/spatial-analysis-showcase`
and the same for `-p examples/deck/spatial-analysis-explorer`. Before: 129 and 33 errors. After: 0 and 0.
No casts were added.

Files changed (about 100). Per kind of change:

- **Import moves to `@luma-contributors/analysis/analysis-core`** (89 files in showcase, explorer and
  `examples/deck/spatial-analysis-shared`). Done by a script that asks TypeScript which names
  `analysis-core` exports and splits each `gpu-spatial-analysis` import statement accordingly; biome
  reformatted the result. Covers `GPUSpatialContext`, `GPUParameterBuffer/Format`, `GPULocalFrame`,
  `createGPULocalFrame`, the earth and WGS84 constants, `GPUInferenceAlternative/Metadata`,
  `GPUMultipleTestingMethod`, `GPU_STUDY_WINDOW_*`, `GPUPolygonGeometryFields`,
  `GPURegressionModelResult`, `getGPUAuthalicSphereRadius`, `GPU_PLANAR_SPATIAL_CONTEXT`.
  `examples/deck/spatial-analysis-shared/` is not in my ownership list but both examples import it
  (`resources.ts`, `hot-spots-analysis.ts`), so it was edited for the import move only.
- **`GPUKMeans` family renamed** (word-boundary replace) in `clusters-mode.ts` (explorer),
  `hurricane-families`, `satellite-constellations`, `nature-clusters`, `regionalization` (compute and
  scene files) and `docs/guides/points.md`, `docs/guides/regression.md`.
- **`validity` to `mask`**: explorer `contours-mode.ts` (4), `suitability-mode.ts` (2),
  `terrain-features-mode.ts` (1); showcase `raster/spectral-texture-display.ts`.
  **`mask` to `validity`** on the point-cloud statistical filter: `terrain/lidar-ground.compute.ts`.
- **`parameters` now required**: `raster/habitat-connectivity.compute.ts` (walking cost; the grass
  model object is now one shared constant passed both as `model` and packed by
  `getGPURasterWalkingCostParameterValues`, because the numbers in `model` are ignored now),
  `raster/pixel-boundaries.compute.ts` and `statistics/chicago-population-support.compute.ts`
  (fractional coverage, packed from the same grid).
- **Line output `count` is lines**: `importPathOutput` in `geometry/b3-common.ts` and the copy in
  explorer `geometry-mode.ts` now write `status: {count: <path count>, overflow, vertexCount:
  <vertices>, requiredVertexCount}`; the `lineCount` option was removed with its three callers
  (`b3-line-views-city.ts`, `buffers-and-shapes.compute.ts`, `geometry-mode.ts`). The explorer
  polygon rectangle clip (`geometry-tools-mode.ts`) gets a ring-count buffer for `count` and passes the
  vertex buffer as `vertexCount`. This was NOT a type error and would have silently changed what the
  helpers report.
- **Linear referencing** status `{candidateOverflow, requiredCandidateCount}`: explorer
  `geometry-mode.ts`, showcase `b3-line-views-city.ts`.
- **Overlay/buffer outputs**: removed `geometry.sourceIds` and renamed `status.requiredCount` to
  `requiredRingCount` in `earth/spherical-boundaries.compute.ts`, `geometry/repair-overlay.compute.ts`,
  `geometry/block-relationships.compute.ts`, `geometry/footprint-shape-lab.compute.ts`. In
  block-relationships the tessellation copy loop reads the now-optional `requiredRingCount`; it throws
  a named error at graph construction if it is missing instead of casting. The unused
  `overlaySourceIds` buffer in repair-overlay was removed (other now-unused source id buffers were left).
- **Shared paths** `leftLineIds/rightLineIds` to `leftSourceIds/rightSourceIds`: `block-relationships.compute.ts`
  (2 sites), `county-linework.compute.ts`.
- `statistics/county-cross-variogram.compute.ts`: `GPUSpaceTimeVariogram` requires `gridSize`; added
  `gridSize: [8, 8]` on the space-time branch only (speed only, 21 counties x periods).
- `scripts/check-cartography.mjs` and `scripts/check-story-steps.mjs`: the esbuild plugin that makes
  the pure CPU contracts real matched `@luma-contributors/analysis/gpu-spatial-analysis`; the
  contracts moved, so it now matches `.../analysis-core`. Without this 22 scenes failed to load
  (`createGPULocalFrame is not a function`).

## 2. Silent changes (text, captions, expected values)

None of the 47 call sites needed an override of a corrected default; no picture was restored to the
old state. No stored oracle or hard-coded expected number in the examples restated an old value that I
could find (the `*-reference.mjs` oracles are only read by browser checks, see 3).

| Where | Old statement | New | Cause (behaviour-changes.md) |
|---|---|---|---|
| `geometry/line-operations.scene.ts` (body, readout help, 3 code snippets) | chunk count "through the `lineCount` output"; snippets `status: {count, overflow}`, `{count, overflow, lineCount}`, `{overflow, candidateCount}` | `status.count` counts lines; snippets show `vertexCount`, `candidateOverflow, requiredCandidateCount` | line-output status rename (w2-geometry) |
| `geometry/buffers-and-shapes.scene.ts` snippet | `requiredCount, lineCount` | `requiredCount, vertexCount` | same |
| `explorer/modes/geometry-mode.ts` chunk note | "lineCount reports it" | "status.count reports it (lines, not vertices)" | same |
| `hydrology/contours.scene.ts` body | saddle: "the cell-centre average decides" | the value of the bilinear surface at its saddle point (asymptotic decider) | `GPUIsolines/Isobands/IsobandRings` saddles (raster-fields-06) |
| explorer `coverage-mode.ts` (description, note, 3 readout labels) and showcase `noding-and-coverage.compute.ts` readout | "crossings found / remain" | "crossings and swept spans" | `GPUCoverageSimplification` `topologyStatistics[0..1]` count swept spans (polygon-topology-02) |
| `docs/guides/hydrology.md` | "`rowDirection: 'south'` (the default)" | read from the sign of `grid.affine[4]`; a contradicting value throws | terrain-surface-02 |
| showcase `terrain/cpu-dem.ts` `hornAt(column,row,border)`; used with `'one-sided'` by `sun-and-shadow-summary.ts` and `relief-visualization.compute.ts` (tooltip) | CPU slope readout repeated the edge cell, matching the old GPU border; GPU `GPUTerrainDerivatives` now defaults to `'one-sided'` | CPU probe gets an optional `'one-sided'` rule (same quadratic extrapolation as `TERRAIN_ONE_SIDED_BORDER_WGSL`); default of `hornAt` stays `'clamp'` for `terrain-basics` (which passes its own `borderMode`) and the other probe users | terrain border (terrain-surface-03). Only border-ring cells differ (e.g. a 11.31 degree plane read 5.71 before) |

Checked and left alone (text does not state the old rule): k-means scenes (all pass `initialization`
and `seed` explicitly; no caption names a seed outcome), Louvain, network K envelope, conditional
simulation, cross-validation folds (text is generic: "one of five folds", no fold numbers), isochrone
bands and `'max'` mode (text describes the mode, no numbers), stretch (percentile/equalize text only
says "accurate to one bin"), coarsening sums (no `65536` claim; `GPU_CELL_DEFAULT_SUM_SCALE` snippet in
`cell-pyramid.scene.ts` is still exported), `rowDirection` (all explicit calls kept; list said 42 agree),
point-cloud grids (`lidar-ground` `RASTER_GRID` and `point-cloud-support` `rasterGrid` are internal and
shared by every contributor, never drawn).

Scenes whose picture or numbers change (computed live, not verifiable without a browser):
terrain border ring of slope/aspect/hillshade/curvature in the 11 explorer and 7 showcase terrain
sites of fix-review-callers 4.1; k-means centres in the 5 sites; Louvain label ids in
`aviation-network-lab` (seed 17); K-function envelopes; cross-validation folds; coarsening sums in
`aviation-network-lab` and `flight-matrix`; isochrone and drive-time band edges (`(a, b]`); equalized
stretch; coverage swept-span counts. These come from the library and are not changed by the examples.

## 3. Verified

- `tsc` showcase: 0 errors; explorer: 0 errors (commands above). `npx biome check --write` on the three
  directories: 3 pre-existing `noExplicitAny` warnings in `docs/live-lens-adapter.ts`, no errors.
- `node scripts/check-upstream-first.mjs`: no violations.
- Specs that import example code: `npx vitest run --project node` on the 8 specs listed by
  `grep -rl examples/deck/spatial-analysis modules/*/test` (parity-wp6-round6-fixtures,
  parity-wp6-reduction-fixtures, parity-wp5-round5/6/7, wgsl-reserved-identifiers, parity-wp1-round5,
  parity-wp1-round6): 8 files, 72 tests passed.
- Showcase check scripts that run without a browser, after the changes: `check-cartography` (exit 0, 6
  style warnings, all of the kind "step count" / "viridis"; `--strict` fails on them, as before),
  `check-story-steps` (exit 0, also under `--strict`), `check-colours`, `check-datasets`,
  `check-wgsl-identifiers`, `check-wgsl-identifiers.test.mjs`, and also exit 0 on the first run:
  `check-bixi-models-reference`, `check-block-tessellation-offsets`, `check-cell-neighborhoods-identities`,
  `check-legend-layout`, `check-map-chart-responsive`, `check-map-legend-spacing`,
  `check-responsive-camera`, `check-route-affine-reference`, `check-street-density-publication`,
  `check-time-bar-layout`.

## 4. Not verified / not run

- 43 other `check-*.mjs` scripts exit 1 for reasons independent of my change: 36 import
  `scripts/playwright/get-playwright-launch-options.mjs` or `collect-page-diagnostics.mjs`, which do not
  exist anywhere in the repository (no git history either); 2 need a dev server on port 5313
  (`check-atlas-navigation`, `check-geometry-observation-batch`). The `*-browser.mjs` scripts and
  `capture-story-previews.mjs` need a browser and a server. So no scene was rendered or captured: all
  picture changes above are unverified visually. The kepler-compute example was not touched.
- The explorer has no check script; only `scripts/build-new-york-land.mjs` (data builder, not run).
- `yarn build`, a vite build, and the rebuilt dist were not exercised; the vite alias file
  (`examples/vite.config.shared.ts`) aliases every package subpath so `analysis-core` resolves.

## 5. Noticed (library or other, not fixed)

- `GPUSpaceTimeVariogram` still requires `gridSize` while `GPUCrossVariogram` made it optional today
  (dense path up to 16384 rows). The two are built from one `commonProps` object in the county scene, so
  the asymmetry forces a branch-specific prop.
- `GPUPolygonOverlayOutput.status.requiredRingCount` is optional, but the old `requiredCount` was required;
  callers that copy it need a presence check.
- Several of the wave-2 list changes were not compile errors: `footprint-shape-lab.compute.ts`
  (`sourceIds`, `requiredCount`) and the line-output helper (`lineCount`, `count` meaning) type-checked
  before the edit, so the typecheck cannot be the authority for the wave 2 renames; I applied the list
  entries there anyway.
- `check-cartography`/`check-story-steps` (and any other esbuild-stub script) silently depend on which
  barrel holds the contracts; they failed only at scene load time, not at startup.
- `scripts/playwright/` is referenced by 36 showcase scripts and absent from the repository.
