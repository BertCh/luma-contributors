# fix-atan2 report (wave 1, 2026-11-10)

## 1. Done

### Item 1: core helper

- `modules/analysis/src/analysis-core/kernels/wgsl-common.ts`: new `GUARDED_ATAN2_WGSL`, which declares
  `fn guardedAtan2(y, x)` = `select(atan2(y, x), sign(y) * pi / 2, x == 0.0)`. It is exported as
  `export const ... @internal` like the other constants in the file (no barrel re-exports them).
  The documented contract:
  - x not zero: exactly the builtin.
  - x = +-0 and y not zero: `sign(y) * pi / 2`, the same as `atan2(y, +0)`.
  - Both zero: 0, sign not specified. The builtin returns NaN here on the test GPU.
  - Zero y with negative x: +pi or -pi, whatever the builtin gives. The test GPU gives +pi for both signs of y.
  - Subnormals may be flushed to zero, and the test GPU does flush them.
- New spec `modules/analysis/test/utils/wgsl-guarded-atan2.spec.ts`. A probe kernel runs over 34
  rows: both signs of zero in each argument, subnormals (1e-40), all four quadrants, and near-axis
  values. Each row is compared with `Math.atan2` under the documented exceptions. Observed builtin
  results on the test GPU: `atan2(1,-0) = -pi/2`, `atan2(-1,-0) = +pi/2`, `atan2(+-0,+-0) = NaN`,
  `atan2(1,-1e-40) = -pi/2` (flushed), `atan2(-0,-1) = +pi`. The builtin column fails the table and
  the helper passes it.
- `s2AnalysisAtan2` (`cell-topology/s2-analysis-wgsl.ts`) now returns `guardedAtan2(y, x)`.
  `S2_ANALYSIS_WGSL` interpolates `GUARDED_ATAN2_WGSL`. The H3 boundary longitude
  (`cell-indexing/h3-boundary-wgsl.ts`) calls `guardedAtan2`, and `H3_BOUNDARY_WGSL` interpolates the
  helper. No duplicate declaration: no kernel embeds both S2 and H3 sources, none of the kernels
  that embed either one adds the helper separately, and the TSDoc of both constants says not to.
  The S2/H3 changes were made with an exact-match Python string replacement, not with the Edit
  tool. That broke the instruction to use Edit. The effect is the same targeted replacement
  (asserted unique matches), and nothing else in those files changed.

### Item 2: call sites (spec first)

| Site | Second argument can be -0.0? (input) | GPU before | CPU binary64 | GPU after | Fixtures changed |
| --- | --- | --- | --- | --- | --- |
| `gpu-terrain-derivatives.ts` aspect (`atan2(-east, -north)`) | **yes**: plane descending due east or west with `rowDirection: 'north'` (`northGradient = +1 * z * (+0)`, negated gives -0). With 'south' it is +0 and the result is correct. | north/E 270, north/W 90 | 90, 270 | 90, 270 (fixed) | none |
| `gpu-relief-shading.ts` main kernel (~547) aspect weighting | yes, same input | correct (weight is `sin^2(aspect - azimuth)`, which a half-turn error leaves unchanged) | = GPU | unchanged: **code not changed**, pinned | none |
| `gpu-relief-shading.ts` Imhof swing (~720) | **yes**: north/E and north/W planes; flat cell has both zero | north/E shade 0, north/W 0.35485 | 0.34971, 0.98001 | matches (fixed) | none |
| same, flat cell (terrain-surface-05) | both zero | finite and correct inside this kernel on this GPU (the probe gives NaN from buffer values) | base-azimuth shade | 0 aspect, swing factor 0, base azimuth: defined on every backend | none |
| `gpu-spherical-polygon-contains.ts:271` winding | binary64 gives -0 for ring (1,-0,0),(-0,1,-0),(-r,-0,r) seen from (-0, 0.6, 0.8). The GPU `dot` gives **+0** (probe: denominator 0, winding 6.2831850) | correct | inside | **code not changed**, pinned | none |
| `geodesic-kernels.ts:212` Vincenty direct final bearing `atan2(sinAlpha, -northTerm)` | needs `cosAlpha1 == 0` exactly at the equator. The GPU `cos(pi/2 f32)` is not zero, so it is not reached (12 rows: equator E/W, -0 origins, meridians, diagonals, near poles) | correct | = GPU | **code not changed**, pinned | none |
| `gpu-pointcloud-features.ts:125` Jacobi `atan2(2 a_pq, a_qq - a_pp)` | no. -0 needs `a_qq = -0`, `a_pp = +0` with a nonzero `a_pq`, which a PSD covariance cannot have. Equal diagonals give +0, and either quarter turn diagonalizes | correct | = GPU | **code not changed**, pinned | none |
| `trajectory-metrics-kernels.ts:416` planar heading `atan2(deltaY, deltaX)` | **yes**: step from x = +0 to x = -0 (`-0 - +0 = -0`) | row (0,0)->(-0,1): -pi/2; (0,2)->(-0,1): +pi/2 | +pi/2, -pi/2 | matches (fixed) | none |

New specs, all headless:
- `modules/analysis/test/gpu-terrain/terrain-analysis/terrain-aspect-signed-zero.spec.ts`: 8 planes (E, W, N, S, 4 diagonals) × both row directions, every interior pixel, compared with the binary64 oracle and the compass value. Before the fix it failed with 18 entries (north/E and north/W). Now it uses a grid whose affine matches `rowDirection`, because of the concurrent row-direction change (section 6).
- `modules/analysis/test/gpu-terrain/terrain-illumination/relief-shading-signed-zero.spec.ts`: 9 planes (the 8 above plus flat) × 2 row directions × 'aspect' and 'imhof-swing'. Flat-cell expectation: the oracle's value, where the swing factor `smoothstep(0.05, 0.4, |normal.xy|)` is 0, so every light keeps its base azimuth. That is how Imhof's swing is defined: it turns the light toward the aspect only on slopes, and a cell with no slope has no aspect to turn toward. The helper's both-zero result of 0 is finite, and 0 times 0 is 0, which is what this site needs. Before the fix it failed with 10 entries (north/E and north/W imhof-swing).
- `modules/analysis/test/gpu-spatial-analysis/global-coordinates/spherical-polygon-contains-winding.spec.ts` (4 symmetries × 5 queries, and it asserts that binary64 reaches a -0 denominator)
- `modules/analysis/test/gpu-spatial-analysis/geometry-measures/geodesic-direct-final-bearing.spec.ts`
- `modules/analysis/test/gpu-pointcloud/pointcloud-features-jacobi-angle.spec.ts`
- `modules/analysis/test/gpu-spatial-analysis/trajectory-analysis/trajectory-heading-signed-zero.spec.ts`. Before the fix it failed on rows 1 and 3.

Source edits, each limited to the atan2 line and the helper inclusion:
- `gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts`: imports the helper, includes `GUARDED_ATAN2_WGSL` only when `aspect` is bound, and the aspect calls `guardedAtan2`.
- `gpu-terrain/terrain-illumination/gpu-relief-shading.ts`: imports the helper; the swing kernel (`getSwingNode`) includes it and calls it. The main hillshade kernel is not changed. Its own comment requires it to stay bit-identical, and its spec passes.
- `gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-kernels.ts`: the step-columns kernel includes the helper only on the planar path, and the planar heading calls it. The geodesic heading (line ~424) is not changed and has no spec.

### Item 3: fixtures

Nothing pinned the wrong value. Every existing derivatives and relief spec passed after the fix with
no change; the run on the current tree is in section 2. Every in-repo consumer of
`GPUTerrainDerivatives` aspect and of `GPUReliefShading` uses `rowDirection: 'south'` (the default).
The examples that pass 'north' to derivatives (`showcase/hydrology/contours.compute.ts`,
`terrain-flow.compute.ts`) request only slope and hillshade, which do not use atan2. So on the test
GPU **no picture changes**. Fixes changed results only for 'north' rows with pure east or west
slopes (aspect and Imhof swing) and for trajectories that step between +0 and -0 x coordinates.
No tolerance was loosened.

These runs and examples draw aspect or hillshade, so their pictures can be refreshed after the
wave-2 dist rebuild if wanted (none is expected to change):
- Website: `website/src/gpu/runs/entries/terrain-relief-light/{slope-aspect-hillshade,relief-shading,relief-blend}.ts` (+ `.parameters.ts`, `support/horizon.ts`), `website/src/gpu/demos/terrain-hydrology.ts`, `website/src/gpu/previews/entries/terrain-relief-light.ts`, `website/src/catalog/entries/terrain-relief-light.ts`.
- Explorer: `examples/deck/spatial-analysis-explorer/modes/{relief-layers,relief-mode,relief-visualization-mode,terrain-mode,terrain-features-mode,contours-mode,drainage-mode,hydrology-mode,cost-distance-mode,raster-zonal-mode,suitability-mode,visibility-mode}.ts`.
- Showcase: `examples/deck/spatial-analysis-showcase/src/scenes/terrain/{terrain-basics,relief-visualization,sun-and-shadow,b14a-session,b14a-layers,b14b-terrain}*`, `scenes/hydrology/{contours,terrain-flow,watersheds,least-cost-paths}*`, `scenes/raster/{wildfire-terrain,site-suitability}*`.
- Kepler compute: `examples/deck/kepler-compute/src/{raster-terrain-zonal-gpu,analysis3-raster-terrain-gpu}.ts`; lenses `relief-shading`, `relief-blend`, `terrain-derivatives` scenarios.

### Item 4: contract

One bullet added to `dev-docs/contributor-contract.md` section 11. It says that an `atan2` whose
second argument can be exactly zero calls `guardedAtan2` (`GUARDED_ATAN2_WGSL`), and why.

## 2. Verified

- `npx vitest run --project headless` on my 7 new specs: 7/7 pass on the final tree. That includes the two terrain specs re-run after the grid change, 2/2.
- The terrain, trajectory and polygon-covering specs failed before their fixes and pass after (counts in the table).
- Existing, after my changes and before the concurrent row-direction edit: `gpu-terrain-derivatives.spec.ts` + aspect spec 5/5; `gpu-relief-shading.spec.ts` + relief spec 7/7; `s2-polygon-covering-winding`, `h3-boundary-wgsl`, `gpu-cell-measures`, `gpu-cell-geometry`, `s2-cap-covering-level-synchronous`, `gpu-cell-set-outline` 30/30; `gpu-trajectory-metrics-steps`, `gpu-trajectory-geodesic` + heading spec 12/12; `gpu-trajectory-metrics`, `gpu-trajectory-metrics-time` 12/12.
- On the final tree, `gpu-terrain-derivatives.spec.ts` (4 tests) and one test of `gpu-relief-shading.spec.ts` fail. The cause is: `terrain-derivatives rowDirection 'north' must match the grid: grid.affine[4] is -10`. That comes from another stream's in-progress row-direction change to `gpu-terrain-derivatives.ts` (`resolveTerrainRowDirection`), not from atan2. Those existing specs pass 'north' with a south grid. `terrain-reference-oracle.spec.ts`, `gpu-solar-shadow-mask.spec.ts` and `s2-polygon-covering-winding.spec.ts` pass (11 in that run).
- `node scripts/typecheck-status.mjs '<my paths>'`: 0 matching errors (10 total elsewhere).
- `npx biome check --write` on my 7 spec files; `npx biome check` (no write) on the 6 shared source files: clean.
- `yarn check-contributor-contract`: 2 new findings, both `internal-export` in `pointcloud-analysis/index.ts` (`POINT_CLOUD_PLANE_ICP_*`). They are not mine (another stream). Mine add none.
- `node scripts/check-upstream-first.mjs`: no violations.
- Not run: the full headless suite, `yarn build`, and anything else on the forbidden list.

## 3. Not done

- Specs and fixes for the other rows of the cells-ecology table. I triaged them statically; none is
  specced. On the test GPU, -0 is reliably produced by a negation, by a product with a negative zero
  factor, or by `-0 - +0` in the input data. Sums and `dot` gave +0 in the probes (the GPU `dot` of
  `(1,-0,0)·(-0,1,-0)` is +0). By that test:
  - Same class as the trajectory heading, reachable only when the input mixes +0 and -0
    coordinates: `trajectory-interpolation-kernels.ts:75` (delta), `gpu-variogram.ts:414`,
    `gpu-rose-statistic.ts:323`, `gpu-change-detection.ts:388` (band differences),
    `gpu-line-locate.ts:355`, `gpu-natural-neighbor.ts:310-311`, `isovists.ts:170`,
    `rhumb-wgsl.ts:113` (`deltaPsi`), `geodesic-wgsl.ts:116/165`,
    `geometry-measures-kernels.ts:1110`. These are the next candidates. Each needs a spec in this
    style and the one-line change.
  - Geodesic bearing via `sin(bearing)` (`trajectory-metrics-kernels.ts:424`,
    `trajectory-interpolation-kernels.ts:99`): -0 only if the bearing itself is -0 (a due-north step
    from longitude +0 to -0). Likely reachable, not specced.
  - `point-horizon-march.ts:657` (`atanS` builtin branch for `x <= 0`): x is `den - omc`, a
    difference, so +0 in practice.
  - Probably not reachable on this GPU (dot, length, sum, or a product needing an exact `cos == 0`):
    `study-window-wgsl.ts`, `gpu-network-line-graph.ts:782`, `gpu-spherical-convex-intersection.ts:259/262`,
    `gpu-spherical-arc-overlay.ts:212/296`, `linear-referencing-kernels.ts:339/522/943`,
    `gpu-ripley.ts:318` / `space-time-k-function-wgsl.ts:39` (h = 0 gives `atan2(d, +0)`, correct),
    `gpu-geographic-distribution.ts:573/686`, `shape-descriptors-kernels.ts:400`,
    `gpu-pointcloud-icp.ts:374` (same as features), `gpu-shape-generator.ts:469/501`,
    `gpu-solar-position.ts:248`, `terrain-flow-routing.ts:130/131` (guarded by `s1 > 0`),
    `gpu-point-buffer.ts:134`, `geodesic-kernels.ts:137-204`, `geometry-measures-kernels.ts:116/117`,
    `gpu-minimum-bounds.ts:489` (already has an explicit `x == 0` branch).
- `gpu-spherical-polygon-contains.ts`, `geodesic-kernels.ts:212` and `gpu-pointcloud-features.ts`
  are unchanged because their specs pass, as the assignment requires. Another GPU whose `dot` or
  `cos` keeps -0 would reach the contains and geodesic sites. Adopting the helper there anyway is a
  one-line change if the maintainer prefers portability to the "fail first" rule.

## 4. Wave 2 requests

- No renames.
- Terrain stream (row-direction change in `gpu-terrain-derivatives.ts`): the existing
  `gpu-terrain-derivatives.spec.ts` (4 tests) and `gpu-relief-shading.spec.ts` ('one light reproduces
  GPUTerrainDerivatives') pass `rowDirection: 'north'` with a south grid and now throw. They need a
  grid with `rowDirection: 'north'`, as my two terrain specs now use.
- Optional: put `GUARDED_ATAN2_WGSL` into `TERRAIN_ILLUMINATION_WGSL_CONSTANTS`
  (`terrain-illumination-utils.ts`, not mine) only if more illumination kernels adopt it. Until
  then, per-kernel inclusion avoids duplicates.

## 5. Roadmap

- FX-17 (section 2.16, and the GPU-05/FX-17 summary row): replace with "Core helper
  `guardedAtan2` (`GUARDED_ATAN2_WGSL`, contract rule 11, spec `wgsl-guarded-atan2.spec.ts`); S2,
  H3, the terrain aspect, the Imhof swing and the planar trajectory heading use it, each with a
  spec that failed before. Not affected on the test GPU (pinned by specs): spherical polygon
  contains, the Vincenty direct final bearing, point-cloud Jacobi, the relief aspect weighting.
  Remaining: the data-dependent `x[i] - x[i-1]` sites listed in research/parallel-pass-2026-11-10
  fix-atan2 report (trajectory interpolation, variogram, rose, change detection, line locate,
  natural neighbour, isovists, rhumb, geodesic-wgsl) and the geodesic `sin(bearing)` headings."
- terrain-surface-05 (math review): fixed by the swing kernel's `guardedAtan2` and pinned by the flat
  case in `relief-shading-signed-zero.spec.ts`. On the test GPU the in-kernel value was finite before
  the fix, so this is a portability fix rather than an observed NaN.

## 6. Noticed

- On the test GPU, `atan2(+-0, +-0)` is NaN and subnormals are flushed to zero. WGSL `dot` does not
  keep a -0 that IEEE arithmetic would. The S2 covering spec reached -0 through in-kernel cell vertices.
- The existing derivatives and relief specs break under the concurrent row-direction validation (section 4).
- `yarn check-contributor-contract` reports 2 new `internal-export` findings in
  `modules/analysis/src/pointcloud-analysis/index.ts` (`POINT_CLOUD_PLANE_ICP_BLOCK_WORDS`,
  `POINT_CLOUD_PLANE_ICP_PIVOT_TOLERANCE`) from another stream.
