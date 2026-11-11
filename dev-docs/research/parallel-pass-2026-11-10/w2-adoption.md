# w2-adoption: STD-10 shared WGSL modules, adoption (non-breaking)

All paths are under `modules/analysis/src/` unless noted. No exported TypeScript name changed and
no fixture or spec changed. Only WGSL function names inside generated shader text changed
(internal), plus one documented requirement change on the exported `getCholeskyWGSL` (see 4).

## 1. Done

### Step 1: float-test copies (results identical: same bit tests, same NaN bit pattern `0x7fc00000`)

| Site | Change |
| --- | --- |
| `gpu-terrain/terrain-processing/terrain-processing-utils.ts` `TERRAIN_FINITE_WGSL` | `finite`/`missing` replaced by `IS_FINITE_VALUE_WGSL` + `GET_NAN_WGSL` (keeps `infinity()`); callers in `gpu-terrain-depression-hierarchy`, `-dem-coregistration`, `terrain-tiled-resident`, `-branching-flow`, `-breach`, `-catchment-summary` and the masking kernel of `terrain-processing-utils` renamed to `isFiniteValue(` / `getNaN()` |
| `gpu-terrain/hydrology/terrain-flow-spill-utils.ts` `isSpillFinite` (`abs bits < inf bits`, equal to the exponent test) | `IS_FINITE_VALUE_WGSL` inside `getSpillCommonWGSL`; callers in `terrain-flow-spill-fill.ts`, `terrain-features/terrain-peak-prominence-tree.ts`, `gpu-terrain-peak-prominence.ts` |
| `gpu-terrain/relief-visualization/relief-visualization-utils.ts:517`, `terrain-illumination/terrain-illumination-utils.ts:22` local `getNaN(index)` (seeded, and a name clash with core `getNaN()`) | `GET_QUIET_NAN_SEEDED_WGSL` (`getQuietNaN(seed)`, same body); 24 calls in 13 relief and illumination files renamed |
| `gpu-terrain/terrain-features/terrain-features-utils.ts` `getNan(seed)` | `GET_QUIET_NAN_SEEDED_WGSL`; calls in `gpu-terrain-peak-snap.ts`, `gpu-terrain-summits.ts` |
| `gpu-terrain/terrain-features/gpu-profile-peaks.ts` `getNanFunction()` / `getNan()` | `GET_NAN_WGSL` / `getNaN()`; private builder deleted |
| `gpu-network/network-optimization/gpu-network-{point-routing,facility-optimization,time-routing,rational-access,louvain}.ts`, `network-time-geography/gpu-network-potential-path-area.ts` (`FINITE_WGSL` deleted) `finite` | `IS_FINITE_VALUE_WGSL`, calls renamed |
| `gpu-network/network-noding/gpu-network-noding.ts` `isFiniteCoordinate` | `IS_FINITE_VALUE_WGSL` (its `getOrderedKey` kept, see 6) |
| `dataframe-analysis/column-classification/scalar-scale-wgsl.ts` `scalarIsFinite` | `IS_FINITE_VALUE_WGSL` in `SCALAR_SCALE_WGSL`; calls in `gpu-scalar-scale.ts` |
| `dataframe-analysis/pair-statistics/pair-statistics-grid.ts` `PAIR_STATISTICS_FLOAT_WGSL` (`isFiniteFloat`, private seeded `getQuietNaN`) | `IS_FINITE_VALUE_WGSL` + `GET_QUIET_NAN_SEEDED_WGSL`; `isFiniteFloat(` renamed in the 10 pair-statistics files |
| `gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample.ts` `sampleFinite` | `IS_FINITE_VALUE_WGSL` |
| `gpu-spatial-analysis/recipes/spatial-regression-recipe.ts:159` `isNaNValue` | `IS_NAN_VALUE_WGSL` (`isNanValue`) |

The four deprecated constants (`GET_QUIET_NAN_WGSL`, `IS_NAN_BITS_WGSL`, `IS_FINITE_BITS_WGSL`,
`IS_FINITE_FLOAT_WGSL`) were NOT deleted: about 110 source files still import them, including the
excluded directories (`raster-analysis/cost-distance`, `raster-time-series`, `remote-sensing`,
`spatial-clustering/*`, `delaunay-tessellation`, `geometry-*`, `line-split`, `minimum-clearance`,
`shape-descriptors`, `vertex-snap`). Those importers use the core constants (not copies), so I left
them for the rename pass.

### Step 2: dense solvers, same arithmetic

- Site 1, `gpu-spatial-analysis/spatial-regression/spatial-regression-solve.ts` `getCholeskyWGSL`:
  the private factor, solve and inverse-diagonal loops are gone. The function now emits
  `getDenseCholeskyWGSL(p)` plus the old names as adapters: `choleskyFactor_p` is
  `factorDenseCholesky_p(a, p, false, 0.0).status == 0u`, and `choleskySolve_p` /
  `choleskyInverseDiagonal_p` call `solveDenseCholesky_p` / `getDenseCholeskyInverseDiagonal_p`
  with a unit-scale record. The record is a literal initializer `array<f32, p>(1.0, ...)`, not a
  fill loop (compiler fact of section 11). There is no equilibration, so the scale is exactly 1
  and every user (OLS, lag/error ML, GM, 2SLS, diagnostics, regimes, restricted regimes, IV
  covariance, probit, MGWR, SUR, panel lag, GTWR, GWR fit, nonstationarity test, GGWR) runs the
  same arithmetic. The module needs `isFiniteValue`, so `IS_FINITE_VALUE_WGSL` is now included once
  in each of the about 20 kernels that use it. With a grid index it is left out, because
  `getGridLookupWGSL` already embeds it (that was the duplicate-declaration trap: the first run
  failed 12 grid specs with `redeclaration of 'isFiniteValue'`, fixed by including it conditionally
  in `gwr-fit-wgsl.ts`, the nonstationarity permuted kernel and the GGWR `common`).
  Edge differences are the ones the map gives: a pivot in `(3e38, inf)` was rejected and is now
  accepted, and a non-finite lower-triangle entry is now rejected before factoring. Both fail
  either way.
- Site 2, `local-weighted-least-squares-wgsl.ts` `getEquilibratedCholeskyWGSL`:
  `factorEquilibrated_size` is `factorDenseCholesky_size(a, size, true, (1e-3)^2)`; it copies the
  module scale into the caller's `scale`. `solveEquilibrated_size` calls `solveDenseCholesky_size`
  with a record built from the caller's scale; the scale, solve, scale order is unchanged.
  `inverseDiagonalEquilibrated_size` keeps its formula: the module's own inverse diagonal scales
  before the solve, so it would round differently. Same equilibration arithmetic. The only
  difference is that the pivot test is now a ratio before the square root, not `L_jj >= 1e-3`
  after it, so a pivot within an ulp of the limit can flip; the pinned limits in
  `math-review-pivot-limit.spec.ts` (7e-4 vs 2e-3) are far from that.
- Site 5, `spatial-interpolation/gpu-kriging.ts` `solveSystem` (augmented Gaussian elimination):
  deleted, replaced by `factorDenseLU_17(&system, n, 0u, RELATIVE_PIVOT_TOLERANCE,
  PIVOT_RATIO_TOLERANCE)` + `solveDenseLU_17`, with the right-hand side in its own vector. The
  module's `largestEntry` equals the old `systemScale` (`max(1, max |gamma|)`, since the border of
  ones is in the matrix). Elimination, forward and back substitution run the same operations in
  the same order. A rejection (status 2 or 3) is `STATUS_UNCERTAIN` as before.

### Step 3: defect-closing sites

Neither site needed a fail-before spec, because both defects are already closed in today's tree:
- FX-09 (GWR family returns status 0 on nearly collinear designs): every GWR-family factor already
  goes through `factorEquilibrated_*` with the OLS limit, which rejects such designs (localStatus
  SINGULAR); `math-review-pivot-limit.spec.ts` pins the limit for probit, SUR, panel and error ML.
  A fail-before spec is impossible. The part not done: reporting the module's `ILL_CONDITIONED`
  (3) separately from SINGULAR. That would add a new GWR status code, a result change this
  assignment does not cover.
- FX-10 (two kriging tolerances, 1e-4 vs 1e-5): `equilibrated-solve-wgsl.ts` already uses
  `KRIGING_PIVOT_RATIO_TOLERANCE` = 1e-4, the same constant as `GPUKriging`. What is left is the
  Ruiz order (site 6), which moves fixtures (step 4).

## 2. Verified

- Baseline before solver edits: `npx vitest run --project headless` on the 43 headless
  spatial-regression specs (non-benchmark): 43 files, 353 tests passed. Same set after site 1:
  first run 12 failed (duplicate `isFiniteValue` with a grid), after the fix 353/353; after site 2:
  353/353; after biome: 353/353 (plus kriging, profile peaks, Ripley: 47 files, 386 passed).
- `npx vitest run --project node modules/analysis/test/analysis-core/kernel-audit
  modules/analysis/test/gpu-spatial-analysis/spatial-regression`: 84 files, 2901 passed. (The
  first audit run after site 1 found the missing `isFiniteValue` includes. Four
  `GPUSphericalPolygonOverlay` build-throw failures in it were not mine and passed in later runs.)
- Kriging: headless `kriging-conditioning`, `gpu-kriging`, `variogram-model-shape`,
  `space-time-kriging`, `kriging-small-lag`: 5 files, 61 passed; node `short-distance-warnings`,
  `spatial-interpolation.node`, kernel audit: 52 files, 2725 passed.
- Terrain: `terrain-processing/*`, `hydrology/terrain-direction-encoding`: 5 files, 55 passed (the
  first run caught one missed `missing()` in `terrain-processing-utils.ts:161`); node
  `terrain-processing.node`: 7 passed. Spill fill and prominence: headless 3 files, 6 passed;
  node 4 files, 25 passed. Relief and illumination (all non-benchmark headless): 21 files,
  144 passed, 1 skipped. Terrain features: headless 10 files, 117 passed; node 6 files, 33 passed.
- Network: noding, optimization (including Louvain visit order and movement-advanced),
  time-geography: headless 6 files, 84 passed; node 3 files, 18 passed;
  `location-allocation/facility-parallel`: 4 passed.
- Dataframe: all 20 non-benchmark headless pair-statistics specs: 490 passed; `gpu-scalar-scale`
  headless 3, node 5 passed.
- Recipes and point process: `spatial-regression-recipe` headless 3 passed;
  `continuous-spatial-sample.node`, `point-process-spatial-context.node`: 3 passed. Other OLS
  consumers: `value-cube/spatiotemporal-wave-two-composition`, `utils/core-device-limits`,
  `spatial-weights/chunked-consumers`: 13 passed.
- `node scripts/typecheck-status.mjs '<my directories>'`: 0 errors (0 total).
  `yarn check-contributor-contract`: no new findings. `node scripts/check-upstream-first.mjs`:
  no violations. `npx biome check --write` on the 128 files in those directories: fixed 12 files
  (import wrapping of my added imports, one blank line).
- Not run: the kepler spec `catalog-platform4-statistics-gpu.spec.ts` (uses `GPUKriging`), lenses
  tests, `typecheck-tests`. `GPUContinuousSpatialSample` has no headless spec; only the node audit
  (undeclared calls) and the node specs cover it.
- One mistake: a command with an empty glob started an unfiltered `vitest run --project headless`
  (the whole suite). I stopped it after about 3 minutes. It changed no files.

## 3. Not done (with reason)

- Binary search (about 25 named, about 110 inline), 64-bit helpers (7 sites), masks (58),
  row accessors, sphere distance (4), ordered keys: not reached, so they are untouched. One thing
  differs from the map: `UINT64_WGSL` is now `(high, low)` and every private copy is `(low, high)`,
  so those swaps are not mechanical.
- Site 6 (`equilibrated-solve-wgsl.ts`: local kriging, local RBF, space-time kriging, Vecchia):
  left, fixtures would move (sequential versus simultaneous Ruiz scaling).
- Sites 7 and 8 (`gpu-spatial-panel-lag-maximum-likelihood.ts` log-determinant by elimination and
  Gauss-Jordan `getInverseWGSL`): not attempted. The map gives a pivot-test change (absolute 1e-10
  to relative or exact zero) and a different inverse algorithm (Hausman moves in the last bits),
  so they are not results-identical.
- Site 11 (`getSymmetricEigenvalueRangeWGSL`): not attempted. The module uses a different stop
  rule (1e-6 off-norm versus 1e-12 on squared sums), so condition numbers move at rounding level.
- Sites 12, 13, 16: need a QR module (not written, as instructed).
- Sites 3, 4 (cooperative and blocked Cholesky), 18 (CPU), the closed-form 2x2: keep.
- The FX-09 status-3 split (see step 3).

### Site table

| # / site | Result |
| --- | --- |
| 1 `getCholeskyWGSL` (and its 15 users) | adopted, results identical |
| 2 `getEquilibratedCholeskyWGSL` (GWR, GTWR, nonstationarity, GGWR, panel) | adopted, results identical (FX-09 was already closed; status-3 split not done) |
| 3 conditional simulation cooperative Cholesky | left: keep (map) |
| 4 blocked Cholesky | left: keep (map) |
| 5 `GPUKriging.solveSystem` | adopted, results identical (FX-10 tolerance was already aligned) |
| 6 `equilibrated-solve-wgsl.ts` | left: fixtures would move |
| 7 panel lag log-determinant | left: fixtures would move (pivot test changes) |
| 8 panel lag `getInverseWGSL` | left: fixtures would move (Gauss-Jordan vs LU rounding) |
| 9 Kalman `inverse4` | left: another agent's directory (`trajectory-*`) |
| 10 point-cloud plane ICP 6x6 | left: another agent's directory |
| 11 eigenvalue range | left: fixtures would move (stop rule) |
| 12 Belsley, 13 local collinearity | left: no QR module |
| 14 point-cloud features Jacobi, 15 point-cloud ICP Jacobi | left: another agent's directory (also fixtures would move) |
| 16 raster Gram-Schmidt | left: another agent's directory; no QR module |
| 17 composite-score power iteration | already adopted by the math pass (shared eigen) |
| 18 CPU Householder; GM `invertPsi` 2x2 | left: keep |
| Float named copies: terrain (2), relief/illumination/features/profile-peaks NaN builders (4), network (7), dataframe (2), point-process (1), recipe (1) | adopted, results identical |
| `edge-bundling-passes.ts:154` `isFiniteValue` (`abs < 3e38`) | left: semantics differ (name collision; rejects magnitudes above 3e38) |
| `gpu-network-k-function.ts:909` `isFiniteCost` | left: semantics differ (`-0` is false) |
| `group-statistics-*` `isFiniteBits(bits: u32)` | left: not reached (signature differs, would be `isFiniteValue(bitcast<f32>(bits))`) |
| `network-isochrones-passes.ts:125`, `cell-cover-kernels.ts:116`, `polygon-edge-slabs.ts:437`, `gpu-bounded-facility-allocation.ts:148` | left: semantics differ (magnitude cap or sign test kept) |
| `shared/spatial-wgsl-point-helpers.ts` `isSpatialFinite`/`getSpatialNaN` | left: another agent's directory uses them (trajectory, spatial-clustering kernels call these names) |
| point-density `gpu-space-time-relative-risk.ts:278`, `gpu-space-time-kernel-density.ts:476`, `space-time-hawkes/gpu-space-time-hawkes.ts:423` NaN builders | left: not reached |
| `point-horizon-march.ts:623` NaN with a parameter payload | left: semantics differ (payload) |
| Raster (8 float sites, `change-detection-wgsl.ts`, ordered keys, distance-field `multiplyWide`), `trajectory-*` (5), `track-similarity` (1), `segment-intersection/segment-table.ts:139`, `geometry-predicates:790`, `spatial-clustering/hierarchical-density-clustering-kernels.ts:63`, `geometry-refinement/gpu-ratio-concave-hull.ts:163`, `geospatial-analysis/gpu-datum-grid-transform.ts:226`, `shape-descriptors-kernels.ts:44,331,386` | left: another agent's directory |
| Inline exponent masks (about 90 lines), all binary searches, 64-bit, masks, row accessors, distance, ordered keys in owned directories | left: not reached |

### Private copies remaining (approximate, whole library)

| Kind | Remaining | Of which in excluded directories |
| --- | --- | --- |
| Named float-test functions | about 22 | about 16 |
| Inline exponent-mask lines | about 90 | about 15 |
| Private NaN builders (named) | 4 + inline `bitcast<f32>(0x7fc00000u | (index & 0u))` in about 10 terrain kernels | 4 |
| Dense solvers (square) | 6 GPU (sites 6, 7, 8, 9, 10, 11) + 2 eigen (14, 15) + 3 non-square (12, 13, 16) | 9, 10, 14, 15, 16 |
| Binary searches | about 25 named + about 110 inline | about 40 |
| 64-bit helper copies | 7 sites | 1 (`distance-field-kernels.ts`) |
| Distance copies | 4 | 0 (`geometry-measures` has 1) |
| Mask tests | 58 sites | 5 (raster) |
| Deprecated float-test constant importers | about 110 files | about 25 |

## 4. Website and examples work list

No exported TypeScript name or result changed, so there is nothing to change in `website/` or
`examples/`. One meaning changed: the exported `getCholeskyWGSL` (spatial-regression barrel) now
also declares `DenseCholesky_p`, `factorDenseCholesky_p` and related functions, and needs
`isFiniteValue` (`IS_FINITE_VALUE_WGSL`) in the same module. It must not be emitted alongside
`getDenseCholeskyWGSL(p)` for the same `p`. `grep` finds no use in `website/src`, `examples`,
`modules/lenses` or `modules/kepler`.

## 5. Roadmap

STD-10 row, replace "Remaining: ..." with: "Adopted 2026-11-11: `getCholeskyWGSL` and
`getEquilibratedCholeskyWGSL` (the GWR and regression family) and `GPUKriging`'s solver now run on
`dense-solve-wgsl.ts` with identical results; float-test copies in terrain, network, pair
statistics, scalar scale, continuous sampling and the regression recipe. FX-09 and FX-10 were
already closed by the 2026-11-10 math pass (the OLS pivot limit in GWR, one kriging ratio constant);
optional: report status 3 separately in GWR. Remaining: site 6 (Ruiz order; moves local kriging,
RBF, space-time kriging, Vecchia), panel lag log-determinant and inverse, eigenvalue range, Kalman
`inverse4`, the point-cloud solvers, a QR module for sites 12, 13, 16; binary searches (about 135),
64-bit copies (7, now a word swap to `(high, low)`), masks (58), row accessors, sphere distance (4),
ordered keys, about 90 inline exponent masks, the remaining named float tests (see the w2-adoption
report), a declaration-deduplicating composer, `GEODESIC_WGSL` into core; the four deprecated
float-test constants still have about 110 importers."

## 6. Noticed

- `gpu-network/network-noding/gpu-network-noding.ts` `getOrderedKey` folds `-0` to `+0`
  (`select(value, 0.0, value == 0.0)`), so the map's "same names: delete" is wrong. Replacing it
  with the core `getOrderedKey` would change the atomic minimum of `-0`.
- `getGridLookupWGSL` (`spatial-regression/gwr-grid-index.ts:200`) embeds `IS_FINITE_VALUE_WGSL`.
  Any GWR kernel that adds it again fails only on the grid path, and the kernel audit runs no grid
  cases (only the headless grid specs caught it). The audit also does not check duplicate
  declarations; adding that check to `kernel-audit-wgsl.ts` would catch this trap in node.
- `terrain-features-utils.ts` still declares `NAN_BITS`; it is unused in WGSL now.
- `GPUSphericalPolygonOverlay` kernel-audit cases ("sourceIds is not written") failed once and
  passed later: another stream was mid-change.
