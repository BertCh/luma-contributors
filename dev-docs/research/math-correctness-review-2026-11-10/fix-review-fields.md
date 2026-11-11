# Fix review: fields scope (network, terrain, raster)

Date: 2026-11-10. Reviewer scope `fields`: `modules/analysis/src/gpu-network`, `gpu-terrain`,
`raster-analysis` and `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts`, against
the Resolution sections of `network-paths.md`, `network-flows.md`, `terrain-surface.md`,
`terrain-hydrology.md`, `raster-imagery.md` and `raster-fields.md`. Base of the diff:
`f5dcdc4119616fcd21b9f44ea84bee571db6b09c`. Paths below are relative to `modules/analysis`.

Method: reading the diff and the changed kernels in context, hand derivations, and two pure-Python
recomputations of fixtures. No spec was run (0 of the 6 allowed): nothing I needed to settle could
be settled by an existing spec, and the repository is read-only for me, so I could not add one.
"CONFIRMED" below therefore means a derivation or a line that plainly cannot do what is claimed,
not a GPU run, unless a row says otherwise.

## 1. Coverage

Source, read in full (the changed kernels with their context):

- `src/gpu-network/network-centrality/network-centrality-passes.ts` (whole file),
  `gpu-network-centrality.ts` (diff), `network-reachability/network-frontier.ts` (round helper,
  bindings, reset).
- `src/gpu-network/network-isochrones/network-isochrones-passes.ts` (splat kernel, inclusive
  breaks), `gpu-network-isochrones.ts` (diff, break consumers).
- `src/gpu-network/network-k-function/gpu-network-k-function.ts` (diff, simulate kernel),
  `analysis-core/kernels/permutation-random.ts` (`nextPhiloxBelow`).
- `src/gpu-network/gpu-network-inhomogeneous-k.ts` (pair kernel).
- `src/gpu-network/recipes/drive-time-catchment-recipe.ts` (diff).
- `src/gpu-network/network-coarsening/gpu-network-coarsening.ts` (diff, vertex and slot kernels),
  `network-workgroup-counters.ts`, `analysis-core/kernels/fixed-point-accumulator.ts` (WGSL).
- `src/gpu-network/network-noding/gpu-network-noding.ts` (CSR section),
  `network-optimization/gpu-network-louvain.ts`, `network-analysis/gpu-network-analytics-columns.ts`,
  `network-accessibility/gpu-network-snapping.ts`, `adjacency-matrix/gpu-adjacency-matrix.ts`.
- `src/gpu-terrain/terrain-curvature/terrain-geomorphometry-utils.ts`,
  `terrain-analysis/gpu-terrain-derivatives.ts`, `terrain-illumination/gpu-terrain-horizon.ts`,
  `gpu-solar-irradiance.ts`, `gpu-terrain-cast-shadow.ts`, `terrain-horizon-sweep.ts` (diffs).
- `src/gpu-terrain/point-horizon/` (all four diffs, the visibility kernel in full).
- `src/gpu-terrain/hydrology/terrain-flow-tracing.ts`.
- `src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts`,
  `gpu-terrain-depression-hierarchy.ts`, `gpu-terrain-flow-fractions.ts`,
  `gpu-terrain-tiled-flow.ts`, `terrain-tiled-resident.ts` (diffs).
- `src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts` (kernel),
  `remote-sensing/gpu-raster-conservative-remapping.ts` (diff),
  `isolines/isolines-utils.ts` and its three users, `flow-texture/gpu-line-integral-convolution.ts`
  (kernel), `gpu-raster-reprojection.ts`, `raster-stretch/gpu-raster-stretch.ts` (diff, finalize
  kernel), `raster-band-utils.ts`.
- Published kernels read for comparison: `node_modules/@luma.gl/graph/dist/gpu-raster/`
  `gpu-raster-contours.js` (saddle decider), `gpu-raster-contrast.js` (equalization),
  `gpu-raster-global-statistics.js` (`GPURasterGlobalPercentile`).

Source, read in part (changed code lines only, not the surrounding kernel):
`terrain-curvature/gpu-terrain-curvature.ts`, `terrain-illumination/gpu-relief-shading.ts`,
`topographic-position/gpu-terrain-vector-ruggedness.ts`, `geomorphons/gpu-geomorphons.ts`,
`terrain-decode/gpu-terrain-spike-repair.ts`, the six hydrology classes that take
`directionEncoding`, `raster-algebra/*`, `gpu-raster-reconstruction.ts`, `raster-sampling/*`,
`gpu-focal-statistics.ts`, the three fractional statistics classes, `cost-distance/gpu-cost-distance.ts`,
`raster-embeddings/*`, `change-detection/*`, `flow-aggregation/*`, `network-statistics`,
`network-service-areas`.

Source, not read (documentation-only diffs according to the Resolution tables; I did not check
that): `gpu-terrain-dem-coregistration.ts`, `gpu-texture-shading.ts`, `terrain-flow-spill-fill.ts`,
`gpu-terrain-flow.ts` beyond its diff, `gpu-raster-cold-monitor.ts`,
`gpu-raster-zonal-statistics.ts`, `gpu-raster-skeleton.ts`, `gpu-raster-watershed-segmentation.ts`,
`gpu-streamlines.ts`, `gpu-particle-advection.ts`, `gpu-lagrangian-flow-map.ts`,
`gpu-finite-time-lyapunov-exponent.ts`, the two `*-parameters.ts` files of isochrones and stretch.

Tests and fixtures:

- Read in full: `test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts`.
- Skimmed (assertions only): `gpu-network-centrality-path-counts.spec.ts`, the diff of
  `gpu-network-centrality.spec.ts`.
- Recomputed: `test/oracles/fixtures/parity-wp1-focal-small.json` (all seven cases, every output,
  by an independent pure-Python implementation) and the CSR of
  `parity-wp3-noding-showcase-large-snap.json`.
- Not read: every other new or changed spec (isochrones max mode, drive-time direct travel, shell
  count, K-function, coarsening, all terrain specs, all raster specs) and the TypeScript oracles.
  Section 4 is therefore short and is not a clean bill for the specs I did not open.
- `docs/api`: only greps for stale defaults (`rowDirection`, band intervals, `borderMode`,
  `connectivity`). The pages were not read.

## 2. Findings

| ID | Severity | Status | Where | What is wrong | Input or reasoning | Fix I would make |
| --- | --- | --- | --- | --- | --- | --- |
| fr-fields-01 | major | CONFIRMED (code reading) | `src/gpu-terrain/point-horizon/point-horizon-march.ts:427`, `:878`; `gpu-point-horizon-visibility.ts:217` | The `rowDirection` rule (default from `grid.affine[4]`, a contradiction throws) was applied to seven classes. `GPUPointHorizonProfile` and `GPUPointHorizonVisibility` take the same `grid` and still resolve `options.rowDirection ?? 'south'` without reading the affine. So the rule is not implemented identically everywhere, and the defect of terrain-surface-02 is still present in two terrain classes. | Planar grid with `affine[4] = +30` (row 0 in the south), `rowDirection` left out: `dirV = -cosA / cellY`, so a ray of azimuth 0 walks toward decreasing rows, which is south on that grid; in visibility `cosA = -deltaY / distance` mirrors the target bearing the same way. Web Mercator is not affected (the packer rejects `affine[4] >= 0`). | Call `resolveTerrainRowDirection(id, grid, options.rowDirection)` in `resolvePointHorizonModel` for the planar projection, add the two classes to `terrain-row-direction.node.spec.ts`, and correct the prop text ("Defaults to `'south'`"). |
| fr-fields-02 | major | CONFIRMED (hand derivation, float32 checked in Node; not run on the GPU) | `src/gpu-network/network-centrality/network-centrality-passes.ts:105-108`; flag text `gpu-network-centrality.ts:103-112` | Dropping `sourceCost < targetCost` makes every edge whose weight is absorbed by float32 (`cost + weight == cost`) tight in both directions on an undirected network, not only zero-weight edges. The kernel comment "only a cycle of zero-weight tight edges can make the tight graph cyclic" is false in float32. One such positive-weight edge now sets `zeroWeightCycle` 1 and `converged` 0, and the betweenness of everything on, after and upstream of it is lost for that pivot. The flag's advice ("give such edges a positive weight") cannot help. Before the pass such an edge was left out of the path counts and the run converged. | Undirected path `s -16777216- u -1- v -1- x`, pivot `s`: `fround(16777216 + 1) = 16777216`, so `u`, `v`, `x` all cost 16777216; `u -> v`, `v -> u`, `v -> x`, `x -> v` are tight; `u` waits for `v` and `v` for `u`, so none is ordered. Realistic scale: costs in metres at 100 km have a float32 step of 0.0078, so `fround(100000 + 0.003) = 100000`: a 3 mm sliver edge 100 km from a pivot; at 1,000 km, any edge under 3 cm; in seconds at one hour, any edge under 0.12 ms. | Keep the strict test for positive weights: tight when `sourceCost + weight == targetCost` and (`sourceCost < targetCost` or `weight == 0.0`). Then only true zero weights can form a cycle and the flag name and advice are right. Add the case above to `gpu-network-centrality-path-counts.spec.ts`. |
| fr-fields-03 | minor | CONFIRMED (code reading) | `src/gpu-network/recipes/drive-time-catchment-recipe.ts:804` | The demand-time kernel loops over every facility inside one invocation, with no cap on `facilities.length`. This is a data-sized loop in one invocation (contract section 2) and a silent all-pairs default (section 3). The result is right for several facilities on one road (the minimum over all of them is taken). | 1,000,000 demand points and 10,000 facilities: 1e10 loop iterations in one dispatch, where the old adapter did 1e6. The Resolution declares the `demandCount * facilityCount` cost but not the missing cap. | Sort facilities by road key (`min(node) * nodeCount + max(node)`) and binary-search the demand's road (`GPUSort` plus the shared binary search), or validate `facilityCount` against a named maximum at construction. |
| fr-fields-04 | minor | CONFIRMED (independent recomputation) | `test/oracles/fixtures/parity-wp1-focal-small.json`; claim in `raster-fields.md` "Fixtures and generators" | The patched expected values are right: my own float64 implementation of the new rule reproduces every value of all seven cases exactly. The description of the patch is not: six `standardDeviation` values in windows that do not reach the infinite cell were also overwritten with the pure-Python result, one float64 step from the committed SciPy value, and the count of changed cells is 54 by my count (48 because of the infinity, 6 by that rounding), not 42. | Case 1 cell 8: `5.58022301426107` became `5.580223014261069`; case 1 and case 4 cell 10; case 2 cells 13, 26, 27. Cell 8 of case 1 has a radius-1 window over rows 0 to 2, the infinity is in row 4. Hand check of three patched cells: case 1 cell 28 (window 1021, 1022, 1029) mean 1024, sum 3072, range 8, count 3; cell 21 sum 5101 of 5; cell 22 sum 8170 of 8. | Restore the six SciPy values (take them from the base commit), or say in the `reference` text that unaffected standard deviations are pure-Python values. Correct the count in the Resolution. |
| fr-fields-05 | minor | SUSPECTED | `src/gpu-network/network-optimization/gpu-network-louvain.ts:184-185`; `test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts` | The unseeded path was cured by removing the order table. The seeded path still declares `var order: array<u32, N>` and fills it with the same identity loop that "read back as zeros"; the cause of that failure was never found. The only evidence that the seeded table survives is 12 seeds at 16 vertices, and the spec reads one bit per pair (which vertex of a pair came later), not the order. Since the seed is a shader literal, the whole shuffle is a compile-time constant, which is the kind of code a compiler bug of that sort would touch. | No failing input found. The unseeded failure was size dependent (10 right, 12 to 24 wrong), so one size says little. | Find the cause (dump `order` to a scratch view for N = 8 to 64, with and without the Philox swaps). Until then pin the seeded order at several sizes with a graph whose labels reveal the full order, or read the seed from the parameters view so the table is not a compile-time constant. |
| fr-fields-06 | minor | CONFIRMED (hand derivation) | `src/gpu-network/gpu-network-inhomogeneous-k.ts:262`, `:275-290` | The tolerance merges a crossing with the node next to it, but two nodes that are both within the tolerance of the shell are each counted. The class text says locations within `1e-5 t` of each other are one location; for an edge shorter than twice the tolerance that is not what happens. In the other direction, a junction within the tolerance below the shell with two outgoing edges has two real crossings and is counted once (this one is declared: "locations closer than `1e-5 * t` are merged"). | Nodes `a` at distance `t (1 - 0.5e-5)` and `b` at `t (1 + 0.5e-5)` joined by an edge of length `1e-5 t`: one real crossing, the node loop counts both, `m` is one too high. Needs an edge under `2e-5 t` (20 cm at 10 km). Integer and ordinary irrational inputs are right. spatstat's `countends` also uses a tolerance, so the approach is the reference practice; only this corner is off. | In the edge loop, when both ends are within the tolerance and the edge is tight (`abs(firstDistance + length - secondDistance) <= tolerance`), subtract one. State in the class text that the tolerance is relative to the pair distance, where spatstat's is relative to the shortest segment. |
| fr-fields-07 | minor | SUSPECTED | `src/raster-analysis/gpu-raster-fractional-coverage.ts:249` | The sliver cure of "extra 01" replaces `(e x - b y) / det` by `x / a` on axis-aligned grids and its comment calls that "one correctly rounded division". GPU float division is not correctly rounded on every device (the brief's own rule): where it is a multiplication by a reciprocal, `x / a` for `x = k a` can be one step off `k` when `a` is not a power of two, and the sliver returns. The spec passes on the test GPU only. | Cell size 0.3, vertex at `x = 3 * 0.3`: IEEE division gives 3 exactly; `x * (1 / 0.3)` in float32 need not. | Snap in integers: `round(x / a)` when `abs(x / a - round(x / a))` is within a few float32 steps, or form the pixel coordinate from an integer lattice as the conservative remapping now does. Remove "correctly rounded" from the comment. |
| fr-fields-08 | minor | SUSPECTED | `src/gpu-terrain/hydrology/terrain-flow-tracing.ts:27-43`, `:120-131` | The three tables are right for rows advancing south (checked code by code, section 3). No hydrology class reads `grid.affine[4]`, so a Whitebox or TauDEM grid on a rows-north grid is decoded with north and south exchanged, silently. The same pass made row direction a checked property of the grid in seven other terrain classes. `'esri'` directions produced by `GPUTerrainFlow` itself are index-space and are not affected. | Rows-north grid (`affine[4] > 0`), Whitebox code 128 (north): decoded to ESRI 64, receiver `row - 1`, which is south on that grid. | In the classes that take `directionEncoding`, throw when the encoding is not `'esri'` and `grid.affine[4] > 0`, or flip the row offsets from the affine sign. |
| fr-fields-09 | minor | CONFIRMED (hand derivation) | `src/gpu-network/network-coarsening/gpu-network-coarsening.ts:237`, class text | With `fixedPointScale` given, the documented "old numbers" no longer include the old clamp. The old quantizer clamped a term at `4e18` units; the new one keeps the low two words of the shared three-word value, so a large term wraps: a term in `[2^47, 2^48)` at scale 65536 sets bit 63 and decodes as a negative sum, a term of `2^87` or more adds nothing at all (the impassable weight 3.4028234e38 on a live slot used to add the clamp, now adds 0). | `fixedPointScale: 65536`, one weight `1.5e14` (`2^47.1`): units `sig * 2^39` with `sig >= 2^23` reach bit 63; `dequantize` reads the high bit as the sign. | Clamp in `quantize` when the third word of `quantizeFixedPoint` is not the sign extension of the second, or document that a pinned sum needs every term below `2^63 / fixedPointScale` (2^47 at 65536). |

No blocker found.

## 3. Fixes checked and found right

- network-paths-01: `'max'` sample cost is `min(cost[u] + t w, cost[v] + (1 - t) wReverse)`. Checked
  a one-way road (no slot `v -> u`: forward term only), a reverse slot of different weight (its own
  weight is used; among parallel slots the largest, as documented), an unreached far end (reverse
  cost not finite: forward only; the slot from the unreached end returns early), a self-loop (never
  its own reverse). `'min'` and the labelled encoding are textually unchanged, and the labelled
  splat is only built in `'min'` mode (constructor throws otherwise). 7 storage bindings.
- network-paths-02: tight rule, unordered detection in the delta initializer, the status words and
  the reset of the delta phase's parameter word per batch. Right for true zero weights; see
  fr-fields-02 for absorbed positive weights.
- network-paths-03: Brandes with the scale, by hand on the diamond `s -> a, b -> t -> u`: `sigma(t)`
  is mantissa 1, exponent 1; `delta(t) = 1`, `delta(a) = 0.5 * (1 + 1) = 1`, as the plain
  recurrence. Mixed exponents (3 + 1 in either order) normalise to mantissa 1, exponent 2. The
  ratio's exponent difference is never positive because the rounded sum is at least each addend.
  The pivot skip, `betweennessScale` and the edge accumulation are unchanged apart from the ratio.
  Sigma and delta rounds bind 4 + 4 frontier views (8), the sigma initializer 8, the delta
  initializer 7, edge accumulate 5. Dependency rounds run with one hop per round, so the non-atomic
  `exponents` word is always written in an earlier dispatch than it is read.
- network-paths-04: two-level draw. Edge units are at most 2^24, a chunk total at most 2^31, the
  shifted chunk totals sum to at most `chunkCount * range <= 2^32 - 1`; both draws use the Lemire
  bounded integer (no modulo); a chunk with a positive coarse total has a positive fine total. An
  edge keeps a unit down to 2^-25 of the longest; a chunk's share is off by at most half a coarse
  unit (documented). 8 bindings in the simulate kernel.
- network-paths-06: live-slot degree, bounded by the row.
- network-paths-07: next-float breaks give `(a, b]` in `GPUIsobands` and the rings; both consumers
  read the shifted view.
- network-paths-08: Fisher-Yates with `nextPhiloxBelow` on a registered purpose (see fr-fields-05
  for what is not established).
- network-flows-01: root algebra (`fromSecond - fromFirst` is twice the peak height above the
  level), both-ends, one-end and unreachable cases; see fr-fields-06 for the corner.
- network-flows-02, -03: direct term sign and direction (`ahead >= 0` forward cost, `ahead <= 0`
  reverse cost), facility fraction mirrored on the opposite slot, start rows on a one-way edge,
  unsnapped facilities never match a demand road. Cost: see fr-fields-03.
- network-flows-04: a loop keeps its start entry; the patched Shapely fixture is consistent with
  its own edge list (one node, six loops, six entries, `offsets [0, 6]`).
- network-flows-05: unit `2^(E + 1 + h - 63)`, a term is below `2^(63 - h)` units, the scale words
  are reduced over the same nodes and slots that are summed, `stats` is zeroed each encoding.
- network-flows-07: half-up rounding without the `+ 0.5` above 2^23.
- network-flows-12 (noding weights), -13 (band `<=`): as described.
- terrain-surface-01: `1 - mean(sin^2(max(h, 0)))`; on an infinite plane of slope `s` the integral
  is `1/2 + cos(s)/2`, so the no-tilt-factor claim holds for that case and for a receiver measured
  per unit of its own surface.
- terrain-surface-02: all seven classes call `resolveTerrainRowDirection` in the constructor and
  use the resolved field in every kernel (no `props.rowDirection ?? 'south'` left in them). Aspect
  and hillshade in `GPUTerrainDerivatives`: `northGradient = ROW_NORTH_SIGN * rawY`, aspect
  `atan2(-east, -north)`, light `(sin az, cos az)`, right for both signs. Horizon sectors and cast
  shadows receive the resolved value; I did not re-derive their kernels.
- terrain-surface-03: the extrapolation is the quadratic through three samples
  (`f(-1) = 3 f0 - 3 f1 + f2`); the centred difference over it is `(-3 f0 + 4 f1 - f2) / (2 h)`
  (second order, exact on a plane and a quadratic) and the second difference is
  `f0 - 2 f1 + f2` (zero on a plane, first order). Corners are separable. The padded index is
  `(row + 1) * (W + 2) + column + 1`. `GPUReliefShading` keeps `'clamp'` by default.
- terrain-surface-06: band `[occluder - tol, occluder + tol)`.
- terrain-surface-09: `N / R` and `M / R` from `1 - e^2 sin^2`, ground bearing to sphere azimuth
  `normalize(M sin T, N cos T)`, the march in sphere metres with `c s^2` and limits divided by `s`,
  results scaled back, the `-BIG` sentinel kept.
- terrain-hydrology-01: zero row for no data, NaN row for an unresolved valid cell; the tiled
  consumer invalidates the cell from `flowDirections`. `GPUTerrainTiledFlow` is the only
  in-library caller of `GPUTerrainBranchingFlow`.
- terrain-hydrology-02: two walks, signed fixed-point elevation sum, headroom from the cell count.
- terrain-hydrology-02b: one kernel reads and writes `output.curves`; every node below `nodeCount`
  is decoded in the last loop, nodes above it and invalid stages hold NaN bits, and nothing else in
  the class reads the view. 8 bindings. It relies, as the old code did, on parents having a larger
  index than their children.
- terrain-hydrology-04: nonzero-except-sentinels stream test in both classes.
- terrain-hydrology-05: ESRI offsets (1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
  north-west, 64 north, 128 north-east) match the published numbering for rows south. Whitebox
  `1 -> 128`, otherwise `word >> 1` (2 east, 4 south-east, 8 south, 16 south-west, 32 west, 64
  north-west, 128 north). TauDEM `1 << ((9 - word) % 8)` (1 east, 2 north-east, 3 north, 4
  north-west, 5 west, 6 south-west, 7 south, 8 south-east). Sentinels pass through. Every read of
  a direction word that feeds `getFlowReceiver` goes through the decoder.
- raster-imagery-01: the sentinel test is skipped for constants in both classes.
- raster-imagery-02: the row ordering does keep each insertion to `order + 1` rotations (the rows
  already in R reach no column beyond `row - 1 + order`), the rotation and right-side updates are
  the Givens recurrences, back substitution reads `R[row, row + k]`. Zero weights skip the weight
  row; fewer than `order + 1` observations (including none) give status 1 and NaN.
- raster-imagery-03: adjacent destination cells share one integer edge, so the partition is
  conservative; the rational and fixed plans stay inside `i32`; a touched cell gets exactly 0.
- raster-imagery-04, -08: chord formula and zero-vector rule.
- raster-imagery-05, -06: same rank rule and same equalization formula as the published kernels
  (`floor((N - 1) p)`, first bin whose cumulative count exceeds it, bin centre; `(cdf - first) /
  (total - first)`).
- raster-fields-01: the step is `direction * stepLength / length(direction / cell)`, tangent in
  world space and `stepLength` pixels long.
- raster-fields-04: origins differenced in float64; `CompiledProjection.destinationOrigin` is
  required by the published type, so the projection branch is always taken with a projection.
- raster-fields-06: the determinant sign rule is the bilinear saddle value against the level; the
  same function serves isolines, isobands and isoband rings; it differs from the published kernel
  only in not scaling below 1e18.
- raster-fields-07: infinities are nodata in sampling and focal statistics; fixture verified
  (fr-fields-04).
- raster-fields-08: default 8, hole filling keeps 4.

## 4. Tests that do not pin their fix

Only what I could establish; most specs were not opened (section 1).

- network-paths-08: `gpu-network-louvain-visit-order.spec.ts` runs one size (16 vertices) and reads
  which vertex of each pair came later, 8 bits per seed. It does fail on the old generator for the
  stuck seed. It does not establish the full order or other sizes (fr-fields-05).
- network-paths-02: no case with a positive weight that float32 absorbs (fr-fields-02). The
  existing "zero cycles" case of `gpu-network-centrality.spec.ts` lost its value assertions for the
  cyclic input and now checks `converged` 0 and the reached counts (declared).
- terrain-surface-02: `terrain-row-direction.node.spec.ts` covers the six classes and geomorphons;
  the two point-horizon classes are not in it (fr-fields-01).
- raster-fields-07: the fixture is right but is no longer SciPy output in six cells (fr-fields-04).
- raster-fields "extra 01": pinned by `rs02-fractional-categories.spec.ts` on a device whose
  division is exact for that input (fr-fields-07).
- Declared by the implementors and not re-examined: network-flows-11 has no pin; the bearing
  conversion of terrain-surface-09 is pinned only by an oracle changed with the same rule; five
  Python generators were edited but not run.

## 5. Undeclared behaviour changes

- `GPUNetworkCentrality`: an undirected edge of positive weight below half a float32 step of the
  cost now sets `status.zeroWeightCycle` and clears `converged` (fr-fields-02). The Resolution
  declares this for zero-weight edges only.
- `GPUNetworkCoarsening` with `fixedPointScale`: large terms wrap or vanish where they were
  clamped (fr-fields-09). The Resolution says the pinned scale gives "the old numbers".
- `GPULineIntegralConvolution`: the Resolution says square cells are unchanged. That holds for
  positive cell sizes. The packer does not reject a negative `outputExtent` cell height; with one,
  the old step was mirrored in y and the new step follows the field, so the picture changes on
  square cells too. Documented input is positive ("row 0 has the smallest y"), so this is a
  correction of an unsupported input, not a defect.
- `GPURasterWhittakerSmoother`: status 2 now also fires when the smallest diagonal of R is below
  1e-6 of the largest. With very few valid observations far from the end of a long series and a
  large `lambda` (three observations at the start of 256 slices, order 2, `lambda = 1e7`: ratio
  about 7e-7 by my estimate) a well-posed system reports status 2. The Resolution states the new
  meaning of status 2 but not that it can reject such inputs. Not verified by a run.
- `preparedWeights.indices` of the conservative remapping, the `-curve-compensation` scratch id,
  the K-function seeds, the new construction throws (`rowDirection` contradiction, sheared
  cost-distance grid, remapping extent) are all declared.

I found no removed or renamed public export in the scope. `createNetworkAnalyticsTopology` was
deleted from `analysis-core` (another scope's file); it was `@internal` and in no entry barrel.

## 6. Not found

No stray debug output, `.only` or `.skip`, bare `assert`, private `@luma.gl` import, or missing
SPDX header in the changed and new files of the scope. No temporary file in the scope directories.
Explicit `rowDirection` callers in `examples/` that I opened agree with their grids; the callers in
`modules/lenses` and `modules/kepler` were not checked against the new construction throw.
