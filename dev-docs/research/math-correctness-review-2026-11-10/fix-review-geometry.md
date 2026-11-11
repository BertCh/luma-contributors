# Fix review: scope `geometry` (2026-11-10)

Independent review of the math-fix pass against the snapshot `f5dcdc4119616fcd21b9f44ea84bee571db6b09c`.
Scope: `analysis-core`, `pointcloud-analysis`, `geospatial-analysis` and the geometry, topology,
join, cell, trajectory and line directories of `gpu-spatial-analysis`, with their specs and docs.
Resolution sections checked: `geometry-measures.md`, `polygon-topology.md`, `cells-joins.md`,
`trajectories-lines.md`, `core-numerics.md`.

Paths are relative to `modules/analysis` unless they start with `modules/`, `docs/` or `dev-docs/`.
Node replicas are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/review-geometry/`
(`are.mjs`, `probe.mjs`, `probe2.mjs`, `probeinside.mjs`, `bulge.mjs`, `icp.mjs`, `rhumb.mjs`).

Result in one paragraph. The numerical content of the fixes in this scope holds up: every item the
lead asked to be proved or broken was replicated or derived and none broke. What is wrong is at
the edges of the pass: two test files that still assert the old behaviour and fail in the node
gate, one byte-pinned kernel text that needs a new baseline, one precision hole in the new rhumb
midpoint at very high latitudes, and a few standards and consistency items. No blocker.

## 1. Coverage

Read in full (current file, in context, plus the diff):

- `src/analysis-core/kernels/addition-rounding-error-wgsl.ts`, `exact-geometry-wgsl.ts`,
  `normal-quantile-wgsl.ts`, `permutation-shuffle.ts`, `uint64-wgsl.ts`, `binary-search-wgsl.ts`,
  `wgsl-poisson-observed-term.ts`
- `src/pointcloud-analysis/pointcloud-plane-icp.ts`, `pointcloud-diamond-morphology.ts` (and the
  published `node_modules/@luma.gl/graph/dist/gpu-raster/gpu-raster-morphology.js` it delegates to)
- `src/gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl.ts`
- `src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts`

Read as the full diff plus the surrounding kernel:

- `analysis-core`: `ring-assembly/gpu-segment-ring-assembly.ts` (match kernel, hole probe),
  `kernels/philox-random.ts`, `kernels/wgsl-common.ts`, `network-topology.ts`, `cell-key-wgsl.ts`,
  `polygon-triangulation/gpu-polygon-triangulation.ts` (comment only)
- `pointcloud-analysis`: `gpu-pointcloud-icp.ts`, `gpu-pointcloud-smrf.ts`, `gpu-pointcloud-ground.ts`,
  `gpu-pointcloud-statistical-filter.ts`, `index.ts`, `pointcloud-limits.ts`
- `geospatial-analysis/gpu-datum-grid-transform.ts` (whole kernel)
- `geometry-measures`: `geometry-measures-kernels.ts` (bulge, authalic, compensated sum, edge
  cross), `geodesic-kernels.ts`, `feature-measures-cooperative.ts` (callers of `addCompensated`),
  `geodesic-wgsl.ts` (`geodesicLongitudeDelta`, `geodesicDestination`)
- `geometry-refinement/gpu-precision-reduction.ts`, `group-geometry/gpu-group-convex-hull.ts`,
  `grid-generators/gpu-shape-generator.ts`, `minimum-bounds/gpu-minimum-bounds.ts`,
  `shape-descriptors/shape-descriptors-kernels.ts`
- `polygon-coverage-topology/coverage-topology-kernels.ts`, `gpu-coverage-dissolve.ts`,
  `gpu-coverage-validity.ts`; `polygon-coverage-simplification/gpu-coverage-simplification.ts`
  (snap, span links, sweep, repair); `geometry-topology/gpu-snap-rounded-noding.ts`,
  `gpu-make-valid.ts`, `gpu-line-noding.ts`; `rectangle-clip/rectangle-clip-kernels.ts`;
  `polygon-buffer/gpu-buffer-surface.ts`; `polygon-overlay/overlay-face-labelling-wgsl.ts` (caller
  of the side probe)
- `trajectory-cleaning/gpu-trajectory-outlier-flags.ts`, `trajectory-mobility/gpu-modal-split.ts`,
  `trajectory-restriction/trajectory-restriction-kernels.ts`,
  `trajectory-interpolation/trajectory-interpolation-kernels.ts`,
  `line-segmentize/line-segmentize-kernels.ts`, `linear-referencing/linear-referencing-kernels.ts`,
  `trajectory-home-range/gpu-utilization-volume-isopleths.ts`,
  `trajectory-estimation/prepare-trajectory-observations.ts`
- `cell-indexing/h3-measures-wgsl.ts`, `h3-index-extended-wgsl.ts`, `gpu-cell-geometry.ts`;
  `cell-table-compare/gpu-cell-table-compare.ts`; `region-statistics/gpu-region-mask.ts`;
  `neighbor-search/neighbor-search-parameters.ts`; `spatial-join/gpu-nearest-feature-weights.ts`

Diff read, documentation only (no arithmetic change): `areal-interpolation`, `cell-aggregation`,
`cell-topology/gpu-cell-grid-path.ts`, `delaunay-tessellation/gpu-voronoi-diagram.ts`,
`geometry-validity/geometry-validity-types.ts`, `label-point`, `gpu-geodesic-pairs.ts`,
`gpu-local-frame-projection.ts`, `gpu-ratio-concave-hull.ts`, `gpu-rectangle-clip.ts`,
`gpu-trajectory-split.ts`, `gpu-trajectory-recursions.ts`, `brownian-bridge-variance-estimation.ts`,
`gpu-zone-events.ts`, `gpu-trajectory-closest-approach.ts`, `encounter-kernels.ts`,
`gpu-pointcloud-features.ts`, `gpu-pointcloud-voxel-merge.ts`.

Specs: read in full `gpu-spatial-predicate-hole-filler.spec.ts`; read the changed or new parts of
`gpu-rhumb.spec.ts`, `global-coordinates.spec.ts` (datum inverse), `polygon-buffer.spec.ts`,
`gpu-cell-measures.spec.ts`, `home-range-recursion-mob07.spec.ts`,
`parity-wp1-round5.parity.node.spec.ts`. Every other spec in scope was not read; for them I relied
on the Resolution text and on the lead's gate logs (`gate-test-node.log`, `gate-typecheck-tests.log`).
Fixtures: `parity-wp1-round5-region-real.json`, `-small.json`, `parity-wp3-noding-showcase-large-snap.json`
and the two rhumb cases of `parity-wp2-round8-geodesics.json` decoded and recomputed (section 4).

Not read: `analysis-core/kernels/dense-solve-wgsl.ts`, `row-access-wgsl.ts`,
`sphere-distance-wgsl.ts`, `symmetric-eigen-wgsl.ts` (untracked before the snapshot, so they have
no baseline to diff; `sphere-distance` changed in words only according to core-numerics-08);
`docs/api` pages beyond targeted greps for defaults and rules; `modules/lenses` and `website`
beyond the one failing binding.

Untracked files have no baseline: the snapshot commit does not hold files that were untracked when
it was taken, so `uint64-wgsl.ts` and `binary-search-wgsl.ts` were reviewed as they stand, not as a
diff.

Spec runs (6 of 6 used, all on the current tree):

- headless, pass: `gpu-spatial-analysis/segment-intersection/exact-geometry-helpers.spec.ts`,
  `utils/addition-rounding-error-wgsl.spec.ts`,
  `gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-hole-filler.spec.ts`,
  `gpu-spatial-analysis/region-statistics/parity-wp1-region-statistics.parity.spec.ts`
- node, fail (findings 01 and 02):
  `gpu-spatial-analysis/neighborhood-summary/parity-wp1-round5.parity.node.spec.ts`,
  `modules/lenses/test/gpu-spatial-analysis/computation-pack/s3-geometry-pack.node.spec.ts`

## 2. Findings

| ID | Severity | Status | `file:line` | What is wrong | Evidence | Fix |
| --- | --- | --- | --- | --- | --- | --- |
| fr-geometry-01 | major | CONFIRMED (spec run) | `test/gpu-spatial-analysis/neighborhood-summary/parity-wp1-round5.parity.node.spec.ts:206-209` | cells-joins-03 changed the two region fixtures to the inclusive lasso, but this node spec still asserts the half-open rule on the patched fixture (`mask[29299]` and `mask[34963]` are 0, `counts[0]` is 2833). It fails, so `yarn test-node` is red. The Resolution names two parity specs as passing and does not mention this third reader of the same fixture. | My run and the lead's `gate-test-node.log:6519`: `expected 1 to be +0` at line 207. The fixture itself is right (section 4): rows 29299 `(12.529296875, 15)` and 34963 `(5, 32.6123046875)` lie exactly on lasso edges, and the inclusive count is 2835. | Expect 1, 1 and 2835, and reword the comment at `:206`. |
| fr-geometry-02 | major | CONFIRMED (spec run) | `modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/geometry-formulas.ts:308`, caused by `src/gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:461-464` | geometry-measures-07 replaced the inline destination formula by a call to `geodesicDestination`, which removed the local `latitude2`. The lens formula of `GPUShapeGenerator` binds `wgsl:latitude2` and its pack spec checks that the identifier exists in the kernel source. `yarn test-node` is red. Not in the Resolution (lenses were out of bounds for the pass, but the break was introduced by it). | My run and `gate-test-node.log:6557`: `GPUShapeGenerator: latitude-out -> wgsl:latitude2: expected ... to match /\blatitude2\b/` (`s3-geometry-pack.node.spec.ts:238`). | Rebind the formula symbol to the shared helper (`wgsl:geodesicDestination`), in the lenses package. |
| fr-geometry-03 | minor | CONFIRMED (Node float32 replica, `rhumb.mjs`) | `src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:146-160`, with the clamp at `:75` | The new midpoint fraction is `rhumbStretchedHalfDelta / rhumbStretchedDelta`. The denominator's `atanh` argument is clamped at 0.999999 (a stretched difference of 7.25 at most) and loses float32 precision as it nears 1; the numerator covers half the latitude range and is neither clamped nor as imprecise. The two no longer describe the same line when it reaches very high latitudes. | `(0, -88)` to `(120, 88)`: longitude 66.96, exact 60 by symmetry. `(0, 0)` to `(90, 89.95)`: 10.93, exact 10.245 (53 km at the midpoint). `(10, 60)` to `(100, 89.99)`: 18.81, exact 17.96. Over 300,000 random pairs (longitude differences up to 60 degrees; error scaled by the cosine of the middle latitude) the worst error is 1.4e-4 degrees while the stretched difference stays under 3 (equator to about latitude 84), 0.02 degrees under 6, 0.1 degrees under 7.2. The same replica gives 37.5392 for `(0, 0)` to `(90, 60)` and -38.866 for New York to London, the Resolution's values. The spec has no high-latitude midpoint. | Take the fraction from the two halves: `h1 / (h1 + h2)` with `h1 = rhumbStretchedHalfDelta(a, b)` and `h2 = -rhumbStretchedHalfDelta(b, a)`. Replica: 60 exactly, 10.277 and 18.03 on the three cases above, 37.5392 unchanged. Add a midpoint row above latitude 85 to the spec. |
| fr-geometry-04 | minor | CONFIRMED (lead's gate log) | `test/gpu-spatial-analysis/computation-pack/local-permutation-kernels.node.spec.ts` | The byte-for-byte pin of the permutation WGSL fails in all 162 configurations. The only differences are the new `philoxOpenUnitFloat` in `PHILOX_WGSL` (`src/analysis-core/kernels/philox-random.ts:156-161`, this scope) and `ANALYTIC_P_VALUE_FLOOR` in `getTwoSidedPValue` (statistics scope). `yarn test-node` is red. | `gate-test-node.log:166-6518`: every added or removed line is one of those two blocks (each 162 times). | Regenerate the pinned text once both changes are accepted. |
| fr-geometry-05 | minor | CONFIRMED (lead's typecheck log) | `test/gpu-spatial-analysis/trajectory-analysis/trajectory-heading-signed-zero.spec.ts:54` | The new spec of the guarded `atan2` calls `getGPUTrajectoryMetricsParameterValues({})`; the settings type requires `stopSpeedThreshold` and `stopMinimumDuration`. `yarn typecheck-tests` reports TS2345, and the two parameters are packed from `undefined`. | `gate-typecheck-tests.log`. | Pass both settings. |
| fr-geometry-06 | minor | CONFIRMED (reading) | `src/gpu-spatial-analysis/trajectory-interpolation/trajectory-interpolation-kernels.ts:76` | The other session guarded the planar step heading in `trajectory-analysis/trajectory-metrics-kernels.ts:419` (`guardedAtan2(deltaY, deltaX)`). The twin heading of the playhead and resample kernels is still `atan2(delta.y, delta.x)` with the same exposure: a due north step from `x = +0` to `x = -0` has `delta.x = -0.0` and reads -90 degrees on the test GPU. The same bare form with a second argument that can be `-0.0` is at `linear-referencing/gpu-line-locate.ts:356` and `linear-referencing-kernels.ts:957` (named as left alone in the trajectories Resolution). | Same expression, same input as `trajectory-heading-signed-zero.spec.ts`. | `guardedAtan2` in the three places (contract section 11). |
| fr-geometry-07 | minor | CONFIRMED (reading) | `src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve.ts:572-594`, `polygon-coverage-simplification/gpu-coverage-simplification.ts:1031-1053` | A vertex outside the exact snap-key range is ORed into `status.overflow`. Contract section 8 defines `overflow` as "rows were dropped" and section 9 says a condition of another kind is not folded into it. A caller that grows capacity on `overflow` loops; the documentation tells it to compare `requiredCount` with the capacity, which is optional in simplification. | The status node writes `overflow | exceeded`. | A status of its own (`approximation`, or a documented new scalar), leaving `overflow` to capacity. |
| fr-geometry-08 | minor | CONFIRMED (arithmetic) | `src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights.ts:388-393` (sibling of polygon-topology-03, outside this scope) | The copy of the snap key in contiguity weights is now relative to the first vertex but still rounds with `floor(steps + 0.5)` over its whole documented range of `2^24` steps. From `2^23` the sum of an odd step and 0.5 is a tie that rounds to even, so an odd cell takes the key of the next even cell. The coverage operators fixed exactly this (`coverage-topology-kernels.ts:51-53`) and their documentation says "as `GPUContiguityWeights`". The polygon-topology Resolution describes that file as still having the absolute key, which is no longer true. | `floor(fround(8388609 + 0.5))` and `floor(fround(8388610 + 0.5))` are both 8388610. | Use `getCoverageSnapKeyWGSL` there. |
| fr-geometry-09 | minor | SUSPECTED | `src/gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-kernels.ts:208-212` | trajectories-lines-09 made the classification exact, but the crossings that create the pieces are still found with float32 cross products (`denominator != 0.0`, `edgeFraction >= 0.0 && edgeFraction <= 1.0`). A segment through a polygon vertex can have `edgeFraction` just outside `[0, 1]` for both adjacent edges; then no boundary is inserted and the whole piece takes the class of its midpoint. Present before the pass; the Resolution's "converged on the exact predicates" covers `inside` only. | Reading. The 128-vertex spec crosses two vertices and passes; no failing input built. | Decide whether a crossing exists with `exactSegmentsCross` and keep the float32 quotient for the parameter only. |
| fr-geometry-10 | minor | CONFIRMED (reading) | `dev-docs/research/math-correctness-review-2026-11-10/trajectories-lines.md:434`, `:454-455` | The trajectories Resolution cites `line-segmentize-kernels.ts:74-125` as the home of `getAdditionRoundingError` (non-negative operands) and `:262`, `:338` as its uses. The function now lives in `analysis-core/kernels/addition-rounding-error-wgsl.ts` (signed); `line-segmentize-kernels.ts:75-77` is a re-export and the uses are at `:214` and `:290`. | Current files. | Update the three references. |
| fr-geometry-11 | minor | CONFIRMED (reading) | `src/analysis-core/kernels/normal-quantile-wgsl.ts:24` | Residue of a scripted replacement: `radius * 2509.0809287301226727 +33430.575583588128105` (no space after `+`). The value is right. | Diff. | Restore the space. |

## 3. Fixes checked and found right

Core numerics.

- **`getAdditionRoundingError`** (lead's task; cures geometry-measures-12, -13, trajectories-lines-03 and
  the line prefix): a bit-for-bit Node replica against exact BigInt arithmetic over 17.7 million pairs
  (random bit patterns, exponent gaps -30 to 30, adversarial significands at every exponent, both signs,
  cancellation to within 1,000 steps, overflow, infinities, NaN, zeros) gives 0 mismatches outside the
  documented case (smaller operand below `2^-103`, where it returns 0). Hand argument for each branch
  (gap above 25, exact cancellation, carry, borrow, `2^25` unit bound, exact `i32` to `f32` conversion,
  normal `ldexp` result) agrees. It also stays correct to first order if the caller's sum was formed by
  a fused multiply-add, because it measures the error of the sum it is given. GPU spec passes.
- **polygon-topology-01, `exactCompareToMidpoint`**: Node replica against BigInt on 17.4 million
  triples (adjacent floats, lattice integers, mixed signs, zero, values above 1e38): 0 mismatches inside
  the documented range. No rounded midpoint is used for the height. Point probes are unchanged
  (`v.y >= point.y`). All four side-probe call sites pass the midpoint of `start`, `end`; the overlay
  face labelling works on lattice integers, where the new test returns the old answer; the candidate
  filters that still use the rounded midpoint (ring assembly box reject, coverage validity BVH query)
  cannot drop an edge the exact ray crosses. GPU spec passes.
- **core-numerics-01, plane ICP**: derived by hand (`J = [a + d x n, n]`, `M' = T M T^T` with
  `T = [[I, D], [0, I]]`, `x = S y`, `t = t_c - (R - I) g`, Rodrigues minus identity as
  `(k k^T - I)(1 - cos) + K sin`), and replicated in Node: transferred moments equal the direct moments
  about the centroid to 2e-14, and the composed motion equals the direct origin-frame step to second
  order in the angle.
- **luma-official 14, SMRF diamond**: the published `'cross'` element is `|dx| + |dy| <= radius`, the
  `'nodata'` border leaves out-of-grid samples out and `'ignore'` skips them, so one pass is the clipped
  diamond and passes compose (Minkowski sum; a clipped rectangle keeps monotone paths). The value path
  `f32(raw) * 1 + 0` is exact. Differs from the old kernel only for non-finite cells, which SMRF
  inpaints away.
- **core-numerics-02, swap-or-not**: both branches stay below `count`; WGSL and TypeScript agree.
- **core-numerics-04, AS 241**: all 45 literals compared as binary64 with CPython's
  `statistics._normal_dist_inv_cdf`, and their positions read against it.
- **core-numerics-12, Poisson term**; **luma-official-05, `uint64` and `'u64'` search**: every
  function re-derived for `(high, low)`; storage stays low word first; the reversed-range guard of the
  other session (`max(first, last)`) is intact. Neither has a caller in `src`.
- **polygon-topology extra-B** (zero-length segments in ring assembly): flag store and candidate skip
  are on every template branch.

Geometry measures.

- **geometry-measures-01, great-circle bulge**: formula derived (area under the arc `2 atan(k t)`,
  chord `k cos^2(h) step`, series and complementary form) and replicated in Node against the closed
  form and a numeric integral: agreement to rounding in binary64. Units match the Lambert cross
  (radians times sine). Great-circle edges are only reachable with the spherical model, so no authalic
  mix. Serial and cooperative kernels share the source.
- **geometry-measures-02, -08**: stretched half difference derived term by term; east-west, meridian
  and antimeridian cases read; `geodesicLongitudeDelta(source, destination)` argument order right at
  the five call sites. See fr-geometry-03 for the high-latitude limit.
- **geometry-measures-04, -09**: `floor` plus `step(0.5, fraction)`; the fraction is exact.
- **geometry-measures-05**: the first update precedes the test; the limit is
  `max(tolerance, 4.8e-7 * largest term)`, which stays at 5e-9 for shifts up to 0.01 degrees; the
  spec's linear case leaves a residual of 1.6e-11 by hand.
- **geometry-measures-06**: coefficients equal Snyder 3-18; the `atanh` series equals
  `s (1 + x/3 + x^2/5 + x^3/7 + x^4/9)` with `x = (e s)^2`, truncation 1e-12 on WGS84.
- **geometry-measures-07**: argument order and units of `geodesicDestination`.

Polygon topology.

- **polygon-topology-03**: key, `2^23` rule and range flag; both users pass the origin; tolerance 0
  keeps the bit key.
- **polygon-topology-04**: closed copy, binary search and `edge - ring` mapping (noding reports the
  line index as ring and the start row as edge); a repeated closing vertex adds no intersection pair
  because the successor of a segment skips degenerate ones.
- **polygon-topology-06**: the cell is at least one float32 step of the largest magnitude (checked for
  magnitudes that are and are not powers of two), and is unchanged when the magnitude is at most four
  extents. `polygon-overlay-float32-lattice.spec.ts` pins one near-zero scene as unchanged.
- **polygon-topology-02** (sweep): completeness of the candidate search (a horizontal line meets the
  chord at most once), the reach reject (the area lies in the hull of the chain), the closing edge, the
  parity test, 8 storage bindings, repair on any non-zero flag.
- **polygon-topology-07, -08, -09, extra-A, extra-C**: read; the validity probe may use either side
  because a clean unmatched segment has no foreign edge through its midpoint.

Cells and joins.

- **cells-joins-01**: `ProbeInside` replicated in Node against an exact rational even-odd test with
  explicit tiny offsets on 800,000 random lattice cases full of collinear and coincident edges: 0
  mismatches. The containment argument holds for the lead's cases (equal polygons true; shared edge
  from inside true, from outside false; hole filler false; inner around a hole false; vertex touch
  decided by the edge test; inscribed hole false; opposite collinear edges handled by the third
  coefficient), which is what DE-9IM `contains` gives. GPU spec passes on three engines.
- **cells-joins-03**: inclusive test; fixtures recomputed (section 4).
- **cells-joins-04, -05, luma-official-06, -09, H3 fallback area** (the determinant identity and the
  l'Huilier denominator): read. A self join never lists the row itself, so the new floor only weighs
  distinct coincident points.

Trajectories and lines.

- **trajectories-lines-01**: every row position of a track of 2, 3, 4 and more rows walked by hand
  for a displaced fix, a sustained jump and both track ends.
- **trajectories-lines-02, -04, -08**: read. **-09**: one call site of `inside`, array bounds, the
  gap branch, the box variant unchanged. See fr-geometry-09.

## 4. Tests that do not pin their fix

- `polygon-buffer.spec.ts` "thin strip" (polygon-topology-09): passes before and after, as its own
  comment says.
- `gpu-coverage-topology-float32.spec.ts` "gap rule" (polygon-topology-05) and
  `gpu-nearest-feature-weights.spec.ts` "default-floor weight" (cells-joins-04): pin unchanged code.
- `gpu-rhumb.spec.ts:204` (geometry-measures-02): three midpoints at low and middle latitudes; it
  does not see fr-geometry-03.
- `parity-wp2-round8-geodesics.json`, rhumb midpoints: all 266 pairs recomputed here with an
  independent closed form and equal to the fixture to the last digit. They are the same definition the
  kernel implements, not a second implementation (PyGeodesy was not run). The hand midpoints and the
  distance-halving check of `gpu-rhumb.spec.ts` are the independent evidence.
- Not run against the old kernel, by the Resolutions' own account: geometry-measures-06, the rhumb
  half of -08, trajectories-lines-01. By reading, the 06 bound (0.9 m over the cosine) is below the old
  4.8 m error, the 08 bound (0.01 m) below the old 1.7 m, and the datum spec's 5e-6 degree shift is
  below the old 1e-5 tolerance, so each would have failed.
- `UINT64_WGSL` and the `'u64'` search are exercised by `utils/wgsl-words-search.spec.ts` only.

Fixtures patched without Shapely, checked by hand:

- `parity-wp1-round5-region-small.json`, query `concave-boundaries`: all 13 rows recomputed with
  exact rationals; the mask, ids, counts `[7, 5, 0]`, statistics `[10, 2, -2, 6]` and histogram agree.
  The four rows that changed are `(1, 1)`, `(3, 1)`, `(1, 3)`, `(1, 2)`, all on the ring.
- `parity-wp1-round5-region-real.json`, query `concave-lasso-selection-histogram`: all 43,557 rows
  recomputed; 0 mismatches; the two changed rows are the only boundary rows; sum 449,002 and mean
  158.378130511 agree. `sha256` matches the decoded data in both files.
- `parity-wp3-noding-showcase-large-snap.json`: six self-loops on one node now have one CSR entry
  each (`offsets [0, 6]`, weights equal to `lengths` in edge order), which is what the changed
  generator line produces.

Closing pass (2026-11-11): Shapely 2.2.0 reproduces all three fixtures exactly and PyGeodesy 26.9.9 the rhumb midpoints to 4.6e-10 degrees; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

## 5. Undeclared behaviour changes

Not in any of the five Resolution tables (the first three look like the other session's edits):

- `GPUDatumGridTransform` throws when both `mask` and `validity` are given
  (`src/geospatial-analysis/gpu-datum-grid-transform.ts:111-115`); before, `mask` won.
- `GPUPointCloudStatisticalFilter` throws when both `validity` and `mask` are given
  (`src/pointcloud-analysis/gpu-pointcloud-statistical-filter.ts:53-57`).
- `prepareGPUTrajectoryObservations` compares the origin as binary64
  (`src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:52-63`): a
  longitude of 180.000001 was accepted and now throws; the message text changed.
- `PHILOX_WGSL` gained a function, so the text of every kernel that embeds it changed
  (fr-geometry-04).
- `createNetworkAnalyticsTopology` and `NetworkAnalyticsTopologyProps` were removed from
  `src/analysis-core/network-topology.ts`. They were `@internal`, in no entry barrel, and no reference
  remains; the removal is declared in `terrain-surface.md`, not in `core-numerics.md`.

Declared, with a consequence worth stating: the inverse-distance floor of `GPUNeighborSearch`
(cells-joins-04) is `1e-6` in position units, so two distinct points at one location now weigh
`1e6^power` against about `1 / distance` for the rest, and take nearly all of a row-standardized
row (before: weight 0 and not a neighbor).
