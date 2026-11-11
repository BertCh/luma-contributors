# Math correctness review: geometry measures, geodesy, shapes and grids

Scope id: `geometry-measures`. Date: 2026-11-10. Reviewer scope: `gpu-spatial-analysis/`
`geometry-refinement`, `geometry-measures`, `lon-lat-polygons`, `outline-geometry`, `label-point`,
`minimum-bounds`, `minimum-clearance`, `concave-hull`, `shape-descriptors`, `group-geometry`,
`global-coordinates`, `bounds-filter`, `grid-generators`, `dot-density`, `shared`, and
`geospatial-analysis/`.

All paths are relative to `modules/analysis/src`. Scratch scripts are in the session scratchpad
under `geometry-measures/` (`check1.mjs` to `check6.mjs`). No spec file was run (0 of 5 runs used).

## 1. Coverage

**Read in full** (kernel bodies and packers):

- `gpu-spatial-analysis/geometry-measures/`: `geodesic-wgsl.ts`, `rhumb-wgsl.ts`,
  `geodesic-kernels.ts`, `gpu-geodesic-pairs.ts`, `gpu-geodesic-destination.ts`,
  `geometry-measures-kernels.ts`, `gpu-geometry-measures.ts`, `feature-measures-cooperative.ts`,
  `gpu-local-frame-projection.ts`.
- `geospatial-analysis/`: `gpu-datum-grid-transform.ts`, `gpu-point-buffer.ts`,
  `projection-spatial-context.ts`, `index.ts`.
- `gpu-spatial-analysis/dot-density/`: all six source files.
- `gpu-spatial-analysis/grid-generators/`: `gpu-grid-generator.ts`, `gpu-shape-generator.ts`.
- `gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds.ts`.
- `gpu-spatial-analysis/label-point/gpu-label-point.ts` (all but the import block).
- `gpu-spatial-analysis/concave-hull/gpu-concave-hull.ts` (all but the import block).
- `gpu-spatial-analysis/shape-descriptors/`: `shape-descriptors-kernels.ts`, `convexity-hull.ts`.
- `gpu-spatial-analysis/lon-lat-polygons/lon-lat-polygon-preparation-kernels.ts`.
- `gpu-spatial-analysis/global-coordinates/`: `spherical-limits.ts`, `assert-unit-vector.ts`,
  `gpu-spherical-convex-contains.ts`.
- `gpu-spatial-analysis/geometry-refinement/`: `gpu-maximum-inscribed-circle.ts`,
  `gpu-ratio-concave-hull.ts`, `gpu-convex-straight-skeleton.ts`, `gpu-segment-snap.ts`.
- `gpu-spatial-analysis/shared/`: `hash-uint32.ts`, `bin-index-wgsl.ts`,
  `complementary-error-function.ts`, `spatial-wgsl-point-helpers.ts`.
- Outside the scope, read because the scope depends on them: `analysis-core/contracts/earth.ts`,
  `analysis-core/contracts/local-frame.ts`, `analysis-core/lon-lat-ring-wgsl.ts`, and the
  `sphericalExcessTriangle` function of `gpu-spatial-analysis/cell-indexing/h3-measures-wgsl.ts`.

**Read in part:**

- `shape-descriptors/gpu-shape-descriptors.ts`: lines 30 to 200 (types and class documentation).
  The node wiring was not read.
- `group-geometry/gpu-group-geometry.ts`: lines 1 to 140 and 300 to 559.
- `group-geometry/gpu-group-convex-hull.ts`: lines 1 to 520 and 980 to 1151 (keys, lattice,
  sorts, extreme-point filter, exact orientation). Lines 520 to 980 (chunk prefilter, chain, bound,
  emit) were not read.
- `minimum-clearance/gpu-minimum-clearance.ts`: lines 30 to 170 and 400 to 560 (documentation and
  the probe kernel). The label and per-feature reduction kernels were not read.
- `lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts`: lines 40 to 420.
- `global-coordinates/gpu-spherical-polygon-contains.ts` (30 to 110, 150 to 285),
  `gpu-spherical-convex-intersection.ts` (150 to 290), `gpu-spherical-polygon-overlay.ts`
  (30 to 80, 150 to 284).
- `outline-geometry/gpu-outline-geometry.ts` (30 to 215, 250 to 427), `gpu-offset-curve.ts`
  (40 to 200, 330 to 520).
- `bounds-filter/gpu-bounds-filter.ts`: lines 20 to 120 and the three predicate expressions
  (226 to 250).
- `geometry-refinement/gpu-precision-reduction.ts`: documentation and the line-rounding kernel
  (20 to 120, 214 to 312). `gpu-vector-straight-skeleton.ts`: header and a formula grep only.
- `shared/point-neighbor-grid-index.ts`: first 90 lines.
- `analysis-core/contracts/spatial-context.ts`, `analysis-core/kernels/philox-random.ts`,
  `sphere-distance-wgsl.ts`, `exact-lattice-wgsl.ts`, `exact-orientation-wgsl.ts`: headers and
  the functions the scope calls.

**Not opened:** `global-coordinates/gpu-spherical-arc-overlay.ts`;
`geometry-refinement/gpu-canonical-geometry-equality.ts`, `gpu-canonical-polygon-equality.ts`,
`rectilinear-subdivision.ts`, `gpu-line-loop-removal.ts`, `gpu-normalize-polygons.ts`,
`gpu-normalize-geometry.ts`, `gpu-offset-curve-components.ts`, `gpu-subdivide-polygons.ts`;
`shared/grid-cell-row-order.ts`, `shared/triangle-edge-adjacency.ts`; every `index.ts` except the
two named above; `lon-lat-polygons/lon-lat-polygon-preparation-parameters.ts` (only its packer
was read). These files are not verified.

There is no ECEF, geodetic or ENU conversion code in this scope. `global-coordinates` takes
caller-supplied unit vectors. There is no local Web Mercator formula in this scope either.

## 2. Findings

| ID | Severity | Status | File:line | Kernel computes | Reference says | Input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geometry-measures-01 | precision | CONFIRMED | `gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:411-417` (uses `cell-indexing/h3-measures-wgsl.ts:10-13`) | Great-circle-edge polygon area as a fan of `sphericalExcessTriangle(u0, ui, ui+1)` on f32 unit vectors. The determinant `dot(c, cross(a, b))` cancels. | Spherical excess of the polygon (the documented `sphere-polygon` model). | 100 m square at (10.3, 45.2): kernel 60,920 m2, true 10,008 m2. At (-122.4, 37.7): -97,560 m2 (wrong sign). 1 km square: 9% to 32% off. 10 km: 0.1%. 100 km: 5e-5. | Sum the precise Lambert chord cross (already computed from degree differences) plus a per-edge great-circle bulge, as the `'geodesic'` mode does. Until then document a minimum polygon size and add a small-polygon spec. |
| geometry-measures-02 | wrong-result | CONFIRMED | `gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:113-117`; output doc `gpu-geodesic-pairs.ts:52-55` | `(lon1 + dLon / 2, lat1 + dLat / 2)`. The comment says longitude is linear along a rhumb line. It is not. | Rhumb midpoint: latitude is linear in distance, longitude is linear in the stretched latitude: `lon1 + dLon * (psi(latMid) - psi(lat1)) / (psi(lat2) - psi(lat1))`. | (0, 0) to (90, 60): kernel longitude 45, rhumb midpoint 37.54 (718 km off). London to Paris: 1.16 km off. New York to London: 140 km off. The kernel point is not on the rhumb line (bearing from origin 55.0 against 50.0). | Use `rhumbStretchedDelta(lat1, latMid) / rhumbStretchedDelta(lat1, lat2)` as the longitude fraction, 0.5 in the east-west limit. Regenerate the oracle with PyGeodesy `rhumbMidpointTo`. |
| geometry-measures-03 | wrong-result | CONFIRMED (CPU replica of the selection rule) | `gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull.ts:237-285`; claim at `:40-45` | Each round removes every candidate that is the largest around all three of its vertices, in parallel. | Class doc: "removed in descending size order". GEOS and JTS `ConcaveHull` remove the largest removable border triangle first, one at a time. | 28 random points, ratios 0.2, 0.35, 0.5: 351 of 900 runs end with a different triangle set. First case (seed 1, ratio 0.2): 27 triangles and area 4674 sequential, 26 triangles and area 4357 parallel. | Either state that the hull is a valid erosion hull that differs from GEOS, or only remove the global maximum plus candidates that provably cannot interact with a larger pending candidate. Add a random-point oracle from `shapely.concave_hull`. |
| geometry-measures-04 | wrong-edge-case | CONFIRMED | `gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction.ts:303` | `floor(value / GRID + 0.5) * GRID` in f32. | Round to the nearest grid multiple; a value on the grid stays. | `GRID = 1`, x = 8388609 (an odd integer, already on the grid): `8388609 + 0.5` rounds to 8388610 in f32, so the vertex moves 1 m. True for every odd integer in [2^23, 2^24), both signs. Web Mercator metres from 75 E to 151 E are in this band. | Use `round(value / GRID)` or skip the `+ 0.5` when `abs(value / GRID) >= 2^23`. |
| geometry-measures-05 | wrong-edge-case | CONFIRMED | `geospatial-analysis/gpu-datum-grid-transform.ts:271-274`; default at `:80` and `:160` | Inverse iteration tests the residual before the first update. The first residual is the grid shift itself. Default tolerance is 1e-5 degrees (1.1 m). | PROJ iterates to about 1e-12 rad. An inverse of a forward shift returns the source. | A grid with a 5e-6 degree shift (0.56 m): forward moves the point, inverse runs 0 iterations, reports success and returns the input unchanged. Any shift under 1.1 m is skipped. | Always take one update before testing, and default the tolerance to a few f32 ulps of the coordinate (far smaller when `positionsLow` is given). |
| geometry-measures-06 | wrong-result | CONFIRMED | `gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:173-177` | Authalic to geodetic latitude series with `17 e^4 / 360` on `sin(4 xi)`. | Snyder eq. 3-18: the inverse series uses `23 e^4 / 360`. `17 / 360` is the forward series coefficient. | Ellipsoidal polygon centroid at latitude 22.5: latitude 5.05 m off. With `23 / 360` the series is within 0.29 m (f32 spacing there is 0.21 m). | Change 17 to 23, add the `e^6` terms. Fix the same constant in the test twin `test/gpu-spatial-analysis/geometry-measures/geometry-measures-oracle.ts:318`. |
| geometry-measures-07 | precision | CONFIRMED | `gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:456-473` | A private spherical destination: `asin` of `sin(lat) cos(d) + cos(lat) sin(d) cos(b)`. | The shared `geodesicDestination` (`geodesic-wgsl.ts:168-193`) uses half-angle forms and reaches the f32 output limit. | 50 m circle, 64 vertices: worst vertex error 0.80 m at latitude 45.2, 1.7 m at 80.3, 3.4 m at 85.2, 10.7 m at 88.6. The shared helper gives 0.21 m to 0.42 m (half an ulp of the output) in every case. | Call `geodesicDestination` from `GEODESIC_WGSL`. |
| geometry-measures-08 | inconsistent | CONFIRMED | `gpu-spatial-analysis/geometry-measures/geodesic-kernels.ts:109`; `rhumb-wgsl.ts:97,106` | Vincenty and rhumb take `geodesicWrapLongitudeDelta(b.x - a.x)`: subtract first, wrap after. | The sphere model uses `geodesicLongitudeDelta` (`geodesic-wgsl.ts:80-98`), written because the other form "loses about 1.6 meters" at the seam. | a = (179.94722, .), b = (-179.99805, .): longitude difference off by 1.53e-5 degrees, 1.7 m at the equator on a 6.1 km separation. | Use `geodesicLongitudeDelta(a.x, b.x)` in `vincentyInverse`, `rhumbAngle` and `rhumbBearingDegrees`. |
| geometry-measures-09 | precision | CONFIRMED | `gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.ts:426-427` | `u32(floor(d * scale + 0.5))` in f32. | Round to nearest lattice point; `d * scale` is already an integer above 2^23. | Extent 1000, so scale 2^19: x = 16 + 2^-19 maps to lattice 8388610, not 8388609. The exactly collinear triple (0, 0), (x, 3), (2x, 6) gets orientation 6, not 0. | `u32(round(d * scale))`, or add 0.5 only below 2^23. Effect is one lattice step (2e-9 of the extent). |
| geometry-measures-10 | doc-mismatch | CONFIRMED | `gpu-spatial-analysis/label-point/gpu-label-point.ts:124-126` | Last refinement round has grid spacing `2 h (2/N) (2/R)^(rounds-1) / R`. | The doc formula has `(2/R)^rounds`. | Defaults (16, 8, 6): actual 3.05e-5 of the half extent, documented 7.6e-6. Factor 4. | Correct the exponent. "About 1e-5 of the bounds" is still close (1.5e-5). |
| geometry-measures-11 | doc-mismatch | SUSPECTED | `gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts:62-69` | Compiles a plan, then replaces `destinationOrigin` with `[0, 0]`. | `compileProjectionPlan` certifies `float32MaxError` and `tolerance` for outputs relative to the bounds centre. | Bounds 270 km from the frame origin: f32 half spacing of the absolute output is 0.016 m, above the default 0.01 m tolerance. The exposed `plan.float32MaxError` no longer describes the output. | Say so on the `tolerance` and `plan` docs, or ask upstream for a `destinationOrigin` option on `compileProjectionPlan` so the certificate is computed for the origin used. |

### Notes on the findings

**01. Great-circle area.** The linear-edge path is careful: it works in degree differences from
a local origin and matches f64 to 1e-8 on the same 100 m square. The great-circle path throws that
away by converting every vertex to an f32 unit vector first. `ringSign` at
`geometry-measures-kernels.ts:749` takes its sign from the same noisy sum, so for small rings a
hole can be added instead of subtracted. The H3 measures doc
(`cell-indexing/h3-measures-wgsl.ts:35-39`) already records this cancellation for small cells
("unusable beyond resolution 5"). The geometry measures doc and `docs/api/gpu-spatial-analysis/operators-lines.md:292-296`
say nothing. Confirmed with an f32 replica of the kernel arithmetic (`check2.mjs`), no fused
multiply-add. A GPU that contracts to FMA will give different noise of the same size. The only
specs of this mode use large triangles.

**02. Rhumb midpoint.** The oracle generator knows: `test/oracles/parity-wp2-round8-geodesics.py:7-10`
says the helper is "linear in unrolled longitude/latitude, not half-distance along a loxodrome"
and pins it with GEOS `line_interpolate_point` on a degree-plane line. The public doc
(`docs/api/gpu-spatial-analysis/operators-lines.md:356-357`) and the output doc call it the rhumb
midpoint. So the spec pins the wrong value on purpose and the documentation claims the right one.

**03. Ratio concave hull.** I replicated the round rule (boundary vertex count, one exposed edge,
interior opposite vertex, size above the threshold, largest around all three vertices, lowest row
on ties) and compared with "always remove the largest eligible triangle" on a brute-force Delaunay
triangulation (`check5.mjs`). I did not run the kernel. The parallel rule lets a locally largest
triangle go before a larger triangle elsewhere cascades into its neighbourhood, which changes
which neighbours stay removable. Both results are connected hulls that keep every vertex. The
fixture `test/oracles/geometry-refinement.py:37-41` has three cases of five to a dozen points,
too small to show it. A second, smaller difference: the kernel removes `size > threshold`
(`:173`), JTS removes `size >= threshold`; they differ only when an edge length equals the
threshold exactly (ratio 0).

**05. Datum inverse.** The independent oracle (`test/oracles/datum-horizontal.py`) uses shifts of
1e-7 to 8e-7 degrees. The spec that checks it passes `tolerance: 1e-10`
(`test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts:670`). With the default
the same rows would return the input.

**06. Authalic series.** The forward mapping in the kernel is the closed form and is right. Only
the inverse used for polygon centroids in the two ellipsoidal modes is affected. Areas and lengths
are not.

**09.** The class doc says the hull "matches a CPU monotone chain bit for bit". That holds only for
a CPU chain that repeats the same f32 rounding. The documented lattice caveats stand.

## 3. Verified correct

- `GEODESIC_WGSL` (`geodesic-wgsl.ts`): central angle (Vincenty sphere form), haversine term,
  initial bearing, destination (half-angle form), slerp, colatitude cosine, the i32 seam
  subtraction. Derived by hand against the standard formulas; the f32 destination replica reaches
  half an output ulp against f64.
- `RHUMB_WGSL` distance, bearing, destination: `atanh((sin b - sin a) / (1 - sin a sin b))`
  identity and the `q = dPhi / dPsi` ratio derived by hand. Matches turf `rhumbDistance`,
  `rhumbBearing`, `rhumbDestination`. (Midpoint: finding 02.)
- `getVincentyWGSL` inverse and direct (`geodesic-kernels.ts`): `A`, `B`, `C`, `deltaSigma`,
  `lambda` update, both azimuths, reduced-latitude gap and its derivative. Compared term by term
  with Vincenty (1975).
- `GPUGeodesicPairs`, `GPUGeodesicDestination`: unit handling (kilometre ellipsoid), final bearing,
  sphere fallback on non-convergence as documented.
- `GPUGeometryMeasures` planar: shoelace sign (counter-clockwise positive), local origin,
  area-weighted centroid `sum((xi + xj) cross) / (3 sum(cross))`, holes by ring position, Neumaier
  sums. Compared with the textbook formulas.
- `GPUGeometryMeasures` spherical, linear edges: shoelace in `(lon, sin lat)` is exactly
  Chamberlain-Duquette (turf `area`), computed from a local origin. f32 replica within 1e-7 of f64.
- `GPUGeometryMeasures` ellipsoidal: authalic constants (`q_p`, authalic radius 6371007.1809 m
  reproduces `earth.ts`), the Simpson integral for `sin xi` differences (integrand derived),
  the chord-plus-4/3-bulge rule for geodesic edges, the adaptive tolerance scaling.
- `GPUGeometryMeasures` group reduction: weights, spherical mean for geographic groups, NaN
  propagation for invalid members.
- `GPUPointBuffer` (`gpu-point-buffer.ts`): destination vector derived in the local frame; both
  branches agree with it.
- `GPUDatumGridTransform` forward: bilinear weights, cell clamp at the last row and column, closed
  coverage test, first-covering-grid rule, vertical sign in the inverse. Units and sign of the
  samples are the caller's by contract (degrees, target minus source).
- `GPUDotDensity`, `GPURandomPointsInPolygon`: `ceil(x - u)` is unbiased stochastic rounding;
  triangle chosen by area through an integer prefix; point `A + sqrt(u1)((1 - u2) AB + u2 AC)` is
  uniform; streams keyed by registered Philox purposes with distinct counters.
- `GPURandomPointsOnLine`: arc-length prefix, binary search, clamp to the last vertex.
- `GPUGridGenerator`: square, pointy and flat hexagon lattices (all six vertex slots checked
  against `R (cos, sin)`, including the flat-top slot map `(7 - k) % 6`), triangle strips,
  counter-clockwise winding, corner counts for small cases, even-odd extent test with the exact
  orientation predicate.
- `GPUShapeGenerator`: circle and sector bearings, sweep rule, ellipse rotation, exact chord
  arc-length table and its quadrant mirror. (Geodesic precision: finding 07.)
- `GPUMinimumBounds`: rotated rectangle per hull edge, caliper monotonicity argument, Welzl
  incremental circle with circumcentre formula, rotating-caliper diameter, envelope.
- `GPUGroupConvexHull`: 64-bit product emulation and sign comparison of the exact orientation;
  power-of-two lattice scale. (Rounding: finding 09.)
- `GPUGroupGeometry`: medoid cost and tie rule. Centres and ellipses are delegated to
  `GPUGeographicDistribution`, which is another reviewer's scope.
- `GPULabelPoint`: signed distance, even-odd crossing, scanline seeds, the Lipschitz skip bound
  `d + sqrt(2) h`. The search is greedy and says so.
- `GPUMaximumInscribedCircle`: best-first branch and bound with the polylabel stopping rule
  `max bound - best <= tolerance`.
- `GPUMinimumClearance` probe: vertex to non-incident segment or other endpoint, GEOS rule. Pinned
  by a shapely fixture.
- `GPUConcaveHull`, `GPUGroupConcaveHull`: alpha-complex triangles by circumradius, as claimed
  ("radius-based alpha-shape semantics, not GEOS's ratio"); capacity `2n - 5`.
- `GPUShapeDescriptors`: Polsby-Popper `4 pi A / P^2`, Schwartzberg as its square root (the
  documented `P_circle / P` form), Green's theorem second moments (coefficients 1/12, 1/12, 1/24),
  principal-axis angle, elongation, convexity by gift wrapping or monotone chain.
- `GPULonLatPolygonPreparation` classify, cap, normalize kernels: winding, turn counting, pole cap
  vertices. Nearer-pole capping is a documented heuristic.
- `GPUSphericalConvexContains`, `GPUSphericalPolygonContains` (winding angle
  `atan2(p . (a x b), a . b - (p . a)(p . b))`, inside when the sum exceeds pi),
  `GPUSphericalConvexIntersection` (candidate set, area by the cancellation-safe excess form),
  `GPUSphericalPolygonOverlay` (gnomonic chart, Boolean duality table checked for all operand
  complements).
- `GPUOutlineGeometry`: winding of discs and quads; the equirectangular offset is documented as
  an approximation. `GPUOffsetCurve`: mitre length `d / cos(theta / 2)`, clipped mitre half length
  `d (1 - m cos(theta / 2)) / sin(theta / 2)`, round join step count.
- `GPUBoundsFilter`: inclusive intersects, within, contains, and the wrapped-box forms.
- `GPUConvexStraightSkeleton`: line intersection by Cramer's rule, collapse time from the shrink
  rate.
- `getComplementaryErrorFunction`: Numerical Recipes `erfcc` coefficients in the right order.

## 4. Standards consistency

**Earth radii and ellipsoid constants.** The scope defines none of its own. Everything resolves to
`analysis-core/contracts/earth.ts`:

| Constant | Value | Defined | Used in scope by |
| --- | --- | --- | --- |
| Mean radius (IUGG R1) | 6371008.8 m | `earth.ts:9` | `getGPUSphereRadius`: geodesic pairs and destination, geometry measures lengths, shape generator, outline geometry; directly by `gpu-point-buffer.ts:18` and by `createGPULocalFrame` |
| Authalic radius | 6371007.180918475 m | `earth.ts:12` | `getGPUAuthalicSphereRadius`: geometry measures linear-edge area. Recomputed from the ellipsoid in `getAuthalicConstants` (`geometry-measures-kernels.ts:47-70`); the two agree to 1e-9 m |
| WGS84 semi-major axis | 6378137 m | `earth.ts:15` | Vincenty, authalic constants |
| WGS84 inverse flattening | 298.257223563 | `earth.ts:18` | Vincenty, authalic constants |
| cuSpatial km per degree | 40000 / 360 | `earth.ts:28` | not used in this scope |

So two sphere radii and one ellipsoid. `geodesic-wgsl.ts:15-21` re-exports three aliases
(`GPU_GEODESIC_*`) of the same values. Note `(2a + b) / 3` is 6371008.7714 m; 6371008.8 is the
usual rounding.

- **Two area radii in one contributor.** `GPUGeometryMeasures` uses the authalic radius for
  linear-edge area and the mean radius for great-circle-edge area
  (`gpu-geometry-measures.ts:277-280`). The areas differ by 5.1e-7 relative. This is documented
  (`docs/api/gpu-spatial-analysis/operators-lines.md:308-310`), so it is a stated convention, but
  it means the same polygon has two "sphere" areas.
- **Four spherical destination implementations.** `geodesicDestination` (shared, best numerics),
  `GPUPointBuffer` (own careful version, fixed mean radius, no spatial context),
  `GPUShapeGenerator.getDestination` (own naive version, finding 07), `GPUOutlineGeometry.offsetPoint`
  (equirectangular, documented). The first three compute the same thing; only the first two agree
  to the output precision.
- **Two longitude-difference helpers.** `geodesicLongitudeDelta` (seam safe) and
  `geodesicWrapLongitudeDelta` plus the identical `lonLatWrapLongitudeStep`
  (`analysis-core/lon-lat-ring-wgsl.ts:21-23`). Finding 08 is a result.
- **Two spherical-excess triangle functions.** `sphericalExcessTriangle`
  (`cell-indexing/h3-measures-wgsl.ts:10-13`, determinant of the raw vectors) and
  `sphereTriangleArea` (`global-coordinates/gpu-spherical-convex-intersection.ts:205-207`,
  determinant of differences from one vertex). Same quantity; the second keeps more digits.
  Finding 01 uses the first.
- **Two pole-of-inaccessibility contributors.** `GPULabelPoint` (greedy fixed schedule, origin
  relative coordinates) and `GPUMaximumInscribedCircle` (branch and bound with an error output,
  absolute coordinates). Already listed as a merge in the library review; noted here because they
  can return different points for the same polygon.
- **Hole rules.** Geometry measures, shape descriptors and dot density use ring position.
  Label point, maximum inscribed circle and the grid extent mask use even-odd. The docs say so.
- **Private copies of things `analysis-core` has** (contract section 11):
  - Exact integer orientation in `gpu-group-convex-hull.ts:1089-1134` (64-bit products of 30-bit
    lattice values). `analysis-core/kernels/exact-lattice-wgsl.ts` covers a 2^22 lattice with 80-bit
    arithmetic; `exact-orientation-wgsl.ts` works on raw f32. The copy is exact for its own lattice,
    so numerically equivalent for its inputs, but it is a third exact-orientation implementation.
  - A hash for the Welzl shuffle, `hashPair` in `gpu-minimum-bounds.ts:576-580` (PCG-style, not
    Philox, not `hashUint32`). The circle does not depend on the order in exact arithmetic, so only
    the running time is affected. Modulo bias of `% (slot + 1)` is irrelevant here.
  - Planar point-to-segment distance three times with two formulations: clamp-and-subtract in
    `gpu-label-point.ts:526-535` and `gpu-maximum-inscribed-circle.ts:160-185`, `cross^2 / length^2`
    in `gpu-minimum-clearance.ts:429-441`.
  - The f32 even-odd crossing test in `gpu-label-point.ts:550-555` and
    `gpu-maximum-inscribed-circle.ts:177-179`, against the exact one in
    `gpu-grid-generator.ts:616-632`. The f32 copies are flagged in comments as sampling-only.
  - Circumcentre in `gpu-minimum-bounds.ts:597-611` and `gpu-concave-hull.ts:210-215`: same
    formula, different degenerate tests (relative determinant against exact orientation).
  - Neumaier addition in `geometry-measures-kernels.ts:488-496` and
    `shape-descriptors-kernels.ts:130-140`: identical.
- **`floor(x + 0.5)` rounding in f32** appears in findings 04 and 09. The library review lists the
  same grid key in `gpu-coverage-dissolve.ts:312`, `gpu-coverage-simplification.ts:412` and
  `spatial-weights/gpu-contiguity-weights.ts:348`. Those are outside this scope and I did not check
  them.
- **Shader literals.** Degree and radian factors are written five ways
  (`0.0174532925199433`, `0.017453292519943295`, `57.2957795130823`, `57.29577951308232`, and
  `x / DEGREES` in `gpu-shape-generator.ts:471`). All round to the same f32 except the division,
  which can differ by one ulp. `SQRT3` is `1.7320508` in two places in `gpu-grid-generator.ts`.

## 5. luma.gl official implementations

- **`GPULocalFrameProjection` delegates** to `compileProjectionPlan` and `GPUProjection` from
  `@luma.gl/graph/gpu-project`, with the frame transform from `@math.gl/geospatial/projection`.
  This is the upstream-first shape. It then overrides `destinationOrigin` on the compiled plan
  (finding 11). The override is consistent with how `GPUProjection` packs patches, but the plan's
  error certificate was computed for the bounds centre. This departure is not in
  `dev-docs/upstream-workarounds.md`. Recommendation: propose a `destinationOrigin` option for
  `compileProjectionPlan` upstream.
- **`GPUGeodesicPairs` (sphere model, distances only) re-implements `GPUHaversineDistance`** from
  `@luma.gl/graph/geospatial`. Both compute the great-circle central angle and agree
  mathematically. Differences:
  - Default radius: upstream 6371 (kilometres), local 6371008.8 m. Relative difference 1.4e-6,
    8.8 m in 6371 km. Neither is wrong; they are two conventions for "the" Earth radius.
  - Form: upstream uses a planar small-angle branch below 1e-4 rad, haversine `asin` to a
    half-versine of 0.5, then the `atan2` form. Local uses the `atan2` (Vincenty sphere) form
    everywhere with cancellation-free terms. Local is the better conditioned of the two at short
    range in f32.
  - Antimeridian: upstream subtracts raw longitudes; the haversine is periodic so the result is
    right, with the f32 loss of finding 08. Local `geodesicLongitudeDelta` avoids that loss.
  - Input: upstream also accepts raw binary64 rows (`uint32x4`) and subtracts in f64. Local is f32
    only, so upstream is more precise for short distances when the caller has f64 positions.
  Recommendation: for sphere distance alone, delegate to the public `GPUHaversineDistance` with
  `radius` from `getGPUSphereRadius`, or propose the seam-safe difference and the bearing,
  midpoint, Vincenty and rhumb outputs upstream. Do not keep two unrelated default radii without
  saying so in the docs.
- **`GPUSinusoidalProjection`** (40000 / 360 km per degree) is not used or re-implemented in this
  scope. `earth.ts` keeps the constant for parity only.
- **`createWebMercatorProjection`** (radius 6378137, latitude clamp 85.05112878) is not used or
  re-implemented in this scope.
- **Geometry entry point.** `@luma.gl/graph/gpu-geometry` (`index.d.ts`) exports only
  `GPUMeshNormals`, `GPUMeshVertexNormals` and `GPUMeshSmoothing`. Nothing in this scope
  duplicates them. The other `geospatial` exports (pairwise point distance, point in polygon,
  point to linestring, spatial query) belong to other scopes; I did not compare them.
- **WGSL conventions.** Upstream geospatial kernels write NaN for non-finite inputs through a bit
  mask; local geodesic pair kernels let NaN propagate through the trigonometry without a guard.
  Results agree (NaN in, NaN out) as far as I read.

## 6. Test oracles

**Pinned by an independent oracle** (generator under `modules/analysis/test/oracles` or a fixture
from a reference library):

- `GPUGeodesicPairs`, `GPUGeodesicDestination`: `parity-wp2-geodesics.py`,
  `parity-wp2-round8-geodesics.py` (pyproj `Geod`, PyGeodesy rhumb), `difficult-geodesics.py`
  (GeographicLib).
- `GPUGeometryMeasures` geodesic area: `geodesic-area-adaptive.py` (GeographicLib). Hole rules:
  `geometry-hole-rules.py` (shapely).
- `GPULocalFrameProjection` and `createGPULocalFrame`: `parity-wp2-local-frame.py`,
  `parity-wp2-round8-local-frame.py` (PROJ).
- `GPUDatumGridTransform`: `datum-horizontal.py` (PROJ `hgridshift`), but only with an explicit
  `tolerance: 1e-10` (finding 05).
- `GPUGroupConvexHull`: `geometry-convex-hull.py` (shapely and Qhull), on coordinates that are
  multiples of 0.25 below 256, which cannot reach the band of finding 09.
- `GPUDotDensity`, `GPURandomPointsInPolygon`: `geometry-dot-density.py` (shapely containment and
  area uniformity).
- `GPUMinimumClearance`: shapely fixture `test/gpu-spatial-analysis/minimum-clearance/shapely-clearance-fixture.ts`.
- `GPULonLatPolygonPreparation`: `parity-wp2-round5-geometry.py`, `parity-wp2-round8-polygons.py`
  (PROJ densification, GEOS clipping).
- Spherical family and datum: `global-coordinates.py` (s2sphere, GeographicLib, pyproj, shapely).
- `GPUVectorStraightSkeleton`: `geometry-straight-skeleton.py` and `.cpp` (CGAL).
- `GPURatioConcaveHull`, `GPUSegmentSnap`, precision reduction, `GPUMaximumInscribedCircle`:
  `geometry-refinement.py`, `geometry-advanced.py` (shapely, including
  `shapely.maximum_inscribed_circle`). The ratio hull has three tiny cases (finding 03).
- `GPUMinimumBounds`: values from shapely `minimum_rotated_rectangle` and
  `minimum_bounding_circle` pinned in `test/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds.spec.ts:22`.
- `GPULabelPoint`: an f64 polylabel in `label-point-oracle.ts` plus two pinned shapely polylabel
  radii (`gpu-label-point.spec.ts:156`). The f64 polylabel is a different algorithm, so it is an
  independent check of the result.
- `GPUGridGenerator`: `grid-generator-geopandas-fixture.ts` (GeoPandas `make_grid`).
- `GPUShapeGenerator`: `turf-shape-values.ts` (real turf `circle` and `sector` at latitudes
  -33 to 41, rounded to 1e-7 degrees). The ellipse values use turf `destination` on the kernel's
  own offsets. No case above latitude 41, so finding 07 is not exercised.
- `GPUOffsetCurve`: the class doc says the spec checks against GEOS; an
  `offset-curve-reference.ts` exists. I did not open either.

**Pinned only by a copy of their own formula, or by a convention chosen to match the kernel:**

- Rhumb midpoint: the oracle uses a degree-plane GEOS interpolation chosen to match the helper
  (finding 02).
- Authalic inverse series and so ellipsoidal polygon centroids: the CPU twin
  `test/gpu-spatial-analysis/geometry-measures/geometry-measures-oracle.ts:318` repeats `17 / 360`
  (finding 06).
- Spherical linear-edge area: a Chamberlain-Duquette CPU function in the guards spec. The formula
  is right (derived by hand), but the check is a twin.
- Great-circle-edge area: a spherical-triangle CPU function on large triangles only (finding 01).

- `GPUShapeDescriptors`: `shape-descriptors-oracle.ts` is a CPU twin; one degenerate convexity
  case cites shapely values (`gpu-shape-descriptors.spec.ts:202`). The formulas were derived by
  hand here.

**No independent check found** (I searched test folders and oracle names, not every spec):
`GPUShapeGenerator` geodesic precision above latitude 41, `GPULinePrecisionReduction` above 2^23
grid steps, `GPUGeometryMeasures` great-circle-edge area below about 100 km, `GPUPointBuffer`,
`GPUGroupGeometry` medoid, `GPUOutlineGeometry` spherical offsets. Treat this list as "check",
not "missing".

## Resolution (2026-11-10)

Stream `geometry-measures`. Paths are relative to `modules/analysis`. Every finding was re-read in
source. Findings 01, 02, 03, 04, 05, 07, 08 (Vincenty) and 09 were also shown on the GPU by a
spec that failed on the old kernel before the fix; 06 and the rhumb half of 08 were changed
before their spec was written, so their old failure rests on the arithmetic in the report (the
06 spec did fail on the first run, on a second defect, see the row). Counts: 11 fixed, 0 not
reproduced, 0 deferred among the report's findings; two items from `luma-official.md` are listed
below the table (one fixed in words, one deferred).

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| geometry-measures-01 | fixed | `src/gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:281-327` (new `getGreatCircleEdgeBulge`), `:492-497` (edge term), import of `SPHERICAL_EXCESS_WGSL` dropped. Area is the Lambert chord cross from the feature origin less twice a cancellation-free per-edge great-circle bulge. | `test/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures-guards.spec.ts:348` (100 m and 1 km squares, a hole; expected by binary64 l'Huilier on the f32 vertices) and `:380` (concave ring across the antimeridian, both windings). Old kernel: 89% error on the first square. | `edgeModel: 'great-circle'` areas of small rings change from noise (60,920 m2, or a wrong sign that added holes) to 1e-7 relative. Rings of 100 km and more change by about 1e-7. |
| geometry-measures-02 | fixed (code, per roadmap 12.3) | `src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:154-183` (new `rhumbStretchedHalfDelta`), `:197-212` (`rhumbMidpoint`); both revised in the review follow-up below. Output documentation `gpu-geodesic-pairs.ts:51-56`; `docs/api/gpu-spatial-analysis/operators-lines.md`. | `test/gpu-spatial-analysis/geometry-measures/gpu-rhumb.spec.ts:220` (three hand-computed midpoints; the midpoint halves the turf-pinned rhumb distance and lies on the rhumb bearing), `:143` (twin corrected to the closed form), `parity-wp2-round8-geodesics.parity.spec.ts` (fixture regenerated, see below). | `GPUGeodesicPairs` `'rhumb'` `midpoints` longitude changes: (0, 0) to (90, 60) gives 37.539 instead of 45; New York to London -38.87 instead of -37.05. Latitude and east-west lines are unchanged. |
| geometry-measures-03 | fixed in the documentation; the kernel is unchanged | `src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull.ts:38-60` (class documentation), `docs/api/gpu-geometry-refinement.md:18`. Confirmed on the GPU first: nine points, kernel area 37.65625, largest-first area 31.75. | `test/gpu-spatial-analysis/geometry-refinement/geometry-topology-refinement.spec.ts:256` (pins the round result on the nine-point case and the documented guarantees: every vertex kept, connected, no removable border triangle above the threshold). | None in results. The documentation no longer claims descending size order; it says the hull can differ from `shapely.concave_hull`. |
| geometry-measures-04 | fixed | `src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction.ts:303-307`: round half up from the exact fraction (`floor` plus `step(0.5, fraction)`). | `test/gpu-spatial-analysis/geometry-refinement/geometry-topology-refinement.spec.ts:382` (odd integers in [2^23, 2^24), both signs; 0.49999997; exact halves). | `GPULinePrecisionReduction`: coordinates with 2^23 or more grid steps that are already on the grid no longer move by one step; the largest f32 below one half rounds to 0, not 1. Exact halves still round up. |
| geometry-measures-05 | fixed | `src/geospatial-analysis/gpu-datum-grid-transform.ts:45-49` (default 5e-9 degrees), `:252-257` (`datumInverseLimit`), `:302-304` (first update before the test), prop documentation `:86-94`; `docs/api/geospatial.md`, `docs/api/gpu-spatial-analysis/global-coordinates.md`. | `test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts:727` (constant 5e-6 degree shift and a linear shift, closed-form inverses, default tolerance). | Inverse with the default tolerance now applies shifts below 1.1 m instead of returning the input. Default tolerance is 5e-9 degrees (0.56 mm), was 1e-5. The convergence test never asks for less than 4.8e-7 of the shift (four f32 spacings), so a tolerance below the residual's own resolution reports success at that floor instead of `inverseExhausted`. |
| geometry-measures-06 | fixed | `src/gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:181-190` (Snyder 3-18 with 23/360 and the e^6 terms; within 2 mm of the exact inverse). Also `:165-179`: `getSinAuthalic` takes `atanh(e s) / e` from its series, because the f32 `log((1 + es) / (1 - es))` lost 1.6 m of latitude (found by the new spec, not in the report). Test twin `test/gpu-spatial-analysis/geometry-measures/geometry-measures-oracle.ts:155-176`, `:340` now uses the exact inverse by Newton iteration in binary64. | `test/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures-guards.spec.ts:411` (box centroids at latitudes 22.5, 67.5, -22.5 and 10 to 35 against the exact inverse). | Ellipsoidal polygon centroid latitudes move by up to 5 m (toward the exact value). Areas and lengths are unchanged. |
| geometry-measures-07 | fixed | `src/gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:463-465`: calls the shared `geodesicDestination`; `GEODESIC_WGSL` is included for geodesic shapes. | `test/gpu-spatial-analysis/grid-generators/gpu-shape-generator.spec.ts:146` (50 m circles at latitudes 45.2 to 88.6 and -86.4 against the binary64 destination formula; bound 0.8 f32 latitude spacings). Old kernel: 1.07 m at latitude 45.2. | Geodesic circle, sector and ellipse vertices move by up to 0.8 m at mid latitudes and 10 m near the poles for small radii. |
| geometry-measures-08 | fixed; converged on the sphere model's `geodesicLongitudeDelta` (the shared helper the siblings use) | `src/gpu-spatial-analysis/geometry-measures/geodesic-kernels.ts:110`, `:264`; `rhumb-wgsl.ts:106`, `:115`, `:198`. | `test/gpu-spatial-analysis/geometry-measures/gpu-geodesic.spec.ts:395` (equatorial WGS84 pair across the antimeridian, closed form `a * delta`; old kernel 1.70 m off), `gpu-rhumb.spec.ts:308` (east-west rhumb pairs across the antimeridian, closed form `R cos(lat) delta`). | Vincenty and rhumb distances, bearings and midpoints of pairs that straddle the antimeridian change by up to 1.5e-5 degrees of longitude difference (1.7 m). |
| geometry-measures-09 | fixed | `src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.ts:423-433`. | `test/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.spec.ts:612` (the report's collinear triple; the old kernel returned the collinear middle point as a hull vertex). | Hulls can lose a spurious vertex that was one lattice step off an edge. The existing binary64 oracle (`group-geometry-oracle.ts:115`) already rounded correctly and is unchanged. |
| geometry-measures-10 | fixed (words) | `src/gpu-spatial-analysis/label-point/gpu-label-point.ts:124-127`: exponent `refinementRounds - 1`, 3.05e-5 of the half extent. | none (documentation). | none. |
| geometry-measures-11 | fixed (words); confirmed by reading `compileProjectionPlan` in published luma.gl, which validates `float32MaxError` against the bounds centre | `src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts:34-41` (`tolerance`), `:55-62` (class), `:68-73` (`plan`). | none (documentation). | none. |

Also done in this stream:

- `luma-official-08` (documentation half): `gpu-geodesic-pairs.ts:100-106` and
  `docs/api/gpu-spatial-analysis/operators-lines.md` now say that `GPUHaversineDistance` exists and
  differs by default radius and unit. Removing the dead constant
  `GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE` is `deferred` (roadmap 12.4: it is an exported symbol, and
  `analysis-core/contracts/earth.ts` is not in this stream).
- `gpu-minimum-bounds.ts:537` (Welzl shuffle): the private PCG hash is replaced by Philox with the
  registered purpose `minimumBoundsCircleShuffle` (`analysis-core/kernels/philox-random.ts:57`). The
  circle is unchanged within its tolerance; `gpu-minimum-bounds.spec.ts` and
  `test/utils/philox-random.node.spec.ts` pass.

Oracles and twins that encoded the old behaviour:

- `test/oracles/parity-wp2-round8-geodesics.py` pinned the degree-plane midpoint on purpose. The
  generator now computes the rhumb midpoint by the closed form `rhumb_midpoint` (mean latitude,
  longitude linear in `asinh(tan(latitude))`) and no longer imports Shapely. PyGeodesy, pyproj,
  NumPy and Shapely are not importable from the Homebrew Python 3.15 used in this pass (see the
  review follow-up for the system Python), so the generator was **not** rerun: only
  the `midpoints` of the two rhumb cases of `fixtures/parity-wp2-round8-geodesics.json` were
  recomputed, with the same function in Python 3.15 (`math` only). 24 values moved by more than
  the fixture tolerance, the largest by 1.41 degrees. The fixture carries a `notes` field saying
  so, and `versions` still describes the earlier full run. A rerun with PyGeodesy should compare
  `rhumb_midpoint` with `rhumbMidpointTo`.
- `gpu-rhumb.spec.ts` `getRhumbPair` (midpoint) and `geometry-measures-oracle.ts` (authalic
  inverse) are corrected as listed in the table.

Not changed, and why:

- **Ratio concave hull order (03).** The round rule cannot give the largest-first order: a
  triangle that is the largest around its own vertices is removed before a larger removal elsewhere
  reaches it. An exact parallel rule exists and fits the kernel's layout (remove a candidate only
  when its exposed edge is longer than every edge that a triangle sharing one of its vertices could
  still be removed for, plus always the global largest candidate), but its round count is unknown
  on large inputs and its parity with GEOS cannot be checked here: Shapely is not installed and
  the committed fixture has three small cases. Following roadmap 12.3 the claim was reworded. A
  random-point Shapely fixture is still worth adding where Shapely is available.
- **`sphericalExcessTriangle`** (`cell-indexing/h3-measures-wgsl.ts`) is untouched and still
  exported; geometry measures no longer uses it. For unit-vector input the form of
  `sphereTriangleArea` (`global-coordinates/gpu-spherical-convex-intersection.ts:207`),
  `dot(a, cross(b - a, c - a))`, keeps a relative error of about `6e-8 / side` instead of
  `6e-8 / side^2`; longitude/latitude input can use `getGreatCircleEdgeBulge`.
- **Ellipsoidal and spherical polygon centroid latitude near the poles.** Both take `asin` of an
  f32 sine, so the latitude carries the sine's rounding divided by `cos(latitude)`: 0.2 m at
  latitude 22, 2 m at 80, 20 m at 89. Not in the report; the 06 spec scales its bound by
  `1 / cos(latitude)` and says so. The cure is the origin-relative latitude step that
  `geodesicDestination` uses.
- **`GPUGeodesicPairs` sphere model is not delegated to `GPUHaversineDistance`.** No parity spec
  was written because the official f32 kernel subtracts raw longitudes
  (`rightDegrees - leftDegrees`), which loses the 1.7 m that finding 08 removes and that
  `parity-wp2-geodesic-seam.parity.spec.ts` pins for the shared helper.

Upstream gaps (what luma.gl would need):

- `GPUHaversineDistance`: a seam-safe longitude difference in the f32 path (subtract on the
  antimeridian side before forming the small separation), so that the sphere model of
  `GPUGeodesicPairs` can delegate without losing 1.5e-5 degrees across +-180.
- `compileProjectionPlan`: a `destinationOrigin` option, so that `tolerance`, `maxError` and
  `float32MaxError` are validated for the origin the output rows are relative to.
  `GPULocalFrameProjection` overrides the origin after compiling, and this departure is not yet in
  `dev-docs/upstream-workarounds.md`.

Gates run by this stream: `node scripts/typecheck-status.mjs` on the stream's paths, 0 errors;
`node scripts/check-upstream-first.mjs`, no violations; `yarn biome check --write` on the changed
source and spec files.

### Follow-up: compensated float sums (same day)

The trajectories stream found that the test GPU's shader compiler (Metal, fast math) folds the
float Neumaier term `(sum - total) + value` to zero. Five files in this stream's directories
mention Neumaier or "compensated"; they hold two distinct implementations, and both were dead.
Each was probed on the GPU with a ring that walks one triangle many times, so that every loop adds
the same f32-exact cross product (4001 * 3989) and the same three edge lengths, and the binary64
totals are closed forms.

| ID | Outcome | What was measured | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- | --- |
| geometry-measures-12 (extra) | fixed | `addCompensated` of `GPUGeometryMeasures`, used for every length, area, moment and line-moment sum of the serial kernel, the lane sums and merges of the cooperative kernel, and the longitude step scan. Serial, 300,000 vertices: perimeter 1.058e-3 off; cooperative, 3.84 million vertices: 2.82e-4 off. Both are the plain f32 running sum to the digits predicted by a CPU replica (1.06e-3, 2.82e-4), so the compensation did nothing. | `src/gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:567-577`: the compensation is now `getAdditionRoundingError(previous, value, total)` from `analysis-core/kernels/addition-rounding-error-wgsl.ts` (the signed version the core stream published), which reads the error off the integer significands. `feature-measures-cooperative.ts` calls the same function and needed no edit. | `test/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures-summation.spec.ts:109` (serial) and `:118` (cooperative): perimeter and area within 2e-6 of the closed form. | Lengths, areas and centroids of features with many vertices change toward the binary64 value: by up to about `vertices * 6e-8` relative in the worst case (1e-3 at 300,000 vertices of equal edges). Small features change in the last bit or not at all. |
| geometry-measures-13 (extra) | fixed | `addNeumaier` of the `GPUShapeDescriptors` moments kernel. With the float term restored for the measurement, the first-ring area of a 300,000-vertex ring failed the 2e-6 bound (plain f32 prediction 1.2e-3). | `src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels.ts:130-138`: same replacement. | `test/gpu-spatial-analysis/shape-descriptors/shape-descriptors-moments-summation.spec.ts:34`: first-ring area and the three second moments within 2e-6 of the closed form (runs `createMomentsNode` with a given centroid). | Elongation and orientation of polygons with many vertices change toward the binary64 value. |

Checked and found alive (no change): the fixes of this pass that depend on an evaluation order or
on a small difference all have a GPU spec that fails without them. The great-circle bulge (01) has
no algebraically zero term and its spec holds 1e-5 where the old kernel was 89% off; the `atanh`
series (06) is a polynomial; the seam-safe longitude difference (08) takes the integer path the
shared helper already uses against reassociation; the rhumb half-delta and the polar colatitude
forms are pinned by the polar pair of `gpu-rhumb.spec.ts`; the `floor` plus `step` rounding (04,
09) and the datum inverse (05) form no error term. The split-word sums of `GPUDatumGridTransform`
(`fma(position, 1, -origin)`) predate this pass and are pinned at 3e-11 by
`global-coordinates.spec.ts`. No other two-sum, two-product or error-free transformation exists in
this stream's directories.

### Review follow-up (same day)

From `fix-review-geometry.md`. Paths are relative to `modules/analysis`.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| fr-geometry-03 | fixed; shown on the GPU first (`(0, 0)` to `(90, 89.95)` returned 10.937, exact 10.2453) | `src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:124-131` (`rhumbAtanh`), `:137-142` (`rhumbIsometricLatitude`), `:154-183` (`rhumbStretchedHalfDelta`), `:197-212` (`rhumbMidpoint`). The longitude fraction is `firstHalf / (firstHalf + secondHalf)` with both halves from `rhumbStretchedHalfDelta`; nothing in it is clamped at `atanh(0.999999)` any more. A half whose `atanh` argument reaches 1/2 is taken as a difference of isometric latitudes `-ln(tan(colatitude / 2))` from the exact colatitudes, because `1 - x` has no float32 precision left near 1. The second half is the first half of the mirrored line (`-b` to `-a`), so a line that is symmetric across the equator makes two calls with identical arguments and the fraction is exactly 1/2. | `test/gpu-spatial-analysis/geometry-measures/gpu-rhumb.spec.ts:220`: the reviewer's cases (`(0, 0)` to `(90, 89.95)`: 10.2453; `(10, 60)` to `(100, 89.99)`: 17.9616; `(0, -88)` to `(120, 88)`: exactly 60), each also checked to halve the turf-pinned rhumb distance on the rhumb bearing, and symmetric pairs `(0, -L)` to `(100, L)` in both directions for L = 20, 60, 80, 85, 88 and 89.99, which must return longitude 50 and latitude 0 exactly. | `'rhumb'` `midpoints` of lines that reach high latitudes move to the loxodrome's midpoint: by up to 1e-4 degrees for lines that stay below latitude 84, by degrees for lines that pass 89. |

Float32 limit of the midpoint, from a float32 replica of the new kernel against binary64 over
400,000 random pairs (scratch `rhumb-mid.mjs`; error in degrees of longitude times the cosine of
the middle latitude): 9e-6 degrees or less for endpoints up to latitude 60, 80, 85, 88, 89, 89.9,
89.99 and 89.9999 alike, which is the float32 spacing of the output longitude. There is no latitude
below the clamp at which float32 limits the midpoint further; above 89.9999 degrees both latitudes
are clamped first. The form the reviewer proposed (two `atanh` halves) was measured too: 9e-5 at
latitude 85, 4e-4 at 88, 0.12 at 89.9, which is why the long-span branch does not use `atanh`.
`rhumbAngle` and `rhumbBearingDegrees` still use the clamped `rhumbStretchedDelta` and lose
precision for lines that pass latitude 89; that is the documented limit of distance and bearing
and was not part of the finding.

Bare `atan2` in this stream's directories. The "twin heading kernel" of fr-geometry-06
(`trajectory-interpolation`, `linear-referencing`) is not in this stream and already calls
`guardedAtan2`. The sites here whose second argument is a difference or a dot product now call
`guardedAtan2` from `analysis-core/kernels/wgsl-common.ts`:

- `src/gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:497` (ellipse offset to
  bearing). This one was a visible defect, shown on the GPU first: a uniform-parameter ellipse
  with a collapsed semi-minor axis has a north offset of `-0.0` in the lower half turn, and the
  vertex at phase 315 degrees came out at x = -4.07 instead of 10.07 (mirrored across the center).
  With both semi-axes zero the builtin returns NaN. Pinned by
  `test/gpu-spatial-analysis/grid-generators/gpu-shape-generator.spec.ts:210`. Callers see collapsed
  uniform ellipses on the right side and zero-radius ellipses at the center instead of NaN.
- `src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels.ts:399` (orientation,
  `xx - yy`); the text pin in `shape-descriptors.node.spec.ts:33-36` follows.
- `src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection.ts:261`, `:264`
  (vertex order angle) and `gpu-spherical-arc-overlay.ts:214` (`arcLength`), `:298` (turn angle).
- `src/geospatial-analysis/gpu-point-buffer.ts:135` (longitude offset).

No failing input was built for the last three files; their specs pass unchanged. Left alone:
`gpu-minimum-bounds.ts:487-491` and `rhumb-wgsl.ts:117-119` already branch on a zero second
argument; `geodesic-kernels.ts` and `gpu-spherical-polygon-contains.ts` are reserved for the
other session, and `geodesic-kernels.ts:213` (`atan2(sinAlpha, -northTerm)`) was still bare when
this was written; the shared `geodesic-wgsl.ts` helpers and the group centroid of
`geometry-measures-kernels.ts` take sums that are `+0.0` when they are zero, and adding the helper
to `GEODESIC_WGSL` would define it twice in every kernel that already includes it.

Resolution line references. The references of rows 01, 02, 06, 07 and 08 above were corrected to
the current files (they had moved with the later edits).

Oracle generator. `/usr/bin/python3` is Python 3.9.6 with NumPy 2.0.2, SciPy 1.13.1, pyproj 3.6.1
and Shapely 2.0.6. PyGeodesy and GeographicLib are not installed there (`requirements.txt` pins
`pygeodesy==26.9.9` and `pyproj==3.8.0`), and `parity-wp2-round8-geodesics.py` imports PyGeodesy
for the rhumb distances, bearings and destinations, so it still cannot run. Nothing was
installed. The hand-recomputed rhumb midpoints stay; the reviewer recomputed all 266 and found
them equal to the last digit. The generator's docstring now names the missing package.
Closing pass (2026-11-11): the generator ran on the pinned set and PyGeodesy 26.9.9 `rhumbMidpointTo` agrees with every midpoint to 4.6e-10 degrees; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

Not done here: fr-geometry-02 (the lens formula of `GPUShapeGenerator` binds `wgsl:latitude2` and
`wgsl:angularDistance`, which geometry-measures-07 removed from the kernel) is in the lenses
package, which is out of bounds for this stream; `yarn test-node` stays red on it until the
formula is rebound to `geodesicDestination`.
