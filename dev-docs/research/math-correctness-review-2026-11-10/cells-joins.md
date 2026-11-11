# Math correctness review: cells, joins, zones (2026-11-10)

Scope id `cells-joins`. Paths are relative to `modules/analysis/src`. Scope directories are under
`gpu-spatial-analysis/`: `cell-indexing`, `cell-topology`, `cell-cover`, `cell-aggregation`,
`healpix-cells`, `hilbert-keys`, `cell-table-compare`, `cell-set-outline`, `spatial-join`,
`neighbor-search`, `pair-gather`, `zonal-statistics`, `region-statistics`, `areal-interpolation`
(about 39,000 lines). Read-only review. No spec was run. Five scratch scripts were run with Node
(see "How findings were confirmed").

Summary: the cell index math (H3, S2, Quadbin, geohash, HEALPix, Hilbert) follows its references.
One real defect was found, in the fast polygon `contains` / `within` kernel. The rest are a
documentation error about tobler, two inconsistent conventions and one edge case.

## 1. Coverage

An unread file is not a verified file. About 60% of the scope lines were read.

**Read in full**

- `cell-indexing`: `cell-index-wgsl.ts`, `h3-index-wgsl.ts`, `h3-index-extended-wgsl.ts`,
  `h3-boundary-wgsl.ts`, `h3-measures-wgsl.ts`, `cell-index-families.ts`, `gpu-point-to-cell.ts`,
  `gpu-cell-measures.ts`, `gpu-cell-geometry.ts`.
- `cell-aggregation`: `cell-keys.ts`.
- `cell-topology`: `cell-topology-wgsl.ts`, `h3-neighbor-wgsl.ts`, `h3-local-ij-wgsl.ts`,
  `s2-analysis-wgsl.ts`, `s2-covering-support.ts`, `gpu-s2-cell-measures.ts`,
  `gpu-s2-cell-hierarchy.ts`, `gpu-s2-cap-covering.ts`, `gpu-s2-polygon-covering.ts`.
- `healpix-cells`: all four source files.
- `hilbert-keys`: `gpu-hilbert-keys.ts`, `bounds-reduction.ts`.
- `cell-table-compare`: `gpu-cell-table-compare.ts`.
- `spatial-join`: `spatial-predicate-wgsl.ts`, `spatial-dwithin-wgsl.ts`,
  `spatial-geodesic-wgsl.ts`.
- `neighbor-search`: `neighbor-search-parameters.ts`, `neighbor-search-derived-parameters.ts`,
  `neighbor-search-kernels.ts`.
- `zonal-statistics`: `zonal-statistics-passes.ts`.
- `areal-interpolation`: `gpu-areal-interpolation.ts`, `gpu-pycnophylactic.ts`,
  `zone-anchor-nodes.ts`.
- Outside the scope directories, because scope code depends on it:
  `analysis-core/cell-key-wgsl.ts:1-80`, `analysis-core/spherical-cell-area.ts`,
  `raster-analysis/polygon-rasterization/polygon-rasterization-parameters.ts:35-66`.

**Read in part**

- `cell-indexing/h3-index-tables.ts`: header and layout. The 540 table values were not checked by
  hand (the header says they were generated against h3-js 4.4.0).
- `cell-topology/h3-neighbor-tables.ts`: digit tables and clockwise-offset faces checked against
  the H3 C source from memory; the 854-entry base cell tables were not checked value by value.
- `cell-topology/gpu-h3-cell-identities.ts:36-143`, `gpu-cell-grid-path.ts:30-130, 225-346`,
  `gpu-cell-compaction.ts` (subtree sizes and pentagon handling only, by grep).
- `cell-aggregation/cell-table.ts:270-300, 440-560, 600-678` (heads, 64-bit accumulate, finish).
- `cell-cover/gpu-cell-cover.ts:40-250`, `cell-cover-kernels.ts:300-520`,
  `cell-cover-h3-constants.ts`.
- `spatial-join/gpu-point-in-polygon-join.ts:190-330, 560-943`, `nearest-wgsl.ts:1-420, 470-616`,
  `spatial-relate-types.ts:60-239`, `gpu-spatial-predicate-join.ts` (engine selection:
  `163-190, 280-306, 620-712, 1195-1220, 1425-1475`), `spatial-join-passes.ts:160-235, 700-790`,
  `gpu-nearest-feature-join.ts:392-470`, `gpu-nearest-feature-weights.ts:30-120, 350`.
- `neighbor-search/gpu-neighbor-search.ts:100-345, 640-790`.
- `zonal-statistics/gpu-zonal-statistics.ts:60-330`.
- `region-statistics/gpu-region-mask.ts:205-405`, `region-statistics-grid.ts:165-200`,
  `gpu-region-statistics.ts:440-475`.
- `pair-gather/gpu-pair-gather.ts:95-150` (documentation only; it has no arithmetic).

**Not opened**

- `cell-set-outline/gpu-cell-set-outline.ts`.
- `cell-cover/cell-cover-scanline.ts`, and `cell-cover-kernels.ts` outside `300-520`
  (`coverContains`, `coverEdgesHitRect`, the H3 dedupe rule).
- `cell-aggregation/gpu-cell-aggregation.ts`, `gpu-cell-rollup.ts`, `gpu-cell-pyramid.ts`,
  `gpu-cell-level-selection.ts`.
- `cell-topology/gpu-cell-topology.ts` (the WGSL it uses was read).
- `spatial-join/spatial-relate-wgsl.ts` (the DE-9IM matrix engine), `nearest-nodes.ts`,
  `gpu-spatial-join-candidates.ts`, `spatial-join-candidates.ts`, `gpu-spatial-join-prepared.ts`,
  `gpu-buffer-selection.ts`, `spatial-join-capacity-plan.ts`, `spatial-join-geometry.ts`, the
  types files, and `spatial-join-passes.ts` outside the two ranges above.
- `region-statistics/region-statistics-readback.ts`, `gpu-pick-region-mask.ts`,
  `region-statistics-types.ts`, and the rest of the three files read in part.
- `areal-interpolation/bounded-key-pair-sort.ts`.

**Not present in scope.** The brief asked about majority and median over categories and fractional
cell coverage. No contributor in these directories computes them. `GPUZonalStatistics` is
points-in-polygons (count, sum, mean, min, max, variance, density). Fractional coverage lives in
`raster-analysis` (other scope). HEALPix has the nested and UNIQ schemes only; there is no ring
scheme.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cells-joins-01 | wrong-result | CONFIRMED | `spatial-join/spatial-predicate-wgsl.ts:586-603` | Fast polygon `contains(outer, inner)` rejects an inner polygon that coincides with a hole only when the f32 midpoint of some inner edge is exactly on the outer boundary. | OGC / GEOS `contains` needs the interiors to intersect. A polygon that fills a hole only touches. | Outer: square (2,2)-(100,100) with the triangular hole (10.1,10.2), (30.3,11.7), (17.9,40.4). Inner: that triangle. Kernel: `contains` true. GEOS: false. | Decide the side with exact orientation signs, or send polygon/polygon `contains` and `within` to the relate engine when an inner edge lies on the outer boundary. |
| cells-joins-02 | doc-mismatch | CONFIRMED | `areal-interpolation/gpu-areal-interpolation.ts:87, 199-203, 265-267`; `docs/api/gpu-spatial-analysis/operators-accessibility.md:134` | Default extensive weight `a_st / A_s` with `A_s` the whole source area (`'zone'`). | The docs call this "tobler". tobler `area_interpolate` defaults to `allocate_total=True`, which divides by the overlap sum. `'zone'` is tobler `allocate_total=False`. | A source half outside the target system, value 100: this library's default transfers 50; tobler's default transfers 100. | Say "tobler with `allocate_total=False`" for `'zone'`, and that tobler's own default is `'overlap'`. |
| cells-joins-03 | inconsistent | CONFIRMED | `region-statistics/gpu-region-mask.ts:231-248` against `253-259` | Rectangle regions are closed on all four sides. Lasso polygons are half-open (left and bottom edges in, right and top edges out). | luma.gl `GPUPointSpatialQuery` includes boundary points for bounds and for polygons. | Point (1, 0.5). Rectangle region [0,0]-[1,1]: selected. The same square as a 4-vertex lasso: not selected. | Pick one rule for both region kinds and state it. Closed matches luma.gl. |
| cells-joins-04 | inconsistent | CONFIRMED | `neighbor-search/neighbor-search-parameters.ts:112-116`, `neighbor-search-kernels.ts:534, 555`; `spatial-join/gpu-nearest-feature-weights.ts:88-95, 350` | Inverse-distance weight of two coincident points: `GPUNeighborSearch` gives 0 (default `distanceFloor` 0, non-finite weight written as 0, and a zero weight is "not a neighbor"). `GPUNearestFeatureWeights` gives `1e6^power` (default `distanceFloor` 1e-6). | One library, one inverse-distance rule. libpysal divides by zero here. | Two points at the same position, `power: 1`, defaults: weight 0 from one contributor, 1,000,000 from the other. | One shared default and one statement of the coincident-point rule. |
| cells-joins-05 | wrong-edge-case | CONFIRMED | `cell-table-compare/gpu-cell-table-compare.ts:588-590` | Poisson z `delta / sqrt(before + after)` is also accepted for `measure: 'sum'`. | The statistic is defined for counts. The doc says only "0 when both are 0". | Sums before = -3, after = 1: `sqrt(-2)`, z = NaN. Sums before = -5, after = 5: denominator 0, z = 0 although delta = 10. | Reject `zScore: 'poisson'` with `measure: 'sum'`, or document it as counts only. |

### cells-joins-01. Fast polygon `contains` / `within` accepts a polygon that fills a hole

`getContainsWGSL` (polygon outer, polygon inner) runs three tests: every inner edge lies in the
closed outer polygon; an inner edge that runs along the outer boundary must have the outer
interior on the inner interior's side; no outer vertex is strictly inside the inner polygon. The
second test is what rejects an inner polygon equal to a hole. It is guarded by
`outerLocate(middle, outer) == 2u`, where `middle = (ia + ib) * 0.5` is rounded to f32
(`spatial-predicate-wgsl.ts:592-593`). `Locate` uses the exact orientation sign, so it returns 2
only if the rounded midpoint is exactly collinear with the edge. For an oblique edge with generic
f32 endpoints the midpoint is usually not representable, `Locate` returns 0 or 1, and the test is
skipped for that edge. If it is skipped for every edge the kernel returns true.

The first and third tests do not catch the case: an inner edge that coincides with a hole edge is
"on the boundary" (`CollinearCovers`, code 1), and the hole vertices are on the inner boundary, not
strictly inside.

Measured with a Node replica of the kernel (f32 arithmetic through `Math.fround`, exact
orientation through BigInt): `contains(outer with hole, polygon equal to the hole)` returned true
for 220 of 400 random triangular holes, 174 of 400 with 4 vertices, 51 of 400 with 8, 14 of 400
with 12, 8 of 400 with 16, 0 of 400 with 24. The reference answer is always false. The same
replica gives the right answer for the axis-aligned square hole.

The only spec for this case uses an integer, axis-aligned square hole
(`test/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.spec.ts:417-430`), where every
midpoint is exact. So the spec passes and the defect is not pinned.

Reach. `within(a, b)` is `contains(b, a)` and shares the kernel. `engine: 'auto'` uses this fast
kernel whenever the product of the two average vertex counts is below 1024
(`gpu-spatial-predicate-join.ts:169, 182-190`), which is exactly where holes have few edges. The
relate engine evaluates the DE-9IM pattern `T*****FF*` and is not affected (its WGSL was not
read). The class documentation says the result "does not depend on the engine"
(`gpu-spatial-predicate-join.ts:296-299`); for this input it does.

A second way into the same hole: with large planar coordinates the probe offset
`normal = (-edge.y, edge.x) * 0.001` (`:595`) is below one ulp, the probe equals the midpoint, both
`innerLocate` calls return 2, and the test is skipped even when the midpoint is exact. Example: a
10 m edge at a UTM northing of 5,000,000 (ulp 0.5 m, offset 0.01 m). This variant is SUSPECTED; it
was not run.

### cells-joins-02. `'zone'` is not tobler's default

`modules/analysis/test/oracles/parity-wp1-areal-interpolation.py:18-19` states it correctly:
"allocate_total=False matches extensive denominator='zone'; True matches 'overlap'. Tobler's
intensive denominator is always overlap." The class documentation and the user documentation say
`'zone'` is tobler. The code is defensible (lost mass is reported in
`sourceTransferredFractions`); only the label is wrong. A user who expects tobler's default
numbers gets smaller extensive totals wherever a source extends beyond the target system.

### cells-joins-03. Two boundary rules in one region mask

Derivation for the lasso (`gpu-region-mask.ts:237-246`). Square (0,0), (1,0), (1,1), (0,1), point
(1, 0.5). The right edge straddles y = 0.5 but its orientation sign is 0, so it counts no
crossing. The left edge straddles and the point is to its right, so no crossing. Result: outside.
A point on the left edge (0, 0.5): the left edge has sign 0, the right edge counts one crossing.
Result: inside. The rectangle region tests `point >= minimum && point <= maximum` and includes
both. The code comment documents the lasso rule ("PNPOLY, half-open in y"), so each rule is
deliberate; the inconsistency is between the two.

### cells-joins-04 and cells-joins-05

Both are stated in full in the table. Neither changes results on ordinary input.

## 3. Verified correct

Each line names what the math was compared against.

- `GPUPointToCell` Quadbin (`cell-aggregation/cell-keys.ts:243-446`): bit layout
  (`0x4000…`, mode 1 at bit 59, zoom at bit 52, x on even and y on odd Morton bits, all-ones
  footer) against the CARTO quadbin-py `tile_to_cell`. Column `floor((lng + 180) 2^z / 360)` with
  longitude 180 wrapping to column 0, and row from `psi = ln(cos t) - ln(sin t)`,
  `t = (90 - |lat|) pi / 360`, against the Mercator formula of `point_to_tile_fraction`. Fixed-point
  scales, the long division, the atanh series and the floor rule for north and south rows were
  re-derived by hand.
- `GPUPointToCell` quadkey and geohash (`cell-indexing/cell-index-wgsl.ts:49-95`): quadkey digit
  `xBit | yBit << 1`; geohash bit counts `ceil(5L/2)` and `floor(5L/2)`, longitude bit first,
  interleave parity for odd and even totals. Derived by hand.
- `GPUPointToCell` S2 (`cell-index-wgsl.ts:112-199`): face choice, the six `uv` formulas, the
  quadratic `uv -> st`, `kPosToIJ` and `kPosToOrientation`, initial orientation `face & 1`, cell id
  layout, against the S2 C++ source from memory. A float64 replica of the kernel formula matched
  the `s2-geometry` npm package on 20,000 random points at levels 1 to 20 (0 mismatches).
- `GPUPointToCell` H3, f32 and extended (`h3-index-wgsl.ts`, `h3-index-extended-wgsl.ts`):
  `_hex2dToCoordIJK`, `_upAp7`, the `_faceIjkToH3` digit walk, unit-ijk to digit, the
  counter-clockwise rotation table, `_h3RotatePent60ccw`, the clockwise-offset rule, the gnomonic
  scaling and the Class III rotation by `-asin(sqrt(3/28))`, line by line against the H3 C library.
  The extended path uses the same steps in double-float arithmetic; its face basis is built from
  the published `dggs` face centres and azimuths with the H3 axis convention.
- H3 cell boundary (`h3-boundary-wgsl.ts`): `_adjustOverageClassII` with the substrate variant,
  `_adjustPentVertOverage`, `_faceIjkToCellBoundary`, `_faceIjkPentToCellBoundary`, the face-edge
  intersection and the IJ / JK / KI edge selection, against H3 `faceijk.c`. One difference: the
  "intersection at a vertex" test is an exact integer test instead of `_v2dAlmostEquals`.
- `GPUCellMeasures` (`h3-measures-wgsl.ts`): triangle fan from the cell centre as in H3
  `cellAreaRads2`; the spherical excess `2 atan2(det, 1 + a.b + b.c + c.a)` and its non-unit form
  on the gnomonic plane; lattice cross product `sqrt(3)/2 (iA jB - jA iB)`; edge length
  `atan2(sqrt(|d|^2 + (qA x d)^2), 1 + qA.qB)`. Derived by hand. Radius is the authalic radius, as
  in H3.
- H3 neighbour step (`h3-neighbor-wgsl.ts`): against H3 `h3NeighborRotations`, including the
  Class II / Class III table naming (the local `CLASS_III` table is H3 `NEW_DIGIT_II`), the deleted
  K direction and the pentagon rotations.
- H3 children and parent (`cell-topology-wgsl.ts:149-197`): pentagon child count
  `(5 * 7^d + 1) / 6` and the rank to digit rule (digit 0 stays on the pentagon chain, digit 1
  deleted). Compaction subtree sizes `p[d] = 7 p[d-1] - 1`. Derived by hand.
- `GPUH3CellIdentities` (`gpu-h3-cell-identities.ts:79-143`): directed-edge and vertex mode bits,
  `vertexRotations`, the direction and vertex-number tables, the `pentagonDirectionFaces` table and
  lowest-index ownership, against H3 `vertex.c` and `directedEdge.c`.
- `GPUCellGridPath` (`h3-local-ij-wgsl.ts`, `gpu-cell-grid-path.ts:240-300`): grid distance and
  `cubeRound` (half away from zero, largest error recomputed) against H3 `localij.c` and
  `algos.c`. The documented deviation at exact ties was measured against h3-js: 6 of 3,999 random
  pairs (8 of 138,491 steps) take the other adjacent cell. This matches the class documentation
  ("about 0.2% of random pairs"), so it is not a finding. The line on `output.cells`
  (`gpu-cell-grid-path.ts:44-46`) says "the cells of H3 `gridPathCells`" without that caveat.
- Quadbin topology (`cell-topology-wgsl.ts:22-133`): parent, child, Chebyshev disk with
  longitude wrap and no latitude wrap. Derived by hand.
- `GPUS2CellMeasures` (`s2-analysis-wgsl.ts:26-48`): `du = 4/3 step (2s + step)` and its mirror,
  the level 0 special case, the two-triangle exact area as in `S2Cell::ExactArea`, the four edge
  lengths. Derived by hand.
- `GPUS2CellHierarchy` (`gpu-s2-cell-hierarchy.ts`): parent `(id & -lsb) | lsb`, children
  `id - lsb + lsb/4 + i lsb/2`, leaf range `id -+ (lsb - 1)`, and the integer face transition for
  edge neighbours with the six S2 face frames. Derived by hand.
- `GPUS2CapCovering` (`gpu-s2-cap-covering.ts`): face cell ids, the exterior and interior
  selection rules, and the pruning rule of the level-synchronous form (a kept parent bounds every
  selected child for both modes). `GPUS2PolygonCovering`: the winding angle
  `atan2(p.(a x b), a.b - (p.a)(p.b))` and the shell / hole rule. Reasoned, not run.
- HEALPix (`healpix-index-wgsl.ts`): equatorial and polar-cap branches, face rule, `ix` / `iy`,
  nested index and UNIQ `4^(order+1) + nested`, against the HEALPix C `ang2pix_nest_z_phi` and
  `xyf2nest`. `1 - |z|` as `2 sin^2(colatitude / 2)` is a sound rewrite.
- `GPUHealpixCellAggregation`: key bit count, run numbering, counts and status. Derived by hand.
- `GPUHilbertKeys`: the parallel-prefix kernel equals the classic `xy2d` loop on 221,364 cells
  across orders 1 to 16, and the join copy (`spatial-join-passes.ts:173-191`) equals it on 200,000
  cells at order 16 (0 mismatches each). The stated start, end and first-move direction hold.
- `getBoundsReductionNodes`: grid-stride coverage and the empty case. Derived by hand.
- `GPUCellTableCompare`: union merge positions, exact 64-bit delta, population standard deviation
  from a centred second pass, ratio and percent change. Derived by hand.
- Cell table sums (`cell-table.ts:440-520`, `analysis-core/cell-key-wgsl.ts`): the two-atomic
  64-bit add carries correctly; the segmented tile scan is a correct Hillis-Steele scan on
  contiguous runs.
- `GPUArealInterpolation`: extensive `a_st / A_s`, intensive `a_st / B_t`, zone and overlap
  denominators, category shares, transferred fraction, and the spherical cell area
  `2 R^2 dLon cos(mid) sin(dLat / 2)` with mirrored rows for north-up rasters. Against tobler
  (with the caveat of cells-joins-02) and by hand.
- `GPUPycnophylactic`: rook mean over in-area neighbours, additive restore `(T - M) / A`, clamp,
  multiplicative rescale, mass as density times cell area, relative convergence test, and the
  buffer parity of the settle step. The differences from Tobler (1979) and from tobler `pycno`
  (no relaxation factor) are documented in the class.
- `GPUZonalStatistics`: mean, weighted mean, two-pass sample variance `ssd / (n - 1)`, reliability
  weights `ssd / (V1 - V2 / V1)` as `numpy.cov(aweights, ddof=1)`, densities, the shoelace area
  relative to the first vertex with holes subtracted. Non-finite values never enter a count or a
  denominator.
- `GPUPointInPolygonJoin`: the even-odd crossing rule and the half-open owner. The owner is the
  parity for the point moved by `(eps^2, eps)`; on an edge's supporting line the moved sign is
  `sign(dx)`, or `sign(-dy)` for a vertical edge, which is what `:685-688` computes. Shared edges
  and vertices of a valid coverage get exactly one owner.
- `GPUSpatialPredicateJoin`: the DE-9IM patterns of all ten named predicates against the OGC and
  JTS definitions (`spatial-relate-types.ts:166-208`); `dwithin` is inclusive (`<=`); the planar
  workgroup `dwithin` is never used with a great-circle metric
  (`gpu-spatial-predicate-join.ts:656-657`).
- `GPUNearestFeatureJoin`: `(distance, feature row)` order, inclusive pruning so ties at the k-th
  distance are resolved by the lowest row, and the great-circle box lower bound
  `hav(latGap) + cos^2(farLat) hav(lonGap)`.
- `GPUNeighborSearch`: inclusive band `d <= radius`, `(d^2, id)` order with a strict stop test,
  3 x 3 cells sufficient for radius-wide cells, the cap longitude half-span
  `asin(sin a / cos lat)`, the spherical stop bound of the great-circle kNN, and the five kernel
  profiles and the adaptive bandwidth factor 1.0000001 against libpysal `Kernel`.
- `GPUCellCover` (documentation and the count / test kernels only): the H3 lattice longitude
  spacing is taken at the most equatorward latitude, which is the conservative choice; Quadbin
  `'full'` and `'intersects'` follow their definitions.

## 4. Standards consistency

Two conventions for one thing:

- Inverse-distance weight at distance 0: see cells-joins-04.
- Region boundary rule: see cells-joins-03. Across the scope there are three point-on-boundary
  rules: half-open owner (`GPUPointInPolygonJoin` default, zonal statistics), closed
  (`GPURegionMask` rectangle, `GPUPairwisePointInPolygon` returns a separate boundary class),
  PNPOLY half-open (`GPURegionMask` lasso). Each is documented where it is used.
- Hilbert quantization: `GPUHilbertKeys` uses cells `floor(u 2^order)`
  (`gpu-hilbert-keys.ts:284-289`); the joins use lattice points `round(u 65535)`
  (`spatial-join-passes.ts:162-167`). The curve is the same. Only the sort order near cell borders
  differs; no result depends on it.
- Zero key as "no cell": every `GPUPointToCell` family uses the zero key for invalid rows. HEALPix
  nested key 0 is a real cell, so there the validity view must be read (documented,
  `gpu-healpix-point-to-cell.ts:31-35`).
- Invalid cell output: `GPUCellGeometry` writes NaN centres; `GPUS2CellMeasures` writes NaN;
  `GPUCellMeasures` writes 0 for invalid cells.
- Units: `GPUCellMeasures` returns metres on the authalic sphere; `GPUS2CellMeasures` returns
  steradians and radians (already noted in the 2026-11-09 review).
- "Distance squared" names: in great-circle mode `pointSegmentDistanceSq`, `segmentDistanceSq` and
  `distanceLimitSq` carry central angles in radians, not squares
  (`spatial-predicate-wgsl.ts:98-102`, `nearest-wgsl.ts:163-166`). The comments say so. Both sides
  of every comparison use the same quantity.

Private copies of something that already exists. All are numerically the same as their twin
unless noted.

- Morton spread: `quadbinSpread16` (`cell-keys.ts:431-438`), `cellTopologyQuadbinSpread16`
  (`cell-topology-wgsl.ts:25-32`), and `SPREAD_BITS_WGSL` in `analysis-core/kernels/wgsl-common.ts`.
  Inverse: `geometryCompactEvenBits` (`gpu-cell-geometry.ts:323-330`) and
  `cellTopologyQuadbinCompact16` (`cell-topology-wgsl.ts:34-41`).
- H3 digit rotation: `cellIndexH3RotateDigitsCcw`, `cellIndexH3RotatePentagonDigitsCcw` and the
  clockwise-offset test in `h3-index-wgsl.ts:67-206`, and the same three in
  `h3-neighbor-wgsl.ts:80-141`. Two clockwise-offset tables with the same content
  (`h3-index-tables.ts`, `h3-neighbor-tables.ts:119-132`).
- S2 encoder: `cellIndexS2FromLngLat` (`cell-index-wgsl.ts:187-197`) and `s2Encode`
  (`gpu-s2-cell-hierarchy.ts:124-135`); the tables are shared, the loop is duplicated.
- Sine and cosine of degrees: `cellIndexS2SinCosDegrees` (terms to x^9 and x^10) and
  `cellIndexH3SinCosDegrees` (to x^11 and x^12). Different truncation, both below f32 rounding on
  [-45, 45] degrees.
- Spherical triangle excess: `sphericalExcessTriangle` (`h3-measures-wgsl.ts:9-14`) and
  `s2AnalysisTriangle` (`s2-analysis-wgsl.ts:23-25`).
- `atan2` with a zero argument (driver issue FX-17): three local fixes,
  `cellIndexH3BoundaryUnitVectorToLngLat`, `geometryS2XyzToLngLat`, `s2AnalysisAtan2`.
- Even-odd point in polygon: `classifyExactly` and `classifySlabs`
  (`gpu-point-in-polygon-join.ts`), `${p}Locate` (`spatial-predicate-wgsl.ts:366-378`),
  `${p}Contains` (`nearest-wgsl.ts:255-268`), the lasso (`gpu-region-mask.ts:237-246`), and
  `coverContains` (not read). The four read use the same half-open y rule and the shared
  `orientSign`.
- Two-word 64-bit helpers: `CELL_KEY_WGSL` and `cellAddI64` re-implement `dggs_u64_*`. The source
  gives the reason (compile cost of the 2.3k-line `dggs` module).
- Lower-bound binary searches over two-word keys in `gpu-cell-table-compare.ts:271-286` and
  `gpu-areal-interpolation.ts:930-942`.
- Grid cell mapping: `getCoordinate` (`region-statistics-grid.ts:171-185`) copies the published
  `GPUGridIndex` mapping. The query is widened by one cell, so a difference cannot lose a match.
- Scan totals written by hand (`gpu-pycnophylactic.ts:399-402`): already in the 2026-11-09 review.

Literals where a named constant exists or should: the Mercator latitude limit `85.051129`
(`cell-keys.ts:201`, `cell-cover-kernels.ts:310`), `0.017453292519943295` in at least six
kernels, `1.5707963267948966` and `3.141592653589793` in the S2 files. The Earth radii come from
`analysis-core/contracts/earth.ts` everywhere in scope (authalic for areas and H3 lengths, mean for
great-circle distances).

Shared numerics used as intended: exact orientation (`orientSign`) in every join predicate,
`GEODESIC_WGSL` for haversine, published `GPUSort`, `GPUScan`, `GPUReduction`, `GPUHistogram`,
`GPUGather`.

## 5. luma.gl official implementations

- `GPUDGGSCellProjection` (`@luma.gl/graph/gpu-dggs`). It covers decoding only: cell key to centre
  longitude/latitude or unit vector, for H3 and A5, with a validity flag and a word-order option.
  It has no forward indexing, boundaries, measures, neighbours, S2, Quadbin or HEALPix.
  `GPUCellGeometry` does not instantiate the class. It calls the same `@luma.gl/shaders` function
  (`dggs_h3_get_center_lnglat`, `dggs_a5_get_center_lnglat`) inside its own kernel, because it
  also writes boundaries in the same pass, so the two agree exactly on H3 and A5 centres. They
  differ on invalid cells: the official class writes zeros plus `validity`; the local one writes
  NaN. No local forward-index kernel duplicates the official class.
- Where local code departs from the published `dggs` decoders, the local version is the better
  one and should go upstream: `geometryS2XyzToLngLat` replaces `dggs_s2_xyz_to_lnglat` on the axes
  (`gpu-cell-geometry.ts:380-395`); `cellIndexH3BoundaryHex2dToUnitVector` replaces the
  `dggs_h3_hex2d_to_lnglat` azimuth-distance chain (`h3-boundary-wgsl.ts:136-152`, stated loss
  about 1e-4 degrees in f32); the H3 boundary adds distortion vertices and pentagons. The H3 and S2
  encoders, H3 neighbour step, local IJK and vertex identities are generic DGGS code with no
  published counterpart. Propose them upstream.
- `GPUPairwisePointInPolygon` (`geospatial`). `GPUSpatialPredicateJoin` delegates to it for
  point/polygon pairs (`gpu-spatial-predicate-join.ts:1311`). `GPUPointInPolygonJoin` re-implements
  the classifier per candidate. The two agree on the crossing rule (a vertex at the point's y
  counts as below) and on the class codes, which the local kernel takes from the published
  `GPU_POINT_IN_POLYGON_CLASSIFICATION`. They differ in three places: the local kernel adds the
  half-open owner bit; an empty polygon part is skipped locally and UNCERTAIN upstream; a ring with
  fewer than three vertices is skipped locally and UNCERTAIN upstream. Neither is wrong. To
  delegate, propose an owner output upstream.
- `GPUPointSpatialQuery` (`geospatial`). It is a one-region query (bounds, radius, polygon) over a
  `GPUGridIndexView`. Its local counterparts are `GPURegionMask` and the grid path of
  `GPURegionStatistics`, which re-implement the candidate range and the predicates. The published
  polygon test is closed and plain f32; the local lasso is half-open with exact signs
  (cells-joins-03). The published radius test is inclusive, as is `GPUNeighborSearch`. It is not a
  batch kNN, so `GPUNeighborSearch` does not duplicate it.
- `GPUGridIndex` (`gpu-core`). `GPUNeighborSearch` delegates: it writes each target's cell centre
  as a unit position and lets the published index bucket it (`gpu-neighbor-search.ts:694-728`), so
  the kernel's own cell formula and the index always agree.
- `GPUBVH` (`gpu-core`). All joins build candidates with it. The two traversals (stackless probe,
  near-first stack) are local.
- `GPUBatchHashJoin` (`gpu-core`). Not used in scope and not needed: attribute filters compare
  keys per candidate.
- No private path (`_internal`, deep `dist`) is imported in the files read.

## 6. Test oracles

Pinned by an independent reference library:

- H3 point to cell: h3-py fixture (`test/oracles/cells-h3.py`, `cell-indexing-h3.json`) and h3-js
  (`h3-index-wgsl.node.spec.ts`, `h3-index-extended.spec.ts`).
- H3 boundary, neighbours, disk, compaction, cover, outline, vertex and directed-edge identities:
  h3-js (`h3-boundary-wgsl.spec.ts`, `h3-neighbor-wgsl.node.spec.ts`, `cell-topology-oracle.ts`,
  `cell-compaction-oracle.ts`, `cell-cover-oracle.ts`, `cell-set-outline-oracle.ts`,
  `geospatial-expansion/network-expansion.spec.ts:228-236`).
- HEALPix: healpix-geo fixture (`rts08-healpix-geo.json`).
- S2 polygon covering and S2 cell area: `s2geometry==0.14.0` fixture
  (`global-coordinates/fixtures/s2-polygon-covering.json`).
- Areal interpolation, dasymetric weights, pycnophylactic: tobler and shapely fixtures
  (`parity-wp1-areal-interpolation.py`, `parity-wp1-areal-modes.py`,
  `dasymetric-redistribution.py`, `parity-wp1-pycnophylactic.py`).
- Zonal statistics, region statistics, nearest weights, buffer selection, cell tables: `parity-wp1`,
  `parity-wp2` and `parity-wp4` fixtures. Their generators were not read.

Pinned by an independent re-implementation written for the tests (not a library):

- S2 point to cell: a BigInt port of `S2CellId::FromFaceIJ` (`cell-indexing-oracle.ts`). This
  review added a check against the `s2-geometry` package (0 mismatches in 20,000).
- Quadkey, geohash: float64 bisection oracles.
- Hilbert keys: the Wikipedia `xy2d` and `d2xy` loops (`hilbert-oracle.ts`).
- Join predicates, nearest, neighbour search: float64 CPU oracles
  (`spatial-predicate-oracle.ts`, `nearest-feature-oracle.ts`, `neighbor-search-oracle.ts`).

Pinned mainly by a copy of its own formula:

- Quadbin row: the BigInt oracle imports `QUADBIN_FIXED_POINT_CONSTANTS` from the source and
  repeats the fixed-point steps (`cell-aggregation-oracle.ts`). One spec compares against the
  float64 formula and reports a mismatch rate (`cell-aggregation.node.spec.ts:92-121`), which is
  the only independent check.
- `GPUCellTableCompare`: the oracle "mirrors the kernel's f32 steps"
  (`cell-table-compare-oracle.ts`). The z-score formulas have no independent check; they are
  simple enough to verify by reading (done above).

Gaps:

- Polygon `contains` / `within` with a hole filler on oblique edges: not covered
  (cells-joins-01). The only case is an integer square.
- `GPUCellGridPath` ties: the fixture origin (`h3-grid-path-fixtures.ts`) was not read.
- `GPUS2CellHierarchy` neighbours across faces and `GPUS2CapCovering`: reference of the recorded
  fixtures not established in this review.

## How findings were confirmed

Scratch scripts, in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/cells-joins/`:

- `contains-hole.mjs`, `contains-lib.mjs`: f32 replica of the fast polygon `contains` kernel with
  exact BigInt orientation (cells-joins-01).
- `gridpath.mjs`, `cube.mjs`: exact cube rounding against h3-js `gridPathCells` and against the
  double arithmetic of H3.
- `hilbert.mjs`: parallel-prefix Hilbert against `xy2d` and the join copy.
- `s2check.mjs`: S2 forward formula against the `s2-geometry` package.

cells-joins-02 was confirmed against the oracle generator's own description of tobler.
cells-joins-03, 04 and 05 were confirmed by hand derivation from the kernel text.

## Needs a second opinion

- cells-joins-01: the replica was written from the WGSL text and not run on a GPU. A headless spec
  with the triangle above would settle it in one run. The relate engine (`spatial-relate-wgsl.ts`)
  was not read; it should be checked for the same input before it is named as the fix.
- The 0.001 probe at large planar coordinates (second paragraph of cells-joins-01).
- `cell-cover` exact tests and the scanline enumeration, `cell-set-outline`, and the DE-9IM matrix
  engine were not reviewed.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. Every spec named below was run on the headless GPU (or
in Node for `*.node.spec.ts`) and passes. Each new GPU spec was run against the old kernel first and
failed there, except where the row says otherwise.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| cells-joins-01 | fixed | `src/gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl.ts:548-591` (new `getProbeInsideWGSL`), `:627-660` (polygon/polygon branch of `getContainsWGSL`). The midpoint and its `0.001` probe are gone. For every outer vertex on the inner boundary the kernel asks whether the point just beyond the vertex along the outer edge, just left and just right of it, is inside the inner polygon and not inside the outer one. The probe is symbolic: each comparison is decided by a coordinate comparison or an exact orientation sign of three input vertices. `docs/api/gpu-spatial-analysis/operators-joins.md` (exactness paragraph). | `test/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-hole-filler.spec.ts` (4 tests, engines `fast`, `auto`, `relate`, both `contains` and `within`). | Fast-engine `contains` / `within` of two polygons: false where it was true for an inner polygon that fills a hole with oblique edges, that is a hole plus adjacent interior, that has a hole of the outer polygon inscribed in it, or that fills a hole at large planar coordinates. |
| cells-joins-02 | fixed (documentation) | `src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts:87-91`, `:203-208`, `:275-279`; `docs/api/gpu-spatial-analysis/operators-accessibility.md` (denominator bullet and the parity sentence). `'zone'` is tobler with `allocate_total=False`; tobler's default is `'overlap'`. | None (words only). The oracle generator already states the mapping. | None. |
| cells-joins-03 | fixed | `src/gpu-spatial-analysis/region-statistics/gpu-region-mask.ts:237-262` (lasso `isInsideRegion`): a point in an edge's bounding box with exact orientation 0 is inside. Class and type documentation (`gpu-region-mask.ts:81-88`, `region-statistics-types.ts:82-85`), `docs/api/gpu-spatial-analysis/operators-statistics.md`. Convention: roadmap 12.3 (inclusive, as `GPUPointSpatialQuery`). | `test/gpu-spatial-analysis/region-statistics/gpu-region-mask-boundary.spec.ts` (2 tests). The Shapely parity specs `parity-wp1-region-statistics.parity.spec.ts` and `parity-wp1-region-selection-modes.parity.spec.ts` pass against the updated fixtures. | `GPURegionMask` and `GPURegionStatistics` with a lasso: a point exactly on a lasso edge or vertex is now selected on every edge (before: left and bottom edges only). Point (1, 0.5) in the unit-square lasso: 0 before, 1 now. |
| cells-joins-04 | fixed | One default, `1e-6`: `src/gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts:21-29` (new internal constant `GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR`), `:56` (schema default), `:219` (packed default); `src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights.ts:240` reads the same constant. Documentation in both classes, `docs/api/gpu-spatial-analysis/operators-weights.md`, `operators-joins.md`. Convention chosen: the `GPUNearestFeatureWeights` default, because a zero weight means "not a neighbor" in this library and a floor of 0 drops the closest possible neighbor (and every touching polygon in the nearest-feature case). Roadmap 12.3 and luma.gl have no rule; libpysal divides by zero. | `test/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.spec.ts` (default-floor block in the adaptive-kernel test), `gpu-neighbor-search.node.spec.ts` (packed default and oracle), `test/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights.spec.ts` ("default-floor weight 1e6"; this one passed before, it pins the shared number). | `GPUNeighborSearch` with `weightKind: 'inverseDistance'` and no `distanceFloor`: two coincident points weigh `1e6^power` (before: 0, and the slot was not a neighbor). An explicit `distanceFloor: 0` keeps the old result. `GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA` default for `distanceFloor` is `1e-6` (before: 0). |
| cells-joins-05 | fixed | `src/gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare.ts:592-599`: the Poisson score is `delta / sqrt(before + after)` only for a positive denominator, 0 when both values are 0, NaN otherwise; `sqrt` is never given a negative number. The combination `measure: 'sum'` with `zScore: 'poisson'` stays accepted (two existing specs use it for non-negative sums); it is documented as a count statistic (`:50-57`, `docs/api/gpu-spatial-analysis/operators-cells.md`). | `test/gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare.spec.ts` ("Poisson z-score of sums is NaN where before + after is not positive", hand-worked values). | Sums `before = -5`, `after = 5`: z-score NaN (before: 0). Negative `before + after`: NaN, now by rule instead of by the device's `sqrt` of a negative number. |
| luma-official-06 | fixed | `src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts:10-47`: `parseDggsFaceTable` requires 20 rows of the expected width with finite numbers and throws `h3-index-extended: dggs shader table <name> has N rows; expected 20 rows of K finite numbers` at module load. | `test/gpu-spatial-analysis/cell-indexing/h3-index-extended.node.spec.ts` (2 new tests). | None with the published shader. A luma.gl release that moves the tables fails at import instead of compiling an empty table. |
| luma-official-09 | fixed | `src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts:76-82` (`output.validity`, `uint32`, 1 valid, 0 invalid), `:221-226`, `:269-276`, `:317`. NaN centres, zero vertex counts and zero boundary rows are unchanged. `docs/api/gpu-spatial-analysis/operators-cells.md`. | `test/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.spec.ts` (2 new tests: every family, and H3 and A5 against published `GPUDGGSCellProjection`: validity equal on every row, valid centres bit-identical). | New optional output only. |
| luma-official-12 | not reproduced (the two classifiers agree) | No source change. | `test/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-classifier-agreement.spec.ts` (4 scenes: shared edges and vertices, the same at projected-metre coordinates, an oblique shared edge with points one float32 step either side, a hole with its filler and a two-vertex ring that the published classifier reports as uncertain). `GPUPointInPolygonJoin` with `boundaryRule: 'closed'` equals `intersects`, `'open'` equals `within`, and both equal hand-written owners. | None. |
| luma-official-02 (the `region-statistics-grid.ts:172` copy) | fixed (pinned, arithmetic unchanged) | No source change. | `test/gpu-spatial-analysis/region-statistics/region-statistics-grid-coordinate-pin.node.spec.ts` compares the copy with the text of the published `GPUGridIndex` kernel, the convention of `trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts`. | None. |
| H3 cell area, fallback cells (from geometry-measures-01) | fixed | `src/gpu-spatial-analysis/cell-indexing/h3-measures-wgsl.ts:153-162` (new `cellMeasuresGetTriangleExcess`, determinant `dot(center, cross(first - center, second - center))`), `:178`. `sphericalExcessTriangle` is left as it was. Accuracy statements updated in `h3-measures-wgsl.ts:33-43`, `gpu-cell-measures.ts:107-113`, `docs/api/gpu-spatial-analysis/operators-cells.md`. | `test/gpu-spatial-analysis/cell-indexing/gpu-cell-measures.spec.ts` (per-resolution area limits for fallback cells up to resolution 10, against python h3). | Areas of pentagons and of cells that straddle an icosahedron edge. Worst relative error against python h3 `cell_area`, before to after: resolution 4, 2.5e-4 to 1.8e-5; 5, 4.1e-3 to 5.5e-5; 6, 1.0e-2 to 9.2e-5; 8, 3.0e-2 to 1.5e-4; 9, a factor of 30 to 5.0e-3; 10, a factor of 47 to 2.3e-3; 14, a factor of 2.9e5 to 0.69. |

Notes.

- **cells-joins-01, why the fix is larger than the suggested one.** The GPU run confirmed the
  finding (the three oblique hole fillers and the hole-plus-triangle case were all reported as
  contained), and also the suspected second way in: a 10 m hole at a northing of 5,000,000 was
  reported as contained. It showed a third case the review did not list: a hole whose vertices lie
  on the inner polygon's edges away from the edge midpoints (inscribed hole). The old probe caught
  that only when a hole vertex was an edge midpoint. The new test covers all three. Argument: once
  every inner edge lies in the closed outer polygon, a part of the outer exterior inside the inner
  polygon is a whole hole; each of its vertices is strictly inside the inner polygon (the existing
  vertex test) or on its boundary, and from such a vertex the hole edge either heads into the
  inner interior or runs along an inner edge with the inner interior on the hole's side. Both
  cases put a point of the inner interior, next to that vertex, outside the outer polygon, which
  is what the probe tests. The relate engine gave the right answer on every case of the new spec
  before and after, so the two engines now agree there. `within` shares the kernel. `covers` and
  `coveredBy` always use the relate engine and were not affected.
- **cells-joins-03, fixtures.** `test/oracles/fixtures/parity-wp1-round5-region-real.json` and
  `-small.json` held Shapely expectations with an explicit half-open adapter for the lasso. Their
  two lasso queries now hold the `shapely.covers` expectation (2 more rows in the Chicago query, 4
  in the small one). Shapely, numpy and pandas are not importable on this machine, so the rows were
  re-derived with standard-library Python: the script first reproduced the stored half-open
  expectations exactly (mask, ids, counts, statistics, histogram), then switched the boundary rule.
  The fixtures say so in `provenance.lassoBoundaryRule`. The generator
  `test/oracles/parity-wp1-round5.py` now calls `shapely.covers` for the lasso; it was **not run**.
  Closing pass (2026-11-11): it was run with Shapely 2.2.0, every value of the patch is reproduced, and both fixtures are now its output (without the `provenance.lassoBoundaryRule` note); see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).
  The test twin `isInsidePolygon` (`test/gpu-spatial-analysis/region-statistics/region-statistics-fixtures.ts`)
  is inclusive too. Script:
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/fix-cells-joins/patch-region-fixtures.py`.
- **cells-joins-04 and cells-joins-05, twins.** The CPU oracles that encoded the old rules were
  changed with the kernels: `test/gpu-spatial-analysis/neighbor-search/neighbor-search-oracle.ts`
  (default floor) and `test/gpu-spatial-analysis/cell-table-compare/cell-table-compare-oracle.ts`
  (Poisson rule).
- **H3 and S2 cell areas at fine resolutions.** Checked against python h3 `cell_area` (the
  committed fixture has cells at resolutions 0 to 15). Hexagons on one icosahedron face, which are
  all but a thin set of cells, never used `sphericalExcessTriangle`: they take an exact lattice
  path and are within 1e-6 at every resolution, 12 to 15 included (the existing spec pins 2e-6).
  Only the fallback cells cancelled, and they are the row above. Their remaining error is the
  resolution of f32 unit vectors (about 0.4 m), so they stay unusable beyond resolution 10; a
  further gain needs the vertices in a face plane, not a better triangle formula. S2 cell areas
  use `s2AnalysisStableMeasures` (the determinant is the exact product of the two cell increments),
  not a triple product of unit vectors; nothing to change. `s2AnalysisTriangle` already has the
  difference form and has no caller.
- **Verified-correct remark on `GPUCellGridPath`.** The `output.cells` line now carries the tie
  caveat of the class documentation (`src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path.ts:43-48`).

Deferred.

- `GPUCellGeometry` centres through `GPUDGGSCellProjection` (luma-official order of work, item 12;
  optional in this pass). Not delegated. The parity spec passes, and delegation is possible through
  `captureGraphCommandNodes`, but the published class writes zeros for invalid cells and has no
  boundary output, so keeping today's outputs would take the published pass, a pass that rewrites
  invalid centres to NaN, and the local kernel again for boundaries, for bit-identical numbers.
- Lens records and lens operator text that restate the old rules were not touched (roadmap 12.4,
  `modules/lenses/**` and `docs/lens-records/**`): `modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/cells.ts:312`
  still says fallback measures are unreliable beyond resolution 5, and the kernel twins under
  `docs/lens-records/twins/` for `cell-table-compare`, `region-mask` and the joins hold the old
  WGSL until they are regenerated.

Upstream gaps.

- `GPUDGGSCellProjection`: an option for the value written to an invalid cell (NaN), and boundary
  and vertex-count outputs in the same pass, are what `GPUCellGeometry` would need to delegate.
- `@luma.gl/shaders` `dggs`: the H3 face centre and axis azimuth tables as exported data, so
  `h3-index-extended-wgsl.ts` does not parse shader text.
- `GPUGridIndex`: a public cell-coordinate WGSL snippet (shared with the other four copies named
  in luma-official-02).
- `GPUPointSpatialQuery` polygon test: it is inclusive but plain f32; the local lasso needs exact
  orientation signs, a per-frame vertex count and a screen transform.

Not checked.

- Test files are not covered by the shared typecheck watcher (`tsconfig.check-src.json` includes
  `modules/*/src` only), and `yarn typecheck-tests` was not run in this pass. The changed and new
  specs run under vitest.
- No shader timing was taken for the new polygon `contains` test. It adds work only for outer
  vertices that lie exactly on the inner boundary.

Compensated sums and error-free transformations (lead's note on the Metal fast-math compiler).

- `grep -rliE "neumaier|kahan|compensat|twoSum|two-sum"` over the scope directories has one hit:
  `src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts`, the private double-float
  library of `GPUPointToCell` `precision: 'extended'` (two-sum, quick two-sum, Veltkamp split,
  Dekker product). No Kahan or Neumaier sum exists in scope.
- That library already routes every intermediate through `cellIndexH3ExtKeep` (an integer OR with
  a runtime zero), written for this compiler behaviour. Checked on the GPU, not with a CPU twin:
  `test/gpu-spatial-analysis/cell-indexing/h3-index-extended.spec.ts` gets 0 mismatches against
  h3-js at every resolution 0 to 15 on uniform, cell-edge and pentagon-neighbourhood points, and
  now also asserts in the same run that the float32 path misses more than 0.5% of the same uniform
  points at resolutions 12 to 15. If the low words were folded away the two paths would be equal,
  so the transformations survive. Outcome: not reproduced, no source change.
- Fixes of this pass that could be affected the same way. The polygon `contains` probe is symbolic
  and uses comparisons and the integer orientation fallback only. The H3 fallback determinant
  `dot(center, cross(first - center, second - center))` is algebraically the old expression, so a
  compiler could in principle expand it back; the headless spec's per-resolution limits fail on
  the old form (4.1e-3 at resolution 5 against a limit of 6e-4), so the difference form is what
  the GPU evaluates today, and the spec will show it if a compiler change undoes that.

### Review follow-up

From `fix-review-geometry.md` (row fr-geometry-01) and the gates.

- **A third reader of the patched region fixtures was missed.**
  `modules/analysis/test/gpu-spatial-analysis/neighborhood-summary/parity-wp1-round5.parity.node.spec.ts:206-217`
  still asserted the half-open lasso on `parity-wp1-round5-region-real.json` (rows 29299 and 34963
  excluded, count 2833) and failed in `yarn test-node`. It now expects the inclusive rule: both
  rows 1, count 2835, and for the small fixture the seven boundary rows and counts `[7, 5, 0]`.
  The cells-joins-03 row above named two parity specs as passing and did not mention this one; I
  had searched `region-statistics` only. A search of `modules/analysis/test`, `modules/lenses`,
  `modules/kepler` and `test` for `round5-region`, `region-real`, `region-small` and for the
  half-open lasso rule finds no other reader: the two `region-statistics` parity specs (already
  updated through the fixture) and this spec are all. `test/oracles/parity-wp1-zonal-half-open.py`
  is the zonal-statistics adapter, a different contributor whose half-open rule did not change.
- **Consequence of the `1e-6` default floor (cells-joins-04), now stated.** Two distinct rows at
  the same position weigh `1e6^power` to each other, so with `rowStandardize` the coincident
  neighbor takes nearly all of the row. Added to the `GPUNeighborSearch` class documentation
  (`src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts`, bullet "Coincident points
  under the default floor"), the `distanceFloor` setting
  (`neighbor-search-parameters.ts`), `docs/api/gpu-spatial-analysis/operators-weights.md` (weights
  bullet and the `distanceFloor` table row) and `operators-joins.md` (`GPUNearestFeatureWeights`).
  The advice in each place: pass a floor on the scale of the data when positions repeat, or
  `distanceFloor: 0` to leave duplicates out. No code change.
