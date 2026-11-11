# Math correctness review: terrain hydrology, processing, features, geomorphons (2026-11-10)

Scope id `terrain-hydrology`. Directories under `modules/analysis/src/gpu-terrain/`: `hydrology`,
`terrain-processing`, `terrain-flow-field`, `terrain-features`, `geomorphons`. Paths below are
relative to `modules/analysis/src`.

Result in one paragraph. The core hydrology is right: the fill, D8, D-infinity, both multiple flow
routings, flat resolution, both accumulation paths, Strahler order, watersheds, height above
drainage, the wetness and stream power indices, prominence and geomorphons all compute what their
references say. I found no `wrong-result`. I found one `wrong-edge-case` (no-data cells block
branching transport), two `precision` findings, one `inconsistent` and three `doc-mismatch`, one
of which (TauDEM and Whitebox direction grids) leads to silently wrong output for a caller who
believes the documentation.

Scratch scripts are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/terrain-hydrology/`
(`branching-nodata.mjs`, `catchment-sum.mjs`, `d8-codes-and-fixpoint.mjs`, `hierarchy-flat.mjs`).
No existing spec file was run (0 of 5 allowed runs used).

## 1. Coverage

Read in full (every kernel body and parameter packer; import blocks skimmed):

- `gpu-terrain/hydrology/`: `gpu-terrain-flow.ts`, `terrain-flow-passes.ts`,
  `terrain-flow-routing.ts`, `terrain-flow-euler-tour.ts`, `terrain-flow-flats.ts`,
  `terrain-flow-spill-fill.ts`, `terrain-flow-spill-tree.ts`, `terrain-flow-spill-utils.ts`,
  `terrain-flow-tracing.ts`, `terrain-flow-unresolved.ts`, `gpu-terrain-watersheds.ts`,
  `gpu-terrain-height-above-drainage.ts`, `gpu-terrain-stream-order.ts`,
  `gpu-terrain-hydrologic-indices.ts`, `gpu-terrain-flow-distance.ts`,
  `gpu-terrain-stream-network.ts`.
- `gpu-terrain/terrain-processing/`: `gpu-terrain-branching-flow.ts`,
  `gpu-terrain-flow-fractions.ts`, `gpu-terrain-breach.ts`, `gpu-terrain-catchment-summary.ts`,
  `gpu-terrain-depression-hierarchy.ts`, `gpu-terrain-dem-coregistration.ts`,
  `gpu-terrain-tiled-flow.ts`, `terrain-tiled-resident.ts`, `terrain-processing-utils.ts`.
- `gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts`.
- `gpu-terrain/geomorphons/gpu-geomorphons.ts`.
- `gpu-terrain/terrain-features/`: `gpu-terrain-critical-points.ts`,
  `gpu-terrain-peak-prominence.ts`, `terrain-peak-prominence-tree.ts`,
  `terrain-features-utils.ts`, `gpu-terrain-summits.ts`, `gpu-terrain-peak-snap.ts`.

Read in part:

- `gpu-terrain/terrain-features/gpu-profile-peaks.ts`: lines 40 to 800 (documentation, candidate
  kernel, suppression kernel). Lines 330 to 440 (constructor checks) skimmed. Lines 800 to 961
  (output and compaction plumbing) not read.
- Dependencies outside the scope, read only as far as the scope needs them:
  `analysis-core/raster-grid.ts` (full), `raster-analysis/cost-distance/raster-grid-utils.ts`
  (full), `raster-analysis/cost-distance/raster-relaxation.ts` lines 150 to 480 (gates and gated
  loops), `raster-analysis/raster-expansion-utils.ts` (the Jacobi loop helper),
  `raster-analysis/raster-band-utils.ts` lines 60 to 240,
  `analysis-core/kernels/parallel-merge-tree.ts` lines 1 to 80.

Not opened, so not verified:

- The five `index.ts` files of the scope.
- The tiled relaxation engine (`raster-relaxation.ts` lines 596 to 1077). The epsilon fill and the
  three flat-resolution passes run on it. I checked only the candidate functions they pass in.
- `parallel-merge-tree.ts` after line 80. Peak prominence builds its join tree there.
- The extrema pyramid that `GPUTerrainSummits` reads (`raster-analysis`).

No file looked half-edited.

## 2. Findings

| ID | Severity | Status | `file:line` | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
|---|---|---|---|---|---|---|---|
| terrain-hydrology-01 | wrong-edge-case | CONFIRMED | `gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts:187-188`, fed by `gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts:150-152` | In `transport` mode a cell reads the fraction row of all eight neighbours. A no-data neighbour has a NaN row, so the cell is never ready. It and everything downstream end with status 2 and NaN. | A no-data cell donates nothing. `GPUTerrainFlow` does exactly that (`terrain-flow-routing.ts:168`, `:251` return 0 for an invalid donor). | 1 x 3 DEM `[NaN, 2, 1]`, D-infinity. `GPUTerrainFlow` accumulation is `[NaN, 1, 2]`. Fractions then branching transport gives status `[0, 2, 2]`, values all NaN. | Publish a zero row for a cell whose surface is not finite and keep NaN rows only for unresolved directions. Or let transport skip a donor whose own state is invalid because it has no data. |
| terrain-hydrology-02 | precision | CONFIRMED | `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:220-243` (sum at `:225`, mean at `:231`, hypsometry at `:240-242`) | Per basin, one thread adds float32 elevations and (under `cellSizeMode`) float32 cell areas one by one. | The mean and the area of the basin. | One basin of 1,000,000 cells at constant 2000 m: mean 2015.58 (error 15.6 m). Random 1500 to 2500 m: mean off by 0.46 m at 1,000,000 cells and 1.0 m at 4,000,000. Non-uniform cell area near 640 m2: area off by 0.6 % at 1,000,000 cells and 2.2 % at 4,000,000. Drainage density and hypsometry inherit the area error. | Use the shared fixed-point accumulator (contract section 11), or sum `value - lower` and count times mean area, or the compensated addition the sibling depression hierarchy already has. |
| terrain-hydrology-03 | precision | CONFIRMED | `gpu-terrain/hydrology/gpu-terrain-stream-network.ts:348-351` | Vertex positions are world coordinates formed in float32 from an affine baked into the shader. | Cell centres. | 1 m cells at northing 4,500,000.3, column 1234: exact 4,501,234.8, kernel 4,501,235.0 (0.2 cell). Longitude -122.123456 with 1e-5 degree cells: 0.35 cell off. 30 m cells are unaffected. | Write positions relative to a declared local origin, or write column and row and let the caller apply the affine in float64 (as the published region measurements do for centroids). |
| terrain-hydrology-04 | inconsistent | CONFIRMED | `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:227`, `:253-257`, `:304`; `gpu-terrain/terrain-processing/terrain-tiled-resident.ts` reach seeds; against `gpu-terrain/hydrology/terrain-flow-tracing.ts:83-85`, `gpu-terrain-stream-order.ts:221-223`, `gpu-terrain-stream-network.ts:150` | Two meanings of "stream cell": `streams == 1` in catchment summary and tiled flow, `streams != 0` in height above drainage, stream order, stream network and flow distance. | One convention. | A mask holding Strahler orders 1 to 3 is a valid stream mask for height above drainage ("non-zero for stream cells"). In the catchment summary only order-1 cells count toward stream length and order 2 and 3 cells get profile status 2. | Pick `!= 0 and not the unresolved word` everywhere, or reject values above 1 with a status. |
| terrain-hydrology-05 | doc-mismatch | CONFIRMED | `gpu-terrain/hydrology/gpu-terrain-watersheds.ts:128-129`, `gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts:109-110`, `docs/api/gpu-terrain/operators.md:1705-1706`; decoder at `gpu-terrain/hydrology/terrain-flow-tracing.ts:44-48` | The decoder accepts only ESRI codes (1 east, clockwise powers of two). | The documentation says grids "imported from TauDEM or Whitebox work as well". TauDEM writes 1 to 8 counter-clockwise from east. Whitebox by default writes powers of two clockwise from north-east. | Whitebox code 1 (north-east) is read as east: every direction is rotated 45 degrees with no error. TauDEM codes 3, 5, 6, 7 are read as terminal, 2 as south-east, 4 as south, 8 as south-west. | Correct the three sentences: ESRI codes only, Whitebox needs `--esri_pntr`, TauDEM needs recoding. Better: add a compile-time `directionEncoding` prop to the shared decoder. |
| terrain-hydrology-06 | doc-mismatch | CONFIRMED | `gpu-terrain/hydrology/gpu-terrain-flow.ts:384-386`, `gpu-terrain/hydrology/terrain-flow-spill-fill.ts:45-49` | Both fill paths compute the largest solution of `W = max(Z, min W(n) + epsilon)`: the relaxation starts at +infinity and only lowers, the spill graph computes the minimax directly. | The comment says the fixpoint is unique. It is not when epsilon is 0. | 5 x 5 grid, border 10, interior 0: interior 10, interior 5 and interior 0 all satisfy the equation. Only 10 is the fill. | Say "the greatest fixpoint, which is the minimax elevation". No code change. It matters to anyone who restarts a relaxation from stale values. |
| terrain-hydrology-07 | doc-mismatch | SUSPECTED | `gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts:66`, `:200` | Correlation `exp(-d / correlationLength)`: an e-folding length. | The prop text calls it the "exponential covariance range". In xDEM (scikit-gstat) the range of an exponential model is the effective range, three times the e-folding length. | A caller who passes an xDEM range of 300 m gets correlation 0.72 at 100 m where xDEM means 0.37, so the volume uncertainty is overstated. | Say "e-folding length (one third of the effective range that xDEM reports)". |

### terrain-hydrology-01: no-data cells block branching transport

`GPUTerrainFlowFractions` writes NaN into all eight fractions of a cell that is not valid
(`valid` is false when the surface is not finite). `GPUTerrainBranchingFlow` in `transport` mode
pulls from donors: for each neighbour it reads that neighbour's fraction toward the cell, and
`if (!finite(fraction) || fraction < 0.0) { ready = false; continue; }`. So every valid cell that
touches a no-data cell is never finalized, and neither is anything downstream of it. The other
four modes read the cell's own fractions and are not affected. `GPUTerrainTiledFlow` with
`routing: 'branching'` uses the same transport and inherits the defect. In D8 mode it does not:
the resident exchange writes a zero row for an invalid code.

How confirmed: `branching-nodata.mjs` replicates the initialize and `tryFinalize` logic line for
line on the 1 x 3 input. The existing specs never put a no-data cell next to a valid one in
transport mode: `terrain-processing.spec.ts` tests NaN rows only for unresolved directions and
for halo cells that are not owned.

The documentation says "cycles, invalid dependencies and insufficient budgets remain unresolved".
That is right for an unresolved direction, whose receiver is unknown. A cell outside the DEM is
not a dependency. The recommended pipeline in `docs/api/gpu-terrain/terrain-processing.md`
("materialize `GPUTerrainFlowFractions` from its filled surface") fails on any DEM with a
coastline or a mask.

### terrain-hydrology-02: serial float32 sums per basin

The statistics kernel runs one invocation per basin and adds elevations in a plain float32
accumulator. Once the sum passes 2^24 times the unit in the last place of an addend, each addition
rounds. For equal or near-equal addends the rounding has the same sign every time, so the error
grows linearly, not as a random walk. `catchment-sum.mjs` replays the arithmetic with
`Math.fround`. In `uniform` mode the area is `f32(count) * cellArea` and is exact; the mean is
affected in every mode, and the area, drainage density and hypsometry under `web-mercator` and
`geographic`.

Three siblings in the same directory sum three ways: this serial sum, the integer-significand
compensation in `gpu-terrain-depression-hierarchy.ts:277-300`, and the `GPUReduction` tree in
`gpu-terrain-dem-coregistration.ts`. The pinned reference (pysheds membership plus NumPy) uses
basins far too small to show it.

### terrain-hydrology-05: direction encodings

This is labelled `doc-mismatch` because the code does what the prop text says ("ESRI D8 codes").
The effect on a user is larger than the label: a Whitebox pointer grid has only valid single-bit
codes, so nothing is rejected and every watershed, drainage height and stream order is computed on
directions rotated by 45 degrees. `d8-codes-and-fixpoint.mjs` prints the full table.

## 3. Verified correct

- `GPUTerrainFlow` direction codes: ESRI powers of two, east first, clockwise, rows advancing
  south. Checked the bit-packed offset tables (`0x901a`, `0x1a9`) by hand against the three array
  copies. They agree.
- `GPUTerrainFlow` D8 direction: steepest drop over ground distance, diagonal
  `sqrt(gx^2 + gy^2)` at the midpoint row, strict drops only, ties to the lowest direction index,
  border and next-to-no-data cells without a lower neighbour are outlets. Compared with
  O'Callaghan and Mark and with the pysheds convention recorded in
  `test/oracles/terrain-hydrology.py`. `cellSizeMode` is honoured.
- Depression fill, spill graph path (epsilon 0): tile fill, labels, implicit spill edges with
  weight `max(L(a), L(b))`, Boruvka with a strict `(key, slot)` order, rooting by Euler tour,
  path maximum, `F = max(L, G(label))`. I rechecked the upper and lower bound proof, the hook rule
  (no cycles, ocean never hooks), the round bound `floor(log2 n) + 2`, and the tour cut at the
  ocean. It is the minimax fill of Barnes 2016, equal to Priority-Flood.
- Depression fill, relaxation path (epsilon above 0): candidate `max(Z(c), W(n) + epsilon)` from
  +infinity with boundary cells fixed. This is Planchon and Darboux. The engine was not read.
- Flat resolution: low edges, high edges, hop counts, component maximum, mask
  `2 * towardLower + (flatMaximum - awayFromHigher)`, route to the equal-height neighbour with the
  strictly smallest mask. Matches Barnes, Lehman and Mulla 2014. I checked that every drainable
  flat cell has a strictly smaller neighbour, so the receiver graph stays a forest.
- D-infinity: eight facets, `s1 = (e0 - e1) / d1`, `s2 = (e1 - e2) / d2`, angle clamped to
  `[0, atan(d2 / d1)]`, diagonal share `r / atan(d2 / d1)`. Matches Tarboton 1997. A cell has a
  facet with positive slope exactly when it has a lower neighbour, so the D8 fallback is used only
  where intended.
- Multiple flow direction: Freeman weights `tan(beta)^p` with `p = 1.1`; Quinn weights
  `(tan(beta) * L)^p` with `L` half the cell side for cardinals and a quarter of the diagonal
  (0.354 cell) for diagonals, `p = 1`. Dividing by the largest score before `pow` does not change
  the fractions.
- Accumulation, Euler tour path: successor rules, Wyllie ranking, subtree size
  `(rank(enter) - rank(exit) + 1) / 2`, round count `ceil(log2(2n)) + 1`. Counts the cell itself.
  Exact integers up to the binding limit.
- Accumulation, pull path: a cell is written once from final donors in fixed direction order.
  Counts the cell itself. `'area'` multiplies by ground area at the row centre.
- Stream mask: accumulation at or above the threshold, unresolved cells never read as 0.
- `GPUTerrainStreamOrder`: I proved that "order at least k + 1 exactly when the subtree holds a
  cell with two children of order at least k" is equivalent to the Strahler recurrence, checked
  the prefix-difference window `[enter, exit)`, the level bound `floor(log2 n)`, and the pull
  fallback rule (`m + 1` only with two donors of the maximum order).
- `GPUTerrainWatersheds`, `GPUTerrainHeightAboveDrainage`: in-place pointer jumping reaches the
  first stop cell; pour point ties go to the lowest index; height is one subtraction (Renno 2008).
- `GPUTerrainHydrologicIndices`: `a = A / b`, wetness `ln(a / tan beta)`, stream power
  `a * tan beta`, slope floor 0.001, zero area gives minus infinity. Matches Beven and Kirkby 1979
  and Moore 1991. The slope is the D8 descent slope, which the class states.
- `GPUTerrainFlowDistance`, `GPUTerrainStreamNetwork`: pointer doubling with summed distance and
  cell counts; `ceil(log2(L))` rounds resolve a path of `L` cells; the ping-pong buffer read after
  an early stop holds the result because a converged round copies. Reach rules (start where the
  donor count is not 1, continue while the receiver has one donor) checked by hand on a
  confluence.
- `GPUTerrainBreach`: layered dynamic programme, cost is the sum of cuts
  `max(z - (base - s * decrement), 0)`. I checked that no interior cell of the chosen path can lie
  below its level, and that a float32 decrement too small to separate two levels is caught by the
  strict-descent check and reported as status 2.
- `GPUTerrainBranchingFlow` recurrences for `expected-distance`, `minimum-distance`,
  `maximum-distance`, `fractional-catchment`, and `transport` on grids without no-data.
- `GPUTerrainTiledFlow`: owned-cell exchange, reach seeds (new id at headwaters, confluences and
  below a branching donor), pointer jumping.
- `GPUTerrainDepressionHierarchy`: sublevel merge tree by activation order `(elevation, index)`;
  stage curves; the integer-significand error term of `positiveAdditionError` reproduces the exact
  rounding error of a positive addition. Flats give zero-persistence nodes (3 nodes for one
  U-shaped flat pit; 327 of 870 nodes on a 64 x 64 integer DEM, `hierarchy-flat.mjs`). The user
  documentation states this, so it is a convention, but those nodes count against `nodeCapacity`.
- `GPUTerrainDEMCoregistration`: median of `moving - reference`, even-count median, NMAD factor
  1.4826022, standard error `sqrt(pi / 2) * NMAD * sqrt(sum of correlations) / n`, volume variance
  with the pairwise covariance and the common bias term. Matches the xDEM vertical shift.
- `GPUTerrainFlowField`: `v = w - (w . g) g / (1 + |g|^2)` is the horizontal part of the
  projection onto the tangent plane (derived by hand). Central differences over ground distances,
  first-order one-sided at the border.
- `GPUGeomorphons`: the 9 x 9 table equals the GRASS `r.geomorphon` table entry for entry
  (minus count by row, plus count by column); direction order NE, N, NW, W, SW, S, SE, E; zenith
  and nadir by cross-multiplied slopes with the nearest sample kept on ties; `anglev1`, `anglev2`
  and `anglev2-distance` reductions; the flat distance rule `|h| > tan(t) * flatDistance`; strict
  search limit; rotation and mirror minimum for the 498 classes. One remark: the search limit is
  multiplied by 0.99999 (`gpu-geomorphons.ts:423`), so a sample whose distance lies within 1e-5 of
  the limit is dropped where GRASS keeps it. On square cells the first case is the diagonal sample
  408 at `searchRadius` 577. Not worth a fix.
- `GPUTerrainCriticalPoints`: ring orders are cyclic for both the 8-ring and the Freudenthal
  6-ring; sign-change counting with index tie-break.
- `GPUTerrainPeakProminence`: edge saddle `min(z(a), z(b))`, descending stable sort, elder rule
  with ties to the lowest index, non-peaks get 0, component elders get +infinity. The join-tree
  builder itself was not read.
- `GPUTerrainSummits`: disc predicate, ring as "disc pixel with a neighbour outside", the
  interior half-width rule `min(e(dy - 1), e(dy), e(dy + 1)) - 1`, strict maximum under
  `(height, -index)`, drop as a lower bound of prominence.
- `GPUTerrainPeakSnap`: scan box covers the disc for any sub-pixel candidate offset, outcome
  precedence as documented, bilinear reference height.
- `GPUProfilePeaks`: parabola vertex `di = 0.5 (a - c) / (a - 2v + c)` and value
  `v - 0.25 (a - c) di` derived by hand; windowed prominence; the suppression rounds decide each
  candidate in agreement with greedy suppression.

No-data: every class that takes a `GPURasterBand` canonicalizes it through the published
`GPURasterNeighborhood` identity kernel, which honours validity, `noDataValue`, scale and offset
and writes a quiet NaN for invalid pixels. So "NaN means invalid" inside those kernels is the
band validity, not only NaN in the source.

## 4. Standards consistency

- **No-data input form.** `GPUTerrainFlow`, the indices, height above drainage, flow field,
  geomorphons and all terrain features take a `GPURasterBand`. The seven terrain-processing
  classes take a bare `float32` view plus an optional `validity` view (for example
  `gpu-terrain-breach.ts:44-49`). The earlier review reported this; it is still present in part.
  The validity view fixed the case where the caller passes one. A DEM with `noDataValue` -9999
  and no validity view is still read as terrain. Contract section 6 asks for the band.
- **Georeference.** The hydrology classes read cell size and row edges from the parameters view
  (`getRasterGridWGSL`). Terrain-processing, flow distance and stream network bake them into the
  shader (`terrain-processing-utils.ts:83-97`, `gpu-terrain-stream-network.ts:348`). Both use the
  same midpoint-row rule and the same WGS84 scale, so the numbers agree. Contract section 6 asks
  for parameters.
- **`cellSizeMode`.** Honoured wherever a slope, distance or area is formed. The nine classes the
  earlier review listed now accept it. The default stays `'uniform'` and nothing compares it with
  the grid's coordinate reference system, so a geographic DEM with the default gives slopes in
  metres per degree without an error.
- **Row scale.** Flow kernels use the midpoint row of each move. Summits, peak snap and
  geomorphons use the centre row for the whole disc or ray. Each states it.
- **Summation.** Three methods in `terrain-processing` (finding 02).
- **Stream mask.** Two truth tests (finding 04). Also two unresolved words for streams: the
  producer writes `0xffffffff`; `gpu-terrain-flow-distance.ts:162` and
  `gpu-terrain-stream-network.ts:185` also treat `0xfffffffe` as unresolved.
- **Private copies, all numerically identical to each other:**
  - D8 neighbour offsets, four times: `raster-grid-utils.ts:163-168` (bit-packed),
    `terrain-flow-tracing.ts:46-47`, `terrain-flow-euler-tour.ts:62-63`,
    `terrain-flow-spill-utils.ts:178-179`.
  - Order-preserving float key, four times: `getSpillKey` (`terrain-flow-spill-utils.ts:192-195`),
    and inline in `gpu-terrain-breach.ts:239`, `gpu-terrain-depression-hierarchy.ts:163-164`,
    `gpu-terrain-dem-coregistration.ts:282-283`. Only two of them normalize `-0`.
  - Finite test and NaN, five spellings: `isFiniteValue` (shared), `finite` and `missing`
    (`terrain-processing-utils.ts:61-63`), `isSpillFinite`, `getNan(seed)`
    (`terrain-features-utils.ts`), and inline bit masks in `terrain-flow-passes.ts:300`, `:492`.
    Same bit test in each.
  - The fill-initialization kernel body exists twice (`terrain-flow-passes.ts:103-120`,
    `terrain-flow-spill-fill.ts:309-330`).
- **Pointer jumping.** Three shapes for one technique: in-place atomic
  (`terrain-flow-tracing.ts`), gated ping-pong (`createRasterJacobiLoopNodes` in flow distance and
  stream network), ungated unrolled ping-pong (`gpu-terrain-catchment-summary.ts:266-288`,
  `terrain-tiled-resident.ts`). All are correct. The round counts are computed three ways
  (`getTerrainJumpRoundCount`, `getSpillRoundCount`, inline `Math.ceil(Math.log2(count))`).
- **Earth model.** One: WGS84 radii of curvature from `analysis-core/raster-grid.ts`. No second
  radius in the scope.

## 5. luma.gl official implementations

- **`GPURasterRegionMeasurements` (entry `gpu-raster`).** It publishes per-region pixel count,
  float32 sum, minimum, maximum, mean and area (count times the affine determinant).
  `GPUTerrainCatchmentSummary` re-implements area, mean, minimum and maximum per basin. The two
  agree on the definitions and on the area unit (squared coordinate units). Neither is exact for
  large regions: the official one uses atomic float32 sums and says they are order-dependent, the
  local one is a fixed-order serial float32 sum (finding 02). The local class adds stream length,
  density, hypsometry and `cellSizeMode`, which the official one lacks. Recommendation: fix the
  local sum with the shared accumulator and propose a deterministic sum upstream; do not delegate
  as is.
- **`GPUFiniteDifference2D` (entry `gpu-core`).** `GPUTerrainFlowField` computes its own
  gradient. Interior stencils agree (centred differences). At the border the official class uses
  second-order one-sided differences and the local kernel first-order. The local kernel also
  needs per-row spacing and no-data, which the official class does not offer (one constant
  spacing pair, no validity). The border difference is a convention, not an error. Candidate for
  an upstream proposal (per-row spacing and a validity input), not for delegation today.
- **`GPURasterGradient`, `GPURasterSobel` (entry `gpu-raster`).** Not used in the scope and not
  re-implemented. The hydrologic indices use the D8 descent slope on purpose.
- **`GPURasterConnectedComponents` (entry `gpu-raster`).** It labels foreground of a binary mask.
  The flat-resolution component maximum needs components of equal elevation, which that class
  cannot express. Not a re-implementation.
- **`GPURasterMorphology` (entry `gpu-raster`).** Erosion and dilation only. Depression filling is
  reconstruction by erosion, which is not published. Not a re-implementation.
- **`GPUSort`, `GPUScan`, `GPUCompaction`, `GPUReduction` (entry `gpu-core`).** Delegated to, with
  arguments that keep the meaning: stream order (exclusive scan), stream network (compaction,
  inclusive scan, sum), coregistration (stable sort for the median, sums), prominence (stable
  descending radix sort; I checked that descending inverts the digit and keeps equal keys in
  input order), depression hierarchy and catchment summary (stable sort).
- **No official list ranking, Euler tour, pointer jumping or minimum spanning tree** exists in
  `gpu-core` or `gpu-graph-algorithms`. The local ones are not copies.
- **WGSL conventions.** The official neighbourhood kernel names its test `isFiniteValue` and
  writes `bitcast<f32>(0x7fc00000u | (i & 0u))` for invalid pixels. The scope uses that spelling
  in places and four others elsewhere (section 4). Row and gradient orientation agree with the
  official "positive y increases downward".

## 6. Test oracles

Pinned by an independent oracle:

- D8 directions and cell accumulation: pysheds, `test/oracles/terrain-hydrology.py`. The GPU is
  compared with a CPU model and the CPU model with pysheds (`terrain-flow-reference.node.spec.ts`).
- Depression fill: pysheds `fill_depressions` (`terrain-fill-depressions.py`) and a literal
  Priority-Flood.
- Height above drainage, wetness and stream power, Strahler order: pysheds, NumPy and a NetworkX
  recurrence (`parity-wp5-hydrology.py`). The generator records that pysheds 0.5 stream order is
  wrong for donor orders `[2, 2, 1]` and does not use it there.
- Watersheds: pysheds (`parity-wp5-watersheds.parity.spec.ts`).
- D-infinity: one analytic plane (slope `(2, 1)`, share `atan(1/2) / (pi/4)`).
- Branching transport on supplied fractions, tiled flow: pysheds `routing='mfd'`
  (`terrain-upstream.py`); the branching modes on a diamond: NetworkX (`terrain-processing.py`).
- DEM coregistration: xDEM 0.1.6 and NumPy.
- Depression hierarchy: SciPy sublevel components, area and volume.
- Breach: WhiteboxTools 2.4.0, corridor only (the elevations differ by design).
- Geomorphons: GRASS 8.5, three comparison modes, with exact-ratio tie certificates.
- Peak prominence: an independent vertex-activation union-find in NumPy.
- Critical points, summits, peak snap, profile peaks: SciPy and composite references
  (`parity-wp5-*.parity.spec.ts`).
- Flow distance: NetworkX path lengths on a 3 x 2 grid.

Pinned only by a copy of the same formula:

- Freeman and Quinn fractions: `terrain-flow-routing-oracle.ts` repeats the kernel formula and
  calls the same distance helper. Property tests (large exponent approaches D8, mass conservation)
  exist. No external value pins the weights. I checked them against the papers by hand.
- Flat resolution: `terrain-flow-flats-oracle.ts` is a sequential port of the same paper written
  for the test. No RichDEM fixture.
- D-infinity away from the analytic plane, and all routings under `web-mercator` and
  `geographic`: CPU copy.
- Flow distance and stream network at scale and under `cellSizeMode`: a serial CPU walk.
- Flow field: `terrain-flow-field-oracle.ts`.
- `cellSizeMode` in catchment summary, coregistration volume and depression hierarchy: values
  computed in the spec from the same scale formula.

Pinned by nothing:

- Branching transport next to a no-data cell (finding 01).
- Catchment statistics on a basin large enough for float32 to matter (finding 02).
- A direction grid in a non-ESRI encoding (finding 05).
- Stream network vertex positions at large coordinates. One spec checks that reach length is
  independent of a large translation; none checks the positions.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. Specs were run on the headless WebGPU device of this
machine (`--project headless`) or in Node (`--project node`). Findings 01 and 02 were first
reproduced on the GPU with the report's inputs: branching status `[0, 2, 2]` on the 1 x 3 DEM, and
a mean of 2015.5799560546875 for a million cells at a constant 2000.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| terrain-hydrology-01 | fixed | `src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts:160-164`: a cell without data gets a zero fraction row (it donates nothing); only a valid cell with an unresolved direction keeps a NaN row. `terrain-tiled-resident.ts:59`: in branching mode a `flowDirections` word of `GPU_TERRAIN_FLOW_NONE` makes the cell invalid, so the no-data cell itself still reports status 0 and NaN (note 1). `GPUTerrainBranchingFlow` is unchanged. | `test/gpu-terrain/terrain-processing/terrain-branching-nodata.spec.ts`: the 1 x 3 DEM `[NaN, 2, 1]` gives `[NaN, 1, 2]`, and a 5 x 5 plane with an interior hole gives a hand-computed accumulation, through `GPUTerrainFlow`, through fractions and `GPUTerrainTiledFlow` with `routing: 'branching'`, and through `GPUTerrainBranchingFlow` with a validity view. Both fail on the old publisher. | `GPUTerrainFlowFractions`: no-data rows are 0 instead of NaN. Cells beside and downstream of no-data were status 2 with NaN and are now resolved. A direct `GPUTerrainBranchingFlow` caller who relied on the NaN row to mark the no-data cell invalid must pass `validity` or NaN `values` (otherwise the no-data cell is a valid outlet holding its own injection). `GPUTerrainTiledFlow` in branching mode reads `0xffffffff` from `flowDirections`; zero placeholders behave as before. |
| terrain-hydrology-02 | fixed | `src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:263-295` (statistics) and `:296-305` (hypsometry): the elevation sum, the ground area and hypsometry areas under `cellSizeMode`, and the stream length are fixed-point sums (`FIXED_POINT_WGSL` of `analysis-core/kernels/fixed-point-accumulator.ts`) in a unit set by the basin's largest term; two walks over the basin's cells instead of one. | `test/gpu-terrain/terrain-processing/terrain-basin-sums.spec.ts`: a million cells at a constant 2000 give mean exactly 2000; a million-cell geographic basin (1500 to 2500 m, cell area varying 3 % over the rows) matches float64 for mean (5e-4 m), area, stream length, drainage density and four hypsometry bins (1e-5 to 2e-5 relative). Both fail on the old kernel (2015.58; area 0.56 % low). | Mean, and area, drainage density, hypsometry and stream length for large basins, change to the correct values. Small basins change by at most a float32 rounding. |
| terrain-hydrology-02b (extra, brief section on compensated sums) | fixed | `src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:177-194`, `:325-402`. The integer-significand compensation did survive the shader compiler, but its correction was accumulated in one float32 word that itself rounds: stage area of one 262,144-cell node with cell area 0.7 was 3.9e-6 off (46 float32 spacings; the plain sum is 2.6e-3 off), and the error grows with the square of the cell count. Replaced by fixed-point sums; `positiveAdditionError` is gone. The low words live in `output.curves` (bound as `u32` inside the kernel until each value is decoded), the middle and high words in two scratch views of the curves' size, so no scratch binding is larger than the caller's curves and the kernel stays at 8 storage bindings. | `terrain-basin-sums.spec.ts` ("stage area and volume of a 262,144-cell depression match float64", 1e-6 relative; failed before). The existing hierarchy specs (nested sinks, SciPy sublevel areas and volumes) pass unchanged. | Stage areas and volumes of large nodes change in the sixth digit. Scratch: two `uint32` views of the curves' size instead of one `float32` view (transient id `<id>-curve-compensation` is now `<id>-curve-sums-middle` and `-high`). |
| terrain-hydrology-03 | fixed | `src/gpu-terrain/hydrology/gpu-terrain-stream-network.ts:67-77` new optional compile-time `localOrigin` (the name and meaning the isoline, sampling and distance-field parameters use), `:126` read-only `localOrigin` field, `:393-397` the translation `affine - localOrigin` is formed in float64. Default `[0, 0]` keeps the absolute output; its precision is stated at `:92-96` and in `docs/api/gpu-terrain/terrain-processing.md`. The affine is still baked into the shader (note 2). | `test/gpu-terrain/hydrology/terrain-direction-encoding.spec.ts` ("positions relative to the grid origin are exact where absolute float32 ones are 0.2 cell off"): 1,300 vertices on 1 m cells at 4,500,000.3 / 5,200,000.7; with the grid origin as `localOrigin` the float64 sum is the exact cell centre (1e-6), the default is 0.2 cell off as the report says. | None by default. With `localOrigin` the positions are origin-relative and exact. |
| terrain-hydrology-04 | fixed | Converged on nonzero (contract section 7 mask rule, and what height above drainage, stream order, stream network and flow distance already did): a stream cell is a nonzero word other than the unresolved sentinels `0xffffffff` and `0xfffffffe`. `gpu-terrain-catchment-summary.ts:186-193`, `:315`, `:318`, `:366-368`; `terrain-tiled-resident.ts:35`, `:58`, `:90`, `:94`. Wording: `gpu-terrain-flow.ts:291`, `docs/api/gpu-terrain/operators.md`, `terrain-processing.md`. | `terrain-direction-encoding.spec.ts` ("a mask of Strahler orders is a stream mask for the catchment summary and tiled flow"; "the unresolved stream words are not stream cells and stay unresolved"). | A stream word above 1 (for example a Strahler order): the catchment summary used to skip its edges and give profile status 2, tiled flow used to mark the cell invalid (status 0); both now treat it as a stream cell. `0xfffffffe` is now unresolved in both, as `0xffffffff` already was. |
| terrain-hydrology-05 | fixed (code and documentation) | `src/gpu-terrain/hydrology/terrain-flow-tracing.ts:26-42` exported type `GPUTerrainFlowDirectionEncoding` (`'esri'` default, `'whitebox'`, `'taudem'`), `:51` validator, `:68-87` the one decoder `getFlowDirectionCode` (WGSL), `:104` `getTerrainFlowTracingWGSL(width, height, directionEncoding)`. New optional compile-time prop `directionEncoding` on `GPUTerrainWatersheds`, `GPUTerrainHeightAboveDrainage`, `GPUTerrainStreamOrder`, `GPUTerrainFlowDistance`, `GPUTerrainStreamNetwork`, `GPUTerrainCatchmentSummary` and `GPUTerrainTiledFlow`; every kernel that reads a caller's direction word passes it through the decoder. The three sentences that claimed TauDEM and Whitebox grids work are replaced (`gpu-terrain-watersheds.ts`, `gpu-terrain-height-above-drainage.ts`, `docs/api/gpu-terrain/operators.md`). | `terrain-direction-encoding.spec.ts`: one 4 x 4 direction grid using all eight directions, written by hand in the three numberings, gives the same hand-traced basins, drainage cells, Strahler orders, terminal cells, accumulation, stream length and reach count in all seven contributors; a Whitebox grid read with the default is not the right answer; zero and the sentinels mean the same in each; an unknown name throws. | None by default (the ESRI decoder is the identity). A caller with a Whitebox or TauDEM grid now has a prop that reads it correctly. |
| terrain-hydrology-06 | fixed (documentation) | `src/gpu-terrain/hydrology/gpu-terrain-flow.ts:388-395`, `terrain-flow-spill-fill.ts:45-53`, `docs/api/gpu-terrain/operators.md` (`GPUTerrainFlow` paragraph): the fill is the greatest fixpoint (the minimax elevation); with epsilon 0 the equation has lower solutions. | none (words) | none |
| terrain-hydrology-07 | fixed (documentation) | `src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts:66-76` (the prop text), `docs/api/gpu-terrain/terrain-processing.md` (DEM comparison): `correlationLength` is the e-folding length `L` of `exp(-d / L)`; the practical range is `3 L`. The formula is unchanged. The factor of three is arithmetic (`exp(-3)` is 0.05); that scikit-gstat and xDEM report the effective range could not be checked here (neither is importable), so the text tells the caller to check the tool's convention. | none (words) | none |
| rowDirection (lead's instruction, roadmap 12.3) | fixed | `src/gpu-terrain/geomorphons/gpu-geomorphons.ts:245`: `GPUGeomorphons` resolves `rowDirection` with `resolveTerrainRowDirection`, the helper of the terrain-surface stream, and has a read-only `rowDirection` field. It is the only class of this scope that takes the prop. | `test/gpu-terrain/geomorphons/geomorphons.node.spec.ts` (default from the affine, both contradictions throw with the contributor ID). The GRASS parity specs pass unchanged (they already build a rows-north grid for their north case). | The default follows `grid.affine[4]`: on a rows-north grid with the prop left out, `pattern` used to be mirrored north to south and is now right (`forms` and `ternary` are mirror-invariant and do not change). An explicit `rowDirection` that contradicts the grid throws. |

### Notes

1. Choice for 01. The lead offered two cures. I took the publisher one, because "a row is NaN" then
   means one thing (the receiver is unknown) and the consumer needs no second input to tell
   no-data from unresolved. Its cost is that a zero row no longer marks the no-data cell itself
   invalid. `GPUTerrainTiledFlow` gets that back from `flowDirections` (which the documented
   pipeline already passes), so its output equals `GPUTerrainFlow` cell for cell including the
   NaN. A direct `GPUTerrainBranchingFlow` caller gets it from `validity` or NaN `values`, which
   the prop documentation now says.
2. Georeference from parameters (contract section 6). The stream network, flow distance and the
   terrain-processing classes still bake the affine into the shader. `localOrigin` removes the
   precision loss without changing that. Moving the affine to a parameters view changes the props
   of seven classes and is not in this pass.
3. Existing tests changed because they encoded old behaviour: one line of
   `test/analysis-core/kernel-audit/cases-terrain.ts` (the `GPUGeomorphons` cases pass
   `rowDirection: 'north'`, so their grid now states north through `withRowDirection`, as the
   terrain-surface stream did for its classes).
4. Not touched, as instructed: `raster-analysis/cost-distance/raster-relaxation.ts` and
   `analysis-core/kernels/parallel-merge-tree.ts`. I did not read them beyond what the report
   lists and found nothing to report in them.
5. Generated lens records (`docs/lens-records/gpu-terrain.terrain-depression-hierarchy.json`,
   `website/src/lens/generated/gpu-terrain.terrain-depression-hierarchy.json`) still name the old
   `-curve-compensation` scratch. They are recordings, outside this pass, and are stale until
   re-recorded.
6. Remaining serial float32 sums in the scope, not in the report and left alone: the reach length
   in `gpu-terrain-stream-network.ts:398-404` (one thread adds the edges of one reach, relative
   error about `n * 6e-8` for a reach of `n` cells), and the profile distances of the catchment
   summary and of `GPUTerrainFlowDistance`, which are pointer-jumping sums in tree order.

### Upstream gaps

- `GPURasterRegionMeasurements` sums float32 values per region with atomic float adds, which is
  order-dependent and loses accuracy for large regions. With a deterministic exact sum (fixed
  point, or a sorted-segment tree) and a per-row cell area, `GPUTerrainCatchmentSummary` could
  delegate its area, mean and extrema.
- `GPUFiniteDifference2D` has one constant spacing pair and no validity input, so
  `GPUTerrainFlowField` cannot delegate its gradient. It would need per-row spacing and a no-data
  rule.
- `gpu-core` has no fixed-point or otherwise exact reduction: `GPUReduction` sums float32 in tree
  order. The contributors here that need an exact sum inside one invocation carry the shared
  `analysis-core` accumulator instead.

### Review follow-up

After the gates and the independent review (`fix-review-fields.md`, row fr-fields-08). The
`file:line` references in the table above were taken before these edits; three of the files grew,
so the current lines are: `gpu-terrain-flow-fractions.ts:163-167` (finding 01),
`terrain-flow-tracing.ts:26-51` (the type), `:60` (validator), `:77-96` (decoder), `:113`
(finding 05), `gpu-terrain-flow.ts:298` (finding 04 wording) and `:399-406` (finding 06).

| Item | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| Test typecheck (two TS2345 in `test/gpu-terrain/hydrology/terrain-direction-encoding.spec.ts`) | fixed | `:78-81`: the narrowed device is captured in a constant, the pattern of `test/geospatial-expansion/raster-terrain-expansion.spec.ts`, because the two helpers are hoisted function declarations. No assertion changed. | The spec itself (11 tests pass). | none |
| fr-fields-08 (direction codes on a rows-north grid) | fixed (contract decided and documented; no kernel change) | The contract is raster-relative: a code names a row and column offset ("south" is `row + 1`), whatever `grid.affine[4]` says. Checked on the GPU: `GPUTerrainFlow` writes the same words (`[2, 4, 1, 0]` for the 2 x 2 DEM `[5, 4, 3, 1]`) on a rows-south and on a rows-north grid, so the producer is already raster-relative and the consumers agree with it. Stated prominently in `src/gpu-terrain/hydrology/terrain-flow-tracing.ts:29-37` (the `GPUTerrainFlowDirectionEncoding` documentation, which also says what to do with a tool's grid held with flipped rows), `gpu-terrain-flow.ts:276-280` (`output.flowDirections`), `gpu-terrain-watersheds.ts`, `gpu-terrain-height-above-drainage.ts`, `gpu-terrain-stream-order.ts:113`, and in bold in `docs/api/gpu-terrain/operators.md` (the `directionEncoding` paragraph). See note 7 for why not compass. | `terrain-direction-encoding.spec.ts:323` (the three numberings give the same hand-traced basins on a rows-south and a rows-north grid) and `:350` (`GPUTerrainFlow` writes the same codes on both grids and `GPUTerrainWatersheds` follows them to the same outlet). The base grid of the spec now states rows south, where the compass names of its direction table are true. | none (words). A caller who holds a Whitebox, TauDEM or ArcGIS direction grid with rows flipped to advance north must flip it back or exchange the north and south codes; that was already so and is now documented. |
| Sheared grids in the users of the diagonal D8 step | fixed | `validateRasterGridOrthogonal` (`raster-analysis/raster-band-utils.ts:89`, the validator of `GPUCostDistance`) is called in the constructors of the three classes of this scope whose kernels use the diagonal of `getD8Distance`: `src/gpu-terrain/hydrology/gpu-terrain-flow.ts:482`, `gpu-terrain-hydrologic-indices.ts:167`, `src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts:76`. It runs after the axis-aligned check, so the existing messages for `'geographic'` and `'web-mercator'` are unchanged. Stated in the `GPUTerrainFlowProps` documentation and in `docs/api/gpu-terrain/operators.md`. | `test/gpu-terrain/hydrology/gpu-terrain-flow.node.spec.ts:161-167`, `gpu-terrain-hydrologic-indices.node.spec.ts:85-89`, `test/gpu-terrain/terrain-processing/terrain-processing.node.spec.ts:439`: a sheared grid throws with the contributor ID, a rotated grid and a rows-north grid pass. | `GPUTerrainFlow`, `GPUTerrainHydrologicIndices` and `GPUTerrainFlowFractions` throw at construction on a sheared grid in `'uniform'` mode (`<id> grid.affine must have orthogonal pixel axes ...`), where they used to measure both diagonals as `sqrt(x^2 + y^2)`. Rotated grids are unaffected. |

7. Why raster-relative. The brief lists "making isolines, isobands or the flow family follow the
   grid affine for row direction" under "Not in this pass" and asks for the current rule to be
   documented instead. The compass contract would also change what `GPUTerrainFlow` writes on a
   rows-north grid and every kernel that turns a code into a row offset (direction, flats, Euler
   tour, routing, fractions, the tiled exchange), for no change on the north-up rasters the three
   tools write. A throw for `'whitebox'` or `'taudem'` on a rows-north grid (the reviewer's
   other suggestion) was not added: under the raster-relative contract such a grid is legal when
   the caller has exchanged the north and south codes, and the class cannot tell.
8. `GPUTerrainFlowField` also uses `getD8Distance`, but only for the row step (no diagonal), so
   it does not get the validator. Its gradient does treat the column and row differences as
   orthogonal components, so on a sheared `'uniform'` grid the deflected wind is wrong in the same
   way. Not changed; a candidate for the same validator.
9. The four other classes of the scope that measure a diagonal move (`GPUTerrainFlowDistance`,
   `GPUTerrainStreamNetwork`, `GPUTerrainCatchmentSummary`, `GPUTerrainBranchingFlow`) use
   `getTerrainGroundMeasureWGSL`, which in `'uniform'` mode takes the length of the affine applied
   to the cell offset. That is right on a sheared grid, so they keep accepting one.
