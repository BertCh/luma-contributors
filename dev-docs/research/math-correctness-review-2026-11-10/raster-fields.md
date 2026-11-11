# Math correctness review: raster fields (2026-11-10)

Scope id: `raster-fields`. Scope: under `raster-analysis/` the folders `cost-distance`, `isolines`,
`raster-sampling`, `distance-field`, `flow-texture`, `polygon-rasterization`,
`lagrangian-diagnostics`, `particle-advection`, `raster-pyramid`, `vector-field-common`,
`raster-grid`, the 17 `.ts` files at the `raster-analysis` root, and
`gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts`. Paths below are relative to
`modules/analysis/src`.

The math in this scope is in good shape. The core kernels (cost distance, walking cost, distance
transform, marching squares, sampling, polygon rasterization, fractional coverage, FTLE, focal
statistics) match their reference methods, and most are pinned by an independent oracle. I found no
result that is wrong on ordinary input. I found three boundary cases, one float32 precision loss
(already known, still present), three inconsistencies and one documentation gap.

Scratch scripts (pure Node, `Math.fround` where float32 matters), all in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/raster-fields/`:
`lic-direction.mjs`, `reprojection-precision.mjs`, `saddle-rule.mjs`, `fill-four-quadrant.mjs`,
`medial-axis.mjs`, `skeleton.mjs`, `watershed-order.mjs`, `small-checks.mjs`. I ran no spec files.
No Python reference library (NumPy, SciPy, scikit-image) is installed on this machine, so
statements about scikit-image and GDAL behaviour come from their source as I know it, except the
GDAL fill, which I checked against the committed rasterio fixture.

Working tree note: none of the files I read looked half-edited.

## 1. Coverage

Read in full (kernel bodies and packers; import blocks skipped):

- `cost-distance/gpu-cost-distance.ts`, `raster-relaxation.ts`, `raster-grid-utils.ts`,
  `gpu-cost-distance-path.ts`
- `distance-field/gpu-distance-field.ts`, `distance-field-kernels.ts`, `distance-field-parameters.ts`
- `isolines/gpu-isolines.ts`, `isolines-utils.ts`, `isolines-parameters.ts`, `isobands-wgsl.ts`,
  `isobands-parameters.ts`
- `raster-sampling/raster-sampling-utils.ts`, `raster-sampling-parameters.ts`,
  `gpu-raster-sampling.ts`
- `flow-texture/gpu-line-integral-convolution.ts`, `line-integral-convolution-parameters.ts`
- `particle-advection/gpu-particle-advection.ts`, `particle-advection-parameters.ts`
- `lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts`, `gpu-lagrangian-flow-map.ts`
- `vector-field-common/vector-field-sampling.ts`
- `polygon-rasterization/polygon-rasterization-parameters.ts`
- `raster-grid/gpu-raster-grid.ts`
- Root: `gpu-raster-walking-cost.ts`, `gpu-raster-reconstruction.ts`, `gpu-raster-skeleton.ts`,
  `raster-medial-axis.ts`, `gpu-raster-watershed-segmentation.ts`, `raster-watershed-forest.ts`,
  `gpu-raster-fill-nodata.ts`, `gpu-raster-rank-filter.ts`, `gpu-raster-reprojection.ts`,
  `gpu-raster-tile-reprojection.ts`, `gpu-raster-spectral-indices.ts`,
  `gpu-raster-fractional-coverage.ts`, `gpu-raster-fractional-zonal-distribution.ts`,
  `gpu-raster-fractional-categorical-statistics.ts`, `raster-band-utils.ts`,
  `raster-expansion-utils.ts`
- `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts`
- Supporting, outside the scope: `analysis-core/raster-grid.ts` (full); published `@luma.gl/graph`
  `gpu-raster/gpu-raster-neighborhood.js` (full), `types.d.ts` (full),
  `gpu-raster-morphology.d.ts` (full).

Read in part:

- `raster-pyramid/gpu-raster-extrema-pyramid.ts`: lines 30-597 (layout, level-0 kernel, fused
  reduction, WGSL helpers). Constructor plumbing skimmed.
- `polygon-rasterization/gpu-polygon-rasterization.ts`: lines 40-190 and 236-695 (semantics, edge,
  emit, sort keys, fill, boundary kernels). The longitude and latitude shift path was read but not
  exercised.
- `polygon-rasterization/gpu-raster-join.ts`: lines 60-130 and the lookup kernel (by search). The
  aggregate and sorted-sum plumbing was not read.
- `isolines/gpu-isobands.ts`: lines 30-170 and the kernel bodies by search. The cell geometry is in
  `isobands-wgsl.ts`, read in full.
- `isolines/gpu-isoband-rings.ts`: lines 20-75 only. It composes `GPUIsobands` and
  `GPUSegmentRingAssembly`.
- `raster-sampling/gpu-raster-profile.ts`: lines 30-210 and every arithmetic line of the kernels (by
  search). `raster-sampling/gpu-raster-profile-scan.ts`: lines 60-200.
- `flow-texture/gpu-streamlines.ts`: lines 60-180 and the trace arithmetic (by search). The pruning
  rounds and publish scans were not read. `flow-texture/streamlines-parameters.ts`: packer lines only.
- Published `gpu-raster-contours.js`, `gpu-raster-overview.js`, `gpu-raster-band-math.js`,
  `gpu-raster-morphology.js`: the policy and formula lines, by search.

Not opened: the `index.ts` files of each folder and `raster-analysis/index.ts`; all spec files
except where named in section 6; published `gpu-raster-convolution.js`,
`gpu-raster-connected-components.js`, `gpu-raster-buffer-to-texture.js`, `gpu-core` kernels.

Not present in this scope (so not reviewed): signed distance, cost allocation, divergence,
vorticity and Okubo-Weiss, RK4 advection, domain wrap, a mean or mode pyramid. `GPUDistanceField`
is unsigned; `GPUCostDistance` lists allocation as a non-goal; the pyramid is min/max only.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raster-fields-01 | wrong-edge-case | CONFIRMED | `raster-analysis/flow-texture/gpu-line-integral-convolution.ts:277`, `:302`, `:306` | Each trace step is `stepLength * (cellWidth * dx, cellHeight * dy)` with `(dx, dy)` the unit flow direction in world space. | A streamline is tangent to the field. The step must be `(dx, dy)` times one scalar length. | Uniform field `(1, 1)`, output cells `0.703 x 0.352` degrees (512 x 512 pixels over 360 x 180): the trace runs at 26.6 degrees, not 45. Cells `1 x 2`: 63.4 degrees. Square cells: exact. | Convert the direction to pixel space with the shared `RASTER_PIXEL_DIRECTION_WGSL` (`analysis-core/raster-grid.ts:280`), or step by a scalar world length as `GPUStreamlines` does (`flow-texture/gpu-streamlines.ts:430-434`). |
| raster-fields-02 | wrong-edge-case | CONFIRMED | `raster-analysis/gpu-raster-fractional-coverage.ts:407-409`, `:425`; `gpu-raster-fractional-categorical-statistics.ts:192-194`, `:300`, `:331`, `:344-346`; `gpu-raster-fractional-zonal-distribution.ts:188-190`, `:273` | The three statistics contributors bake `abs(det(grid.affine))` into the shader as a literal. `GPURasterFractionalCoverage` reads the affine per frame from `parameters` (`:59-74`, `:245`). | One cell area for one graph. | Compile with affine `[10, 0, 0, 0, -10, 0]`. Repack the coverage parameters for `[20, 0, 0, 0, -20, 0]` (same width and height, as its doc invites). A zone covering one cell: fraction 1 (right), `coveredArea` 100 (true 400), density `totals` 4 times too small. | Read the cell area from the same `parameters` view (slot 6 is the determinant), or state that the statistics contributors need a recompile when the cell size changes. |
| raster-fields-03 | wrong-edge-case | CONFIRMED | `raster-analysis/cost-distance/raster-grid-utils.ts:188-195`; `analysis-core/raster-grid.ts:85`; `raster-analysis/raster-band-utils.ts:61-69` | Diagonal step length is `sqrt(hypot(a, d)^2 + hypot(b, e)^2)`. That is the diagonal only when the two pixel axes are orthogonal. `'uniform'` mode accepts any affine. | Length of the world vector of the move, `hypot(a ± b, d ± e)`. | Affine `a = 1, b = 0.5, d = 0, e = 1`: kernel diagonal 1.5 for both diagonals; true lengths 1.803 and 1.118. A rotated (orthogonal) grid is right. | Reject `a*b + d*e != 0` in `GPUCostDistance`, or take the step length from the affine as `GPURasterWalkingCost` does (`gpu-raster-walking-cost.ts:349-358`). |
| raster-fields-04 | precision | CONFIRMED (still present, was G12) | `raster-analysis/gpu-raster-reprojection.ts:133`, `:173` | Destination centres are built in float32 from absolute affine literals, then the absolute source origin is subtracted. With a projection, the float32 `destinationOrigin` is added back before the subtraction. | Source pixel coordinate to well under a cell. | UTM, origin northing 4,600,000 m. 1 m cells, grids offset by 0.25 m: source row wrong by 0.25 cell. 0.25 m cells offset by 0.1 m: wrong by 0.9 cell. Projection path: local y 123.3, origin 4,600,000, source origin 4,600,100.75 gives 22.5, true 22.55. 10 m cells: no error. | Fold `destinationOrigin - sourceOrigin` and the affine origin difference in float64 on the CPU and pass the small difference; keep the per-pixel part relative. |
| raster-fields-05 | inconsistent | CONFIRMED | `raster-analysis/isolines/isolines-parameters.ts:19-23`, `:72-75`; `isobands-parameters.ts:25`, `:85-90`; against `raster-sampling/raster-sampling-parameters.ts:59-75`, `:94-100` | `GPUIsolines` and `GPUIsobands` take `grid: GPURasterMetadata` but read only its width and height. Row 0 is placed at `extent.minY`. The flow family (LIC, streamlines, particle advection, flow map, FTLE) uses the same "row 0 has the smallest y" rule. | The sibling contributors (sampling, profile, distance field, cost distance, polygon rasterization, reprojection, fractional coverage) take the row direction from `grid.affine`. | A north-up grid from `makeGPURasterMetadata` (the default) with its own bounds as `extent`: contours are mirrored north to south, and the documented "high side on the left" becomes the right in world space. The lenses example flips the DEM on the CPU and still passes a north-up `grid` (`modules/lenses/.../gpu-raster/isolines.example.ts:20`, `:36-51`). | Derive the extent and row direction from `grid.affine` in the two packers (signed row step, as sampling does), and reject a `grid` whose affine disagrees with an explicit `extent`. |
| raster-fields-06 | inconsistent | CONFIRMED (still present, was C10) | `raster-analysis/isolines/gpu-isolines.ts:409`, `:960-962`; `isolines/isobands-wgsl.ts:302-304` | Saddles are joined when the mean of the four corners is at or above the level. | Published `GPURasterContours` uses the bilinear asymptotic decider (`node_modules/@luma.gl/graph/dist/gpu-raster/gpu-raster-contours.js:447-448`). | Corners `10, 0, 1.1, 0` (v0..v3), level 1: mean 2.775, so the local rule joins the high corners; saddle value 0.991, so the published rule separates them. | Choose one rule for the library. The asymptotic decider is exact for the bilinear surface; the mean is what GDAL and Matplotlib use. The TSDoc states the local rule, so this is a choice, not a bug. |
| raster-fields-07 | inconsistent | CONFIRMED | `raster-analysis/raster-sampling/raster-sampling-utils.ts:73-79`; `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts:292-293`; against `raster-analysis/isolines/isolines-utils.ts:70-71`, `isolines/isobands-wgsl.ts:370-373` | Sampling and focal statistics treat only NaN as no-data and keep infinities as values. | The published band rule (and isolines, isobands and every contributor that canonicalizes a band) rejects every non-finite sample. | Focal window `{1, +Inf}`: mean `+Inf`, standard deviation NaN, `counts` 2. Bilinear sample between `+Inf` and `-Inf`: NaN with no no-data cell in the support. | Reject non-finite samples in both, or take the published `GPURasterBand` as input. |
| raster-fields-08 | doc-mismatch | CONFIRMED | `raster-analysis/gpu-raster-reconstruction.ts:40`, `:131-132` | `connectivity` defaults to 4. The prop has no doc. | `skimage.morphology.reconstruction` defaults to the 3 x 3 square (8). Published `GPURasterDilation` defaults to `'square'`. The committed oracle passes the cross explicitly. | Marker and mask that touch only diagonally: the default leaves them unconnected, the scikit-image default connects them. | Document the default, or change it to 8 to match the reference and the published kernel. `GPURasterFillHoles` should keep 4 (it matches `scipy.ndimage.binary_fill_holes`). |

Notes on the rows that need them.

**raster-fields-01.** The class doc says the trace runs "along the normalised field (one step is
`stepLength` output pixels)". With square output cells the two statements agree. With non-square
cells the kernel keeps the step at `stepLength` pixels but bends the direction. Output cells are
non-square whenever the viewport aspect differs from the extent aspect, which is common for a
longitude and latitude window. The independent oracle (`test/oracles/parity-wp4-lic.py:110`)
encodes the same per-axis scaling, and its only non-square case is a purely vertical flow, where
the scaling cannot change the direction. So no test sees it. Confirmed by `lic-direction.mjs`.
Second opinion welcome: one can read "pixels per step" as the specification, but then the output
is not a line integral along the field.

**raster-fields-02.** The fractions are right; only the area columns and the density and intensive
totals are affected. `GPURasterFractionalZonalDistribution` is safe when the caller passes
`cellAreas`. If only the origin moves between frames, nothing is wrong.

**raster-fields-04.** The prior review reported this as G12. The class doc says "use local
coordinates for large projected origins", but the kernel adds the origin back in float32 before it
subtracts the source origin, so local coordinates do not help. Confirmed by
`reprojection-precision.mjs`.

**raster-fields-05.** The prior review (C3, item 3) noted that the extent and the grid affine can
disagree. The sharper point is that they disagree for the default grid. Both conventions are
documented, so this is a consistency finding, not a wrong result.

## 3. Verified correct

- `GPUCostDistance`: link cost is the ground length times the mean of the two frictions, `sqrt(gx^2 + gy^2)` on diagonals, as in `r.cost` and ArcGIS. Chaotic min-relaxation of a monotone operator from `+Inf` has one fixed point, which is the float32 Dijkstra result. Tile activation, group gates and sweeps keep that fixed point (read through). Packed D8 offsets checked by script. Back-link codes are the ESRI powers of two, clockwise from east. Bands are `(b[i-1], b[i]]` as documented.
- `GPUCostDistance` ground scale (`analysis-core/raster-grid.ts`): geographic `N cos(lat)` and `M` per degree, Web Mercator `N cos(lat) / a` and `M cos(lat) / a` with `sin(lat) = tanh(y / a)`. Derived by hand; evaluated at the midpoint row, so symmetric.
- `GPUCostDistancePath`: walks the codes to a source (code 0); empty on unresolved, invalid or cyclic links.
- `GPURasterWalkingCost`: Tobler `0.6 * exp(3.5 * |S + 0.05|)` seconds per metre equals `6 exp(-3.5 |S + 0.05|)` km/h (script). Slope is rise over run, positive uphill, in the direction of travel from the source. The GRASS branch matches `r.walk` (`a*dS + b*dH` uphill, `c*dH` moderate downhill, `d*dH` steep downhill, signed `dH`, slope test on `dH / dS`), plus `lambda * mean friction * dS`. Geographic steps use the same midpoint-row ellipsoid scale as cost distance.
- `GPUDistanceField` exact: separable Felzenszwalb lower envelope, exact `u32` keys on square cells, ties to the smaller seed ID, correctly rounded square root. Jump flooding, anisotropic keys and degree units are declared as approximate or non-goals in the TSDoc.
- `GPURasterMedialAxis`: the primitive-lattice-step test equals the brute-force maximal-disk definition on 400 random masks, 17,230 cells, 0 mismatches (`medial-axis.mjs`, exact integers).
- `GPUIsolines`: high is `value >= level`; linear crossing in a canonical edge direction, so both cells of an edge give identical bits; cell-centre registration; high side on the left (checked on cases 1 and 2); no-data corner drops the cell; `t` stays in `[0, 1]` in float32.
- `GPUIsobands`: band `[b[k-1], b[k])`, same crossings and same saddle rule as `GPUIsolines`; partner-edge table checked for masks 5 and 10, joined and separated.
- `GPURasterSampling`: half-pixel centres, `floor` for nearest, bilinear weights, Catmull-Rom weights equal Keys `a = -0.5` exactly (script), edge clamp, strict and renormalize no-data as documented.
- `GPURasterProfile`: planar distances, sample count and segment search, gain and loss as sums of positive and negative differences of consecutive finite samples (part read).
- `GPUPolygonRasterization`: pixel-centre rule, half-open rows `min(y) <= centreY < max(y)` and columns `left <= centreX < right`, even-odd per polygon, smallest feature row wins. `GPURasterJoin` lookup uses the same half-open cell.
- `GPURasterFractionalCoverage`: exact triangle and unit-square clipping in pixel space (Sutherland-Hodgman, at most 7 vertices), inverse affine checked. `GPURasterFractionalZonalStatistics`: coverage-weighted mean, extensive and density totals as documented.
- `GPURasterFractionalZonalDistribution`: weighted quantile equals a transcription of exactextract 0.3.0 `weighted_quantiles.cpp` on 2,000 random cases (script, 4e-14); reliability-weighted sample variance `M2 / (W - W2 / W)`; Kish effective size.
- `GPURasterFractionalCategoricalStatistics`: sums per zone and category, majority and minority tie rules as documented.
- `GPURasterFillNoData` `gdal-four-quadrant`: a Node replica of the kernel matches the committed rasterio/GDAL fixture to 2e-6 on all four cases (`fill-four-quadrant.mjs`). `nearest` and all-donor IDW (`1 / distance`) are as documented.
- `GPURasterRankFilter`: index `min(floor(q * n), n - 1)`, the SciPy rank rule; clamp border equals SciPy `'nearest'`.
- `GPURasterReprojection` formulas: destination centre, inverse affine, `floor(column + 0.5)` for nearest, bilinear taps and strict or renormalize policy. Only the float32 origin handling is wrong (raster-fields-04). `GPURasterTileReprojection`: mosaic index arithmetic.
- `GPURasterReconstruction`: synchronous geodesic dilation or erosion clipped by the mask; the fixed point is the reconstruction. `GPURasterFillHoles`: reconstruction of the background from the border, 4-connected background, as `binary_fill_holes`.
- `GPURasterSkeleton`: neighbour order P2..P9, `2 <= B <= 6`, `A == 1`, and both sub-iteration products match Zhang and Suen (1984); the two sub-iterations are separate parallel passes. Caveat: as in the original algorithm, an isolated 2 x 2 block is deleted completely (`skeleton.mjs`). Worth one line in the TSDoc.
- `GPURasterWatershedSegmentation` FIFO: priority flood by (maximum elevation on the path, discovery age), as documented. It is not the scikit-image queue order, which uses the pixel's own value. On `[5, 2, 2, 2, 2, 2, 5]` with markers at the ends the kernel gives `1 1 1 1 2 2 2` and the scikit-image rule gives `1 1 1 1 1 1 2` (`watershed-order.mjs`). The docs do not claim parity. Forest: rooted Boruvka under a total edge order (maximum endpoint elevation, edge ID); the cut property holds because only unseeded components select.
- `GPURasterNDWI`, `NBR`, `EVI`, `SAVI`: formulas and the scale and offset placement on published `GPURasterBandMath`.
- `GPUParticleAdvection` and `GPULagrangianFlowMap`: midpoint RK2, bilinear field sampling at cell centres with clamp, linear time interpolation, Philox streams with separate purposes.
- `GPUFiniteTimeLyapunovExponent`: `J^T J` entries, closed-form largest eigenvalue, `0.5 * ln(lambda) / |T|`; recovers 0.8 for a linear strain of 0.8 (script).
- `GPUStreamlines` trace: midpoint RK2 with a scalar world step (part read).
- `GPURasterExtremaPyramid`: level-0 footprint `[cS, cS + S + e)`, 2 x 2 tree, sentinel never wins; the hierarchical build equals the direct definition for both footprints (argument in the TSDoc checked).
- `GPUFocalStatistics`: clipped window, circle test `dx^2 + dy^2 <= r^2`, population standard deviation by a second centred pass (no `E[x^2] - E[x]^2`).
- `raster-band-utils.ts` and `raster-grid/gpu-raster-grid.ts`: affine order `[a, b, c, d, e, f]`, `'area'` origin at the outer corner, `'point'` at the centre, raw no-data before calibration. All agree with the published `GPURasterMetadata` and `GPURasterBand` types.

## 4. Standards consistency

- **Row direction and georeference** (raster-fields-05). Two conventions in one family.
- **No-data for infinities** (raster-fields-07). Two rules.
- **Ground step length** is computed twice: `getD8Distance` (`cost-distance/raster-grid-utils.ts:188`) and `getStepDistance` (`gpu-raster-walking-cost.ts:349`). They agree on axis-aligned grids and share `getRasterGroundScaleWGSL`. They differ on sheared grids (raster-fields-03).
- **`cellSizeMode` (contract section 6)** is honoured only by `GPUCostDistance` and `GPURasterWalkingCost` in this scope. Pixel-space or affine-unit distances elsewhere are each documented: `GPUDistanceField` (grid units), `GPURasterProfile` (planar), `GPURasterFillNoData` and `GPUFocalStatistics` circle (cells), `GPURasterMedialAxis` (pixels), fractional areas (planar determinant). On a geographic grid these are not ground distances or areas.
- **Flow family has no declared context.** Velocities are "world units per time unit". The caller converts metres per second to degrees. The lenses do it with a sphere (111,195 m per degree, `modules/lenses/.../line-integral-convolution.lens.ts:23`), while `analysis-core/raster-grid.ts` uses WGS84 radii of curvature. There is no domain wrap: a particle that crosses the antimeridian respawns.
- **Affine in shader literals**: `gpu-raster-reprojection.ts:133`, `:173` and the three fractional statistics contributors (raster-fields-02). `GPURasterWalkingCost` and `GPURasterFractionalCoverage` now read parameters.
- **Private binary searches** where `analysis-core/kernels/binary-search-wgsl.ts` exists: `polygon-rasterization/polygon-rasterization-parameters.ts:85-112` (same result as `upperBound - 1` with a range check), `raster-sampling/gpu-raster-profile.ts` (three inline searches), `raster-sampling/gpu-raster-profile-scan.ts:123-138`, `isolines/gpu-isolines.ts:820-832`. All are upper-bound searches and numerically the same as the shared one.
- **Private ordered float key**: `raster-watershed-forest.ts:91-95` and `gpu-raster-fractional-zonal-distribution.ts:328`. Same bit trick as the shared `getOrderedKey`, except both fold `-0` to `+0` first (the shared key orders `-0 < +0`). The fold is intended.
- **Shader literal**: `PI` at `flow-texture/gpu-line-integral-convolution.ts:251`.
- **Connectivity defaults**: reconstruction 4 (raster-fields-08), fill holes 4, watershed 4, all through `connectivity: 4 | 8`; the published morphology uses `structuringElement: 'cross' | 'square'` with default `'square'`.
- Consistent: the unresolved and none sentinels (`0xfffffffe`, `0xffffffff`) across cost distance, fill holes and watershed; Philox with registered purposes in all three random users; `>=` as "high" in isolines, isobands and the published contours.

## 5. luma.gl official implementations

| Local | Published | Delegates? | Math agreement |
| --- | --- | --- | --- |
| `getRasterBufferBandNodes` (`raster-band-utils.ts:175`) | `GPURasterNeighborhood`, `GPURasterTextureToBuffer` | Yes. Radius 0, kernel `[1]`, `'clamp'`, `'propagate'`, no normalize. | Arguments do not change the meaning: output is the calibrated sample, validity is the published rule. |
| Spectral indices | `GPURasterBandMath` | Yes. | Formulas right. The EVI trick `add` with a `scale: 0, offset: 0` copy keeps raw no-data and validity. |
| `GPUDistanceField` and LIC texture outputs | `GPURasterBufferToTexture` | Yes. | No math. |
| `GPUIsolines`, `GPUIsobands` | `GPURasterContours` | No. Local adds several levels, world coordinates, stitching and bands. | Same registration (cell centres), same `>=` rule, same linear interpolation. Saddle rule differs (raster-fields-06). The published kernel also scales by magnitude before interpolating, which avoids overflow for values near `FLT_MAX`; the local `(level - a) / (b - a)` does not. |
| `GPURasterReconstruction` round | `GPURasterDilation` / `GPURasterErosion` | No. | One round equals the published radius-1 kernel with `noDataPolicy: 'ignore'`; skipping outside cells equals `'clamp'` for min and max. Default footprint differs (raster-fields-08). |
| `GPUFocalStatistics` | `GPURasterNeighborhood`, `GPURasterBoxBlur`, morphology | No. | Mean equals the published kernel with a uniform kernel, `borderMode: 'nodata'`, `'ignore-renormalize'`, `normalize: true` at valid centres. Differences: the local default fills an invalid centre (ArcGIS "DATA" behaviour) while the published kernels always keep an invalid centre invalid; local radius up to 64 against 8. Both are right under their own documentation. |
| `GPURasterRankFilter` | none | - | Uses the published border and no-data names (`'clamp' | 'nodata'`, `'propagate' | 'ignore'`). |
| `GPURasterExtremaPyramid` | `GPURasterOverview` (mean), `GPURasterCategoricalOverview` (mode) | No, different statistic. | Same level alignment: blocks start at pixel 0, `ceil(size / scale)` cells. Propose min and max upstream. |
| `GPURasterFillHoles` | `GPURasterConnectedComponents` | No. | Correct as is; the prior review covers the redesign. |
| `makeGPURasterBand` | `GPURasterBufferBand` type | - | Matches the type. `verticalDatum` is the accepted workaround in `dev-docs/upstream-workarounds.md`. |

No import of a private luma.gl path was found in this scope.

## 6. Test oracles

Independent oracle (reference library or exact arithmetic, not a port of the kernel):

- `GPUCostDistance`, `GPUCostDistancePath`: `skimage.graph.MCP_Geometric`; NetworkX Dijkstra with a pyproj metric for the cell-size modes.
- `GPUDistanceField`: `scipy.ndimage.distance_transform_edt` and `cdist` for ties.
- `GPUIsolines`, `GPUIsobands`, rings, stitching: `skimage.measure.find_contours` composites with Shapely and NetworkX.
- `GPURasterSampling`, `GPURasterProfile`: `scipy.ndimage.map_coordinates` and `CubicHermiteSpline`.
- `GPUPolygonRasterization`, `GPURasterJoin`: exact `Fraction` reference.
- `GPURasterExtremaPyramid`: NumPy footprint extrema taken directly from the pixels.
- `GPUParticleAdvection`, `GPUStreamlines`, `GPULineIntegralConvolution`: SciPy `solve_ivp`, `correlate1d`, `randomgen.Philox`. The LIC oracle shares the per-axis step of raster-fields-01.
- `GPUFiniteTimeLyapunovExponent`, `GPULagrangianFlowMap`: closed-form linear flows plus a float64 port.
- `GPUFocalStatistics`: `scipy.ndimage.generic_filter`, 720 parameter combinations.
- `GPURasterFillNoData`: rasterio `fillnodata` (GDAL) and a cKDTree nearest reference.
- `GPURasterFractionalCoverage` and the three statistics: Shapely intersections and exactextract 0.3.0.
- Spectral indices: spyndex. `GPURasterTileReprojection`: rasterio `reproject`, interior only.
- `GPURasterWatershedSegmentation` forest: an independent Kruskal reference.

Independent but tiny (one fixture of 3 x 3 to 5 x 5):

- `GPURasterRankFilter` (SciPy median), `GPURasterReconstruction` (scikit-image, cross footprint), `GPURasterFillHoles` (SciPy), FIFO watershed (scikit-image, 5 x 3, a case where the two queue orders agree), `GPURasterMedialAxis` radii (SciPy EDT).

Only a copy of the same formula, or structural checks:

- `GPURasterWalkingCost`: a float64 Bellman-Ford twin with the same Tobler and GRASS expressions (`test/geospatial-expansion/raster-walking-cost-parameters.spec.ts:51-82`). No GRASS `r.walk` fixture. I checked the formulas by hand and by script.
- `GPURasterSkeleton`: one 3 x 3 block that thins to its centre. No scikit-image or OpenCV fixture. I checked the conditions against the paper.
- `GPURasterReprojection` precision: no test with projected-scale origins, so raster-fields-04 is not pinned.
- `GPURasterMedialAxis` axis membership: structural tests; my brute-force script is the only independent check I know of.

## Resolution (2026-11-10)

Source paths are relative to `modules/analysis/src`, spec paths to `modules/analysis/test`. Every
spec named below was run on the test GPU (headless Chromium, Metal) or in Node. Only two things
were seen failing on the old behaviour first: the SciPy focal fixture (finding 07) and the
`rs02` polygon-composition spec (extra 01). For the other findings the kernel was changed before
the first run; each of those specs carries a hand-computed expectation that the old arithmetic
does not meet, and the LIC and saddle specs also assert that the old rule gives a different answer
on the same input.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| raster-fields-01 | fixed (code follows the documentation, roadmap 12.3) | `raster-analysis/flow-texture/gpu-line-integral-convolution.ts:284` (`getStepVector`), used at `:316` and `:320`. One step is the unit world direction times one scalar, `stepLength / length(direction / cell)`: tangent to the field in world space and `stepLength` output pixels long. Class documentation, `line-integral-convolution-parameters.ts` (`stepLength`), `docs/api/gpu-raster/operations-analysis.md`. CPU twin `test/oracles/line-integral-convolution-cpu.ts` and the generators `test/oracles/parity-wp4-lic.py`, `parity-wp4-lic-modes.py` changed to the same rule | `gpu-raster/flow-texture/gpu-line-integral-convolution.spec.ts` ("traces tangent to the field on non-square output cells"): uniform field `(1, 1)` on 2 x 1 cells against a closed-form straight line at 45 degrees in world space; it also asserts that the old per-axis picture differs at more than half the pixels | On non-square output cells the streaks follow the field (before: 26.6 degrees for a 45 degree field on 2:1 cells). Square cells and axis-aligned flow are unchanged, so no committed fixture changed |
| raster-fields-02 | fixed | New optional prop `parameters` (the coverage georeference view) on `GPURasterFractionalZonalStatistics` (`raster-analysis/gpu-raster-fractional-coverage.ts:361`, area read at `:433`), `GPURasterFractionalCategoricalStatistics` (`gpu-raster-fractional-categorical-statistics.ts:60`, `:212`) and `GPURasterFractionalZonalDistribution` (`gpu-raster-fractional-zonal-distribution.ts:88`, `:211`). With it the cell area is `abs(parameters[6])` on every encoding; without it the compiled literal is kept and the documentation says so. `docs/api/gpu-raster/expansion.md`, `fractional-categorical-statistics.md`, `fractional-zonal-distribution.md` | `gpu-raster/fractional-statistics-georeference.spec.ts`: one graph compiled for 10-unit cells and repacked for 20-unit cells; a 200 square unit zone keeps 200 in every area column and the density total goes from 700 to 600 (the literal gave 50 and 150) | None unless the new prop is passed. A caller that repacks the coverage georeference for another cell size must now pass the same view to the statistics classes |
| raster-fields-03 | fixed (rejected at construction) | `raster-analysis/raster-band-utils.ts:89` `validateRasterGridOrthogonal` (tolerance `1e-6` on the cosine between the pixel axes), called from the `GPUCostDistance` constructor (`cost-distance/gpu-cost-distance.ts:329`) and from `getGPUCostDistanceParameterValues` (`:107`). Class documentation and `operations-analysis.md` | `gpu-raster/cost-distance/gpu-cost-distance.node.spec.ts` ("raster-fields-03"): the report's sheared grid throws from both; a 2 x 3 grid rotated by 30 degrees constructs and its packed cell sizes give both diagonal lengths | `GPUCostDistance` and its packer throw `... grid.affine must have orthogonal pixel axes ...` for a sheared grid (before: diagonal moves costed with one wrong length). Rotated and axis-aligned grids are unchanged |
| raster-fields-04 | fixed | `raster-analysis/gpu-raster-reprojection.ts:140-145`: without a projection the destination centres are generated relative to the source origin (origins differenced in float64 on the CPU); with a projection the sampler adds `destinationOrigin - sourceOrigin` as one float64 difference (`:194`). Class documentation, `expansion.md` | `gpu-raster/raster-reprojection-precision.spec.ts`: 1 m and 0.25 m grids at a northing of 4.6e6 and a projection case with `destinationOrigin` at 4.6e6, bilinear over the plane `64 * row + column` against the float64 source pixel coordinate (the bound of 0.05 on the plane is 0.001 of a row and 0.05 of a column) and nearest against the float64 cell | Source pixel right to well under a cell at projected-scale origins (before: off by 0.25 cell on the 1 m grid, up to 0.9 cell on the 0.25 m grid). No change for small origins |
| raster-fields-05 | deferred (roadmap 12.4); current rule documented | "Row direction" paragraph in the class documentation of `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings`, `GPULineIntegralConvolution`, `GPUStreamlines`, `GPUParticleAdvection`, `GPULagrangianFlowMap`, `GPUFiniteTimeLyapunovExponent`; `operations-analysis.md` (Conventions bullet and the isolines section), `lagrangian-diagnostics-mob06.md`. No construction-time check, as instructed | none (no formula change) | none |
| raster-fields-06 (and luma-official-01) | fixed (bilinear decider, roadmap 12.3) | `raster-analysis/isolines/isolines-utils.ts:80` `ISOLINES_SADDLE_WGSL` (`isSaddleJoined`), used by `gpu-isolines.ts:421` and `isobands-wgsl.ts:306-307`. Twins `gpu-raster/isolines/isolines-oracle.ts`, `isobands-oracle.ts` use the float64 saddle value; generators `test/oracles/parity-wp4-contours.py`, `parity-wp4-isobands.py` use the determinant. Class documentation, `operations-analysis.md` | `gpu-raster/isolines/isolines-saddle-rule.spec.ts`: the report cell `10, 0, 1.1, 0` and the luma-official cell `100, -9, 1.1, -9` at level 1 against hand-computed segments for `GPUIsolines` and published `GPURasterContours`, band triangle counts and areas for `GPUIsobands`, the mirrored diagonal, a cell where the saddle value is above the level and the mean below, an exact tie, and a 12 x 9 raster of 88 saddles on which the local segments equal the published ones (the spec first checks that the mean and the bilinear rule disagree on more than 10 of those cells) | A saddle whose corners are far from symmetric can connect the other way (corners `10, 0, 1.1, 0` at level 1: joined before, separated now). Symmetric saddles are unchanged. No committed fixture changed: every isoline, isoband and ring parity spec passes on the old fixtures, so none of their cells is decided differently by the two rules |
| raster-fields-07 | fixed (converged on the published band rule: every non-finite sample is nodata) | `raster-analysis/raster-sampling/raster-sampling-utils.ts:77` (`GPURasterSampling`, `GPURasterProfile`), `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts:296`. Twins `gpu-raster/raster-sampling/raster-sampling-oracle.ts`, `gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-oracle.ts`; generator `test/oracles/parity-wp1-round4.py` (`computeFocal`); fixture `test/oracles/fixtures/parity-wp1-focal-small.json` regenerated with the real generator (see "Review follow-up"; a first hand patch was discarded). Prop documentation, `operations-analysis.md`, `gpu-spatial-analysis/operators-statistics.md` | `gpu-raster/raster-sampling/gpu-raster-sampling.spec.ts` and `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.spec.ts` (tests named "treats an infinity as nodata"), hand-computed | An infinite cell is nodata: focal windows leave it out (mean, sum, min, max, standard deviation and `count` change next to an infinity), sampling returns NaN or the renormalized finite neighbours where it returned an infinity |
| raster-fields-08 | fixed (default changed, roadmap decision by the lead) | `raster-analysis/gpu-raster-reconstruction.ts:142`: `connectivity` defaults to 8 and is documented; `GPURasterFillHoles` keeps 4 explicitly (`:300`) and documents it. `expansion.md` | `geospatial-expansion/raster-terrain-expansion.spec.ts`: "reconstruction spreads through corner neighbours by default" and "hole filling keeps a 4-connected background by default", hand-computed. The scikit-image fixture case now passes `connectivity: 4`, the footprint it was generated with | `GPURasterReconstruction` without `connectivity` now floods through corner neighbours (a marker and a mask region that touch diagonally are connected). Pass `connectivity: 4` for the old result. `GPURasterFillHoles` is unchanged |

Counts: 7 fixed, 0 not reproduced, 1 deferred.

### Not in the report

- **Extra 01, coverage slivers on axis-aligned grids (fixed).**
  `geospatial-expansion/rs02-fractional-categories.spec.ts` ("polygon composition retains
  exactextract geometry bounds") failed in this checkout before any edit of mine reached the
  coverage kernel: zone 2 held a tuple of fraction 2^-26 that exactextract does not have, so the
  zone had valid support and `proportions` was 0 where the reference is NaN. The per-frame
  georeference of `GPURasterFractionalCoverage` maps a vertex to pixels as `(e * x - b * y) / det`,
  which rounds the product and the quotient separately (at the last commit the affine was a set of
  shader literals, which the compiler can simplify; I did not run that version). First cure, in
  `point()` of `raster-analysis/gpu-raster-fractional-coverage.ts`: on an axis-aligned grid
  (`b = d = 0`) each axis is the single division `x / a`, `y / e`. That made the spec pass but
  relied on the division being exact for the fixture's ratios; it was replaced by a snap of the
  pixel coordinate, see "Review follow-up" (fr-fields-07).
- **Extra 02, exact saddle ties.** With the corners divided by the largest magnitude before the
  determinant (what published `GPURasterContours` does), the local kernel missed an exact tie of
  `gpu-raster/isolines/gpu-isolines.spec.ts` (random integer field, frame 0, segment 120). The
  local decider takes the differences unscaled unless a magnitude is above 1e18, so the
  determinant is exact on integer rasters. See "Upstream gaps".
- **Compensated sums.** The only mention in these directories is the comment at
  `raster-analysis/gpu-raster-fractional-zonal-distribution.ts:405`, which says the kernel does
  not use a compensation expression: it stripes each zone over 256 partial sums, reduces them in
  a fixed tree and centres the values on an anchor. Tested on the GPU at the largest accepted
  zone (65,536 tuples, a mean of 100,000):
  `gpu-raster/fractional-zonal-distribution/fractional-zonal-distribution.spec.ts` ("sums of 65536
  tuples keep float32 accuracy"). The area and weight sums are within 1e-5 of the float64 values;
  a serial float32 loop over the same terms is off by 6.5e-4 (areas) and 3.5e-4 (squared weights).
  The cure survives; nothing was replaced.
- **Tiled relaxation engine** (`raster-analysis/cost-distance/raster-relaxation.ts:596-1077`),
  read in full. A tile is active in iteration `i` when a tile of its 3 x 3 neighbourhood changed
  in iteration `i - 1` (stamp `>= i + 1`); a tile that changes stamps `i + 2`. A tile that ran and
  stored nothing is at a local fixpoint for the halo it loaded, and that halo is final unless a
  neighbour changed, which re-activates it. So a round without a change is a global fixpoint, in
  tiled and in sweep passes alike (a sweep only adds visibility of earlier tiles of its line). The
  inner loop limit cannot hide work: a tile that hits it has stored a change and is active again.
  The group gate marks `converged` only for a round whose change bit is clear and disarms without
  `converged` at `maxIterations` (contract section 4). I found no defect. Pinned by
  `gpu-raster/cost-distance/gpu-cost-distance.spec.ts` ("sweeps reach the Dijkstra fixpoint and
  report an iteration limit hit mid-sweep"): a 128 x 128 serpentine with sweeps from iteration 2
  reports `converged = 0`, `iterationCount = 7` and NaN for every passable cell at a budget of 7,
  and equals Dijkstra when converged, in fewer iterations than plain tiled passes. Not checked
  here: how the terrain epsilon fill and flat resolution read the status words (their files are
  another stream's).

### Second opinions asked by the report

- **FIFO watershed order.** Code unchanged. The class documentation and `expansion.md` now say
  that the queue order is by path maximum, not scikit-image's order by cell value, and that the
  committed scikit-image fixture is a case where the two agree. The 7-cell row is pinned as an
  example of the documented rule in `geospatial-expansion/raster-terrain-expansion.spec.ts`.
- **Zhang-Suen and the 2 x 2 block.** Documented in the class and in `expansion.md`.
- **Independent oracles.** `raster-terrain-expansion.spec.ts`: Tobler walking cost against the
  published speeds (6 km/h at minus 5 percent, 5.037 km/h on the flat) in both directions; the
  skeleton against a pure-Python implementation written from the 1984 paper on a 2 x 2 block, a
  3 x 7 bar, a line and a T (the expected masks are in the spec).

### Fixtures and generators

- Superseded by "Review follow-up" below. The first pass found no NumPy or SciPy (it looked only
  at the Homebrew Python 3.15), hand-patched `parity-wp1-focal-small.json` with a pure-Python
  implementation and did not run the five edited generators. That patch reproduced the SciPy
  values only to 1e-9: besides the 42 cells an infinity reaches it rewrote 12 standard deviations
  outside that reach by one float64 step (54 cells by exact comparison, not the 42 first
  reported). The fixture has since been regenerated with the real generator and the other
  generators have been run.

### Upstream gaps

- `GPURasterContours` (`@luma.gl/graph/gpu-raster`): one level per instance, pixel coordinates,
  loose segments. To delegate, `GPUIsolines` would need several levels in one pass, an affine or
  extent applied to the vertices, global edge ids for stitching, and a band (two-level) mode for
  `GPUIsobands`.
- `GPURasterContours` divides the corners by the largest magnitude before forming the saddle
  determinant, so on an exact tie its choice depends on rounding, against its own documentation
  ("an exact determinant tie connects the high-valued diagonal"). luma.gl would need to take the
  differences unscaled unless the products can overflow.
- `GPURasterDilation` / `GPURasterErosion`: a radius-1 round clipped by a mask with a GPU-gated
  fixpoint loop is what `GPURasterReconstruction` needs; the published classes have neither the
  mask nor the loop.

### Left for the lead

- `getD8Distance` (`raster-analysis/cost-distance/raster-grid-utils.ts:188`) is also used by the
  terrain hydrology kernels with `cellSizeMode: 'uniform'`. They still accept a sheared grid; only
  `GPUCostDistance` rejects it. Their constructors belong to the terrain streams.
- The lens scenario `gpu-raster.raster-reconstruction` (`modules/lenses`, out of bounds) does not
  pass `connectivity`, so it now runs 8-connected and its recorded numbers may change. The
  isolines and isobands lens records can change on asymmetric saddles.
- The shared typecheck covers `src` only; the new and edited spec files were run, not typechecked.
- `analysis-core/doc-parameters.node.spec.ts` reports `gpu-spatial-analysis/operators-weights.md`
  as stale. That page is another stream's; the pages edited here are in sync.

### Review follow-up (fix-review-fields.md, fr-fields-04 and fr-fields-07)

NumPy 2.0.2, SciPy 1.13.1, scikit-image 0.24.0, Shapely 2.0.6 and Pillow 11.2.1 are importable
under `/usr/bin/python3` (Python 3.9.6, user site-packages). `rasterio`, `randomgen` and
`pointpats` are not. The generators were run unchanged through two scratch drivers that redirect
the fixture output to a scratch directory and register a stand-in only for a missing import:
`rasterio.open().read()` through Pillow, `randomgen.Philox` as Philox4x32-10 written from its
definition, `pointpats` empty (the focal fixtures do not use it). Both working stand-ins are
checked by the comparisons below: the regenerated fixtures hold the decoded inputs and the noise,
and those are bit for bit the committed ones made with the real packages.
Closing pass (2026-11-11): the same generators rerun on the pinned set with the real packages, same results, focal fixture replaced; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

- **fr-fields-04, focal fixture.** `test/oracles/fixtures/parity-wp1-focal-small.json` is now the
  output of `generateFocal` of `parity-wp1-round4.py` (versions recorded in the fixture as the
  generator writes them, with `pointpats` marked as a stand-in). Against the fixture of commit
  `f5dcdc41` it differs in 42 cells, every one in a window that reaches the infinite cell (1, 4,
  8, 9, 4, 0 and 16 cells in the seven cases); every value outside that reach, including the
  standard deviations the hand patch had moved, is bit for bit the earlier NumPy 2.5.4 / SciPy
  1.16.3 value. The sixteen Grand Canyon focal fixtures regenerate with identical payloads and
  were left as committed.
- **Node spec.** `gpu-spatial-analysis/spatial-interpolation/parity-wp1-round4.parity.node.spec.ts`
  ("Focal synthetic cases ...") asserted the old rule on the fixture (count 1 and an infinite
  mean at the infinite cell). It now asserts count 0 and NaN there, and the hand-checked window of
  the radius-1 case (1021, 1022, 1029: count 3, sum 3072, mean 1024, range 8).
- **The other edited generators, run and compared with the committed fixtures.** None was replaced.
  - `parity-wp4-lic-modes.py`: the three fixtures regenerate identically.
  - `parity-wp4-lic.py`: fields, noise, speeds and every LIC value identical. The diagnostic
    `minimumPixelMargin` of the wind case differs by 7e-15 (the rewritten velocity expression
    divides by a norm that is 1 to rounding). The generator of commit `f5dcdc41`, run the same
    way, reproduces the committed fixture exactly, so that is the whole effect of the edit.
  - `parity-wp4-contours.py`: every segment, ring and edge id identical; only a provenance path
    string differs (the committed fixture says `modules/graph/...`).
  - `parity-wp4-isobands.py`: values, classes, triangles and components identical; the boundary
    edges are the same set in another order and 22 areas differ by 1e-16. The generator of commit
    `f5dcdc41` (corner-mean rule) gives the same output as the edited one in this environment, so
    the differences come from Shapely 2.0.6 and scikit-image 0.24.0 against 2.2.0 and 0.26.0, not
    from the saddle rule. No committed case has a saddle the two rules decide differently.
- **fr-fields-07, coverage slivers.** Confirmed, and it does not need a badly rounded division: a
  zone corner at `float32(3 * 0.3)` on a grid of `float32(0.3)` cells has the exact quotient
  2.9999998, so no division puts it on the boundary. The new spec in
  `geospatial-expansion/rs02-fractional-categories.spec.ts` ("a zone edge on a cell boundary
  leaves no sliver when the cell size does not divide exactly": 0.3 x 0.7 cells, two zones, one
  of them over a masked cell) failed on the division cure with 4 tuples instead of 3. Cure,
  `raster-analysis/gpu-raster-fractional-coverage.ts:258` (`snapToLattice`, applied in `point()`
  at `:261` on both the axis-aligned and the general path): a pixel coordinate within four
  float32 steps of an integer (`4 * 2^-23 * max(|k|, 1)`) is that integer. It does not depend on
  how the GPU rounds a division. The spec now gets exactly the three whole cells, fractions 1,
  and the masked zone stays without valid support. Limits, stated in the class documentation and
  in `docs/api/gpu-raster/expansion.md`: the snap moves a vertex by up to four float32 steps of
  its pixel coordinate (0.001 cell at column 2048), and it cannot absorb the rounding of
  `position - origin`, which scales with the origin.
  Behaviour change: a vertex within that distance of a cell boundary is treated as on it.

### Closing pass (2026-11-11)

- **raster-fields-05, row direction: left, with a plan.** Reason: the change has to land together
  with lens, website and example edits that this pass does not own. The study is
  [row-direction-flip-plan.md](./row-direction-flip-plan.md). No source file, spec or fixture was
  changed, so there is no behaviour change and no spec.
  - What it found. Only `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` and the output raster of
    `GPULineIntegralConvolution` receive a grid with an affine. The fields of the four flow classes
    arrive as a bounds tuple and `GPUFiniteTimeLyapunovExponent` receives no placement. Of 108
    constructor call sites in executable code, following the sign of `grid.affine[4]` leaves 76
    unchanged, mirrors 14 until the caller is edited, breaks 15 spec call sites until their grid
    is declared `rowDirection: 'north'`, and 3 are expected to pass either way. No oracle fixture
    changes; 3 lens records, 16 twins and 3 website view models are regenerated.
  - Recommended design. Resolve the row direction once at construction from `grid.affine[4]`, as
    the terrain classes do, with an optional `rowDirection` that throws on a contradiction. Keep
    the packer layouts and the smallest-y rule for bounds-only fields.
  - Evidence. Reading of every kernel and call site. The worked 2 x 2 example was checked against
    `gpu-raster/isolines/isolines-oracle.ts` (today's segment `[1.5, 1.25, 0.5, 1.25]`). The
    signed-cell-height alternative was run on the CPU twins of the sampler, LIC and particle
    advection. Nothing was run on a GPU.
- **Defects noticed, not fixed** (plan section 7): the `GPUFiniteTimeLyapunovExponent`
  documentation written in the first pass says a north-up seed grid must be reordered, but the
  exponent does not depend on the row order; three of the four field packers validate nothing
  about their extent while the flow-map packer rejects non-positive cell sizes; four showcase
  snippets import these classes from `@luma.gl/graph/gpu-raster`.
- **Checked, consistent:** no class of the flow family negates `v`.
- **Fixed by the lead after the study (2026-11-11).**
  - Defect 1: the `GPUFiniteTimeLyapunovExponent` class documentation and
    `docs/api/gpu-raster/lagrangian-diagnostics-mob06.md` no longer say a north-up seed grid must be
    reordered. Reversing the rows flips the sign of one column of `J`, which leaves the eigenvalues
    of `J^T J` unchanged.
  - Defect 2: the particle-advection, LIC and streamlines packers reject a cell size that is zero,
    rounds to zero in float32 or is not finite, through the new `validateNonZeroFiniteNumber` in
    `analysis-core/kernels/gpu-contributor-utils.ts`. A negative cell height is still accepted (the
    flow-map packer refuses it; aligning them is part of the row-direction decision). Spec:
    `test/gpu-raster/flow-texture/flow-field-extent-validation.node.spec.ts` (4 tests; the folder's
    node specs and the `parity-wp4` fixtures pass, 265 tests).
  - Defect 3 (showcase snippets) is with the session that owns `examples/`.
