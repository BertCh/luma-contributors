# Changes for callers: Cells and DGGS, trajectories, urban and landscape, recipes; network; raster and terrain; dataframe, point cloud and computation record

Part of [the record of caller-visible changes](README.md) from the 2026-11-19 and 2026-11-20 fix passes. Each bullet condenses what the implementers of those passes recorded; [README.md](README.md) says how the changes were tested and what that does and does not establish.

Counts in this file: results 118, rejections 116, added 94, deprecated 24, docs 18.

## Results that change

**Cells and DGGS**

- Tiled cell aggregation keeps its full signed sum until the tiles are merged, so cancellation across tiles and `sumOverflow` now match packed aggregation for retained keys. A clamped key that is later dropped no longer marks retained sums saturated, and coarser `GPUCellPyramid` levels inherit fine-level saturation.
- `GPUCellTableCompare` and the period-comparison recipes report sum saturation separately from lost cells and union truncation. Comparisons of saturated sums publish NaN while keys, counts and membership stay exact, and saturated period sums now reach the configured no-data colour.
- Extended-precision H3 indexing in `GPUPointToCell` reduces large longitudes exactly before choosing the quadrant, so clipped large longitudes select the correct cell.
- `GPUHealpixCellAggregation` drops impossible zero UNIQ keys and counts them in `invalidCount`. Nested pixel zero is still a valid cell.
- HEALPix indexing wraps negative residuals from reciprocal division before computing the turn, and wrapped tile indexing keeps canonical subnormal coordinates.
- For H3 and Quadbin, `GPUCellCover` keeps `candidateOverflow` for capacity loss only. Clipped or rejected coverage goes to the new optional `complete` output. Without `complete`, clipping or rejection suppresses the count, so callers that read clipped output before now get a suppressed count.
- `GPUCellCover` scalar invalid and uncertain diagnostics no longer overwrite caller padding beyond the first word.
- `GPUCellSetOutline` and `GPUCellCompaction` suppress nested rings until the whole source and the segment output are complete, including after a closed prefix is truncated. A clean replay restores them.
- GPU and CPU S2 polygon covering (`GPUS2PolygonCovering`, `CPUS2PolygonCoveringOperation`) accept either ring orientation. Clockwise holes are subtracted, and a reversed shell no longer selects its antipodal interior. Very small GPU regions use stable local edge planes, and collapsed zero-area rings return invalid status with zero visits and output instead of a successful exterior cover.
- `CPUS2CapCoveringOperation` treats an interior cap whose radius equals the float32 tolerance as empty, matching the GPU, instead of reporting frontier exhaustion.
- Cell-key encoders reserve the all-ones missing sentinel on output. A key that collides with it is published as missing and counted in `invalidCount`, and decoders reject a payload that reconstructs a missing cell.

**Spherical geometry**

- Unknown `GPUSphericalArcOverlay` and `GPUSphericalPolygonOverlay` operation strings used to fall back silently to symmetric difference. They are now rejected (see below), and the CPU overlay runs the options it validated, including getter-backed operation names.
- `GPUSphericalPolygonContains` keeps exact packed polygon vertices on the boundary even when float32 plane residuals round above tolerance. Open boundaries return contains 0 and closed boundaries return 1, both with status 2. Containment of very small rings is also more stable.
- Spherical count decoders read bounded raw GPU byte windows correctly. Malformed or incomplete counts and containment statuses, including the undocumented status 3, now throw instead of returning plausible geometry.

**Trajectories and movement**

- `GPUTrajectoryKalman` filtering and RTS smoothing keep position high/residual pairs, so small prediction increments and small later measurements far from a segment's first fix are no longer lost. The smoother also rejects non-finite backward corrections as a whole, keeping the filtered state and covariance with status bit 8.
- `GPUBrownianBridgeUtilization` keeps small cell-to-fix and midpoint residuals, so densities no longer shift on translated narrow grids or grids with distant minima. With explicit truncation, the cutoff boundary cell now follows squared float32 cell-relative offsets; the pinned boundary cell changes from CPU zero to a nonzero density.
- `GPUTrajectoryStaypoints` breaks observed dwell at missing fixes. Missing positions no longer count as dwell even when the optional `missingFixCount` output is not requested.
- Staypoint and location centres count equivalent seam longitudes (180, -180, 540) and exact-pole spellings once. Pole-only centres use longitude zero, and raw `[0,180]` now resolves like `[0,-180]`. CPU staypoint centres stay in [-180, 180) after float32 rounding.
- Planar location clustering (`GPUTrajectoryLocations`) keeps exact boundary points inclusive but excludes a neighbour that is representably just outside, such as `(1, 2^-13)` at radius 1. `GPUTrackSimilarity` EDR and LCSS apply the same rule.
- In great-circle contexts, `GPUMobilityIndicators` radius of gyration now defaults to a spherical centre (`gyrationCenter: 'spherical-mean'`). Seam-crossing tracks get the short-arc result and balanced antipodal centres return NaN. Planar contexts and an explicit `'coordinate-mean'` keep arithmetic centring. Stationary tracks give exactly zero on CPU and GPU.
- The CPU mobility fallback returns suppressed outputs before it interprets row indices that producer loss has invalidated.
- `GPUModalSplit` uses float32 exclusive speed bounds on both CPU and GPU, so CPU classifications exactly at a bound can change. The supplied-mode route now returns missing predicted IDs, and modal output length is `trackCount * modeCount` (0 for offsets `[0]`, `modeCount` for `[0,0]`).
- `GPUTrackSimilarity` and `GPUTrackSimilaritySearch` keep extreme representable planar Fréchet distances, and threshold membership no longer depends on the rounded distance. Ambiguous planar thresholds are decided from the stored endpoints, so `(-2^-25,0)` against `(1,0)` no longer matches at radius 1 even though the displayed Fréchet distance is 1. Equality is still included. Search uses the Fréchet producer's decision for retained pairs.
- Track similarity no longer reports invalid coordinates or an unusable Fréchet threshold as clean pairs (new pair status bits 8 and 16). A bad threshold keeps a usable distance.

**Recipes**

- Weighted-point `addHotSpotAnalysisRecipe` runs publish `sumOverflow` on the aggregation stage and the joined table. Saturation or upstream capacity loss now suppresses all dependent inference, including fixed moments, while counts and study inclusion are kept. The joined table reports source loss as `candidateOverflow`.
- In `addSpaceTimeHotSpotsRecipe`, truncation of the sorted groups suppresses inference until the complete input is submitted, with both default and opt-in occupancy.
- CPU choropleth sums and means (`addPointsInPolygonsChoroplethRecipe`) use compensated binary64 accumulation: `[2^55, 1, -2^55]` now gives sum 1 and mean `float32(1/3)` instead of 0 and 0. Get-only proxy settings are honoured, so asking for `sum` of a matched value 7 returns 7 instead of the default count 1.
- Choropleth and zonal-distribution recipes group by polygon row. Empty point vectors with zero or several empty chunks return the documented empty reductions, and nonempty ordinary point vectors and inherited point-source kinds now work through both recipes.
- `addRateClusterMapRecipe` keeps transformed weights metadata through analytic and permutation inference. After missing rates exclude neighbours, surviving row-standardised weights are restandardised, so local lag, Moran and permutation results change.
- `addSpatialRateRecipe` returns NaN by default when every non-self donor is excluded, including donors with negative events or zero-weight slots. `own-rate` remains the explicit fallback to the focus rate.
- `addSpatialRegressionRecipe` suppresses spatial inference when the weights certificate is false, while OLS results stay available.
- `addStopValueCubeRecipe` sets final overflow and suppresses counts when required weights exceed retained weights, even without producer overflow flags, and recovers when a later requirement fits. Invalid live timestamp indices, or a live count above capacity, publish `invalidCount` and suppress counts.
- Recipe stage status views expose only the rows actually written, even when a caller supplies an oversized output. Original overrides and legacy `invalidCount` aliases keep their identities and lengths.
- `addChangeOfSupportRecipe` writes requested nested count, candidate-loss and required-count destinations. Before, they were silently dropped.
- Recipe destination overrides keep published CommonJS graph-view identities and leave omitted optional status keys absent. Raster status tuples and caller view identities survive both the `output` and `outputs` spellings.

**Urban and landscape**

- `addStreetProfilesRecipe` passes invalid segment ownership on to exact facade lengths and building summaries, which are now suppressed.
- `addBuildingAdjacencyRecipe` reads all supplied candidate, run and vertex loss evidence even when overflow is absent. On loss, `complete` becomes 0 and summaries are suppressed. A shared-path count on its own no longer certifies full adjacency; pass the producer's aggregate overflow or the complete set of loss words to get `complete = 1`.
- Reference cell-centre singleton proximity returns a missing nearest neighbour, as the boundary route does. Radius certification compares the stored infinity word, avoiding a Metal float-infinity comparison.
- `addLandscapeMetricsRecipe` with dense labels suppresses measurement and reconciliation outputs when labelling is unfinished or over capacity. `provenance.patches` stays zero-based, and the one-based input is exposed separately as `provenance.denseLabels`.
- A caller-supplied convex tessellation overlay that fits is no longer rejected because unused default `capacity.vertices` or `capacity.rings` would exceed device storage.

**Network**

- `GPUNetworkAccessibility` passes supplied route-completeness and horizon loss on to nested scores and floating-catchment ratios when no public certificate is requested, suppressing them until complete costs arrive. Certificate routes keep incumbent values.
- `addNetworkDisruptionRecipe` keeps route convergence and horizon evidence: incomplete score deltas are NaN. A trip counts as newly unreachable only within the smaller of the baseline and scenario horizons, and noded comparisons pass on source-build loss.
- Without `output.status`, `GPUNetworkTravelTimePercentiles` and its operation forms set every `travelTimes` value to NaN and clear `reachedCounts` if any departure is invalid or unconverged. Requesting status keeps incumbent aggregates. Reached counts test finiteness by exponent bits, so disconnected vertices are not counted under Metal optimisation.
- `GPUNetworkTimeRouting` and the percentile solvers compare elapsed labels instead of absolute clocks. Short positive links and the correct predecessors survive large departure clocks, so route winners can change. The CPU forms also avoid float64 absolute-clock spacing loss.
- `GPUNetworkIsochrones` no longer paints pixels for leading CSR slots that belong to no source row. Very long roads keep every capped sample, including the midpoint, instead of leaving reachable pixels empty, and parameter packing writes exactly one 12-word record without overwriting the records that follow.
- Isochrone triangles take the nearest labelled sample across the whole raster in world units, with lower facility rows winning ties. Labels can change when the nearest sample lies outside the triangle's cell, and large uniform coordinate scales no longer erase a valid label.
- Isochrone cell outlines pass full aggregation loss evidence on and suppress incomplete ring geometry until a clean replay. Drive-time catchments keep cell-stage loss in recipe summaries, and demand joins over incomplete geometry publish `pointFeatureIds` of `0xffffffff`.
- `addDriveTimeCatchmentRecipe` now honours canonical `output` destinations. Returned views are the destinations actually written.
- Noded `GPUNetworkCostMatrix` results exclude spare node-capacity rows. Exact endpoint snapping stays exact when the float32 reciprocal projection is evaluated.
- CPU matrix and sparse-table hop budgets read the source labels captured at the start of each hop, so explicit small budgets can publish a larger, correct incumbent cost (6 instead of 3 in the pinned case). Explicit float64 sparse tables omit costs outside the finite float32 range and clear `complete`.
- With an explicit `costPrecision`, CPU matrices, reachability, service areas and sparse tables apply that precision at cutoffs and in predecessor and label tests. Before, the setting was ignored. Defaults are unchanged.
- CPU table composition `{...table, costLimit}` now honours the captured `builtCostLimit` even when `costLimit` changed.
- When required offsets exceed uint32, GPU `GPUNetworkPathExtraction` batches report loss and suppress publication instead of wrapping. The CPU accepts representable bounded batches it used to reject.
- Exact `GPUNetworkStatistics` readers infer the histogram width, fixing in-degree and total histograms that were decoded as 32 bins when callers chose 40 or 64. Legacy decoding ignores padding beyond the logical layout.
- `GPUNetworkSubgraphFilter` range aliases keep JavaScript comparisons at float32 decimal boundaries. On devices that flush float32 comparisons, ordered bit comparisons keep zero-row membership at subnormal bounds.
- `GPUAdjacencyMatrix` scalar resets keep trailing words.
- Reachability publishes duplicate accepted roots atomically.
- `GPURadiationFlowModel` keeps a zero diagonal for invalid coordinates and reports invalid resident coordinates for every affected origin row instead of inventing flows. Finite-size normalisation no longer overflows at extreme population ratios: the `[2**64, 1]` two-zone case returns `[0,8,6,0]`, and CPU outflows for dominant origins are no longer zero.
- JavaScript `normalization: null` resolves to `'none'` in both radiation operation forms, matching the pure CPU form.
- `GPUKernelFlowSmoothing` accepts both producer candidate-loss spellings, including `pairOverflow`, and withholds completeness when candidate loss is unknown.
- Without a certificate output, `GPUNetworkKernelDensity` sets density and kernel mass to NaN and query status to 1 when matrix loss evidence is supplied.
- Automatic `addVisibilityGraphRecipe` metrics choose feasible pivot batches under lane, device-byte and command limits, so some requests that used to throw now succeed with a sampled plan.

**Raster**

- `GPULagrangianFlowMap` marks a particle lost as soon as it lands on unsupported field, including with signed spacing. `GPULineIntegralConvolution` excludes noise reached after leaving velocity support, and CPU `GPUStreamlines` no longer publish a seed or endpoint that rounds onto an excluded field edge.
- `GPUStreamlines` fills unused CSR offsets with `positionCount`. `count` is still the active-line boundary.
- Raster sampling, profiles and `GPUDistanceField` seeds select the same cell at float32 interior boundaries, including 41 m and 0.85 m spacing and southward rows. Bilinear and bicubic packing use the same float32 step, so values near cell edges can differ slightly.
- `GPUIsolines` segment-only output publishes segment loss to a requested `polylineOverflow` word and clears it on complete or empty reuse.
- `GPURasterProfile` infers sample capacity from the shortest supplied sample column, in either width order.
- `GPURasterTexture` (GLCM) derives bin width from the same rounded float32 endpoints as its origin, and a value exactly on a bin boundary goes to the upper bin. Large-offset ranges change: the pinned case changes contrast from 9 to 4. Exact integer boundaries no longer advance one float32 word.
- Constrained `GPURasterSpectralUnmixing` no longer declares an inaccurate fit solved because spectra use small common units; stopping is unit-independent and residuals keep the original units. Dark-pixel constrained fits can now converge before the iteration budget runs out.
- `GPURasterEmbeddingProjection` PCA fallbacks skip spreads that overflow float32, using the widest usable spread or unit ranges. Wide symmetric stretches map zero to neutral grey.
- `GPURasterEmbeddingSimilarity` ignores non-finite query rows and lets very negative finite dot scores win. An empty eligible set returns NaN maximum, no-label query and zero matches. Requests with a matches output now generate valid WGSL.
- `GPURasterEmbeddingCovariance` and `GPURasterSnapshotCovariance`/EOF projection keep small variation around a nonzero mean, including under Metal fast math, so false temporal modes caused by a rounded absolute mean disappear.
- `GPURasterTemporalResample` fractional interpolation stays inside its bracket and applies `maximumGap` to elapsed time. Host and shader irregular-date conversion share one float32 reciprocal, so fractional coordinates can move by one ULP (the 419-day endpoints change from ±0.5735915303230286 to ±0.5735914707183838) and false outside-support statuses no longer occur.
- `GPURasterMedoidComposite` keeps acquisition provenance at very small or large scales, independently of constant bands. When at most one band varies, CPU and GPU both choose the earliest acquisition in the median interval.
- `GPURasterGeometricMedianComposite` returns singleton and identical observations exactly and updates from displacements on both CPU and GPU.
- The default `GPURasterPhenologyMetrics` second-season threshold now works for ordinary decimal fractions.
- `GPURasterRegionGraph` chooses its fixed-point scale without padding values outside the grid. An unused extreme tail no longer rounds a true mean difference to zero and causes a false merge.
- `GPURasterStreamingAccumulator` and standardised-index calibration centre fixed-point residuals before converting to float. Adjacent observations 10000000 and 10000001 now give a squared-deviation sum of 0.5 in any order, so derived dispersion and Fisk fits can change. Wide moment bounds keep their binary64 exponent, so state retained under the old units must be reset or replayed.
- Streaming observation counts saturate instead of wrapping.
- `GPURasterStandardizedIndex` places the zero atom at original zero for shifted distributions, fixing non-monotone percentiles. An unfinished gamma approximation now returns status code 3 with NaN and `iterationLimit` instead of being published as a probability. Shifted gamma and Fisk arguments no longer overflow, and large Fisk fits that used to return NaN are finite.
- Fractional raster clipping (`GPURasterFractionalCoverage` and consumers) inverts the uploaded affine basis rather than a separately rounded determinant, so near-parallel axes get the correct pixel coordinates and support. The patch-metrics packet restores the original physical area in word 6, and a successful legacy write clears a stale local-frame extension.
- Geographic `GPURasterZonalStatistics` reliability weights use the actual total-weight divisor.
- CPU `GPURasterReconstruction` converts samples, scale and offset to float32 before calibrating, matching the GPU. For example, a marker 33554433 with offset -33554432 is now value 0, validity 1, instead of NaN, validity 0.
- `GPURasterReprojection` accepts CRS JSON regardless of object member order.
- Focal regression and contrast histogram binning keep wide finite ratios on GPUs that flush subnormal reciprocals.

**Terrain**

- On the fixed-bound Euler-tour route, `GPUTerrainStreamOrder` always reports iteration-limit 0, including for odd receiver cycles; convergence still reports structural failure. `GPUTerrainStreamNetwork` suppresses reaches whose junctions depend on unknown donor membership and keeps the definite reaches separate.
- Pull accumulation and stream order retry a spurious weak compare-exchange without spending another round, so they can converge in fewer logical rounds.
- Fractional HAND reports `converged = 0` when drainage is unknown, matching D8 HAND. Unknown membership on its own does not report iteration-limit exhaustion.
- In terrain catchment profile diagnostics, excluded cells take precedence over unknown stream membership: `profileStatus = 0`, and an entirely excluded domain reports `converged = 1`.
- `GPUReliefShading` resident code 2 now computes Imhof swing in the ordinary pass. Unsupported weighting codes, including subnormal words, now invalidate output instead of silently selecting aspect.
- Missing eye or target heights in sight-line visibility produce no-data or add nothing to cumulative visibility, and missing hillshade light angles clear validity.
- Terrain visibility (`GPUTerrainViewshed`, `GPUTerrainLineOfSight`, `GPUTerrainCumulativeViewshed`) gives missing corners zero interpolation weight. Finite endpoints stay usable, off-ray no-data cannot expose a target behind a ridge, and exact integer lattice hits and flat grazing rays classify correctly.
- CPU `GPUProfilePeaks` applies its inclusive minimum and suppression ranking to the reported float32 prominence. Borderline peaks can now be accepted, and rounding ties keep the earliest sample row.
- `GPUTerrainSpikeRepair` releases partially allocated work buffers on failure and can be retried. Caller buffers stay owned by the caller.

**Dataframe**

- `GPUClassBreaks` treats -0 and +0 as equal for membership and maximum-breaks distinctness. Approximate natural breaks keeps finite cuts when the span overflows float32.
- Maximum-breaks classification keeps finite interior edges for extreme opposite-sign values and orders above-range gaps correctly. When a midpoint rounds onto the lower observation, the upper observation becomes the boundary, so `[1, 1+2^-23]` now counts `[1,1]` instead of `[0,2]`. Tiny gaps keep their order on subnormal-flushing backends.
- `GPUClassBreaks` with methods omitted picks compatible methods, so small class capacities and chunked inputs work without a method list. Natural-break row and storage limits no longer reject classifications that do not use natural breaks.
- Density-grid to class-break composition uses only the logical grid prefix. Explicit full views still include every observation.
- `GPUGroupStatistics` and `GPUValueCube` means divide the exact accumulated total by the exact finite count. Finite means now survive overflowing float sums and no longer round twice at subnormal spacing. `GPUKeyJoin` decodes means from integer totals without reciprocal overflow.
- Composite standardisation (`GPUCompositeScore`) keeps small variation even when its public mean cannot represent a half-unit offset, and keeps sparse-column means consistent across tiles.
- CPU quantile and percentile-filter ranks (`GPUColumnQuantiles`) follow the packed float32 rank policy up to 16,777,216 valid rows, so results for probabilities such as 0.7 and 0.07 change. Out-of-domain binary64 probabilities now give NaN instead of an extreme.
- CPU `GPUInequality` keeps very small weighted Gini values and returns exactly zero for constant positive incomes, per zone and pooled. GPU Gini, in both modes, works from scaled adjacent income gaps and keeps the same small values and equality.
- Atkinson returns 1 at epsilon >= 1 for a positive-mean population containing a positive-weight zero income, on CPU and both GPU modes. Empty and all-zero populations stay NaN.
- Standalone Arrow Decimal conversion rounds the original coefficient and scale once, so extreme representable values no longer underflow to zero and scaled zero no longer becomes NaN. Already-converted JavaScript Decimal numbers keep their values.
- Scaled group-sum decoding rounds the exact rational once: words `[1, 2097152]` at scale 3 give 3002399751580331 instead of 3002399751580330.5.
- Paired Arrow category keys exclude decoded null labels and match by label across differently ordered and plain dictionaries.
- `GPUTimeWindowFilter` float membership keeps strict subnormal lanes on flush-to-zero backends, sorted and unsorted. Missing calendar rows return -1 again.

**Point cloud**

- `GPUPointCloudNeighborhoods` keeps in-radius neighbours near the maximum float32 squared distance, and empty slots hold exactly the maximum finite value. Membership and retained rows are resolved at higher precision where rounding intervals overlap, so counts, retained rows and truncation flags can change.
- Neighbourhood halo certification compares the radius with coordinate margins, fixing false completeness at large translated coordinates.
- With a zero standard-deviation multiplier, `GPUPointCloudStatisticalFilter` keeps constant neighbour distances exact, including rows with different support counts.
- The `GPUPointCloudFeatures` status decoder no longer reports convergence for degenerate rows.
- Point-cloud raster metadata uses the surface producer's float32-rounded origin and cell size, so large fractional coordinates map to the correct column.
- `GPUPointCloudVoxelMerge` and `GPUPointCloudChange` keep their construction-time descriptors after caller mutation.

**Computation record**

- New default run IDs include the concrete computational dependency revision, so they differ from older captures. Explicit run IDs are unchanged.
- Invalid timing endpoints no longer poison unrelated nodes or emit non-finite Chrome trace coordinates. Measured zero GPU durations are kept.
- Pipeline annotations record shader-default specialisations as an empty override set, so readers no longer reuse the previous pipeline's overrides.
- Execution records snapshot their plain metadata, so reusing descriptors cannot rewrite earlier records.
- `GPUSnapshotArena` keeps the recorded name-to-value associations and ignores inherited source and metadata entries. Readback maps only the captured prefix and rejects truncated maps instead of publishing empty values.

## Input now rejected at construction

**Cells and DGGS**

- Cell indexing and measurement reject unknown precision tokens before choosing a kernel. High precision is rejected on non-H3 indexing routes. Kepler catalog callers that pass `extended` unconditionally must pass a family-aware value.
- Every form of `GPUHealpixPointToCell` and HEALPix aggregation rejects unsupported HEALPix schemes, including RING, instead of reading them as nested keys.
- Named point and HEALPix indexing with `rowCount: 0` rejects nonempty positions or masks, and supplied empty HEALPix graph names.
- `GPUCellMeasuresOperation` rejects an empty terminal selection.
- The checked HEALPix DGGS-JSON readback builder rejects ambiguous byte-typed property values (`Uint8Array`, `Int8Array`, `ArrayBuffer`, `DataView` and similar); pass decoded numbers. It refuses overflowed aggregation, reporting `requiredCount`.
- The counted-source forms of `GPUCellSetOutline` and `GPUCellCompaction` reject a source whose cells and count disagree, and a `source.status` without `count` and `overflow`.
- Selecting `complete` on the S2 route of `GPUCellCover` is rejected. Malformed `complete` views on counted-cell sources are rejected.
- `GPUCellPyramid` execution plans reject devices without enough storage bindings or workgroup size.
- S2 covering rejects unknown `mode` strings, which used to act as `exterior`. Accepted values are `exterior`/`interior`, cap refinement `uniform`/`level-synchronous` and polygon refinement `serial`/`level-synchronous`. GPU polygon views must match `positionCount` and `ringOffsetCount`.
- Cell-key encoders reject unknown free-bit families. Period-key packing rejects layout/count contradictions, unknown fill or layout values and invalid unsigned key values.

**Spherical geometry**

- Every form of spherical overlay rejects unknown operation strings; the error lists the accepted values.
- `GPUSphericalArcOverlay` rejects count and offset views that share a physical buffer. `GPUSphericalPolygonOverlay` rejects result-status writes that alias boundary-loss or feature-offset reads, as well as assembly-loss count/`candidateOverflow` aliasing and overlapping aligned bindings within a co-written status group. Disjoint packing in separate passes stays valid.
- Spherical count decoders reject NaN, infinite, fractional, negative and above-uint32 words. Containment readbacks must contain exactly one word.

**Geographic inputs**

- `GPUPointBuffer` rejects unsupported or conflicting coordinate declarations, and explicit `null`, so it no longer falls back silently to planar.
- Point-buffer parameter packing rejects radii that become zero or non-finite in float32, and short destinations.
- `GPUDatumGridTransform` rejects unknown operation and direction strings.
- Geographic viewport bounds require an explicit `size` for ambiguous XY/XYZ batches and reject non-canonical longitudes. Some in-repo Kepler fixtures and `examples/deck/kepler-compute/src/source-publication.ts` need `size: 3`.

**Trajectories and movement**

- `GPUBrownianBridgeUtilization` rejects extents whose encoded axis collapses and a resolved positions length that differs from `rowCount`.
- Brownian-bridge `tailPolicy` and CPU `storage` reject unknown values. `storage: 'compact'` requires offsets starting at 0, nondecreasing and ending at `rowCount`.
- `GPUTrajectoryKalman` rejects unknown `covarianceMode` and `timestampFormat`.
- The CPU and named GPU forms of `GPUTrajectoryRestriction` reject incomplete, mixed or simultaneous timestamp/offset pairs (`timestamps`/`trackOffsets` against the legacy `times`/`offsets`).
- `CPUMobilityIndicatorsOperation` requires an explicit CPU entropy cap to be a positive uint32. A cap is ignored when CPU entropy is not requested.
- `GPUMobilityIndicators` with `gyrationCenter: 'spherical-mean'` rejects planar contexts.
- `GPUModalSplit` rejects invalid, ambiguous or collapsed encoded speed tables. Explicit `predictedModes` destinations must be writable uint32 storage of tripleg capacity, even on the supplied-mode route.
- `GPUTrajectoryThinning` rejects conflicting legacy and new destination spellings.
- `GPUTrackSimilaritySearch` rejects destinations for inactive metrics and unknown metric strings.
- Named `GPUTrajectoryRecursions` operations reject unsupported temporal axes, timestamp encodings that do not match `timestampFormat`, and unknown `timestampFormat` strings.
- Trajectory segments that connect bounded movement tables must supply `candidateOverflow` under `output.trips.status` whenever trips are requested, and under `output.triplegs.status` whenever staypoint loss views are supplied.

**Recipes**

- `addSpaceTimeHotSpotsRecipe` rejects a non-boolean `occupiedCellsOnly`, including `null`.
- A supplied `scratch.pointTable.status.sumOverflow` on hot-spot recipes is now validated: it must be one uint32 row.
- Recipes reject conflicting status aliases, unsupported output leaves and destinations for disabled stages. The error names the enabling option. This covers inactive category/source-value, hot-spot colour/permutation, rate-cluster inference, regression local-fit and cluster-outline stages, as well as convex `hullPositions` on the alpha route.
- `addChangeOfSupportRecipe` rejects CSR `distances`/`candidateOverflow` and unwritten diagnostics. `addStraightLineCatchmentsRecipe` rejects moment, dictionary and `statistics.status.incomplete` requests. Outer and nested overflow or `requiredCount` aliases must name the same view.
- Recipe glue kernels reject view scalar formats that disagree with the declared WGSL type.
- `addStopValueCubeRecipe` rejects non-packed timestamps/startRows/finishRows, empty supplied loss views, unsupported strides, unknown time selections and malformed count/`requiredCount` ports. The dictionary must match the weights row count.
- `addPointsInPolygonsChoroplethRecipe` rejects unknown options and a supplied `output.color` without colour settings. Recipe `backend`, `statistic`, `boundaryRule`, `islandPolicy` and `weightMeaning` reject unknown values, such as `grp` or `total`.
- `addSpatialRegressionRecipe` rejects output aliases of the weights `count`/`requiredCount` evidence and certificate aliases that an output write would overwrite.
- A space-time hot-spot settings update that fails validation leaves the previous settings intact.

**Urban and landscape**

- `addStreetProfilesRecipe` checks every supplied output layout before it reserves any result ID.
- `addSegregationDecompositionRecipe` rejects physical aliases, invalid device binding ranges and missing STORAGE usage before adding scenarios. The same ID can be retried after correction.
- Tessellation, simple tessellation, adjacency-with-structures and resistance recipes check overlay device limits, layouts and foreign views before reserving output, and report the dissolve or clip ID.
- `addLandscapeMetricsRecipe` rejects cell widths or heights that round to float32 zero (for example 1e-47). A `cellSize` must have exactly two entries.
- Urban and landscape recipes reject unknown `searchStructure`, `obstacleIndex` and `enclosureMode` values, as well as conditional `output`/`outputs` conflicts.
- `getRasterResistanceConnectivityParameterValues` rejects a sheared update when the optional `cellSizeMode` is geographic or Web Mercator.

**Network**

- `GPUNetworkTravelTimePercentiles` rejects invalid nested routing layouts. The GPU FIFO routing and percentile operations reject invalid declarations and mismatched resolved row counts; the GPU limits are [2,256] and [0,65536].
- Pure CPU FIFO routing and percentiles reject unknown cost models and waiting policies, and affine props that carry piecewise-only controls.
- Line-graph and statistics packers reject controls that overflow their encoded type, lose an integer count or lose a negative U-turn ban.
- Exponential and power decay require an explicit finite non-negative `beta` that stays finite in float32. Before, an omitted rate silently produced constant weights; `beta: 0` is still the constant-weight choice.
- Routing cost precision `null` is rejected in every form. GPU named forms reject `float64`.
- `GPUNetworkCostWeights` rejects `costsComplete` or `costLimit` without `output.status.complete`. The lens example with an optional `matrixCostLimit` must also supply `matrixComplete` and a `complete` destination.
- Noded `GPUNetworkCostMatrix` calls require a `sourceComplete` destination and matching topology identities. CPU operation forms reject `nodedTopology` and `sourceComplete`, and retained noded baselines must carry `status.sourceComplete`.
- Snapping rejects an unknown `startDirection`. Accessibility rejects unknown orientation and decay values, including prototype-inherited decay names.
- `GPUNetworkIsochrones` rejects wrongly formatted start costs, live start counts and cost limits. `startCosts.length` must equal `startNodes.length`, and view-valued `startCount`/`costLimit` must be one row.
- Drive-time raster recipes reject geographic positions; project them first. Explicit planar cell-outline recipes are rejected.
- `addDriveTimeCatchmentRecipe` rejects join `status.unresolvedCount`/`status.complete`, which had no writer, and conflicting destinations. Demand-match outputs without `isochrones.joinDemand`, sum/mean destinations without `demandValues`, and snapping diagnostics without `candidateCapacity` are also rejected. A supplied demand `candidateOverflow` must be one packed uint32 row.
- `GPUNetworkNoding` rejects restricted leaf capacity without a loss destination, and supplied `pieces.status.complete`/`unresolvedCount`, which it cannot write.
- Visibility CSR recipes reject conflicting metric or status leaves across `output` and `outputs`. CSR-only preflight rejects other modes and malformed scalar status ports.
- `GPUNetworkSubgraphFilter` rejects ambiguous legacy range bounds and NaN; use `edgeAttributeRanges`/`nodeAttributeRanges`. This affects `website/src/gpu/runs/entries/network-routing-access/network-subgraph-filter.ts:75`. Packed network readers reject bare `Uint8Array`, `Int8Array` and `Uint8ClampedArray` views. Unknown `degreeBinning`/`textureStatistic` strings are rejected.
- Line-graph settings reject conflicting radian and legacy names.
- `GPUNetworkPathExtraction` and search reject `output.predecessorEdges` without `output.predecessors`. Map-matching packing rejects settings that overflow float32 or whose positive scales underflow.
- Flow-demand preparation rejects short, sparse or missing edge rows instead of inventing endpoints.
- `GPUNetworkRationalAccess` and `GPUNetworkFacilityOptimization` reject unknown `demandMode`, `stepMode` and operation strings. Provenance inputs require a `complete` destination.
- Centrality continuation rejects distinct imported wrapper identities that alias one physical buffer.
- `GPUKernelFlowSmoothing` rejects count-only pair imports and null or malformed statuses; use producer status, or `completeness: 'known-complete'` with `count`. Conflicting `validPairCount` and `count` are rejected.
- CPU `GPURadiationFlowModel` rejects non-finite positions, and GPU operation forms reject unknown `normalization` strings.
- `GPUNetworkKFunction` packing rejects `activeSimulations` above 4,294,967,295, or above the compiled maximum on the instance-bound packer.
- `GPUEdgeBundling` rejects formats other than `uint32` and `float32`.

**Raster**

- Polygon rasterisation and raster joins reject colliding result and status buffers, including multiple imports of one physical buffer.
- `GPUTextureShading.getCommandNodesForTile` rejects origins that do not match the largest decimation scale, and empty, short, sparse or extra-coordinate arrays.
- `getRasterZonalCellArea` rejects explicit v1 shorter than four words and explicit v2 shorter than seven.
- `GPURasterEmbeddingZoneMean` rejects co-bound `tuples.zoneValidity` and writable means/weights in one physical buffer. The embedding constructors reject parent/child format mismatches and malformed chunk topology, and embedding packers reject truncated destinations.
- Five CPU raster operation forms reject empty, duplicate or input-colliding graph value names. Change detection, `GPURasterExtremaPyramid` and `GPURasterStretch` reject selections without a usable terminal, including typo-only and null-only selections.
- `GPURasterWalkingCost` and its packer reject unknown `model.kind`, and `false` is no longer treated as Tobler; omitted models still use Tobler. GRASS coefficient tuples must be well formed.
- `GPURasterReconstruction` rejects misspelled methods, which used to select dilation, and selected empty validity names. CPU walking-cost and watershed forms reject empty input destinations.
- `GPURasterTemporalResample` rejects `targetTimes` combined with `gridEnd`.
- `GPURasterMedoidComposite` and `GPURasterGeometricMedianComposite` CPU forms reject missing required destinations. Seven GPU constructors reject wrong-format or absent required ports.
- Best-available-pixel cloud ramps reject endpoints that are not strictly ordered when encoded.
- Region graphs reject incompatible port formats, supplied completeness evidence without a destination, oversized mandatory planes, too few selected-route bindings and logical input/output overlap. Unsupported Boruvka requests are rejected before tile assembly, so a dense retry with the same ID works.
- Raster and terrain constructors reject misspelled algorithm options, missing parent maps and missing or mismatched named data ports. Raster expansion constructors reject element formats that disagree with the port contract, including band metadata against buffer storage. Required raster GPU ports reject omission.
- `makeGPURasterBand` rejects non-scalar values and non-uint32 validity. CPU raster bands reject sample arrays that disagree with the declared scalar format.
- Fractional zonal `quantity` rejects values outside `extensive`, `density` and `intensive`. Fractional distribution operations reject absent required destinations and unsupported option strings.
- Fractional coverage packing now applies the full raster-context checks: integer width and height in [1, 4294967295], `pixelInterpretation` of `'area'` or `'point'`, six finite affine coefficients, a finite nonzero determinant, and packed axes that do not collapse in float32. A reused options-overload target needs ten words.
- The patch-metrics packer rejects a physical determinant that rounds to zero or infinity.
- Raster time-series and streaming parameter writers reject undersized destinations, malformed slice counts, zero headroom except for an empty reset, active frames above their declared total, and non-positive or underflowing float32 magnitudes. Mixed encoding fields in a plan-based frame and wrong-model or corrupted plans are rejected. In every case the previous packet is kept.
- Raster fixed-point and EOF matrix decoders reject incomplete words or elements instead of truncating.
- Flow-map, LIC and streamline packers reject placement disagreements between the new names and the tuple aliases. Flow maps reject distinct slice times that alias in float32, and streamline packing rejects occupancy reciprocals that are not positive finite float32. Dynamic Kepler epoch-time calls can newly throw.
- Raster policies reject unknown method, border, policy, mode and closure tokens instead of falling back. Inherited names such as `toString` are rejected.
- Weighted overlay rejects active linear ramps whose endpoints collapse in float32 or whose reciprocal is below 2^-126; use a table remap for wider domains. A rejected layer leaves the existing packet unchanged.
- The compact `GPURasterReclassify` packer rejects counts outside uint32 or inexact in float32.
- Tiled reprojection rejects genuine CRS mismatches and invalid projection formats before allocating scratch.
- `GPUIsobandRings` rejects malformed texture bands before reserving resources.
- The isoline decoder rejects undocumented typed arrays, including `Float64Array` coordinates, and segment-only CPU results that cannot certify stitching.

**Terrain**

- The 14 terrain metric packers, plus `GPUTerrainHorizon` and the cast shadow, reject sheared grids and pixel axes that become zero or infinite in float32 (1e-46 or 1e39).
- `GPUPointHorizonVisibility` rejects graphs above 32,768 estimated command nodes; increase `targetsPerDispatch` or partition the targets. Web Mercator windows must have square pixels, columns east and rows south.
- Sight-line `targetIgnoreFraction` must lie in [0,1], and standalone tolerances must stay finite in float32. Scalar solar position and sun tables reject invalid coordinates. Solar irradiance packing rejects overflowing controls and active counts that round or exceed the four-float index layout.
- Relief tint and contrast packers reject ascending ramps whose distinct endpoints collapse in float32. `lights` accepts only `'mdow'` or an array, and `lightWeighting` accepts only `'fixed'`, `'aspect'` or `'imhof-swing'`.
- Contradictory alias pairs are rejected: `rvtDirectionDegrees` with `anisotropyAzimuthDegrees`, `firstAzimuthIndex` with `firstAzimuth`, and `azimuthBinCount` with `azimuthSpan`.
- Packed horizon readbacks reject incomplete words and undersized records.
- `GPUTerrainFlowOperation` requires a per-cell output and rejects empty, status-only or route-inconsistent selections. It also rejects invalid `flowRouting`, `cellSizeMode` and `accumulationUnits`, and the watershed operations reject invalid `directionEncoding`. Inherited budgets above 1024 throw.
- `GPUProfilePeaksOperation` and `GPUSolarPositionOperation` reject empty selections. Status-only `GPUProfilePeaksOperation` runs require an integer `sampleCount >= 1` and a matching `values.length`.

**Dataframe**

- Time-window and indicator parameter packers reject overflow, positive fades or epsilon that round to zero, Palma shares that underflow, and endpoints that lose precision in their packed form. Word-window numeric endpoints must reconstruct exactly; pass bigint beyond the safe-integer range. Kepler time selection and the explorer and showcase playback can now throw for arbitrary fractional times.
- `splitTimestamps` with `{requireExact: true}` rejects the first row whose double-single value changes membership.
- With only approximate natural breaks, the three extra approximate stages can push `GPUClassBreaks` over the 32,768-node limit (for example, 18 classes and 660 chunks). Graph declarations also reject approximate natural-break class capacity above the histogram bins (default 2048), as well as foreign chunked views whose declared format or row count disagrees.
- Custom class edges must be nondecreasing. Ascending opacity ramps whose endpoints collapse in float32 are rejected. The class-breaks `method` and quantile `interpolation` options reject inherited property names such as `'constructor'` or `'toString'`.
- `GPUGroupStatistics` and `GPUKeyJoin` reject conflicting alias destinations or modes, invalid fills, `rightTable` without certification, and unknown group constructor keys.
- CPU and GPU inequality operations reject empty names, duplicate destinations, input/output collisions, unsupported logical ports (including `weightedGini`; select `output.gini`) and unknown `mode`. Named GPU operations validate `zoneCount` and `lorenzKnotCount` even for summary-only requests.
- `GPUCalendarBuckets` rejects units other than `second`, `millisecond`, `microsecond` and `nanosecond`, invalid Gregorian origins and unknown `periods.kind`. Generic dataframe constructors reject misspelled top-level options.
- Arrow preparation rejects invalid precision modes, time units and time field names, ordered dictionaries with duplicate labels, dictionaries whose UTF-8 bytes exceed 2,147,483,647, combined scalar tables too large to address, projection typos and non-float measure sources.
- Word-pair decoding rejects incomplete, overlong or invalid lanes and unsupported typed views.
- `GPUTransferMask` rejects unknown modes and layouts. Snapshot arenas reject unknown slot element types and splitting one encoder at a submission or discard boundary.

**Point cloud**

- `GPUPointCloudChange` rejects positive `confidenceMultiplier` or `halfDepth` values that round to float32 zero, and an unknown `uncertaintyCombination`; accepted values are `linear` and `quadrature`.
- `GPUPointCloudChange`, `GPUPointCloudGround`, `GPUPointCloudICP`, `GPUPointCloudNormalScale`, `GPUPointCloudFeatures`, `GPUPointCloudSMRF` and `GPUPointCloudStatisticalFilter` reject missing or wrongly formatted declared ports.
- `GPUPointCloudICP` rejects `metric` values other than `point-to-point` and `point-to-plane`, and `robust.method` values other than `huber` and `tukey`. It also rejects unequal explicit `iterations` and `maximumIterations`.
- `GPUPointCloudSMRF` rejects conflicting `mask` and `selection` views.
- Change and normal-scale status decoders accept only numbers, the documented typed arrays or raw bytes.

**Computation record**

- `withSynchronousScope`, and the deprecated `withScope`, reject callbacks that return a promise.
- Sidecar validation now reports invalid binding byte ranges, legacy null lanes, cross-category duplicate IDs, non-finite timing endpoints and ambiguous capture references. These are load-time reports.

## Added

**Cells and DGGS**

- `GPUCellTableCompareOutput.status.sumOverflow`. Period-comparison stages expose `sumOverflow`.
- A `'double-single'` precision request for cell indexing and measurement.
- `longitudePolicy: 'wrap'` for point indexing.
- HEALPix DGGS-JSON accepts bigint arrays and adds a checked readback builder, `createHealpixDGGSJSONFromReadback`. `CPUHealpixPointToCellResult` is a named type.
- `GPUCellSetOutline` and `GPUCellCompaction` take a counted-source form (`GPUCountedCellSource`, `GPUCellSourceInput`, `GPUCellSetOutlineInput`, `GPUCellCompactionInput`) and an optional `candidateOverflow` output.
- `GPUCellCover` gains an optional `complete` output for H3 and Quadbin.
- `GPUCellPyramid` gains static route limits, `getCellPyramidExecutionPlan` and optional `levelColumns`.
- An explicit zero-bit identity layout for one-period cell keys, and `unpackGPUCellPeriodKeyChecked`.

**Spherical geometry and geographic inputs**

- `validateUnitVector` and `GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE`.
- Spherical decoders: `decodeGPUSphericalConvexContainsStatus`, `decodeGPUSphericalPolygonContainsStatus`, `decodeGPUSphericalConvexIntersectionStatus` and `decodeGPUSphericalArcOverlayStatus` (with `numericallyUnresolved`), plus convex-intersection and arc-overlay count decoders that accept raw byte readbacks.
- `GPUPointBuffer` accepts `spatialContext`.
- `getGeographicViewportFit` and `GeographicViewportFit` fit compact antimeridian views.
- Datum-grid position and status decoders return `Float64Array` with aligned row statuses.

**Trajectories and movement**

- An optional `locationIterationLimitReached` input on mobility indicators. It suppresses only location-dependent results.
- `GPUTrajectoryStaypoints` gains a mask, a break policy and `missingFixCount`. `GPUTrajectoryLocations` takes producer loss and exposes `complete`.
- `GPUMobilityIndicators` gains a `gyrationCenter` option.
- `GPUModalSplit` gains `modeSpeedUpperBoundsFloat32` and an unknown-ID aggregation policy.
- `GPUTrajectoryThinning` gains `sourceRows` and `sourceSelectionMask` outputs.
- Track similarity pair status bits 8 and 16, with `decodeGPUTrackSimilarityPairStatus`.
- Brownian-bridge `tailPolicy` (full or seven-sigma) and `CPUBrownianBridgeUtilizationResult`.
- Prepared trajectories expose `timesMetadata`, `timesTemporalContext`, `timeWordsTemporalContext` and `spatialContext`.
- CPU restriction accepts `timestamps`/`trackOffsets` (`CPUTrajectoryRestrictionGraphInput`).
- Movement forms accept zero tracks and empty recursion fix arrays. Track similarity accepts zero pairs.
- `GPU_CELL_FLOW_TOPOLOGY_MAXIMUM_PAIRS_PER_PASS` (4096).

**Recipes**

- `occupiedCellsOnly` and `intermediates.cellMask` on `addSpaceTimeHotSpotsRecipe`. Hot-spot stages gain `sumOverflow`.
- Fleet zone-dwell stages expose a scratch-specific `candidateOverflow`.
- Spatial regression stages expose GWR per-row `rowCodes`, and `predictorCount` is inferred from divisible view layouts.
- An `output` spelling on recipe entry points.
- An optional `producerOverflow` on exposure and regional raster-change recipes. Valid zeros are kept only on complete frames.
- Rate and correlogram recipes expose input-weights diagnostics.
- Dasymetric and drainage stages publish `complete`, `zeroEligibleCount` and `unresolvedCount`.
- Packed planar choropleth operation factories, and a stop-value-cube builder (`addStopValueCubeRecipe`).
- `addTrackQualityRecipe` stages include the similarity pair status.

**Urban and landscape**

- `addBuildingTessellationReferenceRecipe` and `addLocalBuildingTessellationRecipe`. An explicit reference route accepts an inline `candidateRadius`.
- An optional `cellSizeMode` on `getRasterResistanceConnectivityParameterValues`.
- `GPUVisibilityGraphReferenceRecipeResult`.

**Network**

- `baselineAccessibilityComplete` and `scenarioAccessibilityComplete` on disruption results, plus a fifth, appended accessibility stage in `status.stages`. Retained baselines take a `costLimit`.
- A noding topology port, a matrix noded handoff, a source certificate and a borrowed result with direction metadata and a catchment adapter. Snapping gains `intersectionAware`.
- `isochroneDemand.status.candidateOverflow`.
- `predecessorEdges` on search, extraction and CPU search results, and `builtCostLimit` on cost tables.
- A bounded path decoder with separate node and edge completeness. Named extraction accepts zero payload capacity.
- `costsComplete`, `costLimit` and `status.complete` on facility optimisation and rational access, with a host feasibility helper for exhaustive facility search.
- An alternative-routes layout helper and decoder. Both route families accept empty query batches.
- Optional `reverseTopology` for centrality distance metrics; outward closeness, harmonic closeness and reached counts no longer need a transpose.
- `CPUNetworkLouvainOperation` returns five named diagnostic arrays with `decodeNetworkLouvainStatus`. `CPUNetworkStructureOperation` accepts a named topology input.
- `elapsedTimes` on time-routing contributors, and `unrepresentable` masks and named result types on CPU routing and percentiles.
- Optional direction certificates on `GPUNetworkPotentialPathArea`, a smoothing `complete` status, and `output.invalidInput` per origin on radiation operations.
- `costPrecision` on CPU matrix, reachability, service-area and sparse-table inputs.
- Visibility preflight and execution plans (`addVisibilityGraphRecipe.executionPlanner.getExecutionPlan`), and qualified `GPUNetworkVisibilityGraphCSR*` type names.
- An instance-bound `GPUNetworkKFunction` packer.
- Packed readers `decodeGPUNetworkStatisticsWords`/`FromBytes`, `decodeGPUNetworkSubgraphFilterCountsWords`/`FromBytes` and `decodeGPUNetworkCoarseningSummaryWords`/`FromBytes`.
- `labelsComplete` and `partitionProvenance` on coarsening.
- Line-graph `turnCostPerRadian`, `straightAngleRadians` and `uTurnAngleRadians`.
- Adjacency weighted maxima and textures work without a public `weightSums`, and analytics accepts extent-only declarations.

**Raster**

- `producerOverflow`, `incompleteSources` and `output.status.incomplete` on raster joins and whole-cell zonal statistics. `populationVariances` on `GPURasterZonalStatistics`.
- `GPUTextureShading.getCommandNodesForTile`.
- `GPURasterBand` texture inputs for fractional scalar/distribution, native uint32 categorical statistics, NDWI, NBR, EVI and SAVI, walking cost and watershed segmentation.
- `GPURasterTemporalResample.targetTimes` (exact Int64 targets).
- `prepareRasterStackInput`, with its props, result and layout types.
- `GPURasterEmbeddingKMeans.getResult(graph)` and `decodeGPURasterEmbeddingKMeansStatus`.
- Region-graph `labelsComplete`, `status.complete`, `GPURasterRegionGraphOutput` and execution plans. Terrain auto exposes the selected route and a cross-device estimate.
- `localOrigin` on particle, flow-map, streamline and LIC packers. Flow maps gain `timeOrigin`, and flow packers gain `fieldPlacement`, `outputPlacement` and `gridPlacement`. The LIC and streamline uint32 packers take no settings object.
- Encoding plans (`createGPURasterStreamingAccumulatorEncodingPlan` and similar), streaming `state.status` with `decodeGPURasterStreamingAccumulatorStatus`, `decodeGPURasterStandardizedIndexStatus` and a one-observation rolling window.
- A `zeroAtom` option on standardised indices.
- A fractional coverage local-origin overload and a fractional categorical execution planner.
- `output.labels` on direct and tiled warps for exact uint32/sint32 category IDs.
- An optional uint32 `activeBreakCount` on reclassify.
- A bounded isoline polyline readback decoder.
- `propagate` as the missing-data policy name for sampling and profiles.

**Terrain**

- `output.status.{converged,invalidCount,iterationLimitReached}` on GPU and CPU terrain catchment summary operations.
- `GPUPointHorizonVisibility.estimatedCommandNodeCount`.
- Prepared solar tables expose active-row counts, capacity and format.
- Viewshed tolerance packing accepts omitted settings.

**Dataframe**

- `status.sumOverflow` on group statistics and key-join aggregates.
- `scaleMode` for independent per-group or per-bin scaling.
- `how`, `probabilities` and `quantiles` spellings.
- Per-column `unmatchedValue`, a grouped `rightTable` with certification, and uint32 count outputs on `GPUKeyJoin`.
- `GPUClassBreaks` static limits and `getExecutionPlan`, with natural-break mode and algorithm provenance. CPU class breaks accept empty inputs.
- Backend-neutral quantile and rational-probability packers. Scalar control preparation returns counts, formats and `minimumDomainCapacity`.
- `CPUInequalityResult.weightedGini`.
- `GPUTimeWordsRelative.mask`, which writes NaN for missing times, and `splitTimestamps(..., {requireExact: true})`.
- Paired Arrow key preparation, nullable Arrow measures, Float16 widening, LargeUtf8 categories, batch preparation, and signed fixed-point group-sum decoding with exact bigint totals.

**Point cloud**

- `GPUPointCloudCoverageValidity`, which produces surface, terrain and canopy validity, and a point-cloud raster metadata helper.
- `decodeGPUPointCloudChangeStatus` and a row decoder, `decodeGPUPointCloudNormalScaleStatus` with flag names, and a feature/neighbourhood status decoder whose bit 8 marks incomplete support.
- Optional capacity inference, `neighborhoodStatus` on features and filtering, and `thresholdComplete` on filtering.
- `GPUPointCloudICP.maximumIterations` and `GPUPointCloudSMRF.mask`.

**Computation record**

- `dependencyRevision` on records and run IDs.
- `gpuTimeMilliseconds` per timing node.
- `declaredBindings` and `entryPoint` on node annotations, and `bindingByteOffset`/`bindingByteLength` on annotations and bindings.
- A lossless binary JSON capture codec on the public record entry.
- `datasetVersion` and `seed` on saved records.
- Affine, pixel-interpretation, frame and CRS metadata on raster captures.
- Snapshot arena checkpoints, discard and `markSubmitted(checkpoint)`.
- `withSynchronousScope` and `withAsyncScope`.

## Deprecated spellings

- `assertUnitVector`: use `validateUnitVector`. The arc-status field `unresolvedJunction` becomes `numericallyUnresolved`.
- Cell precision `'extended'` and `'fp64'`: use `'double-single'`.
- Point-buffer `coordinateSystem`: use `spatialContext`. An omitted viewport `size` is deprecated.
- The implicit-layout overload of `getGPUCellPeriodKeyPeriodBits`: pass an explicit layout. `unpackGPUCellPeriodKey` becomes `unpackGPUCellPeriodKeyChecked`.
- `modeSpeedBreaks`: use `modeSpeedUpperBoundsFloat32`.
- Thinning `rowIds`: use `sourceRows`. Thinning `validity`: use `sourceSelectionMask`.
- Recipe, drive-time, urban and visibility CSR `outputs`: use `output`. The isovist `measures` output becomes `localMeasures`.
- `GPUVisibilityGraphCSRProps`, `GPUVisibilityGraphCSROutput` and `GPUVisibilityGraphCSRStatus`: use the `GPUNetworkVisibilityGraphCSR*` names.
- Subgraph filter `nodeRanges`, `edgeRanges` and `edgeTimeWindow`: use `nodeAttributeRanges`, `edgeAttributeRanges` and `closedEdgeTimeWindow`.
- Line-graph `angleCost`, `straightAngle` and `uTurnAngle`: use `turnCostPerRadian`, `straightAngleRadians` and `uTurnAngleRadians`.
- The previous packed network statistics, subgraph-filter count and coarsening summary readers: use the `...Words` or `...FromBytes` readers.
- Edges-only path inference (`computeNetworkPathExtractionOnCPU` without `predecessorEdges`): supply `predecessorEdges`.
- Flow smoothing `validPairCount`: use `count` with `completeness: 'known-complete'`.
- Region-graph `searchStructure` for provenance: use the execution plan's selected route.
- Flow placement tuple aliases: use `fieldPlacement`, `outputPlacement` and `gridPlacement`.
- `zeroInflated`: use `zeroAtom`.
- Raster sampling and profile `strict`: use `propagate`.
- Relief `imhofSwing`: use `lightWeighting: 'imhof-swing'`.
- `anisotropyAzimuthDegrees`: use `rvtDirectionDegrees`. `firstAzimuth` becomes `firstAzimuthIndex`, and `azimuthSpan` becomes `azimuthBinCount`.
- Group-statistics `percentiles`, both the input and the output: use `probabilities` and `quantiles`. Key-join `kind` becomes `how`, and the column-level `sumOverflow` becomes `status.sumOverflow`.
- `CPUInequalityResult.gini`: use `weightedGini`.
- Point-cloud feature status `complete`: use `supportComplete`.
- SMRF `selection`: use `mask`. ICP `iterations`: use `maximumIterations`.
- `withScope`: use `withSynchronousScope`.

## Documentation and tests

- The cells guide shows counted sorting and deduplication before outlines, retained-input sizing, device-aware pyramid planning and independent sum-saturation checks. S2 precision claims are now based on the actual 10,176-point sample.
- The spherical guides state the validation, capacity, orientation, count-decoding and traversal contracts, and describe the current parallel S2 traversal costs.
- A paste-ready migration note covers the `candidateOverflow` destinations that trajectory segments now require.
- The movement guides distinguish final-window stop rules, location-iteration loss and staypoint loss. ERP is documented as a pseudometric, and great-circle snapping cutoffs use the declared length unit.
- The track guides explain the decision rule, every pair-status bit, and planar versus great-circle search bounds.
- The urban adjacency guide states the exact certificate evidence, including uncertified count-only results.
- The network guides are task-named with legacy fragments kept. They explain certified alternative-route prefixes, window versus total continuation budgets, decimal percentile ranks (probability 0.95 is percentile 95) and clock rebasing.
- Facility optimisation documentation states every simultaneous bound, including the one-facility 64/65 boundary.
- New guidance covers drive-time row domains and units, disruption score evidence and K-function instance-bound packing.
- Zonal documentation states the v2 default and padded layouts, distinguishes authalic-sphere whole-cell weights from WGS84 ellipsoidal fractional areas, and gives the correct reliability-weight conversion.
- The terrain drainage guide uses D8 for its first pipeline and explains translating `0xfffffffe` imported unknowns to `0xffffffff` before `GPUTerrainStreamOrder` and HAND.
- The terrain grid contracts list all 15 guarded contributors and the 32,768-node visibility limit. Relief guidance states the exact quantised offset bounds.
- A new canonical flow-map and particle guide is added. LIC origin packing guidance is corrected.
- A new NDVI-to-temporal-mean recipe example and a neighbourhood-statistic chooser are added, and the constrained-unmixing guidance is corrected.
- The streaming and climate guides cover encoding-plan reuse, saturating count certificates and standardised-index status code 3.
- The raster packer guidance states target minima and how rejected updates are handled.
- `GPUSnapshotTap` has its own documentation heading.
- Every TypeScript example on the raster operations and zonal pages passes strict snippet checking.
