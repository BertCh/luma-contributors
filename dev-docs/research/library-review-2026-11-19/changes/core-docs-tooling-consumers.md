# Changes for callers: Shared core, entry points, catalog and API docs, tooling and gates, lenses, Kepler adapters and examples

Part of [the record of caller-visible changes](README.md) from the 2026-11-19 and 2026-11-20 fix passes. Each bullet condenses what the implementers of those passes recorded; [README.md](README.md) says how the changes were tested and what that does and does not establish.

Counts in this file: results 64, rejections 33, added 31, deprecated 11, docs 20.

## Results that change

**Shared core and contracts**

- Active-count dispatch no longer wraps the workgroup count to zero for the largest accepted uint32
  counts; those runs previously did no work and returned an unwritten result.
- The shared circumcircle construction keeps the center and radius of exactly encoded dyadic
  triangles whose edge subtraction compromised the determinant, so Voronoi and hull consumers no
  longer lose those circles even when the original cross product is nonzero.
- CPU geodesic pairs with very large finite longitude endpoints (for example `2**62` degrees) now
  return the wrapped spherical distance instead of a false zero coincidence.
- CPU network routing picks the least passable parallel CSR slot even when bounded labels are
  incomplete, and overlapping CSR rows no longer inherit another source's tie witness. Reported
  predecessors can change from a wrong root to the one that actually witnesses the child's cost.
- Raster streaming moments and standardized-index calibration (`GPURasterStreamingAccumulator`,
  `GPURasterStandardizedIndexCalibration`) centre squared deviations in fixed-point units and size
  the moment unit from wide squared bounds. Adjacent observations `10000000` and `10000001` now give
  a squared-deviation sum of `0.5` in either order and across chunks, instead of an order-dependent
  `1` or `0`; large squares no longer vanish or clamp. Reset retained state when adopting the change.
- Raster context inverses use the packed-axes determinant, and validation, packing and georeference
  metadata all read one captured affine, so a changing getter or later origin mutation can no longer
  split a resolved context. The signed binary64 physical determinant is kept as `affineDeterminant`.
- Shared regression results: GTWR information criteria now count the variance term and use the
  Hurvich correction with `tr(S) + 1`; panel-error results built from finite-likelihood summaries
  report AIC, Schwarz and AICc as `null` instead of numbers that were not available.
- Kernel recording, graph chunk views and contributor operation lowering now work when ESM and
  CommonJS builds of the package are mixed in one process; these previously failed or lost chunks.

**Statistics and inference**

- Significance planning, summaries and cautions: required permutation budgets and attainability
  comparisons use binary64-scale rounding guards, so large families can now require more draws than
  the earlier undersized budget. An explicit doubled-tail construction at 19 draws reports a `0.10`
  floor; calls that omit the construction keep the old formula.
- Local LISA catalog significance uses the producer's exact permutation-count decision rather than
  rounded p-values: an exact `1/20` now rejects at packed `.05`, and an exact `1/25` no longer
  rejects at packed `.04`.
- GPU local Moran p-values are reconstructed from their integer exceedance counts at the host
  boundary, so equality with an attainable cutoff is classified inclusively. Imported values with
  no known counting convention report `pValueConstruction: 'unknown'` instead of a claimed floor.
- Rate cluster map tie recommendations are corrected at packed floor and cutoff boundaries: the
  boundary counts move from 3 to 4 on CPU and from 30 to 31 on GPU, and GPU packed `.02` at 99
  permutations now reports attainable with `permutationsNeeded = 99`.
- Kepler catalog GWR CPU fits centre and scale each local system, so full-rank translated or scaled
  designs that were rejected or inaccurate now fit; coefficients, fitted values, residuals, standard
  errors and hat diagnostics are returned in caller units. Standard errors keep contributions whose
  raw squared Gaussian weights underflow.
- Kepler OLS reports give the ridge penalty actually applied as `ridgeLambda` (`NaN` when its
  convention is unavailable) and no longer fall back silently to ordinary coefficient inference.
- CPU spacing statistics count prefiltered invalid coordinate pairs exactly once in `invalidCount`,
  including when distance statistics are sampled.
- Kepler CPU catalog KDE now evaluates `bisquare` as the normalized quartic kernel, matching GPU KDE,
  instead of falling back to Epanechnikov. The Gaussian kernel is zero at and beyond five bandwidths.
- Pettitt catalog summaries no longer publish synthetic change or segment evidence for untested or
  constant cells; those rows are unavailable and constant p-values stay 1.
- Vecchia and Fisher-scoring catalog fits report a counted error for an unusable initial fit and a
  counted warning for a retained nonconverged fit, with the actual stop reason, instead of
  presenting a stall as an iteration limit or a failed start as a usable fit.
- Kepler streaming raster results read the observation-count loss certificate; affected mean,
  standard deviation and anomaly values become `NaN` instead of publishing incomplete statistics.
- Kepler calibration and anomaly operations resolve an omitted `calibrationSliceCount` to
  `min(12, sliceCount)`.

**Geometry**

- Kepler planar GPU geometry centres its inputs, so areas and coordinate differences at large
  projected magnitudes stay translation-stable instead of losing unit differences.
- Empty line outputs stay empty, and polygon readers no longer promote a surviving hole to the
  exterior; a polygon whose exterior is unusable is suppressed.
- Polygon triangulation tables (CPU and GPU) keep counted zero-area rows with their identities but
  publish them as line or point geometry with a `degeneracy` column. CPU triangle IDs use the
  original source part index even after empty parts. Sampling frames omit zero-area triangles.
- Kepler GeoArrow geometry readers consume every input chunk; later chunks were silently dropped.
  Source row order, empty features and nesting offsets are preserved.
- Kepler coverage validity keeps thin-overlap classification under translation and when distant
  features are added, and reports duplicate polygons with holes as overlapping.
- Kepler polygon predicate joins reject candidates whose interiors fill a target hole, and the
  bounds prefilter admits contacts within the documented boundary tolerance.
- Kepler GPU geometry validity packs rows in a local frame, so translated closure gaps survive
  float32 conversion. Unrepresentable buffer coordinates are reported as `non-finite-coordinate`
  rows rather than failing the whole validity run.
- Frame selection prefers a candidate that round-trips source coordinates exactly, so representable
  edge lengths such as 0.75 survive noding, isochrone uploads and line chunking.
- CPU circle results report the radius as the clearance at the published centre, which can be zero
  when a searched centre rounds onto a boundary.

**Cells**

- DGGS aggregation, rollup and comparison tables in Kepler publish saturated GPU sums, means and
  sum comparisons as `null`, flagged by the new `sum_overflow` column; counts, extrema, presence and
  geometry are unchanged.
- Kepler spherical containment accepts either winding for shells and holes without including the
  antipodal region, and concave S2 polygon covers keep interior cells when the vertex mean lies
  outside the shell. Zero or non-finite area spherical rings now produce the
  invalid-spherical-polygon diagnostic and no cover instead of a successful cell.
- Kepler catalog GPU indexing selects a supported precision per family, restoring Quadbin, Quadkey,
  Geohash and S2 construction and Quadbin aggregation, rollup and comparison; H3 keeps high precision.

**Trajectories**

- Kepler pooled Brownian bridge and isopleths keep numeric and text track identities distinct even
  when their strings collide; numeric and text IDs no longer create synthetic movement segments.
- Kepler home-range bindings accept a single trajectory fix and keep the requested zero-density grid.
- Kepler ordered trip fitting keeps `+180` endpoints on their branch and source rounding allowances,
  and fits dateline-crossing intervals in Mercator space.

**Raster and terrain**

- Kepler Lagrangian diagnostics internally rebase slice and integration instants, so closely spaced
  epoch times are resolved; omitted start and end times keep their previous absolute defaults.
- Kepler curated kriging separates insufficient-data and invalid-setting cells from ill-conditioned
  or negative-variance cells, and the CPU route (including fallback) reports zero-neighbour cells
  with the insufficient-kriging-neighbors diagnostic instead of dropping them.
- Kepler raster join graphs forward rasterization loss: failed rasterizations no longer publish
  certified counts, assignments, comparisons or bounds, and catalog invocation rejects the failed
  result. Catalog crossing allocation closes each ring and no longer adds cross-ring edges.
- Kepler regional raster change accepts positive completeness certificates; zonal exposure
  withholds statistics on explicit incomplete evidence, and certificate-only loss reports
  recipe-incomplete rather than recipe-overflow.

**Catalog and Kepler adapters**

- PCA and projection catalog results no longer list CPU eigensolver stages in
  `dispatchedContributors`; they remain in full provenance.
- CPU DBSCAN catalog admission accepts its supported 20,000 total points; the curated cluster
  footprint keeps its 6,000-point limit.
- Catalog parameter lists and descriptor-resolved arrays are copied per invocation, so editing one
  run no longer alters registry defaults or another caller's array.
- Geographic metric-grid distance controls display metres; planar controls display coordinate units.
- Saved ICP budgets: a valid canonical `maximumIterations` no longer conflicts with an injected
  legacy default, and editing a restored budget keeps the chosen value.
- Kepler Quantile LISA forwards the selected alternative to the GPU producer for whole and chunked
  weights, so saved legacy, `greater` and `two-sided` requests construct and keep their tail through
  inference, restoration and tooltips. Requested `tested` and `analysisCount` now reach their writers.
- Nonempty Kepler polygon weights construct again, write caller-owned `status.count`, and keep final
  capacity loss, candidate loss and invalid rows separate.
- Curated network-disruption publication waits for both routing convergence certificates and both
  accessibility completeness certificates; scoring loss no longer publishes a comparison.
- Curated drainage refuses results with canonical unresolved counts.
- Catalog status discovery: segment-snap candidate diagnostics no longer appear in scalar queries,
  the unpublished drainage `invalidCount` is removed, and status, loss and exactness metadata for
  shapes, cells, weights transforms, stop-value cubes and allocation adapters are corrected.

**Lenses and views**

- Mapped lens data keeps native missing measurements missing instead of counting them as zero;
  Arrow Int64 values outside JavaScript's safe range become `null`.
- Partial captured lines and polygons keep their original feature keys for values, picking and
  linked selection, including nonzero capture origins.
- Catchment unknown facility counts display as unknown and no longer inflate extrema; comparison
  deltas for those rows are `NaN`, and known change and equality counts exclude them.
- Raster lens comparisons check full sample placement and frame or CRS identity and refuse to
  subtract across mismatched footprints; rotated or nonsquare rasters display in pixel space.
- D8 stencil views report correct ground distances and gradients for rectangular and rotated
  affine grids.
- Raster view measurement levels resolve per operator and port, so nominal FillHoles values no
  longer inherit another operator's numeric scale.
- Line and polygon playback keeps frames whose feature keys, vertices or values change.
- Duration-only GPU measurements, including measured zero, count as execution evidence.
- Lens capture archives use the shared analysis codec, preserving non-finite floats and signed zero.
- Historical workflow runs no longer acquire today's default dependency revision on conversion.
- Inferred point-count captures obey logical capacity and graph identity; renamed raster-join
  captures no longer bypass completeness checks.

**Examples**

- Network, weights, hot-spot, storm, wildfire, cell, LiDAR, regression and trade-area examples and
  showcases now withhold summaries, maps and comparisons while their producer reports candidate,
  capacity, rasterization, saturation or convergence loss, and restore them after a complete run.
- The space-time scan showcase uses the submitted frame's float32 cutoff, Mantel and Knox readouts
  count only usable draws, rate tooltips exclude negative or non-finite events, and quartic kernel
  charts draw the bisquare profile.
- The facility cost-matrix example keeps zero-cost access at source intersections; the aviation
  example publishes community quotients as final only after Louvain converges; earthquake readers
  select the correct cell at float32-sensitive interior edges; satellite clocks show the quantized
  playhead they upload.
- The K-means example requests `centerCoordinates: true`, which changes its graph and can change
  its numbers.

## Input now rejected at construction

**Shared core and contracts**

- Compact outputs that omit the required count or overflow status words.
- Numeric validators reject coercible non-number values such as numeric strings; convert them first.
- GPU operations reject empty logical port keys; CPU row helpers reject vectors whose typed-array
  class disagrees with their format (foreign-realm arrays of the right class remain accepted);
  inconsistent output adapters throw instead of returning a wrongly typed view.
- Fixed-point scale, clear and decode helpers reject strided inputs; decode rejects unknown
  `scaleMode` values and too few scale words.
- The fitting coordinate encoder rejects misspelled, undeclared or branch-inapplicable top-level
  options; generated WGSL builders reject colliding names before declaring resources.
- Sorted segment reductions reject non-packed keys or outputs and contributions shorter than the key
  count (previously zero-filled); `createSegmentSumNode` rejects non-packed direct views, a
  non-integer or negative `segmentCount`, and requested sums or weight sums without contributions.
- Shared status validation accepts a status scalar or stage code only as exactly one uint32 row.
- Regression result builders reject unknown regularization kinds and invalid ridge penalties;
  panel results require an explicit total `k` and GTWR results an explicit `tr(S)`.
- Spatial frames reject known angular frames with native planar metrics and metre declarations that
  contradict known frame units; unknown `purpose`, `anchor`, `precision`, `storage`, units and
  local-frame kinds list the accepted values; CRS origins need two finite elements.
- Spatial context descriptors that are `false`, zero, an empty string or another non-object are
  rejected; nullish primary input keeps the native-planar default. Packed raster coordinate
  determinants must be finite nonzero float32.
- Local-frame array conversions reject partial coordinate rows; compact polygon and trajectory
  tables reject malformed offset hierarchies (zero-parent tables are now accepted).
- Writable alias checks reject overlapping writes to one physical buffer reached through mixed ESM
  and CommonJS wrappers.
- CPU routing rejects an invalid `costPrecision`; direct
  `computeNetworkRoutingPredecessorsWithStatus` calls reject malformed singleton CSR such as `[2]`.

**Statistics and inference**

- CPU spacing functions reject unknown `coordinateModel` strings (accepted: `planar`, `geographic`).
- Attainability and correction-bearing summaries reject unknown correction strings (accepted:
  `none`, `bonferroni`, `holm`, `sidak`, `false-discovery-rate`) and unknown `pValueConstruction`.
- `computeWeightsAlgebra` rejects unknown operations and weight rules.
- Kepler CPU catalog KDE rejects unknown kernel names instead of using Epanechnikov.
- Kepler Quantile LISA rejects contradictory alternative spellings and an explicit `null`
  alternative, and rejects unsupported polygon status destinations and conflicting weights and
  topology status bindings before declaring the graph.
- `suggestKrigingVariogram` refuses empirical weighted-least-squares fitting with
  `longitudeSearch: 'wrapped'`; supply a manual variogram instead.

**Kepler catalog and adapters**

- Convex clip rejects holed, multipart, crossed or invalid clip operands instead of clipping against
  the first shell; subject holes remain supported. Sampling rejects a frame with no positive-area
  triangle.
- Catalog admission rejects zero or non-finite FTLE steps and ten-step durations that round to zero
  in float32, hex, binary or octal numeric text, boolean or array aliases, ambiguous empty entries in
  numeric lists, and conflicting ICP `iterations` and `maximumIterations`.
- Native table reads reject non-scalar cells.
- Kepler buffers require raw source XY to be finite float32 before projection, checked before
  tolerance or workspace construction on both dissolve routes.
- Kepler GeoArrow preparation rejects Float64 values that lose precision under the default
  reject-loss policy, dimensions other than XY, int64-offset columns on the installed Arrow runtime,
  reader-family mismatches, spherical edges and conflicting edge declarations.
- OLS imports require an integer predictor count from 1 to 15.
- `RasterTimeSeriesSettings` rejects non-boolean or conflicting `zeroAtom`/`zeroInflated` values;
  rolling means reject a fractional, negative or larger-than-`sliceCount` `activeSliceCount`; a
  streaming certificate readback that is not exactly `cellCount * slotCount` words refuses the result.
- Fleet dwell and track quality reject numeric and text track IDs with the same spelling before GPU
  allocation; pooled Brownian bridge and isopleths refuse such ambiguity at publication.
- Curated network-disruption publication requires exactly one named certificate per routing and
  scoring stage and refuses conflicting or repeated stage records.
- Catalog output-description calls reject invalid modes, missing or out-of-range sizes and
  impossible uint32 allocation products; task calls reject unknown question IDs.

**Lenses and examples**

- Saving a lens capture rejects non-integer or out-of-range u32 and i32 lanes, partial components
  and invalid capture metadata; loading lens files validates full capture and metric sidecars,
  including unique capture IDs.
- Mapped lens runs reject mismatched column lengths before loading, and refuse the result when a
  custom loader ignores `LensScenarioRunOptions.datasetResolver`; forward the resolver explicitly.
- The Decimal128 and wide-numeric example runners now inherit the public library checks (row
  limit `WIDE_NUMERIC_MAXIMUM_ROW_COUNT`, packed word-aligned chunks, `GPU_MAXIMUM_COMMAND_NODE_COUNT`,
  positive `resultWordCount`, `wordsPerRow` of 2 or 4, device binding limits).
- The network cost-weights example rejects a matrix horizon without its completeness input; the
  showcase raster-join mismatch kernel rejects position offsets or strides not divisible by four.

## Added

**Shared core and contracts**

- Readback conversion of bytes to bounded little-endian uint32 or float32 words, logical byte-range
  packed readers, view predicates, named port descriptors and preflight, a structured validation
  error, and an option validator; `validateOperationGraphNames` for CPU operation authors.
- Typed `getOutput` and checked view getter overloads on contributor operations.
- `planGraphChunkGroups` accepts explicit validation identities; a runtime-independent
  `analysis-validation-error` leaf exports the option and requirement helpers.
- Optional unresolved-work and completeness scalars on shared result and stage status; a noded
  topology handoff that keeps the live node count and build diagnostics.
- Optional regression ridge-penalty and inference metadata, GTWR and panel-error result kinds,
  analytic and Quantile LISA alternative types, and optional p-value construction metadata.
- `costPrecision` on shared CPU routing (`'float64'` default, `'float32'` per edge); empty `[0]` CSR
  is accepted by host routing.
- An optional metric-results purpose for spatial context compatibility, and CRS frames that carry
  a relative storage origin.
- Testing overloads `getPrivateMemoryBytes(recordedKernel)` and `getWorkgroupMemoryBytes(recordedKernel)`.
- Core exports for paired Arrow key preparation, `prepareArrowMeasureColumnsForGPUAnalytics`,
  `prepareArrowBatchesForGPUAnalytics`, `PreparedArrowAnalyticsMeasureColumn` and the Arrow
  analytics scalar, category and dictionary-byte limits; `GPUSpatialContextCompatibilityOptions`.

**Spatial entry**

- Model-specific regression status decoders (24) and `GPUDecodedRegressionStatus`, the GTWR summary
  reader, the panel-error reader, the joined-table training-mask adapter, the ordered-predictor
  design adapter, and OLS constructor input and model result types (also on the regression entry).
- The detailed spacing readback decoder and CPU acceptance function, with `invalidCount` on CPU
  spacing results.
- Spatial and space-time Vecchia likelihood and Fisher-scoring readers, the prepared neighbourhood
  uploader and its limits.
- `decodeGPULocalKrigingStatus`, `decodeGPULocalRBFStatus`, the conformal status decoder and
  input-incomplete bit, and both local-kriging brute-force limits.
- Permutation inference metadata decoders and types, the 1,048,576-draw permutation ceiling, the
  numeric-seed to two-word Philox converter, and pair-block constants.
- Weights producer loss, requirement observation and validation option types; the island-attachment
  planner, routes and planner object; separate automatic-crossover and forced-route maxima constants.
- `GPUMultivariateLocalGearyPermutationTestOutput`, expectation-based scan shape, device and limit
  contracts, `GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT` and
  `MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS`.
- Named CPU results for multiple testing, lag impacts and weights validation, and the spatial-scan
  output contract.
- Counts-and-assignments-only CPU point-in-polygon execution, checked native Arrow geometry
  preparation, bounded geometry readback, a completeness-checked HEALPix DGGS-JSON exporter,
  prepared-index allocation planning, qualified spatial adjacency CSR types and decoder, CPU
  Brownian results and tail-policy type.
- The predicate-join planner, route descriptors and `MAXIMUM_SPATIAL_RELATE_PATTERN_SLOT_COUNT`;
  `decodeGPUBoundedFacilityAllocationStatus`, `GPUBoundedFacilityAllocationOutput`,
  `GPUBufferSelectionMatches`, `MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION` (a per-invocation
  bound, not a total edge limit), `GPU_RANK_MARKOV_ROUTES`, `getGPURankMarkovExecutionPlan` and
  `GPU_RANK_MARKOV_PLANNER`.
- Polygon-row, map-coloring-priority and raster-affine types.

**Dataframe, raster, terrain, point-cloud and network entries**

- Dataframe: the explicit cell-period key layout with one-period identity keys, checked cell-period
  unpacking, signed fixed-point group-sum decoding, the variogram diagnostics fitter and result,
  space-time variogram helpers and types, neutral CPU quantile packers for more than 64 outputs,
  class-break planner limits and props, and point-pattern `indexCellSizeHint` settings.
- Raster and terrain: CPU band, composite, catchment and patch-status types,
  `GPUTerrainHorizonUnorm16Input`, origin-aware LIC, streamline, particle-advection and flow-map
  settings, and the cost-path reference-cell and solar-irradiance sample constants.
- Point cloud: `GPUPointCloudCoverageValidity` and `getRasterMetadataFromPointCloudGrid`.
- Network: the K-function maximum event count (92682), checked readback input types, labelled matrix
  adapters, the path reader with its maximum row constant and `NetworkPathExtractionCompactOutput`
  (optional `status.unrepresentable`), alternative-route types, CPU result types, isochrone outputs,
  and directed and noded GPU cost-matrix props with their shared status type.

**Catalog**

- Optional `GPUAnalysisOperator.forms` with package-qualified CPU and GPU identities, invocation
  contracts, recipe stages and planner locators; route-scoped limits; qualified task discovery
  whose owner preserves `GPUAnalysisPackage`.
- Discovery of the stop-value cube recipe, point-cloud coverage validity, the network-allocation
  adapter and two tessellation recipe identities.
- Optional `pValueConstruction` on attainability, summary and caution helpers.

**Kepler and lenses**

- Kepler: `sum_overflow` on DGGS tables, optional `count` on `KeplerPolygonWeightsStatus`, optional
  stop reason, history and likelihood on `KeplerVariogramFit` and Fisher results, optional
  `longitudeSearch: 'wrapped'` on IDW and kriging settings, `streamingCountStatus` and
  `columns.streamingStatus`, a CPU `degeneracy` column, geographic conformal `input_incomplete_rows`
  and `neighbors_truncated_rows`, `testedLocations` on Pettitt summaries, `empty`, `gridWidth` and
  `gridHeight` on home-range summaries, and drainage `unresolvedCount` and `recipeStatus`.
- Kepler catalog and workflows: execution-stage metadata, unit metadata, `acceptsTable` inputs,
  `kindFrom` outputs, `getWorkflowOutputKind`, a `source-table` host source, qualified discovery,
  and the `deferParameterValidation` option on `parseCatalogInvocation` for restoration only.
- Lenses: `LensScenarioRunOptions.datasetResolver`, `LensDatasetResolver`,
  `LensScenarioInputBinding`, raster lens `frameId` and `crs`, `CompareViewModel.unknownCount`,
  `LensControllerOptions.element`, `WorkflowRunRecord.dependencyRevision` and
  `WorkflowRunnerOptions.dependencyRevision`, `TimingViewModel.durations`, `D8StencilModel.affine`,
  `BufferStripModel.stats.unknownCount`, and the
  `gpu-spatial-analysis.local-building-tessellation-recipe` scenario.
- `GPUComputationNodeMetrics.gpuTimeMs` and `GPUComputationNode.declaredBindings` in recordings.

## Tooling and gates

- `yarn build` detects normal and backdated edits to linked declarations, conditional declaration
  branches, type-affecting JSON and source inputs and compiler or transformer executables, and
  refuses native and transformed output collisions before writing. Older build records require a
  rebuild; JSDoc in TypeScript sources and declarations no longer creates false freshness inputs.
- Build records survive stage-to-live promotion. `yarn build:release --promote` verifies installed
  build records and native entry points, rejects workspace links that load from another checkout,
  and rolls back outputs, compiler caches and the release record on failure.
- `yarn verify-package-exports` exits 1 when a native JavaScript entry or a CommonJS target resolves
  to a different physical file (checked before that entry's ESM module loads), honours Node
  condition ordering, and accepts legal in-package filenames that begin with two dots.
- `yarn check-packed-consumer` authenticates the exact archive bytes it extracts (printing SHA-256
  and byte length with `--keep-artifact`), fails if the archive changes after extraction, resolves
  relative `reference types` and `reference path` directives under NodeNext, requires type-bearing
  module targets, and requires all eleven public analysis entries, rejecting `any`-degraded types
  and ESM/CommonJS chunk, recording, lowering and zonal declaration failures in either direction.
- `yarn verify` reconciles fresh one-worker Node reports with the selected files and writes durable
  full, partial or failed qualification receipts; release stages bind a matching full receipt to the
  retained archive bytes, otherwise they are labelled build and package evidence.
- Typecheck freshness follows compiler-selected JSX, JS and TS dependencies, explicit and implicit
  type-reference modes, extensionless declarations and synthetic tslib imports; prior watcher or
  build evidence may need refreshing.
- Production TypeScript aliases match the eleven supported analysis entries; `/testing` and
  `/test/*` aliases come only from test configurations, and lenses and Kepler test runners resolve
  them through the shared test alias map.
- `yarn check-contributor-contract` inspects every manifest entry, traces declarations through
  default, namespace and indexed-access routes, resolves imported `assert` and luma.gl bindings, and
  applies to `.js`, `.jsx`, `.mjs`, `.cjs`, `.mts` and `.cts` sources; inline serial-loop allowances
  must state a numeric or named bound. Catalog registration checks require participation in the
  public catalog. Importing either checker no longer runs it.
- Layering checks discover parenthesized literal dynamic imports; vocabulary checks cover point-cloud
  APIs and computation records; documentation checks validate rendered heading fragments, reject
  unknown API bindings and unpublished package paths, and flag discarded allocating packers
  (diagnostic 90004).
- Kernel audits reject configurations that produce no shader module; validation audits detect graph
  or resource mutation and the wrong rejection phase.
- Three numerical fixture generators accept `--check-reference`; the clustering regeneration test
  needs `ORACLE_CLUSTERING_PYTHON` and is skipped otherwise.
- Analysis no longer declares `probe.gl` directly; it comes through `@luma.gl/core`.
- Private lenses have a post-build qualification command requiring current output hashes; it does
  not authorize publishing them.

## Deprecated spellings

- Quantile LISA `inference` setting: use the alternative setting (`geoda-legacy`, `greater`,
  `two-sided`).
- `zeroInflated`: use `zeroAtom` in `RasterTimeSeriesSettings`.
- Weights algebra `symmetricDifference`: use `symmetric-difference`.
- Nearest-feature `lowest-id`: use `lowest-row`; saved settings select the same rows.
- Drainage `uncertainCount`: use `unresolvedCount`.
- `requiredTiedMinimumCount`: use `sufficientMinimumPValueTieCount`.
- ICP `iterations` remains accepted beside `maximumIterations`; supplying both with different
  values is refused.
- Showcase `zscore` standardization: use `z-score`.
- Legacy island-attachment crossover constant aliases keep their values beside the separate
  automatic-crossover and forced-route constants; superseded catalog route names carry deprecation notes.
- The synchronous recorder spelling remains documented as deprecated; use the explicit scope API.
- The old network CSR props alias remains beside the qualified visibility props type, and the old
  spatial point-reader import path still works now that the readers live in analysis core.

## Documentation and tests

- The API landing page has four complete GPU first-run examples (terrain, distance, time window,
  point cloud) with a shared device-to-readback lifecycle and a borrowed Deck device route.
- The first-run terrain example uses an east-rising DEM (slope 45 degrees, shade about 1) and the
  time example uses epoch-relative milliseconds.
- The catalog guide explains published versus forwarded status, inherited cell and weights loss,
  per-line diagnostics and that compact matches may omit `requiredCount`.
- Point-in-polygon join docs separate aggregate, intermediate and final capacity loss.
- Rhumb navigation docs use `spatialContext` and state metric and units; line substrings route to
  `GPULineChunk`; NDVI and drive-time imports are corrected.
- Natural-break limits are stated as inclusive source-row limits per exact or monotone route.
- Raster streaming limits describe count saturation and sticky loss evidence, not wraparound.
- Fractional categorical capacity limits include each zone's missing-category slot.
- Arrow preparation docs cover Float16 and LargeUtf8 routes, nullable time and validity handoffs,
  native Polygon and MultiPolygon preparation, and renaming `__proto__` fields before upload.
- Canonical equality docs distinguish per-coordinate and Euclidean tolerance.
- The computation-record guide shows rebuilding a saved run, checkpointing, discarding abandoned
  snapshot suffixes and scoped asynchronous recording.
- Recording guidance documents `gpuTimeMs`, including measured zero and unavailable measurements.
- Point-cloud guidance shows coverage-to-validity conversion and the aligned terrain handoff that
  keeps zero-elevation ground, and names the eleven operations Kepler binds.
- The catalog warns that packed FTLE steps can round to zero even when a ten-step duration is
  representable.
- Grid guidance reports the sampled 6.6% S2 mismatch at level 20 and none observed through level 10.
- The release guide records alpha.16's integer raster calibration precision defect, its host-origin
  workaround and the adoption gate.
- The upstream capability register separates partial PCG and search adoption from proven
  replacements and requires full verification on upgrades even with zero hash differences.
- Agent instructions describe the 8192 MiB heap ceiling and CLI override, release-archive SHA-256
  records and the declaration forms the packed-consumer check reads.
- The regression serial-loop inventory is corrected for GTWR, MGTWR, panel and GWR summaries.
- Reference fixtures for Mantel now encode undefined correlations as JSON `null` under both
  `observedCorrelation` and `observed`; consumers of the fixtures must handle `null`.
