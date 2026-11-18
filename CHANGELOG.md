# Changelog

## Unreleased (0.1.0-alpha.0)

First standalone version, split on 2026-11-07 from the luma.gl `map-contributors` branch.

- Analysis workflows in `@luma-contributors/kepler` (`modules/kepler/src/workflows/`): a serialisable
  step graph over the catalog operations with a planner, a runner whose step status is derived from
  content keys, table glue operations, templates, JSON and TypeScript export and assistant tools. The
  Kepler example gains an analysis workbench (node canvas, palette, inspector, side-panel tab) built
  on it; the example now depends on `@xyflow/react`.
- New packages: `@luma-contributors/analysis` (unreleased alpha), `@luma-contributors/lenses` and
  `@luma-contributors/kepler` (private until dataset licensing is resolved).
- Uses published luma.gl through public entry points only. The integration target, accepted peer
  range and unpublished alpha contract are recorded separately in [release-package.md](dev-docs/release-package.md).
  No luma.gl code is copied or forked. See `dev-docs/upstream-first-plan.md`.
- Analysis requires the `apache-arrow >=17.0.0` peer and ships licence texts plus a source notice
  inventory under `MIT AND Apache-2.0 AND BSD-3-Clause`. An offline packed-consumer checker verifies
  ESM/CommonJS exports, declaration targets, README links, notices and the documented CPU computation.
- Import paths moved from `@luma.gl/graph/*` analysis subpaths to `@luma-contributors/analysis/*`.
  `gpu-raster` analysis became `raster-analysis`, `gpu-dataframe` analysis became
  `dataframe-analysis`, `gpu-pointcloud` analysis became `pointcloud-analysis` and `geospatial` became
  `geospatial-analysis`. Base primitives such as `GPUSort`, `GPUScan`, `GPUGridIndex` and
  `GPUCrossfilter` now come from `@luma.gl/graph`.
- Capabilities dropped or regressed by depending on published luma.gl are listed in
  `dev-docs/upstream-workarounds.md`.
- Restored sparse resistance connectivity (`addSparseResistanceConnectivityRecipe`, EC03) on public
  luma.gl APIs: up to 1,048,576 vertices and 2,097,152 edges, Jacobi-preconditioned,
  Circuitscape-validated; adds node current maps.
- `yarn typecheck-tests` (`scripts/typecheck-tests.mjs`) partitions source/tests, example-backed specs
  and website-backed specs into separate TypeScript runs. In `test/kepler-compute-example-specs.json`,
  `excludedFromRuns` controls runtime exclusions, while `excludedFromTypecheck` controls example-backed
  typecheck exclusions; membership and reasons are explicit per file. These sets need not coincide.
  [release-package.md](dev-docs/release-package.md) gives a command that derives current counts directly
  from the manifest. This release note does not claim that the wide checks have passed.
- CI (`.github/workflows/test.yml`) also runs `yarn check-upstream-first`, `yarn typecheck-tests`, the
  showcase and explorer typechecks, and a `website` job (typecheck, catalog check, build).
- Kernel-audit cases for the 12 point-cloud and datum-grid contributors
  (`kernel-audit/cases-pointcloud-and-datum.ts`); only the four chunked spatial-weights contributors are
  still skipped.
- `GPUNetworkCentrality` exposes `estimatedCommandNodeCount` and its constructor rejects configurations
  above 32,768 command nodes instead of building a very large unrolled graph.
- Internal: the masked weak-components kernel moved from `gpu-network` to
  `analysis-core/kernels/masked-weak-components.ts`, shared by network statistics, network analytics
  columns and sparse resistance connectivity. No public API change.
- Math correctness review (2026-11-10 and 2026-11-11): fixes in the regression family, coverage
  validity, ring assembly, polygon triangulation, the Diggle test, the COLD monitor and others, each
  pinned against a reference library. Every number a caller can see change is listed in
  `dev-docs/research/math-correctness-review-2026-11-10/behaviour-changes.md`; the limits left are in
  [the numerical limits](dev-docs/roadmap.md#known-limits-recorded-by-finished-passes).
- Breaking, from the [contributor contract alignment](dev-docs/contributor-contract.md): the deprecated
  aliases are removed; `parameters` is required on walking cost and fractional coverage; there is one
  variable-length geometry output layout; the raster family follows the `mask` rule (about 40 classes
  renamed props, results unchanged); point-cloud rasters default to `'south'` rows; the k-means
  contributor is `GPUSpatialKMeans`.
- CPU twins and mixed CPU/GPU graphs: 125 contributors have an operation form
  (`GPU<Name>Operation`) for a published luma.gl `GPUGraph`, a float64 CPU twin
  (`CPU<Name>Operation`) over the same named values, and a `compute<Name>OnCPU` function usable
  without a graph. `GPUContributorOperation` (`analysis-core`) lowers a contributor into a `GPUGraph`.
  Rules are in section 14 of `dev-docs/contributor-contract.md`.
- luma.gl usage review ([2026-11-11 and 2026-11-12 evidence](dev-docs/research/luma-usage-review-2026-11-11/README.md)): 21 hand-written
  gather kernels, a flag total, a prefix sum and a high/low split now call the published
  `GPUUint32Gather`, `GPUGather`, `GPUFlagOffsets`, `GPUScan` and `fp64ify`, with equal results.
  `createWGSLKernelNode` takes `modules` and `defines` and assembles them with the published
  `WGSLShaderAssembler` (fp64 modules get the integer variant). 22 raster and terrain contributors
  declare the published `GPURasterHaloStage` contract.
- Changed values, from that review: `requiredHalo` is 2 for `GPUIsobands` when edges are produced
  (0 for classes only), at least 2 for `GPUTerrainDerivatives`, `GPUReliefShading` and the 3x3
  methods of `GPUTerrainCurvature` in the one-sided border mode, and the scaled support for
  `GPUTextureShading` with decimated levels. The former values did not cover what the kernels read.
  `GPUCubeLayoutTranspose` copies raw words, so NaN payloads survive.
- BL-PASS migrations: `GPUNetworkFacilityOptimization` and `GPUNetworkRationalAccess` move the
  packed `output.status` array to `output.summary`; named diagnostic `output.status` ports are
  separate. `GPUGroupConvexHull` and `GPUMinimumBounds` publish boolean `output.status.overflow`.
  Hull reason flags are `groupOverflow` and `vertexOverflow`; minimum bounds has `groupOverflow`.
- `GPURasterReprojection` requires `destroy()` after compiled-graph teardown, including same-CRS
  use. Raster patch metrics parameter buffers contain eight words. `GPUMarkedPointStatistics`
  uses the grid index by default; `searchStructure: 'brute-force'` keeps the reference route.
- SC/IMP scalability: chunk-capable contributors accept `GraphVectorView` inputs with packed
  defaults. `getAnalysisRequiredLimits` and `getAnalysisMaximumRowCount` expose adapter-aware
  sizing. `GPUNetworkPointRouting` defaults to `mode: 'scalable'`; `GPUNetworkFlow` selects its
  parallel solver above reference bounds or when a topology is supplied. `GPUNetworkStructure`
  defaults to `mode: 'auto'`. Reference modes retain their documented behavior and tie order.
- `createNetworkCentralityContinuation` runs bounded host-controlled batches with persistent
  state, allowing runs beyond `GPUNetworkCentrality`'s direct 1,024-round limit. Streaming embedding
  contributors support up to 1,024 dimensions; reference and covariance routes retain separate
  bounds. Fixed-linkage `GPURasterRegionGraph` supports Boruvka merging; D8 `GPUTerrainTiledFlow`
  uses bounded tile scratch and a boundary forest. S2 compaction, center covering and same-level
  outlines are available.
- `addVisibilityGraphRecipe` from `@luma-contributors/analysis/gpu-network` publishes CSR centrality
  metrics; its `gpu-spatial-analysis` counterpart returns CSR adjacency and degree only.
  `GPUGeographicallyWeightedRegression` reports automatic Gaussian-support truncation through
  `output.status.approximation`.
- Analysis experience (AX, `dev-docs/roadmap.md`):
  `@luma-contributors/analysis/gpu-spatial-analysis` exports a host-neutral vocabulary (twelve
  questions, data kinds, measurement levels, field roles, requirements settled as passed, failed or
  unknown, spacing statistics, data-derived default rules, class schemes, result sentences and
  cautions; `docs/api/gpu-spatial-analysis/analysis-experience.md`).
  `@luma-contributors/kepler` adds a dataset profile, hand-written descriptors for all 217 catalog
  bindings, applicability and suggestions, resolved defaults, result presentation and reports.
  All 28 curated analyses have registered bindings and synthetic samples: 16 have CPU paths and
  12 require WebGPU. Host forms and the workbench consume AX roles, defaults, applicability,
  templates and presentation. Browser journeys and latency gates remain open as recorded in the
  roadmap; registration does not qualify them. Existing binding defects and their status are in
  `dev-docs/research/analysis-workflow-ux-2026-11-14/binding-defects.md`.
- Website (Gridlight): 532 catalog entries, each with a live picture, and thirteen how-it-works
  pages. It can be built for a sub-path (`SITE_BASE_PATH`) and `yarn website-deploy` publishes it to
  GitHub Pages.
- Documentation moved from `docs/api-reference/experimental` to `docs/api`. Release notes from the
  luma.gl era are in git history (`dev-docs/archive/luma-era-release-notes.md` at `d9d56f0f`).

### Breaking changes (library review 2026-11-17)

- **Core contracts and status ports**
  - Compact outputs (`GPUCompactOutput`, `createPublishNode`, pair, region, flow, zone, encounter,
    sampled-point and network-path outputs): `status.overflow` now means final output capacity only.
    Loss before the final stage (BVH leaves, scratch, per-track or candidate limits) is reported in
    the new optional `status.candidateOverflow`; `requiredCount` counts accepted final-stage rows
    and is a lower bound after candidate loss. `PublishNodeProps` and `GPUPairGatherProps` gain
    `candidateOverflowSources`. A destination you request is rewritten on every submission.
  - Where a producer can lose candidates, `output.status.candidateOverflow` is now required (cell
    comparison and rollup, `GPUIsobandRings` `edgeOverflow`, volume outputs of map matching, which
    replaces `output.truncated`, and nested consumers of the two compact-pair producers).
  - Weights ports: `weights.status.candidateOverflow` is now read by local and global Moran,
    hot-spot, local Geary, Ward, SKATER, the minimum spanning forest, REDCAP, panel and diagnostic
    operations. A nonzero value suppresses
    statistics, p-values and labels (`NaN`, `0xffffffff` or zero as documented) exactly as
    `overflow` does. `GPUSpatialAutocorrelationStatusPort` and `GPUGlobalSpatialStatisticsOutput`
    gain `candidateOverflow` and `status`.
  - `GPUNearestFeatureWeights` gains `neighborStatus` and publishes upstream loss on
    `output.status.candidateOverflow`; its own `overflow` is final CSR capacity only.
  - Peak and streamline constants: `GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED` and
    `GPU_STREAMLINES_OVERFLOW_UNCONVERGED` removed; use `GPU_PROFILE_PEAKS_CANDIDATE_UNCONVERGED`
    and `GPU_STREAMLINES_CANDIDATE_UNCONVERGED` (value 1).
  - CPU twins now honour explicit iteration, search and capacity budgets instead of ignoring them
    (network reachability, K-shortest paths, flow, cost table, point routing, structure, MGWR, panel
    lag, variogram fit, Ward, AZP, scan location-allocation, raster reconstruction and hole fill,
    offset-curve components, staypoints, track similarity). Exhausting an explicit budget publishes
    the incumbent with `converged = 0` or `iterationLimitReached = 1`. Omitted budgets keep their
    previous behaviour except where a default is documented (for example `maximumPathLength`,
    which now binds with default 1024).
  - `getGPUAnalysisOperator(identity)` takes a string or `GPUAnalysisOperatorIdentity`
    (`{package, exportName}`); an ambiguous bare name now throws. `findGPUAnalysisOperatorPaths`
    and `getGPUAnalysisNextOperators` return qualified identities. Descriptor types are readonly
    and catalog records are frozen. `GPUAnalysisOperator.status` may be `'unknown'`, and
    `GPUAnalysisTier` and `GPUSpatialAnalysisReadback` gain `'unknown'`.
  - Construction snapshots: props and port maps of the core producers, triangulation, coregionalization
    fit, panel operations and many contributors are captured at construction; later mutation of the
    caller's object no longer changes bindings, shapes or schedules. Prepared handles use
    `Parameters = unknown` in place of `any`.
  - Removed public switches, now internal: `extraCounts` (cell aggregation and rollup),
    `forceSequentialRepeatRemoval`, `batchReplicates` (scan statistics), `sampleIndexChunkQueryCount`,
    `fittedScale` and public `getCommandNodes` options of generalized GWR, `rowStart` of spatial
    lag, `separateStatusDispatches` of `GPUNetworkCentrality`; `validateBindings` of centrality and
    cell aggregation is private.
  - `GPUForestInferenceProps`: `features` and `rasterStack` are mutually exclusive source arms;
    `featureLayout` belongs to the `features` arm.
  - `GPUParameterBuffer` initial values and `write` require the array kind that matches the format
    (`GPUParameterValues<Format>`). Spatial parameter schemas reject values that overflow float32.
    `createFillNode` fills every scalar component of a vector view by default.
  - `sliceGraphChunkGroupView` rejects ranges outside the parent view and strided parents.
  - `getGPUSpatialFrameContextUnits` returns `undefined` for kilometre frames (the unit stays in
    `frame.unit`).
  - `GPUComputationResource.ownership` gains `'unknown'` (the default for unannotated resources)
    and `bytesKind`. `getGPUComputationRunId` hashes more inputs, so default run IDs change.
    Records, metrics and captures carry an `executionId` that must match.
  - `makeDecimal128WordVectors` takes `Vector<Decimal>` and returns `{words, validity?, destroy()}`;
    `unusedValidity` is gone. Decimal preparation under exact or zero-budget policies now rejects any
    loss of the original value.
  - `getTimestampCalendarMilliseconds` and the calendar kernels floor negative sub-millisecond
    instants; `getPointCloudLocalOrigin` ignores non-finite points and rejects partial triples.
  - `GPU_MINIMUM_SPANNING_TREE_STATUS.length` 1 → 2: word 1 is `candidateOverflow` (either upstream
    loss cause, 0 or 1); `invalidCount` stays at word 0. MST `output.status` and SKATER `treeStatus`
    need two words.
  - `SkaterRegionsOperationProps.weights.status` and `CPUSkaterRegionsInput.status` accept optional
    `overflow` and `candidateOverflow`; a nonzero flag, or a lost forest word, suppresses labels
    (`0xffffffff`), count (0), cut IDs and gains (`NaN`). REDCAP and the minimum spanning forest accept
    weights that carry status ports.
  - Recipes: every `add...Recipe` function in the spatial and network recipe indexes prepares its
    whole declaration on a separate graph first; a rejected configuration, alias, device-limit or
    contributor-expansion failure leaves the caller graph and its resources unchanged. Consumed
    configuration records returned as metadata are captured copies, not the caller's object. Parameter
    uploads happen after all stages schedule. Validation order can change (glue and device-limit errors
    may precede contributor expansion).
  - Rate-cluster recipe rejects palettes shorter than `GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH` (5), short
    logical colour or quadrant views and aliased independent outputs; the raster zonal recipe rejects
    aliases against its raw inputs.
  - `RateClusterReadback` (kepler curated rate-cluster binding) requires `weightsCandidateOverflow`; a
    failed current readback no longer leaves the previous values.
  - Recipe results list their glue stages in `contributors` as ordered adapter entries next to the
    analysis contributors (drive-time catchment: 5 contributors and 10 adapters; track quality: three raw
    glue stages `reset-flags`, `quality`, `flag-counts`). Node IDs and order are unchanged.

- **Spatial joins and geometry**
  - Identity columns that hold feature rows are renamed `Ids` → `Rows`: `GPUNearestFeatureWeightsProps.neighborIds`
    → `neighborRows`; `GPUBufferSelectionOutput.nearestFeatureIds` → `nearestFeatureRows`;
    `GPUPairGatherNeighbors.ids` → `rows`; compact-pair `leftIds`/`rightIds` → `leftRows`/`rightRows`;
    `GPUTimeWindowFilterOutput.ids` → `rows` (required, defines capacity; `ids` is optional mapped
    identity); `pixelIds` → `pixelRows` and `denseZoneIds` → `denseZoneRows` in raster embedding and
    zone dictionary. The old spellings are rejected at construction. `GPUNearestFeatureJoinOutput`
    keeps `neighborIds` and gains optional `neighborRows`.
  - Point and line `featureOffsets` (multipart features) are accepted by `GPUSpatialPredicateJoin`
    (`intersects`, `within`, `coveredBy`, engine `auto` or `fast`, no `relate` output),
    `GPUNearestFeatureJoin` (neighbors mode), `GPUSpatialJoinCandidates` and prepared joins; other
    predicates, `engine: 'relate'`, nearest weights, nearest selection, point-in-polygon, buffer
    selection and `GPUGeometryPredicates` still throw. Rows are feature rows.
  - Population-support alignment: `validity` (output, operation prop, CPU result, recipe output
    override) → `alignmentCode` (0 matched, 1 missing, 2 ambiguous) plus `matchedMask` (1 only for a
    unique match). Legacy `output.validity` overrides throw a migration error. The recipe `status`
    is now an object with named stages; the packed words are at `result.packedStatus`.
  - Polygonize, make-valid and precision reduction: `status.requiredCount` (ring requirement) →
    `status.requiredRingCount`.
  - `GPUGeometryValidityOutput.mask` → `issueCode` (optional `validity` is nonzero when usable);
    `decodeGPUGeometryValidityIssueCode().structurallyValid` → `hasStructuralIssue`. Segment snap
    `output.validity` diagnostic bits → `output.issueCode`; `validity` now means usable.
  - `GPULonLatPolygonPreparationProps.spatialContext` is required.
  - Offset curves: endpoint coincidence requires equal coordinates (the distance floor is gone);
    `CPUOffsetCurveComponentsOptions` gains `segmentCapacity` and `intersectionCapacity`.
  - Isovists: `rayCount` validates the base count; provenance gains `baseRayCount`.
  - `getGPUCommandNodePipelineKeys` always includes layout, entry point and constants.
  - `GPUGeometryPredicates` `geometry` and `other` are typed `GPUGeometryPredicatesGeometry` (one
    point or line, or a polygon hierarchy) and refuse `featureOffsets` multipart points and lines.
  - `GPUAffineTransform` rejects `featureOffsets` without `lineOffsets` when `geometryType` is
    omitted (the default is lines). `validateGPUFeatureGeometryPort` honours point and line
    `featureOffsets`, validates their format and terminal entry and requires `sourceIds` to have one
    row per feature.
  - Segment snap: a call that can suppress a grid query needs at least one per-line completeness
    port.
  - Nearest-feature ranking uses squared-distance ranks before float32 publication; a row that
    publishes the same rounded distance as another can rank differently.

- **Spatial statistics and regression**
  - OLS: new `predictions` output; model-result readers take `predictorCount` and ignore padded
    tables. Derived `correctedAkaikeInformationCriterion` is `null` when `n <= k + 1`.
  - MGWR: CPU `maximumIterations` binds (default 32); coefficients and fitted values are published
    on code 3 (check status, not `NaN`); new `componentSearchIterationLimitReached` status.
  - `getSpatialWeightsEigenvalues` rejects small nonreversible matrices; `getSpatialExactLogDeterminant`
    uses log absolute factors. Conformal global ranks use the packed float32 alpha exactly
    (`GPU_CONFORMAL_RANK_SLACK` removed).
  - `getNonCentralFUpperTail` throws for noncentrality above 10000 at finite positive F and on
    nonconvergence; `getGeographicalDetectorPValue` inherits this.
  - Variogram model fit throws on unequal `distances`, `semivariances` and `pairCounts` lengths.
  - `GPUSpatialErrorGM`, `GPUSpatialTwoStageLeastSquares` and `GPUSpatialWeightsTraces` reject
    layouts with more than 4096 row tiles (resolved count, including explicit `tileRowCount` and the
    sum of per-chunk counts): `tileCount must be at most 4096`. Lag and error likelihood contributors
    widen the trace tiling by default and reject layouts with more than 4096 nonempty chunks at
    construction (`trace tileCount`). CPU twins are unchanged.

- **Point patterns and optimisation**
  - Sinkhorn divergence: `status.converged` now requires the cross plan and both self plans to meet
    `tolerance`; `estimatedCommandNodeCount` is `10 + 7 * maximumIterations`.
  - Location-allocation: every finite nonnegative cost is reachable (only `+Infinity` is not);
    ties select the lowest facility row. Scan statistics select indexed zone search only for lists
    of at most 65 rows (`nearestZoneSearch`, default `'auto'`).
  - `GPU_MAX_P_STATUS.improvementMoveCount` → `improvementIterationCount` (same offset).
  - Space-time inference metadata: `getInferenceMetadata` returns `GPUSpaceTimeInferenceMetadata`
    (`requestedPermutationCount`, `stoppingRule`); decode with `decodeGPUSpaceTimeInferenceMetadata`.
  - SKATER, Ward, AZP and Max-p reject graphs above 32,768 command nodes at construction. The SKATER
    count includes its loss-processing passes.
  - Hawkes: `logLikelihood` stays the entering-iterate value; use the new `fittedLogLikelihood`.
  - Huff and REDCAP outputs become `NaN` or `0xffffffff` when supplied weights report loss.

- **Trajectories**
  - `geodesicSegmentFraction` projects on the wrapped start-to-end arc and picks the nearest of
    projection and both endpoints; CPU twins match.
  - `addTrajectorySortCheck` counts a non-finite row once, including the first row.
  - `computeTrackSimilarityOnCPU` rejects expanded tracks above `2^32 - 1` vertices.
  - Brownian-motion variance estimators can return exactly zero; the `...WithStatus` variants
    report `lowerBoundReached`.
  - Closest-approach time quantities keep the declared timestamp unit.

- **Network**
  - Cost table: new `output.status.complete` / `CPUNetworkCostTableResult.complete`. In map matching
    `GPUMapMatchingRouteTable.status.overflow` and `CPUMapMatchingInput.routeTable.overflow` →
    `complete`; omitting it disables table acceleration. Matching `status.overflow` is final
    capacity only; traversal loss is `candidateOverflow`.
  - `NetworkCentralityScheduleStep.reset` is `0 | 1`; `runBatch({reset: 0})` accumulates. Travel-time
    percentiles use the exact nearest rank.
  - Isochrone outputs with overlapping logical ranges throw; transient cell tables need capacity of
    at least the node count.
  - Network flow, reachability and K-shortest-path twins consume declared topology and budgets
    (`maximumRelaxations`, `maximumIterations`, `localIterations`, `maximumAugmentations`). Only the
    selected mode's budget binds; reference-mode parallel-only options are refused.
  - `NetworkStructureOperationProps` carries `mode`, `strongComponentsMaximumIterations` and
    `strongComponentsStatus`.
  - Map-matched edge volumes: `traversedEdges` needs a completeness word (`status.candidateOverflow`)
    or `completeness: 'known-complete'`; a requested `transitions.status.candidateOverflow` must be a
    separate buffer from the volume flag.
  - CPU flow in reference mode rejects parallel-only options at construction with an ID-prefixed
    message; GPU operations refuse them at construction, not at lowering.
  - `NetworkPathExtractionNodeOperationOutput` and `NetworkPathExtractionEdgeOperationOutput` replace
    the common output type's row roles.

- **Raster and terrain**
  - `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, `GPURasterSAVI`, `GPURasterEmbeddingKMeans` and
    `GPURasterEmbeddingNearestPixels`: `addToGraph(graph)` removed; use `graph.add(producer)` or
    `getCommandNodes(graph)`.
  - `GPUTerrainHorizon` and `GPUTerrainCastShadow`: `requiredHalo` and `GPURasterHaloStage` removed;
    use `getHaloStage()`, which needs `parameterLayout: 'sweep'`. `sweepOrigin` is a per-frame
    setting (`GPUTerrainHorizonSettings`). `GPUTerrainWeissLandforms.requiredHalo` → `localRadius`;
    `GPUGeomorphons.requiredHalo` → `searchRadius`.
  - Reprojection parameter layout is version 3 (`GPU_RASTER_REPROJECTION_PARAMETER_VERSION`).
  - Zonal packer defaults to seven words with `parameterLayoutVersion: 2`; the contributor default
    stays 1. Fractional scalar statistics take `cellSizeMode` and `cellAreas`.
  - `GPUTerrainDerivatives` rejects sheared grids. Reconstruction and fill-holes advance at most
    `maximumIterations` rounds.
  - Embedding pyramid levels accept an `aggregation` input; tiled reprojection rejects more than
    16,382 tiles.

- **Dataframe**
  - Cell families: `GPUCellFamily` (aggregation) → `GPUCellAggregationFamily`; the indexing family
    type is the public `GPUCellTopologyFamily`.
  - `GPUPeriodComparisonRecipeProps.sumScale` (default 65536) governs both aggregation stages.
  - `GPUClassBreaksOperationProps.rowCount` is optional; `valuesFormat` is inferred on both forms.

- **Point cloud**
  - Eleven producers move their output ports into a required `props.output` with an exported
    `GPUPointCloud<Name>Output` type (Neighborhoods, Features, Ground, Height, Surfaces, SMRF and
    others). `GPUPointCloudStatisticalFilter.validity` → `contributor.props.output.validity`.

- **Geospatial**
  - `GPUGroupConcaveHullProps.layout` (`'fixed'` default, `'packed'`) with a packed output type.
  - Recipe results export concrete `GPU<Name>RecipeOutputs` and `GPU<Name>RecipeIntermediates` types.
    Fleet-dwell stops use `GPUFleetDwellZoneEventsTrajectories` (no `timestampsLow`).

- **Lenses**
  - `RecordedLensFile.record` is `unknown`; `fromRecordedLensFile` returns `DecodedLensFile`.
  - `LadderCoordinator.retainRelations` owns relation mappings; the coordinator emits a `'mapping'`
    event (`CoordinatorEvent` is no longer an exported name).
  - Result views group tables by capture key space; `LensScenarioDescriptor.presentation.primaryOutputCaptureId`
    selects the headline output (a non-final output disables the view).
  - Public entries become named exports. Names: root 36 → 7, `/computation-lenses` 36 → 16,
    `/computation-views` 461 → 8 (`createDefaultViewRegistry`, `LadderCoordinator`,
    `buildDecodedLensTwin`, `buildLensTwin`, `LensController`, `LensControllerOptions`,
    `mountLadderLens`, `defineLadderLensElement`), `/computation-lenses/registry` 6 → 3, each
    `*/computation-pack/index` entry to `createViewPack` (importing it still registers the pack). Every
    name a consumer in the repository used is kept.
  - `LocalPermutationReplayProps.keySpace` removed; replay capture publishes only the first `count`
    valid rows and refuses a missing, invalid or over-capacity count.
  - A scenario whose recorded choices lack a required template now fails before the graph is built or
    submitted; dropped choices no longer produce an incomplete record.
  - Dataframe lens runners declare and describe the graph before submitting it.

- **Packaging and tooling**
  - `apache-arrow >=17.0.0` is a required peer (no longer optional).
  - Package licence expression is `MIT AND Apache-2.0 AND BSD-3-Clause`.
  - Contributor-contract baseline schema is version 2 (per-site `{id, description, reason}`);
    `--update-baseline` needs `--reason` for new sites; `--migrate-baseline` converts once.
  - Oracle fixtures are driven by `scripts/oracle-fixtures.json`; `yarn test-benchmarks` runs all
    benchmark suffixes; test typechecks default to an 8192 MiB heap and stop at the first failure.

### Fixed

- Population-support alignment inverse-convention `validity` flag misread as usable (ERG-01).
- Compact outputs reported candidate loss as final overflow, hiding lost rows (CORE-06, GEO-03).
- Decimal preparation rounded to binary64 before the Float32 check and refused Int64 (CORE-02).
- `sliceGraphChunkGroupView` accepted ranges outside its parent (CORE-01).
- Prepared geometry refreshed incorrectly after discarded encodings (ERG-02).
- Cell-table delta wrapped signed 64-bit subtraction (DGGS-01).
- Cell sum scale overflowed or looped for tiny inputs (DGGS-03).
- Calendar kernels mis-handled signed origins and divisors (DPC-02).
- Point-cloud local origin included non-finite points (DPC-03).
- Negative sub-millisecond timestamps were truncated, not floored (DPC-10).
- Nearest-feature joins fed application IDs to row lookups (GEO-02).
- Offset-curve noding split or dropped valid components (GEO-01).
- Alpha-hull and Voronoi circumcentres failed at large scale and with either winding (GEO-05, GEO-06).
- Counterfactual and isovist outputs accepted overlapping ranges (LAND-01, URB-01).
- Cost-table map matching trusted incomplete tables (NET-01).
- Centrality `reset: 0` discarded totals (NET-02).
- Travel-time percentiles used a fixed rank epsilon (NET-04).
- CPU flow ignored declared CSR ports (NET-05); CPU structure dropped SCC status (NET-06).
- Turn penalties never reached point routing (NET-09); flow solver choice ignored capacity (NET-10).
- Huff probabilities kept stale edges and overflowed for large attractiveness (PTS-03, PTS-09).
- Cross-correlation `testedCount` required `adjustedPValues` (PTS-04).
- Location-allocation treated costs above a cutoff as unreachable (PTS-02).
- Poisson scan baselines overflowed float32 (PTS-07).
- Sinkhorn reported convergence from the cross plan alone (PTS-01).
- Sequential inference reported requested draws as actual (PTS-08).
- Stops lost `temporalContext` and zero-centroid slots consumed candidates (RCP-02, RCP-03).
- Reprojection sample transforms lost precision away from the origin (RT-04).
- Horizon and cast-shadow sweeps depended on window cropping (RT-03, RT-13).
- Zonal windows lost thin cells and snapped quarter cells (RT-01, RT-02).
- Scalar zonal area used only the determinant (RT-10).
- Reconstruction ran an extra unbudgeted round (RT-08).
- GWR packer mutated its target before validation (SSR-05); MGWR CPU ignored `maximumIterations` (SSR-06).
- OLS held-out predictions lost precision at large coordinates (SSR-03).
- Padded model tables shifted coefficient readers (SSR-08).
- Exact log-determinant and conformal ranks used absolute floors and slack (SSR-01, SSR-02, SSR-12).
- Non-central F upper tail returned `1 - CDF` cancellation (SSR-04).
- Trajectory projection used a different arc than the segment (TRJ-01); Brownian variance never chose zero (TRJ-02).
- Track sort check skipped the first row and accepted infinities (TRJ-03).
- Lens result tables joined columns by position and mis-counted vector rows (LENS-01).
- Lens result views dropped triangles at a global stride (LENS-02); controllers removed shared relations (LENS-04).
- Nearest clear shader used an infinite constant that Metal rejects (WKB and nearest-feature
  joins failed to compile).
- Nearest-feature distances overflowed for float32 squares above the float32 range; the
  unbounded query returned no row at coordinates near 1e20.
- Multipart line coverage rounded endpoints through projection fractions and accepted a gap.
- Multipart candidate pruning copied every prefix entry for disjoint inputs.
- Nearest joins with a distinct offset view of equal feature count were accepted by prepared inputs.
- Kepler WKB selection failed on an empty XY chunk (four-byte placeholder, eight-byte row binding).
- Network flow had no final-flow decomposition; CPU phases ordered paths serially.
- Recipes left resources registered on the caller graph after a failure at a later stage.
- Merge loops of error GM, 2SLS and weights traces had no bound on the tile count.
- Shader memory estimator counted unknown typed immutable layouts, for-initializer arrays,
  block-commented declarations and half-float aliases as zero bytes; default-read storage
  declarations were not counted as bindings.
- Snap-rounded noding lost crossing-candidate loss when its intersection BVH was bounded.
- Website and example typechecks failed on stale package output and strict override/index access.
- SKATER ignored lost weights on its CPU twin and operation forms; REDCAP staged private empty CSR
  offsets to work around the old MST refusal.
- Lens records could drift from the registered graph unnoticed: every scenario now has a declaration
  and committed-record signature check.

### Added

- `candidateOverflow` status ports across compact producers, weights status ports and CPU twins;
  `candidateOverflowSources` on `createPublishNode` and `GPUPairGather`.
- `alignmentCode`, `matchedMask`, `POPULATION_SUPPORT_ALIGNMENT_CODES`,
  `decodePopulationSupportAlignmentCode`.
- `decodeGPUGeometryValidityIssueCode`, `GPU_SEGMENT_SNAP_ISSUE`, `decodeGPUSegmentSnapIssueCode`.
- `getGPUKShortestPathsOutputLayout`, `decodeGPUKShortestPaths` and their option and result types;
  optional `nodeRows` and `edgeRows` ports on path extraction.
- `getGPURasterGeometricMedianCompositeOutputLayout`; `GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH`
  (24), `GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH` (32).
- `getGPUIsovistRayCount`; `getPointCloudLocalOriginSummary`; `computePointCloudVoxelMeans`;
  `getTimestampCalendarMilliseconds`; `prepareGPUNetworkPointRoutingTurns` and optional
  `turnPenalties`.
- `GPUScanNearestZoneSearch`, `GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT`, `estimatedFullScanSeparationCount`.
- `decodeGPUGeographicallyWeightedRegressionStatus`, `decodeGPUMultiscaleGeographicallyWeightedRegressionStatus`,
  `GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS`.
- `getConformalPredictionRanks`, `GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT`, `readGeographicalDetectorPValue`.
- `estimateBrownianMotionVarianceWithStatus` and the windowed variant.
- `GPUAnalysisOperatorIdentity`, `normalizeGPUAnalysisClassValue`, `GPUParameterValues`,
  `estimatedCommandNodeCount` on regionalization, tile reprojection and scan contributors.
- Lenses: `buildDecodedLensTwin`, `DecodedLensFile`.
- `CreateGPUComputationRecordProps` gains `executionId`, `parameters` and `implementationRevision`.
- Scripts: `yarn verify`, `yarn check-packed-consumer`, `yarn build:release [--promote]`,
  `yarn typecheck-status --watch`, `yarn check-contributor-contract --check-stale`.
- `GPUMultipartPointGeometryPort`, `GPUMultipartLineGeometryPort`, `GPUMultipartGeometryPort`,
  `validateGPUMultipartGeometryPort`, `createGPUMultipartRowOwnersNode`,
  `GPU_MULTIPART_NO_FEATURE_ROW` from `analysis-core`.
- Multipart point and line operands for `GPUSpatialPredicateJoin` (`intersects`, `within`,
  `coveredBy`), `GPUNearestFeatureJoin` (neighbors mode), `GPUSpatialJoinCandidates` and prepared
  join geometry, geometry measures and affine transform; `leftRows` and `rightRows` identify
  features and capacities count features.
- `computeSpatialPredicateJoinOnCPU`, `computeNearestFeatureJoinOnCPU`, `CPUSpatialJoinGeometry`,
  `MultipartJoinPredicate` (`candidateBoundsTestCount` reports host index work).
- `GPUNetworkFlowOutput.paths` (`GPUNetworkFlowPathOutput`), `NetworkFlowOperationProps.pathCapacity`,
  `CPUNetworkFlowOptions.pathCapacity`, `CPUNetworkFlowResult.paths` (`CPUNetworkFlowPaths`).
- `createLocalPermutationReplay` and `LocalPermutationReplayProps`.
- Scripts: `node scripts/typecheck-tests.mjs --generate-only`; `node scripts/measure-typecheck.mjs`
  (cold and warm runs with sampled process-tree RSS; `--mode check|declarations|build|packages`).
- CI builds packages before any typecheck and checks `examples/deck/kepler-compute`.
