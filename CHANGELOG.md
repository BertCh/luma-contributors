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
  No luma.gl APIs are copied or forked; a few small pinned WGSL helpers are registered in
  `dev-docs/upstream-workarounds.md`. See `dev-docs/upstream-first-plan.md`.
- Targets published luma.gl `10.0.0-alpha.16` with peers `~10.0.0-alpha.16`; aligns grid readers with the public shared coordinate WGSL and forwards Kepler compile options. Node/GPU qualification is tracked in LR-22; segmented sums and PCG remain separate adoption work.
- Fixed `GPUColorScale` and `GPUBivariateClassification` when `output.colors` is omitted: the colorize
  kernel declared palette, position or parameter bindings its counts-only body never read, so the
  dispatch failed WebGPU validation and `classCounts` stayed zero. Both GPU specs now compare
  output subsets against the full run.
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

### Breaking changes before the alpha (review fixes 2026-11-20)

The second library review's fixes (waves 1 to 6 and the known-issues pass) end with one name per
concept and no compatibility aliases: the package is unpublished, so each rename below removes the
old spelling, and every `@deprecated` alias added during the pass is gone. Committed lens scenario
IDs keep their names. Per-change caller notes, including every behaviour fix, are in
[the change records](dev-docs/research/library-review-2026-11-19/changes/); the decisions are in
[the review README](dev-docs/research/library-review-2026-11-19/README.md).

- **Renamed**
  - `GPUSpatialKMeans.k` → `clusterCount`; `k` on `GPUNeighborSearch`, `GPUJacquezTest`,
    `GPUNearestFeatureJoin` and `GPUNearestFeatureWeights` → `neighborCount`. Neighbour-search
    `radius` → `searchRadius` (radius mode) or `maximumDistance` (kNN); `weightKind:
    'inverseDistance'` → `weightType: 'inverse-distance'`; nearest-join `ties: 'lowest-id'` →
    `'lowest-row'`.
  - `GPUSpatialClustering` → `GPUDBSCAN`, `GPUHierarchicalDensityClustering` → `GPUHDBSCAN`,
    `GPUPycnophylactic` → `GPUPycnophylacticInterpolation` (props, outputs, packers and constants
    follow). DBSCAN `minimumPoints` → `minimumSamples` (the point itself counts).
  - `GPUKeyJoin` → `GPUKeyLookup`; `kind` → `compactMatchedLeftRows`. Each matched left row is
    emitted once; pair-join terminology is reserved for an operation that emits every pair.
  - `GPUPairGather` `output.rowCount` → `output.status.count`.
  - `GPU_LOCATION_ALLOCATION_STATUS`, `GPU_MINIMUM_SPANNING_TREE_STATUS`, `GPU_AZP_STATUS`,
    `GPU_WARD_STATUS`, `GPU_MAX_P_STATUS` → the same names ending in `_STATUS_LAYOUT`.
  - Colour scale resident `domainCount` → `classCount` (a threshold-settings overload needs no
    dummy edge count). Hydrologic indices `slope` → `descentTangent`.
  - Spatial weights transform `symmetrize` → `average-reciprocal` (it averages over the existing
    pattern and does not add reverse edges); algebra `symmetricDifference` → `symmetric-difference`.
  - `GPUPointBuffer.coordinateSystem` → `spatialContext`; `CellPrecision` `extended`/`fp64` →
    `double-single`; dot-density `maskExtent` → `maskPlacement`; `getKahanSumWGSL` →
    `getStridedPairwiseSumWGSL`; `unpackGPUCellPeriodKey` → `unpackGPUCellPeriodKeyChecked`; trace
    and island-attachment crossover constants renamed to say automatic or forced.
  - Ontology imports move to `@luma-contributors/analysis/gpu-spatial-analysis/ontology`; the parent
    spatial entry no longer re-exports them.
- **Removed**: `getCholeskyWGSL`, `getSymmetricEigenvalueRangeWGSL` and
  `getSpatialWeightsTracesProbeWGSL` from the regression entry; `reusePreparation` on the conditional
  Gaussian simulation operation (the producer keeps it); implicit one-period cell keys;
  `GPUComputationRecorder.graph` (use `getRecordedGraphDescription`); flat `sumOverflow` views (read
  `status.sumOverflow`); legacy packed or flat status views wherever a named status port exists
  (regression, CPU weights, OPTICS, DBSCAN, HDBSCAN, location allocation, network kernel density,
  depression hierarchy, population support recipe).
- **Changed signatures and types**: `prepareGPUNetworkRestrictionStates({nodeCount, edges,
  restrictions})` with `[source, target]` edge tuples; `GPUTransferMask` bindings typed `uint32`;
  terrain derivatives require one of their outputs and distance fields encode seed/mask
  configurations in props; `GPUPickRegionMask` has its own output type; population support
  alignment splits contributor and recipe props; readonly spectral props and operation descriptors;
  emerging hot spot critical levels use literal keys plus a checked lookup.
- **Newly required inputs** (construction refuses without them): budgeted trajectory thinning and
  staypoints require `status.candidateOverflow` and locations require `status.iterationLimitReached`;
  an anti-only spatial predicate join requires `output.unmatched.status.candidateOverflow` unless
  its capacity provably covers every pair; probit rejects `tolerance` above 0.01.
- **Changed results**: mobility indicators exclude missing or unsettled location keys and count them
  in `status.excludedLocationCount`; the drainage recipe reports geographic and Web Mercator
  distances and reach lengths in ground metres; regression recipe `invalidCount` counts excluded
  rows (the solver outcome is only in `code`).
- **Catalog and packaging**: every catalog record carries `stability`, `'experimental'` for all
  records until the stable-tier rule is met; Kepler discovery walks the full catalog keyed by
  `{package, exportName}` (the `qualified` option and bare-name passes are gone). Tarballs ship
  `dist` only (ESM, CommonJS, declarations, source maps); `src/`, `dist/testing/` and the build
  record are excluded, and relative `import()` types in declarations are rewritten by the build.

### Breaking changes (review fixes 2026-11-19)

- **Inputs now rejected** (construction or packing time; fix the input named)
  - `GPURasterZonalStatistics` defaults `parameterLayoutVersion` to 2, the layout
    `getGPURasterZonalStatisticsParameterValues` writes. A geographic `parameters` view that is not
    exactly seven words now throws unless `parameterLayoutVersion` is given: pass `1` for a legacy
    four-word view or `2` for a padded one. `getRasterZonalCellArea` follows the same default.
  - `GPUContinuousSpatialSample` rejects an area frame whose `bounds` lattice has more cells than
    `maximumCandidates` (raise `maximumCandidates` to the count in the message), and a configuration
    whose serial work per wave exceeds 2^24 inner iterations (lower `maximumCandidates` or the frame
    size).
  - `GPUClassBreaks` in approximate natural-breaks mode rejects `maximumClassCount` above
    `naturalBreaksBinCount`; it used to return at most that many classes. `GPUColumnQuantilesOperation`
    rejects a `rowCount` that differs from the values length.
  - `GPUPointCloudNeighborhoods` rejects runtime `parameters` without `output.status`, as it already
    did for static `supportBounds`.
  - `GPUNetworkKFunction` rejects more than 92,682 events (`GPU_NETWORK_K_FUNCTION_MAXIMUM_EVENT_COUNT`),
    above which its uint32 pair counts could wrap.
  - Fifteen terrain contributors (curvature, geomorphons, vector ruggedness, Weiss landforms, relief
    shading, horizon, cast shadow, viewshed, line of sight, cumulative viewshed, summits, peak snap,
    flow field, point-horizon profile and visibility) reject a sheared grid at construction.
    Resample onto an orthogonal grid first; rotated orthogonal grids are unchanged.
  - `GPUGridGenerator` rejects `columns` and `rows` beyond the sizes whose lattice indices float32
    holds exactly: 2^24 for square and point grids, 2^23 - 1 columns for triangle grids, 5,592,405 to
    8,388,607 for hexagons (larger grids repeated cell positions).
  - `GPUPointHorizonProfile` and `GPUTerrainCumulativeViewshed` reject dispatch settings that would
    declare more than 32,768 command nodes and publish `estimatedCommandNodeCount`.
  - Parameter packers validate the float32 values they return and throw where they used to write a
    zero, infinity or collapsed range: `getGPUVariogramParameterValues`,
    `getGPUSpatialCorrelogramParameterValues`, `getGPURipleyParameterValues`,
    `getGPURipleyDistanceFunctionsParameterValues`, `getGPUPointPatternIndicesParameterValues`,
    `getGPUSpaceTimeVariogramParameterValues`, `getGPUSpaceTimeKFunctionParameterValues` (bounds or a
    time window that lose their strict order in float32), `getGPUSpatialClusteringParameterValues`,
    `getGPUSpatialKMeansParameterValues` and `getGPUNetworkIsochronesParameterValues` (an extent with
    zero or non-finite float32 width, or buffer and walk-cost values that overflow). Messages now end
    "in float32".
  - `fitVariogramModel` and `fitSpaceTimeVariogramModel` (and the CPU and GPU model-fit operations)
    throw on bin arrays of unequal length, naming both lengths, instead of fitting the shortest prefix.
  - `GPULocalFrame.projectArray` and `unprojectArray` throw, naming both lengths, when a supplied
    `target` is too short; they used to drop rows silently.
  - Spatial parameter schemas read only own properties: a field named `constructor`, `toString` or
    `__proto__` uses its declared default. `prepareArrowTableForGPUAnalytics` reads `timeColumns` and
    `dictionaries` the same way.
  - `GPUHuffTradeAreas` treats an infinite attractiveness or weight as invalid and excludes the slot,
    as it did for NaN and non-positive values.

- **Changed results and status values**
  - Chunked `GPUChunkedSpatialWeightsTranspose` output reports loss on the outer
    `outputWeights.status`: `overflow` is 1 when an output chunk dropped slots, `candidateOverflow` is
    1 when the input was incomplete. Chunk descriptions keep `count` and `requiredCount` but carry no
    loss flags, so local Moran, hot spot and the global statistics accept it.
  - `GPURasterLaggedCorrelation` status 2 now means every paired value of one series is equal
    (zero spread); a translated non-constant series is no longer reported as zero variance, and
    identical series correlate 1 at any finite magnitude.
  - `GPUSpaceTimeVariogram` reports status 3, not 0, for a bin with more than `2^32 - 1` pairs
    (`pairCounts` saturates at `0xffffffff`).
  - `GPUTrajectoryLocations` publishes no partial labels at its round limit: every active `locationIds`
    row is `0xffffffff`, `status.count` is 0 and `iterationLimitReached` is set. Check
    `iterationLimitReached` before reading labels.
  - `GPUTrajectoryStaypoints` writes NaN `durations` and `centroids` for every staypoint of a track
    that hit `maximumSearchRows`, with or without `status.candidateOverflow`.
    `GPUMobilityIndicators` writes NaN `realEntropy`, not 0, for a track above
    `maximumEntropySequenceRows` (`computeMobilityIndicatorsOnCPU` takes the same optional cap).
  - Spatial CSR visibility (`mode: 'csr'`) gives an invalid viewpoint an empty row and publishes the
    other rows; one invalid viewpoint no longer empties the graph. `status.invalidCount` still counts them.
  - `GPUSphericalPolygonContains` accepts either ring orientation: clockwise holes (GeoJSON order) are
    no longer ignored and clockwise shells no longer contain nothing.
  - `GPUMapColoring` leaves every row uncoloured with `converged` 0 when its weights status reports
    `overflow` or `candidateOverflow`. `GPUNeighborhoodSummary` (and its operation forms) writes NaN
    for incomplete weights, as does `GPUSpatialMarkov` (counts, probabilities, lag edges, diagnostics
    and `status.converged`); `GPUSegregation` (per scale), `GPUColocationQuotient`,
    `GPULocalCategoryDependence`, `GPUCategoryPairCounts` (so `GPUSpatialEntropy`) and
    `GPULocalOutlierFactor` write NaN, or 0 for counts. `GPULocalKnox` and `GPUNearRepeatKnox`
    suppress results when the pair list reports loss, as `GPUKnoxTest` does; weights-mode
    `GPUEmergingHotSpots` suppresses z-scores, trends and counts and labels cells `UNDETERMINED`.
  - The Poisson `GPUSpatialScanStatistic` reports a frame with a negative, NaN or infinite baseline cell
    as invalid (`status.invalidCount` 1, NaN statistics) instead of rescaling around it.
    `GPUPointPatternIndices` polygon quadrat statistics are NaN when points fall in a quadrat the area
    lattice missed.
  - `GPURasterSkeleton` publishes the state after at most `maximumIterations` iterations; the
    convergence test no longer thins one iteration further.
  - `GPUTerrainHeightAboveDrainage` reports cells on a receiver cycle in caller-supplied D8 directions as
    `GPU_TERRAIN_DRAINAGE_UNRESOLVED` with `converged` 0, not as "no drainage".
  - Computation records: read `node.entryPoint ?? source.entryPoint` and `node.constants`, since
    nodes that share identical WGSL text keep their own entry point and override constants.
    `describeGPUCommandGraphEncoding` gives each render pass its own pass node.
  - The operator catalog's `status` lists changed: 93 operators and recipes now list their count,
    overflow, invalid, uncertain, convergence or approximation status kinds, and `GPULineSegmentize`,
    `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`, `GPURectangleClip` and `GPUOffsetCurve`
    (compact layout) list count, overflow and vertex-count status. An empty list now means no such
    view; queries filtering on `status` return more operators.

- **Tooling**
  - `check-upstream-first` rejects a luma.gl binding re-exported through a local `export {}`,
    `export default` or a value alias; `check-contributor-contract` finds aliased `assert` and internal
    symbols re-exported through an import, invalidates a single-invocation loop allowance when the host
    bound its reason names changes, and no longer accepts a bare inline marker (add the bound).
  - `check-packed-consumer` resolves every relative reference in every packed `.d.ts` under NodeNext and
    type-checks a CommonJS (`.cts`) consumer beside the ES module one; public parameter schemas and
    recipe results no longer reference extensionless declaration paths.
  - Verify and release honour a lower command-line heap over a larger `NODE_OPTIONS` heap. Release
    records identify the checked tarball by SHA-256 and byte length, and promotion verifies it first.

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
- Recipes in the spatial and network recipe indexes left resources registered on the caller graph after a
  failure at a later stage. The 13 urban-morphology and landscape-ecology recipes are outside that
  every-stage preparation; they now validate their options, packed input layouts and caller-owned
  outputs before the first `graph.add`. Device-limit checks inside their overlay planning (tessellation
  and building adjacency) can still reject after nodes were added.
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
- Zonal variances and standard deviations lost accuracy at projected magnitudes: a feature or raster zone
  of 100,000 values near 1e7 with variance 2 reported about 123 (`GPUZonalStatistics`,
  `GPURasterZonalStatistics`; GEO-105).
- `GPUSpatialClustering` and `GPUNeighborSearch` could miss a neighbor exactly one epsilon or radius away
  at projected origins of about 1e5 and above, and a point on the maximum bound could miss itself (CORE-102).
- Voronoi diagrams, concave hulls and minimum bounding circles dropped the circumcircle of a thin triangle
  whose rounded cross product cancelled to zero (CORE-103); minimum bounding circles had radii 50 to 73
  percent too large above extent 2^32 or below 2^-42 (GEO-106).
- Study-window circle fractions (Ripley, inhomogeneous point patterns, kernel density edge correction) were
  wrong at a right-angle boundary vertex on GPUs that flip `atan2(y, -0.0)` (CORE-105); the shared Poisson
  observed term was NaN for 1e9 cases over a 1e-30 baseline (CORE-104); the shared Jacobi eigen solver
  cleared the coupling unrotated when an entry above 2^126 overflowed (CORE-106).
- Cell aggregation, rollup and pyramid sums wrapped negative; they now saturate at `±(2^63 - 1)`, and
  `sumValues` no longer flushes scales up to 2^127 to zero on Metal (DGGS-102, DGGS-107).
  `GPUCellSetOutline` (H3) dropped boundary edges near the poles and for coarse cells (DGGS-101);
  `GPUCellCover` refused feature counts whose capped candidate total reaches 2^32 (DGGS-103).
- `GPUHealpixPointToCell` gave points just west of a 90 degree meridian at the belt-edge latitude a cell on
  the far side of face 4 or invalid face 12 (DGGS-104); `GPUPointToCell` (S2, H3) and
  `GPUHealpixPointToCell` were off by degrees for longitudes at 1e8 and beyond, and now reduce
  longitudes beyond 2^24 degrees exactly modulo 360 (DGGS-106).
- `packGPUCellPeriodKey` turned a missing cell into a groupable key; it now returns `undefined` for the
  all-ones key and for periods outside `[0, periodCount)`, as `GPUCellPeriodKeys` does (DPC-101).
  `GPUCalendarBuckets` decoded a negative time whose floored day is before day -1,000,000,000.
- `GPUCompositeScore` gave a constant indicator column a spurious deviation and scores near 1 at large
  offsets; `GPUGroupStatistics` means, `GPUKeyJoin` `mean` and `GPUValueCube` `means` were not finite when a
  group's sum exceeded the float32 range; `GPUClassBreaks` equal-interval edges collapsed to the maximum
  when the span exceeded float32; `GPUColumnProfile` returned an all-zero histogram for a finite domain
  such as `[-3e38, 3e38]` (DPC-103, DPC-109, DPC-111).
- `GPUSpaceTimeVariogram` centred both columns on all rows, so a row excluded by a non-finite time or
  second value could erase the retained pairs' differences. The inhomogeneous
  `GPUSpaceTimeKFunction` used a fixed 2^-16 pair unit that overstated K by 6.8 % at relative intensity
  100 and dropped every pair from 363; each frame now scales it so every term keeps a relative error of
  at most 2^-17.
- `getKnoxPoissonPValue` returned 0 for small upper tails; it sums the tail directly down to about 1e-300
  (SEAM-101). Recipe stage status records may carry a `code` (the spatial regression recipe publishes its
  OLS and diagnostics codes there) instead of a solver code under `invalidCount` (SEAM-110).
- `GPUNetworkAccessibility` and `GPUNetworkCostWeights` looked complete when the cost matrix stopped at its
  round budget or the requested cutoff exceeded its horizon (NET-101, NET-124); `GPUNetworkKernelDensity`,
  `GPULocationAllocation` and `GPUCatchmentAccessibility` had no way to see either.
- Travel-time percentile ranks read 99.9 % of 1,000 departures as rank 1,000 in the CPU twin; the percentile
  is now read as the decimal written, so the rank is 999 (NET-102).
- `GPURadiationFlowModel` lost the flow of an origin holding nearly all of the population (a two-zone flow of
  100 now stays 100, NET-111); `GPUKernelFlowSmoothing` returned NaN for a bandwidth too small for float32
  and now returns the input (NET-112). The Louvain operation forms rejected an optional status port given
  as `undefined` (NET-114).
- Network snapping and map matching changed which points they admit, and their distances, when a projected
  network was translated by 5e6 to 1e7 metres (NET-106).
- `getRasterEmbeddingStretchRanges` threw on a rank-deficient PCA; constant components now get a usable
  stretch (RT-121). `GPURasterEmbeddingZoneMean` published a partial mean for a truncated tuple list; it now
  reads the coverage producer's `requiredCount`, `overflow`, `candidateOverflow` and `zoneValidity` and
  publishes NaN, and no longer fails normalization for tiny nonzero means (RT-126, RT-127).
- Streamlines (GPU and CPU) emitted a final point outside the field; they end at the last point with
  velocity support (RT-124). `getGPULagrangianFlowMapParameterValues` rejected signed cell sizes, so a
  north-up field now uploads with a negative cell height (RT-129).
- `getGPURasterSnapshotEOFDecomposition` changed explained-variance fractions or dropped a small mode when
  the data were rescaled; it now converges relative to the matrix norm (RT-132).
  `GPURasterLaggedCorrelation` returned 0 or status 2 for identical series above about 1e10 (RT-133).
- Simple and multi-scale local relief changed with a distant first valid elevation; the summed-area table now
  subtracts its anchor exactly, so whole-raster and windowed runs agree (RT-128).
- Kepler location allocation now skips and counts demand rows with negative weight, as documented; Kepler
  predicate join tests geometry relative to the inputs' minimum corner, so translating a polygon and a point
  together no longer changes whether the point lies in a hole; Kepler zonal value distribution reruns on the
  CPU engine with a `gpu-fallback` warning naming the lost stage when the zone assignment overflowed or left
  unproven containments; Kepler GPU kriging reports cells without data (`kriging-cells-without-data`)
  apart from `singular-kriging-system` (SEAM-104, SEAM-106, SEAM-108, SEAM-115).
- `addSegregationDecompositionRecipe` randomization counted exact ties as extreme about half the time and
  biased the p value low; it now counts a replicate against the observed change summed in the replicates'
  own order (URB-101).
- Lens recordings and static twins headline the primary output (the 4,662-cell kriging result, not a
  one-row diagnostic); a result whose headline geometry draws nothing says "geometry not drawn" in its alt
  text, live-region text and thumbnail caption; the make-valid and polygonize lenses describe the ring count
  as capacity to allocate only when candidate overflow is 0; the kernel-source view shows a node's own entry
  point and override constants (LENS-101, LENS-108).
- `formatDecimal128` formatted negative scales with a decimal point (coefficient 12, scale -3 is now
  `12000`); `submitGPUCommandGraphWithTimings` leaked its timestamp query set when command-encoder creation
  threw (DPC-106, DPC-127). `prepareArrowTableForGPUAnalytics` misread columns named `constructor` or
  `toString` (DPC-104, DPC-115).
- The `CommonJS` build rewrote string values that look like import specifiers; it now rewrites only real
  import and export specifiers. `typecheck-status` ignored an edit to a resolved dependency
  declaration (TOOL-108).

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
- `status.sumOverflow` (cell aggregation, rollup and pyramid), `costsComplete` and `costLimit` inputs with `status.complete` outputs on `GPUNetworkAccessibility`,
  `GPUNetworkCostWeights`, `GPUNetworkKernelDensity` (`output.ports.status.complete`) and
  `GPULocationAllocation`, `costsComplete` on `GPUCatchmentAccessibility`, and the optional stage `code`
  of recipe status records.
- `usedThroughOtherEntries` in `scripts/upstream/usage.mjs` output; `scripts/upstream/diff.mjs` takes two
  optional installed `@luma.gl` paths.
