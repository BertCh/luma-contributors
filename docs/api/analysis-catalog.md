---
title: Analysis catalog
description: Browse recorded contributor figures and query the analysis operator catalog.
---

# Analysis catalog

Browse the [interactive catalog](/explore/) for contributors across dataframe, spatial analysis,
network, raster and terrain analysis. Contributor cards include examples, parameters and recorded
input/output figures where available. For example, open [point density](/fn/point-density) for its
recorded GPU computation on a small real dataset. Entries without a recording remain discoverable
in the catalog.

The Kepler example’s [validation and regression catalog](https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/VALIDATION-CATALOG.md) binds the existing GTWR, space-time kriging, Vecchia, applicability, conformal, scoring and FDR contributors with executable forms and diagnostics tables.

Kepler's geometry operations require finite float32 projected XY. Buffer routes additionally
require raw source XY to fit finite float32 before projection, even if a local frame could make an
out-of-range source coordinate fit. Both dissolved and per-feature buffers validate every source
before collection-wide tolerance calculation or GPU workspace allocation. Correct the source coordinates for
this input error. Buffer capacity retries remain available for finite inputs whose arrangement
needs more storage; additional capacity cannot repair NaN, infinity or float32 coordinate overflow.
The geometry validity binding instead retains each offending source row with a
`non-finite-coordinate` diagnostic and still reports valid source rows.

Kepler's native MultiPoint, MultiLineString and MultiPolygon readers consume every GeoArrow chunk
through the public geometry preparation path. Source row order, empty features, actual XY and
cumulative feature, line, polygon and ring offsets are preserved, including zero physical chunks.
The readers accept XY planar geometry of their named family. Float64 coordinates that lose
precision under the default reject-loss policy (for example 0.1) and XYZ input are refused.
The installed Apache Arrow runtime also refuses int64-offset columns because it lacks LargeList.
Spherical edges are refused whether declared at the descriptor top level or in metadata;
conflicting declarations are refused.

Kepler's `polygon-triangles` tables use the same CPU/GPU publication schema. Each counted triangle
retains its source and triangle IDs, including original part indices after empty parts. `degeneracy` is `none`, `line` or `point`, with Polygon,
LineString or Point geometry respectively; the catalog therefore declares mixed geometry.

Kepler's `listCatalogEntries()` lists the full analysis catalog and explicitly qualified binding
extensions. Every row has an `identity` with `package` and `exportName`. Use both fields to join
bindings and lens scenarios and to key UI rows. The network and spatial
`addVisibilityGraphRecipe` exports have separate rows. The `runnable` field reports how an entry
can run in the host, and `reason` describes entries with no runnable route. For entries with
bindings, `engine` reports the execution routes available across those bindings. The `bindingIds`
and `lensScenarioIds` fields identify the registered operations and scenarios for that identity.
`groupCatalogLensScenarios()` returns a map keyed by `JSON.stringify([package, exportName])` for
scenarios with a resolvable identity.

### Kepler binding transport and admission

The Gaussian predictive score, ensemble score and FDR bindings accept native numeric tables as
well as point features. Their input port has `acceptsTable: true`; native transport is
`{type: 'table', table}` through the typed catalog data route (`runDataGpu` for GPU execution).
Feature transport remains a FeatureCollection with numeric properties and optional point geometry.
Native tables declare `rowCount`, matching `columns` and `columnTypes` keys, and that many values
in every column; `geometryColumn` is absent. Required numeric column mappings still apply.
Column-only tables need no synthetic points or geometry vertex count. The binding's
`maximumFeatures` is a total row cap on the table route (2,048 for these validation bindings),
and a feature cap on the feature route; geometry-specific admission applies only to features.
`feature_index` identifies the zero-based input row in either transport, not a caller ID.
Output `geometryFromInput: 'points'` means inherit the selected input's geometry: native tables
remain geometry-free and feature results preserve their point geometry. The older scalar
`geometry` field is a compatibility fallback. Clip output similarly uses
`geometryFromInput: 'features'` to inherit the subject's line or polygon geometry.

Binding parameters accept finite numbers or trimmed decimal/scientific numeric text matching
`^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$`, for example `-0.5`, `.25`, `1.` or `2e-3`.
Scalar numeric parameters reject hexadecimal, binary and octal text, booleans and arrays;
integer parameters additionally require whole numbers. Numeric lists accept arrays of finite
numbers or comma-separated decimal/scientific text, rejecting empty entries. Both explicit spellings of a numeric alias
are validated before comparing them, and conflicting values reject rather than choosing one.
For the generated ten-step FTLE binding, `timeStep` is signed seconds in [-1, 1], excluding zero:
`Math.fround(timeStep * 10)` must also be finite and nonzero. A tiny nonzero double whose duration
rounds to Float32 zero is refused. The flow-map packs the step itself as Float32 too: choose
`Math.fround(timeStep)` finite and nonzero. The current duration-only check can admit `-1e-46`,
which packs to negative zero and cannot advance particles, even though its ten-step duration is
nonzero in Float32. A negative step that remains nonzero in Float32 selects backward integration.

Units follow the selected binding route: geographic metric-grid controls use metres after local
projection; planar controls use coordinate units. Explicit `source-coordinate-units` metadata
means source coordinate units. Legacy `unit: 'crs'` retains the conservative label **CRS units**;
it does not promise metres or a particular projection. Consult each binding's coordinate mode,
column mappings, numeric constraints and capacity limits before executing it.

`result.stages` is optional. Raster embedding PCA and projection supply it with each stage's
CPU/GPU execution location, including the CPU eigensolver. Check for its presence before
iterating. Scoring and FDR results may omit it. When stages are absent, read
`result.dispatchedContributors` if supplied as evidence of actual GPU work; that list
does not reconstruct CPU stages or certify the complete pipeline. If neither field is supplied,
execution-stage evidence is unavailable. A GPU binding may contain CPU stages; do not infer
every stage's location from its engine or reconstruct supplied provenance from contributor names.

Kepler's bucketed CPU DBSCAN route admits at most **20,000 total input points**. The curated
cluster-and-footprint recipe has a separate limit of **6,000 total input points** on both CPU and
GPU routes. These are total-input admission limits, not per-chunk limits or a general GPU DBSCAN
ceiling. The Kepler ICP binding accepts `maximumIterations`, with a default of 10 and a range
of 1–30. The producer's independent default is 20 with range 1–100.

Enabled next actions preserve geometry-free results as typed table text. Keeping a table run
captures its original scalar schema and values in a `source-table` snapshot, including caller
columns named `row_index`; replay does not generate replacement row indices. This source admits
at most 100000 total rows, 64 columns, 1000000 cells and 262144 JSON characters per snapshot;
consuming bindings may enforce smaller row limits. Nonfinite numeric missing-value sentinels and
signed zero survive serialization. The original three host source operations retain their order,
with `source-table` appended.

Saved conflicting spellings open as a blocked editable form. Correct the canonical control or
reset edits before Run; default invocation parsing and every execution route still reject
explicit conflicts.

Find an operation: [spatial](#spatial-analysis-chooser), [network](#network-chooser),
[terrain](#terrain-chooser), [raster](#raster-analysis-chooser),
[dataframe](#dataframe-analysis-chooser), [geospatial](#geospatial-chooser),
[point cloud](#point-cloud-analysis-chooser), or [computation records and views](#computation-records-and-views-chooser).

## The operator catalog

The interactive catalog links to recorded runs. The operator catalog is a data model that lists
every exported contributor class and every `add*Recipe` function of all nine entry
points: `analysis-core`, `computation-record`, `gpu-spatial-analysis`, `gpu-network`,
`gpu-terrain`, `raster-analysis`, `dataframe-analysis`, `pointcloud-analysis` and
`geospatial-analysis`. `analysis-core` includes the four fitting producers
[`GPUFittingCoordinateEncoder`](./analysis-core/fitting-kernels.md#gpufittingcoordinateencoder), [`GPUFittingGradientSums`](./analysis-core/fitting-kernels.md#gpufittinggradientsums), [`GPUFittingMinibatchOrder`](./analysis-core/fitting-kernels.md#gpufittingminibatchorder) and
[`GPUAdamStep`](./analysis-core/fitting-kernels.md#gpuadamstep). A test fails when an exported contributor or recipe has no entry.

Each operator has a capability family named for what it does (`network-routing`,
`terrain-hydrology`, `raster-focal`, `raster-time-series`, `dataframe-aggregation`,
`pointcloud-ground`), its package, input and output data kinds, an optional one-line `summary`,
`dimensionality` (`'2d+time'` for temporal operators, `'3d'` for point clouds),
and the tests and documentation that cover it.

All 261 original spatial descriptors now have summaries from public TSDoc, with editorial overrides
where the opening documentation is absent or chiefly provenance. Both spatial and all-entry text
queries search these summaries. The static metadata is prepared by
`node scripts/generate-spatial-catalog-summaries.mjs`; review its printed module before updating
`spatial-analysis-operator-summaries.ts`. Text queries use these prepared summaries; route metadata
comes from the source-owned planner records described below.
The type remains optional for other catalog extensions; consumers should fall back to `exportName`
when a record has no summary.

An operator's identity is its full entry-point `package` plus `exportName`. The spatial and network
entry points each export `addVisibilityGraphRecipe`: the spatial CSR recipe builds adjacency and
degree, while the network CSR recipe adds exact or sampled network metrics. Pass both fields to
`getGPUAnalysisOperator` for either recipe. A bare name still resolves when unique and throws when
ambiguous. Catalog records, evidence, limits and arrays are frozen and their types are readonly;
copy a descriptor before adding application-specific fields.

The canonical data-kind tuple and `REUSED_ANALYSIS_OPERATORS` are also frozen at declaration;
mutating them cannot change a later guard or discovery call.

Neither visibility recipe produces polygons. The spatial recipe's CSR mode reports `count`,
`requiredCount`, `overflow`, `candidateOverflow` and `invalidCount`; network CSR adds its network
metric statuses, including `zeroWeightCycle`. The dense reference overload retains its packed
status and result layout.

The website inventory retains each overload's props and return fields. Its checker selects the
contract from the example call's inline literal `mode`; for visibility an omitted mode selects
the reference overload. Required options distinguish overloads when multiple modes allow omission.
Explicit CSR calls are checked against CSR return fields. Overload examples
with an unresolved or unknown mode fail validation. Documentation and same-name lens evidence
use the full entry-point identity. Extraction deduplicates actual re-exports and reports eligible
declarations with missing qualified metadata, including public factories.

`getGPUAnalysisCapabilityCatalog()` derives each family's `operators` and `operatorIdentities` from
the finalized records. Use `operatorIdentities` to retain entry-point information. Historical
`GPU_SPATIAL_ANALYSIS_CAPABILITIES` membership stays unchanged.

The all-entry records for `GPUConstrainedDelaunayTriangulation`, `GPUDelaunayTessellation`,
`GPUMinimumBounds` and `GPUPolygonTriangulation` also expose optional `forms.cpu` and `forms.gpu`
identities. For example, the minimum-bounds record links `CPUMinimumBoundsOperation` and
`GPUMinimumBoundsOperation`, each qualified by `@luma-contributors/analysis/gpu-spatial-analysis`.
These are operation forms of the existing contributor, so they add no contributor records.
Polygon triangulation's forms use that public spatial entry even though their implementation
lives in analysis-core. Omitted `forms` means no form metadata was recorded.

### Stability tiers

Every operator and capability record carries a `stability` field. All records are
`'experimental'` today. An experimental API may change or be removed before or after the
alpha without a deprecation period.

A `'stable'` record must meet every part of the R19-22 admission rule.

- An independent correctness oracle.
- A CPU twin or a documented reason for having none.
- A documentation heading for the operator.
- An audited catalog record.
- A translated-scene fixture.
- A completeness fixture.
- Device results on GPUs from two vendors.

### Tier and limits

Every operator has a `tier`:

- `'scale'`: the kernels are parallel over the data and sized by device limits. Limits that still
  exist (for example a neighbor capacity of 64) are listed in `limits`.
- `'reference'`: a single-invocation, exhaustive or small-dense design with a hard maximum size. Use
  these for small problems, oracles and teaching, not for million-row inputs. `limits` lists each
  maximum as `{name, maximum, reason, source}`, where `source` is the file that enforces it, or
  `limitation` describes a serial step that has no numeric maximum.
- `'unknown'`: the execution tier has not been audited. It does not imply a scale implementation.

For unaudited metadata, `status: 'unknown'` differs from a confirmed empty status list and
`readback: 'unknown'` differs from confirmed `'none'`. Queries can select `'unknown'` for `tier`,
`status` or `readback`; a query for a known status excludes unknown records. Audited exceptions
include K-shortest paths' convergence, iteration and candidate-overflow status and the snapshot
arena's `'explicit-adapter'` readback.

In the all-entry catalog, an audited empty `status` list means no one-row status view under a
catalog status name. It does not rule out per-row or packed code words, solver `code`, or counts
under other names such as `caseInvalidCount`. Unaudited metadata is `'unknown'`, including records
whose source has not been inspected; it must not be read as an audited absence. The
[catalog status source check](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/ontology/catalog-status-source.node.spec.ts)
checks named exceptions as well as inspectable output types; it is not an audit of every shader write.

A status list is the union of kinds an operator publishes or forwards, including optional views,
nested table ports and recipe stages. It does not promise that every invocation or mode exposes
every kind. For example:

```ts
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

// Unaudited: the recording contract describes copies, but its catalog status is not audited.
getGPUAnalysisOperator({
  package: '@luma-contributors/analysis/computation-record',
  exportName: 'GPUSnapshotTap'
})?.status; // 'unknown'

// Audited absence of catalog scalar kinds: this operator instead has a status code word.
getGPUAnalysisOperator({
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPUAreaOfApplicability'
})?.status; // []

// Includes GPUCellAggregation, GPUCellRollup and GPUCellPyramid table overflow,
// plus addBuildingTessellationRecipe's local recipe-level and reference per-overlay overflow.
queryGPUAnalysisOperators({status: 'overflow'});
// Includes GPUCellRollup and GPUCellPyramid: inherited source loss reaches these outputs.
queryGPUAnalysisOperators({status: 'candidateOverflow'});
// Selects unaudited records; a query for 'overflow' excludes them.
queryGPUAnalysisOperators({status: 'unknown'});
```

For `addBuildingTessellationRecipe`, local mode exposes the recipe-level flag
`result.status.overflow`, alongside `result.status.count`, `result.status.requiredCount`,
`result.status.candidateOverflow` and `result.status.invalidCount`. The default reference layout
returns one polygon-overlay output per building: `result.outputs[i].status.overflow` diagnoses that
overlay's final capacity, alongside `result.outputs[i].status.count`,
`result.outputs[i].status.candidateOverflow` and `result.outputs[i].status.requiredRingCount`.
The separate `result.outputs[i].boundary.status.requiredCount` is the selected boundary-edge
requirement, not an output feature or geometry ring requirement. A caller-owned overlay may omit
`requiredRingCount` and instead supply optional `result.outputs[i].status.ringCount` (stored rings),
`result.outputs[i].status.uncertainCount` (uncertain intersection pairs) and
`result.outputs[i].status.nodedSegmentRequiredCount` (unclamped noded-piece requirement).
Discovery includes those supplied views as well as the default views. These are per-overlay
polygon/ring and boundary-edge diagnostics, rather than a local recipe-level flag. The reference
recipe's own `result.status` contains packed `sampling` status and per-building `complete` flags.
Thus the mode distinction is the port path and scope, not the existence of all count or overflow
views. Check the selected result shape before reading a catalog-listed kind.

The qualified `gpu-spatial-analysis` identities `addBuildingTessellationReferenceRecipe` and
`addLocalBuildingTessellationRecipe` are also discoverable in the `coverage-urban-morphology`
family. Both consume boundary lines, an enclosure polygon, building-ID attributes and parameters;
both produce polygon geometry, attributes and diagnostic summaries. The reference route returns
**dissolved polygon rings** in `result.outputs[i]` with the per-overlay status above. The local
route returns **fixed-stride convex polygon pieces** in `result.outputs.positions` and `result.outputs.cells`,
with packed per-building `result.outputs.buildingStatus` and the five recipe-level scalar views.
Read each `cells` record's first word for that piece's vertex count; its second word is the required
vertex count, followed by source-building and source-sample IDs. A piece can have more than three
vertices. Internal piece edges remain; this representation has no dissolved shell/hole hierarchy.
See the [urban morphology contract](./gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationrecipe--ur01)
for both named imports and output layouts.

The reference route accepts at most 64 buildings, 32 total enclosure vertices and 128 total
boundary samples per call. The local route accepts at most 1,000,000 buildings, 16,777,216 total
samples, 32 total enclosure vertices and 32 enclosure rings per call; per building it admits at
most 32 own samples and 256 candidates, per convex piece at most 128 vertices, and per indexed
query attempt at most 8,192 visited rows. Device storage limits may lower these bounds;
reference overlay capacities and local certification/overflow still apply.

Publisher and forwarding examples use different port scopes:

- `GPUShapeGenerator` publishes optional `vertexCount` at `output.vertexCount`; it appears in
  `queryGPUAnalysisOperators({status: 'vertexCount'})` even when a caller omits that view.
- `GPUCellAggregation` publishes table `output.status.count`, `overflow` and optional `requiredCount`.
  `GPUCellRollup` publishes the same final-output fields and `candidateOverflow` for inherited
  source loss. `GPUCellPyramid` publishes them under `output.levels[i].status`; coarser levels
  carry inherited `candidateOverflow`. A scalar table count is distinct from the packed level-count array.
- `GPUSpatialWeightsTransform` and `GPUSpatialWeightsTranspose` preserve supplied whole weights
  flags under `outputWeights.status`. `GPUChunkedSpatialWeightsTransform` preserves supplied
  outer flags there and per-chunk flags at `outputWeights.chunks[i].weights.status` independently;
  it synthesizes no combined scalar. Discovery includes these optional forwarded kinds.
  Packed transpose excludes input `count` and `requiredCount`: out-of-range columns can be
  dropped. Read `outputWeights.offsets[columnCount]` for the retained slot count. Forwarded
  loss flags describe input completeness, not destination cardinality.
- `GPUSegmentSnap` has per-line `output.status.candidateOverflow` diagnostics, so its audited
  scalar status list remains `[]`; it is excluded from scalar `candidateOverflow` discovery.
- `GPUEmergingHotSpots` advertises optional `count`: `output.count` aliases `includedCount`, the
  exact included input population, rather than emitted rows or per-cell hot/cold slice counts.
- `addSpatialRegressionRecipe` exposes solver `stage.code` in `result.status.stages` for ordinary
  least squares and spatial diagnostics. `stage.status.invalidCount` counts excluded rows in each
  stage's input domain. Optional GWR stages instead expose per-observation `rowCodes`.


Additional status scopes are independent even when their names match:

| Producer and destination | Population or promise |
| --- | --- |
| `addDrainageNetworkRecipe`, `result.status.stages` entry `stage: 'downstream-distance'` | `status.unresolvedCount` counts cells whose downstream distance remains unresolved. |
| `addDrainageNetworkRecipe`, `result.status.stages` entry `stage: 'stream-network'` | `status.unresolvedCount` counts unresolved stream-network cells, separately from the distance population. Do not add the two counts or reinterpret them as successful fallbacks. |
| `addBuildingAdjacencyRecipe` | `result.status.complete` certifies shared-path coverage for the result, including supplied run vertices and candidates. It is not a count or a generic compact-table flag. |
| `GPUIslandAttachment` | Optional `output.status.count` counts emitted CSR links; `candidateOverflow` forwards inherited final/candidate loss and is zero for loss-free input. `uncertainCount` counts fallback islands, not unresolved work. |
| `GPUCellLevelSelection` | Mandatory `count` and optional `requiredCount` count the selected level's occupied rows, not input cells or all pyramid levels. |
| `GPUTrajectoryLocations` | Optional `iterationCount` is the configured round budget (`maximumIterations`). Final `converged` and `iterationLimitReached` report stability after that budget; the counter is not an early-stop count. |

`queryGPUAnalysisOperators({status: 'complete'})` includes building adjacency;
`{status: 'unresolvedCount'}` includes drainage; `{status: 'candidateOverflow'}` includes island
attachment; `{status: 'requiredCount'}` includes selected-level rows; `{status: 'iterationCount'}`
includes trajectory locations. These queries describe supported destinations, including optional
ones, rather than asserting that a caller supplied every destination. Trajectory locations has
no historical spatial descriptor; the historical spatial discovery API stays array-valued.

The historical `getGPUSpatialAnalysisOperator` and `queryGPUSpatialAnalysisOperators` keep their
array-only status type and known-kind filters for compatibility. An unaudited descriptor is projected to `[]` in that historical API, so its empty arrays
can mean omitted metadata. The all-entry API retains `'unknown'` for that descriptor and uses
`[]` only with affirmative audit evidence. Known-kind queries exclude unknown records in the
all-entry API; the historical query keeps its array-membership behavior.

A `candidateOverflow` always written as 0 is not listed. Other mode-specific kinds include
`GPUOffsetCurve` with `output.layout: 'compact'` and `addPatchProximityRecipe` with
`mode: 'boundary'`. `GPUOPTICS` publishes its kinds under `output.ports.status`.

Audited compact ports advertise `candidateOverflow` independently of final-capacity `overflow`.
For publishers that count retained candidates before compaction, `requiredCount` is a lower bound
after candidate loss. Check the publisher's contract: visibility CSR instead publishes `0xffffffff`
for an unknown requirement after candidate search exhaustion; never use it as an allocation size.
The [visibility CSR regression](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/urban-ecology/visibility-graph-csr.spec.ts),
“CSR failures suppress asymmetric graphs and misleading metrics”, distinguishes four required
directed slots from that unknown sentinel. Polygonize and make-valid publish `requiredRingCount`,
with no feature `requiredCount`. `GPUPolygonPrecisionReduction` also exposes delegated
`output.nodedSegments.status.requiredCount` and `output.directedEdges.status.requiredCount`:
these count noded edges and directed edges respectively, not output features or geometry rings.
Gather publishes clamped `status.count`, final-capacity `overflow`,
optional inherited `candidateOverflow` and optional `matchedCount`, with no `requiredCount`.
Nearest-join `neighborRows` and gather's `neighbors.rows` are original feature rows;
`neighborIds` may be caller identities.

Local Moran, local Geary, hot-spot analysis and global spatial statistics advertise the separate
input-weight `overflow` and `candidateOverflow` flags. Local statistics also publish `invalidCount`.
Point-in-polygon joins expose `candidateOverflow` separately; their top-level `status.overflow`
aggregates BVH, candidate and final-match capacity loss, while `matches.status.overflow` reports
only final-match capacity loss. Optional compact matches publish `matches.status.count` and
`matches.status.overflow`; their `requiredCount` view is optional even when `matches` is supplied.
When supplied, `matches.status.requiredCount` counts retained assignments before final compaction;
it is a lower bound on the full requirement after candidate loss.

Nearest-feature weights publish `outputWeights.status.overflow` and optional
`outputWeights.status.candidateOverflow`; they do not publish `requiredCount`. Local variability
retains the input weights' capacity flags in `status.stages` alongside per-stage `invalidCount`.
Cell-flow topology and surface preparation publish stage `invalidCount`, without an `overflow`
port. Discovery status lists describe these actual output or recipe-stage ports, including optional
ports, rather than inheriting the whole recipe family's status vocabulary.

SKATER, Ward, AZP and Max-p enforce a 32,768 estimated-command-node construction budget. This is
admission for the selected rows and budgets, rather than a universal row cap. SKATER compiles at
most `min(maximumRegionCount - 1, rows - 1)` cuts. Tile reprojection estimates
`3 + 2 * sources.length` nodes, admitting at most 16,382 source tiles under the same budget.

Raster limitations state the selected area and grid conditions: terrain derivatives require
orthogonal axes, and ground cell-size modes require axis alignment. Fractional zonal aggregation
supports uniform planar, ground and explicit cell-area units; explicit `cellAreas` require uniform
mode. Embedding pyramid sums/counts retain original support, including cancelling children;
direction validity is separate from support counts. Evidence may include `additionalTests`
beside its primary `tests` path for distinct supported modes.

For example `GPUNetworkFacilityOptimization` is `'reference'`: it enumerates 2^facilities subsets, or
facilities^demand assignments for the capacitated p-median, and rejects more than 2^20 candidates.
The tier describes the code as it is now. When a contributor is reworked to remove its limit, its entry
in `all-entry-points/operator-annotations.ts` changes with it.

Public annotations are applied at final assembly and checked against exactly one public identity.
Internal contributor annotations are kept separately. Limits for `GPUNetworkInhomogeneousK` apply
in every correction mode: at most 1,024 events, 256 radii, 65,536 nodes, 65,536 edges, 1,048,576
event pairs, and 16,777,216 event-by-node distance rows. Event pairs must also fit the caller's
`maximumPairCount` budget. Their reasons name the enforcing source constants.

`GPUParameterBuffer` is listed because its ID is stable, but it is a helper, not an operation. It has
`role: 'helper'` and is left out of `queryGPUAnalysisOperators` unless `includeHelpers` is true.

### Forms, task search and output plans

Qualified records expose `forms.cpu`, `forms.gpu` and `forms.arrays` where recorded. An array
form names its public `package`, `exportName`, `signature`, `supportedInputs` and `differences`.
Graph and array forms may also carry `contract.signature`, `contract.supportedInputs`,
`contract.defaults`, `contract.limits`, `contract.optionHandling` and `contract.source`.
For example, inspect the qualified `GPUPointToCell` record's `forms.cpu.contract` for the named
operation signature, and `forms.arrays` for `computePointToCellOnCPU`'s packed-array signature.
Source expressions are descriptive; `'unknown'` means undisclosed, including option handling,
and does not certify that no default, limit or validation requirement exists.

Read those differences before substituting it for a GPU form: omitted CPU budgets, precision,
tie rules and device limits can differ. `forms.variants` describes prepared or chunked alternatives;
`forms.recipeStages` qualifies each stage and its `appliesWhen` condition. Missing metadata means
undeclared, rather than no alternative or guaranteed parity.

Import the task search and output planner helpers from the public ontology entry:

```ts
import {
  describeOutputs,
  getGPUAnalysisOperator,
  searchGPUAnalysisTasks
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

const similarPixels = searchGPUAnalysisTasks('find similar pixels');
const changed = searchGPUAnalysisTasks('change point', {question: 'what-changed'});
const ndvi = searchGPUAnalysisTasks('NDVI');
// NDVI has owner: 'upstream' and identity:
// {package: '@luma.gl/graph/gpu-raster', exportName: 'GPURasterNDVI'}.
for (const result of changed) {
  if (result.owner === 'analysis') {
    const operator = getGPUAnalysisOperator(result.identity);
    // Inspect operator?.forms and operator?.routes before choosing an execution route.
  }
}

const ols = describeOutputs({
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPUOrdinaryLeastSquares'
}, 'default', {predictorCount: 2, rowCount: 100});
const lines = describeOutputs({
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPULineChunk'
}, 'chunk', {lineCapacity: 5, vertexCapacity: 20});
```

`searchGPUAnalysisTasks('find similar pixels')` finds embedding-pixel search;
`searchGPUAnalysisTasks('change point', {question: 'what-changed'})` distinguishes single-change
testing from segmentation. `searchGPUAnalysisTasks('NDVI')` returns an upstream-owned task with
`owner: 'upstream'`; it does not add an upstream producer to the analysis census. Task results
carry qualified `identity`, `questions` and ranking `score`. The question facet selects the
analysis question; it does not change the producer's calculation or defaults.

For `describeOutputs`, sizes, capacities and active lengths count format rows: one `float32x2`
row holds two float32 elements. A declaration describes available ports, including optional
ports; it does not request their allocation. Unknown layouts return
`{declaration: 'absent', outputs: []}`. An empty list in that result is not an audited claim that
the contributor has no outputs. A known layout rejects unknown modes or missing/invalid sizes.

| Qualified producer | Mode and sizes | Allocation and readable result |
| --- | --- | --- |
| `@luma-contributors/analysis/gpu-spatial-analysis:GPUOrdinaryLeastSquares` | `'default'`, `{predictorCount: 2, rowCount: 100}` | Coefficients, standard errors and t statistics recommend capacity 16, accept minimumCapacity 3, and expose activeLength 3 (intercept first). Summary length is 22; optional fitted/residual/prediction arrays have 100 rows. Inspect `output.status.code` before interpreting the fit. |
| `@luma-contributors/analysis/gpu-spatial-analysis:GPULineChunk` | `'chunk'`, `{lineCapacity: 5, vertexCapacity: 20}` | Six offset rows are reserved; read the offset prefix through `output.status.count + 1`. `output.status.overflow` diagnoses truncation. Substring mode uses `lineCount` instead of `lineCapacity`. |

OLS capacity 16 is a maximum reservation recommendation, not a minimum allocation. Data beyond
active prefixes is unspecified unless the domain contract promises clearing. Line counts refer to
emitted lines; the terminal readable line offset bounds the vertex prefix. Solver codes and
matrix-level loss flags are distinct from compact row counts and per-row validity.


OLS predictions are available for finite predictor rows including masked rows or missing responses;
training-only centers are retained. Invalid predictor rows and failed fits produce NaN predictions.
By contrast, `output.fitted` and `output.residuals` are NaN outside valid training rows and on a
failed fit. A masked row with finite predictors can therefore have a usable prediction while its
training-row outputs are missing. These row rules are separate from the model-wide solver code.

For lines, optional `output.sourceLineIndices` has the tail sentinel
`0xffffffff past output.status.count`. Other tails follow their declared rules.
`output.status.requiredVertexCount` counts vertices for retained pieces only. It is a complete
vertex requirement only when `requiredCount === count` (no line-piece loss); with dropped lines,
it is a lower bound and cannot size the complete vertex result by itself.

The printed line example omits `lineCount` and `positionCount`, so both feasibility constraints
report `checked: false`. They do not certify that the input will fit. Supplying
`{lineCapacity: 5, vertexCapacity: 20, lineCount: 2, positionCount: 8}` checks
`lineCount * (lineCapacity + 1)` = 12 and `2 * lineCapacity + positionCount` = 18 against
4,294,967,295; those two constraints then report `checked: true`. Device bindings, per-frame
piece counts and vertex output capacity still need their own checks.

`routes` are descriptive constraints with a `name`, `kind`, `quantity`, `unit`, `scope`,
`appliesWhen`, source and accuracy implications. A `hard-limit` is an enforced ceiling;
`auto-crossover` describes route selection; `allocation` describes storage growth. For example,
the point-density route records binding bytes, while geographic-distribution metadata describes
its group-count crossover. A route without a `planner` has no executable feasibility decision:
its presence does not certify construction or replace the contributor's validation and device
binding checks. Consult the linked domain contract for route-specific limits and completeness.


#### Source-validated route constraints

These rows state individual host-enforced conditions or route decisions. They are not complete
feasibility plans: passing one row does not certify all bindings, output capacity or runtime
completeness. Boundaries and units below are checked against the corresponding source and catalog
route records; GPU numerical qualification is separate.
Class-breaks, prepared/predicate joins, neighbor search, island attachment, weights traces,
rank Markov, geographic distribution, spatial interaction, cell pyramid, point density, kernel
density and expectation-based scan identify qualified public planners. Resolve each identity
in its named package and call `getExecutionPlan` with that producer's shape and device-limit
contract. The exact-storage rows identify
`@luma-contributors/analysis/dataframe-analysis:GPUClassBreaks`. The network-qualified `addVisibilityGraphRecipe.executionPlanner` member accepts CSR
props and the graph; it is separate from spatial visibility adjacency. Descriptive byte
expressions alone never certify whole-request feasibility, and Gaussian auto eligibility is
decided per frame.

| Qualified producer | Condition and subject | Source-enforced boundary, unit and alternative |
| --- | --- | --- |
| `gpu-spatial-analysis:GPUSpatialInteraction` | `auto-dense-pair-crossover`, auto method with doubly constrained grid calibration; domain scope | Dense through 4,096 total flow pairs when origins and destinations are each at most 128; exceeding any selects sparse. Other dense requests reject above these limits. Sparse has separate convergence checks. |
| `gpu-spatial-analysis:GPUGeographicDistribution` | `auto-chunked-crossover`, `reduction: 'auto'`; domain scope | Chunked through 64 groups, sorted above 64; at most 16,777,215 total input rows on either route. Reduction order changes float32 rounding. |
| `gpu-spatial-analysis:GPUSpatialPredicateJoin` | `engine: 'auto'` for intersects/contains/within on non-point singlepart sides | Relate at average-left-vertices times average-right-vertices >= 1,024. Explicit relate output or non-legacy predicates force relate; dwithin has its distance route. |
| `gpu-spatial-analysis:GPUNeighborSearch` | Auto search, or k-nearest output | Auto chooses grid, including longitude/latitude; explicit BVH requires planar coordinates and no times. In k-nearest mode `k <= 256` neighbors per output row; radius mode uses distance support instead. |
| `gpu-spatial-analysis:GPUCellPyramid` | `input rows` on every route; domain scope | At most 2,147,483,648 total input rows. The `tile rows` route separately caps scratchTileSize at 1,073,741,824 rows per tile. Requested level-count scratch needs `resolutions.length * 64 * 4` bytes within both device buffer limits; bindings and command budgets remain separate. |
| `gpu-spatial-analysis:GPUPointDensity` | `workgroup` weighted sums/means requested by statistic or output ports, sumAccumulation auto/workgroup; binding scope | `gridSize[0] * gridSize[1] * 12 <= min(134217728, device.maxStorageBufferBindingSize, device.maxBufferSize)` bytes: at most 11,184,810 whole-grid cells before tighter device limits. Input chunking does not lift this limit. Larger weighted grids require explicit atomic accumulation, whose float rounding is order dependent. |
| `gpu-spatial-analysis:GPUKernelDensity` | Every method's index grid | `(indexGridSize[0] * indexGridSize[1] + 1) * 4` bytes must fit maxStorageBufferBindingSize. Binned is Gaussian-only and has additional padded convolution constraints. |
| `dataframe-analysis:GPUClassBreaks` | Natural breaks enabled, exact mode, `materialized` storage; domain scope | Qualified `GPUClassBreaks` planner: at most 1,024 total value rows. Recompute retains exact dynamic programming with less storage and more work. |
| `dataframe-analysis:GPUClassBreaks` | Natural breaks enabled, exact mode, `recompute` storage (or auto above 1,024); domain scope | Qualified `GPUClassBreaks` planner: at most 8,192 total value rows. Monotone and approximate alternatives can change boundaries. |
| `gpu-spatial-analysis:GPUExpectationBasedSpaceTimeScan` | Auto nearest-zone search | Indexed when maximumNeighbors < zone count and <= 65; the producer independently limits maximumNeighbors to 64 and maximumWindowLength to 32. Full scan does not lift either bound. |
| `gpu-terrain:GPUTerrainTiledFlow` | Command budget of the route selected on the graph device | At most 32,768 commands total. For 6,452 one-cell tiles and one sweep, selected resident estimates 12,932 commands; separately named `worstCaseCommandNodeCount` is 32,772. Selected tiled admits 6,451 tiles (32,767 commands), rejects 6,452. |
| `gpu-network:addVisibilityGraphRecipe` | CSR automatic metrics batches | Auto searches smaller feasible whole batches under lane divisibility, allocation/binding bytes and at most 32,768 total recipe commands. Use the network recipe's `executionPlanner` adapter; sampled maximum depth is a lower bound. |
| `raster-analysis:GPURasterFractionalCategoricalStatistics` | Zone/category groups including missing category | `zoneCount * (categoryCapacity + 1) <= 4,294,967,294` for tiled, <= 65,535 for exact; output binding and whole-recipe schedule checks are independent. |

Producer packages are `@luma-contributors/analysis/<entry>`. These route records have source
paths, scopes and accuracy implications; only a validated family planner can certify its own
complete set of admission conditions.

Local Moran's executed route reports `pValueConstruction`: the CPU reference counts absolute
exceedances once (`upper-tail-count`), whereas the GPU uses `doubled-minimum-tail`. With 19 draws
their attainable floors are 0.05 and 0.10 respectively, including CPU fallback. Point-column
`spacing.invalidCount` includes rejected coordinate pairs before distance sampling; `rowCount`
counts source rows, `pointCount` counts accepted locations and `sampleSize` counts the distance
sample.

### Querying it

```ts
import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

// Everything that is limited to small inputs.
queryGPUAnalysisOperators({tier: 'reference'});

// Temporal operators of the raster entry point.
queryGPUAnalysisOperators({
  package: '@luma-contributors/analysis/raster-analysis',
  dimensionality: '2d+time'
});

// One operator, with its limits.
getGPUAnalysisOperator('GPUNetworkFlow')?.limits;

// A name shared by two entry points needs its import subpath.
getGPUAnalysisOperator({
  package: '@luma-contributors/analysis/gpu-network',
  exportName: 'addVisibilityGraphRecipe'
});
```

`queryGPUAnalysisOperators` also accepts the filters of `queryGPUSpatialAnalysisOperators`
(`primaryCapability`, `input`, `output`, `coordinateModel`, `semantics`, `evidence`, `search`),
plus `status`, `readback` and `tier`. Gaussian random fields are discoverable through both point
and surface inputs and outputs, reflecting their point-sampling and grid modes.

`GPU_SPATIAL_ANALYSIS_OPERATORS`, `GPU_SPATIAL_ANALYSIS_CAPABILITIES` and the functions that read them
still describe `gpu-spatial-analysis` (and `GPUTerrainPeakProminence`) only. They remain historical
entry-point APIs; the website inventory generator uses `getGPUAnalysisOperatorCatalog()` and
`getGPUAnalysisCapabilityCatalog()` for all entries. Operators of the other entry points come from
`getGPUAnalysisOperatorCatalog()`. The original operators also carry `tier`,
`limits` and `role` now.

Operators added to the catalog after the first families do not repeat the family scale figures
(`representativeSourceRows`, `transientMemory`, `dispatch`); their `evidence` records tests and
documentation, with `status: 'partial'` and `contract: 'unaudited'` until a contract is assigned.

## Public class reference targets

Use the entry suffix and export name together. An entry suffix below means
`@luma-contributors/analysis/<entry>`; `.` means the package root. These links supplement the
existing class headings, including producers documented in family sections. They are documentation
navigation targets, separate from the catalog's repository-relative `evidence.documentation`
file paths, which remain suitable for file-existence checks. Same-name exports stay qualified;
the two visibility recipes retain distinct spatial and network contracts:

| Entry | Exports | Reference |
| --- | --- | --- |
| `gpu-spatial-analysis` | `addVisibilityGraphRecipe` | [Spatial adjacency](./gpu-spatial-analysis/urban-morphology.md#addisovistsrecipe-and-addvisibilitygraphrecipe--ur05) |
| `gpu-network` | `addVisibilityGraphRecipe` | [Network metrics](./gpu-network/visibility-graph.md#addvisibilitygraphrecipe) |

### Canonical prose contracts

Recording anchors remain available. These qualified links resolve directly to contract prose,
including grouped contracts, rather than a recording marker or generated Parameters heading.

| Entry | Exports | Reference |
| --- | --- | --- |
| `gpu-network` | `GPUNetworkInhomogeneousK` | [Inputs, outputs and limits](./gpu-spatial-analysis/point-process.md#inhomogeneous-network-k) |
| `raster-analysis` | `GPURasterFocalPairs` | [Inputs, outputs and limits](./gpu-raster/focal-pairs.md#focal-cross-band-statistics) |
| `raster-analysis` | `GPURasterConservativeRemapping` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#conservative-remapping-rs03) |
| `raster-analysis` | `GPURasterTexture` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#glcm-texture-rs04) |
| `raster-analysis` | `GPURasterTemporal` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#temporal-stacks-rs06) |
| `raster-analysis` | `GPURasterRegistration` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#translation-registration-rs07) |
| `raster-analysis` | `GPURasterSpectralUnmixing` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#spectral-unmixing-rs10) |
| `raster-analysis` | `GPURasterSeasonalBreak` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#seasonal-decomposition-breaks-and-monitoring-rs11) |
| `raster-analysis` | `GPURasterNDWI` | [Inputs, outputs and limits](./gpu-raster/spectral-indices.md#formula-and-band-contracts) |
| `raster-analysis` | `GPURasterNBR` | [Inputs, outputs and limits](./gpu-raster/spectral-indices.md#formula-and-band-contracts) |
| `raster-analysis` | `GPURasterEVI` | [Inputs, outputs and limits](./gpu-raster/spectral-indices.md#formula-and-band-contracts) |
| `raster-analysis` | `GPURasterSAVI` | [Inputs, outputs and limits](./gpu-raster/spectral-indices.md#formula-and-band-contracts) |
| `gpu-spatial-analysis` | `GPUInstrumentalVariablesCovariance` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#iv-covariance-and-robust-spatial-2sls) |
| `gpu-spatial-analysis` | `GPURestrictedRegimeRegression` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) |
| `gpu-spatial-analysis` | `GPUProbitRegression` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity) |
| `gpu-spatial-analysis` | `GPULocalCollinearity` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity) |
| `gpu-spatial-analysis` | `GPUFuzzyContiguity` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) |
| `gpu-spatial-analysis` | `GPUIslandAttachment` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) |
| `gpu-spatial-analysis` | `GPUSpatialCrossValidation` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) |
| `gpu-spatial-analysis` | `GPUSpatialCrossValidatedOLS` | [Inputs, outputs and limits](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) |
| `gpu-spatial-analysis` | `GPUS2PolygonCovering` | [Inputs, outputs and limits](./gpu-spatial-analysis/global-coordinates.md#adaptive-polygon-covers) |
| `gpu-spatial-analysis` | `GPUSphericalPolygonContains` | [Inputs, outputs and limits](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) |
| `gpu-spatial-analysis` | `GPUSphericalPolygonOverlay` | [Inputs, outputs and limits](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) |
| `gpu-spatial-analysis` | `GPUSphericalArcOverlay` | [Inputs, outputs and limits](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) |
| `gpu-spatial-analysis` | `GPUInhomogeneousPointPattern` | [Inputs, outputs and limits](./gpu-spatial-analysis/point-process.md#intensity-and-polygon-corrections) |
| `gpu-spatial-analysis` | `GPUPointPatternEnvelope` | [Inputs, outputs and limits](./gpu-spatial-analysis/point-process.md#simulation-curves-and-global-tests) |
| `gpu-spatial-analysis` | `GPUSpatialBalancedSample` | [Inputs, outputs and limits](./gpu-spatial-analysis/point-process.md#spatial-probability-sampling) |
| `gpu-spatial-analysis` | `GPUOPTICS` | [Inputs, outputs and limits](./gpu-spatial-analysis/point-process.md#optics-ordering) |
| `gpu-terrain` | `GPUTerrainFlowFractions` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#branching-consumers) |
| `gpu-terrain` | `GPUTerrainDepressionHierarchy` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#depression-hierarchy-and-stage-curves) |
| `gpu-terrain` | `GPUTerrainCatchmentSummary` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#basin-summaries-and-profiles) |
| `gpu-terrain` | `GPUTerrainDEMCoregistration` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#dem-comparison) |
| `gpu-spatial-analysis` | `GPULocalGeary` | [Inputs, outputs and limits](./gpu-spatial-analysis/operators-statistics.md#gpulocalgeary) |
| `gpu-spatial-analysis` | `GPUSpatialPearson` | [Inputs, outputs and limits](./gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic) |
| `gpu-spatial-analysis` | `GPUGammaStatistic` | [Inputs, outputs and limits](./gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic) |
| `gpu-spatial-analysis` | `GPUNearestFeatureWeights` | [Inputs, outputs and limits](./gpu-spatial-analysis/operators-joins.md#gpunearestfeatureweights) |
| `raster-analysis` | `GPURasterFillHoles` | [Inputs, outputs and limits](./gpu-raster/expansion.md#operations-and-contracts) |
| `raster-analysis` | `GPURasterFractionalZonalDistribution` | [Inputs, outputs and limits](./gpu-raster/fractional-zonal-distribution.md#gpurasterfractionalzonaldistribution) |
| `gpu-spatial-analysis` | `GPUSpatialWeightsAlgebra` | [Inputs, outputs and limits](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightsalgebra) |

| `gpu-terrain` | `GPUTerrainBreach` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#breaching-and-explicit-burning) |
| `gpu-terrain` | `GPUTerrainBranchingFlow` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#branching-consumers) |
| `gpu-terrain` | `GPUTerrainTiledFlow` | [Inputs, outputs and limits](./gpu-terrain/terrain-processing.md#cross-tile-d8-and-branching-accumulation-and-reach-ids) |
| `raster-analysis` | `GPURasterRankFilter` | [Inputs, outputs and limits](./gpu-raster/expansion.md#operations-and-contracts) |
| `raster-analysis` | `GPURasterReconstruction` | [Inputs, outputs and limits](./gpu-raster/expansion.md#operations-and-contracts) |
| `raster-analysis` | `GPURasterSkeleton` | [Inputs, outputs and limits](./gpu-raster/expansion.md#operations-and-contracts) |
| `raster-analysis` | `GPURasterWatershedSegmentation` | [Inputs, outputs and limits](./gpu-raster/expansion.md#operations-and-contracts) |
| `raster-analysis` | `GPURasterRegionGraph` | [Inputs, outputs and limits](./gpu-raster/remote-sensing.md#region-adjacency-and-merging-rs08) |

### Fitting producers

| Entry | Exports | Reference |
| --- | --- | --- |
| `analysis-core` | `GPUFittingCoordinateEncoder` | [Contract](./analysis-core/fitting-kernels.md#gpufittingcoordinateencoder) |
| `analysis-core` | `GPUFittingGradientSums` | [Contract](./analysis-core/fitting-kernels.md#gpufittinggradientsums) |
| `analysis-core` | `GPUFittingMinibatchOrder` | [Contract](./analysis-core/fitting-kernels.md#gpufittingminibatchorder) |
| `analysis-core` | `GPUAdamStep` | [Contract](./analysis-core/fitting-kernels.md#gpuadamstep) |

### Producers documented with a family

The linked section describes the named producer's ports, algorithm or supported modes; a generated
Parameters heading alone is not a reference target. Recording taps use the recording contracts.

| Entry | Exports | Reference |
| --- | --- | --- |
| `.` | `GPUSnapshotArena` | [Contract](./computation-record.md#snapshots) |
| `.` | `GPUSnapshotTap` | [Contract](./computation-record.md#snapshots) |
| `.` | `GPUComputationRecorder` | [Contract](./computation-record.md#recording-a-command-graph) |
| `computation-record` | `GPUSnapshotArena` | [Contract](./computation-record.md#snapshots) |
| `computation-record` | `GPUSnapshotTap` | [Contract](./computation-record.md#snapshots) |
| `computation-record` | `GPUComputationRecorder` | [Contract](./computation-record.md#recording-a-command-graph) |
| `dataframe-analysis` | `GPUSpaceTimeKFunction` | [Dataframe pair-statistics contract](./gpu-spatial-analysis/space-time-density-and-k-function.md#space-time-k-pair-correlation-local-statistics-gpuspacetimekfunction) |
| `dataframe-analysis` | `GPUSpaceTimeDiggleTest` | [Dataframe interaction-test contract](./gpu-spatial-analysis/space-time-density-and-k-function.md#diggle-test-of-space-time-interaction-gpuspacetimediggletest) |
| `gpu-spatial-analysis` | `GPUTransitionDiagnostics` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix) |
| `gpu-spatial-analysis` | `GPURandomPointsOnLine` | [Contract](./gpu-spatial-analysis/operators-density.md#gpudotdensity-and-gpurandompointsinpolygon) |
| `gpu-spatial-analysis` | `GPURoseStatistic` | [Contract](./gpu-spatial-analysis/operators-statistics.md#additional-spatial-statistics-and-bounded-optimization) |
| `gpu-spatial-analysis` | `GPULocationAllocation` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocationallocation) |
| `gpu-spatial-analysis` | `GPURegionMask` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics) |
| `gpu-spatial-analysis` | `GPUPickRegionMask` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics) |
| `gpu-spatial-analysis` | `GPULocalRBF` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpulocalrbf-local-radial-basis-interpolation) |
| `gpu-spatial-analysis` | `GPUNaturalNeighbor` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpunaturalneighbor-natural-neighbors) |
| `gpu-spatial-analysis` | `GPUConditionalGaussianSimulation` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpuconditionalgaussiansimulation-conditional-gaussian-simulation) |
| `gpu-spatial-analysis` | `GPUAZPRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuazpregions) |
| `gpu-spatial-analysis` | `GPUWardRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuwardregions) |
| `gpu-spatial-analysis` | `GPUMaxPRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpumaxpregions) |
| `gpu-spatial-analysis` | `GPUSpatialPanelLagMaximumLikelihood` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#random-and-unit-effects-spatial-lag-maximum-likelihood) |
| `gpu-spatial-analysis` | `GPUTrackSimilaritySearch` | [Contract](./gpu-spatial-analysis/track-similarity-measures.md#lower-bounds-and-pruned-search) |
| `gpu-spatial-analysis` | `GPUTrajectoryKalman` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-spatial-analysis` | `GPUTrajectoryRestriction` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-spatial-analysis` | `GPUNestedInterpolationValidation` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#nested-variogram-and-bandwidth-selection) |
| `gpu-spatial-analysis` | `GPUAreaOfApplicability` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#area-of-applicability) |
| `gpu-network` | `GPUNetworkFlow` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-network` | `GPUNetworkTimeRouting` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-network` | `GPUNetworkPointRouting` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-network` | `GPUNetworkFacilityOptimization` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-network` | `GPUNetworkRationalAccess` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) |
| `gpu-network` | `GPUAlternativeRoutes` | [Contract](./gpu-network/alternative-routes.md#alternative-routes) |
| `gpu-network` | `GPUMapMatchedEdgeVolumes` | [Contract](./gpu-network/map-matched-edge-volumes.md#edge-volumes-of-map-matched-tracks) |
| `gpu-network` | `GPUNetworkStructure` | [Contract](./gpu-network/expansion.md#directed-components-and-undirected-resilience) |
| `gpu-network` | `GPUNetworkStronglyConnectedComponents` | [Contract](./gpu-network/expansion.md#large-graph-strongly-connected-components) |
| `raster-analysis` | `GPURasterEmbeddingCovariance` | [Contract](./gpu-raster/raster-embeddings.md#pca-to-rgb) |
| `raster-analysis` | `GPURasterEmbeddingProjection` | [Contract](./gpu-raster/raster-embeddings.md#pca-to-rgb) |
| `raster-analysis` | `GPURasterStandardizedIndexCalibration` | [Contract](./gpu-raster/raster-time-series.md#spi-and-spei) |
| `raster-analysis` | `GPURasterStandardizedIndex` | [Contract](./gpu-raster/raster-time-series.md#spi-and-spei) |

### Infrastructure reference

These helpers manage host state, allocation or graph integration rather than representing an
independent numerical operator. Recording infrastructure uses the same recording contracts as taps.

| Entry | Exports | Reference |
| --- | --- | --- |
| `.` | `GPUSnapshotArena` | [Contract](./computation-record.md#snapshots) |
| `.` | `GPUComputationRecorder` | [Contract](./computation-record.md#recording-a-command-graph) |
| `computation-record` | `GPUSnapshotArena` | [Contract](./computation-record.md#snapshots) |
| `computation-record` | `GPUComputationRecorder` | [Contract](./computation-record.md#recording-a-command-graph) |
| `gpu-spatial-analysis` | `GPURegionStatisticsReadback` | [Contract](#gpuregionstatisticsreadback) |
| `gpu-spatial-analysis/ontology` | `AnalysisGeographicExtentAccumulator` | [Contract](#analysisgeographicextentaccumulator) |
| `raster-analysis` | `FlowFieldViewportTracker` | [Contract](#flowfieldviewporttracker) |
| `analysis-core` | `AnalysisValidationError` | [Contract](#analysisvalidationerror) |
| `analysis-core` | `GPUParameterBuffer` | [Contract](#gpuparameterbuffer) |
| `analysis-core` | `GPUContributorOperation` | [Contract](#gpucontributoroperation) |

#### `AnalysisValidationError`

From `@luma-contributors/analysis/analysis-core`, `AnalysisValidationError` classifies a
construction-time refusal. Inspect `code` and `path` to identify the failed precondition;
`received` preserves the caller's value and `expected` describes the required relationship.
`id` identifies the instance. `contributor` records an explicit producer identity when supplied;
legacy string identities carry that string rather than an inferred class name. The ordinary
error message still begins with the instance ID. Correct the named input before constructing
again; increasing output capacity does not repair an invalid option or malformed input view.

#### `GPUParameterBuffer`

From `analysis-core`, `GPUParameterBuffer` owns a packed `float32`, `uint32` or `sint32` storage
buffer. `importToGraph(graph)` returns its graph view; `write(values, firstElement?)` uploads a
matching typed-array range for subsequent encodings. Destroy compiled graphs that use the buffer
before calling `destroy()`. See [parameter updates](./gpu-spatial-analysis/concepts.md#conventions)
and the [fitting example](./analysis-core/fitting-kernels.md#gpuadamstep).

#### `GPUContributorOperation`

From `analysis-core`, `GPUContributorOperation` adapts a contributor to a named-value `GPUGraph`
operation: it resolves graph inputs and outputs and contributes the GPU command nodes. Use the
concrete domain wrappers listed below, or supply the bridge's contributor factory when authoring
an operation. The [operation forms guide](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu)
describes packed-value support and the asynchronous mixed-execution boundary.

#### `GPURegionStatisticsReadback`

From `gpu-spatial-analysis`, `GPURegionStatisticsReadback` owns a staging ring for the packed
region summary. Call `encodeRead` after encoding the computation; a `null` ticket means
backpressure, so drop that frame. After submitting, `read(ticket)` awaits and decodes the summary.
Call `destroy()` when done. See [region summary layout](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics).

#### `AnalysisGeographicExtentAccumulator`

From `gpu-spatial-analysis/ontology`,
`AnalysisGeographicExtentAccumulator` accepts longitude/latitude with `add`, or literal edge paths
with `addPath(positions, close)`. `getPlain()` returns source bounds and `getCovering()` returns a
covering longitude arc, possibly with west greater than east across the antimeridian. Both return
`undefined` without finite positions. Edges retain their written longitude intervals; they are not
implicitly replaced by shortest great-circle edges. The accumulator retains intervals and sorts
them when read; it is a host helper, with no GPU dispatch or implicit readback.

#### `FlowFieldViewportTracker`

From `raster-analysis`, `FlowFieldViewportTracker.update(desired, active, timestamp)` reports when
a desired field rectangle has settled for 150 ms. It ignores edge changes within 2% of the active
extent to avoid repeatedly reseeding particles during camera jitter. Keep one tracker across
updates and pass timestamps in milliseconds. It tracks viewport changes only; the caller handles
source and filter bound changes separately. It allocates no GPU buffers.

### Operation forms reference

The following shared sections name the concrete exported forms by their owning entry. In each row,
`CPU*Operation` evaluates CPU arrays for named graph values; `GPU*Operation` resolves named values
to the corresponding GPU contributor. Both take the domain operation props. The exceptional
location-allocation wrapper is `GPULocationAllocationGraphOperation`, paired with
`CPULocationAllocationOperation`; `GPULocationAllocationOperation` is the existing union type
`'p-median' | 'maximum-coverage' | 'set-covering'`.
Use the [operation forms guide](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for the
packed-value, transfer, precision and execution contract; consult each contributor's family
reference for output layouts, supported modes and limits. This index does not assert equal CPU/GPU
implementation ceilings or bit-identical arithmetic.

#### Operation forms: gpu-spatial-analysis

Use the [shared lifecycle](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for transfers and execution. Each qualified row links its domain contract; array links open the exported function’s source signature and precision notes.

| Entry | Exports | Reference | Pure CPU function |
| --- | --- | --- | --- |
| `gpu-spatial-analysis` | `CPUPettittTestOperation`, `GPUPettittTestOperation` | [Domain contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#pettitt-test-gpupettitttest) | [`computePettittTestOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation.ts) |
| `gpu-spatial-analysis` | `CPULocationChangePointsOperation`, `GPULocationChangePointsOperation` | [Domain contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#per-location-change-points-gpulocationchangepoints) | [`computeLocationChangePointsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation.ts) |
| `gpu-spatial-analysis` | `CPUPointToCellOperation`, `GPUPointToCellOperation` | [Domain contract](./gpu-spatial-analysis/operators-cells.md#gpupointtocell-and-gpucellgeometry) | [`computePointToCellOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts) |
| `gpu-spatial-analysis` | `CPUCellMeasuresOperation`, `GPUCellMeasuresOperation` | [Domain contract](./gpu-spatial-analysis/operators-cells.md#gpucellgridpath-and-gpucellmeasures) | [`computeCellMeasuresOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation.ts) |
| `gpu-spatial-analysis` | `CPUS2CapCoveringOperation`, `GPUS2CapCoveringOperation` | [Domain contract](./gpu-spatial-analysis/global-coordinates.md#operation-forms-s2-covers-and-intrinsic-arc-overlay) | [`computeS2CapCoveringOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation.ts) |
| `gpu-spatial-analysis` | `CPUS2PolygonCoveringOperation`, `GPUS2PolygonCoveringOperation` | [Domain contract](./gpu-spatial-analysis/global-coordinates.md#adaptive-polygon-covers) | [`computeS2PolygonCoveringOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation.ts) |
| `gpu-spatial-analysis` | `CPUDelaunayTessellationOperation`, `GPUDelaunayTessellationOperation` | [Domain contract](./gpu-spatial-analysis/operators-geometry.md#gpudelaunaytessellation-and-gpuvoronoidiagram) | [`computeDelaunayTessellationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation.ts) |
| `gpu-spatial-analysis` | `CPUTransitionDiagnosticsOperation`, `GPUTransitionDiagnosticsOperation` | [Domain contract](./gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix) | [`computeTransitionDiagnosticsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation.ts) |
| `gpu-spatial-analysis` | `CPURankMarkovOperation`, `GPURankMarkovOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#joint-randomization-and-rank-dynamics) | [`computeRankMarkovOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation.ts) |
| `gpu-spatial-analysis` | `CPULocalFrameProjectionOperation`, `GPULocalFrameProjectionOperation` | [Domain contract](./gpu-spatial-analysis/operators-lines.md#geodesic_wgsl-creategpulocalframe-and-gpulocalframeprojection) | [`computeLocalFrameProjectionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation.ts) |
| `gpu-spatial-analysis` | `CPUGeodesicPairsOperation`, `GPUGeodesicPairsOperation` | [Domain contract](./gpu-spatial-analysis/geodesic-precision.md#gpugeodesicpairs) | [`computeGeodesicPairsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation.ts) |
| `gpu-spatial-analysis` | `CPUHealpixPointToCellOperation`, `GPUHealpixPointToCellOperation` | [Domain contract](./gpu-spatial-analysis/healpix-cells-dggs-json.md#point-to-cell-gpuhealpixpointtocell) | [`computeHealpixPointToCellOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation.ts) |
| `gpu-spatial-analysis` | `CPULabelPointOperation`, `GPULabelPointOperation` | [Domain contract](./gpu-spatial-analysis/operators-geometry.md#gpulabelpoint) | [`computeLabelPointOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/label-point/cpu-label-point-operation.ts) |
| `gpu-spatial-analysis` | `CPULineSimplificationOperation`, `GPULineSimplificationOperation` | [Domain contract](./gpu-spatial-analysis/operators-lines.md#gpulinesimplification) | [`computeLineSimplificationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation.ts) |
| `gpu-spatial-analysis` | `CPULocationAllocationOperation`, `GPULocationAllocationGraphOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpulocationallocation) | [`computeLocationAllocationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation.ts) |
| `gpu-spatial-analysis` | `CPUMapColoringOperation`, `GPUMapColoringOperation` | [Domain contract](./gpu-spatial-analysis/operators-weights.md#gpumapcoloring) | [`computeMapColoringOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation.ts) |
| `gpu-spatial-analysis` | `CPUMinimumBoundsOperation`, `GPUMinimumBoundsOperation` | [Domain contract](./gpu-spatial-analysis/operators-geometry.md#gpuminimumbounds) | [`computeMinimumBoundsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation.ts) |
| `gpu-spatial-analysis` | `CPUMultipleTestingAdjustmentOperation`, `GPUMultipleTestingAdjustmentOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpumultipletestingadjustment) | [`computeMultipleTestingAdjustmentOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation.ts) |
| `gpu-spatial-analysis` | `CPUNeighborhoodSummaryOperation`, `GPUNeighborhoodSummaryOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpuneighborhoodsummary) | [`computeNeighborhoodSummaryOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation.ts) |
| `gpu-spatial-analysis` | `CPUConstrainedDelaunayTriangulationOperation`, `GPUConstrainedDelaunayTriangulationOperation` | [Domain contract](./gpu-raster/expansion.md#operations-and-contracts) | [`computeConstrainedDelaunayTriangulationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation.ts) |
| `gpu-spatial-analysis` | `CPUPolygonTriangulationOperation`, `GPUPolygonTriangulationOperation` | [Domain contract](./gpu-spatial-analysis/operators-geometry.md#gpupolygontriangulation) | [`computePolygonTriangulationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation.ts) |
| `gpu-spatial-analysis` | `CPUEnsembleScoresOperation`, `GPUEnsembleScoresOperation` | [Domain contract](./gpu-spatial-analysis/scoring-rules.md#gpuensemblescores) | [`computeEnsembleScoresOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts) |
| `gpu-spatial-analysis` | `CPUSinkhornDivergenceOperation`, `GPUSinkhornDivergenceOperation` | [Domain contract](./gpu-spatial-analysis/sinkhorn-divergence.md#gpusinkhorndivergence) | [`computeSinkhornDivergenceOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpaceTimeHawkesOperation`, `GPUSpaceTimeHawkesOperation` | [Domain contract](./gpu-spatial-analysis/space-time-hawkes.md#gpuspacetimehawkes) | [`computeSpaceTimeHawkesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation.ts) |
| `gpu-spatial-analysis` | `CPUShapeDescriptorsOperation`, `GPUShapeDescriptorsOperation` | [Domain contract](./gpu-spatial-analysis/operators-geometry.md#gpushapedescriptors) | [`computeShapeDescriptorsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation.ts) |
| `gpu-spatial-analysis` | `CPUMantelTestOperation`, `GPUMantelTestOperation` | [Domain contract](./gpu-spatial-analysis/operators-dynamics.md#gpuknoxtest-and-gpumanteltest) | [`computeMantelTestOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation.ts) |
| `gpu-spatial-analysis` | `CPUOPTICSOperation`, `GPUOPTICSOperation` | [Domain contract](./gpu-spatial-analysis/point-process.md#optics-ordering) | [`computeOPTICSOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation.ts) |
| `gpu-spatial-analysis` | `CPUConditionalGaussianSimulationOperation`, `GPUConditionalGaussianSimulationOperation` | [Domain contract](./gpu-spatial-analysis/interpolation-expansion.md#gpuconditionalgaussiansimulation-conditional-gaussian-simulation) | [`computeConditionalGaussianSimulationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation.ts) |
| `gpu-spatial-analysis` | `CPUVecchiaFisherScoringOperation`, `GPUVecchiaFisherScoringOperation` | [Domain contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchiafisherscoring) | [`computeVecchiaFisherScoringOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation.ts) |
| `gpu-spatial-analysis` | `CPUVecchiaLogLikelihoodOperation`, `GPUVecchiaLogLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchialoglikelihood) | [`computeVecchiaLogLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation.ts) |
| `gpu-spatial-analysis` | `CPUVecchiaSpaceTimeLogLikelihoodOperation`, `GPUVecchiaSpaceTimeLogLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchiaspacetimeloglikelihood) | [`computeVecchiaSpaceTimeLogLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation.ts) |
| `gpu-spatial-analysis` | `CPUWardRegionsOperation`, `GPUWardRegionsOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpuwardregions) | [`computeWardRegionsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation.ts) |
| `gpu-spatial-analysis` | `CPUSkaterRegionsOperation`, `GPUSkaterRegionsOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | [`computeSkaterRegionsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation.ts) |
| `gpu-spatial-analysis` | `CPUAZPRegionsOperation`, `GPUAZPRegionsOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpuazpregions) | [`computeAZPRegionsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation.ts) |
| `gpu-spatial-analysis` | `CPUMaxPRegionsOperation`, `GPUMaxPRegionsOperation` | [Domain contract](./gpu-spatial-analysis/operators-statistics.md#gpumaxpregions) | [`computeMaxPRegionsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation.ts) |
| `gpu-spatial-analysis` | `CPUOrdinaryLeastSquaresOperation`, `GPUOrdinaryLeastSquaresOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuordinaryleastsquares) | [`computeOrdinaryLeastSquaresOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation.ts) |
| `gpu-spatial-analysis` | `CPURestrictedRegimeRegressionOperation`, `GPURestrictedRegimeRegressionOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | [`computeRestrictedRegimeRegressionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation.ts) |
| `gpu-spatial-analysis` | `CPUSeeminglyUnrelatedRegressionOperation`, `GPUSeeminglyUnrelatedRegressionOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | [`computeSeeminglyUnrelatedRegressionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialLagImpactsOperation`, `GPUSpatialLagImpactsOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatiallagimpacts) | [`computeSpatialLagImpactsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialRegressionDiagnosticsOperation`, `GPUSpatialRegressionDiagnosticsOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialregressiondiagnostics) | [`computeSpatialRegressionDiagnosticsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialWeightsTracesOperation`, `GPUSpatialWeightsTracesOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialweightstraces) | [`computeSpatialWeightsTracesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation.ts) |
| `gpu-spatial-analysis` | `CPUMultiscaleGeographicallyWeightedRegressionOperation`, `GPUMultiscaleGeographicallyWeightedRegressionOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#mgwr-bandwidth-selection-and-limits) | [`computeMultiscaleGeographicallyWeightedRegressionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialErrorGMOperation`, `GPUSpatialErrorGMOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialerrorgm) | [`computeSpatialErrorGMOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation.ts) |
| `gpu-spatial-analysis` | `CPUProbitRegressionOperation`, `GPUProbitRegressionOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity) | [`computeProbitRegressionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialTwoStageLeastSquaresOperation`, `GPUSpatialTwoStageLeastSquaresOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialtwostageleastsquares) | [`computeSpatialTwoStageLeastSquaresOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialErrorMaximumLikelihoodOperation`, `GPUSpatialErrorMaximumLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialerrormaximumlikelihood) | [`computeSpatialErrorMaximumLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialLagMaximumLikelihoodOperation`, `GPUSpatialLagMaximumLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/operators-regression.md#gpuspatiallagmaximumlikelihood) | [`computeSpatialLagMaximumLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialPanelLagMaximumLikelihoodOperation`, `GPUSpatialPanelLagMaximumLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#random-and-unit-effects-spatial-lag-maximum-likelihood) | [`computeSpatialPanelLagMaximumLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation.ts) |
| `gpu-spatial-analysis` | `CPUContiguityWeightsOperation`, `GPUContiguityWeightsOperation` | [Domain contract](./gpu-spatial-analysis/operators-weights.md#gpucontiguityweights) | [`computeContiguityWeightsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation.ts), [`computePolygonContiguityOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology.ts) |
| `gpu-spatial-analysis` | `CPUSpatialWeightsValidationOperation`, `GPUSpatialWeightsValidationOperation` | [Domain contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightsvalidation) | [`computeSpatialWeightsValidationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialWeightsAlgebraOperation`, `GPUSpatialWeightsAlgebraOperation` | [Domain contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightsalgebra) | [`computeSpatialWeightsAlgebraOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrackSimilarityOperation`, `GPUTrackSimilarityOperation` | [Domain contract](./gpu-spatial-analysis/operators-lines.md#gputracksimilarity) | [`computeTrackSimilarityOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectorySplitOperation`, `GPUTrajectorySplitOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorysplit) | [`computeTrajectorySplitOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryThinningOperation`, `GPUTrajectoryThinningOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorythinning) | [`computeTrajectoryThinningOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation.ts) |
| `gpu-spatial-analysis` | `CPUBrownianBridgeUtilizationOperation`, `GPUBrownianBridgeUtilizationOperation` | [Domain contract](./gpu-spatial-analysis/home-ranges-and-recursions.md#gpubrownianbridgeutilization) | [`computeBrownianBridgeUtilizationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation.ts) |
| `gpu-spatial-analysis` | `CPUMobilityIndicatorsOperation`, `GPUMobilityIndicatorsOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gpumobilityindicators) | [`computeMobilityIndicatorsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation.ts) |
| `gpu-spatial-analysis` | `CPUModalSplitOperation`, `GPUModalSplitOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gpumodalsplit) | [`computeModalSplitOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryRecursionsOperation`, `GPUTrajectoryRecursionsOperation` | [Domain contract](./gpu-spatial-analysis/home-ranges-and-recursions.md#gputrajectoryrecursions) | [`computeTrajectoryRecursionsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryStaypointsOperation`, `GPUTrajectoryStaypointsOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorystaypoints) | [`computeTrajectoryStaypointsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryLocationsOperation`, `GPUTrajectoryLocationsOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorylocations) | [`computeTrajectoryLocationsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectorySegmentsOperation`, `GPUTrajectorySegmentsOperation` | [Domain contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorysegments) | [`computeTrajectorySegmentsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation.ts) |
| `gpu-spatial-analysis` | `CPUMaximumInscribedCircleOperation`, `GPUMaximumInscribedCircleOperation` | [Domain contract](./gpu-geometry-refinement.md#composition) | [`computeMaximumInscribedCircleOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation.ts) |
| `gpu-spatial-analysis` | `CPUNormalizeGeometryOperation`, `GPUNormalizeGeometryOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeNormalizeGeometryOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation.ts) |
| `gpu-spatial-analysis` | `CPUNormalizePolygonsOperation`, `GPUNormalizePolygonsOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeNormalizePolygonsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation.ts) |
| `gpu-spatial-analysis` | `CPUSegmentSnapOperation`, `GPUSegmentSnapOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeSegmentSnapOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation.ts) |
| `gpu-spatial-analysis` | `CPUConvexStraightSkeletonOperation`, `GPUConvexStraightSkeletonOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeConvexStraightSkeletonOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation.ts) |
| `gpu-spatial-analysis` | `CPULineLoopRemovalOperation`, `GPULineLoopRemovalOperation` | [Domain contract](./gpu-geometry-refinement.md#composition) | [`computeLineLoopRemovalOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation.ts) |
| `gpu-spatial-analysis` | `CPUOffsetCurveComponentsOperation`, `GPUOffsetCurveComponentsOperation` | [Domain contract](./gpu-geometry-refinement.md#composition) | [`computeOffsetCurveComponentsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation.ts) |
| `gpu-spatial-analysis` | `CPURatioConcaveHullOperation`, `GPURatioConcaveHullOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeRatioConcaveHullOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation.ts) |
| `gpu-spatial-analysis` | `CPUVectorStraightSkeletonOperation`, `GPUVectorStraightSkeletonOperation` | [Domain contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | [`computeVectorStraightSkeletonOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation.ts) |
| `gpu-spatial-analysis` | `GPUPopulationSupportAlignmentOperation` | [GPU operation contract](./gpu-spatial-analysis/landscape-ecology.md#gpupopulationsupportalignmentoperation) | [`computePopulationSupportAlignmentOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts) |
| `gpu-spatial-analysis` | `CPUPopulationSupportAlignmentOperation` | [CPU operation contract](./gpu-spatial-analysis/landscape-ecology.md#cpupopulationsupportalignmentoperation) | [`computePopulationSupportAlignmentOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts) |
| `gpu-spatial-analysis` | `CPUContinuousSpatialSampleOperation`, `GPUContinuousSpatialSampleOperation` | [Domain contract](./gpu-spatial-analysis/point-process.md#continuous-line-and-area-survey-frames) | [`computeContinuousSpatialSampleOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialBalancedSampleOperation`, `GPUSpatialBalancedSampleOperation` | [Domain contract](./gpu-spatial-analysis/point-process.md#spatial-probability-sampling) | [`computeSpatialBalancedSampleOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation.ts) |
| `gpu-spatial-analysis` | `CPUPointPatternEnvelopeOperation`, `GPUPointPatternEnvelopeOperation` | [Domain contract](./gpu-spatial-analysis/point-process.md#simulation-curves-and-global-tests) | [`computePointPatternEnvelopeOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryKalmanOperation`, `GPUTrajectoryKalmanOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeTrajectoryKalmanOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation.ts) |
| `gpu-spatial-analysis` | `CPUTrajectoryRestrictionOperation`, `GPUTrajectoryRestrictionOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeTrajectoryRestrictionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialCrossValidationOperation`, `GPUSpatialCrossValidationOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) | [`computeSpatialCrossValidationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation.ts) |
| `gpu-spatial-analysis` | `CPUNestedInterpolationValidationOperation`, `GPUNestedInterpolationValidationOperation` | [Domain contract](./gpu-spatial-analysis/spatial-validation-advanced.md#nested-variogram-and-bandwidth-selection) | [`computeNestedInterpolationValidationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation.ts) |
| `gpu-spatial-analysis` | `CPUSpatialInteractionOperation`, `GPUSpatialInteractionOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#calibrated-spatial-interaction) | [`computeSpatialInteractionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation.ts) |
| `gpu-spatial-analysis` | `CPUSphericalArcOverlayOperation`, `GPUSphericalArcOverlayOperation` | [Domain contract](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) | [`computeSphericalArcOverlayOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation.ts) |

| `gpu-spatial-analysis` | `CPUSpatialPanelErrorMaximumLikelihoodOperation`, `GPUSpatialPanelErrorMaximumLikelihoodOperation` | [Domain contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | [`computeSpatialPanelErrorMaximumLikelihoodOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation.ts) |

#### Operation forms: gpu-network

Use the [shared lifecycle](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for transfers and execution. Each qualified row links its domain contract; array links open the exported function’s source signature and precision notes.

| Entry | Exports | Reference | Pure CPU function |
| --- | --- | --- | --- |
| `gpu-network` | `CPUNetworkTimeRoutingOperation`, `GPUNetworkTimeRoutingOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeNetworkTimeRoutingOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-optimization/cpu-network-time-routing-operation.ts) |
| `gpu-network` | `CPUNetworkPointRoutingOperation`, `GPUNetworkPointRoutingOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeNetworkPointRoutingOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-optimization/cpu-network-point-routing-operation.ts) |
| `gpu-network` | `CPUNetworkFlowOperation`, `GPUNetworkFlowOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeNetworkFlowOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-optimization/cpu-network-flow-operation.ts) |
| `gpu-network` | `CPUNetworkLouvainOperation`, `GPUNetworkLouvainOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#gpunetworklouvain) | [`computeNetworkLouvainOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-optimization/cpu-network-louvain-operation.ts) |
| `gpu-network` | `CPUNetworkRationalAccessOperation`, `GPUNetworkRationalAccessOperation` | [Domain contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | [`computeNetworkRationalAccessOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-optimization/cpu-network-rational-access-operation.ts) |
| `gpu-network` | `CPUAlternativeRoutesOperation`, `GPUAlternativeRoutesOperation` | [Domain contract](./gpu-network/alternative-routes.md#alternative-routes) | [`computeAlternativeRoutesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/alternative-routes/cpu-alternative-routes-operation.ts) |
| `gpu-network` | `CPURadiationFlowModelOperation`, `GPURadiationFlowModelOperation` | [Domain contract](./gpu-network/origin-destination-flow-models.md#gpuradiationflowmodel) | [`computeRadiationFlowModelOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts) |
| `gpu-network` | `CPUMapMatchingOperation`, `GPUMapMatchingOperation` | [Domain contract](./gpu-network/operators.md#gpumapmatching) | [`computeMapMatchingOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/map-matching/cpu-map-matching-operation.ts) |
| `gpu-network` | `CPUNetworkCostMatrixOperation`, `GPUNetworkCostMatrixOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkcostmatrix) | [`computeNetworkCostMatrixOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation.ts) |
| `gpu-network` | `CPUNetworkPathExtractionOperation`, `GPUNetworkPathExtractionOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkpathextraction) | [`computeNetworkPathExtractionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-analysis/cpu-network-path-extraction-operation.ts) |
| `gpu-network` | `CPUNetworkServiceAreasOperation`, `GPUNetworkServiceAreasOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkserviceareas) | [`computeNetworkServiceAreasOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-analysis/cpu-network-service-areas-operation.ts) |
| `gpu-network` | `CPUNetworkCentralityOperation`, `GPUNetworkCentralityOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkcentrality) | [`computeNetworkCentralityOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-centrality/cpu-network-centrality-operation.ts) |
| `gpu-network` | `CPUNetworkCostTableOperation`, `GPUNetworkCostTableOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkcosttable) | [`computeNetworkCostTableOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-cost-table/cpu-network-cost-table-operation.ts) |
| `gpu-network` | `CPUNetworkReachabilityOperation`, `GPUNetworkReachabilityOperation` | [Domain contract](./gpu-network/operators.md#gpunetworkreachability) | [`computeNetworkReachabilityOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-reachability/cpu-network-reachability-operation.ts) |
| `gpu-network` | `CPUNetworkStructureOperation`, `GPUNetworkStructureOperation` | [Domain contract](./gpu-network/expansion.md#directed-components-and-undirected-resilience) | [`computeNetworkStructureOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-structure/cpu-network-structure-operation.ts) |
| `gpu-network` | `CPUNetworkTravelTimePercentilesOperation`, `GPUNetworkTravelTimePercentilesOperation` | [Domain contract](./gpu-network/network-time-geography.md#gpunetworktraveltimepercentiles) | [`computeNetworkTravelTimePercentilesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation.ts) |
| `gpu-network` | `GPUKShortestPathsOperation` | [GPU operation contract](./gpu-network/k-shortest-paths.md#gpukshortestpathsoperation) | [`computeKShortestPathsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.ts) |
| `gpu-network` | `CPUKShortestPathsOperation` | [CPU operation contract](./gpu-network/k-shortest-paths.md#cpukshortestpathsoperation-and-computekshortestpathsoncpu) | [`computeKShortestPathsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.ts) |

#### Operation forms: gpu-terrain

Use the [shared lifecycle](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for transfers and execution. Each qualified row links its domain contract; array links open the exported function’s source signature and precision notes.

| Entry | Exports | Reference | Pure CPU function |
| --- | --- | --- | --- |
| `gpu-terrain` | `CPUTerrainBranchingFlowOperation`, `GPUTerrainBranchingFlowOperation` | [Domain contract](./gpu-terrain/terrain-processing.md#branching-consumers) | [`computeTerrainBranchingFlowOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation.ts) |
| `gpu-terrain` | `CPUTerrainBreachOperation`, `GPUTerrainBreachOperation` | [Domain contract](./gpu-terrain/terrain-processing.md#breaching-and-explicit-burning) | [`computeTerrainBreachOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation.ts) |
| `gpu-terrain` | `CPUTerrainCatchmentSummaryOperation`, `GPUTerrainCatchmentSummaryOperation` | [Domain contract](./gpu-terrain/terrain-processing.md#basin-summaries-and-profiles) | [`computeTerrainCatchmentSummaryOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts) |
| `gpu-terrain` | `CPUTerrainDepressionHierarchyOperation`, `GPUTerrainDepressionHierarchyOperation` | [Domain contract](./gpu-terrain/terrain-processing.md#depression-hierarchy-and-stage-curves) | [`computeTerrainDepressionHierarchyOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-depression-hierarchy-operation.ts) |
| `gpu-terrain` | `CPUTerrainFlowDistanceOperation`, `GPUTerrainFlowDistanceOperation` | [Domain contract](./gpu-terrain/terrain-processing.md#hydrology-path-classes) | [`computeTerrainFlowDistanceOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation.ts) |
| `gpu-terrain` | `CPUTerrainStreamOrderOperation`, `GPUTerrainStreamOrderOperation` | [Domain contract](./gpu-terrain/operators.md#gputerrainstreamorder) | [`computeTerrainStreamOrderOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation.ts) |
| `gpu-terrain` | `CPUTerrainWatershedsOperation`, `GPUTerrainWatershedsOperation` | [Domain contract](./gpu-terrain/operators.md#gputerrainwatersheds) | [`computeTerrainWatershedsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation.ts) |
| `gpu-terrain` | `CPUTerrainFlowOperation`, `GPUTerrainFlowOperation` | [Domain contract](./gpu-terrain/operators.md#gputerrainflow) | [`computeTerrainFlowOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-operation.ts) |
| `gpu-terrain` | `CPUProfilePeaksOperation`, `GPUProfilePeaksOperation` | [Domain contract](./gpu-terrain/operators.md#gpuprofilepeaks) | [`computeProfilePeaksOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-features/cpu-profile-peaks-operation.ts) |
| `gpu-terrain` | `CPUTerrainPeakProminenceOperation`, `GPUTerrainPeakProminenceOperation` | [Domain contract](./gpu-terrain/operators.md#gputerrainpeakprominence) | [`computeTerrainPeakProminenceOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation.ts) |
| `gpu-terrain` | `CPUSolarPositionOperation`, `GPUSolarPositionOperation` | [Domain contract](./gpu-terrain/operators.md#gpusolarposition) | [`computeSolarPositionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/gpu-terrain/terrain-illumination/cpu-solar-position-operation.ts) |

#### Operation forms: dataframe-analysis

Use the [shared lifecycle](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for transfers and execution. Each qualified row links its domain contract; array links open the exported function’s source signature and precision notes.

| Entry | Exports | Reference | Pure CPU function |
| --- | --- | --- | --- |
| `dataframe-analysis` | `CPUClassBreaksOperation`, `GPUClassBreaksOperation` | [Domain contract](./gpu-dataframe-analysis.md#gpuclassbreaks) | [`computeClassBreaksOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/column-classification/cpu-class-breaks-operation.ts) |
| `dataframe-analysis` | `CPUColumnQuantilesOperation`, `GPUColumnQuantilesOperation` | [Domain contract](./gpu-dataframe-analysis.md#gpucolumnquantiles) | [`computeColumnQuantilesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/column-classification/cpu-column-quantiles-operation.ts) |
| `dataframe-analysis` | `CPUInequalityOperation`, `GPUInequalityOperation` | [Domain contract](./gpu-dataframe-analysis.md#gpuinequality) | [`computeInequalityOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/composite-indicators/cpu-inequality-operation.ts) |
| `dataframe-analysis` | `CPUVariogramModelFitOperation`, `GPUVariogramModelFitOperation` | [Domain contract](./gpu-dataframe-analysis.md#gpuvariogrammodelfit) | [`computeVariogramModelFitOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts) |
| `dataframe-analysis` | `CPURipleyEnvelopeOperation`, `GPURipleyEnvelopeOperation` | [Domain contract](./gpu-dataframe-analysis.md#gpuripleyenvelope) | [`computeRipleyEnvelopeOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation.ts) |
| `dataframe-analysis` | `CPUSpaceTimeDiggleTestOperation`, `GPUSpaceTimeDiggleTestOperation` | [Domain contract](./gpu-spatial-analysis/space-time-density-and-k-function.md#diggle-test-of-space-time-interaction-gpuspacetimediggletest) | [`computeSpaceTimeDiggleTestOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation.ts) |
| `dataframe-analysis` | `GPUCoregionalizationFitOperation` | [GPU operation contract](./dataframe-analysis/coregionalization-fit.md#gpucoregionalizationfitoperation) | [`computeCoregionalizationFitOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts) |
| `dataframe-analysis` | `CPUCoregionalizationFitOperation` | [CPU operation contract](./dataframe-analysis/coregionalization-fit.md#cpucoregionalizationfitoperation-and-supplied-kriging-model) | [`computeCoregionalizationFitOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts) |

#### Operation forms: raster-analysis

Use the [shared lifecycle](./operation-forms-and-cpu-twins.md#choosing-cpu-or-gpu) for transfers and execution. Each qualified row links its domain contract; array links open the exported function’s source signature and precision notes.

| Entry | Exports | Reference | Pure CPU function |
| --- | --- | --- | --- |
| `raster-analysis` | `CPUCostDistancePathOperation`, `GPUCostDistancePathOperation` | [Domain contract](./gpu-raster/operations-analysis.md#gpucostdistancepath) | [`computeCostDistancePathOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-path-operation.ts) |
| `raster-analysis` | `CPURasterWalkingCostOperation`, `GPURasterWalkingCostOperation` | [Domain contract](./gpu-raster/expansion.md#gpurasterwalkingcost-georeference) | [`computeRasterWalkingCostOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation.ts) |
| `raster-analysis` | `CPUCostDistanceOperation`, `GPUCostDistanceOperation` | [Domain contract](./gpu-raster/operations-analysis.md#gpucostdistance) | [`computeCostDistanceOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-operation.ts) |
| `raster-analysis` | `CPUStreamlinesOperation`, `GPUStreamlinesOperation` | [Domain contract](./gpu-raster/operations-analysis.md#gpustreamlines) | [`computeStreamlinesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/flow-texture/cpu-streamlines-operation.ts) |
| `raster-analysis` | `CPUIsolinesOperation`, `GPUIsolinesOperation` | [Domain contract](./gpu-raster/operations-analysis.md#gpuisolines) | [`computeIsolinesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/isolines/cpu-isolines-operation.ts) |
| `raster-analysis` | `CPURasterFractionalZonalDistributionOperation`, `GPURasterFractionalZonalDistributionOperation` | [Domain contract](./gpu-raster/fractional-zonal-distribution.md#gpurasterfractionalzonaldistribution) | [`computeRasterFractionalZonalDistributionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation.ts) |
| `raster-analysis` | `CPURasterFractionalZonalStatisticsOperation`, `GPURasterFractionalZonalStatisticsOperation` | [Domain contract](./gpu-raster/expansion.md#operations-and-contracts) | [`computeRasterFractionalZonalStatisticsOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation.ts) |
| `raster-analysis` | `CPURasterRegionGraphOperation`, `GPURasterRegionGraphOperation` | [Domain contract](./gpu-raster/remote-sensing.md#region-adjacency-and-merging-rs08) | [`computeRasterRegionGraphOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation.ts) |
| `raster-analysis` | `CPURasterGeometricMedianCompositeOperation`, `GPURasterGeometricMedianCompositeOperation` | [Domain contract](./gpu-raster/raster-time-series.md#gpurastergeometricmediancomposite) | [`computeRasterGeometricMedianCompositeOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation.ts) |
| `raster-analysis` | `CPURasterMedoidCompositeOperation`, `GPURasterMedoidCompositeOperation` | [Domain contract](./gpu-raster/raster-time-series.md#gpurastermedoidcomposite) | [`computeRasterMedoidCompositeOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation.ts) |
| `raster-analysis` | `CPURasterWatershedSegmentationOperation`, `GPURasterWatershedSegmentationOperation` | [Domain contract](./gpu-raster/expansion.md#operations-and-contracts) | [`computeRasterWatershedSegmentationOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation.ts) |
| `raster-analysis` | `GPURasterReconstructionOperation`, `CPURasterReconstructionOperation` | [Domain contract](./gpu-raster/expansion.md#operations-and-contracts) | [`computeRasterReconstructionOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation.ts) |
| `raster-analysis` | `GPURasterFillHolesOperation`, `CPURasterFillHolesOperation` | [Domain contract](./gpu-raster/expansion.md#operations-and-contracts) | [`computeRasterFillHolesOnCPU`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation.ts) |


## Find an operation

Start with [Selected spatial operator families](./gpu-spatial-analysis/operators.md) for a short
overview, the [tool cross-reference](./gpu-spatial-analysis-cross-reference.md) for GIS names,
or the complete [alphabetical spatial export index](#spatial-export-index).
The selected family table is not the complete export inventory.

### Choose by question

Every spatial identity below is imported from `@luma-contributors/analysis/gpu-spatial-analysis`.
Other owners are printed explicitly. Choose the row matching your inputs and desired result.

| Question | Qualified export and decision | Contract |
| --- | --- | --- |
| Which points fall in which polygon? | `GPUPointInPolygonJoin` chooses one owning polygon per point; `GPUSpatialPredicateJoin` with `predicate: 'coveredBy'` returns every boundary-inclusive pair. | [Read contract](./gpu-spatial-analysis-cross-reference.md#all-matches-or-one-coverage-zone) |
| Drive-time areas? | `@luma-contributors/analysis/gpu-network`: `GPUNetworkIsochrones` starts from road travel costs; independent point snapping is a separate input step. | [Read contract](./gpu-network/operators.md#gpunetworkisochrones) |
| Hot spots over time? | `GPUEmergingHotSpots` consumes an existing time cube; `addSpaceTimeHotSpotsRecipe` constructs the cube first. | [Read contract](./gpu-spatial-analysis/operators-statistics.md#gpuemerginghotspots) |
| Smooth rates for small populations? | `GPUEmpiricalBayesRates({clampPriorVariance: false})` pools all rows for PySAL parity; read `output.smoothedRates`. `GPUSpatialEmpiricalBayesRates` uses neighborhoods; the spatial form always clamps its local prior variance and has no such option. | [Read contract](./gpu-spatial-analysis/operators-statistics.md#gpuempiricalbayesrates) |
| Viewshed? | `@luma-contributors/analysis/gpu-terrain`: `GPUTerrainViewshed` handles one observer; cumulative visibility combines observers. | [Read contract](./gpu-terrain/operators.md#gputerrainviewshed) |
| Fill holes in a raster? | `@luma-contributors/analysis/raster-analysis`: `GPURasterFillHoles` fills enclosed binary background; `GPURasterFillNoData` interpolates missing measurements. | [Read contract](./gpu-raster/expansion.md#operations-and-contracts) |
| Extract point-cloud ground? | `@luma-contributors/analysis/pointcloud-analysis`: compare `GPUPointCloudGround` with morphological `GPUPointCloudSMRF`; retain neighborhood loss status. | [Read contract](./pointcloud-analysis.md) |
| Snap GPS traces to roads? | `@luma-contributors/analysis/gpu-network`: `GPUMapMatching` uses ordered tracks; `GPUNetworkSnapping` locates independent points. | [Read contract](./gpu-network/operators.md#gpumapmatching) |
| Where has vegetation declined? | `@luma.gl/graph/gpu-raster`: `GPURasterNDVI` is an upstream image index. Use temporal change monitoring for a trend; one index image does not establish decline. | [Read contract](./gpu-raster/raster-change-monitoring.md) |
| How high above drainage? | `@luma-contributors/analysis/gpu-terrain`: `GPUTerrainHeightAboveDrainage` follows downstream flow, rather than Euclidean nearest-stream distance. | [Read contract](./gpu-terrain/height-above-drainage.md) |
| Find an alternative route? | `@luma-contributors/analysis/gpu-network`: `GPUAlternativeRoutes` returns edge-distinct simple paths; inspect `complete` and distinguish explicit budgets from omitted CPU budgets. | [Read contract](./gpu-network/alternative-routes.md) |
| Compare two tracks? | `GPUTrackSimilarity` measures supplied pairs; `GPUTrackSimilaritySearch` searches candidate tracks with pruning. | [Read contract](./gpu-spatial-analysis/track-similarity-measures.md#lower-bounds-and-pruned-search) |
| Find a change point? | `GPUPettittTest` tests a single change; `GPULocationChangePoints` partitions each location series into multiple segments. | [Read contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#pettitt-test-gpupettitttest) |
| Summarize land cover by polygon? | `@luma-contributors/analysis/raster-analysis`: `GPURasterFractionalCategoricalStatistics` distinguishes observed class area from missing area. | [Read contract](./gpu-raster/fractional-categorical-statistics.md) |
| Find similar pixels? | `@luma-contributors/analysis/raster-analysis`: `GPURasterEmbeddingNearestPixels` searches embedding vectors and reports missing matches. | [Read contract](./gpu-raster/raster-embeddings.md) |

### Spatial export index

This complete browse index is checked against the public spatial barrel: all 311 producers and
recipes in the review overlay, plus additive exports introduced since that review. Entry suffixes
qualify every export; graph operation wrappers are listed in the forms tables above. The historical 261-descriptor / 49-capability projection is unchanged;
this browse census includes later exports and does not replace the exact website inventory check.
Each row gives the input, result or calculation decision that separates it from its siblings.

| Entry | Exports | Reference | Input or calculation |
| --- | --- | --- | --- |
| `gpu-spatial-analysis` | `addBuildingAdjacencyRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addbuildingadjacencyrecipe--ur03) | Shared-wall adjacency and axial alignment from the existing shared-path topology contributor. |
| `gpu-spatial-analysis` | `addBuildingTessellationRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationrecipe--ur01) | Bounded local route, selected by candidateRadius, or the explicit reference overlay route. |
| `gpu-spatial-analysis` | `addBuildingTessellationReferenceRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationreferencerecipe) | Boundary-sampled reference Voronoi fragments dissolved per building, with an optional simple enclosure clip. |
| `gpu-spatial-analysis` | `addCellFlowTopologyRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#cell-flow-topology) | R11: preserves the original OD table and separately aggregates actual neighboring-cell flows. |
| `gpu-spatial-analysis` | `addChangeOfSupportRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addchangeofsupportrecipe) | Change of support recipe: transfer values between two polygon zone systems by area. |
| `gpu-spatial-analysis` | `addClusterAndOutlineRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addclusterandoutlinerecipe) | Cluster and outline recipe: DBSCAN clusters, per-cluster summaries, convex hulls and their area, perimeter and centroid. |
| `gpu-spatial-analysis` | `addContiguityOrderCorrelogramRecipe` | [Contract](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md#correlogram-over-contiguity-orders-addcontiguityordercorrelogramrecipe) | Spatial correlogram over contiguity orders: global Moran's I with analytic inference for the neighbors at exactly 1, 2, ..., `maximumOrder` steps, each with row-standardised weights (`libpysal` `higher_order` rings and `esda.Moran`). |
| `gpu-spatial-analysis` | `addDasymetricRedistributionRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#recipe-settings-and-stage-props) | Fractional source support -> ancillary eligibility -> weighted areal interpolation -> mass ledger. |
| `gpu-spatial-analysis` | `addDrainageNetworkRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#drainage-network-analysis) | One complete planar D8 domain; graph-owned outputs can feed rendering or explicit readback copies. |
| `gpu-spatial-analysis` | `addExcessRiskRecipe` | [Contract](./gpu-spatial-analysis/rate-mapping.md#rate-mapping-recipes) | Appends observed/expected excess risk E_i / (P_i * sum(E) / sum(P)). |
| `gpu-spatial-analysis` | `addExposureByZoneRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#exposure-by-administrative-area) | Denominators retain population with missing hazards; observed coverage is returned separately. |
| `gpu-spatial-analysis` | `addFleetDwellRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addfleetdwellrecipe) | Fleet dwell recipe (stops variant): where do vehicles stop and for how long, per zone. |
| `gpu-spatial-analysis` | `addFleetDwellZoneEventsRecipe` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpuzoneevents) | Fleet dwell recipe (zone events variant): enter and exit events of tracks against zones, dwell per (track, zone) and a per-zone roll-up. |
| `gpu-spatial-analysis` | `addHotSpotAnalysisRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addhotspotanalysisrecipe) | Hot spot analysis recipe: points on a study area, or a raster, to Getis-Ord Gi* hot and cold spots, optionally confirmed by conditional permutation and colored by class breaks. |
| `gpu-spatial-analysis` | `addInterpolationDiagnosticsRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#interpolation-diagnostics) | Ordinary kriging and leave-one-out diagnostics, with no hidden fitting or readback. |
| `gpu-spatial-analysis` | `addIsovistsRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addisovistsrecipe-and-addvisibilitygraphrecipe--ur05) | Angularly sampled obstacle isovist polygons, counterclockwise from +X, with an explicit circular range boundary. |
| `gpu-spatial-analysis` | `addLandscapeMetricsRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addlandscapemetricsrecipe--ec01) | Named class/landscape metrics from a complete, already labelled categorical raster. |
| `gpu-spatial-analysis` | `addLocalBuildingTessellationRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addlocalbuildingtessellationrecipe) | Bounded, building-parallel sampled Voronoi tessellation for planar coordinates and convex or simple (possibly holed) enclosures. |
| `gpu-spatial-analysis` | `addLocalVariabilityRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#local-variability-comparison) | Compare local mean association with local dispersion on the same explicit graph and complete input mask. |
| `gpu-spatial-analysis` | `addPatchProximityRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addpatchproximityrecipe--ec02) | Sparse boundary mode omits dense matrices/fields; calls without mode retain the reference layout. |
| `gpu-spatial-analysis` | `addPeriodComparisonRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addperiodcomparisonrecipe) | Period comparison recipe: two periods of events to a classified, colored change map. |
| `gpu-spatial-analysis` | `addPointsInPolygonsChoroplethRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addpointsinpolygonschoroplethrecipe) | Points-in-polygons choropleth recipe: points to a per-polygon statistic to class breaks to colors. |
| `gpu-spatial-analysis` | `addPopulationSupportAlignmentRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#explicit-support-alignment-and-counterfactual-generation) | Exact ID alignment. Prefer the CPU operation when support data is already on the host. |
| `gpu-spatial-analysis` | `addRasterResistanceConnectivityRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addrasterresistanceconnectivityrecipe--ec03-r) | Circuit connectivity across a resistance raster, for one source and one sink. |
| `gpu-spatial-analysis` | `addRasterZonalDistributionRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#zonal-distributions) | Whole-cell, unweighted distributions, with native-band validity and calibration applied first. |
| `gpu-spatial-analysis` | `addRateClusterMapRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addrateclustermaprecipe) | Rate cluster map recipe: counts and populations to significant local Moran clusters of empirical-Bayes rates, colored by quadrant (esda `Moran_Local_Rate`). |
| `gpu-spatial-analysis` | `addRegionalRasterChangeRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#regional-raster-change) | Two-period raster differences over common support, with independently reported coverage. |
| `gpu-spatial-analysis` | `addResistanceConnectivityRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addresistanceconnectivityrecipe--ec03-deprecated) | Bounded small-network grounded-Laplacian CG solve, with independently recomputed current residual. |
| `gpu-spatial-analysis` | `addSegregationCounterfactualRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#segregation-counterfactual-generation) | Generates both replacement orders before the existing Shapley decomposition, without host readback. |
| `gpu-spatial-analysis` | `addSegregationDecompositionRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addsegregationdecompositionrecipe--ec04) | Two-factor Shapley decomposition of the dissimilarity-index change, without randomization inference. |
| `gpu-spatial-analysis` | `addSpaceTimeHotSpotsRecipe` | [Contract](./gpu-spatial-analysis/operators-statistics.md#addspacetimehotspotsrecipe) | Space-time hot spot recipe: timestamped events to an ArcGIS-style emerging hot spot map. |
| `gpu-spatial-analysis` | `addSparseResistanceConnectivityRecipe` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#addsparseresistanceconnectivityrecipe--ec03) | Sparse grounded-Laplacian Jacobi-PCG resistance solve with unit current injection, component masking, an independent current-conservation residual and per-edge and per-vertex currents. |
| `gpu-spatial-analysis` | `addSpatialRateRecipe` | [Contract](./gpu-spatial-analysis/rate-mapping.md#rate-mapping-recipes) | Appends scale * (E_i + sum neighbors E) / (P_i + sum neighbors P), never an average of rates. |
| `gpu-spatial-analysis` | `addSpatialRegressionRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addspatialregressionrecipe) | Spatial regression recipe: OLS, spatial dependence diagnostics on its residuals, a GPU-written specification decision, a residual local Moran map, and optionally geographically weighted local fits. |
| `gpu-spatial-analysis` | `addStraightLineCatchmentsRecipe` | [Contract](./gpu-spatial-analysis/recipes.md#addstraightlinecatchmentsrecipe) | Straight-line catchments recipe: nearest-facility zones on a raster and what falls in each. |
| `gpu-spatial-analysis` | `addStopValueCubeRecipe` | [Contract](./gpu-spatial-analysis/movement-semantics.md#stops-to-a-cell-period-cube) | Bins live stops against a stable dictionary. Empty observed bins are zero. Loss suppresses the cube. |
| `gpu-spatial-analysis` | `addStreetProfilesRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addstreetprofilesrecipe--ur02) | Side-aware first-building profiles; missing sides remain range-censored and never become observed walls. |
| `gpu-spatial-analysis` | `addSurfacePreparationRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#surface-preparation-with-provenance) | Continuous-surface repair with original validity and independent fill/filter provenance. |
| `gpu-spatial-analysis` | `addVisibilityGraphRecipe` | [Contract](./gpu-spatial-analysis/urban-morphology.md#addisovistsrecipe-and-addvisibilitygraphrecipe--ur05) | Spatial CSR mode returns obstacle-blocked adjacency and degree. The same recipe name from `@luma-contributors/analysis/gpu-network` adds network metrics; qualify the import before choosing the result. |
| `gpu-spatial-analysis` | `addZonalDistributionRecipe` | [Contract](./gpu-spatial-analysis/expansion-recipes.md#zonal-distributions) | Assigns each point to one feature (the join's deterministic ownership rule), then computes unweighted zone distributions. |
| `gpu-spatial-analysis` | `GPUAdaptiveBandwidthKernelDensity` | [Contract](./gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md#gpuadaptivebandwidthkerneldensity) | Adaptive-bandwidth (variable-kernel) density estimate on a raster: the bandwidth of each point is wider where the data are sparse and narrower where they are dense. |
| `gpu-spatial-analysis` | `GPUAffineTransform` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | Applies 2D affine transforms to many features at once on the GPU (`shapely.affinity.affine_transform`, `translate`, `scale`, `rotate`, `skew`; PostGIS `ST_Affine`, `ST_Translate`, `ST_Scale`, `ST_Rotate`). |
| `gpu-spatial-analysis` | `GPUAnalysisContiguityStatistics` | [Contract](./gpu-spatial-analysis/analysis-experience.md#gpuanalysiscontiguitystatistics) | Composes GPUContiguityWeights, GPUSpatialWeightsSummary and GPUColumnQuantiles for AX profiles. |
| `gpu-spatial-analysis` | `GPUAnalysisSpacing` | [Contract](./gpu-spatial-analysis/analysis-experience.md#gpuanalysisspacing) | Full-population spacing through graph-owned sort, scan, reduction and indexed neighbour search. GPUNeighborSearch composes the published GPUGridIndex; no CPU position or status readback occurs. Distinct locations get a second indexed k=1 search, so duplicate-heavy data retains its spacing. |
| `gpu-spatial-analysis` | `GPUArealInterpolation` | [Contract](./gpu-spatial-analysis/operators-accessibility.md#gpuarealinterpolation) | Area-weighted transfer between two zone systems (the tobler `area_interpolate` and CARTO `ENRICH_POLYGONS` workload) on a common fine raster. |
| `gpu-spatial-analysis` | `GPUAreaOfApplicability` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#area-of-applicability) | Dissimilarity index and area of applicability (Meyer and Pebesma 2021, "Predicting into unknown space? Estimating the area of applicability of spatial prediction models", Methods in Ecology and Evolution 12(9); definition of the R package CAST 1.2.0). |
| `gpu-spatial-analysis` | `GPUAssuncaoReisStandardization` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuassuncaoreisstandardization) | Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind `esda.Moran_Rate` and `esda.Moran_Local_Rate`. |
| `gpu-spatial-analysis` | `GPUAZPRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuazpregions) | Bounded deterministic greedy AZP with batched moves. |
| `gpu-spatial-analysis` | `GPUBoundedFacilityAllocation` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuboundedfacilityallocation) | Enumerates all facility subsets (at most 65536), evaluating subsets independently on the GPU. |
| `gpu-spatial-analysis` | `GPUBoundsFilter` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter) | Selects features whose bounding boxes relate to a per-frame box: a GPU `GeoDataFrame.cx[...]` (bounds level) and `gdf.bounds` comparison. |
| `gpu-spatial-analysis` | `GPUBrownianBridgeUtilization` | [Contract](./gpu-spatial-analysis/home-ranges-and-recursions.md#gpubrownianbridgeutilization) | Brownian bridge utilization distribution of a track (Horne et al., Ecology 88, 2007; the dynamic form of Kranstauber et al., 2012, differs only in the variance given per segment). |
| `gpu-spatial-analysis` | `GPUBufferSelection` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpubufferselection) | Selects the points within a per-frame planar distance of point or polyline features. |
| `gpu-spatial-analysis` | `GPUBufferSurface` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpupolygonoverlay-and-gpubuffersurface) | Builds polygon buffer surfaces for lines and polygons. |
| `gpu-spatial-analysis` | `GPUCanonicalGeometryEquality` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Structural equality of normalized geometry trees, including nested empty collections. |
| `gpu-spatial-analysis` | `GPUCanonicalPolygonEquality` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Structural coordinate equality for GPUNormalizePolygons outputs, without hashing. |
| `gpu-spatial-analysis` | `GPUCatchmentAccessibility` | [Contract](./gpu-spatial-analysis/operators-accessibility.md#gpucatchmentaccessibility) | Two-step and three-step floating catchment area accessibility on any (Euclidean kernel, network-cost kernel, binary band). This is the library's one floating catchment kernel. |
| `gpu-spatial-analysis` | `GPUCategoryPairCounts` | [Contract](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md#spatial-entropy-gpuspatialentropy-and-gpucategorypaircounts) | Counts, over the links of a weights graph, how often a row of category `a` is linked to a row of category `b`, and the rows per category. It is the join-count table of a categorical map and the common input of the spatial entropy and the colocation quotient. |
| `gpu-spatial-analysis` | `GPUCellAggregation` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) | Bins rows into discrete global grid cells and aggregates them per cell into a compact, sorted, capacity-bounded cell table. |
| `gpu-spatial-analysis` | `GPUCellCompaction` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucelltopology-and-gpucellcompaction) | Compacts a cell set into its minimal mixed-resolution cover, or expands a mixed-resolution set to one resolution (Quadbin, H3 and S2). |
| `gpu-spatial-analysis` | `GPUCellCover` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellcover) | Polyfills polygon features with Quadbin, H3 or S2 cells. |
| `gpu-spatial-analysis` | `GPUCellGeometry` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpupointtocell-and-gpucellgeometry) | Decodes 64-bit cell keys to centers and boundary polygons in longitude/latitude degrees. |
| `gpu-spatial-analysis` | `GPUCellGridPath` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellgridpath-and-gpucellmeasures) | Grid distances and paths for H3, Quadbin and S2 cell pairs. |
| `gpu-spatial-analysis` | `GPUCellLevelSelection` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) | Publishes the active pyramid level's count, first row and indirect draw arguments from a per-frame level index, so a layer switches levels by rewriting one `uint32` without rebuilding or re-running the pyramid. Put it in a small per-frame graph next to the draw. |
| `gpu-spatial-analysis` | `GPUCellMeasures` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellgridpath-and-gpucellmeasures) | Spherical area, perimeter, edge lengths and pentagon flag of H3, S2 and Quadbin cells. |
| `gpu-spatial-analysis` | `GPUCellPyramid` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) | Builds a zoom pyramid of cell tables: aggregates rows once at the finest level, then rolls each level up into the next coarser one without re-reading the rows. Accepts chunked input rows with matching chunk boundaries, as documented by ; level tables remain single packed outputs. |
| `gpu-spatial-analysis` | `GPUCellRollup` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) | Rolls a sorted cell table up to a coarser resolution without re-reading the source rows. |
| `gpu-spatial-analysis` | `GPUCellSetOutline` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucellsetoutline) | Boundary edges of a set of H3, S2 or Quadbin cells, the `cellsToMultiPolygon` outline without ring assembly. |
| `gpu-spatial-analysis` | `GPUCellTableCompare` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucelltablecompare) | Outer-join diff of two sorted cell tables (period-over-period or A-versus-B compare maps). |
| `gpu-spatial-analysis` | `GPUCellTopology` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpucelltopology-and-gpucellcompaction) | Per-cell fixed fan-out topology of Quadbin, H3 and S2 cells: `gridDisk`, `gridDiskDistances`, `gridRing`, `cellToParent` and `cellToChildren`. |
| `gpu-spatial-analysis` | `GPUChunkedSpatialWeightsAlgebra` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuchunkedspatialweightsalgebra) | The row-local operations of over chunked weights: `union`, `intersection`, `difference`, `symmetricDifference`, `selfWeight` and `subgraph`. |
| `gpu-spatial-analysis` | `GPUChunkedSpatialWeightsTransform` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuchunkedspatialweightstransform) | of chunked weights. `row`, `binary` and `kernel` are row-local and run the whole-CSR transform once per chunk. |
| `gpu-spatial-analysis` | `GPUChunkedSpatialWeightsTranspose` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuchunkedspatialweightstranspose) | Transposes chunked spatial weights into packed or chunked CSR. |
| `gpu-spatial-analysis` | `GPUChunkedSpatialWeightsValidation` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuchunkedspatialweightsvalidation) | over a chunked CSR. The validation is one implementation that accepts whole or chunked weights (whole weights are a single chunk); this class is the chunked-typed form kept for existing callers, and `new GPUSpatialWeightsValidation({weights})` with chunked weights is equivalent. |
| `gpu-spatial-analysis` | `GPUClassAssignment` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuclassassignment) | Assigns each row of a column the class of the shared breaks, clamped to `0..k-1` (infinities clamp to the end classes). NaN and masked rows get . When `k` is 0 every row gets that value. |
| `gpu-spatial-analysis` | `GPUClassificationFit` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuclassificationfit) | Goodness of fit of a classification, matching mapclassify 2.11 `Classifier.adcm` / `get_adcm()`, `get_gadf()` and `get_tss()`: |
| `gpu-spatial-analysis` | `GPUColocationQuotient` | [Contract](./gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md#gpucolocationquotient) | Colocation quotient of a categorical point (or area) set over a weights graph, after Leslie and Kronenfeld (2011), "The colocation quotient: a new measure of spatial association between categorical subsets of points", Geographical Analysis 43(3). |
| `gpu-spatial-analysis` | `GPUConcaveHull` | [Contract](./gpu-raster/expansion.md#operations-and-contracts) | Planar alpha-complex hull: retain triangles whose circumradius does not exceed the requested radius, then expose their oriented boundary. |
| `gpu-spatial-analysis` | `GPUConditionalGaussianSimulation` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpuconditionalgaussiansimulation-conditional-gaussian-simulation) | Exact dense conditioning for at most 4096 observation and query rows combined. |
| `gpu-spatial-analysis` | `GPUConformalPredictionIntervals` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#gpuconformalpredictionintervals) | Split-conformal and CV+ prediction intervals (Vovk et al. |
| `gpu-spatial-analysis` | `GPUConstrainedDelaunayTriangulation` | [Contract](./gpu-raster/expansion.md#operations-and-contracts) | Triangulates polygon shells/holes, then performs bounded Lawson flips of shared interior edges. |
| `gpu-spatial-analysis` | `GPUContiguityWeights` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpucontiguityweights) | Polygon contiguity weights (the PySAL `Queen` / `Rook` equivalent) written as a CSR whose rows are the polygon FEATURES. |
| `gpu-spatial-analysis` | `GPUContinuousSpatialSample` | [Contract](./gpu-spatial-analysis/point-process.md#continuous-line-and-area-survey-frames) | Continuous random-shift probability sampling. |
| `gpu-spatial-analysis` | `GPUConvexStraightSkeleton` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Vector wavefront prototype for independent strictly convex polygons. |
| `gpu-spatial-analysis` | `GPUCoverageDissolve` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpucoveragedissolve) | Dissolves a polygon coverage by label: GeoPandas `dissolve(by=..., method='coverage')` and Shapely `coverage_union_all` per label. |
| `gpu-spatial-analysis` | `GPUCoverageSimplification` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpucoveragesimplification) | Gap-free polygon-coverage simplification (the Shapely `simplify_coverage` and PostGIS `ST_CoverageSimplify` idea): every arc shared by two polygons is simplified once, so both neighbours keep identical vertices along it. |
| `gpu-spatial-analysis` | `GPUCoverageValidity` | [Contract](./gpu-spatial-analysis/operators-validity.md#gpucoveragevalidity) | Validates a polygon coverage: Shapely `coverage_is_valid` and `coverage_invalid_edges` (GEOS `CoverageValidator`), reported per ring segment. |
| `gpu-spatial-analysis` | `GPUDelaunayTessellation` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpudelaunaytessellation-and-gpuvoronoidiagram) | Builds a planar Delaunay triangulation on the GPU by parallel insertion and flipping. |
| `gpu-spatial-analysis` | `GPUDorlingCartogram` | [Contract](./gpu-spatial-analysis/gpu-dorling-cartogram.md#gpudorlingcartogram) | Planar Dorling circle relaxation, indexed by default above the named cut-over. |
| `gpu-spatial-analysis` | `GPUDotDensity` | [Contract](./gpu-spatial-analysis/operators-density.md#gpudotdensity-and-gpurandompointsinpolygon) | Dasymetric dot-density map: places `value * dotsPerUnit` random dots inside each polygon, per category, without flicker when the dot value changes with zoom. |
| `gpu-spatial-analysis` | `GPUEmergingHotSpots` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuemerginghotspots) | ArcGIS-style emerging hot spot analysis over a dense space-time cube with lattice or CSR neighborhoods. |
| `gpu-spatial-analysis` | `GPUEmpiricalBayesRates` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuempiricalbayesrates) | Global empirical-Bayes rate standardization and smoothing for rate variables, matching PySAL esda `smoothing.assuncao_rate` (used by `Moran_Rate` and `Moran_Local_Rate` with `adjusted=True`) and `smoothing.Empirical_Bayes`. |
| `gpu-spatial-analysis` | `GPUEnsembleScores` | [Contract](./gpu-spatial-analysis/scoring-rules.md#gpuensemblescores) | Proper scoring rules of an ensemble forecast (for example conditional simulations) against an observed field: the continuous ranked probability score per site, the energy score and the variogram score of the whole field. |
| `gpu-spatial-analysis` | `GPUExpectationBasedSpaceTimeScan` | [Contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#expectation-based-scan-gpuexpectationbasedspacetimescan) | Expectation-based space-time scan with Fast Subset Scan (Neill 2012), in the prospective form of surveillance: every time window ends at the last bucket. |
| `gpu-spatial-analysis` | `GPUFocalStatistics` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpufocalstatistics) | Computes moving-window statistics over a float32 raster: mean, sum, min, max, range, population standard deviation, and the count of valid cells. |
| `gpu-spatial-analysis` | `GPUFuzzyContiguity` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) | Distance-based fuzzy polygon contiguity, including multipart geometry and holes. |
| `gpu-spatial-analysis` | `GPUGammaStatistic` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic) | A Gamma-type statistic over directed weighted edges. |
| `gpu-spatial-analysis` | `GPUGaussianPredictiveScores` | [Contract](./gpu-spatial-analysis/scoring-rules.md#gpugaussianpredictivescores) | Gaussian predictive scores with required CRPS and optional negative log density, threshold-event Brier score and central prediction-interval score. |
| `gpu-spatial-analysis` | `GPUGaussianRandomField` | [Contract](./gpu-spatial-analysis/gaussian-random-field.md#gpugaussianrandomfield) | Stationary isotropic Gaussian fields in a native planar frame, without CPU readback. |
| `gpu-spatial-analysis` | `GPUGeneralizedGeographicallyWeightedRegression` | [Contract](./gpu-spatial-analysis/statistical-expansion.md#generalized-geographically-weighted-regression) | Local Poisson log-link or Bernoulli logit regression with a fixed distance or adaptive neighbor bandwidth. |
| `gpu-spatial-analysis` | `GPUGeoConformalPredictionIntervals` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#gpugeoconformalpredictionintervals) | Spatially weighted (localized) split-conformal intervals, the GeoConformal prediction of Lou, Luo and Meng (2024) built on weighted conformal prediction (Tibshirani, Barber, Candes and Ramdas 2019). |
| `gpu-spatial-analysis` | `GPUGeodesicDestination` | [Contract](./gpu-spatial-analysis/geodesic-precision.md#gpugeodesicdestination) | Destination points from origins, bearings and distances (turf `destination`, PostGIS `ST_Project`), for example range rings, sector wedges or projected vessel positions. |
| `gpu-spatial-analysis` | `GPUGeodesicPairs` | [Contract](./gpu-spatial-analysis/geodesic-precision.md#gpugeodesicpairs) | Geodesic distance, initial and final bearing, and midpoint between many origin/target pairs (turf `distance`, `bearing`, `midpoint`; PostGIS `ST_Distance(geography)`, `ST_Azimuth`). |
| `gpu-spatial-analysis` | `GPUGeographicalDetector` | [Contract](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md#geographical-detector-gpugeographicaldetector) | Factor detector of the geographical detector (Wang, Li, Christakos et al. 2010; Wang, Zhang and Fu 2016): the q statistic of a stratification of one response variable, with its non-central F statistic and non-centrality parameter. |
| `gpu-spatial-analysis` | `GPUGeographicallyTemporallyWeightedRegression` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpugeographicallytemporallyweightedregression) | Fits a local weighted regression at each observation using spatial and temporal distance. Choose fixed or adaptive bandwidth and a time-distance scale; this differs from spatial-only GWR. |
| `gpu-spatial-analysis` | `GPUGeographicallyWeightedRegression` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpugeographicallyweightedregression) | Geographically weighted regression (Brunsdon, Fotheringham and Charlton 1996; Fotheringham, Brunsdon and Charlton 2002) with automatic bandwidth selection by AICc, as in mgwr and ArcGIS "Geographically Weighted Regression", local inference, and optional great-circle distances. |
| `gpu-spatial-analysis` | `GPUGeographicallyWeightedRegressionNonstationarityTest` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpugeographicallyweightedregressionnonstationaritytest) | Monte Carlo test of the spatial non-stationarity of GWR coefficients (Brunsdon, Fotheringham and Charlton 1999; the permutation test of GWR4 and mgwr `GWRResults.spatial_variability`). |
| `gpu-spatial-analysis` | `GPUGeographicDistribution` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpugeographicdistribution) | Geographic distribution statistics (the ArcGIS "Measuring Geographic Distributions" family) for planar points, per group or per masked selection, deterministic on the GPU. |
| `gpu-spatial-analysis` | `GPUGeometryCleanup` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | Snaps coordinates to a grid and removes repeated line or polygon vertices on the GPU (`shapely.set_precision(mode='pointwise')`, `shapely.remove_repeated_points`, PostGIS `ST_SnapToGrid`, `ST_RemoveRepeatedPoints`). |
| `gpu-spatial-analysis` | `GPUGeometryMeasures` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | Measures each line or polygon feature, optionally grouped: length or perimeter, area, centroid, bounds and vertex count. Choose the spatial context and coordinate model before interpreting units. |
| `gpu-spatial-analysis` | `GPUGeometryOrientation` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | Reverses rings or orients polygons on the GPU (`shapely.reverse`, `shapely.orient_polygons`, PostGIS `ST_Reverse`, `ST_ForcePolygonCW`, `ST_ForcePolygonCCW`). |
| `gpu-spatial-analysis` | `GPUGeometryPredicates` | [Contract](./gpu-spatial-analysis/operators-validity.md#gpugeometrypredicates) | Returns source-aligned geometry predicates and structural counts. Use aligned geometry pairs for exact-coordinate equality; use a spatial predicate join to discover matching feature pairs. |
| `gpu-spatial-analysis` | `GPUGeometryValidity` | [Contract](./gpu-spatial-analysis/operators-validity.md#gpugeometryvalidity) | Computes a per-feature validity bitmask for polygons, multipolygons, linestrings and points: the GPU analog of the checks behind PostGIS `ST_IsValidDetail`, Shapely `explain_validity` and turf `kinks`. `issueCode` is diagnostic; optional `validity` can be used as the next operation's mask. |
| `gpu-spatial-analysis` | `GPUGlobalPermutationTest` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Global permutation inference (esda `p_sim`, `z_sim`) for Moran's I, Geary's C, Getis-Ord General G, bivariate Moran's I and Lee's L, with the reference distribution and its histogram. |
| `gpu-spatial-analysis` | `GPUGlobalSpatialStatistics` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuglobalspatialstatistics) | Global spatial autocorrelation from a CSR and a value column: Moran's I, Geary's C, Getis-Ord General G, bivariate Moran's I and join counts, each with its analytic expectation, variance, z-score and normal p-value (`alternative`, default two-sided). |
| `gpu-spatial-analysis` | `GPUGreatCircleArcs` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Tessellates great-circle arcs between many origin/destination pairs on the GPU (turf `greatCircle`, the CPU step behind flow maps drawn with `PathLayer`). |
| `gpu-spatial-analysis` | `GPUGridGenerator` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpugridgenerator) | Generates square, hexagon, triangle and point grids from an origin and a cell size (turf `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid`; QGIS "Create grid"; GeoPandas `make_grid`), the "fishnet then join" half of areal aggregation. |
| `gpu-spatial-analysis` | `GPUGroupConcaveHull` | [Contract](./gpu-raster/expansion.md#group-outlines-and-recipes) | Independent alpha complexes for GPU-resident labels. |
| `gpu-spatial-analysis` | `GPUGroupConvexHull` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpugroupconvexhull) | Convex hull of every label group by Andrew's monotone chain, with an exact orientation predicate. It outlines clusters, hot regions and trajectory footprints. |
| `gpu-spatial-analysis` | `GPUGroupGeometry` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpugroupgeometry) | Per-label geometry summaries over a `uint32` label column: bounds, mean and weighted center, medoid, standard deviational ellipse, count. It is the "cluster, then describe it" step after `GPUDBSCAN` or any classifier. |
| `gpu-spatial-analysis` | `GPUH3CellIdentities` | [Contract](./gpu-spatial-analysis/operators-cells.md#cell-system-support) | Exact integer H3 directed edge and canonical vertex identities, including pentagons and faces. |
| `gpu-spatial-analysis` | `GPUHealpixCellAggregation` | [Contract](./gpu-spatial-analysis/healpix-cells-dggs-json.md#aggregation-gpuhealpixcellaggregation) | Aggregates points that `GPUHealpixPointToCell` keyed into HEALPix cells: the distinct cells in key order with their row counts and, for a `values` column, per-cell sums, minima and maxima. |
| `gpu-spatial-analysis` | `GPUHealpixPointToCell` | [Contract](./gpu-spatial-analysis/healpix-cells-dggs-json.md#point-to-cell-gpuhealpixpointtocell) | Converts longitude/latitude points to HEALPix nested cell keys (RTS-08), one row per row, with the same output layout as `GPUPointToCell` (little-endian `(low, high)` words and an optional validity flag; masked and non-finite rows get the zero key). |
| `gpu-spatial-analysis` | `GPUHDBSCAN` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuhdbscan) | HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's `HDBSCAN` with its defaults (`cluster_selection_method='eom'`, `allow_single_cluster=False`, `cluster_selection_epsilon=0`, Euclidean metric) up to a relabeling of the clusters. |
| `gpu-spatial-analysis` | `GPUHilbertKeys` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuhilbertkeys) | Computes 2D Hilbert curve indices for points or feature-box centers within bounds ("Hilbert keys next to `GPUBVH`") and, optionally, the curve-ordered permutation through the gpgpu radix sort. |
| `gpu-spatial-analysis` | `GPUHotSpotAnalysis` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuhotspotanalysis-and-gpulocalmoran) | Getis-Ord Gi* local hot spot analysis on caller-supplied spatial weights. |
| `gpu-spatial-analysis` | `GPUHuffTradeAreas` | [Contract](./gpu-spatial-analysis/operators-accessibility.md#gpuhufftradeareas) | Huff gravity-model trade areas on any : the probability that demand location `i` patronises facility `j`, `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik`, the modal facility per demand location (its trade area), and the expected demand captured by each facility. |
| `gpu-spatial-analysis` | `GPUInhomogeneousPointPattern` | [Contract](./gpu-spatial-analysis/point-process.md#intensity-and-polygon-corrections) | GPU polygon-window intensity estimation and inhomogeneous border K. |
| `gpu-spatial-analysis` | `GPUInstrumentalVariablesCovariance` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#iv-covariance-and-robust-spatial-2sls) | Centered and scaled IV sandwich covariance: A H' diag(u) K diag(u) H A', where A = (Z'P_H Z)^-1 Z'H(H'H)^-1. |
| `gpu-spatial-analysis` | `GPUInverseDistanceWeighting` | [Contract](./gpu-spatial-analysis/operators-density.md#gpuinversedistanceweighting) | Interpolates scattered samples onto a raster with inverse distance weighting, `w = 1 / d^p` (Shepard 1968), or with the optional `'modified-shepard'` taper `w = ((R - d)+ / (R d))^p` (Franke and Nielson 1980), where `R` is the search radius. |
| `gpu-spatial-analysis` | `GPUIslandAttachment` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) | Attach original empty CSR rows to their nearest finite planar representative (ties: lowest ID), retaining original weights and adding binary edges. |
| `gpu-spatial-analysis` | `GPUJacquezTest` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpunearrepeatknox-gpulocalknox-and-gpujacqueztest) | Jacquez k-nearest-neighbor test for space-time interaction (pointpats `jacquez`): the number `J` of (event, neighbor) pairs where the neighbor is among the `k` nearest events of the event in space (the `neighbors` list) and among its `k` nearest events in time. |
| `gpu-spatial-analysis` | `GPUKernelDensity` | [Contract](./gpu-spatial-analysis/operators-density.md#gpukerneldensity) | Planar kernel density estimate on a raster, with the bandwidth in ground units. |
| `gpu-spatial-analysis` | `GPUKnoxTest` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuknoxtest-and-gpumanteltest) | Knox space-time interaction test (pointpats `Knox`): the number `X` of event pairs that are close in space (the `pairs` list) and also close in time (`/t_i - t_j/ <= timeThreshold`), with the classic expectation and a Monte Carlo reference distribution. |
| `gpu-spatial-analysis` | `GPUKriging` | [Contract](./gpu-spatial-analysis/operators-density.md#gpukriging) | Local ordinary kriging on a raster: for every output cell the `k` nearest samples within the search radius form a `(k + 1) x (k + 1)` ordinary kriging system under a fitted variogram model, solved in the invocation by Gaussian elimination with partial pivoting (Cressie 1993, eq. 3.2.12 to 3.2.16). |
| `gpu-spatial-analysis` | `GPULabelPoint` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpulabelpoint) | Pole of inaccessibility (the point of a polygon farthest from its boundary) for many polygons at once, the label anchor of Mapbox `polylabel` and `turf pointOnFeature`-style labeling. |
| `gpu-spatial-analysis` | `GPULatticeWeights` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpulatticeweights) | Regular-grid contiguity weights (the PySAL `lat2W` equivalent) written as a CSR. |
| `gpu-spatial-analysis` | `GPULinearReferencing` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinearreferencing-and-gpulinelocate) | Snaps point queries to polylines and returns the selected line, segment, fraction, foot point, distance and measure. Use GPULineLocate for a position already specified by line and measure. |
| `gpu-spatial-analysis` | `GPULineChunk` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Splits lines into fixed-length chunks, or extracts a measure range from every line, on the GPU. |
| `gpu-spatial-analysis` | `GPULineClipByPolygon` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulineclipbypolygon-and-gpusharedpaths) | Clips linestrings to polygons: GeoPandas `clip(lines, polygons)` and Shapely `intersection(line, polygon)`, or with `mode: 'outside'` Shapely `difference(line, polygon)`. |
| `gpu-spatial-analysis` | `GPULineDensity` | [Contract](./gpu-spatial-analysis/operators-density.md#gpulinedensity) | Line length per grid cell, and that length divided by the cell area (QGIS "Sum line lengths" per grid cell, PostGIS `ST_Length(ST_Intersection(line, cell))` summed). This is not the QGIS or ArcGIS "Line density" tool, which spreads each line over a circular search radius. |
| `gpu-spatial-analysis` | `GPULineLengthPerPolygon` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpulinelengthperpolygon) | Total line length inside each polygon (QGIS "Sum line lengths", PostGIS `sum(ST_Length(ST_Intersection(line, polygon)))` per polygon), optionally weighted by a per-line value, plus the number of segments that contribute. |
| `gpu-spatial-analysis` | `GPULineLocate` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinearreferencing-and-gpulinelocate) | Places events along lines by measure (turf `along`, PostGIS `ST_LineInterpolatePoint` and `ST_LocateAlong` with offset): route events, mile markers, vehicles moving along routes. |
| `gpu-spatial-analysis` | `GPULineLoopRemoval` | [Contract](./gpu-geometry-refinement.md#composition) | Chronological loop erasure of noded open curves. |
| `gpu-spatial-analysis` | `GPULineMerge` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesplit-and-gpulinemerge) | Merges linestrings that share endpoints into maximal chains, like Shapely `linemerge` and turf `lineMerge` (undirected, no direction constraint). |
| `gpu-spatial-analysis` | `GPULineNoding` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpulinenoding-gpupolygonize-and-gpumakevalid) | Nodes arbitrary finite planar linework and emits one atomic row for every segment between consecutive distinct noded vertices (a repeated vertex yields no zero-length row). Crossings, touches and both ends of collinear overlaps reuse the exact-or-uncertain predicates of `GPUSegmentIntersection` through `GPULineSplit`. |
| `gpu-spatial-analysis` | `GPULinePrecisionReduction` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Grid-rounded lines with explicit collapse policy and per-source line status. |
| `gpu-spatial-analysis` | `GPULineSegmentize` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Densifies many lines so that no output segment is longer than a per-frame maximum length (turf-style densify, PostGIS `ST_Segmentize`, the CPU tessellation step of arc and line layers). |
| `gpu-spatial-analysis` | `GPULineSimplification` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesimplification) | Douglas-Peucker (and TD-TR) simplification of many polylines or tracks as a per-vertex importance column plus a per-frame tolerance selection. |
| `gpu-spatial-analysis` | `GPULineSmooth` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Chaikin corner-cutting smoothing of many lines or rings at once (turf `polygonSmooth`, QGIS Smooth, the usual way to soften generalized boundaries or GPS tracks). |
| `gpu-spatial-analysis` | `GPULineSplit` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulinesplit-and-gpulinemerge) | Splits linestrings at every point where they cross, touch or overlap each other or themselves (turf `lineSplit` against the union of the lines, and the noding step of `lineIntersect`). |
| `gpu-spatial-analysis` | `GPULISAMarkov` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuspatialmarkov-and-gpulisamarkov) | LISA Markov (Rey 2001, giddy `LISA_Markov`): transition counts between local Moran quadrants of consecutive periods. Each period runs `GPULocalMoran` on its slice of `values` (moments and significance per period, as esda would per year), and a `GPUTransitionMatrix` counts moves between the states. |
| `gpu-spatial-analysis` | `GPULocalCategoryDependence` | [Contract](./gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md#gpulocalcategorydependence) | Local indicators for categorical data (LICD), local composition part: for each row, how many of its neighbours share its category, and how unusual that is against the category's global share. |
| `gpu-spatial-analysis` | `GPULocalCollinearity` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity) | Weighted, centered local predictor VIF surfaces. |
| `gpu-spatial-analysis` | `GPULocalFrameProjection` | [Contract](./gpu-spatial-analysis/operators-lines.md#geodesic_wgsl-creategpulocalframe-and-gpulocalframeprojection) | Projects GPU-resident longitude/latitude positions into a `createGPULocalFrame` frame on the GPU, so planar operators can consume them without a CPU readback ("project first"). |
| `gpu-spatial-analysis` | `GPULocalGeary` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocalgeary) | Local Geary `c_i = sum_j w_ij (z_i - z_j)^2`, where `z` is standardized by the population variance of the analysis rows. |
| `gpu-spatial-analysis` | `GPULocalKnox` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpunearrepeatknox-gpulocalknox-and-gpujacqueztest) | Local Knox statistic (pointpats `KnoxLocal`): for every event, the number of its spatial neighbors (the row of the `pairs` list) that are also close in time, `/t_i - t_j/ <= timeThreshold`, with conditional permutation inference per event. |
| `gpu-spatial-analysis` | `GPULocalKriging` | [Contract](./gpu-spatial-analysis/operators-density.md#gpulocalkriging) | Local covariance kriging with equilibrated partial-pivot bounded solves. |
| `gpu-spatial-analysis` | `GPULocalMoran` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuhotspotanalysis-and-gpulocalmoran) | Local Moran's I (LISA) on caller-supplied spatial weights, with an analytic z-score and HH/LL/HL/LH quadrants. |
| `gpu-spatial-analysis` | `GPULocalOutlierFactor` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocaloutlierfactor) | Local Outlier Factor (Breunig et al. 2000) over a kNN table: the geo crate's `OutlierDetection` and scikit-learn's `LocalOutlierFactor`, whose `-negative_outlier_factor_` the `lof` output matches. |
| `gpu-spatial-analysis` | `GPULocalPermutationTest` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Conditional-permutation (pseudo p-value) inference for local Moran's I, local Geary (with the optional `classes` output), local Getis-Ord G / G*, local bivariate Moran, Lee's local L and the univariate and bivariate local join counts, all with the same conditional-randomization scheme. |
| `gpu-spatial-analysis` | `GPULocalRBF` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpulocalrbf-local-radial-basis-interpolation) | One GPU augmented kernel/polynomial solve per query. |
| `gpu-spatial-analysis` | `GPULocalSpatialHeteroscedasticity` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocalspatialheteroscedasticity) | LOSH H_i = weightedMean_j(/y_j - weightedMean_k(y_k)/^a) / mean(residual^a). |
| `gpu-spatial-analysis` | `GPULocationAllocation` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpulocationallocation) | Deterministic bounded location-allocation over a caller-owned cost matrix. |
| `gpu-spatial-analysis` | `GPULocationChangePoints` | [Contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#per-location-change-points-gpulocationchangepoints) | Per-location change points by unpruned optimal partitioning (Jackson et al. |
| `gpu-spatial-analysis` | `GPULonLatPolygonPreparation` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpulonlatpolygonpreparation) | Prepares short-way longitude/latitude polygons entirely on the GPU: densify, classify, cap, normalize and clip. |
| `gpu-spatial-analysis` | `GPULOSHResidualPermutationTest` | [Contract](./gpu-spatial-analysis/losh-residual-permutation.md#conditional-losh-residual-inference) | Computes descriptive LOSH and tests its fixed normalized residual field by conditional sampling without replacement. |
| `gpu-spatial-analysis` | `GPUMakeValid` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpulinenoding-gpupolygonize-and-gpumakevalid) | Topology repair that rebuilds valid polygons from arbitrary, possibly invalid polygon boundaries, independently for each source feature. |
| `gpu-spatial-analysis` | `GPUMantelTest` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuknoxtest-and-gpumanteltest) | Mantel test of space-time interaction (pointpats `Mantel`): the Pearson correlation `r` between spatial distances `d_ij` and absolute time differences `/t_i - t_j/` over the listed pairs `i < j`, with a Monte Carlo reference from `P` swap-or-not permutations of the times. |
| `gpu-spatial-analysis` | `GPUMapColoring` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpumapcoloring) | Greedy parallel graph coloring of polygon contiguity (Jones-Plassmann, largest degree first), so adjacent polygons never share a color: the classic "map coloring" for choropleth-free categorical maps. |
| `gpu-spatial-analysis` | `GPUMaximumInscribedCircle` | [Contract](./gpu-geometry-refinement.md#composition) | Best-first branch-and-bound over all rings of each feature under even-odd fill. |
| `gpu-spatial-analysis` | `GPUMaxPRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpumaxpregions) | Parallel multi-seed Max-P regionalization. |
| `gpu-spatial-analysis` | `GPUMinimumBounds` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuminimumbounds) | Computes bounding rectangles, circles, diameters and axis-aligned envelopes per feature or label group. Select the required shapes rather than treating a rectangle and a circle as the same extent. |
| `gpu-spatial-analysis` | `GPUMinimumClearance` | [Contract](./gpu-spatial-analysis/operators-validity.md#gpuminimumclearance) | Computes the minimum clearance of every feature: the GPU analog of `shapely.minimum_clearance` and `shapely.minimum_clearance_line` (GEOS `MinimumClearance`). |
| `gpu-spatial-analysis` | `GPUMobilityIndicators` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gpumobilityindicators) | Per-user mobility indicators on staypoints, following the definitions of scikit-mobility 1.3.1 (`individual` measures) and trackintel (`freq_method`, `temporal_tracking_quality`): |
| `gpu-spatial-analysis` | `GPUModalSplit` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gpumodalsplit) | Modal split of triplegs (trackintel `calculate_modal_split`): one thread per track accumulates count, summed haversine or Euclidean length, or summed duration per (track, mode). With `modeSpeedUpperBoundsFloat32`, the mode comes from trackintel's `simple-coarse` speed classes. |
| `gpu-spatial-analysis` | `GPUMultipleTestingAdjustment` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpumultipletestingadjustment) | Multiple-testing adjustment of per-row p-values on the GPU: Bonferroni, Holm, Benjamini-Hochberg and Benjamini-Yekutieli (see for the exact formulas, which match statsmodels `multipletests`). |
| `gpu-spatial-analysis` | `GPUMultiscaleGeographicallyTemporallyWeightedRegression` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpumultiscalegeographicallytemporallyweightedregression) | Multiscale GTWR by backfitting (the algorithm of mgtwr `multi_bws`, Wu et al. |
| `gpu-spatial-analysis` | `GPUMultiscaleGeographicallyWeightedRegression` | [Contract](./gpu-spatial-analysis/operators-regression.md#mgwr-bandwidth-selection-and-limits) | Gaussian-response multiscale GWR with per-variable distance bandwidths. |
| `gpu-spatial-analysis` | `GPUMultivariateLocalGeary` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#joint-randomization-and-rank-dynamics) | Mean across variables of row-standardized local Geary. |
| `gpu-spatial-analysis` | `GPUMultivariateLocalGearyPermutationTest` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpumultivariatelocalgearypermutationtest) | Per-frame: values, mask, weights and parameter contents (seed and significance). Compile-time: view shapes, variableCount, maximumPermutations, strategy, tile sizes, maximumNeighbors, tail/correction choices and which optional outputs are present. |
| `gpu-spatial-analysis` | `GPUNaturalNeighbor` | [Contract](./gpu-spatial-analysis/interpolation-expansion.md#gpunaturalneighbor-natural-neighbors) | Computes true stolen-Voronoi-area coordinates, independently of triangulation implementation. |
| `gpu-spatial-analysis` | `GPUNearestFeatureJoin` | [Contract](./gpu-spatial-analysis/concepts.md#precision-of-longitudelatitude-positions) | Joins queries to their nearest features by planar or great-circle distance. |
| `gpu-spatial-analysis` | `GPUNearestFeatureWeights` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpunearestfeatureweights) | Adapter from the k-nearest output of to a CSR: libpysal 4.15 `KNN` / `Graph.build_knn` for features of any kind (points, lines, polygons), with the join's exact geometry distances. |
| `gpu-spatial-analysis` | `GPUNearRepeatKnox` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpunearrepeatknox-gpulocalknox-and-gpujacqueztest) | Near-repeat Knox table: for every cell of distance band by time band, the number of event pairs in the cell and a Monte Carlo p-value per cell. The pair counts are exact integers. |
| `gpu-spatial-analysis` | `GPUNeighborhoodSummary` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuneighborhoodsummary) | Per-row summaries of a neighborhood over a CSR: the generalized spatial lag family (momepy `describe`, libpysal `lag_categorical`). |
| `gpu-spatial-analysis` | `GPUNeighborSearch` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuneighborsearch) | Builds CSR neighborhoods from planar or longitude/latitude points. Choose k-nearest neighbors for fixed neighbor counts or radius mode for distance support; the spatial context defines distance units. |
| `gpu-spatial-analysis` | `GPUNestedInterpolationValidation` | [Contract](./gpu-spatial-analysis/spatial-validation-advanced.md#nested-variogram-and-bandwidth-selection) | Resident exact nested search with tiled row scores/metrics and a bounded exhaustive neighbor search. Outer test outcomes never affect inner fits or selection. Fold/split indices are runtime parameters; all folds share the same fixed set of shader sources and pipelines. |
| `gpu-spatial-analysis` | `GPUNormalizeGeometry` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Canonicalizes a mixed geometry collection while preserving its structural hierarchy. Use GPUNormalizePolygons for polygon-only inputs and canonical equality when the desired result is a comparison. |
| `gpu-spatial-analysis` | `GPUNormalizePolygons` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Canonical planar polygon serialization, entirely on the GPU. |
| `gpu-spatial-analysis` | `GPUOffsetCurve` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuoffsetcurve-and-gpuvertexsnap) | One-sided offset of lines and rings (`shapely.offset_curve`, GeoPandas `offset_curve`, geo `OffsetCurve`-style raw offset). |
| `gpu-spatial-analysis` | `GPUOffsetCurveComponents` | [Contract](./gpu-geometry-refinement.md#composition) | Node raw offsets, remove portions inside the source-distance band, select the requested side and compact multipart components. |
| `gpu-spatial-analysis` | `GPUOffsetExpansion` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter) | Expands an offsets view into one owner index (and local index) per child row: the GPU equivalent of Shapely `get_coordinates(..., return_index=True)`, `get_parts(..., return_index=True)` and GeoPandas `explode(index_parts=True)` index maps. |
| `gpu-spatial-analysis` | `GPUOPTICS` | [Contract](./gpu-spatial-analysis/point-process.md#optics-ordering) | Exact bounded OPTICS with deterministic (reachability, input-row) frontier ties. |
| `gpu-spatial-analysis` | `GPUOrdinaryLeastSquares` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuordinaryleastsquares) | Fits a global regression with an intercept and publishes coefficient inference, model diagnostics and optional row outputs. Predictions cover finite predictor rows; residuals and fitted values cover valid training rows. |
| `gpu-spatial-analysis` | `GPUOutlineGeometry` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuoutlinegeometry) | Builds render-only buffer geometry for points, lines and rings: for each input vertex a round join disc and a rectangle toward the next vertex, as non-indexed triangles ready to draw (turf `buffer` and `lineOffset` for the picture only). |
| `gpu-spatial-analysis` | `GPUPairGather` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter) | Merges join output with attribute columns: the dataframe half of GeoPandas `sjoin` and `sjoin_nearest` (`how='inner'` and `how='left'`, `distance_col`). |
| `gpu-spatial-analysis` | `GPUPettittTest` | [Contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#pettitt-test-gpupettitttest) | Pettitt (1979) non-parametric single change point test per location. `U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)` runs in exact integers over the finite series (ties contribute 0; NaN is skipped), one invocation and `sliceCount^2` comparisons per cell. The p-value is the usual asymptotic approximation; it is two-sided. |
| `gpu-spatial-analysis` | `GPUPickRegionMask` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics) | Converts a capacity-bounded index-picking region result into a deduplicated source-aligned 0/1 mask. The application adds `GPUIndexPickingTarget.addRegionPass` before this contributor; the graph orders the two through the hazard on `picks`. |
| `gpu-spatial-analysis` | `GPUPointDensity` | [Contract](./gpu-spatial-analysis/operators-density.md#gpupointdensity) | Bins points into a square or hexagonal grid and writes per-cell count, sum, mean, a heatmap field, its extent and histogram, and an optional `r32float` texture, all on the GPU. |
| `gpu-spatial-analysis` | `GPUPointInPolygonJoin` | [Contract](./gpu-spatial-analysis/operators-joins.md#spatial-sort-default) | Joins planar points to the polygon features that contain them. |
| `gpu-spatial-analysis` | `GPUPointPatternEnvelope` | [Contract](./gpu-spatial-analysis/point-process.md#simulation-curves-and-global-tests) | Pointwise order-statistic envelopes and a distinct global deviation test, composed over GPU simulation curves. |
| `gpu-spatial-analysis` | `GPUPointToCell` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpupointtocell-and-gpucellgeometry) | Converts longitude/latitude points to 64-bit discrete global grid cell keys, one row per row. |
| `gpu-spatial-analysis` | `GPUPolygonize` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpulinenoding-gpupolygonize-and-gpumakevalid) | General bounded polygonization of planar linework. |
| `gpu-spatial-analysis` | `GPUPolygonOverlay` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpupolygonoverlay-and-gpubuffersurface) | General polygon Boolean overlay over a snap-rounded planar arrangement. |
| `gpu-spatial-analysis` | `GPUPolygonPrecisionReduction` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Hot-pixel noding plus even-odd polygon reconstruction. |
| `gpu-spatial-analysis` | `GPUPolygonTriangulation` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpupolygontriangulation) | Triangulates polygons with holes on the GPU: the Earcut algorithm behind deck.gl's polygon tessellation, comparable to `geo` `TriangulateEarcut` and Sedona `ST_TriangulatePolygon`. |
| `gpu-spatial-analysis` | `GPUPopulationSupportAlignment` | [Contract](./gpu-spatial-analysis/landscape-ecology.md#gpupopulationsupportalignment) | Exact support-ID alignment using two sort/index joins and parallel status reductions. |
| `gpu-spatial-analysis` | `GPUProbitRegression` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity) | Nonspatial binary probit with bounded Fisher scoring and monotone likelihood backtracking. Complete cases, centered/scaled predictors, implicit intercept. The log likelihood and score are row-tile sums merged in a fixed pairwise order, and the line search tolerates likelihood noise relative to /loglik/ and sqrt(rows). |
| `gpu-spatial-analysis` | `GPUPycnophylacticInterpolation` | [Contract](./gpu-spatial-analysis/operators-accessibility.md#gpupycnophylacticinterpolation) | Tobler's pycnophylactic (volume-preserving) interpolation on a raster: a smooth density surface whose mass in every zone equals the zone total. Reference: Tobler, W. R. (1979), "Smooth pycnophylactic interpolation for geographical regions", JASA 74(367), 519-530. |
| `gpu-spatial-analysis` | `GPUQuantileLISA` | [Contract](./gpu-spatial-analysis/quantile-lisa.md#gpuquantilelisa) | Univariate or multivariate selected-quantile conditional local join counts. |
| `gpu-spatial-analysis` | `GPURandomPointsInPolygon` | [Contract](./gpu-spatial-analysis/operators-density.md#gpudotdensity-and-gpurandompointsinpolygon) | Places `counts[feature]` uniformly random points inside each polygon feature (holes excluded), optionally thinned by a weight raster. Empty polygons and degenerate rings are skipped on the GPU; zero-area features produce zero points and report requests in `output.status.failedCount`. |
| `gpu-spatial-analysis` | `GPURandomPointsOnLine` | [Contract](./gpu-spatial-analysis/operators-density.md#gpudotdensity-and-gpurandompointsinpolygon) | Places `counts[line]` uniformly random points along each linestring, like GeoPandas `GeoSeries.sample_points(size)` on lines (uniform in arc length) and the line counterpart of . |
| `gpu-spatial-analysis` | `GPURankMarkov` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#joint-randomization-and-rank-dynamics) | giddy full-rank and geographic-rank transition conventions, pinned to giddy 2.3.6 (the version that generated the reference in `test/oracles/fixtures/statistics-upstream.json`; the other Markov contributors are pinned to giddy 2.3.9 in `statistics-markov.json`). |
| `gpu-spatial-analysis` | `GPURasterFalseDiscoveryRateMask` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpurasterfalsediscoveryratemask) | Field significance of a p-value raster: the multiple-testing adjustment of every tested cell (`GPUMultipleTestingAdjustment`: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli) and a significance mask `adjusted <= alpha`, which is statsmodels' `reject`. |
| `gpu-spatial-analysis` | `GPURatioConcaveHull` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Edge-length-ratio hull from Delaunay border erosion. |
| `gpu-spatial-analysis` | `GPURectangleClip` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpurectangleclip) | Clips lines and polygon rings to an axis-aligned rectangle (turf `bboxClip`, Shapely `clip_by_rect`, PostGIS `ST_ClipByBox2D`), with the rectangle read per frame, so a viewport clip runs without recompiling. |
| `gpu-spatial-analysis` | `GPURedcapSingleLinkage` | [Contract](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md#redcap-single-linkage-gpuredcapsinglelinkage) | Contiguity-constrained single-linkage clustering: the single-linkage dendrogram of the contiguity graph cut at `k` regions, the region count chosen per frame. This is the first step of first-order single-linkage REDCAP (Guo 2008), the tree, with a different second step, so it is not the REDCAP partition (see "What this is not"). |
| `gpu-spatial-analysis` | `GPURegionMask` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics) | Writes a 0/1 source-aligned mask of points inside a per-frame rectangle or even-odd lasso polygon, in world space or in screen pixels through a view-projection transform. |
| `gpu-spatial-analysis` | `GPURegionPartitionEvaluation` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | Evaluates a region partition on any label column: within-region and between-region sum of squares, the pseudo-F (Calinski-Harabasz) statistic, region sizes and a contiguity-graph boundary measure. |
| `gpu-spatial-analysis` | `GPURegionStatistics` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics) | Selects points inside a per-frame rectangle, circle, lasso, or pick region and reduces them into counts, value statistics, a histogram, and stable selected IDs. |
| `gpu-spatial-analysis` | `GPURestrictedRegimeRegression` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | Restricted OLS with any common/varying intercept and slopes, plus the joint nested-model Chow F statistic against fully varying regimes on exactly the same complete observations. |
| `gpu-spatial-analysis` | `GPURoseStatistic` | [Contract](./gpu-spatial-analysis/operators-statistics.md#additional-spatial-statistics-and-bounded-optimization) | Directional or axial Rose statistic for planar line segments. |
| `gpu-spatial-analysis` | `GPUS2CapCovering` | [Contract](./gpu-spatial-analysis/global-coordinates.md#s2-identity-measurement-and-covers) | Bounded uniform-level spherical-cap covering. The `'uniform'` enumeration is the reference tier for levels up to 8; `refinement: 'level-synchronous'` refines from the six face cells and is parallel over frontier cells. No host readback or hidden refinement scheduler. |
| `gpu-spatial-analysis` | `GPUS2CellHierarchy` | [Contract](./gpu-spatial-analysis/global-coordinates.md#s2-identity-measurement-and-covers) | Exact S2 integer hierarchy at all 31 levels, including mixed-level input batches. |
| `gpu-spatial-analysis` | `GPUS2CellMeasures` | [Contract](./gpu-spatial-analysis/global-coordinates.md#s2-identity-measurement-and-covers) | S2 spherical cell measurements at levels 0–30 using cancellation-free local increments. Per-frame: cells contents. Compile-time: precision, row count and view layouts. With precision 'double-single', uses the published double-single module for projection and local increments. |
| `gpu-spatial-analysis` | `GPUS2PolygonCovering` | [Contract](./gpu-spatial-analysis/global-coordinates.md#adaptive-polygon-covers) | Adaptive mixed-level S2 interior/exterior cover of one polygon, in a bounded GPU traversal. |
| `gpu-spatial-analysis` | `GPUSeeminglyUnrelatedRegression` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | Nonspatial one-step feasible SUR: equation OLS -> JOINT residual covariance -> system GLS. |
| `gpu-spatial-analysis` | `GPUSegmentIntersection` | [Contract](./gpu-spatial-analysis/operators-validity.md#gpusegmentintersection) | Finds every intersecting pair of line segments between two geometries, or within one. |
| `gpu-spatial-analysis` | `GPUSegmentRingAssembly` | [Contract](./gpu-spatial-analysis/operators-cells.md#gpusegmentringassembly) | Chains directed boundary segments into closed rings on the GPU. |
| `gpu-spatial-analysis` | `GPUSegmentSnap` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Snap endpoints to nearest reference vertices, then insert all reference vertices within tolerance of each snapped segment in projection order. |
| `gpu-spatial-analysis` | `GPUSegregation` | [Contract](./gpu-spatial-analysis/operators-accessibility.md#gpusegregation) | Residential segregation indices over population groups per unit (the PySAL `segregation` package), aspatial or spatial, global or local, and as a multiscale profile. |
| `gpu-spatial-analysis` | `GPUShapeDescriptors` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpushapedescriptors) | Per-polygon shape descriptors: Polsby-Popper and Schwartzberg compactness, convexity, elongation and orientation, plus clockwise and sliver flags (momepy and PySAL `esda` compactness, turf `polygonSmooth` diagnostics, PostGIS `ST_IsPolygonCW`, QGIS "Polygon compactness"). |
| `gpu-spatial-analysis` | `GPUShapeGenerator` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpushapegenerator) | Generates circle, sector and ellipse rings around per-feature centers (turf `circle`, `sector`, `ellipse`; the "buffer a point" workload), one invocation per output vertex. |
| `gpu-spatial-analysis` | `GPUSharedPaths` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpulineclipbypolygon-and-gpusharedpaths) | Finds the paths that two sets of linestrings share (Shapely `shared_paths`, GeoPandas has no direct name), with the direction of the right line relative to the left line. |
| `gpu-spatial-analysis` | `GPUSimilarLocations` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpusimilarlocations) | Feature-space nearest neighbors: ranks every row by its weighted Euclidean distance in standardized attribute space to a reference row, or to the mean of several reference rows (the CARTO `FIND_SIMILAR_LOCATIONS` and ArcGIS Similarity Search model). |
| `gpu-spatial-analysis` | `GPUSinkhornDivergence` | [Contract](./gpu-spatial-analysis/sinkhorn-divergence.md#gpusinkhorndivergence) | Sinkhorn divergence between two spatial distributions on the same grid (Genevay et al. |
| `gpu-spatial-analysis` | `GPUSkaterRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | SKATER regionalization: greedy cuts of a minimum spanning forest into spatially contiguous regions that minimise within-region sum of squares (Assunção et al. 2006; PySAL `spopt` Skater). |
| `gpu-spatial-analysis` | `GPUSpaceTimeCrossCorrelation` | [Contract](./gpu-spatial-analysis/space-time-statistics.md#cross-correlation-per-location-gpuspacetimecrosscorrelation) | Time-series cross-correlation per location over a set of lags (STX-02), with false discovery control across locations and lags. |
| `gpu-spatial-analysis` | `GPUSpaceTimeHawkes` | [Contract](./gpu-spatial-analysis/space-time-hawkes.md#gpuspacetimehawkes) | Space-time Hawkes process fitted by expectation-maximization with bounded parent windows (Veen and Schoenberg 2008 for the EM; the exponential-Gaussian kernel is the fixed-kernel special case of the ETAS family, without magnitudes, which are later work). |
| `gpu-spatial-analysis` | `GPUSpaceTimeKernelDensity` | [Contract](./gpu-spatial-analysis/space-time-density-and-k-function.md#space-time-kernel-density-gpuspacetimekerneldensity) | Space-time kernel density on a voxel cube with a fixed product kernel (sparr 2.3-16 `spattemp.density`, Fernando and Hazelton 2014; fixed bandwidths). |
| `gpu-spatial-analysis` | `GPUSpaceTimeKriging` | [Contract](./gpu-spatial-analysis/operators-density.md#gpuspacetimekriging) | Local space-time kriging with one neighbor search per query, over a per-frame covariance from `SpaceTimeVariogramModel` (the gstat definitions, plus Gneiting). |
| `gpu-spatial-analysis` | `GPUSpaceTimeLocalMoran` | [Contract](./gpu-spatial-analysis/space-time-statistics.md#lagged-and-differential-local-moran-gpuspacetimelocalmoran) | Lagged and differential local Moran of one period of a space-time cube (STX-02). |
| `gpu-spatial-analysis` | `GPUSpaceTimeRelativeRisk` | [Contract](./gpu-spatial-analysis/space-time-density-and-k-function.md#space-time-relative-risk-gpuspacetimerelativerisk) | Compares case and control space-time kernel densities on shared windows, bandwidths and grid support. Probability normalization compares relative distributions; intensity normalization retains population mass. |
| `gpu-spatial-analysis` | `GPUSpatialBalancedSample` | [Contract](./gpu-spatial-analysis/point-process.md#spatial-probability-sampling) | Finite-point stratified spatially ordered probability sampling. |
| `gpu-spatial-analysis` | `GPUDBSCAN` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpudbscan) | GPU DBSCAN density clustering of two-dimensional planar points. |
| `gpu-spatial-analysis` | `GPUSpatialCrossValidatedOLS` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) | Repeated resident OLS fits over spatially buffered block folds. |
| `gpu-spatial-analysis` | `GPUSpatialCrossValidation` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation) | Seeded spatial block holdouts with exact all-pairs planar buffer exclusion. Every valid row appears in exactly one test fold. Invalid rows appear in neither set. This bounded baseline costs O(folds * rows²); it makes no claim to blockCV's balancing or polygon generation. |
| `gpu-spatial-analysis` | `GPUSpatialEmpiricalBayesRates` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialempiricalbayesrates) | Spatial (neighborhood-pooled) rate smoothing, matching PySAL esda 2.10 `smoothing.Spatial_Rate` and `smoothing.Spatial_Empirical_Bayes`. |
| `gpu-spatial-analysis` | `GPUSpatialEntropy` | [Contract](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md#spatial-entropy-gpuspatialentropy-and-gpucategorypaircounts) | Global spatial entropy of a categorical map over a weights graph: the Shannon entropy of the categories, the co-occurrence entropy of the category pairs at the links (Leibovici 2009) and the mutual information between a link's two category labels. |
| `gpu-spatial-analysis` | `GPUSpatialErrorGM` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialerrorgm) | Generalized-moments estimator of the spatial error model `y = X b + u`, `u = lambda W u + e`. |
| `gpu-spatial-analysis` | `GPUSpatialErrorMaximumLikelihood` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialerrormaximumlikelihood) | Maximum-likelihood estimator of the spatial error model `y = X b + u`, `u = lambda W u + e`, `e ~ N(0, s2 I)` (Anselin 1988; spreg `ML_Error` with `method='full'`, whose exact log-determinant is replaced here by a truncated trace series). Nothing is read back. |
| `gpu-spatial-analysis` | `GPUSpatialInteraction` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#calibrated-spatial-interaction) | Calibrated one-covariate Poisson gravity models over a grid or bounded continuous coefficient interval. |
| `gpu-spatial-analysis` | `GPUSpatialJoinCandidates` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpuspatialjoincandidates) | The bounding-box candidate stage of a spatial join on its own: every `(left, right)` feature pair whose bounding boxes overlap (after expanding the left box by `distance`), sorted by left row and then right row. |
| `gpu-spatial-analysis` | `GPUSpatialJoinPrepared` | [Contract](./gpu-spatial-analysis/operators-joins.md#gpuspatialjoinprepared) | A prepared static right-hand side whose geometry snapshot changes only when invalidated. |
| `gpu-spatial-analysis` | `GPUSpatialKMeans` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialkmeans) | Deterministic k-means (Lloyd's algorithm) of planar points. |
| `gpu-spatial-analysis` | `GPUSpatialLag` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatiallag) | Spatial lag `lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`) for a . |
| `gpu-spatial-analysis` | `GPUSpatialLagImpacts` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatiallagimpacts) | Average direct, indirect and total impacts of the predictors of a spatial lag model `y = rho W y + X beta + e` (LeSage and Pace 2009), from a fit table and the power traces of `W`. |
| `gpu-spatial-analysis` | `GPUSpatialLagMaximumLikelihood` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatiallagmaximumlikelihood) | Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e`, `e ~ N(0, s2 I)` (Anselin 1988; spreg `ML_Lag` with `method='full'`, whose exact log-determinant is replaced here by a truncated trace series). Nothing is read back. |
| `gpu-spatial-analysis` | `GPUSpatialMarkov` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gpuspatialmarkov-and-gpulisamarkov) | Spatial Markov (Rey 2001, giddy `Spatial_Markov`): transition counts of a variable's classes conditioned on the class of the spatial lag at the start of each transition. |
| `gpu-spatial-analysis` | `GPUSpatialPanelErrorMaximumLikelihood` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur) | Fixed-effects spatial-error Gaussian ML: y_it = alpha_i + X_it beta + u_it, u_t = lambda W u_t + epsilon_t. |
| `gpu-spatial-analysis` | `GPUSpatialPanelLagMaximumLikelihood` | [Contract](./gpu-spatial-analysis/coverage-statistics.md#random-and-unit-effects-spatial-lag-maximum-likelihood) | Balanced spatial lag ML, random or unit effects. |
| `gpu-spatial-analysis` | `GPUSpatialPanelRegression` | [Contract](./gpu-spatial-analysis/statistical-expansion.md#pooled-spatial-panels-and-regimes) | Pooled spatial-lag panel 2SLS: y_t = rho W y_t + X_t beta + intercept + e_t. |
| `gpu-spatial-analysis` | `GPUSpatialPearson` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic) | Weighted Pearson correlation across directed spatial edges `(i,j)`: the weighted covariance of `x_i` and `y_j`, divided by their weighted edge standard deviations. This definition is bounded to `[-1, 1]`, is valid for asymmetric weights, and returns NaN for zero edge weight or variance. |
| `gpu-spatial-analysis` | `GPUSpatialPredicateJoin` | [Contract](./gpu-spatial-analysis/operators-joins.md#spatial-context-of-gpuspatialpredicatejoin) | Returns matching feature pairs for the selected spatial predicate. Choose boundary-inclusive coveredBy for every containing polygon; use GPUPointInPolygonJoin when each point needs one owning polygon. |
| `gpu-spatial-analysis` | `GPUSpatialRegimeRegression` | [Contract](./gpu-spatial-analysis/statistical-expansion.md#pooled-spatial-panels-and-regimes) | Fully varying spatial-regime OLS: separate intercept and slopes for each supplied region. |
| `gpu-spatial-analysis` | `GPUSpatialRegressionDiagnostics` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialregressiondiagnostics) | Spatial diagnostics for an ordinary least squares fit (spreg `diagnostics_sp`: `LMtests` and `MoranRes`): LM-lag, LM-error, robust LM-lag, robust LM-error, LM-SARMA, and Moran's I of the residuals with its regression-residual mean, variance, normal z and p-value. This is the GeoDa "fit OLS, then which spatial model?" panel. |
| `gpu-spatial-analysis` | `GPUSpatialScanStatistic` | [Contract](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md#prospective-mode-of-gpuspatialscanstatistic) | Kulldorff's spatial and space-time scan statistic (SaTScan; CARTO `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or points, with the Poisson, Bernoulli (Kulldorff 1997) or space-time permutation (Kulldorff et al. 2005) model. |
| `gpu-spatial-analysis` | `GPUSpatialTwoStageLeastSquares` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialtwostageleastsquares) | Spatial two-stage least squares for the lag model `y = rho W y + X b + e` (spreg `GM_Lag` with `w_lags = 1`). The endogenous regressor `W y` is instrumented with `[1, X, W X]` (the lag of every predictor, never of the intercept). |
| `gpu-spatial-analysis` | `GPUSpatialWeightsAlgebra` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightsalgebra) | Algebra on : set operations, higher orders, self weights, subgraphs and block weights, producing a new CSR on the GPU with a bounded output and an overflow flag. |
| `gpu-spatial-analysis` | `GPUSpatialWeightsMinimumSpanningTree` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | Minimum spanning forest over a graph, with edge cost the squared Euclidean distance between the attribute rows of the two neighbors (the metric of SKATER). |
| `gpu-spatial-analysis` | `GPUSpatialWeightsSummary` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightssummary) | Per-frame: weights and symmetryTolerance. Compile-time: view shapes and chunk layout. |
| `gpu-spatial-analysis` | `GPUSpatialWeightsTraces` | [Contract](./gpu-spatial-analysis/operators-regression.md#gpuspatialweightstraces) | Powers of a spatial weights matrix: the traces `tr(W^q)` and the sums `1' W^q 1` for `q = 1..Q`, the inputs of spatial-lag log-determinants and of direct/indirect/total impacts (see and `GPUSpatialLagImpacts`). |
| `gpu-spatial-analysis` | `GPUSpatialWeightsTransform` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightstransform) | Transforms spatial weights on the GPU: row standardization, binarization, distance kernels and symmetrization of a (the PySAL `transform` and `Kernel` equivalents). |
| `gpu-spatial-analysis` | `GPUSpatialWeightsTranspose` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightstranspose) | Per-frame: CSR contents and symmetryTolerance. Compile-time: view shapes, columnCount and which optional outputs are present. Numeric tolerances use the same GPUParameterBuffer layout as getGPUSpatialWeightsTransposeParameterValues. |
| `gpu-spatial-analysis` | `GPUSpatialWeightsValidation` | [Contract](./gpu-spatial-analysis/operators-weights.md#gpuspatialweightsvalidation) | Checks the invariants on the GPU, for CSR that was built elsewhere and uploaded (producers in this package already guarantee them). |
| `gpu-spatial-analysis` | `GPUSphericalArcOverlay` | [Contract](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) | Bounded reference spherical arc arrangement and ring assembly, including non-hemisphere regions. |
| `gpu-spatial-analysis` | `GPUSphericalConvexContains` | [Contract](./gpu-spatial-analysis/global-coordinates.md#convex-spherical-geometry-and-overlap-weights) | Convex spherical containment with explicit boundary and complement semantics. |
| `gpu-spatial-analysis` | `GPUSphericalConvexIntersection` | [Contract](./gpu-spatial-analysis/global-coordinates.md#convex-spherical-geometry-and-overlap-weights) | Intersection of paired convex spherical polygons contained strictly in an open hemisphere. |
| `gpu-spatial-analysis` | `GPUSphericalPolygonContains` | [Contract](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) | Nonconvex spherical winding containment with holes and explicit complementary-region semantics. |
| `gpu-spatial-analysis` | `GPUSphericalPolygonOverlay` | [Contract](./gpu-spatial-analysis/global-coordinates.md#general-spherical-topology) | General nonconvex shell/hole Boolean overlay in a supplied spherical hemisphere chart. |
| `gpu-spatial-analysis` | `GPUStudyWindowPreparation` | [Contract](./gpu-spatial-analysis/operators-density.md#gpustudywindowpreparation) | Prepares one polygon or multipolygon study window: oriented edges with the interior on the left, area, perimeter, finite-vertex bounds, edge slabs and a compact row-major sampling grid. Consumers accept this handle as `studyWindow`; add it to their graph first. |
| `gpu-spatial-analysis` | `GPUSubdividePolygons` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Bounded convex subdivision: triangulate polygon interiors (including holes), then merge edge-adjacent triangles while the union remains convex and within the vertex limit. |
| `gpu-spatial-analysis` | `GPUTrackSimilarity` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputracksimilarity) | Computes Hausdorff, discrete Frechet and maximum vertex distance between pairs of lines or polygon boundaries (rings given as closed vertex sequences). |
| `gpu-spatial-analysis` | `GPUTrackSimilaritySearch` | [Contract](./gpu-spatial-analysis/track-similarity-measures.md#lower-bounds-and-pruned-search) | Batched planar threshold search composing the existing tiled dynamic programs. |
| `gpu-spatial-analysis` | `GPUTrajectoryClosestApproach` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputrajectoryclosestapproach) | Exact encounters between trajectories, the MobilityDB `tdwithin` / `nearestApproachDistance` question on the raw samples: tracks move linearly between their samples, and two tracks are an encounter while their positions at the same instant are at most `distance` apart. |
| `gpu-spatial-analysis` | `GPUTrajectoryEncounters` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputrajectoryencounters) | Finds pairs of tracks that come within a distance of each other in the same time bucket, the MobilityDB `tdwithin` question for resampled trajectories. |
| `gpu-spatial-analysis` | `GPUTrajectoryHampel` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectoryhampel) | Hampel filter per track (the `hampel` package used by PTRAIL). |
| `gpu-spatial-analysis` | `GPUTrajectoryKalman` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | Constant-velocity Kalman filter and optional Rauch–Tung–Striebel smoother, parallel across tracks. |
| `gpu-spatial-analysis` | `GPUTrajectoryLocations` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorylocations) | Staypoint locations per track with trackintel's `generate_locations` (DBSCAN, `num_samples=1`, `agg_level='user'`) semantics: connected components of the graph joining staypoints of one track within `epsilon` (inclusive, haversine in meters for a great-circle context). |
| `gpu-spatial-analysis` | `GPUTrajectoryMetrics` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputrajectorymetrics) | Computes per-track movement metrics and stop (dwell) detection for GPS-style tracks. |
| `gpu-spatial-analysis` | `GPUTrajectoryOutlierFlags` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectoryoutlierflags) | Point-centred speed and acceleration outliers (review finding FX-08): unlike the `'step'` test of `addTrackQualityRecipe`, which flags the row ending a fast step, a fix is an outlier only when both adjacent step speeds exceed the limit, so the valid fix after a spike is left alone. |
| `gpu-spatial-analysis` | `GPUTrajectoryPlayhead` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputrajectoryplayhead-and-gputrajectoryresample) | Interpolates every track at a per-frame playhead time: the "current positions" of a trips map. |
| `gpu-spatial-analysis` | `GPUTrajectoryRecursions` | [Contract](./gpu-spatial-analysis/home-ranges-and-recursions.md#gputrajectoryrecursions) | Recursion analysis of trajectories after Bracis, Bildstein and Mueller (Ecography 2018, `recurse::getRecursions`), and a forward first passage time: for the circle of radius `r` around each fix, how often the track returns, how long it stays, and how long it takes to leave. |
| `gpu-spatial-analysis` | `GPUTrajectoryResample` | [Contract](./gpu-spatial-analysis/operators-lines.md#gputrajectoryplayhead-and-gputrajectoryresample) | Resamples every track to a fixed number of samples, the dense input for trajectory similarity, k-means, and smooth fixed-vertex trails. |
| `gpu-spatial-analysis` | `GPUTrajectoryRestriction` | [Contract](./gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis) | Time/geometry restriction of planar linearly interpolated tracks, with explicit endpoint policy, polygon rings, optional interval union and continuous pairwise within-distance predicates. Positive-duration gaps above a nonzero maximumGap are unknown support, never straight-line bridges. |
| `gpu-spatial-analysis` | `GPUTrajectorySegments` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorysegments) | Triplegs and trips from positionfixes and staypoints, with trackintel's `between_staypoints` tripleg rule and `generate_trips` state machine. |
| `gpu-spatial-analysis` | `GPUTrajectorySplit` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorysplit) | Splits tracks into segments and rewrites the offsets. `'gap'` and `'value-change'` follow MovingPandas 0.23 (pinned by a committed fixture); `'stop'` is the trackintel tripleg rule: |
| `gpu-spatial-analysis` | `GPUTrajectoryStaypoints` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorystaypoints) | Staypoint detection with trackintel's `sliding` semantics (Li et al. 2008), computed in a bounded number of GPU passes. |
| `gpu-spatial-analysis` | `GPUTrajectoryThinning` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorythinning) | Minimum-distance and minimum-time generalization with MovingPandas' sequential semantics in a bounded number of passes: the first and last rows of a track are always kept, and each kept row is followed by the first later row at or beyond the tolerance. |
| `gpu-spatial-analysis` | `GPUTrajectoryTrackRelativeTimes` | [Contract](./gpu-spatial-analysis/movement-semantics.md#gputrajectorytrackrelativetimes) | Int64 word timestamps to `float32` ticks after each track's own first row. |
| `gpu-spatial-analysis` | `GPUTransitionDiagnostics` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix) | Diagnostics of a transition-count tensor: homogeneity of the conditional matrices, steady states of the pooled and conditional chains, and a minimum-cell-count count. |
| `gpu-spatial-analysis` | `GPUTransitionMatrix` | [Contract](./gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix) | Exact class transition counts across periods (giddy `Markov`): how many locations moved from class `a` at period `t` to class `b` at period `t + periodLag`, summed over all `t`, with row-normalized probabilities. |
| `gpu-spatial-analysis` | `GPUUtilizationVolumeIsopleths` | [Contract](./gpu-spatial-analysis/home-ranges-and-recursions.md#gpuutilizationvolumeisopleths) | Volume isopleths of a utilization distribution: the cumulative probability volume of every cell when cells are taken from the densest down, each cell's own mass included, and the mask of the `level` isopleth (the home range at that level). |
| `gpu-spatial-analysis` | `GPUVecchiaFisherScoring` | [Contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchiafisherscoring) | One Fisher scoring evaluation of a Vecchia likelihood (Guinness 2019): the log-likelihood, its score with respect to `theta = (log sill, log effective range, log nugget)` and the expected information. |
| `gpu-spatial-analysis` | `GPUVecchiaLogLikelihood` | [Contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchialoglikelihood) | Log-likelihood of a zero-mean Gaussian process under the Vecchia approximation (Vecchia 1988; Datta et al. 2016): `log L = sum_i -1/2 [log 2 pi + log d_i + r_i^2 / d_i]`, where `d_i` and the mean of `r_i = y_i - mean_i` are the kriging conditional variance and mean of site `i` given its earlier conditioning set. |
| `gpu-spatial-analysis` | `GPUVecchiaSpaceTimeLogLikelihood` | [Contract](./gpu-spatial-analysis/vecchia-likelihood.md#gpuvecchiaspacetimeloglikelihood) | Space-time Vecchia log-likelihood over caller-ordered conditioning sets, with the per-frame `SpaceTimeVariogramModel` covariance of `GPUSpaceTimeKriging`. |
| `gpu-spatial-analysis` | `GPUVectorStraightSkeleton` | [Contract](./gpu-geometry-refinement.md#gpu-geometry-refinement) | Event-driven planar wavefront with explicit linked half-edge topology, reflex splits and hole merges. |
| `gpu-spatial-analysis` | `GPUVertexSnap` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpuoffsetcurve-and-gpuvertexsnap) | Vertex-to-vertex snapping (`shapely.snap`, `shapely.ops.snap`, geo `snap`/Sedona `ST_Snap` subset): every vertex with a reference vertex strictly closer than `tolerance` moves onto the nearest such reference vertex, the lowest reference row winning ties; other vertices are kept bit-for-bit. |
| `gpu-spatial-analysis` | `GPUVoronoiDiagram` | [Contract](./gpu-spatial-analysis/operators-geometry.md#gpudelaunaytessellation-and-gpuvoronoidiagram) | Derives vector Voronoi edges from accepted Delaunay topology entirely on the GPU. |
| `gpu-spatial-analysis` | `GPUWardRegions` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuwardregions) | Spatially constrained Ward agglomeration with an explicit merge budget. |
| `gpu-spatial-analysis` | `GPUZonalStatistics` | [Contract](./gpu-spatial-analysis/operators-statistics.md#gpuzonalstatistics) | Choropleth aggregation: joins points to features and reduces optional per-point values and weights into per-feature statistics, all on the GPU. |
| `gpu-spatial-analysis` | `GPUZoneEvents` | [Contract](./gpu-spatial-analysis/operators-lines.md#gpuzoneevents) | Detects when trajectories enter and leave polygon zones, with interpolated times. |


## Browse by data and task

These chooser links mirror the sidebar records. All existing page URLs, fragments and capability
IDs remain available. Coverage and recordings stay with their related task groups.

### Analysis core chooser

- [Shared fitting kernels](./analysis-core/fitting-kernels.md)

### Spatial analysis chooser

#### Spatial analysis: Start and choose

- [GPU Spatial Analysis](./gpu-spatial-analysis/README.md)
- ["GPU Spatial Analysis concepts"](./gpu-spatial-analysis/concepts.md)
- [Analysis experience vocabulary](./gpu-spatial-analysis/analysis-experience.md)
- [Selected spatial operator families](./gpu-spatial-analysis/operators.md)

#### Spatial analysis: Join and select

- ["GPU Spatial Analysis: Spatial joins"](./gpu-spatial-analysis/operators-joins.md)
- ["GPU Spatial Analysis: Geometry validity & intersections"](./gpu-spatial-analysis/operators-validity.md)

#### Spatial analysis: Measure and construct geometry

- ["GPU Spatial Analysis: Geometry utilities"](./gpu-spatial-analysis/operators-geometry.md)
- ["GPU Spatial Analysis: Lines & trajectories"](./gpu-spatial-analysis/operators-lines.md)
- [Geodesic precision (CR01)](./gpu-spatial-analysis/geodesic-precision.md)
- [Global coordinates, spherical geometry and S2 analysis](./gpu-spatial-analysis/global-coordinates.md)
- [GPU geometry refinement](./gpu-geometry-refinement.md)
- [Geometry refinement computation lenses](./gpu-spatial-analysis/geometry-refinement-lenses.md)

#### Spatial analysis: Build neighborhoods and cells

- ["GPU Spatial Analysis: Spatial weights"](./gpu-spatial-analysis/operators-weights.md)
- [GPUAnalysisContiguityStatistics](./gpu-spatial-analysis/gpu-analysis-contiguity-statistics.md)
- [GPUChunkedSpatialWeightsTranspose](./gpu-spatial-analysis/chunked-spatial-weights-transpose.md)
- [Spatial weights helper conventions](./gpu-spatial-analysis/spatial-weights-audit.md)
- ["GPU Spatial Analysis: Cells (DGGS)"](./gpu-spatial-analysis/operators-cells.md)
- [HEALPix cells and DGGS-JSON zone data](./gpu-spatial-analysis/healpix-cells-dggs-json.md)

#### Spatial analysis: Test spatial patterns

- ["GPU Spatial Analysis: Spatial statistics"](./gpu-spatial-analysis/operators-statistics.md)
- [GPUQuantileLISA](./gpu-spatial-analysis/quantile-lisa.md)
- [Geographical detector, REDCAP, spatial entropy and the contiguity-order correlogram](./gpu-spatial-analysis/geographical-detector-redcap-entropy.md)
- [Colocation quotient, local category dependence and adaptive-bandwidth density](./gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md)
- [Conditional LOSH residual inference](./gpu-spatial-analysis/losh-residual-permutation.md)
- [GPUDorlingCartogram](./gpu-spatial-analysis/gpu-dorling-cartogram.md)
- [Rate mapping recipes](./gpu-spatial-analysis/rate-mapping.md)
- [Point process contributors](./gpu-spatial-analysis/point-process.md)

#### Spatial analysis: Estimate and validate

- ["GPU Spatial Analysis: Density & interpolation"](./gpu-spatial-analysis/operators-density.md)
- ["GPU Spatial Analysis: Regression"](./gpu-spatial-analysis/operators-regression.md)
- [Adaptive generalized geographically weighted regression](./gpu-spatial-analysis/adaptive-generalized-gwr.md)
- [Spatial panel unit fixed effects](./gpu-spatial-analysis/panel-fixed-effects.md)
- [Instrumental-variable robust covariance](./gpu-spatial-analysis/instrumental-variables-covariance.md)
- [Proper scoring rules](./gpu-spatial-analysis/scoring-rules.md)
- [Sinkhorn divergence between spatial distributions (STM-09)](./gpu-spatial-analysis/sinkhorn-divergence.md)
- [Bounded GPU interpolation and simulation](./gpu-spatial-analysis/interpolation-expansion.md)
- [Vecchia likelihood (STM-07)](./gpu-spatial-analysis/vecchia-likelihood.md)
- [Gaussian random fields](./gpu-spatial-analysis/gaussian-random-field.md)
- [Bounded statistical estimation and validation](./gpu-spatial-analysis/coverage-statistics.md)
- [Spatial fold designs and nested interpolation validation](./gpu-spatial-analysis/spatial-validation-advanced.md)
- [Executable interpolation lenses](./gpu-spatial-analysis/interpolation-lenses.md)

#### Spatial analysis: Analyze change over time

- [Temporal context](./gpu-spatial-analysis/temporal-context.md)
- [Space-time weights, lagged and differential local Moran, cross-correlation](./gpu-spatial-analysis/space-time-statistics.md)
- [Space-time clustering](./gpu-spatial-analysis/space-time-clustering.md)
- [Space-time kernel density, relative risk, K function and Diggle test](./gpu-spatial-analysis/space-time-density-and-k-function.md)
- [Change points and the expectation-based scan](./gpu-spatial-analysis/change-points-and-expectation-based-scan.md)
- [Space-time Hawkes process (STM-08)](./gpu-spatial-analysis/space-time-hawkes.md)
- ["GPU Spatial Analysis: Distribution dynamics & space-time tests"](./gpu-spatial-analysis/operators-dynamics.md)

#### Spatial analysis: Analyze tracks and accessibility

- [Home ranges and trajectory recursions](./gpu-spatial-analysis/home-ranges-and-recursions.md)

- [Movement semantics](./gpu-spatial-analysis/movement-semantics.md)
- [Track similarity: EDR, ERP, SSPD and bounded search](./gpu-spatial-analysis/track-similarity-measures.md)
- [Trajectory estimation, restriction and search](./gpu-spatial-analysis/movement-coverage.md)
- ["GPU Spatial Analysis: Accessibility & change of support"](./gpu-spatial-analysis/operators-accessibility.md)

#### Spatial analysis: Compose urban and landscape workflows

- ["GPU Spatial Analysis: Recipes"](./gpu-spatial-analysis/recipes.md)
- [Geospatial workflow recipes](./gpu-spatial-analysis/expansion-recipes.md)
- [Dasymetric redistribution](./gpu-spatial-analysis/dasymetric-redistribution.md)
- [Urban morphology recipes](./gpu-spatial-analysis/urban-morphology.md)
- [Landscape ecology recipes](./gpu-spatial-analysis/landscape-ecology.md)
- [Urban morphology and landscape ecology computation lenses](./gpu-spatial-analysis/urban-ecology-lenses.md)
- [Geospatial expansion computation lenses](./gpu-spatial-analysis/expansion-lenses.md)

#### Spatial analysis: Coordinate and capability coverage

- [Spatial frames](./gpu-spatial-analysis/spatial-frames.md)
- [Geospatial coverage expansion](./gpu-spatial-analysis/coverage-expansion.md)
- [Local variability, point patterns and spatial regression extensions](./gpu-spatial-analysis/statistical-expansion.md)

#### Spatial analysis: Compatibility URLs

- [Home ranges and recursion (compatibility URL)](./gpu-spatial-analysis/home-range-and-recursion-mob07.md)

### Network chooser

#### Network: Start and choose

- [GPU Network](./gpu-network/README.md)
- ["GPU Network concepts"](./gpu-network/concepts.md)
- ["GPU Network operators"](./gpu-network/operators.md)

#### Network: Route and measure access

- [Network time geography](./gpu-network/network-time-geography.md)

- [Alternative routes](./gpu-network/alternative-routes.md)
- [K-shortest simple paths](./gpu-network/k-shortest-paths.md)

#### Network: Analyze topology and spatial patterns

- [GPUNetworkLouvain](./gpu-network/louvain.md)
- [Visibility graph and network centrality](./gpu-network/visibility-graph.md)
- [Network structure, routes and spatial topology extensions](./gpu-network/expansion.md)
- [Network point processes](./gpu-network/point-process.md)

#### Network: Analyze movement and flows

- [Origin-destination flow models](./gpu-network/origin-destination-flow-models.md)
- [Map-matched edge volumes](./gpu-network/map-matched-edge-volumes.md)

- [Bounded network optimization and trajectory analysis](./gpu-network/network-movement-coverage.md)
- [Network and movement measurements](./gpu-network/network-movement-benchmarks.md)

#### Network: Inspect recorded outputs and status

- ["Network status and count checks"](./gpu-network/status-oracles.md)

#### Network: Compatibility URLs

- [Network time geography (compatibility URL)](./gpu-network/network-time-geography-mob09.md)
- [Origin-destination flow models (compatibility URL)](./gpu-network/flow-models-mob10.md)
- [Map-matched edge volumes (compatibility URL)](./gpu-network/map-matched-edge-volumes-mob05.md)

### Terrain chooser

#### Terrain: Start and choose

- [GPU Terrain](./gpu-terrain/README.md)
- ["GPU Terrain concepts"](./gpu-terrain/concepts.md)
- ["GPU Terrain operators"](./gpu-terrain/operators.md)

#### Terrain: Route water and compare surfaces

- [Height above drainage over D-infinity flow](./gpu-terrain/height-above-drainage.md)
- [Terrain processing](./gpu-terrain/terrain-processing.md)
- [Terrain hydrology recordings](./gpu-terrain/expansion-lenses.md)

#### Terrain: Inspect recorded outputs and status

- [Terrain output oracles](./gpu-terrain/output-oracles.md)
- [Terrain status and count evidence](./gpu-terrain/status-and-count-evidence.md)

### Raster analysis chooser

#### Raster analysis: Start and choose

- [Raster analysis](./gpu-raster/README.md)
- [Raster analysis operations](./gpu-raster/operations.md)
- [GPURaster analysis contributors](./gpu-raster/operations-analysis.md)

#### Raster analysis: Prepare surfaces and geometry

- [Raster, terrain and geometry expansion](./gpu-raster/expansion.md)
- [Raster contrast equalization](./gpu-raster/contrast-equalization.md)
- [Nodata filling](./gpu-raster/fill-nodata.md)
- [GPURasterTileReprojection](./gpu-raster/tile-reprojection.md)
- [Raster operation recordings](./gpu-raster/expansion-lenses.md)

#### Raster analysis: Summarize zones and local structure

- [GPURasterFractionalZonalDistribution](./gpu-raster/fractional-zonal-distribution.md)
- [GPURasterFractionalCategoricalStatistics](./gpu-raster/fractional-categorical-statistics.md)
- [Focal cross-band statistics](./gpu-raster/focal-pairs.md)
- [Raster patch metrics and sieving](./gpu-raster/raster-patches.md)
- [GPUIsolines: seam-safe tiles and vector overlays](./gpu-raster/isolines-tiles.md)
- [Spectral raster filtering](./gpu-raster/spectral-filter.md)

#### Raster analysis: Compare spectra and embeddings

- [Calibrated spectral indices](./gpu-raster/spectral-indices.md)
- [Embedding raster codecs](./gpu-raster/raster-embedding-codecs.md)
- [Embedding rasters](./gpu-raster/raster-embeddings.md)
- [Bounded remote-sensing contributors](./gpu-raster/remote-sensing.md)

#### Raster analysis: Analyze time and transport

- [Flow-map and particle diagnostics](./gpu-raster/flow-map-and-particle-diagnostics.md)

- [Raster time series](./gpu-raster/raster-time-series.md)
- [Raster change monitoring](./gpu-raster/raster-change-monitoring.md)
- [Raster EOF analysis and lagged correlation](./gpu-raster/raster-eof-analysis.md)

#### Raster analysis: Compatibility URLs

- [Lagrangian diagnostics (compatibility URL)](./gpu-raster/lagrangian-diagnostics-mob06.md)

### Dataframe analysis chooser

- [GPU Dataframe analysis contributors](./gpu-dataframe-analysis.md)

- [GPUCoregionalizationFit](./dataframe-analysis/coregionalization-fit.md)

- [Forest inference](./forest-inference.md)

- [GPUVariableStandardization](./variable-standardization.md)

### Geospatial chooser

- [Geospatial analysis](./geospatial.md)

### Point cloud analysis chooser

- [Point cloud analysis](./pointcloud-analysis.md)

### Computation records and views chooser

- [Computation record](./computation-record.md)

- [Computation views authoring guide](./computation-views.md)

### Shared guides

- [Analysis modules](./README.md)
- [Operation forms and CPU twins](./operation-forms-and-cpu-twins.md)
- [Source topology and wide numeric profiling](./reused-analysis-preparation.md)
- [GPU spatial analysis cross-reference](./gpu-spatial-analysis-cross-reference.md)

### Recordings index

Browse [all recorded computations](/explore/) or open task-specific collections below. Each
recording sits beside the corresponding operation contract; absence of a recording is not absence
of an export.

- [Geospatial expansion computation lenses](./gpu-spatial-analysis/expansion-lenses.md)
- [Geometry refinement computation lenses](./gpu-spatial-analysis/geometry-refinement-lenses.md)
- [Executable interpolation lenses](./gpu-spatial-analysis/interpolation-lenses.md)
- [Urban morphology and landscape ecology computation lenses](./gpu-spatial-analysis/urban-ecology-lenses.md)
- ["Network status and count checks"](./gpu-network/status-oracles.md)
- [Terrain hydrology recordings](./gpu-terrain/expansion-lenses.md)
- [Terrain output oracles](./gpu-terrain/output-oracles.md)
- [Raster operation recordings](./gpu-raster/expansion-lenses.md)
