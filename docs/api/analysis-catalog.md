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

The Kepler example’s [validation and regression catalog](../../examples/deck/kepler-compute/VALIDATION-CATALOG.md) binds the existing GTWR, space-time kriging, Vecchia, applicability, conformal, scoring and FDR contributors with executable forms and diagnostics tables.

## The operator catalog

The interactive catalog links to recorded runs. The operator catalog is a data model that lists
every exported contributor class and every `add*Recipe` function of all nine entry
points: `analysis-core`, `computation-record`, `gpu-spatial-analysis`, `gpu-network`,
`gpu-terrain`, `raster-analysis`, `dataframe-analysis`, `pointcloud-analysis` and
`geospatial-analysis`. `analysis-core` includes the four fitting producers
`GPUFittingCoordinateEncoder`, `GPUFittingGradientSums`, `GPUFittingMinibatchOrder` and
`GPUAdamStep`. A test fails when an exported contributor or recipe has no entry.

Each operator has a capability family named for what it does (`network-routing`,
`terrain-hydrology`, `raster-focal`, `raster-time-series`, `dataframe-aggregation`,
`pointcloud-ground`), its package, input and output data kinds, an optional one-line `summary`,
`dimensionality` (`'2d+time'` for temporal operators, `'3d'` for point clouds),
and the tests and documentation that cover it.

All 261 original spatial descriptors now have summaries from public TSDoc, with editorial overrides
where the opening documentation is absent or chiefly provenance. Both spatial and all-entry text
queries search these summaries. The static metadata is prepared by
`node scripts/generate-spatial-catalog-summaries.mjs`; review its printed module before updating
`spatial-analysis-operator-summaries.ts`. Runtime discovery imports no constructors or source files.
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

Audited compact ports advertise `candidateOverflow` independently of final-capacity `overflow`.
For publishers that count retained candidates before compaction, `requiredCount` is a lower bound
after candidate loss. Check the publisher's contract: visibility CSR instead publishes `0xffffffff`
for an unknown requirement after candidate search exhaustion; never use it as an allocation size.
The [visibility CSR regression](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/urban-ecology/visibility-graph-csr.spec.ts),
“CSR failures suppress asymmetric graphs and misleading metrics”, distinguishes four required
directed slots from that unknown sentinel. Polygonize and make-valid publish `requiredRingCount`,
with no feature `requiredCount`. Gather publishes clamped `rowCount`, final-capacity `overflow`,
optional inherited `candidateOverflow` and optional `matchedCount`, with no `requiredCount`.
Nearest-join `neighborRows` and gather's `neighbors.rows` are original feature rows;
`neighborIds` may be caller identities.

Local Moran, local Geary, hot-spot analysis and global spatial statistics advertise the separate
input-weight `overflow` and `candidateOverflow` flags. Local statistics also publish `invalidCount`.
Point-in-polygon joins expose `candidateOverflow` separately; their top-level `status.overflow`
aggregates BVH, candidate and final-match capacity loss, while `matches.status.overflow` reports
only final-match capacity loss. Optional compact matches also publish `matches.status.count` and
`matches.status.requiredCount`.

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
