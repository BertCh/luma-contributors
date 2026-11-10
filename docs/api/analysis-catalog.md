---
title: Analysis catalog
description: Every analysis contributor at a glance, one input-to-output figure each, drawn from recorded GPU runs.
---

# Analysis catalog

One figure per contributor across dataframe, spatial analysis, network, raster and terrain analysis.
Each figure is the element's input and output, drawn from a recorded run of the real GPU computation on
small real datasets. Click a figure to open its card: summary, example, parameters, and how it computes.

Elements still without a recorded figure are listed on their entry's overview page.

## The operator catalog

The figures below are generated from recorded runs. Separately, the operator catalog is a data
model that lists every exported contributor class and every `add*Recipe` function of all nine entry
points (`analysis-core` exports none): `computation-record`, `gpu-spatial-analysis`, `gpu-network`,
`gpu-terrain`, `raster-analysis`, `dataframe-analysis`, `pointcloud-analysis` and
`geospatial-analysis`. A test fails when an exported contributor or recipe has no entry.

Each operator has a capability family named for what it does (`network-routing`,
`terrain-hydrology`, `raster-focal`, `raster-time-series`, `dataframe-aggregation`,
`pointcloud-ground`), its package, input and output data kinds, a one-line `summary` taken from the
class documentation, `dimensionality` (`'2d+time'` for temporal operators, `'3d'` for point clouds),
and the tests and documentation that cover it.

### Tier and limits

Every operator has a `tier`:

- `'scale'`: the kernels are parallel over the data and sized by device limits. Limits that still
  exist (for example a neighbor capacity of 64) are listed in `limits`.
- `'reference'`: a single-invocation, exhaustive or small-dense design with a hard maximum size. Use
  these for small problems, oracles and teaching, not for million-row inputs. `limits` lists each
  maximum as `{name, maximum, reason, source}`, where `source` is the file that enforces it, or
  `limitation` describes a serial step that has no numeric maximum.

For example `GPUNetworkFacilityOptimization` is `'reference'`: it enumerates 2^facilities subsets, or
facilities^demand assignments for the capacitated p-median, and rejects more than 2^20 candidates.
The tier describes the code as it is now. When a contributor is reworked to remove its limit, its entry
in `all-entry-points/operator-annotations.ts` changes with it.

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
```

`queryGPUAnalysisOperators` also accepts the filters of `queryGPUSpatialAnalysisOperators`
(`primaryCapability`, `input`, `output`, `coordinateModel`, `semantics`, `evidence`, `search`).

`GPU_SPATIAL_ANALYSIS_OPERATORS`, `GPU_SPATIAL_ANALYSIS_CAPABILITIES` and the functions that read them
still describe `gpu-spatial-analysis` (and `GPUTerrainPeakProminence`) only, because the website
inventory generator and the kepler catalog bindings enumerate them. Operators of the other entry
points come from `getGPUAnalysisOperatorCatalog()`. The original operators also carry `tier`,
`limits` and `role` now.

Operators added to the catalog after the first families do not repeat the family scale figures
(`representativeSourceRows`, `transientMemory`, `dispatch`); their `evidence` records tests and
documentation, with `status: 'partial'` and `contract: 'unaudited'` until a contract is assigned.

::lens-catalog
