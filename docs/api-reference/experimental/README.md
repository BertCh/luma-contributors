---
title: Analysis modules
description: Geospatial and analysis operations in @luma-contributors/analysis, built on published luma.gl.
---

# Analysis modules

`@luma-contributors/analysis` is an extension of [luma.gl](https://luma.gl). It holds only the
geospatial and analysis operations; the GPU infrastructure they run on (`GPUCommandGraph`, GPU Core
operations, `GPUData`/`GPUVector`/`GPUTable`, GPU Dataframe, GPU Graph Algorithms, GPU Project, GPU
DGGS, GPU Trace, physics and simulation, renderers) is owned and documented by luma.gl. Operations
here are built from the public `@luma.gl/core`, `@luma.gl/shaders` and `@luma.gl/graph` APIs.

## Entries

| Entry | Use it for |
| --- | --- |
| [`@luma-contributors/analysis/gpu-spatial-analysis`](/docs/api-reference/experimental/gpu-spatial-analysis) | Spatial weights and statistics, joins, density, cells, lines, geometry, trajectories, and regression. |
| [`@luma-contributors/analysis/gpu-network`](/docs/api-reference/experimental/gpu-network) | CSR network analysis: reachability, service areas, accessibility, flow aggregation, and edge bundling. |
| [`@luma-contributors/analysis/gpu-terrain`](/docs/api-reference/experimental/gpu-terrain) | Elevation-tile analysis: slope, contours, viewshed, hydrology, solar shadows, and relief shading. |
| [`@luma-contributors/analysis/raster-analysis`](/docs/api-reference/experimental/gpu-raster) | Raster analysis: fill nodata, tile reprojection, fractional coverage, focal pairs, spectral indices, regions, contours, and statistics. |
| [`@luma-contributors/analysis/dataframe-analysis`](/docs/api-reference/experimental/gpu-dataframe-analysis) | Analysis contributors over luma.gl GPU Dataframe columns: profiles, group statistics, joins, and standardization. |
| [`@luma-contributors/analysis/geospatial-analysis`](/docs/api-reference/experimental/geospatial) | Geospatial contexts, datum grid transforms and point buffers layered on luma.gl GPU Project. |
| [`@luma-contributors/analysis/pointcloud-analysis`](/docs/api-reference/experimental/pointcloud-analysis) | Point cloud neighborhoods, features, ground classification, registration, and change. |
| [`@luma-contributors/analysis/computation-record`](/docs/api-reference/experimental/computation-record) | Recorded executions of a computation as plain data, for the lens views. |
| [`@luma-contributors/analysis/analysis-core`](/docs/api-reference/experimental/gpu-spatial-analysis/concepts) | Shared kernels and contracts used by the analysis entries. |

See the [analysis catalog](/docs/api-reference/experimental/analysis-catalog) for every contributor
at a glance and the [cross-reference](/docs/api-reference/experimental/gpu-spatial-analysis-cross-reference)
for a map from familiar tool names to contributors.

## What luma.gl owns

Link to the luma.gl documentation rather than to copies here:

- [GPU Core and `GPUCommandGraph`](https://luma.gl/docs/api-reference/experimental/gpu-core)
- [GPU Data and GPU Tables](https://luma.gl/docs/api-reference/graph/gpu-data)
- [GPU Dataframe and GPU SQL](https://luma.gl/docs/api-reference/experimental/gpu-dataframe)
- [GPU Graph Algorithms](https://luma.gl/docs/api-reference/experimental/gpu-graph)
- [GPU Project and GPU DGGS](https://luma.gl/docs/api-reference/experimental/gpu-project)
- [Glossary](https://luma.gl/docs/glossary)

## Conventions

Contributors declare work on a caller-owned `GPUCommandGraph` and return command nodes; callers own
the frame loop, command submission and readback. Entries are WebGPU-first; individual pages state
backend, capacity and precision limits.
