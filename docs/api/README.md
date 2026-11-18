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

## Run your first computation

The [CPU variogram walkthrough](./operation-forms-and-cpu-twins.md#variogram-fitting-without-a-graph)
fits generated data without a GPU. Follow it with the complete
[point-density GPU walkthrough](./operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
to create a device, allocate views, submit work, read typed results and release resources.

## Entries

| Entry | Use it for |
| --- | --- |
| [`@luma-contributors/analysis/gpu-spatial-analysis`](/docs/api/gpu-spatial-analysis) | Spatial weights and statistics, joins, density, cells, lines, geometry, trajectories, and regression. |
| [`@luma-contributors/analysis/gpu-network`](/docs/api/gpu-network) | CSR network analysis: reachability, service areas, accessibility, flow aggregation, and edge bundling. |
| [`@luma-contributors/analysis/gpu-terrain`](/docs/api/gpu-terrain) | Elevation-tile analysis: slope, contours, viewshed, hydrology, solar shadows, and relief shading. |
| [`@luma-contributors/analysis/raster-analysis`](/docs/api/gpu-raster) | Raster analysis: fill nodata, tile reprojection, fractional coverage, focal pairs, spectral indices, regions, contours, and statistics. |
| [`@luma-contributors/analysis/dataframe-analysis`](/docs/api/gpu-dataframe-analysis) | Analysis contributors over luma.gl GPU Dataframe columns: profiles, group statistics, joins, and standardization. |
| [`@luma-contributors/analysis/geospatial-analysis`](/docs/api/geospatial) | Geospatial contexts, datum grid transforms and point buffers layered on luma.gl GPU Project. |
| [`@luma-contributors/analysis/pointcloud-analysis`](/docs/api/pointcloud-analysis) | Point cloud neighborhoods, features, ground classification, registration, and change. |
| [`@luma-contributors/analysis/computation-record`](/docs/api/computation-record) | Recorded executions of a computation as plain data, for the lens views. |
| [`@luma-contributors/analysis/analysis-core`](/docs/api/gpu-spatial-analysis/concepts) | Shared kernels and contracts used by the analysis entries. |

See the [analysis catalog](/docs/api/analysis-catalog) for every contributor
at a glance and the [cross-reference](/docs/api/gpu-spatial-analysis-cross-reference)
for a map from familiar tool names to contributors.

Read [operation forms and CPU twins](/docs/api/operation-forms-and-cpu-twins) to choose between
caller-owned GPU views, named graph operations and CPU execution, with a complete variogram fit.

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
