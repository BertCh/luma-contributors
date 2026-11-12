# @luma-contributors/analysis

GPU geospatial and analysis operations built on published [luma.gl](https://luma.gl)
(`@luma.gl/core`, `@luma.gl/shaders` and `@luma.gl/graph` at `10.0.0-alpha.13`). The GPU
infrastructure itself (command graphs, GPU Core operations, GPU tables, dataframes, graph algorithms)
comes from luma.gl. This package holds only the analysis code.

## Entry points

| Import | Contents | Docs |
| --- | --- | --- |
| `@luma-contributors/analysis/gpu-spatial-analysis` | Spatial weights, autocorrelation, regression, interpolation, clustering, joins, density, geometry, accessibility, landscape ecology | [docs/api/gpu-spatial-analysis](../../docs/api/gpu-spatial-analysis/README.md) |
| `@luma-contributors/analysis/gpu-spatial-analysis/ontology` | Operator catalog and analysis ontology | [analysis catalog](../../docs/api/analysis-catalog.md) |
| `@luma-contributors/analysis/gpu-network` | Network statistics, structure, routing, centrality and movement | [docs/api/gpu-network](../../docs/api/gpu-network/README.md) |
| `@luma-contributors/analysis/gpu-terrain` | Terrain derivatives, hydrology and relief | [docs/api/gpu-terrain](../../docs/api/gpu-terrain/README.md) |
| `@luma-contributors/analysis/raster-analysis` | Raster algebra, cost distance, zonal statistics, patches, remote sensing | [docs/api/gpu-raster](../../docs/api/gpu-raster/README.md) |
| `@luma-contributors/analysis/dataframe-analysis` | Classification, profiles, pair statistics, temporal reduction, joins, standardization | [dataframe analysis](../../docs/api/gpu-dataframe-analysis.md) |
| `@luma-contributors/analysis/pointcloud-analysis` | Ground filtering, ICP, features and change detection | [point cloud analysis](../../docs/api/pointcloud-analysis.md) |
| `@luma-contributors/analysis/geospatial-analysis` | Geospatial helpers, including the datum grid transform | [geospatial](../../docs/api/geospatial.md) |
| `@luma-contributors/analysis/computation-record` | Recording and describing command graphs for computation lenses | [computation record](../../docs/api/computation-record.md) |
| `@luma-contributors/analysis/analysis-core` | Shared contracts, kernel helpers and WGSL utilities | |

What is implemented, function by function, is in [dev-docs/coverage-matrix.md](../../dev-docs/coverage-matrix.md).
Open work is in [dev-docs/roadmap.md](../../dev-docs/roadmap.md).
