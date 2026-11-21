# @luma-contributors/analysis

GPU geospatial and analysis operations built on published [luma.gl](https://luma.gl).
The GPU infrastructure, including command graphs, GPU Core operations, GPU tables, dataframes
and graph algorithms, comes from luma.gl.

## Run your first computation

Start with the [CPU variogram fit](https://github.com/visgl/luma-contributors/blob/master/docs/api/operation-forms-and-cpu-twins.md#variogram-fitting-without-a-graph):
20 generated bins, one public function and the expected exponential model, with no GPU or dataset.
Continue with [point density from device to readback](https://github.com/visgl/luma-contributors/blob/master/docs/api/operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for a complete browser GPU computation using public APIs, including typed readback and cleanup.

From the workspace root after `nvm use` and `yarn install`, execute the literal CPU
walkthrough and display its fitted model and parameters:

```bash
yarn vitest run --project node modules/analysis/test/analysis-core/first-run-docs.node.spec.ts -t 'CPU variogram walkthrough'
```

Find an operation through the [API task chooser](https://github.com/visgl/luma-contributors/blob/master/docs/api/README.md#find-an-operation).

## Installation contract

Compatibility testing targets luma.gl `10.0.0-alpha.16` and Apache Arrow 17. The required peer
ranges are `~10.0.0-alpha.16` for `@luma.gl/core`, `@luma.gl/shaders` and `@luma.gl/graph`
and `>=17.0.0` for `apache-arrow`. Install Arrow even when importing only construction helpers
or network contributors. See the [alpha package contract](https://github.com/visgl/luma-contributors/blob/master/dev-docs/release-package.md)
for release status and offline verification of the packed package.

The package licence is `MIT AND Apache-2.0 AND BSD-3-Clause`. The shipped
[licence and notice inventory](THIRD-PARTY-NOTICES.md) links the applicable licence texts and
lists the third-party adaptations.

## Entry points

| Import | Contents | Docs |
| --- | --- | --- |
| `@luma-contributors/analysis` | Root (`.`): alias of `computation-record` | [computation record](https://github.com/visgl/luma-contributors/blob/master/docs/api/computation-record.md) |
| `@luma-contributors/analysis/gpu-spatial-analysis` | Spatial weights, autocorrelation, regression, interpolation, clustering, joins, density, geometry, accessibility, landscape ecology | [docs/api/gpu-spatial-analysis](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/README.md) |
| `@luma-contributors/analysis/gpu-spatial-analysis/ontology` | Operator catalog and analysis ontology | [analysis catalog](https://github.com/visgl/luma-contributors/blob/master/docs/api/analysis-catalog.md) |
| `@luma-contributors/analysis/gpu-network` | Network statistics, structure, routing, centrality and movement | [docs/api/gpu-network](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-network/README.md) |
| `@luma-contributors/analysis/gpu-terrain` | Terrain derivatives, hydrology and relief | [docs/api/gpu-terrain](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-terrain/README.md) |
| `@luma-contributors/analysis/raster-analysis` | Raster algebra, cost distance, zonal statistics, patches, remote sensing | [docs/api/gpu-raster](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-raster/README.md) |
| `@luma-contributors/analysis/dataframe-analysis` | Classification, profiles, pair statistics, temporal reduction, joins, standardization | [dataframe analysis](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-dataframe-analysis.md) |
| `@luma-contributors/analysis/pointcloud-analysis` | Ground filtering, ICP, features and change detection | [point cloud analysis](https://github.com/visgl/luma-contributors/blob/master/docs/api/pointcloud-analysis.md) |
| `@luma-contributors/analysis/geospatial-analysis` | Geospatial helpers, including the datum grid transform | [geospatial](https://github.com/visgl/luma-contributors/blob/master/docs/api/geospatial.md) |
| `@luma-contributors/analysis/computation-record` | Recording and describing command graphs for computation lenses | [computation record](https://github.com/visgl/luma-contributors/blob/master/docs/api/computation-record.md) |
| `@luma-contributors/analysis/analysis-core` | Shared contracts, kernel helpers and WGSL utilities | [fitting kernels](https://github.com/visgl/luma-contributors/blob/master/docs/api/analysis-core/fitting-kernels.md) |

What is implemented, function by function, is in [dev-docs/coverage-matrix.md](https://github.com/visgl/luma-contributors/blob/master/dev-docs/coverage-matrix.md).
Open work is in [dev-docs/roadmap.md](https://github.com/visgl/luma-contributors/blob/master/dev-docs/roadmap.md).

## Detailed guides

The canonical guides are published with the API documentation. These absolute links also work
when this README is read from an installed package.

| Topic | Guide |
| --- | --- |
| Chunked spatial weights | [Transpose layouts and capacity status](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/chunked-spatial-weights-transpose.md) |
| Coregionalization | [GPU contributor, graph operation and CPU twin](https://github.com/visgl/luma-contributors/blob/master/docs/api/dataframe-analysis/coregionalization-fit.md) |
| Shared fitting | [Gradient sums, Adam, minibatches and coordinate encoders](https://github.com/visgl/luma-contributors/blob/master/docs/api/analysis-core/fitting-kernels.md) |
| Geodesics | [Precision, statuses and measured envelopes](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/geodesic-precision.md) |
| Polygon profiles | [Contiguity statistics and completeness](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/gpu-analysis-contiguity-statistics.md) |
| Network communities | [Sparse Louvain and CPU operation](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-network/louvain.md) |
| Quantile LISA | [Single-variable and multivariate inference](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/quantile-lisa.md) |
| Instrumental variables | [Robust covariance and solver status](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-spatial-analysis/instrumental-variables-covariance.md) |
| Raster contours | [Isoline tiles and vector overlays](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-raster/isolines-tiles.md) |
| Hydrology | [Height above drainage over D-infinity flow](https://github.com/visgl/luma-contributors/blob/master/docs/api/gpu-terrain/height-above-drainage.md) |

For an independent numerical reference, see the
[coregionalization fixture generator](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-dataframe/pair-statistics/fixtures/generate-coregionalization.py).
