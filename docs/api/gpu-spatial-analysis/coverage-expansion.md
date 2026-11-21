# Geospatial coverage expansion

The coverage expansion adds graph contributors with explicit bounded contracts. Import spatial
operations from `@luma-contributors/analysis/gpu-spatial-analysis`; raster, terrain, network, dataframe,
projection and point-cloud operations retain their own package subpaths.

Choose the task contract directly: [landscape metrics and connectivity](./landscape-ecology.md),
[urban morphology and visibility](./urban-morphology.md),
[home ranges and recursions](./home-ranges-and-recursions.md), or
[composed workflows](./recipes.md). Execution forms and recipe-stage mappings are listed on those pages.

These APIs declare GPU work in the caller's graph. Inputs and outputs are GPU views; callers
control submission and readback. Capacity, convergence, missing-data and unsupported-case results
must be inspected before interpreting numerical outputs. An available constructor does not imply
complete compatibility with a reference library or every mode in the maintainer roadmap.

| Area | APIs and contracts |
| --- | --- |
| Geometry | [Polygon normalization, precision reduction, subdivision, reference-vertex insertion, ratio hulls and bounded maximum-inscribed-circle search](../gpu-geometry-refinement.md). Each operation retains its own geometry and numerical limits. |
| Global coordinates | [S2 hierarchy, neighbors, measures and cap coverings; convex spherical predicates and overlap](./global-coordinates.md). Datum grids use `import {GPUDatumGridTransform} from '@luma-contributors/analysis/geospatial-analysis'`; spherical spatial contributors use `@luma-contributors/analysis/gpu-spatial-analysis`. |
| Prediction | [Local kriging, RBF, natural neighbors and conditional Gaussian simulation](./interpolation-expansion.md). Resident variogram fitting lives in `@luma-contributors/analysis/dataframe-analysis`. |
| Statistical contracts | [Robust IV inference, restricted regimes, panel error likelihood, SUR, probit, local diagnostics, permutation inference, rank transitions, fuzzy weights, held-out regression and spatial interaction](./coverage-statistics.md). Training masks must also govern preprocessing and parameter selection. |
| Point processes | [Inhomogeneous window statistics, simulation-curve envelopes, OPTICS and probability sampling](./point-process.md). Network kernel density lives in `gpu-network`. |
| Urban analysis | [Building tessellation, street profiles, shared walls, isovists and visibility graphs](./urban-morphology.md). |
| Ecology | [Landscape summaries, patch proximity, resistance connectivity and segregation decomposition](./landscape-ecology.md). |
| Raster | [Conservative remapping, texture, temporal processing, translation registration, region graphs, spectral unmixing and seasonal breaks](../gpu-raster/remote-sensing.md). |
| Network and movement | [Routing, flow, communities, facilities, rational-agent access, trajectory smoothing, restriction and similarity search](../gpu-network/network-movement-coverage.md). |
| Point clouds | [Indexed neighborhoods, local features, filtering, SMRF, height normalization, ICP and surface change](/docs/api/pointcloud-analysis). |
| Terrain | [Breaching, branching flow consumers, sink hierarchy, catchment summaries, DEM alignment and tiled routing](../gpu-terrain/terrain-processing.md). |

Numerical tests use independent pinned references where available, together with conservation,
covariance, topology, capacity and repeated-execution invariants. A composed reference is labeled
as such. See the domain pages for tested models and remaining modes; measured large-data throughput
and full upstream-package parity are separate claims.
