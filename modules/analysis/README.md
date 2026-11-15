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

## MGWR bandwidth selection and limits

`GPUMultiscaleGeographicallyWeightedRegression` and its GPU/CPU operation forms accept
`bandwidthSelection: {criterion: 'AICc', searchMethod: 'golden-section', maximumSearchIterations: 30}`.
`bandwidthCandidates` then holds two positive finite distance bounds per component, in
`[interceptLower, interceptUpper, predictor0Lower, predictor0Upper, ...]` order, with
`bandwidthCandidateCount: 2`. Equal bounds select a fixed bandwidth; reversed or invalid bounds
invalidate the fit. The existing finite-grid mode remains the default, as does the
`'leave-one-out'` criterion. Candidate/bound values can change per frame.

Each backfit component selects its bandwidth from the scalar regression on its current partial
response, following [PySAL mgwr multi_bw](https://github.com/pysal/mgwr/blob/master/mgwr/search.py).
For AICc, include the focal observation and compute
`n log(max(RSS / n, 1e-30)) + n log(2π) + n (n + tr(H_j)) / (n - tr(H_j) - 2)`.
`H_j` is that scalar partial-response smoother, and `n` counts complete, unmasked rows.
The GPU accumulates power-of-two-scaled residual squares and recovers `RSS / n` using that
included count, so an overflowing intermediate RSS does not reject a representable variance.
A nonpositive denominator or any failed local fit rejects the candidate. Grid ties select the
first candidate; golden-section ties keep the lower interior point. Golden-section shares GWR's
`delta = 0.38197` and score tolerance `max(1e-6, 4e-6 max(abs(scores)))`, with at most 64 search
iterations. MGWR also evaluates the upper endpoint and keeps it if it is better than the
interior winner. A feasible endpoint keeps an initially infeasible interior bracket searchable.
If the search budget is exhausted, it returns the better of that endpoint and the last interior
winner. The
contributor retains OLS initialization (zero with full inference), its relative-contribution
backfit stopping rule, and unrounded distance bandwidths; it does not reproduce PySAL's initial
single-bandwidth GWR fit, SOC stopping rule, or two-decimal bandwidth rounding.

For the **full MGWR model**, the trace is `tr(S) = sum_j tr(R_j)`, where `R_j` is the coupled
component response operator propagated through the backfit, conditional on the selected bandwidth
history (Li and Fotheringham, 2020, [*Computational improvements to multi-scale geographically
weighted regression*](https://doi.org/10.1080/13658816.2020.1720692)). This is the trace in
`inference.output.summary[0]`, and each `tr(R_j)` is in `effectiveParameters`. Summing independent
scalar `H_j` traces would give a different full-model effective parameter count. Standard errors
remain conditional on bandwidth history; bandwidth confidence intervals and full selection
uncertainty are not provided.

Planar bisquare and explicitly truncated Gaussian fits automatically use the shared GWR grid at
1024 rows: square dimensions `ceil(sqrt(rows / 8))`, capped at 1024 per axis. `indexGridSize`
overrides the dimensions, and `false` forces the dense path. Untruncated Gaussian and great-circle
fits retain the dense fallback. Limits are 65,536 dense rows, 1,048,576 indexed rows, 24 predictors,
128 backfit sweeps, 32 finite-grid candidates and the shared 32,768 command-node budget. The
selection score buffer needs `rows * candidateWidth * 16` bytes in one storage binding
(`candidateWidth = 3` for golden-section: two interior points and the upper endpoint), checked against the device limit. Full inference retains
its 128-row and cubic-work limits. The CPU twin uses float64 internally and retains its documented
uncapped backfit behavior; near-tied scores can produce different bandwidths within float32 error.

The independent [NumPy fixture generator](test/gpu-spatial-analysis/spatial-regression/multiscale-bandwidth-selection-fixtures.py)
records the initialization and stopping differences and checks scalar selection plus coupled traces.
Chunked transpose output layouts, capacity status and limits are documented in
[GPUChunkedSpatialWeightsTranspose](docs/chunked-spatial-weights-transpose.md).
