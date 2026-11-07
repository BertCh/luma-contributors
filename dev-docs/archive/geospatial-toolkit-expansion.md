> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/geospatial-toolkit-expansion.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Delivered; remaining gates moved to the roadmap.

# Geospatial toolkit expansion candidates

The next expansion should complete useful analytical workflows using the contributors already
available. Recipes, adapters and additional outputs can deliver much of that value before another
large algorithm family is needed. Start with zonal distributions, interpolation diagnostics and
network disruption scenarios, alongside contained additions such as local spatial heteroscedasticity,
raster nodata filling, downstream flow distance and edge betweenness.

This maintainer proposal covers spatial analysis, dataframe statistics, raster, terrain, network,
cell and trajectory capabilities. The baseline is the working tree on October 15, 2026, including
reference generators under development. The candidate analysis below records the initial proposal. The subsequent implementation maps every
proposal to a public API or recipe under an explicit bounded contract; see the implementation map.
Complexity estimates express relative engineering scope. GPU suitability and speed advantages still
need representative workload measurements.

Related plans:

- [October 23 research and next coverage backlog](./geospatial-coverage-backlog.md): 78 additional
  scoped work items, expanded inspiration libraries, current-tree reconciliation and delivery gates.
- [Function and option coverage matrix](./geospatial-function-coverage.md): implementation evidence,
  partial support, planned modes and deliberate exclusions across the reviewed inventory.
- [Implementation follow up and remaining work](./geospatial-toolkit-expansion-follow-up.md)
- [Spatial analysis architecture and ontology](./gpu-spatial-analysis-roadmap.md)
- [Raster architecture and scope](./gpu-raster-roadmap.md)
- [Contributor algorithms](./gpu-analysis-algorithms-roadmap.md)
- [Contributor quality](./analysis-contributors-quality-roadmap.md)

## Implementation map

All C01–C30 and R01–R12 have implementation paths in this working tree. This means the explicit
contracts below are implemented, not that every related library option, statistical estimator or
large-domain performance goal is supported. Algorithms with small-domain bounds, heuristic
objectives or limited inference state those limits in their API documentation.

| Proposals | Implementation | API contract |
| --- | --- | --- |
| C01–C02 | `GPULocalSpatialHeteroscedasticity`, `GPUMultivariateLocalGeary` | [Statistics](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| C03 | `addZonalDistributionRecipe`, `addRasterZonalDistributionRecipe` | [Recipes](../../docs/api-reference/experimental/gpu-spatial-analysis/expansion-recipes.md) |
| C04–C08 | Fractional coverage and weighted summaries, nodata fill, rank filter, downstream distance, connected reaches | [Raster/terrain](../../docs/api-reference/experimental/gpu-raster/expansion.md) |
| C09 | `GPUNetworkCentrality` edge betweenness output | [Network](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C10 | `GPUConcaveHull`, `GPUGroupConcaveHull` with circumradius alpha semantics | [Geometry](../../docs/api-reference/experimental/gpu-raster/expansion.md) |
| C11–C13 | `GPUMarkedPointStatistics`: cross K/L, supplied-intensity K, annular pair/mark correlation | [Statistics](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| C14–C15 | `GPUCrossVariogram`, `GPUSpaceTimeVariogram` | [Statistics](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| C16–C17 | `GPUNetworkStructure`: strong components, bridges, articulation points, biconnected blocks | [Network](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C18 | `GPUAlternativeRoutes`: bounded simple-path enumeration | [Network](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C19 | `GPUH3CellIdentities`: directed edges and canonical vertices | [Network/cells](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C20 | `GPUTrackSimilarity` LCSS output | [Tracks](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C21–C26 | Walking costs, reconstruction/hole filling, raster reprojection, constrained Delaunay, image watershed, skeleton and medial disks | [Raster/geometry](../../docs/api-reference/experimental/gpu-raster/expansion.md) |
| C27 | `GPUMultiscaleGeographicallyWeightedRegression`: bounded backfitting, candidate-grid CV, conditional submodel errors | [Regression](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| C28 | `GPUGeneralizedGeographicallyWeightedRegression`: separate Poisson/logistic families | [Regression](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| C29 | `GPUBoundedFacilityAllocation`: bounded exact p-center and capacity-feasible greedy allocation | [Allocation](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| C30 | `GPUSpatialPanelRegression` pooled lag 2SLS; `GPUSpatialRegimeRegression` fully varying regime OLS | [Regression](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |

The [recipe reference](../../docs/api-reference/experimental/gpu-spatial-analysis/expansion-recipes.md)
covers all twelve workflows. R07 extends `addClusterAndOutlineRecipe` with an explicit alpha choice;
R02, R04 and R09 are exported by `gpu-network`, while the remaining recipe functions are exported by
`gpu-spatial-analysis`. The discovery ontology registers the new spatial exports without importing
constructors and marks their evidence partial until representative scale/performance audits exist.

Reference generators are included alongside their JSON fixtures. Direct package comparisons and
compositions of trusted lower-level operations are identified separately in the API references.
New integration tests exercise shader execution, empty/missing inputs, bounded failure, topology,
source identity, and cross-stage ownership. The computation packs and executable documentation
now include the new contributors: [spatial lenses](../../docs/api-reference/experimental/gpu-spatial-analysis/expansion-lenses.md), [raster/terrain lenses](../../docs/api-reference/experimental/gpu-raster/expansion-lenses.md), and the network/dataframe expansion cards. These examples build and capture real contributor graphs on explicitly illustrative inputs.

### Implementation verification

- Final focused shader execution: 66 GPU tests passed across seven files: all 49 expansion tests plus 17 existing triangulation regressions after the shared implementation was moved below raster/spatial APIs.
- New executable lens construction: 41 tests passed (20 spatial, 13 raster/terrain, 8 network/dataframe). These construct actual graphs on a null device; they do not substitute for GPU execution.
- Final consolidated integration checks: 198 node tests passed across 11 files, covering maximal kernel cases, layering, canonical vocabulary, explicit index imports, exact computation-pack inventories, centrality compatibility, documentation slots, public examples and table-of-contents coverage. The new kernel area includes 93 variants; no contributors were skipped.
- `nvm use` and `yarn install` completed. Final `yarn lint fix` passed.
- Full node attempt: 11,836 passed, 345 skipped and 10 failed across 1,113 files (690.85 seconds). Expansion integration failures from that attempt were fixed and rechecked by the final 198-test run. The maximum-rounds centrality audit also passed independently after the resource-contention timeout. An independently added terrain parity diagnostic still disagreed with its fixture expectation.
- Full headless attempt: 5,918 passed and 120 failed before the browser connection closed (1,130.37 seconds; suite incomplete). Existing join parity checks account for 64 failures due to global/local frame bookkeeping; terrain visibility requires equality to two disagreeing references (42 cases); geomorphon comparisons account for 12 failures and physical-material rendering for two. Three additional suites encountered a stale cached helper export during live source updates; all 34 tests in those suites passed on a fresh browser afterward. Together with the final 66-test expansion/triangulation run, 100 focused GPU tests passed on the frozen source. A clean full-suite result is not claimed.
- Final `yarn build` passed after all runtime and formatting changes, with no TypeScript or bundler errors. All five geospatial package checks passed ESM, CJS and declaration imports.
- Both post-consolidation website commands passed: `yarn website:build` and `(cd website && yarn build)`. Each generated the production site and validated `llms.txt` plus 1,020 raw documentation pages.

## Reference libraries and current coverage

The [reference generators](../../modules/graph/test/oracles) already use the following
families. These are test references, not proposed runtime dependencies. A library appearing in a
generator does not establish full parity coverage for every GPU operation in its domain.

| Reference family | Existing uses and adjacent opportunities |
| --- | --- |
| Shapely and GEOS, GeoPandas | Geometry topology, construction, measurement, overlay and spatial joins |
| PySAL esda, libpysal, spreg, mgwr, spopt, giddy, segregation, access and tobler | Spatial statistics, weights, regression, regionalization, allocation, accessibility and change of support |
| NumPy, SciPy, scikit-learn, scikit-gstat and statsmodels | Reductions, distributions, clustering, interpolation, numerical solvers and inference |
| Rasterio and GDAL, scikit-image | Raster sampling, morphology, connected regions and segmentation |
| GRASS, pysheds, xdem, RVT and pvlib | Terrain, drainage, visibility, relief and solar analysis |
| NetworkX and spaghetti | Graph structure, routing, centrality and network point patterns |
| H3, pyproj and PyGeodesy | Cell topology, coordinate transformations and geodesic measurements |
| spatstat, tslearn, dtaidistance and similaritymeasures | Point patterns and trajectory similarity |

The following capabilities already exist and should be reused rather than entered as new projects:

- Directional variograms and robust semivariance.
- Ordinary kriging with prediction variance and leave-one-out predictions and variances.
- Exact raster distance transforms and nearest-seed allocation.
- Polygon coverage validity, simplification and dissolve; polygon repair and overlay.
- Minimum bounding circles, rotated rectangles and polygon label points. Label-point search is
  bounded and is not a certified globally optimal maximum-inscribed-circle solver.
- Node betweenness, closeness, harmonic closeness, PageRank and weak connected components.
- P-median, maximum coverage and set covering, with the documented heuristic contracts.
- Spatial lag and error maximum-likelihood regression.
- Discrete Frechet, Hausdorff and dynamic time warping distances.
- Hot-spot, rate-cluster, period-comparison, change-of-support, cluster-outline, fleet-dwell,
  choropleth, regression and catchment recipes.

Implementation anchors are the [spatial exports](../../modules/graph/src/gpu-spatial-analysis/index.ts),
[raster exports](../../modules/graph/src/gpu-raster/index.ts),
[network exports](../../modules/graph/src/gpu-network/index.ts),
[spatial recipes](../../modules/graph/src/gpu-spatial-analysis/recipes/index.ts) and
[network recipes](../../modules/graph/src/gpu-network/recipes/index.ts).
Use implementations and public exports to resolve stale roadmap entries. The ontology's advanced
models opportunity now describes remaining fixed/dynamic panel, SUR, probit and inference work;
the implemented maximum-likelihood and bounded expansion estimators have separate entries.

## Selection and implementation layers

Prioritize a capability when it answers a recurring user question, composes with existing typed
outputs, has an independent reference, and can amortize GPU setup over substantial or repeated work.
Reference-library completeness alone is not a reason to port a function.

| Layer | Use when | Implementation rule |
| --- | --- | --- |
| Recipe | The operation is a recognizable sequence of existing analyses | Use `add<Name>Recipe`; expose intermediate overrides, source identities and stage status |
| Adapter | The main missing step converts compatible representations | Make conversion and allocation explicit; preserve masks, provenance and chunk boundaries |
| Contributor extension | The same traversal or solver can produce another coherent result | Add an optional output or mode without running its extra work when unused |
| New contributor | A distinct algorithm or reusable primitive is required | Give it bounded work, caller-owned outputs and explicit completion evidence |
| Discovery or documentation | The capability exists but is difficult to find or compose | Improve the catalogue and examples without adding a duplicate export |

## First contributor candidates

These ten capabilities have the strongest immediate case. Small means mostly composition or an
output extension; medium means a bounded algorithm or substantial integration; large means topology,
solver or tiling work. The order is a planning preference, not a benchmark result.

| ID | Capability | User value and existing gap | Layer and complexity | Reference |
| --- | --- | --- | --- | --- |
| C01 | Local spatial heteroscedasticity, LOSH | Find neighborhoods with unusually high or low dispersion, complementing mean-based hot spots | New statistic; medium. Reuse weights, lag and reductions; specify inference separately | [esda LOSH][losh] |
| C02 | Multivariate local Geary | Detect local similarity across several demographic or environmental variables; current local Geary takes one value column | Extension; medium. Define scaling, missing rows and joint permutation semantics | [esda multivariate Geary][geary] |
| C03 | Zonal medians and quantiles | Robust elevation, pollution and population summaries; zonal APIs omit order statistics | Recipe and adapters first; small to medium. Reuse grouped sorting | [NumPy quantiles][quantile] |
| C04 | Fractional-cell zonal statistics | Account for partial boundary cells, particularly for small zones and coarse rasters | Coverage adapter plus reductions; large. Emit zone, cell and coverage tuples, including overlapping zones | [Shapely intersection][intersection] plus an independent aggregation generator |
| C05 | Raster nodata filling | Repair bounded gaps in continuous rasters while preserving observed cells | New raster operation; medium. Return filled mask and unresolved count | [Rasterio fillnodata][fill] |
| C06 | Raster median and percentile filters | Remove isolated noise and estimate local background without materializing spatial-weight adjacency | Raster extension; medium. Reuse tile halos; specify footprint, edge and missing-value policies | [SciPy median][median] and [percentile][percentile] |
| C07 | Downstream flow distance | Measure distance to streams or outlets; accept explicit transition costs for weighted distance | Hydrology extension; medium. Start with D8, then define branching-flow semantics | [pysheds][pysheds] and [GRASS stream distance][stream-distance] |
| C08 | Stream-network extraction | Turn drainage rasters into connected reaches with lengths, junctions and stream order | Terrain-to-network adapter; medium to large. Reuse flow, stream masks, scans and compaction | [pysheds river extraction][pysheds] |
| C09 | Edge betweenness | Identify important road segments or river reaches; current centrality exposes node values | Extension; medium. Reuse Brandes accumulation and retain stable edge IDs | [NetworkX centrality][centrality] |
| C10 | Concave hulls | Outline clusters and observed territories without convex hulls enclosing large empty regions | New geometry contributor; medium to large. Reuse Delaunay; select ratio-based hull or alpha-shape semantics explicitly | [Shapely concave hull][concave] |

Concrete API gaps are documented in
[raster zonal statistics](../../modules/graph/src/gpu-raster/raster-zonal-statistics/gpu-raster-zonal-statistics.ts),
[vector zonal statistics](../../modules/graph/src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics.ts)
and [network centrality outputs](../../modules/graph/src/gpu-network/network-centrality/gpu-network-centrality.ts).
The [group statistics contributor](../../modules/graph/src/gpu-dataframe/group-statistics/gpu-group-statistics.ts)
already provides medians, percentiles, modes and unique counts; C03 primarily exposes those results
through spatial workflows.

## Further contributor candidates

| ID | Capability | User value and proposed implementation | Complexity and reference |
| --- | --- | --- | --- |
| C11 | Cross-type K and L functions | Test attraction or avoidance between event categories; extend existing pair traversal with type-pair normalization | Medium; spatstat Kcross and Lcross in its [catalogue][spatstat] |
| C12 | Inhomogeneous K and pair correlation | Separate interaction from changing background intensity; accept caller-supplied intensity first, with explicit low-intensity policy | Medium to large; spatstat Kinhom in its [catalogue][spatstat] |
| C13 | Mark correlation | Measure association between nearby event magnitudes; reuse pair bins with mark-product accumulators | Medium; spatstat markcorr in its [catalogue][spatstat] |
| C14 | Cross-variograms | Measure spatial co-variation of two fields; add pair cross-products before attempting co-kriging | Medium; [scikit-gstat Variogram][variogram] |
| C15 | Space-time variograms | Measure dependence jointly over spatial distance and time lag; bound both neighborhoods and bin capacities | Medium to large; [scikit-gstat SpaceTimeVariogram][space-time-variogram] |
| C16 | Strongly connected components | Find mutually reachable regions in directed road networks; current graph components are weak components | Large; generic graph primitive validated with [NetworkX components][components] |
| C17 | Bridges, articulation points and biconnected components | Identify single points of failure and resilient blocks; start with undirected graphs and define parallel-edge behavior | Large; [NetworkX bridges][bridges] and [components][components] |
| C18 | K-shortest alternative routes | Explore detours and route choices; bound route count and output capacity, and define simple paths and ties | Large; [NetworkX shortest simple paths][alternative-routes] |
| C19 | H3 directed edges and canonical vertices | Preserve cell connection and shared-vertex identities for flows and meshes; test pentagons and face boundaries | Medium; H3 [edge][h3-edges] and [vertex][h3-vertices] APIs |
| C20 | Longest-common-subsequence track similarity | Match partial tracks while permitting unmatched observations; extend the existing dynamic-programming machinery | Medium; [tslearn LCSS][lcss] |
| C21 | Direction-dependent raster movement cost | Model uphill and downhill movement differently; compute directed costs from elevation and friction | Medium to large; [GRASS r.walk][walk]. Its walking-energy model is distinct from Tobler's hiking function |
| C22 | Binary hole filling and morphological reconstruction | Fill enclosed mask holes or grow regions under a mask; retain topology semantics distinct from closing | Medium; SciPy [hole filling][holes] and scikit-image [reconstruction][morphology] |

The [Ripley contributor](../../modules/graph/src/gpu-dataframe/pair-statistics/gpu-ripley.ts)
explicitly excludes inhomogeneous K. The
[graph component contributor](../../modules/graph/src/gpu-graph-algorithms/gpu-graph-connected-components.ts)
computes weak components, and
[raster cost distance](../../modules/graph/src/gpu-raster/cost-distance/gpu-cost-distance.ts)
explicitly excludes slope-dependent costs. These distinctions should remain visible in discovery.

## Larger capabilities requiring an application

| ID | Capability | Value | Prerequisite and reference |
| --- | --- | --- | --- |
| C23 | Raster reprojection and grid alignment | Align rasters for comparison and combined analysis through explicit destination-grid sampling | Large projection/raster bridge. Preserve the raster roadmap's scope boundary; define source-tile selection, CRS validity, nodata and resampling. [Rasterio reprojection][reprojection] |
| C24 | Constrained Delaunay triangulation | Produce analytical meshes respecting polygon boundaries and Delaunay properties | Large topology project. Existing polygon triangulation already serves rendering; require a mesh-quality consumer. [Shapely constrained Delaunay][constrained] |
| C25 | Marker-controlled raster watershed segmentation | Separate adjacent objects in imagery and density surfaces | Large raster algorithm, distinct from drainage watersheds. Define plateau ties, markers and tile seams. [scikit-image watershed][segmentation] |
| C26 | Skeletonization and medial-axis rasters | Derive centerlines and width information from classified corridors | Large topology project. Treat thinning and medial axes separately; vector cleanup is additional work. [scikit-image morphology][morphology] |
| C27 | Multiscale geographically weighted regression | Give predictors separate spatial bandwidths | Large solver extension. Require bounded backfitting, bandwidth searches and inference evidence. [mgwr][mgwr] |
| C28 | Poisson and logistic geographically weighted regression | Fit local count and binary-outcome models | Large solver extension. Add families individually with exposure, separation and convergence contracts. [mgwr GWR implementation][gwr] |
| C29 | Capacitated facility allocation and p-center | Respect service capacities or minimize worst-case distance | Separate optimization contracts. Check feasibility and compare heuristic objectives with small exact solutions. [spopt location models][spopt] |
| C30 | Spatial panel and regime regression | Model repeated observations or region-specific relationships | Large model family. Select one estimator and data layout first. Existing lag/error maximum likelihood is not part of this gap. [spreg][spreg] |

## Recipe and adapter candidates

Recipe names below are provisional design names. A recipe may require small masking, gathering or
summary kernels even when its analytical contributors already exist. Keep those kernels reusable
when another consumer needs them. Do not duplicate an existing recipe merely to change its label.

### R01 Zonal distribution summaries

Inputs are polygon zones and point observations, or a raster value band with zone IDs. Compose
`GPUPointInPolygonJoin` and `GPUPairGather` for point data, or band calibration and masking for raster
data, followed by `GPUGroupStatistics`. Return zone IDs, valid counts, medians, requested quantiles,
modes and unique counts. Preserve empty zones through an explicit alignment step.

This is the first implementation of C03. Start with unweighted observations and whole raster cells.
Fractional or area-weighted quantiles need an explicit weighted empirical-distribution contract and
are not implied by the existing unweighted group quantiles. Define whether overlapping polygons
receive the same observation and how boundary points are assigned. Validate membership with Shapely
and summaries with NumPy/pandas, including empty zones, missing values, even-sized groups and ties.

### R02 Network disruption comparison

Inputs are a baseline directed topology, origin/facility IDs, opportunities and an edge closure mask
or replacement costs. Compose `GPUNetworkSubgraphFilter`, the necessary edge-cost gather or mask
adapter, `GPUNetworkCostMatrix` or `GPUNetworkReachability`, and `GPUNetworkAccessibility`. Retain
the baseline outputs and rerun only the scenario branch when its costs or closures change.

Return baseline/scenario accessibility, finite cost differences, newly unreachable demand and
stable original node/edge IDs. Do not subtract infinities or interpret a truncated search as an
unreachable destination. A filtered adjacency alone does not carry the routing weights: preserve
the original CSR slots with an impassable-cost adapter, or gather weights using compacted provenance.
Validate with NetworkX shortest paths and independently computed opportunity sums. Test one-way
roads, parallel edges, a bridge closure, isolated origins and an unchanged scenario.

### R03 Interpolation diagnostics

Inputs are sample coordinates and values, an output grid and a caller-supplied variogram model.
Compose `GPUKriging` with its existing leave-one-out outputs, residual arithmetic, grouped
reductions and optional spatial weights plus residual Moran analysis. Return predictions, variances,
residuals, bias, mean absolute error, root mean squared error and valid validation counts.

Standardized residuals are undefined when validation variance is zero or unusable; preserve a mask
and count instead of producing infinite scores. Keep model fitting an explicit prior step: this
recipe must not hide CPU fitting or readback. Compare ordinary-kriging predictions with scikit-gstat
under matched model/range/neighborhood conventions and summary arithmetic with NumPy. Validate
duplicate samples, held-out rows, singular neighborhoods and insufficient-neighbor cases separately.

### R04 Facility scenario evaluation

Inputs are demand, a candidate facility set, a reusable cost matrix and scenario facility masks.
Compose `GPUNetworkAccessibility`, `GPUCatchmentAccessibility`, and grouped summaries; optionally
invoke `GPULocationAllocation` when the user asks to select facilities rather than evaluate them.
Return covered demand, uncovered demand, travel-cost distributions and accessibility differences.

Reuse the existing drive-time and straight-line catchment recipes where their inputs fit. The new
piece is scenario comparison with common demand and stable facility identities. Define unreachable
demand, decay, supply and denominator policies. Validate with `access`, NetworkX and NumPy; use
spopt only for an explicitly selected optimization objective. This workflow does not claim the
capacity constraints proposed in C29.

### R05 Exposure by administrative area

Inputs are hazard or environmental values, population or asset weights, and polygon zones. Compose
raster sampling or raster joins, threshold/band arithmetic, zone assignment and grouped reductions.
Return exposed totals, denominator totals, shares and valid-area or valid-population coverage.

Start with aligned grids and whole-cell membership. Add C04 for fractional boundary coverage; require
C23 or explicit upstream alignment when grids differ. Distinguish counts per cell, densities per
area and intensive values: a population count needs a coverage fraction, while a density also needs
cell area. Missing hazard observations remain unknown exposure, not safe area. Validate sampling
with Rasterio, intersections with Shapely and weighted totals with NumPy.

### R06 Regional raster change summaries

Inputs are aligned raster slices, zone IDs and explicit comparison periods. Compose
`GPUChangeDetection`, `GPURasterZonalStatistics` and, where needed, R01. Return per-zone change
summaries, changed area, valid comparison counts and coverage fractions.

Keep this distinct from `addPeriodComparisonRecipe`, which already compares cell tables. The raster
recipe supplies band validity and area accounting. Compare only a defined common support or report
the changed observation footprint separately. Per-pixel significance does not establish a
statistically significant regional mean; an inferential regional output needs its own model.
Validate with Rasterio/NumPy and the relevant SciPy reference for the selected pixel statistic.

### R07 Cluster footprint refinement

Extend `addClusterAndOutlineRecipe` with an explicit outline choice once C10 exists. Preserve the
existing DBSCAN, group summaries and convex-hull path. Return source membership, geometry, measures
and topology status for the selected outline.

The concave mode must expose its ratio or alpha definition, hole policy and degenerate-cluster
behavior. Compare clustering partitions against scikit-learn and footprint geometry against the
chosen Shapely operation. Avoid requiring identical cluster labels or triangle order when equivalent
partitions or topology are the actual contract.

### R08 Drainage network analysis

Inputs are a DEM, its grid metadata, a stream threshold and optional pour points. Compose
`GPUTerrainFlow`, `GPUTerrainStreamOrder`, `GPUTerrainWatersheds`, C07 and C08, then optional network
and grouped statistics. Return raster catchments, connected stream reaches, stream order,
downstream distances and per-catchment summaries.

Begin with one complete D8 domain and consistent nodata/boundary rules. Preserve drainage and
extraction non-convergence throughout the recipe. Branching D-infinity/MFD networks and cross-tile
drainage require separate contracts. Compare raster flow with pysheds/GRASS and reach connectivity
and lengths with independent graph/geometry checks. A thresholded stream mask alone is not a
validated vector network.

### R09 Track quality and route conformity

Inputs are time-ordered tracks, optional reference routes and explicit quality thresholds. Compose
`GPUTrajectoryMetrics`, resampling, `GPUTrackSimilarity`, optional `GPUMapMatching`, and grouped
track summaries. Return speed/time flags, route distances, matching completeness and per-track
quality summaries while retaining original observation IDs.

Use existing Hausdorff, Frechet and DTW first; C20 adds partial matching later. Keep raw-track flags
separate from values derived after resampling. Map-matching route overflow must not become a
plausible off-route score. Validate metrics and matching stages separately, then test propagation
through duplicate timestamps, gaps, reversals and incomplete routes.

### R10 Local variability and hot-spot comparison

Inputs are values, one explicit weight graph and common masks. Compose existing local Moran or
hot-spot analysis with C01, optional permutation inference and multiple-testing adjustment.
Return separate mean-pattern and dispersion statistics, their supported inference outputs and
classification columns. Keep the same neighborhood definition visible in both outputs.

This enables a map of places with similar means but different local variability. Do not invent a
combined significance score. Validate each statistic against esda; specify whether correction is
within each statistic family or across the entire collection of tests. Stage failure leaves the
corresponding classification unknown.

### R11 Cell flow topology

Inputs are cell keys and origin/destination flow rows. Compose existing point-to-cell, aggregation
and flow aggregation where applicable, with C19 for canonical neighboring-cell edge identities.
Return cell totals, neighboring-cell edge flows and shared-boundary geometry.

An H3 directed edge exists only between neighboring cells. Retain arbitrary origin/destination
flows as a separate pair table; do not force them into edge keys or silently route them through
intermediate cells. Compare keys and boundaries with H3 and flow conservation with independent
grouped sums. Test pentagons, repeated pairs and flows crossing the antimeridian.

### R12 Surface preparation with provenance

Inputs are a continuous raster, validity and explicit repair settings. Compose C05, optional C06,
and existing terrain derivative or raster statistics contributors. Return the prepared surface,
original validity, filled/filtered masks, unresolved gaps and downstream validity.

Filling a missing sample and hydrologically filling a depression are different operations and must
remain separately selected. Do not fill categorical rasters using continuous interpolation. Test
preservation of observed values in fill-only mode, maximum gap distance, nodata borders and tiled
equivalence against the complete-domain reference. This recipe can ship with C05 before C06.

## Common recipe contracts

- Recipes construct nodes in the existing command graph. They do not introduce another scheduler,
  hidden submissions, implicit CPU fallbacks or hidden readback.
- Expose requested outputs and useful intermediate overrides. Retain baseline results only while
  their input versions remain valid; changes to topology, costs, masks or CRS invalidate affected
  prepared results explicitly.
- Return source IDs and resolved spatial/grid context. Never silently change units, coordinate
  model, row order, chunk boundaries or ownership.
- Preserve per-stage invalid, uncertain, overflow and convergence evidence. A combined status may
  summarize completeness but cannot erase the cause or make partial results appear complete.
- Keep per-frame thresholds, masks and costs in parameter/data views. Recompile only when the
  algorithm shape or graph topology changes.
- Use typed GPU ports and existing `GPUData`/`GPUVector` ownership. Arrow conversion belongs in
  `@luma.gl/arrow`; generic graph primitives belong below the geospatial recipes.
- Put workflows with network dependencies under the network recipe entry where appropriate. If a
  cross-domain composition would create an import cycle, introduce an explicit composition entry
  above both domains or inject prepared inputs rather than importing recipes back into primitives.
- Add discovery entries and examples only after the public capability has an in-tree consumer and
  its behavior is validated. Proposal names in this document do not imply runtime availability.

## Suggested delivery order

| Stage | Scope | Completion evidence |
| --- | --- | --- |
| 1 | R01 zonal distributions, R02 disruption comparison and R03 interpolation diagnostics | End-to-end fixtures, propagated failure cases and repeated-scenario measurements; no new major analytical solver |
| 2 | C01 LOSH, C05 nodata filling, C07 downstream distance and C09 edge betweenness; R10 and R12 as consumers | Direct reference agreement, numerical contracts and representative browser GPU benchmarks |
| 3 | C02 multivariate Geary, C04 fractional coverage, C06 raster rank filters, C08 stream extraction and C10 concave hulls; R05, R07 and R08 | Boundary/topology invariants, capacity evidence and complete-domain versus tiled checks where supported |
| 4 | R04 facility scenarios and R06 regional change as demand warrants; C11–C22 with matching consumers such as R09 and R11 | Demonstrated workflow demand and benefit over existing compositions |
| 5 | Select individual C23–C30 projects | Specific application, reference semantics, solver/topology prerequisites and measured advantage |

Do not block the existing contributors' quality work on this expansion. Reconcile stale catalogue
claims as part of discovery work, and prefer extensions to existing recipes when they already own
the workflow.

## Validation and performance gates

Every addition needs a committed reference generator with package versions, source data provenance,
coordinate assumptions, input rounding, masks, boundary conventions and tolerances. A direct library
call and a custom composition of library primitives are different kinds of evidence; record which
one produced each result. Avoid treating a reimplementation of the GPU formula as an independent
oracle.

Reference conventions must be selected before comparing results. For example,
[the existing trajectory generator](../../modules/graph/test/oracles/track-similarity-dtw-frechet.py)
distinguishes DTW cost definitions, and
[the variogram generator](../../modules/graph/test/oracles/dataframe-variogram.py)
distinguishes lag-edge inclusion rules. Raster rank filters also need an explicit even-window median
and NaN policy rather than assuming every reference uses the same definition.

Validate each component and its composition. Include empty input, non-finite values, duplicate
coordinates, ties, overlapping support, disconnected graphs and insufficient capacities as relevant.
Geometry needs invariants and canonicalized topology comparisons; optimization needs feasibility
and objective-quality checks; iterative algorithms need forced non-convergence cases. Stochastic
results require matched permutations or distributional evidence rather than merely matching seeds.

For fractional zonal work, separately validate geometric coverage, the meaning of the value band,
area units and conservation under a partition. For statistical recipes, preserve the distinction
between a descriptive summary and an inferential claim. For multi-tile processing, compare the
assembled result with a complete-domain run and ensure halos are not counted twice. Quantiles cannot
be merged by averaging tile quantiles.

Benchmark compile time, upload, execution, optional readback, peak transient memory and repeated
updates. Include small inputs where a CPU library may win, realistic distributions and default
WebGPU device limits. Do not infer speedup from algorithm parallelism alone.

Implementation work must follow AGENTS.md formatting and applicable checks. After final code and
formatting changes, merge preparation requires `yarn build` and `yarn test`; shared/table API changes
also require the focused `yarn test-node` check. Record setup and website-check outcomes when
preparing a merge. This proposal does not substitute for implementation validation.

[losh]: https://pysal.org/esda/stable/generated/esda.LOSH.html
[geary]: https://pysal.org/esda/stable/generated/esda.Geary_Local_MV.html
[quantile]: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
[intersection]: https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html
[fill]: https://rasterio.readthedocs.io/en/stable/api/rasterio.fill.html
[median]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
[percentile]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html
[pysheds]: https://github.com/pysheds/pysheds
[stream-distance]: https://grass.osgeo.org/grass-stable/manuals/addons/r.stream.distance.html
[centrality]: https://networkx.org/documentation/stable/reference/algorithms/centrality.html
[concave]: https://shapely.readthedocs.io/en/stable/reference/shapely.concave_hull.html
[spatstat]: https://spatstat.org/resources/spatstatQuickref.pdf
[variogram]: https://mmaelicke.github.io/scikit-gstat/reference/variogram.html
[space-time-variogram]: https://scikit-gstat.readthedocs.io/en/latest/reference/spacetimevariogram.html
[components]: https://networkx.org/documentation/stable/reference/algorithms/component.html
[bridges]: https://networkx.org/documentation/stable/reference/algorithms/bridges.html
[alternative-routes]: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html
[h3-edges]: https://h3geo.org/docs/api/uniedge/
[h3-vertices]: https://h3geo.org/docs/api/vertex/
[lcss]: https://tslearn.readthedocs.io/en/stable/user_guide/lcss.html
[walk]: https://grass.osgeo.org/grass-stable/manuals/r.walk.html
[holes]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
[morphology]: https://scikit-image.org/docs/stable/api/skimage.morphology.html
[reprojection]: https://rasterio.readthedocs.io/en/stable/topics/reproject.html
[constrained]: https://shapely.readthedocs.io/en/2.1.2/reference/shapely.constrained_delaunay_triangles.html
[segmentation]: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
[mgwr]: https://github.com/pysal/mgwr
[gwr]: https://github.com/pysal/mgwr/blob/main/mgwr/gwr.py
[spopt]: https://pysal.org/spopt/tutorials.html
[spreg]: https://pysal.org/spreg/api.html
