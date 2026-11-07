# Function coverage matrix

Status date: 2026-11-07. Ported from the luma.gl branch matrix ([archive/geospatial-function-coverage.md](./archive/geospatial-function-coverage.md), audit of 2026-10-23, KG rows 2026-10-25) and re-pointed at this repository. This is the living inventory of which reference functions the GPU analysis packages cover. Open work and priorities are in [roadmap.md](./roadmap.md). Dropped capabilities and workarounds are in [upstream-workarounds.md](./upstream-workarounds.md).

## How to read this matrix

- **Implemented**: the specific bounded operation in the row has an implementation and contract. It does not mean every option of the named reference API is supported.
- **Partial**: relevant behavior exists but a named mode, guarantee or integration step is missing.
- **Excluded**: intentionally out of scope; the reason is stated.
- An unlisted function is unassessed, not implemented and not excluded.
- Counts are inventory counts, not a percentage of upstream functionality. Inventory: **142 rows**: 134 implemented subsets, 2 partial, 6 excluded, 0 planned. Two rows are Partial in this repository: KG06 (SQL engine is in published luma.gl) and EC03 (sparse resistance lost in the split).
- Status was carried over from the branch. "Implemented" here means the symbol is defined and exported in this repository (or published luma.gl) and has a source contract. It has not been re-verified by running every test.

## Where to import from

The **LIB entry point** column names the directory under `modules/analysis/src` and the symbols. Import from the matching package entry point.

| Directory | Import path |
| --- | --- |
| `analysis-core` | `@luma-contributors/analysis` (root; `graph-support` has no export yet) |
| `computation-record` | `@luma-contributors/analysis/computation-record` |
| `gpu-spatial-analysis` | `@luma-contributors/analysis/gpu-spatial-analysis` |
| `gpu-network` | `@luma-contributors/analysis/gpu-network` |
| `gpu-terrain` | `@luma-contributors/analysis/gpu-terrain` |
| `dataframe-analysis` | `@luma-contributors/analysis/dataframe-analysis` |
| `raster-analysis` | `@luma-contributors/analysis/raster-analysis` |
| `pointcloud-analysis` | `@luma-contributors/analysis/pointcloud-analysis` |
| `geospatial-analysis` | `@luma-contributors/analysis/geospatial-analysis` |

Rows marked "Published" use `@luma.gl/graph@10.0.0-alpha.10` and have no code here. Every other symbol in this table was found by grep in `modules/analysis/src` and is reachable through its directory barrel (`index.ts`) on 2026-11-07. The `B-` rows are baselines that existed before the expansion; the other IDs come from the backlog.

Reference library links are in the archived matrix.

## Geometry and topology

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-GE01 | Implemented | Shapely intersection/union/difference | Bounded planar polygon overlay; this row does not claim arbitrary geometry-collection parity. | `gpu-spatial-analysis`: `GPUPolygonOverlay` |
| B-GE02 | Implemented | Shapely set_precision: pointwise | Coordinate-grid rounding and cleanup; topology-preserving modes are separate below. | `gpu-spatial-analysis`: `GPUGeometryCleanup` |
| B-GE03 | Implemented | CGAL Delaunay triangulation | Planar point triangulation; bounded outputs and validity remain part of the contract. | `gpu-spatial-analysis`: `GPUDelaunayTessellation` |
| B-GE04 | Implemented | CGAL constrained Delaunay triangulation | Existing constrained polygon triangulation path; not natural-neighbor query weights. | `gpu-spatial-analysis`: `GPUConstrainedDelaunayTriangulation` |
| GE01 | Implemented | Shapely normalize | Postorder points, lines, rings, polygons, multis and nested/empty collections; preserved source IDs, GEOS child order, canonical equality and explicit tolerance. | `gpu-spatial-analysis`: `GPUNormalizeGeometry`, `GPUCanonicalGeometryEquality` |
| GE02 | Implemented | PostGIS ST_Subdivide | Hole-aware convex seeds and bounded axis-aligned queue clipping enforce per-piece vertex limits, source provenance and union reconstruction. | `gpu-spatial-analysis`: `GPUSubdividePolygons` |
| GE03 | Implemented | Shapely set_precision: valid_output, keep_collapsed | Arbitrary positive finite float32 grids, collapsed-line policies and snap-rounded polygon reconstruction; empty results remain successful under repeated composition. | `gpu-spatial-analysis`: `GPULinePrecisionReduction`, `GPUPolygonPrecisionReduction` |
| GE04 | Implemented | Shapely snap | Nearest-reference endpoint snapping, stable row-order ties, duplicate-aware ordered insertion and source provenance; explicit collapse/nonfinite/incomplete validity bits. | `gpu-spatial-analysis`: `GPUSegmentSnap` |
| GE05 | Implemented | Shapely offset_curve | Noded raw offsets, analytic distance-band clipping, collapsed-spur cancellation and directed multipart component assembly with original source edges. | `gpu-spatial-analysis`: `GPUOffsetCurveComponents`, `GPULineLoopRemoval` |
| GE06 | Implemented | Shapely concave_hull: ratio/allow_holes | Delaunay border erosion with distinct edge-ratio and hole policies; explicit convergence budget and triangle-ID tie order. | `gpu-spatial-analysis`: `GPURatioConcaveHull` |
| GE07 | Implemented | Shapely maximum_inscribed_circle: tolerance | Bounded best-first search returns center, radius, boundary witness, remaining geometric error and distinct tolerance-success/exhaustion status. | `gpu-spatial-analysis`: `GPUMaximumInscribedCircle` |
| GE08 | Implemented | CGAL straight skeleton and inward offsets | Linked wavefront edge collapse, reflex split, hole merge, multiple holes, collinear corners and simultaneous terminal ridges; skeleton arcs and linked inward-offset loops. | `gpu-spatial-analysis`: `GPUVectorStraightSkeleton` |

## Global coordinates and cells

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-CR01 | Implemented | GeographicLib-style ellipsoidal measurements | Ellipsoidal area/perimeter modes exist with bounded numerical methods; difficult-case accuracy remains CR01. | `gpu-spatial-analysis`: `GPUGeometryMeasures` |
| B-CR02 | Implemented | H3 identity conversion | Existing H3 cell identity contributor; not a claim that every H3 operation is covered. | `gpu-spatial-analysis`: `GPUH3CellIdentities` |
| CR01 | Implemented | GeographicLib inverse geodesics/polygon area | Existing measurement API adds safeguarded inverse solving and adaptive per-edge area integration, with explicit unresolved/exhausted invalidity on serial/cooperative paths. | `gpu-spatial-analysis`: `GPUGeometryMeasures` |
| CR02 | Implemented | PROJ hgridshift | Caller-prepared regular horizontal grids, bilinear priority/nodata/bounds, target-seeded fixed-point inverse and optional high/low coordinate words. | `geospatial-analysis`: `GPUDatumGridTransform` |
| CR03 | Implemented | PROJ vgridshift | Signed vertical/combined grid corrections, explicit height units and source/target references, split height residuals and source-position sampling in combined inverse. | `geospatial-analysis`: `GPUDatumGridTransform` |
| CR04 | Implemented | S2 spherical predicates/Boolean operations | Nonconvex shell/hole containment with open/closed boundaries; all four Boolean operations through chart or intrinsic global arc noding/ring assembly; explicit complements and reference membership. | `gpu-spatial-analysis`: `GPUSphericalPolygonContains`, `GPUSphericalPolygonOverlay`, `GPUSphericalArcOverlay` |
| CR05 | Implemented | S2 covering, neighbors, measures and crosswalks | Exact hierarchy/neighbors and stable measures through level 30; adaptive polygon interior/exterior covers through 15, uniform cap covers through 8, geometric overlap weights for prepared convex cell boundaries. | `gpu-spatial-analysis`: `GPUS2CellHierarchy`, `GPUS2CellMeasures`, `GPUS2PolygonCovering` |

## Raster and remote sensing

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-RS01 | Implemented | exactextract coverage fractions | Planar triangle/pixel clipping, not supersampling; dense candidates capped at 4,194,304 zone/cell pairs. | `raster-analysis`: `GPURasterFractionalCoverage` |
| B-RS02 | Implemented | exactextract sum/mean/coverage summaries | Fractional intensive means and extensive totals with valid/covered-area outputs. | `raster-analysis`: `GPURasterFractionalZonalStatistics` |
| B-RS03 | Implemented | GDAL FillNodata: four-quadrant search | The explicit four-quadrant mode exists; smoothing and all GDAL edge behavior require separate comparison. | `raster-analysis`: `GPURasterFillNoData` |
| B-RS04 | Implemented | Raster reprojection: tile-routed sampling | Existing tile routing and resampling contract; not conservative cell-overlap remapping. | `raster-analysis`: `GPURasterTileReprojection` |
| B-RS05 | Implemented | Welch comparison, Theil-Sen slope, Mann-Kendall trend | Dense stacks up to 256 slices; Sen slope up to 64; slope uses slice index and single-band trend outputs; time rescaling is explicit. | `raster-analysis`: `GPUChangeDetection` |
| RS01 | Implemented | exactextract quantile/variance/stdev | Exact sorted continuous-value quantiles or inverse weighted CDF, independent observation weights/cell areas, variance and Kish support. | `raster-analysis`: `GPURasterFractionalZonalDistribution` |
| RS02 | Implemented | exactextract majority/minority/frac | Fractional class areas/proportions, stable majority/minority, ties and missing support; up to 65,534 dense classes within the group budget. | `raster-analysis`: `GPURasterFractionalCategoricalStatistics` |
| RS03 | Implemented | xESMF conservative/conservative_normed | Exact Cartesian overlap, three normalizations and caller-owned prepared weights reused across graphs; changing QA/conservation tested at 128², 256² and 512² source grids. | `raster-analysis`: `GPURasterConservativeRemapping` |
| RS04 | Implemented | skimage graycomatrix/graycoprops | Masked directional/symmetric GLCMs and seven texture measures; direct scikit-image properties, three-direction halo/core parity and 128² cells at 32 levels/radius 8 verified. | `raster-analysis`: `GPURasterTexture` |
| RS05 | Implemented | terra focalPairs-style covariance/correlation | Radius-bounded paired-valid covariance, correlation and local OLS with masks/calibration, singular statuses and halo parity. | `raster-analysis`: `GPURasterFocalPairs` |
| RS06.composite | Implemented | terra app/tapp temporal composites | Regular-time masked mean/min/max/median composites with original source-date provenance and effective counts; direct terra app comparisons pass. | `raster-analysis`: `GPURasterTemporal` |
| RS06.filter | Implemented | SciPy savgol_filter | Regular-time missing-data QR Savitzky–Golay fits with effective support, two edge policies and rank-failure status; pinned SciPy comparisons pass. | `raster-analysis`: `GPURasterTemporal` |
| RS07 | Implemented | skimage phase_cross_correlation | Periodic phase correlation/local DFT and masked linear paired-moment NCC/bilinear refinement, with ambiguity/failure/support; direct scikit-image and independent SciPy fixtures pass. | `raster-analysis`: `GPURasterRegistration` |
| RS08 | Implemented | skimage rag_mean_color/merge_hierarchical | Fixed/adaptive mean-updating region merges and explicit GPU tile-core assembly with supplied source equivalences; direct scikit-image, four-core seam parity and 128²/64-region tests pass. | `raster-analysis`: `GPURasterRegionGraph`, `addRasterRegionGraphTiles` |
| RS09 | Implemented | OTB RadiometricIndices | Explicit Gao/McFeeters NDWI, NBR, fixed-coefficient EVI and SAVI with supplied calibration are implemented; numerical GPU fixtures pass. | `raster-analysis`: `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, `GPURasterSAVI` |
| RS10 | Implemented | OTB HyperspectralUnmixing | Supplied-endmember QR, NNLS and simplex-constrained least squares with residual/convergence evidence. All 256 original 224-band pixels match published OTB 9.1.1 UCLS abundances; independent NumPy/SciPy references cover constrained solves. | `raster-analysis`: `GPURasterSpectralUnmixing` |
| RS11.harmonic | Implemented | BFAST harmonic seasonal/break analysis | Regular-time harmonic trend/seasonality plus one level jump selected by BIC gain; independent pinned NumPy cases cover noisy positive/negative/no-break series and QA gaps. | `raster-analysis`: `GPURasterSeasonalBreak` |
| RS11.extended | Implemented | BFAST-inspired dummy seasons/iterative decomposition | Dummy indicators, joint QR or bounded cyclic trend/season fitting, fixed-history persistent residual monitoring, and omit/reject/interpolate regular-slot gaps; independent NumPy references and explicit convergence/exhaustion tests pass. | `raster-analysis`: `GPURasterSeasonalBreak` |

## Terrain and hydrology

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-TE01 | Implemented | pysheds flowdir/accumulation | D8, D-infinity and Freeman/Quinn multiple-flow modes exist; branching downstream consumers are separate. | `gpu-terrain`: `GPUTerrainFlow` |
| B-TE02 | Implemented | pysheds fill_depressions/resolve_flats | Existing bounded filling/flat handling with convergence status. | `gpu-terrain`: `GPUTerrainFlow` |
| B-TE03 | Implemented | pysheds compute_hand | Height above drainage exists; not point-cloud height normalization. | `gpu-terrain`: `GPUTerrainHeightAboveDrainage` |
| B-TE04 | Implemented | GRASS viewshed-style terrain visibility | Raster terrain visibility and observer contracts; not obstacle-polygon isovists. | `gpu-terrain`: `GPUTerrainViewshed` |
| TE01 | Implemented | Whitebox depression breaching/stream burning | Bounded minimum summed-cut paths with explicit burning, provenance and post-edit conflict verification; radius-S search windows and atomic proposals execute 64 pits on 65,536 cells. | `gpu-terrain`: `GPUTerrainBreach` |
| TE02 | Implemented | pysheds distance_to_outlet/catchment with branching routing | Materialized D-infinity/Freeman/Quinn fractions feed parallel resident DAG transport, expected/min/max distance and fractional catchments; direct pysheds D8/MFD fixtures and 16,384-cell forward/reverse chains. | `gpu-terrain`: `GPUTerrainFlowFractions`, `GPUTerrainBranchingFlow` |
| TE03 | Implemented | Whitebox/SAGA depression and basin analysis | GPU-sorted sublevel activation builds a deterministic merge tree with compressed disjoint sets, ownership, outlets and linear-per-stage area/volume propagation; SciPy sublevel topology and 16,384-cell reference coverage. | `gpu-terrain`: `GPUTerrainDepressionHierarchy` |
| TE04 | Implemented | SAGA hypsometry/channel profiles | Physical basin area/elevation distributions, drainage density and pointer-jumped stream profiles; GPU grouping scales to 65,536 cells and 256 basins with composed pysheds/NumPy references. | `gpu-terrain`: `GPUTerrainCatchmentSummary` |
| TE05 | Implemented | xDEM coregistration | Stable-ground median vertical shift and NMAD match pinned xDEM; GPU-sorted robust statistics and explicit correlated volume variance include 1024-cell full covariance and extreme-variance tests. | `gpu-terrain`: `GPUTerrainDEMCoregistration` |
| TE06 | Implemented | pysheds whole-domain routing reference | Resident whole-domain fill/flat preprocessing feeds owned-tile D8 or branching mass and reach stitching; poisoned halos are excluded. Direct pysheds and 65,536-cell tiled/whole-domain equality execute. | `gpu-terrain`: `GPUTerrainTiledFlow` |

## Point clouds

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| PC01 | Implemented | PDAL filters.normal/covariancefeatures | Exact bounded 3D sorted-grid neighborhoods, owned query rows with certified halo-support flags, covariance/Jacobi normals and shape features. Actual PDAL planar-normal reference and uniform/skew scaling through one million candidates. | `pointcloud-analysis`: `GPUPointCloudNeighborhoods`, `GPUPointCloudFeatures` |
| PC02.voxel | Implemented | PDAL voxel reduction filters | Separate sampled representatives and indexed centroids; finite scalar/x2/x3/x4 attributes; disjoint-owned voxel merging with global IDs and overflow flags. Radius masks and linear block-moment statistical masks, with actual PDAL separated-outlier classifications. | Published `@luma.gl/graph/gpu-pointcloud`: `GPUPointCloudVoxelDownsample`, `GPUPointCloudRadiusFilter`; `pointcloud-analysis`: `GPUPointCloudStatisticalFilter`, `GPUPointCloudVoxelMerge` |
| PC02.outlier | Implemented | PDAL filters.outlier | Separate sampled representatives and indexed centroids; finite scalar/x2/x3/x4 attributes; disjoint-owned voxel merging with global IDs and overflow flags. Radius masks and linear block-moment statistical masks, with actual PDAL separated-outlier classifications. | Published `@luma.gl/graph/gpu-pointcloud`: `GPUPointCloudVoxelDownsample`, `GPUPointCloudRadiusFilter`; `pointcloud-analysis`: `GPUPointCloudStatisticalFilter`, `GPUPointCloudVoxelMerge` |
| PC03 | Implemented | PDAL filters.hag_dem/hag_nn/hag_delaunay | Indexed classified DTM/DSM/CHM; signed containing-cell, bilinear and bounded IDW HAG with support/provenance. IDW fills missing query support. Actual PDAL nearest-one comparisons at identical ground/query cell-centre XY. | `pointcloud-analysis`: `GPUPointCloudSurfaces`, `GPUPointCloudHeight` |
| PC04 | Implemented | PDAL filters.smrf | Separate progressive morphology and parameter-compatible SMRF; sorted-cell minima, exact ring-bounded eight-neighbor fill and repeated-cross diamond morphology. Four actual PDAL 2.8.4 fixtures agree completely; measured one-million-point SMRF. | `pointcloud-analysis`: `GPUPointCloudGround`, `GPUPointCloudSMRF` |
| PC05 | Implemented | PDAL filters.icp | Initialized point-to-point Horn and rank-aware point-to-plane Gauss–Newton ICP, static indexed correspondence, block moments, Huber/Tukey rejection, masks and convergence history. Actual PDAL point-to-point transform reference; one-million-point indexed benchmark. | `pointcloud-analysis`: `GPUPointCloudICP` |
| PC06 | Implemented | PDAL filters.m3c2 | Aligned oriented-cylinder changes, sample roughness/support, projected registration covariance and linear/quadrature Gaussian intervals. Optional supplied Gaussian spatial covariance integrates exact pairwise mean variance; independent SciPy oracle and 91 supported actual PDAL M3C2 upstream golden rows. | `pointcloud-analysis`: `GPUPointCloudChange` |

## Interpolation and prediction

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-IN01 | Implemented | Local ordinary kriging | GPU prediction/variance and leave-one-out outputs; at most 16 neighbors; singular/uncertain cases are flagged. | `gpu-spatial-analysis`: `GPUKriging` |
| B-IN02 | Implemented | Empirical variogram bins | Pair-statistic bins on GPU; fitting is an explicit CPU step. | `dataframe-analysis`: `GPUVariogram` |
| IN01 | Implemented | Variogram.fit and model comparison | Resident spherical/exponential/Gaussian fits with constrained parameters, all three existing weighting conventions, convergence/residual diagnostics, direct GSTools fitting and measured readback comparison. | `dataframe-analysis`: `GPUVariogramModelFit` |
| IN02.simple | Implemented | GSTools Simple kriging | Known-mean bounded local prediction/variance and weights; direct GSTools simple-kriging parity. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN02.universal | Implemented | PyKrige UniversalKriging | Default affine or arbitrary supplied eight-column universal trend; 32-neighbor rank-aware solves and direct PyKrige/GSTools parity. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN02.external | Implemented | GSTools ExtDrift / PyKrige external drift | Supplied flat or vec4 observation/query drift, missing-drift and rank status, prediction variance and optional weights; direct PyKrige/GSTools parity. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN03.error | Implemented | GSTools Krige cond_err | Nonnegative observation-error diagonal separate from latent process nugget; direct GSTools and gstat error-aware prediction evidence. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN03.block | Implemented | gstat krige: block | Up to 64 weighted discrete support points, normalized weights, double covariance averaging and direct gstat irregular-block prediction/variance parity. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN04 | Implemented | gstat fit.lmc/predict | Two to four variables, known or separate unknown means, one to four supplied PSD LMC structures, masks and singular status. Direct gstat three-variable/two-structure weighted-block prediction parity. | `gpu-spatial-analysis`: `GPULocalKriging` |
| IN05 | Implemented | SciPy RBFInterpolator | Eight SciPy kernels, degrees −1 through 2, scalar/per-sample smoothing, 32-neighbor solves and resident grid/CSR composition. Direct SciPy parity; 1,024/16,384-site pipeline measurements. | `gpu-spatial-analysis`: `GPULocalRBF` |
| IN06 | Implemented | GSTools CondSRF | Shared GPU storage Cholesky for 128 observations plus 128 joint query sites; Philox batching, exact-hit/semidefinite handling, direct GSTools moments and gstat ensemble joint covariance. | `gpu-spatial-analysis`: `GPUConditionalGaussianSimulation` |
| IN07.coordinates | Implemented | CGAL natural_neighbor_coordinates_2 | Sibson stolen areas with resident Delaunay cavity traversal, CGAL coordinates/prediction parity, hull/sample/duplicate/collinear/capacity contracts. Indexed scaling measured through 65,536 sites. | `gpu-spatial-analysis`: `GPUNaturalNeighbor` |
| IN07.values | Implemented | CGAL linear_interpolation with natural-neighbor weights | Sibson stolen areas with resident Delaunay cavity traversal, CGAL coordinates/prediction parity, hull/sample/duplicate/collinear/capacity contracts. Indexed scaling measured through 65,536 sites. | `gpu-spatial-analysis`: `GPUNaturalNeighbor` |

## Spatial statistics and models

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-ST01 | Implemented | esda Moran_Local | Local Moran statistic exists; permutation conventions and masks remain explicit inputs/contracts. | `gpu-spatial-analysis`: `GPULocalMoran` |
| B-ST02 | Implemented | LOSH conditional residual permutation | Existing fixed-field residual-permutation inference, not raw-value permutation with refitting; not interchangeable with multivariate local randomization. | `gpu-spatial-analysis`: `GPULOSHResidualPermutationTest` |
| B-ST03 | Implemented | Local Poisson/logistic weighted GLM | Fixed/adaptive bandwidth; Gaussian/bisquare kernels; bounded IRLS and status. | `gpu-spatial-analysis`: `GPUGeneralizedGeographicallyWeightedRegression` |
| B-ST04 | Implemented | Unit-effects spatial-lag panel 2SLS | Pooled/unit effects with homoskedastic inference; does not implement spreg panel ML. | `gpu-spatial-analysis`: `GPUSpatialPanelRegression` |
| B-ST05 | Implemented | Spatial scan: Poisson/Bernoulli/space-time permutation | Existing model choices; no new scan-family backlog merely because an older roadmap omitted a mode. | `gpu-spatial-analysis`: `GPUSpatialScanStatistic` |
| ST01 | Implemented | spreg TSLS/GM_Lag robust covariance | HC0/HC1 and admissible symmetric unit-diagonal spatial-HAC full IV covariance; direct spreg 1.8.3 TSLS/robust_vm fixtures. | `gpu-spatial-analysis`: `GPUInstrumentalVariablesCovariance`, `GPUSpatialTwoStageLeastSquares` |
| ST02 | Implemented | spreg OLS_Regimes restrictions/Chow tests | Common/varying coefficients, full covariance, nested Chow F/degrees and per-coefficient/joint Wald chi-square tails; direct OLS_Regimes fixtures. | `gpu-spatial-analysis`: `GPURestrictedRegimeRegression` |
| ST03.ml | Implemented | spreg Panel_FE_Error and pooled panel diagnostics | Balanced unit-FE spatial-error ML plus pooled-panel LM error and robust LM lag/error tests; direct Panel_FE_Error and diagnostics_panel fixtures. | `gpu-spatial-analysis`: `GPUSpatialPanelErrorMaximumLikelihood` |
| ST03.random | Implemented | spreg Panel_RE_Lag/panel diagnostics | Random/unit spatial lag ML, conditional beta/rho covariance, spreg-compatible Hausman diagnostics; actual spreg 1.8.3 fixtures. | `gpu-spatial-analysis`: `GPUSpatialPanelErrorMaximumLikelihood` |
| ST04 | Implemented | spreg SUR/SURerrorGM | Nonspatial one-step SUR and Kelejian–Prucha spatial-error SUR, full joint covariance; direct spreg SUR/SURerrorGM fixtures. | `gpu-spatial-analysis`: `GPUSeeminglyUnrelatedRegression` |
| ST05 | Implemented | spreg Probit | Binary probit likelihood, average effects and expected-information errors plus Pinkse, Kelejian–Prucha and Pinkse–Slade residual diagnostics; direct spreg Probit. | `gpu-spatial-analysis`: `GPUProbitRegression` |
| ST06 | Implemented | mgwr local_collinearity/bandwidth/inference | Local collinearity/VDP, generalized LOO bandwidth search and SE/influence, full MGWR backfit Jacobian uncertainty/effective parameters with both residual-variance conventions; direct mgwr 2.2.1 fixtures. | `gpu-spatial-analysis`: `GPULocalCollinearity`, `GPUGeneralizedGeographicallyWeightedRegression`, `GPUMultiscaleGeographicallyWeightedRegression` |
| ST07 | Implemented | esda Geary_Local_MV randomization | Common-mask whole-vector conditional randomization, replay, equal-tail or esda minimum-tail p-values, classifications and Bonferroni; direct esda observed statistics and replay-tail checks. | `gpu-spatial-analysis`: `GPUMultivariateLocalGearyPermutationTest` |
| ST08 | Implemented | giddy FullRank_Markov/GeoRank_Markov | Full/geographic rank transitions with direct giddy 2.3.6 tie ordering, optional earlier-row tie ordering, common panels, mobility and existing stationary diagnostics. | `gpu-spatial-analysis`: `GPURankMarkov` |
| ST09.fuzzy | Implemented | libpysal fuzzy_contiguity | Planar filled-polygon separation tolerance with holes/multipart containment and edge provenance; direct libpysal fuzzy_contiguity fixtures. | `gpu-spatial-analysis`: `GPUFuzzyContiguity`, `GPUIslandAttachment` |
| ST09.islands | Implemented | libpysal attach_islands | Directed/reciprocal nearest original-island attachment, preserved existing weights and added-edge provenance; direct libpysal attach_islands fixtures. | `gpu-spatial-analysis`: `GPUFuzzyContiguity`, `GPUIslandAttachment` |
| ST10.blocks | Implemented | blockCV cv_spatial | Seeded supplied blocks and exact held-out buffers; repeated training-only OLS, generalized GWR bandwidth search, and reusable nested covariance/variogram or RBF bandwidth selection with outer refits and held-out metrics. Direct blockCV4.0.1 masks and independent nested NumPy/SciPy scores/predictions. | `gpu-spatial-analysis`: `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation` |
| ST10.buffers | Implemented | blockCV cv_buffer | Seeded supplied blocks and exact held-out buffers; repeated training-only OLS, generalized GWR bandwidth search, and reusable nested covariance/variogram or RBF bandwidth selection with outer refits and held-out metrics. Direct blockCV4.0.1 masks and independent nested NumPy/SciPy scores/predictions. | `gpu-spatial-analysis`: `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation` |
| ST10.other | Implemented | blockCV cv_cluster/cv_group/cv_nndm/cv_knndm | Resident group-rank folds, environmental farthest-first/Lloyd clustering, NNDM exclusion radii and kNNDM candidate scoring/automatic spatial clustering; direct blockCV4.0.1 group/cluster/NNDM/kNNDM fixtures. | `gpu-spatial-analysis`: `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation` |
| ST11.gravity | Implemented | spint unconstrained gravity | Unconstrained one-covariate Poisson flows with grid or continuous bounded profile calibration, deviance/marginal residuals and explicit convergence; direct spint GLM fixtures. | `gpu-spatial-analysis`: `GPUSpatialInteraction` |
| ST11.origin | Implemented | spint production-constrained gravity | Origin-constrained one-covariate Poisson flows with grid or continuous bounded profile calibration, deviance/marginal residuals and explicit convergence; direct spint GLM fixtures. | `gpu-spatial-analysis`: `GPUSpatialInteraction` |
| ST11.destination | Implemented | spint attraction-constrained gravity | Destination-constrained one-covariate Poisson flows with grid or continuous bounded profile calibration, deviance/marginal residuals and explicit convergence; direct spint GLM fixtures. | `gpu-spatial-analysis`: `GPUSpatialInteraction` |
| ST11.doubly | Implemented | spint doubly constrained gravity | Doubly constrained one-covariate Poisson flows with grid or continuous bounded profile calibration, deviance/marginal residuals and explicit convergence; direct spint GLM fixtures. | `gpu-spatial-analysis`: `GPUSpatialInteraction` |

## Point processes, sampling and clustering

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-PP01 | Implemented | scikit-learn HDBSCAN | HDBSCAN already exists as a distinct clustering operation; no implication of OPTICS reachability output. | `gpu-spatial-analysis`: `GPUHierarchicalDensityClustering` |
| B-PP02 | Implemented | Random points within polygons | Existing bounded random point generation; no GRTS inclusion-probability design. | `gpu-spatial-analysis`: `GPURandomPointsInPolygon` |
| PP01 | Implemented | spatstat Kinhom/pcf/density.ppp | Supplied/self-included/leave-one-out Epanechnikov intensity; polygon border or translation K; kernel-smoothed pcf with radius/distance divisor and none/reflection origin correction. Direct pinned spatstat and exact clipped-overlap references cover concave, holed and disconnected windows; 4096-event resident execution is verified. | `gpu-spatial-analysis`: `GPUInhomogeneousPointPattern` |
| PP02 | Implemented | spatstat envelope/dclf.test | Resident supplied curves, pointwise envelopes, maximum-deviation/DCLF tests, and two-sided extreme-rank simultaneous envelopes with conservative tie intervals. Direct spatstat DCLF and independent SciPy rank fixtures. | `gpu-spatial-analysis`: `GPUPointPatternEnvelope` |
| PP03 | Implemented | spatstat linear-network density/inhomogeneous patterns | Arc-length quadrature-normalized shortest-path triangular density plus supplied-intensity network K with Ang shell correction and three normalizations; direct spatstat.linnet 3.5.3 branch/cycle/disconnected fixtures. | `gpu-network`: `GPUNetworkKernelDensity`, `GPUNetworkInhomogeneousK` |
| PP04 | Implemented | scikit-learn OPTICS | Complete cooperative GPU core/reachability/predecessor ordering through 4,096 rows, DBSCAN and Xi extraction with nested intervals and predecessor correction; pinned scikit-learn 1.7.2 parity. | `gpu-spatial-analysis`: `GPUOPTICS` |
| PP05.finite | Implemented | spsurvey grts: finite equal probability | Finite-point randomized hierarchical systematic equal-probability sampling with design weights. | `gpu-spatial-analysis`: `GPUSpatialBalancedSample`, `GPUContinuousSpatialSample` |
| PP05.strata | Implemented | spsurvey grts: strata/unequal probability | Separate finite-point strata and integer-ticket unequal inclusion with explicit first-order probabilities. | `gpu-spatial-analysis`: `GPUSpatialBalancedSample`, `GPUContinuousSpatialSample` |
| PP05.frames | Implemented | spsurvey grts: line/area frames and supplementary options | Continuous arc-length and triangle-union random-shift lattice designs, fixed legacy exclusion, minimum spacing, independent complete replacement waves, intensity weights/provenance and explicit capacity/work budgets. | `gpu-spatial-analysis`: `GPUSpatialBalancedSample`, `GPUContinuousSpatialSample` |

## Networks and allocation

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-NW01 | Implemented | Static turn penalties and banned edge-pair turns | Directed line-graph state exists; not arbitrary restriction sequences or time-dependent costs. | `gpu-network`: `GPUNetworkLineGraph` |
| B-NW02 | Implemented | Bounded alternative-route enumeration | Existing explicit alternative-route contract and limits. | `gpu-network`: `GPUAlternativeRoutes` |
| B-NW03 | Implemented | access-style gravity accessibility | Opportunity-weighted static scores with none/exponential/power decay; not flow calibration. | `gpu-network`: `GPUNetworkAccessibility` |
| NW01 | Implemented | pgRouting withPoints | Fractional directed-edge paths with legal turns, same-edge/parallel provenance; direct pgRouting 3.8.0 withPointsCost same-edge, parallel-edge and unreachable fixtures, independent NetworkX oracle, and measured 8/64/128-edge batches. | `gpu-network`: `GPUNetworkPointRouting` |
| NW02 | Implemented | NetworkX maximum_flow/minimum_cut | Integral max flow, residual cut certificate, original-edge conservation; NetworkX exact references and 8/32/128-node measurements. | `gpu-network`: `GPUNetworkFlow` |
| NW03 | Implemented | NetworkX min_cost_flow/network_simplex | Minimum-cost residual cancellation, balanced-demand supernodes, unserved demand, optional exact 64-bit integer objective. | `gpu-network`: `GPUNetworkFlow`, `prepareGPUNetworkFlowDemands` |
| NW04 | Implemented | NetworkX louvain_communities | Weighted directed/undirected modularity, seeded/ascending visitation, quotient levels, objective/convergence trace; NetworkX and 4/16/32-vertex measurements. | `gpu-network`: `GPUNetworkLouvain` |
| NW05.backup | Implemented | spopt LSCPB | Exhaustive bounded backup coverage with minimum-cover then maximum-backup objectives; direct spopt0.7.0 LSCPB/CBC reference. | `gpu-network`: `GPUNetworkFacilityOptimization` |
| NW05.dispersion | Implemented | spopt PDispersion | Exhaustive bounded p-dispersion, completion status and selected facilities; direct spopt0.7.0 PDispersion/CBC reference and 4/8/16-facility measurement. | `gpu-network`: `GPUNetworkFacilityOptimization` |
| NW05.capacity | Implemented | spopt capacitated facility objectives | Exhaustive bounded unsplit capacitated p-median with explicit feasible/complete status; direct spopt0.7.0 PMedian/CBC reference. | `gpu-network`: `GPUNetworkFacilityOptimization` |
| NW06 | Implemented | Valhalla departure-time routing/restriction semantics | Affine/piecewise FIFO arrival, explicit first-sample waiting, reachable overlapping restriction automaton and provenance; time-expanded NetworkX reference. | `gpu-network`: `GPUNetworkTimeRouting`, `prepareGPUNetworkRestrictionStates` |
| NW07 | Implemented | access.Access.raam | Integral/fractional demand, fractional/absolute shift caps, conserved allocations/unserved rows and equilibrium gap; direct access1.1.9 equilibrium fixture and <=64-origin measurements. | `gpu-network`: `GPUNetworkRationalAccess` |

## Movement and temporal geometry

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-MV01 | Implemented | MovingPandas-style resampling/track metrics | Existing resampling and trajectory metrics under explicit time/interpolation contracts. | `gpu-spatial-analysis`: `GPUTrajectoryResample`, `GPUTrajectoryMetrics` |
| B-MV02 | Implemented | DTW/LCSS/discrete Frechet trajectory comparison | Existing DTW and LCSS bands; bounded dynamic programs. | `gpu-spatial-analysis`: `GPUTrackSimilarity` |
| MV01 | Implemented | MovingPandas KalmanSmootherCV | Isotropic/full correlated Joseph/RTS covariance, explicit WGS84 local-plane/bigint epoch preparation; direct MovingPandas0.22.4 and NumPy/pyproj references, <=4096-observation measurements. | `gpu-spatial-analysis`: `GPUTrajectoryKalman`, `prepareGPUTrajectoryObservations` |
| MV02 | Implemented | MobilityDB temporal restriction/relationship operations | Polygon rings/holes, open/closed/half-open intervals, merged compaction, analytic temporal pairwise within-distance and three-valued gaps; direct MovingPandas clipping and MobilityDB 1.2.1 polygon/time-boundary/temporal-distance references. | `gpu-spatial-analysis`: `GPUTrajectoryRestriction` |
| MV03 | Implemented | tslearn/dtaidistance batched/pruned DTW | Admissible metric bounds, stable GPU survivor compaction/capacity status, exact threshold top-k and indexed-pair provenance; direct tslearn/similaritymeasures fixtures and <=4096-pair selectivity measurements. | `gpu-spatial-analysis`: `GPUTrackSimilaritySearch` |

## Urban morphology and change of support

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-UR01 | Implemented | tobler area_interpolate-style redistribution | Existing weighted areal interpolation; ancillary-mask workflow is separate. | `gpu-spatial-analysis`: `GPUArealInterpolation` |
| UR01 | Implemented | momepy morphological_tessellation/enclosed_tessellation | Building-boundary sampling and stable-ID polygon dissolve; convex or simple nonconvex enclosures with holes, GPU clipping and independent overflow status. Direct momepy 0.10.0 concave/holed fixtures pass. | `gpu-spatial-analysis`: `addBuildingTessellationRecipe` |
| UR02 | Implemented | momepy street_profile | Side-aware cross sections, missing-side censoring, height/width and sampled frontage; optional event-split visible facade arc length in a straight street corridor. Direct momepy street-width reference and occlusion tests pass. | `gpu-spatial-analysis`: `addStreetProfilesRecipe` |
| UR03 | Implemented | momepy perimeter_wall/orientation/alignment | Shared-wall adjacency/length and axial differences; cardinal orientation/alignment with explicit Queen/custom support. Optional true component union measures exterior-only perimeter, excluding courtyards. Direct momepy 0.10.0 shared-wall, buffer=0 perimeter, cardinal and isolate fixtures pass. | `gpu-spatial-analysis`: `addBuildingAdjacencyRecipe` |
| UR04 | Implemented | tobler masked_area_interpolate | Retained first-tranche fractional source support, ancillary eligibility, whole-cell target ownership and conservation ledger; direct pinned tobler masked-area reference retained, with 16 native tests rerun. | `gpu-spatial-analysis`: `addDasymetricRedistributionRecipe` |
| UR05.isovist | Implemented | depthmapX isovists | Uniform or sorted vertex-event isovists with explicit inside/outside even-odd/nonzero free-space classification, holes, boundaries, nonfinite evidence and physical ray/polygon measures. | `gpu-spatial-analysis`: `addIsovistsRecipe`, `addVisibilityGraphRecipe` |
| UR05.graph | Implemented | depthmapX visibility graph analysis | <=256-viewpoint visibility with explicit free-space classification, raw/normalized choice and HH/P-value/Teklenburg/Penn metrics using actual BFS maximum depth. Direct compiled depthmapX v0.9.1 mathematical helpers match. | `gpu-spatial-analysis`: `addIsovistsRecipe`, `addVisibilityGraphRecipe` |

## Landscape ecology and equity

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| B-EC01 | Implemented | landscapemetrics-style basic patch geometry | Existing per-patch area/perimeter/bounds; no claim of all landscape indices. | `raster-analysis`: `GPURasterPatchMetrics` |
| B-EC02 | Implemented | PySAL segregation indices | Existing segregation contributor includes multigroup analysis; decomposition remains separate. | `gpu-spatial-analysis`: `GPUSegregation` |
| B-EC03 | Implemented | Gini/Theil/Atkinson/Hoover/Palma/Lorenz and Theil T decomposition | Existing weighted per-zone outputs and between/within-zone Theil T; finite nonnegative values with explicit zero rules. | `dataframe-analysis`: `GPUInequality` |
| EC01 | Implemented | landscapemetrics class/landscape indices | Patch/class area, physical perimeter, adjacency, class and landscape mesh/cohesion/aggregation and two contagion conventions; independent boundary/background controls and GPU reconciliation of globally unique tile patch IDs. Direct landscapemetrics 2.2.1 square-cell/background and singleton-class fixtures pass; pooled landscape cohesion is calculated separately from class scores. | `gpu-spatial-analysis`: `addLandscapeMetricsRecipe` |
| EC02 | Implemented | landscapemetrics core/proximity/nearest-patch indices | Exact separable boundary or cell-center distance fields, physical core or integer four-neighbor erosion, explicit open/edge domain policy and optional GPU seam reconciliation. Direct landscapemetrics core/ENN fixtures pass; two-patch proximity is composed from its direct area/ENN. Native 32²/128²/256² physical-mode measurements retain exact distances. | `gpu-spatial-analysis`: `addPatchProximityRecipe` |
| EC03 | Partial | Circuitscape effective resistance/current maps | Small CG resistance solve with independent residual check (at most 256 vertices, 65,536 edges). Sparse Jacobi-PCG, raster conductance adapter and current density were lost in the split (see [roadmap.md](./roadmap.md) EC03). | `gpu-spatial-analysis`: `addResistanceConnectivityRecipe` only |
| EC04 | Implemented | segregation DecomposeSegregation/counterfactual inference | Explicit common-support Shapley decomposition plus composition/dual-composition/share quantile counterfactuals, stable-ID support alignment and replayable Philox paired-label randomization. Direct segregation 2.5.3 methods and all 16 exchange assignments pass. | `gpu-spatial-analysis`: `addSegregationDecompositionRecipe` |

## Kepler analysis coverage (2026-10-25)

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| KG01 | Implemented | Kepler / GeoDa | Single-variable selected-quantile LISA | `gpu-spatial-analysis`: `GPUQuantileLISA` |
| KG02 | Implemented | Kepler / GeoDa | Bounded Dorling cartogram (1024 rows, 256 passes) | `gpu-spatial-analysis`: `GPUDorlingCartogram` |
| KG03 | Implemented | Kepler / GeoDa | Bounded exact natural breaks (64 classes; 1024 rows materialized, 8192 recomputed) | `dataframe-analysis`: `GPUClassBreaks` |
| KG04 | Implemented | Kepler / GeoDa | Named variable-standardization recipes | `dataframe-analysis`: `GPUVariableStandardization` |
| KG05 | Implemented | Kepler / GeoDa | Excess-risk and spatial-rate recipes | `gpu-spatial-analysis`: `addExcessRiskRecipe`, `addSpatialRateRecipe` |
| KG06 | Partial | DuckDB 1.4.1 | Consumer-required SQL / data wrangling: quoted identifiers, single-key USING, CASE/COALESCE, nullable COUNT done; DOUBLE, WKB, arbitrary strings, right-column projections and UNION open | Published `@luma.gl/graph/gpu-sql` (`GPUSQLContext`); not in this repo |

## Deliberate boundaries (excluded)

| ID | Status | Reference | Supported subset or target | LIB entry point |
| --- | --- | --- | --- | --- |
| X01 | Excluded | GDAL/Open, GeoPandas read_file, PDAL readers | File decoding, cloud catalogues and transport belong in adapters. | None (excluded) |
| X02 | Excluded | PROJ database/grid acquisition and arbitrary pipeline runtime | Host preparation owns CRS resources and network/file resolution. | None (excluded) |
| X03 | Excluded | OTB atmospheric/optical calibration and RPC orthorectification | Requires a sensor-processing roadmap and calibration inputs. | None (excluded) |
| X04 | Excluded | Valhalla transit timetable/service routing | Transit schedules and service operations need dedicated ingestion/state/design. | None (excluded) |
| X05 | Excluded | Hydraulic flood simulation | Shallow-water dynamics, boundary conditions and calibration require a separate roadmap. | None (excluded) |
| X06 | Excluded | CGAL full 3D solid Boolean/volume meshing | Dedicated 3D topology and precision contracts are outside this function backlog. | None (excluded) |

## Rows to re-check

| ID | Note |
| --- | --- |
| EC03 | Sparse resistance is not in this repository. The 256-vertex recipe has a stale comment at `gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts:33`. |
| PC02 | Voxel downsample and radius filter come from published `@luma.gl/graph/gpu-pointcloud`; the statistical-outlier mask and voxel merge are local. |
| KG06 | The SQL engine is published `@luma.gl/graph/gpu-sql`. SQL gaps are upstream work. |
| KG03 | Exact natural breaks now recompute to 8,192 rows (`dataframe-analysis/column-classification/gpu-class-breaks.ts:127-129`); the KG03 row says 1,024 (the materialized limit). |
| IN01 | Resident variogram fitting is slower than CPU on small inputs; kept for pipeline residency. |
| NW-SCC | Strongly connected components are limited to 1,024 vertices (`gpu-network/network-structure/gpu-network-structure.ts:66`). |

## Updating an entry

Promote only the tested function and options, not the whole backlog family. Record the exact API and reference version, supported modes, coordinate and time models, null, tie and boundary rules, capacity and convergence behavior. Attach source contract, pinned generator or fixture, GPU execution evidence and scale evidence separately. If only one mode lands, keep the other suffixed rows partial. Recompute the inventory counts when rows are added or split. When a symbol moves to or from published luma.gl, change the entry point column and the table above in the same commit.
