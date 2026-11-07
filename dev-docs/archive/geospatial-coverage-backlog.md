> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/geospatial-coverage-backlog.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Superseded by the consolidated roadmap; kept as the detailed evidence trail.

# GPU geospatial coverage research and backlog

Research date: October 23, 2026. Baseline: the current working tree, including the uncommitted
geospatial expansion. This is a maintainer backlog, not a statement of shipped API availability.
The original 78 work items below extend the [first expansion](./geospatial-toolkit-expansion.md); they do not
reopen its C01–C30 or R01–R12 implementation paths.

October 25 follow-up: six additional Kepler/GeoDa analysis items, KG01–KG06, are tracked in
[Kepler analysis coverage](#kepler-analysis-coverage-2026-10-25). Their bounded implementations and remaining SQL scope are recorded
[separately](./kepler-analysis-implementation.md) from the original 78-item implementation record.

The [function coverage matrix](./geospatial-function-coverage.md) separates implemented subsets,
partial support, planned functions and deliberate exclusions. It maps every backlog ID to named
reference operations and records evidence without claiming complete upstream-package parity.

The main opportunities are deeper numerical contracts and missing domains: point clouds, remote
sensing, geostatistical prediction, urban morphology, landscape connectivity and temporal geometry.
Common 2D GIS operations already have broad coverage. Prioritize useful extensions and compositions
before building another implementation of those operations.

The GPU implementation survey (local research log, not checked in)
maps selected backlog items to primary GPU code, algorithm papers and CPU semantic references.
It supplies integration/portability findings and proposed benchmarks; it does not promote function
status or replace the [recorded verification](./geospatial-toolkit-expansion-follow-up.md).
See the [roadmap index](./README.md) for document ownership and historical context.

## Research method and limits

Compared official library API catalogues and selected function documentation with public barrels,
implementation contracts, the [cross-reference](../../docs/api-reference/experimental/gpu-spatial-analysis-cross-reference.md),
and [reference generators](../../modules/graph/test/oracles). The source register below
records existing inspiration families and additional references worth adopting. Links were reviewed
on the research date; `stable`, `latest` and `main` links are discovery references, not version pins.
Pin the exact package/version or commit when implementing an oracle. Some upstream pages describe
development APIs. The pgRouting site denied direct page access; its indexed official manual and
migration guide were available, so routing details require a pinned-manual check before implementation.

This is broad coverage research, not an exhaustive enumeration of every function in every package.
Library aliases, file readers, plotting APIs and service orchestration do not each merit GPU kernels.
The proposed GPU approaches, priorities and sizes are engineering judgments, not upstream claims
or measured performance results. No upstream library was installed or executed for this research.

## Inspiration and reference register

“Retain” means an established inspiration family in the repository's cross-reference or prior
proposal; it does not imply an installed package or an independent fixture for every operation.
“Add” means a focused addition to this backlog's reference set, even if mentioned elsewhere in the
repository. References are development/oracle inputs, not new browser runtime dependencies.

| Family | Decision and research finding | Backlog use |
| --- | --- | --- |
| [Shapely/GEOS][shapely], [GeoPandas][geopandas], [sf][sf] | Retain the geometry baseline; add sf as a second interface and spherical-model comparison. GEOS-backed interfaces are not independent geometry engines. | GE01–GE07; canonicalization, precision and topology semantics |
| [PostGIS][postgis], [Turf][turf] | Retain SQL/browser vocabulary. Separate geographic measurements, topology and bounded outputs; subdivision is a meaningful remaining operation. | GE01, GE02, GE07, MV02 |
| [Rust geo][rust-geo], [Apache Sedona][sedona] | Retain existing cross-reference families as additional geometry implementations and SQL contract checks. Do not infer parity from shared function names. | GE01–GE06; differential fixtures where algorithms are independent |
| [QGIS][qgis], [ArcGIS Spatial Analyst][arcgis], [CARTO][carto] | Retain workflow discovery references. Their toolboxes help identify complete user tasks; service APIs, GUI tools and proprietary implementations are not numerical oracles by default. | Raster/terrain compositions and delivery scenarios |
| [PySAL esda][esda], [spreg][spreg], [mgwr][mgwr], [spopt][spopt] | Retain statistical and optimization semantics. More estimators and inferential options remain than the existing family-level coverage suggests. | ST01–ST07, NW05 |
| [libpysal][libpysal], [mapclassify][mapclassify], [access][access], [pointpats][pointpats] | Retain weights, classification, accessibility and point-pattern references. Reuse existing pooled class breaks, weights algebra and floating catchments; fuzzy contiguity and rational-agent access add distinct behavior. | ST09, NW07, PP01–PP02; classification parity remains baseline work |
| [PySAL tobler][tobler], [segregation][segregation], [giddy][giddy] | Retain change-of-support, segregation and dynamics references. Focus on ancillary-data workflows, decomposition and rank dynamics. | ST08, UR04, EC04 |
| [SciPy interpolation][rbf] and [signal processing][savgol], [scikit-learn][optics], [scikit-gstat][variogram] | Retain numerical references alongside existing NumPy/statsmodels fixtures. Useful for model fitting, local interpolation, time-series filters and reachability clustering. | IN01, IN05, RS06, PP04 |
| [GDAL/Rasterio][rasterio], [scikit-image features][texture], [graphs][image-graph], [registration][registration] | Retain raster references. Focal operations already exist; texture, image alignment and region adjacency add distinct outputs. | RS04–RS08 |
| [GRASS][grass], [pysheds][pysheds], [xDEM][xdem], [RVT][rvt], [pvlib][pvlib] | Retain terrain, hydrology and illumination references. Extend routing consumers and DEM comparison rather than re-add slope, HAND, viewsheds or solar position. | TE02–TE06 |
| [NetworkX flow][network-flow] and [communities][communities], [spaghetti][spaghetti] | Retain graph and network-pattern references. Max/min-cost flow and modularity optimization differ from existing routing and label propagation. | NW02–NW04, PP03 |
| [H3][h3], [PROJ horizontal grids][hgrid], [vertical grids][vgrid], [PyGeodesy][pygeodesy] | Retain cell/geodesy references. Add datum-grid semantics without conflating them with map projection or cell-key conversion. | CR01–CR05 |
| [spatstat][spatstat], [tslearn][tslearn], [dtaidistance][dtaidistance], [similaritymeasures][similaritymeasures] | Retain point-process and trajectory references. Existing K/L, DTW, discrete Fréchet, Hausdorff and LCSS are reuse points. | PP01–PP03, MV03 |
| [CGAL][cgal], [natural-neighbor interpolation][natural-neighbor] | Add focused geometry references for straight skeletons, difficult topology and interpolation from Voronoi areas. Keep package-specific licensing and attribution separate from independent implementation. | GE08, IN07 |
| [S2 Geometry][s2], [S2 cell hierarchy][s2-cells], [GeographicLib][geographiclib] | Add global geometry and geodesy references. Spherical topology and ellipsoidal measurement solve different problems. | CR01, CR04, CR05 |
| [exactextract][exactextract], [terra][terra], [xESMF/ESMF][xesmf] | Add fractional summary, raster-stack and conservative-remapping references. Coverage weighting, observation weights and destination normalization must remain distinct. | RS01–RS03, RS05, RS06 |
| [WhiteboxTools hydrology][whitebox], [SAGA][saga] | Add terrain-processing breadth, especially depression breaching and basin analysis. Select named methods rather than claim equivalence to an entire toolbox. | TE01–TE03 |
| [PDAL][pdal] | Add a point-cloud analysis reference beyond ingestion/rendering: local features, ground classification, height normalization, thinning and surface change. | PC01–PC06 |
| [GSTools][gstools], [PyKrige][pykrige], [gstat][gstat], [spdep][spdep] | Add stronger prediction/simulation and independent statistical reference coverage. gstat supports block/multivariable prediction; spdep is useful for checking inference conventions. | IN01–IN06, ST01–ST07 |
| [Orfeo ToolBox][otb] | Add remote-sensing workflows: radiometric indices, spectral unmixing and texture. Sensor calibration/orthorectification are larger separate systems. | RS04, RS09, RS10 |
| [pgRouting][pgrouting], [Valhalla][valhalla] | Add routing semantics for points on edges, flow and departure-time behavior. Existing banned turns do not imply arbitrary restriction sequences or time-dependent search. | NW01, NW06 |
| [MovingPandas][movingpandas], [MobilityDB][mobilitydb] | Add track smoothing and temporal restriction/relationship contracts. Existing stop metrics and closest approach remain the base. | MV01–MV03 |
| [momepy][momepy] | Add urban form as a first-class application: building tessellations, street profiles, shared walls and orientation relationships. | UR01–UR03 |
| [landscapemetrics][landscape], [Circuitscape][circuitscape] | Add class/landscape ecology and resistance connectivity. Current-flow connectivity is not least-cost routing. | EC01–EC03 |
| [cuSpatial][cuspatial] | Add GPU algorithm and benchmark comparison, especially spatial indexing and resident data paths. CUDA measurements are not evidence of browser WebGPU speedups. | Cross-cutting performance evidence; no duplicate join backlog |
| [blockCV][blockcv], [spsurvey GRTS][grts] | Add spatial validation and survey-design references. Ordinary random folds and uniformly random points do not establish spatially separated validation or design-based inclusion probabilities. | ST10, PP05 |
| [PySAL spint][spint] | Add calibrated origin/destination flow models. Existing gravity accessibility and Huff scores do not estimate observed flow-model parameters. | ST11 |
| [BFAST][bfast] | Add seasonal structural-break discovery. Existing difference maps and monotonic trend tests remain useful baselines. Begin with an explicitly restricted model, not full BFAST parity. | RS11 |
| [depthmapX][depthmapx] | Add obstacle-based isovists and visibility graph analysis. Terrain viewsheds and general graph analytics supply different contracts. | UR05 |

## Reconciled baseline: do not add these again

Source code takes precedence over historical roadmap wording. These are implementation-presence
findings, not fresh correctness or performance certification.

| Existing capability | Evidence and consequence |
| --- | --- |
| Overlay, buffering, polygonization, repair, Delaunay/Voronoi, alpha hulls and constrained triangulation | [Spatial exports](../../modules/graph/src/gpu-spatial-analysis/index.ts). GE items below add particular missing semantics. Alpha hulls are not ratio hulls. |
| D8, D-infinity, Freeman/Quinn MFD, filling and flat resolution | [Flow implementation](../../modules/graph/src/gpu-terrain/hydrology/gpu-terrain-flow.ts). TE02 extends branching consumers; it does not invent these routing methods. |
| HAND, hydrologic indices, stream order, downstream distance and stream extraction | [Hydrology exports](../../modules/graph/src/gpu-terrain/hydrology/index.ts). The new terrain work concerns breaching, basin structure, uncertainty and domain stitching. |
| Gaussian GWR search and conditioning | [Gaussian GWR](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression.ts) already provides bandwidth ladder/golden-section search and condition surfaces. ST06 adds remaining diagnostics and estimator-specific search/inference; it does not reopen all bandwidth selection. |
| Adaptive Poisson/logistic GWR | [Generalized GWR](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts). Bandwidth selection, diagnostics and inference are separate extensions. |
| Pooled and unit-effects spatial-lag panels | [Panel implementation](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression.ts). Within-transformed 2SLS does not establish parity with spreg's fixed-effects ML estimators. |
| LOSH residual permutation inference | [Inference contract](../../docs/api-reference/experimental/gpu-spatial-analysis/losh-residual-permutation.md). Do not reopen it as “add LOSH inference”; joint/multivariate conventions remain separate. |
| GDAL-style four-quadrant nodata fill, tile-routed reprojection, bounded alternative routes | [Fill contract](../../docs/api-reference/experimental/gpu-raster/fill-nodata.md), [raster exports](../../modules/graph/src/gpu-raster/index.ts), [routes](../../docs/api-reference/experimental/gpu-network/alternative-routes.md). Scaling and further filters are extensions. |
| Turn angles, penalties and banned edge-pair turns | [Network line graph](../../modules/graph/src/gpu-network/network-line-graph/gpu-network-line-graph.ts). Reuse it for richer routing state. |
| Stops, resampling, zone crossings, encounters and continuous closest approach | [Trajectory metrics](../../modules/graph/src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics.ts), [closest approach](../../modules/graph/src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach.ts). Temporal interval outputs and smoothing are additional contracts. |
| DBSCAN, HDBSCAN, k-means, AZP, max-p, Ward and SKATER regionalization | [Spatial exports](../../modules/graph/src/gpu-spatial-analysis/index.ts). OPTICS remains a distinct candidate; ordinary clustering/regionalization is not missing wholesale. |
| Bernoulli and space-time permutation scans as well as Poisson scans | [Scan implementation](../../modules/graph/src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts). Do not copy old “Bernoulli scan” opportunities into this backlog. |
| Fractional zonal totals/means, multigroup segregation, weighted areal interpolation | [Fractional summaries](../../modules/graph/src/gpu-raster/gpu-raster-fractional-coverage.ts), [segregation](../../modules/graph/src/gpu-spatial-analysis/segregation/gpu-segregation.ts), [areal interpolation](../../modules/graph/src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts). Distribution summaries, decomposition and end-to-end ancillary-data recipes remain useful. |

## Backlog conventions

The research inventory originally proposed the 78 non-KG IDs below. The first implementation tranche supplied
bounded RS01, RS02, RS05, RS09 and UR04 APIs; its evidence remains in the
[tranche validation record](./geospatial-coverage-tranche-one.md). The subsequent
[parallel implementation record](./geospatial-coverage-parallel-implementation.md) maps all 78 IDs
to actual source and explicitly identifies the remaining modes, numerical contracts and evidence.
The other rows are no longer wholly unimplemented design targets, but a scoped first model does
not close the full completion contract. The tables below retain the original targets and acceptance
criteria; use the implementation record and [function matrix](./geospatial-function-coverage.md)
for current status. Only implemented APIs belong in runtime discovery.

- **P1**: first candidates after existing validation gates; broad utility and reusable dependencies.
- **P2**: valuable next coverage, with a named consumer and prerequisite evidence.
- **P3**: research or a large specialist project; benchmark a prototype before committing to an API.
- **N**: new operation; **E**: extend an existing operation; **R**: recipe/adapter; **V**: validation
  or semantic closure. Sizes **S/M/L/XL** are relative engineering scope, not time estimates.

Each table states the proposed function, GPU work/dependencies, and completion contract/oracle.
The common acceptance gates below also apply to every row.

### Kepler analysis coverage (2026-10-25)

Implementation update: KG01–KG05 have bounded implementations; KG06 has two implemented slices
and explicitly open SQL subitems. See the [implementation record](./kepler-analysis-implementation.md)
for numerical contracts, reference evidence, limits and integration checks. The acceptance criteria
below preserve the original target; they do not imply full upstream parity.

Consumer: reuse Kepler's existing frontend analysis workflows with graph contributors. The
[GeoDa tool inventory](https://github.com/geodaai/openassistant/tree/main/packages/tools/geoda),
[LISA tool](https://github.com/geodaai/openassistant/blob/main/packages/tools/geoda/src/lisa/tool.ts),
[rate tutorial](https://github.com/keplergl/kepler.gl/blob/master/docs/spatial-analysis-tutorial/rate-mapping.md)
and [DuckDB explorer](https://github.com/keplergl/kepler.gl/blob/master/docs/user-guides/sql-data-explorer.md)
are discovery references. Kepler documentation spans development features and different assistant
versions; pin the actual host and tool packages before claiming tool availability or parity.
This source review ran no upstream tools or numerical comparisons.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| KG01 | P1 R/M | Quantile LISA, beginning with a single variable and selected quantile | Compose existing column quantiles, indicator masks, spatial weights, local join counts and permutation machinery; assess multivariate joint membership separately | Match pinned GeoDa `quantileLisa` quantile boundaries/ties, conditional permutations, islands, missing values, p-value tails and cluster labels. Test constant/tied columns, empty selected classes and replayable inference; primitives alone do not establish a supported recipe. |
| KG02 | P2 N/L | Dorling cartogram | Iterative GPU circle placement with spatial neighbor queries, area-derived radii and explicit iteration budgets | Define planar coordinates, initialization, area scale, zero/negative/missing values and source IDs. Report residual overlap, displacement and convergence/budget status. Compare pinned GeoDa cartogram behavior and invariants; heuristic layouts need not have identical coordinates. Preserve chunk ownership and keep convergence control on the GPU. |
| KG03 | P2 E/L | Exact natural-breaks classification mode and GeoDa-compatible class assignment | Extend `GPUClassBreaks` with a bounded globally optimal Fisher–Jenks path over sorted values; retain the existing approximate mode | Existing histogram/refinement natural breaks are explicitly approximate. Specify objective, capacities, repeated values, deterministic optimal-partition ties and upper/lower class-edge inclusion. Compare objective and assignments against pinned GeoDa and an independent exact oracle, including cases where local refinement misses the optimum. No silent approximation when exact mode exceeds limits. |
| KG04 | P1 R/M | Named variable-standardization recipes | Reuse column profiles, reductions and dataframe expressions for deviation from mean, z-score, MAD standardization, range adjust and range standardize | Audit each pinned GeoDa formula before adding a wrapper: MAD must not be guessed to mean median rather than mean absolute deviation. Specify population/sample conventions, masks, constants, nulls and nonfinite inputs. Test whole-column and chunked results; add only missing recipe behavior, not duplicate reduction kernels. |
| KG05 | P1 R/S | Complete rate-tool recipes, particularly excess risk and spatial rates | Reuse `GPUEmpiricalBayesRates`, `GPUSpatialEmpiricalBayesRates`, spatial lag and dataframe expressions | Preserve existing raw/global/spatial EB outputs. Add missing named compositions after auditing GeoDa: excess risk uses the pooled event/population rate; spatial rates must distinguish a ratio of weighted sums from an average of row rates. Verify zero denominators, islands, masks and scale factors against the pinned rate tool. |
| KG06 | P2 E/XL | Selected SQL/data-wrangling gaps required by Kepler analysis tools | First capture the actual queries emitted by the pinned host, then extend `LuSQLContext` and GPU dataframe lowering only for the missing expressions/operators | Produce a supported/unsupported query ledger and scoped subitems before implementation. Cover required casts, string/category and null behavior, derived columns and query composition only where the consumer demonstrates a gap. Compare with pinned DuckDB on identical Arrow inputs. Do not imply general DuckDB parity, file ingestion, remote querying or hidden CPU fallback. |

Compatibility work is separate from missing functions: existing weights, Moran/Geary/Getis–Ord,
regression, spatial joins, geometry operations and density aggregation require a tool-contract
audit before substitution. Record parameter defaults, row identity, normalization, inference,
boundary rules, coordinate models and output schemas, then add only demonstrated missing modes to
their existing owners. In particular, coverage dissolve is not proof of arbitrary grouped dissolve.
Uploads, charts, assistant orchestration, map configuration and external geocoding/data services
remain host responsibilities; they are not new GPU contributors. Arrow conversion stays in
`@luma.gl/arrow`, with no Kepler or GeoDa runtime dependency added to the graph package.

### Geometry and topology

Owner: `gpu-spatial-analysis`, with shared topology in `analysis-core`. Reuse
[geometry editing](../../modules/graph/src/gpu-spatial-analysis/geometry-edit),
[noding](../../modules/graph/src/gpu-spatial-analysis/geometry-topology),
[label points](../../modules/graph/src/gpu-spatial-analysis/label-point) and
[offsets](../../modules/graph/src/gpu-spatial-analysis/offset-expansion).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| GE01 | P1 N/M | Normalize geometry and compare canonical geometry; reproducible caching/deduplication | Segmented ring rotation, orientation and lexicographic part sorting | Preserve empty/collection structure and source IDs; idempotence and canonical equality against [Shapely normalize][normalize]. Equality still specifies precision. |
| GE02 | P1 N/L | Subdivide large geometries into bounded pieces for selective joins | Queue/scan-based rectilinear splits using clipping and polygonization | Per-piece vertex limits, holes and source provenance; union reconstructs input. [PostGIS ST_Subdivide][subdivide]; benchmark candidate-count reduction against unsplit joins. |
| GE03 | P1 E/L | Precision reduction with valid-output and keep-collapsed policies | Extend cleanup through snap-rounded noding and repair | Separate pointwise rounding from topology reconstruction; test collapsed lines/rings, repeated points and idempotence with [set_precision][precision]. |
| GE04 | P2 E/L | Snap segments to reference vertices with inserted vertices | Candidate queries, sorted insertion positions and bounded compaction; extend vertex snapping | Explicit tolerance/tie policy and validity output. [Shapely snap][snap] uses safety heuristics and does not guarantee valid output; do not promise universal topology preservation. |
| GE05 | P2 E/L | Remove loops and select valid components from offset curves | Reuse noding and traversal after existing offset generation | Compare inner corners, self-crossings, side/orientation and multipart output with [GEOS/Shapely offset_curve][shapely]. Preserve original segment provenance. |
| GE06 | P2 E/L | Ratio-based concave hulls | Delaunay border-triangle pruning beside the existing alpha mode | Specify ratio and hole policy; test ratio endpoints and disconnected/degenerate inputs against [Shapely concave_hull][shapely]. Never reinterpret the existing circumradius parameter. |
| GE07 | P2 E/L | Maximum inscribed circle with a requested error bound | Bounded branch-and-bound using label-point distance bounds | Center, boundary witness, radius and remaining error bound; distinguish budget exhaustion from tolerance success. [Shapely maximum_inscribed_circle][inscribed]. |
| GE08 | P3 N/XL | Vector straight skeleton and inward polygon offsets | Event processing over explicit half-edge topology; prototype many independent small polygons first | Handle holes, simultaneous events and topology changes. Compare against [CGAL straight skeleton][cgal]; distinct from raster skeletonization or a medial axis. |

### Global coordinates and cells

Owners: `gpu-project` for transforms and `gpu-spatial-analysis` for geometry/cells. Extend the
[projection boundary](../../modules/graph/src/gpu-project/projection-crs-lowering.ts)
and [cell topology](../../modules/graph/src/gpu-spatial-analysis/cell-topology).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| CR01 | P2 V/L | Extend difficult-geodesic accuracy and error reporting | Reuse existing ellipsoidal areas/perimeters; investigate adaptive edge integration and inverse-geodesic fallback | Current geodesic area uses bounded sampling and perimeter uses Vincenty. Validate tolerance envelopes, holes, poles and near-antipodes against [GeographicLib][geographiclib]; report unsupported/uncertain cases. This is not a new ellipsoidal measurement API. |
| CR02 | P2 N/L | Horizontal datum-grid transforms | Sample caller-provided grids inside an explicitly prepared projection pipeline | Match grid priority, bounds, inverse iteration and missing-grid failure against [PROJ hgridshift][hgrid]. Grid acquisition stays outside kernels; no silent identity fallback. |
| CR03 | P2 N/M | Vertical datum/geoid corrections | Raster interpolation plus height arithmetic following CR02's grid contract | Explicit height units, vertical reference, sign and validity; compare [PROJ vgridshift][vgrid]. Do not treat orthometric height as ellipsoidal height. |
| CR04 | P3 N/XL | Spherical polygon predicates and overlay | Robust great-circle arc intersections and spherical ring assembly | Closed/open boundary choice, hemisphere/complement semantics, poles and antimeridian; [S2][s2] and its [validation design][s2-validation]. Planar longitude unwrapping is insufficient. |
| CR05 | P3 E/XL | Complete S2 analysis-cell adapter: topology, covering, measures and crosswalks | Reuse key storage and cell tables; add missing operations beyond any existing indexing/shader support | Face transitions and exact IDs against [S2 hierarchy][s2-cells]; compare cover guarantees separately from [H3 centroid containment][h3]. Cross-family transfer needs geometric overlap weights, not key relabeling. |

### Raster and remote sensing

Owner: `gpu-raster`; statistics kernels may live in `gpu-dataframe`/`analysis-core`. Reuse
[raster exports](../../modules/graph/src/gpu-raster/index.ts), fractional tuples, tile leases,
halos, grouping and FFT primitives. An aligned-grid contract comes before new reprojection behavior.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| RS01 | P1 E/M | Fractional zonal quantiles, variance and effective support | Sort/group existing zone-cell-fraction tuples; weighted cumulative reductions | Specify weighted CDF/interpolation and population/sample variance; empty and partial zones, nodata and partition conservation against [exactextract operations][exactextract]. Never average tile quantiles. |
| RS02 | P1 E/M | Fractional categorical composition, majority/minority and class area | Group by zone and category, accumulate coverage/area | Stable category ties, empty classes and separate missing area; [exactextract frac/majority/minority][exactextract]. Weights must not accidentally become counts. |
| RS03 | P2 N/L | Conservative raster remapping | Explicit source/destination cell overlap weights followed by sparse multiplication | Begin with rectilinear grids; distinguish total-preserving and area-normalized intensive outputs. Validate constant fields, masks and mass against [xESMF/ESMF conservative modes][xesmf]. Curvilinear/spherical grids are later scopes. |
| RS04 | P1 N/M | GLCM/Haralick texture maps | Quantize, build local pair histograms and reduce per direction/window | Levels, offsets, symmetry, normalization and halo exclusion against [scikit-image graycomatrix/graycoprops][texture] and [OTB texture][otb]. Bound levels squared times active windows. |
| RS05 | P1 N/M | Focal cross-band covariance, correlation and local regression | Reuse window sums/masks; small batched solves | Paired-valid counts, constant bands, singular neighborhoods and boundary modes against [terra focalPairs/focalReg][terra]. Fit statistics remain descriptive unless inference is separately defined. |
| RS06 | P1 R/M | Masked temporal composites and Savitzky–Golay smoothing of raster stacks | Time-axis segmented reductions/convolution using common grid and QA masks | Observation count, chosen-date provenance and edge policy; [terra temporal operations][terra], [SciPy savgol_filter][savgol]. Irregular dates require explicit resampling or a different fit. |
| RS07 | P2 N/L | Subpixel translation registration between raster dates | Phase correlation using existing FFTs and validity windows | Shift convention, overlap mask and subpixel refinement against [scikit-image phase_cross_correlation][registration]. Report failed/ambiguous alignment; do not imply nonrigid registration. |
| RS08 | P2 R/L | Region adjacency graph and bounded region merging | Extract component boundary pairs, aggregate weights, merge via explicit rounds | Reuse existing segmentation/components; stable source-region mapping, seam equivalence and merge stopping criteria against [scikit-image RAG/merge_hierarchical][image-graph]. |
| RS09 | P1 R/S | Calibrated spectral-index recipes beyond NDVI | Compose band math, supplied calibration and quality masks | Start with named NDWI, NBR, EVI and SAVI recipes; document band wavelengths, scaling and denominator masks against [OTB RadiometricIndices][otb]. Existing NDVI is not a new kernel. |
| RS10 | P3 N/L | Linear spectral unmixing with supplied endmembers | Batched small least-squares/NNLS solves; optional sum-to-one constraint | Abundances, residuals and constraint/convergence status against [OTB HyperspectralUnmixing][otb]. Endmember discovery and atmospheric correction are separate projects. |
| RS11 | P1 E/L | Seasonal trend decomposition and structural-break detection in raster stacks | Per-pixel harmonic design matrices and bounded segmented fits; reuse change maps, masks and solver diagnostics | Start with regular-time harmonic seasonality and one documented breakpoint criterion; report break dates/magnitudes, minimum segment length, missing-data policy and iteration limits. Compare the selected model with [BFAST][bfast] or a pinned equivalent reference; dummy-season, iterative decomposition and online monitoring require separate promotion. |

### Terrain and hydrology

Owner: `gpu-terrain`. Reuse existing flow, reach extraction, HAND, derivatives, visibility and
[terrain exports](../../modules/graph/src/gpu-terrain/index.ts).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| TE01 | P1 N/L | Least-cost depression breaching and explicit stream burning | Bounded search/relaxation plus elevation edits before existing routing | Limit cut depth/length, retain modification masks and unresolved pits; [Whitebox BreachDepressionsLeastCost][whitebox]. Burning and filling remain individually selected actions. |
| TE02 | P2 E/L | Branching-flow travel distance, transport and fractional catchments | Consume existing D-infinity/MFD receivers and fractions; weighted DAG reductions | Define expected/minimum/maximum distance separately; conserve injected mass and report cycles/unresolved receivers. [pysheds][pysheds], [GRASS r.watershed][grass]. Existing D8 distance is the baseline. |
| TE03 | P2 N/L | Depression hierarchy and stage-area-volume curves | Label nested sinks, discover spill saddles, sort elevation bins and reduce area/volume | Preserve real closed basins, nodata barriers, nested ownership and spill connectivity; [Whitebox sink tools][whitebox] and [SAGA terrain tools][saga]. Not a dynamic flood solver. |
| TE04 | P1 R/M | Catchment hypsometry, drainage density and longitudinal stream profiles | Existing watersheds/reaches plus grouped elevation distributions, lengths and slopes | Area units, reach direction, zero-area basins and mask accounting; compare composed [GRASS][grass]/[pysheds][pysheds] outputs with independent reductions. Mark this as a composed oracle. |
| TE05 | P2 N/L | DEM coregistration, stable-ground bias and uncertainty-aware differencing | Robust reductions and bounded shift/bias fits on stable masks | Match one [xDEM coregistration][xdem] model first; report residuals and uncertainty assumptions. Propagate spatial error correlation before claiming volume-change uncertainty. |
| TE06 | P2 E/XL | Cross-tile hydrology and reach stitching | Exchange boundary receiver summaries and iterate explicit tile dependencies | Complete-domain parity for D8 first, then branching flow; no treating tile edges as outlets or halo samples as fresh runoff. Validate with [pysheds][pysheds]/[GRASS][grass] and existing single-domain operators. |

### Point clouds and 3D terrain observations

Proposed home: a focused experimental point-cloud analysis entry after validating the shared
3D-neighborhood contract. Reuse generic GPU indexing/reductions and raster adapters; do not hide
LAS/LAZ/COPC decoding in a compute contributor. These are new analytical workflows, not new renderers.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| PC01 | P1 N/L | 3D neighborhood covariance, normals and eigenvalue shape features | Radius/kNN indexing, segmented moments and symmetric 3×3 eigensolve | Unit normals, orientation policy, degeneracy and neighborhood truncation against [PDAL normal/covariancefeatures][pdal]. Existing planar neighbor APIs need an explicit 3D adapter. |
| PC02 | P1 N/M | Voxel reduction and statistical/radius outlier filtering | 3D key sort/group plus PC01 neighborhoods | Representative/centroid choice, attribute aggregation, original IDs and partition halos against [PDAL voxel filters/outlier][pdal]. Sampling and averaging are separate outputs. |
| PC03 | P1 R/M | Height above ground and DTM/DSM/CHM construction | Classified ground interpolation or raster sampling, vertical differences and cell reductions | Preserve vertical datum, ground coverage and negative heights; [PDAL hag_dem/hag_nn/hag_delaunay][pdal]. Select one method first; expose interpolation provenance. |
| PC04 | P2 N/L | Progressive morphological ground classification | Multi-scale raster morphology and point-to-ground residual tests | Match [PDAL SMRF][pdal] parameters and class codes; retain original classifications, edge support and slope sensitivity. Do not equate the output with universal ground truth. |
| PC05 | P3 N/L | Point-cloud registration with bounded ICP | Repeated nearest matches, robust rejection and rigid-transform solve | Initial transform, metric, inlier masks and residual/convergence history against [PDAL ICP][pdal]. Batch independent tiles/scans; local optimum is not global alignment. |
| PC06 | P3 N/L | M3C2-style surface change and confidence intervals | PC01 normals, cylindrical neighborhoods and paired reductions | Signed normal distances, roughness, registration error and sample support against [PDAL m3c2][pdal]. Require PC05 or an explicit supplied alignment. |

### Interpolation and geostatistical prediction

Owner: `gpu-spatial-analysis`, pair estimation in `gpu-dataframe`, small solves in `analysis-core`.
Start from [ordinary kriging](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging.ts)
and [pair-statistics exports](../../modules/graph/src/gpu-dataframe/pair-statistics/index.ts).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| IN01 | P1 E/M | GPU-resident variogram model fitting and model-selection diagnostics | Bounded candidate fits over existing empirical bins; shared optimization; current fitVariogramModel is an explicit CPU helper | Preserve existing spherical/exponential/Gaussian and Cressie/pair/unweighted conventions; constrain admissible parameters, expose residual/convergence evidence and benchmark whether avoiding readback benefits repeated fits. [scikit-gstat Variogram.fit][variogram], [GSTools][gstools]. |
| IN02 | P1 E/L | Simple, universal and external-drift kriging | Extend neighborhood solves with known means or drift design matrices | Coefficients, prediction variance, rank failure and missing drift against [PyKrige][pykrige]/[GSTools Krige][gstools-krige]. Each mode needs its own contract. |
| IN03 | P2 E/L | Measurement-error kriging and block-support predictions | Observation-error diagonal plus explicit support quadrature | Separate smoothing nugget from process nugget; distinguish point and block variance. [GSTools cond_err][gstools-krige], [gstat block kriging][gstat-krige]. Report quadrature resolution. |
| IN04 | P3 N/XL | Co-kriging with a valid coregionalization model | Reuse cross-variograms and bounded block covariance solves | Start with supplied positive-semidefinite model; variable masks, singular systems and cross-covariance validity against [gstat fit.lmc/predict][gstat]. Cross-variograms alone do not supply a valid model. |
| IN05 | P2 N/L | Local RBF/thin-plate interpolation | Neighbor queries plus reusable batched dense solves | Kernel, polynomial degree, smoothing, extrapolation and rank contracts against [SciPy RBFInterpolator][rbf]. Bound neighbors; avoid global quadratic storage. |
| IN06 | P3 N/XL | Conditional Gaussian spatial simulation | Shared covariance model, bounded factorization or spectral simulation, then conditioning | Reproducible random stream and ensemble mean/covariance evidence against [GSTools conditioned fields][gstools]. Match distributions, not independent-library random sequences. |
| IN07 | P2 N/L | 2D natural-neighbor interpolation with explicit hull behavior | Reuse Delaunay/Voronoi topology; bounded query cavities and stolen-area weights | [CGAL natural_neighbor_coordinates_2 and linear_interpolation][natural-neighbor]: nonnegative weights, partition of unity, linear precision and sample reproduction. Define duplicates, collinear inputs, hull boundaries and outside-hull invalidity; nearest-triangle interpolation is not equivalent. |

### Spatial statistics and models

Owner: `gpu-spatial-analysis`; reuse dataframe statistics and solver primitives. Baselines are
[regression exports](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/index.ts),
[local statistics](../../modules/graph/src/gpu-spatial-analysis/spatial-autocorrelation/index.ts)
and [distribution dynamics](../../modules/graph/src/gpu-spatial-analysis/distribution-dynamics).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| ST01 | P1 E/L | Heteroskedastic and spatial-HAC covariance for spatial 2SLS | Sandwich matrix products over residuals and supplied kernel weights | Select HC convention, finite-sample correction and HAC weight requirements; [spreg TSLS/GM_Lag][spreg]. Existing conditioning diagnostics do not substitute for robust inference. |
| ST02 | P2 E/L | Regime restrictions, coefficient covariance and Chow tests | Restricted/unrestricted designs and shared small solves | Retain current fully varying OLS; add common versus regime coefficients and joint tests against [spreg OLS_Regimes][spreg]. Singular/empty regimes must remain explicit. |
| ST03 | P2 E/XL | Spatial panel ML/random effects and panel diagnostics | Time-block sparse operations and bounded likelihood optimization | Select one estimator first: [spreg Panel_FE_Error/Panel_RE_Lag and panel tests][spreg]. Keep existing unit-effects 2SLS distinct; clustered covariance is a separate subtask. |
| ST04 | P3 N/XL | Seemingly unrelated regression and spatial SUR | Cross-equation residual covariance and block solves | Start with nonspatial SUR, then one spatial variant; align observation IDs and masks. [spreg SUR/SURerrorGM][spreg]. No reuse of single-equation errors as joint covariance. |
| ST05 | P2 N/L | Probit with spatial residual diagnostics | Batched likelihood derivatives and bounded optimization | [spreg Probit][spreg] is classic nonspatial probit with spatial diagnostics, not a spatial-lag latent probit model. Match likelihood, marginal effects and diagnostic assumptions. |
| ST06 | P1 E/L | GWR collinearity surfaces, bandwidth selection and model-specific inference | Reuse local solves; expose VIF/variance decomposition and bounded candidate scoring | Match [mgwr local_collinearity/selection/inference][mgwr]; adaptive generalized GWR already exists. Full MGWR backfit uncertainty requires additional evidence beyond conditional submodel errors. |
| ST07 | P2 E/L | Joint multivariate local randomization and classifications | Reuse permutation replay with common row masks and multivariate statistic adapters | [esda Geary_Local_MV][esda], checked against [spdep][spdep] where semantics overlap. Define what is permuted and correction families; preserve LOSH's existing separate inference. |
| ST08 | P2 E/M | Full-rank/geographic-rank Markov and mobility summaries | Segmented ranking, transition tables and existing Markov diagnostics | Rank ties, missing panels, irreducibility and stationary-state assumptions against [giddy FullRank_Markov/GeoRank_Markov][giddy]. Do not duplicate current class/LISA transitions. |
| ST09 | P1 E/M | Fuzzy contiguity and explicit island attachment | Reuse indexed distance/predicate joins and weight builders | Match [libpysal fuzzy_contiguity/attach_islands][libpysal] under explicit buffer/distance rules; current quantized vertex snapping is not distance-based tolerance. Preserve added-edge provenance, directedness and island policy. |
| ST10 | P1 R/L | Spatially separated cross-validation folds and leakage-safe model evaluation | Spatial block IDs, buffered neighbor exclusions and fold masks; explicit repeated-fit orchestration over GPU estimators | Start with [blockCV cv_spatial/cv_buffer][blockcv]. Keep preprocessing and bandwidth/variogram selection inside training folds; test separation, complete test coverage, empty folds and held-out metrics. Environmental clustering and NNDM are separately scoped modes; expose seed and fold provenance. |
| ST11 | P2 N/L | Calibrated gravity/spatial-interaction flow models | Sparse origin-destination reductions, batched likelihood derivatives and bounded balancing/optimization | Stage unconstrained, origin-constrained, destination-constrained and doubly constrained models against [PySAL spint][spint]. Report fitted coefficients, predictions, deviance, marginal residuals and convergence. Distinguish structural zeros from missing pairs; existing accessibility gravity scores do not fit observed flows. |

### Point processes, sampling and clustering

Owners: pair kernels in `gpu-dataframe`, spatial windows and recipes in `gpu-spatial-analysis`,
network-specific consumers in `gpu-network`. Reuse K/L/F/G/J, marked statistics and study windows.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| PP01 | P1 E/L | Intensity estimation and polygon-window edge corrections for K and pair correlation | Reuse KDE and marked pairs; add window-overlap/border support | Declare estimated versus supplied intensity, self-contribution and leave-one-out policy; [spatstat density.ppp/Kinhom/pcf][spatstat]. Existing rectangular translation correction is only one option. |
| PP02 | P2 R/L | Monte Carlo envelopes and global point-pattern tests | Bounded simulation batches feeding existing summary curves and reductions | Distinguish pointwise envelopes from a global test; preserve null model, fitted-parameter handling, simulation counts and p-value resolution. [spatstat envelope/dclf.test][spatstat]. |
| PP03 | P2 E/L | Network kernel density and inhomogeneous network point patterns | Bounded network-distance neighborhoods and edge-aware integration | Edge-length normalization, junction correction and disconnected support against [spatstat linear-network tools][spatstat]/[spaghetti][spaghetti]. Reuse existing network K rather than re-add it. |
| PP04 | P3 N/L | OPTICS reachability ordering and cluster extraction | Reuse neighbor/core-distance machinery; bounded frontier processing | Ordering, predecessor/reachability outputs and chosen xi/DBSCAN extraction against [scikit-learn OPTICS][optics]. Demonstrate benefit despite sequential expansion; HDBSCAN is already implemented. |
| PP05 | P1 N/L | Spatially balanced probability sampling with design weights | Hierarchical spatial ordering with randomized traversal and bounded selection; reuse cell keys and scans | Begin with finite-point equal-probability [spsurvey grts][grts], then strata and unequal inclusion probabilities. Return frame IDs, inclusion probabilities, weights and random-stream provenance. Validate empirical inclusion frequencies and spatial balance, not identical random draws. Continuous line/area frames, legacy sites, distance constraints and replacements need separate contracts. |

### Networks and allocation

Owner: `gpu-network`, with generic flow/graph primitives below recipes and facility objectives in
spatial allocation. Reuse [network exports](../../modules/graph/src/gpu-network/index.ts).

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| NW01 | P1 R/L | Route between points on edges while preserving turn state | Virtual split-node/edge adapter over snapping, line graph and cost matrix | Partial-edge costs, same-edge travel, direction, parallel edges and provenance. [pgRouting withPoints][pgrouting]; pin the accessible source/manual before oracle implementation. |
| NW02 | P2 N/L | Maximum flow, minimum cut and edge-disjoint route capacity | Residual graph and bounded push/relabel or augmenting passes | Feasible flows, capacity limits, conservation and cut certificate against [NetworkX flow][network-flow]. Distinct from shortest paths and observed flow aggregation. |
| NW03 | P2 N/XL | Minimum-cost flow and capacity-constrained demand assignment | Residual costs plus feasibility checks; reuse NW02 | Match exact small cases with [NetworkX min_cost_flow/network_simplex][network-flow]; expose unserved demand, cost and optimality evidence. A budget-limited heuristic cannot certify infeasibility. |
| NW04 | P2 E/L | Louvain modularity communities | Graph coarsening and bounded local-move rounds | Weighted/directed/self-loop semantics, seed and objective trace against [NetworkX louvain_communities][communities]. Existing label propagation is not modularity optimization; compare objective and partition invariants. |
| NW05 | P2 E/L | Backup coverage, p-dispersion and certified capacitated facility cases | Reuse cost tables and bounded objective evaluation; small exact reference cases | Separate [spopt LSCPB/PDispersion][spopt] objectives and capacity constraints. Return feasibility, objective and bound/gap when available; existing p-center and greedy capacity allocation remain distinct. |
| NW06 | P3 N/XL | Departure-time routing and multi-edge turn restrictions | Expanded routing state, time-dependent relaxation and bounded labels | Begin with FIFO costs and explicit waiting policy; use [Valhalla routing semantics][valhalla] plus an independent tiny time-expanded graph oracle. Existing per-frame static costs do not evolve during a route. |
| NW07 | P2 N/L | Rational-agent access with congestion-sensitive allocation | Reuse origin/facility costs and bounded demand-shift iterations | Travel/congestion scales, supply normalization, unreachable demand and convergence against [access.Access.raam][raam]. Preserve demand totals; existing gravity and floating catchment scores do not model this iterative assignment. |

### Movement and temporal geometry

Owner: spatial trajectory primitives; recipes needing routing belong in `gpu-network`.
Reuse existing int64 timestamp-word support, track offsets, event crossings and closest approach.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| MV01 | P1 N/M | Constant-velocity Kalman filtering/smoothing with uncertainty | Small state solves, parallel across tracks, forward/backward time passes | Explicit process/measurement noise, gap resets, covariance and projection; [MovingPandas KalmanSmootherCV][kalman]. Keep raw observations and filtered estimates distinct. |
| MV02 | P1 R/L | Restrict tracks to time/geometry and return ever/always relationship intervals | Compose crossing segments, time clipping and compact interval algebra | Endpoint inclusion, interpolation model, gaps and zero common support against [MobilityDB temporal geometry][mobilitydb]. Existing zone events/closest approach supply pieces but not the general interval API. |
| MV03 | P2 E/L | Pruned and batched DTW, Fréchet and LCSS search at useful scale | Reuse existing DTW/LCSS bands; add admissible lower bounds, query batching and pruning | Existing bands are not new work. Preserve each metric's contract and exact/approximate status using [tslearn][tslearn] and [dtaidistance][dtaidistance]. Never prune a valid match with an unproved bound. |

### Urban morphology and change of support

Owner: spatial recipes with explicit network/raster dependencies at a composition boundary.
Reuse polygon topology, nearest features, building attributes and grouped statistics.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| UR01 | P2 N/L | Building-based morphological/enclosed tessellation | Boundary sampling, Voronoi construction and clipping to barriers | Sampling tolerance, enclosures, slivers and stable building IDs against [momepy morphological_tessellation/enclosed_tessellation][momepy]. Point Voronoi alone is not building tessellation. |
| UR02 | P1 R/M | Street profiles, frontage and street-canyon measures | Sample cross-sections and intersect building boundaries/heights | Width/open-space measures, side assignment, absent buildings and sample spacing against [momepy street_profile][momepy]. Explicit planar/height units. |
| UR03 | P1 R/M | Building adjacency, shared-wall length and orientation alignment | Reuse shared paths, contiguity and grouped measures | Touching at a point versus a shared wall; angular wrap and missing heights. [momepy perimeter_wall/orientation/alignment][momepy]. Audit existing shape descriptors before adding scalar formulas. |
| UR04 | P1 R/M | Ancillary-raster dasymetric redistribution with conservation | Compose masks, fractional coverage and existing weighted areal interpolation | Source totals, excluded land, zero eligible area and target coverage against [tobler masked_area_interpolate][tobler]. Weighted interpolation already exists; this adds the complete provenance-preserving recipe. |
| UR05 | P2 N/L | 2D obstacle isovists and sampled visibility graphs | Indexed ray/segment intersections and angular ordering; bounded visible-pair graph plus existing network analytics | Compare [depthmapX isovists/VGA][depthmapx]. Declare obstacle/range boundaries, viewpoint-on-edge behavior, angular approximation and sampling resolution; report graph capacity exhaustion. Isovist polygons, graph connectivity and integration/choice metrics are separate deliverables; terrain viewsheds do not establish these semantics. |

### Landscape ecology and spatial equity

Owners: raster patch metrics, spatial statistics, and explicit cross-domain recipes. Reuse
components, distance fields, sparse solvers, segregation and accessibility.

| ID | Priority/type/size | Proposed function and use | GPU approach and dependencies | Completion contract and reference |
| --- | --- | --- | --- | --- |
| EC01 | P1 E/M | Class/landscape aggregation, contagion, cohesion and effective mesh size | Extend patch outputs with adjacency counts and class/landscape reductions | Adopt named [landscapemetrics][landscape] definitions; specify 4/8 connectivity, border/edge accounting and background. Existing per-patch geometry summaries remain reusable. |
| EC02 | P2 E/M | Patch core area, nearest-patch distance and proximity metrics | Distance fields, edge erosion and nearest-feature queries | Exclude same-patch hits; physical edge depth, search radius and nodata barriers against [landscapemetrics][landscape]. Compute across tile seams before summarizing. |
| EC03 | P3 N/XL | Effective resistance and current-density connectivity maps | Conductance graph, grounded Laplacian and existing sparse iterative solver | Residual tolerance, disconnected components, conductance/resistance conversion and current conservation against [Circuitscape][circuitscape]. No inference of equivalence to least-cost paths. |
| EC04 | P2 R/L | Segregation change decomposition and counterfactual comparison | Reuse segregation at common support plus explicit counterfactual population tables | Separate composition/spatial components and randomization assumptions against [PySAL DecomposeSegregation/inference][segregation]. Shared boundaries and population totals must be aligned first. |

## Delivery sequence and dependencies

The original 78 rows comprise 8 geometry, 5 global/cell, 11 raster, 6 terrain, 6 point-cloud, 7 interpolation,
11 statistics/weights, 5 point-process/sampling, 7 network, 3 movement, 5 urban and 4 ecology work items. Rows
can contain related modes, but completion must record each supported mode separately.
KG01–KG06 add six follow-up items, for 84 backlog IDs in total. Five bounded implementations and
the partial SQL extension are covered by the [Kepler implementation record](./kepler-analysis-implementation.md),
not the historical implementation tranches below.

| Tranche | Candidate slice | Useful demonstration and exit evidence |
| --- | --- | --- |
| 0 | Existing validation closure; baseline capability ledger | Follow [expansion follow up](./geospatial-toolkit-expansion-follow-up.md). Record frozen-tree tests and performance envelopes before claiming wider readiness. |
| 1 | RS01–RS02, RS05, RS09, UR04; GE01, ST10, PP05 | Fractional population/exposure summaries, reproducible geometry, spatially separated model evaluation and balanced monitoring samples. Require conservation, leakage checks, sampling-design evidence and repeated-scenario benchmarks. |
| 2 | RS04, RS06, RS11, TE01, TE04, UR02–UR03, MV01 | Land-cover texture and seasonal disturbance, prepared drainage and urban/track metrics. Require window/halo, breakpoint and provenance evidence; masks stay visible throughout recipes. |
| 3 | PC01–PC03, IN01–IN02, ST01, ST06, NW01 | Point-cloud surface analysis, prediction and routing/model precision. Shared neighborhood/solver contracts and small-to-large workload curves. |
| 4 | Select P2 tasks by consumer: GE02–GE03, RS03/RS07/RS08, TE02/TE06, ST02/ST11, IN07, UR05, NW02–NW05 | Advance topology, remapping and cross-domain analysis only after their bounded output and convergence contracts pass. P1 means value, not an automatic dependency order. |
| 5 | Select individual P3 tasks | Prototype spherical overlay, straight skeletons, conditional simulation, spectral unmixing, time-dependent routing or circuit connectivity against an explicit application. Stop if browser cost/memory cannot meet the use case. |

Key dependencies: topology/precision → GE02–GE08 and CR04; 3D neighborhoods → PC01–PC06;
fractional tuples and weighted reductions → RS01–RS03/UR04; stable solver diagnostics →
IN02–IN06/ST01–ST06; residual graph infrastructure → NW02–NW03; interval algebra → MV02;
cross-tile equivalence → TE06/EC02; fold-safe model fitting → ST10; temporal solves → RS11;
Delaunay query cavities → IN07; probability-design metadata → PP05; sparse OD balancing → ST11;
visibility adjacency → UR05. Reuse existing sparse solvers for EC03 before inventing another
solver abstraction. Cross-domain recipes must sit above their dependencies and avoid import cycles.

## Acceptance gates for every addition

1. **Exact scope.** Record the chosen external function/version, supported options, coordinate and
   temporal models, null/boundary/tie rules, units and parameter limits. Distinguish an exact bounded
   result, a numerical approximation, a heuristic objective and incomplete output.
2. **GPU contract.** Specify input/output shape, source IDs, capacities, required counts and status
   ports. No hidden CPU fallback, submission, readback or scheduler. Respect default WebGPU binding
   limits. Dynamic thresholds/masks/costs belong in data views where practical.
3. **Numerical and semantic evidence.** Commit version-pinned reference generators and fixtures;
   label direct upstream calls versus composed references. Add independent invariants such as
   area/mass conservation, topology validity, flow feasibility or covariance behavior. Two wrappers
   over GEOS, PROJ or ESMF are not independent algorithmic confirmation.
4. **Adversarial cases.** Empty, null, tied, duplicate and non-finite data; degeneracy and singular
   systems; tiny and insufficient capacities; forced iteration exhaustion. Global geometry adds
   poles/antimeridian/antipodes; temporal operations add gaps and coincident endpoints. Preserve
   unknown results instead of mapping them to false, zero exposure or unreachable.
5. **Partition and lifecycle evidence.** Preserve chunks and source-batch boundaries; packing stays
   explicit. Verify whole-domain versus tiled results, halo exclusion, prepared-input invalidation
   and ownership. `GPUData` owns/borrows one buffer, vectors hold ordered chunks, and Arrow adapters
   stay in `@luma.gl/arrow`.
6. **Measured value.** Report compile, upload, execution, optional readback, peak scratch memory and
   repeated-update costs. Benchmark small inputs where CPU execution can win and representative
   skew/large domains. Publish numerical error and supported workload/device envelopes alongside
   throughput. CUDA/cpu timings do not establish WebGPU performance.
7. **Integration.** Add documentation, typed ports, public exports, discovery descriptors and an
   executable consumer only when implemented. Record semantic/fixture/GPU/scaling evidence
   separately; a constructor test does not validate numerical execution.

Before implementation promotion, follow AGENTS.md: `nvm use`, dependencies as needed, and
`yarn lint fix` after edits. Merge preparation requires final `yarn build` and `yarn test`; shared
table/API changes also require `yarn test-node`. Record `yarn install`, `yarn website:build` and
`(cd website && yarn build)` outcomes as applicable. This documentation research does not rerun or
supersede the prior expansion's test record.

## Keep outside this function backlog

- File formats, cloud catalogues, databases, geocoding, routing services and remote data acquisition
  belong in adapters/services. This backlog accepts already decoded, explicitly prepared inputs.
- General atmospheric/radiometric calibration, sensor RPC orthorectification, transit timetable
  routing, hydraulic flood simulation and full 3D solid/mesh Boolean processing need dedicated
  domain roadmaps. Related primitive kernels here do not establish those end-to-end capabilities.
- A full GPU clone of GEOS, PROJ, PySAL or a GIS desktop is not the coverage target. Coverage means
  useful workflows with explicit semantics and measured envelopes, including documented exclusions.
- Adding a reference does not authorize copying its implementation or changing first-party
  licensing. Preserve attribution and review the actual source license if reuse is proposed.

## Maintaining the coverage ledger

Maintain the [function coverage matrix](./geospatial-function-coverage.md) alongside these IDs.
When taking an ID, attach its consumer, API design, selected reference/version, prerequisite IDs and
expected capacities. On delivery, link implementation and fixtures here and mark individual modes
implemented, limited or deferred. Keep correctness, inference and scale evidence separate. Reconcile
the public cross-reference only against actual exports; retain unavailable functions here rather
than advertising them in the runtime catalogue. Refresh moving upstream documentation links during
implementation, especially development manuals and pgRouting pages unavailable during this review.

[shapely]: https://shapely.readthedocs.io/en/stable/_reference.html
[geopandas]: https://geopandas.org/en/stable/docs/reference.html
[sf]: https://r-spatial.github.io/sf/reference/index.html
[postgis]: https://postgis.net/docs/reference.html
[rust-geo]: https://docs.rs/geo/latest/geo/
[sedona]: https://sedona.apache.org/latest/api/sql/Geometry-Functions/
[turf]: https://turfjs.org/docs/api/area
[qgis]: https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/index.html
[arcgis]: https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/an-overview-of-the-spatial-analyst-toolbox.html
[carto]: https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery
[esda]: https://pysal.org/esda/stable/api.html
[spreg]: https://pysal.org/spreg/api.html
[mgwr]: https://github.com/pysal/mgwr/blob/main/mgwr/gwr.py
[spopt]: https://pysal.org/spopt/api.html
[libpysal]: https://pysal.org/libpysal/stable/api.html
[mapclassify]: https://pysal.org/mapclassify/api.html
[access]: https://pysal.org/access/api.html
[raam]: https://pysal.org/access/generated/access.Access.raam.html
[pointpats]: https://pysal.org/pointpats/api.html
[tobler]: https://pysal.org/tobler/stable/api.html
[segregation]: https://pysal.org/segregation/api.html
[giddy]: https://pysal.org/giddy/api.html
[rbf]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
[savgol]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
[optics]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
[variogram]: https://scikit-gstat.readthedocs.io/en/latest/reference/variogram.html
[rasterio]: https://rasterio.readthedocs.io/en/stable/topics/resampling.html
[texture]: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
[image-graph]: https://scikit-image.org/docs/stable/api/skimage.graph.html
[registration]: https://scikit-image.org/docs/stable/api/skimage.registration.html
[grass]: https://grass.osgeo.org/grass-stable/manuals/raster.html
[pysheds]: https://github.com/pysheds/pysheds
[xdem]: https://xdem.readthedocs.io/en/stable/coregistration.html
[rvt]: https://rvt-py.readthedocs.io/en/latest/
[pvlib]: https://pvlib-python.readthedocs.io/en/stable/reference/index.html
[network-flow]: https://networkx.org/documentation/stable/reference/algorithms/flow.html
[communities]: https://networkx.org/documentation/stable/reference/algorithms/community.html
[spaghetti]: https://pysal.org/spaghetti/api.html
[h3]: https://h3geo.org/docs/api/regions/
[hgrid]: https://proj.org/en/stable/operations/transformations/hgridshift.html
[vgrid]: https://proj.org/en/stable/operations/transformations/vgridshift.html
[pygeodesy]: https://github.com/mrJean1/PyGeodesy
[spatstat]: https://spatstat.org/resources/spatstatQuickref.pdf
[tslearn]: https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
[dtaidistance]: https://dtaidistance.readthedocs.io/en/latest/modules/dtw.html
[similaritymeasures]: https://github.com/cjekel/similarity_measures
[cgal]: https://doc.cgal.org/latest/Manual/packages.html
[s2]: https://s2geometry.io/about/overview
[s2-cells]: https://s2geometry.io/devguide/s2cell_hierarchy.html
[s2-validation]: https://github.com/google/s2geometry/blob/master/doc/design/index_validation.md
[geographiclib]: https://geographiclib.sourceforge.io/
[exactextract]: https://isciences.github.io/exactextract/operations.html
[terra]: https://rspatial.github.io/terra/reference/index.html
[xesmf]: https://xesmf.readthedocs.io/en/stable/notebooks/Compare_algorithms.html
[whitebox]: https://jblindsay.github.io/wbt_book/available_tools/hydrological_analysis.html
[saga]: https://saga-gis.sourceforge.io/saga_tool_doc/9.9.0/index.html
[pdal]: https://pdal.io/en/latest/stages/filters.html
[gstools]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/
[gstools-krige]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.krige.Krige.html
[pykrige]: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/
[gstat]: https://r-spatial.github.io/gstat/reference/index.html
[gstat-krige]: https://r-spatial.github.io/gstat/reference/krige.html
[spdep]: https://r-spatial.github.io/spdep/reference/index.html
[otb]: https://www.orfeo-toolbox.org/CookBook/Applications.html
[pgrouting]: https://docs.pgrouting.org/latest/en/migration.html
[valhalla]: https://github.com/valhalla/valhalla/blob/master/docs/docs/api/route/api-reference.md
[movingpandas]: https://movingpandas.readthedocs.io/en/main/api/api.html
[kalman]: https://movingpandas.readthedocs.io/en/main/api/trajectorysmoother.html
[mobilitydb]: https://mobilitydb.github.io/MobilityDB/master/ch08.html
[momepy]: https://docs.momepy.org/stable/api.html
[landscape]: https://r-spatialecology.github.io/landscapemetrics/reference/index.html
[circuitscape]: https://docs.circuitscape.org/Circuitscape.jl/latest/
[cuspatial]: https://docs.rapids.ai/api/cuspatial/stable/
[normalize]: https://shapely.readthedocs.io/en/stable/reference/shapely.normalize.html
[subdivide]: https://postgis.net/docs/ST_Subdivide.html
[precision]: https://shapely.readthedocs.io/en/stable/reference/shapely.set_precision.html
[snap]: https://shapely.readthedocs.io/en/stable/reference/shapely.snap.html
[inscribed]: https://shapely.readthedocs.io/en/stable/reference/shapely.maximum_inscribed_circle.html

[blockcv]: https://github.com/rvalavi/blockCV
[grts]: https://usepa.github.io/spsurvey/reference/grts.html
[spint]: https://github.com/pysal/spint
[bfast]: https://search.r-project.org/CRAN/refmans/bfast/html/bfast.html
[natural-neighbor]: https://doc.cgal.org/latest/Interpolation/index.html
[depthmapx]: https://github.com/SpaceGroupUCL/depthmapX
