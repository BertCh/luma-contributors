> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/geospatial-coverage-parallel-implementation.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Implementation complete; remaining items are tracked in the roadmap.

# Geospatial coverage: parallel implementation record

Implementation review: October 23, 2026. Baseline: the shared working tree, including uncommitted
parallel implementation. Ten implementation streams added scoped GPU behavior across the
[78-item backlog](./geospatial-coverage-backlog.md). This record maps **all 78 IDs** to actual
implementation paths, bounded contracts and their executed evidence. Advanced modes, direct
reference comparisons and measured scaling are recorded separately; implementation presence does
not establish complete upstream-library parity or released availability.

The [first tranche](./geospatial-coverage-tranche-one.md) already supplied RS01, RS02, RS05, RS09
and UR04; those implementations are retained rather than counted as newly built. CR01 is an
extension of failure handling and numerical evidence in existing measurement code. The other
rows identify actual bounded additions or extensions, including deliberately narrow first models.

## Status and evidence policy

Each table records **implementation presence and scope**, independently of executed validation.
The [function matrix](./geospatial-function-coverage.md) tracks each named option independently,
including advanced seasonal, statistical, validation and sampling modes. A row is promoted only
with its own implementation and evidence; bounded capacities and explicitly unsupported numerical
models remain visible in the contract. Source links below point to implementations;
the linked domain pages define layouts, capacities, statuses, approximation and ownership.

Kernels and recipes retain graph-owned execution with explicit submission/readback. The routing
restriction-state helper is explicit CPU topology preparation, not a hidden route solver. Tests and
oracles must distinguish direct upstream execution, independent composed references, analytical
invariants and shader-structure checks. None substitutes for final repository gates, complete-domain
seam tests, production-scale benchmarks or cross-device evidence.

## Geometry and topology

[API contracts](../../docs/api-reference/experimental/gpu-geometry-refinement.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| GE01 | [`GPUNormalizeGeometry`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry.ts), [`GPUCanonicalGeometryEquality`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality.ts), polygon-specialized variants | Postorder points, open/closed lines, polygons, multis and nested/empty collections; stable source IDs, GEOS child ordering, explicit equality tolerance and idempotence. Direct Shapely tree fixtures and executable lenses pass. | Nesting is bounded at 64; normalization assumes disjoint leaf ranges and valid polygon ring structure. No universal cross-device numerical certificate. |
| GE02 | [`GPUSubdividePolygons`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-subdivide-polygons.ts), [`rectilinear queue`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/rectilinear-subdivision.ts) | Hole-aware convex seeds and bounded axis-aligned clipping; vertex limits, source provenance and work/capacity status. Twelve direct PostgreSQL 16.6/PostGIS 3.5.3 ST_Subdivide cases at limits 5/9/17 pass validity, exact union and 841 membership checks each; GPU multi-feature unions match all cases. At 16/64/256 polygons, GPU join candidates halve, with timings recording both improvement and regression. | Seed decomposition and piece ordering differ from ST_Subdivide; union/provenance and per-piece limits are the verified contract. Engine pieces omit closing duplicates (GPU limits 4/8/16 correspond to PostGIS 5/9/17). Candidate reduction is not a universal speedup. |
| GE03 | [`GPULinePrecisionReduction`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction.ts), `GPUPolygonPrecisionReduction` | Arbitrary positive float32 grid steps, normalized-grid snap-rounded noding and polygon reconstruction, valid-output/keep-collapsed policies, repeated/empty behavior and idempotent composition. Direct Shapely fixtures include grids 0.1/0.3/2.5, holes and full collapse. | Actual lattice remains explicit if representable extent requires a coarser step. Double-precision identity grid 0 is outside this float32 contract. |
| GE04 | [`GPUSegmentSnap`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap.ts) | Nearest-reference endpoint snapping, stable row-order ties, duplicate-aware ordered insertion, source edges and explicit collapse/nonfinite/incomplete validity bits; direct Shapely supported-case fixture and adversarial tie checks. | GEOS safety heuristics are deliberately a different policy; neither policy promises universal topology preservation. |
| GE05 | [`GPUOffsetCurveComponents`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components.ts), [`GPULineLoopRemoval`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal.ts) | Node raw offsets, analytically clip source distance bands, cancel collapsed spurs and compact directed multipart components with source provenance. All 36 direct Shapely signed mitre/round/bevel cases pass component, length, bidirectional distance and orientation checks; incomplete components are not published. | Raw offsets must align with the signed source-line generation contract; chord-error tolerance and bounded noding/fragment capacities are explicit. Large component assembly remains a scan-based algorithm. |
| GE06 | [`GPURatioConcaveHull`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull.ts) | Delaunay border erosion with separate edge-ratio and hole policies; 42 direct Shapely endpoint/intermediate, duplicate, point, collinear and separated-cluster cases pass area checks, plus existing hole fixtures. | Returns area boundaries: point/collinear results have no area edges. Triangle-ID tie breaking is explicit; identical cross-engine triangulation ordering is not promised. |
| GE07 | [`GPUMaximumInscribedCircle`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle.ts) | Bounded best-first search, nearest boundary witness, remaining geometric search error and distinct success/exhaustion states. Direct Shapely concave/hole/thin-polygon radius bounds and exhausted budgets pass; 16/64/256-feature complete GPU batches are measured. | The bound excludes float32 arithmetic error; interval-arithmetic certification was not the geometric-search contract. Device timings are observations, not universal limits. |
| GE08 | [`GPUVectorStraightSkeleton`](../../modules/graph/src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton.ts), convex-specialized variant | Linked event wavefronts handle reflex splits, hole merges, multiple holes, simultaneous ridges and collinear corners. All arcs and inward-offset vertices match ten direct CGAL 6.0.1 fixtures; complete concave batches of 16/64/256 polygons pass. | Valid planar polygons, maximum 1024 vertices/4096 events per polygon, explicit input/work/capacity/numerical flags. Float32 event grouping is not CGAL exact-arithmetic equivalence for every near-degenerate input. |

## Global coordinates and cells

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/global-coordinates.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| CR01 | [`GPUGeometryMeasures`](../../modules/graph/src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures.ts) | Safeguarded auxiliary-longitude inverse and adaptive per-edge area integration with explicit invalid/exhaustion reporting. Direct GeographicLib: 256 difficult/global pairs and 12 holed polygons, serial/cooperative paths. | f32 validated envelopes are not certified bounds; unresolved singular inverse roots and pole-winding ellipsoidal polygon areas explicitly remain invalid. |
| CR02 | [`GPUDatumGridTransform`](../../modules/graph/src/gpu-project/gpu-datum-grid-transform.ts) | Prepared regular horizontal grids, bilinear priority/nodata/bounds, target-seeded inverse, optional high/low coordinates. Direct PROJ CTable2 forward/inverse fixtures agree within 1e-9 degrees, including priority discontinuity and outer-boundary failure. | Acquisition/CRS database/epoch/subgrid selection are external preparation in the original backlog. Samples remain f32; no general PROJ pipeline or certified float64 claim. |
| CR03 | [`GPUDatumGridTransform`](../../modules/graph/src/gpu-project/gpu-datum-grid-transform.ts) | Signed vertical/combined corrections, explicit references and height units, high/low height residuals, direct PROJ GTX comparison and combined inverse tests. | Named geoid/epoch acquisition and full PROJ pipeline construction stay external; f32 grid samples constrain accuracy. |
| CR04 | [`GPUSphericalPolygonContains`](../../modules/graph/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains.ts), [`GPUSphericalPolygonOverlay`](../../modules/graph/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay.ts), [`GPUSphericalArcOverlay`](../../modules/graph/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts) | Nonconvex shell/hole predicates; all four Boolean operations via chart or intrinsic global arc arrangement; complements, assembled rings, boundary policy. Direct S2: 1,032 classifications, holed chart overlays and 12 non-hemisphere global/complement overlays. | Tolerance-based f32 predicates, explicit uncertain topology; intrinsic input ≤128 vertices/output ≤1024. Prepared ring nesting and anchor membership are caller contracts; exact S2 predicates and unbounded production topology are not claimed. |
| CR05 | [`GPUS2CellHierarchy`](../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy.ts), [`GPUS2CellMeasures`](../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures.ts), [`GPUS2PolygonCovering`](../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering.ts) | Exact hierarchy/neighbors and stable area/perimeter through level 30; adaptive mixed-level interior/exterior polygon covers through 15, uniform cap covers and prepared-boundary geometric cross-family overlap weights. Direct S2 IDs/measures and level-5 cover guarantees; measured 166/16384-row and level-4/6 workloads. | Tolerance-aware cover predicates; adaptive cover is bounded DFS, not S2 priority cover minimization. Automatic cross-family acquisition/broad phase and optimized CPU timing comparison remain integration/performance extensions. |

CR01–CR05 round-two evidence: 23 focused GPU runtime cases plus nine executable lenses passed;
35 focused node cases include 24 maximum-option kernel audits, nine lens graphs and two
structure/alias checks. Four new operators have domain descriptors, public examples and lazy
capture scenarios. The [domain contract](../../docs/api-reference/experimental/gpu-spatial-analysis/global-coordinates.md)
records direct GeographicLib 2.1, PROJ 9.5.1 / pyproj 3.7.2 and S2Geometry 0.14.0 generators,
thresholds and failure modes. The [14-workload measured record](../benchmarks/data/global-coordinates-expanded-2026-10-23.json)
separates upload, compile, first/warm/updated execution, validation/readback and explicit allocation.
It uses a shared dirty source tree with no frozen revision or optimized CPU baseline; no general
performance advantage or complete upstream-library parity follows from these measurements.

## Raster and remote sensing

[API contracts](../../docs/api-reference/experimental/gpu-raster/remote-sensing.md). Retained first-tranche rows keep their [separate evidence record](./geospatial-coverage-tranche-one.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| RS01 | [`GPURasterFractionalZonalDistribution`](../../modules/graph/src/gpu-raster/gpu-raster-fractional-zonal-distribution.ts) | Exact sorted continuous-value interpolation or inverse weighted CDF; independent observation weights and supplied per-cell areas; population/reliability variance and Kish support. Direct exactextract 0.3.0/NumPy references and 16,384-row nonbinary-weight scale tests; fixed partial reduction tree limits drift. | Explicit 65,536-tuple/32-probability bounds. Caller supplies geographic areas and complete tile tuple unions; no averaged tile quantiles or automatic geodesic-area calculation. |
| RS02 | [`GPURasterFractionalCategoricalStatistics`](../../modules/graph/src/gpu-raster/gpu-raster-fractional-categorical-statistics.ts) | Fractional class areas/proportions and stable majority/minority/missing support; direct exactextract parity. Capacity expanded to 65,534 classes within the 65,535-group budget; 32,768-class GPU conservation/tie test passes. | Dense numeric IDs are caller encoded; sparse/string discovery is not required by the original row. Allocation/group bounds remain explicit. |
| RS03 | [`GPURasterConservativeRemapping`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-conservative-remapping.ts) | Exact Cartesian overlap with three normalizations and caller-owned prepared weights reused across graphs. Direct ESMPy/ESMF 8.9.1 conservative parity passes nine coarsened/masked/partial/disjoint cases, each with two dates and weight reuse. Conservation and changing QA masks pass at 128², 256² and 512² source sizes. | Reuse emits only accumulation. Curvilinear/spherical grids and reprojection are explicitly later scopes in the original row. |
| RS04 | [`GPURasterTexture`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-texture.ts) | Masked directional/symmetric GLCMs and seven measures. Direct scikit-image 0.25.2 properties, three-direction halo/core parity, and 128² cells at 32 levels/radius 8 pass on GPU. | Directions are explicit separate calls. Direct OTB comparison unavailable on native Apple Silicon; no automatic direction averaging or hardware-independent throughput claim. |
| RS05 | [`GPURasterFocalPairs`](../../modules/graph/src/gpu-raster/focal-pairs/gpu-raster-focal-pairs.ts) | Paired covariance/correlation/local OLS, masks/calibration and singular statuses. Direct terra 1.9.50/R 4.4.3 focalPairs/focalReg comparisons pass for six boundary/NA modes; independent NumPy adversarial and halo tests retained. | Clamp/reflect parity uses explicit padded terra input, then core cropping. Fit statistics remain descriptive; inferential models were explicitly separate in the original row. |
| RS06 | [`GPURasterTemporal`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-temporal.ts) | Regular-time masked composites, original-date provenance and missing-data QR Savitzky–Golay with two edge policies. Direct terra composites and pinned SciPy smoothing comparisons pass. | Irregular dates require the explicit resampling/different model stated in the original row. No implicit resampling or unspecified arbitrary quality ranking. |
| RS07 | [`GPURasterRegistration`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-registration.ts) | Periodic phase correlation/local DFT refinement and masked linear NCC/bilinear refinement, with ambiguity/failure/support. Direct scikit-image subpixel fixtures and independent masked SciPy NCC cases pass. | Mode objectives differ and precision is bounded by the selected refinement grid. Nonrigid registration is explicitly excluded by the original row; broad sensor-specific noise certification is not claimed. |
| RS08 | [`GPURasterRegionGraph`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-region-graph.ts), [`addRasterRegionGraphTiles`](../../modules/graph/src/gpu-raster/remote-sensing/raster-region-tiles.ts) | Fixed/adaptive mean-updating merges plus explicit GPU tile-core assembly using stable source equivalences. Direct scikit-image hierarchical partitions, four-core seam equivalence and 128²/64-region adjacency/convergence tests pass. | Caller supplies local-to-source equivalence IDs, e.g. existing cross-tile component output. Arbitrary callbacks/vector linkage are additional models; bounded stopping status remains explicit. |
| RS09 | [`GPURasterNDWI`](../../modules/graph/src/gpu-raster/gpu-raster-spectral-indices.ts), [`GPURasterNBR`](../../modules/graph/src/gpu-raster/gpu-raster-spectral-indices.ts), [`GPURasterEVI`](../../modules/graph/src/gpu-raster/gpu-raster-spectral-indices.ts), [`GPURasterSAVI`](../../modules/graph/src/gpu-raster/gpu-raster-spectral-indices.ts) | Supplied-calibration named indices, wavelengths/scaling and denominator masks; direct pinned Spyndex 0.12.0 formula fixtures reexecuted and GPU verified. | Direct OTB execution unavailable on native Apple Silicon. Automatic sensor selection and spectral-response equivalence remain outside supplied-band recipes. |
| RS10 | [`GPURasterSpectralUnmixing`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-spectral-unmixing.ts) | Supplied-endmember QR, NNLS and simplex-constrained solutions with residual/convergence evidence. All 1,280 abundances from the complete 256-pixel/224-band OTB 9.1.1 UCLS golden image match GPU results within `2e-5`; pinned NumPy/SciPy references cover constrained solutions. All three modes pass at both ends of the joint workspace bounds (up to 256 bands or eight endmembers), including changed signed inputs. | Published OTB baseline and original inputs are SHA256-verified at commit `154726b379fdb39b481ee8fd793fbd3d584884e6`; OTB was not executed locally. ISRA/MDMD-NMF equivalence, endmember discovery and atmospheric correction are not claimed. |
| RS11 | [`GPURasterSeasonalBreak`](../../modules/graph/src/gpu-raster/remote-sensing/gpu-raster-seasonal-break.ts) | Harmonic or dummy-season trends with one BIC level break; joint QR or bounded cyclic trend/season decomposition; fixed-history persistent residual monitoring and omit/reject/interpolate gap policies. Pinned independent NumPy references, converged/exhausted/rank/QA/future-isolation GPU cases, 8 cells ×128 dates/12 phases, and a real dummy-monitor lens pass. | Explicit 256-slot, 8 KiB private-workspace and iterative-work bounds. Monitoring is fixed-history and reports original alarm dates; no hidden stream state, multiple break-count search, calibrated p-values or full BFAST equivalence. |

## Terrain and hydrology

[API contracts](../../docs/api-reference/experimental/gpu-terrain/terrain-processing.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| TE01 | [`GPUTerrainBreach`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-breach.ts) | Bounded minimum summed-cut paths, explicit burning, provenance and post-edit conflict verification. Exact radius-S search windows plus atomic minimum proposals scale independently of the whole DEM search extent; 64 pits on 65,536 cells pass. Executed Whitebox 2.4.0 corridor comparison and overlapping-path adversarial case. | The named cut objective preserves the starting pit; Whitebox lowers it in the reference fixture. Corridor/invariants are compared explicitly, not general Whitebox elevation identity. Overlap conflicts remain visible and unresolved by contract. |
| TE02 | [`GPUTerrainFlowFractions`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts), [`GPUTerrainBranchingFlow`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts) | Existing D-infinity/Freeman/Quinn fractions feed atomic DAG transport, expected/min/max distance and fractional catchments. GPU-gated successor walks use one readiness/value word per cell. Executed pysheds 0.5 D8/MFD parity, 65,536-cell split/merge mass conservation and 16,384-cell forward/reverse chains pass. | Explicit sweep/walk budgets bound execution; cycles, unknown incoming fractions and exhausted dependencies retain NaN/status instead of renormalization. Completed values use fixed-order float32 sums; the scale mass error is about 1.55 ppm. |
| TE03 | [`GPUTerrainDepressionHierarchy`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts) | Stable GPU sort, compressed disjoint-set activation, nested ownership, explicit closed/outlet components and linear-per-stage area/volume reductions. Executed SciPy sublevel components independently verify topology-equivalent active basins. 16,384-cell hierarchy/volume checks pass with integer-significand summation correction. | The activation pass remains serial on the GPU; it no longer rescans/relabels the raster quadratically. Dynamic flood simulation is outside the row. Equal-height zero-persistence refinements are compared through active components. |
| TE04 | [`GPUTerrainCatchmentSummary`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts) | Sorted basin ranges replace basin-by-raster scans; pointer jumping replaces per-cell downstream walks while preserving maximumSteps semantics. Physical area/elevation distributions, density and profiles pass composed executed pysheds/NumPy references and 65,536-cell/256-basin checks. | Units are those of the supplied planar affine; geographic-degree conversion is caller preparation. The oracle is explicitly composed membership plus independent reductions, not whole-toolbox identity. |
| TE05 | [`GPUTerrainDEMCoregistration`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts) | GPU-sorted stable-ground median/NMAD matches executed xDEM 0.1.6 VerticalShift. Independent-error processing passes 16,384 cells. Parallel covariance rows match a full 1,024-cell sheared-grid heterogeneous-variance NumPy matrix before integrated volume uncertainty. | The chosen model is vertical shift, satisfying the named-model scope; horizontal/Nuth–Kääb/tilt models are additional models. Nonzero correlation retains quadratic exact covariance arithmetic. Gaussian median-error and independent calibration-error assumptions remain explicit. |
| TE06 | [`GPUTerrainTiledFlow`](../../modules/graph/src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts) | Owned-cell GPU exchange into a resident global DAG, atomic D8/branching transport and logarithmic reach pointer jumping. Whole-domain equality passes at 65,536 cells across four poisoned-halo tiles. Global fill/flat-resolution → D8/MFD fraction preparation → owned-tile processing runs in one graph and matches existing whole-domain operators. | Residency is explicit: approximately 60 bytes of internal global-cell scratch plus caller tile buffers; the 32-byte-per-cell fraction binding must fit device limits. This is resident full-domain composition, not out-of-core streaming or independently filled tiles. |

## Point clouds

[API contracts](../../docs/api-reference/gpgpu/gpu-pointcloud.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| PC01 | [`GPUPointCloudNeighborhoods`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-neighborhoods.ts), [`GPUPointCloudFeatures`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-features.ts) | Exact bounded 3D sorted-grid neighborhoods, owned query rows with certified halo-support flags, covariance/Jacobi normals and shape features. Actual PDAL planar-normal reference and uniform/skew scaling through one million candidates. | Viewpoint/up orientation is explicit; no globally propagated orientation or unlimited stored support. Coincident occupancy remains linear per query; the million-point skew benchmark uses 64 queries. |
| PC02 | [`GPUPointCloudVoxelDownsample`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-voxel-downsample.ts), [`GPUPointCloudRadiusFilter`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-radius-filter.ts), [`GPUPointCloudStatisticalFilter`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-statistical-filter.ts), [`GPUPointCloudVoxelMerge`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-voxel-merge.ts) | Separate sampled representatives and indexed centroids; finite scalar/x2/x3/x4 attributes; disjoint-owned voxel merging with global IDs and overflow flags. Radius masks and linear block-moment statistical masks, with actual PDAL separated-outlier classifications. | Caller excludes duplicate halo rows before voxel merging and supplies complete neighborhoods. Population standard deviation is explicit and differs from PDAL sample standard deviation; distributed global threshold reconciliation is not automatic. |
| PC03 | [`GPUPointCloudSurfaces`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-surfaces.ts), [`GPUPointCloudHeight`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-surfaces.ts) | Indexed classified DTM/DSM/CHM; signed containing-cell, bilinear and bounded IDW HAG with support/provenance. IDW fills missing query support. Actual PDAL nearest-one comparisons at identical ground/query cell-centre XY. | Direct ground-point Delaunay/nearest interpolation, arbitrary extrapolation and vertical-datum transforms are outside the raster-sampling contract. |
| PC04 | [`GPUPointCloudGround`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-ground.ts), [`GPUPointCloudSMRF`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-smrf.ts) | Separate progressive morphology and parameter-compatible SMRF; sorted-cell minima, exact ring-bounded eight-neighbor fill and repeated-cross diamond morphology. Four actual PDAL 2.8.4 fixtures agree completely; measured one-million-point SMRF. | Float32 threshold/tie differences remain possible. Window/grid bounds are explicit; broad unsupported holes can require large searches. Recorded production-scale evidence uses window4 and a 256×256 grid. |
| PC05 | [`GPUPointCloudICP`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-icp.ts) | Initialized point-to-point Horn and rank-aware point-to-plane Gauss–Newton ICP, static indexed correspondence, block moments, Huber/Tukey rejection, masks and convergence history. Actual PDAL point-to-point transform reference; one-million-point indexed benchmark. | Local initialization remains caller-owned; no global alignment guarantee or trimmed-order-statistic objective. Rank-deficient point-to-plane systems return explicit failure. |
| PC06 | [`GPUPointCloudChange`](../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-change.ts) | Aligned oriented-cylinder changes, sample roughness/support, projected registration covariance and linear/quadrature Gaussian intervals. Optional supplied Gaussian spatial covariance integrates exact pairwise mean variance; independent SciPy oracle plus 91 supported rows from the pinned actual PDAL M3C2 upstream golden dataset (distance/counts and explicit population/sample uncertainty conversion). | Upstream golden comparison passed two GPU batches; no local PDAL M3C2 binary execution is claimed. Twelve no-solution golden rows are outside the minimum sample-variance support contract. No automatic normal-scale selection or fitted unknown correlation model; pairwise covariance is bounded-support quadratic work. |

## Interpolation and simulation

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/interpolation-expansion.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| IN01 | [`GPUVariogramModelFit`](../../modules/graph/src/gpu-dataframe/pair-statistics/gpu-variogram-model-fit.ts) | Resident spherical/exponential/Gaussian fits with constrained parameters, all three existing weighting conventions, convergence/residual diagnostics, direct GSTools fitting and measured readback comparison. | Original bounded-fit acceptance covered. No readback-avoidance benefit at measured 20/256 bins: CPU round-trip medians 1.3/3.2 ms versus resident 4.3/51.3 ms. Global nonlinear optimum is not promised. |
| IN02 | [`GPULocalKriging`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts) | Simple, ordinary, affine or arbitrary supplied universal drift, and external drift; 32 neighbors, eight columns, variance/weights and rank/missing-drift status. Direct GSTools/PyKrige prediction/variance parity. | Original bounded local modes covered. Candidate selection scans observations; unlimited drift dimension and global indexed kriging are not claimed. |
| IN03 | [`GPULocalKriging`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts) | Separate measurement-error/process-nugget covariance; up to 64 nonnegative weighted quadrature points, support-averaged covariance and latent block variance. Direct gstat irregular-block parity. | Original discrete-support acceptance covered. Resolution is explicit and caller-selected; exact polygon integration is outside this quadrature contract. |
| IN04 | [`GPULocalKriging`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts) | Two to four variables, known or separate unknown means, one to four supplied PSD LMC structures, masks and singular status. Direct gstat three-variable/two-structure weighted-block prediction parity. | Original supplied-valid-model acceptance covered. LMC fitting and more than four variables/structures are not implemented; no claim that cross-variograms alone define a valid model. |
| IN05 | [`GPULocalRBF`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf.ts) | Eight SciPy kernels, degrees −1 through 2, scalar/per-sample smoothing, 32-neighbor solves and resident grid/CSR composition. Direct SciPy parity; 1,024/16,384-site pipeline measurements. | Original bounded-local acceptance covered. Higher degrees and global dense RBF systems remain outside this API; extrapolation is explicit. |
| IN06 | [`GPUConditionalGaussianSimulation`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts) | Shared GPU storage Cholesky for 128 observations plus 128 joint query sites; Philox batching, exact-hit/semidefinite handling, direct GSTools moments and gstat ensemble joint covariance. | Original bounded-factorization acceptance covered. At maximum measured size, sequential factorization costs 146.2 ms and 267,272 transient bytes; separate calls do not preserve joint covariance. GSTools CondSRF joint covariance differs by construction, documented explicitly. |
| IN07 | [`GPUNaturalNeighbor`](../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor.ts) | Sibson stolen areas with resident Delaunay cavity traversal, CGAL coordinates/prediction parity, hull/sample/duplicate/collinear/capacity contracts. Indexed scaling measured through 65,536 sites. | Original bounded-cavity acceptance covered. Caller supplies valid Delaunay topology and containing-triangle seeds; 64 cavity sites/64 polygon vertices and explicit exhaustion. Index-build/seed-location costs are excluded from interpolation timing. |

Interpolation round-two evidence: the original 25 numerical GPU tests and 21 advanced tests passed
on native Apple WebGPU/Metal; 19 maximal kernel-audit cases pass core binding/memory checks.
Pinned direct references are GSTools 1.7.0, PyKrige 1.7.3, SciPy 1.15.3, CGAL 6.0.1/Boost 1.87.0,
and gstat 2.1-6 under R 4.4.3. Generators are committed under `modules/graph/test/oracles`.
The expanded measurement artifact (local research log, not checked in) records 17
workloads plus two actual GPU-versus-CPU readback/fit/reupload comparisons, input-update checks,
phase/memory accounting and capture-time source hashes. Indexed Sibson warm medians were
0.3/0.3/0.4 ms at 1,024/16,384/65,536 sites for 128 queries; resident grid/RBF pipelines used
5.6/6.9 ms at 1,024/16,384 sites. These results establish supported bounded operations and their
observed costs, not general parity with every reference-library option or a GPU speedup.

## Spatial statistics and models

The matrix marks the completed selected contracts ST01, ST02, ST03.ml, ST04, ST05, ST07, ST08, ST09.fuzzy/islands and all four ST11 modes **Implemented**. Their stated exclusions do not enlarge the selected acceptance contract. ST03.random and ST06 are now **Implemented** with bounded random/unit spatial lag ML, spreg-compatible Hausman diagnostics, and complete MGWR backfit response-Jacobian uncertainty. ST10 is tracked separately in its dedicated cross-validation update.

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/coverage-statistics.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| ST01 | [`GPUInstrumentalVariablesCovariance`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance.ts), [`GPUSpatialTwoStageLeastSquares`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts) | HC0/HC1 and admissible symmetric unit-diagonal spatial-HAC full IV covariance; direct spreg 1.8.3 TSLS/robust_vm fixtures. | HC2/HC3, clustered covariance and automatic PSD certification are outside the selected HC/HAC contract; the kernel is caller supplied. |
| ST02 | [`GPURestrictedRegimeRegression`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts) | Common/varying coefficients, full covariance, nested Chow F/degrees and per-coefficient/joint Wald chi-square tails; direct OLS_Regimes fixtures. | Spatially robust restriction inference is outside the selected homoskedastic contract. |
| ST03 | [`GPUSpatialPanelErrorMaximumLikelihood`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood.ts) | Balanced unit-FE spatial-error ML plus pooled-panel LM error and robust LM lag/error tests; direct Panel_FE_Error and diagnostics_panel fixtures. | Random/unit spatial lag effects and Hausman are delivered separately under ST03.random; time effects and clustered covariance are outside the selected FE-error contract. |
| ST03.random | [`GPUSpatialPanelLagMaximumLikelihood`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts) | Complete balanced random/unit spatial lag ML, conditional beta/rho covariance, quasi-demeaned predictions, exact determinant profiling, and additional FE fit for spreg-compatible Hausman inference. Direct pinned spreg estimates/covariance/likelihood/diagnostics and adversarial failure tests. | 2–32 entities/periods, 1–7 predictors; rho [-.999,.999], random phi [1e-5,1]. Conditional covariance excludes phi/sigma². Hausman uses spreg RE.k df rather than contrast rank and rejects indefinite covariance differences. |
| ST04 | [`GPUSeeminglyUnrelatedRegression`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts) | Nonspatial one-step SUR and Kelejian–Prucha spatial-error SUR, full joint covariance; direct spreg SUR/SURerrorGM fixtures. | Spatial mode requires aligned complete rows, nonnegative row-standardized zero-diagonal weights without islands. Iterated SUR is outside the selected spatial variant. |
| ST05 | [`GPUProbitRegression`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression.ts) | Binary probit likelihood, average effects and expected-information errors plus Pinkse, Kelejian–Prucha and Pinkse–Slade residual diagnostics; direct spreg Probit. | Latent spatial probit and alternative robust covariance are outside this classic-probit contract. |
| ST06 | [`GPULocalCollinearity`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity.ts) , [`GPUGeneralizedGeographicallyWeightedRegression`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts), [`GPUMultiscaleGeographicallyWeightedRegression`](../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts) | Local VIF, centered condition, intercept-inclusive design condition/VDP; generalized exact LOO search and Poisson/logistic SE/influence; full MGWR backfit uncertainty and effective parameters; direct mgwr fixtures. | Full MGWR inference propagates the coupled response Jacobian, estimates sigma² under either residual-df convention, and reports per-component effective parameters. It conditions on selected bandwidth history; finite-grid LOO selection differs from Sel_BW AICc/continuous search. Explicit n²p storage/cubic work bounds apply. |
| ST07 | [`GPUMultivariateLocalGearyPermutationTest`](../../modules/graph/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test.ts) | Common-mask whole-vector conditional randomization, replay, equal-tail or esda minimum-tail p-values, classifications and Bonferroni; direct esda observed statistics and replay-tail checks. | Additional correction families and identical upstream RNG/label numbering are outside the selected documented conventions; columns are never permuted independently. |
| ST08 | [`GPURankMarkov`](../../modules/graph/src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts) | Full/geographic rank transitions with direct giddy 2.3.6 tie ordering, optional earlier-row tie ordering, common panels, mobility and existing stationary diagnostics. | Fractional ties, missing-panel interpolation and more than 64 states remain unsupported; stationary interpretations retain irreducibility/ergodicity assumptions. |
| ST09 | [`GPUFuzzyContiguity`](../../modules/graph/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts), [`GPUIslandAttachment`](../../modules/graph/src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts) | Filled-polygon distance tolerance and directed/reciprocal original-island attachment with provenance; direct libpysal fuzzy_contiguity/attach_islands fixtures. | Planar valid polygons and supplied representatives only; geographic geometry, topology repair and robust GEOS degeneracy parity remain outside this selected contract. |
| ST10 | [`GPUSpatialCrossValidation`](../../modules/graph/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts), [`GPUSpatialCrossValidatedOLS`](../../modules/graph/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols.ts), [`GPUNestedInterpolationValidation`](../../modules/graph/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts) | Seeded blocks/buffers plus group/environmental/NNDM/kNNDM designs; OLS and nested generalized GWR or covariance/variogram/RBF selection, outer training-only refits and held-out metrics. Direct blockCV4.0.1 designs and independent NumPy/SciPy complete nested scores, predictions and metrics. | Explicit finite model ladders and bounded designs: 256 rows, NNDM64, local neighbors32. Automatic Lloyd candidate generation differs from blockCV hierarchical/merge/RNG and KS random fallback; no response balancing or continuous empirical-variogram optimizer is claimed. |
| ST11 | [`GPUSpatialInteraction`](../../modules/graph/src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction.ts) | All four one-covariate Poisson constraint modes with grid or continuous bounded profile calibration, IPF, deviance/marginal residuals and explicit exhaustion; direct spint GLM fixtures. | Multiple covariates, other count distributions and coefficient uncertainty are outside the selected calibrated flow contract; finite f32 search intervals are explicit. |

Direct reference execution: [pinned Python generator](../../modules/graph/test/oracles/statistics-upstream.py), [upstream fixture](../../modules/graph/test/oracles/fixtures/statistics-upstream.json), [blockCV R generator](../../modules/graph/test/oracles/statistics-blockcv.R), and [blockCV fixture](../../modules/graph/test/oracles/fixtures/statistics-blockcv.json). The separate NumPy/SciPy oracle remains labeled as a composed numerical reference. Native Apple WebGPU verification passed 38 tests (25 direct/adversarial numerical cases and 13 executable statistics lenses), covering direct models, full covariance, masked-response perturbation, forced optimizer exhaustion, invalid spatial weights and blockCV masks. Focused node verification passed 64 tests including maximal variants and expanded lenses. Maximal-option shader audit includes generalized selection/inference, spatial SUR and continuous flow calibration.

[Six measured workloads](../../modules/graph/test/oracles/fixtures/statistics-performance-baseline.json) separate preparation, compile, first/repeated resident execution, readback and graph transient memory. Resident medians were 1.8/4.8/63.8 ms for SUR at 64/1,024/16,384 rows and 2.5/7.0/17.5 ms for three-candidate Poisson LOO search at 32/128/512 rows on the recorded Apple WebGPU device. These are bounded measurements with cached pipelines, not cross-device throughput or peak driver-memory claims. The advanced extension additionally executes actual spreg 1.8.3 random/unit spatial lag ML and Hausman, and mgwr 2.2.1 complete smoother inference, including a finite-grid candidate policy composed from actual local fits. Native advanced/lens verification passes 30 cases; maximal shader audits include random/unit/Hausman and full MGWR candidate/inference variants. Broad statistical coverage studies remain separate research evidence.

ST10 advanced validation closes the separately staged group/environmental/NNDM/kNNDM modes under [explicit contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/spatial-validation-advanced.md). The [pinned blockCV4.0.1 generator](../../modules/graph/test/oracles/spatial-validation-advanced.R) and [NumPy2.2.6/SciPy1.15.3 nested reference](../../modules/graph/test/oracles/nested-validation-oracle.py) verify actual GPU results. Fourteen native GPU tests pass, including the executable lens, held-out-response replay, invalid/exhausted/empty designs, mask overlap and arithmetic overflow; six node checks cover maximal kernels and diagnostic/output alias rejection. Ten measured workloads (local research log, not checked in) record compile/upload/first/warm/updated/readback and memory separately. Warm resident medians at the largest measured sizes are 5.2 ms for 256-row groups, 0.7 ms for 256-row environmental clustering, 9.2 ms for 64-row NNDM, 42.4 ms for 256-row/four-candidate kNNDM and 2.1 ms for 256-row/two-outer/two-inner/two-model nested selection. These are cached-device observations, not speedup or combined-maximum latency claims.

## Point processes and probability sampling

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/point-process.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| PP01 | [`GPUInhomogeneousPointPattern`](../../modules/graph/src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern.ts) | Supplied/self-included/leave-one-out Epanechnikov intensity, polygon border or translation K, and smoothed Epanechnikov pcf with explicit divisor/origin correction. GPU signed triangle intersections preserve concavity, holes and components, exact metric fallback prevents determinant cancellation, and binary reductions bound accumulation error. Direct spatstat 3.8.2 plus exact clipped-overlap/composed kernel references and Shapely 2.1.2 25-hole fixture pass; 4096-event resident pair cache and input updates execute. | Selected all-pairs path supports 4096 events/256 radii; allocation/work budgets and unresolved f32 slivers are explicit. Supplied intensity and plug-in estimates remain distinct. Distance-divisor coincident events reject inference, whereas upstream omits their nonfinite terms. Native new suite21/21, combined regression/lens suite38/38 and focused node/audit31/31. |
| PP02 | [`GPUPointPatternEnvelope`](../../modules/graph/src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope.ts) | Resident supplied curves, pointwise envelopes, maximum-deviation/DCLF tests, and two-sided extreme-rank simultaneous envelopes with conservative tie intervals. Direct spatstat DCLF and independent SciPy rank fixtures. | Simulation/fitted-model refitting must be upstream; exchangeability is a caller assumption. Rank work has an explicit quadratic budget; extreme-rank-length tie breaking is not implemented. |
| PP03 | [`GPUNetworkKernelDensity`](../../modules/graph/src/gpu-network/gpu-network-kernel-density.ts), [`GPUNetworkInhomogeneousK`](../../modules/graph/src/gpu-network/gpu-network-inhomogeneous-k.ts) | Arc-length quadrature-normalized shortest-path triangular density plus supplied-intensity network K with Ang shell correction and three normalizations; direct spatstat.linnet 3.5.3 branch/cycle/disconnected fixtures. | Events lie at graph vertices; continuous density normalization is quadrature, not analytic integration or equal-split diffusion. Caller supplies complete shortest-path distances and admissible intensities. |
| PP04 | [`GPUOPTICS`](../../modules/graph/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts) | Complete cooperative GPU core/reachability/predecessor ordering through 4,096 rows, DBSCAN and Xi extraction with nested intervals and predecessor correction; pinned scikit-learn 1.7.2 parity. | Brute-force bounded neighborhood work; no indexed large-cloud frontier. Explicit partial iteration budgets remain available but are never reported as completed ordering. Performance evidence uses an independent exhaustive CPU reference. |
| PP05 | [`GPUSpatialBalancedSample`](../../modules/graph/src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample.ts), [`GPUContinuousSpatialSample`](../../modules/graph/src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample.ts) | Finite equal/stratified/integer-ticket probability designs plus continuous random-shift line/area frames, fixed legacy exclusions, minimum spacing and complete independent replacement waves. Exact ideal-design intensity and reciprocal weights, provenance, bounded-status and GPU empirical mass/moment/frequency checks. | Explicit alternatives to spsurvey GRTS. Continuous line spacing is arc length; legacy-excluded neighborhoods are outside the weighted target frame. Selective response-dependent replacement and general design variance are not claimed; numerical phases use 24 bits and float32 geometry. |

## Networks and allocation

[API contracts](../../docs/api-reference/experimental/gpu-network/network-movement-coverage.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| NW01 | `GPUNetworkPointRouting` | Fractional directed-edge paths with legal turns, same-edge/parallel provenance; direct pgRouting 3.8.0 withPointsCost same-edge, parallel-edge and unreachable fixtures, independent NetworkX oracle, and measured 8/64/128-edge batches. | Caller supplies snapped fractions, reverse-edge representation and legal turn topology; direct PostgreSQL 16.6/PostGIS 3.5.3 server parity passes for the bounded directed contract. |
| NW02 | `GPUNetworkFlow` | Integral max flow, residual cut certificate, original-edge conservation; NetworkX exact references and 8/32/128-node measurements. | Bounded 128 nodes/1024 edges; no path decomposition or production sparse push/relabel solver. |
| NW03 | `GPUNetworkFlow`, `prepareGPUNetworkFlowDemands` | Minimum-cost residual cancellation, balanced-demand supernodes, unserved demand, optional exact 64-bit integer objective. | Nonnegative original costs; negative-cost circulation and lower-bound constraints remain unsupported. |
| NW04 | `GPUNetworkLouvain` | Weighted directed/undirected modularity, seeded/ascending visitation, quotient levels, objective/convergence trace; NetworkX and 4/16/32-vertex measurements. | Dense <=32 vertices; local-move modularity heuristic, no global-optimum promise or scalable sparse implementation. |
| NW05 | `GPUNetworkFacilityOptimization` | Exhaustive backup coverage, p-dispersion and unsplit capacitated p-median; feasibility/completion certificates, direct spopt0.7.0/PuLP3.3.0 CBC references and 4/8/16-facility measurements. | Bounded candidate enumeration, no general mixed-integer branch-and-bound or fractional allocation. |
| NW06 | `GPUNetworkTimeRouting`, `prepareGPUNetworkRestrictionStates` | Affine/piecewise FIFO arrival, explicit first-sample waiting, reachable overlapping restriction automaton and provenance; time-expanded NetworkX reference. | Non-FIFO schedules/general time windows and direct Valhalla server parity remain outside the supported model. |
| NW07 | `GPUNetworkRationalAccess` | Integral/fractional demand, fractional/absolute shift caps, conserved allocations/unserved rows and equilibrium gap; direct access1.1.9 equilibrium fixture and <=64-origin measurements. | f32 fractional conservation tolerance; full cross-version option parity is not claimed. Rounding stalls remain unconverged. |

## Movement and temporal geometry

[API contracts](../../docs/api-reference/experimental/gpu-network/network-movement-coverage.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| MV01 | `GPUTrajectoryKalman`, `prepareGPUTrajectoryObservations` | Isotropic/full correlated Joseph/RTS covariance, explicit WGS84 local-plane/bigint epoch preparation; direct MovingPandas0.22.4 and NumPy/pyproj references, <=4096-observation measurements. | Sequential per-track solve; caller chooses appropriate projection. No associative long-track scan. |
| MV02 | `GPUTrajectoryRestriction` | Polygon rings/holes, open/closed/half-open intervals, merged compaction, analytic temporal pairwise within-distance and three-valued gaps; direct MovingPandas clipping and MobilityDB 1.2.1 polygon/time-boundary/temporal-distance references. | Planar f32 tolerance/bounded slots; no general temporal SQL algebra or geodesic interpolation. Always semantics match always_eq(tDwithin(...),true); a captured MobilityDB 1.2.1 aDwithin crossing anomaly is explicitly not reproduced. |
| MV03 | `GPUTrackSimilaritySearch` | Admissible metric bounds, stable GPU survivor compaction/capacity status, exact threshold top-k and indexed-pair provenance; direct tslearn/similaritymeasures fixtures and <=4096-pair selectivity measurements. | Candidate index supplied explicitly; no implicit all-pairs/spatial-index construction, densification/geographic metrics or CPU speedup claim. |

## Urban morphology and redistribution

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/urban-morphology.md). Retained first-tranche rows keep their [separate evidence record](./geospatial-coverage-tranche-one.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| UR01 | [`addBuildingTessellationRecipe`](../../modules/graph/src/gpu-spatial-analysis/urban-morphology/building-tessellation.ts) | Building-boundary sampling and stable-ID polygon dissolve; convex or simple nonconvex enclosures with holes, GPU clipping and independent overflow status. Direct momepy 0.10.0 concave/holed fixtures pass. | Explicit 128-sample/64-building capacities and snap lattice; automatic footprint shrink/repair and street-enclosure construction remain caller preprocessing. No city-wide tessellation throughput claim. |
| UR02 | [`addStreetProfilesRecipe`](../../modules/graph/src/gpu-spatial-analysis/urban-morphology/street-profiles.ts) | Side-aware cross sections, missing-side censoring, height/width and sampled frontage; optional event-split visible facade arc length in a straight street corridor. Direct momepy street-width reference and occlusion tests pass. | Caller supplies street samples/normals and subdivides curved streets into corridors; full momepy workflow equivalence is not claimed. |
| UR03 | [`addBuildingAdjacencyRecipe`](../../modules/graph/src/gpu-spatial-analysis/urban-morphology/building-adjacency.ts) | Shared-wall adjacency/length and axial differences; cardinal orientation/alignment with explicit Queen/custom support. Optional true component union measures exterior-only perimeter, excluding courtyards. Direct momepy 0.10.0 shared-wall, buffer=0 perimeter, cardinal and isolate fixtures pass. | Adjacency supports 1,024 buildings; union structures support 64 with explicit snap/capacity status. Shape orientation is reused from supplied descriptors; positive buffer/repair remains explicit upstream geometry preparation. |
| UR04 | [`addDasymetricRedistributionRecipe`](../../modules/graph/src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe.ts) | Retained first-tranche fractional source support, ancillary eligibility, whole-cell target ownership and conservation ledger; direct pinned tobler masked-area reference retained, with 16 native tests rerun. | Independent source/target pixel fractions are not exact joint overlap; arbitrary target polygon subdivision and larger streaming packing remain explicit. |
| UR05 | [`addIsovistsRecipe`](../../modules/graph/src/gpu-spatial-analysis/urban-morphology/isovists.ts), [`addVisibilityGraphRecipe`](../../modules/graph/src/gpu-spatial-analysis/urban-morphology/visibility-graph.ts) | Uniform/vertex-event isovists with explicit inside/outside even-odd/nonzero free-space classification; bounded dense visibility, raw/normalized choice, HH/P-value/Teklenburg/Penn metrics using actual BFS depth. Pinned depthmapX v0.9.1 math helpers compiled/executed directly; holes/boundaries/nonfinite fixtures pass. | Angular epsilon and finite-range polygon approximation remain explicit; <=256 visibility vertices and <=4,096 vertex-event rays. Direct helper execution does not certify full depthmapX graph-import/GUI parity; angular-segment NACh/NAIn are separate models. |

## Landscape ecology and equity

[API contracts](../../docs/api-reference/experimental/gpu-spatial-analysis/landscape-ecology.md).

| ID | Actual implementation | Delivered scope | Remaining scope or evidence |
| --- | --- | --- | --- |
| EC01 | [`addLandscapeMetricsRecipe`](../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/landscape-metrics.ts) | Patch/class area, physical perimeter, adjacency, class and landscape mesh/cohesion/aggregation and two contagion conventions; independent boundary/background controls and GPU reconciliation of globally unique tile patch IDs. Direct landscapemetrics 2.2.1 square-cell/background and singleton-class fixtures pass; pooled landscape cohesion is calculated separately from class scores. | Reconciliation needs resident halo-free complete support and globally unique IDs; no out-of-core fetching. Rectangular-cell physical lengths are independently defined; no blanket package parity. |
| EC02 | [`addPatchProximityRecipe`](../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/patch-proximity.ts) | Exact separable boundary or cell-center distance fields, physical core or integer four-neighbor erosion, explicit open/edge domain policy and optional GPU seam reconciliation. Direct landscapemetrics core/ENN fixtures pass; two-patch proximity is composed from its direct area/ENN. Native 32²/128²/256² physical-mode measurements retain exact distances. | O(P*C) scratch and O(P*C*(W+H)+P²*C) work; high patch-count workloads unmeasured. landscapemetrics2.2.1 has no proximity-index function, so that comparison is explicitly composed. Euclidean proximity is not least-cost movement. |
| EC03 | [`addResistanceConnectivityRecipe`](../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts); `addSparseResistanceConnectivityRecipe`, `addRasterResistanceConnectivityRecipe` | Small CG plus sorted-CSR Jacobi-PCG sparse resistance, independent true residual/energy checks, raster 4/8-neighbor conductance adapter and current density. Independent SciPy nonuniform-network reference, six primary Circuitscape golden voltage/current comparisons and 65,538-vertex measured network pass. | Authentic pinned Circuitscape upstream regression outputs were compared; Julia was not executed locally and the legacy golden-producing binary version is unspecified. Serial GPU component reconciliation and graph-stage count remain scaling costs; measured parallel-route topology does not establish arbitrary conditioning, ensemble orchestration or multigrid. |
| EC04 | [`addSegregationDecompositionRecipe`](../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/segregation-decomposition.ts); `addSegregationCounterfactualRecipe`, `addPopulationSupportAlignmentRecipe` | Explicit common-support Shapley decomposition plus composition/dual-composition/share quantile counterfactuals, stable-ID support alignment and replayable Philox paired-label randomization. Direct segregation 2.5.3 methods and all 16 exchange assignments pass. | Counterfactual generation bounded to 2,048 units; support ID alignment does not infer spatial interpolation. Other segregation statistics and null models remain separate; paired exchange requires its stated assumption. |

Urban/ecology round-two verification passed 56 native GPU tests across the original, advanced and
13-recipe executable-lens suites, plus a separate 34-test advanced/retained-dasymetric run.
Direct reference generators are `modules/graph/test/oracles/urban-ecology-advanced.py`
(momepy 0.10.0, segregation 2.5.3 and SciPy 1.15.3) and `urban-landscape.R`
(landscapemetrics 2.2.1, terra 1.9.50), plus `urban-depthmap.py` (compiled upstream v0.9.1 normalization helpers with source SHA-256). Maximum-capacity audits and owned node/lens tests passed 35 tests. The
urban/ecology execution record (local research log, not checked in) measures sparse
resistance at 1,026/8,194/65,538 vertices and exact proximity at 32²/128²/256² cells, separating
preparation, graph compilation, resident execution, readback and scratch. These narrow topology and
patch-distribution measurements do not establish general CPU speedup or arbitrary solver conditioning.

## Recorded scale evidence

Device-specific records separate preparation, compilation, upload, resident execution, repeated
updates, readback and memory where applicable. Measurements cover the recorded data distributions
and algorithm bounds; they do not imply speedup over every CPU implementation or other devices.

| Domain | Executed scale evidence |
| --- | --- |
| Geometry | [Join and refinement measurements](../../modules/graph/test/oracles/fixtures/geometry-scaling-evidence.json), including 16/64/256-feature batches and both timing improvements and regressions. |
| Global coordinates | [Expanded 14-workload record](../benchmarks/data/global-coordinates-expanded-2026-10-23.json), including 16,384-row cell workloads and bounded adaptive covers. |
| Terrain | Twelve phase measurements (local research log, not checked in), covering six operation families at 64 and 16,384 cells, native GPU timestamps, resident updates, readback, transient memory and a best-case CPU transport baseline. |
| Raster | Twenty measured workloads (local research log, not checked in) cover categorical summaries, remapping, texture, temporal composites/smoothing, registration, region merging, spectral recipes/unmixing and seasonal breaks. Initial, warm and changed-input checks accompany separate upload/compile/readback timings, native timestamps and transient memory; retained fractional/focal measurements are in the first-tranche record. |
| Point clouds | Advanced record (local research log, not checked in), including million-point SMRF, indexed ICP and skewed neighborhoods; the earlier indexed-pipeline record (local research log, not checked in) remains available. |
| Interpolation | Advanced record (local research log, not checked in), covering local models, conditional simulation and variogram fitting, with CPU comparisons and measured regressions. |
| Point processes | Complete-ordering record (local research log, not checked in), including complete 4,096-row OPTICS with every core/reachability value checked. The partial-frontier record (local research log, not checked in) is historical only. |
| Spatial validation | Advanced design and nested-model record (local research log, not checked in), including 256-row group, environmental and kNNDM designs, 64-row NNDM and 256-row nested interpolation. |
| Spatial statistics | [Measured model record](../../modules/graph/test/oracles/fixtures/statistics-performance-baseline.json), with explicit model sizes and phase timings. |
| Networks and movement | [Execution and comparison record](../../docs/api-reference/experimental/gpu-network/network-movement-benchmarks.md), including bounded flow/facility/routing and 4,096-observation/pair workloads. |
| Urban and ecology | October 24 execution record (local research log, not checked in), including 65,538-vertex sparse resistance, exact 256²-cell proximity and 256²-cell class/landscape summaries; earlier execution record (local research log, not checked in). |

Raster and terrain domain tests also validate complete results at the sizes recorded in their rows:
512² conservative remapping, 65,536-cell branching/tiled flow, 256-basin summaries and 64-pit breach
batches. Large bounded tests establish correctness for those workloads, not universal throughput.
Some captures use the shared dirty tree without a frozen source revision; that provenance limitation
is retained in each artifact.

Public-API computation lenses exercise the contributors and recipes on native WebGPU. The current
integration capture includes 87 scenarios, including advanced geometry/global/urban operators,
generalized GWR, random-effects panels, nested interpolation validation and continuous sampling. Static twins retain captions and provenance. Compact array thumbnails show
at most four selected panels, disclose omitted panels, and retain complete captures in the full result.
All 87 captures and the seven static-twin regeneration checks passed.

## Contract boundaries

- Numerical failure, capacity and approximation states remain visible when composing outputs.
  An exhausted budget or incomplete edge list does not establish convergence or complete geometry.
- Direct executed references, pinned upstream golden outputs, independent composed references and
  analytical invariants are distinguished in the domain evidence. Golden outputs do not imply that
  their upstream binary ran locally; no row claims complete upstream-library parity.
- Small bounded exact algorithms and quadratic/serial GPU baselines have no implied large-data
  speedup. External grid acquisition, general topology repair and out-of-core residency remain
  explicit boundaries where the backlog places them outside the selected operation.
- Focused node tests and native numerical cases complement the full repository build, test and
  website gates recorded below.

## Integrated verification record

Validation covers the combined, uncommitted `map-contributors` working tree based on
`e6375c5d683ee8a6289789f07ea8e24e7b87c21e`. That commit alone does not reproduce the tested
sources: the tree includes preserved earlier work and these ten implementation streams. Native
browser checks use Chromium 151 and the local Apple Metal WebGPU backend.

Integration fixes retain the existing numerical assertions and rendering tolerance. GJK retries a
bounded set of initial directions to escape a cycling simplex. The Three adapter matches upstream
32-bit index uploads while preserving CPU index/readback types. Public examples and capture
budgets were repaired without removing their numerical checks.

The [headless runner](../../scripts/run-headless-tests.mjs) executes eight sequential fresh browser
processes, covers every configured shard even after an ordinary failure, and fails if any shard fails.
It does not retry failed tests or remove the large inflation and density-tree probes. Browser setup
explicitly [destroys cached test devices](../../modules/test-utils/src/create-test-device.ts) after each
file's own cleanup. On macOS the runner enables Vitest's existing low-disk Chromium garbage-
collection check, preserving explicit environment overrides and the upstream disk threshold.
The three previously disconnecting files passed unchanged in isolated runs; all four tests also
passed with collection deliberately triggered. Cleanup and runner regression tests passed.

| Check | Result |
| --- | --- |
| `nvm use` | Passed; Node 22.22.1. |
| `yarn install` | Passed; existing peer warnings remain. |
| `yarn lint fix` | Passed; 7,883 files checked with no final formatting changes. |
| `yarn build` | Passed after the final source changes, including cross-package CommonJS identity checks. |
| Full `yarn test-node`, invoked by `yarn test` | 1,219 files and 12,462 tests passed; 2 files and 433 tests skipped. |
| Full `yarn test` | Passed with exit code 0: Node plus all eight headless shards. Browser totals: 1,073 files and 12,089 tests passed; 8 files and 836 tests skipped. No assertion failures or unhandled errors. |
| `yarn website:build` | Passed; 1,164 raw documentation pages validated. |
| `(cd website && yarn build)` | Passed; 1,164 raw documentation pages validated. |
| Computation lenses | 87 native captures and 7 static-twin regeneration checks passed; record budgets and public-example checks passed. |
| Roadmap evidence links | All 455 checked local links resolved. |
| SPDX declarations | All 625 changed source files carrying SPDX license markers also carry copyright declarations. |
| Working-tree whitespace | `git diff --check` passed. |

The completed third browser shard reported a Chromium shutdown timeout after its tests passed;
Vitest used its process-cleanup fallback. This warning is retained separately from numerical
results. Website builds retain upstream dynamic-import and bundle-size warnings. The website
verification was repeated after package compilation completed to avoid resolving outputs while
the package build was regenerating them.

Detailed command logs for this execution are retained locally under
`/tmp/luma-contributors-round-two/`; that temporary directory is not a portable repository artifact.
Earlier dated evidence remains in the [first-tranche record](./geospatial-coverage-tranche-one.md)
and [original expansion follow-up](./geospatial-toolkit-expansion-follow-up.md), whose results do
not recertify these later additions.

## Additional primary golden-reference gates

EC03 and PC06 additionally passed eight GPU cases against [pinned upstream regression fixtures](../../modules/graph/test/oracles/upstream-golden/README.md), plus three node provenance checks.
Circuitscape comparison uses six `sgNetworkVerify2` source/sink pairs with potentials, branch
currents, unit-source resistance and independent residual/energy checks. PDAL M3C2 comparison
uses all 91 supported Autzen golden rows, exact counts and signed distance within 0.01 coordinate
units; population/sample roughness conversion is explicit, and 12 upstream no-solution rows are
listed separately. Original files, exact commits, SHA-256 and upstream licenses are retained.
Neither Julia nor the PDAL M3C2 binary ran locally; this is primary upstream golden evidence,
not a relabelled composed oracle or a claim of freshly executed reference binaries. No runtime
algorithm changes were required by these comparisons.

## Subsequent final code review

The [ten-reviewer final review](./geospatial-coverage-final-review.md) records additional
correctness fixes and their post-review merge gates. Its final results supersede the
pre-review totals above. A [copyable merge description](./geospatial-coverage-merge-description.md)
summarizes the changes against current upstream master.

### October 24 coverage completion audit

Ten further domain assignments checked the original 78 requirements against the combined source,
reference fixtures, executable consumers and documented limits. This audit completed EC01's
landscape-level aggregation and pooled cohesion outputs, including direct landscapemetrics
fixtures, executable lens assertions and updated measurements (local research log, not checked in).
The terrain phase record (local research log, not checked in)
adds twelve measured workloads separating preparation, compilation, repeated resident execution,
native GPU timestamps, readback and transient allocation; hierarchy and DEM costs remain visible.
The raster phase record (local research log, not checked in) closes the corresponding
measurement gap with twenty small/larger workloads, independent initial/warm/updated result
checks, CPU reference timings and checksummed source identities. Its structured inputs do not
establish arbitrary-distribution or cross-device performance.

Additional regressions cover canonical winding at large translations, invalid S2 cover rings,
unusable raster-registration spectra, near-sample point-cloud interpolation and height overflow,
extreme-coordinate island attachment and survey sampling, Geary status-buffer ownership, and
same-edge returning-route provenance. Straight-skeleton fixtures now check linked offset cycles,
and terrain tests verify that explicit burning works without requested pits or implicit filling.
Focused numerical and ownership checks passed before integration. Counts overlap the final review
and are not additive. Final repository gates remain owned by the linked final review record.
