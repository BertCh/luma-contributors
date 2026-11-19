var e={slug:`api/gpu-spatial-analysis/expansion-lenses`,title:`Geospatial expansion computation lenses`,html:`<h1 id="geospatial-expansion-computation-lenses">Geospatial expansion computation lenses</h1>
<p>These executable examples use small, explicitly synthetic inputs to inspect each contributor’s actual GPU command graph and captured outputs. They illustrate API contracts and status behavior; they are not performance benchmarks or observed geographic datasets. Statistical definitions and bounded algorithm contracts are documented in <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">Statistical expansion</a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-recipes">Expansion recipes</a>, and the raster and network expansion references.</p>
<h2 id="gpulocalspatialheteroscedasticity"><code>GPULocalSpatialHeteroscedasticity</code></h2>
<p>Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.</p>
<p>LOSH divides a weighted neighborhood residual moment by the study-wide residual moment; the graph exposes normalization and invalid-row status.</p>
<p>{/* lens:GPULocalSpatialHeteroscedasticity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-spatial-heteroscedasticity rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-spatial-heteroscedasticity rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpumultivariatelocalgeary"><code>GPUMultivariateLocalGeary</code></h2>
<p>Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded. Descriptive reductions default to
tiled; mode reference retains its previous limit. Permutation inference retains its own ceiling.</p>
<p>Complete-case, variance-standardized multivariate squared differences are averaged over each weighted neighborhood.</p>
<p>{/* lens:GPUMultivariateLocalGeary */}</p>
<p>::lens{scenario=gpu-spatial-analysis.multivariate-local-geary rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.multivariate-local-geary rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpugeneralizedgeographicallyweightedregression"><code>GPUGeneralizedGeographicallyWeightedRegression</code></h2>
<p>This synthetic Poisson fit uses adaptive bandwidthMode with six neighbors and a six-entry selection capacity. Family-specific local IRLS exposes convergence status; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a>.</p>
<p>{/* lens:GPUGeneralizedGeographicallyWeightedRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.generalized-geographically-weighted-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.generalized-geographically-weighted-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpumultiscalegeographicallyweightedregression"><code>GPUMultiscaleGeographicallyWeightedRegression</code></h2>
<p>Additive multiscale backfitting alternates coefficient updates with explicitly selected bandwidths and bounded convergence checks.</p>
<p>{/* lens:GPUMultiscaleGeographicallyWeightedRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.multiscale-geographically-weighted-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.multiscale-geographically-weighted-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuspatialpanelregression"><code>GPUSpatialPanelRegression</code></h2>
<p>This balanced synthetic panel uses effects: unit, exposing within transformation before spatial-lag two-stage least squares. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/panel-fixed-effects">Unit fixed effects</a> for absorbed intercepts and diagnostics.</p>
<p>{/* lens:GPUSpatialPanelRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuspatialregimeregression"><code>GPUSpatialRegimeRegression</code></h2>
<p>Separate fully varying ordinary least-squares models are solved per caller-defined regime, retaining empty and singular group status.</p>
<p>{/* lens:GPUSpatialRegimeRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regime-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regime-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuconcavehull"><code>GPUConcaveHull</code></h2>
<p>Triangle reservation is bounded by 715,827,882 to keep six boundary words per triangle within uint32
addressing; device bindings impose lower limits.</p>
<p>An alpha complex selects Delaunay triangles by maximum circumradius and derives the exposed boundary edges; graph and validity are the appropriate structural explanation.
The circumcircle solve independently scales its two edge equations by powers of two, protecting
products at large, small and unequal coordinate scales. Degenerate triangles and unrepresentable final
radii are excluded. Float32 accuracy still limits nearly collinear circumcircles.</p>
<p>{/* lens:GPUConcaveHull */}</p>
<p>::lens{scenario=gpu-spatial-analysis.concave-hull rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.concave-hull rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpugroupconcavehull"><code>GPUGroupConcaveHull</code></h2>
<p>Independent alpha complexes preserve original point IDs and empty group slots; boundary and triangle capacities remain explicit.</p>
<p>The default <code>layout: 'fixed'</code> reserves <code>T = max(0, 2*n-5)</code> triangle rows per group and <code>3*T</code>
boundary-edge rows per group, where <code>n</code> is the total input position count. Validity masks select
retained rows. With <code>layout: 'packed'</code>, reserve only <code>T</code> triangle rows and <code>3*T</code> boundary-edge
rows across all groups. Each list is compact and group-major; its validity is one in the used
prefix and zero in padding. <code>output.groups.triangleOffsets</code>, <code>triangleCounts</code>, <code>edgeOffsets</code> and
<code>edgeCounts</code> each have <code>groupCount</code> rows and describe ranges in the corresponding compact list.
An empty or fully removed group has count zero and the current prefix offset.
When the total input has fewer than three positions, reserve zero geometry rows; packed counts
and every group range are still written as zero. Resubmitting after all triangles are removed
clears the old compact geometry, validity and counts.
Zero-length logical views still need one physical row of backing storage for the published binding
helper: 12 bytes for <code>uint32x3</code> triangles, 8 for <code>uint32x2</code> edges, and 4 for scalar validity.</p>
<p>Packed <code>output.status.count</code> and <code>requiredCount</code> count boundary-edge rows, and <code>overflow</code> is zero
because this layout requires the full linear reservation. <code>output.status.converged</code> still has one
row per group: a nonconverged group contributes no triangles or edges. The packed reservation is
bounded by 715,827,882 total triangle rows for uint32 addressing, with device bindings imposing
lower limits. Both layouts use the same group-aware Delaunay build and retain original position rows.
All output fields, including the four packed range views, must use distinct buffers disjoint from
the inputs. Missing packed fields fail at construction with a contributor-prefixed diagnostic.
The grouped contributor snapshots its layout and nested output maps at construction while retaining
the supplied GPU views; later edits to the caller's props/maps do not change the validated graph.</p>
<p>{/* lens:GPUGroupConcaveHull */}</p>
<p>::lens{scenario=gpu-spatial-analysis.group-concave-hull rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.group-concave-hull rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuconstraineddelaunaytriangulation"><code>GPUConstrainedDelaunayTriangulation</code></h2>
<p>flipMethod auto selects parallel candidate discovery and adjacency repair when the polygon and
command budget fit; serial retains reference flip ordering. Serial maximumFlips is 4,294,967,294;
parallel polygon and triangle counts are bounded by 536,870,911 and 1,431,655,765 respectively, with
device bindings imposing lower limits.</p>
<p>Polygon triangulation preserves ring constraints and performs legal Lawson edge flips, retaining validity and convergence diagnostics.</p>
<p>{/* lens:GPUConstrainedDelaunayTriangulation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.constrained-delaunay-triangulation rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.constrained-delaunay-triangulation rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuh3cellidentities"><code>GPUH3CellIdentities</code></h2>
<p>Integer H3 topology derives six canonical directed-edge or vertex keys, including deleted pentagon slots; numerical interpolation is not involved.</p>
<p>{/* lens:GPUH3CellIdentities */}</p>
<p>::lens{scenario=gpu-spatial-analysis.h3-cell-identities rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.h3-cell-identities rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuboundedfacilityallocation"><code>GPUBoundedFacilityAllocation</code></h2>
<p>Bounded facility-subset enumeration separates exact p-center certification from demand-order greedy capacitated allocation.</p>
<p>{/* lens:GPUBoundedFacilityAllocation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounded-facility-allocation rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounded-facility-allocation rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addzonaldistributionrecipe"><code>addZonalDistributionRecipe</code></h2>
<p>Composes finite-value filtering, dense zone grouping, ordered quantiles and deterministic modes; a graph explains the separate reductions.</p>
<p>{/* lens:addZonalDistributionRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.zonal-distribution-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.zonal-distribution-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addrasterzonaldistributionrecipe"><code>addRasterZonalDistributionRecipe</code></h2>
<p>Native nodata is filtered before scale/offset calibration, then the zonal distribution stages run on calibrated values.</p>
<p>{/* lens:addRasterZonalDistributionRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.raster-zonal-distribution-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.raster-zonal-distribution-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addexposurebyzonerecipe"><code>addExposureByZoneRecipe</code></h2>
<p>Exposure separates total population, hazard-observed population and threshold-exposed population, with explicit missing-support diagnostics.</p>
<p>{/* lens:addExposureByZoneRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.exposure-by-zone-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.exposure-by-zone-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addregionalrasterchangerecipe"><code>addRegionalRasterChangeRecipe</code></h2>
<p>Aligned slice change detection feeds zone summaries on common support; cell areas, missing support and threshold validity remain separate outputs.</p>
<p>{/* lens:addRegionalRasterChangeRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.regional-raster-change-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.regional-raster-change-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addsurfacepreparationrecipe"><code>addSurfacePreparationRecipe</code></h2>
<p>Bounded filling and optional rank filtering preserve original support provenance and report filled, filtered and unresolved cells separately.</p>
<p>{/* lens:addSurfacePreparationRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.surface-preparation-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.surface-preparation-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addinterpolationdiagnosticsrecipe"><code>addInterpolationDiagnosticsRecipe</code></h2>
<p>Leave-one-out kriging produces residuals and variance-aware validation summaries with explicit invalid and standardized-count diagnostics.</p>
<p>{/* lens:addInterpolationDiagnosticsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.interpolation-diagnostics-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.interpolation-diagnostics-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="adddrainagenetworkrecipe"><code>addDrainageNetworkRecipe</code></h2>
<p>D8 routing, stream extraction, downstream distance and reach assembly form a multi-stage drainage graph with convergence and capacity status.</p>
<p>{/* lens:addDrainageNetworkRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.drainage-network-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.drainage-network-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addlocalvariabilityrecipe"><code>addLocalVariabilityRecipe</code></h2>
<p>Shared weights feed descriptive LOSH and analytic local Moran inference, preserving separate inference families and row validity.</p>
<p>{/* lens:addLocalVariabilityRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-variability-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-variability-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="addcellflowtopologyrecipe"><code>addCellFlowTopologyRecipe</code></h2>
<p>Original OD pairs feed conserved cell totals and a separate immediate-neighbor edge aggregation with exact H3 identities and boundaries.</p>
<p>{/* lens:addCellFlowTopologyRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-flow-topology-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-flow-topology-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}</p>
<h2 id="gpuloshresidualpermutationtest"><code>GPULOSHResidualPermutationTest</code></h2>
<p>Inspect conditional randomization of a fixed LOSH normalized residual field with an explicit greater tail. This does not refit residuals or implement esda LOSH inference.</p>
<p>{/* lens:GPULOSHResidualPermutationTest */}</p>
<p>::lens{scenario=gpu-spatial-analysis.losh-residual-permutation-test rung=result caption="Inspect synthetic outputs and validity."}</p>
<p>::lens{scenario=gpu-spatial-analysis.losh-residual-permutation-test rung=graph caption="Inspect the actual GPU command graph."}</p>`};export{e as default};