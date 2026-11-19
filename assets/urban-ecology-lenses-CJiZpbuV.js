var e={slug:`api/gpu-spatial-analysis/urban-ecology-lenses`,title:`Urban morphology and landscape ecology computation lenses`,html:`<h1 id="urban-morphology-and-landscape-ecology-computation-lenses">Urban morphology and landscape ecology computation lenses</h1>
<p>These executable examples use explicitly synthetic planar geometry, categorical cells and population counts. Each lens runs the actual recipe on a caller-owned GPU command graph and captures its output and status views. They illustrate contracts; they are not observed city data or performance evidence.</p>
<h2 id="addstreetprofilesrecipe"><code>addStreetProfilesRecipe</code></h2>
<p>A street sample faces walls two and three units away, with building heights six and nine. The captured canyon width is five and the mean height/width ratio is 1.5; the validity mask records observation on both sides.</p>
<p>{/* lens:addStreetProfilesRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addbuildingadjacencyrecipe"><code>addBuildingAdjacencyRecipe</code></h2>
<p>Two buildings share a wall of length two. Their axial orientations straddle the half-turn boundary and differ by 0.02π radians. The docs-sized fixture contains only this adjacent pair; their dissolved structure has exterior perimeter eight. The pair table, component labels, morphology and completeness evidence are all captured.</p>
<p>{/* lens:addBuildingAdjacencyRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addsparseresistanceconnectivityrecipe"><code>addSparseResistanceConnectivityRecipe</code></h2>
<p>A 3 × 3 lattice of unit conductances with one weak diagonal chord carries unit current from one corner to the opposite corner. The sparse grounded-Laplacian Jacobi-PCG solve returns the potentials, the edge currents and the Circuitscape-style node currents (one at the source and the sink); the status and the independently recomputed true residual gate the effective resistance.</p>
<p>{/* lens:addSparseResistanceConnectivityRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=graph caption="Follow the sort, CSR assembly and the unrolled conjugate gradient stages."}</p>
<h2 id="addrasterresistanceconnectivityrecipe"><code>addRasterResistanceConnectivityRecipe</code></h2>
<p>A 6 × 5 resistance raster has a no-data wall in its fourth column with a one-cell gap, and a high-resistance cell beside the gap. Eight-neighbour conductances <code>2 / (R(a) + R(b)) / edgeLength</code> are built on the GPU, and unit current flows from a cell on the west edge to a cell on the east edge: the whole current crosses the wall through the gap (its node current is one), and the wall cells have no potential.</p>
<p>{/* lens:addRasterResistanceConnectivityRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=result caption="Inspect the potentials, the current map and the edge conductances."}</p>
<p>::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=graph caption="Follow the edge builder, the invalid-cell count and the sparse solve."}</p>
<h2 id="addisovistsrecipe"><code>addIsovistsRecipe</code></h2>
<p>Eight rays from the centre recover the square enclosure area and perimeter of sixteen. A second observer lies on the boundary and produces invalid measures, so the result also demonstrates the boundary mask.</p>
<p>{/* lens:addIsovistsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addvisibilitygraphrecipe"><code>addVisibilityGraphRecipe</code></h2>
<p>The <code>gpu-network</code> recipe of the same name supports budgeted exact CSR all-source metrics or deterministic sampled estimates via
metricsMode auto/exact. Auto samples when the exact command budget cannot fit; sampled maximum depth
is a lower bound. Approximation, overflow and convergence are separate statuses; explicit exact
rejects an insufficient budget.</p>
<p>Three collinear observers form a path graph. The middle observer has betweenness one. An edge capacity of one deliberately truncates the two-edge output; the status reports both the written and required counts while dense adjacency still supports the metrics.</p>
<p>This scenario uses the dense reference mode. In CSR mode this spatial recipe provides
adjacency and degree only; <a href="/luma-contributors/docs/api/gpu-network/visibility-graph">the gpu-network visibility recipe</a>
adds pivot betweenness, closeness and reached counts.</p>
<p>{/* lens:addVisibilityGraphRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addbuildingtessellationrecipe"><code>addBuildingTessellationRecipe</code></h2>
<p>Local mode uses an adaptive candidate index and publishes convex-piece triangle unions for bounded
simple nonconvex or holed enclosures (at most 32 rings). Reference mode retains dissolved rings, 64
buildings and 128 samples. Device bindings and certification/overflow status still apply.</p>
<p>Eight boundary samples from two rectangular buildings are clipped to a rectangular enclosure and dissolved by building. The two complementary cells each have area twelve. Sampling, overlay counts and completeness accompany the variable-length geometry.</p>
<p>For these four rectangular fragments per building, the recording uses capacities of 64 intersections,
32 noded segments, 16 output vertices and four rings per dissolve. These are fixture-specific
capacities, not recipe limits. Every output geometry, boundary, count and loss flag remains captured.</p>
<p>{/* lens:addBuildingTessellationRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addlandscapemetricsrecipe"><code>addLandscapeMetricsRecipe</code></h2>
<p>Two 2×2 patches fill a 4×2 raster with two classes. Each class has area four, perimeter eight and aggregation one hundred percent. Class adjacency and the complete-domain landscape statistics are captured separately.</p>
<p>{/* lens:addLandscapeMetricsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.landscape-metrics-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.landscape-metrics-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addpatchproximityrecipe"><code>addPatchProximityRecipe</code></h2>
<p>Two two-cell patches are separated by one background cell. With edge depth 0.4 and search radius two, each has core area two, nearest boundary distance one and proximity two. Background remains excluded from the core mask.</p>
<p>{/* lens:addPatchProximityRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.patch-proximity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.patch-proximity-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addresistanceconnectivityrecipe"><code>addResistanceConnectivityRecipe</code></h2>
<p>A three-node network carries unit current between the first and last nodes. Potentials are one, one-half and zero; effective resistance is one. The status and independently recomputed current residual gate interpretation of the solver result.</p>
<p>{/* lens:addResistanceConnectivityRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.resistance-connectivity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.resistance-connectivity-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addsegregationdecompositionrecipe"><code>addSegregationDecompositionRecipe</code></h2>
<p>Four supplied population tables share the same two zones. Their dissimilarities are 0.8, 0.6, 0.4 and zero. The two-factor decomposition attributes −0.3 to composition and −0.5 to allocation, with zero reconstruction residual; no counterfactual is inferred by the recipe.</p>
<p>{/* lens:addSegregationDecompositionRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation-decomposition-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation-decomposition-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}</p>
<h2 id="addsegregationcounterfactualrecipe"><code>addSegregationCounterfactualRecipe</code></h2>
<p>Empirical-quantile population counterfactuals and Shapley components. This executes a synthetic fixture and preserves status independently from numerical output.</p>
<p>{/* lens:addSegregationCounterfactualRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=graph caption="Follow graph-native construction and execution."}</p>
<h2 id="addpopulationsupportalignmentrecipe"><code>addPopulationSupportAlignmentRecipe</code></h2>
<p>Explicit population support identifiers reordered with completeness evidence. This executes a synthetic fixture and preserves status independently from numerical output.</p>
<p>{/* lens:addPopulationSupportAlignmentRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}</p>
<p>::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=graph caption="Follow graph-native construction and execution."}</p>`};export{e as default};