var e={slug:`api/gpu-network/visibility-graph`,title:`Visibility graph and network centrality`,html:`<h1 id="visibility-graph-and-network-centrality">Visibility graph and network centrality</h1>
<h2 id="addvisibilitygraphrecipe"><code>addVisibilityGraphRecipe</code></h2>
<p>CSR visibility supports budgeted exact all-source metrics or deterministic sampled estimates via
<code>metricsMode: 'auto' | 'exact'</code>. Auto samples when the exact command budget cannot fit; sampled maximum depth
is a lower bound. <code>status.approximation</code>, overflow and convergence are separate statuses; explicit exact
rejects an insufficient budget.</p>
<p>{/* lens:addVisibilityGraphRecipe */}</p>
<p>The <code>@luma-contributors/analysis/gpu-network</code> recipe composes indexed visibility with
batched pivot centrality in <code>mode: 'csr'</code>. Three collinear observers at unit spacing
and a visibility radius of 1.1 form a path: degree is <code>[1, 2, 1]</code>, betweenness is
<code>[0, 1, 0]</code>, and closeness is <code>[2/3, 1, 2/3]</code>. The wall lies outside their sight lines.
All three observers are pivots, so approximation is zero. Four directed CSR slots
store the two undirected links without overflow.</p>
<p>The spatial <a href="/luma-contributors/docs/api/gpu-spatial-analysis/urban-ecology-lenses">urban-morphology recipe</a>
provides adjacency and degree only in CSR mode; its dense reference mode retains
all-source metrics. Import the network recipe for CSR betweenness, pivot closeness
and reached counts. With fewer pivots, betweenness is scaled by node count divided
by pivot count, closeness follows pivot order, and approximation is one. Capacity,
candidate overflow, invalid input and convergence remain separate status fields.</p>
<p>::lens{scenario=gpu-network.visibility-graph-recipe rung=result caption="Inspect CSR degree, pivot centrality and completeness status."}</p>
<p>::lens{scenario=gpu-network.visibility-graph-recipe rung=graph caption="Follow indexed visibility, symmetric CSR construction and batched network searches."}</p>`};export{e as default};