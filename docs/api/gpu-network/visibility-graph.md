# Visibility graph and network centrality

## `addVisibilityGraphRecipe`

CSR visibility supports budgeted exact all-source metrics or deterministic sampled estimates via
`metricsMode: 'auto' | 'exact'`. Auto samples when the exact command budget cannot fit; sampled maximum depth
is a lower bound. `status.approximation`, overflow and convergence are separate statuses; explicit exact
rejects an insufficient budget.

{/* lens:addVisibilityGraphRecipe */}

The `@luma-contributors/analysis/gpu-network` recipe composes indexed visibility with
batched pivot centrality in `mode: 'csr'`. Three collinear observers at unit spacing
and a visibility radius of 1.1 form a path: degree is `[1, 2, 1]`, betweenness is
`[0, 1, 0]`, and closeness is `[2/3, 1, 2/3]`. The wall lies outside their sight lines.
All three observers are pivots, so approximation is zero. Four directed CSR slots
store the two undirected links without overflow.

The spatial [urban-morphology recipe](../gpu-spatial-analysis/urban-ecology-lenses.md)
provides adjacency and degree only in CSR mode; its dense reference mode retains
all-source metrics. Import the network recipe for CSR betweenness, pivot closeness
and reached counts. With fewer pivots, betweenness is scaled by node count divided
by pivot count, closeness follows pivot order, and approximation is one. Capacity,
candidate overflow, invalid input and convergence remain separate status fields.

::lens{scenario=gpu-network.visibility-graph-recipe rung=result caption="Inspect CSR degree, pivot centrality and completeness status."}

::lens{scenario=gpu-network.visibility-graph-recipe rung=graph caption="Follow indexed visibility, symmetric CSR construction and batched network searches."}
