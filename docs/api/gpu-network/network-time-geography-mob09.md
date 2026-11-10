# Network time geography

Two contributors on top of the routing already in the package.

## `GPUNetworkPotentialPathArea`

The network potential path area of a space-time prism: the nodes and directed edges a traveller
who leaves `a` and must be at `b` within a budget could have used. Inputs are the two trees:
`forwardCosts` (cost from `a`, a `GPUNetworkReachability` run) and `reverseCosts` (cost to `b`, the
same contributor on the reversed network), plus the per-frame `budget` row.

- An edge `(u, v)` is in the prism when `forward[u] + weight + reverse[v] <= budget`; `edgeSlack` is
  the difference. A node is in it when `forward[v] + reverse[v] <= budget`.
- The edge is whole: a directed edge is entered at `u` and left at `v`, so every point on it has
  the same slack. Two directed edges make an undirected street. Clipping partial edges applies to
  anchors in the middle of an edge, which is not supported.
- `output.summary` is `[nodes in the prism, edges in the prism]`.

Verified against NetworkX 3.7 `single_source_dijkstra_path_length` on the network and its reverse
(three random directed networks, fixture `network-time-geography-mob09.json`), with the trees
computed both by NetworkX and by two `GPUNetworkReachability` runs.

{/* lens:GPUNetworkPotentialPathArea */}

## `GPUNetworkTravelTimePercentiles`

Travel-time percentiles over a departure window, the analogue of r5r `time_window`. The window is
`window = [first departure, step]` with `departureCount <= 60`. Each departure runs one
`GPUNetworkTimeRouting` (FIFO earliest arrival on the restriction automaton) in the same graph. A
vertex's travel time is its earliest arrival over its states minus the departure; per vertex the
values are sorted and the `p`-th percentile is the value of rank `ceil(p / 100 * n)` (nearest rank).
A departure that cannot reach the vertex counts as `+Infinity`. `output.reachedCounts` and
`output.status` (`[valid, converged]` per departure) say how complete the numbers are.

The percentile rule is nearest rank. r5r's exact index rule was not checked against its source, so
for small windows its percentiles may differ by one rank.

Reference tier: each routing is a single invocation (at most 256 states and 4096 transitions), so
a window costs `departureCount` such dispatches, and arbitrary departure lists are not supported.

Verified against a binary64 FIFO Dijkstra with affine arrival `t + base + slope * t` (NetworkX has
no time-dependent routing), and, for slope 0, against NetworkX distances.

{/* lens:GPUNetworkTravelTimePercentiles */}
