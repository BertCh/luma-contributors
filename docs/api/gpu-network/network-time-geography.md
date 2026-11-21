# Network time geography

This is the canonical task guide. The [legacy URL](./network-time-geography-mob09.md) remains available.

Import from `@luma-contributors/analysis/gpu-network`.

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

Pass both `forwardComplete` and `reverseComplete` from the corresponding reachability
`status.converged` views. Either supplied zero certificate suppresses prism memberships, slack
and summary counts. Optional `output.status` contains one-word `complete` and `invalidInput` views;
malformed resident CSR offsets also suppress the result. Omitted optional tree evidence is still
accepted, so the caller remains responsible for supplying complete trees. The queued three-node
chain regression toggles each certificate independently, then restores both; it has not been run
on a GPU in this pass.

Every encoding resets edge output capacity before writing the live CSR ranges. Slots outside
those ranges have `edgeSlack = -Infinity` and `edgeInPrism = 0`, including slots that were live
on the previous submission. The summary counts only live memberships after that reset.

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

CPU and GPU ranks use the same exact nearest-rank host calculation for the shortest round-trip
decimal `String(p)`, without an epsilon. Thus 99.9% of 1,000 CPU departures selects rank 999;
the 1,000-departure example exceeds the GPU departure cap. For travel times `[5, 90]`, percentiles immediately below 50%, at
50%, and immediately above 50% select `[5, 5, 90]`. Integer boundaries stay exact: 56% of 25
departures selects rank 14, and the next representable percentile above 56% selects rank 15.

Percentiles and output shapes are captured when a contributor or operation is constructed;
later changes to the caller's percentile array do not change that computation. Every percentile
slot must contain a finite number in `(0, 100]`; sparse lists are rejected with a named error.
Declared properties are read by name, so inherited getters (including a zero routing budget)
have the same meaning as own properties. Caller-owned GPU views retain their identities.
Both backends honor `maximumRelaxations` for each departure (default: the number of states).
At zero budget, the routing incumbent reaches only the origin and leaves other travel times at
`+Infinity`, with `[valid, converged] = [1, 0]` per departure. When `output.status` is requested,
percentile aggregates of those incumbents remain available; inspect every departure certificate
before treating them as final. When status is omitted, any invalid or unconverged departure
suppresses **every** percentile travel-time row with NaN and clears every requested
`reachedCounts` row to zero. A suppressed zero count does not establish that no departure reached
the vertex. For three departures from vertex 1 at zero sweeps, requested status retains counts
`[0,3,0]`; omitting status gives `[0,0,0]` and all-NaN travel times. The pure CPU function always returns
status and incumbents; the CPU graph operation follows the requested-status suppression rule.

The percentile rule is nearest rank. r5r's exact index rule was not checked against its source, so
for small windows its percentiles may differ by one rank.

GPU declaration preflight runs before scratch allocation: 1–256 states, 0–4,096 transitions,
2–256 samples per piecewise transition, 0–65,536 sweeps per departure (default state count),
1–60 departures and 1–16 ascending percentile levels. Affine routing does not take sample controls.
These are per-window/per-departure limits, not total work across independent windows. The CPU twin
drops those GPU dimension caps, including the departure and percentile caps, while an explicit
`maximumRelaxations` still bounds each departure's sweeps.

Reference tier: each departure uses one cooperative 64-lane workgroup, committing transitions in
input order with a barrier per transition. Each departure also has a separate absolute-arrival
publication dispatch: a window declares `2 + 2 * departureCount` nodes, including window preparation
and final percentile publication. Three departures therefore declare eight nodes.
`estimatedCommandNodeCount` matches this declaration; it is not an execution-time or
dispatch-completion certificate. Arbitrary departure lists are not supported. This describes the
implementation; fresh GPU qualification is pending.

Reached counts include only finite elapsed labels; a disconnected vertex keeps an infinite
travel time and receives no reached departure. Finiteness is classified from float32 exponent bits.

Verified against a binary64 FIFO Dijkstra with affine arrival `t + base + slope * t` (NetworkX has
no time-dependent routing), and, for slope 0, against NetworkX distances.

{/* lens:GPUNetworkTravelTimePercentiles */}

## Clock units and rebasing

Percentiles are **percentages** in `(0, 100]`: write `95` for the 95th percentile, rather than the
probability fraction `0.95` (which requests the 0.95th percentile). Departures, window step, costs,
arrival samples and travel-time outputs use one consistent caller-selected time unit.

To rebase an absolute clock by `origin`, use `t′ = t - origin`. For affine arrival
`A(t) = t + base + slope * t`, use `base′ = base + slope * origin` and preserve `slope`:
`A′(t′) = t′ + base′ + slope * t′`. For example, `[base, slope] = [10, 0.5]`
at departure 100 becomes `[60, 0.5]` at departure 0 and preserves travel time 60.
For piecewise arrival samples `(departure, arrival)`, subtract
`origin` from **both** columns. Shift the window's first departure by the same origin; its step
is unchanged. Choose an origin that preserves nonnegative departures and admissible FIFO samples.
The GPU solve uses elapsed labels and publishes absolute arrivals separately, so rounding an
absolute arrival does not choose its predecessor. The new arithmetic's GPU regression remains queued.

Pure CPU routing and percentile results include `unrepresentable`: one uint32 word per routing
state or per vertex-major selected-percentile row, respectively. It is 1 when a finite internal
float64 value overflows its float32 output payload. Two `3e38` edges can reach a destination whose
payload is Infinity and mask is 1; disconnected Infinity has mask 0. Reached counts and convergence
are separate evidence. Both CPU graph operations accept optional `output.unrepresentable`; the GPU
operations reject that CPU-only port.
