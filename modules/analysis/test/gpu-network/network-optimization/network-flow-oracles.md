# Reproducing the flow oracle records

Run from the checkout root, using the installed offline environment:

```sh
/usr/bin/python3 modules/analysis/test/gpu-network/network-optimization/generate-network-flow-oracles.py --output-directory .typecheck/r19/scratch/network-construction-and-flow/oracles
```

The verified environment is Python 3.9.6, NetworkX 3.2.1, SciPy 1.13.1 and NumPy 2.0.2.
The script enforces the NetworkX and SciPy versions and prints all four actual versions.
It writes fresh records to the requested scratch directory, then compares parsed values and exact
JSON bytes against all three committed fixtures. It refuses to overwrite the fixture directory.
No GPU, repository flow implementation or CPU twin participates in generation.

- `network-flow-fan-oracle.json`: deterministic 100,000-node unit-capacity directed fan.
  Source 0 connects to vertices 1 through 99,998, each connecting to sink 99,999.
  Edge rows list all source-to-leaf edges, then all leaf-to-sink edges. There is no seed.
- `network-flow-grid-oracle.json`: deterministic 250-row by 400-column unit-capacity directed grid.
  Vertices are row-major; each lists its right neighbor, then its down neighbor, when present.
  Source is 0 and sink is 99,999. There is no seed.
- Both large records use SciPy's Dinic maximum flow. The generator derives the cut by traversing
  positive residual capacities, including reverse arcs, and independently sums its capacity.
- `network-flow-primal-dual-oracle.json`: explicit integer edge tables in
  `network-flow-primal-dual-oracle.inputs.json` construct each graph. This separate input manifest
  contains no expected values. The first three are the named hand-built cases. The nine `random-*`
  tables retain the original record's seed, 93703, as provenance. The historical random generation
  algorithm was not checked in; regeneration replays those exact tables and does not claim to
  reconstruct them from the seed. Neither a new seed nor a guessed random algorithm is used.
  Every edge row is a NetworkX `MultiDiGraph` edge with its row number as the multiedge key,
  preserving parallel edges, antiparallel edges and self-loops. A capacity-aggregated `DiGraph`
  establishes maximum value; `network_simplex` solves the keyed graph at that value or the smaller
  finite request. The generator checks every keyed capacity, node balance and objective before
  emitting the record. `0xffffffff` requests maximum flow; other requests retain any unserved
  amount explicitly. Nonunique optimal edge assignments are allowed; these records store values
  and objectives rather than pinning one assignment.

Fresh output from the command above matched all three committed files byte for byte on 2026-11-19.
The tiny reverse-cancellation case additionally has an exhaustive, dependency-free enumeration of
all 256 integral edge assignments in `network-flow-integral-enumeration.test-utils.ts`, exercised
by the Node and GPU flow specs. Its complete optimum has value 2 and cost 4; the finite unit request
has cost 1 and cannot certify a minimum cut because a second unit remains feasible.
