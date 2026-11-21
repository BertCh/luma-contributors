# Network analysis from an edge list to a trustworthy answer

## Summary

The CPU twins agreed with independent references for ordinary shortest-path costs, OD costs, facility allocation, centrality, strong components, integral flow and map matching. The most serious failures appeared **between** operations.

A converged shortest-path search followed by path extraction returned the wrong parallel road. The reported route cost was `0.3`, while its published edge IDs described a route costing `9.1`.

The accessibility example seeds both ends of a one-way street before searching the reverse graph. The path example omits the convergence input that prevents a truncated predecessor chain from looking like a successful route. Both examples survive in the staging copy.

The graph-construction instructions omit a precondition of the recommended upstream converter. An ordinary unsorted two-edge list can change a shortest-path cost from `5` to `3`.

The most valuable change is **one executable, asymmetric road-network walkthrough** that starts with external edge IDs and finishes with a decoded route, OD matrix and closest facility. Use a one-way street, parallel edges, a reordered facility list and an isolated vertex. These four features expose errors that a symmetric street grid hides.

## Census

`S` means `modules/analysis/src/gpu-network`, `D` means `docs/api/gpu-network`, and `KI` means the newer copy under `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`. Line numbers without a `KI` qualifier refer to r19.

I bundled the public source barrel with `node_modules/.bin/esbuild --bundle --platform=node --format=esm --loader=ts --log-level=warning` and piped it to `node --input-type=module -`. I counted exported constructors by prototype, inspected their constructor-prop declarations, and counted actual `new ClassName(` expressions in TypeScript fences. The documentation counts cover the network directory, including specialized guides.

| Measured variant | Count | How counted |
|---|---:|---|
| Public GPU command-node producers | 41 | Exported constructor with `prototype.getCommandNodes` |
| Additional GPU operation classes / CPU operation classes / pure CPU functions | 17 / 17 / 17 | Public exports ending in `Operation`, or matching `^compute.*OnCPU$` |
| Producers requiring a top-level `topology` | 22 | Constructor props, including the inherited `GPUNetworkCentralityScheduleStep` props |
| Producers taking a conditional or optional top-level `topology` | 2 | `GPUNetworkLouvain`, `GPUNetworkFlow` |
| Producers with no top-level `topology` | 17 | Named under **Departures by name** |
| Producers constructed in a network-guide TypeScript fence | 19 of 41 | `new ClassName(` across 14 r19 pages; also 19 across 15 KI pages |
| Pure CPU functions named anywhere / called in a TypeScript fence | 4 of 17 / 0 of 17 | Same result in r19 and KI |
| Public `decode*` functions | 4 | Three accept `ArrayLike<number>` words; one checks typed-array formats |

### The nine first-use journeys

I used the directed road `0 → 1` costing `2`, `1 → 2` costing `3`, and isolated vertex `3`, unless noted. `u` and `f` below construct `Uint32Array` and `Float32Array`. I compared the public functions with `/usr/bin/python3` heap-based shortest paths, hand-enumerated flow and projection, and installed NetworkX 3.2.1 for the centrality comparison.

| Journey | Code suggested by the guide or its vocabulary | Observed answer or obstacle |
|---|---|---|
| Build from an edge list | `new GPUCOOToCSR({rowIndices, columnIndices, values, ...outputs, rows})` | Requires source-sorted rows and a shared permutation of all edge columns. A transpose needs its own ordering. UNW-04. |
| Shortest paths from sources | `computeNetworkReachabilityOnCPU({topology, startNodes: u(2, 0)})` | Costs `[0,2,0,∞]`, matching the minimum over sources. The result has one cost per node. |
| Origin–destination costs | `computeNetworkCostMatrixOnCPU({topology, startNodes: u(2,0), targetNodes: u(1,0)}, {rowCount: 2})` | Row-major `[∞,∞,2,0]`, matching Python. Column zero denotes **target-list row zero**, vertex `1`. A target ID outside the graph also produces `∞` with `converged=1`. |
| Service areas | `computeNetworkServiceAreasOnCPU({topology, facilities: u(2,0)})` | Assignments `[1,1,0,NONE]`, costs `[0,2,0,∞]`, counts `[1,2]`. Assignment `1` denotes the facility at vertex `0`. |
| Closest facility | Same call with `topology: reverseTopology` | Assignments `[1,0,0,NONE]`, costs `[0,3,0,∞]`. At intersection `1`, the nearest facility you **can drive to** differs from the nearest facility that **can drive to you**. |
| Centrality | `computeNetworkCentralityOnCPU({topology, reverseTopology, pivotNodes: u(0,1,2,3)})` | Betweenness `[0,1,0,0]`. Pivot `0` has outward closeness `2/7`, harmonic closeness `0.7`. Both match weighted NetworkX. |
| Connected components | `computeNetworkStructureOnCPU({offsets, neighbors})` | Strong labels `[0,1,2,3]`. Python weak labels `[0,0,0,3]`. The guide’s weak-label GPU entry is `GPUNetworkAnalyticsColumns`; `GPUNetworkStatistics` gives a summary. |
| Flow | `const p = prepareGPUNetworkFlowDemands(problem); computeNetworkFlowOnCPU(p, p)` | One arc `0→1`, capacity `2`, unit cost `3`, net demands `[-2,2]`: expanded edge flows `[2,2,2]`, exact objective `[6,0]`, unserved `0`. `summary[5]` reads as the integer `1086324736` until its bits are interpreted as float32. |
| Map matching | `computeMapMatchingOnCPU({topology, positions, trackOffsets, parameters}, {cellSize: 2})` | Road `(0,0)→(10,0)`, fixes `(2,0),(8,0)`, `sigma=beta=1`: edges `[0,0]`, fractions `[0.2,0.8]`, breaks `[0,0]`, log-likelihood `−1.837877`, agreeing with projection and `−log(2π)`. |

`NONE` above is `0xffffffff`. A completed bounded search establishes reachability **within its cutoff**. The OD probe also shows why an infinity value cannot validate an external-ID-to-row mapping.

## Proposed convention

Every public network workflow should have an executable example that starts with identified, directed edges, names the cost unit and travel direction, and decodes the answer back to the input IDs. A documentation check should run the CPU example and declare the GPU composition, then test a one-way edge, parallel edges, an isolated node and an exhausted budget against explicit expected outputs.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UNW-01 | W | ORDINARY | BREAKING | S | CPU path extraction | A cheaper parallel edge disappears when rounded costs are used to recover its identity. |
| UNW-02 | W | ORDINARY | ADDITIVE | S | Snapping → reverse OD example | The example allows travel against a one-way street. |
| UNW-03 | W | ORDINARY | ADDITIVE | S | Reachability → path example | An exhausted predecessor search becomes a successful one-vertex route. |
| UNW-04 | W | ORDINARY | ADDITIVE | S | Edge-list construction | The recommended COO converter requires sorted rows; the guide omits that step. |
| UNW-05 | W | ORDINARY | BREAKING | S | Three readback decoders | The type accepts raw readback bytes and silently treats each byte as a word. |
| UNW-06 | W | ORDINARY | ADDITIVE | S | Service areas / closest facility | The same “nearest facility” wording covers two different directed answers. |
| UNW-07 | W | ORDINARY | ADDITIVE | S | Statistics readback | A valid nondefault histogram can be silently decoded at the wrong stride. |
| UNW-08 | W/S | EXTREME | ADDITIVE | M | Path compaction | The GPU accepts an unrepresentable total; the CPU rejects a representable bounded one. |
| UNW-09 | W | UNUSUAL | ADDITIVE | S | Flow-demand preparation | A missing endpoint becomes vertex zero and can produce a certified feasible flow. |
| UNW-10 | D | ORDINARY | ADDITIVE | S | Centrality example | The sampled estimate in the example reports `approximation=0` if that flag is requested. |
| UNW-11 | D | ORDINARY | ADDITIVE | S | NetworkX cross-reference | Both printed reference calls omit the edge-weight argument. |
| UNW-12 | C | UNUSUAL | ADDITIVE | S | Map-matching parameters | The public packer accepts impossible noise scales and can produce input its own twin rejects. |
| UNW-13 | E | ORDINARY | ADDITIVE | M | Centrality preparation | Closeness alone requires a transpose; the missing-input error names the supplied topology. |
| UNW-14 | E/C | ORDINARY | ADDITIVE | S | Network front page / CPU structure | Zero of 17 CPU functions have a fenced first run, and the obvious SCC call fails. |
| UNW-15 | E | ORDINARY | ADDITIVE | S | Facility, K and SCC limits | Three important boundaries have three different preflight experiences. |

### UNW-01: A certified path can name a road that costs thirty times too much

**Where.** `S/network-analysis/cpu-network-path-extraction-operation.ts:193`, especially the equality and fallback at lines 199–204. The search rounds its published distances in `S/network-reachability/cpu-network-reachability-operation.ts:153`. The existing fallback assertion is in `modules/analysis/test/gpu-network/network-analysis/cpu-network-path-extraction-operation.node.spec.ts:84`.

**Evidence — EXECUTED.** Use `offsets=[0,1,3,3]`, `neighbors=[1,2,2]`, `weights=[0.1,9,0.2]`. Search from `0`, then pass `costs`, `predecessors` and `converged` directly to extraction for target `2`. Result: `pathFound=[1]`, edge slots `[0,1]`, cost `0.3000000119`, candidate and edge overflow both zero. Slots `[0,1]` cost approximately `9.1`. Python, `computeKShortestPathsOnCPU` and `computeAlternativeRoutesOnCPU` all select `[0,2]`. The staged reachability twin produced the same failing composition.

**Now.** `costs[tail] + weights[edge] === costs[head]` compares a binary64 sum with an already rounded float32 label. The fallback chooses the first parallel slot, regardless of impedance.

**Wanted.** For a parent–child pair from a shortest-path forest, select the minimum passable parallel weight, breaking equal weights by slot. For exact cross-backend provenance, allow the search to publish `predecessorEdges` and the extractor to consume it directly. Pin the three-edge example end to end.

**In-repository callers that change.** Zero signature edits for the resolver correction. `rg` found 18 direct calls to the CPU function, including its operation lowerer and tests. The explicitly documented smallest-slot fallback changes meaning; update the fallback expectation and comments. **Value and cost.** A route can be used for road-level joins, tolls or closures with a trustworthy edge identity. S for the correction.

### UNW-02: The reverse OD example admits an illegal trip; the directed mode also loses exact-intersection starts

**Where.** `D/operators.md:683`, KI `D/operators.md:705`; `S/network-accessibility/gpu-network-snapping.ts:176,687,744,794`. Compare the endpoint handling in `S/recipes/drive-time-catchment-recipe.ts:684`; the KI recipe also handles coincident endpoints.

**Evidence — SOURCE for snapping, EXECUTED for downstream routing.** The example omits `startDirection`, whose default enables both endpoints. For a job halfway along one-way `0→1`, cost `10`, the published seed formula gives `[0,1]` at costs `[5,5]`. I fed those seeds to the real CPU search on the transpose. It returned `[5,5]`; the legal costs **to the job** are `[5,∞]`. Matrix convergence cannot detect an illegal seed.

There is a second case. At the source intersection of `0→1` costing `10` and `0→2` costing `1`, snapping to the first slot with `'forward'` emits only node `1`, cost `10`. The executed downstream costs were `[∞,10,∞]`; starting at the intersection gives `[0,10,1]`.

**Now.** `new GPUNetworkSnapping({positions: jobs, topology, output})` followed by `new GPUNetworkCostMatrix({topology: reverseTopology, ...})`.

**Wanted.** Set `startDirection: 'reverse'` in the one-way **to-job** example. Add an intersection-aware seed option that emits a zero-cost seed for a coincident endpoint in either search direction. State that `'both'` assumes the selected road’s cost applies in both directions. The smallest GPU confirmation is one midpoint on one one-way edge, followed by the two-edge intersection fixture; read the two seed slots and the matrix row.

**In-repository callers that change.** None required for an additive endpoint option. The network cost-matrix lens explicitly uses `'forward'` and is a concrete adopter at `modules/lenses/src/computation-lenses/scenarios/gpu-network/network-cost-matrix.example.ts:38`. **Value and cost.** One-way accessibility and intersection-origin routing become reproducible. S.

### UNW-03: The path example discards its own search certificate

**Where.** `D/operators.md:244–249`, KI `D/operators.md:251–255`; `S/network-analysis/cpu-network-path-extraction-operation.ts:165`. The paragraph immediately above the example describes the required wiring.

**Evidence — EXECUTED in both trees.** On `0→1→2→3` with zero weights, start `0`, `localIterations=1`, `maximumTieIterations=1`, reachability returned costs `[0,0,0,0]`, `converged=0`, `unresolvedCount=2`. Extracting target `3` exactly as composed on the page returned `pathFound=1`, nodes `[3]`, `candidateOverflow=0`. Adding `predecessorsComplete: search.converged` returned `pathFound=0`, nodes `[]`.

**Now.** `output: {costs, predecessors}` feeds an extractor that treats an unqualified `NONE` parent as a root.

**Wanted.** Request `output.status.converged`, bind it to `predecessorsComplete`, and show that an incomplete search yields an unresolved answer. Test the **printed composition**. This adds a distinct failing path example to KNOWN(NET-101), which concerned matrix consumers.

**In-repository callers that change.** None; one guide example in both copies. **Value and cost.** Prevents displaying the destination alone as a successful itinerary. S.

### UNW-04: “Build CSR with `GPUCOOToCSR`” leaves out the sort that determines the graph

**Where.** `D/operators.md:37,83`, KI `D/operators.md:40,85`; `D/concepts.md:66`. The installed upstream implementation states “row-sorted” at `node_modules/@luma.gl/graph/src/gpu-core/gpu-coo-to-csr.ts:27`, copies columns and values at line 77 and binary-searches row indices at line 119. An in-repository preparation example already sorts directed edges in `modules/lenses/src/computation-lenses/scenarios/gpu-network/construction-lens-support.ts:98`.

**Evidence — SOURCE for conversion, EXECUTED for its consequence.** Edge list in arrival order `[(1,2,3),(0,1,2)]`. Python transcribed the converter’s lower bounds on `[1,0]` as `[0,2,2,2]`. The real CPU twin on those offsets and unchanged columns returned `[0,2,3]` from node `0`. After sorting the entire edge records, the twin and Python Dijkstra both returned `[0,2,5]`. The smallest GPU check is these two COO entries; read the four offsets.

**Now.** Swapping COO source and target columns to construct the transpose also invalidates the original source sort. Casting application IDs straight to `Uint32Array` is independently unsafe: I executed starts `0` and `4294967296`; they became the same start.

**Wanted.** Show external-ID-to-dense-row mapping, stable source sort, permutation of weights **and original edge IDs**, then an independent sort by reversed source. Preserve isolated vertices through an explicit vertex list. A small host preparation helper can return forward and reverse CSR plus the two slot-to-edge mappings. This gives KNOWN(DIR-103) a concrete ingestion contract and regression fixture.

**In-repository callers that change.** None. **Value and cost.** A GIS edge table can reach routing without acquiring new connectivity or losing feature identity. S for the example, M for a reusable helper.

### UNW-05: Three decoders accept the byte buffer returned by readback and produce plausible counts

**Where.** `S/network-statistics/gpu-network-statistics.ts:148`, `S/network-coarsening/gpu-network-coarsening.ts:112`, `S/network-subgraph-filter/network-subgraph-filter-parameters.ts:107`. The fourth decoder, `S/k-shortest-paths/k-shortest-paths-output.ts:119`, checks the scalar type.

**Evidence — EXECUTED.** I encoded four test words `[300,17,3,9]` in correctly sized buffers and passed `new Uint8Array(words.buffer)` to each decoder. All three returned the first four fields as `[44,1,0,0]`. Passing the word view returned `[300,17,3,9]`. The subgraph decoder also accepted `Uint32Array.of(3)` and returned three `undefined` fields declared as numbers.

**Now.** `decodeGPUNetworkStatistics(await buffer.readAsync())` is type-compatible. The statistics example correctly reinterprets the bytes, but the signature allows callers to omit that step.

**Wanted.** Accept `Uint32Array | readonly number[]` for word decoders and check the required length. Provide an explicit byte-readback entry that respects `byteOffset` and `byteLength`. Use the same convention for all three.

**In-repository callers that change.** `rg` found 24 calls across the three names. The showcase helper explicitly returns `Uint32Array`, and the tests include ordinary number arrays. No raw-byte migration was identified. Narrowing the published `ArrayLike<number>` signature is **breaking**. **Value and cost.** Prevents silently reporting 44 roads or vertices where the device wrote 300. S.

### UNW-06: “Nearest facility” needs a direction and two row identities

**Where.** `D/operators.md:240,317,975`, KI service-area description at `D/operators.md:246`; `S/network-analysis/gpu-network-service-areas.ts:76,125`; `S/network-analysis/cpu-network-service-areas-operation.ts:69`. I also checked the KI service-area twin.

**Evidence — EXECUTED in both trees.** Facilities `[2,0]` on the asymmetric fixture assign intersection `1` to facility **row 1**, cost `2`, using forward CSR. Reverse CSR assigns it to **row 0**, cost `3`. Python independently reproduced both. These correspond to dispatching a vehicle from a station and driving a patient to a station, respectively.

**Now.** `facilities[result.assignments[1]]` is the facility’s vertex row. `result.assignments[1]` alone is the facility-table row. Either can be a valid-looking array index.

**Wanted.** Put “from facilities” and “to facilities” in two adjacent worked examples. Decode `facilityVertex = facilities[assignment]` only after checking the sentinel and convergence. Label the drive-time recipe’s illustrated direction **facility to demand**. The earlier review notes transpose semantics; the new evidence is the documented service-area journey selecting a different named facility on the same input.

**In-repository callers that change.** None. **Value and cost.** Closest-facility users can choose the correct hospital or dispatch station on their first run. S.

### UNW-07: Statistics can silently shift one histogram into another

**Where.** `S/network-statistics/gpu-network-statistics.ts:77,148,312,345`; `D/operators.md:408`, KI `D/operators.md:422`.

**Evidence — EXECUTED.** For the graph `0→1, 0→2`, I encoded its degree histograms in a valid 64-bin layout. Calling `decodeGPUNetworkStatistics(words)` gave in-degree bins `0..2 = [0,0,0]` and total-degree bins `[1,2,0]`. With `{degreeBinCount:64}`, the answers were `[1,2,0]` and `[0,2,1]`. The producer requires exactly `16 + 3*bins` words. The decoder defaults to 32 and checks only a lower length bound. `{degreeBinCount:NaN}` returned three empty histograms.

**Now.** Changing the contributor’s bin count while retaining the documented decoder call can change the meaning of every histogram after the first.

**Wanted.** Add a strict exact-view reader that infers `(words.length - 16) / 3` and checks the constructor’s integer range `1..4096`. Show it in the example. Keep an explicit layout for padded buffers. Validate explicit layout values in the existing reader.

**In-repository callers that change.** None required for an additive reader; 16 existing calls to the current decoder. **Value and cost.** A histogram either decodes at its producing stride or gives a layout error. S.

### UNW-08: Path extraction applies opposite size rules on the two backends

**Where.** `S/network-analysis/cpu-network-path-extraction-operation.ts:104,233`; `S/network-analysis/gpu-network-path-extraction.ts:354,420`; `S/network-analysis/network-path-extraction-passes.ts:205`.

**Evidence — EXECUTED host; SOURCE device.** With 65,536 vertices, 65,536 targets and `maximumPathLength:1`, the CPU function threw `targets.length * predecessors.length must be at most 4294967295`. At most 65,536 rows can be emitted. The GPU contributor declared eight nodes on a null device. With `maximumPathLength:65536`, it **also** declared eight nodes. If every target requests the last vertex of a full chain, the true node total is exactly `2^32`. The scan and `lastStart + lastLength` are uint32.

**Now.** The CPU rejects a modest real output, while a GPU total can wrap before final-capacity reporting. No device result is claimed.

**Wanted.** Use `activeTargetUpperBound * min(nodeCount, maximumPathLength)` for the inexpensive safe case. For larger products, use a checked or wide sum before publishing counts and offsets; expose unrepresentability explicitly. Test the GPU total-publication step with `lastStart=4294901760`, `lastLength=65536`. A four-billion-step device run is unnecessary.

**In-repository callers that change.** None for checked publication and accepting the safe CPU case. **Value and cost.** Large OD route batches get consistent admission and reliable required capacity. M.

### UNW-09: A malformed flow edge can manufacture a feasible shipment

**Where.** `S/network-optimization/prepare-flow-demands.ts:62,66,69,162`.

**Evidence — EXECUTED.** For two vertices, `edges:[[1]]`, `capacities:[2]`, `costs:[3]`, `demands:[2,-2]`, the preparer emitted edges `[1,0, 0,3, 2,1]`. The CPU solver returned `valid=1`, `complete=1`, flows `[2,2,2]`, exact objective `6`. The input supplied no destination for its sole road. `edge.every(...)` visits only present entries; conversion to uint32 turns the missing value into zero.

**Now.** A TypeScript tuple protects authored literals. Imported JSON and JavaScript tables still pass through this public host adapter.

**Wanted.** Before flattening, require exactly two present integer endpoints, and include the edge row in the error: `prepareGPUNetworkFlowDemands edges[0] must contain [source, target]`. Test a short tuple, a sparse tuple and a valid self-loop.

**In-repository callers that change.** None with valid inputs. **Value and cost.** Invalid network data fails at ingestion instead of authorizing a shipment over an invented arc. S.

### UNW-10: The centrality example computes a sampled estimate without marking it sampled

**Where.** `D/operators.md:124,164`, KI `D/operators.md:173`; `S/network-centrality/gpu-network-centrality.ts:214,648`; `S/network-centrality/cpu-network-centrality-operation.ts:230`, KI line 273.

**Evidence — EXECUTED in both trees.** One pivot `[0]`, `betweennessScale:4`, and the example’s omission of `sampling` gave betweenness `[0,4,0,0]`, `approximation=0`, `converged=1`. All four pivots gave `[0,1,0,0]`. Supplying four copies of pivot zero also produced `[0,4,0,0]` with `approximation=0`. Thus pivot count alone cannot certify full coverage.

**Now.** `approximation` echoes a caller label. The sample sets the estimate’s scale but leaves the default label `'exact'`.

**Wanted.** Set `sampling:'sampled'` alongside `betweennessScale` and request `status.approximation`. Describe the word as declared sampling provenance. Reserve a whole-network exactness claim for a producer or host helper that supplies every vertex once.

**In-repository callers that change.** None. **Value and cost.** Consumers can distinguish a converged estimate from all-pivot betweenness. S.

### UNW-11: The documented NetworkX oracle computes hop centrality

**Where.** `D/operators.md:131–135`, KI `D/operators.md:137`; `S/network-centrality/gpu-network-centrality.ts:256`. The checked-in oracle already supplies the missing keyword at `modules/analysis/test/oracles/network-centrality.py:214,218`.

**Evidence — EXECUTED.** On the weighted fixture, the two printed NetworkX calls returned closeness `0.6666667`, harmonic `1.5`. Adding `distance='weight'` returned `0.2857143` and `0.7`, matching the CPU twin.

**Wanted.** Print `nx.closeness_centrality(G.reverse(), u=p, distance='weight', wf_improved=False)` and `nx.harmonic_centrality(G.reverse(), nbunch=[p], distance='weight')[p]`.

**In-repository callers that change.** None; correct the guide and TSDoc. **Value and cost.** A NetworkX user can reproduce both metrics without diagnosing a nonexistent algorithm discrepancy. S.

### UNW-12: The map-matching packer can emit settings its own twin rejects

**Where.** `S/map-matching/gpu-map-matching.ts:158,180–193`; `S/map-matching/cpu-map-matching-operation.ts:311,333`; `D/operators.md:1374`.

**Evidence — EXECUTED.** `getGPUMapMatchingParameterValues({sigma:1e40,beta:1,searchRadius:4})` produced `Infinity` in word zero. Passing its output directly to the twin threw `parameters settings must be finite`. With `sigma:-1,beta:0`, packing succeeded and the two-fix match reported likelihood `39.608654` and two matched fixes. Both scales are documented as positive. This is a different packer and failure from KNOWN(NET-117).

**Wanted.** Validate the packed float32 values once in the public packer. Require finite positive `sigma` and `beta`, and finite nonnegative search radius and routing allowances. Name the setting in each error. Retain the runtime kernel’s defensive handling of caller-written buffers.

**In-repository callers that change.** None with valid settings. **Value and cost.** Invalid model calibration fails where the user specifies it. S.

### UNW-13: Outward closeness alone costs the caller an unnecessary transpose

**Where.** `S/network-centrality/gpu-network-centrality.ts:155,433,770`; `S/network-centrality/gpu-network-centrality-operation.ts:18`; `S/network-centrality/cpu-network-centrality-operation.ts:77,140`, KI lines 100,172.

**Evidence — EXECUTED.** A closeness-only constructor with a valid forward topology and no `reverseTopology` failed with `network-centrality topology is required`. In the CPU twin, the closeness calculation finishes before any transpose edge walk. For betweenness the transpose genuinely matters: passing the forward graph twice on the directed fixture gave `[0,0,0,0]`, `converged=1`; the correct transpose gave `[0,1,0,0]`. The source documents the exact-transpose precondition.

**Wanted.** Make `reverseTopology` and the operation’s `reverseEdgeCount` optional for closeness, harmonic closeness and reached counts. Require them for node or edge betweenness with the message `reverseTopology is required for betweenness`. Prefer the validated forward/transpose pair from UNW-04 for the latter.

**In-repository callers that change.** None for an optional-input extension. **Value and cost.** A user measuring access from stations can avoid sorting, uploading and binding a second network. M.

### UNW-14: The family guide hides its easiest first run and overstates the uniformity of its input

**Where.** `D/README.md:50`, `D/concepts.md:70`, `D/expansion.md:68` and its KI copy; `S/network-structure/cpu-network-structure-operation.ts:30,67`; `S/network-structure/gpu-network-structure-operation.ts:16`.

**Evidence — COUNT, EXECUTED.** Only `computeKShortestPathsOnCPU`, `computeNetworkFlowOnCPU`, `computeNetworkLouvainOnCPU` and `computeNetworkStructureOnCPU` appear by name in the family pages. None of the 17 pure functions is called in a TypeScript fence. The concepts page says every contributor takes `topology`; the census gives 22 required, two conditional and 17 absent. `computeNetworkStructureOnCPU({topology})` threw `Cannot read properties of undefined (reading 'length')`. `computeNetworkStructureOnCPU(topology)` returned correct labels. The GPU operation uses `{topology}`.

**Wanted.** Put the CPU shortest-path and OD calls from the census directly beside the existing GPU fixture. Give a task lookup for weak labels, strong labels, component summary, route geometry and OD costs. For the two simple-path enumerators, identify the distinguishing contracts: `GPUAlternativeRoutes` offers lexicographic edge-slot ties and per-query completeness; `GPUKShortestPaths` offers bounded Yen search and batch status. Accept `{topology, reverseEdgeSlots?}` as an additional CPU structure input.

**In-repository callers that change.** None. **Value and cost.** Users can check their data and identify the correct operation before allocating a buffer. S. This extends KNOWN(ERG-05) without reopening the corrected GPU quick-start fixture.

### UNW-15: Publish the facility feasibility calculation before users allocate graph views

**Where.** `S/network-optimization/gpu-network-facility-optimization.ts:43,142`; `S/network-k-function/index.ts:8`; `S/network-structure/gpu-network-structure.ts:53,134`; `S/network-structure/index.ts:4`; `D/network-movement-coverage.md:71`.

**Evidence — EXECUTED host construction.**

| Request | Result in this checkout |
|---|---|
| One-facility capacitated p-median, 64 / 65 demands | 64 accepted with **one candidate**; 65 threw `demand rows must be at most 64`. No facility maximum or feasibility helper is exported by the network barrel. |
| Network K, 92,682 / 92,683 events | 92,682 accepted in the probed configuration; 92,683 threw `eventCount must be at most 92682`. The barrel **does export** `GPU_NETWORK_K_FUNCTION_MAXIMUM_EVENT_COUNT`, superseding the older open barrel question. |
| SCC, 1,025 vertices | `mode:'exact'` threw `nodeCount must be an integer in [1, 1024]`; `mode:'auto'` selected `parallel`. The exact-route node and edge maxima exist in source and are absent from the network barrel. |

**Wanted.** Add `getNetworkFacilitySearchLimits({operation,facilityCount,demandCount})` returning the candidate count, applicable limits and the limiting dimension. Export the existing exact SCC maxima and mention `mode:'auto'` in the exact-mode error. At 65 demands, explain that exhaustive joint facility selection cannot be recovered by partitioning demands. Point users with **already selected facilities** to `GPUNetworkServiceAreas`; identify heuristic location-allocation alternatives as a change of solver contract.

**In-repository callers that change.** None. **Value and cost.** Applications can explain feasibility before upload and suggest a mathematically appropriate route. S. The numerical caps are KNOWN(NET-105) and KNOWN(NET-110); the new finding is the unequal public preflight and the verified current barrel state.

## Departures by name

- **Required top-level network, 22 producers.** `GPUAdjacencyMatrix`, `GPUAlternativeRoutes`, `GPUKShortestPaths`, `GPUMapMatching`, `GPUNetworkAnalyticsColumns`, `GPUNetworkCentrality`, `GPUNetworkCentralityScheduleStep`, `GPUNetworkCoarsening`, `GPUNetworkCostMatrix`, `GPUNetworkCostTable`, `GPUNetworkIsochrones`, `GPUNetworkKFunction`, `GPUNetworkLineGraph`, `GPUNetworkNeighborhood`, `GPUNetworkPotentialPathArea`, `GPUNetworkReachability`, `GPUNetworkServiceAreas`, `GPUNetworkSnapping`, `GPUNetworkStatistics`, `GPUNetworkStronglyConnectedComponents`, `GPUNetworkStructure`, `GPUNetworkSubgraphFilter`.
- **Conditional or optional top-level network, two.** `GPUNetworkFlow`, `GPUNetworkLouvain`.
- **Other primary input, 17.** `GPUAdjacencyMatrixOrder`, `GPUEdgeBundling`, `GPUFlowAggregation`, `GPUKernelFlowSmoothing`, `GPUMapMatchedEdgeVolumes`, `GPUNetworkAccessibility`, `GPUNetworkCostWeights`, `GPUNetworkFacilityOptimization`, `GPUNetworkInhomogeneousK`, `GPUNetworkKernelDensity`, `GPUNetworkNoding`, `GPUNetworkPathExtraction`, `GPUNetworkPointRouting`, `GPUNetworkRationalAccess`, `GPUNetworkTimeRouting`, `GPUNetworkTravelTimePercentiles`, `GPURadiationFlowModel`. These input differences are often appropriate. The departure is the universal claim in the guide.
- **Ambiguous word readers, three of four.** `decodeGPUNetworkStatistics`, `decodeGPUNetworkCoarseningSummary`, `decodeGPUNetworkSubgraphFilterCounts`. Only the statistics reader also depends on a separately repeated variable stride.
- **Pure CPU functions absent even by name from the network pages, 13.** `computeAlternativeRoutesOnCPU`, `computeMapMatchingOnCPU`, `computeNetworkCentralityOnCPU`, `computeNetworkCostMatrixOnCPU`, `computeNetworkCostTableOnCPU`, `computeNetworkPathExtractionOnCPU`, `computeNetworkPointRoutingOnCPU`, `computeNetworkRationalAccessOnCPU`, `computeNetworkReachabilityOnCPU`, `computeNetworkServiceAreasOnCPU`, `computeNetworkTimeRoutingOnCPU`, `computeNetworkTravelTimePercentilesOnCPU`, `computeRadiationFlowModelOnCPU`.

## Quick wins

The first eight changes below and the first two larger proposals are the ten I would prioritize.

- [ ] Wire `predecessorsComplete` in the printed path composition and run its zero-weight-chain counterexample.
- [ ] Correct the reverse OD snippet and publish the two legal midpoint seed costs for a one-way road.
- [ ] Add the source-sort, transpose-sort and stable edge-ID steps to graph preparation.
- [ ] Show “from facilities” and “to facilities” beside one another with the same reordered facility list.
- [ ] Add an exact-view histogram reader and reject invalid bin counts.
- [ ] Correct both NetworkX calls and mark the sampled centrality example `'sampled'`.
- [ ] Reject malformed flow tuples and invalid packed map-matching scales at preparation time.
- [ ] Put a CPU shortest-path and compact OD first run on the network landing page, including the expected unreachable entry.

## Larger proposals

1. **Fix path edge identity before alpha.** Correct the fallback first and add predecessor-edge provenance where the producing backend can supply it. The public edge sequence must describe the published cost. UNW-01 is small enough to land independently of a routing redesign.
2. **Standardize network readback before alpha.** Use checked word decoders and an explicit byte-readback path. Add a flow summary reader that exposes `cost`, `exactCost` as a `bigint`, `unserved` and `cutCertified`. Three existing readers and the flow result already demonstrate the need.
3. **Add one host road-network preparation function.** Preserve application IDs, parallel-edge identity, directed impedance and the forward/transpose relationship. Reuse the published luma.gl graph primitives for GPU work. This supports OD, closest facility and centrality together.
4. **Give path counts a shared representability rule.** The CPU and GPU should admit the same safe bounded output, and the GPU must detect a total that cannot fit its publication format.
5. **Keep distinct routing algorithms, but make the choice visible.** The two K-route implementations agreed on costs `[0.3,9.1]` and the cheapest parallel slot in the probe. Their ordering promises, budgets and completion scopes justify separate implementations. A task-level comparison is cheaper and clearer than merging them.

## Not read

I read the network landing page, public barrel, flow-demand adapter and shared CPU routing logic through their relevant complete paths. I traced the nine workflows through the operator and specialized guides, public input/result types, CPU numerical branches and selected GPU declaration, seeding, scan and publication code. I sampled the corresponding tests and lens callers, including the checked-in centrality oracle and path fallback assertion. I checked staging ownership and compared the relevant KI guide sections and routing twins; the report distinguishes fixes already present there.

I did not audit every optimization shader, K-function simulation, community-detection phase, spatial-index kernel, benchmark or downstream renderer. The numerical executions were host-side. Null-device declarations establish admission and graph shape only. The GPU-dependent findings identify the smallest device probes needed to confirm them.