# Thinking-partner dialogue (2026-11-17)

Two rounds between the lead reviewer (Claude Fable 5.2) and a Codex `gpt-6.2-astra` session with read access to the repository. The prompts state the lead's hypotheses at the time; several were corrected by the replies and by the area reports, so read [README.md](./README.md) for the conclusions.

## Round 1: lead

You are my thinking partner (not a sub-reviewer) on a comprehensive review of the core of this
repository: `@luma-contributors/analysis` (`modules/analysis`, ~540k lines of TypeScript+WGSL source,
~2260 spec files, hundreds of `GPU*` "contributor" classes that emit luma.gl `GPUCommandGraph` command
nodes) and `@luma-contributors/lenses`. Out of scope: `modules/kepler`, `examples/`, `website/`.
The maintainer asked: "think about improving the existing state, planning for the future, making the
lib more ergonomic and consistent, and general goodness." Thirteen separate reviewers are doing the
area-by-area reading; you and I are doing the thinking. I want your independent judgement and I want
you to disagree with me where the code says I am wrong.

Rules: read-only. The machine is low on disk and swap: do NOT run tsc, vitest, yarn build/test/lint/
typecheck. `rg`, `sed -n`, `wc`, `git log`, small `node -e`/python3 census probes are fine. Read
`AGENTS.md` and `dev-docs/contributor-contract.md` first; `dev-docs/roadmap.md` is the backlog
(long lines; grep it). Upstream-first is a hard rule: only published `@luma.gl/*` public entry points,
never re-implement or re-export luma.gl; when luma.gl lacks something: use the public API even if
verbose, adapt, drop the feature, or (last resort) a small helper named for the analysis domain.
luma.gl improvements go upstream as luma.gl pull requests.

My working hypotheses after a first pass (test them against the code; each may be wrong):

H1 "No front door." There is no short path from data to a result. A contributor takes caller-owned
   `GraphDataView`s for every input and every output; the caller builds a `GPUCommandGraph`, imports
   buffers, creates views of exactly the right format and length for each output, compiles, encodes,
   submits, reads back, destroys. Evidence: `new GPUCommandGraph(` appears 3203 times in 1303 test
   files; tests have ten separate `*Rig`/`*Harness` classes (for example
   `modules/analysis/test/gpu-spatial-analysis/spatial-weights/spatial-weights-harness.ts`) that each
   re-invent `input(values, format, length)`, `output(format, length)`, `run(...producers)`;
   `importGraphBuffer` exists in `analysis-core/kernels/gpu-contributor-utils.ts` but is exported
   only through `src/testing`, which is excluded from the published files; `docs/` has no end to end
   "hello world". Question: what is the right front door given upstream-first? Candidates: (a) the
   operation forms (`GPU<Name>Operation` + luma.gl `GPUGraph`, contract section 14) become THE user
   API and get generated for every contributor; (b) a tiny domain-named session/runner helper in
   analysis-core; (c) propose the runner upstream; (d) contributors learn to allocate their own
   outputs (`static createOutput(graph, props)` or an output-shape descriptor). Which, and why?

H2 "Hand-written everything, no single source of truth." Each contributor hand-writes prop
   validation (see `gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor.ts` constructor:
   format check + length check + hand-written message per view), then separately: a catalog entry in
   `gpu-spatial-analysis/ontology` (~20k lines), a docs page section, a kernel-audit case, a lens
   (modules/lenses has 840 files of computation-lenses), optionally an operation form and a CPU twin.
   A declarative port/prop descriptor per contributor (format, length relation, optional, per-frame
   vs compile-time, limits) could generate validation, output allocation, the operation form, docs
   tables, catalog rows and lens scaffolding. Is that the right consolidating move, or would it be a
   big-bang abstraction that fights WGSL-shaped reality? What is the smallest version that pays?

H3 "Breadth has outrun the structure." The cost of one more contributor is now dominated by
   registry/lens/docs/audit obligations and by merge conflicts on shared registry files. Typecheck
   needs 8 to 16 GB heaps; the suite cannot run in CI (there is none). Should the next 6 to 12 months
   be consolidation (fewer, more uniform, more composable contributors; generated metadata; project
   references; a release) rather than more operators? What would you cut or freeze?

H4 "Three composition mechanisms." Contributor classes (`getCommandNodes`), operation forms in a
   luma.gl `GPUGraph`, and recipes (`add<Name>Recipe` via `RecipeBuilder`). Do all three earn their
   keep?

H5 "Not yet a product." Version 0.1.0-alpha.0, docs disagree about the luma.gl pin (alpha.12, .13,
   .14 in different files), `upstream-proposals.md` has 324 lines of proposals and "nothing has been
   filed". What is the path to a first release that a stranger could use, and what should be filed
   upstream first because the library's ergonomics depend on it?

Please: (1) spend real effort checking H1 to H5 in the code (cite path:line for what you rely on);
(2) tell me where I am wrong or shallow; (3) give me the two or three strategic questions I have not
asked; (4) sketch, concretely, the API you would want as a user for three very different cases: a
local Moran's I on 1e6 points, a cost-distance raster, and a k-shortest-paths network query: show
what the code looks like today (from tests) and what it should look like; (5) end with your ranked
top 7 moves. Be opinionated and specific. About 1500 to 2500 words.

## Round 1: partner

**I’d spend the next release making three analyses usable from a clean install.** The repository already has a short path for some calculations, a composition model, CPU implementations, and CI. The gap is that a newcomer must know which path to choose, which outputs to allocate, and whether the returned numbers are fit to use. The most telling example is in the Local Moran test itself. The neighbor search writes an overflow flag, but the harness passes Moran a separately assembled weights object that omits that flag. The two stages share the right buffers while losing a fact that determines whether the statistic is valid. That is a stronger argument for a better front door than the number of graph constructors. [spatial-autocorrelation-harness.ts:168](../../../modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-harness.ts:168) · [gpu-local-moran.ts:131](../../../modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts:131)

## Where the hypotheses hold

**H1. There are several entrances, and the package README gives a newcomer little help choosing one.** Within `modules/analysis/test`, I count 3,203 literal `new GPUCommandGraph(` occurrences in 1,290 TypeScript files and ten class names ending in `Rig` or `Harness`. Some duplication is deliberate. `WeightsRig` poisons output buffers to catch missing writes, behavior you would never want in a user runner. The README lists entry points, then goes straight into MGWR bandwidth selection. Meanwhile, there *is* an end-to-end tutorial under `docs/`. It creates synthetic variogram bins, fits them with one CPU function, and runs the same fit inside a mixed graph given a device. I’d put the first half of that tutorial on the package landing page today. [spatial-weights-harness.ts:58](../../../modules/analysis/test/gpu-spatial-analysis/spatial-weights/spatial-weights-harness.ts:58) · [analysis/README.md:8](../../../modules/analysis/README.md:8) · [operation-forms-and-cpu-twins.md:46](../../../docs/api/operation-forms-and-cpu-twins.md:46)

Of your candidates, **I’d start with (d), as a pure output-layout function, then use (b) only where the domain supplies real behavior**. A layout can tell a caller what to allocate while leaving ownership with the caller, a recipe, or luma.gl. A cost-distance helper also knows when to submit another relaxation window and when the surface is usable. There’s precedent in `createNetworkCentralityContinuation`, which keeps frontier state resident and reads one status word per window. A generic `input/output/run` session has much less justification. The unpublished `importGraphBuffer` boils down to two public calls, `graph.importBuffer` and `graph.createDataView`. Publishing a second graph vocabulary to save those lines would saddle this project with the wrong API. [gpu-contributor-utils.ts:110](../../../modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts:110) · [analysis/package.json:75](../../../modules/analysis/package.json:75) · [network-centrality-continuation.ts:639](../../../modules/analysis/src/gpu-network/network-centrality/network-centrality-continuation.ts:639)

**H2. The duplication is real. Its smallest profitable unit is an output layout for one family.** Start with k-shortest paths. The contributor and the operation independently encode the same arithmetic. Vertices and edges need `queries × routes × pathLength` rows, costs and lengths need `queries × routes`, counts need `queries`, and status flags are scalars. Extract `getKShortestPathsOutputLayout(options)`. Have the constructor validate against it and the operation declare outputs from it. Use the same data to render the docs table. Keep the fixture’s expected sizes independently written so a wrong layout can still fail a test. That removes two copies of an actual decision without introducing a schema language. [gpu-k-shortest-paths.ts:173](../../../modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts:173) · [gpu-k-shortest-paths-operation.ts:38](../../../modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths-operation.ts:38)

LOF shows where to stop. Format, row count, and optionality are mechanical. Whether a radius neighborhood is legitimate input to LOF, whether the density floor uses compatible distance units, and whether two writable views alias are different questions. Preserve the domain checks and their diagnostics. Reuse the parameter schema that already knows word offsets, defaults, units, and mutability. Likewise, a Chicago Moran lens records the tract dataset, weight standardization, correction method, and meaning of the mapped variable. None of those can be inferred from `float32[rows]`. Generate the mechanical table once. Keep the interpretation, worked scenario, and oracle independently authored. [gpu-local-outlier-factor.ts:130](../../../modules/analysis/src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor.ts:130) · [parameter-schema.ts:8](../../../modules/analysis/src/analysis-core/contracts/parameter-schema.ts:8) · [local-moran.lens.ts:73](../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-moran.lens.ts:73)

**H3. Consolidate, but two premises are wrong.** CI is configured for pull requests and master. It requests contract checks, typechecks, a build, package export verification, Node tests, and headless GPU tests. Linux CI requests SwiftShader. Package-level TypeScript project references also already exist. The source check still includes all module sources, and the test driver gives its main test group a 16 GB heap ceiling. Those facts justify measuring elapsed time and peak memory by group. They do not establish that CI passes, that it cannot pass, or that registry conflicts dominate contributor cost. My review was read-only, with no build, test, or live CI run. [test.yml:16](../../../.github/workflows/test.yml:16) · [vitest.config.ts:66](../../../vitest.config.ts:66) · [lenses/tsconfig.json:15](../../../modules/lenses/tsconfig.json:15) · [tsconfig.check-src.json:1](../../../tsconfig.check-src.json:1) · [typecheck-tests.mjs:152](../../../scripts/typecheck-tests.mjs:152)

I’d freeze new operator families, blanket operation-form coverage, and retrospective lens coverage for one release cycle. Keep admitting correctness fixes and work needed by an identified consumer. I would also change one catalog decision early. The all-entry-point assembler defaults a missing tier annotation to `scale` while marking the contract `unaudited`. Those fields can coexist technically, but a user choosing a million-row method needs to know the largest demonstrated workload, its shape, and the adapter. Record that evidence or leave the capacity claim unqualified. Sharding registry files can follow demonstrated editing contention. [analysis-operator-catalog.ts:46](../../../modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts:46)

**H4. All three earn their keep, for different jobs.** A contributor is the right unit for a host that owns buffers and repeatedly encodes a resident graph. An operation earns its place when named values, a CPU stage, or a swappable twin matter. Today the bridge accepts one packed chunk per value, rebuilds the contributor on each run, and requires explicit destruction of the last resource-owning contributor. Generating hundreds of wrappers would make that a poor default for some of the library’s best workloads. A recipe earns its place when it expresses a method. The rate-cluster recipe chooses how empirical-Bayes rates, weights, Moran, permutation inference, and quadrant colors fit together, and returns status by stage. `RecipeBuilder.add` itself just calls `graph.add` and records the contributor. Keep the scientific recipes and the internal builder. Teach users one recipe by the question it answers. [gpu-contributor-operation.ts:79](../../../modules/analysis/src/analysis-core/operations/gpu-contributor-operation.ts:79) · [recipe-utils.ts:53](../../../modules/analysis/src/gpu-spatial-analysis/recipes/recipe-utils.ts:53) · [rate-cluster-map-recipe.ts:253](../../../modules/analysis/src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe.ts:253)

**H5. Release the analysis alpha independently of lenses.** Root development dependencies pin luma.gl `.14`, analysis declares `~10.0.0-alpha.14` peers, and the package README still tells installers `.13`. The `.12` transfer timings are dated measurements worth preserving with their original version. The proposals register says nothing has been filed. For usability, I’d send the packed CPU/GPU transfer fast path upstream first. There is a small reproducible benchmark, a historical ~90 ms boundary cost for one million `float32x2` rows on one Mac, and a row-by-row loop still visible in the installed `.14` implementation. Second, propose a producer-to-operation port and ownership contract using one real contributor as the example. Both can proceed independently of the release. [package.json:42](../../../package.json:42) · [analysis/package.json:88](../../../modules/analysis/package.json:88) · [upstream-proposals.md:3](../../../dev-docs/upstream-proposals.md:3) · [CPU measurements:23](../../../dev-docs/research/cpu-suitability-2026-11-11/README.md:23) · [installed gpu-cpu-execution.js:232](../../../node_modules/@luma.gl/graph/dist/gpu-core/gpu-cpu-execution.js:232)

## Three APIs I would actually use

The “today” snippets condense the cited tests. The “proposed” names are API sketches. GPU examples assume an existing luma.gl device.

### Local Moran’s I on one million points

Today the integration test uses 2,000 points, changes the radius across runs, compares against an independent oracle, and checks that the contributor was built once. The pleasantly short call below conceals setup, allocation, and extensive readback in a test-only harness. Its `160` slots per point are a fixture choice. At one million points that choice alone would take **640 MB each** for neighbor IDs and weights, before offsets, distances, or scratch. Moran already accepts chunked CSR weights. A packed point-to-weights producer is a separate capacity decision. [gpu-local-moran.spec.ts:171](../../../modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.spec.ts:171) · [spatial-autocorrelation-harness.ts:147](../../../modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-harness.ts:147) · [gpu-local-moran.ts:127](../../../modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts:127)

```ts
// Today, in the test
const harness = createSpatialAutocorrelationHarness(device, {
  contributor: 'local-moran', scene,
  weights: {kind: 'neighbor-search', positions: scene.positions,
    parameters: {bounds: BOUNDS, radius: 3, rowStandardize: true},
    capacity: 2000 * 160}
});
try { const result = await harness.run(); } finally { harness.destroy(); }

// Proposed, with projected point coordinates in metres
const moran = await preparePointLocalMoran(device, {
  positions, bounds, spatialContext,
  weights: {kind: 'distance-band', radius: 250, transform: 'row'},
  islands: 'exclude', neighborCapacity: 8_000_000,
  outputs: ['localI']
});
try {
  const result = await moran.run({values});
  if (result.status.weightsOverflow) reportCapacity(result.status.requiredCount);
  else consume(await result.readLocalI());
} finally { moran.destroy(); }
```

The proposed object owns its allocations, retains weights across value-only updates, and reports required slots on overflow. Eight million is an explicit budget. Success depends on point density and device limits. A second overload should take existing weights, preserving their transform, chunking, and status. The current capacity planner already specifies how to grow from an observed requirement. Also, `localI` is optional today while `zScores` is required. Someone asking for I should be able to request I alone, even if the first implementation needs an internal z-score buffer. [spatial-weights.ts:362](../../../modules/analysis/src/gpu-spatial-analysis/spatial-weights/spatial-weights.ts:362) · [gpu-local-moran.ts:73](../../../modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts:73)

### Cost-distance raster

Today there is already an excellent first calculation. The CPU fixture has two competing sources and a zero-friction plateau. Its four costs are `[0, 1, 1, 0.5]`. Make that the runnable introductory example. The GPU operation also handles buffer-backed raster bands. The interesting missing convenience is bounded GPU continuation, which the operation form expressly excludes. [cpu-cost-distance-operation.node.spec.ts:18](../../../modules/analysis/test/gpu-raster/cost-distance/cpu-cost-distance-operation.node.spec.ts:18) · [gpu-cost-distance-operation.ts:12](../../../modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-operation.ts:12)

```ts
// Today, with the fixture's grid and friction band
const result = computeCostDistanceOnCPU({
  friction, parameters: getGPUCostDistanceParameterValues({}, grid),
  sources: Uint32Array.of(0, 3), sourceCosts: Float32Array.of(0, 0.5),
  costBreaks: Float32Array.of(0.5, 1, 2)
}, {grid});

// Proposed GPU convenience, with the band's validity and grid metadata intact
const surface = await computeCostDistance(device, {
  grid, friction, sources, sourceCosts,
  roundsPerSubmission: 64, maximumSubmissions: 8
});
if (surface.converged) drawCosts(surface.costs);
else reportUnresolved(surface.iterationCount);
```

For this one-shot API I’d return CPU-owned costs and read them once, on convergence. Internally, continuation owns persistent state and polls a scalar between submissions. A passable cell has `NaN` cost until relaxation converges, and continuation excludes back-links. The API should make costs available on the converged branch and reject an incompatible back-link request up front. The test already demonstrates repeated 64-round windows against one 512-round execution. [gpu-cost-distance.ts:267](../../../modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance.ts:267) · [gpu-cost-distance-continuation.spec.ts:133](../../../modules/analysis/test/gpu-raster/cost-distance/gpu-cost-distance-continuation.spec.ts:133)

### K-shortest paths on a directed network

For a small network I would keep the existing CPU entry point. The input is already ordinary CSR and endpoint arrays. The awkward part comes *after* the call, when the user must decode padded slots and remember that parallel edges distinguish routes with identical vertex sequences. The GPU test adds nine named output values and a mixed-graph runner to obtain the same layout. [cpu-k-shortest-paths-operation.node.spec.ts:16](../../../modules/analysis/test/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.node.spec.ts:16) · [gpu-k-shortest-paths.spec.ts:31](../../../modules/analysis/test/gpu-network/k-shortest-paths/gpu-k-shortest-paths.spec.ts:31)

```ts
// Today, for the first query
const options = {routeCount: 4, maximumPathLength: 32};
const result = computeKShortestPathsOnCPU({topology, sources, targets}, options);
for (let route = 0; route < result.counts[0]; route++) {
  const start = route * options.maximumPathLength;
  const edges = result.edges.subarray(start, start + result.lengths[route] - 1);
  drawRoute(edges, result.costs[route]);
}

// Proposed decoder over either backend's packed result
const paths = describeKShortestPaths(result, options);
for (const route of paths.routesForQuery(0)) drawRoute(route.edges, route.cost);
if (!paths.batchStatus.converged) reportPartial(paths.batchStatus);
```

That decoder should return typed-array views and expose batch-wide status. On candidate overflow or a search limit, previously certified routes remain usable. A complete result can also contain fewer than four paths. Treating “fewer than requested” as an error would discard exactly the distinction the algorithm worked to preserve. [k-shortest-paths.md:44](../../../docs/api/gpu-network/k-shortest-paths.md:44)

## Three strategic questions

1. **What promise travels with a result?** Moran can lose validity through upstream weight overflow, cost distance can be unresolved everywhere passable, and Yen can return a certified prefix. The common contract should require that callers can discover completeness and its cause. The meaning of a partial result belongs to the method.
2. **Which resource is the user trying to preserve?** A map host wants resident buffers and predictable frame time. An analyst with typed arrays wants an answer and no resource leak. Time each supported workflow from its actual starting point through consumption, including upload, compilation, repeated execution, and selected readback. Publish workload shape and peak bytes alongside row count.
3. **What does a lens certify?** A scenario can capture a real execution, adapter, and dataset. It can also be tied closely to an embedded dataset. The current user-input path serializes runs through module-scoped state and supports three dataset IDs for dataframe/SQL scenarios. I’d use lenses as worked, inspectable evidence for selected methods. Each release example should have a synthetic fixture, an independently expected answer, and a record tied to the shipped version. [run-lens-scenario.ts:26](../../../modules/lenses/src/computation-lenses/run-lens-scenario.ts:26) · [user-input.ts:68](../../../modules/lenses/src/computation-lenses/user-input.ts:68)

## Ranked top seven moves

1. **Define the analysis alpha by three clean-install journeys.** Put a CPU answer with literal expected values first in the README. Give each GPU journey a public-import example, device setup, completion check, and cleanup.
2. **Make invalid and partial results hard to misread.** Wire Moran to the producer’s `outputWeights`, deliberately overflow a small fixture, and test what a downstream consumer sees. Add equivalent budget-stop examples for raster and routing.
3. **Extract the k-shortest-paths output layout and decoder.** Reuse the layout in validation, operation declaration, and docs. Try one optional-output raster family before standardizing anything across the package.
4. **Ship the two domain conveniences that earn their cost.** Prepare reusable point weights for Moran and own bounded continuation for cost distance. Preserve the direct contributor path for callers with resident data and output buffers.
5. **Qualify the packed artifact.** Align active version instructions, install the tarball outside the workspace, exercise documented peer combinations, and run the journeys there. The current export check resolves workspace packages. Keep lenses private while its manifest includes the dataset trees covered by the redistribution notice. [verify-package-exports.mjs:14](../../../scripts/verify-package-exports.mjs:14) · [lenses/package.json:47](../../../modules/lenses/package.json:47) · [THIRD-PARTY-DATA.md:229](../../../THIRD-PARTY-DATA.md:229)
6. **Qualify one million points on named hardware and measure CI by stage.** Publish density, edge count, binding limits, peak allocation, cold time, and warm time. Use measured memory and duration to choose the first internal TypeScript split.
7. **File two focused luma.gl PRs and hold the breadth freeze through the alpha.** Lead with packed transfer and its benchmark. Follow with the producer port/lifetime contract and a real consumer. Resume adding families when a stranger can install the package, obtain the expected answer, and recognize an incomplete one.

## Round 2: lead

Round 2. Thank you: your corrections held up when I checked them (the CI workflow exists, though by
the maintainer's own notes GitHub Actions has never produced a run in this private repository; the
catalog defaults unannotated operators to tier 'scale' with contract 'unaudited'; your "what promise
travels with a result" point turned out to be the spine of the whole review).

The thirteen area reviewers have reported. Their reports are in
the `area-*.md` files of this folder (read any you like; `api-ergonomics.md`, `core-contract.md`,
`ontology-catalog.md`, `tooling-tests.md`, `lenses.md`, `docs-devdocs.md` are the structural ones).
Totals: 38 findings graded P0, 83 P1, 44 P2. Seven independent verifiers are trying to refute the
P0s right now, so treat severities as provisional; I spot-checked five in source and the evidence was
accurate each time, but several are extreme-input or documented-legacy cases that I expect to regrade.

My synthesis so far. Attack it.

T1 "Seams, not kernels." The defects cluster where one correct piece meets another, not inside
kernels: (a) completeness does not travel: pair gather drops a join's candidateOverflow (GEO-03), map
matching reads an unfinished cost table as "no route" (NET-01), your Moran harness example; (b) one
`uint32` view type carries five meanings: feature row vs application id (GEO-02 nearest join → gather
looks up attributes by id), validity polarity (ERG-01: an output named `validity` where 0 = matched),
failure bitmasks under filter-compatible names (GEO-08), key spaces joined by array position in lens
tables (LENS-01); (c) CPU twin and GPU form disagree on inputs or statuses (ERG-07, NET-05, NET-06,
SSR-06, SSR-09, DOC-01); (d) outputs that have no writer on reuse or when requested alone (PTS-03,
PTS-04); (e) tiled vs whole, padded vs exact-length, texture vs buffer (RT-03, SSR-08, RT-09, RT-12).
Specs test one contributor on fresh exact-length buffers, so all of this passes. Proposed response:
a small set of semantic port aliases (row index / source id / mask / validity / code) with the
meaning in the type name; completeness carried on the data port (as GPUSpatialWeights could carry its
producer's status) with consumers required to fold it in; and a "composition suite" of cheap
invariants: producer→consumer graphs, resubmission into prefilled outputs with fewer live rows, each
optional output alone, padded capacity, unit rescale and translation invariance, one props object
through both twins.

T2 "The float32 border." Host code validates JS doubles and then packs float32: finite-only fields
that pack to Infinity (CORE-04), a significance level that packs to the default (SSR-05), parameter
buffers that accept the wrong typed array (CORE-03), finite sentinels (1e30 "unreachable", PTS-02),
absolute tolerances that make results unit-dependent (GEO-01, PTS-07, PTS-09, SSR-01). There are 214
hand-written `getGPU*ParameterValues` packers and a declarative parameter schema used by 6 files.
Proposed response: validate the rounded value in one place, migrate packers to the schema
opportunistically, add rescale/translate fixtures.

T3 "Gates that pass when nothing ran." The export check passes against a stale dist (156 sources
have no emitted JS); a layering spec iterates an empty map; a runtime-boundary spec has no assertion;
81 specs define local closeness comparators and at least two accept NaN; some headless specs return
silently with no device; the oracle runner exits 0 having compared nothing; the contract baseline
compares counts so a replacement regression passes; typecheck-status trusts a dead watcher; the
catalog registration check is a text search.

T4 "Many inventories." Catalog (554 hand-maintained records, 63 missing from their families, bare-name
identity that collapses two visibility recipes, `status: []`/`readback: 'none'` defaults that read as
facts), docs (107 pages, 98 in the sidebar, 66 snippet-checked), coverage matrix, lens coverage, kernel
audit, roadmap (12 of 15 sampled rows stale; 3 duplicate IDs), version pins (alpha.12/.13/.14 in docs,
.14 pinned, .16 published today). Proposed response: package-qualified identity now; sibling
descriptors feeding one generated manifest later; "unknown" as an explicit value; roadmap as records.

T5 "Not yet installable by a stranger": Arrow optional in metadata but imported by 4 of 11 entry
points; MIT-only metadata over 19 files with Apache/BSD expressions; README links that leave the
package; no packed-consumer check; five lens loaders with directory imports.

T6 "The loop is too heavy": test typecheck overrides the caller's heap with 16 GB and pulls 223 MB of
JSON fixtures into the program; build always cold; no single verify command; the library grew from
317k to 540k source lines between the 2026-11-09 review and today, and that review's align/merge/
rework rows are mostly still open.

Your round-1 answers I am adopting: output-layout functions rather than a generic runner; domain
conveniences only where they own real behaviour (prepared Moran weights, cost-distance continuation,
a k-shortest-paths decoder); keep all three composition forms; breadth freeze through a first alpha;
first-run docs led by the CPU variogram example.

Questions:
1. Is T1 really one theme or am I forcing five things into a slogan? If you had to pick ONE mechanism
   to adopt library-wide from the T1 response, which, and what does it cost per contributor? Would
   branded port types actually have caught GEO-02 and ERG-01 given everything is `GraphDataView<'uint32'>`
   from luma.gl (check how views are typed and whether a brand survives `graph.createDataView`)?
2. "Completeness travels with the data": sketch the smallest concrete TypeScript shape, using
   `GPUSpatialWeights` (it already has an optional `information` field) and `GPUCompactPairPort` as the
   two pilots. Who writes the fold, a helper or each consumer? What about consumers that are luma.gl
   primitives?
3. Order of work. I am inclined to: (A) fix verified wrong-result defects and vacuous gates, about a
   week; (B) make an installable alpha (packed-consumer check, Arrow decision, licences, first run,
   luma.gl bump); (C) composition contract v2 + composition suite, piloted on joins→gather and
   weights→statistics; (D) single-source metadata; (E) typecheck/test split. Is B before C right, or
   does releasing before C just publish the traps?
4. What would you NOT do that the reviewers proposed? Name the proposals that are over-engineering
   for a ten-day-old alpha with one maintainer and AI agents doing most of the writing.
5. This library is being written at about 28k lines a day by agents, reviewed by agents. What review
   or gating practice would change the defect profile most, given T1 and T3? Be concrete (for example:
   every new contributor lands with one composition fixture and one invariance fixture; a reviewer
   brief that forbids fresh-buffer-only tests).
6. Anything important the thirteen reports collectively missed?

Same rules as before (read-only, no tsc/vitest/build). 1200 to 2000 words, opinionated.

## Round 2: partner

**Start packaging now, and put the two composition pilots ahead of publication.** I’d change your sequence to **A plus the small part of E that makes verification runnable → B and the two C pilots in parallel → publish the tested tarball → decide how much of C and D to generalize**. A stranger should be able to trust the paths you teach on day one. That requires less work than migrating the whole library and more work than making its imports resolve.

### 1. T1 is a good review strategy and a weak defect taxonomy

Your strongest claim is that **testing from the next consumer’s point of view exposes failures that isolated numerical agreement misses**. I would drop the claim that the defects mostly occur between correct pieces. The reports are a selected sample, several findings duplicate the same defect, and there are clear counterexamples. A conformal kernel can choose the wrong order statistic with exactly representable inputs. An optional output with no writer is a defect inside its producer. Neither needs a second contributor to go wrong. Conversely, the Moran test already composes two real contributors and resubmits the graph. What it lacks is a case where the distinction between *sharing the weights buffers* and *passing the producer’s complete weights port* is observable. [Conformal-rank finding:39](../../../area-spatial-stats.md:39) · [Moran integration test:171](../../../modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.spec.ts:171) · [Moran harness:186](../../../modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-harness.ts:186)

My **one library-wide mechanism** would be **one discriminating test for each new or changed public result contract**. Choose a realistic consumer and a fixture on which the documented meaning and a plausible wrong meaning yield different answers. For a terminal operator, the consumer can be the public decoder or a five-line user calculation. Often the cost is one extra case in an existing spec, one small literal expectation, and, for a GPU reuse claim, a second submission of the same graph. The producer and consumer can share that regression. I would spend nothing on a cross-product of every operator, output, and execution mode.

**Brands would help at a narrower boundary than you suggest.** `GraphDataView<T>` parameterizes storage format. `graph.createDataView(..., {format: 'uint32', ...})` returns `GraphDataView<'uint32'>`; its signature has nowhere to express row identity or polarity. Two type aliases of that type remain interchangeable. A required `unique symbol` brand can distinguish two already-branded values, but a fresh view has neither brand, and passing a branded value to an existing raw-view input is still legal. Branding at allocation also asks the allocator to certify what a subsequent shader writes. [GraphDataView declaration:125](../../../node_modules/@luma.gl/graph/dist/gpu-core/gpu-command-graph-types.d.ts:125) · [createDataView signature:230](../../../node_modules/@luma.gl/graph/dist/gpu-core/gpu-command-graph.d.ts:230)

For **GEO-02**, nearest already has an internal `writeFeatureRows` path. Expose the actual rows separately from the remapped application IDs and let gather accept the producer’s row-valued port. Test IDs `[1, 0]` with different attributes. Both IDs are legal offsets, so bounds checks give no protection. For **ERG-01**, preserve the legacy bytes under a deprecated spelling, expose `alignmentCode`, and produce a real `matchedMask` from the predicate the kernel already computes. Branding today’s `validity` as valid would certify the inversion. Pass that mask to a real filter and assert **which target survived**. A count of one passes in both directions. [Nearest publisher:1055](../../../modules/analysis/src/gpu-spatial-analysis/spatial-join/nearest-nodes.ts:1055) · [Existing row-valued use:970](../../../modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join.ts:970) · [Alignment predicate:188](../../../modules/analysis/src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment.ts:188)

### 2. Both pilots already have the transport

`GPUCompactPairPort.status` already carries `overflow` and optional `candidateOverflow`. `GPUSpatialWeights` inherits an optional `status` from `GPUSpatialWeightsPort`; its `information` describes how the weights were built. Keep changing GPU status out of that CPU metadata. Crucially, predicate join writes **both** pair and weights overflow from `pairIncomplete`, and both candidate flags from `candidateIncomplete`. Gather reads only the first pair flag. Moran likewise documents and consumes `weights.status.overflow`. Candidate-only loss is therefore a useful test for **both** pilots. For Moran, use a square self-join whose rows and neighbor IDs refer to the same observations. [Port contracts:136](../../../modules/analysis/src/analysis-core/contracts/ports.ts:136) · [Predicate publisher:1863](../../../modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:1863) · [Gather’s inherited flags:393](../../../modules/analysis/src/gpu-spatial-analysis/pair-gather/gpu-pair-gather.ts:393) · [Moran input contract:131](../../../modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts:131)

I’d prototype this as a **stronger return type for candidate-bounded producers**. Existing consumers and caller-supplied input types can stay source-compatible.

```ts
type CandidateBoundedResult<Port> = Port & {
  readonly status: {
    readonly overflow: GPUStatusScalar;
    readonly candidateOverflow: GPUStatusScalar;
  };
};

type PredicatePairs = CandidateBoundedResult<GPUCompactPairPort>;
type PredicateWeights = CandidateBoundedResult<GPUSpatialWeights>;
```

That intersection retains the pair count and existing diagnostics. Its guarantee is modest and enforceable: a composed candidate-bounded producer supplies both signal views, including graph-owned scalars when the caller has no interest in reading them back. A caller uploading a known-complete CSR can still provide plain weights. An old bounded producer with an unwired diagnostic has **unknown** candidate completeness. An exact-result convenience accepting that producer must obtain its aggregate guarantee or reject that connection. An absent view cannot establish that a GPU flag was zero.

**Share the mechanical fold; give each consumer responsibility for the consequence.** A small helper can collect the applicable loss views for binding, validation, alias checks, and a GPU-side OR. Gather already batches flags when a publication node would exceed eight bindings. It can retain matched rows, preserve the candidate diagnostic, and honor its historical aggregate `overflow` promise. Moran should suppress inference on incomplete neighborhoods. Map matching has a better recovery: bypass the unfinished table and run its existing A* path. A larger final buffer will do nothing for lost candidates, and `requiredCount` after candidate loss may be only a lower bound. Preserve those distinctions in the published result. [Gather flag batching:554](../../../modules/analysis/src/gpu-spatial-analysis/pair-gather/gpu-pair-gather.ts:554) · [Weights count semantics:250](../../../modules/analysis/src/gpu-spatial-analysis/spatial-weights/spatial-weights.ts:250) · [Map-matching fallback:819](../../../modules/analysis/src/gpu-network/map-matching/gpu-map-matching.ts:819)

For a luma.gl primitive, the **enclosing analysis recipe** owns that decision. A primitive can safely reduce a retained prefix while the recipe carries its incompleteness to publication. If partial input makes the answer meaningless, use supported gating or invalidate the published answer. No upstream primitive needs to learn the semantics of a missing spatial neighbor.

### 3. The release boundary cuts through B and C

- **A and a small piece of E first.** Honor the caller’s heap limit and establish one bounded local verification command for the release paths. Fix gates that could certify those paths vacuously. Give the oracle checker negative controls for zero comparisons and one missing expected file among several successful comparisons. A nonzero total alone misses the second case. I would budget A by reachable wrong answers and verified regressions. A week is a planning allowance, not an acceptance criterion. [Oracle exit paths:315](../../../scripts/run-oracle-fixture.mjs:315) · [Typecheck heap override:152](../../../scripts/typecheck-tests.mjs:152)
- **Start B immediately.** Build once, pack once, exercise the exact tarball outside the workspace, and promote those same bytes. For the first alpha I’d declare Arrow required if the four eager entry points remain. Splitting them can follow consumer demand. Resolve attribution and README destinations, and fix the documented variogram input discrepancy before making it the first example. Keep the luma.gl bump independent unless it fixes a release blocker. Changing the substrate while qualifying numerical fixes makes regressions harder to attribute. [Packed-artifact finding:83](../../../area-tooling-tests.md:83) · [Variogram discrepancy:16](../../../area-docs-devdocs.md:16)
- **Finish the pilots before publishing.** Require a successful ordinary case, a candidate-only loss case, and a return to success on reuse. For other verified wrong answers reachable in documented supported configurations, fix them or explicitly restrict the affected configuration. Publish once the artifact and its promises agree. Full contract migration, generated metadata, and project-reference restructuring can follow.

### 4. Work I would decline this cycle

I’d defer the ontology report’s neutral catalog entry plus checked-in manifest, the docs report’s generated roadmap views, and the lenses report’s additional workspace and incremental replay dependency system. Each solves a real maintenance problem. Together they create several new systems for one maintainer to reconcile during a correctness release. Fix the identity collision now with **public import subpath plus export name**. Both visibility recipes are in the same npm package, so package name alone still collides. Fix stale backlog rows directly. Keep lenses private. [Ontology proposals:226](../../../area-ontology-catalog.md:226) · [Docs proposals:208](../../../area-docs-devdocs.md:208) · [Lens proposals:151](../../../area-lenses.md:151)

I’d also decline a 214-packer migration and a blanket rescaling requirement. T2 covers three different defects. Validate the caller’s value **and its encoded value** against the field’s domain before mutating a supplied target. Check the actual typed-array representation separately when writing raw bytes. Unit-dependent tolerances require a mathematical correction and a fixture that rescales every dimensional parameter consistently. Routing all three through the parameter schema would fix only the first category. [Schema packing:98](../../../modules/analysis/src/analysis-core/contracts/parameter-schema.ts:98) · [Parameter buffer input:27](../../../modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts:27)

### 5. Change what counts as review evidence

Give the reviewer the public contract and a concrete instruction: **“Name a plausible wrong implementation that would pass the author’s ordinary fixture. Supply the smallest input that distinguishes it, and write down the expected consumer-visible result before inspecting the fix.”** For a bug fix, require a recorded failure against the parent revision and success against the candidate. For new code, temporarily remove the relevant write, swap the row mapping, or ignore the candidate flag and show that the fixture fails for the stated reason. One targeted check is enough; a repository-wide mutation-testing service would add little here.

For joins, my fixture would give one left row two genuine matches, enough final capacity for both, and candidate capacity for one. Assert one retained match **and** incomplete status. On the same compiled graph change the data so only one candidate exists. Assert the correct row and cleared status. That catches the dropped flag, a sticky flag, and a test that never submitted its second frame. For CPU/GPU twins, use a hand-computed expectation on a shared valid input; require parity only where the documented contracts agree. Disciplined disagreement with an independent expectation is more valuable than two implementations reproducing the same mistake.

### 6. One omission worth adding to the review

**How does a user know that a saved result predates a correctness fix?** The default `runId` hashes graph and node IDs, backend and variant, dataset version, and seed. It omits parameter values and implementation revision. Two runs can therefore use different radii, or different corrected shader code, and receive the same default ID. Capture and metric validation compares that ID. With matching shapes, the identity check alone cannot distinguish the sidecars. This is visible from the source; I did not execute a mismatched-sidecar test. [Run ID calculation:89](../../../modules/analysis/src/computation-record/create-computation-record.ts:89) · [Sidecar validation:370](../../../modules/analysis/src/computation-record/validate-computation-record.ts:370)

Keep the stable scenario permalink. Give each new capture and its sidecars a shared capture identifier, record the producing analysis version or revision, and have a correctness fix name the affected operator and input condition in the changelog. That gives a maintainer a concrete answer when someone asks, “Do I need to rerun my analysis?” It also extends your central question to the point where the user needs it most: what promise still travels with a result **after the library changes**?
