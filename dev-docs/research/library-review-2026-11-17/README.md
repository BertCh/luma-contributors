# Library review: ergonomics, consistency, release readiness and direction (2026-11-17)

A review of the core of this repository at commit `bdc715ea` (clean working tree): `@luma-contributors/analysis` (`modules/analysis/src`: 1,710 TypeScript files, about 542,000 lines, 631 exported `GPU*` classes of which 492 are public command-node producers, 128 operation forms with 128 CPU twins, 42 recipe exports; `modules/analysis/test`: 2,264 spec files) and `@luma-contributors/lenses` (1,028 source files, about 108,000 lines, 124 spec files). `modules/kepler`, `examples/` and `website/` were out of scope and were read only as consumers.

The question was: what would improve the existing state, make the library more ergonomic and consistent, and set it up for the future. The [library review of 2026-11-09](../library-review-2026-11-09/README.md) measured every contributor against the contract and GPU scale; this review does not repeat that. It looks at the library from the outside: what a user or a second contributor meets. The work items are in [../../roadmap.md](../../roadmap.md) under "Library review 2026-11-17 (LR)". This directory is the evidence.

## Summary

1. **Wrong results are reachable through the public API, and the specs do not see them.** Reviewers reported 38 defects that give a wrong result. Independent verifiers tried to refute each one and refuted none: 35 confirmed, 3 confirmed in part (36 distinct defects after two duplicates; 12 of them on ordinary input). About half sit inside one function: a stopping rule, an order-statistic rank, a tail probability, a tolerance tied to the units of the data. The other half only appear when one correct piece is connected to another, reused, or given a view with spare capacity: a join's lost candidates are not reported by the gather that consumes it, a nearest join hands application IDs to a consumer that reads rows, an output named `validity` uses 0 for valid, a second submission leaves old values in slots that no longer have a writer, a tiled run differs from the whole. The specs test one contributor at a time, on fresh buffers of exact length, at one scale, so both kinds pass.
2. **A caller cannot always tell whether a result is complete.** Completeness lives in optional status views that the next stage is free to ignore. The ports that could carry it already exist (`GPUCompactPairPort.status`, `GPUSpatialWeightsPort.status`); the consumers do not fold them in. This is the single most useful contract change: a producer whose result can be incomplete returns a port that says so, and every consumer either propagates it or refuses.
3. **There is no short path from data to a result, and the first one that exists is hidden.** Every use needs about 35 lines of graph, buffer and readback code around the contributor. The tests wrote that code 3,203 times (`new GPUCommandGraph(` in 1,290 files) and built ten separate rigs for it. The one complete walkthrough (the variogram fit in `docs/api/operation-forms-and-cpu-twins.md`) is not linked from either README; the five family quick starts stop before submission and readback, and the network one is wrong (its three spare start slots also seed node 0, and its cost breaks are never written).
4. **The calling convention is already uniform; the vocabulary around it is not.** 489 of 492 public producers have the same `getCommandNodes(graph)` shape and 470 of 491 take `output`. The exceptions are few and nameable: six raster classes with only `addToGraph`, eleven point-cloud classes with outputs at the top level, status in five shapes, per-row status codes whose values mean different things in sibling models (1 is "singular" in GWR and "excluded" in MGWR), eight internal switches visible in public props.
5. **Several gates pass when nothing ran.** The export check passes against a build that is missing 156 current source files. One structural spec iterates an empty map and another has no assertion. At least two numeric comparators accept NaN where a finite value is expected, and 81 specs define their own. Seven named headless cases return green with no device. The oracle runner exits 0 after comparing nothing. The contract baseline compares counts, so replacing one allowed loop with another passes.
6. **Facts are written down in many places and have drifted.** The operator catalog has 554 hand-maintained records; 63 of them are missing from their capability family, two different public recipes share one record because identity is the bare export name, and 226 records say "no status, no readback" as a default, not as a fact. Docs: 107 API pages, 98 in the sidebar, 66 snippet-checked. Roadmap: 12 of 15 sampled rows describe work that has shipped or changed. luma.gl is named as alpha.12, alpha.13 and alpha.14 in different documents; the pins are alpha.14 and alpha.16 was published on the day of this review.
7. **A stranger could not install it yet.** `apache-arrow` is an optional peer but four of eleven entry points import it at load. The manifest says MIT while 19 shipped source files carry Apache-2.0 or BSD-3-Clause expressions. Every relative link in the package README leaves the published files. Nothing checks the packed tarball. The package has never been published.
8. **The work loop is too heavy for the machine it runs on.** `yarn typecheck-tests` replaces the caller's heap limit with 16 GB and pulls 223 MB of JSON fixtures into the type program. The build always starts cold. There is no single verify command. This review itself could not run the heavy gates (see Method).
9. **Growth has outrun review.** Source grew from about 317,000 to about 542,000 lines in the eight days since the last library review, and many of that review's align and merge rows (STD, DUP) are still open. The next period should finish and connect what exists: fix the verified defects, make an installable alpha with three working journeys, and adopt one new habit, a test that would fail if a plausible wrong implementation were substituted, before adding more operators.

## Method and what was checked

- **Who.** The lead reviewer (Claude Fable 5.2) set the briefs, did a first census, cross-checked claims and wrote this document. Thirteen Codex `gpt-6.2-astra` sessions each reviewed one area, read-only, against one brief, and wrote the `area-*.md` files. Seven more `gpt-6.2-astra` sessions, which had not seen the code before, tried to refute the 38 findings graded P0 ([verification.md](./verification.md)). A separate `gpt-6.2-astra` session acted as thinking partner for two rounds ([thinking-partner.md](./thinking-partner.md)); it changed the conclusions in several places, noted below.
- **Run.** `node scripts/check-upstream-first.mjs` (no violations). `node scripts/check-contributor-contract.mjs` (no new findings; the baseline holds 118 loop sites in 85 files, 21 root files, 9 multi-class files and 1 storage preflight). Census scripts over source (syntax parsing and pattern counts, no type analysis). CPU reproductions: reviewers and verifiers executed the public CPU functions (`compute*OnCPU`, packers, the Arrow preparation, catalog helpers) by transpiling them in memory, and printed the numbers quoted in the reports.
- **Not run.** `yarn typecheck`, `yarn typecheck-tests`, `yarn build`, `yarn lint`, `yarn test-node`, `yarn test-headless`. The machine had 4.0 GB of disk free and 18 of 19 GB of swap in use when the review started, mostly held by two orphaned `tsc` processes from earlier sessions; a typecheck or test run would have filled the disk. No GPU code was executed by anyone. Every statement about what a shader returns is traced from source and marked UNVERIFIED in the reports.
- **Re-checked by the lead in source.** The `validity` codes of population alignment, the missing AICc denominator guard, the unchecked slice helper, pair gather's inherited flags, the catalog tier default, the Moran harness that rebuilds the weights port without its status, the CI workflow, the version pins, and that `importGraphBuffer` is exported only from the unpublished `testing` folder.
- **Hypotheses that were wrong.** The lead's first pass said there was no CI (a workflow exists; by the maintainer's notes GitHub Actions has never produced a run in this repository), that docs had no end-to-end example (the variogram walkthrough is one), and that a generic runner was the missing front door (the partner argued, with the code, that `importGraphBuffer` is two public luma.gl calls and that a second graph vocabulary here would be the wrong API; the proposal below is narrower).

## Files

| File | Scope | Findings (P0 / P1 / P2 as reported) |
| --- | --- | --- |
| [area-api-ergonomics.md](./area-api-ergonomics.md) | Six end-to-end user workflows, naming and shape census over 930 props and output types, entry points, errors | 3 / 5 / 4 |
| [area-core-contract.md](./area-core-contract.md) | `analysis-core`, the contract, the checker and its baseline, structural specs | 5 / 7 / 3 |
| [area-spatial-stats.md](./area-spatial-stats.md) | Regression, weights, autocorrelation, inference, validation, 19 directories | 4 / 7 / 3 |
| [area-spatial-geometry.md](./area-spatial-geometry.md) | Geometry, lines, polygons, joins, search: 80 contributors with a layout table | 4 / 7 / 1 |
| [area-spatial-points.md](./area-spatial-points.md) | Density, clustering, regionalization, interpolation, scan, space-time | 3 / 6 / 3 |
| [area-spatial-cells-trajectories.md](./area-spatial-cells-trajectories.md) | Cells and DGGS, trajectories, urban, landscape, recipes | 5 / 7 / 3 |
| [area-network.md](./area-network.md) | `gpu-network`: 41 contributors with an input table, overlap with `@luma.gl/graph` | 4 / 7 / 0 |
| [area-raster-terrain.md](./area-raster-terrain.md) | `raster-analysis` and `gpu-terrain` with a port matrix | 5 / 6 / 4 |
| [area-dataframe-misc.md](./area-dataframe-misc.md) | Dataframe, point cloud, geospatial, computation record, Arrow helpers | 3 / 5 / 4 |
| [area-ontology-catalog.md](./area-ontology-catalog.md) | Catalog, coverage matrix, API docs: reconciliation of exports, records, headings | 1 / 7 / 4 |
| [area-lenses.md](./area-lenses.md) | `@luma-contributors/lenses`: vocabulary, 413 records and 1,858 twins, coverage cost | 1 / 5 / 4 |
| [area-tooling-tests.md](./area-tooling-tests.md) | Build, packaging, typecheck, test census, checkers, release path | 0 / 9 / 6 |
| [area-docs-devdocs.md](./area-docs-devdocs.md) | READMEs, docs, dev-docs: 40 factual claims checked, roadmap census, 15 rows verified | 0 / 5 / 5 |
| [verification.md](./verification.md) | The refutation attempts on the 38 P0 findings | |
| [thinking-partner.md](./thinking-partner.md) | Two rounds of hypotheses and replies | |

Finding IDs below (ERG, CORE, SSR, GEO, PTS, DGGS, TRJ, RCP, NET, RT, DPC, CAT, LENS, TOOL, DOC) are the IDs in those files. Each finding there has the lines, the reproduction and the proposed change.

## Verified defects

The lead's grade is not the reviewers': it weighs how ordinary the input is and whether anything tells the caller. "CPU" means the wrong number was reproduced by running the public CPU function; "source" means it was traced in source and the GPU result is unverified.

**Fix first: ordinary use, nothing tells the caller.**

| ID | Where | What goes wrong | Evidence |
| --- | --- | --- | --- |
| ERG-01 | `GPUPopulationSupportAlignment` and its twin | Output `validity` means 0 matched, 1 missing, 2 ambiguous. The contract says nonzero is valid. Passed as a mask to the next operator it keeps only the rows that failed; a fully matched table loses every row. | CPU |
| GEO-02 | `GPUNearestFeatureJoin` into `GPUPairGather` | With `features.sourceIds` the join publishes application IDs; gather reads the same slot as a feature row. IDs `[1, 0]` gather the other feature's attributes with no flag. | source, mirrored on CPU |
| GEO-03 | `GPUPairGather` after a predicate join | Gather inherits `pairs.status.overflow` and ignores `candidateOverflow`. One left feature with nine matches at the default candidate capacity of eight yields eight rows and overflow 0. | source, mirrored on CPU |
| NET-01 | `GPUMapMatching` with `GPUNetworkCostTable` | A cost table that stopped at its round budget marks unreached transitions resolved; the matcher reads them as "no route". The public composition example wires only `overflow`. | source, CPU matcher |
| RT-03 | Terrain horizon sweep | The same terrain cropped with the advertised halo follows a different ray: a blocker three cells inside the window changes a horizon from about 84 degrees to 0. | CPU oracles |
| TRJ-02 | Brownian bridge motion variance | The optimizer excludes zero. For a track whose movement is smaller than its location error, zero is the maximum; a positive floor is returned with no diagnostic. | CPU |
| SSR-03 | `GPUSpatialCrossValidatedOLS`, regime regression | Held-out prediction uses the float32 raw intercept instead of the centered form the fit uses. With projected-coordinate-sized predictors the prediction of 1 becomes 0, and cross-validation reports it as model error. | arithmetic |
| SSR-04 | Geographical detector p-value | The upper tail is computed as one minus a CDF and returns exactly 0 for probabilities near 1e-18; reachable with three strata, 303 observations and q = 0.4. A separate extreme input does not terminate. | CPU |
| PTS-01 | Sinkhorn divergence (CPU and GPU) | Convergence is tested on the cross term only. A two-cell case returns 0.00074 for a divergence of 0.00030 with `converged = 1` at any tolerance. | CPU |
| PTS-03 | `GPUHuffTradeAreas` on resubmission | When the CSR shrinks between submissions, probability slots past the new end keep the old values; the output promises zeros. | source |
| DPC-03 | Point-cloud local origin helper | One non-finite coordinate makes the recommended origin NaN and turns every finite point non-finite, in that cloud and in any other cloud that shares the origin. | CPU |
| LENS-01 | Lens result tables | Columns from different key spaces are joined by row position: station rainfall, cell variance and a run-wide count on one row. 47 of 53 committed result tables mix key spaces. | committed twins |

**Fix next: valid but uncommon input, option or call sequence; nothing tells the caller.**

| ID | Where | What goes wrong | Evidence |
| --- | --- | --- | --- |
| ERG-02 | `GPUSpatialJoinPrepared`, `GPUStudyWindowPreparation` | The dirty flag is cleared when the graph is encoded, not when it is submitted. Encode, discard, encode again: the rebuild is skipped. | extracted latch |
| ERG-03, CORE-03 | `GPUParameterBuffer` | A `float32` buffer accepts a `Uint32Array` (and the reverse) and writes the bytes: 1 becomes 1.4e-45 or 1,065,353,216. Initial values skip the range check. | host |
| CORE-01 | `sliceGraphChunkGroupView` (public) | No bounds check against the parent view: a slice one row past the end is a usable view of the neighbouring allocation. | host |
| CORE-02, DPC-01 | Arrow Decimal128 preparation, `exact` mode | The coefficient is rounded to a double before the exactness test: 1.0000000000000001 becomes 1 with reported error 0. | CPU |
| CORE-05 | `createGPURegressionModelResult` | Corrected AIC divides by `n - k - 1` without a sign check: infinite at `n = k + 1`, and smaller than AIC below it. | CPU |
| GEO-01 | Offset curve components (CPU and GPU) | Reversed-fragment cancellation uses an absolute floor of 1e-5. A 20 km line with two 3 cm edges loses vertices at a 10 km offset; at small coordinate scales a straight line becomes empty with required size 0. | CPU |
| RT-01 | Fractional zonal area, geographic grids | Signed row spacing is lost on a one-row window: a valid cell gets area 0. Reproduced at 70 degrees latitude. | CPU twin |
| RT-04 | Raster reprojection, nearest | Source and destination affines are packed to float32 separately; on a wide sheared grid an identical-grid reprojection reads the adjacent column. | arithmetic |
| RT-05 | `GPUTerrainDerivatives` | Sheared grids are accepted and the slope is the norm of two non-orthogonal differences. | source |
| NET-02 | Centrality schedule and continuation | The planner emits `reset: 0`; the runner tests `!== false`, so every batch after the first replaces the totals and clears an earlier failure. | host |
| NET-03 | `GPUNetworkIsochrones` | Two promised outputs may share a writable range; outputs are checked against inputs but not against each other. | source |
| NET-04 | Travel-time percentiles | A fixed `1e-9` subtracted before the nearest-rank ceiling selects the adjacent observation for some exact ranks. | CPU |
| DGGS-01 | `GPUCellTableCompare` | The signed difference of two exact sums wraps modulo 2^64: 1,048,576 rows at opposite values at the default scale give a negative delta for a positive change. | oracle |
| TRJ-01 | Track similarity, geographic | Point-to-segment projection across the antimeridian returns 180 degrees when both endpoints are within 100. | CPU |
| SSR-01 | Exact log-determinant spectrum helper | The reversibility check is absolute for small weights: a 47% mismatch passes for weights near 0.001 and the spectrum of a different matrix is returned. | CPU |
| SSR-02 | Conformal prediction intervals | A relative slack on the order-statistic rank moves it by several observations at about 8 million calibration rows, narrowing the interval. | arithmetic |

**Lower: extreme input, a visible status, or documented behaviour.** CORE-04 (a finite 1e39 packs to infinity in a finite-only schema field; also a default equal to its maximum crosses it after rounding), DPC-02 (period index at 2^31 hours), PTS-02 (a finite 1e30 used as "unreachable" makes dense and sparse location-allocation disagree), CAT-01 (validity tally counts only 0 and 1; reports the rest as unrecognized), DGGS-02 (a column named `__proto__` is dropped from DGGS-JSON), GEO-04 (the bounds filter treats longitude as plain numbers, as documented; the suggestion to feed it unwrapped bounds from geometry measures is what is wrong), RT-02 (fractional clipping snaps as documented; the tolerance policy erases a polygon at pixel index 2^20), RCP-01 (hot-spot permutation follows its documented family; the returned `testedRowsDefinition` describes a different one).

The 83 findings reported as P1 were not put through refutation. The ones that recur across areas are grouped in the next sections; the rest are in the area files.

## What the defects have in common

No single mechanism explains all of them (the conformal rank and the Sinkhorn stopping rule are wrong inside one kernel). What they share is how they would be found: by looking at the result from the next consumer's side. Five patterns cover about half of the verified list and most of the P1 findings.

1. **Completeness does not travel.** GEO-03, NET-01; also PTS-08 (an early-stopped permutation test reports 150 draws for 10), CORE-06 (the shared compact publisher folds candidate loss into `overflow` while contract section 9 says to keep it separate), and the Moran test harness, which rebuilds the weights port from buffers and drops the search's status. Growing the output buffer cannot recover lost candidates, so the two flags are not interchangeable.
2. **One view type, several meanings.** A `GraphDataView<'uint32'>` is a feature row, an application ID, a keep-mask, a validity flag, a failure bitmask or a diagnostic code (GEO-02, ERG-01, GEO-08, GEO-07, PTS-10, LENS-01). Branded types would not help: luma.gl's `createDataView` returns the plain view and a brand at allocation would certify what a shader writes later. What helps is a separate named port per meaning (`neighborRows` beside `neighborIds`, `matchedMask` beside `alignmentCode`) and a decoder for packed codes.
3. **Two forms of one operation disagree.** A CPU twin accepts input its GPU form rejects, ignores a declared port, or handles the budget differently (ERG-07, NET-05, NET-06, SSR-06, SSR-09, DOC-01, where the first documented computation silently truncates unequal arrays). The contract says one props object builds both; nothing checks it.
4. **Reuse and partial requests.** Outputs with no writer on a second submission or when requested alone (PTS-03, PTS-04), a latch that fires on encode (ERG-02), props objects kept by reference and re-read after validation (CORE-08: 390 `this.props = props` assignments).
5. **Same data, different framing.** Tiled against whole (RT-03, RT-06), padded against exact length (SSR-08: a regression reader labels padding as a coefficient; RT-12), texture against buffer bands (RT-09), a change of units or origin (GEO-01, PTS-07, PTS-09, SSR-01, SSR-03, GEO-05).

A sixth, narrower pattern is the float32 border: host code validates a JavaScript double and then packs float32 (CORE-04, SSR-05, where a significance level just below 1 packs to 1 and selects the default 0.05). There are 214 hand-written `getGPU*ParameterValues` packers and a declarative schema used by six files. The fix is not a migration: validate the encoded value before writing, in the schema and in the packers that are touched.

## Ergonomics and consistency

**The first run.** Six workflows written from the public docs alone (point density, network reachability, terrain derivatives, distance field, time-window filter, point-cloud statistical filter) each needed 42 to 44 lines, of which 35 are the same scaffold: device, buffers with usage flags, `importBuffer` plus `createDataView` per view, `graph.add`, compile, encode, submit, typed readback, cleanup. The guides show 12 to 23 of those lines. Tests carry the rest in helpers that are not published: `importGraphBuffer` in 676 test files, `createOutputBuffer` in 625, `submitGraph` in 755.

What to do about it, in order of cost:

- Link the variogram walkthrough from both READMEs as "run your first computation", move the MGWR section of the package README to the regression guide, and finish one complete GPU walkthrough (point density) that the five family pages link to instead of each showing a fragment. Fix ERG-05.
- **Output layout functions.** A contributor and its operation form each encode how long every output is (k-shortest paths: `queries x routes x pathLength` for vertices and edges, `queries x routes` for costs). One `get<Name>OutputLayout(options)` used by the constructor's validation, by the operation's declared outputs and by the docs table removes the duplicate and tells the caller what to allocate, while ownership stays with the caller. Pilot on k-shortest paths and one optional-output raster family before generalising.
- **Decoders.** `decodeGPUAnalysisSpacingStatistics` is the model: it checks the layout version, respects the byte offset and returns named fields. K-shortest paths, the regression model readers and every packed status word want the same.
- **A few domain conveniences that own real behaviour**, not a generic runner: prepared point weights for local Moran that keep the search's status, a cost-distance helper that owns bounded continuation, perhaps a one-shot point density. `createNetworkCentralityContinuation` is the precedent.
- **Upstream:** a typed raw-buffer view import (`graph.importBufferView`) and a typed-array constructor for CPU graph vectors are generic and belong in luma.gl.

**Three forms.** Contributor, operation form with CPU twin, and recipe each do a different job and should stay. What is missing is telling the reader which exists for a given operator (an operation form exists for 26% of producers) and one worked recipe taught by the question it answers.

**Naming standard for new and changed APIs** (from the census in [area-api-ergonomics.md](./area-api-ergonomics.md); existing names change only where the meaning is the same):

| Meaning | Dominant today | Standard |
| --- | --- | --- |
| Row count | derived from the view; `rowCount`, `cellCount`, `nodeCount` | Derive from `view.length`; `rowCount` for an explicit host count; `<entity>Count` otherwise; `status.count` for produced rows |
| Capacity | 45 spellings | `<entity>Capacity`, with what occupies one slot documented |
| Maximum | `maximum*` 459 against `max*` 21; `maximumNeighborCount` and `maximumNeighbors` coexist | `maximum<Entity>Count`, `maximumIterations` |
| Row selection and validity | `mask` 139; `validity`, `outputValidity`, `outputMask` | `mask` selects input rows; `validity` is nonzero-usable output; `<reason>Code` for diagnostics |
| Identity | `sourceIds`, `rowIds`, `indices`, `ids` | `<role>Ids` for caller identities, `<role>Rows` for row indices, never one name for both |
| Coordinates | `positions` 97, `points` 7 | `positions` or `<role>Positions` |
| Outputs | `output` 483, `outputs` 10 | `output` on a producer, `outputs` on a recipe result |
| Status | `output.status` in five shapes | named `output.status` for whole-operation conditions; `<role>Status` for per-row codes with an exported decoder |

**The short exception list** (the package is `0.1.0-alpha.0` and unpublished, so these can change now): six raster classes with only `addToGraph` (NDWI, NBR, EVI, SAVI, embedding k-means, embedding nearest pixels); eleven point-cloud producers without `output`; eight `@internal` props reachable from public props types (ERG-09); `rebuildWhen: (parameters: any)` on the two prepared handles; `GPULonLatPolygonPreparation`, whose optional `spatialContext` always throws when omitted (GEO-09); forest inference's either-or inputs that should be a union; errors that say "outputs must not share buffers with inputs" without naming the pair.

**Structure.** `analysis-core` is a sound shared layer whose public barrel mixes four things: the authoring contract, Arrow conversion, fitting contributors and whole geometry implementations (30% of its lines). An author who imports `validateCapacity` loads `apache-arrow`. Its most used helper, `validateSinglePackedView` (1,790 calls in 435 files), is not exported. The "root holds only `index.ts`" rule is checked for five of eight roots; the other three hold 30 loose files. The catalog for every entry point lives under `gpu-spatial-analysis/ontology` for historical reasons.

## Inventories and documents

- **Catalog.** Fix now: identity by entry-point subpath plus export name (both visibility recipes are in one npm package, so the package name is not enough); derive family membership from the records (63 are missing); make "unknown" an explicit value for status, readback and tier instead of `[]`, `'none'` and `'scale'`; freeze returned records; implement the kind guard against the canonical list (it rejects `time-series`). Later, and only if hand maintenance keeps failing: sibling descriptor files feeding a generated manifest. That design is in the catalog report and is deliberately not in the near plan.
- **Docs.** One discovery of API pages for the sidebar, the snippet check and the parameters generator (nine pages are not in the sidebar, 17 to 25 code fences are unchecked). The generated Parameters blocks assign a packer to any class whose file imports it, so `GPUAnalysisContiguityStatistics` documents four settings the caller cannot supply, and two point-cloud blocks sit under the wrong class. Ten package-local guides are outside both the website loader and the package files.
- **Stated limits.** Several published limits describe a superseded route: forest inference (16,777,216 rows and "single packed buffers" against chunked input up to 2^32 - 1), embedding codecs (128 against 1,024 dimensions), strongly connected components (1,024 vertices is the exact route only). A limit needs a subject and a condition: total or per chunk, which route, which layout.
- **Contract.** Section 14 attributes the one-chunk restriction to the luma.gl executor at alpha.12; the installed executor handles chunks and the restriction is in the local lowering (`chunks.length === 1`). Section 6 points to a roadmap row that no longer exists. Section 11 carries compiler history that belongs in its cited evidence. A table in the core report says, section by section, what is machine-checked.
- **Roadmap.** 214 prioritised rows (10 H, 90 M, 99 L, 15 decisions) plus five pass narratives with their own tables. Three IDs are used twice for different things. Of 15 rows checked, 12 need closing or narrowing (centrality continuation, 1,024-dimension embeddings, Boruvka rounds, S2 compaction, parallel depression activation, leak specs all exist). The rule "delete completed rows" is not being followed by the passes that land work.
- **No authoring walkthrough.** AGENTS.md and the contract give rules; nothing walks one real contributor from props to catalog entry, audit case and docs page. Coregionalization fit is a complete example to write it around.

## Gates and tooling

| Gate | How it passes without checking | Finding |
| --- | --- | --- |
| `yarn verify-package-exports` | Imports workspace packages; passes against a stale `dist`; compares export counts, not names; skips wildcard entries | TOOL-02 |
| `analysis-layering` spec | `DOCUMENTED_ENTRIES = {}`; the documented-contributor loop has nothing to iterate | CORE-11 |
| `analysis-runtime-boundaries` spec | Walks imports and never asserts against a forbidden target | CORE-11 |
| Numeric comparators | `Math.abs(value - expected) > tolerance` is false for NaN; 81 specs define their own comparator | TOOL-01 |
| Headless specs | `if (!device) return` at four sites, seven named cases | TOOL-09 |
| `yarn oracle-fixture` | Exit 0 when no output was written or an expected file is missing | TOOL-06 |
| Contract baseline | A file's count may not grow; a different loop in place of a removed one passes; 24 of 85 files have no reason | CORE-10, TOOL-13 |
| Registry check | A text search of the ontology source; a comment satisfies it | TOOL-12 |
| `yarn typecheck-status` | Trusts the last line of a log; a dead watcher's "Found 0 errors" from 2026-11-11 still reads as success | TOOL-05 |
| Shader memory estimate | An array sized by a named constant counts as 4 bytes; used in about 50 specs | CORE-12 |

Also: `yarn typecheck-tests` hard-codes heaps of 16, 16 and 8 GB whatever `NODE_OPTIONS` says, and AGENTS.md says the opposite; ten benchmark files run in ordinary headless runs and 19 of 29 listed benchmarks skip under `yarn test-benchmarks`; the build deletes its incremental state and starts every CommonJS transform at once; Biome does not cover `scripts/`; there is no `yarn verify`.

## Release readiness

To publish `@luma-contributors/analysis` as an alpha a stranger can use:

1. Build once, pack once, install that tarball in an empty consumer outside the workspace, import every entry point as ESM and CommonJS, resolve the types, and run the documented first computations there. Promote those bytes.
2. Decide Arrow: declare it a required peer for this alpha (smallest change), or move the Arrow adapters behind their own subpath and prove the other entry points load without it.
3. Licences: ship the licence texts and a notice inventory for the 19 files that are not MIT-only (H3 tables, d3-hexagon and others); keep their SPDX expressions.
4. README: stable URLs instead of links into the repository, the tested luma.gl version stated once, the first computation at the top.
5. Peer range: `~10.0.0-alpha.14` also accepts 10.0.0 and 10.0.x; state the tested version and the intended range separately. Decide the repository URL (`visgl` in the manifest, a private remote in fact).
6. Fix the verified defects on the paths the release documents, or restrict those configurations in the docs.

Lenses stays private: its files include the datasets under third-party licences, five built loaders import a directory (`computation-pack.js` does not exist), 91 dynamic JSON imports lack import attributes, and one registration module is missing from `sideEffects`.

## Keep

- Caller-owned execution as the low-level API, and the uniform `getCommandNodes(graph)` shape.
- One kernel builder (`createWGSLKernelNode`, 2,396 calls) and the named-limit validators with ID-prefixed messages.
- The split between host planning and graph declaration in `graph-chunk-dispatch` and `graph-chunk-gather`, with documented ownership.
- The model-result, inference and temporal-context contracts.
- Independent oracles with provenance (PySAL, SciPy, NumPy, closed forms; 282 Python generators with recorded environments) and specs that state which outputs are compared exactly, absolutely and relatively.
- Separate public classes where the statistical model differs, with a compatibility table (`kriging-compatibility.md` is the model for KDE and clustering families).
- The discriminated union in geometry cleanup, `makeGPURasterBand`, the spacing decoder, HDBSCAN's pairing of a construction-time node estimate with execution gates.
- The variogram walkthrough and its spec that allows no implicit inputs.
- A catalog that can be read without constructing a contributor, and the independent check of the catalog against the barrels.
- The pure view contract in lenses and capture as an opt-in step.
- The narrow package root, family subpaths and one-to-one CommonJS layout.

## Plan

The order came out of the second partner round: packaging can start at once, but the two completeness pilots should land before anything is published, or the release teaches the traps.

| Phase | What | Roadmap rows |
| --- | --- | --- |
| A. Make results and gates trustworthy | Fix the verified defects, most reachable first, each with a test that fails before the fix. Repair the gates that pass when nothing ran. One bounded local verify command that honours the caller's heap limit. | LR-01 to LR-03 |
| B. An installable alpha (in parallel with C pilots) | Packed-consumer check, Arrow decision, licences, README and first-run path, documentation facts. | LR-04 to LR-06 |
| C. The result contract | Completeness pilots on join to gather, weights to statistics and cost table to map matching. Named ports for rows, IDs, masks and codes. Twin parity. The float32 border. Output layout functions and decoders. The API exception list. | LR-07 to LR-13 |
| Publish | The tarball that passed B, once the C pilots are in and the documented paths have no known wrong result. | |
| D. Inventories | Catalog identity and derived membership, one docs inventory, roadmap reconciliation, structure. | LR-14 to LR-18 |
| E. The loop and the rest | Typecheck and test memory, lenses fixes, record identity, luma.gl bump cadence, upstream filings. | LR-19 to LR-23 |

**One habit to adopt with phase A** (it changes the defect profile more than any structural work): every new or changed public result contract lands with one discriminating test. Name a plausible wrong implementation that would pass the ordinary fixture (ignore the candidate flag, swap row and ID, skip the tail write, read the raw intercept), give the smallest input that tells them apart, and assert the consumer-visible result. For a fix, the test is recorded failing on the parent revision. For a join the fixture is: one left row, two true matches, final capacity two, candidate capacity one; assert one row and an incomplete status; then on the same compiled graph change the data so one candidate exists and assert the row and a cleared status. That catches the dropped flag, a sticky flag and a second frame that was never submitted. Reviewer briefs should ask for exactly this instead of "check the tests pass".

**Not now.** These were proposed by reviewers and solve real problems, but together they are several new systems for one maintainer during a correctness release: a neutral catalog entry with a generated checked-in manifest, generated roadmap views, a separate lenses workspace with incremental replay tracking, migrating the 214 packers to the schema, a rescaling fixture for every contributor, operation forms for every producer, family-level TypeScript project references before the test-side memory work has been measured. New operator families should also wait until the alpha is out, except for an identified consumer.

## Decisions for the maintainer

1. **Freeze breadth until the alpha?** Recommended: yes. Correctness fixes and work a named consumer needs continue.
2. **Arrow: required peer, or optional with an isolated subpath?** Recommended: required for the first alpha.
3. **Use the unpublished state to break names now?** The exception list above, `alignmentCode` and `matchedMask` replacing the legacy `validity`, `neighborRows`, separate final and candidate overflow in the shared publisher. Recommended: yes, in one pass per family, keeping deprecated aliases only where lenses or kepler read them.
4. **What does an incomplete input do to a consumer?** Propagate and continue (gather), suppress inference (Moran), fall back (map matching to A*)? Recommended: each consumer documents one of the three; none may ignore the flag.
5. **CPU budget semantics.** Does an explicit `maximumIterations` bind a CPU twin as it binds the GPU form (MGWR ignores it by design)? Recommended: yes; name the unlimited policy if it is wanted.
6. **Lens coverage rule.** One captured scenario per public class (89 pending), or a named family scenario for variants? Recommended: family scenarios.
7. **CI.** GitHub Actions has not run here. Either find out why, or accept that `yarn verify` on one machine is the gate and size it to that machine (state a peak memory ceiling).
8. **Who files upstream.** Nothing in [../../upstream-proposals.md](../../upstream-proposals.md) has been filed. Recommended first: the packed CPU/GPU transfer path with its benchmark (about 90 ms per million `float32x2` rows at alpha.12, and a row-by-row loop still present at alpha.14), the `GPUHistogram` edge and domain cases (a committed spec holds the expected assertions), graph algorithms on caller CSR with masks (retires local topology code), then a typed buffer-view import.

## Limits of this review

- Nothing was compiled, tested or run on a GPU. The heavy gates were skipped for lack of disk and swap; their last recorded results are in the roadmap's pass notes, not here.
- No reviewer read all of its scope. Each area file lists what was read in full, sampled and not opened. The regression family, the catalog, analysis-core and the tooling scripts were read most closely; emerging hot spots, point process, global coordinates, grid generators and most WGSL kernel bodies least.
- Counts are syntax or pattern matches, not type analysis. Where two reports count the same thing differently (producers 491, 492 or 495; API code fences 184 or 192) the difference is in what was included, and each report says.
- The 38 P0 findings were independently attacked; the 83 P1 and 44 P2 findings were not. Treat a P1 as a lead with evidence, not a verified defect.
- Verifiers confirmed CPU behaviour and traced GPU source. A GPU path that mirrors a confirmed CPU defect is likely wrong too, but that is an inference until a headless spec shows it. In this repository reviewers' statements about what a GPU returns have been wrong before.
- The reviewers and verifiers were all the same model family. An error they share would not have been caught by the refutation round; the lead's own source checks covered nine items.
