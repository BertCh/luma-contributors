# Documentation and developer documents

**The repository already has a good first computation, but neither landing README leads the reader to it.** The more consequential problem is that the same API facts are maintained independently in guides, package READMEs, the coverage matrix and the roadmap. I found **13 failures in 40 targeted factual checks**. One reaches runtime behavior: the CPU variogram fit silently discards unmatched input bins that the GPU constructor rejects.

The roadmap has **214 formally prioritized rows**. Of 15 deliberately selected rows checked against source or configuration, **12 need closure, a narrower scope, or corrected evidence**. These are targeted samples weighted toward recent changes and suspected drift. Neither percentage estimates the defect rate of the whole documentation set.

## What I checked

- Read the root README, changelog, AGENTS instructions, both package READMEs, contributor contract, developer-doc index and research index in full. Traced the CPU variogram walkthrough from public import through fitting and the GPU walkthrough through graph cleanup.
- Counted all 107 `docs/api` Markdown pages, all 10 package-local analysis guides, all roadmap domain rows, and all 18 research directories. Sampled the largest API page, `docs/api/gpu-spatial-analysis/operators-statistics.md` at 2,617 lines, the latest SC/IMP/AX changes, and ordinary guides on HAND, embedding codecs, model fitting, centrality and computation records. The 40-claim and 15-row ledgers below identify the checks individually.
- Used the installed alpha.14 source, package manifests, public barrels, constructors, and committed specs as evidence. Read the earlier review indexes and searched the roadmap by subject and ID. Read website and example code only to establish how they consume documentation or research.
- The check was read-only at HEAD `bdc715ea`. I did not compile, run tests, execute shaders, check npm publication, verify external issue trackers, or validate the numerical contents of the 2,271 generated lens-record files. Runtime and cross-device results are **UNVERIFIED** in this review.

## Findings

### DOC-01 The first CPU computation silently accepts bin arrays the GPU rejects [P1] [correctness]

- **Evidence.** The walkthrough specifies equal-length distances, semivariances and pair counts. The contributor contract says a pure twin checks lengths that must agree and names the offending argument. The CPU fit instead takes `Math.min` of all three lengths and passes only that prefix to the fitter. Its graph wrapper calls the same function. The GPU constructor checks both length equalities and throws. `docs/api/operation-forms-and-cpu-twins.md:181`, `dev-docs/contributor-contract.md:98`, `modules/analysis/src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:77`, `:120`, `:210`, `modules/analysis/src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts:100`.
- **Why it matters.** A missing count at the end of an otherwise usable dataset changes the observations used for the fit. The CPU path can return ordinary parameters and a normal fit status for the prefix. Switching the example to the GPU path changes how the same input error surfaces.
- **Proposed change.** Check both lengths at entry to `computeVariogramModelFitOnCPU`, report the expected and supplied length by argument name, and add a node regression with valid leading bins and one missing trailing count. The wrapper inherits the fix. **S, behavior change for malformed input.**
- **Confidence.** High. This follows directly from the host code.

### DOC-02 The README-to-result path stops before the existing walkthrough [P1] [ergonomics]

- **Evidence.** The root README gives example directories, a repository map and maintainer build commands. Its `docs/api` entry is plain code-formatted text. The analysis README links domain references, then spends lines 26–74 on MGWR. The actual first computation creates its own 20 bins, calls one public function and shows the expected parameter row. Its GPU continuation assumes an existing `Device`. The spatial overview also assumes `device`, four graph views and viewport bounds. `README.md:12`, `:28`, `:36`, `modules/analysis/README.md:23`, `:26`, `:73`, `docs/api/operation-forms-and-cpu-twins.md:46`, `:68`, `:80`, `docs/api/gpu-spatial-analysis/README.md:40`.
- **Why it matters.** A new user can get a CPU result after finding the right page. A reader who follows the prominently documented setup instead installs and builds a large contributor workspace before learning what to import. The GPU examples are usable integration fragments for someone who already owns the missing resources.
- **Proposed change.** Put a clickable **Run your first computation** link in both READMEs. Give the CPU example an installation or workspace invocation and the expected model. Add one browser entry point that creates the device, calls `fitVariogramInGraph`, displays the result and releases resources. Move the MGWR discussion intact to the regression guide and put the transpose link with spatial weights. **S for the README and move, M for the browser example. Nonbreaking.**
- **Confidence.** High on source and navigation. Fresh-install execution is **UNVERIFIED**.

### DOC-03 The roadmap can assign work that has already shipped [P1] [architecture]

- **Evidence.** The roadmap says “Delete completed rows.” It also says SC and IMP are uncommitted, although the implementation and documentation landed in `69305eb4` and `bdc715ea`. The IMP summary reports centrality continuation, 1,024-dimensional embeddings and Borůvka region graphs. Their domain rows still describe an upstream blocker, “Not implemented,” and outstanding Borůvka rounds. Of 28 AX status rows, 18 have an empty `Open` cell. `dev-docs/roadmap.md:9`, `:66`, `:88`, `:90`, `:148`, `:301`, `:434`, `:477`. The row-by-row source checks are below.
- **Why it matters.** Two readers can make opposite prioritization decisions depending on whether they start at the summary, pass narrative or domain table. The high-priority test-exclusion row is off by two orders of magnitude on the run exclusions.
- **Proposed change.** Keep one authoritative open record per ID with an unmet acceptance condition, current evidence, domain and priority. Generate the high-priority summary. Give decisions distinct IDs and link them to implementation items. Delete landed wave tables after putting caller-facing changes in the changelog and retaining one completion-log line. Reconcile the 15 sampled rows first. **M, nonbreaking.**
- **Confidence.** High. `KNOWN(NW-CENT)`, `KNOWN(EMB-02)`, `KNOWN(GPU-13)`, `KNOWN(KE-TESTS)` have materially stale premises.

### DOC-04 The public reference understates capabilities that callers can use today [P1] [docs]

- **Evidence.** The embedding codec page says both “1–1,024” with chunked input and “1–128 dimensions” with single packed buffers. The source distinguishes a 1,024 streaming limit from a 128 reference limit, and the scale spec enumerates **192, 256, 384 and 1,024**. The coverage matrix gives SCC a universal 1,024-vertex ceiling, while `GPUNetworkStructure` selects a parallel SCC route above that exact-mode limit. The catalog introduction says “`analysis-core` exports none” immediately before explaining how to discover contributors. Four public fitting contributors are already registered there. `docs/api/gpu-raster/raster-embedding-codecs.md:9`, `:87`, `modules/analysis/src/raster-analysis/raster-embeddings/raster-embedding-common.ts:15`, `modules/analysis/test/gpu-raster/raster-embeddings/embedding-scale.node.spec.ts:178`, `dev-docs/coverage-matrix.md:256`, `modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts:85`, `docs/api/analysis-catalog.md:19`, `modules/analysis/src/gpu-spatial-analysis/ontology/fitting-kernel-catalog.ts:23`.
- **Why it matters.** A caller can unnecessarily repack or tile data, rule out the library for a larger network, or build a fitting primitive that already has a public import.
- **Proposed change.** Correct these three pages now. Generate the entry-point inventory from the catalog. Give family limits separate entries for execution mode, input layout, and binding assumptions. Use the codec scale spec as the source for its supported-width examples. **S for corrections, M for generation. Nonbreaking.**
- **Confidence.** High. `KNOWN(EMB-02)` and `KNOWN(STD-13)` understate the issue because shipped capabilities are already misrepresented in user documentation.

### DOC-05 The contributor contract assigns a bridge restriction to the upstream executor [P1] [docs]

- **Evidence.** Section 14 says a graph with a CPU stage or output requires “one packed chunk each” and uses that restriction to determine eligibility for an operation form. The alpha.14 executor checks declared chunk lengths against multiple CPU chunks and accepts multi-chunk callback results. The **local GPU contributor bridge** explicitly checks `chunks.length === 1`. The local CPU-row helper calls `input.pack()` when multiple chunks arrive. The published executor does still reject textures and state vectors. `dev-docs/contributor-contract.md:101`, `node_modules/@luma.gl/graph/src/gpu-core/gpu-cpu-execution.ts:37`, `:136`, `:203`, `modules/analysis/src/analysis-core/operations/gpu-contributor-operation.ts:220`, `modules/analysis/src/analysis-core/operations/cpu-operation-rows.ts:27`.
- **Why it matters.** An author deciding whether to write a CPU operation, a GPU wrapper or a chunk-capable contributor gets the wrong design boundary. A CPU callback can also incur a packing copy that the current contract gives the reader little reason to expect.
- **Proposed change.** Replace the alpha.12 paragraph with a small support table for the published executor, local GPU lowering and local CPU-row adapter. Include where a copy occurs. Keep the explicitly dated alpha.12 timing measurements in section 14 as historical measurements. **S, nonbreaking.**
- **Confidence.** High on the control flow. Runtime behavior on a particular GPU is **UNVERIFIED**.

### DOC-06 Navigation, page publication and snippet checks use different inventories [P2] [testing]

- **Evidence.** There are **107** `docs/api` pages, **98** distinct TOC entries and **66** pages in the shared analysis snippet/parameter list. The sidebar consumes the TOC, while the page loader can load all Markdown under `docs`. All 98 TOC targets exist. Nine pages lack sidebar entries, including forest inference, k-shortest paths, visibility graphs, embedding codecs and Gaussian random fields. Of 192 `ts` or `typescript` fences under `docs/api`, the generic snippet scanner can discover 167 before explicit opt-outs. Its page list misses 19 fences and its exact ` ```ts ` spelling misses six more on listed pages. `docs/table-of-contents.json:5`, `website/src/docs.ts:18`, `:33`, `modules/analysis/test/analysis-core/doc-program.ts:21`, `:87`, `modules/analysis/test/analysis-core/doc-snippets.node.spec.ts:57`.
- **Why it matters.** A new guide can render at a direct URL, be absent from navigation and miss the primary snippet check. An existing test proves TOC membership for the variogram walkthrough, so the repository already demonstrates the desired invariant for one page.
- **Proposed change.** Discover Markdown pages once. Give intentional sidebar omissions an explicit reason, accept both TypeScript fence spellings, and retain per-fragment opt-outs. Preserve the stricter no-implicit-input test for complete walkthroughs. **M, nonbreaking.** `modules/analysis/test/analysis-core/operation-forms-docs.node.spec.ts:27`, `:36`.
- **Confidence.** High. The fence count describes this scanner’s coverage, not the outcome of running tests.

### DOC-07 Package-local guides are outside the docs loader and package file list [P2] [packaging]

- **Evidence.** Ten guides live in `modules/analysis/docs`. The website loader reads `docs/**/*.md`; the analysis package file list includes `src`, `dist` and `README.md`. The package README links a relative `docs/chunked-spatial-weights-transpose.md` and a `test/` fixture generator. Neither target is in that file list. The live operator metadata names package-local documentation paths, including the coregionalization guide, while the main API also describes the same limits. `modules/analysis/package.json:75`, `modules/analysis/README.md:73`, `:76`, `website/src/docs.ts:18`, `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/entry-point-operator-details.ts:1605`, `modules/analysis/docs/coregionalization-fit.md:111`, `docs/api/gpu-dataframe-analysis.md:1784`.
- **Why it matters.** A link valid in a repository checkout can lead an installed-package reader to a missing file. Maintainers also have two places to update the same numerical contract.
- **Proposed change.** Move the ten guides into their owning `docs/api` families, merge duplicated sections and update the catalog `documentation` paths in the same change. Link the distributed README to stable documentation URLs and fixture permalinks. Delete old copies after checking their consumers. **M, nonbreaking for the library API.** Preserve redirects where a public URL exists.
- **Confidence.** High on manifests and the loader. `KNOWN(STD-12)` is under-prioritized where the split affects distributed README links.

### DOC-08 Setup and release documents describe an older checkout [P2] [consistency]

- **Evidence.** The root README, analysis README, AGENTS, changelog, upstream-first plan and roadmap still say alpha.13. Root pins and resolutions are alpha.14; the analysis peer declarations are `~10.0.0-alpha.14`. The lenses README describes graph-only upstream dependencies, while its manifest declares core, graph and shaders and its view registry imports core directly. `README.md:4`, `modules/analysis/README.md:4`, `AGENTS.md:21`, `CHANGELOG.md:14`, `dev-docs/upstream-first-plan.md:3`, `dev-docs/roadmap.md:3`, `package.json:42`, `:74`, `modules/analysis/package.json:89`, `modules/lenses/README.md:12`, `modules/lenses/package.json:63`, `modules/lenses/src/computation-views/registry/view-registry.ts:5`.
- **Why it matters.** Users are directed to a different prerelease than the one used by the workspace. Contributors also get incorrect advice about tuning the test typecheck. The runner overrides the caller’s heap separately for all three processes. Several changelog references point to roadmap sections that no longer exist. `AGENTS.md:7`, `scripts/typecheck-tests.mjs:152`, `:161`, `CHANGELOG.md:44`, `:45`, `:55`.
- **Proposed change.** Keep the supported release and peer range in one compatibility block sourced from the manifests. Keep historical release dates in the changelog. Point test-coverage instructions to the runner and exclusion manifest, and replace the numbered roadmap references with durable documents or commit links. **S, nonbreaking.**
- **Confidence.** High.

### DOC-09 Authors get the rules but no worked path through a complete change [P2] [ergonomics]

- **Evidence.** AGENTS points to the contract. The contract specifies the class, named limits, catalog entry, docs heading and kernel-audit case, but gives no worked authoring sequence. It also sends authors to roadmap `TC-01`, which is absent from the current roadmap even though a temporal-context guide documents the resolver. A complete example already exists across the coregionalization constructor, budget check, catalog record, audit case and API page. `AGENTS.md:54`, `dev-docs/contributor-contract.md:9`, `:38`, `:80`, `docs/api/gpu-spatial-analysis/temporal-context.md:35`, `modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts:79`, `:157`, `modules/analysis/test/analysis-core/kernel-audit/cases-coregionalization-fit.ts:42`, `docs/api/gpu-dataframe-analysis.md:1772`.
- **Why it matters.** A first-time author can build a correct kernel and still miss the public barrel, catalog metadata, size-bound audit or user-facing page. Those omissions appear in separate systems.
- **Proposed change.** Add `dev-docs/authoring-a-contributor.md`. Walk through one real contributor from props and caller-owned output through constructor validation, device preflight where applicable, graph nodes, public export, oracle, audit registration and documentation. Show the CPU twin as an optional branch. Link it from AGENTS and the contract, and replace `TC-01` with the resolver guide. **M, nonbreaking.**
- **Confidence.** High.

### DOC-10 The research retention rule needs to distinguish executable dependencies from citations [P2] [architecture]

- **Evidence.** Research contains **178 files in the tree, across 18 subdirectories plus the index**. The index omits the scalability audit, which the roadmap cites. The upgrade procedure runs three programs from a dated review directory. Showcase scripts parse oracle JSON from another. A core kernel comment cites an extended-precision report, and numerical specs cite math-review explanations. These are distinct obligations when moving or deleting material. `dev-docs/research/README.md:39`, `dev-docs/roadmap.md:55`, `dev-docs/upstream-workarounds.md:144`, `dev-docs/research/luma-usage-review-2026-11-11/scripts/exports.mjs:5`, `examples/deck/spatial-analysis-showcase/scripts/check-data-topology-reference.mjs:13`, `modules/analysis/src/analysis-core/kernels/wgsl-kernel-nodes.ts:361`, `modules/analysis/test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-rounded-directions.spec.ts:7`.
- **Why it matters.** Deleting an oracle breaks a consumer. Deleting a cited compiler investigation loses the rationale for unusual code. A closed coordination report can often be recovered from Git at little ongoing cost.
- **Proposed change.** Record the consumer and relationship for retained artifacts: executable input, maintained tool, implementation rationale, decision evidence or historical snapshot. Move the three upgrade utilities to `scripts/upstream/`; leave measured baselines with their provenance. Index the scalability audit. **M, nonbreaking if consumers move together.**
- **Confidence.** High for the identified dependencies. Absence of a literal-path reference alone is insufficient evidence for deletion.

### Evidence ledger — 40 factual claims

**27 passed, 13 failed, 32.5% failed.** I counted the repeated alpha pin once, the changelog’s obsolete section-number scheme once, and distinct API limits separately. Some failures were suggested in the brief. The sample also includes positive checks on ordinary, less frequently discussed documentation.

For compactness, `S/` means `modules/analysis/src/`, `D/` means `docs/api/`, `A/` means `modules/analysis/docs/`, and `R/` means `dev-docs/`. Paths after a shared directory name are relative to that directory.

| # | Documented claim | Result and verification |
|---:|---|---|
| 1 | `README.md:4` — current luma.gl is alpha.13 | **Fail.** Workspace pins are alpha.14, `package.json:42`. Other occurrences are listed in DOC-08. |
| 2 | `CHANGELOG.md:3` — analysis version is `0.1.0-alpha.0` | Pass. `modules/analysis/package.json:3`. |
| 3 | `README.md:43` — build produces ESM, CommonJS and declarations | Pass by build-script inspection. `scripts/build.mjs:5`, `:43`. |
| 4 | `modules/lenses/README.md:3` — lenses is private | Pass. `modules/lenses/package.json:4`. |
| 5 | `modules/lenses/README.md:12` — only analysis and published graph subpaths are dependencies | **Fail.** Direct math.gl dependencies and core/shaders peers are declared; core is imported by production source. `modules/lenses/package.json:58`, `:63`, `modules/lenses/src/computation-views/registry/view-registry.ts:5`. |
| 6 | `modules/lenses/README.md:20` — scenarios/index is the registry entry | Pass. `modules/lenses/package.json:32`, `modules/lenses/src/computation-lenses/scenarios/index.ts:7`. |
| 7 | `modules/lenses/README.md:19` — scenario registration is side-effectful and marked accordingly | Pass. `modules/lenses/src/computation-lenses/define-lens-scenarios.ts:38`, `modules/lenses/package.json:52`. |
| 8 | `modules/lenses/README.md:24` — the license field points to the dataset README | Pass. `modules/lenses/package.json:11`. |
| 9 | `AGENTS.md:9` — export verification loads “every package entry point” | **Fail.** The verifier skips wildcard and string entries. Lenses exposes wildcard data and computation-pack entries. `scripts/verify-package-exports.mjs:17`, `modules/lenses/package.json:41`. Document the scope or test representative concrete wildcard paths. |
| 10 | `AGENTS.md:7` — the test typecheck needs the caller’s same `NODE_OPTIONS` | **Fail.** It explicitly assigns 16,384, 16,384 and 8,192 MiB to its children. `scripts/typecheck-tests.mjs:152`, `:161`. |
| 11 | `README.md:42` — test typechecking omits the Kepler example specs as a group | **Fail.** A separate configuration includes example-backed files after filtering its named exceptions. `scripts/typecheck-tests.mjs:109`, `:114`. |
| 12 | `CHANGELOG.md:26` — the current exclusion description is 271 specs with the same exclusions as Vitest | **Fail.** The manifest has **1** `excludedFromRuns` key and **58** `excludedFromTypecheck` keys, and the runner treats them separately. `test/kepler-compute-example-specs.json:3`, `:6`, `scripts/typecheck-tests.mjs:103`. |
| 13 | `CHANGELOG.md:44`, `:45`, `:55` — current roadmap sections 12.7, 11.6, 11.8 and 16 locate the cited decisions | **Fail.** The roadmap now uses named domain and pass headings. `R/roadmap.md:96`, `:412`, `:525`, `:558`. |
| 14 | `R/coverage-matrix.md:11` — 144 inventory rows, 137 implemented, one partial, six excluded | Pass. Counting the status column gives exactly those values. The partial is at `R/coverage-matrix.md:235`; exclusions start at `:241`. |
| 15 | `R/coverage-matrix.md:20` — the analysis root exports only computation-record | Pass. `S/index.ts:5`. |
| 16 | `R/coverage-matrix.md:256` — SCC is limited to 1,024 vertices | **Fail.** This is the exact route’s cap. Larger auto-mode graphs select parallel SCC. `S/gpu-network/network-structure/gpu-network-structure.ts:53`, `:85`. |
| 17 | `R/coverage-matrix.md:254` — exact natural breaks can recompute through 8,192 rows | Pass. The source distinguishes materialized 1,024 and recomputed 8,192. `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:151`. |
| 18 | `D/analysis-catalog.md:19` — `analysis-core` exports no contributors | **Fail.** Four fitting producers are exported and catalogued. `S/analysis-core/index.ts:55`, `S/gpu-spatial-analysis/ontology/fitting-kernel-catalog.ts:23`. |
| 19 | `R/contributor-contract.md:5` — the named validators and kernel helpers are available from analysis-core | Pass. `S/analysis-core/index.ts:27`, `:44`, and `:7` for the factory. |
| 20 | `R/contributor-contract.md:101` — the mixed executor requires one packed chunk per value | **Fail.** Multi-chunk CPU values are handled upstream; the local GPU lowering enforces single-chunk ports. DOC-05 gives both code paths. |
| 21 | `D/operation-forms-and-cpu-twins.md:41` — mixed execution excludes textures | Pass. `node_modules/@luma.gl/graph/src/gpu-core/gpu-cpu-execution.ts:37`. |
| 22 | Same page, `:41` — mixed execution excludes state vectors | Pass. The same constructor assertion checks the vector role. |
| 23 | Same page, `:42` — CPU input validity bitmaps and list offsets are unsupported | Pass. `node_modules/@luma.gl/graph/src/gpu-core/gpu-cpu-execution.ts:132`. |
| 24 | `R/contributor-contract.md:98` — pure twins reject lengths that must agree, applied to the equal-length variogram bins in `D/operation-forms-and-cpu-twins.md:181` | **Fail.** The CPU fit truncates to the shortest length. `S/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:77`. |
| 25 | `D/operation-forms-and-cpu-twins.md:74` — the CPU variogram twin delegates to `fitVariogramModel` | Pass. `S/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:120`. |
| 26 | Same page, `:189` — the parameter tuple is nugget, partial sill, range, weighted residual | Pass. Same source, `:126`, `:137`. |
| 27 | `modules/analysis/README.md:45` — MGWR golden-section search has a 64-iteration maximum | Pass. `S/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts:37`, `mgwr-bandwidth-selection.ts:80` in the same directory. |
| 28 | `modules/analysis/README.md:58` — the full-model trace is in summary slot zero | Pass. `S/gpu-spatial-analysis/spatial-regression/multiscale-inference.ts:87`. |
| 29 | `A/coregionalization-fit.md:111` — two to four variables and one to four structures | Pass. `S/dataframe-analysis/pair-statistics/coregionalization-parameters.ts:30`, `:53`, `:60`. |
| 30 | Same page, `:111` — GPU fit admits zero through 256 lag matrices | Pass. Same parameters source, `:34`; `S/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts:83`. |
| 31 | Same page, `:111` — GPU sweep budget is zero through 1,024 | Pass. Same parameters source, `:36`, `:67`. |
| 32 | Same page, `:114` — the node estimate is `2 * maximumIterations + 3` | Pass. `S/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts:157`. |
| 33 | `A/terrain-dinf-hand.md:5` — exactly one of directions and fractions is required | Pass. `S/gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts:213`. |
| 34 | Same page, `:38` — fractional and D8 budgets default to 128 and 32 | Pass. Same source, `:200`. |
| 35 | Same page, `:30` — multiple first drainage cells use `0xfffffffd` | Pass. Same source, `:70`. |
| 36 | `D/gpu-raster/raster-embedding-codecs.md:87` — codec vectors stop at 128 dimensions | **Fail.** Streaming maximum is 1,024. `S/raster-analysis/raster-embeddings/raster-embedding-common.ts:16`, `gpu-raster-embedding-dequantize.ts:120` in the same directory. |
| 37 | Same page, `:88` — codec views are single packed buffers | **Fail.** Pixel fields support complete-pixel chunk boundaries. `S/raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts:97`; the guide itself gives the correct contract at `:9`. |
| 38 | Same page, `:22` — TESSERA treats signed byte −128 as a value | Pass by shader-source inspection. Only the AlphaEarth branch sets invalidity for −128. Same implementation, `:270`. |
| 39 | `R/upstream-first-recording.md:25` — `GPUComputationRecorder` is public from computation-record | Pass. `S/computation-record/index.ts:88`. |
| 40 | `R/upstream-workarounds.md:144` — the upgrade inventory programs accept the documented file arguments | Pass by source inspection. `R/research/luma-usage-review-2026-11-11/scripts/exports.mjs:5`, `usage.mjs:4`, `:48`, `:67`, `diff.mjs:2`. |

### Roadmap census and 15-row verification

The **214** count is the number of data rows in the five-column *Open backlog by domain* tables, including maintainer decisions. It counts rows, so an ID pair in one row counts once. H/M/L qualifiers are folded into their base priority; the explicit `H (decision)` row counts as H. The high-priority summary, historical pass gates, AX status table and known-limits register use different schemas. Adding them would double-count work. There is no reliable whole-file count of *unique, genuinely outstanding* tasks until the separate pass tables are reconciled. `R/roadmap.md:24`, `:96`, `:215`, `:258`, `:301`, `:504`, `:525`.

| Domain | H | M | L | Decision |
|---|---:|---:|---:|---:|
| Repository, examples, integration | 0 | 5 | 4 | 0 |
| Spatial statistics, regression, local statistics | 0 | 1 | 11 | 0 |
| Point processes, sampling, clustering | 0 | 0 | 2 | 0 |
| Interpolation and simulation | 0 | 0 | 3 | 0 |
| Network | 1 | 0 | 5 | 0 |
| Raster, terrain, hydrology | 0 | 3 | 13 | 0 |
| Dataframe, point cloud, projection, geometry | 0 | 2 | 14 | 0 |
| Kepler | 1 | 6 | 2 | 0 |
| Lenses, recording, CPU forms | 1 | 9 | 7 | 0 |
| Website pictures and interaction | 1 | 12 | 9 | 0 |
| Website explanation pages | 0 | 3 | 3 | 0 |
| Spatiotemporal and platform | 1 | 4 | 6 | 0 |
| Contributor contracts and GPU scale | 4 | 20 | 8 | 0 |
| Precision | 0 | 3 | 0 | 0 |
| New contributor options | 0 | 22 | 12 | 0 |
| Maintainer decisions | 1 | 0 | 0 | 15 |
| **Total** | **10** | **90** | **99** | **15** |

Exactly three first-column labels repeat verbatim across these tables: `KE-TESTS` at lines 201/517, `PLAT-03` at 406/510, and `STD-05` at 417/515. The paired rows often describe different decisions. For instance, `STD-05` means both runtime parameter adoption and accepting a walking-cost rounding difference. Reusing the ID makes commit and issue references ambiguous.

| Sampled roadmap row | Disposition and source evidence |
|---|---|
| `KNOWN(NW-CENT)` `:148` | **Rewrite.** The original contributor still has its budget. A public host-controlled continuation already windows longer runs. Keep the upstream request for generic device-side continuation. `S/gpu-network/network-centrality/index.ts:31`, `network-centrality-continuation.ts:78` in the same directory. |
| `KNOWN(NW02)` `:149` | **Narrow.** 128 nodes/1,024 edges describe the reference solver. Auto selects parallel above the bounds. Track path decomposition separately. `S/gpu-network/network-optimization/gpu-network-flow.ts:101`, `:196`, `:217`. |
| `KNOWN(TE03)` `:171` | **Close the stated task.** The row requests parallel activation. The producer describes a parallel ranked-edge forest and schedules the activation nodes. `S/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:120`, `:276`. |
| `KNOWN(TE06)` `:169` | **Narrow.** D8 has an explicit tiled boundary route; branching remains resident. Independently filled tiles and host out-of-core transport are separate acceptance conditions. `S/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts:111`, `:301`. |
| `KNOWN(GPU-10)` `:431` | **Remove the point-routing prerequisite.** Scalable routing is the default, with same-edge and predecessor semantics documented in source. Keep the independent facility/time-routing work. `S/gpu-network/network-optimization/gpu-network-point-routing.ts:72`, `:80`, `:112`. |
| `KNOWN(GPU-13)` `:434` | **Remove Borůvka rounds.** The region-graph producer calls `createRasterRegionBoruvkaNodes`. Other clauses require their own gates. `S/raster-analysis/remote-sensing/gpu-raster-region-graph.ts:421`. |
| `KNOWN(DUP-03)` `:436` | **Remove S2 compaction from the remaining work.** The compaction contributor has an S2 layout and compaction nodes. Keep the shared type and separately verified parity gaps. `S/gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts:26`, `:44`. |
| `KNOWN(STD-14)` `:426` | **Re-census.** The row still asks whether joins need chunking. Point-in-polygon already documents chunked positions and cross-chunk output identity. Reframe the row around contributors and scratch buffers that still require packing. `S/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts:171`, `:225`. |
| `KNOWN(EMB-02)` `:477` | **Close for streaming contributors.** The spec enumerates every width requested by the row. File a separate covariance expansion only if a consumer needs it. `modules/analysis/test/gpu-raster/raster-embeddings/embedding-scale.node.spec.ts:178`, `S/raster-analysis/raster-embeddings/raster-embedding-common.ts:19`. |
| `KNOWN(LUM-30)` `:447` | **Rewrite the premise.** “No spec checks for leaks” is false. Analysis spacing snapshots active buffers and checks the count after teardown. Expand fixture and resource-type coverage. `modules/analysis/test/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.spec.ts:155`, `:220`. |
| `KNOWN(KE-TESTS)` `:201` | **Regenerate the counts.** The stated 131 run exclusions and 59 additional typecheck exclusions are now 1 and 58 in the named manifest. The manifest does not establish the status of the separate DuckDB cold-cache concern. `test/kepler-compute-example-specs.json:3`, `:6`. |
| `KNOWN(IN04)` `:140` | **Keep the limit, fix the API name.** The four-variable/four-structure bounds exist. The CPU class is `CPUCoregionalizationFitOperation`, and the pure export is `computeCoregionalizationFitOnCPU`. `S/dataframe-analysis/pair-statistics/coregionalization-parameters.ts:30`, `S/dataframe-analysis/pair-statistics/index.ts:191`. |
| `TE05` `:173` | **Keep.** The current DEM coregistration class describes a vertical-shift model. Horizontal and tilt registration remain meaningful requests. `S/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts:103`. |
| `PLAT-03` decision `:510` | **Keep as a decision with its own ID.** HDBSCAN defines and validates a maximum of 32 samples. `S/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts:95`, `:296`. |
| `ST06` `:119` | **Keep.** MGWR exposes conditional inference with standard errors, effective parameters, summary and status. Bandwidth-selection confidence intervals remain a distinct request. `S/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts:89`. |

The primary census can be reproduced without building anything:

```python
from collections import Counter, defaultdict
from pathlib import Path

rows = defaultdict(Counter)
for number, line in enumerate(Path("dev-docs/roadmap.md").read_text().splitlines(), 1):
    if line.startswith("### "):
        domain = line[4:]
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    if 96 < number < 525 and len(cells) == 5 and cells[0] not in {"ID", "---"}:
        rows[domain][cells[3].split()[0]] += 1
```

Other census inputs were `rg --files docs/api -g '*.md'`, the `api/…` strings in `docs/table-of-contents.json`, `ANALYSIS_DOC_PAGES` in `modules/analysis/test/analysis-core/doc-program.ts`, and opening fences matching ``^```(ts|typescript)\s*$``. Research consumers were located with `rg -l -F "/<research-directory>/" scripts modules/analysis modules/lenses docs/api examples website`, followed by inspection of reads, writes and citations. A separate local-link census resolved **724** inline file/route references under `docs/api` with **zero missing local targets**, using `/docs/...`, `.md` and directory-README resolution. It did not check fragments, external URLs or the deployed site.

## Cross-cutting patterns

- **Give each fact an owner.** Package manifests own compatibility, public barrels and the catalog own import names, constructors own limits, and family guides own the explanation of what happens at a limit. A summary should link to that explanation or include generated fields. The embedding guide demonstrates how inserting a correct paragraph can leave a contradictory paragraph intact. `D/gpu-raster/raster-embedding-codecs.md:9`, `:87`.
- **Separate the execution routes in every limit statement.** Exact versus parallel SCC, streaming versus reference embedding widths, host-controlled versus unrolled centrality, and GPU lowering versus CPU packing all change what a caller can do. A useful limit names the route, default selection, input layout and exhaustion behavior. `S/gpu-network/network-structure/gpu-network-structure.ts:85`, `D/gpu-network/operators.md:114`, `:199`, `S/analysis-core/operations/cpu-operation-rows.ts:27`.
- **Separate research according to what consumes it.** Keep the showcase oracle JSON, upgrade scripts, and protected AX baseline paths as operational dependencies. Preserve the compiler investigations, spatiotemporal experiments, precision measurements and benchmark evidence where code, specs or API pages cite them. The capability matrix has a generator, so moving it requires updating its destination. Library/split/fork reviews, CPU surveys, candidate studies and picture scores primarily provide dated decision evidence. Keep their conclusions and referenced evidence while a decision is open; remove superseded coordination prose once closed. `examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs:109`, `examples/deck/kepler-compute/scripts/capability-matrix.mjs:8`, `R/upstream-workarounds.md:144`, `D/gpu-spatial-analysis/space-time-hawkes.md:131`, `R/research/README.md:16`, `:42`.

## Keep

- **The variogram walkthrough’s structure.** It supplies real inputs, gives an expected answer, explains why an existing device is needed for the graph form, checks result status and shows teardown. Its dedicated spec combines the fences and allows no implicit imports or `any` inputs. Use this standard for pages labelled “getting started.” `D/operation-forms-and-cpu-twins.md:55`, `:139`, `:159`, `modules/analysis/test/analysis-core/operation-forms-docs.node.spec.ts:36`.
- **Method-specific interpretation and provenance.** The HAND guide distinguishes missing drainage, unresolved routing and multiple valid destinations. The MGWR notes explain which trace is reported and what uncertainty the computation covers. Preserve that material during moves. `A/terrain-dinf-hand.md:28`, `modules/analysis/README.md:54`.
- **Existing generated parameter sections and the coverage matrix’s definition of “implemented.”** The API already marks TSDoc-derived blocks; the matrix explicitly limits coverage claims to tested subsets, and its inventory totals reconcile. Both are better foundations than writing another reference system. `D/gpu-spatial-analysis/analysis-experience.md:23`, `R/coverage-matrix.md:7`, `:260`.

## Future

1. **Build one user documentation path.** Root README → first computation → concepts for ownership, execution, coordinates, masks, capacity/status and readback → per-family guides → generated per-symbol reference. Put hand-written algorithm interpretation, precision qualifications and oracle evidence beside generated props. Start with embeddings and regression. The trade-off is maintaining redirects and catalog paths while moving existing material.
2. **Create a contributor guide built around a real change.** Keep AGENTS short, keep the contract normative, and link both to one worked implementation and its source, audit and specs. A live example can evolve with the library and avoids a second implementation template drifting independently. It does require an owner when the exemplar changes.
3. **Keep one roadmap index and split the editable backlog by domain.** Generate the priority and decision views from the same records. Move caller-visible completion notes to `CHANGELOG.md`, delete finished acceptance rows, and keep one dated line per completed pass. This adds a small generator but removes the need to hand-reconcile BL-PASS, SC, IMP, KE-PORT, KE-WF and AX against the domain tables.
4. **File three narrowly scoped upstream proposals first.** The register reports that nothing has been filed; external filing status is **UNVERIFIED** here. First, file the `GPUHistogram` edge/domain issue: the committed spec has explicit literal/view cases and expected assertions to change after a fix. Second, file caller-owned CSR with node/slot masks and a self-loop convention: there is a concrete local consumer and a clear retirement target. Third, propose the kernel-node factory using one representative binding/dispatch example and the fork implementation. Keep generic continuation on the queue with its updated rationale now that local host-controlled centrality exists. Histogram has the shortest path to a decision; CSR and the factory offer the larger long-term reduction in local code. `R/upstream-proposals.md:3`, `:13`, `:77`, `:126`, `:299`, `modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts:169`, `S/gpu-network/network-statistics/gpu-network-statistics.ts:197`, `S/analysis-core/kernels/wgsl-kernel-nodes.ts:368`. `KNOWN(UPS-01)` deserves earlier filing; `KNOWN(PLAT-04)` needs the revised local baseline.

## Top 5

1. Enforce equal-length variogram bins in the pure CPU function and pin CPU/GPU input-contract parity.
2. Put the existing CPU computation on both README landing paths and finish its browser/device entry point.
3. Reconcile the 15 sampled roadmap rows, then generate priority summaries from the open records.
4. Correct the public embedding, SCC, fitting-kernel and mixed-execution contracts.
5. Unify page discovery and snippet coverage, then move the ten package-local guides into the user documentation tree.

## Questions for the lead

- Can the CPU-twins stream own the unequal-length variogram fix? The documentation change depends on which behavior the library commits to support.
- Which URL should be stable for readers of an installed package README? That determines the link and redirect policy before the ten guides move.
- Who can own the upstream histogram and CSR submissions and update the retirement register when upstream responds?