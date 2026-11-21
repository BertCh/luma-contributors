# Roadmap and developer documents (2026-11-19)

The working roadmap contains **291 rows and 15,681 words**. The committed version the slice reviewers started from had 284 rows and 13,425 words. Four obsolete rows have already been removed. The ledger below covers both versions and preserves those deletions.
The most pressing correction is the release boundary. The alpha.16 gates passed as separate steps on November 18. The November 19 known-issues pass has its own staging copy, incomplete re-reviews and some failed verifications. Its completion labels cannot qualify a release from this checkout.
Several remaining tasks would send an implementor to build an existing feature. Three contributors already expose the status path awaiting a decision, the packed-view validator is public, and the Rose API guide already documents both angle origins.
CPU probes confirmed two consequential documentation errors. Capacitated p-median enforces **both** a 64-demand limit and an exponential candidate limit. A recording helper drops an upstream duration-only measurement and produces the same node metrics as a report with no GPU measurement.
The first computation, canonical API guides, route-qualified SCC documentation and mixed-execution contract are substantially improved. The largest remaining maintenance cost is keeping the same operational or numerical fact synchronized across prose, code, tests and research.
Before publication, reconcile the candidate revision, caller migrations and release gate. Afterward, keep one short core roadmap, give consumer backlogs their own files, and move persistent limits and dated measurements to their respective owners.

## Method

I served as area lead (`gpt-6.2-astra`). I read all six `gpt-6.1-sol` slice reports in full, including their limits. The slices covered LR and domain rows; platform, contracts and decisions; consumer planning and changelog; known limits, new options and inventories; document ownership and links; and agent rules and research retention. Together they accounted for 275 of the original physical rows. I reconciled the nine original index rows, the complete working-copy delta, the new KI table, and the seams between the reports.

The committed reference is `bd8279a49a924bc55c29c807bb46929de69adc56`. The working `dev-docs/roadmap.md` reviewed here had Git blob hash **`7ee5b7435319495fb83be22c791ebf7b3a60494d`**. Roadmap line citations below use that working version unless marked *committed*. I read the scoped Git history and diff from `bdc715ea` across 22 maintained documents, 2,835 insertions and 192 deletions. I also compared the roadmap with uncommitted source changes. The distinction matters: in two places the working roadmap describes a defect in the committed code after another session has changed the implementation.

Across the six slices, the two indexes, AGENTS, changelog, coverage matrix, regression inventory, recording guide, four upstream drafts, assigned roadmap rows and earlier documentation findings were read in full. Public exports, constructor branches, selected WGSL, consumer/parser implementations, tests, installed alpha.16 sources and research evidence were read where they support a specific claim. The 33 proposed contributors were searched by name and compared with the closest existing operations. Consumer qualification was limited to named-file existence, manifest membership and identified documentary dependencies.

I independently ran the roadmap census, a Markdown parser, the local-link inventory, and stdin-esbuild CPU probes for variogram fitting, facility admission, flow decomposition, path extraction, record ownership, preflight metadata and timing conversion. The reviewers also ran the two permitted static checkers, lightweight inventories and additional CPU probes for triangulation, continuation, alignment, tile layout and quantile admission. **None of the six slices submitted a new W or S finding.** I retained the timing loss as a contract finding: the measurement disappears, but the helper does not publish a false duration. No tests, gates, builds, browser, GPU work, external requests or writes were performed for this report.

## Census

| Measure | Result | Reproduction |
|---|---:|---|
| Working roadmap | 550 lines, 15,681 words, 117,149 bytes | `wc -lwc dev-docs/roadmap.md` |
| Committed roadmap | 529 lines, 13,425 words, 101,107 bytes | `git show bd8279a4:dev-docs/roadmap.md \| wc -lwc` |
| Physical data rows | 291 working, 284 committed | Count lines starting `\| `, excluding headers beginning `ID` or `Area` and delimiter rows |
| Rendered data rows | 289 working | Installed `markdown-it` yields 312 `tr_open` and 23 `table_open` tokens. KE-WF-K/L are outside the table after a blank line |
| Priority | 10 H, 95 M, 97 L, 15 Decision, 74 without a priority cell | Read cell four of five-column rows and fold parenthetical qualifiers. One H is also a decision |
| Primary planning kind | 145 core/repository, 21 Kepler, 22 examples, 30 website, 9 lenses, 14 upstream, 16 decision, 24 limit, 10 index | One primary owner per physical row, as encoded in the ledger. Cross-links and secondary owners do not add rows |
| Maintained portfolio | 22 documents | `dev-docs/*.md` (13), `dev-docs/upstream-drafts/*.md` (5), research index, root README, AGENTS and changelog |
| Research | 195 tracked files, 7,857,438 bytes, 19 immediate directories and two root files | `git ls-files -z dev-docs/research`; sum file sizes. The live review had 9 additional files, 383,732 bytes, when rechecked |
| Coverage matrix | 144 reference-operation rows: 137 implemented subsets, 1 partial, 6 excluded | Count the status cell in `dev-docs/coverage-matrix.md:34–246`. The export walk found all 142 distinct local directory/symbol pairs, 174 mentions |
| Regression checker | Historical 21 sites in 15 files; current 29 in 17, zero new | Compare `scripts/contributor-contract-baseline.json` at `bdc715ea` and HEAD; scoped `single-invocation-data-loop` checker over `spatial-regression` |
| Exclusions | 1 runtime, 58 test-typecheck, zero intersection; all 59 paths exist | `Object.keys` on the two maps in `test/kepler-compute-example-specs.json:2`. Consumers are `vitest.config.ts:17` and `scripts/typecheck-tests.mjs:129` |

Physical rows by section are independently reproducible with this read-only command:

```sh
node -e '
const lines = require("fs").readFileSync("dev-docs/roadmap.md","utf8").split("\n");
const sections = {}, priorities = {};
let section;
for (const line of lines) {
  if (/^#{2,4} /.test(line)) section = line.replace(/^#+ /,"");
  if (!/^\| /.test(line) || /^\| (ID |Area |---)/.test(line)) continue;
  const cells = line.slice(1,-1).split("|").map(cell => cell.trim());
  const priority = cells.length === 5 ? cells[3].split(" ")[0] : "unrated";
  sections[section] = (sections[section] || 0) + 1;
  priorities[priority] = (priorities[priority] || 0) + 1;
}
console.log(sections, priorities)'
```

| Section | Rows | Section | Rows |
|---|---:|---|---:|
| Summary | 10 | LR | 12 |
| KI | 10 | Repository and integration | 9 |
| Statistics | 11 | Point processes | 2 |
| Interpolation | 3 | Network | 4 |
| Raster and terrain | 14 | Dataframe, points and geometry | 13 |
| Kepler domain | 9 | KE-PORT | 9 |
| KE-WF | 12 | AX | 9 |
| Lenses, recording and CPU forms | 17 | Website pictures | 22 |
| Website explanation | 6 | Spatiotemporal and platform | 11 |
| Contracts and GPU scale | 32 | Precision | 3 |
| New options | 33 | Decisions | 16 |
| Known limits | 24 | **Total** | **291** |

### Factual checks on documents without another primary verifier

The samples were targeted. The rates measure agreement for the listed claims, not the fraction of all prose that is correct. Paths and individual misses are given so the samples can be extended.

| Document | Claims checked | Fully supported | What was checked and every exception |
|---|---:|---:|---|
| Root `README.md` | 17 | 14 | Target, CPU fixture, GPU walkthrough scope, package roles, Arrow requirement, semver range, workspace membership, TOC consumer, Node/Yarn, source typecheck, build outputs, headless configuration, test stages, exclusions/licensing. **Two misses:** “no copies” at `:26`, and blanket “kepler-compute is parked” at `:45` despite workspace registration, a start script and open qualification gates. **One unresolved:** public-registry status. See `package.json:11`, `examples/deck/kepler-compute/package.json:7`, `dev-docs/roadmap.md:191`. The seventeenth check was added by the lead. |
| `dev-docs/README.md` | 15 | 15 | Purpose of roadmap, contract, walkthrough, test guide, script guide, regression inventory, matrix, release guide, upstream plan, recording guide; website table parser; filing-draft distinction; executable research; maintained upgrade tools; retention classes. Evidence includes `dev-docs/README.md:7–26,36–42`, `website/src/components/meta/pages/upstream-first/generate-data.ts:57`, `scripts/upstream/README.md:1`. External filing was not checked. |
| `dev-docs/upstream-first-recording.md` | 18 | 15 | Public graph facts, order/allocation, preflight, encode stats, returned shape, hazard granularity, recorder edges, opt-in storage, unobserved factory behavior, text omission, override order, public export, scopes, repeated IDs, taps/pass inference. **Three misses:** endpoints on duration-only backends at `:12`, ownership fallback at `:17`, and already-available optional upstream kernel metadata at `:57`. |
| `AGENTS.md` | 25 | 21 | Commands, version selection, heaps, project grouping/configs, generation-only mode, sequential verification, partial runs, disk, RSS, packed isolation, workers, report membership, opt-ins, benchmarks, incremental builds, records, staging, rollback and peer range. **One ambiguous/inaccurate instruction:** “source roots separately” at `:25` can imply family source compilers; `package.json:18` starts one combined source project. **Three partial:** assets resolve rather than load at `:10`; ten website engine imports lie outside the stated examples/tests rule at `:123`; a pinned WGSL copy survives the rule at `:128`. |
| `dev-docs/coverage-matrix.md` | 20 | 19 | Four status/denominator checks, root export, import entries, local symbol reachability, upstream point-cloud exports, natural-breaks classes and both exact routes, three Dorling bounds, resistance export, SCC limits/selection/exhaustion, dated IN01 timings. **Miss:** `:30` describes published rows against alpha.10 without a clear historical qualification. Installed target is alpha.16 (`package.json:47`). |
| `dev-docs/regression-loop-inventory.md` | 23 | 22 | Historical baseline; GTWR, GWR, MGWR, MGTWR and restricted-regime bounds; panel outputs/reference bounds; search rows, inference budget, diagnostic axes; wrapper depth, guards and allowances; IV axes; 4,096-tile limits; tiled panel layout; fold loops; real-source checker coverage. **Miss:** `:15–17` assigns removal of restricted-regime baseline entries already absent. The 21/15 count remains valid for the old baseline. |
| `dev-docs/research/README.md` | 25 | 24 | Recovery revision, split/fork/workflow consumers, AX note counts, review counts and implementation record, scale index, spatiotemporal scripts, math evidence, CPU survey, usage report/probes, precision, candidate research, pictures, Kepler matrix, showcase readers and size policy. **Miss:** `:26` describes upgrade execution from historical research scripts. The maintained procedure calls `scripts/upstream/` at `dev-docs/upstream-workarounds.md:294`. The parity report is also absent from the index. |

## Follow-through on the 2026-11-17 findings

The original ten findings are in `dev-docs/research/library-review-2026-11-17/area-docs-devdocs.md:16–84`. Its `verification.md` has no independent `DOC-*` entries. A pin below means the assertion discriminates the named change by inspection. Parent-revision failure logs were not available for all fixes.

| ID | 2026-11-17 claim | Verdict | Fix, pin and what it would catch |
|---|---|---|---|
| DOC-01 | First CPU fit silently truncated unmatched bins | **FIXED** | `modules/analysis/src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:29`; `modules/analysis/test/gpu-dataframe/pair-statistics/cpu-variogram-model-fit-operation.node.spec.ts:395` checks shorter/longer arrays. Lead probe with the guide’s 20 bins produced `[0.2000000775,1.5,2.2000002861,2.76e-13]`, status 0. With 19 counts it threw `variogram-model-fit invalid model or bin topology`. Restoring shortest-prefix fitting changes this outcome. |
| DOC-02 | Neither landing README led to a first result | **FIXED** | `README.md:6`, `modules/analysis/README.md`, `modules/analysis/test/analysis-core/first-run-docs.node.spec.ts:40`, `first-run-review.node.spec.ts:71`. Removing, reordering or breaking an entry link is detectable. CPU output was reproduced. Browser execution and the eventual public host were not qualified here. |
| DOC-03 | Roadmap assigns work already shipped | **PARTLY** | Prior sampled closures for NW-CENT, TE03 and EMB-02 held. Host continuation, parallel terrain activation and widths 192/256/384/1,024 have code/spec evidence. GPU-10, GPU-13, DUP-03, STD-14, LUM-30, KE-TESTS, IN04 and the old PLAT-03 decision were corrected. TE06, TE05 and ST06 retain real limits. NW02 closed in the working copy. UF-10, INT-06, several status decisions and new review claims still need reconciliation. `modules/analysis/test/analysis-core/roadmap-reconciliation.node.spec.ts:20,37,98,217` covers selected rows, not all records. |
| DOC-04 | Reference understated usable capabilities | **FIXED for the three cited cases** | `dev-docs/coverage-matrix.md:255` distinguishes exact and parallel SCC; automatic route selection is in `modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts:134`. `docs/api/analysis-catalog.md:22` names the fitting producers. Streaming/reference codec widths are distinct at `modules/analysis/src/raster-analysis/raster-embeddings/raster-embedding-common.ts:15`. `documentation-facts.node.spec.ts:150–180` rejects obsolete wording; `embedding-scale.node.spec.ts:222` would reject a restored universal 128 cap. |
| DOC-05 | Contract attributed local one-chunk lowering to upstream | **FIXED** | `dev-docs/contributor-contract.md:176–178`, `modules/analysis/src/analysis-core/operations/gpu-contributor-operation.ts:218`, `cpu-operation-rows.ts:27`. `cpu-callback-copies.node.spec.ts:12` distinguishes callback input copies from local packing. Retain separate entries for executor, GPU lowering and CPU-row adaptation. |
| DOC-06 | Sidebar, snippets and API pages used incompatible inventories | **FIXED for discovery** | `modules/analysis/test/analysis-core/doc-program.ts:17,35` discovers pages and checks exactly one sidebar entry. `doc-snippets.node.spec.ts:85,316` covers fence spelling and discovers snippets from that list. `doc-publication.node.spec.ts:65` rejects a missing sidebar page. Two intra-page fragments still fail; these are a different check. |
| DOC-07 | Package-local guides unreachable through the publication path | **FIXED locally** | Ten legacy files remain as short redirect pointers. `modules/analysis/test/analysis-core/doc-publication.node.spec.ts:72,95,115` rejects duplicated long guides, absent substantive targets, stale catalog paths and relative README links to unpackaged fixtures. External availability of the eventual repository URL remains LR-04. |
| DOC-08 | Setup and release statements described an older tree | **PARTLY** | Target, peer policy, test heap, exclusions, changelog roadmap references and wildcard handling improved. `release-package.node.spec.ts:43,61`, `documentation-facts.node.spec.ts:108,126`, `gate-scripts.node.spec.ts:153,180` distinguish old wording and missing wildcard/string-target checks. Residues include the matrix pin, the root repository map and AGENTS’ asset-loading description. |
| DOC-09 | No worked authoring route | **PARTLY** | `dev-docs/authoring-walkthrough.md:3` connects real props, kernels, exports, catalog, audit, oracle, docs and lens. `authoring-walkthrough.node.spec.ts:20,95` inspects numbered steps and a concrete example. `TC-01` was replaced at `dev-docs/contributor-contract.md:85`. `rg -n authoring-walkthrough AGENTS.md dev-docs/contributor-contract.md` finds neither requested entry link. |
| DOC-10 | Retention conflated executable inputs and citations | **PARTLY** | The policy at `dev-docs/README.md:26,36–42` distinguishes five relationships; scalability is indexed. `upstream-inventory-tools.node.spec.ts:31,57,95` exercises maintained tools. It would catch broken import/export discovery, but cannot catch the obsolete index direction at `dev-docs/research/README.md:26`. |

## Findings

**Final grading:** 11 C, 2 T and 2 M. Fourteen are ORDINARY and one is UNUSUAL. There are zero confirmed W or S findings in this documentary area. Related defects and qualifications in family reports are cited under their own IDs and are not counted again.

| ID | Kind | Reach | Evidence | Where | What goes wrong | Proposed change |
|---|---|---|---|---|---|---|
| ROAD-101 | C | ORDINARY | SOURCE, COUNT | `dev-docs/roadmap.md:41,64,80–91`; `dev-docs/research/library-review-2026-11-19/ki-pass-digest.md:14,30–34,293` | An implementor can read a staged “done” as a verified fix. The digest records failed v2 runs in two “done” multipart streams and an interrupted second review of cost-distance. LR-22’s “has a cause” is stronger than the digest’s sufficient mechanism for the historical failures. | Tie release decisions to a revision, a passing artifact and the disposition of each outstanding check. Keep staging status and family findings in a dated evidence record. `KNOWN(LR-22, KI-03, KI-04)` |
| ROAD-102 | C | ORDINARY | SOURCE | `dev-docs/roadmap.md:60,121,504`; `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/segregation-decomposition.ts:88,142`; `docs/api/gpu-spatial-analysis/operators-statistics.md:1001` | Even the new roadmap mixes revisions. It still says randomization is validated after graph mutation, both Rose origins are undocumented, and inhomogeneous K uses a fixed scale that the working source has changed. | Tag committed evidence and pending changes separately. On landing, remove these three obsolete premises after their respective pins are verified. `KNOWN(LR-10, ROSE-01)` |
| ROAD-103 | C | ORDINARY | EXECUTED-CPU, SOURCE | `dev-docs/roadmap.md:145,510–511,514`; `modules/analysis/src/gpu-network/network-optimization/gpu-network-facility-optimization.ts:134`; `modules/analysis/src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts:401` | Effective limits lose an independent constraint, route or unit. NW05 still implies the 64-demand ceiling belongs only to the other operations. CPU walk budgets and output tails differ by output. Word-mode temporal keys consume eight bytes per slot. | State subject, formula, route, device constraint and observable response. See the ledger for replacement text. |
| ROAD-104 | C | ORDINARY | SOURCE | `dev-docs/roadmap.md:388–389,393,397,482`; `modules/analysis/src/analysis-core/index.ts:46` | The roadmap asks whether to export a public validator and add status paths that already exist. It labels runtime point-cloud settings as baked and asks for an already-corrected Dorling catalog limit. | Delete shipped clauses. Leave compatibility decisions for the legacy views and enumerate actual remaining props. `KNOWN(STD-04/05/09/13)` |
| ROAD-105 | C | ORDINARY | COUNT, SOURCE | `dev-docs/roadmap.md:99–107,194,319–320`; `test/kepler-compute-example-specs.json:2`; `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-permutation-test.lens.ts:29` | Old work remains assignable: missing guides, old example-name prefix, alpha.10 exclusion reasons, an unpublished import, and pre-alpha.16 recapture. There is still one genuinely missing showcase helper. | Delete the obsolete records and keep the actual missing file, current manifest, caption checks and canonical-URL decision. |
| ROAD-106 | C | ORDINARY | EXECUTED-CPU, SOURCE | `dev-docs/upstream-first-recording.md:17,57`; `dev-docs/upstream-workarounds.md:34`; `modules/analysis/src/computation-record/describe-command-graph.ts:129,259` | Unknown ownership is documented as imported. Alpha.16 already exposes opt-in kernel metadata for declaring nodes, including scan, while the local describer ignores that field. | Document `unknown` and lower-bound bytes. Distinguish local adoption, upstream primitive coverage and domain annotations. `KNOWN(LN-REC)` |
| ROAD-107 | C | ORDINARY | SOURCE, COUNT | `README.md:26,45`; `AGENTS.md:10,25,123,128`; `dev-docs/upstream-first-plan.md:15`; `scripts/verify-package-exports.mjs:149` | Agent/setup policy overstates export loading and copy removal, leaves source typecheck grouping ambiguous and excludes site rendering in prose while the checker omits the site. The root map calls an active example workspace parked. | Correct scope at each entry point. Resolve the site-engine and existing non-grid-copy policies explicitly. Keep publication distinct from public metadata. |
| ROAD-108 | C | ORDINARY | COUNT, SOURCE | `dev-docs/coverage-matrix.md:30`; `dev-docs/regression-loop-inventory.md:15`; `dev-docs/research/README.md:26` | Three inventories direct work against the wrong version, an already-clean baseline, or historical scripts. | Link compatibility to the release guide, date the old baseline and point upgrades to `scripts/upstream/`. |
| ROAD-109 | C | ORDINARY | COUNT | `dev-docs/README.md:9`; `dev-docs/contributor-contract.md:9`; `AGENTS.md:160`; `docs/api/gpu-spatial-analysis/operators.md:19`; `dev-docs/roadmap.md:24` | The worked guide lacks its two author entry links. Two KMeans fragments target a removed heading. The in-progress review index is already a roadmap dependency and had no target at capture. | Link both author entry points, repair the two fragments and install the review index with its evidence. Preserve historical links by revision. |
| ROAD-110 | C | ORDINARY | COUNT, SOURCE | `dev-docs/roadmap.md:249–261,339,482–493`; `modules/analysis/test/analysis-core/roadmap-reconciliation.node.spec.ts:20` | Eight component IDs denote both implementation and decision records. The final two workflow rows are physical text outside the rendered table. First-match readers do not disambiguate record roles. | Give decisions distinct IDs with old-ID cross-references, repair the KE-WF table and resolve records by section and ID during migration. |
| ROAD-111 | C | UNUSUAL | EXECUTED-CPU, SOURCE | `dev-docs/upstream-first-recording.md:12,51`; `docs/api/computation-record.md:215`; `modules/analysis/src/computation-record/computation-metrics.ts:49–70` | A supported duration-only timing becomes indistinguishable from unavailable GPU timing. With endpoints the helper uses milliseconds, although the API guide promises nanoseconds. | Specify endpoint and duration support separately. Preserve duration in a named field if supported by the record contract. Keep endpoint absence representable. |
| ROAD-112 | T | ORDINARY | SOURCE | `modules/analysis/test/analysis-core/roadmap-reconciliation.node.spec.ts:62,135,182,217`; `root-files-roadmap-handoff.node.spec.ts:51` | Several reconciliation tests pin phrases or source statements. The priority check accepts a substring for any compound ID and excludes LR/KI from its domain scan. These checks can pass with a stale roadmap or without testing the numerical boundary. | Resolve exact record identity, preserve meaningful behavioral assertions, and label editorial checks as such. |
| ROAD-113 | T | ORDINARY | SOURCE | `modules/analysis/test/gpu-dataframe/column-classification/gpu-column-quantiles-rational.node.spec.ts:131`; `dev-docs/roadmap.md:51–53,78` | Parent-failure evidence is not established for every LR change. Some claimed boundaries have only one side pinned. Optional-output comparisons can catch a no-writer regression while leaving shared numerical errors undetected. | Record the fixture, wrong implementation, parent result and candidate result for each changed contract. Keep oracle provenance per output. `KNOWN(KI-01)` |
| ROAD-114 | M | ORDINARY | COUNT, SOURCE | `dev-docs/roadmap.md:200–306,429–473,529–550`; `CHANGELOG.md:7–116` | Consumer stages, parked options, pass histories and caller migrations compete with release work. There are 31 cells of 50 or more words in the working roadmap. | Move work by owner, keep a short index, and put history and migrations in their respective documents. |
| ROAD-115 | M | ORDINARY | COUNT, SOURCE | `AGENTS.md:20–119`; `website/src/components/meta/pages/upstream-first/generate-data.ts:57`; `modules/analysis/test/analysis-core/upstream-drafts.node.spec.ts:128` | One hundred lines of gate/build mechanics precede the architecture rules. Other documents double as parser inputs through table-cell order, heading spelling or fence position. | Shorten AGENTS after moving operational details. Change the identified document readers in the same patch as any restructuring. |

**Executed examples.** A four-node diamond with edges `0→1→3` and `0→2→3`, flows `[3,3,2,2]` and path capacity one produced edge rows `[0,1,4294967295]`, volume `[3]`, `count=1`, `requiredCount=2`, `overflow=1`, `converged=1`. This is the expected bounded decomposition. Its public writer is scheduled at `modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts:616`.

For capacitated p-median, the real constructor accepted `(facilities,demand)=(1,64),(2,20),(4,10),(16,5)` with candidate counts `1,1048576,1048576,1048576`. It rejected `(1,65)` on demand rows and `(2,21),(4,11),(16,6)` on candidate count. The graph used the repository’s CPU-only null device and transient views; no commands were encoded. For path extraction, predecessors `[0xffffffff,0,1]`, costs `[0,1,2]`, target `[2]` and output capacity five produced `count=0,candidateOverflow=1` at walk budget one and `count=3,ids=[0,1,2,0,0],candidateOverflow=0` at budget three. `modules/analysis/src/gpu-network/network-analysis/cpu-network-path-extraction-operation.ts:69–84` describes this distinction.

For ROAD-106 a synthetic preflight node with an unobserved 16-byte buffer and optional kernel facts produced `ownership:"unknown"`, `bytesKind:"lower-bound"`, no source records and no workgroup facts. Installed `node_modules/@luma.gl/graph/src/gpu-core/gpu-scan.ts:632` supplies kernel metadata; collection must be requested. For ROAD-111 an input node with `cpuEncodeTimeMilliseconds:1,gpuTimeMilliseconds:2` produced `timebase:"ns",nodes.n:{cpuEncodeMs:1}`. Removing the GPU measurement produced exactly the same output. Adding start 10 and end 12 milliseconds produced `timebase:"ms",nodes.n:{cpuEncodeMs:1,gpuBegin:0,gpuEnd:2}`. The installed non-WebGPU branch returns the duration-only shape at `node_modules/@luma.gl/graph/src/gpu-core/gpu-command-graph.ts:609`.

## Refuted or downgraded

- NW02 is already gone from the working roadmap. Keep the bounds in LR-25. The CPU diamond and scheduled public writer refute the claim that path decomposition is unimplemented.
- “Only the other operations enforce 64 demand rows” is false. A one-facility capacitated p-median demonstrates the independent guard.
- ROSE-01’s present documentation premise is false. `docs/api/gpu-spatial-analysis/operators-statistics.md:1001` already names both origins and both units. A third convention needs its own consumer.
- The committed segregation example rejected invalid randomization late. In the working file the check is at line 88 and the first `graph.add` is at 142. This does not establish transactional rollback for every recipe.
- The historical fixed-scale argument for inhomogeneous K needs a revision label. The working implementation introduces per-frame intensity scaling at `modules/analysis/src/dataframe-analysis/pair-statistics/space-time-k-function-wgsl.ts:6` and `space-time-k-function-nodes.ts:332`. Current GPU numerical qualification is **UNVERIFIED**.
- The source supports a mechanism for schedule-dependent cost-distance counts. Neither source nor matching emulator counts identify the actual historical Metal scheduling order. `ki-pass-digest.md:293` makes this distinction.
- Shipped Strahler order, compact geometry, A5 decoding, candidate-capacity forwarding and option sequencing already have substantial working-roadmap corrections. Preserve them. A bounded breach from explicit pits still does not establish automatic whole-DEM pit discovery.
- Upstream CPU callbacks support chunks. The local GPU contributor lowering imposes a separate single-chunk restriction. Optional public shader metadata also leaves legitimate local formula, provenance and binding needs.
- Consumer files and 427 recorded captures establish existence or recorded completion. They do not qualify current screenshots, captions, numerical values, latency or device tiers.
- The tracked research corpus obeys its individual-file size rule. A missing literal path is insufficient to justify retirement where a reader constructs paths or protects a baseline by realpath.

## Tests

Five weak **evidence chains** in this area give better regression priorities than ranking contributors by test-file count.

| Contributor or surface | Existing protection | Discriminating check still needed |
|---|---|---|
| `GPUComputationRecorder` / metrics | `modules/analysis/test/computation-record/computation-recorder.node.spec.ts:107` checks unknown ownership; `computation-record.node.spec.ts:384,440` distinguishes ns/ms endpoint inputs | Duration-only versus absent timing, and public preflight kernel metadata versus an explicit local annotation with overlapping fields |
| `GPUColumnQuantiles` | Rational-mode spec rejects `2^31`; roadmap spec constructs class breaks on both sides of `2^24` | Accept `2^31−1` on the rational unfiltered constructor, reject the next row, and retain the float-filter limit. Constructor admission alone says nothing about feasible device allocation |
| `GPUSpatialLagMaximumLikelihood` and its nested traces | `root-files-roadmap-handoff.node.spec.ts:51` checks spelling of three limits. The nested trace helper coarsens width then rejects above 4,096 | Construct equal-row layouts with different nonempty chunk counts. Distinguish main likelihood tiles from nested traces and show where increasing tile width stops helping |
| `GPUTemporalReduction` | Source enforces the u32 slot bound and separately preflights bindings at `gpu-temporal-reduction.ts:215,401` | Use the same logical slot count with two binding maxima. Demonstrate which bound rejects a word-mode layout |
| `GPUTrajectoryRestriction` | Pair-merge spec pins counting an invalid segment once | A fixture that independently exercises the gap-pair and nonfinite-arithmetic increments. Preserve the documented mixed-unit meaning of `unknown` |

Stronger pins include the hand-computed network diamond and separate capacity/budget cases in `network-flow-paths.node.spec.ts:48,78,106`; exact coordinates, source rows and chainage in `line-smooth-metadata.node.spec.ts:80`; an independent integer CDF oracle in `triangle-cdf.spec.ts:123`; and sparse zone IDs in `raster-zone-dictionary.spec.ts:75`. The post-bump color fix compares output subsets against a nonempty full run in `gpu-color-scale.spec.ts:710` and `gpu-bivariate-classification.spec.ts:399`. These would expose the reported no-writer regression. The full-run comparison alone is not an independent classifier oracle.

Four specs read `dev-docs/roadmap.md` by literal path: `documentation-facts.node.spec.ts`, `release-package.node.spec.ts`, `roadmap-reconciliation.node.spec.ts`, `root-files-roadmap-handoff.node.spec.ts`, all under `modules/analysis/test/analysis-core/`. The last two use first matching rows. The reconciliation spec also expects BL-PASS migration phrases and parses sections between exact headings. Move these readers with the content. Preserve the AX distinction between a 500 ms **map update** and a fast table preview, and between a 24-row form fixture and the million-row profile gate.

## Maintainability

### Document ownership and consumers

`AC` below means `modules/analysis/test/analysis-core/`. Spec names in this table end in `.node.spec.ts`. “No reader found” refers to executable Markdown-content readers in the scoped search across `scripts`, `website`, `modules`, `test`, `examples`, `docs` and `dev-docs`; links and comments were counted separately.

| Document | Purpose and audience | Consumers, duplicated facts and owner |
|---|---|---|
| `README.md` | First result and orientation for callers | AC `first-run-docs:40`, `documentation-facts:109`, `release-package:70`. Link to installation compatibility in the release guide and behavioral details in the API guide. |
| `AGENTS.md` | Instructions a contributor needs before acting | AC `documentation-facts:110` expects runner/manifest paths; `release-package:43` extracts a section by heading. Gate machinery belongs in script/release guides. |
| `CHANGELOG.md` | Upgrade contract for package callers | AC `release-package:61` checks Unreleased and exclusion wording. Keep migrations here and link dated verification elsewhere. |
| `dev-docs/README.md` | Navigation and retention rules | Root instructions refer to it; no executable reader found. Own the policy allowing package-specific backlogs under one roadmap index. |
| `roadmap.md` | Decisions and open acceptance | Four AC specs above. Keep tasks here once; code owns bounds and the package contract owns qualification. |
| `authoring-walkthrough.md` | Worked coregionalization change | Developer index, AC `authoring-walkthrough:20`, `doc-publication:112`. Numbered steps are parsed. Own procedure, link normative rules. |
| `contributor-contract.md` | Normative contributor requirements | Walkthrough, upstream plan, roadmap; AC `documentation-facts:102` checks phrases. Checker implementation owns enforcement scope. |
| `testing-gates.md` | Numeric, device and structural test conventions | Developer index; no direct content reader found. Keep assertion conventions here. |
| `script-gates.md` | What local automation verifies | Contract, upstream plan, oracle guide; AC `analysis-layering:484` reads migration wording. Own verification mechanics. |
| `regression-loop-inventory.md` | Interprets serial axes, routes and costs for regression reviewers | Contract `:56`, developer index. No executable reader found. Checker/baseline own site identities; inventory owns interpretation. |
| `coverage-matrix.md` | Reference-operation coverage and import mapping | Analysis README, developer index, roadmap; AC `documentation-facts:171` reads `NW-SCC`. Historical `research/library-review-2026-11-09/census/crossref.mjs:59` reads IDs. Keep reference scope and oracle provenance here. |
| `release-package.md` | Installation, peer range, package bytes and release readiness | Package/root READMEs, AGENTS, upstream plan; AC `root-files-roadmap-handoff:83` parses snapshot columns. Own target, range and artifact procedure. |
| `upstream-first-plan.md` | Package boundary and exception policy | Root/package guides, register and roadmap. No content parser found. Resolve its grid-only exception against the non-grid pin. |
| `upstream-first-recording.md` | How public graph data becomes records | Index, upstream plan/proposals/register; AC `release-package:45` checks the opening. Keep recorder semantics in API/source, upstream availability in the register. |
| `upstream-workarounds.md` | Capability differences and retirement conditions | Website generator `website/src/components/meta/pages/upstream-first/generate-data.ts:57–74` parses the **last nonempty table cell** by leading status word. Preserve `dropped`, `workaround`, `accepted regression`, `obsolete`, `unmeasured`, or change the parser in the same patch. |
| `upstream-proposals.md` | Upstream requests with evidence | Roadmap, register, recording guide, draft index; AC `roadmap-reconciliation:174` reads continuation wording. Own requested changes, link draft bodies. |
| `upstream-drafts/README.md` | Filing order and locally recorded status | LR-23 and proposals; no executable reader found. Record a filing URL here when available. |
| `upstream-drafts/01-packed-cpu-gpu-transfer.md` | Packed transfer proposal and dated benchmark | Draft index and AC `upstream-drafts:231`, which extracts the first TypeScript fence. |
| `upstream-drafts/02-histogram-edges-and-domains.md` | Histogram reproducer and requested behavior | Draft index/proposals; no direct Markdown reader found. Preserve literal expected edges. |
| `upstream-drafts/03-caller-csr-and-masks.md` | Caller-owned topology proposal | AC `upstream-drafts:16,56,128` extracts positional fences and acceptance phrases. Keep the public reproduction executable. |
| `upstream-drafts/04-typed-buffer-and-cpu-vector.md` | Typed import and CPU construction proposal | AC `upstream-drafts:79` locates the `validateImportLayout` fence. |
| `research/README.md` | Revision and retention index | Developer index; no executable index reader found. Several descendants are executable inputs or generated outputs. |

### AGENTS adherence and proposed replacement

The upstream-first checker traverses the three packages’ source/tests and `examples/deck`, permits the root forwarding `index.ts`, and checks re-exports under `modules/*/src`. It does not traverse website, scripts or all Markdown. The ordinary contributor checker skips `testing` and `computation-record`; its separate public-surface check has different coverage. Current output was zero upstream-first violations, zero new contract findings, with 127 scalar-loop findings, 3 internal exports, 15 multiple-class files and 21 root files under existing allowances. `scripts/check-upstream-first.mjs:49–103,127`, `scripts/check-contributor-contract.mjs:25,153–208`.

| Rule sampled | Count and method | Exceptions |
|---|---|---|
| Paired SPDX headers | First 1,800 characters of 11,149 tracked TS/JS/Python/shell-family files in modules, examples, scripts, test, website. All 11,070 files with an SPDX licence line had a copyright line | This checks presence, not holder completeness. Files without an SPDX licence line are outside that specific implication. |
| Arrow imports | Per-file TypeScript AST, module specifier `apache-arrow`: `modules/*/src` had 10 named, zero namespace/default. Elsewhere 326 namespace, 11 named, zero default | Ten Kepler example files: `coordinate-dimension-names`, `geojson-trips`, `geometry-gpu-vectors`, `geometry-native-geoarrow` (two), `geometry-wkb`, `polygon-contiguity-topology`, `polygon-positions-low`, `tabular-trips`, `resident-spatial-join-merge`, `resident-spatial-join-polygons`. The outside-source rule is a preference. |
| Bare `assert(` | Text scan of non-comment lines: 643 occurrences in 211 files, zero in analysis source | Largest counts were `trip-markers.mjs` 24, `validate-saved-maps.mjs` 23, `s2-extract.mjs` 17, `gpu-polygon-extrusion.ts` 16, `styles.mjs` 15. Includes Node assertion APIs and fixture text. |
| Names | 50,695 simple variable/parameter declarations across 1,772 analysis TS files. Exact short-name list `tmp arr obj val idx buf num len acc res cfg opts params coords src dst pos ptr args ctx ref prev curr max min` matched 3. Separately, 268 single-letter declarations | `params` in `analysis-core/kernels/polygon-edge-slabs.ts:369`; `ref` in `computation-record/computation-node-ref.ts:11` and `validate-computation-record.ts:264`. Zero lexical casing exceptions in 3,792 type/class/interface and 3,762 function declarations. This cannot decide whether a mathematical name is appropriate. |
| Upstream boundary | One literal helper in the official WGSL-copy list; ten website files import engine outside checker traversal | `official-wgsl-copies.node.spec.ts:52`, `upstream-workarounds.md:207`. Site files include five meta scenes, demo kit, preview runtime and shared kernels in geometry-construction-overlay, regression-weights and space-time-events. Decide website scope without broadening module policy by implication. |

Reproduce inventory roots with `git ls-files modules examples scripts test website` and `git ls-files modules/analysis/src`. For Arrow/names, call `typescript.createSourceFile` one file at a time and count `ImportDeclaration`, `VariableDeclaration` and `Parameter` nodes with simple identifiers. For bare assertions use `rg -n '(^|[^[:alnum:]_.])assert[[:space:]]*\('` and distinguish calls, text and comments before acting.

| AGENTS lines | Proposed disposition |
|---|---|
| 3–18 | Keep commands, build-before-package-check order, local entry points and the two approved checker commands. Correct “loads every entry point.” |
| 20–32 | Keep runner/manifest links, combined source project and test heap default. Move grouping, Vite ambient types, manifests, configs and caches to `script-gates.md`. |
| 34–55 | Keep full-versus-partial meaning, one-heavy-command rule, 8 GiB free disk, sampled 12 GiB RSS and 8,192 MiB heap maximum. Move order, sampling, parsing and child precedence details to `script-gates.md`. |
| 56–75 | Move packed-worker isolation, report membership, flags and benchmark suffix details to script/testing guides. Keep the ordinary-versus-opt-in distinction. |
| 76–92 | Move incremental repair, hashing and concurrent CommonJS behavior to script gates. |
| 93–119 | Move snapshot, promotion, lock and interrupted rollback details **before shortening** to `release-package.md`. Keep staging versus release qualification and a recovery link. |
| 121–146 | Keep public entry points, dependency direction, allowed implementation roots and upstream-first order. Clarify the root barrel and the two unresolved policy mismatches. Keep `## Relationship to luma.gl` and `## Plans and docs`, or update the section parser. |
| 148–169 | Keep lint-fix, full verify before merge, strict TS style, attribution, descriptive names, named validation, Arrow convention and dataset restrictions. Add the worked authoring link. |

**Proposed shorter AGENTS text**, after the detailed guides are ready:

```markdown
# AGENTS.md

## Work locally

Use `nvm use`, then `yarn install`. `yarn typecheck` checks the combined source
project. `yarn typecheck-tests` stages the test families and example projects.
See the [runner](scripts/typecheck-tests.mjs) and
[exclusion manifest](test/kepler-compute-example-specs.json). The runner
preserves caller options and defaults to an 8192 MiB heap.

Run `yarn lint-fix` after changes. For focused checks use
`yarn check-upstream-first`, `yarn check-contributor-contract`,
`yarn test-node` or `yarn test-headless`. `yarn test-benchmarks` is separate.
Use `yarn build` before `yarn verify-package-exports` and `yarn check-packed-consumer`.
The export checker loads JavaScript entries and resolves asset targets.
Start the showcase with `cd examples/deck/spatial-analysis-showcase && yarn start`.

Run one heavyweight checkout command at a time. Run a full `yarn verify`
before merging. Resumed and single-step runs provide partial evidence.
Verify requires at least 8 GiB free checkout disk and enforces a sampled
12 GiB process-tree RSS limit and an 8192 MiB V8 heap maximum. Allow room
on the packed-consumer temporary filesystem. See [script gates](dev-docs/script-gates.md).

`yarn build:release` stages a clean snapshot and checked tarball. Read the
[release and recovery procedure](dev-docs/release-package.md) before promotion.
Inspect the lock owner and promotion journal before recovery. Publish the
checked bytes only after metadata is settled and the release gate passes.
Keep packages containing unresolved restricted datasets private.

## Relationship to luma.gl

Use published core, shaders and graph public entries. Engine is restricted
to the allowed rendering consumers under the upstream-first policy. Keep
private paths, deep dist imports, generic forks and re-exports out of modules.
Use the public API, adapt the design, drop the feature, then consider a small
analysis-domain helper. Send generic improvements upstream. Consult
[the package contract](dev-docs/release-package.md) and
[the boundary and exception policy](dev-docs/upstream-first-plan.md).

Keep analysis implementation under analysis-core, computation-record,
gpu-spatial-analysis, gpu-network, gpu-terrain, testing, dataframe-analysis,
raster-analysis, pointcloud-analysis or geospatial-analysis. The source root
holds its forwarding index. Dependency direction is kepler → lenses → analysis
→ published luma.gl.

## Plans and docs

Start at [the roadmap](dev-docs/roadmap.md). Follow the
[contributor contract](dev-docs/contributor-contract.md) and
[worked authoring guide](dev-docs/authoring-walkthrough.md). Read the
[retention policy](dev-docs/README.md#research-retention) before moving evidence.

Use strict TypeScript, semicolons, single quotes and descriptive names.
Preserve each SPDX expression and holder attribution. Validate props once
at construction with contributor-ID-prefixed named-limit helpers. Avoid bare
assertions and per-row or per-frame CPU validation. Use individual Arrow
imports in module source and prefer the Arrow namespace in tests, examples
and docs. Consult THIRD-PARTY-DATA.md before distributing example or lens data.
```

### Link failures and research retention

The lead’s current Markdown scan covered **274 files**: 20 live/root guides, 116 under `docs/`, 129 tracked research Markdown files, and nine incoming reports. It checked parsed local links, images and reference definitions, skipped code and external URLs, decoded destinations, resolved repository docs routes and checked heading fragments separately. On the working tree there were **441 historical missing-target occurrences and ten live occurrences of one pending target**. There were no missing file targets under `docs/`. The two non-doc routes `/explore/` and `/fn/point-density` have route definitions; deployment was not exercised.

- The live pending target is `dev-docs/research/library-review-2026-11-19/README.md`, defined at `dev-docs/roadmap.md:24` and used at lines 60, 64, 388, 390, 402, 405, 480, 504 and 507. This is an assembly dependency for the lead reviewer.
- Two live fragments are broken: `docs/api/gpu-spatial-analysis/operators.md:19` and `docs/api/gpu-spatial-analysis-cross-reference.md:126` use `#gpukmeans`. The heading at `docs/api/gpu-spatial-analysis/operators-statistics.md:1562` resolves as `#gpuspatialkmeans`.
- Two historical fragments at `research/picture-rescore-2026-11-10/README.md:3` and `research/spatiotemporal-review-2026-11-09/README.md:3` point to old numbered roadmap headings, `#76-open-picture-items-web-pic` and `#215-spatial-and-spatiotemporal-directions-2026-11-09`.
- Seven historical `#L` destinations exceed current file length, all from `dev-docs/research/library-review-2026-11-17/`: `area-api-ergonomics.md:107,127,160` to pointcloud `index.ts#L29`, analysis `README.md#L80`, and raster spectral indices `#L130`; `area-tooling-tests.md:20,100,169,190` to `vitest.config.ts#L95,#L92,#L104` and the lenses computation-pack index `#L175`. An eighth exceeded the committed 529-line roadmap; its `#L548` now exists but points into a different document state.
- The 441 historical misses comprise 401 distinct missing path targets. The source files, with occurrence counts, are: October geospatial implementation survey `README` 11, `geometry` 9, `network-movement` 18, `raster` 17, `shared-patterns` 13, `statistics` 10, `terrain` 11; Kepler sandbox `README` 2 and `verification` 4; November 17 `area-api-ergonomics` 5, `area-lenses` 84, `area-network` 104, `area-tooling-tests` 1, `thinking-partner` 70 and `verification` 60; picture-rescore `README` 1; scalability `README` 21. At `bdc715ea`, 305 of the November 17 set’s 324 resolve after removing literal `:line` suffixes, as do all 21 scalability targets. Use revision permalinks with `#L…`.

The following is the expandable historical broken-link list. It prints each offending original target beside its source block, including repeated references in table rows. This keeps all 441 occurrences recoverable without making the working backlog depend on an expanded historical dump.

```sh
node -e '
const fs = require("fs"), path = require("path"), md = new (require("markdown-it"))();
function walk(dir) {
  return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e => {
    const file = path.join(dir,e.name);
    return e.isDirectory() ? walk(file) : file.endsWith(".md") ? [file] : [];
  });
}
for (const file of walk("dev-docs/research")) {
  const env = {}, tokens = md.parse(fs.readFileSync(file,"utf8"),env);
  let line = 1;
  for (const token of tokens) {
    if (token.map) line = token.map[0] + 1;
    for (const child of token.children || []) {
      const url = child.type === "link_open" ? child.attrGet("href")
        : child.type === "image" ? child.attrGet("src") : "";
      if (!url || /^(?:[a-z][\w+.-]*:|\/|#)/i.test(url)) continue;
      const target = decodeURIComponent(url.split(/[?#]/)[0]);
      if (target && !fs.existsSync(path.resolve(path.dirname(file),target)))
        console.log(file + ":" + line, url);
    }
  }
}'
```

**Retention ledger.** Paths are under `dev-docs/research/`. File counts and bytes are tracked unless marked live. “Keep part” authorizes pruning only after the specified evidence or consumer has a durable home.

| Directory or root file | Files / bytes | Consumer, relationship and disposition |
|---|---:|---|
| `README.md` | 1 / 6,691 | **Keep.** Maintained index linked at `dev-docs/README.md:19`. Correct the script direction and add the parity report and November 19 index. |
| `kepler-source-parity-2026-11-14.md` | 1 / 27,018 | **Keep.** `dev-docs/roadmap.md:306` links the historical audit. Preserve distinction between first-round and subsequent verification. |
| `analysis-workflow-ux-2026-11-14` | 23 / 876,294 | **Keep part.** AX design, decisions, binding findings and baseline. `examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs:109,1709` resolves/protects the historical directory and reads `journeys.json`. Move realpath protections and tests with all eight baseline files, including images. |
| `cpu-suitability-2026-11-11` | 4 / 458,991 | **Keep.** CPU-03, contributor contract and draft 01 use survey rows, measurements and environment. |
| `evidence` | 14 / 1,206,272 | **Keep.** Performance/API guides refer to measurement captures. Preserve the six Kepler resident records, hashes and CPU profile; see `evidence/README.md:8–12`. |
| `extended-precision-2026-11-11` | 2 / 20,096 | **Keep.** `modules/analysis/src/analysis-core/kernels/wgsl-kernel-nodes.ts:366` cites the arithmetic rationale; preserve probe and environment. |
| `fork-comparison-2026-11-13` | 1 / 25,349 | **Keep.** Boundary plan `:27`, proposals `:351`, UF-03 and INT-07 use the fork-to-public comparison. |
| `gpu-geospatial-implementations-2026-10-23` | 7 / 107,903 | **Keep part.** Preserve reference/licence provenance and open design choices. Historical ordering can retire after transfer. No maintained executable consumer identified. |
| `gpu-geospatial-performance-2026-10-23` | 1 / 7,648 | **Keep.** `docs/api/gpu-spatial-analysis/interpolation-expansion.md:161` cites it. Preserve the original branch/revision location. |
| `kepler-sandbox-2026-10-28` | 4 / 251,845 | **Keep part.** `examples/deck/kepler-compute/scripts/capability-matrix.mjs:8,37` writes/checks the capability matrix. Move the generator destination and links together. Keep the verification log dated. |
| `kepler-workflows-2026-11-14` | 1 / 41,926 | **Keep.** `modules/kepler/src/workflows/README.md:5` and KE-WF cite design choices. |
| `library-review-2026-11-09` | 12 / 449,376 | **Keep part.** Contract and STD rows cite rationale. Keep two census utilities as snapshots; port and test them before making them operational. |
| `library-review-2026-11-17` | 17 / 667,926 | **Keep.** LR decisions, verification, unread fix rounds and release evidence underpin current work. Anchor historical links to the reviewed revisions. |
| `library-review-2026-11-19` | live 9 / 383,732 | **Keep during assembly.** Add revision, scope, execution record and index. Do not retire incoming reports. |
| `luma-usage-review-2026-11-11` | 19 / 358,691 | **Keep part.** Dated baselines, probes, costs and original three scripts are evidence. `scripts/upstream/README.md:23` identifies maintained replacements. |
| `math-correctness-review-2026-11-10` | 32 / 1,542,752 | **Keep part.** Compiler reductions, closing reviews, method differences and row-direction decisions have contract, API and spec citations. Transfer citations before pruning coordination. |
| `new-contributors-2026-11-11` | 11 / 240,661 | **Keep.** Options, rejected candidates and three archived probes support the maintainer’s choice. Freeze the dated census. |
| `picture-rescore-2026-11-10` | 2 / 121,338 | **Keep pending comparable replacement.** WEB-PIC-12 consumes dated scores. |
| `scalability-limits-2026-11-15` | 1 / 331,088 | **Keep.** Route-specific scale evidence underlies STD-14; indexed at `dev-docs/README.md:33`. |
| `showcase-overhaul-2026-10-25` | 28 / 646,704 | **Keep.** The index and 27 JSON files include executable fixture inputs. `examples/deck/spatial-analysis-showcase/scripts/check-data-topology-reference.mjs:13–15` constructs the fire paths; research README `:13–32` maps other readers and generators. |
| `spatiotemporal-review-2026-11-09` | 13 / 454,853 | **Keep part.** Keep two float32 experiments, settings, external comparison gaps and method evidence. Hawkes/Vecchia API guides and the Vecchia study spec cite them. |
| `split-review-2026-11-08` | 1 / 14,016 | **Keep pending INT-07.** Its 18 historical leads need an explicit disposition. |

No full directory met the conditions for unconditional retirement. Largest tracked research file was 763,812 bytes. The tracked inventory contained no archive or bytecode file.

### Changelog

`CHANGELOG.md` marks breaking changes and includes usable old-to-new mappings. Its first 116 lines contain 27 bullets before the breaking-change heading. The next 240 lines contain 84 nested bullets. There are 24 literal before/after arrows. These counts describe the reading burden; they do not measure migration completeness. The post-bump optional-colors correction is already recorded at `CHANGELOG.md:18–21`, and the BL-PASS migration handoff is present at `:77–83`.

Use one Unreleased section with **Compatibility and installation → Required migrations by family → Changed result semantics → Added capabilities → Fixes → Qualification and known limits**. Move private consumer changes behind links. Replace the ambiguous “125 contributors” with a dated, defined inventory. A CPU operation class, pure function and GPU contributor are different census units.

| Family | Migration text suitable for the alpha |
|---|---|
| Network | Facility optimization and rational access publish packed words in `output.summary`. Request named diagnostics through `output.status`. See `CHANGELOG.md:77`. |
| Spatial joins | Request and pass `neighborRows` for weights. Use `neighborIds` for application identity. Propagate producer completeness through `neighborStatus` and allocate the consuming candidate-loss port where required. See `CHANGELOG.md:200–206`. |
| Alignment | Replace diagnostic `validity` with `alignmentCode`. A unique match has code zero. Select usable rows with nonzero `matchedMask`. Recipe-level packed diagnostics moved to `packedStatus`. See `CHANGELOG.md:212–215`. |
| Raster | Repack old manual reprojection layouts with `getGPURasterReprojectionParameterValues`. Destroy the compiled graph before `contributor.destroy()`, including same-CRS use. See `CHANGELOG.md:81–83,307`. |
| Classification | `GPUColorScale` and `GPUBivariateClassification` can omit colors and still write requested classes/counts. The fix is in `73e8f638`. No rename is required. |

## Roadmap rows

**Proposed table of contents.** Keep `dev-docs/roadmap.md` as the entry point with four short sections: **Before the alpha**, **Core backlog by domain**, **Decisions**, **Completion references**. Link proposed `dev-docs/limits.md`, `dev-docs/kepler-roadmap.md`, `dev-docs/lenses-roadmap.md`, `dev-docs/website-roadmap.md`, and the existing upstream register. Put example-only work with its owning host, including a small examples subsection for the non-Kepler examples. The known-limits register earns its own file because limits survive completion and are consulted by callers planning inputs. Each entry needs route, units, bound or formula, failure/status behavior, source, spec and dated measurement where relevant.

**How to apply the ledger.** Every numbered line below is one physical row in `dev-docs/roadmap.md` at the working hash. The original section is recoverable directly from the line and section counts above. `I` = index, `A` = core/repository, `K` = Kepler, `E` = examples, `W` = website, `L` = lenses, `U` = upstream, `D` = decision, `B` = known limit. H/M/L are existing priorities; `d` is the Decision priority, `–` means none. `KEEP` preserves the stated acceptance or qualification. `NARROW` gives the replacement limit in the evidence column. `UP` means reprioritise as specified. `CLOSE` deletes the open row, preserving the stated destination when needed. The primary kind supplies the default destination; `now`, `limits`, `decision`, `research`, `API` and cross-IDs override it.

To keep the ledger usable, source filenames below are repository-relative through these conventions: implementation stems resolve under `modules/analysis/src`, spec stems under `modules/analysis/test`, `AC` is `modules/analysis/test/analysis-core`, `E/` is `examples/deck/kepler-compute/`, and `D/` is `dev-docs/`. A unique basename and line can be expanded with `rg --files modules/analysis | rg '/<basename>\.(ts|md)$'`. Source IDs in the original roadmap remain part of each row’s evidence. Consumer-file checks establish existence only.

| Line / old ID | Kind · priority | Verdict / destination | Evidence or replacement Current limit |
|---|---|---|---|
| 32 MOB-03 | I – | KEEP → decision | `track-quality-recipe.ts:45` defaults to step. Index the unresolved default choice. |
| 33 WEB-PIC-12 | I – | KEEP → W | Link the website’s two-theme rescore; keep scores dated. |
| 34 KE-TESTS | I – | KEEP → K | Link the separate manifest populations, `test/kepler-compute-example-specs.json:2`. |
| 35 UPS-CPU-1 | I – | KEEP → U | Link draft 01 and CPU-08; no new measurement in the index. |
| 36 AX-01, AX-20, AX-40, AX-70 | I – | KEEP → K | Retain four distinct host gates at lines 294,296,299,302. |
| 37 STD-04, STD-06, STD-07 | I – | NARROW → A | Link remaining named classes, coordinate and band decisions. Remove completed port design from the summary. |
| 38 GPU-05 | I – | KEEP → A | Large-radius work survives. Working edit removed the blanket snapping prerequisite. |
| 39 LUM-09 | I – | KEEP → D | Decide what iteration count means after algorithm substitution. |
| 40 LR-04, LR-22, LR-23 | I – | NARROW → now | Accept checked bytes, decide canonical URL and qualify the candidate. Track upstream filing separately unless the maintainer makes it a release condition. `D/release-package.md:82,91`. |
| 41 KI-01 to KI-10 | I – | NARROW → now | Record candidate inclusion and verification owner; put stream counts and timestamps only in the KI digest. |
| 57 LR-02 | A M | KEEP | Numerical regeneration remains. Runner/inventory tests do not recreate GRASS/R/GDAL values. `scripts/oracle-fixtures.json:1066`. |
| 58 LR-04 | A H | KEEP → now | Resolve repository URL and release version, identify checked tarball, qualify selected revision. `D/release-package.md:3,82,91`. |
| 59 LR-08 | A M | KEEP → KI-06 | Three multipart predicates, neighbors and preparation ship. Weights consumes neighbor rows; line length takes linestrings. Preserve remaining feature-level reductions/refusals. `gpu-spatial-predicate-join.ts:679`, `gpu-nearest-feature-weights.ts:88`. |
| 60 LR-10 | A L | NARROW → limits | Prepared recipe failures precede caller mutation; final graph mutation lacks rollback. Separate other barrels and mark the late-randomization example as committed-revision evidence. `prepared-recipe.ts:138`, `segregation-decomposition.ts:88,142`. |
| 61 LR-17 | A L | KEEP | 16 raster and five network root files remain in `scripts/contributor-contract-baseline.json:121`. |
| 62 LR-19 | A M | KEEP | Require ≥20% sampled RSS improvement, ≤10% cold/warm regression, same heap and idle machine. Prior timing lacked RSS. `D/release-package.md:171–209`. |
| 63 LR-20 | L M | KEEP → L/D | Structural signatures and 427 dated captures exist. Licensing, workspace and numeric/caption qualification have different owners. `scenario-declaration.node.spec.ts:55`. |
| 64 LR-22 | A M | UP → H, now | KI-04 supplies a sufficient scheduling mechanism. Verify contract and costs on the release candidate. Keep segmented sums/PCG adoption at M. `raster-relaxation.ts:887,948`, KI digest `:293`. |
| 65 LR-23 | U M | KEEP → U | Four draft bodies exist. Record filing evidence once available. `D/upstream-drafts/README.md:13`. |
| 66 LR-24 | A L | NARROW → LR-08/KI-06 | Multipart nearest currently requires neighbor output. Keep one nearest-mode restriction, linked to its issuer. `gpu-nearest-feature-join.ts:513`. |
| 67 LR-25 | A L | KEEP → limits | Decomposition ships; ≤128 expanded vertices, ≤1,024 edges, path capacity 1…1,024. `network-flow-paths.ts:55`, `gpu-network-flow.ts:616`. |
| 68 LR-26 | A L | NARROW → limits/KI-07 | GM, 2SLS and traces enforce a tile limit. ML main tiles lack that cap; nested traces coarsen to at most one tile per nonempty chunk then reject above 4,096. Cite `regression-trace-layout.ts:9,17–25`, not only the default-tile constant. |
| 80 KI-01 | A – | KEEP → now, staged | Record each public output’s oracle and failing parent evidence. `gpu-bivariate-classification.spec.ts:399` illustrates output-presence versus independent-value coverage. |
| 81 KI-02 | A – | KEEP → statistics audit | Reconcile audited folders, model, oracle, current revision and residual findings. Source population and stage progress are separate. KI digest `:35–48`. |
| 82 KI-03 | A – | UP → now | Name every reread, including tranche-one second fix round, with finding/commit/verifier. Staging review completion is not sufficient. November 17 `implementation.md:85–94,116,137`. |
| 83 KI-04 | A – | NARROW → LR-22 | Pin fixed-order counts in an emulator, device costs and per-budget bounds in correctness tests; repeated sampling supplies diagnostics. KI digest `:293`. |
| 84 KI-05 | A – | KEEP → ST03 | Proposed panel-error effects. Current public route is unit fixed effects; preserve oracle/qualification per effect. `gpu-spatial-panel-error-maximum-likelihood.ts:70`. |
| 85 KI-06 | A – | KEEP → LR-08/24 | Preserve per-predicate, DE-9IM, nearest, weights and refusal acceptance. Staged nearest/point-join verify failures remain explicit. KI digest `:30–33`. |
| 86 KI-07 | A – | KEEP → LR-26 | Proposed two-level merge; require old-bound equivalence, new bound and chunk-seam cases. `regression-trace-layout.ts:25`. |
| 87 KI-08 | A – | KEEP → ST02 | White/spatial HAC restriction inference is distinct from pooled Chow/Wald. `gpu-restricted-regime-regression.ts:51`. |
| 88 KI-09 | A – | KEEP → ST04 | Iterated feasible GLS needs budget and status; landed SUR is single-pass. `gpu-seemingly-unrelated-regression.ts:79`. |
| 89 KI-10 | A – | KEEP → ST05 | Qualify the spatial-lag GMM probit against a reference; ordinary binary probit does not close it. `gpu-probit-regression.ts:185`. |
| 99 UF-01 | A M | NARROW → LR-04/D | One canonical repository owner decision for metadata, README and header. `modules/analysis/package.json:10`, `website/src/components/brand/site-header.tsx:8`. |
| 100 UF-03 | A M | NARROW | Compare named current primitive/route, fixture and adapter with a dated baseline. Alpha.13 absence is historical. `D/upstream-workarounds.md:58`. |
| 101 UF-12 | E M | NARROW | Local example typechecks exist. Missing import is `scripts/playwright/collect-page-diagnostics.mjs` at `examples/deck/spatial-analysis-showcase/scripts/check-sprint-scenes.mjs:12`. Explorer check and visual acceptance remain. |
| 102 UF-10 | A L | CLOSE → LR-04 | Authoring and release guides exist (`D/authoring-walkthrough.md:1`, `D/release-package.md:1`). Public access metadata alone cannot establish publication. |
| 103 SH-DATA | E L | KEEP → decision | Size/distribution choice for showcase assets. Preserve provenance and licensing in `THIRD-PARTY-DATA.md`. |
| 104 SH-NAMES | E L | CLOSE | All three `examples/deck/*/package.json:2` use `luma-contributors-examples-*`. Example package versions are a separate decision. |
| 105 INT-07 | A M | NARROW → research | Triage each of the 18 split findings against a named current path and acceptance case. “Re-check dropped files” cannot close as one task. `D/research/split-review-2026-11-08/README.md:114`. |
| 106 INT-08 | A M | NARROW → decision | Website CI typecheck/build jobs exist. Decide Pages policy and record execution evidence. `.github/workflows/test.yml:83,106,115`. |
| 107 INT-09 | W L | CLOSE → LR-04/D | Merge header URL choice with UF-01; retain the site consumer of the decision. |
| 113 ST02 | A L | KEEP → KI-08 | Robust covariance remains distinct from existing restricted-regime inference. `gpu-restricted-regime-regression.ts:51`. |
| 114 ST03 | A L | KEEP → KI-05 | Error unit FE and lag FE/RE differ. Effects, clustered covariance and dynamic models require separate acceptance. `gpu-spatial-panel-lag-maximum-likelihood.ts:166`. |
| 115 ST04 / ST05 | A L | KEEP → KI-09/10 | Preserve both IDs and separate iterated SUR from latent spatial probit. |
| 116 ST06 | A M | KEEP | Full-backfit inference exists; bandwidth-selection uncertainty remains. `gpu-multiscale-geographically-weighted-regression.ts:169`. |
| 117 ST08 | A L | KEEP → limits/backlog | Sorted ranks exceed 64 states; optional reference diagnostics retain 64. Missing-panel/fractional-tie work survives. `gpu-rank-markov.ts:74,100`. |
| 118 ST09 | A L | KEEP | Island attachment uses planar distances; distances output explicitly refused. `gpu-island-attachment.ts:121,201`. |
| 119 ST10 | A L | NARROW | Existing block/group/cluster/NNDM routes leave hierarchical/merged blocks, response balance and continuous variogram optimization open. `gpu-spatial-cross-validation.ts:75`. |
| 120 ST11 | A L | NARROW | One-covariate calibrated Poisson exists. Additional covariates, distributions and coefficient uncertainty remain. `gpu-spatial-interaction.ts:103`. |
| 121 ROSE-01 | A L | NARROW | Radians/degrees and both origins ship **and are documented**. Defer further conventions until one is requested. `gpu-rose-statistic.ts:164`, `operators-statistics.md:1001`. |
| 122 KG02 | A L | KEEP → limits | Dorling has a ≤256-pass budget, exhaustion status and no GeoDa parity guarantee. Retain a named parity fixture. `gpu-dorling-cartogram.ts:47,171`. |
| 123 KG03 | A L | KEEP → limits | Natural breaks has 1,024 materialized, 8,192 recomputed rows and 64 classes. `natural-breaks-limits.ts:6–8`. |
| 129 PP01 / PP02 | A L | KEEP | Border/translation corrections and rank intervals exist. Remaining targets are inhomogeneous isotropic correction and extreme-rank-length ties. `gpu-inhomogeneous-point-pattern.ts:60`, `gpu-point-pattern-envelope.ts:89`. |
| 130 PP05 | A L | KEEP | Response-dependent replacement and design variance remain distinct from independent replacement waves. `gpu-spatial-balanced-sample.ts:83`. |
| 136 IN04 | A L | KEEP → limits | 2–4 variables, 1–4 structures. CPU/GPU public names now agree. `coregionalization-parameters.ts:30,53`. |
| 137 IN05 / IN02 | A L | KEEP | Separate the 32-neighbor kriging system, usable-sample RBF indexing and global solvers. `AC/roadmap-reconciliation.node.spec.ts:72`. |
| 138 IN06 | A L | KEEP → limits | Joint covariance applies within one query set. Independent calls have no shared joint simulation. `gpu-conditional-gaussian-simulation.ts:94`. |
| 144 NW03 | A L | KEEP | Negative-cost circulation and lower-bound constraints remain distinct from nonnegative costed flow. `gpu-network-flow.ts:108,467`. |
| 145 NW05 | A L | NARROW | **Every operation:** `F≤16,D≤64`. Subset routes use `2^F`; capacitated p-median additionally requires `F^D≤2^20`. A smaller examination budget does not admit a larger model. `gpu-network-facility-optimization.ts:118–154`. |
| 146 NW06 / X04 | A L | KEEP | FIFO, 256 states and 4,096 transitions on reference route. Separate non-FIFO/time windows from transit-service ingestion. `gpu-network-time-routing.ts:70,85`. |
| 147 NW-GA | A L | NARROW → upstream/design | Identify a consumer, dense traversal and public primitive whose current topology contract prevents use. Network optimization public index and installed `gpu-graph/index.ts:19`. |
| 153 R6.4 | A M | NARROW | Indexed/tiled fractional coverage exists. Name the outstanding whole-versus-tiled fixture, conservation tolerance and status. `gpu-raster-fractional-coverage.ts:162,247`. |
| 154 R7.2 | A L | KEEP → limits | FIFO flood ≤16,384 cells; rooted forest <2³⁰ and device-bound. Scikit-image plateau order is a separate comparison. `gpu-raster-watershed-segmentation.ts:77,122`. |
| 155 R7.3 | A L | KEEP → deferred | 2D CLAHE exists. Require a named volumetric microscopy consumer. `raster-analysis/index.ts:34`. |
| 156 RS02 | A L | KEEP | Sparse uint32 dictionaries ship for whole-cell zonal statistics. Strings and sparse fractional categorical input remain. `gpu-raster-zonal-statistics.ts:233,246`. |
| 157 RS03 | A L | KEEP | Regular axis-aligned conservative remapping; curvilinear/spherical cells remain. `gpu-raster-conservative-remapping.ts:228`. |
| 158 RS06 | A L | KEEP | Retain per-class date-axis inventory. Irregular-date resampling already exists. `gpu-raster-temporal-resample.ts:80`, `gpu-raster-temporal.ts:172`. |
| 159 RS10 | A L | KEEP | Supplied-endmember QR/NNLS exists. Discovery, ISRA and NMF remain. `gpu-raster-spectral-unmixing.ts:55`. |
| 160 RS11 | A L | NARROW | BFAST Monitor and seasonal-break fitting exist. Multiple breaks, full BFAST and calibrated inference remain distinct. `gpu-raster-bfast-monitor.ts:88,109`. |
| 161 R-REPROJ | A L | KEEP → limits | Caller fetches windows/halos; bounded mosaic assembly supports nearest/bilinear. `gpu-raster-tile-reprojection.ts:54,223`. |
| 162 TE06 | A M | KEEP | Bounded D8 tile scratch ships. Host paging, independent filling and branching transport need separate fixtures. `gpu-terrain-tiled-flow.ts:111,145`. |
| 163 RELIEF-01 | A L | KEEP → deferred | Published histogram/percentile composition supplies approximate interior bounds. One-call exact interior percentiles are a distinct request. `gpu-relief-blend.ts:206`. |
| 164 A-18 | U L | NARROW → register | Spike repair already owns and destroys persistent buffers. Identify the missing upstream **caller-topology/view** contract if the purpose is to retire that workaround. `gpu-terrain-spike-repair.ts:234,354,963`, `D/upstream-workarounds.md:318`. |
| 165 TE05 | A L | KEEP | Current DEM coregistration is a robust vertical shift. Horizontal/Nuth–Kääb/tilt remain. `gpu-terrain-dem-coregistration.ts:103`. |
| 166 LUM-24 | A M | NARROW | Default tiled classifier exceeds 4,096 training rows. Brute-force retains the cap. Measure comparable scratch for both routes before assigning a migration cost or saving. `gpu-raster-embedding-knn-classifier.ts:109,156`, `embedding-scale.spec.ts:446`. |
| 172 KG06 | U L | KEEP | SQL lives in installed graph `gpu-sql/index.ts:5`. Qualify listed gaps individually upstream. |
| 173 DF-DESC | U M | KEEP | Public dataframe query surface exists; intent/shader/formula description was absent from sampled exports. Installed `gpu-dataframe/index.ts:6`. |
| 174 ARROW-01 | A L | NARROW | Existing normalizer supports narrow numerics, dictionaries/strings and time words. Name unmet nested/list/struct consumers. `arrow-analytics-normalization.ts:149,180`. |
| 175 PC01 | A L | KEEP | Per-query viewpoint/up orientation exists; graph propagation remains. `gpu-pointcloud-features.ts:84,264`. |
| 176 PC02 | A L | KEEP | Current filter computes input-global moments; distributed threshold reconciliation needs a contract. `gpu-pointcloud-statistical-filter.ts:64`. |
| 177 PC03 | A L | NARROW | Cell, bilinear and bounded IDW height interpolation exist. Request ground-point Delaunay/nearest or a vertical datum explicitly. `gpu-pointcloud-height.ts:118,282`. |
| 178 PC05 | A L | KEEP | Initialized robust point/plane ICP exists. Global registration and a trimmed objective remain. `gpu-pointcloud-icp.ts:189,273`. |
| 179 CR02 / CR03 | A M | KEEP | Supplied grids remain resident; separate epoch selection from bounded acquisition/residency. `gpu-datum-grid-transform.ts:80,142`. |
| 180 CR04 / CR05 | A L | KEEP | Spherical predicates, S2 covering and compaction exist. Exact adaptive guarantees and cell-budget minimization remain. `gpu-s2-cap-covering.ts:121`, `D/coverage-matrix.md:60`. |
| 181 P.7a | U L | KEEP | ECEF/static Helmert absent from sampled published projection exports. Installed `gpu-project/index.ts:32`. |
| 182 GE01 / GE08 / UR01-UR05 | A L | NARROW → limits | Split depth 64, general skeleton 1,024, convex skeleton 64 and distinct urban capacities/statuses. `gpu-normalize-geometry.ts:106`, `gpu-vector-straight-skeleton.ts:68`, `gpu-convex-straight-skeleton.ts:78`. |
| 183 EC02 | A L | KEEP | High-count correctness fixtures differ from high-count timing. Record patch count, storage route and memory. `patch-proximity.ts:37,106`. |
| 184 EC04 | A L | NARROW | Existing dissimilarity/Shapley/paired-label null is named. Require a specified additional statistic, null and reference before promotion. `segregation-decomposition.ts:61`, `segregation-layout.ts:9`. |
| 190 KE-TESTS | K H | KEEP | 1 runtime/58 typecheck exclusions; separately qualify cold-cache DuckDB. |
| 191 KE-EX | E M | KEEP → Kepler port | `E/package.json` exists. Port acceptance remains. |
| 192 KE-UI | E M | KEEP → Kepler | Preserve picking, labels and hardware acceptance. |
| 193 KE-DECK | E L | KEEP → Kepler | Cluster, flow and globe layer integration belongs with external host consumers. |
| 194 INT-04 | K M | CLOSE → KE-TESTS | Scoped manifest/tests contain no old alpha.10/physical-transient rationale. Preserve current per-file reasons. |
| 195 CPU-06 | K M | KEEP | `modules/kepler/src/analysis/catalog-gpu-snapping.ts` exists. Qualify fallback replacement there. |
| 196 LUM-31 | K L | KEEP | Require null-key group and nonfinite count parity. |
| 197 LUM-13 / LUM-32 | K M | KEEP | Preserve submit-confirmation condition for filter re-encoding. |
| 198 LUM-14 / LUM-33 | K M | KEEP | Preserve explorer pass-through and readback-length acceptance. |
| 206 KE-PORT-0 | E – | KEEP → port | Workspace exists. Qualify dependencies, aliases and launch. |
| 207 KE-PORT-1 | E – | KEEP → port | `E/gpu-layers/` resolves, 31 files. |
| 208 KE-PORT-2 | E – | KEEP → port | `E/src/geometry-preparation/` resolves. |
| 209 KE-PORT-3 | E – | KEEP → port | Public helpers exist. Example import/build qualification remains. |
| 210 KE-PORT-4 | E – | KEEP → port | Keep app/viewer build qualification. |
| 211 KE-PORT-5 | E – | KEEP → port | `E/scripts/smoke.mjs` and named driver scripts resolve. |
| 212 KE-PORT-6 | K – | NARROW → port | Qualify current runtime and typecheck sets. Drop the historical 43-file tier as current inventory. |
| 213 KE-PORT-7 | K – | KEEP → port | Preserve independent review and full-gate acceptance. |
| 214 KE-PORT-SAMPLES | E – | KEEP → port | Preserve remote admission, rollback, offline and attribution checks. `E/scripts/kepler-demo-data.mjs` resolves. |
| 249 KE-WF-A | K – | KEEP → workbench | `workflow-types.ts` and `workbench-contract.ts` resolve. |
| 250 KE-WF-B | K – | KEEP → workbench | Data, expression and document files resolve under `modules/kepler/src/workflows/`. |
| 251 KE-WF-C | K – | KEEP → workbench | Registry, source and table operations resolve. Keep chain acceptance. |
| 252 KE-WF-D | K – | KEEP → workbench | Plan/edit/history files resolve. |
| 253 KE-WF-E | K – | KEEP → workbench | Runner exists. Preserve cache, invalidation, cancel and publish checks. |
| 254 KE-WF-F | K – | KEEP → workbench | Template/export/assistant files resolve. Preserve CPU template gate. |
| 255 KE-WF-G | E – | KEEP → workbench | Host/store/extension files resolve. Preserve replay gate. |
| 256 KE-WF-H | E – | KEEP → workbench | 16 `workbench-*.tsx` matches. Interaction unqualified. |
| 257 KE-WF-I | E – | KEEP → workbench | Eight `workflow-*.mjs` matches. Preserve screenshots and keyboard acceptance. |
| 258 KE-WF-J | K – | KEEP → workbench | Retain independent-review gate. |
| 260 KE-WF-K | K – | KEEP → workbench | SQL/suggestions/report files resolve. Repair detached Markdown row. |
| 261 KE-WF-L | K – | KEEP → workbench | Recommendations file resolves. Repair detached row and preserve replacement-operation check. |
| 294 AX-01 | E – | KEEP → AX | Two-action form interaction qualification. |
| 295 AX-04 | E – | KEEP → AX | Independent screenshot review. |
| 296 AX-20 | K – | KEEP → AX | One million points/eight columns and staged latency thresholds. AC `roadmap-reconciliation:112`. |
| 297 AX-31 | K – | KEEP → AX/D | Real-data licensing/admission remains distinct from the 28 synthetic examples. |
| 298 AX-33 | W – | KEEP → W | Chain-intermediate capture; `website/scripts/snapshot-workflow-intermediates.mjs` exists. |
| 299 AX-40 | E – | KEEP → AX | All wave-one maps, including KDE, within 500 ms. AC `roadmap-reconciliation:98`. |
| 300 AX-50 | K – | KEEP → workbench | GPU-resident handoff remains KE-WF-08. |
| 301 AX-60 | E – | KEEP → AX | Browser publication-state qualification remains. |
| 302 AX-70 | E – | KEEP → AX | Release journeys, fixture hashes and unresolved browser failures. |
| 312 LN-REC | L M | NARROW | Recount captions/timings against the dated record set. Alpha.16 provides **some** primitive shader facts. Local adoption, uncovered primitives and domain annotations are separate. `D/upstream-workarounds.md:34`. |
| 313 LN-SEL | L M | KEEP | Named `gpu-crossfilter-selection-backend.ts` resolves. Preserve external-mask parity. |
| 314 LN-DQ | L M | KEEP | Link query scenarios to upstream DF-DESC. |
| 315 LN-STAB | L L | KEEP → deferred | Product proposal for stability/purpose views. |
| 316 LN-V5 | L L | KEEP → deferred | Product proposal for archive visualization. |
| 317 LN-ENC | U L | KEEP → U/L | Rendering/trace packs depend on upstream labels. |
| 318 LN-ST | L M | NARROW | Recount the per-entry exception assignments. The slice counted 89 assignments against the dated 85. `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts:156`. |
| 319 INT-01 / LN-REC2 | L M | CLOSE | All 427 scenarios were recaptured on alpha.16 per implementation record `:167–173`. Reopen for a named later mismatch. |
| 320 INT-06 | L L | CLOSE | Local-permutation replay imports from the public spatial entry at `local-permutation-test.lens.ts:29–38`. |
| 321 CPU-03 | A L | NARROW → research | Preserve the dated survey. Promote a twin for a named consumer; triage existing leads separately. `D/research/cpu-suitability-2026-11-11/README.md:23`. |
| 322 CPU-04 | A M | NARROW | Population alignment forms ship. Counterfactual/tessellation mixed forms remain. Freeze dense resistance pending a named migration. `landscape-ecology/index.ts:47`, `resistance-connectivity.ts:52`. |
| 323 CPU-05 / INT-03 | A M | KEEP | Catalog twin metadata and class-based operation discovery remain. `website/scripts/extract-inventory.ts:410` uses filename form. |
| 324 CPU-08 | A L | KEEP | Preserve UPS-CPU-1 dependency for an automatic choice. Named current crossover measurements can proceed independently. |
| 325 UPS-CPU-1 | U H | NARROW | Installed fixed-width transfer loops copy rows. Keep alpha.12 timing dated; remeasure alpha.16 with stride/offset/chunk semantics. Draft 01 `:35`. |
| 326 UPS-CPU-2 | U M | KEEP | Installed `gpu-cpu-execution.ts:37` rejects textures/state vectors. |
| 327 UPS-CPU-3 | U L | KEEP | Installed `gpu-cpu-execution.ts:58–88` builds a graph per GPU operation. |
| 328 UPS-CPU-4 | U M | KEEP | Local GPU contributor bridge remains; installed mixed executor requires its operation contract. `gpu-contributor-operation.ts:216`. |
| 334 WEB-INT-08 | W M | KEEP → pictures | Touch steering, `website/src/gpu/canvas-pointer.ts` exists. |
| 335 WEB-INT-09 | W M | KEEP → pictures | Shared-device latency and device-loss experiment. |
| 336 WEB-INT-13 | W L | KEEP → pictures | `define-run.ts` and `gpu-canvas.tsx` resolve. |
| 337 WEB-INT-14 | W M | KEEP → pictures | Keyboard path; pointer file resolves. |
| 338 WEB-INT-15 | W L | KEEP → pictures | Pointer-to-frame measurement; verifier resolves. |
| 339 WEB-INT-16 / WEB-COV-04 | W M | KEEP → pictures/D | Preserve label support, scale checks and separate font decision. |
| 340 WEB-INT-17 | W L | KEEP → pictures | Shared highlight option; runs directory resolves. |
| 341 WEB-INT-18 | W L | KEEP → pictures | `record-demo.ts` resolves. |
| 342 WEB-PIC-09 | W M | KEEP → pictures | Real asset provenance/licensing; `website/build/demo-data-sources.ts` resolves. |
| 343 WEB-PIC-12 | W H | NARROW → pictures | Rescore both themes and gestures. Keep suspected computation problems as separately verified family leads. |
| 344 WEB-PIC-13 | W M | KEEP → pictures | Real-input and spike-repair investigation. |
| 345 WEB-PIC-14 | W M | KEEP → pictures | Whole-group/light-theme/poster acceptance. |
| 346 WEB-PIC-15 | W L | KEEP → pictures | Distinguish sibling layouts. |
| 347 WEB-PIC-16 | W L | KEEP → pictures | Improve named weak pictures. |
| 348 WEB-PIC-17 | W M | KEEP → pictures | Verify input-domain explanation before alleging a computation defect. |
| 349 WEB-PIC-18 | W L | KEEP → pictures | Decide real versus constructed scenario suitability. |
| 350 WEB-COV-01 | W M | KEEP → pictures | Local panel helpers; entries directory exists. |
| 351 WEB-COV-02 | W M | KEEP → pictures | Segment-value rendering. |
| 352 WEB-COV-03 | W L | KEEP → pictures | Shared-helper consolidation; `_lib/` resolves. |
| 353 LUM-27 | W M | KEEP → pictures | Display reshaping and device-loss qualification. Treat 85 paths as dated. |
| 354 LUM-34 | W L | KEEP → W/D | Choose capture settlement/failure policy. |
| 355 LUM-35 | W M | KEEP → W/K | Batch readbacks; re-count dated 48/six sites before implementation. |
| 361 WEB-HIW-01 | W M | KEEP → explanation | Distinguish captured timings, public metadata and physical allocations. |
| 362 WEB-HIW-02 | W M | KEEP → explanation | Raster view-model placement; extractor resolves. Recount old fractions before use. |
| 363 WEB-HIW-03 | W M | KEEP → explanation | Use actual chain outputs. Workflows page directory resolves. |
| 364 WEB-HIW-04 | W L | KEEP → explanation | Primitive attribution and usage-count semantics. |
| 365 WEB-HIW-05 | W L | KEEP → explanation | Phone text-size/device acceptance. |
| 366 WEB-HIW-06 | W L | KEEP → explanation | Live-card product decision. |
| 372 STX-08 | A L | KEEP | REDCAP’s direct cut and SKATER’s SSD/floor differ. `gpu-redcap-single-linkage.ts:146`, `gpu-skater-regions.ts:205`. |
| 373 STM-07 | A M | KEEP | External GpGp comparison remains. Separate Fisher fit/stall/budget tests from dated nugget evidence. `vecchia-fisher-fit.ts:144,178`. |
| 374 MOB-03 / FX-08 | A H | UP → D | Choose the default; point-centred implementation ships. TD-TR word-time conversion is a separate input constraint. `track-quality-recipe.ts:45`. |
| 375 MOB-07 | A L | KEEP | dBBMM change-point step remains outside `brownian-bridge-variance-estimation.ts:294`. |
| 376 MOB-10 | A L | NARROW | Both reference routes cap zones at 1,024. Smoothing implements a product-kernel weighted mean. Identify the paper estimator before claiming parity. `gpu-radiation-flow-model.ts:42`, `gpu-kernel-flow-smoothing.ts:134`. |
| 377 RTS-08 | A L | KEEP | A5 geometry decoding ships; point-to-cell, aggregation and topology remain separate. HEALPix JSON identifiers require mapping. `gpu-cell-geometry.ts:49`, `healpix-dggs-json.ts:85`. |
| 378 PLAT-03 | A M | KEEP | Share only traversal with matching eligibility, distance, tie and overflow contracts. Builder adoption does not establish this. `gpu-kernel-density.ts:889`. |
| 379 PLAT-04 | U M | KEEP → U | Host-controlled batches ship. Generic reusable GPU-gated iteration remains upstream. `network-centrality-continuation.ts:66`, `D/upstream-proposals.md:331`. |
| 380 PLAT-05 | U L | NARROW | Numeric required limits and subgroup reduction selection exist. Track limits-only adapter-max policy, qualified subgroup paths and portable `buffer_view` fallback independently. `D/upstream-proposals.md:341`. |
| 381 PLAT-06 | A M | KEEP | List external generator/version/spec and distinguish definitions from actual external-library output. `modules/analysis/test/oracles/GENERATORS.md:51,71,81`. |
| 382 PLAT-07 | A L | NARROW | SF100 has 600M trips: 4.8 GB float32x2 or 9.6 GB float64x2 position payload before scratch. Packed benchmark skips over-binding cases. Qualify chunked/real-zone/KNN separately. `spatialbench-scale-factors.ts:28`, `spatialbench-scale-benchmark.spec.ts:34`. |
| 388 STD-04 | A H | NARROW | Depression hierarchy, OPTICS and network KDE have `output.ports.status`. Retain compatibility choices and named missing diagnostics. KDE legacy status is per-query. `gpu-terrain-depression-hierarchy.ts:91`, `gpu-optics.ts:181`, `gpu-network-kernel-density.ts:53`. |
| 389 STD-05 | A M | NARROW | Neighborhood radius/support, grid/height IDW settings and surface ground class have runtime routes. Inventory remaining props plus shape exceptions. `gpu-pointcloud-neighborhoods.ts:108`, `gpu-pointcloud-height.ts:110`, `gpu-pointcloud-surfaces.ts:151`. |
| 390 STD-06 | A H | KEEP | Shared context exists. Close only with explicit affine/row-direction behavior for each remaining consumer. `analysis-core/contracts/raster-grid.ts:75`. Novel shear results remain with terrain lead. |
| 391 STD-07 | A H | KEEP | Published band validity, output validity, outlier flags and label-bearing seed masks have different meanings. `gpu-raster-cell-statistics.ts:82`, `raster-spectral-index.ts:56`, `gpu-distance-field.ts:92`. |
| 392 STD-08 | A M | KEEP | Compact layouts ship; representative retained-boundary provenance does not give full source membership. `gpu-offset-curve.ts:181`, `gpu-outline-geometry.ts:154`, `gpu-polygon-overlay.ts:146`. |
| 393 STD-09 | A L | NARROW | `validateSinglePackedView` is public. Move the raster gate or family helpers only for a second compatible consumer. `analysis-core/index.ts:46`, `raster-relaxation.ts:80`. |
| 394 STD-10 | A M | KEEP | Keep shared-solver/search work by named call site. The ~140 count is historical. `panel-lag-tiled-nodes.ts:318`, `binary-search-wgsl.ts:11`. |
| 395 STD-11 | A M | KEEP | Preserve independent overload and optional-stage result checks. `segregation-counterfactual.ts:253`, `D/contributor-contract.md:42`. |
| 396 STD-12 | A L | NARROW | LR-17 owns remaining 21 root files. Geospatial source has datum-grid, point-buffer and viewport families. Name remaining class/test/API moves; canonical guide migration has landed. `geospatial-analysis/index.ts:5`. |
| 397 STD-13 | A M | NARROW | Retain 14 `expansion-`/`coverage-` capability aliases and synchronization work. Dorling already distinguishes indexed/all-pairs envelopes. Expand `ontology/all-entry-points/` to `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/`. `kepler-analysis-catalog.ts:51`. |
| 398 STD-14 | A M | KEEP | Record packed-only ports and scratch by route. Point-in-polygon accepts chunks; KDE chunking is method-dependent. `gpu-point-in-polygon-join.ts:171`, `gpu-kernel-density.ts:279`. |
| 399 STD-15 | A L | KEEP → D then A | Decide statistic count/minimum vocabulary before coordinated change. `gpu-raster-cell-statistics.ts:48`, `gpu-raster-zonal-statistics.ts:149`. |
| 400 GPU-01 | A M | KEEP → audit | Parallel inequality route exists. Require evidence per remaining loop and route. `inequality-parallel-statistics.ts:1`. |
| 401 GPU-05 | A H | KEEP | Large-radius work remains. Candidate capacity exists in snapping and current callers forward it. Preserve local point-cloud grid contract. `gpu-network-snapping.ts:166`, `drive-time-catchment-recipe.ts:644`. |
| 402 GPU-09 | A M | KEEP | Distinguish dense panel determinant, tiled means and serial diagnostic loops. `panel-lag-tiled-nodes.ts:39`, regression inventory `:81`. |
| 403 GPU-10 | A M | KEEP | Scalable point routing and bounded decomposition ship. Retain facility consolidation and time-routing scale. `gpu-network-point-routing.ts:117`, `gpu-network-flow.ts:616`. |
| 404 GPU-11 | A M | KEEP | Local holed tessellation and budgeted network CSR metrics exist. Larger capacities/dissolved route need separate evidence. `building-tessellation-local.ts:54`, `visibility-graph.ts:90`. |
| 405 GPU-12 | A M | KEEP → domain backlog | Preserve partial-result semantics for S2 and distinguish constrained/unconstrained flips. `gpu-s2-polygon-covering.ts:124`. Remaining algorithms need separate acceptance. |
| 406 GPU-13 | A M | KEEP | Fixed-linkage Borůvka ships. Class breaks and filtered quantiles retain `2^24`; rational unfiltered column quantiles admit `2^31−1`. `gpu-class-breaks.ts:324`, `gpu-column-quantiles.ts:198`, `gpu-raster-region-graph.ts:85`. |
| 407 DUP-01 | A M | KEEP → design | Shared cross-variogram kernels and interpolation helpers ship. Establish duplicate, distance and exact-hit compatibility before combining kriging APIs. November 9 review `README.md:93`. |
| 408 DUP-03 | A M | KEEP | S2 compaction, covering, outline and aggregation have distinct support. `cell-keys.ts:13`, `gpu-cell-cover.ts:324`, `gpu-cell-set-outline.ts:106`. |
| 409 UPS-01 | U M | KEEP → draft 03 | Published graph algorithms take physical topology/vectors; caller views, costs and masks remain contract work. `D/upstream-proposals.md:133`. |
| 410 UPS-02 | U L | KEEP | Offer eligible generic pieces individually. Existing upstream algorithms do not establish equivalent local contracts. `D/upstream-proposals.md:137`. |
| 411 UPS-03 | A M | KEEP | Record the per-operator boundary, halo and global-state requirements. `D/upstream-workarounds.md:245`. |
| 412 GAP-01 | A M | KEEP → deferred | Preserve shipped line splitting, dissolve and points-along-lines. Voronoi edges differ from complete polygons. `gpu-voronoi-diagram.ts:60,108`. |
| 413 GAP-02 | A M | NARROW | Strahler and bounded breach **from caller-supplied pits** exist. Define automatic pit discovery, larger search or out-of-core processing separately. `gpu-terrain-stream-order.ts:137`, `gpu-terrain-breach.ts:56,80`. |
| 414 GAP-03 | A M | KEEP → deferred | Name a consumer, measures and expected outputs for edge demand, nearest POIs or street statistics. `gpu-network/index.ts:1`. |
| 415 GAP-04 | A L | KEEP → deferred | Existing masked-row prediction and time filtering do not close new-support regression, general windows or as-of joins. `time-window-filter/index.ts:5`. |
| 416 GAP-05 | A L | KEEP → deferred | Voxel merge takes supplied centroids; neighborhood bound remains 64. `gpu-pointcloud-voxel-merge.ts:70`, `gpu-pointcloud-neighborhoods.ts:122`. |
| 417 LUM-21 | A M | KEEP → deferred | Use measured dependency-order cost per kernel. Counts and 5–17 ms are dated. `research/luma-usage-review-2026-11-11/shaders-core.md:127,152`. |
| 418 LUM-28 | K L | KEEP → K/L experiment | No measured `createExecution` benefit established. Preserve as host experiment. |
| 419 LUM-30 | A L | KEEP | Extend the existing active-buffer snapshot to other owning classes, repeated execution and textures. `gpu-analysis-spacing.spec.ts:155,220`. |
| 425 FP64-02 / LUM-20 | A M | NARROW | The ~3× result was a chained arithmetic probe (380 vs 132 ms) on Apple Metal. Time the actual point-to-cell route before replacing it. `research/extended-precision-2026-11-11/README.md:23`. |
| 426 FP64-03 | A M | KEEP | Cell measures expose optional fp64. Other S2/geodesic/containment routes require their own qualification. `gpu-cell-measures.ts:122,154`. |
| 427 FP64-05 | A M | KEEP | Obtain a comparable non-Apple measurement. Precision study `README.md:40` excludes other devices. |
| 441 LRN-03 | A M | KEEP → research | No public fitted neural field found; coordinate/fitting primitives at `analysis-core/index.ts:60`. |
| 442 LRN-04 | A M | KEEP → research | No Gaussian-mixture contributor found; public forest inference is a different model. `dataframe-analysis/index.ts:19`. |
| 443 LRN-05 | A M | KEEP → research | No SOM/GeoSOM contributor found. Closest embedding clustering export is `raster-embeddings/index.ts:15`. |
| 444 LRN-06 | A M | KEEP → research | Forest prediction exists; no permutation-importance output found. `forest-inference/index.ts:5`. |
| 445 LRN-07 | A L | KEEP → research | Existing forest inference does not train boosted trees. Same public barrel. |
| 446 LRN-08 | A L | KEEP → research | Forest prediction lacks proposed SHAP attribution. Same public barrel. |
| 447 LRN-09 | A L | KEEP → research | Conformal operations exist. Require a fitted-network ensemble producer and consumer. `gpu-spatial-analysis/spatial-validation/`. |
| 448 EMB-01 | A M | NARROW → research | Existing codecs include AlphaEarth int8, TESSERA int8 and float16. Presto uint16 needs a verified scaling formula. `gpu-raster-embedding-dequantize.ts:59,126`. |
| 449 EMB-03 | A M | KEEP → research | Pairwise year change exists; multi-year trajectory outputs remain distinct. `gpu-raster-embedding-change.ts:65`. |
| 450 EMB-04 | A M | KEEP → research | Covariance and nearest-pixel operations are potential inputs. No proposed anomaly output found. `raster-embeddings/index.ts:13,18`. |
| 451 EMB-05 | A M | KEEP → research | Linear probe produces labels/probabilities; mean-reference similarity and regression output remain. `gpu-raster-embedding-linear-probe.ts:45`, `gpu-raster-embedding-similarity.ts:110`. |
| 452 EMB-06 | A L | KEEP → research | Define the spatial-neighbor embedding edge product against existing texture/change operations. `D/coverage-matrix.md:75`. |
| 453 SIM-02 | A M | KEEP → research | Random-field primitive ships; no proposed terrain ensemble recipe found. `spatial-interpolation/index.ts:175`. |
| 454 SIM-03 | A M | KEEP → research | Weighted overlay is a consumer candidate. No Sobol/Morris contributor found. `raster-algebra/gpu-weighted-overlay.ts`. |
| 455 SIM-04 | A M | KEEP → ownership decision | EOF uses a host Jacobi step. Decide generic SVD ownership. `raster-snapshot-eof-decomposition.ts:46`. |
| 456 SIM-05 | A M | KEEP → research | GWR neighbor infrastructure exists; no local-PCA contributor found. `gpu-geographically-weighted-regression.ts:1116`. |
| 457 SIM-06 | A M | NARROW → research | NMF adds learned factorization/endmembers. Supplied-endmember unmixing already exists. Link the remaining RS10 clause. `D/coverage-matrix.md:82`. |
| 458 SIM-07 | A L | KEEP → research | Interaction balancing is related; multi-zone integerized raking remains distinct. `spatial-interaction/interaction-balancing-wgsl.ts`. |
| 459 SIM-08 | A L | KEEP → research | Distribution-dynamics Markov classes do not provide a track HMM/Viterbi result. |
| 460 SIM-09 | A L | KEEP → research | Random point generation exists; no named location-obfuscation operation found. `D/coverage-matrix.md:166`. |
| 461 SIM-10 | A M | KEEP → research | Indexed local kriging exists; proposed FFT structured-kernel mean solver remains distinct. `gpu-space-time-kriging.ts:139`. |
| 462 CART-01 | A M | KEEP → research | Dorling circle relaxation differs from density-equalizing flow cartograms. `gpu-dorling-cartogram.ts:107`. |
| 463 CART-02 | A M | KEEP → research | No point-label placement export found; representative points alone do not resolve label conflicts. |
| 464 CART-04 | A M | KEEP → research | Sinkhorn divergence exists; no displacement-interpolation product found under `gpu-spatial-analysis/optimal-transport/`. |
| 465 CART-05 | A M | KEEP → research | Existing clustering does not establish the proposed multizoom hierarchy or blue-noise sample. Require algorithm design review. |
| 466 CART-06 | A M | KEEP → research | Accessibility and isochrones are inputs to the proposed deformation. `D/coverage-matrix.md:181`. |
| 467 CART-07 | A M | NARROW → research | Value-by-alpha already ships. Check it against the uncertainty consumer; split contour, exceedance, vision, filtering, density and pansharpening requests. `gpu-bivariate-classification.ts:90,104`. |
| 468 CART-08 | A L | KEEP → research | Existing embedding projection is distinct from field t-SNE. `raster-embeddings/index.ts:20`. |
| 469 CART-09 | A L | KEEP → research | A line graph does not establish road-stroke thinning. `D/coverage-matrix.md:179`. |
| 470 ENC-01 | A M | KEEP → research/D | Coordinate encoder primitive exists. SatCLIP model, weight extraction and usage decision remain. `analysis-core/index.ts:65`. |
| 471 ENC-02 | A L | KEEP → research/D | No public transformer/Galileo contributor found. Decide generic ownership first. |
| 472 ENC-03 | A L | KEEP → research/D | Presto model execution and weight rights remain distinct from dequantization. `gpu-raster-embedding-dequantize.ts:59`. |
| 473 ENC-04 | E L | UP → examples experiment | Name an owner and GPU tensor-sharing acceptance condition before considering an optional package. |
| 479 FX-03 / FX-05 | D d | KEEP | Choose sporadic default and negative-General-G handling per route. Global NaN, permutation flags and analytic Gi* differ. `gpu-global-spatial-statistics.ts:231,280`. |
| 480 CUBE-OCCUPANCY | D d | KEEP | Decide treatment/mask for never-occupied cells. Current dense evaluation: `gpu-emerging-hot-spots.ts:618`. External parity unverified here. |
| 481 DEC-HDBSCAN | D d | KEEP | Maintain 32 or raise with private-storage audit and rejection changes. `gpu-hierarchical-density-clustering.ts:95,293`. |
| 482 STD-04 / STD-09 | D d | NARROW | Decide retention/renaming of existing packed depression/OPTICS views and network KDE per-query status. Drop the validator publication question. See ROAD-104. |
| 483 STD-07 / STD-15 | D d | KEEP | Give the vocabulary decision a distinct ID. Link implementation IDs and preserve validity polarity and labels. `analysis-core/contracts/status.ts:49`. |
| 484 DUP-02 | D d | KEEP | `GPUSpatialKMeans` is exported. Decide naming/deprecation only. `spatial-clustering/index.ts:25`. |
| 485 raster-fields-05 / STD-06 | D d | NARROW | Decide affine-following placement and compatibility. Refresh the 2026-11-11 count of 108 sites before changing callers. `row-direction-flip-plan.md:34`. |
| 486 DEC-WALK-ROUNDING | D d | KEEP | Decide whether the dated one-ULP GRASS-route difference is acceptable. Runtime parameters ship. `gpu-raster-walking-cost.ts:74,189`. |
| 487 DEC-VARIANCE | D d | UP → accepted limit if approved | Retain documented cell/zonal divisors unless a concrete interoperability requirement demands alignment. `gpu-raster-cell-statistics.ts:108`, `gpu-raster-zonal-statistics.ts:388` at the reviewed revision. |
| 488 DEC-ARROW-CONSUMERS | D d | KEEP | Decide which parked paths to restore using the current exclusion manifest and registered copy. `D/upstream-workarounds.md:205`. |
| 489 WEB-INT-16 | D d | KEEP → website licence decision | Modified font and unresolved interpretation at `THIRD-PARTY-DATA.md:48`. Give this decision a distinct ID. |
| 490 LUM-09 | D H | KEEP | Decide iteration-count comparability before published labeling adoption. `gpu-raster-reconstruction.ts:73,295`, `D/upstream-workarounds.md:289`. |
| 491 DEC-LAG | D d | KEEP | CPU retains a finite rejected iterate; GPU clears fit table/summary. Decide diagnostic exposure. `cpu-spatial-lag-maximum-likelihood-operation.ts:200`, `gpu-spatial-lag-maximum-likelihood.ts:1540`. |
| 492 dataframe-10 | D d | KEEP | Separate K tie-rule compatibility from reference provenance/licensing. `modules/analysis/test/oracles/space-time-diggle-splancs-port.py:8`. |
| 493 PLAT-04 / PLAT-05 | D d | NARROW | Choose filing owners for drafts/compiler reports. Continuation and device-limit work already have IDs. Compensated-sum guidance exists in `D/contributor-contract.md:120`. |
| 494 LRN / EMB / SIM / CART / ENC | D d | KEEP | Choose scope only for a consumer, including SVD/transformer ownership and model restrictions. Fitting foundations already ship at `analysis-core/index.ts:60`. |
| 500 Lag ML | B – | NARROW → limits/evidence | Dated fixture measured ~33× inflation; current test bounds it below 45×. State fixture and status. `math-review-lag-conditioning.spec.ts:781–805`. |
| 501 Panel random effects | B – | NARROW → limits/evidence | Reference route measured 1.1% at the dated near-boundary fixture; test permits 3.5% and separately checks boundary status. `math-review-panel-lag-conditioning.spec.ts:345,404`. |
| 502 Probit | B – | KEEP → limits | Positive finite tolerance has no upper guard; distinguish stopping status from exact separation. `gpu-probit-regression.ts:185,318`. |
| 503 GPU triangulation | B – | KEEP → limits/evidence | Historical Earcut-only failure, default monotone retry, actual rescue unverified on device. `gpu-polygon-triangulation-search.spec.ts:273`. |
| 504 Space-time K / Diggle | B – | NARROW → API/evidence | Preserve tie/cap/reference-port distinctions. Mark fixed-scale intensity loss as committed-revision evidence pending qualification of new scaling. `space-time-diggle-splancs.spec.ts:5,153`, `space-time-k-function-wgsl.ts:6`. |
| 505 COLD | B – | NARROW → limits | Record refit/window, LASSO, Tmask and partial tail differences from pyxccd. `rts04-cold-pyxccd.spec.ts:211–265`. |
| 506 Sky view | B – | NARROW → limits/evidence | Retain 0.11/0.57 as dated observations for the stated march/sweep, directions and edges. Matched sampling has its own comparison. `gpu-terrain-horizon-rvt.spec.ts:378`. |
| 507 Panel error ML | B – | CLOSE → research/domain | Review coverage belongs in dated research. Link actual diagnostic findings to the statistics backlog; retain the likelihood convention in the API. `gpu-spatial-panel-error-maximum-likelihood.ts:339`. |
| 508 Triangulation twins | B – | KEEP → CPU/API | Current edit correctly distinguishes budgeted Earcut, monotone route and exact/float64 CPU predicates. Document constrained twin flip-budget exception. `cpu-polygon-triangulation-operation.ts:628`, `cpu-constrained-delaunay-triangulation-operation.ts:38`. |
| 509 CPU arithmetic | B – | KEEP → CPU/API | State per-operation effects on threshold/tie decisions. `cpu-network-path-extraction-operation.ts:73`. |
| 510 CPU budgets | B – | NARROW → CPU/API | Some GPU capacity limits are omitted; explicit `maximumWork` and `maximumPathLength` can bind both forms. `cpu-polygon-triangulation-operation.ts:628`, `cpu-network-path-extraction-operation.ts:69`. |
| 511 CPU output tails | B – | NARROW → CPU/API | Path node tails are zero on CPU, unwritten on GPU. CPU edge padding is `0xffffffff` for nonempty target tables, zero for empty. Consume only the prefix. `cpu-network-path-extraction-operation.ts:76`. |
| 512 IN01 | B – | KEEP → research/API | Preserve residency rationale. Existing 1.3/3.2 versus 4.3/51 ms predate the current fitting route. `interpolation-performance-README.md:88`. |
| 513 Space-time variogram | B – | CLOSE → API | Required grid size and eight parameters are an input contract. Document the deprecated cross-variogram default separately. `gpu-space-time-variogram.ts:84,132`. |
| 514 Temporal reduction | B – | NARROW → limits | Logical slot ceiling `0xfffffffe`. Word time keys require 8 bytes per slot, hence ≤16,777,216 at a 128 MiB binding before other constraints. `gpu-temporal-reduction.ts:215,401`. |
| 515 Trajectory restriction | B – | KEEP → API | Invalid segments, overlapping gap pairs and nonfinite arithmetic events contribute different units. Working wording corrected. `trajectory-restriction-pair-merge-kernels.ts:189,218`. |
| 516 Space-time kriging | B – | KEEP → limits | Auto switch at 4,096, explicit scan cap 1,048,576, indexed bindings device-bounded. `gpu-space-time-kriging.ts:133,385`. |
| 517 OPTICS xi | B – | KEEP → limits | Xi extraction 65,536, grid route 2²⁰. `gpu-optics.ts:66,421`. |
| 518 Urban tessellation | B – | KEEP → limits | Separate local enclosure/radius certification from dissolved reference; include candidate loss. `building-tessellation-local.ts:44,90`. |
| 519 Visibility graph | B – | KEEP → limits | Spatial CSR gives adjacency/degree. Network metrics use all-source or sampled pivots with budget/status. `visibility-graph-csr.ts:86`, `gpu-network/visibility-graph/visibility-graph.ts:90`. |
| 520 Network flow | B – | NARROW → limits | Max flow uses preflow push-relabel. Costed flow also selects primal-dual when `maximumCostPhases` is supplied without mode. Preserve partial-result status. `gpu-network-flow.ts:120`, `network-flow-parallel.ts:63`. |
| 521 Chunked input | B – | NARROW → limits | Each route needs maximum binding, bytes/row and preflight. Replace unmeasured “most” with the named routes that still retain row-proportional scratch. `device-limits.ts:89`, STD-14. |
| 522 FP64-04 | B – | CLOSE → workaround register | Accepted local rounding-error helpers remain used. `D/upstream-workarounds.md:234,308`, `exact-geometry-wgsl.ts:5`. |
| 523 Published fp64 | B – | KEEP → limits | Approximate double-single precision with float32 exponent range. Sampled accuracy is not a uniform 48-bit guarantee. `shader-fp64-module.spec.ts:154`. |

**Four committed rows no longer present**, completing the mapping from the original 284: NW02 (committed `:119`) **CLOSE** into LR-25; GEOM-01 (`:161`) **CLOSE**, optional polygon hierarchy from `gpu-rectangle-clip.ts:146,365`; GEOM-02 (`:162`) **CLOSE**, source rows/chainage from `gpu-line-smooth.ts:121`, `line-smooth-metadata.ts:101`; LUM-23 (`:163`) **CLOSE**, segmented CDF ordering and scan at `dot-density-triangle-cdf.ts:22,114`. The working version adds ten KI rows and one KI index reference. Thus `284 − 4 + 11 = 291`.

**Structural edits accompanying the mapping.**

- Exact two-column summary repeats are `WEB-PIC-12`, `KE-TESTS`, `UPS-CPU-1`, `GPU-05`, `LUM-09`. These are index references. The eight implementation/decision collisions after splitting compound IDs are `WEB-INT-16`, `PLAT-04`, `PLAT-05`, `STD-04`, `STD-06`, `STD-07`, `STD-09`, `STD-15`. Retain existing implementation IDs and map distinct decision IDs back to them. The earlier PLAT-03, STD-05 and KE-TESTS decision collisions were fixed.
- The reproducible ≥50-word cell list is the KI summary, LR-08/10/19/20/22/26, KI-01/02/06, IN05 / IN02, KE-PORT-SAMPLES, STX-08, PLAT-03/04/06, STD-04/06/07/11/14, GPU-05/10/11/12/13, DUP-01/03, Space-time K / Diggle, Urban tessellation and Visibility graph. Preserve the compact contract or acceptance sentence; transfer run chronology and measurements by link.
- Rows especially lacking a falsifiable finish condition include INT-07, NW-GA, R6.4, EC04, the grouped GE/UR capacity request, GPU-01’s audit list and GAP-03/04/05. The 33 options have proposals and some oracle names, but no assigned consumer per row. Keep them out of the “now” list.
- Move `roadmap.md:202,216–228` to port history, `:234–245` to workflow design/decisions, `:267–288` to AX architecture/history and `:304–306` to dated journey evidence. Preserve the provisional canvas and host-persistence decisions, data/font licensing and actual browser failures. Reduce `:529–550` to one line per pass. Attach the historical migrations at `:535` to the changelog, updating the BL-PASS text reader at the same time.
- Resolve source shorthand by its declared base. `ontology/all-entry-points/` is wrong under the roadmap’s `modules/analysis/src/` omission rule. In contrast, `gpu-layers/`, `scripts/smoke.mjs` and `scripts/kepler-demo-data.mjs` resolve under the Kepler example. The showcase `collect-page-diagnostics.mjs` import really is absent. A global basename search alone cannot distinguish these cases.

## Next steps

### Before the alpha

1. **Finish release qualification — M, release maintainer and family owners.** Select the candidate revision and checked tarball, resolve the canonical repository URL and release version, and explicitly include or defer each KI change. Record independent re-reviews for the previously unread rounds. For LR-22, qualify final costs and the documented count/budget contract on the candidate. Run the full local gate at the end of the iteration. Keep the dated separate-step passes visible as historical evidence.
2. **Finish caller-facing corrections — S/M, package callers, recording and numerical owners.** Correct duration-only timing, ownership, partial metadata availability, p-median bounds, CPU budgets/tails, status-path decisions and route-specific tile/slot limits. Pair each changed observable contract with a discriminating fixture. Put the family-specific upgrade mappings at the front of the changelog.
3. **Consolidate the actionable roadmap — M, maintainer and agent authors.** Apply the row mapping against the working hash, preserve the concurrent closures, and move consumer sections with their readers. Keep the initial “now” list to release selection/metadata, KI-03 re-review, KI-04 qualification and remaining demonstrated caller-impacting defects from family reports. Link the full authoring path and complete the review index. The roadmap split need not delay publication once the release-critical facts and decisions are accurate.

### After

1. **Connect documentation to evidence — M.** Give status/limit records and executable snippets stable identities. Start with the workaround table parser, the four roadmap-reading specs and positional draft fences. Use generated views where an existing parser already treats prose as data; introduce a broader generator only if hand maintenance continues to fail.
2. **Consolidate limits and scale evidence — M.** API users need route, input layout, per-binding arithmetic, exhaustion behavior, validating code and oracle. Start with network optimization, regression tiles, temporal reduction and chunked scratch. Date device measurements and commission second-device measurements only for actual candidate changes.
3. **Freeze speculative expansion and retire history selectively — S per decision, M per move.** Require a named consumer and observable acceptance case before promoting an option. Keep showcase inputs, AX protected baselines, generated Kepler matrix, compiler rationale and numerical probes with their consumers. Retire only coordination prose whose decisions and dependencies have been transferred.

## Limits of this report

The ledger adjudicates planning and documentation, not numerical correctness of every contributor it names. The 33 option searches establish absence of proposed public names and selected equivalents. Entire WGSL families, consumer implementations, staged branch contents, raw research datasets, independent statistical libraries, copyright ownership, external filing/publication and deployed URLs were not comprehensively inspected. Family reports retain responsibility for their W/S grades. Staged claims lacking independent evidence on the selected release revision remain **UNVERIFIED**.

Read-only CPU and null-device probes establish host behavior and constructor admission. They do not establish allocation feasibility, GPU output, performance or backend frequency. A source/spec citation establishes what the assertion would discriminate; it does not establish that it ran on the parent or passed today. No current full-gate or release certification follows from this report.