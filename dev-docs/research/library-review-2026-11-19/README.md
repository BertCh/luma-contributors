# Library review: state after the fixes, new defects, and what was corrected the same day (2026-11-19)

A second review of `@luma-contributors/analysis` (`modules/analysis`: 1,772 source files, 565,068 lines, 2,498 specs, 498 public command-node producers, 761 public classes), with consumer seams in `modules/lenses`, `modules/kepler`, `examples/` and `website/` sampled. It reviewed commit `bd8279a4` plus the uncommitted working tree of 2026-11-19: 368 modified files (+10,847 / -1,804 lines, `git diff --shortstat`) and 73 untracked files (17 are the files of this folder). The tree holds the same-day fixes, roadmap and documentation edits, and the review files; nothing is committed.

The previous review is [library-review-2026-11-17](../library-review-2026-11-17/README.md). This one asks whether the fixes made after it held, what first reads of the same code still find, and what changed in gates, package and documents. The plan is in [../../roadmap.md](../../roadmap.md); this document is evidence and state, and ends with the decisions and open items the plan depends on.

## Summary

1. **The 2026-11-17 fixes held where the reports could check them.** 231 follow-through rows: 188 fixed (some for a stated scope), 35 partly, 8 limits or open. Fixed / partly / limit or open per report: statistics 13 / 4 / 0; geometry 11 / 1 / 1; points 14 / 2 / 1; cells, trajectories, recipes 14 / 1 / 1; network 10 / 0 / 1; raster and terrain 13 / 2 / 0; dataframe, point cloud, record 18 / 0 / 2; core 12 / 3 / 0; API, docs, catalog 25 / 9 / 0 (report text: 24 / 10); tooling 14 / 4 / 0; luma.gl and scale 32 / 2 / 1; seams 6 / 3 / 1; roadmap and developer documents 6 / 4 / 0. "Fixed" means a pinning assertion was read, not rerun. The fixes left their classes alive elsewhere (the unscaled circumcentre in `GPUMinimumBounds`, unwrapped longitude in geographic group bounds, optional candidate-loss flags at join producers).
2. **First reads still found wrong results in every family.** 118 W or S findings (95 wrong result, 23 visible failure) in 12 of 13 area reports. Reach as graded by the reports: 23 ORDINARY, 83 UNUSUAL, 12 EXTREME. W / S per report: statistics 10 / 1, geometry 9 / 1, points 9 / 3, cells and trajectories 11 / 5, network 10 / 1, raster and terrain 10 / 4, dataframe and record 14 / 1, core 7 / 2, API and catalog 1 / 2, tooling 1 / 2, luma.gl 3 / 1, seams 10 / 0. 23 are leads or GPU-unverified by their own report's label. The reports also list 142 contract, 50 test and 33 maintainability findings.
3. **Seven classes explain most of them** (grouping and counts in "What the defects have in common"): completeness that does not travel (17: weights loss in about 18 consumers, cost matrices, AZP, zone assignment); float32 range and cancellation (28: zonal variance 123 where the truth is 2 at 1e7, a grid writer and reader that disagree at a projected origin, panel LM sums +15 percent at 2^24 rows); silent loss behind an omitted status view (8: trajectory thinning, staypoints, entropy); packers that validate a double and write float32 (4: clustering radius, isochrone extents); outputs without a writer (RT-122; KI counts 137 ports); tail probabilities (Knox Poisson tail returns 0); catalog and documentation claims (a catalog `status: []` on 99 records that publish one).
4. **Focused agents fixed or recorded 152 findings the same day; no independent reviewer has read the diff.** Verdicts in 32 `fix-*.md` records (session scratch folder): 123 fixed, 9 partly, 18 deferred (the known-issues staging pass owns the file), 2 waiting for a decision (DPC-110, SSR-112); none refuted whole. Narrower than reported: NET-102, URB-101, SSR-102, RT-102, DOC-102, CORE-114; NET-112 needed a different fix; TOOL-104 was made by the review (two junk files from a review agent's shell). Of the 118 W and S findings: 85 fixed (41 with a GPU spec file PASS in the scoped run, 36 node or host specs only, 4 device PASS per fix record only, 3 spec failed in the run and re-ran PASS per fix record, 1 mixed), 3 partly, 1 awaiting decision, 17 deferred, 12 open. By `git diff --stat`: 155 source files (+4,091 / -784), 121 test files (+4,820 / -633), 44 `docs/` and `dev-docs/` files (+408 / -228), including the lead's documentation edits.
5. **The known-issues pass (KI) recorded 116 defects and landed none.** It is an unlanded staging clone (`luma-contributors-ki`, base `9e93cbc7`, 65 streams). Landing analysis at 18:28: 0 streams with a clean review; 15 ready with unreviewed fixes, 28 not ready, 7 read-only census, 15 not started. Under a whole-stream rule 31 files qualify (stream oracle-network-twins, defects D079, D080); the lead did not land them. Fourteen defects found today lie in staging-owned files that neither tree fixes.
6. **Gates and package could pass without checking; the worst cases are closed in the working tree.** NodeNext resolution failed on 22 relative type references in 12 emitted declarations, including public parameter schemas, and the packed-consumer gate used a different schema (TOOL-102). All 22 are fixed in source (21 by the fix agent, the last by the lead in a staging-owned file) and a staged release build now resolves 4,069 of 4,069 relative references in the packed declarations, with ESM and CommonJS type consumers. NaN passed six numerical comparisons (TOOL-109) and nine cases returned green with no device (TOOL-110); both are closed. The kernel audit discarded resource-budget findings (CORE-114); run strictly over 3,103 cases, no registered kernel exceeds a budget (the 32,768 > 16,384 byte case was the reviewer's synthetic probe). Local gate cost: 90.8 minutes, reconstructed from separate runs.
7. **The catalog misdescribed status for 99 records, corrected today.** 555 records, none with CPU-twin linkage, 242 stating "no status" (six counterexamples were the finding); 6 records fixed by one agent and 93 by a second. The default for unaudited records is a decision.
8. **Scale: 90 of 498 producers are chunked or tiled, none pages from the host** (408 packed, 78 chunked, 12 tiled scratch, 0 host-paged). A valid two-chunk call of 33,554,432 rows is rejected for a buffer it never allocates (LUMA-104, deferred); two cleanup failures leak GPU resources (LUMA-116 deferred, LUMA-117 open).
9. **Documentation: 58 corrections applied (9, 25, 24 by batch).** 33 deferred to staging-owned pages, 33 not applied for a stated reason. 472 of 761 public classes appear in a class heading (289 do not: 32 producers, 250 operation forms, 7 others); 58 of 116 API pages have no TypeScript fence.
10. **The roadmap has 291 rows and 15,681 words** (284 and 13,425 at the start). The clerk log has 120 entries: 32 narrowed, 4 closed (INT-04, INT-06, INT-01 / LN-REC2, FP64-04 closed then restored), 15 sentences added, 27 not applied with a reason, 32 already correct, 10 other. The reconciliation spec passed 9 of 9.
11. **Direction.** Breadth is the strength; depth is uneven (130 of 633 GPU classes have a CPU twin, 370 of 548 catalog operators are `unaudited`) and there is no front door yet. Bets: five finished journeys with a facade, GeoArrow and a two-vendor benchmark; a stable core with tiers and three gate levels; a machine-readable contract. Stops: new operator families before external use, a generic runner, retrospective lens coverage.
12. **Outcome, 2026-11-20.** In a staging copy, six waves of implementer streams, each with one independent review and a fix round, carried these findings through, the known-issues pass was merged, and the reports show all 21 breaking proposals of the second round carried out, one of them differently. Every gate step then passed on the staged tree, run in chunks rather than as one uninterrupted `yarn verify` (see the roadmap's R19-01 row), all device evidence is still from one Apple Metal adapter, package identity is still the maintainer's, and the alpha stays held; see "Outcome (2026-11-20)".

## Outcome (2026-11-20)

The work was done in a staging copy on 2026-11-19 and 2026-11-20; the lead is gating and landing it as this is written. Six waves of Codex implementers (`gpt-6.1-sol`, then `gpt-6.2-sol`) worked the findings of this review, of the second round in [coherence/](coherence/README.md) and of the known-issues pass; each stream was followed by one independent reviewer (`gpt-6.2-astra`) and a fix round, and in wave 6 Claude coordinators ran each area's GPU specs on the device. The 21 breaking proposals and this README's open decisions were decided on 2026-11-20 by the lead under the maintainer's instruction to "make the right decisions" ([decisions-2026-11-20.md](decisions-2026-11-20.md)). Two decisions stay with the maintainer or with hardware. Every gate step passed on the staged tree before landing, but in chunks: one uninterrupted `yarn verify` on master is still owed, so the alpha stays held. Caller-visible changes accompany this README: old to new names in [changes/renames-before-the-alpha.md](changes/renames-before-the-alpha.md), and one record per family in [changes/](changes/) (statistics; geometry and points; cells, network, raster and dataframe; core, docs, tooling and consumers; the known-issues pass).

This section supersedes, without editing them: Summary point 4 and the "Fixes no independent reviewer has read" limit (each stream now had one independent review); Summary point 5, the section "The known-issues pass", decision 8 and the "KI pass" row of "What is still open" (the pass is finished and merged); and the decision rows below whose recommendation was taken.

### What was fixed

**Waves 1 to 5** tracked 1,975 items. Implementers' verdicts: 1,638 fixed, 240 partly, 49 deferred, 46 already fixed, 2 refuted. A read-only triage of 487 rows (the 289 partly or deferred items, the planner's open list, and 155 handoffs that no applier could place) found 338 done, 25 not defects, 12 upstream, 1 needing a decision and 111 open: 60 fixable from code, 51 needing a device run.

**Known-issues pass.** Finished (65 streams) and merged on 2026-11-20: 29 merge conflicts were resolved by hand, and an audit of 40 files that merged without a conflict found 16 wrong in combination; those were fixed in later waves.

**Second review round.** 32 reports (16 cross-family themes, 8 family-as-newcomer reviews, 8 defect-class sweeps) with 493 findings by the synthesis's recount ([coherence/README.md](coherence/README.md)). Conventions are collected in [coherence/conventions-proposed.md](coherence/conventions-proposed.md), the 21 breaking proposals in [coherence/decisions-for-maintainer.md](coherence/decisions-for-maintainer.md), the 17 result corrections decided and implemented in [coherence/lead-decisions.md](coherence/lead-decisions.md), and 14 website items in [coherence/website-items.md](coherence/website-items.md).

**Wave 6** took the decisions, the leftovers of waves 1 to 5 and every failure of the last full run. Counts are from each area report; a spec file can count in more than one area.

| Area | Streams | Device spec files (final runs pass) | Left open by the report |
|---|---:|---:|---|
| Renames and removals | 14 | about 310 | Source file and directory names keep their old stems by rule; generated blocks, catalog summaries and inventory need regeneration |
| Statistics | 6 | about 110 | Point density near 2^126 (below); EXTREME items deferred; forwards to other areas not confirmed applied |
| Geometry, joins, trajectories, recipes, cells | 8 | 44 | Offset-component timing at capacity 512 unmeasured; typecheck rerun after the renames |
| Raster, terrain, network | 10 | 47 | Three EXTREME items (network KDE excluded-event word, service areas on overlapping CSR, an embedding sweep that would pass with zero kernels) |
| Core, catalog, dataframe, point cloud, record, package | 8 | 9 | 32 stale Parameters pages, catalog summaries, website inventory, one roadmap cell, a clean `yarn build:release` |
| Lenses, Kepler, examples, website | 7 | 24 | Record-only failures: 171 lens declaration mismatches to recapture, one first capture; website GPU runs |
| Leftovers of waves 1 to 5 | 7 | 59 | B7-RING-01, three MGWR cases, the Mantel byte check, two forwards |

The most consequential corrections, by family:

- **Statistics, float32 scale.** Weighted emerging hot spots combined each window's Gi* moments in mixed coefficient scales; they now use a common scale with compensated accumulation, and hot or cold categories can change where rounding had moved a score across a cutoff.
- **Trajectories, loss behind an omitted status view (decision 9).** Thinning and staypoints now require `output.status.candidateOverflow` and locations `output.status.iterationLimitReached`; mobility indicators exclude missing and unsettled location keys and count them in `status.excludedLocationCount`.
- **Joins, completeness (GEO-103).** An anti-only predicate join must bind `output.unmatched.status.candidateOverflow` unless its capacity covers every pair; on the device the row whose candidates were dropped is now withheld from the unmatched output.
- **Terrain, units (FRM-06).** The drainage recipe reported downstream distances and reach lengths in affine units on geographic and Web Mercator grids; they are now WGS84 ground metres.
- **Core, float32 range.** The grid coordinate kernels mapped x = 0 over [-3e38, 3e38] to column 0; wide bounds are now scaled by a power of two.
- **Package (TOOL-102, decision 28).** A post-emit pass in the build rewrites relative type imports in declarations: a release run resolved all 4,669 relative references in 2,088 packed declaration files (45 failed before) and the tarball ships `dist` only. Its final step refused publication because other sessions edited sources during the build.
- **Consumers (UCO-10).** Kepler discovery enumerates the qualified catalog once by `{package, exportName}`; the 22-name network supplement and three side passes are removed, and the Kepler node project passes 8,060 tests in 514 files.

**Test state.** The last full run before wave 6 had 218 failing Node tests of 38,774 and 128 failing headless files of 2,018 (two specs hung); wave 6 was assigned all of them. The full gate on the final committed tree had not run when this was written; no pass count is given for it.

### Decisions

The full text is [decisions-2026-11-20.md](decisions-2026-11-20.md). The governing rule: the package is unpublished, so before the alpha there is one name per concept and no compatibility alias; a rename is made directly, the old name removed and every in-repository caller migrated in the same wave. Serialized identifiers that exist outside code (lens scenario IDs, committed record names) are not renamed.

The 21 breaking proposals, with the outcome the area reports show:

| ID | Decision | Outcome |
|---|---|---|
| NAM-13 | `k` becomes `clusterCount` (k-means) and `neighborCount` (four neighbour APIs) | Done |
| NAM-14 | `GPUDBSCAN`, `GPUHDBSCAN`, `GPUPycnophylacticInterpolation`; `minimumSamples` | Done |
| OUT-11 | Pair gather `rowCount` becomes `status.count` | Done |
| STA-13 | Five `_STATUS` maps become `_STATUS_LAYOUT` | Done |
| ING-20 | Object form of `prepareGPUNetworkRestrictionStates` | Done |
| FRM-06 | Drainage recipe forwards `cellSizeMode` | Done |
| CMP-13 | Resident colour scale `classCount`; threshold overload | Done |
| TYP-09 | Transfer masks typed `GraphDataView<'uint32'>` | Done |
| TYP-11 | Valid configurations encoded in props | Done differently: the baseline already rejected both cases; only the mask-only narrowing was added |
| TYP-12 | Literal critical levels plus a checked lookup | Done |
| TYP-13 | `Readonly` spectral props and operation descriptors | Done |
| TYP-14 | `recorder.graph` replaced by an inspection lookup | Done (`getRecordedGraphDescription`) |
| TYP-15 | Alignment props require four outputs; separate recipe props | Done |
| ENT-10 | Ontology re-export removed from the spatial barrel | Done (73 named imports and 2 namespace references migrated; the proposal counted 69) |
| ENT-13 | Three WGSL getters out of the regression barrel | Done |
| LIF-09 | `reusePreparation` removed from the simulation operation form | Done |
| WRT-13 | `GPUPickRegionMaskOutput` with an overflow status | Done |
| UST-15 | `symmetrize` becomes `average-reciprocal` | Done |
| URT-10 | Hydrologic `slope` becomes `descentTangent` | Done |
| UDF-14 | `GPUKeyJoin` becomes `GPUKeyLookup` with `compactMatchedLeftRows` | Done |
| UCO-10 | Kepler discovery keyed on the qualified catalog | Done |

Twenty are done, one done differently, none open; generated pages and lens records that still show old names wait for regeneration and recapture at landing. The planner leftovers were decided and carried out: the implicit one-period cell-period packing and unchecked unpacker removed (EGR-04/05), `symmetricDifference` removed (NAM-09), and the location-allocation admission limit (W2-03) and the explicit-grid Gaussian GGWR cost (SSR-121) kept and stated on their pages. Every `@deprecated` alias the pass added since `bd8279a4` (about 186) was retired; the audit leaves one, the route record `addResistanceConnectivityRecipe`, kept because committed lens records reference it.

Of this README's 28 decisions, two remain open: **1, package identity**, which is the maintainer's (owner, URL and name of a public package), and **6, a second GPU vendor and Safari and Firefox runs**, which needs hardware the lead does not have. Decision 2: the alpha stays held until this work lands and one full gate passes on the committed tree. Decision 7: keep LRN-06, EMB-03, EMB-04 and SIM-03 and drop the other LRN, EMB, SIM, CART and ENC rows; the roadmap was edited accordingly (29 rows deleted). The rest are recorded in [decisions-2026-11-20.md](decisions-2026-11-20.md).

### What the pass taught

Three kernel fixes for EXTREME-magnitude defects changed ordinary results, and only second reviews found it, since the device runs and tolerance-based specs passed: line noding returned 0.9531249 for an exact 61/64, the Gram traces changed the last bit of about 4,500 of 10,000 ordinary words, and the change-point slope kind published the other of two exactly tied splits after a translation. The rule adopted from then on: an extreme-only fix keeps ordinary output bit-identical behind a magnitude gate computed from operand exponents, proven by an old-versus-new float32 word sweep and confirmed by an independent review, or it is reverted and recorded open. Implementers without a GPU twice shipped shaders that did not compile on the device (a constant infinity bitcast, which zeroed every MGWR full-inference output, and a reserved word); device runs found both.

### Still open

The roadmap ([../../roadmap.md](../../roadmap.md), section "Library review 2026-11-19 (R19)") carries these as rows.

- **Package identity** (decision 1). The maintainer's.
- **A second GPU vendor, Safari and Firefox** (decision 6). All device evidence is one Apple Metal adapter in Chromium.
- **The full gate and the alpha** (decision 2). One full gate on the committed tree, and a clean `yarn build:release` on a quiescent tree; the last release run passed every gate but refused publication over concurrent edits.
- **Landing work.** Regenerate 32 stale Parameters pages, the spatial catalog summaries and the website inventory; recapture 171 lens records with declaration mismatches plus the records named in the area reports, and make one first capture (`local-building-tessellation-recipe`).
- **B7-RING-01.** Ring assembly flags overflow for finite coordinates spanning ±2^127; a partial fix let a closed ring through with a NaN area and clear status and was reverted.
- **Three MGWR EXTREME cases.** Variance at the overflow midpoint published as the float32 maximum with status 0, a derivative-norm overflow, and a lost subnormal variance.
- **`describeGPUSpatialWeights`** drops an inherited completeness certificate (`spatial-weights.ts:378`); forwarded with a reproduction, not applied.
- **Metal density limit near 2^126.** Rectangular point density counts and sums disagree with their keys for bounds whose extent reaches 2^126 on Metal with alpha.16; documented, and Kepler rejects endpoints of magnitude 2^125 or more.
- **Mantel oracle byte check (NUM-03).** All 4,815 retained finite draws regenerate within 8.2e-16; the byte comparison needs the recorded scientific package versions.
- **Landscape oracle (URB-105).** A pinned R run must establish the class and patch mapping and perimeters before the recipe's output is accepted.
- **Device measurements not yet recorded.** Spherical overlay at 8 to 128 combined vertices (DGGS-110) and datum-grid selection at 1 to 512 descriptors (DGGS-111) as benchmark series, routing latency at the published cap (FLT-12), a measured entropy tolerance (NUM-14), Voronoi and Delaunay probe readbacks (GEO-123, GEO-130), and upstream evidence for LUMA-101, 103 and 111. The stream for these had not started.
- **14 website items** (12 distinct IDs, [coherence/website-items.md](coherence/website-items.md)), and the website GPU runs and deck smoke the headless harness cannot run.
- **Decisions not yet asked.** The area reports mark no item NEEDS DECISION (the one raised, rounded facility ties, was decided: the device was right); the second-round synthesis lists 33 findings tagged breaking that appear in neither decision file.

## Method and what was checked

- **Who.** A lead reviewer (Claude) set briefs and cross-checked. Six Claude Fable area leads with Opus subagents wrote first passes of statistics (3 slices), geometry (5), points (6), cells, trajectories and recipes (4), direction, and thin first passes of network and raster and terrain; the geometry report calls its lead a "Sonnet-tier agent". Nine Codex pipelines, each one `gpt-6.2-astra` lead and six `gpt-6.1-sol` slice reviewers, covered dataframe, core, API/docs/catalog, tooling, luma.gl and scale, seams, roadmap and developer documents, and gave second, deeper passes of network and raster and terrain. Reason: the session's limit of twenty concurrent subagents left first-pass leads without subagents. The statistics report says slices 4 and 5 were never delegated; the cells report says the limit delayed two slices, so the lead read recipes itself and merged the trajectory slice.
- **"Executed".** CPU runs of public functions (`compute*OnCPU`, packers, Arrow preparation, catalog helpers), float32 transcriptions of shader arithmetic in JavaScript, NumPy or Python, WGSL on the `wgsl_reflect` interpreter, null-device graph builds. No reviewer ran GPU code. A text match on the Evidence column: 78 of 118 W and S findings cite a CPU run or model, 40 are source traces (approximate).
- **Run afterwards, all scoped.** Fix agents ran single Node spec files and both checkers (`check-upstream-first`: no violations; `check-contributor-contract`: no new findings). GPU: the 73 headless spec files the fixes wrote or changed, WebGPU in Chromium on Apple Metal: 66 passed, 5 failed, 2 are benchmark files outside the ordinary selection. The five failures were sent back and each file was rerun whole by its owner and passes: three `DPC-109` files (the control rows of the fixture were wrong, not the fix: 41/41, 11/11, 4/4), `gpu-network-cost-weights.spec.ts` (a duplicate resource id in the test: 6/6) and `symmetric-eigen-wgsl.spec.ts` (the first CORE-106 change had no effect on Metal and was replaced: 5/5). The core-numerics agent ran its five shared-kernel specs with the fix and with only that kernel restored to the committed version: each passes with the fix and fails without it. The whole set was not rerun as one batch after the last change. Node: 261 spec files (every Node spec changed or added, plus the core, catalog, lens and documentation specs): 8,800 of 8,824 tests passed; the 22 failures were in five files and each was resolved and rerun (kernel-audit cases for the new K-function limit; the website inventory and lens view models, regenerated; the release-declarations spec; 14 lens scenario records, recaptured with 67 twins). Typecheck: source passes with 0 errors after one lead fix; the test typecheck over all 240 groups found 20 errors in seven spec files written today, which were being corrected when this was written. Build: `yarn build:release` staged a clean build that passes the export check (55 entry points) and the packed-consumer check. The documentation Parameters blocks were regenerated and `doc-parameters.node.spec.ts` passes; `roadmap-reconciliation.node.spec.ts` passes 9 of 9.
- **Not run.** The full Node suite, the full headless suite, `yarn verify` as one command, benchmarks, a full lens recapture (34 records hold stale shader text that no spec flags), hosted CI, a second GPU vendor or browser.
- **Known failures on the tree when this was written.** The 20 test type errors above. Nothing else known; the full suites have not run on this tree.

## Files

In `dev-docs/research/library-review-2026-11-19/`. Counts are from each report's Findings table (W wrong result, S visible failure, C contract or documentation, T test, M maintainability).

| File | KB | Scope | W | S | C | T | M | Total |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `area-spatial-stats.md` | 48 | Statistics, regression, weights, autocorrelation, permutation, validation (about 100,000 lines) | 10 | 1 | 15 | 2 | 3 | 31 |
| `area-spatial-geometry.md` | 46 | Geometry, joins, search, zonal statistics, grids | 9 | 1 | 19 | 1 | 1 | 31 |
| `area-spatial-points.md` | 45 | Point patterns, clustering, interpolation, regionalization, space-time, pair statistics | 9 | 3 | 26 | 6 | 1 | 45 |
| `area-cells-trajectories-recipes.md` | 51 | Cells and DGGS, trajectories, urban and landscape, recipes | 11 | 5 | 13 | 3 | 4 | 36 |
| `area-network.md` | 73 | Network analysis (Codex second pass) | 10 | 1 | 8 | 2 | 4 | 25 |
| `area-raster-terrain.md` | 83 | Raster and terrain (Codex second pass) | 10 | 4 | 11 | 3 | 3 | 31 |
| `area-dataframe-pointcloud-record.md` | 79 | Dataframe, point cloud, computation record, Arrow | 14 | 1 | 6 | 3 | 3 | 27 |
| `area-core-contract.md` | 79 | Shared core, contributor contract, kernel audit | 7 | 2 | 4 | 7 | 2 | 22 |
| `area-api-docs-catalog.md` | 81 | Public API, documentation, operator catalog | 1 | 2 | 9 | 5 | 2 | 19 |
| `area-tooling-tests-release.md` | 87 | Gates, tests, build, package, release | 1 | 2 | 5 | 11 | 2 | 21 |
| `area-luma-upstream-scale.md` | 93 | luma.gl integration, workarounds, GPU scale | 3 | 1 | 7 | 3 | 4 | 18 |
| `area-consumer-seams-lenses.md` | 83 | Lenses, Kepler, examples, website seams | 10 | 0 | 8 | 2 | 2 | 22 |
| `area-roadmap-devdocs.md` | 110 | Roadmap (291 rows) and developer documents | 0 | 0 | 11 | 2 | 2 | 15 |
| `area-direction.md` | 70 | Direction for two quarters (options DIR-101..109, risks DIR-110..116) | - | - | - | - | - | 16 |
| `ki-pass-digest.md` | 35 | Digest of the unlanded known-issues pass | - | - | - | - | - | - |
| `roadmap-changes.md` | 25 | Roadmap edits made today, source checked, decisions | - | - | - | - | - | - |
| `doc-corrections.md` | 50 | Three batches of documentation corrections | - | - | - | - | - | - |
| **Sum, 13 tables** | | | **95** | **23** | **142** | **50** | **33** | **343** |

Where a report's own count differs from its table, the table is used: geometry says 33 findings (21 C) and lists 31; seams says 9 verified W and lists 8 plus 2 leads; statistics says all but SSR-08 and SSR-13 are fixed and its table also marks SSR-14 PARTLY; cells says 15 FIXED and its table 14 plus one OPEN row; the API report says 24 FIXED / 10 PARTLY and its table 25 / 9.

## Follow-through on the 2026-11-17 findings

Rows are the rows of each report's follow-through table (some add related items from the 2026-11-18 rounds). "Fixed" includes "fixed for the stated scope" and means the report read a current assertion that fails on the old behaviour; none was rerun. The tooling report sampled 30 public behaviour changes: 30 have a discriminating assertion, 26 a located historical failure (final-flow decomposition, old grid reader on a device, colour-scale counts and bivariate counts do not).

| Area | Rows | Fixed | Partly | Limit or open | Exceptions |
|---|---:|---:|---:|---:|---|
| Statistics | 17 | 13 | 4 | 0 | SSR-08 padded readers: OLS reader missed (SSR-106). SSR-13 loop baseline: 7 of 19 entries generic. SSR-14: GTWR docs still say 32,768 rows (SSR-122). PTS-08 early-stopped draws: Knox only (SSR-114) |
| Geometry | 13 | 11 | 1 | 1 | GEO-04 longitude is a documented limit; GEO-12 multipart joins for three predicates |
| Points | 17 | 14 | 2 | 1 | PTS-06, DOC-01 partly; Vecchia over-width conditioning sets open |
| Cells, trajectories, recipes | 16 | 14 | 1 | 1 | RCP-01 partly; shared STD-09 note open; the "every recipe is atomic" claim fails for one of 13 urban recipes (RCP-101) |
| Network | 11 | 10 | 0 | 1 | NET-08 sparse cost-table size, documented limit |
| Raster and terrain | 15 | 13 | 2 | 0 | RT-09, RT-14 partly |
| Dataframe, point cloud, record | 20 | 18 | 0 | 2 | DPC-07 (Arrow required) limit by decision; ERG-09 public WGSL attribute helpers still exported |
| Core | 15 | 12 | 3 | 0 | CORE-10 loop allowance survives a changed host bound (CORE-118); CORE-13 adapter misattributes chunk restriction (CORE-112); CORE-14 vocabulary guard omits point-cloud analysis |
| API, docs, catalog | 34 | 25 | 9 | 0 | ERG-06, 09, 12, CAT-04, 09, 10, DOC-03, 04, 10 partly |
| Tooling | 18 | 14 | 4 | 0 | TOOL-02 release record names the tarball by path only (TOOL-105); TOOL-09 nine early returns (TOOL-110); TOOL-14, TOOL-15 |
| luma.gl and scale | 35 | 32 | 2 | 1 | CORE-13 / DOC-05, RCP-04 partly; NET-08 limit |
| Seams, lenses | 10 | 6 | 3 | 1 | LENS-01 (LENS-101), LENS-06, LENS-09 partly; LENS-08 restricted data limit |
| Roadmap, developer documents | 10 | 6 | 4 | 0 | DOC-03, 08, 09, 10 partly |
| **Total** | **231** | **188** | **35** | **8** | |

## New findings

All 118 W and S findings of every report, in three tables. Kind: W wrong result, S visible failure. Evidence is the report's label: CPU executed on the CPU, SRC source trace, "f32 model" and "transcript" arithmetic transcriptions, "unverified" or "lead" not confirmed on a GPU. Reach is the report's grade (ORDINARY normal input, UNUSUAL particular input or sequence, EXTREME near a numeric limit). State comes from the fix records and `wave1.summary.txt`:

- **FIXED, GPU PASS**: fixed in the working tree; a headless spec file covering it passed in the scoped run (one Apple Metal adapter). The file passing does not say which case ran.
- **FIXED, node spec only**: fixed with Node specs; not run on a GPU.
- **FIXED; device PASS per fix record** and **spec FAILED in run, re-run PASS per fix record**: the run summary does not show the pass.
- **DEFERRED**: the staging pass owns the file; the row says whether the staging copy has the fix.
- **OPEN**: no fix record.

No fix has been read by anyone other than its agent.

### Ordinary reach (23)

| ID | Kind | Where | What goes wrong | Evidence | State today |
|---|---|---|---|---|---|
| CAT-101 | W | `catalog records` | Catalog claims `status: []` for operators that publish one; overflow search omits six line operators | CPU, CNT | FIXED, node specs; 99 records corrected, default is decision 10 |
| DOC-101 | S | `cross-reference page` | Documented entry points for raster and dataframe fail module resolution | CPU | FIXED, node spec only |
| DOC-102 | S | `cross-reference page` | Documented spherical option is ignored and the planar route throws | CPU | FIXED, node spec only; narrowed: docs wrong, code right |
| SEAM-101 | W | `space-time-parameters.ts` | Knox Poisson upper tail returns 0 for representable positive probabilities | CPU | FIXED, node spec only |
| SEAM-103 | W | `extract-lens-view-model.ts` | Saved maps colour missing or suppressed values as the measured minimum (1,150 nodes) | CPU | OPEN, no fix record; website, peer session |
| SEAM-104 | W | `catalog-platform3-b-clusters.ts` | Kepler facility objective reverses on negative demand | CPU | FIXED, node spec only |
| SEAM-105 | W | `website segment-snap` | Dragged reference labelled "inserted" when snapping excludes it | CPU | OPEN, no fix record; website, peer session |
| LENS-101 | W | `lenses static twin` | Saved kriging record read alone shows a one-row diagnostic instead of 4,662 cells | CPU | FIXED, node spec only; recapture pending |
| SEAM-109 | W lead | `gpu-emerging-hot-spots.ts` | Committed score path lacks weights-loss suppression (lead) | SRC, GPU unverified | FIXED, GPU PASS via PTS-104 |
| CORE-102 | W | `point-neighbor-grid-index.ts` | Projected origin: grid writer and 3x3 reader disagree on a neighbour's cell, DBSCAN misses a pair one radius apart | SRC, GPU unverified | FIXED; device PASS per fix record; test not in run summary |
| DPC-101 | W | `cell-period-key-packing.ts` | CPU packer turns the documented missing cell into a valid group key | CPU | FIXED, node spec only |
| DPC-102 | W | `create-kernel-annotation.ts` | Two nodes sharing WGSL lose the second node's entry point and constants in the record | CPU | FIXED, node spec only |
| DPC-103 | W | `gpu-composite-score.ts` | Constant indicator acquires positive variance and a nonzero score | SRC, GPU unverified | FIXED, GPU PASS |
| LUMA-101 | W lead | `@luma.gl histogram` | Equal-width histogram quotient can put an exact edge value in the preceding bin (lead) | SRC, UNVERIFIED | OPEN, no fix record; upstream draft 02 |
| NET-106 | W | `network-spatial-context.ts` | Absolute projected point formed before subtracting the query loses the local residual | SRC, CPU control | FIXED, GPU PASS |
| NET-107 | W | `gpu-network-potential-path-area.ts` | Reused edge slot keeps its old mask and slack and is still counted | SRC | DEFERRED; staging has fix |
| NET-113 | W | `drive-time-catchment-recipe.ts` | Lon/lat recipe admits the raster branch with a planar child | CPU | DEFERRED; staging lacks fix |
| RT-124 | W | `cpu-streamlines-operation.ts` | RK2 accepts a point after leaving usable field support | CPU | FIXED, GPU PASS |
| RT-126 | W | `gpu-raster-embedding-zone-mean.ts` | Fractional coverage prefix becomes an apparently complete embedding mean | CPU graph | FIXED, GPU PASS |
| RT-121 | S | `raster-embedding-principal-components.ts` | PCA stretch of a constant component has zero width; the projection packer rejects it | CPU | FIXED, node spec only |
| GEO-102 | W | `cpu-offset-curve-components-operation.ts` | CPU twin publishes zero-length fragments on oblique lines (CPU twin only) | CPU | DEFERRED; staging lacks fix |
| PTS-112 | S | `gpu-continuous-spatial-sample.ts` | Area lattice larger than `maximumCandidates` is reported only by a runtime flag | SRC | FIXED, node spec only |
| TOOL-102 | S | `12 emitted .d.ts` | 22 relative type references fail native NodeNext resolution, including public parameter schemas | CPU, CNT | PARTLY: 21 of 22 edges, unverified until a build; last edge in staging file |

### Uncommon input or sequence (68)

| ID | Kind | Where | What goes wrong | Evidence | State today |
|---|---|---|---|---|---|
| DGGS-101 | W | `gpu-cell-set-outline.ts` | H3 outline picks the neighbour across an edge by planar lon/lat distance (poles, coarse cells) | CPU | FIXED, GPU PASS |
| DGGS-102 | W | `cell-key-wgsl.ts` | Cell sums clamp above 7.04e13 and wrap negative above 2^63/scale at the default scale | CPU | FIXED, GPU PASS |
| DGGS-103 | S | `gpu-cell-cover.ts` | Feature-count by capacity product rule rejects valid calls (S) | SRC | FIXED, GPU PASS |
| DGGS-104 | W | `healpix-index-wgsl.ts` | HEALPix belt wrap rounds to the wrong cell in f32 near longitude -4.6e-5 degrees | CPU f32 transcription | FIXED, GPU PASS |
| DGGS-105 | W | `gpu-spherical-polygon-contains.ts` | Clockwise holes are ignored and clockwise shells contain nothing, status 0 | SRC, CPU f64 | FIXED, GPU PASS; both orientations accepted (decision 21) |
| TRJ-101 | W | `gpu-trajectory-locations.ts` | Non-root labels published at the 32-round limit with no flag | SRC, NumPy model | FIXED, GPU PASS |
| TRJ-102 | W | `gpu-trajectory-thinning.ts` | Thinning and staypoints change their answer past `maximumSearchRows` (4,096 by default) | CPU | PARTLY: staypoints fixed, GPU PASS; thinning needs decision 9 |
| TRJ-103 | W | `gpu-mobility-indicators.ts` | `realEntropy` is 0 for tracks over 1,024 staypoints | SRC | FIXED, GPU PASS |
| TRJ-104 | W | `showcase satellite scene` | Scene ignores the candidate-overflow word; band entry counts can be incomplete | SRC | FIXED, node spec only |
| URB-103 | S | `visibility-graph-csr.ts` | One invalid viewpoint zeroes every offset and count (S) | SRC | FIXED, GPU PASS |
| RCP-101 | S | `segregation-decomposition.ts` | Rejected `randomization.replicates: 0` leaves 61 `graph.add` calls on the caller graph (S) | CPU | FIXED, node spec only |
| SEAM-102 | W | `gpu-azp-regions-operation.ts` | Loss-bearing weights accepted, graph lowering drops the flags; CPU form reports convergence | CPU, SRC | OPEN, no fix record; same files as PTS-103 |
| SEAM-106 | W | `catalog-platform3-a-join.ts` | Translating a polygon and a point together changes hole membership | CPU | FIXED, node spec only |
| SEAM-107 | W | `showcase joins scene` | Candidate-only loss reported as a definite success fraction and "no overflow" | CPU, SRC | FIXED, node spec only |
| CORE-101 | W | `local-frame.ts` | Short destination silently drops projected or unprojected rows | CPU | FIXED, node spec only |
| CORE-108 | S | `parameter-schema.ts` | Field named like a prototype property fails to use its declared default (S) | CPU | FIXED, node spec only |
| DPC-104 | W | `arrow-analytics-normalization.ts` | Empty option map reads inherited keys: Int64 column `constructor` becomes a time column | CPU | FIXED, node spec only |
| DPC-105 | W | `describe-command-graph.ts` | Render node attributed to a compute pass in a mixed graph | CPU | FIXED, node spec only |
| DPC-106 | W | `wide-numeric-gpu-decimal-reference` | Negative decimal scale formats the coefficient followed by a point | CPU | FIXED, node spec only |
| DPC-107 | W | `pointcloud-attribute.ts` | Public WGSL attribute generators change meaning of compound arguments | CPU | FIXED, node spec only; ERG-09 stays open |
| DPC-108 | W | `space-time-k-function-nodes.ts` | Packers accept values that lose width, order or finiteness in float32 | CPU | FIXED, node spec only; time-window and composite packers still open |
| DPC-115 | S | `arrow-analytics-normalization.ts` | Empty dictionary-options map rejects a UTF-8 column named `constructor` (S) | CPU | FIXED, node spec only |
| LUMA-104 | S | `gpu-chunked-spatial-weights-transform.ts` | Valid chunked call rejects 33,554,432 rows for nonexistent global scratch (S) | CPU | DEFERRED; staging lacks fix |
| NET-101 | W | `gpu-network-cost-matrix.ts` | Budget-limited dense costs enter numerical consumers as final | SRC, CPU controls | FIXED; 3 of 4 GPU files PASS, cost-weights file failed (see NET-124) |
| NET-108 | W | `network-isochrones-parameters.ts` | Positive finite extent packs to zero-width or non-finite float32 | CPU | FIXED, node spec only |
| NET-109 | W | `network-isochrones-passes.ts` | Raster work item before first CSR offset maps to node 0 and draws an unowned road | SRC | DEFERRED; staging lacks fix |
| NET-110 | W | `gpu-network-k-function.ts` | Event counts can wrap the uint32 pair histogram; K looks finite | SRC, CPU | FIXED, node spec only |
| NET-111 | W | `gpu-radiation-flow-model.ts` | Finite-size normalization drops the population outside the origin (flow 100 becomes 0) | SRC, CPU | FIXED, GPU PASS |
| NET-124 | W | `gpu-network-cost-weights.ts` | `complete` ignores the matrix cutoff against the requested cutoff | SRC, CPU | FIXED; spec FAILED in run, re-run PASS per fix record; failure was a test id clash |
| NET-114 | S | `gpu-network-louvain-operation.ts` | Status property present with value `undefined` blocks construction (S) | CPU | FIXED, node spec only |
| RT-120 | W | `raster-snapshot-eof-decomposition.ts` | Unit-scale floor in Jacobi convergence loses a rank-one mode at small magnitude | CPU | FIXED, node spec only |
| RT-122 | W | `gpu-terrain-flow.ts` | Flat-status outputs have no writer when only filled elevation is requested | CPU graph | FIXED, node spec only; writer spec held in scratch |
| RT-123 | W | `terrain-flow-tracing.ts` | Pointer doubling collapses a rootless cycle into a false terminal | SRC, CPU model | FIXED, GPU PASS; HAND only |
| RT-128 | W | `terrain-summed-area-table.ts` | Distant first-valid elevation changes local relief of a haloed core | SRC, CPU model | FIXED, GPU PASS |
| RT-130 | W | `raster-irregular-date-axis.ts` | Interpolation fraction can exceed 1 on strictly increasing dates | SRC, CPU model | DEFERRED; staging lacks fix |
| RT-132 | W | `gpu-raster-lagged-correlation.ts` | Finite centered sums overflow; identical series get correlation 0 with success | SRC, f32 model | FIXED, GPU PASS |
| RT-134 | W | `gpu-raster-skeleton.ts` | Verification iteration thins the published skeleton past the budget | CPU graph | FIXED, GPU PASS |
| RT-129 | S | `gpu-lagrangian-flow-map.ts` | Signed-cell contract and validator disagree: north-up negative height throws (S) | CPU | FIXED, GPU PASS |
| RT-131 | S | `gpu-raster-temporal-resample.ts` | Gap to the next sample invalidates an exact observation (S) | SRC | DEFERRED; staging has fix |
| RT-133 | S | `gpu-raster-lagged-correlation.ts` | Translated nonconstant series fails the variance check (S) | SRC, f32 model | FIXED, GPU PASS |
| GEO-101 | W | `cpu-offset-curve-components-operation.ts` | Offset with tolerance 0 drops the interior at projected magnitudes | CPU | DEFERRED; staging lacks fix |
| GEO-103 | W | `gpu-spatial-predicate-join.ts` | `output.pairs` accepted with no `candidateOverflow` or status port | SRC, CPU | OPEN, no fix record |
| GEO-104 | W | `geometry-measures-kernels.ts` | Geographic group bounds are per-feature min/max on separate longitude branches | SRC, arithmetic copy | OPEN, no fix record |
| GEO-105 | W | `zonal-statistics-passes.ts` | Zonal variance near 1e7 reports 123 where the truth is 2 | numpy f32 model | FIXED, GPU PASS; raster zonal also |
| GEO-107 | W | `gpu-map-coloring.ts` | Map colouring ignores weights loss and can report a colouring with conflicts | SRC | FIXED, GPU PASS |
| GEO-110 | S | `gpu-segment-intersection.ts` | Nested segment intersection requires a candidate-overflow port when leaf capacity is small (S) | SRC | OPEN, no fix record |
| PTS-101 | W | `gpu-neighborhood-summary.ts` | Reads `overflow` only; pairs dropped by `pairCapacity` give ordinary numbers | CPU | FIXED, GPU PASS |
| PTS-102 | W | `cpu-max-p-regions-operation.ts` | Enclave rounds stopped by `maximumGrowthSteps` leave non-adjacent parts under one label | CPU, GPU trace | OPEN, no fix record |
| PTS-103 | W | `gpu-azp-regions.ts` | Regions built or scored on truncated adjacency without a status change | SRC | DEFERRED; staging has fix |
| PTS-104 | W | `local and near-repeat Knox, EHS` | Counts and Gi* computed on truncated rows with no flag | SRC | FIXED, GPU PASS |
| PTS-105 | W | `variogram-model.ts` | Public fits cut bin arrays to the shortest length | CPU | FIXED, node spec only |
| PTS-106 | W | `space-time-k-function-nodes.ts` | Inhomogeneous space-time K unit 2^-16: +6.8 percent at intensity 100, -100 percent from 363 | WGSL transcript | FIXED, GPU PASS |
| PTS-107 | W | `gpu-spatial-scan-statistic.ts` | `[1,1,NaN,1]` baseline normalises to `[0,0,1.5,0]` | WGSL transcript | FIXED, GPU PASS |
| PTS-108 | W | `gpu-point-pattern-indices.ts` | Zero-sample polygon quadrats skipped but expected counts use all points | SRC | FIXED, GPU PASS; result is NaN (decision 19) |
| PTS-109 | W | `gpu-point-pattern-indices.ts` | Quadrat index via reciprocal: 6 of 9 interior edges land in the lower quadrat on Metal | SRC, emulation | FIXED, GPU PASS |
| PTS-110 | S | `gpu-huff-trade-areas.ts` | Infinite attractiveness passes `> 0` and reaches `frexp(inf)` (S) | SRC, interpreter | FIXED, GPU PASS |
| PTS-111 | S | `gpu-continuous-spatial-sample.ts` | One serial invocation per wave allows about 1.7e9 inner iterations (S) | SRC | FIXED, node spec only |
| SSR-101 | W | `gpu-spatial-markov.ts` | Rebuilds weights without `status` before the lag | SRC | FIXED, node spec only |
| SSR-102 | W | `18 weights consumers` | Take weights and never read `status`; truncated table gives clean results | SRC grep | PARTLY: 6 fixed, 1 consumer refuted, rest deferred |
| SSR-103 | W | `7 consumers` | Read `overflow` only, ignore `candidateOverflow` and chunk flags | SRC | DEFERRED; staging has fix (except multivariate Geary, panel regression) |
| SSR-104 | W | `gpu-island-attachment.ts` | Ignores input weights status; emptied rows become islands with invented edges | SRC | DEFERRED; staging has fix |
| SSR-105 | S | `gpu-chunked-spatial-weights-transpose.ts` | Chunked transpose output is refused by the statistics it should feed (S) | SRC | FIXED, node spec only; status shape changed (decision 17) |
| SSR-106 | W | `ordinary-least-squares-statistics.ts` | OLS reader takes predictor count from the table length: df 6 for 8, AIC 28 for 24 | CPU | DEFERRED; staging has fix |
| SSR-107 | W | `panel-diagnostics.ts` | LM test sums N*T rows serially in float32: +15 percent at 2^24 rows | SRC, f32 emulation | DEFERRED; staging lacks fix |
| SSR-108 | W | `spatial-sur-nodes.ts` | Spatial SUR residuals from the raw float32 intercept: [0,0,-2,-2,0] for [0.25,-0.25,0.25,-0.25,0] | f32 emulation | DEFERRED; staging lacks fix (partly) |
| SSR-109 | W | `gpu-probit-regression.ts` | Probit tolerance has no upper limit: tolerance 0.5 gives a 16 percent slope error | CPU twin | DEFERRED; staging lacks fix; limit value is decision 13 |
| TOOL-101 | W | `build-commonjs.mjs` | CommonJS rewrite changes strings that contain import-like text | CPU | FIXED, node spec only; unverified until a build |
| TOOL-103 | S | `verification-policy.mjs` | Verify and release reject a valid lower heap when the environment value is over the ceiling (S) | CPU | FIXED, node spec only |

### Extreme reach, and unverified leads (27)

The 12 EXTREME rows, plus 15 UNUSUAL rows that the report marks as a lead or GPU-unverified.

| ID | Kind, reach | What goes wrong | State today |
|---|---|---|---|
| DGGS-106 | W, extreme | Longitude above about 1e8 degrees: f32 quadrant reduction wrong | FIXED, GPU PASS |
| DGGS-107 | W, unusual | `sumValues` at scale 2^127: Metal reciprocal flushes to zero (lead) | FIXED, GPU PASS; no 2^127 case |
| DGGS-110 | S, unusual | O(n^3) in one invocation, about 1.4 s extrapolated at 128 vertices (S, lead) | OPEN, no fix record |
| DGGS-111 | S, unusual | No cap on `grids.length`; each grid inlines a WGSL block (S, lead) | OPEN, no fix record |
| URB-101 | W, unusual | Randomization tie test biased low: tree-reduced observed vs serial replicates (lead) | FIXED, GPU PASS; narrower than reported |
| SEAM-108 | W, unusual | Zone assignment loss discarded; statistics published (lead) | FIXED, GPU PASS |
| CORE-103 | W, unusual | Exact nonzero orientation discarded when two rounded products cancel | FIXED; device PASS per fix record; test not in run summary |
| CORE-105 | W, unusual | Unguarded `atan2` at a boundary when the dot product is -0 | FIXED; device PASS per fix record; test not in run summary |
| CORE-104 | W, extreme | Case/baseline ratio overflows and suppresses a finite score | FIXED; device PASS per fix record; test not in run summary |
| CORE-106 | W, extreme | Diagonal difference overflows in Jacobi rotation; coupling cleared | FIXED; spec FAILED in run, re-run PASS per fix record; first version failed on Metal |
| CORE-107 | W, extreme | Radius packer writes zero or infinity for a finite positive radius | FIXED, node spec only |
| CORE-109 | S, extreme | Rejection after six graph resources are declared; corrected call with same id fails (S) | FIXED, node spec only |
| DPC-109 | W, unusual | Representable mean overflows because the total is narrowed before division | FIXED; spec FAILED in run, re-run PASS per fix record; failure was a test fixture error |
| DPC-110 | W, unusual | Class breaks and colour scale disagree at a signed-zero break | NEEDS DECISION; no code |
| DPC-111 | W, unusual | Equal-interval cut or histogram moves when extrema span overflows float32 | FIXED, GPU PASS |
| DPC-112 | W, unusual | Row excluded from every pair still shifts the centre and erases differences | FIXED, GPU PASS |
| DPC-113 | W, unusual | Wide pair count saturates a public count with success status | FIXED, GPU PASS |
| DPC-114 | W, extreme | Day-limit check precedes flooring: one extra day for a negative remainder | FIXED, GPU PASS |
| LUMA-102 | W, unusual | Finite endpoints give an infinite float32 span; counting branch skips every row (lead) | FIXED, GPU PASS; natural breaks not covered |
| LUMA-103 | W, unusual | Bounds query reconstructs a cell edge on the other side of the point (lead) | OPEN, no fix record; upstream |
| NET-112 | W, extreme | Positive bandwidth gives zero squared denominator and an undefined self weight | FIXED, GPU PASS; different fix than requested |
| RT-127 | W, extreme | Squaring a tiny nonzero mean underflows and blocks normalization | FIXED, GPU PASS |
| GEO-106 | W, extreme | Circumcentre unscaled: radius 50 percent too large above 2^32, 73 percent below 2^-42 | FIXED, GPU PASS |
| GEO-108 | W, extreme | GPU `within` for line-in-line differs from the CPU twin at a zero-length inner line | OPEN, no fix record |
| GEO-109 | W, extreme | `columns` and `rows` up to `MAX_SAFE_INTEGER` repeat lattice positions past 2^24 | FIXED, node spec only |
| SSR-110 | W, unusual | GWR/GTWR sum `w z y` with the raw response: slope 7 percent off at offset 1e6 (lead) | DEFERRED; staging lacks fix (GTWR only) |
| SSR-111 | W, extreme | Permutation p-value counts NaN permutations in the denominator (lead) | DEFERRED; staging has fix |

By state, the 118: 41 FIXED GPU PASS; 36 fixed with node specs only; 4 device PASS per fix record (CORE-102 to 105); 3 spec failed in the run (DPC-109, NET-124, CORE-106); 1 mixed (NET-101); 3 partly (TRJ-102, SSR-102, TOOL-102); 1 needing a decision (DPC-110); 17 deferred; 12 open (SEAM-102, 103, 105, GEO-103, 104, 108, 110, PTS-102, LUMA-101, 103, DGGS-110, 111).

## What the defects have in common

Grouped by reading the reports' descriptions (the grouping is mine; counts are W and S findings). The last column quotes proposals from the reports and fix records.

| Class | Findings | Today | What would stop it recurring |
|---|---|---|---|
| Completeness does not travel: a consumer ignores the loss flags of a weights port, cost matrix, pair list or zone assignment | 17: SSR-101 to 104; PTS-101, 103, 104; GEO-103, 107; SEAM-102, 107, 108, 109; NET-101, 124; RT-126; TRJ-104 | 11 fixed, 1 partly, 3 deferred, 2 open | One suppression sweep or construction-time refusal (SSR-102); one loss vocabulary and a "not tested" class (SSR-113); a spec per consumer with only `candidateOverflow` set, and chains lossy weights into island attachment (SSR-125); decide which bounded-loss operations must carry completeness evidence (CORE-110) |
| Float32 range and cancellation (projected magnitudes 1e5 to 1e7, near 2^24, near exponent limits) | 28: GEO-101, 105, 106; NET-106, 112; CORE-102 to 104, 106; SSR-107, 108, 110; RT-120, 127, 128, 130, 132, 133; DPC-103, 109, 111, 112; DGGS-104, 106, 107; PTS-106, 109; LUMA-102 | 23 fixed, 5 deferred | A translation fixture at origins 0, 5,000,000 and 2^24 per coordinate producer (SSR-03 form, SSR-108, GEO-105); writer and reader from one cell definition (CORE-102); an independent two-point device result for a projected route (core report); a second GPU vendor, since Metal reciprocal division recurs (PTS-109, DPC-109; DIR-111) |
| Silent loss behind a status view the caller may omit | 8: TRJ-101 to 103; PTS-102, 112; DGGS-102; DPC-113; NET-110 | 6 fixed, 1 partly, 1 open | Require the flag at construction as trajectory segments do (TRJ-102, TRJ-105) or write NaN (TRJ-103); decision 9 |
| Packers validate a double, write float32 | 4: CORE-107, NET-108, DPC-108, RT-130 | 3 fixed, 1 deferred | Validate the encoded value and relations after packing (CORE-107, NET-108, DPC-108); time-window and composite packers still need it |
| Outputs without a writer | RT-122; KI counts 16 entries (137 ports) plus 13 destinations | RT-122 fixed | Schedule the producer or reject the combination (RT-122); an independent oracle per output port (KI-01 census: 3,865 ports, 622 with none); counts-only fixtures (TOOL-114) |
| Tail probabilities | SEAM-101 (Knox tail 0), SSR-111; related SSR-124, KI D001, D009, D011, D013 | 1 fixed, 1 deferred | Evaluate the tail directly and pin a relative-error fixture (SEAM-101); the log-space detector test at 2^60 is the model (SSR-04); valid-permutation count (SSR-111) |
| Catalog and documentation claims | CAT-101, DOC-101, DOC-102 | 3 fixed | Finalize metadata once per qualified identity (CAT-102); check prose entry-point lists against the export map (DOC-101); typed `spatialContext` examples (DOC-102); one record joining forms, status and docs (API priority 3) |

## Consistency, documentation and catalog

**API census movement** (`area-api-docs-catalog.md`, 2026-11-17 to today): source files 1,710 to 1,772; lines 541,841 to 565,068; producers 492 to 498; producers accepting `props.output` 471 of 492 to 488 of 498 (point cloud 0 of 11 to 11 of 11); `estimatedCommandNodeCount` 171 to 177; explicit `any` in public props and output aliases 2 to 0 (930 to 942 aliases); output `status` 177 to 188 (157 named objects, 31 views); API pages 107 to 116, all in the sidebar and snippet-checked (was 98 and 66); `ts` fences 192 to 203, no opt-outs; 472 of 761 classes have a heading. 413 of 498 constructors call a named limit helper; 384 of 498 are unclassified for chunk or packed support by class documentation. Of 40 sampled error messages, 34 named contributor, subject and relation.

**Vocabularies.** Regression status: six code vocabularies with two meanings of code 1 and of 3, GGWR codes 5 and 6 undecoded, literal codes in Kepler, lenses and the explorer (SSR-112); staging adds three more. Mask and validity polarity differs by name (19 `outputValidity` aliases; population alignment matched code 0, keep-mask 1). Row-count spellings, 45 capacity spellings and `maximumNeighborCount` against `maximumNeighbors` remain.

**Catalog.** 555 records, 554 names, 92 families, no duplicate identities. Status 91 non-empty, 222 unknown, 242 confirmed empty; readback 8 explicit; tier 324 scale, 30 reference, 201 unknown; `cpuTwin` 0 populated; 414 identities lie outside the 144-row coverage matrix. Today 6 records (`fix-api-catalog.md`) and 93 (`fix-catalog-status.md`) were corrected; that agent confirmed the others as `[]` with a named reason. Default for unaudited records: decision 10.

**Documentation** (`doc-corrections.md`): 58 applied (9, 25, 24); 33 deferred to staging-owned pages (12, 7, 14); 33 not applied (9, 11, 13); 22 rows on generated blocks and source comments. No generated block was hand-edited. `docs/api/gpu-network/operators.md` is staging-owned and was edited by exception (six hunks; hand merge needed). The Parameters blocks are stale after today's props; `UPDATE_DOC_PARAMETERS=1` rewrites every stale page, staging-owned ones too. Text wanted on staging-owned pages (`operators-weights.md`, `operators-geometry.md`, regression pages, `global-coordinates.md`, `operators-cells.md`) is in the fix records.

## Gates, tests and package

From `area-tooling-tests-release.md` (one Mac, Metal).

| Gate | Passes when | Can pass without checking | After today's fixes |
|---|---|---|---|
| `yarn verify` | ten steps exit 0, headless inventory matches | compares filenames, not bytes; no release identity | unchanged, not run |
| Contributor contract | no new finding across 11 heuristic rules | aliased `assert`, indirect internal exports, a different loop in the same file | assert aliases, indirect exports, quote-dependent edges closed (TOOL-117, partly); cited host bounds now verified (CORE-118) |
| Packed consumer, export check | 11 entries load in ESM and CJS | a `.d.ts` with unresolved references; no `.cts` consumer | declaration-edge census and `.cts` consumer added; unverified until a build |
| Node suite (1,543.9 s) | one worker, specs pass | `allowOnly` focus (TOOL-111); NaN in 6 comparisons (TOOL-109) | both closed; suite not run |
| Headless suite (3,568.7 s, 1,409 files) | each file has a passed case | a passed case with no assertion; 9 cases return without a device (TOOL-110) | 9 returns replaced; scoped run 66 of 73 PASS |
| Kernel audit | a case per discovered export name | filter dropped budget findings; two recipes share one name; 2 producers undiscovered | filter removed (CORE-114), discovery extended (CORE-115), identity fixed (TOOL-112); 3 `GPUNetworkKFunction` cases fail |
| Numeric helpers | finite tolerances | 81 specs define their own; `NaN > x` is false | six sites, dense solve and eigen fixed (TOOL-109, CORE-117); compensation test backend-neutral (CORE-120) |
| Build record, promotion | input hashes, rollback, digests | tarball named by path; lockfile and compiler not hashed | tarball SHA-256 verified (TOOL-105); declaration inputs fingerprinted (TOOL-108); source identity and receipt not done |

**Cost.** 90.8 minutes reconstructed from separate runs: headless 59.5, Node 25.7, test typecheck 4.8 warm (13.2 earlier), source typecheck 0.24, build 0.17, exports 0.06, packed 0.13; no uninterrupted run was observed. **Proposed tiers:** 5 minutes (lint, checkers, source typecheck, warm build, export and packed checks, the changed Node case and one GPU regression; about 47 s of non-test work); 30 minutes (adds the one-worker Node suite and selected GPU cases); full gate (ten steps, about 91 warm minutes, durable receipt).

**Release readiness.**

| Item | State |
|---|---|
| Staged release, export check, packed consumer, rollback; 2026-11-18 gates as separate runs | Done, but on different trees and modes |
| TOOL-102 declaration edges; TOOL-101 CJS strings; TOOL-103 heap; TOOL-105, 108, 115 | Fixed today, unverified until `yarn build` and the packed check; one edge sits in a staging-owned file (annotated by the lead) |
| Two junk files in the package selection (TOOL-104) | Deleted; made by a review agent, not a repository defect |
| Full `yarn verify` on the exact tree | Open |
| Repository identity (`visgl` in the package, `BertCh` as remote); tag; compatibility statement; Arrow `>=17` | Open; decisions 1, 28 |
| Source and lock identity, dependency versions, qualification receipt | Open |
| Lenses private until `THIRD-PARTY-DATA.md` licensing is resolved | Open |

## luma.gl and scale

**Usage.** 1,658 published import declarations in 1,086 source files, 156 distinct (entry, name) pairs, no `_internal`, deep `dist`, default or namespace imports, no direct re-exports. By entry: `graph/gpu-core` 1,268, `graph/gpu-raster` 159, `core` 127, `graph` 41. The five grid readers use the alpha.16 coordinate function. `usage.mjs` joined names across entries; it now keys by entry (LUMA-105, fixed), and `diff.mjs` compares declaration hashes (LUMA-112, fixed); `scripts/upstream/README.md` still needs text.

**Alpha.16 offers, and verdicts** (luma report):

| Published | Verdict |
|---|---|
| Scan, sort, gather, reduction, compaction, COO to CSR, hash index | Adopted for matching contracts; keep local saturation, bounded publication, exact accumulation |
| Segmented reduction, segmented scan, scatter | Adopt now for ungated sums (about 60 lines, 23 calls; size S, M for the 3 gated calls) after empty, prefix, tail and order fixtures |
| Segmented sort, batch sort, segment offsets, unique, run-length | Contract change needed (dense keys, global sort, wide keys, GPU-produced segments) |
| `GPUJacobiPCG`, SpMV, CSR matrix | Cannot replace the resistance solve: a two-row system declares 51, 85, 153 nodes at budgets 1, 2, 4; 34,833 at 1,024; size L |
| Grid index, binning, BVH, point filter; histogram, group aggregation, hash join | Adopted for compatible cases; contract changes for batched candidates, stable order, exact or wide keys (LUMA-101 to 103) |
| FFT2D, convolution, transpose | Already adopted where matching; matrix products lack validity and calibration |
| Graph algorithms, traversal | Cannot adopt for caller-owned live CSR with sentinels and completeness (upstream draft 03); PageRank, core number: after CSR-view support |
| Arrow, tables, query, SQL, raster tile and morphology, similarity, DGGS, projection | Adopt matching conversions and morphology; fill holes, PMF radius, classifier ties and fused cell products need adapters |

**Register and drafts.** Four drafts are ready and none is filed: 02 histogram edge quotient (investigation first), 03 caller CSR and masks (retires about 101 lines), 01 bulk packed transfer (benefit unmeasured), 04 typed imports (up to 17 lines). Proposal 1 (public grid formula) can close; 2, 6, 9 stay; 3 to 5 narrow. Register rows 173 to 289 were not re-dispositioned; only rows a report called wrong were rewritten.

**alpha.17** was published 2026-11-18, one day after the pin (`beta` = alpha.17, `latest` = alpha.5; direction and luma reports). The peer range `~10.0.0-alpha.16` accepts it. Nobody assessed it in this review.

**Scale census** (luma report, 498 producers; the API census assigns 4 producers to `analysis-core` and 272 to spatial, this report 6 and 270):

| Family | Producers | Packed | Chunked | Tiled scratch | Host-paged |
|---|---:|---:|---:|---:|---:|
| spatial | 270 | 231 | 31 | 8 | 0 |
| raster | 86 | 58 | 27 | 1 | 0 |
| terrain | 46 | 45 | 0 | 1 | 0 |
| network | 41 | 40 | 1 | 0 | 0 |
| dataframe | 34 | 20 | 12 | 2 | 0 |
| point cloud | 11 | 7 | 4 | 0 | 0 |
| core, record, geospatial | 10 | 7 | 3 | 0 | 0 |
| **Total** | **498** | **408** | **78** | **12** | **0** |

At least 13 of 39 spatial chunk-capable producers still keep a global primary-axis allocation. First limits at the default 128 MiB binding: point-in-polygon join 33,554,422 points with two chunks; space-time kriging 8,388,608 samples; network cost table 87,838 vertices at 191 lanes; k-nearest classifier 524,288 pixels per chunk; region graph 5,792 regions. Today the 32,768-node admission rule was added where batch settings could declare 40,000 nodes (LUMA-118, fixed).

## The known-issues pass

**What it is.** A staging clone, `luma-contributors-ki` (base `9e93cbc7`), started 2026-11-18 20:07 after the previous pass showed that passing gates had not meant correct output. Roadmap rows KI-01 to KI-10. Codex streams: implementor `gpt-6.1-sol`, reviewer `gpt-6.2-astra`; verification is automated (biome, scoped `tsc`, named Node specs, both checkers, named headless specs on Metal). `ki-pass-digest.md` was read at 17:43 to 17:57 and `landing-plan.md` at 18:28; the pass kept running, so counts may have moved.

**116 defects in existing code.** Families: regression and inference 61, statistics and weights 9, network 10, geometry, topology, joins 27, raster, terrain, cost 9. Kinds: wrong result 52, wrong status 11, port accepted but never written 16 entries (137 ports), retained caller props 9, invalid input accepted 11, ignored loss flag 10, CPU twin 7. Fix state: claimed 88, confirmed by a later round 13, partly 10, open 5. Most serious: D014 (exactly affine data gets finite spatial-error inference, on Metal), D026 (error-ML p-value 1.8 percent off, on Metal), D034 (panel-lag within-lag loses 20 percent), D053 (panel-error ML reports success on exact fits, 3 of 98 GPU cases), D062 (Gamma and Pearson from truncated edges). Not fixed: 13 more unwritten destinations and a census backlog of 3,865 ports (622 with no independent oracle). New capability held: 2^24 regression tiles (LR-26), panel error effects (ST03), iterated SUR (ST04), spatial-lag probit (ST05), robust restriction (ST02), multipart predicates and forms (LR-08), binding CPU cost-distance budgets.

**Why nothing was landed.** Of 65 streams: 15 ready with unreviewed fixes (39 defects), 28 not ready, 7 read-only census, 15 not started. No stream ended with a clean review: all 29 first-round reviews returned FIX (97 blockers), and on re-read 30 of 53 first-round findings were fully fixed, 23 partly. Round-2 reviews died for 11 streams (an orchestrator fault; never retried). No log shows `yarn typecheck`, test typecheck, build, export checks, a whole Node project or `yarn verify` on the clone. 1,056 files differ from base (386 modified, 670 new). Under the strict file rule and a "stream lands whole" rule, 31 files qualify (oracle-network-twins: 7 source, 2 docs, 22 specs; D079, D080); the lead decided not to land them. The user has not answered whether the lead should take over the pass (decision 8).

**Hand merges.** Ten files changed in both trees at 18:28 (`landing-plan.md` section d; later edits may add more): `dev-docs/roadmap.md` (the clone adds one self-contained section), six pages under `docs/api/gpu-spatial-analysis/` (`operators-accessibility`, `chunked-spatial-weights-transpose`, `geographical-detector-redcap-entropy`, `movement-semantics`, `operators-dynamics`, `point-process`), two source files (`gpu-chunked-spatial-weights-transpose.ts`, `gpu-trajectory-locations.ts`, real conflicts with unfinished streams) and `website/src/catalog/entries/geometry-processing.ts`. Two lead edits to staging-owned files also need a merge: `docs/api/gpu-network/operators.md` (six hunks) and `spatial-autocorrelation-parameters.ts` (annotated `GPUSpatialParameterSchema<'float32'>`, the last TOOL-102 edge).

**Fourteen defects neither tree fixes** (changes are in the named record): SSR-107 panel LM sums, SSR-108 spatial SUR residuals, SSR-109 probit tolerance, SSR-110 plain GWR (`fix-stats-regression.md`); GEO-101 offset at tolerance 0, GEO-102 CPU zero-length fragments (`fix-geometry.md`); multivariate Geary and panel regression weights loss (`fix-stats-weights.md`); NET-109, NET-113 (`fix-network-geometry.md`); RT-130 (`fix-raster-timeseries.md`); `GPURasterWalkingCost` shear guard (`fix-raster-terrain.md`); LUMA-104, LUMA-116 (`fix-luma-scale.md`). Two test follow-ups: the staging test for AZP and Max-p loss never clears the flag; the RT-122 writer spec (`fixes/fix-terrain-hydrology/gpu-terrain-flow-output-writers.node.spec.ts`) must be copied in.

## Consumers

**Found** (`area-consumer-seams-lenses.md`). The saved website network-neighbourhood map gives the minimum colour to 1,150 nodes the recording marks as outside the network, and suppressed hot-spot scores likewise (SEAM-103). The Knox helper returns 0 for representable tails (SEAM-101); AZP graph operations drop weights loss and the CPU form can report convergence on the incomplete graph (SEAM-102); the showcase reported one of four true join pairs as 25 percent success and "no overflow" (SEAM-107). Lens tables held across 365 inspected saved tables; the kriging headline depended on process-global registration (LENS-101). Kepler has 229 bindings; a 40-binding audit found two handwritten CPU errors (negative demand, translated hole) and one unverified GPU completeness issue. The archive has 427 records and 384 scenarios with the full structural comparison; it does not establish numerical correctness, and the 43 network scenarios bypass the richer comparison (LENS-105).

**Fixed today.** SEAM-101, 104, 106, 107, 108, 115; LENS-101, 103, 105, 106, 108; SEAM-110 partly (additive `code`; count semantics are decision 14); the lenses kernel-source view reads each node's entry point and constants; the satellite scene reads candidate overflow (TRJ-104); the sampling-frames option was relabelled after PTS-112. Kepler: zonal distribution reruns on the CPU engine with a warning after zone-assignment loss.

**Needs a lens recapture.** 14 `scenario-declaration` cases (`VITE_CAPTURE_LENS_RECORDS=1`), then twins; six records gain a `presentation` field. Known graph changes: map colouring (`incomplete` nodes), zonal variance node ids, neighbourhood summary loss nodes, map-matching and snapping shader hashes. Four records kept run IDs across an upstream shader change (SEAM-113, open). More records change when the staging pass lands (OLS, GM, diagnostics, panel lag, exact ML, precision reduction).

**For the website's owner** (a peer session; not touched): SEAM-103, SEAM-105, SEAM-111 (the catalog describes MGWR as `{count, overflow}` and "zero means complete"), SEAM-112 (the display reads a cut capacity without `cutCertified`), website lens models and inventory regeneration.

## Direction

(`area-direction.md`; no GPU code run; only the luma.gl dates were re-checked.) **Position.** A kernel library, not a runtime: 504 contributors and 129 operation forms add WGSL kernels to a caller-owned `GPUCommandGraph`. Breadth is the strength; by script 130 of 633 classes have a CPU twin, 361 appear in a fixture-backed test, 385 in a lens scenario, 240 have a Kepler binding; 370 of 548 catalog operators are `unaudited`, 201 have no tier. The first GPU result takes about 55 lines; there is no file reader, GeoArrow adapter, worker story or Node GPU path, and no benchmark against a CPU library off one Apple GPU. The library requires no WebGPU features, so reach follows browser support (on by default in Chrome, Edge, Safari 26, Firefox on Windows and Apple Silicon). **Comparables.** cuSpatial is archived and the rest of RAPIDS has no geospatial operators and needs CUDA; GeoPandas and Shapely give exact GEOS predicates and I/O without a GPU; PySAL (v26.07) is the citable reference for spatial statistics, CPU only; WhiteboxTools, GRASS and Earth Engine cover terrain, raster and planetary scale without browser GPU compute. **Three bets.** Five finished journeys with a facade, GeoArrow and a two-vendor benchmark (DIR-101 to 104); a stable core with tiers, an admission rule and three gate levels (DIR-105, 108); a machine-readable contract on that core (DIR-109). **Three stops.** New operator families before external users (except three in DIR-107); a generic `input/output/run` runner (DIR-103); retrospective lens and website coverage outside the journeys. **Not re-checked.** All comparables (versions, dates, links) came from a web-research subagent; the browser-limit figures (web3dsurvey, undated) and whether a deck.gl layer can bind a caller-owned buffer without readback are unverified; the journeys rest on oracle availability, not demand.

## Decisions for the maintainer

Sources: lead notes (L), fix records (F), `roadmap-changes.md` (R), direction report (D). **Applied** means already in the working tree; the row says how to reverse it.

| # | Decision | Options and recommendation (by source) | Src |
|---:|---|---|---|
| 1 | Canonical package identity: the package and README name `visgl/luma-contributors`, the remote is the private `BertCh` repository; `0.1.0-alpha.0` availability | Choose owner, URL, name before publishing. Tooling: raise UF-01 to H. Direction: first of six decisions; a rename after publication costs every consumer | R, D |
| 2 | Hold the alpha | Tooling headline: hold until TOOL-102 is fixed and a `.cts` check exists. Both now pass on a staged release build, so the remaining holds are the package identity (1), the unreviewed fix diff and the known-issues pass (8) | R |
| 3 | Stability tiers and admission rule | Direction: adopt the stable and experimental split (DIR-105) | D |
| 4 | Which five journeys | Direction chose by oracle availability, not demand | D |
| 5 | luma.gl cadence and alpha.17 | Pin and bump on a schedule with a nightly run on `beta` (direction); record skipped versions (luma report) | D |
| 6 | Second GPU machine, Safari and Firefox runs | Needed before portability claims (DIR-104, DIR-111, FP64-05) | D, R |
| 7 | `LRN`, `EMB`, `SIM`, `CART`, `ENC` rows | Confirm three (DIR-107), drop the rest | D |
| 8 | KI pass: lead takes over now, after this review, or not; land by whole stream or by file | Lead recommends after the review | L |
| 9 | Required completeness status on trajectory thinning, staypoints, locations, mobility (breaks operation forms, Kepler bindings, specs, scenes); and whether unsettled locations suppress mobility indicators or a missing location key keeps counting as a category (changes semantics) | Fix agent lists callers, no recommendation; for thinning it is the only remedy left | F |
| 10 | Catalog default for unaudited records: (a) `'unknown'` in the all-entry catalog only (about 230 records change); (b) widen the legacy type (public type change); (c) keep `[]` and audit | Report proposes `unknown`; fix agent offers the three options | F, D |
| 11 | Regression status decoders (SSR-112): (a) additive per-model decoders, GGWR 0 to 6, `statusName`; (b) also a shared union; (c) one renumbered vocabulary | Fix agent recommends (a), not (c) | F |
| 12 | Signed zero in class breaks (DPC-110): (A) canonicalize zero in class breaks; (B) ordered keys in colour consumers; (C) keep both, document | Fix agent recommends (A) | F |
| 13 | Probit tolerance upper limit (SSR-109) | Proposed 1e-2 through `validateRequirement`; in a staging file | L |
| 14 | Regression recipe `invalidCount` holding a solver code (SEAM-110): (a) drop it; (b) publish a real excluded-row count | `code` added; no recommendation | F |
| 15 | p-median `costLimit` makes `complete` 0 for any finite matrix limit | Documented; alternative compares `uncoveredDemandCount`, weaker. Not blocking | F |
| 16 | 109 of 127 loop allowances cite no named bound | (A) add `bound` and `route` fields; (B) leave | F |
| 17 | **Applied.** Chunked transpose status shape (SSR-105) | Revert: one hunk in `gpu-chunked-spatial-weights-transpose.ts`; no in-repo caller read the old fields | L, F |
| 18 | **Applied, breaking.** `GPURasterZonalStatistics` default `parameterLayoutVersion` 2 (RT-101) | Revert: restore default 1; explicit `parameterLayoutVersion: 1` keeps legacy decoding | L, F |
| 19 | **Applied.** Quadrats with points but no sampled area give NaN (PTS-108) | Alternative: sampled-only total (drops points); a one-kernel change | L, F |
| 20 | **Applied.** Suppressed emerging hot spots labelled `UNDETERMINED` | No rename or removed export | L, F |
| 21 | **Applied.** Both ring orientations accepted (DGGS-105) | Alternative: reject clockwise (turns GeoJSON holes into status 1); `abs(winding)` is wrong | F |
| 22 | **Applied (checker).** A CAPITAL_CASE alias of a luma.gl constant counts as a domain limit | Otherwise two findings (`MAXIMUM_PERIODIC_REGISTRATION_DIMENSION`, `MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION`) | F |
| 23 | Rows to reprioritise before the alpha: LUM-30, LR-D4, R6.4 (H), GPU-13 docs, NW-SCC, GPU-11 and urban tessellation docs, STD-13 (high), FP64-05, UF-01 (H); defer LN-STAB, LN-V5 | Requests in `roadmap-changes.md`; none applied to the roadmap by the clerk | R |
| 24 | Roadmap structure: four sections, split limits and consumer backlogs, component IDs that mean both implementation and decision, 31 long cells; shorten `AGENTS.md` (ROAD-115) | Proposed by the roadmap report; the reconciliation spec pins phrases | R |
| 25 | Upstream policy (PLAT-04, PLAT-05) | File batch iteration, dry-run planning and public-only defects upstream; keep geospatial semantics here | R |
| 26 | STD-04 / STD-09 legacy status views; STD-07 / STD-15 naming list | Report text answers a different question than the rows | R |
| 27 | **Applied.** FP64-04 limit restored (roadmap-devdocs said close, core said keep); INT-06 and INT-01 closed | Lead may reverse | R |
| 28 | Publication boundary: lenses private; compatibility statement for alpha.16, later prereleases and Arrow `>=17`; whether source and the build record ship | Keep lenses private; decide the rest before the tag | R |

## What is still open

Size: S under a day, M one to two weeks, L more.

| Item | Why it matters | Size | Evidence |
|---|---|---|---|
| Independent review of today's fix diff (368 modified files, 56 new outside this folder) | 32 agents worked from reports; no reviewer has read it | L | `fixes/*.md` |
| Full Node suite and full headless suite on the tree that is committed | Only scoped runs were made (Method) | M | Method |
| Full lens recapture at integration | 14 records and 67 twins were recaptured; 34 more records hold stale shader text that no spec flags, and six gain a `presentation` field | S | `fix-lenses.md` |
| 18 deferred findings (14 fixed in neither tree) | Real defects in staging-owned files | M | Known-issues section |
| 12 open W and S findings with no record: SEAM-102, 103, 105, GEO-103, 104, 108, 110, PTS-102, DGGS-110, 111, LUMA-101, 103 | Nobody assigned; SEAM-103, 105 are the website's | M | Tables |
| Contract, test and maintainability findings with no record (of 142 C, 50 T, 33 M; for example CORE-110, 111, 119, 122, LUMA-107 to 111, LUMA-117, SEAM-111 to 113) | Gaps the plan may batch | L | Area reports |
| Unverified leads (23) and the device probes wanted | Two-point DBSCAN at 1e6 (CORE-102, done once), thin triangle (CORE-103), constant indicator, overflowing mean, variogram with one excluded high row (DPC-103, 109, 112), two-cell D8 cycle, one-iteration skeleton, exact observation beside a gap, distant-anchor relief (RT-123, 134, 131, 128), 4,096 by 1,024 panel LM (SSR-107), 4,000-row GWR at 1e6 offset (SSR-110), histogram edges on alpha.16 (LUMA-101), grid-index bounds query (LUMA-103), quadrat edges (PTS-109), zone-assignment loss on a real submission (SEAM-108, 109) | M | Area reports |
| Follow-ups named by fix agents and not done | `GPUS2PolygonCovering` has the same sign-only winding rule; time-window and composite packers (DPC-108); natural breaks share the DPC-111 span overflow; in-repo composers do not pass `costsComplete`; flow map marks a particle valid when the last step lands on a NaN cell; stream order and flow distance unaudited for the RT-123 cycle; overlay device-limit checks run after nodes (RCP-102); `GPULocalKriging` grid normalization | M | `fixes/*.md` |
| Partial fixes: TRJ-102 thinning, SSR-102 (about 12 consumers), TOOL-117 and 118 residuals, CORE-116 allowlist, DPC-123, CORE-118 baseline | Each named in its record | M | `fix-trajectories.md`, `fix-core-audit.md` |
| Roadmap restructure and the reprioritisations the reports ask for | The R19 section and plan are written; structure and priorities are the maintainer's | M | `roadmap-changes.md` |
| Alpha.17 assessment and bump cadence | Accepted by the peer range, unassessed | S | luma report |
| Second GPU vendor, Safari, Firefox | All GPU evidence is one Apple Metal adapter | M | DIR-111 |
| Upstream drafts 01 to 04 | Not filed | S each | `dev-docs/upstream-drafts/` |
| KI pass: finish, re-review the 11 died rounds, land | 116 defects held; 21 streams not started or just started | L | `ki-pass-digest.md` |

## Limits of this review

- **Partial first-pass coverage.** Each report lists what it did not open. Statistics: validation, scoring, dynamics, rate smoothing, segregation, entropy, region statistics, outlier detection and block cross-validation were sampled by the lead, not audited; GWR grid index, bandwidth search, most of MGTWR, GM, 2SLS and IV kernel bodies and tiled global WGSL were not opened. Geometry: line merge, simplification, straight skeleton, ratio hull, canonical equality, Delaunay insertion, overlay labelling, coverage dissolve and DE-9IM WGSL were not opened. Points: HDBSCAN, k-means, Ward, Dorling, natural neighbour, Vecchia conditioning and about 60 percent of the scan statistic were not opened or sampled. Cells and trajectories: encounters, zone-event and restriction kernels, Kalman, H3 boundary and fp64 measure WGSL and 14 of 24 recipe bodies. Network and raster second passes did not exhaust hydrology (15,646 lines), temporal files or 1,201 network test bodies. Other reports state their samples (62 claims, 40 error messages, 30 behaviours, 40 of 229 Kepler bindings, 466 unresolved binding arrays).
- **Two model families, no human reading.** Claude Fable leads with Opus subagents, Codex `gpt-6.2-astra` with `gpt-6.1-sol`, Claude agents for the fixes; the geometry report names a Sonnet-tier lead. Reports disagree with their own tables in places (under the Files table).
- **One GPU.** WebGPU in Chromium on Apple Metal only; no other vendor, Safari or Firefox. Metal reciprocal-division and reassociation behaviour produced several defect classes. Reviewers ran no GPU code; device results are the fix agents' and the scoped run's.
- **Fixes no independent reviewer has read.** 123 fixed findings rest on their agents' specs. A test fixture error caused three of the five run failures (DPC-109); one fix changed a limit and broke three audit cases (NET-110). A PASS says a file passed, not that each new case ran.
- **The review's own error rate.** No finding was refuted whole. Six were narrower than reported: NET-102 (GPU form unaffected for 1 to 60 departures, CPU twin only), URB-101 (real only for tables with equal rows or small n), SSR-102 (one of about 18 consumers was compliant), RT-102 (six terrain classes handle shear; the guard went on 15), DOC-102 (docs wrong, code right), CORE-114 (no registered kernel exceeds a budget). NET-112's requested fix was impossible. TOOL-104 was made by the review. The roadmap clerk could not reproduce counts in several rows (STD-07 61 of 132 against 129; 748 against 790 fixtures; 15 terrain classes against 17 class files).
- **Units and patterns.** The direction report counts 548 operators and 633 GPU classes; the API report 555 records, 631 public GPU classes and 498 producers. Counts come from `rg`, regexes and token scripts; the evidence-type split and my class grouping are approximate; follow-through totals add rows the reports define differently.
- **State.** The Method and open-items sections were brought up to date by the lead at the end of the day; the tables of findings carry the state the fix records gave when the draft was written, and the staging pass was still running.
- **Web facts** in the direction report come from a research subagent (accessed 2026-11-19); only luma.gl release dates were re-checked. A search that found nothing does not show that no comparable exists.
