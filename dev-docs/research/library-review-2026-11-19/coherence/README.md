# Library review, second round: a coherent and ergonomic library (2026-11-19 and 2026-11-20)

Thirty-two independent read-only reviewers read the package under `modules/analysis` for coherence and ergonomics: one cross-family theme each, one family each as a newcomer, and one defect class each across the whole library. This file synthesizes their reports; the first round, mostly about wrong results, is in [../README.md](../README.md).

## Summary

1. **Wrong results are still the largest kind of finding, and most are reachable by ordinary input.** Recounted from the 32 findings tables: 493 findings, of which 241 are W (wrong result), 34 S (visible failure), 65 C, 65 E, 30 D, 5 T, 4 M, and 49 carry a mixed kind such as `C/W` or `W, S` (25 of those name a W). Reach: 343 ORDINARY, 117 UNUSUAL, 33 EXTREME. Change: 422 additive, 71 breaking. Of the single-kind W and S rows, 158 of 275 are ORDINARY; the first round graded 23 of its 118 W and S findings ORDINARY. The reviewers were asked to find incoherence, and incoherence mostly turned out to change answers.
2. **The ordinary-reach wrong results worth knowing about** (each input and result is quoted from the finding):
   - NAM-01: `percentiles: [.95]` means the 95th percentile in grouped statistics and the 0.95th percentile in network travel times; on a two-departure fixture the answers were 5 and 90.
   - NAM-03: `ties: 'lowest-id'` selects the lowest feature row; with IDs `[99, 1]` at equal distance the nearest join returned 99 with both loss flags clear.
   - CMP-01: a group table one row short of capacity fed a key join and OLS; the slope moved from 10.2 to 2, solver status 0 in both cases.
   - CMP-03: an absent cell gathered from a sparse count table became 4,294,967,296 observations at the emerging hot-spot input.
   - UNW-01: shortest-path search then path extraction named edges costing about 9.1 for a route whose reported cost was 0.3, because a binary64 sum was compared with a rounded float32 label.
   - UDF-01: two Arrow string columns prepared independently both coded `['A','B']` and `['B','A']` as `[0,1]`; with right values `[20,10]` the join returned `[20,10]` where `[10,20]` is correct.
   - UCO-01: a native field `[null, 0, NaN, undefined, 12]` mapped to `[0, 0, null, null, 12]`; the expected result is `[null, 0, null, null, 12]`.
   - UST-02: reusing the pre-standardization weights handle for masked Moran gave I = +0.02857143; re-standardized, -0.03571429.
   - STA-01: two adjacent squares with candidate loss; passing the producer result intact to the weights algebra cleared the flag.
   - ING-01: a documented Vecchia query outside the sample extent returned the same site three times and omitted two nearer ones.
   - UPT-01: the Hawkes CPU twin reported `converged=1` at iteration 25 while reducing only the decay rate by 10 percent raised the log likelihood from -8.7896780 to -8.7031654.
   - FLT-08 and FLT-10: translating the grid origin to 10,000,000 changed a Brownian-bridge density by +171.83 percent and a line-integral-convolution value by +160.34 percent (host model, device result predicted).
   - EGR-01: bytes from `Buffer.readAsync()` passed to a public decoder gave 44 nodes for 300.
3. **A small number of causes recur across themes.** Evidence of loss does not travel with the result (CPL 16 findings; STA, CMP, FST, UGE add more). Float32 range and origin (FLT 15, WGS 15). Packers that validate a double and write something else (PCK 16, VAL, LIF). Outputs accepted but not written (WRT 13, OUT, FRM, TYP). Byte-versus-word readback (OUT-02, EGR-01, UNW-05 describe the same defect). One word with two meanings (NAM, UNI, CMP-13). Retained state reinterpreted by a later call (LIF-01, LIF-02, SCL-02). The same defect was found from different themes: LIF-02 and SCL-02 are one streaming-unit defect, with two probes that disagree on the decoded value (3 where 2 is expected in LIF-02; 1 or 2 in SCL-02).
4. **The cross-family seams are weaker than the single operators.** The f-reports found single operators agreeing with independent references and the failures between them: "the most serious failures appeared between operations" (f05). Of 493 findings, 247 come from the cross-family theme reports, 128 from the family reviews and 118 from the sweeps.
5. **Newcomers meet a wrong default or wrong example before a wrong kernel** (family reports; first finding each):
   - Statistics: the weights tutorial passes the wrong handle (UST-02); two empirical-Bayes smoothers accept negative counts, giving a smoothed rate of -1.033333 for events `[-10,2,2,2]` (UST-03).
   - Geometry: identity multipart offsets change polygon counts, and spare feature slots become unmatched features (UGE-02).
   - Points: the Hawkes fit above (UPT-01); SKATER labels `0` and `2` size a downstream evaluation for two labels.
   - Cells and trajectories: raster patch labels passed unchanged count background as a patch; a ten-minute stay spans a missing GPS fix (UCT-04).
   - Network: an unsorted two-edge list changes a cost from 5 to 3 (UNW-04); the accessibility example seeds both ends of a one-way street.
   - Raster and terrain: NDVI of the geometric median was 0.26006 and median NDVI 0.33333 for the same three observations, and a guide interchanges the two workflows (URT-01).
   - Dataframe and point cloud: UDF-01 above; a statistical-filter mask addresses a different point on every row (UDF-03).
   - Consumers: UCO-01 above; Kepler blocks a CPU clustering job at 6,324 rows that its own runner completed at 8,000, and its discovery panel omits 176 catalog records (UCO-10).
6. **The defect-class sweeps found 118 findings in eight classes** (table in "Recurring defect classes"). The strongest single results: five packers overwrite the next record in a caller-owned array and 17 more drop the last word of their own record while reporting success (d03); 19 of 59 CPU twins accepted a zero-row batch and 40 rejected it (d07); an untested Pettitt cell contributing `p = 1` moved a neighbouring cell's adjusted p-value from 0.03141 to 0.06282 (DGN-01).
7. **Validation reaches the constructor for counts and stops short of element types.** Of 409 producers tested with a replacement packed view, 52 accepted a format outside the port's declared contract; none of 1,117 wrong-format refusals named the supplied format (c06). Of 498 producers, 45 accept `{}` for the output type despite a runtime requirement to select an output (c05).
8. **Completeness has no common spelling.** Of 498 producers, 197 accept a named `output.status`, 94 a raw status view and 207 neither; of 291 conventional destinations, 99 are optional (c03). Exactly one producer declares both `status.count` and `validity`, and there the two index different row domains (c02).
9. **What is strong** (report statements, not inferred):
   - c09: all 128 CPU graph operations have a GPU graph operation, a GPU contributor and a public array function; 127 pairs name the same props type. 25 lens examples matched their recorded output.
   - c11: all 498 producers export a `<Class>Props` type; all 67 generic producers default every type parameter; of 421 directly typed `GraphDataView` fields only `GPUTransferMask` exposes the unrestricted format.
   - c12: all 112 named bindings in 56 explicit analysis imports in API code fences resolved from the stated entry.
   - f01 to f05: CPU twins and oracles matched independent references for OLS, spatial 2SLS, both spatial ML fits, local Moran, Gi*, density clustering (every assignment equal to scikit-learn after mapping `0xffffffff` to -1), shortest-path costs, OD costs, centrality, integral flow and map matching, and the CPU predicate join, nearest join, clipped length and triangulators against Shapely.
   - f08: all 229 registered Kepler bindings resolve to a hand-written descriptor; the generic fallback accounts for none.
   - c08: `GPUKShortestPaths` shares allocation and decoding dimensions and keeps interruption flags; it is the reviewers' model decoder.
   - c01: GPU and CPU class prefixes are consistent; the census found five `criterion: 'queen'` fields and zero `'Queen'`.
10. **For the maintainer:** 21 proposals remove or rename a public name or change an output's shape or type; 17 result corrections were decided; 12 distinct findings have their fix in `website/`. 33 findings tagged breaking in their own table appear in neither decision file (listed in "For the maintainer").

## Method and limits

- **Who.** 32 Codex `gpt-6.2-astra` reviewers, one theme each, briefed for "a big focus on a coherent and ergonomic lib". Sixteen cross-family themes (`c01` to `c16`), eight family reviews written from a newcomer's side (`f01` to `f08`), eight defect-class sweeps (`d01` to `d08`). Four further files (`conventions-proposed.md`, `decisions-for-maintainer.md`, `lead-decisions.md`, `website-items.md`) collect or route the reports' content.
- **Read-only.** The reports state that no files were written. One exception to check: d07 reports that an initial unbounded diagonal label-point probe was still running as Node PID `28095` at its last check and that the sandbox denied termination; later probes used an 800 ms deadline.
- **"Executed".** Host runs of public functions, CPU twins (`compute*OnCPU`), parameter packers, Arrow preparation and catalog helpers; graph declarations on the null device; float32 transcriptions of shader arithmetic; generated WGSL read as text. By text match on the finding subsections, 465 of 493 contain the word EXECUTED, 248 contain SOURCE (a shader-result prediction with a named device probe), 227 contain both, 21 contain SOURCE only, 7 neither. The match does not say how much of a claim was executed.
- **No GPU ran.** No report ran a GPU kernel, a browser, a build, a compiler gate or a test suite. Every device-result claim is a prediction. Several reports say so per finding (`SOURCE` plus a probe); f08 and c09 note that the GPU probes remain unverified on the review machine.
- **Counts depend on a moving tree.** c06 and d05 state that the checkout (or the staging copy) changed during their review, so line numbers and totals describe observed snapshots. Reports checked the newer staging copy (`KI`) for staging-owned files; those fixes are not in the main tree.
- **Findings are claims, not verdicts.** Findings are not deduplicated across reports, and the reports do not describe a second reader confirming individual findings.
- **Not read** (aggregated from the 32 `## Not read` sections). No report read every shader, recipe, lens, catalog description, test or example. Specific gaps: c03 cannot say whether its 222 catalog records of unknown status lack loss semantics; c06 and c05 did not read all 498 implementations; c09 did not prove every stated tolerance; c12 did not read API pages line by line; c14 did not review all 761 classes semantically; c16 did not audit all 359 catalog bounds or every private kernel; d03's count of 17 short-target packers is a demonstrated count, not an upper bound; d06 treated 652 array declarations and 477 barrier calls as lexical leads and established no out-of-bounds read or divergent barrier; d08 executed 79 of 128 CPU twins; d05 did not audit regression, likelihood, GWR or autocorrelation; f08 did not read all 229 runner bodies. The family reviews (f01 to f08) sampled the long reference pages by user journey.

## Files

In `dev-docs/research/library-review-2026-11-19/coherence/`. Counts come from each file's findings table. W, S, C, E, D, T, M count rows with a single kind; Mix counts rows with two kinds (`C/W`, `W, S`); Total is all rows; ORD is reach ORDINARY; Brk is any change cell that begins with BREAKING.

| File | Theme | W | S | C | E | D | T | M | Mix | Total | ORD | Brk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c01-naming.md` | Naming | 9 | 1 | 4 | 2 | 0 | 0 | 0 | 0 | 16 | 13 | 4 |
| `c02-output-shapes.md` | Output shapes | 7 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 14 | 12 | 1 |
| `c03-status-vocabulary.md` | Status vocabulary | 7 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 14 | 12 | 1 |
| `c04-units-coordinates.md` | Units and coordinates | 14 | 0 | 2 | 1 | 2 | 0 | 0 | 0 | 19 | 15 | 1 |
| `c05-defaults-required.md` | Defaults, required inputs | 5 | 3 | 3 | 5 | 0 | 0 | 0 | 0 | 16 | 16 | 5 |
| `c06-validation-errors.md` | Validation, errors | 6 | 2 | 2 | 2 | 0 | 1 | 0 | 1 | 14 | 10 | 0 |
| `c07-ingress-packers.md` | Ingress, packers | 12 | 3 | 1 | 3 | 1 | 0 | 0 | 0 | 20 | 9 | 1 |
| `c08-egress-decoders.md` | Egress, decoders | 7 | 2 | 1 | 5 | 0 | 0 | 0 | 0 | 15 | 9 | 2 |
| `c09-forms-parity.md` | CPU, GPU, graph forms | 5 | 2 | 3 | 3 | 1 | 1 | 0 | 0 | 15 | 15 | 2 |
| `c10-composition.md` | Composition | 8 | 0 | 1 | 2 | 0 | 0 | 0 | 3 | 14 | 10 | 1 |
| `c11-types.md` | Types | 6 | 0 | 7 | 3 | 0 | 0 | 0 | 0 | 16 | 12 | 9 |
| `c12-entry-points.md` | Entry points | 3 | 2 | 3 | 3 | 1 | 1 | 1 | 0 | 14 | 10 | 3 |
| `c13-first-run.md` | First run | 7 | 0 | 1 | 4 | 1 | 1 | 0 | 0 | 14 | 12 | 0 |
| `c14-docs-structure.md` | Documentation structure | 3 | 3 | 1 | 4 | 3 | 0 | 0 | 1 | 15 | 15 | 0 |
| `c15-lifecycle-reuse.md` | Lifecycle, reuse | 5 | 1 | 2 | 3 | 3 | 0 | 1 | 0 | 15 | 7 | 1 |
| `c16-scale-variants.md` | Scale variants | 3 | 4 | 1 | 2 | 4 | 0 | 0 | 2 | 16 | 11 | 1 |
| `d01-float32-magnitude.md` | Float32 magnitude | 14 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 15 | 11 | 1 |
| `d02-completeness-travel.md` | Completeness travel | 12 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 16 | 9 | 1 |
| `d03-host-packers-borders.md` | Host packers, borders | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 16 | 7 | 1 |
| `d04-writers-reuse.md` | Writers, reuse | 6 | 0 | 3 | 3 | 0 | 1 | 0 | 0 | 13 | 11 | 1 |
| `d05-statistical-numerics.md` | Statistical numerics | 10 | 0 | 1 | 0 | 1 | 0 | 0 | 3 | 15 | 6 | 9 |
| `d06-wgsl-hazards.md` | WGSL hazards | 11 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 15 | 3 | 0 |
| `d07-degenerate-inputs.md` | Degenerate inputs | 7 | 2 | 2 | 0 | 1 | 0 | 0 | 2 | 14 | 8 | 2 |
| `d08-twin-disagreement.md` | Twin disagreement | 7 | 1 | 0 | 1 | 1 | 0 | 0 | 4 | 14 | 5 | 9 |
| `f01-statistics.md` | Statistics | 7 | 0 | 3 | 1 | 0 | 0 | 1 | 6 | 18 | 13 | 1 |
| `f02-geometry.md` | Geometry | 7 | 0 | 1 | 3 | 1 | 0 | 0 | 4 | 16 | 11 | 5 |
| `f03-points.md` | Points | 5 | 2 | 4 | 2 | 3 | 0 | 0 | 0 | 16 | 11 | 1 |
| `f04-cells-trajectories.md` | Cells, trajectories | 4 | 0 | 0 | 0 | 2 | 0 | 0 | 10 | 16 | 14 | 2 |
| `f05-network.md` | Network | 8 | 0 | 1 | 2 | 2 | 0 | 0 | 2 | 15 | 12 | 2 |
| `f06-raster-terrain.md` | Raster, terrain | 5 | 1 | 2 | 0 | 0 | 0 | 0 | 6 | 14 | 11 | 2 |
| `f07-dataframe-pointcloud.md` | Dataframe, point cloud | 12 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 16 | 10 | 1 |
| `f08-consumers-view.md` | Consumers, Kepler, lenses | 8 | 2 | 2 | 4 | 1 | 0 | 0 | 0 | 17 | 13 | 1 |
| **Sum, 32 tables** | | **241** | **34** | **65** | **65** | **30** | **5** | **4** | **49** | **493** | **343** | **71** |

Sub-totals: c reports 247 findings (188 ORDINARY, 32 breaking), d reports 118 (60, 24), f reports 128 (95, 15). Also in the folder: `conventions-proposed.md` (each coherence reviewer's convention, collected), `decisions-for-maintainer.md` (21 proposals), `lead-decisions.md` (17 result corrections), `website-items.md` (14 entries, 12 distinct IDs).

Cross-check against the lead's totals (514 findings; 240 W, 34 S, 65 C, 65 E, 30 D, 5 T, 4 M; reach 305, 107, 31; 378 additive, 65 breaking). The lead's kind counts add to 443, not 514. They match the 444 single-kind rows here to within one row (241 W, not 240; 306 ORDINARY among those rows, not 305; S, C, E, D, T, M and the 107 and 31 agree), so they leave out the 49 mixed-kind rows. Breaking: 65 among the single-kind rows, 71 over all rows (62 plain BREAKING and 9 qualified, such as "BREAKING results" or "BREAKING numeric correction"). Additive: 379 among single-kind rows, 422 over all rows. I could not reproduce 514: the tables hold 493 rows, and the `###` finding subsections also number 493 (c10 has an extra heading for its journeys).

## Conventions proposed

The reviewers each proposed one convention; `conventions-proposed.md` collects them. Distilled, with the departure each reviewer counted. The implementation waves adopted these as rules for code they touched, in additive form only: old names kept and marked deprecated.

| Convention | Rule | Departure counted |
|---|---|---|
| Options and values (c01) | Name an input for role, unit and index space; one spelling and one finite value set where those agree; lowercase hyphenated strings for multiword options; the reference algorithm's searchable name for a class. | 16 findings; 1,297 distinct top-level prop names; 668 class-prop pairs take string literals |
| Units and index space (c04, d01) | Every spatial or temporal port declares quantity, representation, unit or source of units, and missing-value rule; fixed-unit scalars get a suffix; differences are taken in a local frame before narrowing. | 19 findings, 14 W; FLT 15 |
| Outputs, counts, validity, tails (c02, d04) | Each writable port declares format, row domain and order, readable extent, tail policy and missing-value rule; `status.count` bounds the compact prefix, `...Rows` indexes a source, `...Ids` carries identities; every advertised optional port has a writer, a declared pass-through or construction-time rejection. | 14 findings; 1 of 498 declares both `status.count` and `validity`; 356 producers have optional outputs |
| Status vocabulary (c03, d02) | A producer that can return a plausible partial answer exposes `status.complete` for its declared scope with separate causes; a consumer forwards evidence, suppresses the dependent result or refuses. | 197 named, 94 raw, 207 neither; 99 of 291 destinations optional |
| Decoders (c08) | `decode<Producer><Output>(readback, layout?)` beside each producer whose output depends on packing, counts, offsets or status; bytes and elements distinguished. | 10 of 20 traced producers have a byte-ambiguous decoder |
| Validation and structured errors (c06, d03, d07) | Check every host-knowable requirement at construction, before any graph or destination mutation; refusals are one `AnalysisValidationError` with `code`, `contributor`, `id`, `path`, `received`, `expected`; validate a whole packer record before the first write; accept logical empty batches. | 52 of 409 accepted a wrong format; 0 of 1,117 named it; 8 packers return an undersized destination as success |
| Defaults shared across forms (c05, c09) | A minimal typed call computes the selected mode or names the unmet requirement; one selection fixes both accepted props and computation; CPU and GPU forms resolve defaults alike. | 45 producers where `{}` satisfies the output type; 13 CPU operations share a destination name between ports |
| Ingress (c07) | Adapters named `prepare<Domain>Input` return `{arrays, formats, counts, metadata}` with live rows distinct from capacity. | 214 packers, 7 public `prepare*` functions |
| Limits and scale variants (c16) | `execution` for equivalent routes, `method` or `accuracy` for changed guarantees; constants `<OPERATION>_<ROUTE>_MAXIMUM_<QUANTITY>`, `..._AUTO_..._THRESHOLD_...`, `DEFAULT_<QUANTITY>`; limits queryable by a side-effect-free plan. | 16 findings; three planners reject work that fits an existing route |
| Types (c11) | A constructor accepts its exported props type; selected ports captured once; unknown fixed-schema keys rejected; declared and returned keys agree. | 16 findings, 9 breaking |
| Lifecycle (c15) | Capture validated values at construction; bind retained results to the metadata that gives them meaning; define reset, invalidation, cleanup owner and teardown order. | 7 of 498 producers expose `destroy()` |
| Entry points and first run (c12, c13) | One canonical import per supported declaration; `import` and `require` agree on identity; five runnable first-run programs with discriminating inputs. | 11 entries, 200 barrels, 5,416 declarations; 22 of 70 family pages omit the package path; 1 of 5 first-run tasks has an end-to-end program in the docs |
| Documentation structure (c14, f01 to f08) | One task-to-contract record (question, qualified export, mode, result shape, section) generating the chooser, cross-reference and catalog aliases; one operation-entry template. | 165 of 311 contributors and recipes named in the spatial index; 94 of 120 API pages reachable by body links; 1 of 15 task queries returned a result from catalog search |
| Numerical contract (d05, d08) | State the admissible sample, estimator, tail construction and undefined-result policy; discrete decisions (ranks, selected IDs, masks, exceedance counts) are exact on CPU and GPU. | NUM 15, TWN 14; 79 of 128 twins executed, 14 issues in 15 twins |

## Recurring defect classes

Counts are the sweep's own findings; "also seen" lists IDs from other themes describing the same class (I did not search for every instance).

| Class | Sweep | Also seen | Check that would catch it |
|---|---|---|---|
| Completeness evidence lost at a handoff | CPL 16 | STA-01 to STA-04, CMP-01, CMP-07, CMP-09, FST-05, UGE-02 | Pass the producer result unchanged to the consumer in paired clean and lossy fixtures; assert the downstream number and the surviving cause |
| Float32 range, origin and cancellation; WGSL arithmetic | FLT 15, WGS 15 (9 EXTREME) | UNI-03, UNI-09, PCK-05, UPT-03, SCL-01, UCO-03 | Run each fixture twice, translated by 1e7 and scaled by 2^30; assert discrete decisions exactly |
| Packer writes something other than it validated | PCK 16 | VAL-02, VAL-06, LIF-03, DEF-04, ING-08 to ING-10 | Target with nonzero offset and distinct sentinels around it; repeat with an invalid final field |
| Output accepted but not written | WRT 13 | OUT-01, OUT-04, FRM-01, FRM-05, TYP-03 | Poison the destination, submit a larger then a smaller result, check values, status and neighbouring guards |
| Byte versus word readback | (c08: 10 of 20 traced) | OUT-02, OUT-06, EGR-01, EGR-02, UNW-05, UNW-07 | Decode starting from the exact `Uint8Array` that `readAsync()` returns |
| Same word, two meanings | (c01) | NAM-01, NAM-04, NAM-05, UNI-04, UNI-05, CMP-13, NUM-05 | Pass a nondefault value through both sibling entry points with a fixture whose interpretations differ |
| Unknown option selects a different algorithm | (c01, c06) | NAM-06, NAM-07, NAM-09, VAL-04, URT-06, UCT-06, FST-04 | Pass an unknown value through the JavaScript boundary |
| Retained state reinterpreted | (c15, c16) | LIF-01, LIF-02, SCL-02, LIF-04, FRM-04 | Two-submission test that changes one setting between submissions |
| Failed construction leaves residue | (c06, c15) | VAL-08, SCL-08, LIF-11, EGR-09, URT-07 | Correct the prop and retry on the same graph and ID |
| Degenerate and boundary inputs | DGN 14 | ING-13, NUM-03, NUM-04 | Zero rows, one-short capacity, constant column, `180` versus `-180`, an all-missing cell |
| CPU and GPU twin disagree on a discrete decision | TWN 14 | FRM-02, FRM-03, DEF-13 | Independent exact expectation for ranks, IDs, masks, counts, with tolerance only on floats |
| Statistical convention | NUM 15 | DEF-01, DEF-02, UST-04 | Reference-value fixture at the boundary that separates plausible formulas |
| Documentation example gives the wrong answer | (c13, c14) | FST-01 to FST-05, DST-01 to DST-05, UNW-02 to UNW-04, UST-02, CMP-10 | Typecheck and run the displayed source with discriminating inputs |

## For the maintainer

`decisions-for-maintainer.md` holds 21 proposals; the reviewer's recommendation is given.

**Renames (9).** Take the rename before alpha in each case.
- NAM-13: `GPUSpatialKMeans.k` to `clusterCount`; neighbour APIs may accept `neighborCount`. 47 spellings.
- NAM-14: `GPUDBSCAN`, `GPUHDBSCAN`, `GPUPycnophylacticInterpolation`, with DBSCAN `minimumSamples`. 138 spellings.
- OUT-11: `GPUPairGather` `output.rowCount` to `output.status.count`. 41 sites.
- STA-13: five `_STATUS` index maps to `_STATUS_LAYOUT`, with aliases first. 265 references.
- CMP-13: resident colour-scale `domainCount` to `classCount`. 20 sites.
- UST-15: `symmetrize` to `averageReciprocalWeights`. 3 direct calls.
- URT-10: hydrologic `slope` to `descentTangent`. 5 constructor sites.
- UDF-14: `GPUKeyJoin` to `GPUKeyLookup`. 51 constructors.
- ING-20: positional `prepareGPUNetworkRestrictionStates` to an options object. 3 calls.

**Removals (4).**
- ENT-10: parent barrel's deprecated re-export of 196 ontology names; 69 named imports migrate.
- ENT-13: three regression WGSL exports; no public-import call sites.
- LIF-09: `reusePreparation` on the conditional-simulation operation; no edits needed.
- UCO-10: replace Kepler's bare-name catalog discovery with `{package, exportName}` and drop the supplemental passes; 13 calls.

**Shape or type changes (8).**
- FRM-06: forward `cellSizeMode` in the drainage recipe, which changes distance units for geographic input.
- TYP-09: narrow five transfer-mask bindings to `uint32`.
- TYP-11: require one of five derivative outputs; type the distance-field seed alternatives.
- TYP-12: literal keys and checked lookups for three definite-looking lookups.
- TYP-13: `Readonly` on frozen spectral props and operation descriptors.
- TYP-14: replace the writable `recorder.graph` getter with an inspection lookup.
- TYP-15: require all four outputs in `GPUPopulationSupportAlignmentProps`; split a recipe props type.
- WRT-13: narrow `GPUPickRegionMask` output to `mask` plus overflow status.

Disagreement to note: NAM-03 is BREAKING in the c01 table and its larger proposal says to rename the selector, while `website-items.md` carries an additive form (`'lowest-row'` canonical, `'lowest-id'` deprecated). OUT-11, NAM-13 and STA-13 each list an additive alias as an alternative that the reviewer declines.

**33 findings are tagged BREAKING in their table and appear in neither decision file:** NAM-03, NAM-10, UNI-07, DEF-01, DEF-03, DEF-07, EGR-04, EGR-05, FRM-02, TYP-02, TYP-04, TYP-06, ENT-01, CPL-09, PCK-04, NUM-02, NUM-03, NUM-06, NUM-10, NUM-12, NUM-13, NUM-15, TWN-04, TWN-08, UGE-03 to UGE-07, UCT-03, UCT-08, UNW-01, UNW-05. Most are result corrections; I did not check which were routed elsewhere.

**Decided result corrections (17), `lead-decisions.md`.** The file states the choice for DEF-08, DEF-13 and FLT-04; for the others it records the reviewer's recommendation under the instruction to implement every form as a recorded behaviour fix.

| ID | What changes for a caller |
|---|---|
| DEF-08 | Exponential or power network decay without `beta` is refused; explicit `beta: 0` stays valid |
| DEF-13 | Hawkes CPU and GPU share the GPU default iteration budget (100, not 1,000 on CPU) |
| SCL-01 | Streaming variance corrected; `[10000000, 10000001]` must decode to 0.5 in both orders |
| FLT-04 | `GPUSpatialKMeans.centerCoordinates` defaults to true (254 becomes 254/255 in the 255-row case) |
| NUM-08 | Mobility centre is spherical by default; arithmetic centring by name (radius 19,904 km becomes about 111 km) |
| NUM-09 | Atkinson index is 1 at `epsilon >= 1` for a positive mean with a zero income |
| DGN-01 | Untested Pettitt cell returns NaN instead of `p = 1` |
| DGN-02 | Staypoint and location centres give `180`, `-180` and `540` one vote |
| TWN-01 | Packed quantile ranks round the product to float32 first (`0.7`: row 7, not 6) |
| TWN-02 | One-band medoid takes the earliest valid acquisition on an exact tie |
| TWN-03 | Envelope exceedance keeps a residual (GPU `p` 1 becomes 0.05, as on CPU) |
| TWN-05 | Track match, location ID and fold usability use the retained-precision radius decision |
| TWN-06 | A sole retained observation is returned directly in geometric median |
| TWN-09 | Profile peaks use float32 prominence for acceptance and ordering |
| TWN-10 | Radiation normalization sums non-origin population directly |
| UPT-01 | Hawkes decay update includes the finite-window compensator derivative |
| URT-02 | Visibility samples only positive-weight corners; viewshed and line of sight agree |

**Website.** `website-items.md` lists 14 entries (12 distinct IDs: NAM-03, NAM-04, DEF-01, EGR-04, EGR-05, FRM-13, DST-07, DST-13, CPL-09, PCK-04, DGN-01, URT-01) in `website/`, which this pass did not edit.

## Larger proposals

Deduplicated from the 32 "Larger proposals" sections. Reports that asked for each are in brackets.

1. **One declarative description of each operation and port** [c06, c09, c11, c05, c02, c16, f08]. Named constructor port descriptors (format, presence, topology, row relationship) that also produce the structured error and the negative test (c06); one validated snapshot of selected ports driving declared outputs, lowering bindings and CPU return keys (c11); entry-qualified forms and semantic options shared by constructors, catalog, family page and parity test (c09); a qualified versioned invocation contract for Kepler (f08); row-domain metadata beside output declarations (c02); `Contributor.limits` (c16).
2. **Structured validation error** [c06]. `AnalysisValidationError` before alpha, with end-anchored message assertions migrated first, plus single-prop mutation probes in contributor acceptance.
3. **Completeness that travels, and a generic status decoder** [c03, d02, c10, c02, d04, f02, f04]. `decodeGPUAnalysisStatus(layout, readback, row?)` returning `complete`, `incomplete` or `unknown` with scoped causes (c03); producer-specific writable status types, starting with the three weights builders (c02, d04); a consumable result port carrying rows, identity and evidence (c10, d02, f02); evidence retrievable when destinations are omitted (c03).
4. **Pure decoders beside producers** [c08, f05, f01]. Shared output layouts for the ten byte-ambiguous producers, modelled on `GPUKShortestPaths`; checked word and byte readback and a flow summary reader (f05); descriptive readers for global statistics (f01).
5. **Domain input adapters** [c07, f07, f05, f03, c13]. GeoArrow geometry ingress first (c07); role-aware Arrow keys and measures with shared dictionaries (f07); a road-network preparation function keeping application IDs and parallel edges (f05); a contributor-bound packer that checks frame settings against the grid (f03); a direction-explicit facility-search topology helper (c13).
6. **A front door and first-run path** [c13, c14, c12, f01, f05, f06]. Five runnable entry modules with discriminating inputs shown from their source (c13); one task-to-contract record generating chooser, cross-reference and catalog aliases, and one operation-entry template (c14); an asymmetric road-network walkthrough (f05); a weights walkthrough from `W` and an indexed column to a named result (f01); a checked raster and cube journey (f06); import guidance generated from the manifest (c12).
7. **Frame, unit and placement metadata carried through pipelines** [c04, f06, f04, c01, f03, d01]. Frame and validity through classified points, DEM, flow and recorded raster (c04); origin-aware frame identity and a metric-result comparison (c04); named raster placement objects with `origin` and `cellSize` (c01, NAM-04; line-density raster metadata in f03); one signed raster edge-ownership calculation and retained local state (d01).
8. **Side-effect-free execution plan** [c16, f08]. `getExecutionPlan(shape, deviceLimits)` reporting route, guarantees, limiting port, bytes and commands, introduced for visibility, class breaks, terrain flow, raster region graph and prepared joins (c16); output plans starting with OLS and line geometry (f08).
9. **Bind metadata to retained data** [c15, c16, d03]. Publish the cost table's built cutoff; capture raster stream settings at reset; encode and validate a whole packer record before publishing (d03).
10. **Lossless capture codec in analysis** [f07]. Promote the binary codec from lenses so NaN survives JSON capture.
11. **Stop-to-cube recipe** [c10]. Explicit event time, stable cell dictionary and weights order.
12. **Test practices** [d04, d08, d01, d07, c12, c15]. Poisoned second-submission writer tests; exact discrete-decision parity between twins; translated-fixture pairs; logical empty batches; mixed `import` and `require` objects in packed release verification; failure injection at each owned allocation.

<!-- OUTCOME: appended by the lead after the final wave -->
