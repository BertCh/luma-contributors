# Spatial statistics and regression (2026-11-19)

Scope: 18 directories of `modules/analysis/src/gpu-spatial-analysis` (about 100,000 lines), their specs and pages. All 2026-11-17 SSR findings except SSR-08 (padded readers) and SSR-13 (loop baseline) are fixed. SSR-08 is fixed in three readers and still open in a fourth (the OLS reader), and SSR-13 is partly done. The formulas read and run this time (Moran, Geary and G moments, local Moran z, permutation p-values, conformal ranks, multiple testing, the geographical-detector tail, the interaction fit, ensemble scores, Student t) match independent references. The remaining defects are at seams.

- The weights-completeness rule (LR-D4) is met by about 20 consumers and not met by about 25. Seven read only `overflow`, about eighteen read no weights status, island attachment drops the input status, and chunked transpose output is refused by the statistics it should feed.
- Row-status codes still mean different things in each regression model. There is no shared decoder, and GGWR codes 5 and 6 have none at all.
- Three numerical defects at the float32 border are source-traced or emulated: panel LM diagnostics sum N·T rows serially, spatial SUR residuals use raw coefficients, and GWR/GTWR centre predictors but not the response.
- The OLS model-result reader still infers the predictor count from the table length.
- Panel error ML was read for correctness. Under its stated N·T convention, an independent reference matches the recorded GPU readback in all estimates. Its log-likelihood is not spreg's, and exact mode needs N·T eigenvalues.
- No GPU code ran. Every shader claim is a lead marked `SOURCE` with a GPU probe note.

## Method

- **Lead:** census, the diff `bdc715ea..HEAD` (about 110 source files, +3,500 lines in this area), re-runs of SSR-02, -03 (cross-validated OLS), -04, -10 and -11, the weights-status check on the descriptive families, and verification of subagent findings.
- **Subagent slice 1:** OLS, GWR, GGWR, MGWR, GTWR, MGTWR and the model-result readers.
- **Subagent slice 2:** ML, GM, 2SLS, panel, SUR, probit, regimes, diagnostics and the log-determinant. This slice includes panel error ML read for correctness.
- **Subagent slice 3:** weights, autocorrelation, permutation inference and global statistics, plus a consumer map of every weights port in the library.
- **No subagents for slices 4 and 5.** The 20-subagent limit was full on every retry. The lead read validation, multiple testing, scoring, detector, dynamics, rate smoothing and the small descriptive families only partly. Those families are sampled, not audited, and the report says so below.
- **Executed (CPU/Python):**
  - Detector tail at 8 inputs (compared with analytic central F; no scipy was available).
  - Multiple-testing CPU twin at family size 2^31+1 and 1e7.
  - The interaction CPU twin against a planted coefficient of -0.7.
  - Geodetector non-centrality shift check (Python).
  - Slice probes:
    - OLS readers on capacity tables.
    - Exact log-determinant.
    - Probit tolerance.
    - Float32 serial-sum and GWR emulations.
    - SUR residual emulation.
    - Moment enumeration over 9! permutations of a 3x3 rook lattice.
    - Count-division helpers over 200,000 random cases.
    - Panel error ML against NumPy.
  - Probes are in `scratchpad/review19/spatial-stats/{lead,s1,s2,s3}/`.
- **Not run:** any test, gate, typecheck, GPU or browser.
- **Read in full, by slice:**
  - Slice 1: Student t, incomplete beta, GWR reductions and fit WGSL, GTWR indexed summary, the GWR and OLS readers and statistics, centered prediction, the table layout.
  - Slice 2: panel error ML, panel within nodes, the exact log-determinant, row tiles, trace layout, the completeness helper and table layout.
  - Slice 3: the weights status, loss and suppression helpers, permutation kernels, weights validation, counts WGSL and permutation parameters.
  - Lead: the detector p-value, multiple-testing, conformal ranks and the CPU ensemble scores.
- **Not opened:** `gwr-grid-index`, `gwr-golden-section-nodes`, `generalized-gwr-selection`, `gtwr-bandwidth-search`, most of MGTWR, GM/2SLS/IV kernel bodies, the impacts and traces bodies, tiled global WGSL, fuzzy and lattice contiguity kernels, `UINT64_WGSL`, neighbour-search ties, the bodies of segregation, entropy, colocation, LOF, region statistics, rose and distribution-dynamics kernels, rate-smoothing kernels beyond the summary formulas, and block cross-validation fold assignment.

## Census

Commands run from `modules/analysis/src/gpu-spatial-analysis` unless noted.

| Item | Count | Command |
|---|---|---|
| Directories, source files, lines | 18 dirs, 259 files, about 100,000 lines. spatial-regression 112 files/42,292 lines; weights 47/12,610; autocorrelation 17/6,414; permutation 8/4,555; global 5/2,738; validation 17/4,368; dynamics 15/3,944; others smaller | `find <dir> -type f \| wc -l`; `cat ... \| wc -l` |
| Spec files in matching test dirs | regression 206, weights 66, autocorrelation 37, permutation 28, global 16, dynamics 24, validation 12, multiple-testing 9, region statistics 18, geographic distribution 13, rate smoothing 12, scoring 10, segregation 6, outlier 4, interaction 2; detector, entropy and association have no dir of their own | `find ../../test/gpu-spatial-analysis/<dir> -type f \| wc -l` |
| Exported `GPU*` classes | 109 | `rg -c "^export (abstract )?class GPU\w+"` |
| Files with `getCommandNodes(` | 62 | `rg -l "getCommandNodes\("` |
| `compute*OnCPU` / `CPU*Operation` | 24 / 23 | `rg -c "^export (async )?function compute\w+OnCPU"`; `rg -c "^export class CPU\w+Operation"` |
| Bare `assert(` | 0 | `rg -n "\bassert\("` |
| `validate*` helper calls | 1,080 (539 `validateCapacity`) | `rg -c "validate(Capacity\|MaximumCount\|Requirement\|FiniteNumber\|SinglePackedView\|ViewLength)\("` |
| Named status output (`output.status` 531, `status:` 263), `invalidCount` 108, `overflow` 543, `candidateOverflow` 254, `converged` 224, `iterationLimitReached` 15 | | `rg -o` per term |
| Mask / validity polarity | `mask` 969 uses; `validity` 54 uses, mostly in `gpu-raster-false-discovery-rate-mask.ts` (13) and `gpu-analysis-contiguity-statistics.ts` (14), where nonzero means usable | `rg -c "\bvalidity\b"` |
| Files over 1,500 lines | `gpu-geographically-weighted-regression.ts` 1,987; `gpu-global-spatial-statistics.ts` 1,860; `gpu-spatial-lag-maximum-likelihood.ts` 1,643; `gpu-spatial-two-stage-least-squares.ts` 1,637; `gpu-global-permutation-test.ts` 1,610; `gpu-local-permutation-test.ts` 1,555; `gpu-spatial-error-gm.ts` 1,511; `gpu-geographic-distribution.ts` 1,505 | `wc -l \| sort -rn` |
| `check-contributor-contract --all`, slice-3 sites | 3 findings: `multiple-classes` in `gpu-additional-spatial-statistics.ts:206` (x3) and `gpu-local-variability.ts:140` (x2); `single-invocation-data-loop` in `gpu-global-permutation-test.ts:1315` (x2). Slice 1: 7 sites, 10 instances, all `single-invocation-data-loop`. Slice 2: 19 baseline entries | `node scripts/check-contributor-contract.mjs --all` |
| Public classes with docs heading, catalog record, kernel-audit case | Slice 2: all 13 regression contributors have all three. Slice 1: all 8 contributor classes have all three. Lead sample of 22 descriptive/validation classes: all have specs, docs and catalog entries. No kernel-audit case was located by name for any of them | `rg -l "\bClass\b"` over tests, `docs/api`, ontology |
| Classes with no docs mention | 7 in weights (`CPU/GPUSpatialWeightsValidationOperation`, `CPU/GPUContiguityWeightsOperation`, `CPU/GPUSpatialWeightsAlgebraOperation`, `GPULocalPermutationReplay`), 4 regression operation forms (OLS and MGWR, CPU and GPU), and no `###` heading for `GPUSpatialPanelLagMaximumLikelihood` | `rg -l` over `docs/api` |
| Output-length policy | "at least" for GWR, OLS, ML, GM, 2SLS; "exactly" for MGWR, GTWR, MGTWR | slice 1 |
| Status vocabularies | Six row-status code classes in regression (table below), no shared decoder | slice 1/2 |

Row-status codes in regression, from slices 1 and 2:

| Model | Codes |
|---|---|
| GWR, GTWR, MGTWR `localStatus` | 0 ok, 1 singular, 2 excluded. Decoder `decodeGPUGeographicallyWeightedRegressionStatus` |
| GGWR `codes` | 0 converged, 1 excluded, 2 singular, 3 iteration limit, 4 numerical failure, 5 degenerate response, 6 separation. No decoder |
| MGWR `codes` | same 0 to 4 as GGWR with its own decoder and a duplicate constant. Code 1 means excluded here and singular in GWR |
| OLS | 0 ok, 1 singular, 2 too few rows |
| Lag/error ML, GM, 2SLS | 0 ok, 1 singular, 2 too few rows, 3 non-finite or incomplete weights (lag ML also 3 for rejected first-order condition) |
| Diagnostics | 0 ok, 1 singular, 2 too few, 3 degenerate or incomplete weights, 4 non-finite |
| SUR | 0 ok, 1 insufficient rows, 2 singular design, 3 singular error covariance, 4 inadmissible weights |
| Probit | 0 converged, 1 insufficient, 2 singular, 3 not converged (iterate written), 4 separated |
| Panel lag | 0 ok, 1 invalid or incomplete, 2 singular, 3 not converged |
| Regimes | restricted 0/1 insufficient/2 restricted singular/3 unrestricted singular; fully varying per regime 1 too few, 2 singular |

## Follow-through on the 2026-11-17 findings

| ID | 2026-11-17 claim (short) | Verdict | Evidence |
|---|---|---|---|
| SSR-01 | Rescaled nonreversible weights accepted by the exact spectrum helper | FIXED | Dimensionless check, no floor: `spatial-exact-log-determinant.ts:151-158`. Slice 2 probe: the 3-row fixture throws at scales 1, 1e-6, 1e-3. Pinned by `spatial-exact-log-determinant.node.spec.ts` (joint W/rho rescaling plus a reversible control); it would catch restoring the `max(1, ...)` floor. |
| SSR-02 | Conformal rank slack moves ranks inward | FIXED | New `conformal-prediction-ranks.ts` computes `floor(alpha(n+1))` from the float32 significand with integer arithmetic, host and WGSL; upper rank is `size - lower`. Pinned by `conformal-prediction-ranks.node.spec.ts:8` (8,388,607 rows, alpha 0.25 gives 2,097,152 and 6,291,456) and the 0xfffffffe cases; a relative-slack implementation fails it. Decimal-boundary cases are no longer snapped (float32 0.7 at n=9 gives lower 6, a wider interval, documented in the file). |
| SSR-03 | Held-out OLS and regime predictions use rounded raw intercepts | FIXED (CV OLS), FIXED but weakly pinned (regime) | CV OLS now writes `predictions` through the OLS contributor (`gpu-spatial-cross-validated-ols.ts:162`), and the separate held-out kernel is gone. Pinned by `spatial-validation-advanced.spec.ts:81` (origins 0, 5,000,000, 2^24; a raw-coefficient prediction would fail). Regime: `gpu-spatial-regime-regression.ts:230-260`, pinned only by WGSL source text (`heldout-prediction.node.spec.ts:16-68`); the scenes in `spatial-validation-advanced.spec.ts:147-216` have representable raw intercepts, so a semantic revert that kept the text would pass (SSR-123). Restricted-regime `fitted` still uses raw coefficients: LIMIT (`gpu-restricted-regime-regression.ts:97`). |
| SSR-04 | Detector upper tail loses small probabilities, can hang | FIXED | Log-space complemented beta mixture (`geographical-detector-p-value.ts`). The lead ran `getNonCentralFUpperTail(2**60,2,2,0)` = 8.67361737988405e-19 (analytic 1/(1+2^60) = 8.67e-19). Non-centrality above 10000 now throws; `readGeographicalDetectorPValue` returns `{pValue: null, unavailableReason}`; kepler uses the reader. Pinned by `geographical-detector-p-value.node.spec.ts` (10000 boundary, 2^55, a recomputed mixture at 197-217). See SSR-124 for the new limit. |
| SSR-05 | GWR packer accepts unrepresentable significance level | FIXED | Float32 validation before mutating the target: `geographically-weighted-regression-parameters.ts:326-352`. Pinned by `stats-gwr-review.node.spec.ts:118-148` (`1-2^-27`, `MIN_VALUE`, `1e100`, NaN throw, target stays 123). Would catch pack-then-validate order. |
| SSR-06 | MGWR drops last fit on GPU, CPU ignores budget | FIXED | GPU publishes code 0 or 3 (`gpu-multiscale-geographically-weighted-regression.ts:799`), inference stays gated; CPU honours the budget (`cpu-multiscale-...-operation.ts:251-254`), history `Set` removed. Pinned on both backends by `stats-gwr-review.spec.ts:315-404` and `.node.spec.ts:150-189`. |
| SSR-07 | MGWR explicit grid bypasses the dense limit | FIXED for MGWR | `gpu-multiscale-geographically-weighted-regression.ts:201-214`; `stats-gwr-review.node.spec.ts:191-224` (65,537 rows with explicit grid and no cutoff throws). GGWR keeps the same unbounded traversal (SSR-121). |
| SSR-08 | Capacity-sized tables decode padding as coefficients | PARTLY | `predictorCount` option added to lag ML, error ML, GM and 2SLS readers (`regression-table-layout.ts`), pinned by `regression-padding.node.spec.ts` (nonzero padding 99/98/97/96). Without the option a capacity table still decodes from its length (the exact-length overload is preserved). The OLS reader was missed (SSR-106). |
| SSR-09 | CPU panel-lag graph ignores the eigenvalue port | FIXED | CPU graph reads and validates the spectrum (`cpu-spatial-panel-lag-maximum-likelihood-operation.ts:325,399-401,470,652`). Pinned by `panel-lag-eigenvalues.node.spec.ts:206+`: a zero spectrum and a ring spectrum must change rho; wrong length at `:306`. |
| SSR-10 | One p-value triggers a 16 GiB allocation | FIXED | `getMultipleTestingAdjustedPValues` stores only supplied rows; Euler-Maclaurin harmonic number above 64 terms. Lead probe: one p-value at family size 2^31+1 returns in 0 ms; at family size 1e7 the Benjamini-Yekutieli value 16.7 caps to 1. |
| SSR-11 | Gaussian CRPS overflows for large standardized error | FIXED (SOURCE) | `gpu-gaussian-predictive-scores.ts`: leading term is `abs(error) * erf(z/sqrt2)`; beyond `5.6568542495 * deviation` the kernel returns `abs(error) - deviation/sqrt(pi)` before dividing. Not run on the GPU. The product `5.65 * deviation` can overflow for deviation near 1e38 (EXTREME). Pinning spec for the 1e32/1e-20 case was not located by the lead. |
| SSR-12 | Exact log-determinant disagrees with absolute-determinant contract | FIXED | `spatial-exact-log-determinant.ts:171-187`. Probe: rho=2, spectrum [1,-1] gives ln 3 = 1.0986; rho=1 gives -Infinity. Pinned at `spatial-exact-log-determinant.node.spec.ts:157-159`. |
| SSR-13 | Serial-loop baseline mixes bounded and data-dependent work | PARTLY | 12 of 19 regression baseline entries now state loop axis and cap (IV, GM, 2SLS, traces, SUR). The restricted-regime entries are gone. Seven remain generic. Only panel LM diagnostics (SSR-107) and the panel-lag reference route are real data-row loops. |
| SSR-14 | Catalog and roadmap misstate GTWR status port, ST03, ST08 | PARTLY | GTWR catalog text corrected (`operator-annotations.ts:332`); ST03 and ST08 roadmap rows were rewritten and match source today. GTWR docs still say 32,768 rows and omit the grid route (SSR-122). |
| PTS-08 (README row) | Early-stopped draws misreported | PARTLY | Fixed for Knox (`gpu-knox-test.ts:269-281`); open for local and global permutation tests (SSR-114). |
| Moran harness rebuilt weights without status (README) | | FIXED | `spatial-autocorrelation-harness.ts:229` uses `search.outputWeights`; pinned by `moran-weights-status.node.spec.ts` and `global-weights-status*.spec.ts`. |
| CORE-05 (AICc sign) | | FIXED | `analysis-core/contracts/model-result.ts:211-218` returns `null` when the denominator is not positive. Pinning test not looked up. |

## Findings

Kinds: W wrong result, S visible failure, C contract, T test, M maintainability. Evidence: EC executed on CPU, SRC source trace, CNT count. Items marked LEAD need a GPU probe and are not confirmed.

| ID | Kind | Reach | Evidence | Where | What goes wrong | Proposed change |
|---|---|---|---|---|---|---|
| SSR-101 | W | UNUSUAL | SRC (lead opened) | `distribution-dynamics/gpu-spatial-markov.ts:241-247` | `GPUSpatialMarkov` rebuilds the weights without `status` before its per-period `GPUSpatialLag`, so the lag cannot suppress on incomplete weights and the transition counts have no suppression of their own. Comment says status is "not consumed". | Pass `status` through and suppress or refuse; spec with `candidateOverflow` only. |
| SSR-102 | W | UNUSUAL | SRC (lead grep of status reads) | GPUSegregation `segregation/gpu-segregation.ts:122,252`; GPUSpatialEmpiricalBayesRates `rate-smoothing/...:78,131`; GPUColocationQuotient `:57,121`; GPULocalCategoryDependence `:65,128`; GPUCategoryPairCounts `:50,94`; GPUSpatialEntropy `:74,137`; GPULocalOutlierFactor `outlier-detection/...:75,143`; GPUMapColoring; GPUAZPRegions; GPUMaxPRegions; GPURegionPartitionEvaluation; GPUSpaceTimeLocalMoran; GPUEmergingHotSpots weights mode; GPULocalKnox; GPUNearRepeatKnox; GPULocalCollinearity; GPUSpatialWeightsSummary | About 18 consumers take a weights or neighbour port and never read `status`. A truncated table gives neighbourhood statistics over missing neighbours with clean status. LR-D4 says propagate, suppress or refuse. | One sweep with `createSpatialWeightsSuppressionNodes` or construction-time refusal of status-carrying inputs; one spec per consumer. |
| SSR-103 | W | UNUSUAL | SRC (lead opened three) | Multivariate Geary exact `gpu-multivariate-local-geary-permutation-test.ts:407-411` and tiled `:134-135,162`; `neighborhood-summary/gpu-neighborhood-summary.ts:244-263`; probit diagnostics `gpu-probit-regression.ts:932-941`; SUR `gpu-seemingly-unrelated-regression.ts:116,186`, `spatial-sur-nodes.ts:315-328`; panel regression `:236-246`; panel error ML and panel diagnostics `panel-diagnostics.ts:95`; IV covariance `:179,199` | Seven consumers read only `weights.status.overflow`. `candidateOverflow` and chunk-level flags are ignored. Contiguity weights set only `candidateOverflow` when the pair list exceeds `pairCapacity` (`gpu-contiguity-weights.ts:181,784`); algebra does so for lossy operands. The ML/GM/2SLS family honours both. | Use `getGPUSpatialWeightsLossViews` or `getRegressionLossViews` in each; a `candidateOverflow`-only spec per consumer. |
| SSR-104 | W | UNUSUAL | SRC (lead opened) | `spatial-weights/gpu-island-attachment.ts:298-310` | `props.weights.status` is never read. Rows emptied by an input truncation look like islands and receive nearest-neighbour edges, and the output status reports `overflow` 0. A downstream Moran runs on invented neighbours with clean status. Algebra does carry operand loss (`gpu-spatial-weights-algebra.ts:369`). | Fold input loss into `output.status.candidateOverflow`. |
| SSR-105 | S | UNUSUAL | SRC (lead opened) | `gpu-chunked-spatial-weights-transpose.ts:237-246,283-298`; refusal at `spatial-autocorrelation-kernels.ts:136-143`, `gpu-global-spatial-statistics.ts:346-347` | Output of the chunked transpose cannot feed local Moran, hot spot, additional statistics, global statistics or another chunked transpose: construction throws "loss flags must be aggregated on the outer weights.status port". No public helper aggregates chunk flags. The local permutation test accepts the same input and multivariate Geary ignores chunk flags: two rules for one object. | Aggregate chunk flags into the outer status in the transpose (as `gpu-chunked-spatial-weights-algebra.ts:208,328-336` does). |
| SSR-106 | W | UNUSUAL | EC | `ordinary-least-squares-statistics.ts:192-205` | The OLS model-result reader takes the predictor count from the whole coefficient table. A 4-slot capacity table for a 1-predictor fit gives df 6 instead of 8, AIC 28 instead of 24, AICc 36 instead of 25.71, Breusch-Pagan df 3 instead of 1, and two extra "predictors" with p = 1. Lead re-ran the probe. This is SSR-08 in a fifth reader. | `predictorCount` option as in the other four readers; spec with a capacity table. |
| SSR-107 | W | UNUSUAL | SRC + float32 emulation (GPU probe wanted) | `panel-diagnostics.ts:134,147`, reached from `gpu-spatial-panel-error-maximum-likelihood.ts:319` | The LM-test kernel runs with `invocationCount: 1` and sums N·T rows serially in float32 (`residualLag`, `responseLag`, `trace`, `projectedSquares`). Emulating the trace term (0.4 per row): at 1e6 rows the sum is 403,833 vs 400,000 (+0.96%); at 4e6 it is 1,538,099 vs 1,600,000 (-3.9%); at 2^24 it is 7,740,356 vs 6,710,886 (+15.3%). The LM statistic scales by the same factor and diagnostic status stays 0. It is also a single-invocation scan of up to 16M rows. | Tiled partials with pairwise merge (`getRowTileSumNodes`), as the ML contributors do. GPU probe: N=4096, T=1024, ring of 5 neighbours at 0.2; compare the first test with a float64 recomputation. |
| SSR-108 | W | UNUSUAL | float32 emulation of the kernel (GPU probe wanted) | `spatial-sur-nodes.ts:257-265` | Spatial SUR structural residuals evaluate `y - b0 - sum x b` from the raw float32 intercept. Ordinary SUR was fixed to use the centered fit (`ordinary-sur-nodes.ts:418-432`). With x = 2^24 + {-2,-2,2,2,0}, y = 1 + 1.5(x - 2^24) +/- 0.25, the published residuals are [0,0,-2,-2,0] against true [0.25,-0.25,0.25,-0.25,0]. The SSR-03 class at a new seam. | Evaluate from the centered solution and means; translation fixture. |
| SSR-109 | W | UNUSUAL | EC (CPU twin) KNOWN(Probit row, adds a number) | `gpu-probit-regression.ts:319-322`, `cpu-probit-regression-operation.ts:53-54` | Only `tolerance > 0` is checked. 400 rows, true slope 1.2: tolerance 1e-5 gives status 0 and [0.4340, 0.9701]; tolerance 0.5 or 10 gives status 0 after one iteration and [0.3644, 0.8177], a 16% slope error. | Upper limit through `validateRequirement` on both backends. |
| SSR-110 | W | UNUSUAL | LEAD: float32 emulation, UNVERIFIED on GPU | `gwr-fit-wgsl.ts:168-171`; `gpu-geographically-temporally-weighted-regression.ts:594-597` | GWR and GTWR centre predictors on the focal row but sum `w z y` with the raw response. Emulated serial float32, 4,000 rows, 2x + noise: offset 1e6 gives slope 2.141 against 1.9946 (7% off); with centered y it matches. Needs a mean-to-residual ratio of about 1e5 or more. | Centre y before accumulating; fitted value adds the offset back. GPU probe: 4,000 rows, fixed bandwidth, y = 1e6 + 2x + noise; read `coefficients[focus*2+1]`; a slope off by about 1e-2 confirms. |
| SSR-111 | W | EXTREME | LEAD: SRC | `gpu-geographically-weighted-regression-nonstationarity-test.ts:1036-1058` | The permutation p-value is `(g+1)/(P+1)` with P the requested count. A permutation with NaN spread counts in the denominator and never in g, so p is too small. | Use the valid-permutation count or report invalid permutations. |
| SSR-112 | C | ORDINARY | SRC (slice 1/2) | Status table in Census | Six code vocabularies, two meanings of code 1 and of code 3, GGWR codes 5 and 6 without a decoder, duplicate MGWR constant, literal codes in kepler, lenses, explorer and indexed GTWR (`gtwr-indexed-nodes.ts:287,295,355`; `catalog-validation-regression.ts:238-239`; `regression-mode.ts:1053`). Each current reader is right for its own model; nothing prevents applying one model's constant to another's output. `GPUSpatialLagImpacts` NaNs on any nonzero status. | One decoder per model now (GGWR included, delete the MGWR duplicate, replace literals); add a `status.reason` string to the model result. |
| SSR-113 | C | UNUSUAL | SRC | `gpu-local-permutation-test.ts:1001-1023,1115-1116,1308`; `gpu-global-permutation-test.ts:218-219,361`; `gpu-spatial-lag.ts:204,295`; `gpu-quantile-lisa.ts:420` | Each consumer signals incomplete weights differently: the local permutation test folds weights loss, neighbour-cap overflow, invalid permutation count and negative G into one `output.status.overflow`; analytic statistics report the two flags separately; the global permutation test and the spatial lag return NaN with no flag; quantile LISA gives untested rows class 0 ("not significant"). | One loss/invalid vocabulary and a "not tested" class in quantile LISA. |
| SSR-114 | C | ORDINARY | SRC | `gpu-local-permutation-test.ts:813-829`; `gpu-global-permutation-test.ts:508-523` | With `stopAfterExceedances`, the inference metadata still reports the requested `permutationCount` and nothing about stopping. Per-row counts are right; only the metadata is wrong. Fixed for Knox only. | Apply the Knox `requestedPermutationCount` pattern. |
| SSR-115 | C | UNUSUAL | SRC | `gpu-global-spatial-statistics.ts:1727-1728,1734`; `gpu-local-permutation-test.ts:1299-1310` | Negative data for G: the global G block is NaN with no status word, while local G/G* permutation sets `overflow` and `invalidCount`. | Add an invalid count to the global status. KNOWN(FX-05). |
| SSR-116 | C | UNUSUAL | SRC | `gpu-spatial-weights-validation.ts:152-163`; `gpu-neighbor-search.ts` | `outputWeights.status.invalidCount` is attached but no consumer reads it; `GPUNeighborSearch` rows dropped for coordinates outside `bounds` silently become islands downstream. | Refuse in consumers or document as host-readback only. |
| SSR-117 | C | UNUSUAL | SRC | `spatial-regression/gpu-geographically-temporally-weighted-regression.ts:698-709`; `gtwr-indexed-summary.ts:110-119`; `geographically-weighted-regression-model-result.ts:74` | GTWR summaries document the GWR summary layout but leave slot 5 (`HAS_VALID_CANDIDATE`) NaN. Passing a GTWR summary to the GWR reader reports a failed fit with null log-likelihood, AIC and AICc (EC by slice 1). | Write slot 5 or add a GTWR reader. |
| SSR-118 | C | UNUSUAL | SRC | `gpu-spatial-panel-error-maximum-likelihood.ts:145`, `spatial-exact-log-determinant.ts:298-311` | Exact mode needs N·T repeated eigenvalues and N·T <= 1024; panel lag takes one per entity. No doc says so; the natural N-length spectrum throws "one value per row". | Accept N entity eigenvalues; bound N. |
| SSR-119 | C | UNUSUAL | SRC + recorded readback | `statistics-upstream.spec.ts:357-358`; panel error doc | The doc says "spreg Panel_FE_Error concentrated likelihood". lambda, beta and sigma squared match (NumPy reference matches the recorded GPU readback). The log-likelihood at spreg's own estimates is -56.584 against spreg's recorded -149.493. The spec checks only beta and lambda; sigma squared, log-likelihood and standard errors are unchecked. Reusing the cross-section ML reader gives a finite AIC the contributor deliberately writes as NaN, and df = NT - (p+2) which does not count the N fixed effects. | Panel reader or an `informationCriterion: null` option; document the convention; assert sigma squared and log-likelihood. |
| SSR-120 | C | UNUSUAL | SRC | `regression-row-tiles.ts` and the callers named in SSR-126 | The 4,096-tile cap is enforced at construction by GM (`gpu-spatial-error-gm.ts:361`), 2SLS (`:397`), traces (`gpu-spatial-weights-traces.ts:345`) and the trace layout, and not by lag ML (`gpu-spatial-lag-maximum-likelihood.ts:566-569`) or error ML (`:413-417`). The merge is pairwise, so this is cost, not correctness. KNOWN(LR-26). | Enforce or document one number. |
| SSR-121 | C | UNUSUAL | SRC | `gpu-multiscale-geographically-weighted-regression.ts:236-300`; GGWR `:269-279` vs MGWR `:201-214` | MGWR messages ("output length mismatch", "inference shape mismatch") name neither prop nor limit; MGWR demands exact output lengths while GWR/OLS accept longer. GGWR still accepts an explicit grid with an untruncated fixed Gaussian kernel up to 1,048,576 rows with a full row scan; MGWR now caps the same traversal at 65,536. | `validateCapacity` with named limits; use the dense cap for GGWR fixed Gaussian. |
| SSR-122 | C | ORDINARY | SRC | `docs/api/gpu-spatial-analysis/operators-regression.md:1086`; `gpu-seemingly-unrelated-regression.ts:78-84`; ROSE docs | GTWR docs say "at most 32,768 rows" and omit the grid route, `searchStructure`, `gaussianCutoff` and the 16,777,216 limit; the SUR class comment says "not spatial SUR" although the `spatial` option exists; the docs page does not mention the rose `positive-y-clockwise` origin or degrees; panel family limits differ (rows 2^24-1 and T 1024 in panel regression; rows 2^24 and T 2^24 in panel error); the `ST08` text was fixed but the docs page for panel lag has no `###` heading. | Docs edit. |
| SSR-123 | T | ORDINARY | SRC | `heldout-prediction.node.spec.ts:16-68` | The regime-prediction fix is pinned by source text, and the GPU scenes have representable raw intercepts: raw-coefficient prediction would give the same outputs. | Add the SSR-03 case y = 1 + 1.5(x - 2^24). |
| SSR-124 | C | UNUSUAL | EC + doc | `geographical-detector-p-value.ts`; `docs/.../geographical-detector-redcap-entropy.md:22-46` | p-value is unavailable (throws, or null from the reader) above non-centrality 10,000. The non-centrality is the published geodetector formula `sum mean^2 - (sum sqrt(N_h) mean_h)^2/N` over sigma squared. It is not shift-invariant for unequal strata: lead run on y=[1,2,10,11,12,13], strata sizes 2 and 4 gives 1.27 at offset 0, 14.8 at +100, 2361 at +1000. Documented (docs:45) and tested against the offset fixture; a response with a large mean therefore reaches the cap and returns null. | Keep the documented behaviour. Consider a finite bound above 10,000 so ordinary offset responses return a number. |
| SSR-125 | T | ORDINARY | CNT | specs | No spec sets only `candidateOverflow` for the overflow-only consumers, none chains lossy weights into island attachment, none chains chunked transpose into Moran, none feeds loss into spatial Markov or emerging hot spots. No GWR or GTWR spec uses a large-offset response. The OLS reader is only tested with exact-length tables. | Add these fixtures. |
| SSR-126 | M | - | CNT | `gpu-global-spatial-statistics.ts:1546` vs `permutation-inference-kernels.ts:700` (column sums); `regression-weights-completeness.ts:18-29` vs `gpu-chunked-spatial-weights.ts:102-112`; catchment `:490` and Huff `:587` have their own loss collectors; `MAXIMUM_DEFAULT_TILE_COUNT = 4096` redeclared in six regression files | Four or five copies of loss collection/suppression and six of the tile layout. `getKahanSumWGSL` is a pairwise sum. `getEquilibratedCholeskyWGSL` lives in a GWR file but panel lag uses it. | Consolidate into `spatial-weights-loss.ts` and `getRegressionTileLayout`. |
| SSR-127 | M | - | SRC | `T/stats-gwr-independent-review.node.spec.ts:197-212` | An analysis spec dynamically imports a website source file, a dependency the wrong way round, while another session edits `website/`. | Move into a website test. |
| SSR-128 | C | UNUSUAL | SRC | `gpu-global-permutation-test.ts:1160` | `pZSimulated` is always one-sided whatever `alternative` is. esda's convention; undocumented. | One doc sentence. |
| SSR-129 | M | - | SRC | `gpu-global-spatial-statistics.ts:1686-1692` with `gpu-chunked-spatial-weights.ts:97` | `getWeightsViews` drops views with `index % 4` but loss views are appended after chunk views. Harmless today. | Filter by field. |
| SSR-130 | C | EXTREME | SRC | `permutation-inference-kernels.ts:164-167,191-192` | A `candidateOverflow` view of the wrong format passes construction checks; binding ranges omit `candidateOverflow` and chunk flags. | Call `validateWeightsLossSources`. |
| SSR-131 | C | EXTREME | SRC | `gpu-gaussian-predictive-scores.ts` | `5.6568542495 * deviation` overflows for a deviation near 1e38. | Compare in the ratio form. |

SSR-130 and SSR-131 have the lowest reach (EXTREME).

### Reproductions

**SSR-101, SSR-102, SSR-103, SSR-104, SSR-105 (weights completeness).** SOURCE, GPU probe wanted.
- SSR-103: 2x2 grid of square polygons, queen contiguity with `pairCapacity: 1`. Read `candidateOverflow = 1` and `overflow = 0`, then feed `outputWeights` to `GPUMultivariateLocalGearyPermutationTest`. A finite statistic confirms; NaN refutes.
- SSR-104: CSR for 3 rows on a line with capacity 1 (so `overflow = 1`); run `GPUIslandAttachment`; `outputWeights.status.overflow = 0` confirms.
- SSR-105: host-only. Construct a `GPULocalMoran` over the chunked output of `GPUChunkedSpatialWeightsTranspose`; a construction throw confirms.
- SSR-101/102: build weights with `status.overflow = [1]`, run `GPUSpatialMarkov` or `GPUSegregation`; non-NaN output with nonzero input flag confirms. Expected under LR-D4: NaN, a status word, or a construction refusal.

**SSR-106 (executed).** Probe `s1/probe-ols.ts`. Input: coefficients [1,2], standard errors [0.5,0.8], t [2,2.5], n 10, log-likelihood -10. Exact 2-slot buffer: p(x0) = 0.03694, df 8, k 2, AIC 24, AICc 25.714, BP df 1. 4-slot buffer padded with zeros: p(x0) = 0.04653, df 6, k 4, AIC 28, AICc 36, BP df 3, extra x1 and x2 at p = 1. Expected: the second equals the first.

**SSR-107 (emulation).** `s2/serial_sum.py`, `serial_sum2.py` mirror the serial float32 add of the per-row trace term 0.4: 1e6 rows 403,833 (exact 400,000), 4e6 rows 1,538,099 (1,600,000), 2^24 rows 7,740,356 (6,710,886). GPU probe: N=4096, T=1024, ring of 5 neighbours at 0.2; read `tests[0]`; deviation beyond 1e-3 relative from the float64 value confirms.

**SSR-108 (emulation).** `s2/sur_resid.py`: x = 2^24 + {-2,-2,2,2,0}, y = 1 + 1.5(x - 2^24) +/- 0.25. Published [0,0,-2,-2,0]; true [0.25,-0.25,0.25,-0.25,0]. GPU probe: spatial SUR on this scene; residuals equal to the first vector confirm.

**SSR-109 (executed).** Probit CPU twin, 400 rows, true slope 1.2. Tolerance 1e-5: status 0, [0.4340, 0.9701]; tolerance 0.5: status 0, [0.3644, 0.8177], 1 iteration; tolerance 10: same. Expected: status 3 or a construction rejection.

**SSR-110 (lead).** `s1/f32-gwr.py`: 4,000 rows, Gaussian bandwidth 30, y = offset + 2x + N(0, 0.5). Float64 slopes 2.00514, 1.99995, 1.99455 at offsets 1e3, 1e5, 1e6; float32 raw y gives 2.00501, 2.00575, 2.14100; float32 centred y gives 2.00514, 1.99993, 1.99455. The emulation does not include the equilibrated Cholesky; the error sits in `X'Wy`.

**SSR-124 (executed).** Python: y = [1,2,10,11,12,13], strata [0,0,1,1,1,1]; non-centrality 1.267 at offset 0, 14.79 at +100, 2360.98 at +1000. With equal strata it is shift-invariant.

## Refuted or downgraded

- SSR-03 CV OLS, SSR-02, SSR-04, SSR-10: re-run today; they hold.
- Detector tail: the lead compared against the analytic central F(2,2) value only; no non-central reference (scipy) was available, so the non-central accuracy at moderate values rests on the pinned spec at `geographical-detector-p-value.node.spec.ts:197-217`.
- Student-t tail and critical value: executed against SciPy by slice 1, relative error at most 1e-9 for df 1 to 1e9.
- AICc: the GWR candidate form and the GTWR form agree algebraically; match mgwr.
- Hat diagonal and intercept standard errors with focal centring: correct.
- OLS with df <= 0: status 2, NaN p-values.
- Panel error ML estimates under the N·T convention: match NumPy and the recorded GPU readback; the downgrade is only the log-likelihood and df convention (SSR-119).
- Slice 2's panel error `estimatedCommandNodeCount` undercount: refuted (actual 414 and 451 nodes against 647).
- Moments: binary-weight V[I] 0.0516975309, V[C] 0.0557569621, V[G] 0.002185981941 and row-standardised V[I] 0.0555957008, V[C] 0.0492654473, V[G] 0.000199759912 equal full enumeration to 10+ digits. Local Moran conditional z matches enumeration (-0.742926594561, 1.856770457731).
- Count-division helpers over (2^24, 2^32): correct in 200,000 of 200,000 cases.
- Ensemble CRPS, energy and variogram score formulas, read in `cpu-ensemble-scores-operation.ts`, match closed forms.
- `computeSpatialInteractionOnCPU` recovers beta = -0.7 as -0.7000027 and -0.6999953 (bounded interval), predictions within 1.4e-5 relative.
- SSR-04 non-termination at 2^55: now throws a stated error; the wrong p-value of 0 is gone.

## Tests

Independent oracles exist for the global and local autocorrelation statistics (esda JSON fixtures with generator scripts), OLS (SciPy fixture), GTWR/MGTWR (external results with a documented AICc convention difference), MGWR (hand-computed budget case), cross-validated OLS (origin sweep) and conformal ranks (rational arithmetic). The weakest five:

1. **Panel error ML.** Checks beta and lambda to three decimals. sigma squared and log-likelihood from the fixture are unused, no standard errors, no `candidateOverflow` case, no large-row case for the diagnostics.
2. **Spatial regime prediction and spatial SUR residuals.** Regime fix pinned by source text; SUR residuals tested at small scale only. A translation fixture is missing in both.
3. **The OLS model-result reader.** Only exact-length tables. The defect SSR-106 passed.
4. **The weights-consuming family** (spatial Markov, segregation, entropy, colocation, LOF, spatial EB rates, island attachment). No spec sets a loss flag. `GPUIslandAttachment`'s oracle is written from the same reading. `GPUMultivariateLocalGearyPermutationTest` has an oracle in `multivariate-geary-runtime-oracle.ts`, one status spec with a single `expect`, no `candidateOverflow` case.
5. **GWR/GTWR numerics.** No large-offset response; the nonstationarity test oracle mirrors the same shuffle and reading and has no NaN-permutation case. Also: GGWR codes 5 and 6 are tested only by constant identity; MGTWR has two fixed-bandwidth configurations that converge in five iterations; the CPU twin and GPU contributor of `GPULocalPermutationReplay`, the six weights operation forms and `GPUAnalysisContiguityStatistics` rest on twin-versus-contributor checks written from one reading.

The discriminating-test rule is followed for SSR-01, -02, -03 (CV), -05, -06, -07, -09, -10 and -12. It is not followed for the regime half of SSR-03 and for SSR-08 as a whole. Tiled-versus-whole equality has GPU specs for chunked GM, lag ML, error ML and traces (`weights7-chunked-*.spec.ts`) and none for 2SLS. Fixtures at one scale only: most regression specs except the origin sweep in cross-validated OLS and GGWR predictor offsets.

## Maintainability

- **Duplicates, named:**
  - Student t: `gwr-student-t.ts` and `ordinary-least-squares-statistics.ts:115`.
  - AICc: `gpu-geographically-weighted-regression.ts:1421`, `gpu-geographically-temporally-weighted-regression.ts:697`, `gtwr-indexed-summary.ts:109`.
  - GTWR summary: dense and indexed copies disagree on slot 5.
  - MGWR status constant duplicates `GPU_ITERATIVE_REGRESSION_STATUS`.
  - Tile layout in six files.
  - Weights loss collection in the transpose pipeline, regression completeness, catchment and Huff.
  - Column sums in global statistics and permutation kernels.
- **Over 1,500 lines mixing concerns:** the eight files listed in the census. `gpu-global-spatial-statistics.ts` mixes tiled and reference modes, finalize WGSL, symmetry and layout.
- **Dead exports:** not systematically checked; slice 2's full scan timed out. 8 named exports checked all had references. `getStudentTSurvival` has only a few callers outside its directory (check before removal).
- **Internal switches in public props:** none found beyond documented route options.
- **Helpers that belong elsewhere:** `getEquilibratedCholeskyWGSL` (in a GWR file, used by panel lag) belongs in `analysis-core`; the weights-loss helpers are `@internal` and used by only about 10 of roughly 45 consumers, and a public combine-loss helper is missing.
- **Wrapper forms:** `GPU/CPU*Operation` wrappers have no docs, catalog or audit entries by design for most; the four regression forms (OLS, MGWR) have zero references.

## Roadmap rows

| Row ID | Verdict | Evidence | Replacement text if NARROW |
|---|---|---|---|
| ST02 | KEEP | No robust or HAC option in `gpu-restricted-regime-regression.ts`. | |
| ST03 | KEEP | `gpu-spatial-panel-error-maximum-likelihood.ts:75` supports unit effects only; panel lag fixed and random. Text is accurate. | |
| ST04 / ST05 | KEEP | `gpu-seemingly-unrelated-regression.ts:84` single pass; probit `:282` says it is not a latent spatial probit. | |
| ST06 | KEEP | No bandwidth confidence interval in the MGWR files; other variance conventions rejected at `gpu-multiscale-geographically-weighted-regression.ts:284-289`. | |
| ST08 | KEEP | Sorted ranking above 64 states exists; diagnostics stay capped at 64 states (`gpu-rank-markov.ts:53,74,97`); fractional ties and interpolation absent. | |
| ST10 | KEEP | `gpu-spatial-cross-validation.ts` has seeded block folds and exact planar buffer exclusion only; no hierarchical blocks, merging, balancing or variogram optimiser. | |
| ST11 | KEEP | `gpu-spatial-interaction.ts` supports one covariate and Poisson with golden-section calibration; no coefficient uncertainty. | |
| ROSE-01 | NARROW | `gpu-rose-statistic.ts:89-90,164-167` accepts `angleUnit: 'radians' \| 'degrees'` and origins `positive-x-counter-clockwise` and `positive-y-clockwise`; specs `gpu-rose-statistic-improvements.*.spec.ts`. | "Radians or degrees, with a positive-x counter-clockwise or positive-y clockwise origin. Other conventions (for example clockwise from the x axis) are unsupported; the docs page does not yet describe the two origins." |
| LR-26 | NARROW | See SSR-120. | "GM, 2SLS, weights traces and the nested trace layout reject more than 4096 row tiles at construction. Lag and error ML main tiles have no cap (serial merge cost only). Chunked equals whole is tested for GM, lag ML, error ML and traces, not 2SLS. No hierarchical merge exists." |
| GPU-09 | NARROW | MGWR global fit is tiled (`multiscale-global-fit.ts:36-130`). Remaining data-sized single-invocation loops: panel-lag dense determinant and reference-route fits, panel error LM diagnostics (`panel-diagnostics.ts:147`), GTWR, MGTWR, the GWR nonstationarity test, GWR grid index, GTWR bandwidth search, `gpu-global-permutation-test.ts:1315`. | "MGWR global fit, panel means and SUR weight validation are tiled. Panel-lag dense determinant, panel error LM diagnostics (N*T rows in one invocation, SSR-107), the GWR nonstationarity test, GTWR/MGTWR searches and the global permutation sequential walk remain." |
| STD-04 | NARROW | `spatial-autocorrelation-status.ts:30`: local Moran, hot spot and Local Geary ports exist. The global permutation test has no loss flag; spatial lag, weights summary and multivariate Geary permutation have no status; the local permutation test folds every cause into `overflow` (SSR-113); about 18 consumers read no weights status (SSR-102). | Append: "In spatial statistics, weights loss is read by about 20 consumers and ignored by about 25 (SSR-101 to SSR-105); decide one loss vocabulary for local and global permutation tests." |
| STD-05 | KEEP (claims verified) | `selfWeight` view (`gpu-hot-spot-analysis.ts:150`), LOSH `residualPower` (`gpu-local-variability.ts:118`), summary `symmetryTolerance` (`gpu-spatial-weights-summary.ts:89`), transpose tolerance (`gpu-spatial-weights-transpose.ts:52`), multivariate Geary seed and significance (`multivariate-geary-parameters.ts:11-56`), lattice `cellSize` (`gpu-lattice-weights.ts:81`), kernel bandwidth (`gpu-spatial-weights-transform.ts:168`) are runtime. | |
| STD-10 (slice part) | NARROW | Regression Cholesky wraps the shared `getDenseCholeskyWGSL` (`spatial-regression-solve.ts:5,38`); panel lag uses `getDenseLUWGSL`. Remaining: panel-lag `getInverseWGSL` (`:761`) and the equilibrated Cholesky in a GWR file. | |
| GAP-04 | NARROW | No Durbin, SLX, SDEM or SARAR code. OLS has `output.predictions` (`gpu-ordinary-least-squares.ts:88`); GGWR predicts at masked rows with `predictExcluded` (`gpu-generalized-geographically-weighted-regression.ts:150,692`). | "Durbin/SLX/SDEM, SARAR, GWR/MGWR/GTWR prediction at new locations (GGWR predicts masked rows; OLS has `output.predictions`), multi-attribute clustering, windows, group regression and as-of joins." |
| DEC-LAG | KEEP | CPU twin publishes the rejected iterate (`cpu-spatial-lag-maximum-likelihood-operation.ts:200-202`); GPU clears it (`gpu-spatial-lag-maximum-likelihood.ts:384`). Also record that code 3 is overloaded (SSR-112). | |
| FX-03 / FX-05 | KEEP, NARROW text | Sporadic rule and the opt-in older rule confirmed at `gpu-emerging-hot-spots.ts:260-268,1534,1591`. | "Negative data: the global G block is NaN with no status word (SSR-115); local G/G* permutation sets `overflow` and `invalidCount`; analytic Gi* accepts negative values. Decide host rejection or a status word." |
| Known limit: Lag ML | KEEP | Documented at `gpu-spatial-lag-maximum-likelihood.ts:385-399`; no flag. Not re-measured. | |
| Known limit: Panel random effects | KEEP | Documented at the panel lag header; numbers not re-measured. | |
| Known limit: Probit | NARROW | `math-review-probit-bound.spec.ts` | "Uncertified separation can end as status 3 or 2 rather than 4; `tolerance` has no upper bound (CPU: tolerance 0.5 gives status 0 after one iteration with a 16% slope error, SSR-109)." |
| Known limit: Panel error ML "not read for correctness" | CLOSE, replace | Read 2026-11-19 (slice 2). | "Read 2026-11-19: estimates and standard errors equal the N*T concentrated likelihood on a recorded GPU readback; log-likelihood is not spreg's `logll`; exact log-determinant needs N*T repeated eigenvalues; LM diagnostics sum serially in float32 (SSR-107) and ignore `candidateOverflow` (SSR-103)." |
| SIM-05 | KEEP | No geographically weighted PCA class. | |
| New: LR-D4 conformance | ADD | SSR-101 to SSR-105. | "Weights consumers that read only `overflow` (7), read nothing (about 18), drop loss (island attachment) or cannot accept chunked transpose output." |

## Next steps

Before the alpha:

1. **Weights completeness sweep (M).** One pass over about 25 consumers with the shared loss helpers, one spec each that sets only `candidateOverflow` (SSR-101 to SSR-105, -125). For whoever publishes: without it a truncated neighbour table gives statistics that look valid. Also make the chunked transpose aggregate its outer flag.
2. **Regression readers (S).** OLS `predictorCount` (SSR-106); one decoder per model including GGWR; GTWR summary slot 5 and a decision on what the model result says about status scope (SSR-112, -117).
3. **Three numerical fixes (S each).** Tile the panel LM diagnostics (SSR-107), centred spatial SUR residuals (SSR-108), probit tolerance bound (SSR-109). Each is a one-contributor change with a discriminating fixture.
4. **GPU probes (S).** Run the probes for SSR-103, -104, -105, -107, -108 and -110 on a device; each is a single small scene listed above.
5. **Docs and roadmap (S).** Apply the NARROW texts above; correct the GTWR page, SUR class comment, rose origins, panel limits.

After:

- **Consolidate (M):** weights loss collection, tile layout constants, Student t and AICc.
- **One loss and invalid vocabulary** across statistics (SSR-113, -115), then the shared status vocabulary for models.
- **Stopping metadata** for the permutation tests (SSR-114).
- **Panel family (M):** reader, log-likelihood convention, entity-spectrum log-determinant, then time effects (ST03, L).
- **Add (L, consumer-driven):** Durbin/SLX/SDEM and prediction at new locations (GAP-04); MGWR bandwidth intervals (ST06).
- **Freeze:** nothing to delete. The wrapper forms with zero references (OLS and MGWR operation classes) could be frozen as internal.
- **Family steps:**
  - Regression: finish status and readers; scale (panel LM tiling); connect (model-result capability field); L: panel time effects.
  - Weights: finish and connect (loss sweep, S/M); consolidate helpers (M).
  - Autocorrelation and permutation: finish (stopping metadata, S); consolidate vocabulary (M).
  - Validation and multiple testing: finish (regime test, S); nothing else pressing.
  - Detector: decide the finite bound above non-centrality 10,000 (S).
  - Descriptive families: finish (weights status, S per class); add specs where only indirect ones exist.

## Limits of this report

- No test, gate, GPU or browser ran. Every shader statement is SOURCE or emulation and written as a lead; GPU probes are listed.
- Slices 4 and 5 were not delegated (subagent limit) and the lead covered them only partly. Validation, multiple testing, scoring, the detector, conformal ranks and the interaction CPU twin were run; distribution dynamics, rate smoothing, segregation, entropy, association, region statistics, geographic distribution, outlier detection and block cross-validation were read in headers, formulas and status handling only. Closed-form checks of segregation, entropy, colocation, standard distance, rose, LOF, empirical Bayes and region statistics were not done. Float32 accumulation at 1e6 rows in those families was not examined beyond a grep for local-origin handling in geographic distribution.
- Slice 1 did not open `gwr-grid-index`, `gwr-golden-section-nodes`, `generalized-gwr-selection`, `gtwr-bandwidth-search`, `gtwr-adaptive-grid-wgsl`, the local-weights and collinearity WGSL, most of MGTWR.
- Slice 2 did not open GM and 2SLS kernel bodies beyond tile caps and residuals, the IV covariance body, the impacts and traces bodies, `spatial-series-tail.ts`, `chunked-weights-kernels.ts`, nor the CPU twins for GM, 2SLS, SUR and restricted regimes.
- Slice 3 did not open tiled global WGSL, fuzzy and lattice contiguity kernels, summary and algebra kernels, `UINT64_WGSL`, or neighbour-search tie handling; the 64-bit count helpers were assumed exact.
- The detector tail was compared with the analytic central case only; no non-central reference was available (scipy absent).
- Pinning tests were named from reading the spec; none were run, and the pinning spec for the CRPS 1e32/1e-20 case was not located.
- Consumer counts (about 20 and about 25) come from `rg` status greps and the slice-3 map; a consumer that reads status under another name would be miscounted.
- Counts of 'weights consumers' cover the whole library, not only these directories.
