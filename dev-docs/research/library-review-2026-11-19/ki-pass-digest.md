# Known-issues pass (KI): digest for the library review

Written 2026-11-19 from the staging clone `/Users/robertchristie/Documents/GitHub/luma-contributors-ki` (base `9e93cbc7`, nothing landed). Read-only: status files, reviews, logs, orchestrator scripts, `git diff`. First read 17:43 EST, counts and ledger 17:57; streams kept moving. "Claimed" = a stream's own status file. "Confirmed" = a later reviewer or the lead's verify log. I read six failing GPU logs (audit-likelihood v1-1, audit-gm v2-1, panel-error-effects v1-1, oracle-make-valid v1-1, oracle-isochrones v2-1, tile-merge v1-1); each matches its status file.

## 1. What the pass is

Purpose (`COMMON.md`): after the 2026-11-17 pass, passing gates had repeatedly not meant correct (class counts stayed zero while the GPU suite passed). Five issues: independent oracle for every output port; spatial statistics never audited line by line; last fix rounds never re-reviewed; intermittent cost-distance continuation failure; functional limits (panel-error effects, multipart predicates, 4,096 row-tile ceiling, robust restriction, iterated SUR, latent spatial probit). Roadmap section KI-01 to KI-10. Rules: independent expected values (Python literals, closed form, hand fixture); a fix needs a test that failed on old code; no edits to lead-owned files (roadmap, CHANGELOG, lens records, lenses, kepler, website, examples); streams cannot run GPU specs.

Mechanics (`orch.py`, `orch2.py`, `orch3.py`, `round2.py`):
- `COMMON.md` written 2026-11-18 20:02; `orch.py` started 20:07 (42 streams, 8 at once), `orch2.py` 21:13 (14 oracle streams, 5 at once), `orch3.py` 21:41 (9 recheck streams, 2 at once), `round2.py` 21:42 (4 at once). About 21.5 h elapsed. Implementor `gpt-6.1-sol` (workspace-write), reviewer `gpt-6.2-astra` (read-only). Work is held below 8 GB free disk or above 10 GB swap (41 GB free now).
- Phases: `new`, `implemented`, up to 2 continuation rounds while NOT DONE has content, `continued`, verify v1 (4 runs, 3 implementor fix rounds), `verified1`, reviewer (CLEAN or FIX), `reviewed`, implementor review-fix, `fixed`, verify v2, `done` or `needs-lead` (final). Census streams stop after continuation.
- Verification is automated, no person: biome, scoped `tsc` (errors outside the manifest ignored), named node specs, the two checkers (only findings in manifest files count), then named headless specs (Chromium, WebGPU on Metal, chunks of 12, 2,400 s). "The lead runs GPU specs" means this script. 
- Round 2 (`round2.py`): every `done` stream with verdict FIX gets a second review (3,600 s); if FIX, review2-fix, verify v3, third confirmation. Each orchestrator has its own gates (2 scoped checks, 1 headless run): up to 3 Chromium runs at once.
- By hand: landing, lens recapture, CHANGELOG, roadmap rows, lead-owned files, cross-stream failures (`lead/TODO.md`), resetting `needs-lead` streams (multipart-nearest and multipart-point-joins are `done` with `why` "verification still failing" and all v2 logs FAIL), and the full gate. The roadmap says `yarn verify` is not part of this pass.
- Orchestrator fault: after a Codex timeout, SIGKILL raises `PermissionError` (macOS, group already gone). 85 stream threads died this way and restart from the stored phase (multipart-geometry-forms: 8 restarts before its first review ended at 17:30); 14 round-2 reviews died. For 11 streams `round2=EXCEPTION` with an empty `out/*.review2.md` and `round2.py` never retries: tile-merge, panel-error-effects, iterated-sur, spatial-lag-probit, multipart-predicates, -nearest, -point-joins, cost-distance-flake, oracle-network-structure, -twins, oracle-make-valid. Their round-1 fixes were never re-read, the third issue of the pass.

Phases at 17:57 (65 streams): done 29 (9 census, 20 other), reviewed 5, fixed 2, verified1 2, continued 1, implemented 1, new 4 (implementor running), not started 21. Verify logs: 112 (PASS 49, FAIL 63).

## 2. Stream ledger

Verify: log count then last result (P/F) for v1 / v2 / v3. Reviews: round, verdict, blockers/shoulds/nits. Rd 2/3: "died" = review killed, none exists. `*` = set to done by hand. Files: existing manifest files outside `.typecheck`; census-only streams have none.

| # | Stream | Roadmap | Phase | Verify | Reviews | Rd 2/3 | Files | Outcome |
|--|--|--|--|--|--|--|--|--|
| 1 | tile-merge | LR-26 | done | 2P / 1P | 1:FIX 3/5/0 | died | 25 | merge to 2^24 tiles; <=4096 graphs unchanged |
| 2 | panel-error-effects | ST03 (KI-05) | done | 2P / 1P | 1:FIX 5/1/0 | died | 17 | effects unit/time/twoWay (random refused); new operations |
| 3 | robust-restriction | ST02 (KI-08) | done | 4F / 1P | 1:FIX 4/5/0; 2:FIX 4/3/0 | FIX | 15 | robust HC0/HC1/HAC option; 5 of 9 fixes partial in round 2 |
| 4 | iterated-sur | ST04 (KI-09) | done | 4F / 4P | 1:FIX 4/5/0 | died | 19 | iteration option, iterationCount; 6 existing defects fixed |
| 5 | spatial-lag-probit | ST05 | done | 4F / 1P | 1:FIX 7/4/0 | died | 20 | spatialLag GMM option; 7 blockers fixed; reference match unverified |
| 6 | multipart-predicates | LR-08 (KI-06) | done | 4P / 1P | 1:FIX 5/3/0 | died | 24 | all predicates, DE-9IM, anti join on multipart |
| 7 | multipart-nearest | LR-24 (KI-06) | done* | 4F / 4F | 1:FIX 2/3/0 | died | 13 | multipart nearest; verify never passed (stale shared cases, not rerun) |
| 8 | multipart-point-joins | LR-08 (KI-06) | done* | 2P / 4F | 1:FIX 6/4/0 | died | 35 | multipart buffer/point/line/choropleth; 4 pure CPU functions |
| 9 | multipart-geometry-forms | LR-08 (KI-06) | reviewed | 2P | 1:FIX 4/7/1 |  | 68 | 12 contributors take multipart; 68 files; fix round running |
| 10 | cost-distance-flake | LR-22 (KI-04) | done | 1P / 1P | 1:FIX 2/3/0 | died | 17 | in-place halo visibility (sufficient cause); contract changed |
| 11 | census-raster-a | KI-01 | done | 2P / 1P | 1:FIX 1/5/0; 2:FIX 1/3/0 | FIX | 15 | 295 ports; weak ports given oracles; path validation fix |
| 12 | census-raster-b | KI-01 | done | - | - |  |  | 188 ports, 20 weak (read-only) |
| 13 | census-terrain | KI-01 | done | 1P / 1P / 1P | 1:FIX 2/5/0; 2:FIX 3/3/1 | FIX | 22 | 276 ports; flat status, CPU budget fixed; round-2 fix running |
| 14 | census-network | KI-01 | done | - | - |  |  | 551 ports, 137 weak (read-only) |
| 15 | census-dataframe | KI-01 | done | - | - |  |  | 325 ports, 3 weak (read-only) |
| 16 | census-spatial-a | KI-01 | done | - | - |  |  | 603 ports, 182 weak (read-only) |
| 17 | census-spatial-b | KI-01 | done | - | - |  |  | 317 ports, 82 weak (read-only) |
| 18 | census-spatial-c | KI-01 | done | - | - |  |  | 328 ports, 70 weak (read-only) |
| 19 | census-spatial-d | KI-01 | done | - | - |  |  | 982 ports; 13 unwritten destinations found (read-only) |
| 20 | audit-gwr-a | KI-02 | fixed | 4F / 1F | 1:FIX 5/8/0 |  | 34 | 14 fixes; v1 failed 4 times; review-fix running |
| 21 | audit-gwr-b | KI-02 | done | 2P / 1P | 1:FIX 2/7/0; 2:FIX 2/5/1 | FIX | 23 | Poisson/GTWR/MGWR fixes; round 2 found 2 new blockers |
| 22 | audit-likelihood | KI-02 | done | 2P / 2P | 1:FIX 2/4/0 |  | 43 | real-GPU cancellation, scalar W, tail, near-pole fixes |
| 23 | audit-panel | KI-02 | reviewed | 3P | 1:FIX 6/3/0 |  | 28 | packing, pole, CSR fixes; review running |
| 24 | audit-gm | KI-02 | done | 1P / 2P | 1:FIX 6/5/0 |  | 39 | exact-affine false success fixed (GPU); 11 findings fixed |
| 25 | audit-ols | KI-02 | done | 1P / 1P / 2P | 1:FIX 4/5/0; 2:FIX 1/4/1; 3:FIX 0/0/0 | FIX / FIX | 28 | 11 defects fixed; round 3 still FIX (t tail underflow) |
| 26 | audit-autocorrelation | KI-02 | verified1 | 2P | - |  | 45 | loss flags, Pearson, QuantileLISA fixes; verifying |
| 27 | audit-weights-a | KI-02 | implemented | - | - |  | 20 | implementor running; diagonal metadata defect |
| 28 | audit-weights-b | KI-02 | new | - | - |  | 22 | just started; summary ignores loss flags |
| 29 | audit-permutation | KI-02 | new | - | - |  | 12 | just started |
| 30 | audit-regionalization-a | KI-02 | new | - | - |  | 11 | just started |
| 31 | audit-regionalization-b | KI-02 | not started | | | | |  |
| 32 | audit-dynamics | KI-02 | not started | | | | |  |
| 33 | audit-validation | KI-02 | not started | | | | |  |
| 34 | audit-scan | KI-02 | not started | | | | |  |
| 35 | audit-misc | KI-02 | not started | | | | |  |
| 36 | rr-joins | KI-03 | not started | | | | |  |
| 37 | rr-weights-regression | KI-03 | not started | | | | |  |
| 38 | rr-network-recipes | KI-03 | not started | | | | |  |
| 39 | rr-lenses | KI-03 | not started | | | | |  |
| 40 | rr-tooling-apps | KI-03 | not started | | | | |  |
| 41 | rr-round4-landing | KI-03 | not started | | | | |  |
| 42 | rr-bump | KI-03 | not started | | | | |  |
| 43 | oracle-isochrones | KI-01 | done | 1P / 2P | 1:FIX 2/5/0; 2:FIX 1/3/1 | FIX | 15 | recipe status forwarding fixed (GPU); 4 ports refused |
| 44 | oracle-network-structure | KI-01 | done | 1P / 1P | 1:FIX 3/6/0 | died | 29 | 35 ports pinned, 11 refused; 3 defects fixed |
| 45 | oracle-network-twins | KI-01 | done | 1P / 1P | 1:FIX 3/6/0 | died | 32 | 33 ports pinned; CPU twins honor budgets |
| 46 | oracle-make-valid | KI-01 | done | 3P / 1P | 1:FIX 1/4/0 | died | 11 | 31 pinned, 15 refused; real-GPU diagnostics fixed |
| 47 | oracle-noding-polygonize | KI-01 | done | 3P / 2P | 1:FIX 2/6/0; 2:FIX 3/4/1 | FIX | 19 | 21 pinned, 25 refused; round 2 found 4 partial |
| 48 | oracle-geometry-refinement | KI-01 | done | 3P / 2P | 1:FIX 4/3/0 |  | 22 | 57 pinned, 15 refused; v2 verify failing |
| 49 | oracle-cells-tessellation | KI-01 | reviewed | 2P | 1:FIX 3/7/0 |  | 31 | unwritten requiredCount/diagnostics fixed; v1 test defects |
| 50 | oracle-lines | KI-01 | fixed | 2P / 1F | 1:FIX 3/4/0 |  | 28 | 17 pinned, 20 refused; lineOrders sentinel; fix running |
| 51 | oracle-global-misc | KI-01 | reviewed | 2P | 1:FIX 3/6/0 |  | 18 | 34 pinned, 11 refused; v1 one test failure |
| 52 | oracle-polygons | KI-01 | verified1 | 2P | - |  | 23 | 29 pinned, 26 refused; v1 failed 17:52 |
| 53 | oracle-point-process | KI-01 | new | - | - |  | 26 | just started |
| 54 | oracle-joins-interpolation | KI-01 | not started | | | | | waits for multipart streams |
| 55 | oracle-trajectories | KI-01 | not started | | | | |  |
| 56 | oracle-dataframe-raster | KI-01 | not started | | | | |  |
| 57 | recheck-status-raster-a | KI-01 | reviewed | 1P | 1:FIX 2/4/0 |  | 26 | 120 ports: 92 ok, 28 strengthened; CPU budgets bind |
| 58 | recheck-status-raster-b | KI-01 | done | 1P / 1P / 1P | 1:FIX 1/3/0; 2:FIX 2/1/0; 3:FIX 0/0/0 | FIX / FIX | 18 | 58 ports; resample endpoint fix + kepler line |
| 59 | recheck-status-terrain | KI-01 | continued | 1F | - |  | 18 | 23 of 103 ports done; running |
| 60 | recheck-status-network | KI-01 | not started | | | | | waits for 3 oracle streams |
| 61 | recheck-status-dataframe | KI-01 | not started | | | | |  |
| 62 | recheck-status-spatial-a | KI-01 | not started | | | | |  |
| 63 | recheck-status-spatial-b | KI-01 | not started | | | | |  |
| 64 | recheck-status-spatial-c | KI-01 | not started | | | | |  |
| 65 | recheck-status-spatial-d | KI-01 | not started | | | | | waits for 5 streams |

## 3. Defects found in existing code (commit 9e93cbc7)

From the audit, oracle, census and recheck streams, plus defects capability streams hit in existing code. Kind: W wrong result, S false or wrong status, U port accepted but never written (xN ports; now refused at construction or written), P caller props retained after construction, V invalid input accepted, L loss flag ignored, T CPU twin ignores a budget or disagrees. Evidence: G✓ real WebGPU failure in the lead's log (I read it), G quoted from a status file, N node test the stream says failed on old code, R reviewer reproduction, R2/R3 later review round. Fix: F claimed, no later review yet; C confirmed by a later round; Q a later round says partly fixed; O open.

116 entries. Family: regression and inference 61, statistics and weights 9, network 10, geometry/topology/joins 27, raster/terrain/cost 9. Kind: W 52, S 11, U 16 entries (137 ports), P 9, V 11, L 10, T 7. Fix: F 88, C 13, Q 10, O 5. Not fixed: 13 more accepted-but-unwritten destinations found by census-spatial-d (7 `GPUNearestFeatureWeights` status fields, 5 predicate-join weights status fields, locations `iterationCount`), and the census backlog (3,865 ports: 622 `none`, 65 `self`, 46 `shape`, 243 `twin` before any oracle stream). Five most serious: D014 (exactly affine data gets finite spatial-error inference, on Metal), D026 (error-ML p-value 1.8% off, on Metal), D034 (panel-lag within-lag loses 20%), D053 (panel-error ML reports success on exact fits, 3 of 98 GPU cases), D062 (Gamma and Pearson statistics from truncated edges). No implementor rejected a reviewer finding. Later reviewers disputed fixes: of 53 round-1 findings re-read in 7 streams, 30 fixed and 23 partly (partly: audit-ols 5 of 9, robust-restriction 5 of 9, audit-gwr-b 4 of 9, noding-polygonize 4 of 8, census-terrain 2, isochrones 2, recheck-raster-b 1). Round 3 on audit-ols: R10 to R13 fixed, R14 partly (FIX).

**audit-ols**
- D001 W getChiSquareSurvival: survival at +Infinity df gives NaN, true 0 [N,F]
- D002 P GPUOrdinaryLeastSquares,..: retain caller props: validated predictorCount 1 became 2 [N,F]
- D003 W GPUOrdinaryLeastSquares ..: Breusch-Pagan NaN/NaN when squared residuals are constant positive; true 0/1 [R,C]
- D004 W GPUOrdinaryLeastSquares ..: held-out R2 of translated outcomes 0, true -1 (metrics WGSL origin) [R,C]
- D005 W cpu-regression-linear-al..: never terminates on rank-deficient input (>2000 ms deadline) [R,C]
- D006 V named OLS operations: rowCount 8 on 4 rows, and 0, accepted [R,C]
- D007 W ordinary-least-squares-s..: padded readback invents a coefficient: df 6->5, sigma2 4/3->1.6 [R,Q]
- D008 P GPUSpatialRegimeRegression: drops an inherited mask getter [R,C]
- D009 W chi-square helper: sf(10000,10000)=0.50053, SciPy 0.49812 (unconverged series) [R,C]
- D010 W OLS residual diagnostics: kurtosis/Jarque-Bera nonfinite at response scale 2^32 and 2^-40 [R,Q]
- D011 W Student-t probability: t=1e-6 df=65534 gives 1 (0.99999920); t=1e160 df=1 gives 0 (6.4e-161) [R,Q]
- D012 W round-1 fix of the scale..: 5 new defects from round-1 fixes: KB decision reversed, BP raw-RSS gate, Jacobi guard, accessor reread, t at df 1e20 gives p>1 [R2,C]
- D013 W Student-t df>=1e10: t=38.48 df=1e10 returns 0, true 6.08e-324 (round 3 R14, still open) [R3,O]

**audit-gm**
- D014 S GPUSpatialErrorGM: exactly affine data: code 0 and finite lambda inference, expected code 3 (lead v2-1: 2 of 181 failed) [G✓,F]
- D015 V robust 2SLS covariance: 13-row residual capacity throws design length 39 [N,F]
- D016 W regression diagnostics: response offset 4096 erases projected norm (16 to 0), LM tests lost [R,F]
- D017 W 2SLS first stage: saturated first stage reports weak-instrument F=0, expected NaN [N,F]
- D018 V GPUInstrumentalVariables..: accepts a constant instrument (centred norm 4.4e-16 passes) [R,F]
- D019 S robust model readback: labelled maximum-likelihood; covariance failure code lost [N,F]
- D020 W Moran variance in diagno..: +-2.8e-17 instead of 0 gives a finite p [N,F]
- D021 W homoskedastic GM weighting: covariance determinant 4.78e40 becomes inf after response scale 2^18 [R,F]
- D022 V three GPU 2SLS operations: ignore rowCount (8 vs 12 rows accepted silently) [N,F]
- D023 V HAC kernel validation: nonzero first CSR offset accepted, covariance code 0 (expected 4) [N,F]
- D024 L 2SLS/HAC: kernel overflow/candidateOverflow loss flags ignored: status 0, expected 4 [N,F]
- D025 P GPUInstrumentalVariables..: retains caller maps: coefficientCount 15 after construction, validated 1 [N,F]

**audit-likelihood**
- D026 W error ML, scalar-profile..: c=-3: intercept p-value rel. error 0.01849 vs 0.005 (v1-1: 4 of 150 failed); Gram cancellation [G✓,F]
- D027 W error ML, W=cI: lambda SE 0.1381 (Q=3), 0.1515 (Q=17), true NaN: flat profile, singular information [N,F]
- D028 S lag ML, W=3I: varianceDefined=1 and significant rho on a constant likelihood [R,F]
- D029 W tail extrapolation bound: published bound 0.00165 below observed error 0.00198 [R,F]
- D030 W pseudo-R2: NaN at response scale 2^-40 / 2^34 and for weak predictions [N,F]
- D031 W exact determinant near a..: -36.7368 vs exact -36.6190 [R,F]
- D032 S profile search at a stat..: reflected case upper {.25,false} vs lower {-.25,true} [R,F]
- D033 P GPUSpatialLagImpacts + o..: retain caller props (CPU key "changed", GPU Q=1 vs 8) [N,F]

**audit-panel**
- D034 W panel lag ML pack: lag before demeaning loses 20% of within-lag (.00048828 vs .00061035) [G,F]
- D035 W panel lag ML profile: random affine residual error .00142 vs .00073; ring rho error 1.8e-5 [G,F]
- D036 S panel lag ML poles: zero-RSS pole reported status 0, rho .49999997, loglik 117.8 [N,F]
- D037 P GPUSpatialPanelRegression: retains caller props (periodCount 4 to 2) [N,F]
- D038 V GPUSpatialPanelRegression: CSR offsets [0,2,4,6,9] and [0,2,1,6,8] silently clamped to a valid operator [N,F]

**audit-gwr-a**
- D039 P GWR, nonstationarity tes..: retain caller props (predictorCount 2 to 1) [N,F]
- D040 L GPULocalCollinearity: weights loss flags and truncated counts ignored: partial diagnostics with valid statuses [N,F]
- D041 W GWR local/global R2: finite R2 for a constant response; translated singular design status 0 on all nine rows [R,F]
- D042 W nonstationarity test: identical draws finite z; undefined spreads give p=1/3; permutationCount=0 acts as one draw [N,F]
- D043 W GWR adaptive search: default bounds select 23 neighbours, 8 is maximum [R,F]
- D044 W GWR candidate accounting: invalid bandwidths counted as 8 fitted observations, true 0 [R,F]
- D045 W Student-t helper: tail(1e200,1)=NaN (6.4e-201); critical(1e-25,1)=1.15e18 vs 6.37e24 [N,F]
- D046 V GWR parameters, local di..: Object.prototype names overwrite targets; diagnostic outputs may alias [R,F]

**audit-gwr-b**
- D047 W GPUGeneralizedGWR: masked-query leverage 0.1762 expected 0; prediction-only Poisson offset 100 status 0, fitted Infinity (expected status 4) [N,C]
- D048 W generalized GWR selectio..: Poisson selection picks bandwidth 10, true 0.5; exposure shifts diverge; held-out predictor looks singular [R,C]
- D049 W GTWR: y=1e6+x loses slope (response not centred) [R,C]
- D050 S CPU MGWR; generalized GW..: infinite variance marked valid; aliased outputs accepted; props reread [R,Q]
- D051 W round-1 fixes: new Poisson initializer rejects a fit the old solved; leave-one-out rejects every valid bandwidth (R2 blockers R3, R4) [R2,O]

**panel-error-effects**
- D052 L GPUSpatialPanelErrorMaxi..: weights.status.candidateOverflow ignored: status 0, now 1 [N,F]
- D053 S panel error ML, exact fits: exact y=3+2x published success (v1-1: 3 of 98 failed, [0] vs [3]); CPU skipped trace SEs (0 vs 2.2696) [G✓,F]

**robust-restriction**
- D054 P GPUSpatialRegimeRegression: retains caller props [R,C]

**iterated-sur**
- D055 S spatial SUR: all-island: equations [4,4], lambdas NaN, joint status 0 [N,F]
- D056 W GPU SUR covariance: residual product overflows: Sigma inf, true diag 2^125 [N,F]
- D057 L spatial SUR CPU/named fo..: spatial weight loss flags discarded [N,F]
- D058 W spatial SUR: near-unit weights: slope moves 0.01055 under +65536 response shift [R,F]
- D059 V spatial SUR validator, o..: GPU accepts nonzero first CSR offset; exact-fit equation status 0 (true 3) [R,F]

**spatial-lag-probit**
- D060 S GPUProbitRegression spat..: negative edge on masked row: status 0, six finite statistics (true status 2, NaN) [R,F]
- D061 L probit diagnostics: requiredCount>capacity in weights status unseen (status 0) [R,F]

**audit-autocorrelation**
- D062 L GPUGammaStatistic, GPUSp..: finite statistic from truncated edges despite loss flags (6 of 6 tests failed) [N,F]
- D063 W GPUSpatialPearson: x=(i+1)2^32 ring: f32 product overflow gives 0, truth 1 [N,F]
- D064 U GPUQuantileLISA: output.analysisCount accepted, never forwarded or written [N,F]
- D065 L multivariate Geary: candidate-only loss ignored; significance 1-2^-26 and MIN_VALUE accepted (pack to 1/0) [N,F]
- D066 W bivariate Moran: constant second variable: variance 0.0923, true NaN [N,F]
- D067 V GPUSpatialPearson, LOSH ..: missing secondValues falls back; maximumPermutations=0 late; significance 1 accepted [N,F]
- D068 T CPU twin p-value: no f32 floor: z=20 gives 5.5e-89, GPU 1.18e-38 [N,F]

**audit-weights-a**
- D069 W binary weights algebra: possible-diagonal metadata hard-coded false (union/difference/xor/intersection) [N,F]

**audit-weights-b**
- D070 L GPUSpatialWeightsSummary: ignores matrix and chunk loss flags (zero suppression gates) [N,O]

**oracle-isochrones**
- D071 U drive-time-catchment-rec..: isochrone-band-rings status.candidateOverflow never forwarded (lead v2-1: 1 of 94 failed) [G✓,C]
- D072 W raster triangle facility..: pixel-index distance on rectangular pixels: label 2, true 5 [R,Q]
- D073 W cell table/outline, CPU ..: negative-cost nodes enter table; CPU raster publishes negative/Infinity [R,Q]
- D074 U x4 GPUNetworkIsochrones tab..: accepted at construction, no values input exists [N,C]

**oracle-network-structure**
- D075 L GPUNetworkNoding: overflow stays 0 after split candidate loss (incomplete topology) [R,F]
- D076 W GPUNetworkPotentialPathA..: inactive edge capacity keeps previous membership [R,F]
- D077 V neighbourhood/subgraph c..: four loss outputs absent from disjointness checks [R,F]
- D078 U x11 GPUMapMatchedEdgeVolumes..: zone weights and pieces.geometry.sourceIds + status fields accepted, never written [C,F]

**oracle-network-twins**
- D079 T CPU cost matrix, central..: explicit budgets ignored: cost 2 vs Infinity; closeness 2/3 vs 1; routes 1 vs 0; 1025 rounds accepted [N,F]
- D080 P matrix operations: one-to-eight round change after construction: 16 relax nodes vs 2; getter read 3 times [N,F]

**oracle-make-valid**
- D081 W GPUMakeValid duplicate g..: dangle diagnostics [0,1,0,1,2] vs [0,2,0,2,2]; bridge [8,1,1,0,2] vs [8,2,2,0,2] (lead v1-1) [G✓,F]
- D082 W GPUMakeValid exact route: retraced two-point ring edge ids [0,0] vs [0,1] [R,F]
- D083 U x15 GPUMakeValid: invalidCount, iterationCount, converged, iterationLimitReached, approximation x 3 levels accepted, unwritten [N,F]

**oracle-noding-polygonize**
- D084 U x25 GPULineNoding, GPUSnapRo..: 5+5+15 generic status fields accepted, never written [N,F]
- D085 W GPULineNoding: traversal source rows reassigned geometrically; extreme edges NONE/[0,0] vs 0/[0,1] [R,F]
- D086 W GPUPolygonize: inward-spurred square loses its area-16 face [R,Q]
- D087 W x3 GPUPolygonize, offset CP..: round 2: short terminal edge still destroys square; reordering two lines changes bounded face; CPU offset keeps wrong source edge [R2,O]

**oracle-geometry-refinement**
- D088 W GPUPolygonPrecisionReduc..: implicit closure shifts edge ids ([5,6,7,8] vs [4,5,6,7]); .1-step grid rectangle loses its face [R,F]
- D089 W GPUVectorStraightSkeleton: terminal ridge source pair {0,3}, true {0,2}; clockwise corner {0,3} vs {2,3} [R,F]
- D090 T CPU convex/vector skelet..: explicit limits ignored (status [5,5,4,0] vs [0,0,4,4]) [N,F]
- D091 U x15 GPUPolygonPrecisionReduc..: 5 status fields x 3 levels accepted unwritten; props retained [N,F]

**oracle-lines**
- D092 U GPULineMerge: lineOrders row for lines under two vertices never initialised (93 vs 4294967295) [N,F]
- D093 U x18 GPULineClipByPolygon, GP..: geometry.sourceIds + 8 nested status fields accepted unwritten [N,F]

**oracle-cells-tessellation**
- D094 U GPUCellLevelSelection: status.requiredCount never written (poison 2779096485 vs 17) [N,F]
- D095 U x8 GPUDelaunayTessellation,..: optional diagnostics keep stale caller bytes (8 ports) [N,F]

**oracle-global-misc**
- D096 U x11 GPUSphericalPolygonOverlay: 11 inherited result/boundary status fields accepted unwritten [N,F]

**oracle-polygons**
- D097 U x23 GPUBufferSurface, GPUPol..: generic status fields (+ boundary uncertainCount) accepted unwritten [N,F]

**multipart-predicates**
- D098 W CPU feature join distance: exact distance at threshold 6.5 not inclusive; translated disjoint squares matched radius 0/0.02 [R,F]
- D099 W DE-9IM: narrow-hole 10F0FF212 vs 1010FF212; one-ulp squares FFFF1FFF2 vs 2FFF1FFF2; part order changes class [R,F]

**multipart-nearest**
- D100 U GPUNearestFeatureJoin ne..: status.candidateCount never written (3735928559 vs 0) [N,F]
- D101 W nearest traversal: cancellation makes near hits exact; tied edges give inconsistent points [R,F]

**multipart-point-joins**
- D102 W x5 GPUBufferSelection, line..: CPU drops zero-radius exact hit; shared-edge length split inconsistently; invalid edges use candidate slots; empty multiline batch fails [R,F]

**multipart-geometry-forms**
- D103 S GPUGeometryPredicates: isSimple=1 after dropped crossings ([0,1], capacity 1, required 2) [N,F]
- D104 L GPULineLoopRemoval CPU: explicit intersectionCapacity ignored [N,F]
- D105 U GPULineClipByPolygon: vertexSourceEdges never written [N,F]
- D106 W single-part sourceIds: caller ids ignored ([0,1] vs [17,88]) [N,F]
- D107 W GPUMinimumClearance: multiline parts counted as separate features [N,F]

**census-raster-a**
- D108 V GPUCostDistancePath: candidateOverflow view not validated (float, two rows, overlap accepted) [N,F]

**cost-distance-flake**
- D109 T CPUCostDistanceOperation: maximumIterations/maximumTieIterations ignored: budget 1 gives [0,1], count 2, converged 1 [N,F]

**census-terrain**
- D110 U GPUTerrainFlow: flat statuses never written (graph omits flow-flats-finalize) [N,Q]
- D111 T CPU terrain watersheds a..: explicit budget ignored (converged 1, true 0); CPU flow certifies flats after exhausted budget [N,Q]

**recheck-status-raster-a**
- D112 T CPU raster watershed: one pop leaves cell 2 undiscovered yet converged=[1], limit=[0] [N,F]
- D113 T CPU raster walking cost: maximumIterations ignored: [6,1,0,0] vs [2,0,1,0] [N,F]
- D114 P GPU+CPU watershed/walkin..: budgets reread after construction [N,F]

**recheck-status-raster-b**
- D115 W GPURasterTemporalResample: observation at start of a long gap dropped: [..NaN,11,NaN]/[1,2,2,0,1] vs [NaN,7,NaN,11,NaN]/[1,0,2,0,1] [N,Q]
- D116 S x2 GPURasterTemporalResampl..: date-helper endpoints status [1,1] vs [0,0]; kepler default grid loses last observation [R2,O]

## 4. New capability and API changes in the clone

Additive options and outputs (behaviour unchanged when absent unless stated):
- tile-merge: exported `REGRESSION_TILE_MAXIMUM_TILE_COUNT = 2^24` replaces the 4,096 refusal (GM, 2SLS, weights traces, GWR, likelihoods); a group node above 4,096 tiles; `estimatedCommandNodeCount` with a 32,768-command limit, so some newly accepted layouts are refused; GWR above 4,096 uses pairwise grouping (float32 rounding changes).
- panel-error: `effects: 'unit'|'time'|'twoWay'|'random'` (default unit; `random` refused, covariance not representable); new GPU/CPU `SpatialPanelErrorMaximumLikelihoodOperation`, `computeSpatialPanelErrorMaximumLikelihoodOnCPU`, three diagnostic status constants, `entityCount`; GPU time/twoWay capped at predictorCount 15.
- robust restriction: `robust` (HC0, HC1, Kelejian-Prucha HAC) on `GPURestrictedRegimeRegression`, operation, CPU twin; exports `RestrictedRegimeRobustOptions`, `RestrictedRegimeRobustOutput`, `RESTRICTED_REGIME_ROBUST_*`.
- iterated SUR: `iteration {maximumIterations 1-256, tolerance}`, runtime `parameters` view, `output.iterationCount`, `GPU_SUR_STATUS_INVALID_PARAMETERS=5`.
- probit: `spatialLag {weights, output {coefficients, standardErrors, status}}` (linearized GMM), `GPU_PROBIT_SPATIAL_LAG_STATUS` 0 to 5, weights status count/requiredCount. Stream simulation (2,000 rows, 50 seeds): mean rho 0.217 at 0.2, 0.855 at 0.6.
- multipart: every predicate with `pattern`, `distance`, `engine`, `how`, opt-in `relate`, anti-join `unmatchedRows`; nearest modes; point-in-polygon `featurePolygonMembership`, `pointMatchCounts`; `GPULineLengthPerPolygon.featureOffsets`; pure `computeBufferSelectionOnCPU`, `computePointInPolygonJoinOnCPU`, `computeLineLengthPerPolygonOnCPU`, `computePointsInPolygonsChoroplethOnCPU` with operation factories; `featureOffsets` on 14 geometry contributors.
- CPU budgets now bind: `CPUCostDistanceOptions` (`maximumIterations`, `maximumTieIterations`, `sweepAfterIteration`), network matrix/centrality/alternative routes, raster watershed/walking, terrain watersheds (`maximumIterations`), skeleton/circle/hull limits.
Removed or narrowed (no alias): `GPUTopologyStatusPort` loses invalidCount, iterationCount, converged, iterationLimitReached, approximation (refused for noding, snap noding, polygonize, make-valid, precision reduction, buffer surface, overlay, spherical overlay, line clip/split); `pieces.geometry.sourceIds` of noding; map-matched `zoneOutWeights/zoneInWeights`; isochrone `table.sums/sumValues/minimums/maximums` throw. Also `getChiSquareSurvival(+Inf)=0`, OLS decoder `predictorCount`, `robust.kernel.candidateOverflow`, robust readback `covarianceInference`. Published numbers change where a defect was fixed (OLS diagnostics, GM exact-affine code 3, GTWR, panel lag, Pearson).
Lead-owned areas touched: one line in `modules/kepler/src/analysis/raster-timeseries.ts` (`gridEnd`, recheck-status-raster-b) and two website catalog files (multipart-geometry-forms: the catalog said multipart is rejected).
Docs: 33 tracked `docs/` files (+1,459/-134), new `docs/api/gpu-terrain/output-oracles.md` with a `docs/table-of-contents.json` entry. `dev-docs/regression-loop-inventory.md`: three tile-ceiling rows rewritten. `dev-docs/roadmap.md`: +21 lines, only a new "Known-issues pass 2026-11-18 (KI)" section (KI-01 to KI-10, no full `yarn verify`); no existing row (LR-08, LR-22, LR-24, LR-26, ST02 to ST05, Known limits) updated. `CHANGELOG.md` untouched; streams request entries.

## 5. cost-distance-flake conclusion

Symptom: `gpu-cost-distance-continuation.spec.ts` failed 2 of 10 runs on alpha.16 (167 and 163 rounds against 168, equal costs). Conclusion: a sufficient cause, not the proven historical one. Halo cells are read in place with atomics, so a tile can see neighbours' writes of the same round; the fixed point is deterministic, the round count is not. Evidence: lead v1-1 forced-visibility GPU controls (2 versus 9 rounds, identical corridor costs); float32 host model of the 256-cell spiral gives 168 versus 163 rounds at 192 versus 256 tiles visible first round, both surfaces SHA256 `9f69bcb1...4526`, equal to a Python float32 Dijkstra over 33,984 reachable cells; serpentine 344 rounds with the sweep switch at 96, 405 without. Not observed: the Metal workgroup order; no frequencies or timings exist. Change: contract, not kernel. Counts (`iterationCount`, `settledAt`, terrain fill/flat/pull) documented as schedule-dependent with bounded increments; specs assert convergence, independent Dijkstra costs and per-submission bounds (previous < count <= previous + budget, before any convergence branch) instead of count equality; forced schedules keep exact counts; the 40-repeat matrix moved to `gpu-cost-distance-round-determinism-benchmark.spec.ts` (opt-in report). Also fixed: `CPUCostDistanceOperation` ignored budgets (budget 1 gave costs [0,1], count 2, converged 1); new `cpu-raster-relaxation.ts`. No GPU node changed. Round 2 died; v2-1 PASS.

## 6. Not finished or not verified

- Not started (21): audit-regionalization-b, -dynamics, -validation, -scan, -misc; all seven rr-* re-review streams (KI-03 untouched); oracle-joins-interpolation (waits for the multipart streams), oracle-trajectories, oracle-dataframe-raster; recheck-status-network, -dataframe, -spatial-a to -d. Running, early: audit-weights-a/b, audit-permutation, audit-regionalization-a, oracle-point-process.
- Verification never passed: multipart-nearest (8 FAIL), multipart-point-joins (v2 4 FAIL), audit-gwr-a (v1 4 FAIL), oracle-geometry-refinement v2-1, oracle-global-misc v1-1, oracle-polygons v1-1 (17:52). The four stale `x1 X3 validity rejects multipart` cases in `lead/TODO.md` were rewritten by multipart-point-joins; multipart-nearest was not rerun. Outside causes blocked many streams: a missing terrain sidebar entry (robust-restriction, iterated-sur, spatial-lag-probit v1 x4) and a `GPUQuantileLISA` doc-owner clash (lead edited).
- Open after review: round-2 FIX in robust-restriction (4B/3S), census-terrain (3B/3S), audit-gwr-b (2B/5S), audit-ols (round 3 FIX), oracle-isochrones, oracle-noding-polygonize (3B/4S), recheck-status-raster-b (round 3 FIX at 17:53); the 11 died round-2 streams. All 29 first-round reviews returned FIX (97 blockers, 136 shoulds).
- GPU specs written, no device run yet (file mtime newer than the stream's last PASS verify, or no verify; my inference): audit-gwr-a 8, audit-panel 9, audit-autocorrelation 10, audit-weights 14, oracle-cells-tessellation 13, oracle-global-misc 11, oracle-polygons 12, recheck-status-terrain 17, plus review-fix cases of multipart-geometry-forms, oracle-lines (5), oracle-noding-polygonize, recheck-status-raster-b. Older "GPU pending" lines are stale where a later v2 PASS exists.
- Unverified: probit versus R `spprobit`/spreg; the cost-distance cause; recheck-status-terrain did 23 of 103 ports.
- Lead actions pending: lens recapture (OLS, GM, diagnostics, spatial-regression recipe, panel lag ML, panel regression, local collinearity with weights flags, exact lag/error ML, precision reduction); `examples/deck/spatial-analysis-explorer/modes/regression-result-readouts.ts:79` robust label; `cost-distance*.lens.ts` wording; website NaN p-value readouts; CHANGELOG; roadmap LR-22, LR-26.
- Gaps by decision: no production CPU twin or operation for GWR, nonstationarity test, local collinearity; panel-error random effects refused; no stream builds the KI-01 coverage record and node gate.
- Same-file edits: 35 of 756 manifest files are in more than one manifest (briefs allow it): `spatial-regression/index.ts` (6 streams), noding, polygonize, snap-noding, topology-status, line-clip, shared-paths, line-split, buffer-surface (multipart-geometry-forms with oracle streams), `cpu-feature-join.ts`, `spatial-join/index.ts`, `gpu-spatial-error-gm.ts` and other regression files (audit-* with tile-merge), `gpu-terrain-flow.ts`. One cross-stream break so far (GM requested workgroupSize 1; fixed by its owner).
- Landing needs more than ran. In the clone: per-stream biome, scoped tsc, named node specs, global upstream-first and contributor-contract checkers, named headless specs on Metal. No log shows `yarn typecheck`, test typecheck, `yarn build`, export checks, a whole node project or `yarn verify`.

## 7. Size (17:57)

`git diff --shortstat`: 308 files changed, 15,709 insertions, 2,677 deletions; 553 untracked files. By area (861 files): `modules/analysis/src` 236, `modules/analysis/test` 583 (most new; 74 oracle `.py`), `docs` 36, `dev-docs` 2, `modules/kepler` 2, `website` 2 (catalog, by multipart-geometry-forms), `modules/lenses`, `examples`, `scripts`, `docs/lens-records` 0.
