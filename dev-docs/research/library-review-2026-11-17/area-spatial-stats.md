# Spatial statistics and regression contributors

The first fixes I would merge are the weights-spectrum check, conformal rank calculation, centered prediction, and geographical-detector upper tail. Each has a small counterexample with an independently calculable answer. The first and fourth reproduced in extracted CPU functions. The shader findings are marked `UNVERIFIED` where I checked the arithmetic and source but did not execute the GPU path.

## What I checked

I reviewed commit `bdc715ea`. The 19 requested directories contain **259 TypeScript source files, 86,166 source lines, and 332 same-directory specs**. Spatial regression accounts for **106 files, 40,726 lines, and 142 specs**, including 65 `.node.spec.ts` files. Specs elsewhere also cover these contributors, so directory counts are lower bounds on coverage.

I use `S` for `modules/analysis/src/gpu-spatial-analysis`, `R` for `S/spatial-regression`, and `T` for `modules/analysis/test/gpu-spatial-analysis`.

- **Read in full** — regression runtime kernel selection, large-sample inference, scalability utilities, binding plans, output dispatches, GWR model result, MGWR graph operation, and the exact-determinant Node spec. Outside regression I read the geographical-detector p-value helper, CPU multiple-testing operation, cross-validated OLS contributor, and spatial-statistics count arithmetic in full.
- **Read selected paths end to end** — parameter packing, fit, status publication, and readback in the largest GWR source and its local-model siblings. I followed the recent indexed GTWR and tiled panel-lag paths from validation into their kernels and specs, and sampled ML, GM, 2SLS, probit, both regime layouts, permutation inference, autocorrelation, weights, smoothing, scoring, and distribution dynamics.
- **Limits** — the remaining directories received a census and targeted reads, not a line-by-line mathematical audit. I did not establish an independent oracle for every model, inspect generated shader binaries, measure GPU time, run the repository checker, compile, or run tests. The loop counts below are from the committed baseline. No files changed.

### Regression family at the public boundary

| Family | Fit and parameter shape | Outputs, inference, and status | Evidence |
|---|---|---|---|
| GWR | Runtime kernel, bandwidth mode, ladder, and significance in a packed view | Local coefficients required. Fitted values, residuals, SEs, tests, hat diagonal, summary, and bandwidth selection optional. Per-row `localStatus` and approximation status have different purposes. | `R/gpu-geographically-weighted-regression.ts:112`, `:149`, `:191`, `:208` |
| Generalized GWR | Poisson/logistic family, optional offset, packed bandwidth and tolerance, separate bandwidth-selection object | Coefficients and fitted values required. `inference` takes SE and influence views directly. `status.codes` and `iterationCount` are both **per row**. | `R/gpu-generalized-geographically-weighted-regression.ts:99`, `:118`, `:158`, `:187` |
| MGWR | Packed tolerance and per-component bandwidths. Optional candidate view and selection policy | Coefficients and fitted values required. Known-variance conditional SEs live under `output`. Coupled backfit inference lives under `inference.output` with its own status. `status.codes` is per row, `iterationCount` is **one scalar**. | `R/gpu-multiscale-geographically-weighted-regression.ts:76`, `:87`, `:101` |
| GTWR | Relative times with explicit temporal context, compile-time kernel, packed bandwidth/time scale | Coefficients, fitted, hat diagonal, `localStatus`, and summary required. Residuals optional. The optional `status` describes bandwidth search. | `R/gpu-geographically-temporally-weighted-regression.ts:151`, `:170`, `:185`; `R/gtwr-bandwidth-search.ts:52` |
| MGTWR | Packed initial and component bandwidth/time pairs, fixed `iterationCount` | Last-iteration coefficients, fitted, residuals, `localStatus`, and a score per sweep. Optional `status` aggregates **search** convergence. Consumers infer backfit convergence from `scores`. | `R/gpu-multiscale-geographically-temporally-weighted-regression.ts:85`, `:104`, `:115`, `:137` |
| Panel lag / panel error ML | Balanced period-major panel and entity weights. Lag selects random/unit effects and reference/tiled fit. Error implements unit effects. | Lag publishes coefficients, covariance, summary, scalar status, transformed fitted values and innovations, plus optional Hausman comparison. Error inherits the cross-sectional ML shape and adds optional pooled diagnostics. | `R/gpu-spatial-panel-lag-maximum-likelihood.ts:80`, `:98`, `:116`; `R/gpu-spatial-panel-error-maximum-likelihood.ts:57`, `:69` |
| Probit | Binary response, mask, iteration budget, optional weights for diagnostics | Global coefficients, probability-scale fitted values, and converged-fit SEs. Status 3 publishes the last fit. Optional spatial tests have their own status. | `R/gpu-probit-regression.ts:114`, `:126`, `:135` |
| Fully varying / restricted regimes | Regime IDs and counts. Restricted fit adds a `common` flag per source column. | Fully varying coefficients are **regime-major** with per-regime status. Restricted coefficients are **source-column-major**, with pooled covariance, Chow test, optional stability tests, and model-level status. | `R/gpu-spatial-regime-regression.ts:46`; `R/gpu-restricted-regime-regression.ts:43` |

Two differences are especially easy for a consumer to miss. **Status 1 means singular in GWR/GTWR/MGTWR and excluded in GGWR/MGWR.** And a successful search, successful local fit, and converged backfit are three separate facts. A common decoder would help immediately. Renumbering existing buffers would break consumers. `R/geographically-weighted-regression-parameters.ts:122`; `R/gpu-generalized-geographically-weighted-regression.ts:88`; `R/gpu-multiscale-geographically-temporally-weighted-regression.ts:96`.

## Findings

### SSR-01 Rescaling a nonreversible weights matrix makes the exact-spectrum helper accept it [P0] [correctness]

- **Evidence** — The helper promises relative tolerance, checks `Math.abs(a - b) > 1e-4 * Math.max(1, Math.abs(a), Math.abs(b))`, then replaces both entries by their average. I used the existing three-row nonreversible fixture. Its original weights throw. Multiplying every weight by `1e-6` passes. At `rho = 500000`, the returned spectrum gives log determinant **−0.468227430**. Direct evaluation of the 3×3 determinant gives **−0.227842072**. The product `rho W` is identical to the original matrix at `rho = 0.5`. `R/spatial-exact-log-determinant.ts:27`, `:60`, `:79`, `:82`; `T/spatial-regression/spatial-exact-log-determinant.node.spec.ts:81`.
- **Why it matters** — The public helper silently manufactures a spectrum for a different matrix. That spectrum is an input to exact likelihood fitting.
- **Proposed change** — Check positive finite similarity scales and use a dimensionless reversibility residual. Extend the existing fixture with joint `W`/`rho` rescalings and a reversible control. **S**, corrective rejection of previously accepted invalid input.
- **Confidence** — High. Reproduced in the extracted public CPU functions.

### SSR-02 Conformal rank slack moves valid interval endpoints inward by multiple observations [P0] [correctness]

- **Evidence** — The documented endpoints use `ceil((1−α)(n+1))` and `floor(α(n+1))`. The shader multiplies the products by `1−2^-20` and `1+2^-20` before rounding. For **8,388,607** calibration rows and **α = 0.25**, the correct upper/lower ranks are **6,291,456 / 2,097,152**. The implemented arithmetic gives **6,291,450 / 2,097,154**. Both unadjusted products are exactly representable. The helper comment says the slack moves a rank by one. `S/spatial-validation/gpu-conformal-prediction-intervals.ts:100`, `:108`, `:306`; `S/spatial-validation/conformal-prediction-nodes.ts:29`, `:123`.
- **Why it matters** — Both shifts can narrow the published interval. The documented finite-sample coverage guarantee depends on the order statistic actually selected.
- **Proposed change** — Extract a rank helper using the packed alpha significand and integer count, with an explicit rule for decimal boundary cases. Test it directly against integer/rational ranks, including this example and the existing `0.1` cases. A multi-million-row GPU allocation is unnecessary for the regression test. **M**, corrected outputs, same port shapes.
- **Confidence** — High for the arithmetic. GPU execution `UNVERIFIED`.

### SSR-03 Held-out OLS and regime predictions discard the numerical protection of the fit [P0] [correctness]

- **Evidence** — OLS deliberately evaluates fitted values in centered coordinates and carries a correction for the stored means. Cross-validated OLS fits each training fold with that contributor, then predicts using the rounded raw intercept and `value += predictor * coefficient`. Fully varying regime regression does the same after solving centered moments. For `y = 1 + 1.5(x − 2^24)`, the exact raw intercept is `−25,165,823`, which rounds to `−25,165,824` in float32. Even with an exact slope, raw-coefficient prediction at `x = 2^24` gives **0 instead of 1**. `R/gpu-ordinary-least-squares.ts:148`; `S/spatial-validation/gpu-spatial-cross-validated-ols.ts:153`, `:197`, `:215`; `R/gpu-spatial-regime-regression.ts:202`, `:223`.
- **Why it matters** — Adding a constant to a predictor can change held-out errors and regime fitted values. Cross-validation then reports the discrepancy as model error.
- **Proposed change** — Retain the training-fold centers and correction for held-out prediction. Reuse regime moments for the regime prediction pass. Keep the raw public coefficient representation. Add translation-invariance fixtures with representable predictor increments. Probit already documents and implements this distinction. **M**, same API. `R/gpu-probit-regression.ts:137`.
- **Confidence** — High for the representation error. End-to-end GPU result `UNVERIFIED`.

### SSR-04 The geographical-detector upper tail loses small probabilities and can fail to terminate [P0] [correctness]

- **Evidence** — `getNonCentralFUpperTail(2**60, 2, 2, 0)` returned **0**. The central `F(2,2)` survival probability is `1/(1+2**60)`, approximately **8.67×10⁻¹⁹**, representable in Float64. The implementation forms a CDF argument that rounds to one and subtracts the CDF from one. Separately, at noncentrality `2**55`, the loop starts at `mode = 2**54`, where `index++` leaves the JS number unchanged. The loop bound remains true and its `index > rate` exit remains false. `S/geographical-detector/geographical-detector-p-value.ts:88`, `:93`, `:96`, `:106`, `:116`.
- **Why it matters** — A public p-value becomes exactly zero. A finite input can also hang the calling thread.
- **Proposed change** — Evaluate the complemented beta tail directly, sum the survival terms, and bound iterations with a separate small integer counter. Define the supported large-noncentrality range and provide a diagnosed failure or validated approximation beyond it. Add a relative-error tail fixture and a termination-boundary fixture. **M**, behavioral fix.
- **Confidence** — High. Reproduced the zero and checked the nonadvancing cursor independently, without running the infinite loop.

### SSR-05 GWR accepts an inference level that becomes the default after packing [P1] [correctness]

- **Evidence** — The GWR packer validates a JS number in `(0,1)` and writes it into a `Float32Array`. `significanceLevel: 1 - 2**-27` passes and packs as `1`. The shader then selects the default **0.05**. Likewise, finite `bandwidth: 1e100` passes the host finite check and packs as infinity. GTWR already rounds before validating. The GWR packer also clears the caller’s target before finishing validation. `R/geographically-weighted-regression-parameters.ts:255`, `:305`, `:311`, `:327`; `R/gpu-geographically-weighted-regression.ts:1502`; `R/gpu-geographically-temporally-weighted-regression.ts:95`.
- **Why it matters** — A caller can receive significance masks for a materially different inference level without an error.
- **Proposed change** — Validate the `Math.fround` representation of every supplied value before mutating the target. Reserve the default encoding for an omitted level. **S**, newly rejects unrepresentable settings.
- **Confidence** — High for host packing and branch selection. GPU output `UNVERIFIED`.

### SSR-06 MGWR drops the last fit on GPU and ignores the caller’s iteration budget on CPU [P1] [consistency]

- **Evidence** — The contributor contract says an iteration-limit result “publishes what it has.” GPU MGWR retains status 3 for a finite unfinished sweep, then publishes a coefficient only for `code == 0u`. GGWR explicitly publishes its finite limit iterate. CPU MGWR documents that it ignores `maximumIterations`, runs until convergence, failure, or repetition, and retains a serialized copy of every distinct complete state in `Set<string>`. `dev-docs/contributor-contract.md:27`; `R/gpu-multiscale-geographically-weighted-regression.ts:673`, `:681`, `:732`; `R/gpu-generalized-geographically-weighted-regression.ts:195`; `R/cpu-multiscale-geographically-weighted-regression-operation.ts:140`, `:245`, `:390`, `:404`.
- **Why it matters** — The same explicit budget can yield NaNs on GPU and a converged fit after many more sweeps on CPU. CPU history storage grows with rows × coefficients × sweeps.
- **Proposed change** — Publish finite coefficients and fitted values for status 3 on both backends. Keep converged-fit inference gated. Honor an explicitly supplied budget on CPU. If unlimited CPU fitting must remain available, name that policy and use bounded-memory cycle detection. **M**, behavioral migration for callers using NaN as their convergence test.
- **Confidence** — High for source and documented CPU behavior. GPU execution `UNVERIFIED`.

### SSR-07 An explicit MGWR grid bypasses the documented dense-work limit [P1] [performance]

- **Evidence** — MGWR documents **65,536 dense** and **1,048,576 indexed** rows. Its constructor retains an explicitly supplied grid for untruncated Gaussian fits. Shared validation raises the row limit whenever `indexGridSize` is present. The neighborhood kernel sets `useGrid` false for that kernel and scans `0..ROW_COUNT` for each focus. A planar Gaussian fit with 65,537 rows, an explicit grid, and no cutoff passes this row check and entails over **4.29 billion candidate visits per component per sweep** for complete rows. `R/gpu-multiscale-geographically-weighted-regression.ts:137`, `:164`, `:420`, `:565`; `R/gpu-generalized-geographically-weighted-regression.ts:1048`; `R/local-weighted-least-squares-wgsl.ts:147`, `:187`.
- **Why it matters** — An indexing option lifts the guard precisely when the selected kernel cannot use the index to prune work.
- **Proposed change** — Resolve the traversal first, then select the capacity from the resolved traversal. Test explicit grid plus Gaussian on either side of the dense limit at construction. **S**, rejects configurations exceeding the documented limit.
- **Confidence** — High for control flow and operation count. Runtime and device timing `UNVERIFIED`.

### SSR-08 Capacity-sized regression tables decode padding as model coefficients [P1] [ergonomics]

- **Evidence** — Four contributors accept output tables **at least** as long as the active result. Three readback helpers infer the predictor count from the **entire** table length. A one-predictor lag fit occupies 12 floats. Reading a valid 16-float capacity buffer without slicing it labels the real spatial coefficient as another predictor and labels the padding `W_y`. ML also derives the information-criterion parameter count from that inflated coefficient list. `R/gpu-spatial-lag-maximum-likelihood.ts:476`; `R/gpu-spatial-error-maximum-likelihood.ts:353`; `R/gpu-spatial-error-gm.ts:311`; `R/gpu-spatial-two-stage-least-squares.ts:311`; `R/spatial-maximum-likelihood-model-result.ts:110`, `:162`; `R/spatial-error-gm-model-result.ts:63`; `R/spatial-two-stage-least-squares-model-result.ts:106`.
- **Why it matters** — Both calls look valid separately. Their natural composition produces a plausible, mislabeled model report. Exact-length readbacks are unaffected.
- **Proposed change** — Add `predictorCount` or a fit-layout argument to the readers and decode only the active prefix. Preserve the exact-length overload. Test with distinctive nonzero padding, with and without predictor names. **S**, additive.
- **Confidence** — High.

### SSR-09 The CPU panel-lag graph accepts an eigenvalue port it never reads [P1] [consistency]

- **Evidence** — The GPU operation declares and forwards `eigenvalues`. Tiled panel lag validates their finiteness/range and uses them in the likelihood. The CPU operation copies the GPU input declarations, then its `execute` reads the CSR, predictors, and response and calls its fit without reading the eigenvalue input. The new tiled spec covers three valid ring scenes against the production CPU fit and compares the first two of six summary slots. `R/gpu-spatial-panel-lag-maximum-likelihood-operation.ts:66`, `:108`; `R/panel-lag-tiled-nodes.ts:211`, `:297`; `R/cpu-spatial-panel-lag-maximum-likelihood-operation.ts:654`, `:664`, `:695`; `T/spatial-regression/panel-lag-tiled.spec.ts:63`, `:123`.
- **Why it matters** — A broken upstream spectrum can be invisible in CPU execution. Agreement on valid rings does not exercise this graph dependency.
- **Proposed change** — Have the CPU graph operation consume and validate the selected mode’s spectrum and use it in the objective. Keep the dense CSR determinant as an independent test oracle. Add valid, wrong-length, and nonfinite spectrum cases through both graph entry points. **M**, corrected backend behavior.
- **Confidence** — High for the missing CPU dependency. GPU failure status `UNVERIFIED`.

### SSR-10 One observed p-value can trigger a 16 GiB CPU allocation [P1] [performance]

- **Evidence** — At `minimumFamilySize = 2**31 + 1`, the CPU operation models an enlarged family with a proxy containing implicit `p=1` rows. The helper immediately executes `new Float64Array(rows).fill(NaN)` and collects every tested index. That is **16 GiB plus 8 bytes** for the result alone, even for one real p-value. Below that boundary, Benjamini–Yekutieli also loops over the entire logical family to compute its harmonic factor. `S/multiple-testing/cpu-multiple-testing-adjustment-operation.ts:47`, `:54`, `:67`; `S/multiple-testing/gpu-multiple-testing-adjustment.ts:50`, `:155`, `:187`.
- **Why it matters** — Family size is a statistical parameter. Callers supplying it can exhaust memory independently of the size of their dataset.
- **Proposed change** — Pass logical family size separately to the adjustment formula, sort only observed valid p-values, and use a bounded, error-checked harmonic-number calculation for large families. Test one p-value immediately below and above the boundary. **S–M**, same API.
- **Confidence** — High from allocation and loop bounds. I did not attempt the allocation.

### SSR-11 Gaussian CRPS can overflow an intermediate when the correct score fits in float32 [P1] [correctness]

- **Evidence** — The kernel forms `z = abs(error) / deviation`, then evaluates `deviation * (z * erf(...) + ...)`. With error `1e32` and variance `1e-20`, both inputs and the correct score of approximately `1e32` are representable in float32. The standardized error is about `1e42`. `S/scoring-rules/gpu-gaussian-predictive-scores.ts:179`, `:193`, `:199`.
- **Why it matters** — A finite, near-deterministic forecast can receive a nonfinite score, affecting model comparisons and downstream averages.
- **Proposed change** — Evaluate the leading term as `abs(error) * erf(...)`. Use the existing large-argument branch before forming an overflowing ratio and scale only the bounded correction by the deviation. Add a closed-form large-residual fixture. **S**, same API.
- **Confidence** — High for the overflow boundary. GPU execution `UNVERIFIED`.

### SSR-12 The public exact log-determinant helper disagrees with its absolute-determinant contract [P2] [correctness]

- **Evidence** — The exported helper documents `ln|I - rho W|` and sums `Math.log(1 - rho * eigenvalue)`. At `rho=2`, spectrum `[1,-1]`, the determinant is `−3`. The documented result is `log(3)` and the implementation yields NaN. This example concerns the public helper independently of the fitter’s admissible search interval. `R/spatial-exact-log-determinant.ts:88`; `R/index.ts:375`.
- **Why it matters** — Standalone callers cannot tell whether a NaN denotes a domain restriction or a numerical failure.
- **Proposed change** — Implement log absolute determinant, retaining `−Infinity` at singularity and using `log1p` where appropriate. If the intended API is restricted to positive factors, document and validate that precondition instead. **S**, behavioral clarification.
- **Confidence** — High.

### SSR-13 The serial-loop baseline mixes bounded coefficient work with data-dependent work [P2] [architecture]

- **Evidence** — The committed `single-invocation-data-loop` baseline contains **21 sites in 15 regression files**, **34 in 25 files** across this stream. Two restricted-regime sites are marked pending review, yet the constructor caps the solve at 16 coefficients and the finish loop at 256 covariance entries. Conversely, panel-lag wraps a kernel with `invocationCount: count` and calls it with `1` to scan up to 1,024 pivot candidates. The checker looks for a direct kernel construction with a statically single invocation. Cross-validated OLS also scans all rows per fold with `invocationCount: folds`. `scripts/contributor-contract-baseline.json:101`, `:172`; `scripts/check-contributor-contract.mjs:1083`, `:1093`, `:1113`; `R/gpu-restricted-regime-regression.ts:119`, `:434`, `:531`; `R/panel-lag-tiled-nodes.ts:179`, `:325`; `S/spatial-validation/gpu-spatial-cross-validated-ols.ts:205`, `:215`.
- **Why it matters** — The count alone misdirects optimization work. It can make a small bounded solve look more urgent than a serial data scan.
- **Proposed change** — Record the loop axis, enforced maximum, and route for each reviewed entry. Teach the checker the small local kernel-wrapper pattern. Start by resolving the two restricted-regime reasons and recording the tiled pivot bound. **M**, tooling only.
- **Confidence** — High. `KNOWN(GPU-09)` understates the distinction between the dense determinant, bounded tiled pivot selection, and fixed-size coefficient work.

### SSR-14 The catalog and roadmap misstate three shipped capabilities [P2] [docs]

- **Evidence** — The GTWR catalog says “Gaussian cutoff publishes approximation.” Its optional status port is for profile search, and providing it without a search is rejected. `ST03` describes random-effects panel error as existing, whereas the contributor explicitly supports unit effects only. `ST08` lists more than 64 rank-Markov states as unimplemented, whereas sorted ranking already supports them and the optional reference diagnostics retain the 64-state cap. `S/ontology/all-entry-points/operator-annotations.ts:279`; `R/gpu-geographically-temporally-weighted-regression.ts:178`; `R/gtwr-bandwidth-search.ts:52`; `dev-docs/roadmap.md:117`, `:120`; `R/gpu-spatial-panel-error-maximum-likelihood.ts:69`; `S/distribution-dynamics/gpu-rank-markov.ts:53`, `:74`, `:97`.
- **Why it matters** — Consumers can request an unsupported status port. Maintainers can plan an already-shipped ranking feature or assume a missing estimator exists.
- **Proposed change** — Correct the GTWR catalog wording, split panel error and panel lag effects in `ST03`, and narrow `ST08` to its remaining tie, missing-data, and diagnostic limitations. **S**, documentation only.
- **Confidence** — High. `KNOWN(ST03)` and `KNOWN(ST08)` are stale.

## Cross-cutting patterns

**Sharing is substantial. The best remaining extraction targets are validation and selected numerical kernels.** Of 106 regression sources, 23 are `gpu-` non-operation files, 13 are GPU operation files, 13 are CPU operation files, and 52 have neither prefix and are not the barrel. Of those 52, **34 have at least two in-directory importers** excluding the barrel. `spatial-regression-solve` has 18, `spatial-weights-traces-parameters` 16, `regression-output-dispatches` 15, and `regression-row-tiles` 14.

I found **223 WGSL function definitions** by scanning line-start `fn` definitions and balancing braces. Only **one definition was whitespace-identical across files**, the two `getLogDeterminantDerivative` bodies. This measures exact copies only. Dense and indexed GTWR still separately assemble the same centered normal equations and perform the same equilibrated solve under different variable names. By comparison, six named validation helpers account for **437 call sites**, **305 distinct normalized texts**, and **52 texts repeated across files covering 182 sites**. The CPU lag/error likelihood option validators are a good first extraction into `validateCPUSpatialLikelihoodOptions`. `R/spatial-exact-log-determinant.ts:248`; `R/spatial-maximum-likelihood-wgsl.ts:66`; `R/gpu-geographically-temporally-weighted-regression.ts:592`; `R/gtwr-indexed-nodes.ts:345`; `R/cpu-spatial-lag-maximum-likelihood-operation.ts:80`; `R/cpu-spatial-error-maximum-likelihood-operation.ts:75`.

**The statistical evidence is strong for several established fits and thinner at backend and route boundaries.** Of 142 regression specs, **48 directly import JSON fixtures** and **29 reference a `compute…OnCPU` function**. Those sets overlap, and imported reference helpers make the first count a lower bound on fixture use. There are 13 Python files in the regression test tree. PySAL fixtures identify `spreg`, `esda`, and `libpysal` versions and how weights were transferred. The OLS fixture generator uses SciPy independently of the production twin. GTWR/MGTWR checks external fitted results and explicitly accounts for a known AICc convention difference. In contrast, the new tiled-panel cases all use ring graphs and the production CPU fitter. I would add one irregular reversible graph, an independent determinant, and assertions on boundary flags and convergence fields. `T/spatial-regression/spreg-reference.ts:70`; `T/spatial-regression/ordinary-least-squares-operation-fixtures.py:7`; `T/spatial-regression/gtwr-mgtwr.spec.ts:121`, `:141`, `:308`; `T/spatial-regression/panel-lag-tiled.spec.ts:63`, `:125`.

**Several reference caps are real, and the distinction between capacity and work matters.** GTWR enforces 32,768 rows for exhaustive traversal and 16,777,216 for the indexed route. MGTWR explicitly selects exhaustive initialization and documents quadratic row work. Panel lag enforces separate reference and tiled entity/period limits, including the tiled total-row limit. SSR-07 is a specific exception. Across these classes, capacity should be derived from the same resolved route that constructs the kernels. `R/gpu-geographically-temporally-weighted-regression.ts:63`, `:326`; `R/gpu-multiscale-geographically-temporally-weighted-regression.ts:137`, `:147`; `R/gpu-spatial-panel-lag-maximum-likelihood.ts:130`, `:223`.

**Numerical policy works best when tied to the quantity.** Tiled regression explains its `2^24` count limit and uses a fixed pairwise order for floating sums. Spatial-statistics count arithmetic preserves integer bits above that threshold. Colocation uses a deterministic fixed-point accumulator with a justified scale because each neighbor fraction is in `[0,1]`. These are sound, different choices. Parameter packers, ranks, and prediction adapters need the same precision rationale at their boundaries. `R/regression-row-tiles.ts:26`, `:165`; `S/spatial-weights/spatial-statistics-counts-wgsl.ts:7`, `:39`; `S/categorical-association/gpu-colocation-quotient.ts:42`, `:97`.

<details>
<summary>Census commands</summary>

From the repository root, these reproduce the regression source, spec, fixture-import, CPU-reference, and helper-import counts.

```sh
R=modules/analysis/src/gpu-spatial-analysis/spatial-regression
T=modules/analysis/test/gpu-spatial-analysis/spatial-regression
rg --files "$R" -g '*.ts' | wc -l
rg --files "$T" -g '*.spec.ts' | wc -l
rg -l "(from\s+|require\(\s*)['\"][^'\"]+\.json['\"]" "$T" -g '*.spec.ts' | wc -l
rg -l '\bcompute\w+OnCPU\b' "$T" -g '*.spec.ts' | wc -l
```

The following is the exact-copy census. It preserves interpolated WGSL identifiers and removes whitespace. It does not detect renamed or rearranged implementations.

```sh
python3 - "$R" <<'PY'
from collections import Counter, defaultdict
from pathlib import Path
import re, sys

files = list(Path(sys.argv[1]).glob("*.ts"))
helpers = {p.stem for p in files
           if p.name != "index.ts" and not p.name.startswith(("gpu-", "cpu-"))}
imports = Counter()
validators = r"\bvalidate(?:Requirement|Capacity|SinglePackedView|ViewLength|FiniteNumber|MaximumCount)\s*\("
groups = {"fn": defaultdict(list), "validation": defaultdict(list)}
for p in files:
    s = p.read_text()
    if p.name != "index.ts":
        imports.update(set(re.findall(r"""from ['"]\./([^'"]+)['"]""", s)) & helpers)
    for name, pattern, left, right in [
        ("fn", r"^[ \t]*fn [^\n(]+\(", "{", "}"),
        ("validation", validators, "(", ")")
    ]:
        for m in re.finditer(pattern, s, re.M):
            i = m.end() - 1 if left == "(" else m.end() + re.search(r"(?<!\$)\{", s[m.end():]).start()
            j, depth = i + 1, 1
            while depth:
                depth += (s[j] == left) - (s[j] == right)
                j += 1
            groups[name][re.sub(r"\s+", "", s[m.start():j])].append(p.name)
print(len(helpers), sum(n >= 2 for n in imports.values()), imports.most_common(8))
for name, g in groups.items():
    repeated = [paths for paths in g.values() if len(set(paths)) > 1]
    print(name, sum(map(len, g.values())), len(g), len(repeated), sum(map(len, repeated)))
PY
```

For the scope totals I applied `Path.rglob("*.ts")`, `len(read_text().splitlines())`, and `Path.rglob("*.spec.ts")` to each of the 19 named source/test directory pairs. For the baseline I summed the integer values in `scripts/contributor-contract-baseline.json` under `single-invocation-data-loop`, filtering by those same directory prefixes.

</details>

## Keep

- **Keep the distinction between conditional and coupled MGWR uncertainty.** The public type says which uncertainty each output includes, and the implementation documents conditioning on bandwidth history. Preserve this when normalizing inference ports. `R/gpu-multiscale-geographically-weighted-regression.ts:87`, `:113`, `:126`.
- **Copy the centered-prediction design from OLS and probit.** Both explain why raw coefficients remain useful for reporting while prediction needs a more accurate internal representation. `R/gpu-ordinary-least-squares.ts:148`; `R/gpu-probit-regression.ts:137`.
- **Keep external-reference provenance beside the assertion.** The GTWR AICc spec explains exactly why its expected result differs from the upstream package. That is more valuable than a looser tolerance. `T/spatial-regression/gtwr-mgtwr.spec.ts:141`.

## Future

1. **Extend the existing model-result contract with fit layout and capabilities.** Six model kinds already participate. Add coefficient axes, active dimensions, status scope, inference method and conditioning, and optional prediction capability. Pilot on GGWR, MGWR, and the two regime layouts. This is worth doing as additive adapters. A shared estimator base class would couple unrelated solvers without solving the readback problem. `modules/analysis/src/analysis-core/contracts/model-result.ts:7`, `:19`.
2. **Make centered prediction an internal regression primitive.** Let a fit provide training centers, correction, and slopes to a prediction node with an explicit query view. Start with held-out OLS and regime prediction. The cost is a little retained fit state and a few bindings. It also gives future prediction work a numerically sound starting point. `R/gpu-ordinary-least-squares.ts:148`; `S/spatial-validation/gpu-spatial-cross-validated-ols.ts:190`.
3. **Describe capacities per execution route.** Have the constructor resolve traversal, supported kernel, row and workspace bounds, and iteration policy once. Use that description for validation and catalog text. Pilot with GTWR and panel lag, where meaningful routes already exist. Device-specific binding limits still need runtime checks. `R/gpu-geographically-temporally-weighted-regression.ts:326`; `R/gpu-spatial-panel-lag-maximum-likelihood.ts:223`.
4. **Build a small regression fixture set around invariants that survive implementation changes.** Include weights rescaling, predictor translation, exact rank arithmetic, padded readback, and CPU/GPU agreement on invalid graph inputs. These cases complement the external statistical fixtures and are cheap enough to attach to every new route. Reserve large-device scenes for actual resource and dispatch claims. `T/spatial-regression/spatial-exact-log-determinant.node.spec.ts:81`; `T/spatial-regression/panel-lag-tiled.spec.ts:63`.

## Top 5

1. **SSR-01** — Reject nonreversible spectra independently of weight scale.
2. **SSR-02** — Compute conformal order-statistic ranks without relative slack.
3. **SSR-03** — Carry the centered fit through held-out and regime prediction.
4. **SSR-04** — Repair upper-tail evaluation and guarantee termination.
5. **SSR-06** — Give MGWR a consistent iteration-budget and last-fit contract across backends.

## Questions for the lead

- Does an explicitly supplied CPU iteration budget carry the same semantics as on GPU? MGWR is a concrete decision point because its current divergence is intentional and documented.
- Can the existing model-result contract own coefficient axes, active lengths, and status scope? That would resolve several consumer traps without changing packed layouts.
- Should the public exact log-determinant helper cover the full nonsingular real domain, as its notation suggests, or expose the fitter’s narrower domain explicitly?