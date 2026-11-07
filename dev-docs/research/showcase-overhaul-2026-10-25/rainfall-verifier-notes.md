# Rainfall fit and conditional LOO verifier handoff

## Scope

`check-rainfall-fit-loo.mjs` is ready for the parent-authorized serialized browser/GPU slot. It navigates to the existing `rainfall-interpolation` scene, leaves the source frozen, selects `prcp` and the existing 300 km maximum lag, then captures all 18 combinations of robust estimator, weighting and model from the actual scene readouts. It finishes on the default Matheron / Cressie / spherical fit and checks the conditional LOO diagnostics.

The script reuses `statistics-browser-check.mjs`, including its served-bundle hash evidence, WebGPU error capture, report writing and browser cleanup. It does not build or serve the site. Invoke it only after the showcase is served at `http://127.0.0.1:5313/` (or pass another base URL):

```sh
node dev-docs/research/showcase-overhaul-2026-10-25/check-rainfall-fit-loo.mjs [base-url] [report-directory]
```

The independent reference runs under `/tmp/showcase-statistics-reference-env/bin/python`. On success or failure, the browser helper writes `report.json`; the script also saves its complete numerical request and SciPy output as `reference-input.json` and `reference-output.json` in the report directory. The request includes each exact 16-row bin array and the full 6,135-row LOO table.

## Fit assertions

The check waits until the robust toggle, weighting selector and model selector show the requested values, the fit readout has completed with raw CPU and GPU parameter tuples, and the exact empirical-bin readout has its 16 lag rows. Those rows are parsed from the seven-significant-digit `distance_m, semivariance, pair_count` source readout and passed unchanged as JSON numbers to the reference. The selected model/weighting raw readouts are independently scored against that same input, so a result attached to the wrong input will fail the objective/prediction check.

`reference-rainfall-fit-loo.py` profiles nonnegative nugget and partial sill with SciPy NNLS, a 200-step log-range scan and bounded scalar refinement. Cressie weighting is re-estimated through the three documented passes. It uses these effective-range definitions from `variogram-model.ts`: spherical reaches its sill at `range`; exponential and Gaussian reach 95% at `range`, through `1-exp(-3h/range)` and `1-exp(-3(h/range)^2)` respectively. It does not import or reimplement the production closed-form fitting routine.

Accepted GPU fit statuses are 0 (interior) and 3 only when the readout explicitly identifies a boundary/iteration-budget candidate. Status 3 is retained as a candidate and does not assert range identification. Statuses 1 or 2 fail this verification lane. For every combination, the script independently recomputes objective from both raw backend parameter tuples and checks each against the SciPy profile within 2% (or `1e-5` absolute for a near-zero objective). Weighted model-prediction RMS must be at most the greater of 0.03 semivariance units and 4% of the reference weighted residual RMS (objective divided by total objective weight before taking its square root). Parameter equality is deliberately not required for flat or boundary fits.

## LOO assertions

The script checks that the selected controls return to `prcp`, Matheron, Cressie and spherical, and waits for the completed 6,135-row table. The Python oracle reads the bundled `position.bin` and `prcp.bin`, applies the documented spherical AEQD frame centered on `[-83, 32]`, and builds exact Euclidean neighborhoods with a SciPy KDTree, 120,000 m radius and `k=12`. It verifies all station IDs and observations against the bundled CSV/binary row order, excludes each target row before neighbor selection, requires at least three distinct locations, coalesces duplicate coordinates, and solves ordinary kriging using the spherical model and the actual float32 nugget/sill/range reported from the kriging parameter buffer; independent fitting is checked separately. It checks 24 deterministic positive-variance probes: the first three eligible source rows plus farthest-point spatial coverage.

All-row residual and standardized-residual identities are checked against explicit seven-significant-digit rounding bounds plus float32 arithmetic error. The readout's three-significant-digit bias, MAE and RMSE are compared with independent recomputation from every table row using half a display unit plus `2e-4` computational tolerance; each probe must have positive variance. Probe predictions allow the greater of 0.08 measurement units or 1.5% relative error; variance allows the greater of 0.05 squared units or 4% relative error. The all-row summary is recomputed from readout values, while the 24 spatial probes are independently solved from source binaries.

These are **conditional** LOO diagnostics: the variogram is fitted on the full gauge set before any station is omitted from its local kriging solve. They do not establish leakage-resistant cross-validation or deployment error. The LOO oracle uses binary64 SciPy linear solves and float32-projected station coordinates; tolerance covers GPU float arithmetic and seven-significant-digit readout formatting. Projection identity and difficult near-singular neighborhoods remain the main numerical uncertainty.

## Run record

Prepared only. `node --check` is the sole permitted verification before handoff; do not launch this browser script until the parent releases the serialized GPU slot. No browser, GPU, root build, lint, install or source changes were performed for this lane.

Planner review corrected the verifier before execution: actual kriging parameter floors are now read from the GPU buffer disclosure, objective RMS is normalized by total weight, metric tolerances reflect actual display precision, and the full input request is saved before invoking Python so oracle failures remain reproducible. No runtime results have been accepted yet.
