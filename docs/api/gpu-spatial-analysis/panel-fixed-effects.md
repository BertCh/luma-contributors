# Spatial panel unit fixed effects

`GPUSpatialPanelRegression` accepts `effects: 'pooled' | 'unit'`; the default
remains pooled spatial-lag 2SLS. Unit effects estimate
`y_it = alpha_i + X_it beta + rho (W y_t)_i + u_it`.
Supply a balanced, complete period-major panel: row `t * entityCount + entity`,
then predictor column. The same entity weights apply to every period. There is no
mask or missing-row compaction. Nonfinite observations are failures, never omitted.
Malformed CSR bounds (a nonzero first offset, a decreasing row offset, an offset beyond the active terminal
offset, or a terminal offset beyond edge capacity) and active neighbor indices outside the entity
axis invalidate the fit. Expansion bounds
memory accesses and preserves that invalidity; it does not fit silently clamped weights.
Estimator settings, weight metadata and output port maps are captured at construction.
Later edits to those JavaScript objects require constructing a new contributor.

Unit effects subtract each entity's temporal mean from every predictor and response
on GPU. The shared split-center transform computes tiled two-part means, then subtracts
the high and low parts separately. This preserves within residual variation when entity
offsets are large relative to that variation. This projection commutes with the period-block spatial weights, including
directed and non-row-standardized weights. The existing spatial 2SLS solver then
uses transformed X and its spatial lags as instruments for transformed Wy. No
observation buffers are read back and there is no CPU estimation fallback.

The output retains the pooled table allocation and order: intercept, supplied
predictors, rho, with four values per row (coefficient, standard error, normal z,
two-sided normal p-value). The absorbed intercept row is NaN. Entity intercepts are
not returned. Time-invariant predictors are absorbed and cause singular status;
the estimator does not silently remove columns. Remove such predictors yourself.
Residuals are the within structural residuals in original period-major row order.

For unit effects the residual degrees of freedom are
`entityCount * (periodCount - 1) - predictorCount - 1`.
The residual variance and homoskedastic coefficient covariance use this divisor.
The first-stage F uses `n - entityCount - instrumentCount` residual degrees.
Standard errors are not clustered or serial-correlation robust. Summary pseudo
R-squared and conditioning describe the within problem. Summary slots 4–6
(pooled Moran/Anselin-Kelejian inference) are NaN: that pooled test does not account
for absorption. Other summary indices retain the spatial 2SLS layout.

Status codes retain the underlying solver contract: 0 success, 1 singular
instruments/projected regressors, 2 nonpositive residual degrees, 3 nonfinite
moments. Nonfinite inputs can also yield singular status when the factorization
fails first; status 1 must not be interpreted as proof of finite inputs.
Failed coefficient tables and residuals are NaN. Inputs and outputs must be
disjoint; outputs are caller owned. For this pooled/unit 2SLS contributor, `periodCount` is 2–1,024 and
`entityCount * periodCount <= 16,777,215` (`2^24 - 1`). Repeated CSR allocations and device
bindings can lower these limits. These are total panel rows, not rows per period.
The separate [panel-error ML](./coverage-statistics.md#regimes-panels-and-sur) route accepts up
to `2^24` total rows and periods subject to its own design/binding limits; the separate
[panel-lag ML](./coverage-statistics.md#gpuspatialpanellagmaximumlikelihood) reference route
supports 2–32 entities and 2–32 periods, while tiled unit effects with an entity spectrum
support up to 1,024 entities and 16,777,216 total rows. These ML bounds do not enlarge this
2SLS contributor's route.

Time effects, two-way effects, random effects, dynamic panels and clustered
inference are unsupported. In particular, subtracting time means from X and y
before applying general W is not a correct two-way IV transformation.

The independent fixture generator
`modules/analysis/test/oracles/spatial-panel-fixed-effects.py` fits a full
entity-dummy instrumental-variable system with NumPy least squares. It records
NumPy version, ordering and covariance convention, and is a reference composition
rather than a direct panel-library call. GPU specs check coefficients, standard
errors, residuals, absorbed intercepts and singular time-invariant predictors.
The additional `panel-regression-audit.oracle.py` beside the specs records all table,
summary and residual values for a 16-row panel, including an island. Its headless
cases check response rescaling, absorbed entity shifts and malformed CSR refusal.

Panel-error ML exact determinants accept either `entityCount` eigenvalues of W or the
expanded `entityCount * periodCount` block spectrum. The short representation repeats
the log determinant and both trace derivatives `periodCount` times. The GPU exact
spectrum limit is 1,024 supplied eigenvalues: it bounds entities on the short route
and total panel rows on the expanded route. These are dimensionless eigenvalues;
series mode does not take an eigenvalue port.

The named GPU and CPU panel-error operations also accept an optional
`weights.status.complete` scalar port. Zero suppresses the fit (code 3) and pooled
diagnostics (code 1); restoring a positive certificate permits recomputation when
other loss checks are clear. The spectrum representation is inferred from its
length in every panel-error form.

The named GPU and CPU panel-lag operations accept an optional graph name in
`weights.status.overflow`. The pure CPU function accepts an optional `Uint32Array` at
the same path. A nonzero first word suppresses the fit with code 1.
