# Spatial panel unit fixed effects

`GPUSpatialPanelRegression` accepts `effects: 'pooled' | 'unit'`; the default
remains pooled spatial-lag 2SLS. Unit effects estimate
`y_it = alpha_i + X_it beta + rho (W y_t)_i + u_it`.
Supply a balanced, complete period-major panel: row `t * entityCount + entity`,
then predictor column. The same entity weights apply to every period. There is no
mask or missing-row compaction. Nonfinite observations are failures, never omitted.

Unit effects subtract each entity's temporal mean from every predictor and response
on GPU. This projection commutes with the period-block spatial weights, including
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
disjoint; outputs are caller owned. Period count remains bounded at 2–1024.

Time effects, two-way effects, random effects, dynamic panels and clustered
inference are unsupported. In particular, subtracting time means from X and y
before applying general W is not a correct two-way IV transformation.

The independent fixture generator
`modules/graph/test/oracles/spatial-panel-fixed-effects.py` fits a full
entity-dummy instrumental-variable system with NumPy least squares. It records
NumPy version, ordering and covariance convention, and is a reference composition
rather than a direct panel-library call. GPU specs check coefficients, standard
errors, residuals, absorbed intercepts and singular time-invariant predictors.
