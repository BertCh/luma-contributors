# Instrumental-variable robust covariance

See also the [spatial regression API](operators-regression.md#gpuspatialtwostageleastsquares).

`GPUInstrumentalVariablesCovariance` and `GPUSpatialTwoStageLeastSquares.robust` support
`HC0`, `HC1`, `HC2`, `HC3`, `HAC` and one-way `CLUSTER`. The graph operation and
`CPUSpatialTwoStageLeastSquaresOperation` accept the same method and cluster options.
No new estimator or public contributor class is introduced.

## Formula and corrections

Let `Z` be the structural design, `H` the instruments, both including an intercept,
`P = H (H'H)^-1 H'`, `B = (Z'PZ)^-1`, and `A = B Z'H (H'H)^-1`.
The structural residual is `u = y - Z delta`; influence score `s_i = A H_i u_i`.
This is the IV sandwich used by [PySAL spreg's robust_vm](https://pysal.org/spreg/_modules/spreg/robust.html),
rather than the OLS sandwich on endogenous `Z`.

- HC0: `sum_i s_i s_i'`.
- HC1: HC0 times `n / (n-p)`, where `p` includes the intercept, consistent with
  spreg's `sig2n_k=True` White correction. HAC has no finite-sample correction.
- HC2: replace `u_i` by `u_i / sqrt(1-h_i)`.
- HC3: replace `u_i` by `u_i / (1-h_i)`.
- CLUSTER: `G/(G-1) * (n-1)/(n-p) * sum_g S_g S_g'`, `S_g = sum_{i in g} s_i`.
  This is one-way CR1. `G` counts occupied groups, not the supplied key capacity.
  No extra HC1 factor is applied. With singleton groups it equals HC1.

HC2/HC3 are extensions of spreg's sandwich, with the explicitly selected **projected-design**
leverage `h_i = (PZ)_i B (PZ)_i'`, including the intercept. This orthogonal second-stage hat
matrix is distinct from the oblique structural IV hat matrix. No leverage convention is inferred
from the endogenous raw design. `h_i < -1e-6`, nonfinite leverage, or `1-h_i <= 1e-6` fails
inference, including when `u_i = 0`; values are never clipped.

## Cluster input and limits

`clusters` is one packed uint32 key per estimation row. `clusterCount` is a compile-time dense
key capacity in `[1,n]`, with active keys in `[0,clusterCount)`. Unused slots and unsorted keys
are supported. At least two groups must be occupied. `0xffffffff` and all other out-of-range
keys fail inference; no observations are silently deleted. Stable sort and shared segmented
reductions sum scores by group. Empty slots contribute zero outer products and never increase `G`.
There is no CPU scan over GPU inputs and no all-pairs cluster kernel.

The GPU factors at most 15 structural columns excluding the intercept and 16 instrument columns
excluding the intercept. Rows are at most `2^24`, further limited by device storage bindings.
All shape/method checks run at construction; device binding checks run before graph nodes are made.
The contributor exposes and validates `estimatedCommandNodeCount`, including a conservative
allowance for published sort/gather internals. CPU twins retain float64 QR and drop GPU matrix caps.

## PSD certificate and status

Every method symmetrizes its computed covariance `V`. For nonzero finite `V`, it factors
`V / max(abs(V)) + 1e-5 I` using the shared Cholesky solver. A successful factorization certifies
PSD **within this float32 relative tolerance**; it does not assert exact arithmetic PSD.
Zero and rank-deficient matrices are accepted. The published covariance is not shifted or
projected onto the PSD cone. A failed certificate returns NaNs, including robust table inference,
while the point estimates, structural residuals and classical diagnostics remain available.
This certifies the resulting small covariance, not every eigenvalue of a potentially huge HAC kernel.
HAC callers must still supply a symmetric unit-diagonal nonnegative PSD kernel.

Use `output.status: {code: statusView}` for the named solver port. The legacy scalar
`output.status: statusView` remains supported. `GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS`
exports these names:

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | SUCCESS | Covariance passed the numerical certificate |
| 1 | SINGULAR_PROJECTION | Projection system could not be solved |
| 2 | INSUFFICIENT_ROWS | `n <= p` |
| 3 | NONFINITE_INPUT | Input or computed covariance is nonfinite |
| 4 | INVALID_KERNEL | Invalid HAC structure, symmetry, diagonal or overflow |
| 5 | INVALID_LEVERAGE | HC2/HC3 denominator is unresolved or invalid |
| 6 | INVALID_CLUSTERS | Invalid key or fewer than two occupied groups |
| 7 | NOT_POSITIVE_SEMIDEFINITE | Covariance failed the shifted Cholesky certificate |

The existing `robust.output.status` graph-value name maps to this covariance code. The point-fit
solver code remains separate. GPU projection failures take precedence over leverage/cluster
failures; invalid HAC kernels take precedence over projection failures, preserving existing behavior.
The CPU twin uses the same numeric tolerances, with float64 arithmetic before its single output
rounding. It can differ near the thresholds and retains its documented QR/rank differences.

## Independent verification

`spatial-two-stage-least-squares-operation-fixtures.py` constructs a NumPy SVD projector,
dense sandwich matrices and occupied-group CR1 correction without calling first-party solvers.
It covers overidentification, second-order instruments, gapped keys and indefinite HAC covariance.
Node specs compare CPU matrices, tables and diagnostics against those fixtures. Headless specs
compare the direct contributor against NumPy and a hand-derived rank-one sandwich, and compare
GPU/CPU operations on every output, including covariance failures. Zero covariance, unit leverage,
invalid/missing keys, empty cluster slots and residual scales from `1e-12` to `1e12` are covered.
