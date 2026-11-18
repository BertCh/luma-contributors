# Sinkhorn divergence between spatial distributions (STM-09)

`GPUSinkhornDivergence` measures how different two mass fields on the same grid are, in the units of squared distance: the debiased entropic optimal transport cost (the Sinkhorn divergence of Genevay et al. 2018 and Feydy et al. 2019). It is zero for identical fields and grows as mass has to move farther, which makes it a verification score for forecast and observed fields that does not suffer the "double penalty" of cell-by-cell error, and a comparison of two density maps.

```ts
import {
  getGPUSinkhornDivergenceParameterValues,
  GPUSinkhornDivergence
} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

The classes are in `src/gpu-spatial-analysis/optimal-transport/`.

## Definition

For masses `a` and `b` on the grid, normalized to sum to one, and the squared Euclidean cost `C` between cell centers,

```
OT(a, b) = min over couplings P of  <P, C> + epsilon KL(P | a x b)
S(a, b)  = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2
```

`S` is zero when `a = b`, never negative, and symmetric. Plain `OT(a, a)` is not zero (the entropic bias), which is why the divergence subtracts it. Both masses are normalized, so a difference in total mass is not seen, and there is no unbalanced (KL-marginal) variant. The POT function `ot.bregman.empirical_sinkhorn_divergence` (0.9.7.post1) is a different quantity: its documentation says it does not include the entropic terms, and it differs from `S` by 0.05 on the `epsilon` 0.5 fixture case.

## `GPUSinkhornDivergence`

```ts
const sinkhorn = new GPUSinkhornDivergence({
  grid,             // GPURasterMetadata: only width and height are used
  source,           // GraphDataView<'float32'>, width * height masses in raster row order
  target,
  parameters,       // GraphDataView<'float32'> from getGPUSinkhornDivergenceParameterValues
  maximumIterations: 100,
  output: {
    divergence,     // 1 row
    components,     // optional, 3 rows: OT(a, b), OT(a, a), OT(b, b)
    marginalError,  // optional, 1 row
    status: {converged, iterationLimitReached, iterationCount, invalidCount, approximation}
  }
});
graph.add(sinkhorn);

getGPUSinkhornDivergenceParameterValues({
  cellWidth: 1,
  cellHeight: 1,
  epsilon: 2,
  tolerance: 1e-4,
  epsilonStart: 16,
  scalingFactor: 0.8
});
```

Per-frame (no recompile): the contents of `parameters` (cell width and height, `epsilon`, `tolerance`, `epsilonStart`, `scalingFactor`) and of both mass fields. Compile-time (a new graph): the grid size, `maximumIterations`, `blurRadius`, and which optional outputs exist. Extra prop: `id`, `spatialContext` (planar/native only).

- `parameters` is `[cellWidth, cellHeight, epsilon, tolerance, epsilonStart, scalingFactor]` (`GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH` is 6). `epsilon` is in squared position units. Iteration `k` uses `max(epsilon, epsilonStart * scalingFactor^k)`; the defaults (`epsilonStart = epsilon`, `scalingFactor = 1`) are no schedule. `tolerance` defaults to `GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE` (1e-4). The packer throws on a value that is not positive and finite, on `epsilonStart < epsilon` and on `scalingFactor` outside `(0, 1]`.
- Masses must be non-negative and finite. A negative or non-finite cell is counted in `status.invalidCount` and treated as zero. If either field has no mass, or the parameters are unusable, `divergence` and the other values are NaN, `iterationCount` is 0 and neither `converged` nor `iterationLimitReached` is set.
- Each field is normalized independently after rescaling by a power of two chosen from its largest finite mass. The multiplier stays normal and finite and every rescaled mass is below four, so even a raw total beyond float32 range is supported. For example, `[3, 2]` and `[2, 3]` give the same distributions when either or both fields are multiplied by `2^126`; every input remains finite. The CPU twin uses float64 totals and supports the same case.
- `components` are reported because the divergence is a difference of them and carries their rounding.
- `marginalError` is the L1 distance between the cross plan's target marginal and the normalized target masses, measured before the last target-side update. `converged` is one only when the cross plan and both symmetric self plans each have L1 marginal error at most `tolerance` before their updates, with the regularization at its final value. The symmetric plans have equal row and column residuals. A small `marginalError` alone does not certify convergence. `iterationLimitReached` is one when `maximumIterations` (1 to `GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS`, 1000; default `GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS`, 100) ran out first, in which case the latest potentials are still used.
- The residual pass must use the exact packed final `epsilon`. A distinct `epsilonStart` with `scalingFactor = 1` never reaches it and exhausts the budget, even if it differs by only one float32 step. The website scene displays `iteration limit reached (estimate)` alongside its distance and cost readout when this flag is set; it clears the indication after a converged run.
- `blurRadius` (compile-time, default the longer side minus one, which is exact) limits each 1D blur window. A smaller radius drops Gibbs kernel terms beyond it. It is exact when the dropped terms are negligible against the nearest mass and sets `status.approximation`; a cell with no mass in its window gets a huge potential, so keep the window large against the gaps between masses.
- A positive-mass cell unreachable through the blur window contributes its entire normalized mass to the residual. Repeated unreachable potentials do not count as convergence. For example, `[1, 0, 0]` to `[0, 0, 1]` on a `3 x 1` grid with `blurRadius = 1` has cross marginal error `1` and reaches the iteration limit for tolerance below `1`.
- Grid sides are limited to `GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE` (2048) because one blur invocation walks a row or a column; cells to 2^24 (float32 counts). `estimatedCommandNodeCount` is `10 + 7 * maximumIterations`, an upper bound including both optional publication passes. The two additional fixed nodes find the per-field maxima before the rescaled mass sums.

{/* lens:GPUSinkhornDivergence */}

{/* parameters: GPUSinkhornDivergence generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSinkhornDivergence` parameters

- Packer: `getGPUSinkhornDivergenceParameterValues(settings)`, writes `GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH` (6) `float32` values.

Settings of `getGPUSinkhornDivergenceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `cellWidth` | `number` | Cell width in position units. Positive and finite. |
| `cellHeight` | `number` | Cell height in position units. Positive and finite. |
| `epsilon` | `number` | Entropic regularization, in squared position units (the cost is the squared distance between cell centers). Positive. In float32 the log-domain exponents reach `maximum cost / epsilon`, and each carries an absolute rounding error of about `6e-8` times that, so keep `epsilon` at least about `1e-3` times the squared grid diagonal. |
| `tolerance` (optional) | `number` | Largest L1 marginal error allowed in each of the cross plan and both symmetric self plans before their updates at the final epsilon. Defaults to `GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE`. |
| `epsilonStart` (optional) | `number` | First regularization of the epsilon schedule, at least `epsilon`. Defaults to `epsilon` (no scaling). |
| `scalingFactor` (optional) | `number` | Factor in `(0, 1]` applied to the regularization each iteration until it reaches `epsilon`: iteration `k` uses `max(epsilon, epsilonStart * scalingFactor^k)`. Defaults to 1. |

{/* /parameters */}

## How it runs

In the log domain the dual iteration is `g_j = -epsilon * LSE_i(f_i / epsilon + log a_i - C_ij / epsilon)` and the same for `f`. The Gibbs kernel `exp(-C / epsilon)` is separable in the two grid axes, so each log-sum-exp over all cells is two 1D passes, one along rows and one along columns: the raster convolution pattern in the log semiring, at `O(cells * (width + height))` per update instead of `O(cells^2)`. The three problems `(a, b)`, `(a, a)` and `(b, b)` run in one batch, the symmetric ones with the averaged update `f <- (f + T f) / 2`. Each first half-iteration measures all three marginal errors before applying the updates, using `mass * |exp((old - T(old)) / epsilon) - 1|`. A tiled reduction sums each plan's residual separately. A one-thread node requires all three to meet tolerance at the final epsilon, applies the schedule and disarms a GPU dispatch gate, so finished iterations launch nothing; zero-mass cells are masked rather than given a log of minus infinity. The divergence is `sum a (f_ab - f_aa) + sum b (g_ab - g_bb)`, with the differences formed before the tiled sum (plain float32 inside tiles of a few hundred cells, then a fixed pairwise merge; not a compensated float sum, which the test GPU's shader compiler removes).

`GPUSinkhornDivergenceOperation` and `CPUSinkhornDivergenceOperation` use the same three-plan stopping rule and explicit iteration budget. The CPU twin evaluates the full kernel in float64; near the tolerance, rounding can change the stopping iteration between the two forms.

## Single precision

The exponents reach `maximum cost / epsilon` and each carries a relative rounding of 6e-8, so keep `epsilon` at least about 1e-3 times the squared grid diagonal (the exponent range at most 1000; the specs run at most 200). The marginal-error test cannot resolve changes below about `6e-8 * maximum cost / epsilon`, so a tolerance below that never converges (at an exponent range of 100 and a tolerance of 1e-5, several random cases did not converge in 800 iterations even for identical inputs; they still agreed to 4e-6 of the cost).

## Validation

- **POT.** POT 0.9.7.post1 (MIT) was unpacked into a scratch directory, not installed; the generator is `test/oracles/stm09-sinkhorn-pot.py` and the fixture `fixtures/stm09-sinkhorn-pot.json` (seven grids from 8 by 8 to 16 by 16, the largest exponent range 196, with zero cells, anisotropic cells, identical inputs and point masses). The cost is evaluated from POT's coupling (`ot.sinkhorn`, `sinkhorn_log`, marginal tolerance 1e-13) as `<P, C> + epsilon KL(P | a x b)`. A converged GPU run agrees with POT to 3e-7 of the cost or better in the divergence and in each component (the spec allows 1e-6). The independent binary64 dense reference in the test directory agrees with POT to 1e-7.
- **Closed form.** Two point masses at squared distance `d2` give `S = d2` for any `epsilon` (one coupling cell, no entropy, `OT(a, a) = 0`); checked on three cell sizes.
- **Zero for identical inputs and non-negative.** On ten random sparse pairs (grids 5 to 7 cells a side, exponent range 30 to 150) with 400 iterations and the default tolerance, `S(a, a)` and the negative part of `S(a, b)` stay within `1.1 * 2^-24 * 150`, about 9.8e-6, of `OT(a, b)`; the plain `OT(a, a)` is above 0.1, so the debiasing is what makes it vanish. Converged runs are far closer: 1e-9 to 4e-7 of the cost. Distinct inputs are positive by at least 1e-3 of the cost, and `S(a, b)` and `S(b, a)` agree within the same bound.
- Also checked: invariance to the scale of each field, invalid cells, unusable inputs, the iteration limit, the epsilon schedule, a short blur window against the exact one (and a window of three cells that is clearly wrong), growth with distance, and bitwise repeatability.
- Not measured: GPU speed at 2048 by 2048.
