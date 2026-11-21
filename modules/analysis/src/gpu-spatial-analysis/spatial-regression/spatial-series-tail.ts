// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Tail extrapolation for the truncated trace series of the spatial maximum-likelihood estimators.
 *
 * The series `ln|I - rho W| = -sum_q t^q c_q / q` (`t = rho s`, `c_q = tr((W / s)^q)`) is truncated
 * after `Q` terms. For an irreducible aperiodic row-standardized component the traces `c_q`
 * converge to 1; periodic components can keep them oscillating. When the dominant modes have
 * stabilized, the omitted tail can be far smaller
 * than the worst case `n |t|^(Q+1) / ((Q + 1) (1 - |t|))` that bounds the plain truncation. The extrapolation
 * models the omitted traces as a geometric sequence `c_(Q+k) = level * ratio^k`, `k >= 1`, with
 * the level and ratio estimated from the last three stored traces, plus a second component
 * `alternatingLevel * alternatingRatio^k` with a negative ratio for the alternation that an
 * eigenvalue at -1 (a bipartite weights graph, such as a rook lattice) adds. The closed-form tail
 * (or, for the log-determinant itself, a bounded term loop) of both components is added.
 *
 * It is an approximation, not an exact method: it is accurate when the dominant eigenvalue
 * structure has converged by `Q`, and it keeps the log-likelihood usable up to the stationarity
 * bound (`0.99 / ||W||_inf`) where the plain series must stop at about `0.93` for `Q = 128`.
 * For strongly autocorrelated problems on small `n`, an eigenvalue-based log-determinant remains
 * the exact reference.
 */

/** Most extra terms summed explicitly for the log-determinant tail. */
export const SPATIAL_SERIES_TAIL_TERM_LIMIT = 4096;

/** Smallest `maximumPower` for which the tail model can be estimated (three stored traces). */
export const SPATIAL_SERIES_TAIL_MINIMUM_POWER = 3;

/** Largest geometric ratio in the tail, which keeps `|t * ratio| < 1`. */
const MAXIMUM_TAIL_MAGNITUDE = 0.9999;

/**
 * Geometric tail model `c_(Q+k) = level * ratio^k + alternatingLevel * alternatingRatio^k`, with
 * `ratio` in `[0, 1]` and `alternatingRatio` in `[-1, 0]`.
 */
export type SpatialSeriesTailModel = {
  level: number;
  ratio: number;
  alternatingLevel: number;
  alternatingRatio: number;
};

/**
 * Estimates the tail model from `powerTraces` (`c_q` at index `q - 1`, at least three values).
 * The smooth part is half the last pair sum, with the clamped quotient of the last two pair sums as
 * ratio; the alternating part is half the last difference, with the negated quotient of the last
 * two differences as magnitude when they alternate in sign (else it is dropped).
 */
export function getSpatialSeriesTailModel(powerTraces: ArrayLike<number>): SpatialSeriesTailModel {
  const count = powerTraces.length;
  const last = powerTraces[count - 1];
  const before = powerTraces[count - 2];
  const first = powerTraces[count - 3];
  const pairNow = last + before;
  const pairBefore = before + first;
  let ratio = 0;
  if (pairBefore > 1e-30 && pairNow > 0) {
    ratio = Math.min(pairNow / pairBefore, 1);
  }
  const differenceNow = last - before;
  const differenceBefore = before - first;
  let alternatingRatio = 0;
  if (Math.abs(differenceBefore) > 1e-30 && differenceNow * differenceBefore < 0) {
    alternatingRatio = Math.max(differenceNow / differenceBefore, -1);
  }
  return {
    level: 0.5 * pairNow,
    ratio,
    alternatingLevel: alternatingRatio === 0 ? 0 : 0.5 * differenceNow,
    alternatingRatio
  };
}

function getTailPower(t: number, exponent: number): number {
  return t === 0 ? 0 : t ** exponent;
}

function clampTailRatio(value: number): number {
  return Math.max(-MAXIMUM_TAIL_MAGNITUDE, Math.min(MAXIMUM_TAIL_MAGNITUDE, value));
}

/** Omitted `sum_{q>Q} t^(q-1) c_q` of the log-determinant derivative series (without the factor `-s`). */
export function getSpatialSeriesDerivativeTail(t: number, powerTraces: ArrayLike<number>): number {
  const model = getSpatialSeriesTailModel(powerTraces);
  const scale = getTailPower(t, powerTraces.length - 1);
  const component = (level: number, ratio: number) => {
    const u = clampTailRatio(t * ratio);
    return (level * scale * u) / (1 - u);
  };
  return (
    component(model.level, model.ratio) + component(model.alternatingLevel, model.alternatingRatio)
  );
}

/** Omitted `sum_{m>=Q-1} (m + 1) t^m c_(m+2)` of the second trace series (without the factor `s^2`). */
export function getSpatialSeriesSecondTail(t: number, powerTraces: ArrayLike<number>): number {
  const maximumPower = powerTraces.length;
  const model = getSpatialSeriesTailModel(powerTraces);
  const scale = getTailPower(t, maximumPower - 2);
  const component = (level: number, ratio: number) => {
    const u = clampTailRatio(t * ratio);
    const geometric = u / (1 - u);
    return level * scale * ((maximumPower - 1) * geometric + geometric / (1 - u));
  };
  return (
    component(model.level, model.ratio) + component(model.alternatingLevel, model.alternatingRatio)
  );
}

/** Omitted `-sum_{q>Q} t^q c_q / q` of the log-determinant series. */
export function getSpatialSeriesLogDeterminantTail(
  t: number,
  powerTraces: ArrayLike<number>
): number {
  const maximumPower = powerTraces.length;
  const model = getSpatialSeriesTailModel(powerTraces);
  const scale = getTailPower(t, maximumPower);
  const component = (level: number, ratio: number) => {
    const u = clampTailRatio(t * ratio);
    let power = u;
    let sum = 0;
    let step = 1;
    for (; step <= SPATIAL_SERIES_TAIL_TERM_LIMIT; step++) {
      sum += power / (maximumPower + step);
      power *= u;
      if (Math.abs(power) < 1e-9) {
        step++;
        break;
      }
    }
    sum += power / ((1 - u) * (maximumPower + step));
    return level * scale * sum;
  };
  return -(
    component(model.level, model.ratio) + component(model.alternatingLevel, model.alternatingRatio)
  );
}

/**
 * WGSL helpers for the tail of the trace series, to be appended after the Horner functions of
 * `getTraceSeriesWGSL` / `getSpatialLogDeterminantWGSL`. They define `getSeriesTailDerivative(t)`,
 * `getSeriesTailSecond(t)` and `getSeriesTailLogDeterminant(t)`, the float32 forms of the CPU
 * functions above, reading `c_q` from the f32 binding `bindingName`.
 */
export function getSpatialSeriesTailWGSL(maximumPower: number, bindingName: string): string {
  // Internal helper: callers validate maximumPower and fixed binding names, a violation is a library bug.
  if (maximumPower < SPATIAL_SERIES_TAIL_MINIMUM_POWER) {
    throw new Error('tail extrapolation needs maximumPower >= 3'); // library invariant
  }
  const trace = (q: number) => `${bindingName}[${bindingName}Offset + ${q - 1}u]`;
  return /* wgsl */ `
fn getSeriesTailPower(t: f32, exponent: u32) -> f32 {
  if (t == 0.0) {
    return 0.0;
  }
  let magnitude = pow(abs(t), f32(exponent));
  return select(magnitude, -magnitude, t < 0.0 && (exponent & 1u) == 1u);
}

// (level, ratio, alternatingLevel, alternatingRatio) of the tail c_(Q+k) = level * ratio^k + alternatingLevel * alternatingRatio^k.
fn getSeriesTailModel() -> vec4f {
  let last = ${trace(maximumPower)};
  let before = ${trace(maximumPower - 1)};
  let first = ${trace(maximumPower - 2)};
  let pairNow = last + before;
  let pairBefore = before + first;
  var ratio = 0.0;
  if (pairBefore > 1.0e-30 && pairNow > 0.0) {
    ratio = min(pairNow / pairBefore, 1.0);
  }
  let differenceNow = last - before;
  let differenceBefore = before - first;
  var alternatingRatio = 0.0;
  if (abs(differenceBefore) > 1.0e-30 && differenceNow * differenceBefore < 0.0) {
    alternatingRatio = max(differenceNow / differenceBefore, -1.0);
  }
  return vec4f(0.5 * pairNow, ratio, select(0.5 * differenceNow, 0.0, alternatingRatio == 0.0), alternatingRatio);
}

fn getSeriesTailRatio(t: f32, ratio: f32) -> f32 {
  return clamp(t * ratio, -0.9999, 0.9999);
}

fn getSeriesTailDerivativeComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  return level * scale * u / (1.0 - u);
}

fn getSeriesTailDerivative(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${maximumPower - 1}u);
  return getSeriesTailDerivativeComponent(scale, t, model.x, model.y)
    + getSeriesTailDerivativeComponent(scale, t, model.z, model.w);
}

fn getSeriesTailSecondComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  let geometric = u / (1.0 - u);
  return level * scale * (${maximumPower - 1}.0 * geometric + geometric / (1.0 - u));
}

fn getSeriesTailSecond(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${maximumPower - 2}u);
  return getSeriesTailSecondComponent(scale, t, model.x, model.y)
    + getSeriesTailSecondComponent(scale, t, model.z, model.w);
}

fn getSeriesTailLogDeterminantComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  var power = u;
  var sum = 0.0;
  var step = 1u;
  for (; step <= ${SPATIAL_SERIES_TAIL_TERM_LIMIT}u; step++) {
    sum += power / f32(${maximumPower}u + step);
    power *= u;
    if (abs(power) < 1.0e-9) {
      step++;
      break;
    }
  }
  sum += power / ((1.0 - u) * f32(${maximumPower}u + step));
  return level * scale * sum;
}

fn getSeriesTailLogDeterminant(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${maximumPower}u);
  return -(getSeriesTailLogDeterminantComponent(scale, t, model.x, model.y)
    + getSeriesTailLogDeterminantComponent(scale, t, model.z, model.w));
}
`;
}
