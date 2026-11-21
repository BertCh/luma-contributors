// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** How a quantile that falls between two order statistics is resolved (numpy `method` names). */
export type GPUColumnQuantileInterpolation = 'lower' | 'higher' | 'nearest' | 'linear' | 'midpoint';

/** Parameter-buffer code of each {@link GPUColumnQuantileInterpolation}. */
export const GPU_COLUMN_QUANTILE_INTERPOLATION_CODES: Record<
  GPUColumnQuantileInterpolation,
  number
> = {
  lower: 0,
  higher: 1,
  nearest: 2,
  linear: 3,
  midpoint: 4
};

/** Number of float32 header elements before the probabilities. */
export const GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH = 4;

/** Largest compile-time quantile count. */
export const GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT = 64;

/** Number of float32 elements of a parameter view for `quantileCount` probabilities. */
export function getGPUColumnQuantilesParameterLength(quantileCount: number): number {
  return GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH + quantileCount;
}

/** Inputs of {@link getGPUColumnQuantilesParameterValues}. */
export type GPUColumnQuantilesSettings = {
  /** Probabilities in `[0, 1]`; others (and NaN) produce a NaN quantile. */
  quantiles: ArrayLike<number>;
  /** Defaults to `'linear'`. */
  interpolation?: GPUColumnQuantileInterpolation;
  /** Percentile filter as fractions `[lower, upper]` in `[0, 1]`. Defaults to `[0, 1]`. */
  filterRange?: [number, number];
  /** Number of probability slots. Defaults to `quantiles.length`. Unused slots become NaN. */
  quantileCount?: number;
};

/**
 * Packs per-frame parameters of `GPUColumnQuantiles`.
 *
 * Layout (float32): `[interpolationCode, filterLower, filterUpper, 0, p_0, ..., p_{quantileCount-1}]`.
 * Probability slots beyond `quantiles.length` are NaN, so their outputs are NaN.
 *
 * @param settings Probabilities, interpolation, filter range, and slot count.
 * @param target Optional destination of at least {@link getGPUColumnQuantilesParameterLength} elements.
 */
export function getGPUColumnQuantilesParameterValues(
  settings: GPUColumnQuantilesSettings,
  target?: Float32Array
): Float32Array {
  validateCapacity(
    'GPUColumnQuantiles',
    'quantileCount',
    settings.quantileCount ?? settings.quantiles.length,
    1,
    GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
  );
  return getColumnQuantilesParameterValues(settings, target);
}

/**
 * Packs the same float32 layout as getGPUColumnQuantilesParameterValues for either backend,
 * without the GPU's 64-slot limit. quantileCount must be a positive uint32 count; the target
 * must hold 4 + quantileCount words. Out-of-domain binary64 probabilities become NaN before
 * narrowing, and unused slots are NaN. Interpolation defaults to linear.
 */
export function getColumnQuantilesParameterValues(
  settings: GPUColumnQuantilesSettings,
  target?: Float32Array
): Float32Array {
  const quantileCount = settings.quantileCount ?? settings.quantiles.length;
  validateCapacity('GPUColumnQuantiles', 'quantileCount', quantileCount, 1);
  if (settings.quantiles.length > quantileCount) {
    throw new Error('GPUColumnQuantiles probabilities exceed quantileCount');
  }
  const length = getGPUColumnQuantilesParameterLength(quantileCount);
  const values = target ?? new Float32Array(length);
  if (values.length < length) {
    throw new Error(`GPUColumnQuantiles parameter target must hold ${length} elements`);
  }
  const interpolation = settings.interpolation ?? 'linear';
  const code = GPU_COLUMN_QUANTILE_INTERPOLATION_CODES[interpolation];
  if (!Object.hasOwn(GPU_COLUMN_QUANTILE_INTERPOLATION_CODES, interpolation)) {
    throw new Error(
      `GPUColumnQuantiles unknown interpolation ${String(interpolation)}; interpolation must be lower, higher, nearest, linear or midpoint`
    );
  }
  // Capture controls before writing: quantiles may be a view into the reusable target.
  const probabilities = Array.from(settings.quantiles);
  const [filterLower, filterUpper] = settings.filterRange ?? [0, 1];
  values[0] = code;
  values[1] = filterLower;
  values[2] = filterUpper;
  values[3] = 0;
  for (let index = 0; index < quantileCount; index++) {
    const probability = index < probabilities.length ? probabilities[index] : NaN;
    values[GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH + index] =
      probability >= 0 && probability <= 1 ? probability : NaN;
  }
  return values;
}

/** Largest numerator or denominator of a rational probability: the terms are `uint32` words. */
export const GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM = 0xffffffff;

/** A probability as the exact ratio `numerator / denominator` of two integers. */
export type GPUColumnQuantileRationalProbability = readonly [
  numerator: number,
  denominator: number
];

/** Number of uint32 words of a `rationalProbabilities` view for `quantileCount` probabilities. */
export function getGPUColumnQuantilesRationalProbabilityLength(quantileCount: number): number {
  return 2 * quantileCount;
}

/**
 * Packs the `rationalProbabilities` view of `GPUColumnQuantiles`: the probabilities as exact
 * ratios, for the integer rank that stays exact at any row count.
 *
 * Layout (uint32): `[numerator_0, denominator_0, ..., numerator_{quantileCount-1},
 * denominator_{quantileCount-1}]`. Slots beyond `probabilities.length` are `[0, 0]`, which is not
 * a probability, so their outputs are NaN. A ratio above 1 (`numerator > denominator`) is packed
 * as given and also produces NaN, like a float probability outside `[0, 1]`. The ratio need not
 * be reduced: `[95, 100]` and `[19, 20]` give the same quantile.
 *
 * @param probabilities One `[numerator, denominator]` per quantile; both integers in
 * `[0, 2^32 - 1]`, the denominator at least 1.
 * @param quantileCount Number of probability slots. Defaults to `probabilities.length`.
 * @param target Optional destination of at least
 * {@link getGPUColumnQuantilesRationalProbabilityLength} elements.
 */
export function getGPUColumnQuantilesRationalProbabilityValues(
  probabilities: readonly GPUColumnQuantileRationalProbability[],
  quantileCount: number = probabilities.length,
  target?: Uint32Array
): Uint32Array {
  validateCapacity(
    'GPUColumnQuantiles',
    'quantileCount',
    quantileCount,
    1,
    GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
  );
  return getColumnQuantilesRationalProbabilityValues(probabilities, quantileCount, target);
}

/**
 * Packs the same uint32 pairs as getGPUColumnQuantilesRationalProbabilityValues for either
 * backend, without the GPU's 64-slot limit. quantileCount must be a positive uint32 count;
 * target must hold 2 * quantileCount words. Validates all pairs before any target write.
 * Unused slots are [0, 0] and produce NaN; ratios above 1 also produce NaN.
 */
export function getColumnQuantilesRationalProbabilityValues(
  probabilities: readonly GPUColumnQuantileRationalProbability[],
  quantileCount: number = probabilities.length,
  target?: Uint32Array
): Uint32Array {
  validateCapacity('GPUColumnQuantiles', 'quantileCount', quantileCount, 1);
  if (probabilities.length > quantileCount) {
    throw new Error('GPUColumnQuantiles rational probabilities exceed quantileCount');
  }
  const length = getGPUColumnQuantilesRationalProbabilityLength(quantileCount);
  const values = target ?? new Uint32Array(length);
  if (values.length < length) {
    throw new Error(`GPUColumnQuantiles rational probability target must hold ${length} elements`);
  }
  for (let index = 0; index < quantileCount; index++) {
    const [numerator, denominator] = index < probabilities.length ? probabilities[index] : [0, 0];
    if (index < probabilities.length) {
      validateCapacity(
        'GPUColumnQuantiles',
        `rational probability ${index} numerator`,
        numerator,
        0,
        GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM
      );
      validateCapacity(
        'GPUColumnQuantiles',
        `rational probability ${index} denominator`,
        denominator,
        1,
        GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM
      );
    }
  }
  for (let index = 0; index < quantileCount; index++) {
    const [numerator, denominator] = index < probabilities.length ? probabilities[index] : [0, 0];
    values[2 * index] = numerator;
    values[2 * index + 1] = denominator;
  }
  return values;
}
