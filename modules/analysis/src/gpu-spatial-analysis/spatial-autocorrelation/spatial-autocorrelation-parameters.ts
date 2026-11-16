// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineGPUSpatialParameterSchema} from '../../analysis-core/contracts/index';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a spatial-autocorrelation parameter buffer. */
export const GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH = 8;

/** Extended fixed population counts use two exact 16-bit words, bounded by uint32. */
export const GPU_SPATIAL_AUTOCORRELATION_MAXIMUM_FIXED_COUNT = 0xffffffff;

/** Float32 represents every integer through this count; legacy slot-2 updates remain exact. */
export const GPU_SPATIAL_AUTOCORRELATION_MAXIMUM_LEGACY_FIXED_COUNT = 2 ** 24;

/** Declarative layout shared by local Moran and hot-spot analysis. */
export const GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA = defineGPUSpatialParameterSchema({
  id: 'spatial-autocorrelation',
  format: 'float32',
  wordLength: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  fields: [
    {
      name: 'significanceLevel',
      format: 'float32',
      wordOffset: 0,
      defaultValue: 0.05,
      minimum: Number.MIN_VALUE,
      maximum: 1,
      dynamic: true
    },
    {
      name: 'useFixedMoments',
      format: 'float32',
      wordOffset: 1,
      defaultValue: 0,
      minimum: 0,
      maximum: 1,
      dynamic: true
    },
    {
      name: 'fixedCount',
      format: 'float32',
      wordOffset: 2,
      defaultValue: 0,
      minimum: 0,
      dynamic: true
    },
    {name: 'fixedMean', format: 'float32', wordOffset: 3, defaultValue: 0, dynamic: true},
    {
      name: 'fixedVariance',
      format: 'float32',
      wordOffset: 4,
      defaultValue: 0,
      minimum: 0,
      dynamic: true
    },
    {
      name: 'fixedCountHigh',
      format: 'float32',
      wordOffset: 5,
      defaultValue: 0,
      minimum: 0,
      maximum: 65535,
      dynamic: true
    },
    {
      name: 'fixedCountLow',
      format: 'float32',
      wordOffset: 6,
      defaultValue: 0,
      minimum: 0,
      maximum: 65535,
      dynamic: true
    },
    {
      name: 'useExtendedFixedCount',
      format: 'float32',
      wordOffset: 7,
      defaultValue: 0,
      minimum: 0,
      maximum: 1,
      dynamic: true
    }
  ]
});

/**
 * Number of float32 values written to an optional `globalStatistics` output:
 * `[n, mean, variance, standardDeviation, islandCount]`.
 */
export const GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH = 5;

/**
 * Two-sided standard normal critical values of the Getis-Ord Gi* confidence bins, in bin order
 * `1, 2, 3`: 90%, 95% and 99% confidence (`p <= 0.10, 0.05, 0.01`).
 */
export const GPU_HOT_SPOT_CRITICAL_Z_SCORES = [
  1.6448536269514722, 1.959963984540054, 2.5758293035489004
] as const;

/** Significance levels of the Getis-Ord Gi* confidence bins `1, 2, 3`. */
export const GPU_HOT_SPOT_SIGNIFICANCE_LEVELS = [0.1, 0.05, 0.01] as const;

/** Local Moran quadrant codes written by `GPULocalMoran` (esda numbering). */
export const GPU_LOCAL_MORAN_QUADRANT = {
  /** Not significant, invalid, or undefined. */
  NOT_SIGNIFICANT: 0,
  /** High value surrounded by high values (hot spot). */
  HIGH_HIGH: 1,
  /** Low value surrounded by high values (outlier). */
  LOW_HIGH: 2,
  /** Low value surrounded by low values (cold spot). */
  LOW_LOW: 3,
  /** High value surrounded by low values (outlier). */
  HIGH_LOW: 4
} as const;

/**
 * Global moments that replace the moments computed from the current rows.
 *
 * Use them to pin the reference distribution, for example to moments measured once over the
 * full dataset, so that z-scores do not change when a viewport mask changes the row set.
 */
export type GPUSpatialAutocorrelationFixedMoments = {
  /** Population size `n`, at least 2. */
  count: number;
  /** Population mean. */
  mean: number;
  /** Population variance (denominator `n`), positive. */
  variance: number;
};

/**
 * CPU settings packed into the per-frame `parameters` of `GPUHotSpotAnalysis`, `GPULocalMoran` and
 * `GPULocalGeary` by {@link getGPUSpatialAutocorrelationParameterValues}, and described by their
 * `getInferenceMetadata(settings)`.
 *
 * Neighborhoods and weight values come from the `weights` prop, not from the parameters. Every
 * field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPUSpatialAutocorrelationSettings = {
  /**
   * Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the
   * adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis`
   * ignores it and always reports 90/95/99% bins.
   */
  significanceLevel?: number;
  /** Optional moments that replace the moments of the current rows. */
  fixedMoments?: GPUSpatialAutocorrelationFixedMoments;
};

/**
 * Packs spatial-autocorrelation settings into the 8-element float32 layout read by
 * `GPUHotSpotAnalysis`, `GPULocalMoran` and `GPULocalGeary`.
 *
 * Layout: `[significanceLevel, useFixedMoments, fixedCount, fixedMean, fixedVariance, fixedCountHigh, fixedCountLow, useExtendedFixedCount]`.
 * Counts through 2^24 retain the legacy layout: words 5–7 are zero and updates to word 2 take effect.
 * Larger counts explicitly select the exact words 5–6 with word 7 equal to one. To return to legacy
 * word-2 updates, set word 7 to zero; the packer does this automatically for smaller counts.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If `significanceLevel` is not finite or outside `(0, 1)`, fixed moments are invalid, or
 * `target` is too short.
 */
export function getGPUSpatialAutocorrelationParameterValues(
  settings: GPUSpatialAutocorrelationSettings = {},
  target: Float32Array = new Float32Array(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH)
): Float32Array {
  validateCapacity(
    'getGPUSpatialAutocorrelationParameterValues',
    'target length',
    target.length,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const significanceLevel = settings.significanceLevel ?? 0.05;
  if (!(Math.fround(significanceLevel) > 0 && Math.fround(significanceLevel) < 1)) {
    throw new Error('Spatial autocorrelation significanceLevel must be in (0, 1)');
  }
  const fixedMoments = settings.fixedMoments;
  if (fixedMoments) {
    validateCapacity(
      'spatial-autocorrelation',
      'fixedMoments.count',
      fixedMoments.count,
      2,
      GPU_SPATIAL_AUTOCORRELATION_MAXIMUM_FIXED_COUNT
    );
    if (
      !Number.isFinite(fixedMoments.count) ||
      !Number.isFinite(Math.fround(fixedMoments.mean)) ||
      !Number.isFinite(Math.fround(fixedMoments.variance)) ||
      fixedMoments.count < 2 ||
      Math.fround(fixedMoments.variance) <= 0
    ) {
      throw new Error('Spatial autocorrelation fixedMoments are invalid');
    }
  }
  const useExtendedFixedCount = Boolean(
    fixedMoments && fixedMoments.count > GPU_SPATIAL_AUTOCORRELATION_MAXIMUM_LEGACY_FIXED_COUNT
  );
  target.set([
    significanceLevel,
    fixedMoments ? 1 : 0,
    fixedMoments?.count ?? 0,
    fixedMoments?.mean ?? 0,
    fixedMoments?.variance ?? 0,
    useExtendedFixedCount ? Math.floor(fixedMoments!.count / 65536) : 0,
    useExtendedFixedCount ? fixedMoments!.count % 65536 : 0,
    useExtendedFixedCount ? 1 : 0
  ]);
  return target;
}
