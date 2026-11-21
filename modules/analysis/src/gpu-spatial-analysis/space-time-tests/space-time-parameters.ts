// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUInferenceMetadata} from '../../analysis-core/contracts/index';
import {
  getPackedFloat32Values,
  validateCapacity,
  validateOption,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of uint32 elements in a space-time test parameter view. */
export const GPU_SPACE_TIME_PARAMETER_LENGTH = 4;

/** Number of float32 elements in the `summary` output of the space-time tests. */
export const GPU_SPACE_TIME_SUMMARY_LENGTH = 12;

/**
 * Indices of the `summary` output of `GPUKnoxTest` and `GPUMantelTest`. A summary reports the
 * statistic of the observed times and the Monte Carlo reference distribution of the `P`
 * swap-or-not-permuted time assignments.
 */
export const GPU_SPACE_TIME_SUMMARY = {
  /** Observed statistic: the Knox count, or Mantel's `r`. */
  observed: 0,
  /** Spatial pairs `i < j` of the pair list (the pairs the statistic runs over). */
  pairCount: 1,
  /** Knox: all pairs of rows within the time threshold of each other (0 for Mantel). */
  timeClosePairs: 2,
  /** Knox: `pairCount * timeClosePairs / (n (n - 1) / 2)`, the permutation mean (0 for Mantel). */
  expected: 3,
  /** Mean of the permuted statistics. */
  permutationMean: 4,
  /** Sample variance (`P - 1` denominator) of the permuted statistics. */
  permutationVariance: 5,
  /** Permutations with statistic `>=` observed. */
  greaterCount: 6,
  /** Permutations with statistic `<=` observed. */
  lesserCount: 7,
  /**
   * `(greaterCount + 1) / (P + 1)`: pseudo p-value for positive space-time interaction. This is the
   * Knox p-value (the hypothesis is an excess of close pairs) and the Mantel p-value used by
   * pointpats.
   */
  pseudoPGreater: 8,
  /** `(lesserCount + 1) / (P + 1)`. */
  pseudoPLesser: 9,
  /** `min(1, 2 min(pseudoPGreater, pseudoPLesser))`. */
  pseudoPTwoSided: 10,
  /** `(observed - permutationMean) / sqrt(permutationVariance)`, NaN when the variance is 0. */
  zScore: 11
} as const;

const {observed: mantelObservedCorrelationSlot, ...mantelSummarySlots} = GPU_SPACE_TIME_SUMMARY;

/** Mantel names over the unchanged shared summary slots. */
export const GPU_MANTEL_TEST_SUMMARY = {
  ...mantelSummarySlots,
  /** Pearson correlation; NaN for an undefined observation. */
  observedCorrelation: mantelObservedCorrelationSlot
} as const;

/** Named readback of the unchanged 12-word Mantel summary, accepting GPU bytes and float words. */
export function decodeGPUMantelTestSummary(values: Float32Array | Uint8Array | ArrayBuffer) {
  const tag = Object.prototype.toString.call(values);
  validateRequirement(
    'decodeGPUMantelTestSummary',
    ['[object Float32Array]', '[object Uint8Array]', '[object ArrayBuffer]'].includes(tag),
    `values must be Float32Array, Uint8Array or ArrayBuffer; received ${tag}`
  );
  const words = getPackedFloat32Values(values, GPU_SPACE_TIME_SUMMARY_LENGTH);
  return {
    observedCorrelation: words[0],
    pairCount: words[1],
    timeClosePairs: words[2],
    expected: words[3],
    permutationMean: words[4],
    permutationVariance: words[5],
    greaterCount: words[6],
    lesserCount: words[7],
    pseudoPGreater: words[8],
    pseudoPLesser: words[9],
    pseudoPTwoSided: words[10],
    zScore: words[11]
  };
}

/** Per-frame settings of `GPUKnoxTest` and `GPUMantelTest`, packed by {@link getGPUSpaceTimeParameterValues}. */
export type GPUSpaceTimeSettings = {
  /** Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. */
  seed: number;
  /** Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. */
  permutationCount: number;
  /**
   * Knox only: two rows are close in time when `|t_i - t_j| <= timeThreshold` (f32), in the same
   * units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by
   * `GPUMantelTest`.
   */
  timeThreshold?: number;
};

/**
 * Packs {@link GPUSpaceTimeSettings} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutationCount, float32 bits of timeThreshold]`.
 */
export function getGPUSpaceTimeParameterValues(settings: GPUSpaceTimeSettings): Uint32Array {
  const {seed, permutationCount} = settings;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('space-time seed must be a non-negative safe integer');
  }
  validateCapacity('space-time', 'permutationCount', permutationCount, 1, 2 ** 31 - 1);
  const timeThreshold = settings.timeThreshold ?? 0;
  if (Number.isNaN(timeThreshold) || timeThreshold < 0) {
    throw new Error('space-time timeThreshold must be non-negative');
  }
  const values = new Uint32Array(GPU_SPACE_TIME_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutationCount;
  values[3] = new Uint32Array(new Float32Array([timeThreshold]).buffer)[0];
  return values;
}

/**
 * Upper-tail p-value `P(X >= observed)` of a Poisson distribution with mean `expected`, the
 * classic Knox (1964) test. CPU helper for the `observed` and `expected` of `GPUKnoxTest`'s
 * `summary`. When `observed` exceeds `expected` the tail terms are summed directly, so a small
 * p-value keeps its relative precision down to the float64 range (about 1e-300) instead of
 * rounding to 0; otherwise it is one minus the lower tail. A fractional `observed` is rounded up.
 * Returns NaN when either input is nonfinite, including a suppressed GPU summary.
 *
 * @param observed Observed Knox count.
 * @param expected Expected count under no interaction.
 */
export function getKnoxPoissonPValue(observed: number, expected: number): number {
  if (!Number.isFinite(observed) || !Number.isFinite(expected)) return NaN;
  if (observed <= 0) {
    return 1;
  }
  if (!(expected > 0)) {
    return 0;
  }
  const threshold = Math.ceil(observed);
  if (threshold > expected) {
    // Right tail: sum P(X = k) for k >= threshold directly, so a small tail keeps its relative
    // precision instead of vanishing in `1 - P(X < threshold)`.
    return Math.min(1, sumPoissonUpperTerms(threshold, expected));
  }
  // The upper tail is at least about one half here; summing the short lower tail is exact enough.
  const lowerTail = sumPoissonLowerTerms(threshold - 1, expected);
  return Math.max(0, Math.min(1, 1 - lowerTail));
}

const STIRLING_ERROR_SMALL = [
  0, 0.0810614667953272582196702, 0.0413406959554092940938221, 0.02767792568499833914878929,
  0.02079067210376509311152277, 0.01664469118982119216319487, 0.01387612882307074799874573,
  0.01189670994589177009505572, 0.010411265261972096497478567, 0.009255462182712732917728637,
  0.00833056343336287125646931, 0.007573675487951840794972024, 0.006942840107209529865664152,
  0.006408994188004207068439631, 0.005951370112758847735624416, 0.00555473355196280137103869
];

/** `log(n!) - ((n + 1/2) log(n) - n + log(2 pi) / 2)` for integer `n >= 0` (Loader 2000). */
function getStirlingError(count: number): number {
  if (count < STIRLING_ERROR_SMALL.length) return STIRLING_ERROR_SMALL[count];
  const squared = count * count;
  if (count > 500) return (1 / 12 - 1 / 360 / squared) / count;
  if (count > 80) return (1 / 12 - (1 / 360 - 1 / 1260 / squared) / squared) / count;
  if (count > 35) {
    return (1 / 12 - (1 / 360 - (1 / 1260 - 1 / 1680 / squared) / squared) / squared) / count;
  }
  return (
    (1 / 12 -
      (1 / 360 - (1 / 1260 - (1 / 1680 - 1 / 1188 / squared) / squared) / squared) / squared) /
    count
  );
}

/** Deviance term `x log(x / mean) + mean - x`, accurate when `x` is close to `mean` (Loader 2000). */
function getPoissonDeviance(count: number, mean: number): number {
  if (Math.abs(count - mean) < 0.1 * (count + mean)) {
    let ratio = (count - mean) / (count + mean);
    let sum = (count - mean) * ratio;
    let term = 2 * count * ratio;
    ratio *= ratio;
    for (let index = 1; index < 1000; index++) {
      term *= ratio;
      const next = sum + term / (2 * index + 1);
      if (next === sum) return next;
      sum = next;
    }
    return sum;
  }
  // Separate logarithms preserve a representable tail when a subnormal mean overflows the ratio.
  const ratio = count / mean;
  const logRatio = Number.isFinite(ratio) ? Math.log(ratio) : Math.log(count) - Math.log(mean);
  return count * logRatio + mean - count;
}

/** Poisson probability mass `P(X = count)` for integer `count >= 0`, without log-factorial cancellation. */
function getPoissonProbability(count: number, mean: number): number {
  if (count === 0) return Math.exp(-mean);
  return (
    Math.exp(-getStirlingError(count) - getPoissonDeviance(count, mean)) /
    Math.sqrt(2 * Math.PI * count)
  );
}

/** `P(X >= threshold)` for `threshold > mean`: terms decrease geometrically from the first. */
function sumPoissonUpperTerms(threshold: number, mean: number): number {
  const first = getPoissonProbability(threshold, mean);
  if (first === 0) return 0;
  let ratioSum = 1;
  let term = 1;
  for (let count = threshold + 1; ; count++) {
    term *= mean / count;
    const next = ratioSum + term;
    if (next === ratioSum) break;
    ratioSum = next;
  }
  return first * ratioSum;
}

/** `P(X <= last)` for `last < mean`: terms decrease from `last` downward. */
function sumPoissonLowerTerms(last: number, mean: number): number {
  if (last < 0) return 0;
  const first = getPoissonProbability(last, mean);
  if (first === 0) return 0;
  let ratioSum = 1;
  let term = 1;
  for (let count = last; count > 0; count--) {
    term *= count / mean;
    const next = ratioSum + term;
    if (next === ratioSum) break;
    ratioSum = next;
  }
  return first * ratioSum;
}

/** Result of {@link getGPUSpaceTimeRelativeTimes}. */
export type GPUSpaceTimeRelativeTimes = {
  /** `time - origin` rounded to float32, ready for `GPUKnoxTest` and `GPUMantelTest`. */
  times: Float32Array;
  /** The float64 origin that was subtracted, in the original time unit. */
  origin: number;
  /**
   * Spacing of float32 values at the largest relative time, in the original time unit: times
   * closer than this can collapse to the same value, so `timeThreshold` should be several times
   * larger. 0 when all times equal the origin.
   */
  resolution: number;
};

/**
 * Subtracts a float64 origin from event times on the CPU and rounds the result to float32.
 *
 * float32 has 24 bits of mantissa, so raw Unix epoch seconds (about 1.7e9) are spaced 128 s apart
 * and epoch milliseconds are unusable. Relative times keep full precision near the origin: the
 * spacing at a span `S` is `2^(floor(log2 S) - 23)`, for example 0.5 ms for 1 hour, 0.03 s for 3
 * days and 2 s for 4.6 months of seconds. Use the same unit for `timeThreshold`.
 *
 * @param times Event times in any unit; every value must be finite.
 * @param origin Float64 origin to subtract. Defaults to the minimum time.
 * @param requiredResolution Optional largest acceptable float32 spacing, in the time unit. Throws
 *   when the span makes the spacing coarser, so a lost resolution is never silent.
 */
export function getGPUSpaceTimeRelativeTimes(
  times: ArrayLike<number>,
  origin?: number,
  requiredResolution?: number
): GPUSpaceTimeRelativeTimes {
  const count = times.length;
  let minimum = Infinity;
  for (let index = 0; index < count; index++) {
    const time = times[index];
    if (!Number.isFinite(time)) {
      throw new Error('space-time times must be finite');
    }
    minimum = Math.min(minimum, time);
  }
  const resolvedOrigin = origin ?? (count > 0 ? minimum : 0);
  if (!Number.isFinite(resolvedOrigin)) {
    throw new Error('space-time origin must be finite');
  }
  const relative = new Float32Array(count);
  let largest = 0;
  let hasNonzeroDifference = false;
  for (let index = 0; index < count; index++) {
    const value = times[index] - resolvedOrigin;
    relative[index] = value;
    if (!Number.isFinite(relative[index])) {
      throw new Error(
        `space-time times[${index}] must encode as finite float32; received ${value}`
      );
    }
    hasNonzeroDifference ||= value !== 0;
    largest = Math.max(largest, Math.abs(relative[index]));
  }
  const resolution = hasNonzeroDifference
    ? Math.max(2 ** -149, largest > 0 ? 2 ** (Math.floor(Math.log2(largest)) - 23) : 0)
    : 0;
  if (requiredResolution !== undefined && resolution > requiredResolution) {
    throw new Error(`space-time float32 resolution ${resolution} exceeds ${requiredResolution}`);
  }
  return {times: relative, origin: resolvedOrigin, resolution};
}

/** Selects a probability word without confusing scientific sidedness and its counting rule. */
export type GPUSpaceTimePValueField = 'pseudoPGreater' | 'pseudoPLesser' | 'pseudoPTwoSided';

/** Settings-derived inference description; no GPU result is read to construct it. */
export type GPUSpaceTimeInferenceMetadata = Omit<GPUInferenceMetadata, 'permutationCount'> & {
  /** Requested draw count after clamping to `maximumPermutations`, before early stopping. */
  requestedPermutationCount: number;
  /** Mantel conditional inference uses defined null correlations, not the attempted prefix. */
  inferenceCountDefinition?: 'usable-correlations';
  stoppingRule: {method: 'fixed-count'} | {method: 'besag-clifford'; stopAfterExceedances: number};
  /**
   * Present only for fixed-count Knox/Jacquez inference. Mantel settings never publish a
   * data-dependent inference count. For final records use
   * {@link decodeGPUSpaceTimeInferenceMetadata} with the published draw count.
   */
  permutationCount?: number;
};

/** Mantel settings cannot certify data-dependent usable draws, even with a fixed budget. */
export type GPUMantelTestInferenceMetadata = Omit<
  GPUSpaceTimeInferenceMetadata,
  'permutationCount'
> & {
  inferenceCountDefinition: 'usable-correlations';
  permutationCount?: never;
};

/** A methods record distinguishes the statistics prefix from the inference denominator. */
export type GPUSpaceTimeDecodedInferenceMetadata = Omit<
  GPUSpaceTimeInferenceMetadata,
  'permutationCount'
> & {
  /** Attempted prefix length, excluding observed slot 0. */
  permutationCount: number;
  usablePermutationCount: number;
  available: boolean;
  /** Metadata to pass to probability-resolution helpers: its count is the usable denominator. */
  inference: GPUInferenceMetadata;
};

/**
 * Decode caller-read counts without GPU readback. permutationCount is the attempted prefix;
 * Mantel also requires usablePermutationCount for a positive prefix. The nested inference
 * metadata contains the usable denominator for significance-resolution helpers. Zero usable
 * draws mean available=false, including a defined observation with no defined null draws.
 */
export function decodeGPUSpaceTimeInferenceMetadata(
  metadata: GPUSpaceTimeInferenceMetadata,
  permutationCount: number,
  usablePermutationCount?: number
): GPUSpaceTimeDecodedInferenceMetadata {
  validateRequirement(
    'space-time',
    !('permutationCountDefinition' in metadata),
    'permutationCountDefinition was removed. Use inferenceCountDefinition'
  );
  const conditional = metadata.inferenceCountDefinition === 'usable-correlations';
  validateCapacity(
    'space-time',
    'permutationCount',
    permutationCount,
    0,
    metadata.requestedPermutationCount
  );
  if (permutationCount > 0)
    validateCapacity(
      'space-time',
      'permutationCount',
      permutationCount,
      metadata.stoppingRule.method === 'fixed-count'
        ? metadata.requestedPermutationCount
        : Math.min(metadata.requestedPermutationCount, metadata.stoppingRule.stopAfterExceedances),
      metadata.requestedPermutationCount
    );
  validateRequirement(
    'space-time',
    !conditional || permutationCount === 0 || usablePermutationCount !== undefined,
    'usablePermutationCount is required for Mantel inference; read output.usablePermutationCount'
  );
  const usable = usablePermutationCount ?? permutationCount;
  validateCapacity('space-time', 'usablePermutationCount', usable, 0, permutationCount);
  if (!conditional)
    validateRequirement(
      'space-time',
      usable === permutationCount,
      `usablePermutationCount must equal permutationCount ${permutationCount}; received ${usable}`
    );
  return {
    ...metadata,
    permutationCount,
    usablePermutationCount: usable,
    available: usable > 0,
    inference: {...metadata, permutationCount: usable}
  };
}

/**
 * CPU-side inference metadata of the space-time tests: Monte Carlo over swap-or-not shuffle
 * permutations of the times, alternative `'greater'` for both tests (Knox: excess close pairs;
 * Mantel: positive space-time correlation), no multiple-testing correction. `settings` is
 * validated as {@link getGPUSpaceTimeParameterValues} does and the count is clamped to
 * `maximumPermutations`, as the kernels do.
 *
 * @internal
 */
export function getGPUSpaceTimeInferenceMetadata(
  test: 'knox',
  settings: GPUSpaceTimeSettings,
  maximumPermutations: number
): GPUSpaceTimeInferenceMetadata & GPUInferenceMetadata;
/** @internal */
export function getGPUSpaceTimeInferenceMetadata(
  test: 'mantel',
  settings: GPUSpaceTimeSettings,
  maximumPermutations: number,
  stopAfterExceedances?: number,
  summaryField?: GPUSpaceTimePValueField
): GPUMantelTestInferenceMetadata;
/** @internal */
export function getGPUSpaceTimeInferenceMetadata(
  test: 'knox' | 'mantel',
  settings: GPUSpaceTimeSettings,
  maximumPermutations: number,
  stopAfterExceedances?: number,
  summaryField?: GPUSpaceTimePValueField
): GPUSpaceTimeInferenceMetadata;
export function getGPUSpaceTimeInferenceMetadata(
  test: 'knox' | 'mantel',
  settings: GPUSpaceTimeSettings,
  maximumPermutations: number,
  stopAfterExceedances?: number,
  summaryField: GPUSpaceTimePValueField = 'pseudoPGreater'
): GPUSpaceTimeInferenceMetadata {
  getGPUSpaceTimeParameterValues(settings);
  validateOption('space-time', 'summaryField', summaryField, [
    'pseudoPGreater',
    'pseudoPLesser',
    'pseudoPTwoSided'
  ]);
  const requestedPermutationCount = Math.min(settings.permutationCount, maximumPermutations);
  return {
    alternative:
      summaryField === 'pseudoPTwoSided'
        ? 'two-sided'
        : summaryField === 'pseudoPLesser'
          ? 'lesser'
          : 'greater',
    pValueConstruction:
      summaryField === 'pseudoPTwoSided' ? 'doubled-minimum-tail' : 'upper-tail-count',
    ...(test === 'mantel' ? {inferenceCountDefinition: 'usable-correlations' as const} : {}),
    requestedPermutationCount,
    ...(stopAfterExceedances === undefined
      ? {
          ...(test === 'mantel' ? {} : {permutationCount: requestedPermutationCount}),
          stoppingRule: {method: 'fixed-count' as const}
        }
      : {stoppingRule: {method: 'besag-clifford' as const, stopAfterExceedances}}),
    seed: settings.seed,
    multipleTesting: 'none',
    testedRowsDefinition:
      test === 'knox'
        ? 'all unordered event pairs listed in the spatial pair list, counted once'
        : 'all unordered event pairs listed in the distance band, counted once'
  };
}
