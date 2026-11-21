// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of uint32 elements in a scan statistic parameter view. */
export const GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH = 8;

/**
 * Largest size of each of the two per-center neighbor tables (zone indices and separation keys),
 * in words: `zones * listLength <= 2^24`, so 64 MiB per table. All zones fit up to 4096 zones; beyond
 * that the compile-time `maximumWindowZones` capacity shrinks and truncated windows are reported.
 */
export const GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS = 2 ** 24;

/** Largest compile-time `timeBuckets`. */
export const GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS = 32;

/** Largest compile-time `maximumClusters`. */
export const GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_CLUSTERS = 64;

/**
 * Largest compile-time `maximumPermutations` (SaTScan-style 99999 replicates). The replicates run in
 * batches, so only the unrolled batch count (at most 1024) bounds it further, see
 * `GPUSpatialScanStatisticProps.maximumPermutations`.
 */
export const GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS = 99999;

/** Words per cluster in the `clusterIndices` output: center zone, zone count, first and last bucket. */
export const GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS = 4;

/** Words per cluster in the `clusterStatistics` output, see `GPU_SPATIAL_SCAN_STATISTIC_CLUSTER`. */
export const GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS = 8;

/** Number of uint32 elements in the `summary` output. */
export const GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH = 4;

/** Probability models of {@link GPUSpatialScanStatistic}. */
export type GPUSpatialScanModel = 'poisson' | 'bernoulli' | 'space-time-permutation';

/** Direction of the rate anomaly a scan looks for. */
export const GPU_SPATIAL_SCAN_STATISTIC_RATE_DIRECTION = {
  /** Windows whose rate exceeds the rate outside (`c > e`). */
  high: 0,
  /** Windows whose rate is below the rate outside (`c < e`). */
  low: 1,
  /** Both. */
  both: 2
} as const;

/** Rate direction name of {@link GPUSpatialScanStatisticSettings}. */
export type GPUSpatialScanRateDirection = keyof typeof GPU_SPATIAL_SCAN_STATISTIC_RATE_DIRECTION;

/** Compile-time layout of the per-center neighbor tables. */
export type GPUSpatialScanStatisticLayout = {
  /** Windows may hold up to this many zones (the compile-time capacity). */
  usableZones: number;
  /**
   * Rows per center of the neighbor tables: `usableZones`, plus one look-ahead zone that tells
   * whether the last window ends between two distances, when not every zone fits.
   */
  listLength: number;
};

/**
 * Resolves the compile-time neighbor-table layout. `maximumWindowZones` defaults to every zone
 * when `zones * zones <= GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS`, else to the largest capacity
 * that fits. Throws when an explicit capacity does not fit.
 */
export function getGPUSpatialScanStatisticLayout(
  zones: number,
  maximumWindowZones?: number,
  id = 'scan statistic'
): GPUSpatialScanStatisticLayout {
  validateCapacity(id, 'zones', zones);
  const words = GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS;
  const defaultZones = zones * zones <= words ? zones : Math.max(1, Math.floor(words / zones) - 1);
  const usableZones = Math.min(maximumWindowZones ?? defaultZones, zones);
  validateCapacity(id, 'maximumWindowZones (effective)', usableZones);
  const listLength = usableZones < zones ? usableZones + 1 : zones;
  validateMaximumCount(
    id,
    'zones * listLength (reduce maximumWindowZones)',
    listLength * zones,
    words
  );
  return {usableZones, listLength};
}

/** Indices into one `clusterIndices` record. */
export const GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX = {
  /** Zone at the center of the circular window. */
  center: 0,
  /** Number of zones in the window: the center's `zoneCount` nearest zones, itself included. */
  zoneCount: 1,
  /** First time bucket of the cylinder, inclusive. */
  firstBucket: 2,
  /** Last time bucket of the cylinder, inclusive. */
  lastBucket: 3
} as const;

/** Indices into one `clusterStatistics` record. */
export const GPU_SPATIAL_SCAN_STATISTIC_CLUSTER = {
  /** Log-likelihood ratio of the cluster under the contributor's model. */
  logLikelihoodRatio: 0,
  /** Observed cases in the cluster. */
  observedCases: 1,
  /** Expected cases in the cluster under the null (baseline scaled to the total case count). */
  expectedCases: 2,
  /** Monte Carlo p-value `(1 + #{replicate maxima >= LLR}) / (P + 1)`. */
  pValue: 3,
  /** Radius from the center to its farthest zone, in the spatial context's length unit. */
  radius: 4,
  /** `observedCases / expectedCases`. */
  observedOverExpected: 5,
  /**
   * 1 when the compile-time `maximumWindowZones` capacity cut the windows of the cluster's center
   * (larger windows were within the population fraction but not evaluated), else 0.
   */
  truncatedByZoneCap: 6
} as const;

/** Indices into the `summary` output. */
export const GPU_SPATIAL_SCAN_STATISTIC_SUMMARY = {
  /** Clusters written to the outputs (at most `maximumClusters`). */
  clusterCount: 0,
  /** Total observed cases `C` over all zones and buckets. */
  totalCases: 1,
  /** Monte Carlo replicates evaluated; zero for an invalid frame whose batches were suppressed. */
  permutationCount: 2,
  /**
   * `overflow`: centers whose windows were cut by the compile-time `maximumWindowZones` capacity
   * rather than by the population fraction or the per-frame limits. 0 means no window was truncated.
   */
  overflow: 3
} as const;

/** Spatial window shapes. */
export const GPU_SPATIAL_SCAN_STATISTIC_WINDOW_SHAPE = {
  /** Circles: a window never splits zones at the same distance from the center. */
  circle: 0,
  /** The `k` nearest zones for every `k`, ties broken by zone index. */
  nearest: 1
} as const;

/** Window shape name of {@link GPUSpatialScanStatisticSettings}. */
export type GPUSpatialScanWindowShape = keyof typeof GPU_SPATIAL_SCAN_STATISTIC_WINDOW_SHAPE;

/** Per-frame settings of `GPUSpatialScanStatistic`, packed by {@link getGPUSpatialScanStatisticParameterValues}. */
export type GPUSpatialScanStatisticSettings = {
  /** Random seed, an integer in `[0, 2^53)`. The result is a pure function of seed and inputs. */
  seed: number;
  /** Monte Carlo replicates `P`, at most the contributor's `maximumPermutations`. */
  permutationCount: number;
  /**
   * Largest spatial window as a fraction of the total population, remaining in `(0, 1]` after float32 conversion. A window grows until
   * adding the next zone would exceed it. Defaults to 0.5.
   */
  maximumPopulationFraction?: number;
  /**
   * Finite positive requested zone limit, rounded down (minimum 1); evaluated windows are also
   * bounded by compile-time capacity. Omission leaves capacity as a storage constraint, with the
   * population fraction governing eligible windows. An explicit limit also excludes larger
   * windows from capacity loss diagnostics, even when it equals capacity.
   */
  maximumWindowZones?: number;
  /**
   * Finite positive cylinder length in buckets, rounded down (minimum 1) and clamped to `timeBuckets`. Defaults to 50% of the study
   * period, `max(1, floor(timeBuckets / 2))`, as in SaTScan.
   */
  maximumTimeBuckets?: number;
  /** Window shape. Defaults to `'circle'`. */
  windowShape?: GPUSpatialScanWindowShape;
  /** Which anomalies count as clusters. Defaults to `'high'`. */
  rateDirection?: GPUSpatialScanRateDirection;
};

/**
 * Packs {@link GPUSpatialScanStatisticSettings} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutationCount, float32 bits of maximumPopulationFraction,
 * maximumWindowZones, maximumTimeBuckets (0: 50% of the study period), windowShape, rateDirection]`.
 */
export function getGPUSpatialScanStatisticParameterValues(
  settings: GPUSpatialScanStatisticSettings
): Uint32Array {
  const {
    seed,
    permutationCount,
    maximumPopulationFraction,
    maximumWindowZones,
    maximumTimeBuckets,
    windowShape,
    rateDirection
  } = settings;
  validateCapacity('scan statistic', 'seed', seed, 0, Number.MAX_SAFE_INTEGER);
  validateCapacity('scan statistic', 'permutationCount', permutationCount, 1, 2 ** 31 - 1);
  const fraction = maximumPopulationFraction ?? 0.5;
  validateRequirement(
    'scan statistic',
    fraction > 0 && fraction <= 1 && Math.fround(fraction) > 0,
    `maximumPopulationFraction received ${fraction}; must remain in (0, 1] after float32 conversion`
  );
  for (const [name, value] of [
    ['maximumWindowZones', maximumWindowZones],
    ['maximumTimeBuckets', maximumTimeBuckets]
  ] as const) {
    validateRequirement(
      'scan statistic',
      value === undefined || (Number.isFinite(value) && value > 0),
      `${name} received ${String(value)}; must be finite and positive when supplied`
    );
  }
  for (const [field, received, accepted] of [
    ['windowShape', windowShape, ['circle', 'nearest']],
    ['rateDirection', rateDirection, ['high', 'low', 'both']]
  ] as const) {
    validateRequirement(
      'scan statistic',
      received === undefined || accepted.some(value => value === received),
      `${field} received ${String(received)}; expected ${accepted.join(', ')}`,
      {code: 'INVALID_OPTION', path: field, received, expected: {values: accepted}}
    );
  }
  const values = new Uint32Array(GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutationCount;
  values[3] = new Uint32Array(new Float32Array([fraction]).buffer)[0];
  // Preserve over-capacity requests for shader clamping instead of wrapping modulo 2^32.
  values[4] = Math.min(0xffffffff, Math.max(1, Math.floor(maximumWindowZones ?? 2 ** 31)));
  values[5] =
    maximumTimeBuckets === undefined
      ? 0
      : Math.min(0xffffffff, Math.max(1, Math.floor(maximumTimeBuckets)));
  values[6] = GPU_SPATIAL_SCAN_STATISTIC_WINDOW_SHAPE[windowShape ?? 'circle'];
  values[7] = GPU_SPATIAL_SCAN_STATISTIC_RATE_DIRECTION[rateDirection ?? 'high'];
  return values;
}

/**
 * Recurrence interval of a prospective scan alarm: `analysisInterval / pValue`, in the unit of
 * `analysisInterval` (Kulldorff 2001), the mean time between alarms this extreme under the null
 * when the analysis repeats every `analysisInterval`. Repeated analyses are not adjusted for.
 */
export function getGPUScanRecurrenceInterval(pValue: number, analysisInterval = 1): number {
  validateRequirement(
    'scan recurrence interval',
    pValue > 0 && pValue <= 1,
    `pValue received ${pValue}; must be in (0, 1]`
  );
  validateRequirement(
    'scan recurrence interval',
    analysisInterval > 0,
    `analysisInterval received ${analysisInterval}; must be positive`
  );
  return analysisInterval / pValue;
}
