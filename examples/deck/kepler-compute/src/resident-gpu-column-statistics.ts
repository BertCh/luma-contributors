// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ColumnStats, ColumnStatsType} from '@kepler.gl/utils';
import type {GPUColumnStatistics} from '@luma-contributors/kepler';

/** Quantile stops of the native numeric column statistics (labels match computeColumnStats). */
const QUANTILE_STOPS = [0, 0.25, 0.5, 0.75, 1] as const;

/**
 * Linear interpolation of a quantile inside an equal-width histogram. Min and Max are exact; the
 * interior quantiles are approximations bounded by one histogram bin width, and the caller labels
 * the result accordingly.
 */
export function estimateHistogramQuantile(
  histogram: ArrayLike<number>,
  minimum: number,
  maximum: number,
  probability: number
): number {
  if (probability <= 0) return minimum;
  if (probability >= 1) return maximum;
  let total = 0;
  for (let index = 0; index < histogram.length; index++) total += histogram[index];
  if (!total || !(maximum > minimum)) return minimum;
  const width = (maximum - minimum) / histogram.length;
  const target = probability * total;
  let seen = 0;
  for (let index = 0; index < histogram.length; index++) {
    const next = seen + histogram[index];
    if (next >= target && histogram[index] > 0) {
      return minimum + width * (index + (target - seen) / histogram[index]);
    }
    seen = next;
  }
  return maximum;
}

const finiteOrNull = (value: number) => (Number.isFinite(value) ? value : null);

/**
 * Kepler-shaped numeric column statistics from a GPU profile. Mean, deviation, range, null share and
 * histogram counts are exact (histogram bins differ from d3's tick-aligned bins); quantiles between
 * Min and Max are histogram estimates.
 */
export function makeNumericColumnStatsFromGPU(
  statistics: GPUColumnStatistics
): ColumnStats & {approximateQuantiles: true; source: 'gpu'} {
  const {histogram, minimum, maximum, count, populationRows} = statistics;
  const width = histogram.length ? (maximum - minimum) / histogram.length : 0;
  const hasRange = Number.isFinite(minimum) && Number.isFinite(maximum);
  return {
    // String enum member; avoids a value import of @kepler.gl/utils in node-tested code.
    type: 'numeric' as ColumnStatsType.numeric,
    bins: hasRange
      ? Array.from(histogram, (binCount, index) => ({
          count: binCount,
          indexes: [],
          x0: minimum + width * index,
          x1: index === histogram.length - 1 ? maximum : minimum + width * (index + 1)
        }))
      : [],
    quantiles: QUANTILE_STOPS.map(probability => ({
      label: probability === 0 ? 'Min' : probability === 1 ? 'Max' : `${probability * 100}%`,
      value:
        count && hasRange
          ? finiteOrNull(estimateHistogramQuantile(histogram, minimum, maximum, probability))
          : null
    })),
    mean: count ? finiteOrNull(statistics.mean) : null,
    std: count > 1 ? finiteOrNull(Math.sqrt(statistics.sampleVariance)) : null,
    percentNulls: populationRows > 0 ? (populationRows - count) / populationRows : null,
    approximateQuantiles: true,
    source: 'gpu'
  } as ColumnStats & {approximateQuantiles: true; source: 'gpu'};
}
