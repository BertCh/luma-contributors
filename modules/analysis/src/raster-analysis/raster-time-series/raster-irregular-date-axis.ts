// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import type {ArrowAnalyticsTimeUnit} from '../../analysis-core/graph-support/arrow-analytics-normalization';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../dataframe-analysis/time-window-filter/time-words';

/** Length of the Gregorian mean year, 365.2425 days, in seconds. */
export const GPU_RASTER_SECONDS_PER_YEAR = 31556952;

const TICKS_PER_SECOND: Readonly<Record<ArrowAnalyticsTimeUnit, number>> = {
  second: 1,
  millisecond: 1e3,
  microsecond: 1e6,
  nanosecond: 1e9
};

/**
 * Acquisition dates of a raster stack, one per slice. Times must be strictly increasing; this is a
 * precondition, not checked on the GPU. The axis is centred fractional years in float32: years of
 * 365.2425 days after the midpoint of the first and last acquisition, so trend columns are O(1)
 * and harmonic columns stay in [-1, 1].
 */
export type GPURasterIrregularDateAxis = {
  /** One `uint32x2` Int64 `(low, high)` row per slice, in the unit of `temporalContext`. */
  acquisitionTimes: GraphDataView<'uint32x2'>;
  /** Must be `int64-words` with `{kind: 'irregular'}`; any unit and any epoch. */
  temporalContext: GPUTemporalContext;
};

/** Validates a date axis and returns the ticks per year of its unit. */
export function resolveGPURasterIrregularDateAxis(
  id: string,
  axis: GPURasterIrregularDateAxis,
  sliceCount: number
): number {
  const context = resolveGPUTemporalContext(id, axis.temporalContext, {
    encodings: ['int64-words'],
    axisKinds: ['irregular']
  });
  validateSinglePackedView(axis.acquisitionTimes, ['uint32x2'], `${id} dateAxis.acquisitionTimes`);
  validateRequirement(
    id,
    axis.acquisitionTimes.length >= sliceCount,
    'dateAxis.acquisitionTimes must hold one row per slice'
  );
  return getGPURasterTicksPerYear(context.unit);
}

/** Checks that the acquisition-time view belongs to `graph`. */
export function validateGPURasterDateAxisGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  axis: GPURasterIrregularDateAxis
): void {
  validateGraphViewsBelongToGraph(id, graph, [axis.acquisitionTimes]);
}

/** Ticks of `unit` in one Gregorian mean year. */
export function getGPURasterTicksPerYear(unit: ArrowAnalyticsTimeUnit): number {
  return GPU_RASTER_SECONDS_PER_YEAR * TICKS_PER_SECOND[unit];
}

/**
 * Centred fractional years of a strictly increasing acquisition series, for CPU-side use such as
 * choosing a regular output grid. Matches the axis the kernels build from the same words.
 */
export function getGPURasterCentredFractionalYears(
  times: BigInt64Array,
  unit: ArrowAnalyticsTimeUnit
): Float32Array {
  const ticksPerYear = getGPURasterTicksPerYear(unit);
  const midpoint = times[0] + (times[times.length - 1] - times[0]) / 2n;
  return Float32Array.from(times, time => Number(time - midpoint) / ticksPerYear);
}

/** Binding the declarations of {@link getGPURasterDateAxisDeclarations} read. */
export function getGPURasterDateAxisBinding(axis: GPURasterIrregularDateAxis): WGSLKernelBinding {
  return {name: 'acquisitionTimes', view: axis.acquisitionTimes, type: 'u32', access: 'read'};
}

/**
 * WGSL declaring `rasterSliceYears(slice)`, the centred fractional year of a slice, and
 * `rasterYearsBetween(later, earlier)`, the exact elapsed years between two slices. Elapsed years
 * come straight from the Int64 words, so a difference of nearby dates keeps full float32 precision
 * even far from the centre. Requires the binding of {@link getGPURasterDateAxisBinding}.
 */
export function getGPURasterDateAxisDeclarations(ticksPerYear: number, sliceCount: number): string {
  return `${TIME_WORDS_WGSL}
const RASTER_AXIS_TICKS_PER_YEAR: f32 = ${ticksPerYear}.0;
fn rasterAxisWords(slice: u32) -> vec2<u32> {
  return vec2<u32>(acquisitionTimes[acquisitionTimesOffset + 2u * slice], acquisitionTimes[acquisitionTimesOffset + 2u * slice + 1u]);
}
fn rasterAxisMidpoint() -> vec2<u32> {
  let first = rasterAxisWords(0u);
  let difference = timeWordsSubtract(rasterAxisWords(${sliceCount - 1}u), first);
  let low = first.x + ((difference.x >> 1u) | (difference.y << 31u));
  return vec2<u32>(low, first.y + (difference.y >> 1u) + select(0u, 1u, low < first.x));
}
fn rasterSliceYears(slice: u32) -> f32 {
  return timeWordsDifferenceToF32(rasterAxisWords(slice), rasterAxisMidpoint()) / RASTER_AXIS_TICKS_PER_YEAR;
}
fn rasterYearsBetween(later: u32, earlier: u32) -> f32 {
  return timeWordsDifferenceToF32(rasterAxisWords(later), rasterAxisWords(earlier)) / RASTER_AXIS_TICKS_PER_YEAR;
}
`;
}
