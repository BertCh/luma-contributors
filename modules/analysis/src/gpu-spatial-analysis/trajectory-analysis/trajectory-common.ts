// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';
import {TIME_WORDS_WGSL} from '../../dataframe-analysis/time-window-filter/time-words';

/** Float32 maximum: a `maximumGap` of 0 becomes this limit, which no finite step exceeds. */
const FLOAT32_MAXIMUM_WGSL = '3.402823466e+38';

/**
 * WGSL of the effective gap limit: `raw`, or no limit when `raw` is 0. For parameter arrays that
 * otherwise require a positive `maximumGap`; negative and NaN limits stay invalid there. @internal
 */
export function getEffectiveMaximumGapWGSL(raw: string): string {
  return `select(${raw}, ${FLOAT32_MAXIMUM_WGSL}, ${raw} == 0.0)`;
}

/**
 * Validates an optional trajectory `temporalContext` against the bound time views: `uint32x2` is
 * `int64-words`, `float32` is `float32-relative` (`double-single` with `timestampsLow`), and the
 * axis is irregular. @internal
 */
export function resolveGPUTrajectoryTemporalContext(
  id: string,
  timestamps: GraphDataView | undefined,
  context: GPUTemporalContext | undefined,
  timestampsLow?: GraphDataView
): GPUTemporalContext | undefined {
  if (!context) {
    return undefined;
  }
  if (!timestamps) {
    return resolveGPUTemporalContext(id, context, {axisKinds: ['irregular']});
  }
  const encoding =
    timestamps.format === 'uint32x2'
      ? 'int64-words'
      : timestampsLow
        ? 'double-single'
        : 'float32-relative';
  return resolveGPUTemporalContext(id, context, {encodings: [encoding], axisKinds: ['irregular']});
}

/** Trajectory time columns under the names Kalman and restriction have always taken. */
export type GPUTrajectoryTimeColumns = {
  /** Time rows, as `timestamps` of the trajectory port. */
  times: GPUTrajectoryGeometryFields['timestamps'];
  /** Track-to-row offsets with one terminal entry, as `trackOffsets` of the trajectory port. */
  offsets: GraphDataView<'uint32'>;
};

/**
 * Time and offset columns of Kalman and restriction: either the trajectory port's `timestamps` and
 * `trackOffsets`, or the equivalent `times` and `offsets`. Not both.
 */
export type GPUTrajectoryTimeFields = {
  /** Optional declaration of the time column; see `GPUTrajectoryGeometryPort`. */
  temporalContext?: GPUTemporalContext;
} & (
  | (GPUTrajectoryTimeColumns & {timestamps?: never; trackOffsets?: never})
  | (Pick<GPUTrajectoryGeometryFields, 'timestamps' | 'trackOffsets'> & {
      times?: never;
      offsets?: never;
    })
);

/** Reads the time and offset columns of either shape of {@link GPUTrajectoryTimeFields}. @internal */
export function getGPUTrajectoryTimeColumns(
  id: string,
  fields: GPUTrajectoryTimeFields
): GPUTrajectoryTimeColumns {
  const hasLegacyFields = Boolean(fields.times || fields.offsets);
  const hasPortFields = Boolean(fields.timestamps || fields.trackOffsets);
  const times = fields.times ?? fields.timestamps;
  const offsets = fields.offsets ?? fields.trackOffsets;
  if (hasLegacyFields === hasPortFields || !times || !offsets) {
    throw new Error(`${id} takes either timestamps and trackOffsets, or times and offsets`);
  }
  return {times, offsets};
}

/** Number of radix key bits needed to sort keys up to `maximumKey`. @internal */
export function getKeyBitCount(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}

/**
 * WGSL `rowTimeDifference(a, b)` in the timestamp unit: f32 values, or exact two-word
 * timestamps (low, high) when `isWordMode`. @internal
 */
export function getTimestampSourceWGSL(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsDifferenceToF32(wordsA, wordsB);
}`
    : /* wgsl */ `
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`;
}

/** Throws unless a parameter target holds the required element count. @internal */
export function checkParameterTarget(
  target: Float32Array | Uint32Array,
  length: number,
  name: string
): void {
  if (target.length < length) {
    throw new Error(`${name} target must hold ${length} elements`);
  }
}
