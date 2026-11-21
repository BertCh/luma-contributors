// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64ify} from '@luma.gl/shaders';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a time-window parameter buffer. */
export const GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH = 8;

/**
 * CPU description of one per-frame time window, the settings of {@link GPUTimeWindowFilter}.
 *
 * Times use the same unit and epoch as the row timestamps, for example seconds since an
 * application epoch, or absolute epoch values when timestamps are split with
 * {@link splitTimestamps}.
 */
export type GPUTimeWindowFilterSettings = {
  /** Inclusive window start. */
  start: number;
  /** Inclusive window end. A window with `end < start` accepts nothing. */
  end: number;
  /** Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. */
  startFadeDuration?: number;
  /** Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. */
  endFadeDuration?: number;
};

/** High and low float32 parts of double-precision timestamps, `value ≈ high + low`. */
export type GPUSplitTimestamps = {
  /** `Math.fround(value)`. */
  high: Float32Array;
  /** `Math.fround(value - high)`. */
  low: Float32Array;
};

/**
 * Packs a time window into the 8-element float32 layout read by `GPUTimeWindowFilter`.
 *
 * Layout: `[startHigh, startLow, endHigh, endLow, startFadeDuration, endFadeDuration, 0, 0]`.
 * Start and end are split into high and low float32 parts so absolute epoch times keep sub-ulp
 * precision.
 *
 * @param settings Time window to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a split lane is not finite, an endpoint changes after packing, a fade is
 * negative or not finite in float32, a positive fade rounds to zero, or `target` is too short.
 * Exactly representable equal endpoints and reversed windows remain legal. Validation leaves `target` unchanged.
 */
export function getGPUTimeWindowFilterParameterValues(
  settings: GPUTimeWindowFilterSettings,
  target: Float32Array = new Float32Array(GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH) {
    throw new Error(
      `Time window target must hold ${GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} elements`
    );
  }
  const startFadeDuration = settings.startFadeDuration ?? 0;
  const endFadeDuration = settings.endFadeDuration ?? 0;
  for (const value of [settings.start, settings.end, startFadeDuration, endFadeDuration]) {
    if (!Number.isFinite(value)) {
      throw new Error('Time window values must be finite');
    }
  }
  if (startFadeDuration < 0 || endFadeDuration < 0) {
    throw new Error('Time window fade durations must be non-negative');
  }
  const [startHigh, startLow] = fp64ify(settings.start);
  const [endHigh, endLow] = fp64ify(settings.end);
  const packed = new Float32Array([
    startHigh,
    startLow,
    endHigh,
    endLow,
    startFadeDuration,
    endFadeDuration,
    0,
    0
  ]);
  if (!packed.subarray(0, 4).every(Number.isFinite)) {
    throw new Error('Time window split lanes must be finite in float32');
  }
  // A rounded low lane can move an inclusive edge onto a representable row outside the
  // requested window even when the start/end order survives. Reject any endpoint loss.
  if (packed[1] !== startLow || packed[3] !== endLow) {
    throw new Error('Time window endpoints must be exactly representable as two float32 lanes');
  }
  // Match the GPU's lexicographic double-single comparison, without adding away low lanes.
  const originalOrder = Math.sign(settings.end - settings.start);
  const packedOrder = Math.sign(packed[2] - packed[0]) || Math.sign(packed[3] - packed[1]);
  if (originalOrder !== packedOrder) {
    throw new Error('Time window endpoint order must survive float32 packing');
  }
  for (const [duration, rounded] of [
    [startFadeDuration, packed[4]],
    [endFadeDuration, packed[5]]
  ]) {
    if (!Number.isFinite(rounded) || (duration > 0 && !(rounded > 0))) {
      throw new Error(
        'Time window positive fade durations must remain finite and positive in float32'
      );
    }
  }
  target.set(packed);
  return target;
}

/**
 * Splits double-precision timestamps into float32 high and low parts for double-single comparison.
 *
 * `fp64ify` gives `high = Math.fround(value)` and `low = value - high`; the float32 arrays round the
 * low part. By default values are not validated; membership is exact for the encoded pairs,
 * not necessarily the original doubles. Set `requireExact` to reject the first row that cannot
 * be represented exactly by two finite float32 lanes. Use Int64 words for integer timestamps.
 */
export function splitTimestamps(
  values: ArrayLike<number>,
  options: {requireExact?: boolean} = {}
): GPUSplitTimestamps {
  const high = new Float32Array(values.length);
  const low = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const [highPart, lowPart] = fp64ify(values[index]);
    high[index] = highPart;
    low[index] = lowPart;
    if (
      options.requireExact &&
      (!Number.isFinite(high[index]) || !Number.isFinite(low[index]) || low[index] !== lowPart)
    ) {
      validateRequirement(
        'splitTimestamps',
        false,
        `values row ${index} value ${values[index]} must be exactly representable as two finite float32 lanes`
      );
    }
  }
  return {high, low};
}
