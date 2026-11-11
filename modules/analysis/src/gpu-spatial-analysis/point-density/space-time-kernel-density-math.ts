// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getComplementaryErrorFunction} from '../shared/complementary-error-function';
import {
  getGPUKernelDensityParameterValues,
  type GPUKernelDensitySettings
} from './gpu-kernel-density';
import {GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION} from './kernel-density-math';

/** Symmetric 1D kernel of the time axis of {@link GPUSpaceTimeKernelDensity}. */
export type GPUSpaceTimeTemporalKernel = 'gaussian' | 'quartic' | 'epanechnikov';

/**
 * Number of float32 elements in a {@link GPUSpaceTimeKernelDensity} parameter view. Slots 0 to 8
 * are those of `getGPUKernelDensityParameterValues`; then `[temporalBandwidth, firstSliceTime,
 * sliceInterval, timeWindowStart, timeWindowEnd]`.
 */
export const GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH = 14;

/** Per-frame settings of {@link GPUSpaceTimeKernelDensity}. */
export type GPUSpaceTimeKernelDensitySettings = GPUKernelDensitySettings & {
  /** Temporal kernel bandwidth, positive and finite, in the unit of the relative `times`. */
  temporalBandwidth: number;
  /** Time of slice 0 in the unit and origin of `times`. Slice `k` is evaluated at `firstSliceTime + k * sliceInterval`. */
  firstSliceTime: number;
  /** Time between slices, positive and finite. */
  sliceInterval: number;
  /**
   * `[start, end]` of the temporal edge correction window, `start < end`. Defaults to the first
   * and last slice time. Ignored without `temporalEdgeCorrection: 'uniform'`. With one slice the
   * default is the instant of that slice, which holds no kernel mass: under the uniform temporal
   * correction every voxel is then NaN, so give one slice an explicit window.
   */
  timeWindow?: readonly [number, number];
};

/**
 * Packs per-frame {@link GPUSpaceTimeKernelDensity} parameters.
 *
 * Layout (float32): the nine values of `getGPUKernelDensityParameterValues` (grid corners,
 * spatial bandwidth, spatial window), then `[temporalBandwidth, firstSliceTime, sliceInterval,
 * timeWindowStart, timeWindowEnd]`.
 *
 * @param settings Bandwidths, slice times and optional windows.
 * @param grid The output grid, as for `getGPUKernelDensityParameterValues`.
 * @param sliceCount Number of slices (a positive integer), to default the time window.
 * @param target Optional destination of at least {@link GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH} elements.
 * @throws If a spatial value is invalid (see `getGPUKernelDensityParameterValues`), the temporal
 * bandwidth or slice interval is not positive and finite, the first slice time is not finite,
 * `sliceCount` is not a positive integer, or an explicit time window is not finite with
 * `start < end`. A polygon `'study-window'` is not supported.
 */
export function getGPUSpaceTimeKernelDensityParameterValues(
  settings: GPUSpaceTimeKernelDensitySettings,
  grid: GPURasterMetadata,
  sliceCount: number,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUSpaceTimeKernelDensityParameterValues';
  if (target.length < GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `${name} target must hold ${GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  if (settings.window === 'study-window') {
    throw new Error(`${name} does not support a polygon study window`);
  }
  if (!Number.isFinite(settings.temporalBandwidth) || settings.temporalBandwidth <= 0) {
    throw new Error(`${name} temporalBandwidth must be positive and finite`);
  }
  if (!Number.isFinite(settings.firstSliceTime)) {
    throw new Error(`${name} firstSliceTime must be finite`);
  }
  if (!Number.isFinite(settings.sliceInterval) || settings.sliceInterval <= 0) {
    throw new Error(`${name} sliceInterval must be positive and finite`);
  }
  validateCapacity(name, 'sliceCount', sliceCount, 1, Number.MAX_SAFE_INTEGER);
  const timeWindow = settings.timeWindow ?? [
    settings.firstSliceTime,
    settings.firstSliceTime + (sliceCount - 1) * settings.sliceInterval
  ];
  // The default window of one slice is the instant of that slice, start equal to end. It is
  // packed as it is: the window is not read without the uniform temporal correction, and with it
  // an instant has no kernel mass, so every voxel is NaN. An explicit window must have a length.
  const isDefaultInstant = settings.timeWindow === undefined && sliceCount === 1;
  if (
    timeWindow.length !== 2 ||
    !timeWindow.every(Number.isFinite) ||
    !(timeWindow[0] < timeWindow[1] || isDefaultInstant)
  ) {
    throw new Error(`${name} timeWindow must be finite [start, end] with start < end`);
  }
  getGPUKernelDensityParameterValues(settings, grid, target);
  target[9] = settings.temporalBandwidth;
  target[10] = settings.firstSliceTime;
  target[11] = settings.sliceInterval;
  target[12] = timeWindow[0];
  target[13] = timeWindow[1];
  return target;
}

/**
 * Radius in time units beyond which the temporal kernel is zero: `bandwidth` for the compact
 * kernels, `5 * bandwidth` for `'gaussian'` (the truncation of the spatial gaussian).
 */
export function getGPUSpaceTimeTemporalSupportRadius(
  kernel: GPUSpaceTimeTemporalKernel,
  bandwidth: number
): number {
  return kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION * bandwidth : bandwidth;
}

/**
 * Temporal kernel `K_l(d) = K(d / l) / l` of a time offset `d`, integrating to one over the line:
 * `'gaussian'` `exp(-s^2 / 2) / sqrt(2 pi)` (zero for `s >= 5`), `'epanechnikov'`
 * `3 / 4 (1 - s^2)` and `'quartic'` (biweight) `15 / 16 (1 - s^2)^2`, both zero for `s >= 1`,
 * with `s = |d| / l` (Silverman 1986, Table 3.1 for d = 1).
 */
export function evaluateGPUSpaceTimeTemporalKernel(
  kernel: GPUSpaceTimeTemporalKernel,
  bandwidth: number,
  offset: number
): number {
  const s = Math.abs(offset) / bandwidth;
  if (kernel === 'gaussian') {
    return s < GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION
      ? Math.exp(-0.5 * s * s) / (Math.sqrt(2 * Math.PI) * bandwidth)
      : 0;
  }
  if (!(s < 1)) {
    return 0;
  }
  const complement = 1 - s * s;
  return kernel === 'epanechnikov'
    ? (0.75 * complement) / bandwidth
    : (0.9375 * complement * complement) / bandwidth;
}

/** Cumulative distribution of the standardised temporal kernel at `s` (untruncated gaussian). */
function getTemporalKernelCumulative(kernel: GPUSpaceTimeTemporalKernel, s: number): number {
  if (kernel === 'gaussian') {
    return 0.5 * getComplementaryErrorFunction(-s * Math.SQRT1_2);
  }
  const clamped = Math.max(-1, Math.min(1, s));
  return kernel === 'epanechnikov'
    ? 0.5 + 0.75 * (clamped - clamped ** 3 / 3)
    : 0.5 + 0.9375 * (clamped - (2 * clamped ** 3) / 3 + clamped ** 5 / 5);
}

/**
 * Temporal kernel mass inside a window, `e(t) = integral from start to end of K_l(t - v) dv`, the
 * quantity the `'uniform'` temporal edge correction divides by. It is 1 when the whole kernel lies
 * inside the window. This is the float64 reference of the WGSL used on the GPU.
 */
export function getGPUSpaceTimeTemporalWindowMass(
  kernel: GPUSpaceTimeTemporalKernel,
  bandwidth: number,
  time: number,
  window: readonly [number, number]
): number {
  return (
    getTemporalKernelCumulative(kernel, (window[1] - time) / bandwidth) -
    getTemporalKernelCumulative(kernel, (window[0] - time) / bandwidth)
  );
}
