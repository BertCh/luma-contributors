// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getNormalIntervalMass} from '../shared/complementary-error-function';
import {resolveKernelShapeName} from '../shared/kernel-shape-name';
import {
  getGPUKernelDensityParameterValues,
  type GPUKernelDensitySettings
} from './gpu-kernel-density';
import {createDensitySliceTime, findDensityRepeatedSlice} from './density-slice-times';
import {GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION} from './kernel-density-math';

/** Symmetric 1D kernel of the time axis of {@link GPUSpaceTimeKernelDensity}. */
export type GPUSpaceTimeTemporalKernel = 'gaussian' | 'quartic' | 'epanechnikov';
/** Accepted temporal shape names; bisquare is the quartic alias. */
export type GPUSpaceTimeTemporalKernelName = GPUSpaceTimeTemporalKernel | 'bisquare';

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
  /** Time of slice 0 in the unit and origin of `times`. Slice `k` uses one float32 fused multiply-add, `fma(float32(k), sliceInterval, firstSliceTime)`. */
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
  validateRequirement(
    name,
    target.length >= GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH,
    `target must hold ${GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH} elements; received ${target.length}`
  );
  validateRequirement(
    name,
    settings.window !== 'study-window',
    'window does not support a polygon study window'
  );
  const temporalBandwidth = Math.fround(settings.temporalBandwidth);
  const first = Math.fround(settings.firstSliceTime);
  const interval = Math.fround(settings.sliceInterval);
  validateRequirement(
    name,
    Number.isFinite(temporalBandwidth) && temporalBandwidth > 0,
    `temporalBandwidth must be positive and finite in float32; received ${settings.temporalBandwidth} encoded as ${temporalBandwidth}`
  );
  validateRequirement(
    name,
    Number.isFinite(first),
    `firstSliceTime must be finite in float32; received ${settings.firstSliceTime}`
  );
  validateRequirement(
    name,
    Number.isFinite(interval) && interval > 0,
    `sliceInterval must be positive and finite in float32; received ${settings.sliceInterval}`
  );
  validateCapacity(name, 'sliceCount', sliceCount, 1, Number.MAX_SAFE_INTEGER);
  const getSliceTime = createDensitySliceTime(first, interval);
  const repeated = findDensityRepeatedSlice(getSliceTime, interval, sliceCount);
  if (repeated !== undefined) {
    validateRequirement(
      name,
      false,
      `slice ${repeated} time must advance in float32; received ${getSliceTime(repeated - 1)}, ${getSliceTime(repeated)} from firstSliceTime ${first} and sliceInterval ${interval}; move the time origin closer to the observations`
    );
  }
  const last = getSliceTime(sliceCount - 1);
  const timeWindow = settings.timeWindow
    ? Array.from(settings.timeWindow, Math.fround)
    : [first, last];
  const isDefaultInstant = settings.timeWindow === undefined && sliceCount === 1;
  validateRequirement(
    name,
    timeWindow.length === 2 &&
      timeWindow.every(Number.isFinite) &&
      (timeWindow[0] < timeWindow[1] || isDefaultInstant),
    `timeWindow must be finite [start, end] with start < end in float32; received ${settings.timeWindow} encoded as ${timeWindow}; move the time origin closer to the observations`
  );
  // Validate the temporal controls before the spatial packer publishes anything.
  getGPUKernelDensityParameterValues(settings, grid, target);
  target.set([temporalBandwidth, first, interval, timeWindow[0], timeWindow[1]], 9);
  return target;
}

/**
 * Radius in time units beyond which the temporal kernel is zero: `bandwidth` for the compact
 * kernels, `5 * bandwidth` for `'gaussian'` (the truncation of the spatial gaussian).
 */
export function getGPUSpaceTimeTemporalSupportRadius(
  kernel: GPUSpaceTimeTemporalKernelName,
  bandwidth: number
): number {
  kernel = resolveKernelShapeName('getGPUSpaceTimeTemporalSupportRadius', kernel, [
    'gaussian',
    'quartic',
    'epanechnikov'
  ]);
  return kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION * bandwidth : bandwidth;
}

/**
 * Temporal kernel `K_l(d) = K(d / l) / l` of a time offset `d`, integrating to one over the line:
 * `'gaussian'` `exp(-s^2 / 2) / sqrt(2 pi)` (zero for `s >= 5`), `'epanechnikov'`
 * `3 / 4 (1 - s^2)` and `'quartic'` (biweight) `15 / 16 (1 - s^2)^2`, both zero for `s >= 1`,
 * with `s = |d| / l` (Silverman 1986, Table 3.1 for d = 1).
 */
export function evaluateGPUSpaceTimeTemporalKernel(
  kernel: GPUSpaceTimeTemporalKernelName,
  bandwidth: number,
  offset: number
): number {
  kernel = resolveKernelShapeName('evaluateGPUSpaceTimeTemporalKernel', kernel, [
    'gaussian',
    'quartic',
    'epanechnikov'
  ]);
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

/** Stable compact-kernel survival mass, in remaining support width. */
function getTemporalCompactTail(kernel: GPUSpaceTimeTemporalKernelName, value: number): number {
  const remaining = Math.max(0, 1 - Math.min(1, value));
  return kernel === 'epanechnikov'
    ? 0.25 * remaining ** 2 * (3 - remaining)
    : 0.0625 * remaining ** 3 * (20 - 15 * remaining + 3 * remaining ** 2);
}

/**
 * Temporal kernel mass inside a window, `e(t) = integral from start to end of K_l(t - v) dv`, the
 * quantity the `'uniform'` temporal edge correction divides by. It is 1 when the whole kernel lies
 * inside the window. This is the float64 reference of the WGSL used on the GPU.
 */
export function getGPUSpaceTimeTemporalWindowOverlap(
  kernel: GPUSpaceTimeTemporalKernelName,
  bandwidth: number,
  time: number,
  window: readonly [number, number]
): number {
  kernel = resolveKernelShapeName('getGPUSpaceTimeTemporalWindowOverlap', kernel, [
    'gaussian',
    'quartic',
    'epanechnikov'
  ]);
  const low = (window[0] - time) / bandwidth;
  const high = (window[1] - time) / bandwidth;
  if (kernel === 'gaussian') return getNormalIntervalMass(low, high);
  if (low >= -0.5 && high <= 0.5) {
    const width = (window[1] - window[0]) / bandwidth;
    const midpoint = low + width / 2;
    const squared = midpoint * midpoint;
    return kernel === 'epanechnikov'
      ? 0.75 * width * (1 - squared - (width * width) / 12)
      : 0.9375 *
          width *
          ((1 - squared) ** 2 + ((6 * squared - 2) * width * width) / 12 + width ** 4 / 80);
  }
  if (low >= 0) return getTemporalCompactTail(kernel, low) - getTemporalCompactTail(kernel, high);
  if (high <= 0)
    return getTemporalCompactTail(kernel, -high) - getTemporalCompactTail(kernel, -low);
  return 1 - getTemporalCompactTail(kernel, -low) - getTemporalCompactTail(kernel, high);
}
