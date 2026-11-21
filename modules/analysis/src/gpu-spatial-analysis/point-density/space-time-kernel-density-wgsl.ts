// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {DENSITY_NORMAL_INTERVAL_WGSL} from './density-normal-interval-wgsl';
import {GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION} from './kernel-density-math';
import type {GPUSpaceTimeTemporalKernel} from './space-time-kernel-density-math';

/**
 * WGSL `getTemporalKernelValue(offset, bandwidth)`: the float32 mirror of
 * {@link evaluateGPUSpaceTimeTemporalKernel}, zero beyond its support.
 * @internal
 */
export function getTemporalKernelValueWGSL(kernel: GPUSpaceTimeTemporalKernel): string {
  if (kernel === 'gaussian') {
    const scale = getWGSLFloatLiteral(1 / Math.sqrt(2 * Math.PI));
    const truncation = getWGSLFloatLiteral(GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION ** 2);
    return /* wgsl */ `
fn getTemporalKernelValue(offset: f32, bandwidth: f32) -> f32 {
  let s = offset / bandwidth;
  let t = s * s;
  if (!(t < ${truncation})) { return 0.0; }
  return ${scale} * exp(-0.5 * t) / bandwidth;
}`;
  }
  const scale = getWGSLFloatLiteral(kernel === 'epanechnikov' ? 0.75 : 0.9375);
  return /* wgsl */ `
fn getTemporalKernelValue(offset: f32, bandwidth: f32) -> f32 {
  let s = offset / bandwidth;
  let t = s * s;
  if (!(t < 1.0)) { return 0.0; }
  let complement = 1.0 - t;
  return ${scale} * ${kernel === 'epanechnikov' ? 'complement' : 'complement * complement'} / bandwidth;
}`;
}

/**
 * WGSL `getTemporalWindowMass(time, windowStart, windowEnd, bandwidth)`: the float32 mirror of
 * {@link getGPUSpaceTimeTemporalWindowOverlap}. `includeErfc` is false when the module already holds
 * `ERFC_VALUE_WGSL` (the spatial window mass of a gaussian kernel declares it).
 * @internal
 */
export function getTemporalWindowMassWGSL(
  kernel: GPUSpaceTimeTemporalKernel,
  includeErfc: boolean
): string {
  if (kernel === 'gaussian') {
    return /* wgsl */ `
${includeErfc ? DENSITY_NORMAL_INTERVAL_WGSL : ''}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  return getDensityNormalIntervalMassWithWidth((windowStart - time) / bandwidth, (windowEnd - time) / bandwidth, (windowEnd - windowStart) / bandwidth);
}`;
  }
  const tail =
    kernel === 'epanechnikov'
      ? '0.25 * remaining * remaining * (3.0 - remaining)'
      : '0.0625 * remaining * remaining * remaining * (20.0 - 15.0 * remaining + 3.0 * remaining * remaining)';
  return /* wgsl */ `
fn getTemporalCompactTail(value: f32) -> f32 {
  let remaining = max(0.0, 1.0 - min(1.0, value));
  return ${tail};
}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  let low = (windowStart - time) / bandwidth;
  let high = (windowEnd - time) / bandwidth;
  if (low >= -0.5 && high <= 0.5) {
    let width = (windowEnd - windowStart) / bandwidth;
    let midpoint = low + 0.5 * width;
    let squared = midpoint * midpoint;
    ${kernel === 'epanechnikov' ? 'return 0.75 * width * (1.0 - squared - width * width / 12.0);' : 'return 0.9375 * width * ((1.0 - squared) * (1.0 - squared) + (6.0 * squared - 2.0) * width * width / 12.0 + width * width * width * width / 80.0);'}
  }
  if (low >= 0.0) { return getTemporalCompactTail(low) - getTemporalCompactTail(high); }
  if (high <= 0.0) { return getTemporalCompactTail(-high) - getTemporalCompactTail(-low); }
  return 1.0 - getTemporalCompactTail(-low) - getTemporalCompactTail(high);
}`;
}
