// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {ERFC_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
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
 * {@link getGPUSpaceTimeTemporalWindowMass}. `includeErfc` is false when the module already holds
 * `ERFC_VALUE_WGSL` (the spatial window mass of a gaussian kernel declares it).
 * @internal
 */
export function getTemporalWindowMassWGSL(
  kernel: GPUSpaceTimeTemporalKernel,
  includeErfc: boolean
): string {
  if (kernel === 'gaussian') {
    return /* wgsl */ `
${includeErfc ? ERFC_VALUE_WGSL : ''}
fn getTemporalUpperTail(z: f32) -> f32 { return 0.5 * erfcValue(z * 0.70710678); }
fn getTemporalNormalMass(low: f32, high: f32) -> f32 {
  if (low > 0.0) { return getTemporalUpperTail(low) - getTemporalUpperTail(high); }
  if (high < 0.0) { return getTemporalUpperTail(-high) - getTemporalUpperTail(-low); }
  return 1.0 - getTemporalUpperTail(-low) - getTemporalUpperTail(high);
}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  return getTemporalNormalMass((windowStart - time) / bandwidth, (windowEnd - time) / bandwidth);
}`;
  }
  const cumulative =
    kernel === 'epanechnikov'
      ? '0.5 + 0.75 * (c - c * c * c / 3.0)'
      : '0.5 + 0.9375 * (c - 2.0 * c * c * c / 3.0 + c * c * c * c * c / 5.0)';
  return /* wgsl */ `
fn getTemporalCumulative(s: f32) -> f32 {
  let c = clamp(s, -1.0, 1.0);
  return ${cumulative};
}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  return getTemporalCumulative((windowEnd - time) / bandwidth) -
    getTemporalCumulative((windowStart - time) / bandwidth);
}`;
}
