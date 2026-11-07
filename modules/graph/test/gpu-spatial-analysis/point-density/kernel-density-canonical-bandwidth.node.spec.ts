// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  evaluateGPUKernelDensityKernel,
  getGPUKernelDensityCanonicalBandwidthScale,
  type GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';

/** Numerical `R(K) = integral K^2` and per-axis `mu2 = integral x^2 K` of a unit-bandwidth kernel. */
function getKernelMoments(kernel: GPUKernelDensityKernel): {
  roughness: number;
  secondMoment: number;
} {
  const extent = kernel === 'gaussian' ? 10 : 1;
  const steps = 200000;
  const step = extent / steps;
  let roughness = 0;
  let secondMoment = 0;
  for (let index = 0; index < steps; index++) {
    const radius = (index + 0.5) * step;
    const value = evaluateGPUKernelDensityKernel(kernel, 1, radius);
    roughness += value * value * 2 * Math.PI * radius * step;
    secondMoment += 0.5 * radius * radius * value * 2 * Math.PI * radius * step;
  }
  return {roughness, secondMoment};
}

it('the canonical bandwidth scale equalizes smoothing across kernels (Gaussian-scale rule input)', () => {
  const canonicalBandwidth = (kernel: GPUKernelDensityKernel) => {
    const {roughness, secondMoment} = getKernelMoments(kernel);
    return (roughness / secondMoment ** 2) ** (1 / 6);
  };
  const gaussian = canonicalBandwidth('gaussian');
  expect(getGPUKernelDensityCanonicalBandwidthScale('gaussian')).toBe(1);
  for (const kernel of ['epanechnikov', 'quartic'] as const) {
    expect(getGPUKernelDensityCanonicalBandwidthScale(kernel)).toBeCloseTo(
      canonicalBandwidth(kernel) / gaussian,
      3
    );
    // Compact kernels need a larger radius than the Gaussian sigma for equal smoothing.
    expect(getGPUKernelDensityCanonicalBandwidthScale(kernel)).toBeGreaterThan(2);
  }
});
