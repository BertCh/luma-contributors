// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUKernelDensityMethod} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {getBandwidthOracle} from './kernel-density-bandwidth-oracle';
import {getBinnedDensityModel, getExactDensityModel} from './kernel-density-binned-oracle';

/**
 * Independent SciPy normal-CDF integral for this fixture at every cell center:
 * prod(ndtr((2 - center) / h) - ndtr(-center / h)), h = 0.4638694609608677.
 * Symmetry makes the rectangular Gaussian mass identical at all four centers.
 */
const SHARED_OUTPUT_GAUSSIAN_WINDOW_MASS = 0.7376188520917182;

/** Shares the GPU fixture and its explicitly matched edge-correction oracle with node checks. */
export function getSharedOutputDensityFixture(
  method: GPUKernelDensityMethod,
  edgeCorrection?: 'none' | 'uniform'
) {
  const points = [0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5, Number.NaN, 0.5];
  const weights = [1, 2, 3, 4, 5];
  const grid = makeExtentGrid(2, 2, [0, 0, 2, 2]);
  const bandwidth = getBandwidthOracle(points, weights).isotropic;
  const input = {
    points,
    weights,
    width: 2,
    height: 2,
    start: [0, 0] as const,
    cellSize: [1, 1] as const,
    bandwidth,
    padding: 8
  };
  // Here 5 h < 32 cells, so 'auto' selects exact gather even with sufficient padding.
  const intensity =
    method === 'binned' ? getBinnedDensityModel(input) : getExactDensityModel(input);
  const resolvedEdgeCorrection = edgeCorrection ?? 'uniform';
  const mass = resolvedEdgeCorrection === 'uniform' ? SHARED_OUTPUT_GAUSSIAN_WINDOW_MASS : 1;
  const weightSum = 10; // The fifth row has a non-finite position and contributes no weight.
  return {
    points,
    weights,
    grid,
    bandwidth,
    weightSum,
    resolvedEdgeCorrection,
    settings: {method, edgeCorrection},
    expectedValues: Array.from(intensity, value => value / (weightSum * mass))
  };
}
