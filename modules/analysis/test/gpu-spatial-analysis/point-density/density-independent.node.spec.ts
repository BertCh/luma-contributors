// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPU_GRID_COORDINATE_WGSL
} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUAdaptiveBandwidthKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getGPUKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {evaluateGPUKernelDensityKernel} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {TRANSLATED_DENSITY_FIXTURE as translatedFixture} from './density-translated-fixture';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {expect, it} from 'vitest';
import {getSpaceTimeDensityReference} from './space-time-kernel-density-reference';
import {SPACE_TIME_INDEPENDENT_VALUES} from './density-independent-fixtures';

it('PTS-140 pins independent product-kernel interior and temporal boundary values', () => {
  const result = getSpaceTimeDensityReference({
    scene: {positions: [0, 0, 0, 0], times: [0.5, 1.5]},
    columns: 1,
    rows: 1,
    extent: [0, -1, 2, 1],
    bandwidth: 2,
    temporalBandwidth: 2,
    firstSliceTime: 0,
    sliceInterval: 2,
    sliceCount: 2,
    timeWindow: [0, 4],
    spatialKernel: 'epanechnikov',
    temporalKernel: 'epanechnikov',
    spatialEdge: false,
    temporalEdge: true,
    normalization: 'intensity'
  });
  expect(result.values).toHaveLength(SPACE_TIME_INDEPENDENT_VALUES.length);
  result.values.forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(SPACE_TIME_INDEPENDENT_VALUES[index], 10);
  });
});

it('PTS-139 packs translated cells and pins compact-kernel values at represented distances', () => {
  const originX = 500000;
  const originY = 5000000;
  const {width, height, point, bandwidth} = translatedFixture;
  const grid = makeExtentGrid(width, height, [originX, originY, originX + width, originY + height]);
  const parameters = getGPUKernelDensityParameterValues({bandwidth}, grid);
  expect(Array.from(parameters.slice(0, 5))).toEqual([500000, 5000000, 500012, 5000008, 1.5]);
  const values: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const centerX = Math.fround(parameters[0] + column + 0.5);
      const centerY = Math.fround(parameters[1] + row + 0.5);
      values.push(
        evaluateGPUKernelDensityKernel(
          'epanechnikov',
          bandwidth,
          Math.hypot(centerX - (originX + point[0]), centerY - (originY + point[1]))
        )
      );
    }
  }
  expect(values).toHaveLength(translatedFixture.values.length);
  values.forEach((value, cell) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(value, `cell ${cell}`).toBeCloseTo(translatedFixture.values[cell], 14);
  });
  expect(values.reduce((sum, value) => sum + value, 0)).toBeCloseTo(translatedFixture.mass, 14);
});

it.each([
  'adaptive',
  'space-time'
] as const)('shared density helpers preserve parseable %s kernels and the published grid mapping', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const common = {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', kind === 'adaptive' ? 10 : 14),
    grid: makeExtentGrid(3, 1, [-1, -0.5, 5, 0.5]),
    indexBounds: [-1, -1, 5, 1] as const,
    indexGridSize: [6, 2] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 3)}
  };
  const density =
    kind === 'adaptive'
      ? new GPUAdaptiveBandwidthKernelDensity(common)
      : new GPUSpaceTimeKernelDensity({
          ...common,
          sliceCount: 1,
          times: createTransientView(graph, 'times', 'float32', 4)
        });
  const sources = getGPUCommandNodeKernelSources(density.getCommandNodes(graph));
  expect(sources.some(kernel => kernel.source.includes(GPU_GRID_COORDINATE_WGSL))).toBe(true);
  for (const kernel of sources)
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
  device.destroy();
});
