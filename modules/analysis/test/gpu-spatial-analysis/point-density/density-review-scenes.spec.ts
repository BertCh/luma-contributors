// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {TRANSLATED_DENSITY_FIXTURE as translatedFixture} from './density-translated-fixture';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

it('PTS-131 diagnoses below, at and above three cells on both unequal axes on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(4, 3, [0, 0, 4, 6]); // cells 1 x 2
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {positions: {format: 'float32x2', values: new Float32Array([1, 2])}},
    outputs: {values: {format: 'float32', length: 12}, resolution: {format: 'uint32', length: 1}},
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid,
        indexBounds: [0, 0, 4, 6],
        indexGridSize: [2, 2],
        method: 'binned',
        binnedPadding: 32,
        edgeCorrection: 'none',
        output: {
          values: outputs['values'] as never,
          status: {bandwidthBelowThreeCells: outputs['resolution'] as never}
        }
      })
  });
  try {
    for (const [bandwidth, expected] of [
      [2.5, 1],
      [3, 1],
      [5.5, 1],
      [6, 0],
      [6.5, 0],
      [2.5, 1]
    ]) {
      const result = await fixture.run(getGPUKernelDensityParameterValues({bandwidth}, grid));
      expect(result['resolution']).toEqual([expected]);
      expect(result['values'].every(Number.isFinite)).toBe(true);
    }
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('PTS-139 places fixed KDE query cells at a projected origin with literal density and mass', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // All endpoints, positions and centers are exactly representable as float32, even at 5e6.
  for (const [originX, originY] of [
    [0, 0],
    [500000, 5000000]
  ]) {
    const {width, height} = translatedFixture;
    const extent = [originX, originY, originX + width, originY + height] as const;
    const grid = makeExtentGrid(width, height, extent);
    const fixture = createGeometryFixture(device, {
      parameterLength: 9,
      inputs: {
        positions: {
          format: 'float32x2',
          values: new Float32Array([
            originX + translatedFixture.point[0],
            originY + translatedFixture.point[1]
          ])
        }
      },
      outputs: {
        values: {format: 'float32', length: width * height},
        weightSum: {format: 'float32', length: 1}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUKernelDensity({
          positions: inputs['positions'] as never,
          parameters,
          grid,
          indexBounds: extent,
          indexGridSize: [width, height],
          kernel: 'epanechnikov',
          edgeCorrection: 'none',
          output: {values: outputs['values'] as never, weightSum: outputs['weightSum'] as never}
        })
    });
    try {
      const result = await fixture.run(
        getGPUKernelDensityParameterValues({bandwidth: translatedFixture.bandwidth}, grid)
      );
      expect(result['values']).toHaveLength(translatedFixture.values.length);
      result['values'].forEach((value, cell) => {
        expect(Number.isFinite(value), `cell ${cell}`).toBe(true);
        expect(value, `cell ${cell}`).toBeCloseTo(translatedFixture.values[cell], 6);
      });
      expect(result['values'].reduce((sum, value) => sum + value, 0)).toBeCloseTo(
        translatedFixture.mass,
        6
      );
      expect(result['weightSum']).toEqual([1]);
    } finally {
      fixture.destroy();
    }
  }
});

it('PTS-131 diagnoses the GPU-selected effective bandwidth and recovers on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(4, 3, [-2, -3, 2, 3]);
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2])}
    },
    outputs: {
      values: {format: 'float32', length: 12},
      bandwidth: {format: 'float32', length: 1},
      resolution: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid,
        indexBounds: [-50, -50, 50, 50],
        indexGridSize: [2, 2],
        bandwidthRule: 'scott',
        method: 'binned',
        binnedPadding: 64,
        edgeCorrection: 'none',
        output: {
          values: outputs['values'] as never,
          bandwidth: outputs['bandwidth'] as never,
          status: {bandwidthBelowThreeCells: outputs['resolution'] as never}
        }
      })
  });
  try {
    for (const [scale, expected] of [
      [2, 1],
      [8, 0],
      [2, 1]
    ]) {
      fixture.writeInput(
        'positions',
        new Float32Array([-scale, -scale, scale, -scale, -scale, scale, scale, scale])
      );
      const result = await fixture.run(getGPUKernelDensityParameterValues({bandwidth: 99}, grid));
      // Four corners have unbiased variance 4*a²/3 on each axis; Scott h=4^(-1/6)*sqrt(4*a²/3).
      expect(Number.isFinite(result['bandwidth'][0])).toBe(true);
      expect(result['bandwidth'][0]).toBeCloseTo(0.916486424665735 * scale, 4);
      expect(result['resolution']).toEqual([expected]);
    }
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});
