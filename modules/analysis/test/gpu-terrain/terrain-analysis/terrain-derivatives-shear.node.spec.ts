// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

for (const scale of [0, 1e-200, 1e200]) {
  for (const entryPoint of ['constructor', 'packer'] as const) {
    it(`R5 rejects degenerate or unrepresentable axes at ${entryPoint}, scale ${scale}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const grid = {
        ...makeGPURasterMetadata({width: 3, height: 3}),
        affine: [scale, scale, 0, 0, -scale, 0] as const
      };
      try {
        expect(() =>
          entryPoint === 'packer'
            ? getGPUTerrainDerivativesParameterValues({}, grid)
            : new GPUTerrainDerivatives({
                id: 'rescaled-shear',
                grid,
                elevation: createBand(graph, 'elevation', 9),
                parameters: createTransientView(graph, 'parameters', 'float32', 8),
                output: {slope: createTransientView(graph, 'slope', 'float32', 9)}
              })
        ).toThrow(/^(rescaled-shear|getGPUTerrainDerivativesParameterValues) grid.affine/);
      } finally {
        device.destroy();
      }
    });
  }
}

for (const entryPoint of ['constructor', 'packer'] as const) {
  it(`rejects the world-X plane's sheared grid at the ${entryPoint}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const grid = {
      ...makeGPURasterMetadata({width: 3, height: 3}),
      affine: [1, 1, 0, 0, -1, 0] as const
    };
    try {
      expect(() =>
        entryPoint === 'packer'
          ? getGPUTerrainDerivativesParameterValues({}, grid)
          : new GPUTerrainDerivatives({
              id: 'shear-test',
              grid,
              elevation: createBand(graph, 'elevation', 9),
              parameters: createTransientView(graph, 'parameters', 'float32', 8),
              output: {slope: createTransientView(graph, 'slope', 'float32', 9)}
            })
      ).toThrow(/^(shear-test|getGPUTerrainDerivativesParameterValues) grid.affine.*orthogonal/);
    } finally {
      device.destroy();
    }
  });
}

it('accepts rotated orthogonal axes at both entry points', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    ...makeGPURasterMetadata({width: 3, height: 3}),
    affine: [2, 1, 0, 1, -2, 0] as const
  };
  try {
    const packed = getGPUTerrainDerivativesParameterValues({}, grid);
    expect(packed[0]).toBeCloseTo(Math.sqrt(5));
    expect(packed[1]).toBeCloseTo(Math.sqrt(5));
    expect(
      () =>
        new GPUTerrainDerivatives({
          grid,
          elevation: createBand(graph, 'elevation', 9),
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          output: {slope: createTransientView(graph, 'slope', 'float32', 9)}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
