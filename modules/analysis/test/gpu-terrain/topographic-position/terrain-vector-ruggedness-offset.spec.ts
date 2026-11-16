// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUTerrainVectorRuggedness,
  getGPUTerrainVectorRuggednessParameterValues
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';
import {computeVectorRuggedness} from './terrain-ruggedness-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

it('preserves local Horn normals on a curved surface at a million-unit elevation offset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const width = 7;
  const height = 7;
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, -1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const residuals = Float32Array.from(
    {length: width * height},
    (_, cell) => ((cell % width) ** 2 + Math.floor(cell / width) ** 2) / 16
  );
  const oracle = computeVectorRuggedness(
    residuals,
    undefined,
    width,
    height,
    getGPUTerrainVectorRuggednessParameterValues({radius: 1}, grid),
    {radius: 1, borderMode: 'clamp', rowDirection: 'south'}
  );
  const outputs = [0, 1000000].map(offset => {
    const result = fixture.output('float32', residuals.length);
    fixture.graph.add(
      new GPUTerrainVectorRuggedness({
        id: `offset-${offset}`,
        grid,
        elevation: {
          id: `elevation-${offset}`,
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input(
              'float32',
              Float32Array.from(residuals, value => value + offset)
            ).view
          }
        },
        parameters: fixture.input(
          'float32',
          getGPUTerrainVectorRuggednessParameterValues({radius: 1}, grid)
        ).view,
        output: {vectorRuggedness: result.view}
      })
    );
    return result;
  });
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    compiled = fixture.run();
    for (const result of outputs) {
      const actual = await result.read();
      for (let row = 1; row < height - 1; row++)
        for (let column = 1; column < width - 1; column++) {
          const cell = row * width + column;
          expect(Math.abs(actual[cell] - oracle.vectorRuggedness[cell])).toBeLessThan(2e-6);
        }
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

for (const [cellSize, elevationScale, zFactor] of [
  [1e38, 1e38, 1],
  [1e-30, 1e-36, 1e38]
]) {
  it(`preserves Horn normals at extreme finite cell size ${cellSize}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const width = 7;
    const height = 7;
    const grid = {
      width,
      height,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const elevation = Float32Array.from(
      {length: width * height},
      (_, cell) =>
        ((((cell % width) - 3) ** 2 + (Math.floor(cell / width) - 3) ** 2) / 64) * elevationScale
    );
    const parameters = getGPUTerrainVectorRuggednessParameterValues({radius: 1}, grid);
    // Uniform cell sizes and zFactor are runtime settings; latitude slots are unused here.
    parameters[0] = cellSize;
    parameters[1] = cellSize;
    parameters[2] = zFactor;
    const oracle = computeVectorRuggedness(elevation, undefined, width, height, parameters, {
      radius: 1,
      borderMode: 'clamp',
      rowDirection: 'south'
    });
    const output = fixture.output('float32', elevation.length);
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      fixture.graph.add(
        new GPUTerrainVectorRuggedness({
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {kind: 'buffer', values: fixture.input('float32', elevation).view}
          },
          parameters: fixture.input('float32', parameters).view,
          output: {vectorRuggedness: output.view}
        })
      );
      compiled = fixture.run();
      const actual = await output.read();
      for (let cell = 0; cell < actual.length; cell++) {
        expect(Number.isFinite(oracle.vectorRuggedness[cell])).toBe(true);
        expect(Math.abs(actual[cell] - oracle.vectorRuggedness[cell])).toBeLessThan(2e-6);
      }
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}
