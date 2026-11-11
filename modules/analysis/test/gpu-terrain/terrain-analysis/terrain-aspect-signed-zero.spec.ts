// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainDerivatives} from './terrain-analysis-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const WIDTH = 5;
const HEIGHT = 5;
const PIXEL_COUNT = WIDTH * HEIGHT;
/** A grid whose affine states the same row direction as the `rowDirection` prop. */
const GRIDS = {
  south: makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10}),
  north: makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10, rowDirection: 'north'})
};

/** Planes named by the compass direction they descend towards, with the expected aspect. */
const PLANES: {name: string; east: number; north: number; aspect: number}[] = [
  {name: 'east', east: -1, north: 0, aspect: 90},
  {name: 'west', east: 1, north: 0, aspect: 270},
  {name: 'north', east: 0, north: -1, aspect: 0},
  {name: 'south', east: 0, north: 1, aspect: 180},
  {name: 'north-east', east: -1, north: -1, aspect: 45},
  {name: 'south-east', east: -1, north: 1, aspect: 135},
  {name: 'south-west', east: 1, north: 1, aspect: 225},
  {name: 'north-west', east: 1, north: -1, aspect: 315}
];

function getPlane(east: number, north: number, rowDirection: 'south' | 'north'): Float32Array {
  const rowNorthSign = rowDirection === 'south' ? -1 : 1;
  return Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const column = index % WIDTH;
    const row = Math.floor(index / WIDTH);
    return 100 + 3 * east * column + 3 * north * rowNorthSign * row;
  });
}

function getAngularDifference(left: number, right: number): number {
  const difference = Math.abs(left - right) % 360;
  return Math.min(difference, 360 - difference);
}

// FX-17: a plane descending purely east or west has a north gradient computed from equal values,
// so with one row direction `-northGradient` is -0.0 and the builtin `atan2(y, -0.0)` on the test
// GPU returns the opposite half-turn. Compare every interior pixel with the binary64 oracle and
// the compass value, for both row directions and the four diagonals.
it('GPUTerrainDerivatives aspect is correct for pure east, west and diagonal planes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const failures: string[] = [];
  for (const rowDirection of ['south', 'north'] as const) {
    const grid = GRIDS[rowDirection];
    const settingsValues = getGPUTerrainDerivativesParameterValues({}, grid);
    for (const plane of PLANES) {
      const elevation = getPlane(plane.east, plane.north, rowDirection);
      const graph = new GPUCommandGraph(device, {id: 'terrain-aspect-signed-zero'});
      const elevationBuffer = createInputBuffer(device, elevation);
      const aspectBuffer = createOutputBuffer(device, PIXEL_COUNT);
      const settings = new GPUParameterBuffer(device, {
        id: 'aspect-settings',
        format: 'float32',
        length: 8,
        values: settingsValues
      });
      graph.add(
        new GPUTerrainDerivatives({
          grid,
          rowDirection,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
            }
          },
          parameters: settings.importToGraph(graph),
          output: {
            aspect: importGraphBuffer(graph, 'aspect', aspectBuffer, 'float32', PIXEL_COUNT)
          }
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const aspect = await readFloat32(aspectBuffer, PIXEL_COUNT);
      compiled.destroy();
      settings.destroy();
      elevationBuffer.destroy();
      aspectBuffer.destroy();
      const oracle = computeTerrainDerivatives(
        elevation,
        undefined,
        WIDTH,
        HEIGHT,
        settingsValues,
        {rowDirection}
      );
      for (let row = 1; row < HEIGHT - 1; row++) {
        for (let column = 1; column < WIDTH - 1; column++) {
          const index = row * WIDTH + column;
          const reference = oracle.aspect[index];
          if (
            getAngularDifference(reference, plane.aspect) > 1e-9 ||
            getAngularDifference(aspect[index], reference) > 1e-3
          ) {
            failures.push(
              `${rowDirection} ${plane.name} pixel ${index}: GPU ${aspect[index]}, CPU ${reference}`
            );
          }
        }
      }
    }
  }
  expect(failures).toEqual([]);
});
