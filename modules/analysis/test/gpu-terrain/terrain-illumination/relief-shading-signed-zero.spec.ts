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
  getGPUReliefShadingParameterValues,
  GPUReliefShading,
  type GPUReliefShadingSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeReliefShading} from './relief-shading-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const WIDTH = 7;
const HEIGHT = 7;
const PIXEL_COUNT = WIDTH * HEIGHT;
/** A grid whose affine states the same row direction as the `rowDirection` prop. */
const GRIDS = {
  south: makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10}),
  north: makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10, rowDirection: 'north'})
};
const SHADE_TOLERANCE = 1e-5;

/** Planes named by the compass direction they descend towards; `flat` has no slope at all. */
const PLANES: {name: string; east: number; north: number}[] = [
  {name: 'east', east: -1, north: 0},
  {name: 'west', east: 1, north: 0},
  {name: 'north', east: 0, north: -1},
  {name: 'south', east: 0, north: 1},
  {name: 'north-east', east: -1, north: -1},
  {name: 'south-east', east: -1, north: 1},
  {name: 'south-west', east: 1, north: 1},
  {name: 'north-west', east: 1, north: -1},
  {name: 'flat', east: 0, north: 0}
];

const STYLES: {name: string; settings: GPUReliefShadingSettings}[] = [
  {name: 'aspect', settings: {lights: 'mdow', lightWeighting: 'aspect'}},
  {name: 'imhof-swing', settings: {lights: 'mdow', lightWeighting: 'imhof-swing'}}
];

function getPlane(east: number, north: number, rowDirection: 'south' | 'north'): Float32Array {
  const rowNorthSign = rowDirection === 'south' ? -1 : 1;
  // 20 m per 10 m pixel: steep enough for the full Imhof swing factor.
  return Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const column = index % WIDTH;
    const row = Math.floor(index / WIDTH);
    return 500 + 20 * east * column + 20 * north * rowNorthSign * row;
  });
}

// FX-17 and math review terrain-surface-05. The relief aspect `atan2(-eastGradient,
// -northGradient)` gets a -0.0 second argument on pure east and west slopes with north row
// order, and both arguments zero on a flat cell, where WGSL leaves atan2 undefined (a probe of
// atan2(0, 0) from buffer values returns NaN on the test GPU; inside this kernel it happened to
// return a finite value before the fix). The
// aspect weighting squares sin(aspect - azimuth), so a half-turn error cancels there; the Imhof
// swing does not. A flat cell has no aspect: its swing factor is zero, so every light keeps its
// base azimuth (the binary64 oracle gives exactly that), and its shade must be finite.
it('GPUReliefShading aspect and Imhof swing match the oracle on axis, diagonal and flat planes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const failures: string[] = [];
  for (const rowDirection of ['south', 'north'] as const) {
    const grid = GRIDS[rowDirection];
    for (const plane of PLANES) {
      const elevation = getPlane(plane.east, plane.north, rowDirection);
      const elevationBuffer = createInputBuffer(device, elevation);
      const hillshadeBuffer = createOutputBuffer(device, PIXEL_COUNT);
      const settings = new GPUParameterBuffer(device, {
        id: 'relief-signed-zero-settings',
        format: 'float32',
        length: 80,
        values: getGPUReliefShadingParameterValues(STYLES[0].settings, grid)
      });
      const graph = new GPUCommandGraph(device, {id: 'relief-signed-zero'});
      graph.add(
        new GPUReliefShading({
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
            hillshade: importGraphBuffer(
              graph,
              'hillshade',
              hillshadeBuffer,
              'float32',
              PIXEL_COUNT
            )
          }
        })
      );
      const compiled = graph.compile();
      for (const style of STYLES) {
        settings.write(getGPUReliefShadingParameterValues(style.settings, grid));
        submitGraph(device, compiled, undefined);
        const hillshade = await readFloat32(hillshadeBuffer, PIXEL_COUNT);
        const expected = computeReliefShading({
          width: WIDTH,
          height: HEIGHT,
          elevation,
          settings: style.settings,
          cellSize: [10, 10],
          rowDirection
        });
        for (let row = 1; row < HEIGHT - 1; row++) {
          for (let column = 1; column < WIDTH - 1; column++) {
            const index = row * WIDTH + column;
            const reference = expected.hillshade[index];
            if (
              !Number.isFinite(hillshade[index]) ||
              Math.abs(hillshade[index] - reference) > SHADE_TOLERANCE
            ) {
              failures.push(
                `${rowDirection} ${plane.name} ${style.name} pixel ${index}: GPU ${hillshade[index]}, CPU ${reference}`
              );
              break;
            }
          }
        }
      }
      compiled.destroy();
      settings.destroy();
      elevationBuffer.destroy();
      hillshadeBuffer.destroy();
    }
  }
  expect(failures).toEqual([]);
});
