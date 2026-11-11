// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {
  getGPUTerrainHorizonParameterValues,
  GPUTerrainHorizon,
  type GPUTerrainHorizonAlgorithm
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Rows-north grids with `rowDirection` left out: the contributors must take north from the grid
// affine. Expected values are closed forms of the geometry, not a model of the kernel.

const SIZE = 21;
const PIXEL_COUNT = SIZE * SIZE;
const CELL_SIZE = 10;
const CENTER = 10 * SIZE + 10;
/** Row 0 is the southernmost row: `affine[4] = +10`. */
const NORTH_GRID = makeGPURasterMetadata({
  width: SIZE,
  height: SIZE,
  cellSize: CELL_SIZE,
  rowDirection: 'north'
});
/** The same samples read as a north-up raster: `affine[4] = -10`. */
const SOUTH_GRID = makeGPURasterMetadata({width: SIZE, height: SIZE, cellSize: CELL_SIZE});

/**
 * Flat ground at 100 m with a 30 m plateau on rows 15 and above. On the rows-north grid the
 * plateau is north of the centre pixel, 5 rows (50 m) away: the horizon there is
 * atan(30 / 50) = 30.964 degrees, and 0 in the other three cardinal directions.
 */
const PLATEAU = Float32Array.from({length: PIXEL_COUNT}, (_, pixel) =>
  Math.floor(pixel / SIZE) >= 15 ? 130 : 100
);
const PLATEAU_HORIZON = (Math.atan(30 / 50) * 180) / Math.PI;

async function runHorizon(
  device: Device,
  grid: typeof NORTH_GRID,
  algorithm: GPUTerrainHorizonAlgorithm
): Promise<number[]> {
  const graph = new GPUCommandGraph(device, {id: 'rows-north-horizon'});
  const elevationBuffer = createInputBuffer(device, PLATEAU);
  const horizonBuffer = createOutputBuffer(device, PIXEL_COUNT * 4);
  const settings = new GPUParameterBuffer(device, {
    id: 'horizon-parameters',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHorizonParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainHorizon({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      parameters: settings.importToGraph(graph),
      directionCount: 4,
      maximumRadius: 8,
      algorithm,
      output: {
        horizon: importGraphBuffer(graph, 'horizon', horizonBuffer, 'float32', PIXEL_COUNT * 4)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const horizon = await readFloat32(horizonBuffer, PIXEL_COUNT * 4);
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  horizonBuffer.destroy();
  return horizon.slice(CENTER * 4, CENTER * 4 + 4);
}

async function runCastShadow(
  device: Device,
  grid: typeof NORTH_GRID,
  azimuthDegrees: number
): Promise<{sunVisibility: number; horizonAngle: number}> {
  const graph = new GPUCommandGraph(device, {id: 'rows-north-cast-shadow'});
  const elevationBuffer = createInputBuffer(device, PLATEAU);
  const visibilityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const horizonBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUParameterBuffer(device, {
    id: 'cast-shadow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
    // A hard-edged sun at 20 degrees: below the 30.964 degree plateau horizon, above flat ground.
    values: getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees, altitudeDegrees: 20, angularRadiusDegrees: 0},
      grid
    )
  });
  graph.add(
    new GPUTerrainCastShadow({
      grid,
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
        sunVisibility: importGraphBuffer(
          graph,
          'visibility',
          visibilityBuffer,
          'float32',
          PIXEL_COUNT
        ),
        horizonAngle: importGraphBuffer(graph, 'horizon', horizonBuffer, 'float32', PIXEL_COUNT)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    sunVisibility: (await readFloat32(visibilityBuffer, PIXEL_COUNT))[CENTER],
    horizonAngle: (await readFloat32(horizonBuffer, PIXEL_COUNT))[CENTER]
  };
  compiled.destroy();
  settings.destroy();
  for (const buffer of [elevationBuffer, visibilityBuffer, horizonBuffer]) {
    buffer.destroy();
  }
  return result;
}

it('GPUTerrainHorizon sectors follow the grid affine on a rows-north grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const algorithm of ['march', 'sweep'] as const) {
    // Sectors are compass azimuths 0 (north), 90, 180, 270.
    const northRows = await runHorizon(device, NORTH_GRID, algorithm);
    expect(Math.abs(northRows[0] - PLATEAU_HORIZON), `${algorithm} north`).toBeLessThan(2e-3);
    expect(northRows.slice(1), algorithm).toEqual([0, 0, 0]);
    // The same samples on a north-up grid put the plateau to the south.
    const southRows = await runHorizon(device, SOUTH_GRID, algorithm);
    expect(Math.abs(southRows[2] - PLATEAU_HORIZON), `${algorithm} south`).toBeLessThan(2e-3);
    expect([southRows[0], southRows[1], southRows[3]], algorithm).toEqual([0, 0, 0]);
  }
});

it('GPUTerrainCastShadow follows the grid affine on a rows-north grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Sun in the north, behind the plateau: the centre is in shadow and the horizon toward the sun
  // is the plateau edge.
  const sunNorth = await runCastShadow(device, NORTH_GRID, 0);
  expect(sunNorth.sunVisibility).toBe(0);
  expect(Math.abs(sunNorth.horizonAngle - PLATEAU_HORIZON)).toBeLessThan(2e-3);
  // Sun in the south, over flat ground: lit.
  const sunSouth = await runCastShadow(device, NORTH_GRID, 180);
  expect(sunSouth.sunVisibility).toBe(1);
  expect(sunSouth.horizonAngle).toBe(0);
  // Read as a north-up raster the plateau is in the south, and the two suns swap.
  expect((await runCastShadow(device, SOUTH_GRID, 0)).sunVisibility).toBe(1);
  expect((await runCastShadow(device, SOUTH_GRID, 180)).sunVisibility).toBe(0);
});
