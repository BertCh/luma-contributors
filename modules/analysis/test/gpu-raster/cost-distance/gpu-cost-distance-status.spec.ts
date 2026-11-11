// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUCostDistanceParameterValues,
  GPUCostDistance
} from '../../../src/raster-analysis/cost-distance/index';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const SIZE = 128;
const CELLS = SIZE * SIZE;
const GRID = makeGPURasterMetadata({width: SIZE, height: SIZE});

/** Nested square corridors with one gap per wall ring on alternating sides: a long winding path. */
function createSpiralFriction(): Float32Array {
  const friction = new Float32Array(CELLS).fill(1);
  for (let row = 0; row < SIZE; row++) {
    for (let column = 0; column < SIZE; column++) {
      const ring = Math.min(row, column, SIZE - 1 - row, SIZE - 1 - column);
      if ((ring & 7) < 4) continue;
      const gapOnLeft = (ring >> 3) % 2 === 0;
      const inGap =
        Math.abs(row - (SIZE >> 1)) < 2 && (gapOnLeft ? column < SIZE >> 1 : column >= SIZE >> 1);
      if (!inGap) friction[row * SIZE + column] = NaN;
    }
  }
  return friction;
}

it('GPUCostDistance publishes iterationLimitReached as the complement of converged', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const frictionBuffer = createInputBuffer(device, createSpiralFriction());
  const parameterBuffer = createInputBuffer(device, getGPUCostDistanceParameterValues({}, GRID));
  const sourceBuffer = createInputBuffer(device, Uint32Array.of(0));
  for (const backLinks of [false, true]) {
    for (const maximumIterations of [2, 512]) {
      const graph = new GPUCommandGraph(device, {id: 'cost-distance-status'});
      const owned = {
        costs: createOutputBuffer(device, CELLS),
        backLinks: createOutputBuffer(device, CELLS),
        converged: createOutputBuffer(device, 1),
        iterationLimitReached: createOutputBuffer(device, 1)
      };
      graph.add(
        new GPUCostDistance({
          id: 'cost-distance',
          grid: GRID,
          friction: makeGPURasterBand(
            importGraphBuffer(graph, 'friction', frictionBuffer, 'float32', CELLS),
            GRID,
            {id: 'friction'}
          ),
          parameters: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 8),
          sources: importGraphBuffer(graph, 'sources', sourceBuffer, 'uint32', 1),
          maximumIterations,
          output: {
            costs: importGraphBuffer(graph, 'costs', owned.costs, 'float32', CELLS),
            backLinks: backLinks
              ? importGraphBuffer(graph, 'back-links', owned.backLinks, 'uint32', CELLS)
              : undefined,
            status: {
              converged: importGraphBuffer(graph, 'converged', owned.converged, 'uint32', 1),
              iterationLimitReached: importGraphBuffer(
                graph,
                'iteration-limit-reached',
                owned.iterationLimitReached,
                'uint32',
                1
              )
            }
          }
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const [[converged], [iterationLimitReached]] = await Promise.all([
        readUint32(owned.converged, 1),
        readUint32(owned.iterationLimitReached, 1)
      ]);
      const expectedConverged = maximumIterations === 512 ? 1 : 0;
      expect({backLinks, maximumIterations, converged, iterationLimitReached}).toEqual({
        backLinks,
        maximumIterations,
        converged: expectedConverged,
        iterationLimitReached: 1 - expectedConverged
      });
      compiled.destroy();
      Object.values(owned).forEach(buffer => buffer.destroy());
    }
  }
  [frictionBuffer, parameterBuffer, sourceBuffer].forEach(buffer => buffer.destroy());
});
