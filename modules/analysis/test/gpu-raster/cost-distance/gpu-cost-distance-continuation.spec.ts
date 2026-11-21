// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUCostDistanceParameterValues,
  getGPURasterRelaxationContinuationStateLength,
  GPUCostDistance,
  GPU_RASTER_RELAXATION_RESTART_WORD
} from '../../../src/raster-analysis/cost-distance/index';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeCostDistance} from './cost-distance-oracle';

const SIZE = 256;
const CELLS = SIZE * SIZE;
const GRID = makeGPURasterMetadata({width: SIZE, height: SIZE});

/** Nested square corridors 4 cells wide; each wall ring has one gap on alternating sides. */
function createSpiralFriction(size: number): Float32Array {
  const friction = new Float32Array(size * size).fill(1);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const ring = Math.min(row, column, size - 1 - row, size - 1 - column);
      if ((ring & 7) < 4) continue;
      const gapOnLeft = (ring >> 3) % 2 === 0;
      const inGap =
        Math.abs(row - (size >> 1)) < 2 && (gapOnLeft ? column < size >> 1 : column >= size >> 1);
      if (!inGap) friction[row * size + column] = NaN;
    }
  }
  return friction;
}

/** Builds one cost-distance graph over the shared friction buffer; `continued` adds a continuation. */
function createRun(device: Device, frictionBuffer: Buffer, continued: boolean) {
  const graph = new GPUCommandGraph(device, {id: continued ? 'cd-continued' : 'cd-one-shot'});
  const costs = createOutputBuffer(device, CELLS);
  const converged = createOutputBuffer(device, 1);
  const iterationCount = createOutputBuffer(device, 1);
  const state = createOutputBuffer(device, getGPURasterRelaxationContinuationStateLength(GRID));
  const values = createOutputBuffer(device, CELLS);
  const settings = new GPUParameterBuffer(device, {
    id: 'cd-settings',
    format: 'float32',
    length: 8,
    values: getGPUCostDistanceParameterValues({}, GRID)
  });
  const sources = new GPUParameterBuffer(device, {
    id: 'cd-sources',
    format: 'uint32',
    length: 1,
    values: Uint32Array.of(0)
  });
  graph.add(
    new GPUCostDistance({
      id: 'cd',
      grid: GRID,
      friction: makeGPURasterBand(
        importGraphBuffer(graph, 'cd-friction', frictionBuffer, 'float32', CELLS),
        GRID,
        {id: 'friction'}
      ),
      parameters: settings.importToGraph(graph),
      sources: sources.importToGraph(graph),
      continuation: continued
        ? {
            state: importGraphBuffer(
              graph,
              'cd-state',
              state,
              'uint32',
              getGPURasterRelaxationContinuationStateLength(GRID)
            ),
            values: importGraphBuffer(graph, 'cd-values', values, 'float32', CELLS)
          }
        : undefined,
      // Continued runs use the 64-round default; the one-shot run unrolls 512 rounds.
      maximumIterations: continued ? undefined : 512,
      output: {
        costs: importGraphBuffer(graph, 'cd-costs', costs, 'float32', CELLS),
        status: {
          converged: importGraphBuffer(graph, 'cd-converged', converged, 'uint32', 1),
          iterationCount: importGraphBuffer(
            graph,
            'cd-iteration-count',
            iterationCount,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  const history: {converged: number; iterationCount: number; state: number[]}[] = [];
  return {
    state,
    history,
    execute: async (restartRequested = false) => {
      submitGraph(device, compiled, undefined);
      const [[isConverged], [count], stateWords] = await Promise.all([
        readUint32(converged, 1),
        readUint32(iterationCount, 1),
        continued ? readUint32(state, 6) : Promise.resolve([] as number[])
      ]);
      const previous = history.at(-1);
      const previousCount =
        restartRequested || !previous || previous.converged ? 0 : previous.iterationCount;
      expect(count).toBeGreaterThan(previousCount);
      expect(count).toBeLessThanOrEqual(previousCount + (continued ? 64 : 512));
      if (!isConverged) expect(count).toBe(previousCount + (continued ? 64 : 512));
      history.push({converged: isConverged, iterationCount: count, state: stateWords});
      if (continued) {
        expect(count, 'published count must match caller-owned continuation state').toBe(
          stateWords[2]
        );
        expect(
          isConverged,
          'published convergence must match caller-owned continuation state'
        ).toBe(stateWords[3]);
      }
      return {converged: isConverged, iterationCount: count};
    },
    readCosts: () => readFloat32(costs, CELLS),
    destroy: () => {
      compiled.destroy();
      settings.destroy();
      sources.destroy();
      for (const buffer of [costs, converged, iterationCount, state, values]) buffer.destroy();
    }
  };
}

async function runToConvergence(run: ReturnType<typeof createRun>, maximumExecutions: number) {
  let status = {converged: 0, iterationCount: 0};
  let executions = 0;
  while (!status.converged && executions < maximumExecutions) {
    status = await run.execute();
    executions++;
  }
  return {...status, executions};
}

it('GPUCostDistance continuation: repeated 64-round executions equal the one-shot 512-round result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const friction = createSpiralFriction(SIZE);
  const frictionBuffer = createInputBuffer(device, friction);
  const oneShot = createRun(device, frictionBuffer, false);
  const continued = createRun(device, frictionBuffer, true);

  const expected = await runToConvergence(oneShot, 1);
  expect(expected.converged).toBe(1);
  const oneShotCosts = await oneShot.readCosts();

  const result = await runToConvergence(continued, 64);
  expect(result.converged).toBe(1);
  // Unconverged executions publish NaN for every passable cell; the converging one is exact.
  const continuedCosts = await continued.readCosts();
  expect(continuedCosts.length).toBe(oneShotCosts.length);
  for (const [cell, cost] of oneShotCosts.entries()) {
    expect(Object.is(continuedCosts[cell], cost), `cell ${cell}`).toBe(true);
  }
  // Independent Dijkstra integration on unit cells. Float32 path additions accumulate rounding;
  // 3e-5 relative bounds that error here. Check finiteness first so NaN cannot pass the comparison.
  const oracle = computeCostDistance({
    width: SIZE,
    height: SIZE,
    friction,
    georeference: {cellSize: [1, 1], northEdge: 0, southEdge: 0},
    sources: [{cell: 0, cost: 0}]
  });
  // Literal checkpoints from gpu-cost-distance-round-determinism.oracle.py, whose float32
  // Dijkstra uses no production geometry or relaxation helpers. Include an interior tail,
  // the opposite raster corner, sources and barriers rather than checking only the rim.
  const checkpoints = [
    [0, 0],
    [1, 1],
    [255, 255],
    [256, 1],
    [65280, 255],
    [65535, 505.8995056152344],
    [32896, 3777.5693359375],
    [32639, Infinity],
    [32888, 3758.497802734375],
    [33124, 3620.283203125],
    [1028, Infinity]
  ];
  for (const values of [oneShotCosts, continuedCosts]) {
    for (const [cell, cost] of values.entries()) {
      if (oracle[cell] === Infinity) expect(cost).toBe(Infinity);
      else {
        expect(Number.isFinite(cost), `cell ${cell}`).toBe(true);
        expect(Math.abs(cost - oracle[cell])).toBeLessThanOrEqual(3e-5 * Math.max(1, oracle[cell]));
      }
    }
    for (const [cell, expectedCost] of checkpoints) {
      if (expectedCost === Infinity) expect(values[cell]).toBe(Infinity);
      else {
        expect(Number.isFinite(values[cell]), `independent checkpoint ${cell}`).toBe(true);
        expect(Math.abs(values[cell] - expectedCost)).toBeLessThanOrEqual(
          3e-5 * Math.max(1, expectedCost)
        );
      }
    }
  }
  // In-place halo visibility changes propagation speed, including between submissions.
  expect(expected.iterationCount).toBeLessThanOrEqual(512);
  expect(result.iterationCount).toBeLessThanOrEqual(result.executions * 64);

  oneShot.destroy();
  continued.destroy();
  frictionBuffer.destroy();
}, 240000);

it('GPUCostDistance continuation: an unconverged execution reports NaN, and a restart follows an input change', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const friction = createSpiralFriction(SIZE);
  const frictionBuffer = createInputBuffer(device, friction);
  const continued = createRun(device, frictionBuffer, true);

  const first = await continued.execute();
  expect(first.converged).toBe(0);
  expect(first.iterationCount).toBe(64);
  const partial = await continued.readCosts();
  expect(Number.isNaN(partial[1])).toBe(true);
  const second = await continued.execute();
  expect(second.iterationCount).toBe(128);

  // Change the input mid-run and request a restart: the result is the new surface's one-shot result.
  const changed = Float32Array.from(friction, (value, cell) => (cell % 11 === 0 ? 3 : value));
  frictionBuffer.write(changed);
  continued.state.write(Uint32Array.of(1), GPU_RASTER_RELAXATION_RESTART_WORD * 4);
  const restarted = await continued.execute(true);
  expect(restarted.iterationCount).toBe(64);
  const finalStatus = await runToConvergence(continued, 64);
  expect(finalStatus.converged).toBe(1);
  const continuedCosts = await continued.readCosts();

  const reference = createRun(device, frictionBuffer, false);
  expect((await reference.execute()).converged).toBe(1);
  const referenceCosts = await reference.readCosts();
  for (const [cell, cost] of referenceCosts.entries()) {
    expect(Object.is(continuedCosts[cell], cost), `cell ${cell}`).toBe(true);
  }

  // A converged run restarts by itself: the next execution is again the first of a new run.
  const afterConverged = await continued.execute();
  expect(afterConverged.iterationCount).toBe(64);

  reference.destroy();
  continued.destroy();
  frictionBuffer.destroy();
}, 240000);
