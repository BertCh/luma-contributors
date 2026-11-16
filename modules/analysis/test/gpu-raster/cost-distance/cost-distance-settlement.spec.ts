// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCostDistance,
  getGPUCostDistanceParameterValues
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {
  getGPURasterRelaxationContinuationStateLength,
  GPU_RASTER_RELAXATION_RESTART_WORD
} from '../../../src/raster-analysis/cost-distance/raster-relaxation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('retains cumulative settlement rounds across submissions and resets them with the cost run', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 256;
  const grid = {
    width,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const graph = new GPUCommandGraph(device);
  const friction = createInputBuffer(device, new Float32Array(width).fill(1));
  const settings = createInputBuffer(device, getGPUCostDistanceParameterValues({}, grid));
  const sources = createInputBuffer(device, Uint32Array.of(0));
  const costs = createOutputBuffer(device, width);
  const working = createOutputBuffer(device, width);
  const stamps = createOutputBuffer(device, width);
  const stateLength = getGPURasterRelaxationContinuationStateLength(grid);
  const state = createOutputBuffer(device, stateLength);
  const status = createOutputBuffer(device, 1);
  const iterations = createOutputBuffer(device, 1);
  graph.add(
    new GPUCostDistance({
      grid,
      friction: {
        id: 'friction',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'friction', friction, 'float32', width)
        }
      },
      parameters: importGraphBuffer(graph, 'settings', settings, 'float32', 8),
      sources: importGraphBuffer(graph, 'sources', sources, 'uint32', 1),
      maximumIterations: 1,
      sweepAfterIteration: 1024,
      continuation: {
        state: importGraphBuffer(graph, 'state', state, 'uint32', stateLength),
        values: importGraphBuffer(graph, 'working', working, 'float32', width)
      },
      output: {
        costs: importGraphBuffer(graph, 'costs', costs, 'float32', width),
        settledAt: importGraphBuffer(graph, 'stamps', stamps, 'uint32', width),
        status: {
          converged: importGraphBuffer(graph, 'converged', status, 'uint32', 1),
          iterationCount: importGraphBuffer(graph, 'iterations', iterations, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  const execute = async () => {
    submitGraph(device, compiled, undefined);
    const [[converged], [iterationCount], settlement, values] = await Promise.all([
      readUint32(status, 1),
      readUint32(iterations, 1),
      readUint32(stamps, width),
      readFloat32(working, width)
    ]);
    return {converged, iterationCount, settlement, values};
  };
  try {
    let previous = await execute();
    expect(previous.converged).toBe(0);
    expect(previous.iterationCount).toBe(1);
    expect(previous.settlement[0]).toBe(0);
    expect(previous.settlement[1]).toBe(1);
    for (let execution = 0; !previous.converged && execution < 64; execution++) {
      const next = await execute();
      expect(next.iterationCount).toBeGreaterThan(previous.iterationCount);
      for (let cell = 0; cell < width; cell++) {
        if (next.values[cell] < previous.values[cell]) {
          expect(next.settlement[cell]).toBeGreaterThan(previous.iterationCount);
          expect(next.settlement[cell]).toBeLessThanOrEqual(next.iterationCount);
        } else {
          expect(next.settlement[cell]).toBe(previous.settlement[cell]);
        }
      }
      previous = next;
    }
    expect(previous.converged).toBe(1);
    // Unit friction on a one-dimensional grid has the closed-form path cost |cell - source|.
    expect(Array.from(await readFloat32(costs, width))).toEqual(
      Array.from({length: width}, (_, cell) => cell)
    );
    expect(previous.settlement[width - 1]).toBeGreaterThan(0);

    // A submission after convergence restarts automatically, including diagnostic outputs.
    stamps.write(new Uint32Array(width).fill(0xffffffff));
    working.write(new Float32Array(width).fill(-1));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(iterations, 1)).toEqual([1]);
    expect((await readUint32(stamps, width))[0]).toBe(0);
    expect((await readFloat32(working, width)).every(value => value >= 0)).toBe(true);
    expect((await readFloat32(costs, width)).every(Number.isNaN)).toBe(true);

    sources.write(Uint32Array.of(width - 1));
    state.write(Uint32Array.of(1), GPU_RASTER_RELAXATION_RESTART_WORD * 4);
    let restarted = await execute();
    expect(restarted.iterationCount).toBe(1);
    expect(restarted.settlement[width - 1]).toBe(0);
    for (let execution = 0; !restarted.converged && execution < 64; execution++) {
      restarted = await execute();
    }
    expect(restarted.converged).toBe(1);
    expect(Array.from(await readFloat32(costs, width))).toEqual(
      Array.from({length: width}, (_, cell) => width - 1 - cell)
    );
    expect(restarted.settlement[width - 1]).toBe(0);
    expect(restarted.settlement[width - 2]).toBe(1);
  } finally {
    compiled.destroy();
    for (const buffer of [
      friction,
      settings,
      sources,
      costs,
      working,
      stamps,
      state,
      status,
      iterations
    ])
      buffer.destroy();
  }
});

function createSettlementRun(
  device: Device,
  width: number,
  height: number,
  budget: number,
  offset = 0
) {
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const length = width * height;
  const graph = new GPUCommandGraph(device);
  const frictionValues = Float32Array.from({length}, (_, cell) => 1 + (cell % 5) / 4);
  const friction = createInputBuffer(device, frictionValues);
  const settings = createInputBuffer(device, getGPUCostDistanceParameterValues({}, grid));
  const sources = createInputBuffer(device, Uint32Array.of(length - 1));
  const sourceCount = createInputBuffer(device, Uint32Array.of(1));
  const costs = createOutputBuffer(device, length);
  const working = createOutputBuffer(device, length);
  const stateLength = getGPURasterRelaxationContinuationStateLength(grid);
  const stateWords = new Uint32Array(stateLength + offset * 2).fill(0xdeadbeef);
  stateWords.fill(0, offset, offset + stateLength);
  const state = createOutputBuffer(device, stateWords.length);
  state.write(stateWords);
  const stampWords = new Uint32Array(length + offset * 2).fill(0xdeadbeef);
  stampWords.fill(0xffffffff, offset, offset + length);
  const stamps = createOutputBuffer(device, stampWords.length);
  stamps.write(stampWords);
  const status = createOutputBuffer(device, 1);
  const iterations = createOutputBuffer(device, 1);
  const stateView = importGraphBuffer(graph, 'state', state, 'uint32', stateWords.length);
  const stampView = importGraphBuffer(graph, 'stamps', stamps, 'uint32', stampWords.length);
  graph.add(
    new GPUCostDistance({
      grid,
      friction: {
        id: 'friction',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'friction', friction, 'float32', length)
        }
      },
      parameters: importGraphBuffer(graph, 'settings', settings, 'float32', 8),
      sources: importGraphBuffer(graph, 'sources', sources, 'uint32', 1),
      sourceCount: importGraphBuffer(graph, 'source-count', sourceCount, 'uint32', 1),
      maximumIterations: budget,
      sweepAfterIteration: 0,
      continuation: {
        state: graph.createDataView(stateView.buffer, {
          format: 'uint32',
          byteOffset: offset * 4,
          length: stateLength
        }),
        values: importGraphBuffer(graph, 'working', working, 'float32', length)
      },
      output: {
        costs: importGraphBuffer(graph, 'costs', costs, 'float32', length),
        settledAt: graph.createDataView(stampView.buffer, {
          format: 'uint32',
          byteOffset: offset * 4,
          length
        }),
        status: {
          converged: importGraphBuffer(graph, 'converged', status, 'uint32', 1),
          iterationCount: importGraphBuffer(graph, 'iterations', iterations, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    frictionValues,
    sourceCount,
    restart() {
      state.write(Uint32Array.of(1), (offset + GPU_RASTER_RELAXATION_RESTART_WORD) * 4);
    },
    async execute() {
      submitGraph(device, compiled, undefined);
      const [[converged], [iterationCount], values, settlement, stateValues] = await Promise.all([
        readUint32(status, 1),
        readUint32(iterations, 1),
        readFloat32(costs, length),
        readUint32(stamps, stampWords.length),
        readUint32(state, stateWords.length)
      ]);
      if (offset > 0) {
        expect(settlement.slice(0, offset)).toEqual(Array(offset).fill(0xdeadbeef));
        expect(settlement.slice(offset + length)).toEqual(Array(offset).fill(0xdeadbeef));
        expect(stateValues.slice(0, offset)).toEqual(Array(offset).fill(0xdeadbeef));
        expect(stateValues.slice(offset + stateLength)).toEqual(Array(offset).fill(0xdeadbeef));
      }
      return {
        converged,
        iterationCount,
        values,
        settlement: settlement.slice(offset, offset + length)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [
        friction,
        settings,
        sources,
        sourceCount,
        costs,
        working,
        state,
        stamps,
        status,
        iterations
      ])
        buffer.destroy();
    }
  };
}

function computeShortestPaths(width: number, height: number, friction: Float32Array): Float64Array {
  const distances = new Float64Array(width * height).fill(Infinity);
  const visited = new Uint8Array(distances.length);
  distances[distances.length - 1] = 0;
  for (let iteration = 0; iteration < distances.length; iteration++) {
    let cell = -1;
    for (let candidate = 0; candidate < distances.length; candidate++) {
      if (!visited[candidate] && (cell < 0 || distances[candidate] < distances[cell]))
        cell = candidate;
    }
    if (cell < 0 || !Number.isFinite(distances[cell])) break;
    visited[cell] = 1;
    const row = Math.floor(cell / width);
    const column = cell % width;
    for (let rowDelta = -1; rowDelta <= 1; rowDelta++) {
      for (let columnDelta = -1; columnDelta <= 1; columnDelta++) {
        if (rowDelta === 0 && columnDelta === 0) continue;
        const neighborRow = row + rowDelta;
        const neighborColumn = column + columnDelta;
        if (
          neighborRow < 0 ||
          neighborRow >= height ||
          neighborColumn < 0 ||
          neighborColumn >= width
        )
          continue;
        const neighbor = neighborRow * width + neighborColumn;
        const edge =
          (Math.hypot(rowDelta, columnDelta) * (friction[cell] + friction[neighbor])) / 2;
        distances[neighbor] = Math.min(distances[neighbor], distances[cell] + edge);
      }
    }
  }
  return distances;
}

for (const width of [15, 16, 17, 31, 32, 33]) {
  for (const budget of [4, 5]) {
    it(`matches independent shortest paths across tile boundaries with sweeps: width ${width}, budget ${budget}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const run = createSettlementRun(device, width, 3, budget, 64);
      try {
        let result = await run.execute();
        for (let execution = 0; !result.converged && execution < 32; execution++)
          result = await run.execute();
        expect(result.converged).toBe(1);
        const expected = computeShortestPaths(width, 3, run.frictionValues);
        for (const [cell, cost] of result.values.entries())
          expect(cost).toBeCloseTo(expected[cell], 4);
        expect(result.settlement[width * 3 - 1]).toBe(0);
      } finally {
        run.destroy();
      }
    });
  }
}

it('clears old settlement stamps when an unconverged restart removes every source', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createSettlementRun(device, 33, 3, 1, 64);
  try {
    const first = await run.execute();
    expect(first.converged).toBe(0);
    expect(first.settlement.some(stamp => stamp > 0 && stamp !== 0xffffffff)).toBe(true);
    run.sourceCount.write(Uint32Array.of(0));
    run.restart();
    const restarted = await run.execute();
    expect(restarted.converged).toBe(1);
    expect(restarted.iterationCount).toBe(1);
    expect(restarted.values).toEqual(Array(99).fill(Infinity));
    expect(restarted.settlement).toEqual(Array(99).fill(0));
  } finally {
    run.destroy();
  }
});
