// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {computeTerrainStreamOrderOnCPU} from '../../../src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation';
import {CPUTerrainCatchmentSummaryOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('an odd stream cycle is structural failure, not a tour iteration limit (B1, RT-123)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  const directions = createTransientView(graph, 'directions', 'uint32', 4);
  const streams = createTransientView(graph, 'streams', 'uint32', 4);
  const limit = createTransientView(graph, 'limit', 'uint32', 1);
  const words = Uint32Array.of(1, 8, 64, 0);
  const mask = new Uint32Array(4).fill(1);
  // 0 -> 1 -> 2 -> 0 has no root. Cell 3 is a separate leaf/outlet, hence order 1.
  const cpu = computeTerrainStreamOrderOnCPU(words, mask, {grid});
  expect(Array.from(cpu.streamOrder)).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 1]);
  expect(Array.from(cpu.converged)).toEqual([0]);
  expect(Array.from(cpu.iterationLimitReached)).toEqual([0]);
  const nodes = new GPUTerrainStreamOrder({
    grid,
    flowDirections: directions,
    streams,
    output: {
      streamOrder: createTransientView(graph, 'order', 'uint32', 4),
      status: {iterationLimitReached: limit}
    }
  }).getCommandNodes(graph);
  const limitIndex = nodes.findIndex(node => node.id.endsWith('-order-limit'));
  expect(limitIndex).toBeGreaterThan(0);
  const memory = createCPUGraphMemory();
  memory.write(directions, words);
  memory.write(streams, mask);
  memory.write(limit, Uint32Array.of(12345));
  try {
    // Execute the actual link, ranking, gate and limit kernels. No scan/barrier emulation:
    // the fixed tour bound may end with a changing odd cycle, but it has no caller budget.
    executeCommandNodesOnCPU(nodes.slice(0, limitIndex + 1), memory);
    expect(Array.from(memory.readUint32(limit))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

for (const unknownWord of [0, 0xfffffffe, 0xffffffff]) {
  it(`uncertain donor ${unknownWord.toString(16)} cannot certify reach endpoints (B3)`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const grid = makeGPURasterMetadata({width: 6, height: 2});
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length = 12
    ) => createTransientView(graph, name, format, length);
    const directions = view('directions', 'uint32');
    const streams = view('streams', 'uint32');
    const nodes = new GPUTerrainStreamNetwork({
      grid,
      flowDirections: directions,
      streams,
      output: {
        sourceCells: view('source', 'uint32'),
        targetCells: view('target', 'uint32'),
        vertexOffsets: view('offsets', 'uint32', 13),
        positions: view('positions', 'float32x2'),
        lengths: view('lengths', 'float32'),
        reachCount: view('reaches', 'uint32', 1),
        vertexCount: view('vertices', 'uint32', 1),
        unresolvedCount: view('unresolved', 'uint32', 1),
        status: {overflow: view('overflow', 'uint32', 1)}
      }
    }).getCommandNodes(graph);
    const reachesIndex = nodes.findIndex(node => node.id.endsWith('-reaches'));
    expect(reachesIndex).toBeGreaterThan(0);
    const resources = nodes[reachesIndex].resources!;
    const bufferAt = (index: number) => {
      const resource = resources[index];
      if (!('buffer' in resource)) throw new Error('expected a graph buffer');
      return resource.buffer as GraphDataView;
    };
    const memory = createCPUGraphMemory();
    memory.write(directions, Uint32Array.of(1, 1, 0, 0, 1, 0, 128, 0, 0, 0, 0, 0));
    memory.write(streams, Uint32Array.of(1, 1, 1, 0, 1, 1, unknownWord, 0, 0, 0, 0, 0));
    try {
      executeCommandNodesOnCPU(nodes.slice(0, reachesIndex + 1), memory);
      // Unknown cell 6 may donate to 1. The 0->1->2 segmentation is therefore uncertain;
      // suppress its reaches and count cells 0,1,6 unresolved. Definite reach 4->5 survives.
      expect(Array.from(memory.readUint32(bufferAt(3)))).toEqual([
        unknownWord ? 0 : 3,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]);
      expect(Array.from(memory.readUint32(bufferAt(4)))).toEqual(
        unknownWord ? [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] : new Array(12).fill(0)
      );
    } finally {
      device.destroy();
    }
  });
}

for (const testCase of [
  {
    name: 'cycle and excluded cell',
    directions: [1, 16, 0, 0],
    streams: [1, 1, 0, 1],
    maximumSteps: 4,
    expected: [0, 1, 1]
  },
  {
    name: 'unknown input without budget exhaustion',
    directions: [0, 0, 0, 0],
    streams: [0xfffffffe, 0, 0, 1],
    maximumSteps: 4,
    expected: [0, 2, 0]
  },
  {
    name: 'short path budget',
    directions: [1, 1, 1, 0],
    streams: [1, 1, 1, 1],
    maximumSteps: 1,
    expected: [0, 0, 1]
  },
  {
    name: 'complete path',
    directions: [1, 1, 1, 0],
    streams: [1, 1, 1, 1],
    maximumSteps: 4,
    expected: [1, 0, 0]
  }
]) {
  it(`CPU catchment optional status ports publish ${testCase.name} (consumer note)`, async () => {
    const operation = new CPUTerrainCatchmentSummaryOperation({
      grid: makeGPURasterMetadata({width: 4, height: 1}),
      basinCount: 1,
      histogramBins: 1,
      maximumSteps: testCase.maximumSteps,
      elevation: 'elevation',
      basinIds: 'basins',
      streams: 'streams',
      flowDirections: 'directions',
      output: {
        statistics: 'statistics',
        hypsometry: 'hypsometry',
        profiles: 'profiles',
        profileStatus: 'profileStatus',
        status: {
          converged: 'converged',
          invalidCount: 'invalidCount',
          iterationLimitReached: 'limit'
        }
      }
    });
    // Counts come from the four hand-specified cells. Unknown input causes state 2 without a
    // budget failure; the cycle and exclusive one-edge limit do exhaust the path computation.
    expect(Object.keys(operation.outputs).sort()).toEqual([
      'converged',
      'hypsometry',
      'invalidCount',
      'limit',
      'profileStatus',
      'profiles',
      'statistics'
    ]);
    const vector = (values: Uint32Array | Float32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const result = await operation.execute({
      inputs: {
        elevation: vector(new Float32Array(4).fill(1)),
        basinIds: vector(new Uint32Array(4)),
        flowDirections: vector(Uint32Array.from(testCase.directions)),
        streams: vector(Uint32Array.from(testCase.streams))
      }
    });
    expect(result['converged']).toEqual(Uint32Array.of(testCase.expected[0]));
    expect(result['invalidCount']).toEqual(Uint32Array.of(testCase.expected[1]));
    expect(result['limit']).toEqual(Uint32Array.of(testCase.expected[2]));
  });
}
