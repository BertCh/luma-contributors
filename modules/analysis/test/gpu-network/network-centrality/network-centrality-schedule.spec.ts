// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkCentralityScheduleStep,
  planNetworkCentralitySchedule
} from '../../../src/gpu-network/network-centrality/index';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildCentralityCSRs, CENTRALITY_REFERENCE} from './network-centrality-oracle';

type Topology = {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};

async function runSchedule(
  device: Device,
  forward: Topology,
  reverse: Topology,
  pivots: readonly number[],
  options: {
    laneCount: number;
    scale?: number;
    sampled?: boolean;
    maximumIterations?: number;
    vertexPositions?: boolean;
    nonzeroViews?: boolean;
    queueOnly?: boolean;
    resetPivots?: readonly number[];
  }
) {
  const nodeCount = forward.offsets.length - 1;
  const graph = new GPUCommandGraph(device);
  const byteOffset = options.nonzeroViews ? 256 : 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    graph: GPUCommandGraph,
    id: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ) => {
    const count = length ?? (buffer.byteLength - byteOffset) / (format === 'float32x2' ? 8 : 4);
    const imported = importGraphBuffer(graph, id, buffer, format, count);
    return graph.createDataView(imported.buffer, {format, length: count, byteOffset});
  };
  const readFloats = async (buffer: Buffer, length: number) => {
    const bytes = await buffer.readAsync(byteOffset, length * 4);
    return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, length));
  };
  const readWords = async (buffer: Buffer, length: number) => {
    const bytes = await buffer.readAsync(byteOffset, length * 4);
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, length));
  };
  const buffers: Buffer[] = [];
  let nextId = 0;
  const input = (values: Uint32Array | Float32Array) => {
    const padded = new Uint8Array(byteOffset + values.byteLength);
    padded.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), byteOffset);
    const buffer = byteOffset
      ? device.createBuffer({data: padded, usage: Buffer.STORAGE | Buffer.COPY_DST})
      : createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length + byteOffset / 4);
    buffers.push(buffer);
    return buffer;
  };
  const topology = (network: Topology) => ({
    offsets: view(graph, `input-${nextId++}`, input(network.offsets), 'uint32'),
    neighbors: view(graph, `input-${nextId++}`, input(network.neighbors), 'uint32'),
    weights: view(graph, `input-${nextId++}`, input(network.weights), 'float32'),
    nodePositions: options.vertexPositions
      ? (() => {
          const buffer = device.createBuffer({
            byteLength: byteOffset + nodeCount * 8,
            usage: Buffer.VERTEX
          });
          buffers.push(buffer);
          return view(graph, `positions-${nextId++}`, buffer, 'float32x2', nodeCount);
        })()
      : undefined
  });
  const pivotBuffer = input(new Uint32Array(options.laneCount));
  const resetBuffer = input(Uint32Array.of(1));
  const betweenness = output(nodeCount);
  const edgeBetweenness = output(forward.neighbors.length);
  const closeness = output(options.laneCount);
  const harmonic = output(options.laneCount);
  const reached = output(options.laneCount);
  const status = {
    converged: output(1),
    zeroWeightCycle: output(1),
    iterationLimitReached: output(1),
    approximation: output(1)
  };
  graph.add(
    new GPUNetworkCentralityScheduleStep({
      id: 'schedule',
      topology: topology(forward),
      reverseTopology: topology(reverse),
      pivotNodes: view(graph, 'pivots', pivotBuffer, 'uint32'),
      reset: view(graph, 'reset', resetBuffer, 'uint32'),
      maximumIterations: options.maximumIterations ?? 64,
      laneCount: options.laneCount,
      sampling: options.sampled ? 'sampled' : 'exact',
      betweennessScale: options.scale ?? 1,
      output: {
        betweenness: view(graph, 'between', betweenness, 'float32'),
        edgeBetweenness: view(graph, 'edges', edgeBetweenness, 'float32'),
        closeness: view(graph, 'close', closeness, 'float32'),
        harmonicCloseness: view(graph, 'harmonic', harmonic, 'float32'),
        reachedCounts: view(graph, 'reached', reached, 'uint32'),
        status: Object.fromEntries(
          Object.entries(status).map(([name, buffer]) => [
            name,
            view(graph, name, buffer, 'uint32')
          ])
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    // Poison totals: reset must replace them without reading a NaN into the sum.
    betweenness.write(new Float32Array(nodeCount).fill(NaN), byteOffset);
    edgeBetweenness.write(new Float32Array(forward.neighbors.length).fill(NaN), byteOffset);
    const perPivot = {closeness: [] as number[], harmonic: [] as number[], reached: [] as number[]};
    for (const step of planNetworkCentralitySchedule({
      pivotCount: pivots.length,
      laneCount: options.laneCount
    })) {
      const batch = new Uint32Array(step.laneCount).fill(0xffffffff);
      batch.set(pivots.slice(step.pivotOffset, step.pivotOffset + step.pivotCount));
      pivotBuffer.write(batch, byteOffset);
      resetBuffer.write(Uint32Array.of(step.reset), byteOffset);
      submitGraph(device, compiled);
      if (!options.queueOnly) {
        perPivot.closeness.push(...(await readFloats(closeness, step.pivotCount)));
        perPivot.harmonic.push(...(await readFloats(harmonic, step.pivotCount)));
        perPivot.reached.push(...(await readWords(reached, step.pivotCount)));
      }
    }
    if (options.queueOnly) {
      perPivot.closeness.push(...(await readFloats(closeness, options.laneCount)));
      perPivot.harmonic.push(...(await readFloats(harmonic, options.laneCount)));
      perPivot.reached.push(...(await readWords(reached, options.laneCount)));
    }
    const result = {
      betweenness: await readFloats(betweenness, nodeCount),
      edgeBetweenness: await readFloats(edgeBetweenness, forward.neighbors.length),
      status: Object.fromEntries(
        await Promise.all(
          Object.entries(status).map(async ([name, buffer]) => [
            name,
            (await readWords(buffer, 1))[0]
          ])
        )
      ),
      ...perPivot
    };
    // Reset after a completed or failed run, into either an empty or independently checked valid run.
    const resetPivots = new Uint32Array(options.laneCount).fill(0xffffffff);
    if (options.resetPivots) resetPivots.set(options.resetPivots);
    pivotBuffer.write(resetPivots, byteOffset);
    resetBuffer.write(Uint32Array.of(1), byteOffset);
    submitGraph(device, compiled);
    const resetResult = {
      betweenness: await readFloats(betweenness, nodeCount),
      edgeBetweenness: await readFloats(edgeBetweenness, forward.neighbors.length),
      closeness: await readFloats(closeness, options.laneCount),
      harmonic: await readFloats(harmonic, options.laneCount),
      reached: await readWords(reached, options.laneCount),
      status: Object.fromEntries(
        await Promise.all(
          Object.entries(status).map(async ([name, buffer]) => [
            name,
            (await readWords(buffer, 1))[0]
          ])
        )
      )
    };
    if (!options.resetPivots) {
      expect(resetResult.betweenness).toEqual(new Array(nodeCount).fill(0));
      expect(resetResult.edgeBetweenness).toEqual(new Array(forward.neighbors.length).fill(0));
      expect(resetResult.closeness).toEqual(new Array(options.laneCount).fill(0));
      expect(resetResult.harmonic).toEqual(new Array(options.laneCount).fill(0));
      expect(resetResult.reached).toEqual(new Array(options.laneCount).fill(0));
      expect(resetResult.status).toEqual({
        converged: 1,
        zeroWeightCycle: 0,
        iterationLimitReached: 0,
        approximation: Number(Boolean(options.sampled))
      });
    }
    return {...result, resetResult};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectRelative(actual: readonly number[], expected: readonly number[], tolerance = 2e-5) {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < actual.length; row++)
    expect(Math.abs(actual[row] - expected[row]), `row ${row}`).toBeLessThanOrEqual(
      tolerance * Math.abs(expected[row])
    );
}

it('reuses one scheduled graph for exact and sampled results against committed networkx values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = CENTRALITY_REFERENCE.scenes['hand'];
  const {forward, reverse} = buildCentralityCSRs(scene.nodeCount, scene.edges);
  for (const sampled of [false, true]) {
    const pivots = sampled
      ? scene.sampledPivots
      : Array.from({length: scene.nodeCount}, (_, node) => node);
    const scale = sampled ? scene.sampledScale : 1;
    const result = await runSchedule(device, forward, reverse, pivots, {
      laneCount: 2,
      scale,
      sampled
    });
    expectRelative(
      result.betweenness,
      (sampled ? scene.betweenness.sampled : scene.betweenness.exact).map(value => scale * value)
    );
    expectRelative(
      result.closeness,
      pivots.map(pivot => scene.closeness.unbounded.closeness[pivot])
    );
    expectRelative(
      result.harmonic,
      pivots.map(pivot => scene.closeness.unbounded.harmonicCloseness[pivot])
    );
    expect(result.reached).toEqual(
      pivots.map(pivot => scene.closeness.unbounded.reachedCounts[pivot])
    );
    expect(result.status).toEqual({
      converged: 1,
      zeroWeightCycle: 0,
      iterationLimitReached: 0,
      approximation: Number(sampled)
    });
  }
});

it('preserves an earlier cycle and round-limit failure after later converged batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {forward, reverse} = buildCentralityCSRs(13, [
    [0, 1, 1],
    [1, 2, 0],
    [2, 1, 0],
    [2, 3, 2],
    ...Array.from({length: 8}, (_, node) => [node + 4, node + 5, 1] as [number, number, number])
  ]);
  const result = await runSchedule(device, forward, reverse, [0, 4, 12, 99], {
    laneCount: 1,
    maximumIterations: 4,
    resetPivots: [10]
  });
  expectRelative(
    result.resetResult.betweenness,
    Array.from({length: 13}, (_, node) => Number(node === 11))
  );
  expectRelative(
    result.resetResult.edgeBetweenness,
    Array.from({length: 12}, (_, edge) => (edge === 10 ? 2 : edge === 11 ? 1 : 0))
  );
  expectRelative(result.resetResult.closeness, [2 / 3]);
  expectRelative(result.resetResult.harmonic, [1.5]);
  expect(result.resetResult.reached).toEqual([3]);
  expect(result.resetResult.status).toEqual({
    converged: 1,
    zeroWeightCycle: 0,
    iterationLimitReached: 0,
    approximation: 0
  });
  expect(result.status).toEqual({
    converged: 0,
    zeroWeightCycle: 1,
    iterationLimitReached: 1,
    approximation: 0
  });
});

it('computes sampled centrality on one million directed edges against star closed forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edgeCount = 1_000_000;
  const nodeCount = edgeCount + 1;
  const forward: Topology = {
    offsets: new Uint32Array(nodeCount + 1).fill(edgeCount, 1),
    neighbors: Uint32Array.from({length: edgeCount}, (_, edge) => edge + 1),
    weights: new Float32Array(edgeCount).fill(1)
  };
  const reverse: Topology = {
    offsets: Uint32Array.from({length: nodeCount + 1}, (_, node) => Math.max(0, node - 1)),
    neighbors: new Uint32Array(edgeCount),
    weights: new Float32Array(edgeCount).fill(1)
  };
  const scale = nodeCount / 2;
  const result = await runSchedule(device, forward, reverse, [0, 1], {
    laneCount: 1,
    scale,
    sampled: true,
    maximumIterations: 4
  });
  expect(result.betweenness.every(value => value === 0)).toBe(true);
  expectRelative(result.edgeBetweenness, new Array(edgeCount).fill(scale));
  expect(result.closeness).toEqual([1, 0]);
  expect(result.harmonic).toEqual([edgeCount, 0]);
  expect(result.reached).toEqual([nodeCount, 1]);
  expect(result.status).toEqual({
    converged: 1,
    zeroWeightCycle: 0,
    iterationLimitReached: 0,
    approximation: 1
  });
}, 120_000);

it('accumulates independent hand-enumerated node and edge dependencies across encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {forward, reverse} = buildCentralityCSRs(5, [
    [0, 1, 1],
    [0, 2, 2],
    [1, 3, 1],
    [2, 3, 0],
    [3, 4, 3]
  ]);
  const result = await runSchedule(device, forward, reverse, [0, 0, 99], {laneCount: 1});
  // Each of the two valid pivots gives the two branches half of targets 3 and 4.
  expectRelative(result.betweenness, [0, 2, 2, 2, 0]);
  expectRelative(result.edgeBetweenness, [4, 4, 2, 2, 2]);
  expect(result.reached).toEqual([5, 5, 0]);
});

it('accepts vertex-only optional positions and preserves closed-form outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {forward, reverse} = buildCentralityCSRs(3, [
    [0, 1, 1],
    [1, 2, 1]
  ]);
  const baseline = await runSchedule(device, forward, reverse, [0], {laneCount: 1});
  const positioned = await runSchedule(device, forward, reverse, [0], {
    laneCount: 1,
    vertexPositions: true
  });
  expect(positioned).toEqual(baseline);
  expectRelative(positioned.betweenness, [0, 1, 0]);
  expectRelative(positioned.edgeBetweenness, [2, 1]);
  expectRelative(positioned.closeness, [2 / 3]);
  expect(positioned.reached).toEqual([3]);
});

it('queues a schedule beyond the old source-node budget without intermediate readbacks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 40,
    pivotCount = 257;
  // With every output/status, the old graph has 3*40+10 nodes per batch, plus fixed nodes.
  expect(pivotCount * (3 * nodeCount + 10)).toBeGreaterThan(32768);
  const {forward, reverse} = buildCentralityCSRs(
    nodeCount,
    Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1])
  );
  const result = await runSchedule(device, forward, reverse, new Array(pivotCount).fill(0), {
    laneCount: 1,
    maximumIterations: nodeCount,
    queueOnly: true
  });
  const dependencyTolerance = ((nodeCount - 1) * 8 + pivotCount + 2) * 2 ** -24;
  expectRelative(
    result.betweenness,
    Array.from({length: nodeCount}, (_, node) =>
      node === 0 ? 0 : pivotCount * (nodeCount - node - 1)
    ),
    dependencyTolerance
  );
  expectRelative(
    result.edgeBetweenness,
    Array.from({length: nodeCount - 1}, (_, edge) => pivotCount * (nodeCount - edge - 1)),
    dependencyTolerance
  );
  expectRelative(result.closeness, [2 / nodeCount]);
  expect(result.reached).toEqual([nodeCount]);
  expect(result.status).toEqual({
    converged: 1,
    zeroWeightCycle: 0,
    iterationLimitReached: 0,
    approximation: 0
  });
}, 120_000);

for (const laneCount of [255, 256, 257]) {
  it(`handles ${laneCount} lanes, a padded tail and nonzero binding offsets`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {forward, reverse} = buildCentralityCSRs(3, [
      [0, 1, 1],
      [1, 2, 1]
    ]);
    const pivotCount = laneCount + 3;
    const result = await runSchedule(device, forward, reverse, new Array(pivotCount).fill(0), {
      laneCount,
      nonzeroViews: true
    });
    expectRelative(result.betweenness, [0, pivotCount, 0]);
    expectRelative(result.edgeBetweenness, [2 * pivotCount, pivotCount]);
    expectRelative(result.closeness, new Array(pivotCount).fill(2 / 3));
    expectRelative(result.harmonic, new Array(pivotCount).fill(1.5));
    expect(result.reached).toEqual(new Array(pivotCount).fill(3));
    expect(result.status).toEqual({
      converged: 1,
      zeroWeightCycle: 0,
      iterationLimitReached: 0,
      approximation: 0
    });
  });
}
