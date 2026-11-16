// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/network-topology';
import {
  GPUNetworkCentrality,
  computeNetworkCentralityOnCPU,
  createNetworkCentralityContinuation
} from '../../../src/gpu-network/network-centrality/index';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildCentralityCSRs, CENTRALITY_REFERENCE} from './network-centrality-oracle';

async function runContinuation(
  device: Device,
  nodeCount: number,
  edges: readonly (readonly [number, number, number])[],
  pivots: readonly number[],
  options: {
    roundsPerStep: number;
    maximumIterations?: number;
    compareSingleGraph?: boolean;
    sampled?: boolean;
    scale?: number;
    radius?: number;
    localIterations?: number;
    nonzeroViews?: boolean;
    packedOutputs?: boolean;
    batches?: readonly {pivots: readonly number[]; reset: boolean}[];
  }
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const byteOffset = options.nonzeroViews ? 256 : 0;
  const importView = <Format extends 'float32' | 'uint32'>(
    buffer: Buffer,
    format: Format,
    length: number,
    offset = byteOffset
  ) => {
    const imported = importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
    return graph.createDataView(imported.buffer, {format, length, byteOffset: offset});
  };
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const padded = new Uint8Array(byteOffset + values.byteLength);
    padded.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), byteOffset);
    const buffer = byteOffset
      ? device.createBuffer({data: padded, usage: Buffer.STORAGE | Buffer.COPY_DST})
      : createInputBuffer(device, values);
    buffers.push(buffer);
    return importView(buffer, format, values.length);
  };
  const packed = options.packedOutputs
    ? createOutputBuffer(
        device,
        byteOffset / 4 + nodeCount + edges.length + 3 * pivots.length + 4 + 256
      )
    : undefined;
  if (packed) buffers.push(packed);
  let outputIndex = 0,
    packedOffset = byteOffset;
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    if (outputIndex >= 2 && outputIndex <= 4) packedOffset = Math.ceil(packedOffset / 256) * 256;
    outputIndex++;
    const offset = packed ? packedOffset : byteOffset;
    packedOffset += length * 4;
    const buffer = packed ?? createOutputBuffer(device, length + byteOffset / 4);
    if (!packed) buffers.push(buffer);
    return {buffer, byteOffset: offset, view: importView(buffer, format, length, offset)};
  };
  const readFloats = async (value: {buffer: Buffer; byteOffset: number}, length: number) => {
    const bytes = await value.buffer.readAsync(value.byteOffset, length * 4);
    return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, length));
  };
  const readWords = async (value: {buffer: Buffer; byteOffset: number}, length: number) => {
    const bytes = await value.buffer.readAsync(value.byteOffset, length * 4);
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, length));
  };
  const {forward, reverse} = buildCentralityCSRs(nodeCount, edges);
  const topology = (network: typeof forward) => ({
    offsets: input('uint32', network.offsets),
    neighbors: input('uint32', network.neighbors),
    weights: input('float32', network.weights)
  });
  const betweenness = output('float32', nodeCount),
    edgeBetweenness = output('float32', edges.length);
  const closeness = output('float32', pivots.length),
    harmonic = output('float32', pivots.length),
    reached = output('uint32', pivots.length);
  const status = {
    converged: output('uint32', 1),
    zeroWeightCycle: output('uint32', 1),
    iterationLimitReached: output('uint32', 1),
    approximation: output('uint32', 1)
  };
  const props = {
    topology: topology(forward),
    reverseTopology: topology(reverse),
    pivotNodes: input('uint32', Uint32Array.from(pivots)),
    costLimit:
      options.radius === undefined ? undefined : input('float32', Float32Array.of(options.radius)),
    roundsPerStep: options.roundsPerStep,
    maximumIterations: options.maximumIterations,
    localIterations: options.localIterations ?? 1,
    sampling: options.sampled ? ('sampled' as const) : ('exact' as const),
    betweennessScale: options.scale ?? 1,
    output: {
      betweenness: betweenness.view,
      edgeBetweenness: edgeBetweenness.view,
      closeness: closeness.view,
      harmonicCloseness: harmonic.view,
      reachedCounts: reached.view,
      status: Object.fromEntries(Object.entries(status).map(([name, value]) => [name, value.view]))
    }
  };
  const read = async () => ({
    betweenness: await readFloats(betweenness, nodeCount),
    edgeBetweenness: await readFloats(edgeBetweenness, edges.length),
    closeness: await readFloats(closeness, pivots.length),
    harmonicCloseness: await readFloats(harmonic, pivots.length),
    reachedCounts: await readWords(reached, pivots.length),
    ...Object.fromEntries(
      await Promise.all(
        Object.entries(status).map(async ([name, value]) => [name, await readWords(value, 1)])
      )
    )
  });
  let baseline: Awaited<ReturnType<typeof read>> | undefined;
  if (options.compareSingleGraph) {
    graph.add(
      new GPUNetworkCentrality({
        ...props,
        maximumIterations: options.maximumIterations ?? nodeCount,
        laneCount: pivots.length
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      baseline = await read();
    } finally {
      compiled.destroy();
    }
  }
  const runner = createNetworkCentralityContinuation(graph, props);
  try {
    const snapshots: Awaited<ReturnType<typeof read>>[] = [];
    for (const batch of options.batches ?? [{pivots, reset: true}]) {
      const values = new Uint32Array(pivots.length).fill(0xffffffff);
      values.set(batch.pivots);
      getCoreDefaultBuffer(props.pivotNodes.buffer.defaultBuffer!).write(values, byteOffset);
      await runner.runBatch({reset: batch.reset});
      snapshots.push(await read());
    }
    const result = snapshots[snapshots.length - 1];
    const cpu = computeNetworkCentralityOnCPU(
      {
        topology: forward,
        reverseTopology: reverse,
        pivotNodes: Uint32Array.from(pivots),
        costLimit: options.radius === undefined ? undefined : Float32Array.of(options.radius)
      },
      {sampling: props.sampling, betweennessScale: props.betweennessScale}
    );
    return {result, baseline, cpu, snapshots};
  } finally {
    runner.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectRelative(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 2e-5) {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < expected.length; row++)
    expect(Math.abs(actual[row] - expected[row]), `row ${row}`).toBeLessThanOrEqual(
      tolerance * Math.abs(expected[row])
    );
}

it('continues a 2050-hop weighted road path across bounded graphs with closed-form centralities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 2051;
  const edges = Array.from(
    {length: nodeCount - 1},
    (_, node) => [node, node + 1, 2] as [number, number, number]
  );
  // Odd-sized windows exercise the boundary queue copy as well as the last short window.
  const {result} = await runContinuation(device, nodeCount, edges, [0], {roundsPerStep: 31});
  // Parent accuracy bound h*(7+K)*2^-24, with tight out-degree K=1 and one pivot.
  const dependencyTolerance = ((nodeCount - 1) * 8 + 3) * 2 ** -24;
  expectRelative(
    result.betweenness,
    Array.from({length: nodeCount}, (_, node) => (node === 0 ? 0 : nodeCount - node - 1)),
    dependencyTolerance
  );
  expectRelative(
    result.edgeBetweenness,
    Array.from({length: nodeCount - 1}, (_, edge) => nodeCount - edge - 1),
    dependencyTolerance
  );
  expectRelative(result.closeness, [1 / nodeCount]);
  expectRelative(result.harmonicCloseness, [
    Array.from({length: nodeCount - 1}, (_, row) => 1 / (2 * (row + 1))).reduce(
      (sum, value) => sum + value,
      0
    )
  ]);
  expect(result.reachedCounts).toEqual([nodeCount]);
  expect(result.converged).toEqual([1]);
  expect(result.iterationLimitReached).toEqual([0]);
  expect(result.zeroWeightCycle).toEqual([0]);
  expect(result.approximation).toEqual([0]);
}, 120_000);

for (const roundsPerStep of [1, 2, 3, 7]) {
  it(`matches the single-graph exact route and networkx with ${roundsPerStep}-round windows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = CENTRALITY_REFERENCE.scenes['hand'];
    const pivots = Array.from({length: scene.nodeCount}, (_, row) => row);
    const {result, baseline, cpu} = await runContinuation(
      device,
      scene.nodeCount,
      scene.edges,
      pivots,
      {roundsPerStep, compareSingleGraph: true}
    );
    expectRelative(result.betweenness, scene.betweenness.exact);
    expectRelative(result.betweenness, cpu.betweenness);
    for (const name of [
      'betweenness',
      'edgeBetweenness',
      'closeness',
      'harmonicCloseness'
    ] as const)
      expectRelative(result[name], baseline![name]);
    for (const name of [
      'reachedCounts',
      'converged',
      'zeroWeightCycle',
      'iterationLimitReached',
      'approximation'
    ] as const)
      expect(result[name]).toEqual(baseline![name]);
  });
}

it('honors a non-window-aligned total budget and defers cycle certification until sigma drains', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges: [number, number, number][] = [
    [0, 1, 1],
    [1, 2, 0],
    [2, 1, 0],
    [2, 3, 2],
    ...Array.from({length: 8}, (_, node) => [node + 4, node + 5, 1] as [number, number, number])
  ];
  const {result} = await runContinuation(device, 13, edges, [0, 4, 12], {
    roundsPerStep: 3,
    maximumIterations: 4,
    compareSingleGraph: true,
    sampled: true,
    scale: 13 / 3
  });
  expect(result.converged).toEqual([0]);
  expect(result.zeroWeightCycle).toEqual([0]);
  expect(result.iterationLimitReached).toEqual([1]);
  expect(result.approximation).toEqual([1]);
  expect(result.reachedCounts).toEqual([4, 5, 1]);
});

it('certifies a zero cycle after an exhausted frontier without reporting a round-limit exit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {result} = await runContinuation(
    device,
    5,
    [
      [0, 1, 1],
      [1, 2, 0],
      [2, 1, 0],
      [2, 3, 2]
    ],
    [0, 4],
    {roundsPerStep: 3, maximumIterations: 4, compareSingleGraph: true}
  );
  expect(result.converged).toEqual([0]);
  expect(result.zeroWeightCycle).toEqual([1]);
  expect(result.iterationLimitReached).toEqual([0]);
  expect(result.reachedCounts).toEqual([4, 1]);
});

for (const roundsPerStep of [2, 3]) {
  it(`preserves the inclusive cost limit across ${roundsPerStep}-round carry windows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const edges = Array.from(
      {length: 11},
      (_, node) => [node, node + 1, 1] as [number, number, number]
    );
    const {result, baseline} = await runContinuation(device, 12, edges, [0], {
      roundsPerStep,
      radius: 5,
      compareSingleGraph: true
    });
    expectRelative(
      result.betweenness,
      Array.from({length: 12}, (_, node) => (node > 0 && node < 5 ? 5 - node : 0))
    );
    expectRelative(
      result.edgeBetweenness,
      Array.from({length: 11}, (_, edge) => (edge < 5 ? 5 - edge : 0))
    );
    expectRelative(result.closeness, [1 / 3]);
    expect(result.reachedCounts).toEqual([6]);
    expect(result.converged).toEqual([1]);
    expectRelative(result.betweenness, baseline!.betweenness);
  });
}

it.each([
  false,
  true
])('reuses packed continuation outputs across batches, accumulates, fails and resets every field (nonzero views: %s)', async nonzeroViews => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges = Array.from(
    {length: 12},
    (_, node) => [node, node + 1, 1] as [number, number, number]
  );
  const {snapshots} = await runContinuation(device, 13, edges, [10, 0xffffffff, 0xffffffff], {
    roundsPerStep: 3,
    maximumIterations: 4,
    packedOutputs: true,
    nonzeroViews,
    sampled: true,
    batches: [
      {pivots: [10], reset: true},
      {pivots: [11], reset: false},
      {pivots: [4], reset: false},
      {pivots: [10], reset: true}
    ]
  });
  const nodeValues = (failed: boolean) =>
    Array.from({length: 13}, (_, node) =>
      node === 11 ? 1 : failed && node === 5 ? 2 : failed && node === 6 ? 1 : 0
    );
  const edgeValues = (second: boolean, failed: boolean) =>
    Array.from({length: 12}, (_, edge) =>
      edge === 10
        ? 2
        : edge === 11
          ? second
            ? 2
            : 1
          : failed && edge >= 4 && edge <= 6
            ? 7 - edge
            : 0
    );
  for (const [index, snapshot] of snapshots.entries()) {
    const failed = index === 2,
      second = index === 1 || failed;
    expectRelative(snapshot.betweenness, nodeValues(failed));
    expectRelative(snapshot.edgeBetweenness, edgeValues(second, failed));
    expectRelative(snapshot.closeness, [failed ? 0.4 : second ? 1 : 2 / 3, 0, 0]);
    expectRelative(snapshot.harmonicCloseness, [
      failed ? 1 + 1 / 2 + 1 / 3 + 1 / 4 : second ? 1 : 1.5,
      0,
      0
    ]);
    expect(snapshot.reachedCounts).toEqual([failed ? 5 : second ? 2 : 3, 0, 0]);
    expect(snapshot.converged).toEqual([Number(!failed)]);
    expect(snapshot.iterationLimitReached).toEqual([Number(failed)]);
    expect(snapshot.zeroWeightCycle).toEqual([0]);
    expect(snapshot.approximation).toEqual([1]);
  }
});

function createJoinedFrontier(width: number, roundsPerStep: number) {
  const join = roundsPerStep + width;
  const nodeCount = width + 2 * roundsPerStep;
  const edges: [number, number, number][] = [];
  for (let node = 0; node < roundsPerStep - 1; node++) edges.push([node, node + 1, 1]);
  for (let lane = 0; lane < width; lane++) edges.push([roundsPerStep - 1, roundsPerStep + lane, 1]);
  for (let lane = 0; lane < width; lane++) edges.push([roundsPerStep + lane, join, 1]);
  for (let node = join; node < nodeCount - 1; node++) edges.push([node, node + 1, 1]);
  return {
    nodeCount,
    edges,
    nodes: Array.from({length: nodeCount}, (_, node) =>
      node === 0
        ? 0
        : node < roundsPerStep
          ? nodeCount - node - 1
          : node < join
            ? roundsPerStep / width
            : nodeCount - node - 1
    ),
    edgeValues: edges.map(([source, destination]) =>
      source < roundsPerStep - 1 || source >= join
        ? nodeCount - source - 1
        : destination < join
          ? 1 + roundsPerStep / width
          : roundsPerStep / width
    )
  };
}

for (const width of [255, 256, 257])
  for (const roundsPerStep of [2, 3]) {
    it(`carries a ${width}-node joined frontier through ${roundsPerStep}-round windows and a final one-round window`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scene = createJoinedFrontier(width, roundsPerStep);
      const {result, cpu} = await runContinuation(device, scene.nodeCount, scene.edges, [0], {
        roundsPerStep,
        maximumIterations: 2 * roundsPerStep + 1
      });
      const tolerance = (2 * roundsPerStep * (7 + width) + 3) * 2 ** -24;
      expectRelative(result.betweenness, scene.nodes, tolerance);
      expectRelative(result.edgeBetweenness, scene.edgeValues, tolerance);
      expectRelative(result.closeness, cpu.closeness);
      expectRelative(result.harmonicCloseness, cpu.harmonicCloseness);
      expect(result.reachedCounts).toEqual([scene.nodeCount]);
      expect(result.converged).toEqual([1]);
      expect(result.iterationLimitReached).toEqual([0]);
      expect(result.zeroWeightCycle).toEqual([0]);
      expect(result.approximation).toEqual([0]);
    });
  }

it('combines localIterations 16, nonzero views and padded invalid pivots in continuation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createJoinedFrontier(257, 3);
  const {result, baseline, cpu} = await runContinuation(
    device,
    scene.nodeCount,
    scene.edges,
    [0, 0xffffffff, 0xffffffff],
    {
      roundsPerStep: 3,
      localIterations: 16,
      nonzeroViews: true,
      maximumIterations: 7,
      compareSingleGraph: true
    }
  );
  const tolerance = (6 * (7 + 257) + 5) * 2 ** -24;
  expectRelative(result.betweenness, scene.nodes, tolerance);
  expectRelative(result.edgeBetweenness, scene.edgeValues, tolerance);
  expectRelative(result.betweenness, baseline!.betweenness, tolerance);
  expectRelative(result.closeness, cpu.closeness);
  expectRelative(result.harmonicCloseness, cpu.harmonicCloseness);
  expect(result.reachedCounts).toEqual([scene.nodeCount, 0, 0]);
  expect(result.converged).toEqual([1]);
  expect(result.iterationLimitReached).toEqual([0]);
  expect(result.zeroWeightCycle).toEqual([0]);
  expect(result.approximation).toEqual([0]);
});
