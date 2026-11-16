// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  decodeGPUNetworkCoarseningSummary,
  GPUNetworkCoarsening
} from '../../../src/gpu-network/network-coarsening/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCoarseningCSR,
  computeCoarseningOracle,
  createRandom,
  createRandomCoarseningEdges,
  type CoarseningCSR,
  type CoarseningOracleResult
} from './network-coarsening-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type FixtureOptions = {
  nodeCount: number;
  csr: CoarseningCSR;
  labels: Uint32Array;
  groupCapacity: number;
  edgeCapacity: number;
  directed?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  positions?: Float32Array;
  vertexValues?: Float32Array;
  weighted?: boolean;
  fixedPointScale?: number;
};

type Result = {
  groupVertexCount: number[];
  groupIntraEdgeCount: number[];
  groupIntraWeight: number[];
  groupCentroid: number[];
  groupBounds: number[];
  groupValueSum: number[];
  count: number;
  total: number;
  overflow: number;
  edgeIds: number[];
  edgeTargets: number[];
  edgeCounts: number[];
  edgeWeights: number[];
  summary: number[];
};

class Fixture {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  readonly contributor: GPUNetworkCoarsening;
  readonly out: Record<string, Buffer> = {};
  compiled?: CompiledGPUCommandGraph<void>;

  constructor(
    readonly device: Device,
    readonly options: FixtureOptions
  ) {
    const {nodeCount, csr, groupCapacity, edgeCapacity} = options;
    this.graph = new GPUCommandGraph(device, {id: 'coarsening'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      buffer: Buffer,
      length: number
    ) => {
      this.buffers.push(buffer);
      return importGraphBuffer(this.graph, `c-${name}`, buffer, format, length);
    };
    const input = (name: string, format: GPUVectorFormat, values: Uint32Array | Float32Array) =>
      view(name, format as 'uint32', createInputBuffer(device, values), values.length);
    const output = (name: string, format: GPUVectorFormat, rows: number, words: number) => {
      const buffer = createOutputBuffer(device, rows * words);
      this.out[name] = buffer;
      return view(name, format as 'uint32', buffer, rows);
    };
    const {positions, vertexValues} = options;
    this.contributor = new GPUNetworkCoarsening({
      id: 'coarse',
      topology: {
        offsets: input('offsets', 'uint32', csr.offsets) as never,
        neighbors: input('neighbors', 'uint32', csr.neighbors) as never,
        weights: options.weighted ? (input('weights', 'float32', csr.weights) as never) : undefined,
        nodePositions:
          positions &&
          (view('positions', 'float32x2', createInputBuffer(device, positions), nodeCount) as never)
      },
      directed: options.directed,
      nodeMask: options.vertexMask && (input('vertex-mask', 'uint32', options.vertexMask) as never),
      edgeMask: options.edgeMask && (input('edge-mask', 'uint32', options.edgeMask) as never),
      labels: input('labels', 'uint32', options.labels) as never,
      nodeValues: vertexValues && (input('values', 'float32', vertexValues) as never),
      groupCapacity,
      fixedPointScale: options.fixedPointScale,
      output: {
        groupNodeCount: output('group-counts', 'uint32', groupCapacity, 1) as never,
        groupIntraEdgeCount: output('intra-count', 'uint32', groupCapacity, 1) as never,
        groupIntraWeight: output('intra-weight', 'float32', groupCapacity, 1) as never,
        groupCentroid: positions && (output('centroid', 'float32x2', groupCapacity, 2) as never),
        groupBounds: positions && (output('bounds', 'float32x4', groupCapacity, 4) as never),
        groupValueSum: vertexValues && (output('value-sum', 'float32', groupCapacity, 1) as never),
        edges: {
          ids: output('edge-ids', 'uint32', edgeCapacity, 1) as never,
          status: {
            count: output('edge-count', 'uint32', 1, 1) as never,
            overflow: output('edge-overflow', 'uint32', 1, 1) as never,
            requiredCount: output('edge-total', 'uint32', 1, 1) as never
          }
        },
        edgeTargetGroups: output('edge-targets', 'uint32', edgeCapacity, 1) as never,
        edgeCounts: output('edge-counts', 'uint32', edgeCapacity, 1) as never,
        edgeWeights: output('edge-weights', 'float32', edgeCapacity, 1) as never,
        summary: output('summary', 'uint32', 8, 1) as never
      }
    });
    this.graph.add(this.contributor);
  }

  async run(): Promise<Result> {
    this.compiled ??= this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
    const {groupCapacity: groups, edgeCapacity: edges} = this.options;
    const u = (name: string, length: number) => readUint32(this.out[name], length);
    const f = (name: string, length: number) => readFloat32(this.out[name], length);
    return {
      groupVertexCount: await u('group-counts', groups),
      groupIntraEdgeCount: await u('intra-count', groups),
      groupIntraWeight: await f('intra-weight', groups),
      groupCentroid: this.out['centroid'] ? await f('centroid', groups * 2) : [],
      groupBounds: this.out['bounds'] ? await f('bounds', groups * 4) : [],
      groupValueSum: this.out['value-sum'] ? await f('value-sum', groups) : [],
      count: (await u('edge-count', 1))[0],
      total: (await u('edge-total', 1))[0],
      overflow: (await u('edge-overflow', 1))[0],
      edgeIds: await u('edge-ids', edges),
      edgeTargets: await u('edge-targets', edges),
      edgeCounts: await u('edge-counts', edges),
      edgeWeights: await f('edge-weights', edges),
      summary: await u('summary', 8)
    };
  }

  oracle(): CoarseningOracleResult {
    const {options} = this;
    return computeCoarseningOracle({
      nodeCount: options.nodeCount,
      csr: options.csr,
      labels: options.labels,
      groupCapacity: options.groupCapacity,
      directed: options.directed,
      vertexMask: options.vertexMask,
      edgeMask: options.edgeMask,
      positions: options.positions,
      vertexValues: options.vertexValues,
      unweighted: !options.weighted,
      fixedPointScale: options.fixedPointScale,
      edgeCapacity: options.edgeCapacity
    });
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of [...this.buffers, ...Object.values(this.out)]) buffer.destroy();
  }
}

function close(actual: number[], expected: ArrayLike<number>, relative = 2e-6): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index++) {
    const tolerance = relative * Math.max(1, Math.abs(expected[index]));
    expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(tolerance);
  }
}

/** Compares a GPU result to the oracle, honoring the superedge capacity. */
function expectMatches(actual: Result, fixture: Fixture): CoarseningOracleResult {
  const expected = fixture.oracle();
  const {edgeCapacity} = fixture.options;
  expect(actual.groupVertexCount).toEqual(Array.from(expected.groupVertexCount));
  expect(actual.groupIntraEdgeCount).toEqual(Array.from(expected.groupIntraEdgeCount));
  close(actual.groupIntraWeight, expected.groupIntraWeight);
  if (fixture.options.positions) {
    close(actual.groupCentroid, expected.groupCentroid);
    expect(actual.groupBounds).toEqual(Array.from(expected.groupBounds));
  }
  if (fixture.options.vertexValues) close(actual.groupValueSum, expected.groupValueSum);
  expect(actual.summary).toEqual(expected.summary);
  expect(actual.total).toBe(expected.edges.length);
  expect(actual.count).toBe(Math.min(expected.edges.length, edgeCapacity));
  expect(actual.overflow).toBe(expected.overflow ? 1 : 0);
  const kept = expected.edges.slice(0, edgeCapacity);
  expect(actual.edgeIds.slice(0, actual.count)).toEqual(kept.map(edge => edge.source));
  expect(actual.edgeTargets.slice(0, actual.count)).toEqual(kept.map(edge => edge.target));
  expect(actual.edgeCounts.slice(0, actual.count)).toEqual(kept.map(edge => edge.count));
  close(
    actual.edgeWeights.slice(0, actual.count),
    kept.map(edge => edge.weight)
  );
  // Rows past the count are zero.
  expect(actual.edgeIds.slice(actual.count).every(value => value === 0)).toBe(true);
  expect(actual.edgeCounts.slice(actual.count).every(value => value === 0)).toBe(true);
  return expected;
}

function createLabels(nodeCount: number, groups: number, seed: number, wild = 0): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: nodeCount}, () =>
    random() < wild ? 1000 + Math.floor(random() * 50) : Math.floor(random() * groups)
  );
}

function createMask(length: number, deadFraction: number, seed: number): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length}, () => (random() < deadFraction ? 0 : 1));
}

function createPositions(nodeCount: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: nodeCount * 2}, () => (random() - 0.5) * 360);
}

function createValues(nodeCount: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: nodeCount}, () => Math.round((random() - 0.3) * 40) / 8);
}

function createCase(
  device: Device,
  seed: number,
  directed: boolean,
  overrides: Partial<FixtureOptions> = {}
): Fixture {
  const nodeCount = 300;
  const edges = createRandomCoarseningEdges(seed, nodeCount, 700, directed);
  const csr = buildCoarseningCSR(nodeCount, edges);
  return new Fixture(device, {
    nodeCount,
    csr,
    labels: createLabels(nodeCount, 16, seed + 1),
    groupCapacity: 16,
    edgeCapacity: 400,
    directed,
    positions: createPositions(nodeCount, seed + 2),
    vertexValues: createValues(nodeCount, seed + 3),
    weighted: true,
    ...overrides
  });
}

for (const directed of [false, true]) {
  it(`GPUNetworkCoarsening matches the oracle on a random ${directed ? 'directed' : 'undirected'} graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createCase(device, 11, directed);
    const result = await fixture.run();
    const expected = expectMatches(result, fixture);
    expect(expected.edges.length).toBeGreaterThan(20);
    expect(expected.summary[4]).toBeGreaterThan(0);
    const summary = decodeGPUNetworkCoarseningSummary(result.summary);
    expect(summary.liveNodeCount).toBe(300);
    expect(summary.groupCount).toBe(16);
    if (!directed) {
      expect(result.edgeIds.slice(0, result.count).every((s, i) => s < result.edgeTargets[i])).toBe(
        true
      );
    }
    fixture.destroy();
  });
}

it('GPUNetworkCoarsening honors vertex and edge masks and is repeatable', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 300;
  const slotCount = buildCoarseningCSR(
    nodeCount,
    createRandomCoarseningEdges(21, nodeCount, 700, true)
  ).neighbors.length;
  const fixture = createCase(device, 21, true, {
    vertexMask: createMask(nodeCount, 0.2, 5),
    edgeMask: createMask(slotCount, 0.3, 6)
  });
  const first = await fixture.run();
  const expected = expectMatches(first, fixture);
  expect(expected.summary[0]).toBeLessThan(nodeCount);
  const second = await fixture.run();
  expect(second).toEqual(first);
  fixture.destroy();
});

it('GPUNetworkCoarsening handles self-loops and unweighted undirected graphs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = buildCoarseningCSR(5, [
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 2, 1],
    [2, 1, 1],
    [3, 4, 1],
    [4, 3, 1]
  ]);
  const fixture = new Fixture(device, {
    nodeCount: 5,
    csr,
    labels: Uint32Array.from([0, 0, 1, 2, 2]),
    groupCapacity: 3,
    edgeCapacity: 4
  });
  const result = await fixture.run();
  expectMatches(result, fixture);
  expect(result.groupIntraEdgeCount).toEqual([3, 0, 1]);
  expect(result.edgeIds.slice(0, result.count)).toEqual([0]);
  expect(result.edgeTargets.slice(0, result.count)).toEqual([1]);
  expect(result.edgeCounts.slice(0, result.count)).toEqual([1]);
  expect(result.edgeWeights[0]).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening reports label overflow and excludes those vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createCase(device, 31, false, {
    labels: createLabels(300, 16, 32, 0.1)
  });
  const result = await fixture.run();
  const expected = expectMatches(result, fixture);
  expect(expected.summary[1]).toBeGreaterThan(0);
  expect(expected.summary[3]).toBeGreaterThan(0);
  expect(result.overflow).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening clamps the superedge list and keeps the smallest keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createCase(device, 41, false, {edgeCapacity: 25});
  const result = await fixture.run();
  const expected = expectMatches(result, fixture);
  expect(expected.edges.length).toBeGreaterThan(25);
  expect(result.count).toBe(25);
  expect(result.total).toBe(expected.edges.length);
  expect(result.overflow).toBe(1);
  fixture.destroy();
});

it('GPUNetworkCoarsening is exact on a larger graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 5000;
  const csr = buildCoarseningCSR(
    nodeCount,
    createRandomCoarseningEdges(51, nodeCount, 20000, false)
  );
  const fixture = new Fixture(device, {
    nodeCount,
    csr,
    labels: createLabels(nodeCount, 200, 52),
    groupCapacity: 256,
    edgeCapacity: 4096,
    positions: createPositions(nodeCount, 53),
    vertexValues: createValues(nodeCount, 54),
    weighted: true
  });
  expectMatches(await fixture.run(), fixture);
  fixture.destroy();
});

for (const directed of [false, true]) {
  it(`GPUNetworkCoarsening matches the oracle with few heavy groups (${directed ? 'directed' : 'undirected'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Three groups and thousands of edges: every superedge is a run far longer than one
    // invocation's chunk of sorted slots, and every row folds many intra-group slots.
    const nodeCount = 600;
    const edges = createRandomCoarseningEdges(77, nodeCount, 6000, directed);
    const fixture = new Fixture(device, {
      nodeCount,
      csr: buildCoarseningCSR(nodeCount, edges),
      labels: createLabels(nodeCount, 3, 78),
      groupCapacity: 3,
      edgeCapacity: 8,
      directed,
      positions: createPositions(nodeCount, 79),
      vertexValues: createValues(nodeCount, 80),
      weighted: true
    });
    const result = await fixture.run();
    const expected = expectMatches(result, fixture);
    expect(expected.edges.length).toBeGreaterThan(0);
    expect(expected.edges[0].count).toBeGreaterThan(100);
    fixture.destroy();
  });
}

/**
 * Two groups of 1,000 nodes with small terms everywhere: 1,000 edges of weight 1e-4 between the
 * groups, a ring of 1,000 edges of weight 1e-5 inside group 0, node values of 1e-3 and 2.5e-6,
 * and coordinates of a few 1e-4. One more node has a label above the group capacity, huge
 * coordinates and a huge value, and the edge to it has the largest float32 as its weight: it is
 * dropped, so none of that may enter a sum or set its unit.
 */
function createSmallTermsFixture(device: Device, fixedPointScale?: number): Fixture {
  const groupSize = 1000;
  const nodeCount = 2 * groupSize + 1;
  const edges: [number, number, number][] = [];
  for (let node = 0; node < groupSize; node++) {
    edges.push([node, groupSize + node, 1e-4], [node, (node + 1) % groupSize, 1e-5]);
  }
  edges.push([0, 2 * groupSize, 3.4028234e38]);
  const positions = new Float32Array(2 * nodeCount);
  const vertexValues = new Float32Array(nodeCount);
  const labels = new Uint32Array(nodeCount);
  for (let node = 0; node < 2 * groupSize; node++) {
    const group = node < groupSize ? 0 : 1;
    labels[node] = group;
    positions[2 * node] = (node % groupSize) * 1e-6;
    positions[2 * node + 1] = group === 0 ? -3e-7 : 2e-4;
    vertexValues[node] = group === 0 ? 1e-3 : 2.5e-6;
  }
  labels[2 * groupSize] = 1000;
  positions.set([1e20, -1e20], 4 * groupSize);
  vertexValues[2 * groupSize] = 1e30;
  return new Fixture(device, {
    nodeCount,
    csr: buildCoarseningCSR(nodeCount, edges),
    labels,
    groupCapacity: 2,
    edgeCapacity: 4,
    directed: true,
    positions,
    vertexValues,
    weighted: true,
    fixedPointScale
  });
}

function expectRelative(actual: number, expected: number, relative = 1e-5): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(relative * Math.abs(expected));
}

it('GPUNetworkCoarsening sums small weights, values and coordinates without rounding them away', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSmallTermsFixture(device);
  const result = await fixture.run();
  // By hand: 1,000 * 1e-4 between the groups and 1,000 * 1e-5 inside group 0. A unit of 1/65536
  // gave 1,000 * 7/65536 = 0.1068 and 1,000 * 1/65536 = 0.0153.
  expect(result.count).toBe(1);
  expect(result.edgeCounts[0]).toBe(1000);
  expectRelative(result.edgeWeights[0], 0.1);
  expect(result.groupIntraEdgeCount).toEqual([1000, 0]);
  expectRelative(result.groupIntraWeight[0], 0.01);
  expect(result.groupIntraWeight[1]).toBe(0);
  // 1,000 * 1e-3 and 1,000 * 2.5e-6.
  expectRelative(result.groupValueSum[0], 1);
  expectRelative(result.groupValueSum[1], 0.0025);
  // Mean of 0, 1e-6, ..., 999e-6 is 499.5e-6; y is one value per group.
  expectRelative(result.groupCentroid[0], 499.5e-6);
  expectRelative(result.groupCentroid[1], -3e-7);
  expectRelative(result.groupCentroid[2], 499.5e-6);
  expectRelative(result.groupCentroid[3], 2e-4);
  // The node above the group capacity and its edge are reported, not summed.
  const summary = decodeGPUNetworkCoarseningSummary(result.summary);
  expect(summary.overflowedNodeCount).toBe(1);
  expect(summary.droppedEdgeCount).toBe(1);
  expectMatches(result, fixture);
  fixture.destroy();
});

it('GPUNetworkCoarsening keeps the fixed rounding step when fixedPointScale is given', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSmallTermsFixture(device, 65536);
  const result = await fixture.run();
  // Each addend is rounded to 1/65536: 1e-4 is 6.55 steps and rounds to 7, 1e-5 is 0.66 steps
  // and rounds to 1, 1e-3 is 65.5 steps and rounds to 66.
  expect(result.edgeWeights[0]).toBe(Math.fround(7000 / 65536));
  expect(result.groupIntraWeight[0]).toBe(Math.fround(1000 / 65536));
  expect(result.groupValueSum[0]).toBe(Math.fround(66000 / 65536));
  expectMatches(result, fixture);
  fixture.destroy();
});

it('GPUNetworkCoarsening saturates a term too large for a fixed fixedPointScale', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // With a unit of 1/65536 a term saturates at 4e18 units (the float32 nearest 4e18), about
  // 6.1e13. 1.5e14 is 9.8e18 units, past 2^63: it must add the saturated value, not wrap into a
  // negative sum. The largest float32 (the impassable weight some topologies carry) saturates
  // too, and a negative term saturates with its sign.
  const edges: [number, number, number][] = [
    [0, 2, 1.5e14],
    [0, 1, 3.4028234e38],
    [2, 3, -2e14]
  ];
  const fixture = new Fixture(device, {
    nodeCount: 4,
    csr: buildCoarseningCSR(4, edges),
    labels: Uint32Array.from([0, 0, 1, 1]),
    groupCapacity: 2,
    edgeCapacity: 4,
    directed: true,
    weighted: true,
    fixedPointScale: 65536
  });
  const result = await fixture.run();
  const saturated = Math.fround(Math.fround(4.0e18) / 65536);
  expect(saturated).toBeGreaterThan(6.1e13);
  expect(result.count).toBe(1);
  expect(result.edgeWeights[0]).toBe(saturated);
  expect(result.groupIntraWeight).toEqual([saturated, -saturated]);
  expectMatches(result, fixture);
  fixture.destroy();
});

it('coarsens above 65535 groups with separate source and target keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const labels = new Uint32Array([0, 32768, 65536, 131071, 65536, 0]);
  const csr = buildCoarseningCSR(6, [
    [0, 1, 2],
    [2, 1, 3],
    [2, 1, 4],
    [1, 3, 5],
    [4, 5, 6],
    [3, 2, 7],
    [0, 0, 9]
  ]);
  for (const directed of [false, true]) {
    for (const edgeCapacity of [2, 8]) {
      const fixture = new Fixture(device, {
        nodeCount: 6,
        csr,
        labels,
        groupCapacity: 131072,
        edgeCapacity,
        directed,
        weighted: true
      });
      try {
        const result = await fixture.run();
        expectMatches(result, fixture);
        if (directed) {
          // Independent expected tuple order, including two source IDs whose old packed keys collide.
          expect(result.total).toBe(5);
          expect(result.edgeIds.slice(0, result.count)).toEqual(
            [0, 32768, 65536, 65536, 131071].slice(0, edgeCapacity)
          );
          expect(result.edgeTargets.slice(0, result.count)).toEqual(
            [32768, 131071, 0, 32768, 65536].slice(0, edgeCapacity)
          );
          expect(result.edgeCounts.slice(0, result.count)).toEqual(
            [1, 1, 1, 2, 1].slice(0, edgeCapacity)
          );
          expect(result.edgeWeights.slice(0, result.count)).toEqual(
            [2, 5, 6, 7, 7].slice(0, edgeCapacity)
          );
        }
        expect(result.groupVertexCount[65536]).toBe(2);
        expect(result.groupVertexCount[0]).toBe(2);
        expectMatches(await fixture.run(), fixture);
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('coarsening tuple runs cross 15/16, 255/256 and 511/512 with truncation and repeated encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const labels = new Uint32Array([0, 32768, 65536, 131071]);
  for (const lengths of [
    [18, 13],
    [18, 240, 7, 257]
  ]) {
    const tuples = [
      [0, 1],
      [1, 2],
      [2, 0],
      [2, 3]
    ];
    const edges: [number, number, number][] = [];
    for (const [group, length] of lengths.entries())
      for (let row = 0; row < length; row++)
        edges.push([tuples[group][0], tuples[group][1], 0.125]);
    for (const edgeCapacity of [1, 6]) {
      const fixture = new Fixture(device, {
        nodeCount: 4,
        csr: buildCoarseningCSR(4, edges),
        labels,
        groupCapacity: 131072,
        edgeCapacity,
        directed: true,
        weighted: true
      });
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          const result = await fixture.run();
          expectMatches(result, fixture);
          const count = Math.min(edgeCapacity, lengths.length);
          expect(result.total).toBe(lengths.length);
          expect(result.count).toBe(count);
          expect(result.edgeIds.slice(0, count)).toEqual([0, 32768, 65536, 65536].slice(0, count));
          expect(result.edgeTargets.slice(0, count)).toEqual(
            [32768, 65536, 0, 131071].slice(0, count)
          );
          expect(result.edgeCounts.slice(0, count)).toEqual(lengths.slice(0, count));
          expect(result.edgeWeights.slice(0, count)).toEqual(
            lengths.slice(0, count).map(length => length / 8)
          );
        }
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('coarsening shared edge buffers reject adjacent slices and accept aligned independent columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  for (const stride of [4, 256]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = (name: string, values: Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, 'uint32', values.length);
    };
    const output = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const columns = output('shared-edge-columns', 129);
    const count = output('shared-edge-count', 1);
    const overflow = output('shared-edge-overflow', 1);
    const groups = output('shared-group-counts', 131072);
    const props = {
      groupCapacity: 131072,
      directed: true,
      topology: {
        offsets: input('shared-offsets', new Uint32Array([0, 1, 1])),
        neighbors: input('shared-neighbors', new Uint32Array([1]))
      },
      labels: input('shared-labels', new Uint32Array([65536, 131071])),
      output: {
        groupNodeCount: groups.view,
        edges: {
          ids: graph.createDataView(columns.view.buffer, {format: 'uint32' as const, length: 1}),
          status: {count: count.view, overflow: overflow.view}
        },
        edgeTargetGroups: graph.createDataView(columns.view.buffer, {
          format: 'uint32' as const,
          byteOffset: stride,
          length: 1
        }),
        edgeCounts: graph.createDataView(columns.view.buffer, {
          format: 'uint32' as const,
          byteOffset: 512,
          length: 1
        })
      }
    };
    try {
      if (stride === 4) {
        expect(() => new GPUNetworkCoarsening(props)).toThrow(/aligned binding byte ranges/);
      } else {
        graph.add(new GPUNetworkCoarsening(props));
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const values = await readUint32(columns.buffer, 129);
          expect([values[0], values[64], values[128]]).toEqual([65536, 131071, 1]);
          expect(await readUint32(count.buffer, 1)).toEqual([1]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        } finally {
          compiled.destroy();
        }
      }
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
