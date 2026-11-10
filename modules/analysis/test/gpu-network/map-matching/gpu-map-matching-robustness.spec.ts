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
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues,
  type GPUMapMatchingSettings
} from '../../../src/gpu-network/map-matching/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** A chain of one-way edges: edge `i` runs from node `i` to node `i + 1` along the x axis. */
type ChainFixture = {
  /** Node x positions (y is 0). */
  nodeXs: number[];
  /** Fix positions of all tracks, concatenated. */
  points: [number, number][];
  trackOffsets: number[];
  settings: GPUMapMatchingSettings;
  minimumFixSpacingSigmas?: number;
  traversedEdgeCapacity?: number;
};

type ChainResult = {
  edges: number[];
  breaks: number[];
  breakCount: number;
  matchedCount: number;
  logLikelihoods: number[];
  snapDistances: number[];
  overflow: number;
  traversedEdges: number[];
  traversedCounts: number[];
  traversedOverflow: number;
};

async function runChain(device: Device, fixture: ChainFixture): Promise<ChainResult> {
  const graph = new GPUCommandGraph(device, {id: 'map-matching-robustness'});
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? length * 2 : length);
    buffers.push(buffer);
    outputs[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const nodeCount = fixture.nodeXs.length;
  const edgeCount = nodeCount - 1;
  const pointCount = fixture.points.length;
  const trackCount = fixture.trackOffsets.length - 1;
  const capacity = fixture.traversedEdgeCapacity;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'map-matching-robustness-parameters',
    format: 'float32',
    length: 8,
    values: getGPUMapMatchingParameterValues(fixture.settings)
  });
  // Node i has the single out-edge i, except the last node.
  const offsets = Uint32Array.from({length: nodeCount + 1}, (_, node) => Math.min(node, edgeCount));
  const neighbors = Uint32Array.from({length: edgeCount}, (_, edge) => edge + 1);
  graph.add(
    new GPUMapMatching({
      id: 'match',
      positions: input('points', Float32Array.from(fixture.points.flat()), 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.from(fixture.trackOffsets), 'uint32'),
      topology: {
        nodePositions: input(
          'nodes',
          Float32Array.from(fixture.nodeXs.flatMap(x => [x, 0])),
          'float32x2'
        ),
        offsets: input('offsets', offsets, 'uint32'),
        neighbors: input('targets', neighbors, 'uint32')
      },
      parameters: parameterBuffer.importToGraph(graph),
      // Cells wide enough that no edge spans more than 16 of them.
      cellSize: 300,
      bounds: {
        minimum: [Math.min(...fixture.nodeXs) - 20, -20],
        maximum: [Math.max(...fixture.nodeXs) + 20, 20]
      },
      minimumFixSpacingSigmas: fixture.minimumFixSpacingSigmas,
      traversedEdgeCapacity: capacity,
      output: {
        matchedEdges: output('edges', 'uint32', pointCount),
        breaks: output('breaks', 'uint32', pointCount),
        snapDistances: output('distances', 'float32', pointCount),
        trackLogLikelihoods: output('ll', 'float32', trackCount),
        matchedCount: output('matched-count', 'uint32', 1),
        breakCount: output('break-count', 'uint32', 1),
        status: {overflow: output('overflow', 'uint32', 1)},
        ...(capacity
          ? {
              traversedEdges: {
                edges: output('traversed-edges', 'uint32', trackCount * capacity),
                counts: output('traversed-counts', 'uint32', trackCount),
                status: {overflow: output('traversed-overflow', 'uint32', 1)}
              }
            }
          : {})
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: ChainResult = {
    edges: [...(await readUint32(outputs['edges'], pointCount))],
    breaks: [...(await readUint32(outputs['breaks'], pointCount))],
    breakCount: (await readUint32(outputs['break-count'], 1))[0],
    matchedCount: (await readUint32(outputs['matched-count'], 1))[0],
    logLikelihoods: [...(await readFloat32(outputs['ll'], trackCount))],
    snapDistances: [...(await readFloat32(outputs['distances'], pointCount))],
    overflow: (await readUint32(outputs['overflow'], 1))[0],
    traversedEdges: capacity
      ? [...(await readUint32(outputs['traversed-edges'], trackCount * capacity))]
      : [],
    traversedCounts: capacity
      ? [...(await readUint32(outputs['traversed-counts'], trackCount))]
      : [],
    traversedOverflow: capacity ? (await readUint32(outputs['traversed-overflow'], 1))[0] : 0
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

const JITTER: ChainFixture = {
  // One one-way edge from x = 0 to x = 100. The third fix jitters 1.5 backward: within 2 sigma.
  nodeXs: [0, 100],
  points: [
    [20, 1],
    [40, 1],
    [38.5, 1],
    [60, 1],
    [80, 1]
  ],
  trackOffsets: [0, 5],
  settings: {sigma: 1, beta: 10, searchRadius: 5}
};

it('GPUMapMatching: backward jitter on a one-way edge breaks the track without the fix filter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, JITTER);
  // No route leaves the one-way edge and returns to it, so the backward fix has no transition.
  expect(result.breaks).toEqual([0, 0, 1, 0, 0]);
  expect(result.breakCount).toBe(1);
});

it('GPUMapMatching: minimumFixSpacingSigmas drops the jitter fix and keeps the track whole', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, {...JITTER, minimumFixSpacingSigmas: 2});
  expect(result.breaks).toEqual([0, 0, 0, 0, 0]);
  expect(result.breakCount).toBe(0);
  // The dropped fix is unmatched; the others sit on the only edge.
  expect(result.edges).toEqual([0, 0, GPU_MAP_MATCHING_NONE, 0, 0]);
  expect(result.matchedCount).toBe(4);
  expect(result.overflow).toBe(0);
});

it('GPUMapMatching: fixes are measured from the last retained fix, and the first is always kept', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, {
    ...JITTER,
    // Steps of 1.5 sigma: the second fix is dropped, the third is 3 sigma from the retained first.
    points: [
      [20, 1],
      [21.5, 1],
      [23, 1],
      [24.5, 1]
    ],
    trackOffsets: [0, 4],
    minimumFixSpacingSigmas: 2
  });
  expect(result.edges).toEqual([0, GPU_MAP_MATCHING_NONE, 0, GPU_MAP_MATCHING_NONE]);
  expect(result.breaks).toEqual([0, 0, 0, 0]);
});

const THREE_EDGES: ChainFixture = {
  // Edges 0, 1 and 2 on x = 0..300; fixes land on edges 0 and 2 only.
  nodeXs: [0, 100, 200, 300],
  points: [
    [50, 1],
    [250, 1],
    [150, 1]
  ],
  trackOffsets: [0, 2, 3],
  settings: {sigma: 5, beta: 20, searchRadius: 10, routeFactor: 3, routeSlack: 50},
  traversedEdgeCapacity: 8
};

it('GPUMapMatching: traversedEdges lists the intermediate edge that no fix lands on', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, THREE_EDGES);
  expect(result.edges).toEqual([0, 2, 1]);
  expect(result.traversedCounts).toEqual([3, 1]);
  expect(result.traversedEdges.slice(0, 3)).toEqual([0, 1, 2]);
  expect(result.traversedEdges.slice(8, 9)).toEqual([1]);
  expect(result.traversedOverflow).toBe(0);
});

it('GPUMapMatching: traversedEdges lists an edge once for fixes along it and separates sub-tracks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, {
    ...THREE_EDGES,
    // Track 0: three fixes on edge 0, one fix far from any edge (unmatched), then a fix on edge 2.
    points: [
      [20, 1],
      [50, 1],
      [80, 1],
      [150, 100],
      [280, 1]
    ],
    trackOffsets: [0, 5]
  });
  expect(result.edges).toEqual([0, 0, 0, GPU_MAP_MATCHING_NONE, 2]);
  expect(result.breaks).toEqual([0, 0, 0, 1, 1]);
  expect(result.traversedCounts).toEqual([3]);
  expect(result.traversedEdges.slice(0, 3)).toEqual([0, GPU_MAP_MATCHING_NONE, 2]);
});

it('GPUMapMatching: traversedEdges reports the needed count and overflow past its capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runChain(device, {...THREE_EDGES, traversedEdgeCapacity: 2});
  expect(result.traversedCounts).toEqual([3, 1]);
  expect(result.traversedOverflow).toBe(1);
  // Entries past the capacity are not stored and the next track's rows are untouched.
  expect(result.traversedEdges.slice(0, 2)).toEqual([0, 1]);
  expect(result.traversedEdges.slice(2, 3)).toEqual([1]);
});

it('GPUMapMatching: the log-likelihood of a long track stays accurate as scores grow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointCount = 2000;
  const sigma = Math.fround(0.01);
  const beta = Math.fround(1000);
  const points: [number, number][] = Array.from({length: pointCount}, (_, index) => [
    1 + 1.3 * index,
    1
  ]);
  const result = await runChain(device, {
    nodeXs: [0, 4000],
    points,
    trackOffsets: [0, pointCount],
    // Each step scores about -5003: the unnormalized total passes 1e7, where one float32 ULP is 1.
    settings: {sigma, beta, searchRadius: 5}
  });
  expect(result.breaks.every(value => value === 0)).toBe(true);
  // The model in float64 from the snap distances the matcher reports (a long edge makes the
  // shader's own distances differ from 1 by a few percent, which sigma = 0.01 magnifies).
  const emissionConstant = -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
  const expected =
    result.snapDistances.reduce(
      (sum, distance) => sum + emissionConstant - 0.5 * (distance / sigma) ** 2,
      0
    ) +
    (pointCount - 1) * -Math.log(beta);
  expect(Math.abs(expected)).toBeGreaterThan(9e6);
  expect(Math.abs(result.logLikelihoods[0] - expected)).toBeLessThan(8);
});
