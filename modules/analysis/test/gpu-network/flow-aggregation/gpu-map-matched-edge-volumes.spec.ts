// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/index';
import {
  GPUMapMatchedEdgeVolumes,
  getGPUFlowPairZones
} from '../../../src/gpu-network/flow-aggregation/index';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';

const SIDE = 4;
const SPACING = 100;
/** Directed edges of a 4 x 4 grid in CSR order (right, left, up, down per node). */
function buildGrid() {
  const nodePositions: number[] = [];
  const offsets = [0];
  const neighbors: number[] = [];
  const edgeIds = new Map<string, number>();
  for (let row = 0; row < SIDE; row++) {
    for (let column = 0; column < SIDE; column++) {
      nodePositions.push(column * SPACING, row * SPACING);
      const node = row * SIDE + column;
      const targets: number[] = [];
      if (column + 1 < SIDE) targets.push(node + 1);
      if (column > 0) targets.push(node - 1);
      if (row + 1 < SIDE) targets.push(node + SIDE);
      if (row > 0) targets.push(node - SIDE);
      for (const target of targets) {
        edgeIds.set(`${node}>${target}`, neighbors.length);
        neighbors.push(target);
      }
      offsets.push(neighbors.length);
    }
  }
  return {nodePositions, offsets, neighbors, edgeIds};
}

const GRID = buildGrid();
const node = (column: number, row: number) => row * SIDE + column;

/** Ground truth: the directed edge rows of a node path. */
function getPathEdges(path: number[]): number[] {
  return path.slice(1).map((target, index) => GRID.edgeIds.get(`${path[index]}>${target}`)!);
}

/** Fix one meter off the middle of a directed edge. */
function getEdgeFix(edge: number, path: number[]): [number, number] {
  const edgeIndex = getPathEdges(path).indexOf(edge);
  const [from, to] = [path[edgeIndex], path[edgeIndex + 1]];
  const [x0, y0] = [GRID.nodePositions[2 * from], GRID.nodePositions[2 * from + 1]];
  const [x1, y1] = [GRID.nodePositions[2 * to], GRID.nodePositions[2 * to + 1]];
  const normal = [-(y1 - y0), x1 - x0].map(value => value / SPACING);
  return [(x0 + x1) / 2 + normal[0], (y0 + y1) / 2 + normal[1]];
}

type Drive = {
  /** Node path of one sub-track. */
  path: number[];
  /** Indices (into the path's edges) of the edges that get a fix. */
  fixedEdges: number[];
};

/** Fixes for a track made of sub-tracks, with a fix far from the network between them. */
function createTrack(drives: Drive[]) {
  const fixes: [number, number][] = [];
  const expectedEntries: number[] = [];
  drives.forEach((drive, driveIndex) => {
    const edges = getPathEdges(drive.path);
    for (const edgeIndex of drive.fixedEdges) fixes.push(getEdgeFix(edges[edgeIndex], drive.path));
    // The route runs from the first to the last fixed edge.
    expectedEntries.push(...edges.slice(drive.fixedEdges[0], drive.fixedEdges.at(-1)! + 1));
    if (driveIndex + 1 < drives.length) {
      fixes.push([5000, 5000]);
      expectedEntries.push(GPU_MAP_MATCHING_NONE);
    }
  });
  return {fixes, expectedEntries};
}

const TRACKS = [
  // Along the bottom row and up the right column; fixes on every other edge, so the hops cross
  // the edges between them.
  createTrack([
    {
      path: [node(0, 0), node(1, 0), node(2, 0), node(3, 0), node(3, 1), node(3, 2), node(3, 3)],
      fixedEdges: [0, 2, 4]
    }
  ]),
  // The right column downward and left along the bottom row (the opposite directions).
  createTrack([
    {
      path: [node(3, 3), node(3, 2), node(3, 1), node(3, 0), node(2, 0), node(1, 0)],
      fixedEdges: [0, 2, 4]
    }
  ]),
  // Two sub-tracks separated by a fix off the network: no adjacency across the break.
  createTrack([
    {path: [node(0, 1), node(1, 1), node(2, 1), node(3, 1)], fixedEdges: [0, 2]},
    {path: [node(0, 3), node(0, 2), node(0, 1), node(0, 0)], fixedEdges: [0, 2]}
  ]),
  // The first track's first two edges again, so the shared edges get a volume of two.
  createTrack([{path: [node(0, 0), node(1, 0), node(2, 0)], fixedEdges: [0, 1]}])
];
const CAPACITY = 16;

/** Reference volumes and movements of the ground-truth edge lists, by plain counting. */
function countReference() {
  const volumes = new Array(GRID.neighbors.length).fill(0);
  const transitions = new Map<string, number>();
  for (const {expectedEntries} of TRACKS) {
    expectedEntries.forEach((edge, index) => {
      if (edge === GPU_MAP_MATCHING_NONE) return;
      volumes[edge]++;
      const next = expectedEntries[index + 1];
      if (next !== undefined && next !== GPU_MAP_MATCHING_NONE) {
        const key = `${edge}>${next}`;
        transitions.set(key, (transitions.get(key) ?? 0) + 1);
      }
    });
  }
  return {volumes, transitions};
}

async function runMatchedVolumes(options: {capacity: number; transitions: boolean}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const points = TRACKS.flatMap(track => track.fixes);
  const trackOffsets = [0];
  for (const track of TRACKS) trackOffsets.push(trackOffsets.at(-1)! + track.fixes.length);
  const trackCount = TRACKS.length;
  const edgeCount = GRID.neighbors.length;
  const edges = harness.output('uint32', trackCount * options.capacity);
  const counts = harness.output('uint32', trackCount);
  const matchedOverflow = harness.output('uint32', 1);
  const pointCount = points.length;
  const matching = new GPUMapMatching({
    positions: harness.input('float32x2', Float32Array.from(points.flat())),
    trackOffsets: harness.input('uint32', Uint32Array.from(trackOffsets)),
    topology: {
      nodePositions: harness.input('float32x2', Float32Array.from(GRID.nodePositions)),
      offsets: harness.input('uint32', Uint32Array.from(GRID.offsets)),
      neighbors: harness.input('uint32', Uint32Array.from(GRID.neighbors))
    },
    parameters: harness.input(
      'float32',
      getGPUMapMatchingParameterValues({sigma: 5, beta: 20, searchRadius: 20})
    ),
    cellSize: 50,
    bounds: {minimum: [-20, -20], maximum: [320, 320]},
    traversedEdgeCapacity: options.capacity,
    output: {
      matchedEdges: harness.output('uint32', pointCount),
      breaks: harness.output('uint32', pointCount),
      snapDistances: harness.output('float32', pointCount),
      trackLogLikelihoods: harness.output('float32', trackCount),
      matchedCount: harness.output('uint32', 1),
      breakCount: harness.output('uint32', 1),
      status: {overflow: matchedOverflow},
      traversedEdges: {edges, counts}
    }
  });
  const edgeVolumes = harness.output('uint32', edgeCount);
  const truncated = harness.output('uint32', 1);
  const pairCapacity = 64;
  const K = 32;
  const transitionOutput = options.transitions
    ? {
        ids: harness.output('uint32', K),
        flowOriginZoneIds: harness.output('uint32', K),
        flowDestinationZoneIds: harness.output('uint32', K),
        flowCounts: harness.output('uint32', K),
        zoneOutCounts: harness.output('uint32', edgeCount),
        zoneInCounts: harness.output('uint32', edgeCount),
        status: {
          count: harness.output('uint32', 1),
          overflow: harness.output('uint32', 1),
          requiredCount: harness.output('uint32', 1),
          pairOverflow: harness.output('uint32', 1)
        }
      }
    : undefined;
  const volumes = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges, counts},
    trackCount,
    traversedEdgeCapacity: options.capacity,
    edgeCount,
    transitionPairCapacity: options.transitions ? pairCapacity : undefined,
    output: {edgeVolumes, truncated, transitions: transitionOutput}
  });
  const chain = {
    id: 'map-matching-then-edge-volumes',
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
      ...matching.getCommandNodes(graph),
      ...volumes.getCommandNodes(graph)
    ]
  } as GPUCommandNodeProducer;
  const result = await harness.run(chain);
  return {
    edges: result(edges),
    counts: result(counts),
    volumes: result(edgeVolumes),
    truncated: result(truncated)[0],
    transitions: transitionOutput && {
      count: result(transitionOutput.status.count)[0],
      ids: result(transitionOutput.ids),
      flowCounts: result(transitionOutput.flowCounts),
      zoneOutCounts: result(transitionOutput.zoneOutCounts),
      zoneInCounts: result(transitionOutput.zoneInCounts)
    }
  };
}

it('GPUMapMatchedEdgeVolumes counts the traversals of map-matched tracks including intermediate edges', async () => {
  const result = await runMatchedVolumes({capacity: CAPACITY, transitions: true});
  const reference = countReference();
  TRACKS.forEach((track, index) => {
    // The map matching recovers the driven route edge for edge, separators included.
    expect(
      result.edges.slice(index * CAPACITY, index * CAPACITY + track.expectedEntries.length),
      `track ${index} route`
    ).toEqual(track.expectedEntries);
    expect(result.counts[index]).toBe(track.expectedEntries.length);
  });
  expect(result.volumes).toEqual(reference.volumes);
  // Teeth: edges on both directions, an edge used twice, and edges no fix landed on.
  expect(Math.max(...reference.volumes)).toBe(2);
  expect(reference.volumes.filter(volume => volume > 0).length).toBeGreaterThan(15);
  expect(result.truncated).toBe(0);

  const transitions = result.transitions!;
  expect(transitions.count).toBe(reference.transitions.size);
  const edgeCount = GRID.neighbors.length;
  const reported = new Map<string, number>();
  for (let flow = 0; flow < transitions.count; flow++) {
    const [from, to] = getGPUFlowPairZones(transitions.ids[flow], edgeCount);
    reported.set(`${from}>${to}`, transitions.flowCounts[flow]);
  }
  expect(reported).toEqual(reference.transitions);
  // Movements leaving and entering each edge are volumes minus the sub-track ends and starts.
  const referenceOut = new Array(edgeCount).fill(0);
  const referenceIn = new Array(edgeCount).fill(0);
  for (const [key, count] of reference.transitions) {
    const [from, to] = key.split('>').map(Number);
    referenceOut[from] += count;
    referenceIn[to] += count;
  }
  expect(transitions.zoneOutCounts).toEqual(referenceOut);
  expect(transitions.zoneInCounts).toEqual(referenceIn);
  // No movement crosses the separator between the two sub-tracks of the third track.
  const firstSubTrackEnd = GRID.edgeIds.get(`${node(2, 1)}>${node(3, 1)}`)!;
  const secondSubTrackStart = GRID.edgeIds.get(`${node(0, 3)}>${node(0, 2)}`)!;
  expect(reported.has(`${firstSubTrackEnd}>${secondSubTrackStart}`)).toBe(false);
});

it('GPUMapMatchedEdgeVolumes reports truncated tracks and counts only the stored entries', async () => {
  const result = await runMatchedVolumes({capacity: 3, transitions: false});
  expect(result.truncated).toBe(1);
  // Track 0 needs five entries; three are stored and counted.
  expect(result.counts[0]).toBe(5);
  const stored = TRACKS.flatMap((track, index) =>
    result.edges
      .slice(index * 3, index * 3 + Math.min(track.expectedEntries.length, 3))
      .filter(edge => edge !== GPU_MAP_MATCHING_NONE)
  );
  const expected = new Array(GRID.neighbors.length).fill(0);
  for (const edge of stored) expected[edge]++;
  expect(result.volumes).toEqual(expected);
});
