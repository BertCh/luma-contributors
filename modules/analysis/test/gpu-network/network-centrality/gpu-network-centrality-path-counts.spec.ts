// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkCentrality} from '../../../src/gpu-network/network-centrality/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import type {NetworkEdge} from '../network-reachability/network-reachability-oracle';
import {buildCentralityCSRs} from './network-centrality-oracle';

type PathCountRun = {
  betweenness: Float32Array;
  edgeBetweenness: Float32Array;
  converged: number;
  zeroWeightCycle: number;
};

/** Runs betweenness and edge betweenness with both status flags. */
async function runBetweenness(
  device: Device,
  nodeCount: number,
  edges: readonly NetworkEdge[],
  pivots: readonly number[],
  maximumIterations: number
): Promise<PathCountRun> {
  const graph = new GPUCommandGraph(device, {id: 'centrality-path-counts'});
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const {forward, reverse} = buildCentralityCSRs(nodeCount, edges);
  const topologyViews = (name: string, csr: typeof forward) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      input(csr.offsets),
      'uint32',
      nodeCount + 1
    ),
    neighbors: importGraphBuffer(
      graph,
      `${name}-neighbors`,
      input(csr.neighbors),
      'uint32',
      csr.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      `${name}-weights`,
      input(csr.weights),
      'float32',
      csr.weights.length
    )
  });
  const betweennessBuffer = output(nodeCount);
  const edgeBetweennessBuffer = output(edges.length);
  const convergedBuffer = output(1);
  const zeroWeightCycleBuffer = output(1);
  graph.add(
    new GPUNetworkCentrality({
      id: 'centrality',
      topology: topologyViews('forward', forward),
      reverseTopology: topologyViews('reverse', reverse),
      pivotNodes: importGraphBuffer(
        graph,
        'pivots',
        input(Uint32Array.from(pivots)),
        'uint32',
        pivots.length
      ),
      laneCount: pivots.length,
      maximumIterations,
      output: {
        betweenness: importGraphBuffer(
          graph,
          'betweenness',
          betweennessBuffer,
          'float32',
          nodeCount
        ),
        edgeBetweenness: importGraphBuffer(
          graph,
          'edge-betweenness',
          edgeBetweennessBuffer,
          'float32',
          edges.length
        ),
        status: {
          converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1),
          zeroWeightCycle: importGraphBuffer(
            graph,
            'zero-weight-cycle',
            zeroWeightCycleBuffer,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: PathCountRun = {
    betweenness: Float32Array.from(await readFloat32(betweennessBuffer, nodeCount)),
    edgeBetweenness: Float32Array.from(await readFloat32(edgeBetweennessBuffer, edges.length)),
    converged: (await readUint32(convergedBuffer, 1))[0],
    zeroWeightCycle: (await readUint32(zeroWeightCycleBuffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUNetworkCentrality counts paths across a zero-weight edge like networkx (network-paths-02)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Review network-paths-02: s -1-> a -1-> m -0-> t -1-> u with pivot s. Every node lies on the
  // one shortest path to each node after it, so the dependencies of s are the numbers of nodes
  // after each node: a 3, m 2, t 1. networkx.betweenness_centrality_subset(G, sources=[s],
  // targets=list(G), weight='weight') gives the same. The kernel used to give a 1, m 0, t 1.
  const edges: NetworkEdge[] = [
    [0, 1, 1],
    [1, 2, 1],
    [2, 3, 0],
    [3, 4, 1]
  ];
  const run = await runBetweenness(device, 5, edges, [0], 16);
  expect(run.converged).toBe(1);
  expect(run.zeroWeightCycle).toBe(0);
  expect(Array.from(run.betweenness)).toEqual([0, 3, 2, 1, 0]);
  // Edge dependencies: the number of nodes at or after the edge's far end.
  expect(Array.from(run.edgeBetweenness)).toEqual([4, 3, 2, 1]);
});

it('GPUNetworkCentrality counts every path of an acyclic zero-weight plateau', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // s -1-> m, then zero-weight edges m -> t, m -> x, x -> t, then t -1-> u. Nodes s, m, x, t, u are
  // 0 to 4. Shortest paths from s: one to m and to x; two to t (direct, and through x) and to u.
  // Dependencies of s by hand: m is on every path to x, t and u: 3. x is on one of the two paths
  // to t and to u: 1/2 + 1/2 = 1. t is on both paths to u: 1. A self-loop of weight 0 on m is
  // never on a shortest path and changes nothing.
  const edges: NetworkEdge[] = [
    [0, 1, 1],
    [1, 3, 0],
    [1, 2, 0],
    [2, 3, 0],
    [3, 4, 1],
    [1, 1, 0]
  ];
  const run = await runBetweenness(device, 5, edges, [0], 16);
  expect(run.converged).toBe(1);
  expect(run.zeroWeightCycle).toBe(0);
  expect(Array.from(run.betweenness)).toEqual([0, 3, 1, 1, 0]);
});

it('GPUNetworkCentrality reports a zero-weight cycle instead of a wrong betweenness', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // s -1-> a, an undirected zero-weight edge a <-> b (both directions listed), b -1-> c. The two
  // zero-weight slots form a cycle between nodes of equal cost: the path count is not defined, and
  // the result must say so. A converged flag of 1 here was the silent failure of the review.
  const edges: NetworkEdge[] = [
    [0, 1, 1],
    [1, 2, 0],
    [2, 1, 0],
    [2, 3, 1]
  ];
  const run = await runBetweenness(device, 4, edges, [0], 16);
  expect(run.zeroWeightCycle).toBe(1);
  expect(run.converged).toBe(0);
  // The same network with a positive weight on the connector is fine: a 2, b 1.
  const positive = await runBetweenness(
    device,
    4,
    edges.map(([from, to, weight]): NetworkEdge => [from, to, weight === 0 ? 0.5 : weight]),
    [0],
    16
  );
  expect(positive.zeroWeightCycle).toBe(0);
  expect(positive.converged).toBe(1);
  expect(Array.from(positive.betweenness)).toEqual([0, 2, 1, 0]);
});

it('GPUNetworkCentrality leaves an absorbed positive weight out instead of calling it a zero-weight cycle (fr-fields-02)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fix review fr-fields-02. Undirected path b -1- a -1- s -16777216- u -1- v -1- x, nodes 0 to 5,
  // pivot s. Float32 cannot add 1 at 2^24: fround(16777216 + 1) = 16777216, so u, v and x all cost
  // 16777216. The weights of 1 beyond u are positive, so their edges are not tight in either
  // direction: v and x are roots with one path, nothing is unordered, and the flag stays 0.
  // (With every edge of equal float32 cost taken as tight, u and v waited for each other, the
  // flag was 1, converged 0, and no positive weight could cure it.)
  // By hand under the documented rule: a lies on the path to b, so its dependency is 1; u has no
  // tight edge beyond it, so 0. Exact arithmetic would give u 2 and v 1: that is the float32
  // limit the class documents, not something the status flags report.
  expect(Math.fround(16777216 + 1)).toBe(16777216);
  const undirected = (from: number, to: number, weight: number): NetworkEdge[] => [
    [from, to, weight],
    [to, from, weight]
  ];
  const edges: NetworkEdge[] = [
    ...undirected(0, 1, 1),
    ...undirected(1, 2, 1),
    ...undirected(2, 3, 16777216),
    ...undirected(3, 4, 1),
    ...undirected(4, 5, 1)
  ];
  const run = await runBetweenness(device, 6, edges, [2], 16);
  expect(run.zeroWeightCycle).toBe(0);
  expect(run.converged).toBe(1);
  expect(Array.from(run.betweenness)).toEqual([0, 1, 0, 0, 0, 0]);
});

it('GPUNetworkCentrality does not mistake a round limit for a zero-weight cycle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A chain of 12 unit edges needs 13 one-hop rounds; 4 rounds truncate the path-count phase.
  const edges: NetworkEdge[] = Array.from({length: 12}, (_, node) => [node, node + 1, 1]);
  const run = await runBetweenness(device, 13, edges, [0], 4);
  expect(run.converged).toBe(0);
  expect(run.zeroWeightCycle).toBe(0);
});

/** Natural logarithms of the factorials `0!` to `count!` in float64. */
function getLogFactorials(count: number): Float64Array {
  const logFactorials = new Float64Array(count + 1);
  for (let value = 1; value <= count; value++) {
    logFactorials[value] = logFactorials[value - 1] + Math.log(value);
  }
  return logFactorials;
}

/**
 * Dependency of the corner `(0, 0)` of a `size x size` unit lattice on every node, straight from
 * the definition (not the Brandes recursion): the shortest paths between two lattice nodes are the
 * monotone staircases, `binomial(dx + dy, dx)` of them, so the dependency on `v = (column, row)` is
 * the sum over the targets `t` beyond `v` of `paths(corner, v) * paths(v, t) / paths(corner, t)`.
 * Float64 with log-factorials; the corner itself gets 0.
 */
function getCornerDependencies(size: number): Float64Array {
  const logFactorials = getLogFactorials(2 * size);
  const logPaths = (columns: number, rows: number) =>
    logFactorials[columns + rows] - logFactorials[columns] - logFactorials[rows];
  const dependencies = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      if (row === 0 && column === 0) {
        continue;
      }
      const logToNode = logPaths(column, row);
      let total = 0;
      for (let targetRow = row; targetRow < size; targetRow++) {
        for (let targetColumn = column; targetColumn < size; targetColumn++) {
          if (targetRow === row && targetColumn === column) {
            continue;
          }
          total += Math.exp(
            logToNode +
              logPaths(targetColumn - column, targetRow - row) -
              logPaths(targetColumn, targetRow)
          );
        }
      }
      dependencies[row * size + column] = total;
    }
  }
  return dependencies;
}

it('GPUNetworkCentrality keeps path counts beyond the float32 range on a lattice (network-paths-03)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Review network-paths-03: on a unit lattice of 67 x 67 nodes the corner-to-corner path count,
  // binomial(132, 66), passes the float32 maximum; the f32 path counts became Infinity and the
  // dependencies NaN with converged 1. Here 72 x 72: binomial(142, 71) is about 9e41.
  const size = 72;
  const nodeCount = size * size;
  const edges: NetworkEdge[] = [];
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const node = row * size + column;
      if (column + 1 < size) {
        edges.push([node, node + 1, 1], [node + 1, node, 1]);
      }
      if (row + 1 < size) {
        edges.push([node, node + size, 1], [node + size, node, 1]);
      }
    }
  }
  // Pivots: the two opposite corners. The longest tight chain has 2 * (size - 1) hops.
  const farCorner = nodeCount - 1;
  const run = await runBetweenness(device, nodeCount, edges, [0, farCorner], 2 * size);
  expect(run.converged).toBe(1);
  expect(run.zeroWeightCycle).toBe(0);

  const cornerDependencies = getCornerDependencies(size);
  let total = 0;
  let largestRelativeError = 0;
  for (let node = 0; node < nodeCount; node++) {
    // The far corner's dependencies are the near corner's under the point reflection of the lattice.
    const expected =
      (node === 0 ? 0 : cornerDependencies[node]) +
      (node === farCorner ? 0 : cornerDependencies[farCorner - node]);
    const actual = run.betweenness[node];
    expect(Number.isFinite(actual), `node ${node}`).toBe(true);
    total += actual;
    if (expected > 0) {
      largestRelativeError = Math.max(largestRelativeError, Math.abs(actual - expected) / expected);
    } else {
      expect(actual).toBe(0);
    }
  }
  // Documented bound h (7 + K) 2^-24 with h = 142 hops and tight out-degree K = 2, plus the pivots.
  expect(largestRelativeError).toBeLessThan((142 * 9 + 4) * 2 ** -24);
  // Closed form of the sum: each target at distance d from a pivot has d - 1 inner path nodes, so
  // one corner contributes sum(d - 1) = size^2 (size - 1) - (size^2 - 1).
  const expectedTotal = 2 * (size * size * (size - 1) - (size * size - 1));
  expect(Math.abs(total - expectedTotal) / expectedTotal).toBeLessThan(1e-4);
  // Symmetry: the lattice and the pivot pair are unchanged by transposition and by point reflection.
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const value = run.betweenness[row * size + column];
      const transposed = run.betweenness[column * size + row];
      const reflected = run.betweenness[farCorner - (row * size + column)];
      expect(Math.abs(value - transposed)).toBeLessThanOrEqual(1e-4 * Math.abs(value));
      expect(Math.abs(value - reflected)).toBeLessThanOrEqual(1e-4 * Math.abs(value));
    }
  }
});
