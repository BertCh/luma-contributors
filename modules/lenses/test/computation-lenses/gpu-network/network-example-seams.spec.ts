// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUElementwise} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkCoarsening} from '@luma-contributors/analysis/gpu-network';
import {requireTestDevice} from '../../../../analysis/src/testing/require-test-device';
import {createOracleContext} from '../../../../analysis/test/gpu-network/adjacency-matrix/network-output-oracle-test-utils';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-network/network-cost-matrix.example';

it.each([
  false,
  true
])('matrix example preserves intersection access and downstream-only interiors (interior=%s, UNW-02)', async interior => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  try {
    const example = await runExample(device, {
      graph,
      roads: {
        offsets: input('offsets', 'uint32', [0, 2, 2, 2]),
        neighbors: input('neighbors', 'uint32', [1, 2]),
        weights: input('weights', 'float32', [10, 1]),
        nodePositions: input('nodes', 'float32x2', [0, 0, 0.01, 0, 0, 0.01])
      },
      facilityPositions: input('facility', 'float32x2', interior ? [0.005, 0] : [0, 0])
    });
    const observed = output('observed-costs', 'float32', 3);
    graph.add(new GPUElementwise({operation: 'copy', input: example.costs, output: observed}));
    context.run();
    const costs = await read(observed);
    expect(costs).toHaveLength(3);
    // Hand routes: intersection 0 can take either outgoing edge. The one-way midpoint
    // can reach only node 1, paying half its edge's cost; nodes 0 and 2 are unreachable.
    if (interior) {
      expect(costs[0]).toBe(Infinity);
      expect(Number.isFinite(costs[1])).toBe(true);
      expect(costs[1]).toBeCloseTo(5, 4);
      expect(costs[2]).toBe(Infinity);
    } else {
      expect(costs.every(Number.isFinite)).toBe(true);
      expect(costs).toEqual([0, 10, 1]);
    }
  } finally {
    context.destroy();
  }
});

it.each([
  true,
  false
])('a valid iteration-limited partition keeps its quotient provisional without overflow (directed=%s, CPL-14)', async directed => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  try {
    // Solver summary fixture: valid partition, not converged, one level and one sweep.
    const solverSummary = input('solver-summary', 'uint32', [1, 0, 1, 1]);
    const labelsComplete = graph.createDataView(solverSummary.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const complete = output('labels-complete', 'uint32');
    const overflow = output('overflow', 'uint32');
    const candidateOverflow = output('candidate-overflow', 'uint32');
    const groupNodeCount = output('group-counts', 'uint32', 2);
    const count = output('count', 'uint32');
    const summary = output('quotient-summary', 'uint32', 8);
    const intraWeights = output('intra-weights', 'float32', 2);
    graph.add(
      new GPUNetworkCoarsening({
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 2]),
          neighbors: input('neighbors', 'uint32', [1, 0]),
          // Match the showcase's symmetric quotient: one counted unordered pair
          // carries both directed arc weights, preserving total scheduled weight 4.
          weights: input('weights', 'float32', directed ? [2, 2] : [4, 4])
        },
        directed,
        labels: input('labels', 'uint32', [0, 0]),
        labelsComplete,
        groupCapacity: 2,
        output: {
          groupNodeCount,
          groupIntraWeight: intraWeights,
          summary,
          labelsComplete: complete,
          edges: {
            ids: output('source-groups', 'uint32', 2),
            status: {count, overflow, candidateOverflow}
          },
          edgeTargetGroups: output('target-groups', 'uint32', 2),
          edgeCounts: output('edge-counts', 'uint32', 2)
        }
      })
    );
    context.run();
    expect(await read(groupNodeCount)).toEqual([2, 0]);
    expect(await read(count)).toEqual([0]);
    const countedEdges = directed ? 2 : 1;
    expect(await read(summary)).toEqual([2, 0, countedEdges, 0, countedEdges, 0, 1, 0]);
    const weights = await read(intraWeights);
    expect(weights.every(Number.isFinite)).toBe(true);
    expect(weights).toEqual([4, 0]);
    expect(await read(overflow)).toEqual([0]);
    expect(await read(candidateOverflow)).toEqual([0]);
    expect(await read(complete)).toEqual([0]);
    context.write(solverSummary, [1, 1, 1, 1]);
    context.run();
    expect(await read(complete)).toEqual([1]);
  } finally {
    context.destroy();
  }
});
