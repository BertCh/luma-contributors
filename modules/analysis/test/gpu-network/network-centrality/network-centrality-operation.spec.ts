// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';
import {
  GPUNetworkCentralityOperation,
  type NetworkCentralityOperationProps
} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildCentralityCSRs} from './network-centrality-oracle';

for (const scene of ['weighted tied paths, multiple pivots', 'zero cycle', 'negative radius']) {
  it(`centrality operation twins agree on every output (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const zeroCycle = scene === 'zero cycle';
    const edges: [number, number, number][] = zeroCycle
      ? [
          [0, 1, 1],
          [1, 2, 0],
          [2, 1, 0],
          [2, 3, 2]
        ]
      : [
          [0, 1, 1],
          [0, 2, 2],
          [1, 3, 1],
          [2, 3, 0],
          [3, 4, 3],
          [4, 0, 1]
        ];
    const {forward, reverse} = buildCentralityCSRs(5, edges);
    const props: NetworkCentralityOperationProps = {
      id: 'centrality',
      nodeCount: 5,
      edgeCount: edges.length,
      reverseEdgeCount: edges.length,
      pivotCount: 3,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      reverseTopology: {
        offsets: 'reverseOffsets',
        neighbors: 'reverseNeighbors',
        weights: 'reverseWeights'
      },
      pivotNodes: 'pivots',
      costLimit: 'radius',
      laneCount: 2,
      maximumIterations: 8,
      betweennessScale: 1.5,
      output: {
        betweenness: 'between',
        edgeBetweenness: 'edges',
        closeness: 'close',
        harmonicCloseness: 'harmonic',
        reachedCounts: 'reached',
        status: {
          converged: 'converged',
          zeroWeightCycle: 'cycle',
          iterationLimitReached: 'limitReached'
        }
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: forward.offsets},
      neighbors: {format: 'uint32' as const, values: forward.neighbors},
      weights: {format: 'float32' as const, values: forward.weights},
      reverseOffsets: {format: 'uint32' as const, values: reverse.offsets},
      reverseNeighbors: {format: 'uint32' as const, values: reverse.neighbors},
      reverseWeights: {format: 'float32' as const, values: reverse.weights},
      pivots: {format: 'uint32' as const, values: Uint32Array.of(0, 3, 99)},
      radius: {
        format: 'float32' as const,
        values: Float32Array.of(scene === 'negative radius' ? -1 : 5)
      }
    };
    const results: Record<string, Record<string, OperationGraphRows>> = {};
    const outputs = [
      'between',
      'edges',
      'close',
      'harmonic',
      'reached',
      'converged',
      'cycle',
      'limitReached'
    ];
    for (const location of ['gpu', 'cpu'])
      results[location] = await runOperationGraph(device, {
        inputs,
        operations: [
          location === 'gpu'
            ? new GPUNetworkCentralityOperation(props)
            : new CPUNetworkCentralityOperation(props)
        ],
        outputs
      });
    for (const port of outputs) {
      expect(results['gpu'][port].length).toBe(results['cpu'][port].length);
      if (['reached', 'converged', 'cycle', 'limitReached'].includes(port))
        expect(Array.from(results['gpu'][port])).toEqual(Array.from(results['cpu'][port]));
      else
        for (let row = 0; row < results['cpu'][port].length; row++) {
          const expected = results['cpu'][port][row];
          // Relative 2e-5; exact zeros must remain numerically zero (no sign comparison).
          expect(Math.abs(results['gpu'][port][row] - expected)).toBeLessThanOrEqual(
            2e-5 * Math.abs(expected)
          );
        }
    }
    expect(results['cpu']['limitReached'][0]).toBe(0);
    expect(results['cpu']['cycle'][0]).toBe(Number(zeroCycle));
    expect(results['cpu']['converged'][0]).toBe(Number(!zeroCycle));
    expect(results['cpu']['reached'][2]).toBe(0);
    if (scene === 'negative radius')
      expect(Array.from(results['cpu']['reached'])).toEqual([0, 0, 0]);
    else if (!zeroCycle) expect(results['cpu']['between'].some(value => value > 0)).toBe(true);
  });
}
