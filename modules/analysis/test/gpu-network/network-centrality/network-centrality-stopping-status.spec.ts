// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUNetworkCentrality} from '../../../src/gpu-network/network-centrality/gpu-network-centrality';
import {createCentralityBatchStatusNode} from '../../../src/gpu-network/network-centrality/network-centrality-passes';
import {
  createFrontierState,
  FRONTIER_CONTROL_TRUNCATED,
  FRONTIER_CONTROL_PARAMETER_0
} from '../../../src/gpu-network/network-reachability/network-frontier';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {buildCentralityCSRs} from './network-centrality-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Scene =
  | 'closeness-limit'
  | 'search-limit'
  | 'sigma-limit'
  | 'cycle'
  | 'complete'
  | 'mixed'
  | 'empty';

function createScene(scene: Scene) {
  const nodeCount = scene === 'mixed' ? 13 : scene === 'cycle' ? 5 : 9;
  const cycleEdges: [number, number, number][] = [
    [0, 1, 1],
    [1, 2, 0],
    [2, 1, 0],
    [2, 3, 2]
  ];
  const edges: [number, number, number][] =
    scene === 'cycle'
      ? cycleEdges
      : scene === 'mixed'
        ? [
            ...cycleEdges,
            ...Array.from(
              {length: 8},
              (_, node) => [node + 4, node + 5, 1] as [number, number, number]
            )
          ]
        : Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1]);
  return {
    ...buildCentralityCSRs(nodeCount, edges),
    nodeCount,
    pivots: Uint32Array.from(
      scene === 'mixed'
        ? [0, 4, 99]
        : scene === 'cycle'
          ? [0, 3, 99]
          : scene === 'empty'
            ? [99, 99, 99]
            : [0, 8, 99]
    ),
    maximumIterations: scene.endsWith('limit') ? 1 : scene === 'mixed' ? 4 : 16,
    localIterations: scene === 'closeness-limit' || scene === 'search-limit' ? 1 : 64
  };
}

for (const scene of [
  'closeness-limit',
  'search-limit',
  'sigma-limit',
  'cycle',
  'complete',
  'mixed',
  'empty'
] as const) {
  for (const withConverged of [false, true]) {
    it(`centrality distinguishes round exhaustion and zero cycles (${scene}, converged: ${withConverged})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createOptimizationHarness(device);
      const {forward, reverse, nodeCount, pivots, maximumIterations, localIterations} =
        createScene(scene);
      const hasBetweenness = scene !== 'closeness-limit';
      const status = {
        converged: withConverged ? harness.output('uint32', 1) : undefined,
        zeroWeightCycle: hasBetweenness ? harness.output('uint32', 1) : undefined,
        iterationLimitReached: harness.output('uint32', 1)
      };
      const output = {
        betweenness: hasBetweenness ? harness.output('float32', nodeCount) : undefined,
        closeness: harness.output('float32', pivots.length),
        reachedCounts: harness.output('uint32', pivots.length),
        status
      };
      const costLimit = harness.input('float32', Float32Array.of(Infinity));
      const result = await harness.run(
        new GPUNetworkCentrality({
          topology: {
            offsets: harness.input('uint32', forward.offsets),
            neighbors: harness.input('uint32', forward.neighbors),
            weights: harness.input('float32', forward.weights)
          },
          reverseTopology: {
            offsets: harness.input('uint32', reverse.offsets),
            neighbors: harness.input('uint32', reverse.neighbors),
            weights: harness.input('float32', reverse.weights)
          },
          pivotNodes: harness.input('uint32', pivots),
          costLimit,
          // Mixed uses distinct batches for cycle, round limit, then an inactive pivot. Other
          // scenes have a short final batch, whose inactive lane must not affect the reductions.
          laneCount: scene === 'mixed' ? 1 : 2,
          maximumIterations,
          localIterations,
          output
        }),
        first => {
          const stopped = scene.endsWith('limit') || scene === 'mixed';
          const cycle = scene === 'cycle' || scene === 'mixed';
          expect(first(status.iterationLimitReached)).toEqual([Number(stopped)]);
          if (status.zeroWeightCycle)
            expect(first(status.zeroWeightCycle)).toEqual([Number(cycle)]);
          if (status.converged)
            expect(first(status.converged)).toEqual([Number(!stopped && !cycle)]);
          if (scene === 'complete') {
            expect(first(output.reachedCounts)).toEqual([9, 1, 0]);
            // For the sole nontrivial chain pivot, node i serves the 8-i targets beyond it.
            const values = first(output.betweenness!);
            for (let node = 0; node < nodeCount; node++) {
              const expected = node === 0 ? 0 : nodeCount - node - 1;
              expect(Math.abs(values[node] - expected)).toBeLessThanOrEqual(2e-5 * expected);
            }
          } else if (scene === 'empty') expect(first(output.reachedCounts)).toEqual([0, 0, 0]);
          else if (scene === 'search-limit' || scene === 'closeness-limit') {
            expect(first(output.reachedCounts)).toEqual([2, 1, 0]);
          }
          harness.rewrite(costLimit, Float32Array.of(-1));
        }
      );
      expect(result(status.iterationLimitReached)).toEqual([0]);
      if (status.zeroWeightCycle) expect(result(status.zeroWeightCycle)).toEqual([0]);
      if (status.converged) expect(result(status.converged)).toEqual([1]);
      expect(result(output.reachedCounts)).toEqual([0, 0, 0]);
      expect(result(output.closeness)).toEqual([0, 0, 0]);
      if (output.betweenness)
        expect(result(output.betweenness)).toEqual(new Array(nodeCount).fill(0));
    });
  }
}

for (const [searchConverged, sigmaTruncated, deltaTruncated, unordered] of [
  [0, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 0, 1, 0],
  [1, 0, 0, 1],
  [1, 0, 0, 0],
  [0, 0, 0, 1]
]) {
  it(`centrality batch status preserves independent phase reasons (${searchConverged}, ${sigmaTruncated}, ${deltaTruncated}, ${unordered})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const controlValues = new Uint32Array(32);
    controlValues[FRONTIER_CONTROL_TRUNCATED] = sigmaTruncated;
    controlValues[16 + FRONTIER_CONTROL_TRUNCATED] = deltaTruncated;
    controlValues[16 + FRONTIER_CONTROL_PARAMETER_0] = unordered;
    const control = harness.input('uint32', controlValues);
    const batchStatus = harness.output('uint32', 3);
    const result = await harness.run({
      getCommandNodes(graph) {
        const state = createFrontierState(graph, 'probe', {
          nodeCount: 1,
          controlWordCount: 32,
          dispatchSlotCount: 1
        });
        return [
          createFillNode(graph, {
            id: 'seed',
            operation: 'status-probe',
            view: batchStatus,
            type: 'u32',
            value: `${searchConverged}u`
          }),
          createCentralityBatchStatusNode(graph, {
            id: 'status',
            state: {...state, control},
            batchStatus,
            sigmaPhase: {maxRounds: 1, stampBase: 0, controlWordOffset: 0, dispatchSlotOffset: 0},
            deltaPhase: {maxRounds: 1, stampBase: 1, controlWordOffset: 16, dispatchSlotOffset: 0}
          })
        ];
      }
    });
    expect(result(batchStatus)).toEqual([
      Number(Boolean(searchConverged && !sigmaTruncated && !deltaTruncated && !unordered)),
      Number(Boolean(unordered && !sigmaTruncated)),
      Number(Boolean(!searchConverged || sigmaTruncated || deltaTruncated))
    ]);
  });
}
