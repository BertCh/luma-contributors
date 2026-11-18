// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {
  GPUNetworkReachabilityOperation,
  type NetworkReachabilityOperationProps
} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('network:G6 zero active rounds leave empty and seeded searches uncertified on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const startCount of [0, 1])
    for (const Operation of [GPUNetworkReachabilityOperation, CPUNetworkReachabilityOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 1)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1)},
          weights: {format: 'float32', values: Float32Array.of(1)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          startCount: {format: 'uint32', values: Uint32Array.of(startCount)},
          active: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [
          new Operation({
            nodeCount: 2,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            startNodes: 'starts',
            startCount: 'startCount',
            activeIterations: 'active',
            output: {
              costs: 'costs',
              status: {
                converged: 'converged',
                iterationLimitReached: 'limited',
                iterationCount: 'rounds'
              }
            }
          })
        ],
        outputs: ['costs', 'converged', 'limited', 'rounds']
      });
      expect(rows['costs']).toEqual(Float32Array.of(startCount ? 0 : Infinity, Infinity));
      expect(rows['converged']).toEqual(Uint32Array.of(0));
      expect(rows['limited']).toEqual(Uint32Array.of(1));
      expect(rows['rounds']).toEqual(Uint32Array.of(0));
    }
});
it('network:G7 reachability backends retain constructor rounds while activeIterations stays live', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUNetworkReachabilityOperation, CPUNetworkReachabilityOperation])
    for (const active of [3, 0]) {
      const props: NetworkReachabilityOperationProps = {
        nodeCount: 3,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        maximumIterations: 1,
        localIterations: 1,
        activeIterations: 'active',
        output: {costs: 'costs', status: {converged: 'complete'}}
      };
      const operation = new Operation(props);
      props.maximumIterations = 3;
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(active)}
        },
        operations: [operation],
        outputs: ['costs', 'complete']
      });
      expect(Array.from(result['costs'])).toEqual(
        active ? [0, 1, Infinity] : [0, Infinity, Infinity]
      );
      expect(Array.from(result['complete'])).toEqual([0]);
    }
});
import {
  getOverlappingRoutingTopology,
  getRoutingTopology
} from './network-routing-operation-test-utils';

for (const empty of [false, true])
  it(`reachability twins agree on every requested port (no valid starts: ${empty})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const topology = getRoutingTopology();
    const props: NetworkReachabilityOperationProps = {
      nodeCount: 9,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      startNodes: 'starts',
      startCosts: 'startCosts',
      startCount: 'startCount',
      costLimit: 'cutoff',
      costBreaks: 'breaks',
      costBreakCount: 4,
      maximumIterations: 8,
      maximumTieIterations: 8,
      output: {
        costs: 'costs',
        predecessors: 'parents',
        bands: 'bands',
        bandCounts: 'bandCounts',
        unresolvedCount: 'unresolved',
        status: {
          converged: 'converged',
          iterationLimitReached: 'limitReached',
          iterationCount: 'iterationRounds',
          relaxationCount: 'relaxedEdges',
          reachedEdgeCount: 'reachedEdges'
        }
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: topology.offsets},
      neighbors: {format: 'uint32' as const, values: topology.neighbors},
      weights: {format: 'float32' as const, values: topology.weights},
      starts: {format: 'uint32' as const, values: Uint32Array.of(empty ? 99 : 0, 7, 1)},
      startCosts: {format: 'float32' as const, values: Float32Array.of(0.5, 2, 0)},
      startCount: {format: 'uint32' as const, values: Uint32Array.of(empty ? 0 : 2)},
      cutoff: {format: 'float32' as const, values: Float32Array.of(6)},
      breaks: {format: 'float32' as const, values: Float32Array.of(1, 2, 3, 6)}
    };
    const outputs = [
      'costs',
      'parents',
      'bands',
      'bandCounts',
      'unresolved',
      'converged',
      'limitReached',
      'reachedEdges',
      'iterationRounds',
      'relaxedEdges'
    ];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkReachabilityOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUNetworkReachabilityOperation(props)],
      outputs
    });
    // Exact binary fractions give zero tolerance on results. Execution-dependent counters
    // agree exactly for an empty frontier; ordinary input checks each backend's work below.
    for (const output of outputs) {
      if (!empty && ['iterationRounds', 'relaxedEdges'].includes(output)) continue;
      expect(gpuRows[output].length).toBe(cpuRows[output].length);
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    }
    expect(Array.from(cpuRows['iterationRounds'])).toEqual([1]);
    if (empty) expect(Array.from(cpuRows['relaxedEdges'])).toEqual([0, 0]);
    else {
      expect(cpuRows['relaxedEdges'][0]).toBeGreaterThanOrEqual(13);
      expect(cpuRows['relaxedEdges'][1]).toBe(0);
    }
    if (empty) {
      expect(Array.from(gpuRows['iterationRounds'])).toEqual([1]);
      expect(Array.from(gpuRows['relaxedEdges'])).toEqual([0, 0]);
    } else {
      expect(gpuRows['iterationRounds'].length).toBe(1);
      expect(gpuRows['iterationRounds'][0]).toBeGreaterThanOrEqual(1);
      expect(gpuRows['iterationRounds'][0]).toBeLessThanOrEqual(props.maximumIterations!);
      expect(gpuRows['relaxedEdges'].length).toBe(2);
      expect(gpuRows['relaxedEdges'][1]).toBe(0);
      // Convergence scans all 13 slots at least once; chaotic revisits can add work.
      const relaxationCount = gpuRows['relaxedEdges'][0] + 2 ** 32 * gpuRows['relaxedEdges'][1];
      expect(relaxationCount).toBeGreaterThanOrEqual(13);
    }
    expect(Array.from(cpuRows['parents'])).toEqual(
      empty ? Array(9).fill(0xffffffff) : [0xffffffff, 2, 0, 1, 2, 3, 7, 0xffffffff, 0xffffffff]
    );
  });

it('reuses the same slot-indexed tie flag across overlapping CSR rows on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const topology = getOverlappingRoutingTopology();
  const props: NetworkReachabilityOperationProps = {
    nodeCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    startCosts: 'startCosts',
    maximumIterations: 8,
    maximumTieIterations: 8,
    localIterations: 4,
    output: {
      costs: 'costs',
      predecessors: 'parents',
      unresolvedCount: 'unresolved',
      status: {converged: 'converged', iterationLimitReached: 'limitReached'}
    }
  };
  const inputs = {
    offsets: {format: 'uint32' as const, values: topology.offsets},
    neighbors: {format: 'uint32' as const, values: topology.neighbors},
    weights: {format: 'float32' as const, values: topology.weights},
    starts: {format: 'uint32' as const, values: Uint32Array.of(0, 2)},
    startCosts: {format: 'float32' as const, values: Float32Array.of(1, 0)}
  };
  // The shared slot is marked by row 2, then reused by the smaller root 0 despite its higher cost.
  const expected: Record<string, number[]> = {
    costs: [1, 0, 0],
    parents: [0xffffffff, 0, 0xffffffff],
    unresolved: [0],
    converged: [1],
    limitReached: [0]
  };
  const outputs = Object.keys(expected);
  const gpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new GPUNetworkReachabilityOperation(props)],
    outputs
  });
  const cpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new CPUNetworkReachabilityOperation(props)],
    outputs
  });
  // Integer costs and all metadata compare exactly, using numeric equality for zero signs.
  for (const output of outputs) {
    expect(gpuRows[output].length).toBe(expected[output].length);
    expect(cpuRows[output].length).toBe(expected[output].length);
    for (let index = 0; index < expected[output].length; index++) {
      expect(gpuRows[output][index] === expected[output][index]).toBe(true);
      expect(cpuRows[output][index] === expected[output][index]).toBe(true);
    }
  }
});
