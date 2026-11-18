// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPU_NETWORK_FLOW_WORD} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';

for (const mode of ['reference', 'uncosted', 'primal-dual', 'successive-shortest-path'] as const)
  it(`network:G4 flow twins use the ${mode} controlling budget at zero and one`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const maximumAugmentations of [0, 1]) {
      for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
        const rows = await runOperationGraph(device, {
          inputs: {
            edges: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
            capacities: {format: 'uint32', values: Uint32Array.of(1)},
            parameters: {format: 'uint32', values: Uint32Array.of(0, 1, 1)},
            ...(mode === 'primal-dual' || mode === 'successive-shortest-path'
              ? {costs: {format: 'uint32' as const, values: Uint32Array.of(2)}}
              : {})
          },
          operations: [
            new Operation({
              nodeCount: 2,
              edgeCount: 1,
              solver: mode === 'reference' ? 'reference' : 'parallel',
              mode:
                mode === 'primal-dual' || mode === 'successive-shortest-path' ? mode : undefined,
              costs:
                mode === 'primal-dual' || mode === 'successive-shortest-path' ? 'costs' : undefined,
              maximumAugmentations:
                mode === 'uncosted' || mode === 'primal-dual' ? 0 : maximumAugmentations,
              maximumRelabels: mode === 'uncosted' ? maximumAugmentations : undefined,
              maximumCostPhases: mode === 'primal-dual' ? maximumAugmentations : undefined,
              edges: 'edges',
              capacities: 'capacities',
              parameters: 'parameters',
              output: {
                flow: 'flow',
                cut: 'cut',
                summary: 'summary',
                status: {converged: 'complete', iterationLimitReached: 'limited'}
              }
            })
          ],
          outputs: ['flow', 'summary', 'complete', 'limited']
        });
        expect(rows['flow']).toEqual(Uint32Array.of(maximumAugmentations));
        expect(rows['summary'][GPU_NETWORK_FLOW_WORD.complete]).toBe(maximumAugmentations);
        expect(rows['summary'][GPU_NETWORK_FLOW_WORD.augmentations]).toBe(maximumAugmentations);
        expect(rows['summary'][GPU_NETWORK_FLOW_WORD.unserved]).toBe(1 - maximumAugmentations);
        expect(rows['complete']).toEqual(Uint32Array.of(maximumAugmentations));
        expect(rows['limited']).toEqual(Uint32Array.of(1 - maximumAugmentations));
      }
    }
  });

it('solves expanded demands using the selected reference or parallel solver', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [nodeCount, capacity] of [
    [126, 1],
    [127, 1],
    [2, 0x1000000]
  ]) {
    const demands = new Int32Array(nodeCount);
    demands[0] = -1;
    demands[1] = 1;
    const prepared = prepareGPUNetworkFlowDemands({
      nodeCount,
      demands,
      edges: [[0, 1]],
      capacities: [capacity]
    });
    const props: NetworkFlowOperationProps = {
      nodeCount: prepared.nodeCount,
      edgeCount: prepared.capacities.length,
      solver: prepared.solver,
      edges: 'edges',
      capacities: 'capacities',
      parameters: 'parameters',
      output: {
        flow: 'flow',
        cut: 'cut',
        summary: 'summary',
        status: {invalidCount: 'invalid', converged: 'converged'}
      }
    };
    const rows = await runOperationGraph(device, {
      inputs: {
        edges: {format: 'uint32x2', values: prepared.edges},
        capacities: {format: 'uint32', values: prepared.capacities},
        parameters: {format: 'uint32', values: prepared.parameters}
      },
      operations: [new GPUNetworkFlowOperation(props)],
      outputs: ['flow', 'summary', 'invalid', 'converged']
    });
    expect(rows['flow']).toEqual(Uint32Array.of(1, 1, 1));
    expect(rows['summary'][GPU_NETWORK_FLOW_WORD.value]).toBe(1);
    expect(rows['summary'][GPU_NETWORK_FLOW_WORD.unserved]).toBe(0);
    expect(rows['invalid']).toEqual(Uint32Array.of(0));
    expect(rows['converged']).toEqual(Uint32Array.of(1));
  }
});

it('network:G4 a parallel phase commits multiple equal-length paths on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const costed of [false, true])
    for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 0, 1)},
          capacities: {format: 'uint32', values: Uint32Array.of(1, 1)},
          parameters: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
          ...(costed ? {costs: {format: 'uint32' as const, values: Uint32Array.of(2, 2)}} : {})
        },
        operations: [
          new Operation({
            nodeCount: 2,
            edgeCount: 2,
            solver: 'parallel',
            mode: costed ? 'primal-dual' : undefined,
            maximumAugmentations: 0,
            maximumCostPhases: costed ? 1 : undefined,
            maximumRelabels: costed ? undefined : 1,
            edges: 'edges',
            capacities: 'capacities',
            parameters: 'parameters',
            costs: costed ? 'costs' : undefined,
            output: {
              flow: 'flow',
              cut: 'cut',
              summary: 'summary',
              status: {converged: 'complete', iterationCount: 'phases'}
            }
          })
        ],
        outputs: ['flow', 'complete', 'phases']
      });
      expect(rows['flow']).toEqual(Uint32Array.of(1, 1));
      expect(rows['complete']).toEqual(Uint32Array.of(1));
      expect(rows['phases']).toEqual(Uint32Array.of(1));
    }
});

it('flow twins reject inconsistent CSR offsets and matching-length wrong neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: NetworkFlowOperationProps = {
    nodeCount: 3,
    edgeCount: 2,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    output: {
      flow: 'flow',
      cut: 'cut',
      summary: 'summary',
      exactCost: 'exact',
      status: {invalidCount: 'invalid', converged: 'converged', iterationLimitReached: 'limited'}
    }
  };
  for (const [offsets, neighbors] of [
    [
      [0, 2, 1, 2],
      [1, 2]
    ],
    [
      [0, 1, 1, 1],
      [1, 2]
    ],
    [
      [0, 1, 2, 2],
      [2, 2]
    ]
  ]) {
    for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 1, 2)},
          capacities: {format: 'uint32', values: Uint32Array.of(1, 1)},
          parameters: {format: 'uint32', values: Uint32Array.of(0, 2, 0xffffffff)},
          offsets: {format: 'uint32', values: Uint32Array.from(offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(neighbors)}
        },
        operations: [new Operation(props)],
        outputs: ['flow', 'cut', 'summary', 'exact', 'invalid', 'converged', 'limited']
      });
      expect(rows['flow']).toEqual(new Uint32Array(2));
      expect(rows['cut']).toEqual(new Uint32Array(3));
      expect(rows['summary']).toEqual(new Uint32Array(8));
      expect(rows['exact']).toEqual(new Uint32Array(2));
      expect(rows['invalid']).toEqual(Uint32Array.of(1));
      expect(rows['converged']).toEqual(Uint32Array.of(0));
      expect(rows['limited']).toEqual(Uint32Array.of(0));
    }
  }
});

for (const scene of [
  'reroute',
  'finite-request',
  'zero-capacity',
  'invalid',
  'large-cost',
  'uncosted'
] as const) {
  it(`flow twins compare every output (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const large = scene === 'large-cost';
    const props: NetworkFlowOperationProps = {
      id: 'flow',
      nodeCount: large ? 2 : 6,
      edgeCount: large ? 1 : 8,
      edges: 'edges',
      capacities: 'capacities',
      costs: scene === 'uncosted' ? undefined : 'costs',
      parameters: 'parameters',
      maximumAugmentations: 8,
      output: {
        flow: 'flow',
        cut: 'cut',
        summary: 'summary',
        exactCost: 'exact',
        status: {
          invalidCount: 'invalidCount',
          converged: 'converged',
          iterationCount: 'iterationCount',
          iterationLimitReached: 'iterationLimitReached'
        }
      }
    };
    const inputs = {
      edges: {
        format: 'uint32x2' as const,
        values: large
          ? Uint32Array.of(0, 1)
          : Uint32Array.of(
              0,
              scene === 'invalid' ? 99 : 1,
              0,
              2,
              1,
              3,
              1,
              4,
              2,
              3,
              2,
              4,
              3,
              5,
              4,
              5
            )
      },
      capacities: {
        format: 'uint32' as const,
        values: large
          ? Uint32Array.of(0xffffff)
          : new Uint32Array(8).fill(scene === 'zero-capacity' ? 0 : 1)
      },
      costs: {
        format: 'uint32' as const,
        values: large ? Uint32Array.of(32767) : Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0)
      },
      parameters: {
        format: 'uint32' as const,
        values: Uint32Array.of(0, large ? 1 : 5, scene === 'finite-request' ? 1 : 0xffffffff)
      }
    };
    const outputs = [
      'flow',
      'cut',
      'summary',
      'exact',
      'invalidCount',
      'converged',
      'iterationCount',
      'iterationLimitReached'
    ];
    const results: Record<string, Record<string, OperationGraphRows>> = {};
    for (const location of ['gpu', 'cpu'])
      results[location] = await runOperationGraph(device, {
        inputs,
        operations: [
          location === 'gpu'
            ? new GPUNetworkFlowOperation(props)
            : new CPUNetworkFlowOperation(props)
        ],
        outputs
      });
    for (const port of outputs) {
      expect(results['gpu'][port].length).toBe(results['cpu'][port].length);
      for (let row = 0; row < results['cpu'][port].length; row++) {
        const expected = results['cpu'][port][row];
        const actual = results['gpu'][port][row];
        if (port === 'summary' && row === GPU_NETWORK_FLOW_WORD.cost) {
          // Only the packed float32 objective permits rounding differences (relative 2e-5).
          const expectedCost = new Float32Array(Uint32Array.of(expected).buffer)[0];
          const actualCost = new Float32Array(Uint32Array.of(actual).buffer)[0];
          expect(Math.abs(actualCost - expectedCost)).toBeLessThanOrEqual(
            2e-5 * Math.abs(expectedCost)
          );
        } else expect(actual).toBe(expected);
      }
    }
    if (scene === 'reroute')
      expect(Array.from(results['cpu']['flow'])).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
    if (scene === 'large-cost')
      expect(BigInt(results['cpu']['exact'][0]) + (BigInt(results['cpu']['exact'][1]) << 32n)).toBe(
        0xffffffn * 32767n
      );
  });
}
