// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createProps(
  nodeCount: number,
  edgeCount: number,
  pathCapacity: number
): NetworkFlowOperationProps {
  return {
    nodeCount,
    edgeCount,
    pathCapacity,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    output: {
      flow: 'flow',
      cut: 'cut',
      summary: 'summary',
      status: {
        converged: 'converged',
        iterationLimitReached: 'limited',
        iterationCount: 'iterations',
        invalidCount: 'invalid'
      },
      paths: {
        edgeRows: 'pathEdges',
        lengths: 'pathLengths',
        volumes: 'pathVolumes',
        status: {
          count: 'pathCount',
          requiredCount: 'pathRequired',
          overflow: 'pathOverflow',
          converged: 'pathConverged',
          iterationLimitReached: 'pathLimited',
          invalidCount: 'pathInvalid'
        }
      }
    }
  };
}
const OUTPUTS = [
  'flow',
  'cut',
  'summary',
  'converged',
  'limited',
  'iterations',
  'invalid',
  'pathEdges',
  'pathLengths',
  'pathVolumes',
  'pathCount',
  'pathRequired',
  'pathOverflow',
  'pathConverged',
  'pathLimited',
  'pathInvalid'
];

for (const costed of [false, true]) {
  for (const budget of [1, 2])
    it(`R1 ${costed ? 'primal-dual' : 'uncosted'} asymmetric diamond needs two phases (budget ${budget})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const prepared = prepareGPUNetworkFlowDemands({
        nodeCount: 4,
        edges: [
          [0, 1],
          [0, 2],
          [1, 3],
          [2, 3]
        ],
        capacities: [2, 2, 1, 1],
        demands: [-2, 0, 0, 2],
        costs: costed ? [0, 0, 0, 0] : undefined
      });
      const props: NetworkFlowOperationProps = {
        ...createProps(prepared.nodeCount, prepared.capacities.length, 2),
        solver: 'parallel',
        mode: costed ? 'primal-dual' : undefined,
        costs: costed ? 'costs' : undefined,
        maximumCostPhases: costed ? budget : undefined,
        maximumRelabels: costed ? undefined : budget
      };
      for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
        const rows = await runOperationGraph(device, {
          inputs: {
            edges: {format: 'uint32x2', values: prepared.edges},
            capacities: {format: 'uint32', values: prepared.capacities},
            parameters: {format: 'uint32', values: prepared.parameters},
            ...(prepared.costs ? {costs: {format: 'uint32' as const, values: prepared.costs}} : {})
          },
          operations: [new Operation(props)],
          outputs: OUTPUTS
        });
        expect(rows['pathCount']).toEqual(Uint32Array.of(budget));
        expect(rows['pathRequired']).toEqual(Uint32Array.of(budget));
        expect(rows['pathLengths']).toEqual(Uint32Array.of(4, budget === 2 ? 4 : 0));
        expect(rows['pathVolumes']).toEqual(Uint32Array.of(1, budget - 1));
        expect(rows['summary'][2]).toBe(budget);
        expect(rows['summary'][3]).toBe(2 - budget);
        expect(rows['iterations']).toEqual(Uint32Array.of(budget));
        for (const name of ['converged', 'pathConverged'])
          expect(rows[name]).toEqual(Uint32Array.of(budget - 1));
        for (const name of ['limited', 'pathLimited'])
          expect(rows[name]).toEqual(Uint32Array.of(2 - budget));
        for (const name of ['invalid', 'pathInvalid', 'pathOverflow'])
          expect(rows[name]).toEqual(Uint32Array.of(0));
        // Atomic scheduling may choose either first branch; each retained path must be one of
        // these two source-to-sink routes and its edge flow must be feasible after excess return.
        const branches = [
          Uint32Array.of(4, 0, 2, 5, 0xffffffff),
          Uint32Array.of(4, 1, 3, 5, 0xffffffff)
        ];
        const pathEdges = rows['pathEdges'] as Uint32Array;
        for (let path = 0; path < budget; path++)
          expect(
            branches.some(branch =>
              branch.every((edge, slot) => edge === pathEdges[path * 5 + slot])
            )
          ).toBe(true);
        if (budget === 1) {
          expect([
            [1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1]
          ]).toContainEqual(Array.from(rows['flow']));
          expect(pathEdges.slice(5)).toEqual(new Uint32Array(5).fill(0xffffffff));
        } else {
          expect(rows['flow']).toEqual(Uint32Array.of(1, 1, 1, 1, 2, 2));
          expect(pathEdges.slice(0, 5)).not.toEqual(pathEdges.slice(5));
        }
      }
    });
  it(`R2 ${costed ? 'primal-dual' : 'uncosted'} sink search serves the request despite a source dead end`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: NetworkFlowOperationProps = {
      ...createProps(4, 3, 1),
      solver: 'parallel',
      mode: costed ? 'primal-dual' : undefined,
      costs: costed ? 'costs' : undefined,
      maximumRelaxationRounds: 2,
      maximumCostPhases: costed ? 1 : undefined,
      maximumRelabels: costed ? undefined : 1
    };
    for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 0, 2, 2, 3)},
          capacities: {format: 'uint32', values: Uint32Array.of(1, 1, 1)},
          parameters: {format: 'uint32', values: Uint32Array.of(0, 1, 1)},
          ...(costed ? {costs: {format: 'uint32' as const, values: new Uint32Array(3)}} : {})
        },
        operations: [new Operation(props)],
        outputs: OUTPUTS
      });
      expect(rows['flow']).toEqual(Uint32Array.of(1, 0, 0));
      expect(rows['pathEdges']).toEqual(Uint32Array.of(0, 0xffffffff, 0xffffffff));
      expect(rows['pathCount']).toEqual(Uint32Array.of(1));
      expect(rows['pathLengths']).toEqual(Uint32Array.of(1));
      expect(rows['pathVolumes']).toEqual(Uint32Array.of(1));
      for (const name of ['converged', 'pathConverged'])
        expect(rows[name]).toEqual(Uint32Array.of(1));
      for (const name of ['limited', 'pathLimited']) expect(rows[name]).toEqual(Uint32Array.of(0));
      expect(rows['cut']).toEqual(Uint32Array.of(1, 0, 1, 1));
      expect(rows['summary'][7]).toBe(0); // The bounded cut is not certified; the served request is.
    }
  });
}

it('R5 BFS capacity truncation retains the direct edge instead of the first DFS route', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 1, 2, 0, 2)},
        capacities: {format: 'uint32', values: Uint32Array.of(1, 1, 1)},
        parameters: {format: 'uint32', values: Uint32Array.of(0, 2, 2)}
      },
      operations: [new Operation(createProps(3, 3, 1))],
      outputs: OUTPUTS
    });
    expect(rows['pathEdges']).toEqual(Uint32Array.of(2, 0xffffffff));
    expect(rows['pathLengths']).toEqual(Uint32Array.of(1));
    expect(rows['pathVolumes']).toEqual(Uint32Array.of(1));
    expect(rows['pathRequired']).toEqual(Uint32Array.of(2));
    expect(rows['pathOverflow']).toEqual(Uint32Array.of(1));
  }
});
