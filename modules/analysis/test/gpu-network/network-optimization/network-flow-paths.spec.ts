// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  CPUNetworkFlowOperation,
  computeNetworkFlowOnCPU
} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const solver of ['reference', 'parallel'] as const)
  for (const scene of [
    'diamond',
    'budget-one',
    'budget-zero',
    'zero-demand',
    ...(solver === 'parallel' ? ['search-limit' as const] : []),
    'overflow',
    'invalid',
    'nonunique',
    'rerouted'
  ] as const)
    it(`NET-10 ${solver} flow paths match every twin output (${scene})`, async () => {
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
        capacities: [1, 1, 1, 1],
        demands: scene === 'zero-demand' ? [0, 0, 0, 0] : [-2, 0, 0, 2],
        costs: [0, 0, 0, 0]
      });
      if (scene === 'nonunique') {
        prepared.nodeCount = 7;
        prepared.edges = Uint32Array.of(0, 1, 0, 2, 1, 3, 2, 3, 3, 4, 3, 5, 4, 6, 5, 6);
        prepared.capacities = new Uint32Array(8).fill(1);
        prepared.costs = new Uint32Array(8);
        prepared.parameters = Uint32Array.of(0, 6, 2);
      }
      if (scene === 'rerouted') {
        prepared.nodeCount = 6;
        prepared.edges = Uint32Array.of(0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5);
        prepared.capacities = new Uint32Array(8).fill(1);
        prepared.costs = Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0);
        prepared.parameters = Uint32Array.of(0, 5, 2);
      }
      if (scene === 'invalid') prepared.edges[0] = 99;
      const pathCapacity = scene === 'overflow' ? 1 : 3;
      const maximumAugmentations = scene === 'budget-zero' ? 0 : scene === 'budget-one' ? 1 : 8;
      const props: NetworkFlowOperationProps = {
        nodeCount: prepared.nodeCount,
        edgeCount: prepared.capacities.length,
        pathCapacity,
        solver,
        mode: solver === 'parallel' ? 'successive-shortest-path' : undefined,
        maximumAugmentations,
        maximumRelaxationRounds: solver === 'parallel' && scene === 'search-limit' ? 1 : undefined,
        edges: 'edges',
        capacities: 'capacities',
        costs: 'costs',
        parameters: 'parameters',
        output: {
          flow: 'flow',
          cut: 'cut',
          summary: 'summary',
          exactCost: 'exact',
          status: {
            invalidCount: 'invalid',
            converged: 'converged',
            iterationCount: 'iterations',
            iterationLimitReached: 'limited'
          },
          paths: {
            edgeRows: 'pathEdges',
            lengths: 'pathLengths',
            volumes: 'pathVolumes',
            status: {
              count: 'pathCount',
              requiredCount: 'pathRequired',
              overflow: 'pathOverflow',
              invalidCount: 'pathInvalid',
              converged: 'pathConverged',
              iterationLimitReached: 'pathLimited'
            }
          }
        }
      };
      const expected = computeNetworkFlowOnCPU(prepared, {...props});
      const expectedPaths = expected.paths!;
      const expectedOutputs = {
        flow: expected.flow,
        cut: expected.cut,
        summary: expected.summary,
        exact: expected.exactCost,
        invalid: expected.status.invalidCount,
        converged: expected.status.converged,
        iterations: expected.status.iterationCount,
        limited: expected.status.iterationLimitReached,
        pathEdges: expectedPaths.edgeRows,
        pathLengths: expectedPaths.lengths,
        pathVolumes: expectedPaths.volumes,
        pathCount: expectedPaths.status.count,
        pathRequired: expectedPaths.status.requiredCount,
        pathOverflow: expectedPaths.status.overflow,
        pathInvalid: expectedPaths.status.invalidCount,
        pathConverged: expectedPaths.status.converged,
        pathLimited: expectedPaths.status.iterationLimitReached
      };
      const expectedCount =
        scene === 'zero-demand' ||
        scene === 'budget-zero' ||
        scene === 'invalid' ||
        (solver === 'parallel' && scene === 'search-limit')
          ? 0
          : scene === 'budget-one' || scene === 'overflow'
            ? 1
            : 2;
      // Hand-worked oracle: two unit paths, except when solving/truncation retains only one.
      expect(expectedPaths.status.count).toEqual(Uint32Array.of(expectedCount));
      expect(expectedPaths.volumes).toEqual(
        Uint32Array.from({length: pathCapacity}, (_, row) => Number(row < expectedCount))
      );
      if (scene === 'diamond')
        expect(expectedPaths.edgeRows.slice(0, 10)).toEqual(
          Uint32Array.of(4, 0, 2, 5, 0xffffffff, 4, 1, 3, 5, 0xffffffff)
        );
      if (scene === 'nonunique')
        expect(expectedPaths.edgeRows.slice(0, 12)).toEqual(
          Uint32Array.of(0, 2, 4, 6, 0xffffffff, 0xffffffff, 1, 3, 5, 7, 0xffffffff, 0xffffffff)
        );
      if (scene === 'rerouted')
        expect(expectedPaths.edgeRows.slice(0, 10)).toEqual(
          Uint32Array.of(0, 3, 7, 0xffffffff, 0xffffffff, 1, 4, 6, 0xffffffff, 0xffffffff)
        );
      for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
        const rows = await runOperationGraph(device, {
          inputs: {
            edges: {format: 'uint32x2', values: prepared.edges},
            capacities: {format: 'uint32', values: prepared.capacities},
            costs: {format: 'uint32', values: prepared.costs!},
            parameters: {format: 'uint32', values: prepared.parameters}
          },
          operations: [new Operation(props)],
          outputs: Object.keys(expectedOutputs)
        });
        // All numbers here are exact uint32 (the objective f32 bits are also exact at these sizes).
        for (const [name, values] of Object.entries(expectedOutputs))
          expect(rows[name], name).toEqual(values);
      }
    });

for (const mode of ['uncosted', 'primal-dual'] as const)
  for (const budget of [0, 1])
    it(`NET-10 ${mode} path decomposition propagates its phase budget ${budget}`, async () => {
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
        capacities: [1, 1, 1, 1],
        demands: [-2, 0, 0, 2],
        costs: mode === 'primal-dual' ? [0, 0, 0, 0] : undefined
      });
      for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
        const props: NetworkFlowOperationProps = {
          solver: 'parallel',
          nodeCount: prepared.nodeCount,
          edgeCount: prepared.capacities.length,
          pathCapacity: 2,
          mode: mode === 'primal-dual' ? 'primal-dual' : undefined,
          maximumAugmentations: 0,
          maximumRelabels: mode === 'uncosted' ? budget : undefined,
          maximumCostPhases: mode === 'primal-dual' ? budget : undefined,
          edges: 'edges',
          capacities: 'capacities',
          parameters: 'parameters',
          costs: prepared.costs ? 'costs' : undefined,
          output: {
            flow: 'flow',
            cut: 'cut',
            summary: 'summary',
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
        const expected = computeNetworkFlowOnCPU(prepared, props).paths!;
        const expectedOutputs = {
          pathEdges: expected.edgeRows,
          pathLengths: expected.lengths,
          pathVolumes: expected.volumes,
          pathCount: expected.status.count,
          pathRequired: expected.status.requiredCount,
          pathOverflow: expected.status.overflow,
          pathConverged: expected.status.converged,
          pathLimited: expected.status.iterationLimitReached,
          pathInvalid: expected.status.invalidCount
        };
        expect(expected.status.count).toEqual(Uint32Array.of(2 * budget));
        expect(expected.volumes).toEqual(Uint32Array.of(budget, budget));
        expect(expected.status.converged).toEqual(Uint32Array.of(budget));
        expect(expected.status.iterationLimitReached).toEqual(Uint32Array.of(1 - budget));
        const rows = await runOperationGraph(device, {
          inputs: {
            edges: {format: 'uint32x2', values: prepared.edges},
            capacities: {format: 'uint32', values: prepared.capacities},
            parameters: {format: 'uint32', values: prepared.parameters},
            ...(prepared.costs ? {costs: {format: 'uint32' as const, values: prepared.costs}} : {})
          },
          operations: [new Operation(props)],
          outputs: Object.keys(expectedOutputs)
        });
        for (const [name, values] of Object.entries(expectedOutputs))
          expect(rows[name], name).toEqual(values);
      }
    });
