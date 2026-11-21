// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing-operation';
import {CPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {GPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {CPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const scene of ['complete', 'truncated', 'invalid', 'stopped'] as const) {
  it(`pins poisoned solver and path status independently on both backends (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Two parallel unit edges imply exactly two unit paths. A one-row path
    // capacity truncates one, without invalidating the completed solver.
    // Zero augmentations leaves both named paths unserved; source=sink is invalid.
    for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
      let operation = new Operation({
        nodeCount: 2,
        edgeCount: 2,
        pathCapacity: scene === 'truncated' ? 1 : 2,
        edges: 'edges',
        capacities: 'capacities',
        parameters: 'parameters',
        maximumAugmentations: scene === 'stopped' ? 0 : 2,
        output: {
          flow: 'flow',
          cut: 'cut',
          summary: 'summary',
          status: {
            converged: 'converged',
            iterationLimitReached: 'limit',
            invalidCount: 'invalid',
            iterationCount: 'iterations'
          },
          paths: {
            edgeRows: 'pathEdges',
            lengths: 'lengths',
            volumes: 'volumes',
            status: {
              count: 'count',
              requiredCount: 'required',
              overflow: 'overflow',
              converged: 'pathConverged',
              iterationLimitReached: 'pathLimit',
              invalidCount: 'pathInvalid'
            }
          }
        }
      });
      if (operation instanceof GPUNetworkFlowOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 0, 1)},
          capacities: {format: 'uint32', values: Uint32Array.of(1, 1)},
          parameters: {format: 'uint32', values: Uint32Array.of(0, scene === 'invalid' ? 0 : 1, 2)}
        },
        operations: [operation],
        outputs: [
          'converged',
          'limit',
          'invalid',
          'iterations',
          'count',
          'required',
          'overflow',
          'pathConverged',
          'pathLimit',
          'pathInvalid'
        ]
      });
      expect(['converged', 'limit', 'invalid', 'iterations'].map(name => rows[name][0])).toEqual(
        scene === 'invalid' ? [0, 0, 1, 0] : scene === 'stopped' ? [0, 1, 0, 0] : [1, 0, 0, 2]
      );
      expect(
        ['count', 'required', 'overflow', 'pathConverged', 'pathLimit', 'pathInvalid'].map(
          name => rows[name][0]
        )
      ).toEqual(
        {
          complete: [2, 2, 0, 1, 0, 0],
          truncated: [1, 2, 1, 1, 0, 0],
          invalid: [0, 0, 0, 0, 0, 1],
          stopped: [0, 0, 0, 0, 1, 0]
        }[scene]
      );
    }
  });
}

for (const mode of ['reference', 'scalable'] as const)
  for (const maximumRelaxations of [0, 4])
    it(`pins poisoned point-routing statuses 0 1 and 2 (${mode}, ${maximumRelaxations})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // One turn 0->1, costs 1 and 2, fractions 0 and 1: cost 3 after
      // settlement. An invalid source 99 is status 1 regardless of budget;
      // budget zero cannot examine the required turn and is status 2.
      for (const Operation of [GPUNetworkPointRoutingOperation, CPUNetworkPointRoutingOperation]) {
        let operation = new Operation({
          edgeCount: 2,
          queryCount: 2,
          mode,
          maximumRelaxations,
          turns: 'turns',
          costs: 'inputCosts',
          queries: 'queries',
          fractions: 'fractions',
          output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
        });
        if (operation instanceof GPUNetworkPointRoutingOperation)
          operation = poisonOperationOutputs(operation);
        const rows = await runOperationGraph(device, {
          inputs: {
            turns: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
            inputCosts: {format: 'float32', values: Float32Array.of(1, 2)},
            queries: {format: 'uint32x2', values: Uint32Array.of(0, 1, 99, 1)},
            fractions: {format: 'float32x2', values: Float32Array.of(0, 1, 0, 1)}
          },
          operations: [operation],
          outputs: ['costs', 'status']
        });
        expect(Array.from(rows['status'])).toEqual(maximumRelaxations === 0 ? [2, 1] : [0, 1]);
        expect(Array.from(rows['costs'])).toEqual(
          maximumRelaxations === 0 ? [Infinity, Infinity] : [3, Infinity]
        );
      }
    });

for (const maximumRelaxations of [1, 3])
  it(`pins both time-routing stopping outcomes after poisoning (${maximumRelaxations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Reverse-listed chain 0->1->2: after sweep 1, target 2 remains
    // unreachable; sweep 2 reaches it, sweep 3 certifies unchanged labels.
    for (const Operation of [GPUNetworkTimeRoutingOperation, CPUNetworkTimeRoutingOperation]) {
      let operation = new Operation({
        stateCount: 3,
        transitionCount: 2,
        stateVertices: 'vertices',
        transitions: 'transitions',
        costs: 'inputCosts',
        endpoints: 'endpoints',
        departure: 'departure',
        maximumRelaxations,
        output: {
          arrivals: 'arrivals',
          predecessorTransitions: 'parents',
          summary: 'summary',
          status: {
            converged: 'converged',
            iterationCount: 'iterations',
            iterationLimitReached: 'limit'
          }
        }
      });
      if (operation instanceof GPUNetworkTimeRoutingOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          vertices: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
          transitions: {format: 'uint32x2', values: Uint32Array.of(1, 2, 0, 1)},
          inputCosts: {format: 'float32x2', values: Float32Array.of(1, 0, 1, 0)},
          endpoints: {format: 'uint32', values: Uint32Array.of(0, 2)},
          departure: {format: 'float32', values: Float32Array.of(0)}
        },
        operations: [operation],
        outputs: ['converged', 'iterations', 'limit']
      });
      expect(['converged', 'iterations', 'limit'].map(name => rows[name][0])).toEqual(
        maximumRelaxations === 1 ? [0, 1, 1] : [1, 3, 0]
      );
    }
  });
