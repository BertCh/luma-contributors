// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';
import {
  FLOW_REUSE_EDGES,
  FLOW_REUSE_COSTS,
  FLOW_REUSE_CAPACITIES,
  expectFlowReuseRows
} from './network-flow-reuse.test-utils';

for (const solver of ['reference', 'parallel'] as const)
  for (const scenario of ['complete', 'path-overflow', 'solver-limit'] as const)
    it(`flow reuse ${solver} ${scenario}: two paths, one path, empty, invalid`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createOptimizationHarness(device);
      const pathCapacity = scenario === 'path-overflow' ? 1 : 2;
      const limited = scenario === 'solver-limit';
      const capacities = harness.input('uint32', FLOW_REUSE_CAPACITIES[0]);
      const parameters = harness.input('uint32', Uint32Array.of(0, 3, 0xffffffff));
      const rows = Object.fromEntries(
        Object.entries({
          flow: 4,
          invalid: 1,
          converged: 1,
          limited: 1,
          pathEdges: pathCapacity * 3,
          pathLengths: pathCapacity,
          pathVolumes: pathCapacity,
          pathCount: 1,
          pathRequired: 1,
          pathOverflow: 1,
          pathInvalid: 1,
          pathConverged: 1,
          pathLimited: 1
        }).map(([name, length]) => [name, harness.output('uint32', length)])
      );
      const contributor = new GPUNetworkFlow({
        nodeCount: 4,
        solver,
        mode: solver === 'parallel' ? 'successive-shortest-path' : undefined,
        maximumAugmentations: limited ? 1 : 4,
        maximumRelaxationRounds: solver === 'parallel' ? 4 : undefined,
        costs: harness.input('uint32', FLOW_REUSE_COSTS),
        edges: harness.input('uint32x2', FLOW_REUSE_EDGES),
        capacities,
        parameters,
        output: {
          flow: rows['flow'],
          cut: harness.output('uint32', 4),
          summary: harness.output('uint32', 8),
          status: {
            invalidCount: rows['invalid'],
            converged: rows['converged'],
            iterationLimitReached: rows['limited']
          },
          paths: {
            edgeRows: rows['pathEdges'],
            lengths: rows['pathLengths'],
            volumes: rows['pathVolumes'],
            status: {
              count: rows['pathCount'],
              requiredCount: rows['pathRequired'],
              overflow: rows['pathOverflow'],
              invalidCount: rows['pathInvalid'],
              converged: rows['pathConverged'],
              iterationLimitReached: rows['pathLimited']
            }
          }
        }
      });
      const finalRead = await harness.run(
        contributor,
        (read, stage) => {
          expectFlowReuseRows(
            (name, expected) => expect(read(rows[name]), `${stage} ${name}`).toEqual(expected),
            stage,
            pathCapacity,
            limited
          );
          harness.rewrite(capacities, FLOW_REUSE_CAPACITIES[stage + 1]);
          if (stage === 2) harness.rewrite(parameters, Uint32Array.of(99, 3, 0xffffffff));
        },
        4
      );
      expectFlowReuseRows(
        (name, expected) => expect(finalRead(rows[name]), name).toEqual(expected),
        3,
        pathCapacity,
        limited
      );
    });
