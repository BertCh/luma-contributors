// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {buildCSR} from '../network-reachability/network-reachability-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

for (const phase of ['cost', 'label', 'tie', 'zero-active', 'complete', 'empty'] as const) {
  for (const withConverged of [false, true]) {
    it(`service-area stopping ports cover ${phase} with converged requested: ${withConverged}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createOptimizationHarness(device);
      const nodeCount = 33;
      const topology = buildCSR(
        nodeCount,
        Array.from(
          {length: nodeCount - 1},
          (_, node) => [node, node + 1, phase === 'tie' ? 0 : 1] as const
        )
      );
      const status = {
        converged: withConverged ? harness.output('uint32', 1) : undefined,
        iterationLimitReached: harness.output('uint32', 1)
      };
      const output = {
        assignments: harness.output('uint32', nodeCount),
        costs: harness.output('float32', nodeCount),
        predecessors: harness.output('uint32', nodeCount),
        status
      };
      const facilityCount = harness.input('uint32', Uint32Array.of(phase === 'empty' ? 0 : 1));
      const activeIterations = harness.input(
        'uint32',
        Uint32Array.of(phase === 'zero-active' ? 0 : 48)
      );
      const result = await harness.run(
        new GPUNetworkServiceAreas({
          topology: {
            offsets: harness.input('uint32', topology.offsets),
            neighbors: harness.input('uint32', topology.neighbors),
            weights: harness.input('float32', topology.weights)
          },
          // The second facility row is padding and must never seed any phase.
          facilities: harness.input('uint32', Uint32Array.of(0, 0xffffffff)),
          facilityCount,
          activeIterations,
          maximumIterations: phase === 'cost' ? 1 : 48,
          maximumLabelIterations: phase === 'label' ? 1 : 48,
          maximumTieIterations: phase === 'tie' ? 1 : 48,
          localIterations: 1,
          output
        }),
        first => {
          const stopped =
            phase === 'cost' || phase === 'label' || phase === 'tie' || phase === 'zero-active';
          expect(first(status.iterationLimitReached)).toEqual([Number(stopped)]);
          if (status.converged) expect(first(status.converged)).toEqual([Number(!stopped)]);
          const costs = first(output.costs);
          const assignments = first(output.assignments);
          const predecessors = first(output.predecessors);
          if (phase === 'cost' || phase === 'zero-active') {
            expect(costs[0]).toBe(0);
            expect(costs[nodeCount - 1]).toBe(Infinity);
          } else if (phase === 'empty') {
            expect(costs).toEqual(new Array(nodeCount).fill(Infinity));
            expect(assignments).toEqual(new Array(nodeCount).fill(0xffffffff));
          } else {
            // A single directed chain has hand-derived distances and a unique facility label.
            expect(costs).toEqual(
              Array.from({length: nodeCount}, (_, node) => (phase === 'tie' ? 0 : node))
            );
            if (phase === 'label') expect(assignments[nodeCount - 1]).toBe(0xffffffff);
            else expect(assignments).toEqual(new Array(nodeCount).fill(0));
            if (phase === 'tie') expect(predecessors[nodeCount - 1]).toBe(0xffffffff);
            else if (phase === 'complete') {
              expect(predecessors).toEqual(
                Array.from({length: nodeCount}, (_, node) => (node === 0 ? 0xffffffff : node - 1))
              );
            }
          }
          // An empty facility set has no pending work, even under the same short phase budgets.
          // A second encoding must clear an earlier limit flag and every incumbent output.
          harness.rewrite(facilityCount, Uint32Array.of(0));
          harness.rewrite(activeIterations, Uint32Array.of(48));
        }
      );
      expect(result(status.iterationLimitReached)).toEqual([0]);
      if (status.converged) expect(result(status.converged)).toEqual([1]);
      expect(result(output.costs)).toEqual(new Array(nodeCount).fill(Infinity));
      expect(result(output.assignments)).toEqual(new Array(nodeCount).fill(0xffffffff));
      expect(result(output.predecessors)).toEqual(new Array(nodeCount).fill(0xffffffff));
    });
  }
}
