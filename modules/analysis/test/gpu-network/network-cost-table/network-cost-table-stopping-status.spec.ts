// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {buildCSR} from '../network-reachability/network-reachability-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

for (const stopped of [false, true]) {
  for (const capacity of [3, 81]) {
    for (const withConverged of [false, true]) {
      it(`cost-table round exhaustion is independent of capacity (stopped: ${stopped}, capacity: ${capacity}, converged: ${withConverged})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const harness = createOptimizationHarness(device);
        const nodeCount = 9;
        const topology = buildCSR(
          nodeCount,
          Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1] as const)
        );
        const costLimit = harness.input('float32', Float32Array.of(Infinity));
        const status = {
          converged: withConverged ? harness.output('uint32', 1) : undefined,
          iterationLimitReached: harness.output('uint32', 1),
          overflow: harness.output('uint32', 1),
          requiredCount: harness.output('uint32', 1),
          complete: harness.output('uint32', 1)
        };
        const output = {
          offsets: harness.output('uint32', nodeCount + 1),
          targets: harness.output('uint32', capacity),
          costs: harness.output('float32', capacity),
          status
        };
        const result = await harness.run(
          new GPUNetworkCostTable({
            topology: {
              offsets: harness.input('uint32', topology.offsets),
              neighbors: harness.input('uint32', topology.neighbors),
              weights: harness.input('float32', topology.weights)
            },
            costLimit,
            // The final batch has three padding lanes, which must not affect either flag.
            laneCount: 4,
            maximumIterations: stopped ? 1 : 16,
            localIterations: 1,
            output
          }),
          first => {
            const offsets = [0];
            const targets: number[] = [];
            const costs: number[] = [];
            for (let source = 0; source < nodeCount; source++) {
              // One-hop incumbents after one round, or every downstream node at the fixpoint.
              const last = stopped ? Math.min(source + 1, nodeCount - 1) : nodeCount - 1;
              for (let target = source; target <= last; target++) {
                targets.push(target);
                costs.push(target - source);
              }
              offsets.push(Math.min(targets.length, capacity));
            }
            expect(first(output.offsets)).toEqual(offsets);
            expect(first(output.targets).slice(0, Math.min(targets.length, capacity))).toEqual(
              targets.slice(0, capacity)
            );
            expect(first(output.costs).slice(0, Math.min(costs.length, capacity))).toEqual(
              costs.slice(0, capacity)
            );
            expect(first(status.requiredCount)).toEqual([targets.length]);
            expect(first(status.overflow)).toEqual([Number(targets.length > capacity)]);
            expect(first(status.iterationLimitReached)).toEqual([Number(stopped)]);
            expect(first(status.complete)).toEqual([
              Number(!stopped && targets.length <= capacity)
            ]);
            if (status.converged) expect(first(status.converged)).toEqual([Number(!stopped)]);
            // A negative cutoff rejects all starts: the next encoding is empty and converged.
            harness.rewrite(costLimit, Float32Array.of(-1));
          }
        );
        expect(result(output.offsets)).toEqual(new Array(nodeCount + 1).fill(0));
        expect(result(status.requiredCount)).toEqual([0]);
        expect(result(status.overflow)).toEqual([0]);
        expect(result(status.iterationLimitReached)).toEqual([0]);
        expect(result(status.complete)).toEqual([1]);
        if (status.converged) expect(result(status.converged)).toEqual([1]);
      });
    }
  }
}
