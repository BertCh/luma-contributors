// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {buildCSR} from '../network-reachability/network-reachability-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

for (const withAssignments of [false, true]) {
  for (const withConverged of [false, true]) {
    for (const stopped of [false, true]) {
      it(`isochrone stopping ports follow the ${withAssignments ? 'service-area' : 'reachability'} search (stopped: ${stopped}, converged: ${withConverged})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const harness = createOptimizationHarness(device);
        const nodeCount = 33;
        const topology = buildCSR(
          nodeCount,
          Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1] as const)
        );
        const status = {
          converged: withConverged ? harness.output('uint32', 1) : undefined,
          iterationLimitReached: harness.output('uint32', 1)
        };
        const costs = harness.output('float32', nodeCount);
        const assignments = withAssignments ? harness.output('uint32', nodeCount) : undefined;
        const startCount = harness.input('uint32', Uint32Array.of(1));
        const count = harness.output('uint32', 1);
        const overflow = harness.output('uint32', 1);
        const result = await harness.run(
          new GPUNetworkIsochrones({
            topology: {
              offsets: harness.input('uint32', topology.offsets),
              neighbors: harness.input('uint32', topology.neighbors),
              weights: harness.input('float32', topology.weights),
              nodePositions: harness.input(
                'float32x2',
                Float32Array.from({length: 2 * nodeCount}, (_, component) =>
                  component % 2 === 0 ? component / 2 : 0.5
                )
              )
            },
            startNodes: harness.input('uint32', Uint32Array.of(0, 0xffffffff)),
            startCount,
            costs,
            assignments,
            maximumIterations: stopped && !withAssignments ? 1 : 48,
            maximumLabelIterations: withAssignments ? (stopped ? 1 : 48) : undefined,
            localIterations: 1,
            costBreaks: harness.input('float32', Float32Array.of(100)),
            parameters: harness.input(
              'float32',
              getGPUNetworkIsochronesParameterValues({
                breakCount: 1,
                extent: [0, 0, 32, 1],
                bufferRadius: 0.5,
                walkCostPerUnit: 0
              })
            ),
            output: {status},
            raster: {
              width: 2,
              height: 2,
              maximumBufferPixels: 1,
              output: {
                triangles: harness.output('float32x2', 3 * 64),
                triangleBands: harness.output('uint32', 64),
                status: {count, overflow}
              }
            }
          }),
          first => {
            expect(first(status.iterationLimitReached)).toEqual([Number(stopped)]);
            if (status.converged) expect(first(status.converged)).toEqual([Number(!stopped)]);
            if (stopped && !withAssignments) {
              expect(first(costs)[0]).toBe(0);
              expect(first(costs)[nodeCount - 1]).toBe(Infinity);
            } else expect(first(costs)).toEqual(Array.from({length: nodeCount}, (_, node) => node));
            if (assignments) {
              if (stopped) expect(first(assignments)[nodeCount - 1]).toBe(0xffffffff);
              else expect(first(assignments)).toEqual(new Array(nodeCount).fill(0));
            }
            expect(first(count)[0]).toBeLessThanOrEqual(64);
            expect(first(overflow)).toEqual([0]);
            harness.rewrite(startCount, Uint32Array.of(0));
          }
        );
        expect(result(status.iterationLimitReached)).toEqual([0]);
        if (status.converged) expect(result(status.converged)).toEqual([1]);
        expect(result(costs)).toEqual(new Array(nodeCount).fill(Infinity));
        if (assignments) expect(result(assignments)).toEqual(new Array(nodeCount).fill(0xffffffff));
      });
    }
  }
}
