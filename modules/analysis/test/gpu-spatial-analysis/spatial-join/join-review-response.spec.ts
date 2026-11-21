// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {createJoinReviewCases} from './join-review-response-fixture';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const entry of createJoinReviewCases())
  it(`WebGPU ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: entry.inputs,
      outputs: entry.outputs,
      create: ({inputs, outputs}) => ({
        getCommandNodes: graph => entry.declare(graph, inputs, outputs)
      })
    });
    try {
      for (let submission = 0; submission < 2; submission++) {
        const actual = await fixture.run();
        for (const [name, expected] of Object.entries(entry.expected))
          expected.forEach((value, row) => {
            expect(Number.isFinite(actual[name][row]), name).toBe(true);
            expect(Math.abs(actual[name][row] - value), name).toBeLessThanOrEqual(
              entry.tolerance ?? 0
            );
          });
      }
    } finally {
      fixture.destroy();
    }
  });

for (const reverse of [false, true])
  it(`WebGPU R7 threshold excludes road reverse=${reverse} in both operations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const end = Math.fround(0.01),
      road = reverse ? [end, end, -8191, -8190] : [-8191, -8190, end, end];
    for (const create of [
      createSpatialPredicateJoinOperation,
      createCPUSpatialPredicateJoinOperation
    ]) {
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      const operation = create({
        left: {kind: 'points', positions: {value: 'left', length: 1}},
        right: {
          kind: 'lines',
          positions: {value: 'right', length: 2},
          lineOffsets: {value: 'offsets', length: 2}
        },
        predicate: 'dwithin',
        distance: 8.633243737676821e-7,
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      });
      const actual = await runOperationGraph(device, {
        inputs: {
          left: {format: 'float32x2', values: Float32Array.of(0, 0)},
          right: {format: 'float32x2', values: Float32Array.from(road)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 2)}
        },
        operations: [operation],
        outputs: Object.values(output)
      });
      for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'])
        expect(Array.from(actual[name]), name).toEqual([0]);
    }
  });
