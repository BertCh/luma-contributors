// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {createJoinReview2Cases} from './join-review2-fixture';

for (const entry of createJoinReview2Cases())
  it(`WebGPU review2 ${entry.name}`, async () => {
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
        const result = await fixture.run();
        for (const [name, expected] of Object.entries(entry.expected)) {
          if (entry.outputs[name].format === 'uint32')
            expect(Array.from(result[name]), name).toEqual(expected);
          else
            expected.forEach((value, row) => {
              expect(Number.isFinite(result[name][row]), name).toBe(true);
              expect(Math.abs(result[name][row] - value), name).toBeLessThanOrEqual(
                entry.tolerance ?? 0
              );
            });
        }
      }
    } finally {
      fixture.destroy();
    }
  });
