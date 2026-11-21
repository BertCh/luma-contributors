// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUTrajectoryHampel,
  getGPUTrajectoryHampelParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-hampel';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';

// Hand oracle, window size 3: [1, 9, 1] has median 1 and MAD 0, so only row 2
// is replaced. Every other centred window has its centre equal to its median;
// track ends pass through. A constant second track contributes no replacements.
// Integer outputs are exact, including every bit of the documented 0/1 flags.
for (const scenario of [
  {
    name: 'one replacement',
    values: [1, 1, 9, 1, 1, 2, 2, 2, 2, 2],
    flags: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    count: [1]
  },
  {
    name: 'no replacements',
    values: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    flags: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    count: [0]
  }
]) {
  it(`Hampel writes exact flags and count over poison: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, `hampel-status-${scenario.name}`);
    const flags = harness.output('flags', 'uint32', 10);
    const count = harness.output('count', 'uint32', 1);
    flags.buffer.write(new Uint32Array(10).fill(0xdeadbeef));
    count.buffer.write(new Uint32Array([0xdeadbeef]));
    harness.graph.add(
      new GPUTrajectoryHampel({
        values: harness.input('values', Float32Array.from(scenario.values), 'float32', 10),
        trackOffsets: harness.input('offsets', new Uint32Array([0, 5, 10]), 'uint32', 3),
        windowSize: 3,
        parameters: harness.input(
          'parameters',
          getGPUTrajectoryHampelParameterValues({numberOfSigmas: 3}),
          'float32',
          4
        ),
        output: {
          filtered: harness.output('filtered', 'float32', 10).view,
          flags: flags.view,
          outlierCount: count.view
        }
      })
    );
    const result = await harness.run(async () => ({
      flags: await readUint32(flags.buffer, 10),
      count: await readUint32(count.buffer, 1)
    }));
    expect(result.flags).toEqual(scenario.flags);
    expect(result.count).toEqual(scenario.count);
  });
}
