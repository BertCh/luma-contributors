// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUTrajectoryOutlierFlags,
  getGPUTrajectoryOutlierFlagsParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-outlier-flags';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

// Hand oracle: track 0 has step speeds 1, sqrt(10001), sqrt(10001), 1.
// Only fix 2 has two adjacent speeds > 10 and a rise/fall of magnitude > 10.
// Track 1 moves at constant speed 1. Counts count flagged ROWS, not flag bits:
// enabling both diagnostics still gives [1, 0], never [2, 0]. Exact uint32 words
// observe all bits, including reserved bits; poison makes unwritten zeros fail.
for (const scenario of [
  {name: 'speed', maximumSpeed: 10, maximumAcceleration: 0, flag: 1, count: 1},
  {name: 'acceleration', maximumSpeed: 0, maximumAcceleration: 10, flag: 2, count: 1},
  {name: 'both', maximumSpeed: 10, maximumAcceleration: 10, flag: 3, count: 1},
  {name: 'disabled', maximumSpeed: 0, maximumAcceleration: 0, flag: 0, count: 0}
]) {
  it(`outliers overwrite poisoned whole flag words and track counts: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, `outlier-status-${scenario.name}`);
    const trajectory = importTrajectory(harness, {
      positions: [
        [0, 0],
        [1, 0],
        [2, 100],
        [3, 0],
        [4, 0],
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0]
      ],
      times: new Float32Array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
      trackOffsets: [0, 5, 10]
    });
    const flags = harness.output('flags', 'uint32', 10);
    const counts = harness.output('counts', 'uint32', 2);
    flags.buffer.write(new Uint32Array(10).fill(0xdeadbeef));
    counts.buffer.write(new Uint32Array(2).fill(0xdeadbeef));
    harness.graph.add(
      new GPUTrajectoryOutlierFlags({
        positions: trajectory.positions,
        timestamps: trajectory.timestamps,
        trackOffsets: trajectory.trackOffsets,
        parameters: harness.input(
          'parameters',
          getGPUTrajectoryOutlierFlagsParameterValues(scenario),
          'float32',
          4
        ),
        output: {flags: flags.view, trackFlaggedCounts: counts.view}
      })
    );
    const result = await harness.run(async () => ({
      flags: await readUint32(flags.buffer, 10),
      counts: await readUint32(counts.buffer, 2)
    }));
    expect(result.flags).toEqual([0, 0, scenario.flag, 0, 0, 0, 0, 0, 0, 0]);
    expect(result.counts).toEqual([scenario.count, 0]);
  });
}
