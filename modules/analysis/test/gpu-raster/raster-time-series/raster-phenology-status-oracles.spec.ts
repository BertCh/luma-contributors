// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterPhenologyMetrics,
  getGPURasterPhenologyMetricsParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';

// Generated independently by the adjacent .oracle.py using the committed NumPy oracle.
// Entries are chronological, with fields in the public ten-metric order. All finite
// expectations are dyadic rationals and exactly representable in float32.
const CASES = [
  {
    name: 'start-at-other-season',
    curve: [0, 4, 2, 3, 0],
    startFraction: 0.25,
    endFraction: 0.75,
    flags: [0, 4],
    metrics: [0.25, 1.5, 1, 1.25, 4, 0, 4, 4, -2, 3.625, 2, 3, 3, 1, 3, 0, 3, 1, NaN, 2.5]
  },
  {
    name: 'end-at-other-season',
    curve: [0, 3, 2, 4, 0],
    startFraction: 0.75,
    endFraction: 0.25,
    flags: [8, 0],
    metrics: [1, 2, 1, 1, 3, 0, 3, NaN, -1, 2.5, 2.5, 3.75, 3, 1.25, 4, 0, 4, 2, -4, 3.625]
  }
];

for (const fixture of CASES) {
  it(`retains the other-season phenology boundary flag: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const flags = harness.output('flags', 'uint32', 2);
    const status = harness.output('status', 'uint32', 1);
    (flags.buffer.defaultBuffer as Buffer).write(new Uint32Array(2).fill(0xdeadbeef));
    (status.buffer.defaultBuffer as Buffer).write(Uint32Array.of(0xdeadbeef));
    const result = await harness.run(
      new GPURasterPhenologyMetrics({
        cellCount: 1,
        sliceCount: 5,
        maximumSeasonCount: 2,
        smoothed: harness.floats(fixture.curve),
        parameters: harness.floats(
          getGPURasterPhenologyMetricsParameterValues({
            startFraction: fixture.startFraction,
            endFraction: fixture.endFraction,
            minimumAmplitude: 0.1,
            secondSeasonMinimumFraction: 0.75
          })
        ),
        output: {
          metrics: harness.output('metrics', 'float32', 20),
          flags,
          seasonCount: harness.output('count', 'uint32', 1),
          status
        }
      })
    );
    expect(result['flags']).toEqual(fixture.flags);
    expect(result['count']).toEqual([2]);
    expect(result['status']).toEqual([0]);
    expect(result['metrics']).toEqual(fixture.metrics);
  });
}
