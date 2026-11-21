// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runHDBSCAN} from './hierarchical-density-clustering-harness';

for (const coreDistanceSearch of ['bvh', 'neighbor-search'] as const) {
  it(`uses the 32nd self-inclusive neighbor for omitted minimumSamples (${coreDistanceSearch})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    try {
      // Rows x=0..47 on a line. At row zero the self-inclusive rank 32 is x=31,
      // distance 31; an uncapped minimumClusterSize=40 would instead give 39.
      const positions = Float32Array.from({length: 96}, (_, index) => (index % 2 ? 0 : index / 2));
      const actual = await runHDBSCAN(device, positions, {
        minimumClusterSize: 40,
        coreDistanceSearch
      });
      expect(Number.isFinite(actual.coreDistances[0])).toBe(true);
      expect(actual.coreDistances[0]).toBe(31);
      expect(actual.coreDistances[0]).not.toBe(39);
    } finally {
      device.destroy();
    }
  });
}
