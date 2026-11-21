// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';
import {createSmoothOracleFixture} from './line-smooth-output-oracle-fixture';

it('smooth publishes two lines and their source rows with each optional output isolated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['required', 'sources', 'absent'] as const) {
    const harness = createLineOracleHarness(device);
    const fixture = createSmoothOracleFixture(harness.view, optional);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      expect(await harness.read(fixture.count)).toEqual([2]);
      expect(await harness.read(fixture.overflow)).toEqual([0]);
      expect(await harness.read(fixture.lineOffsets)).toEqual([0, 4, 8]);
      expect(await harness.read(fixture.positions)).toEqual([
        0, 0, 1, 0, 3, 0, 4, 0, 10, 2, 11, 2, 13, 2, 14, 2
      ]);
      if (fixture.requiredCount) expect(await harness.read(fixture.requiredCount)).toEqual([2]);
      if (fixture.sourceLineIndices)
        expect(await harness.read(fixture.sourceLineIndices)).toEqual([0, 1]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});
