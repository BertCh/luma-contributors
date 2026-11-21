// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCollapsedClearanceReviewCase} from '../geometry-predicates/multipart-review-fixtures';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([
  false,
  true
])('GPU diagonal clearance chooses row 2 partner with sorting=%s', async spatialSort => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const fixture = createCollapsedClearanceReviewCase(
    harness.graph,
    (name, format, length, values) =>
      harness.allocate(
        name,
        format,
        length,
        name === 'collapsed-clearance-input' ? [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1] : values
      ),
    false,
    true,
    spatialSort
  );
  try {
    await harness.run(fixture.nodes);
    expect(await harness.read(fixture.output.lines)).toEqual([0, 0, 1, 1]);
    expect(await harness.read(fixture.output.vertexIds)).toEqual([0]);
    const distance = (await harness.read(fixture.output.clearances))[0];
    expect(Number.isFinite(distance)).toBe(true);
    expect(distance).toBeCloseTo(1.4142135623730951, 6);
  } finally {
    harness.destroy();
  }
});

// Review B2: collapsed parts at (0,0) and (1,1) beside the ordinary segment (-1,-1)->(-3,-3).
// Row 0 ties at sqrt(1²+1²)=sqrt(2) with (1,1) (rows 2-3) and with the segment endpoint
// (-1,-1) (row 4); the lowest partner row wins whether or not the segment pass found its tie first.
it.each([
  false,
  true
])('GPU mixed-part clearance tie keeps the row 2 partner with sorting=%s', async spatialSort => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const fixture = createCollapsedClearanceReviewCase(
    harness.graph,
    (name, format, length, values) =>
      harness.allocate(
        name,
        format,
        length,
        name === 'collapsed-clearance-input' ? [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -3, -3] : values
      ),
    false,
    true,
    spatialSort
  );
  try {
    await harness.run(fixture.nodes);
    expect(await harness.read(fixture.output.lines)).toEqual([0, 0, 1, 1]);
    expect(await harness.read(fixture.output.vertexIds)).toEqual([0]);
    const distance = (await harness.read(fixture.output.clearances))[0];
    expect(Number.isFinite(distance)).toBe(true);
    expect(distance).toBeCloseTo(1.4142135623730951, 6);
  } finally {
    harness.destroy();
  }
});
