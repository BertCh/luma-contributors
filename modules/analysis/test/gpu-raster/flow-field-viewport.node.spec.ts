// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  FlowFieldViewportTracker,
  getFlowFieldViewportBounds
} from '../../src/raster-analysis/flow-texture/flow-field-viewport';

const FIELD = [0, 0, 100, 50] as const;

describe('flow field viewport domain', () => {
  test('pads by a tenth, clips to the data and handles shifted world copies', () => {
    expect(getFlowFieldViewportBounds([40, 20, 60, 30], FIELD)).toEqual([38, 19, 62, 31]);
    expect(getFlowFieldViewportBounds([400, 20, 420, 30], FIELD)).toEqual([38, 19, 62, 31]);
    expect(getFlowFieldViewportBounds([-5, -5, 5, 5], FIELD)).toEqual([0, 0, 6, 6]);
  });

  test('uses the whole field for missing, disjoint, invalid or world-wide viewports', () => {
    for (const viewport of [
      undefined,
      [200, 20, 220, 30],
      [0, 60, 100, 80],
      [0, 0, 360, 50],
      [10, 0, 0, 50],
      [0, 10, 100, 0],
      [NaN, 0, 100, 50]
    ] as const) {
      expect(getFlowFieldViewportBounds(viewport, FIELD)).toBe(FIELD);
    }
  });

  test('waits for a settled camera and restarts the delay on a material pan', () => {
    const tracker = new FlowFieldViewportTracker();
    const first = [38, 19, 62, 31] as const;
    const second = [48, 19, 72, 31] as const;
    expect(tracker.update(first, FIELD, 0)).toBe(false);
    expect(tracker.update(first, FIELD, 149)).toBe(false);
    expect(tracker.update(second, FIELD, 150)).toBe(false);
    expect(tracker.update(second, FIELD, 299)).toBe(false);
    expect(tracker.update(second, FIELD, 300)).toBe(true);
  });

  test('ignores jitter under 2% of the active span and handles zooming back out', () => {
    const tracker = new FlowFieldViewportTracker();
    const active = [38, 19, 62, 31] as const;
    expect(tracker.update([38.1, 19.1, 62.1, 31.1], active, 0)).toBe(false);
    expect(tracker.update([38.1, 19.1, 62.1, 31.1], active, 1000)).toBe(false);
    expect(tracker.update(FIELD, active, 1100)).toBe(false);
    expect(tracker.update(FIELD, active, 1250)).toBe(true);
    expect(tracker.update(FIELD, FIELD, 1300)).toBe(false);
  });
});

test('a stationary committed viewport never re-arms settling, including the hysteresis boundary', () => {
  const tracker = new FlowFieldViewportTracker();
  const active = [38, 19, 62, 31] as const;
  // Material movement just beyond the 2% threshold commits once after settling.
  const desired = [38.49, 19, 62.49, 31] as const;
  expect(tracker.update(desired, active, 0)).toBe(false);
  expect(tracker.update(desired, active, 150)).toBe(true);
  for (let timestamp = 166; timestamp < 4000; timestamp += 16)
    expect(tracker.update(desired, desired, timestamp)).toBe(false);
});
