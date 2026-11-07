// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  getBoundedWorkgroupDispatchLayout
} from '../../src/gpu-core/gpu-dispatch-utils';

test('getBoundedDispatchLayout spreads large element counts across dimensions', () => {
  // 5 million rows at 64 per workgroup exceeds 65535 workgroups in one dimension.
  const layout = getBoundedDispatchLayout('test', 5_000_000, 64, 65_535);
  expect(layout.x).toBe(65_535);
  expect(layout.y).toBeGreaterThan(1);
  expect(layout.x * layout.y * layout.z * 64).toBeGreaterThanOrEqual(5_000_000);
});

test('getBoundedWorkgroupDispatchLayout rejects counts beyond the 3D limit', () => {
  expect(getBoundedWorkgroupDispatchLayout('test', 17, 4)).toEqual({x: 4, y: 4, z: 2});
  expect(() => getBoundedWorkgroupDispatchLayout('test', 65, 4)).toThrow(/3D dispatch limit/);
});

test('override helpers expose the layout as pipeline constants', () => {
  expect(getBoundedDispatchOverrides({x: 3, y: 2, z: 1})).toEqual({DISPATCH_X: 3, DISPATCH_Y: 2});
  const source = getBoundedInvocationIndexOverrideSource(64);
  expect(source.declarations).toContain('override DISPATCH_X: u32;');
  expect(source.body).toContain('workgroupIndex * 64u + localInvocationIndex');
});
