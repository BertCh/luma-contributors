// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectExtremeICPReadback} from './pointcloud-icp-extreme-assertions';

const TRANSFORM = [
  0.9987502603949663, 0.04997916927067833, 0, 0, -0.04997916927067833, 0.9987502603949663, 0, 0, 0,
  0, 1, 0, 0, 0, 0, 1
];
const HISTORY = [3, 6.96e17, 0];

it('S1: extreme ICP checks reject incomplete, oversized and zeroed-column readbacks', () => {
  expect(() => expectExtremeICPReadback(TRANSFORM, HISTORY)).not.toThrow();
  expect(() => expectExtremeICPReadback(TRANSFORM.slice(0, 15), HISTORY)).toThrow();
  expect(() => expectExtremeICPReadback([...TRANSFORM, 0], HISTORY)).toThrow();
  expect(() => expectExtremeICPReadback(TRANSFORM, [3])).toThrow();
  expect(() => expectExtremeICPReadback(TRANSFORM, [...HISTORY, 0])).toThrow();
  const missingColumn = [...TRANSFORM];
  missingColumn[10] = 0;
  expect(() => expectExtremeICPReadback(missingColumn, HISTORY)).toThrow();
});
