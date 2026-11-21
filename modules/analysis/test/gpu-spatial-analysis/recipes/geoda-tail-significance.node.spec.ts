// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {isGeodaTailSignificant} from './geoda-tail-significance';

it('compares the exact tail ratio against the packed cutoff', () => {
  // 1/25 exceeds float32 .04 = .03999999910593033; 1/32 is exactly .03125.
  expect(isGeodaTailSignificant(0, 24, 0.04)).toBe(false);
  expect(isGeodaTailSignificant(0, 31, 0.03125)).toBe(true);
  expect(isGeodaTailSignificant(0xffffffff, 31, 1)).toBe(false);
});
