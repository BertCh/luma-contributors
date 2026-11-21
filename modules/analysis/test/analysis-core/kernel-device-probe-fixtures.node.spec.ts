// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GRID_QUERY_INITIAL_CANDIDATES,
  createSegmentParityFixture,
  validateCompensatedCancellationReadback
} from './kernel-device-probe-fixtures';

it('review B1 an empty scatter leaves a preseeded sentinel candidate vector', () => {
  // No selected ranks means no scatter writes; the literal empty result must already be seeded.
  expect(GRID_QUERY_INITIAL_CANDIDATES.slice()).toEqual([0xffffffff]);
});

it('review S2 gather parity requires sorting and distinct populated totals', () => {
  const fixture = createSegmentParityFixture();
  expect(fixture.keys.slice(0, 5)).toEqual([1, 3, 1, 3, 1]);
  expect(fixture.expected).toEqual([0, 3, 0, 7]);
  // Stable gathering must retain the cancellation tree's order in group 1.
  const gathered = fixture.values.filter((_, index) => fixture.keys[index] === 1);
  expect(gathered).toHaveLength(257);
  expect([gathered[0], gathered[1], gathered[128], gathered[256]]).toEqual([
    16777216, 1, -16777216, 2
  ]);
  expect(fixture.values.filter((_, index) => fixture.keys[index] === 3)).toEqual([9, -2]);
});

it('review S4 cancellation evidence rejects extra and missing storage words', () => {
  expect(() => validateCompensatedCancellationReadback([12345, 0, 1, 1, 0, 12345])).not.toThrow();
  expect(() => validateCompensatedCancellationReadback([12345, 0, 1, 1, 0, 12345, 7])).toThrow();
  expect(() => validateCompensatedCancellationReadback([12345, 0, 1, 1, 0])).toThrow();
});
