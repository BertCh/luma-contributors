// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import fixture from './fixtures/parity-wp5-round5-peaks-synthetic.json';

it('TWN-09 accepts a reported prominence equal to the inclusive minimum', () => {
  const peak = fixture.cases.find(entry => entry.name === 'float32-inclusive-prominence');
  expect(peak).toBeDefined();
  // Exact prominence 1-2^-26 rounds to f32 1, meeting minimumProminence=1.
  expect(peak!.values).toEqual([2 ** -26, 1, 2 ** -26]);
  expect(peak!.minimumProminence).toBe(1);
  expect(peak!.expectedMask).toEqual([0, 1, 0]);
  expect(peak!.expectedIds).toEqual([1]);
  expect(peak!.expected.prominence).toEqual([null, 1, null]);
});

it('TWN-09 prefers the earlier index when distinct exact prominences round equal', () => {
  const peak = fixture.cases.find(entry => entry.name === 'float32-prominence-suppression-tie');
  expect(peak).toBeDefined();
  // 1-2^-25 < 1-2^-26, but both round to f32 1 (the former ties to even).
  // A gap separates the basins; the two peaks are four samples apart, within radius 5.
  expect(peak!.values).toEqual([2 ** -25, 1, 2 ** -25, null, 2 ** -26, 1, 2 ** -26]);
  expect(peak!.candidateCount).toBe(2);
  expect(peak!.suppressionRadius).toBe(5);
  expect(peak!.expectedMask).toEqual([0, 1, 0, 0, 0, 0, 0]);
  expect(peak!.expectedIds).toEqual([1]);
  expect(peak!.expected.prominence).toEqual([null, 1, null, null, null, null, null]);
});
