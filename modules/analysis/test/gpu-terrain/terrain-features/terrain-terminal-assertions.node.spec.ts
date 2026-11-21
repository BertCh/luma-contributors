// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectProfileSingletonStatus,
  expectSingleProfilePeak,
  expectCapturedSolarAltitude,
  expectSolarAngle,
  expectFlatRelief,
  expectReliefColor
} from './terrain-terminal-assertions';

for (const terminal of ['converged', 'iterationLimitReached'] as const) {
  for (const rounds of [0, 16] as const) {
    it(`B2 status oracle rejects the opposite ${terminal} word at ${rounds} rounds`, () => {
      // A candidate is undecided with no rounds and decided with sixteen rounds.
      const expected = terminal === 'converged' ? Number(rounds > 0) : Number(rounds === 0);
      expect(() =>
        expectProfileSingletonStatus(Uint32Array.of(1 - expected), terminal, rounds)
      ).toThrow();
      expect(() =>
        expectProfileSingletonStatus(Uint32Array.of(expected), terminal, rounds)
      ).not.toThrow();
    });
  }
}

it('S1 mutation oracle rejects un-refracted altitude against the independent SPA reference', () => {
  // These measured controls straddle the independent NREL SPA 39.88838 degree reference.
  expect(() => expectCapturedSolarAltitude(Float32Array.of(39.87207))).toThrow();
  expect(() => expectCapturedSolarAltitude(Float32Array.of(39.891361))).not.toThrow();
});

for (const field of ['count', 'ids', 'prominence'] as const) {
  it(`S2 profile oracle rejects an extra ${field} row`, () => {
    const rows = {
      mask: Uint32Array.of(0, 1, 0),
      count: Uint32Array.of(1),
      ids: Uint32Array.of(1),
      prominence: Float32Array.of(NaN, 1, NaN)
    };
    expect(() =>
      expectSingleProfilePeak(
        {...rows, [field]: Float32Array.from([...rows[field], 99])},
        [0, 1, 0]
      )
    ).toThrow();
    expect(() => expectSingleProfilePeak(rows, [0, 1, 0])).not.toThrow();
  });
}

it('S2 solar angle oracle rejects an extra row', () => {
  expect(() => expectSolarAngle(Float32Array.of(194.34024, 99), 194.34024)).toThrow();
  expect(() => expectSolarAngle(Float32Array.of(194.34024), 194.34024)).not.toThrow();
});

it('S2 relief scalar oracle rejects an empty readback', () => {
  expect(() => expectFlatRelief(new Float32Array(), 0.5)).toThrow();
  expect(() => expectFlatRelief(new Float32Array(9).fill(0.5), 0.5)).not.toThrow();
});

it('S2 relief color oracle rejects an empty readback', () => {
  expect(() => expectReliefColor(new Uint32Array())).toThrow();
  // Literal little-endian RGBA bytes [51,26,13,255].
  expect(() => expectReliefColor(new Uint32Array(9).fill(0xff0d1a33))).not.toThrow();
});
