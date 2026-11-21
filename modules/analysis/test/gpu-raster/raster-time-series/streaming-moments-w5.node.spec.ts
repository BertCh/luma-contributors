// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {decodeMomentReadback, runHostMomentProbe} from './streaming-moments-w5.test-utils';
import {
  getGPURasterFixedPointValues,
  createGPURasterStreamingAccumulatorEncodingPlan
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {createGPURasterStandardizedIndexCalibrationEncodingPlan} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';

for (const magnitude of ['adjacent', 'large', 'wide-bound'] as const) {
  for (const route of ['streaming', 'calibration'] as const) {
    for (const split of [false, true]) {
      for (const reversed of [false, true]) {
        it(`SCL-01 ${route} ${magnitude} centered moments split=${split} reversed=${reversed}`, () => {
          const observations: [number, number] =
            magnitude === 'adjacent'
              ? [10000000, 10000001]
              : magnitude === 'wide-bound'
                ? [2 ** 69, 2 ** 70]
                : route === 'streaming'
                  ? [2 ** 70, -(2 ** 70)]
                  : [2 ** 70, 2 ** 71];
          if (reversed) observations.reverse();
          const result = runHostMomentProbe({
            route,
            observations,
            split,
            maximumStreamSliceCount: magnitude === 'wide-bound' ? 2 : undefined
          });
          // Closed form M2=(b-a)^2/2; adjacent mean=10000000.5 gives .5.
          const expected =
            magnitude === 'adjacent'
              ? 0.5
              : magnitude === 'wide-bound'
                ? 2 ** 137
                : route === 'streaming'
                  ? 2 ** 141
                  : 2 ** 139;
          expect(result.moments).toHaveLength(1);
          expect(Number.isFinite(result.moments[0])).toBe(true);
          expect([...result.moments]).toEqual([expected]);
          if (magnitude === 'wide-bound') expect(result.momentUnitExponent).toBe(50);
        });
      }
    }
  }
}

it('B1 retains the wide squared magnitude bound in both encoding plans', () => {
  const settings = {maximumValueMagnitude: 2 ** 70, maximumStreamSliceCount: 2};
  // floor(log2(4*(2^70)^2)) + 1 + bitLength(2) - 95 = 50.
  expect(createGPURasterStreamingAccumulatorEncodingPlan(settings, 2).momentUnitExponent).toBe(50);
  expect(
    createGPURasterStandardizedIndexCalibrationEncodingPlan(settings, 2).momentUnitExponent
  ).toBe(50);
});

for (const split of [false, true]) {
  for (const reversed of [false, true]) {
    it(`B2 retains shifted low bits split=${split} reversed=${reversed}`, () => {
      const observations: [number, number] = [16, 16 + 2 ** -19];
      if (reversed) observations.reverse();
      const result = runHostMomentProbe({
        route: 'calibration',
        observations,
        split,
        location: -64,
        maximumValueMagnitude: 81,
        distribution: 'fisk'
      });
      // Translation leaves M2=(2^-19)^2/2=2^-39; shifted sum=160+2^-19.
      expect([...result.moments]).toEqual([2 ** -39]);
      expect([
        ...getGPURasterFixedPointValues(result.fields.moments.slice(0, 3), result.sumUnitExponent)
      ]).toEqual([160 + 2 ** -19]);
      expect([...result.fields.counts]).toEqual([2, 0, 2]);
      expect(result.fields.fit).toHaveLength(3);
      expect(result.fields.fit.every(Number.isFinite)).toBe(true);
      expect(Math.abs(result.fields.fit[0] / 152152520.3859001 - 1)).toBeLessThan(3e-7);
      expect(Math.abs(result.fields.fit[1] / 80.00000095367432 - 1)).toBeLessThan(3e-7);
    });
    it(`Fisk preserves a finite large-offset fit split=${split} reversed=${reversed}`, () => {
      const observations: [number, number] = [2 ** 46, 2 ** 47];
      if (reversed) observations.reverse();
      const result = runHostMomentProbe({
        route: 'calibration',
        observations,
        split,
        distribution: 'fisk'
      });
      // mean=3*2^45, population SD=2^45; shape=pi*sqrt(3); scale=mean/(1+1/18).
      expect([...result.moments]).toEqual([2 ** 91]);
      expect(result.fields.fit).toHaveLength(3);
      expect(result.fields.fit.every(Number.isFinite)).toBe(true);
      expect(result.fields.fit[0]).toBeCloseTo(5.441398092702653, 5);
      expect(Math.abs(result.fields.fit[1] / 99997689094575.16 - 1)).toBeLessThan(3e-7);
      expect(result.fields.fit[2]).toBe(0);
    });
  }
}

for (const route of ['streaming', 'calibration'] as const) {
  it(`S1 rejects extra ${route} moment words`, () => {
    const length = route === 'streaming' ? 3 : 9;
    for (const size of [0, length - 1, length + 1, length + 3]) {
      expect(() => decodeMomentReadback(new Uint32Array(size), route, 0)).toThrow();
    }
  });
}

it('S2 qualifies unavailable-calibration documentation by valid active input', () => {
  const page = readFileSync(
    new URL('../../../../../docs/api/gpu-raster/raster-time-series.md', import.meta.url),
    'utf8'
  );
  expect(page).not.toContain('a slot without a usable fit marks every output NaN');
  expect(page).toContain('active, finite, unmasked');
});

for (const split of [false, true]) {
  for (const reversed of [false, true]) {
    it(`B2 preserves nearby large-location subtraction split=${split} reversed=${reversed}`, () => {
      const observations: [number, number] = [2 ** 70 + 2 ** 47, 2 ** 70 + 2 ** 48];
      if (reversed) observations.reverse();
      const result = runHostMomentProbe({
        route: 'calibration',
        observations,
        split,
        location: 2 ** 70,
        maximumValueMagnitude: 2 ** 48
      });
      // Sterbenz-exact differences 2^47 and 2^48, whose M2 is (2^47)^2/2=2^93.
      expect([...result.moments]).toEqual([2 ** 93]);
    });
  }
}
