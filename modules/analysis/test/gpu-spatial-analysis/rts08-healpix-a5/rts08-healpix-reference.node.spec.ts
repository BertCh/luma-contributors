// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/rts08-healpix-geo.json';
import {
  getAngleDegrees,
  getHealpixNestedCenter,
  getHealpixNestedPixel
} from './rts08-healpix-reference';

/** Cells that meet at the equator at longitudes of 0, 45, 90, ... degrees. */
const isVertexPoint = (longitude: number, latitude: number) =>
  latitude === 0 && longitude % 45 === 0;
const getCellWidthDegrees = (depth: number) => 58.6 / 2 ** depth;

describe('binary64 HEALPix nested reference', () => {
  for (const depth of fixture.depths) {
    it(`equals healpix-geo ${fixture.versions['healpix-geo']} at depth ${depth}`, () => {
      const expected = fixture.nestedHex[String(depth) as keyof typeof fixture.nestedHex];
      let mismatches = 0;
      let tiedMismatches = 0;
      fixture.longitude.forEach((longitude, row) => {
        const latitude = fixture.latitude[row];
        const pixel = getHealpixNestedPixel(longitude, latitude, depth);
        if (pixel.toString(16) === expected[row]) return;
        // A point exactly on a cell vertex (equator, longitude a multiple of 45) belongs to
        // several cells; the libraries break the tie differently. The tied cell must still be
        // one of the cells that meet there (its center within one cell diagonal).
        if (isVertexPoint(longitude, latitude)) {
          tiedMismatches++;
          const reference = getHealpixNestedCenter(BigInt(`0x${expected[row]}`), depth);
          const ours = getHealpixNestedCenter(pixel, depth);
          expect(getAngleDegrees(reference, ours)).toBeLessThan(2 * getCellWidthDegrees(depth));
        } else {
          mismatches++;
        }
      });
      expect(mismatches).toBe(0);
      expect(tiedMismatches).toBeLessThanOrEqual(6);
    });
  }

  it('maps the center of a cell back to the cell', () => {
    for (const depth of [0, 1, 3, 7, 12]) {
      const cells = 12 * 4 ** depth;
      for (let sample = 0; sample < 400; sample++) {
        const pixel = BigInt(Math.floor((sample * 2654435761) % cells));
        const center = getHealpixNestedCenter(pixel, depth);
        expect(getHealpixNestedPixel(center.longitude, center.latitude, depth)).toBe(pixel);
      }
    }
  });

  it('has cell centers one cell width from their neighbours', () => {
    const first = getHealpixNestedCenter(0n, 5);
    const second = getHealpixNestedCenter(1n, 5);
    expect(getAngleDegrees(first, second)).toBeGreaterThan(0);
    expect(getAngleDegrees(first, second)).toBeLessThan(4);
  });
});
