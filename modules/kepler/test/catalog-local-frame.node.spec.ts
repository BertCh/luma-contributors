// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import type {CatalogDiagnostic} from '../src/analysis/catalog-geojson';
import {addLocalFrameDiagnostic, makeLocalFrame} from '../src/analysis/catalog-local-frame';

const POSITIONS: [number, number][] = [
  [10, 45],
  [11, 45],
  [11, 46],
  [10, 46]
];

describe('makeLocalFrame', () => {
  test('planar data is an identity frame unless shifted', () => {
    const identity = makeLocalFrame('planar', POSITIONS);
    expect(identity.gpuFrame).toBeNull();
    expect(identity.forward([3, 4])).toEqual([3, 4]);
    expect(identity.inverse(3, 4)).toEqual([3, 4]);
    const minimum = makeLocalFrame('planar', POSITIONS, {shift: 'minimum'});
    expect(minimum.forward([11, 46])).toEqual([1, 1]);
    expect(minimum.inverse(1, 1)).toEqual([11, 46]);
    const center = makeLocalFrame('planar', POSITIONS, {shift: 'center'});
    expect(center.forward([10.5, 45.5])).toEqual([0, 0]);
  });

  test('geographic frames are centered on the bounding box and round trip', () => {
    const frame = makeLocalFrame('EPSG:4326', POSITIONS);
    expect(frame.gpuFrame?.origin).toEqual([10.5, 45.5]);
    expect(frame.gpuFrame?.kind).toBe('azimuthal-equidistant');
    const [x, y] = frame.forward([10.5, 45.5]);
    expect(Math.hypot(x, y)).toBeLessThan(1e-6);
    const [longitude, latitude] = frame.inverse(...frame.forward([10.9, 45.2]));
    expect(longitude).toBeCloseTo(10.9, 9);
    expect(latitude).toBeCloseTo(45.2, 9);
  });

  test('minimum shift keeps geographic frame coordinates non-negative', () => {
    const frame = makeLocalFrame('EPSG:4326', POSITIONS, {shift: 'minimum'});
    for (const position of POSITIONS) {
      const [x, y] = frame.forward(position);
      expect(x).toBeGreaterThanOrEqual(-1e-6);
      expect(y).toBeGreaterThanOrEqual(-1e-6);
    }
    const [longitude, latitude] = frame.inverse(...frame.forward([10.2, 45.7]));
    expect(longitude).toBeCloseTo(10.2, 9);
    expect(latitude).toBeCloseTo(45.7, 9);
  });

  test('azimuthal equidistant keeps distances and Lambert equal-area keeps areas', () => {
    const equidistant = makeLocalFrame('EPSG:4326', POSITIONS);
    const [x0, y0] = equidistant.forward([10, 45]);
    const [x1, y1] = equidistant.forward([11, 46]);
    const greatCircleMeters = 135_786;
    expect(Math.hypot(x1 - x0, y1 - y0)).toBeGreaterThan(greatCircleMeters * 0.9999);
    expect(Math.hypot(x1 - x0, y1 - y0)).toBeLessThan(greatCircleMeters * 1.0001);
    const equalArea = makeLocalFrame('EPSG:4326', POSITIONS, {kind: 'lambert-equal-area'});
    expect(equalArea.gpuFrame?.kind).toBe('lambert-equal-area');
    const ring = POSITIONS.map(position => equalArea.forward(position));
    const area =
      Math.abs(
        ring.reduce(
          (sum, [x, y], index) =>
            sum + x * ring[(index + 1) % ring.length][1] - ring[(index + 1) % ring.length][0] * y,
          0
        )
      ) / 2;
    // One degree square near 45.5 degrees north: about 8.7e9 square meters.
    expect(area).toBeGreaterThan(8.6e9);
    expect(area).toBeLessThan(8.8e9);
  });

  test('empty positions give a frame at the origin', () => {
    const frame = makeLocalFrame('EPSG:4326', [], {shift: 'minimum'});
    expect(frame.gpuFrame?.origin).toEqual([0, 0]);
    expect(frame.forward([0, 0])).toEqual([0, 0]);
  });
});

describe('makeLocalFrame antimeridian and antipodes', () => {
  const ACROSS: [number, number][] = [
    [179, 10],
    [-179, 12],
    [179.5, 11],
    [-179.5, 11]
  ];

  test('data from 179 E to 179 W is centered on the antimeridian and round trips', () => {
    const frame = makeLocalFrame('EPSG:4326', ACROSS);
    const [originLongitude, originLatitude] = frame.gpuFrame!.origin;
    expect(Math.abs(originLongitude)).toBeCloseTo(180, 9);
    expect(originLatitude).toBeCloseTo(11, 9);
    for (const position of ACROSS) {
      const [x, y] = frame.forward(position);
      expect(Math.abs(x)).toBeLessThan(150_000);
      expect(Math.abs(y)).toBeLessThan(150_000);
      const [longitude, latitude] = frame.inverse(x, y);
      const longitudeDelta = Math.abs(
        ((((longitude - position[0] + 540) % 360) + 360) % 360) - 180
      );
      expect(longitudeDelta).toBeLessThan(1e-9);
      expect(latitude).toBeCloseTo(position[1], 9);
    }
    // The old plain bounding box origin was at 0 degrees, antipodal to the data.
    expect(frame.getNonFiniteCount()).toBe(0);
  });

  test('wrap awareness keeps ordinary extents and shifted frames working', () => {
    const frame = makeLocalFrame(
      'EPSG:4326',
      [
        [-10, 0],
        [20, 5]
      ],
      {shift: 'minimum'}
    );
    expect(frame.gpuFrame!.origin[0]).toBeCloseTo(5, 9);
    const crossing = makeLocalFrame('EPSG:4326', ACROSS, {shift: 'minimum'});
    for (const position of ACROSS) {
      const [x, y] = crossing.forward(position);
      expect(x).toBeGreaterThanOrEqual(-1e-6);
      expect(y).toBeGreaterThanOrEqual(-1e-6);
    }
  });

  test('antipodal positions project to NaN and are counted', () => {
    const frame = makeLocalFrame('EPSG:4326', POSITIONS);
    const [x, y] = frame.forward([-169.5, -45.5]);
    expect(Number.isNaN(x) || Number.isNaN(y)).toBe(true);
    expect(frame.getNonFiniteCount()).toBe(1);
    frame.forward([10.5, 45.5]);
    expect(frame.getNonFiniteCount()).toBe(1);
    const diagnostics: CatalogDiagnostic[] = [];
    addLocalFrameDiagnostic(diagnostics, frame);
    expect(diagnostics).toEqual([expect.objectContaining({code: 'projection-nan', count: 1})]);
  });

  test('an explicit origin overrides the data center', () => {
    const frame = makeLocalFrame('EPSG:4326', POSITIONS, {origin: [0, 0]});
    expect(frame.gpuFrame!.origin).toEqual([0, 0]);
  });
});
