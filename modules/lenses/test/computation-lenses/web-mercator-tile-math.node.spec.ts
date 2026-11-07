// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {lngLatToWorld, worldToLngLat} from '@math.gl/geospatial';

const WORLD_SIZE = 512;

function getLegacyTile(lng: number, lat: number, z: number): [number, number] {
  const tiles = 2 ** z;
  const sine = Math.sin((lat * Math.PI) / 180);
  return [
    Math.floor(((lng + 180) / 360) * tiles),
    Math.floor((0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * tiles)
  ];
}

test('math.gl world coordinates reproduce hand-rolled Web Mercator tile math', () => {
  for (const zoom of [0, 3, 10, 16]) {
    for (const [lng, lat] of [
      [-87.65, 41.9],
      [0.001, -0.001],
      [179.9, 84],
      [-179.9, -84]
    ] as const) {
      const tiles = 2 ** zoom;
      const [worldX, worldY] = lngLatToWorld([lng, lat]);
      expect([
        Math.floor((worldX / WORLD_SIZE) * tiles),
        Math.floor((1 - worldY / WORLD_SIZE) * tiles)
      ]).toEqual(getLegacyTile(lng, lat, zoom));
    }
  }
});

test('math.gl tile centre matches the closed-form inverse', () => {
  const tiles = 2 ** 10;
  const [longitude, latitude] = worldToLngLat([
    ((300 + 0.5) / tiles) * WORLD_SIZE,
    (1 - (400 + 0.5) / tiles) * WORLD_SIZE
  ]);
  expect(longitude).toBeCloseTo((300.5 / tiles) * 360 - 180, 9);
  expect(latitude).toBeCloseTo(
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * 400.5) / tiles))) * 180) / Math.PI,
    9
  );
});
