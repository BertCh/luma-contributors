// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

export function makePointCloudFixture(
  coordinates: number[][],
  classifications?: number[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, row) => ({
      id: `point-${row}`,
      geometry: {type: 'Point', coordinates: position},
      properties: {classification: classifications?.[row] ?? 2, intensity: row * 10}
    }))
  };
}

export const POINT_CLOUD_PLANE = makePointCloudFixture(
  Array.from({length: 9}, (_, row) => [1000000 + (row % 3), 4000000 + Math.floor(row / 3), 100])
);
export const POINT_CLOUD_CANOPY = makePointCloudFixture(
  [
    ...POINT_CLOUD_PLANE.features.map(feature => feature.geometry!.coordinates as number[]),
    [1000001, 4000001, 103]
  ],
  [...Array<number>(9).fill(2), 1]
);
