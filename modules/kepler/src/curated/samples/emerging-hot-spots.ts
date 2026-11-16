// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';

export const EMERGING_HOT_SPOTS_SAMPLE_LICENSE = {
  license: 'MIT',
  attribution:
    'vis.gl contributors; repository-authored synthetic events, no third-party observations',
  reference: 'THIRD-PARTY-DATA.md#curated-emerging-hot-spots-synthetic-sample'
} as const;

/** Same seeded event construction as space-time-hot-spots-recipe.spec.ts: 6 × 5 cells, 12 months. */
export function createEmergingHotSpotsSample(seedOffset = 0): CatalogFeatureCollection {
  let state = (31 + seedOffset) >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  const features: CatalogFeatureCollection['features'] = [];
  for (let cell = 0; cell < 30; cell++) {
    const column = cell % 6;
    const row = Math.floor(cell / 6);
    for (let month = 0; month < 12; month++) {
      let count = 8 + Math.floor(random() * 5);
      if (column >= 1 && column <= 3 && row >= 1 && row <= 2) count += 6 * month;
      else if (column >= 4 && row >= 3) count = Math.floor(random() * 2);
      for (let event = 0; event < count; event++) {
        const horizontal = column + 0.05 + 0.9 * random();
        const vertical = row + 0.05 + 0.9 * random();
        const observed = Date.UTC(2023, month, 1 + Math.floor(random() * 28), 12);
        features.push({
          id: `event-${features.length}`,
          geometry: {type: 'Point', coordinates: [horizontal, vertical]},
          properties: {observed}
        });
      }
    }
  }
  return {type: 'FeatureCollection', features};
}
