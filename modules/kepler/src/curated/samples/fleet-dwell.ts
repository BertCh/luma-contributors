// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';

/** Repository-authored observations; no real vehicles, locations or third-party records. */
export const FLEET_DWELL_SAMPLE_LICENSE = {
  id: 'fleet-dwell',
  reference: 'modules/kepler/src/curated/samples/fleet-dwell.ts',
  license: 'MIT',
  attribution: 'vis.gl contributors'
} as const;

/** 32 tracks, each with four stationary runs; two inside zones and two outside. */
export function createFleetDwellSample(seedOffset = 0): Record<string, CatalogFeatureCollection> {
  const project = (horizontal: number, vertical: number) => [
    -87.65 + horizontal / 83000,
    41.85 + vertical / 111320
  ];
  const tracks: CatalogFeatureCollection = {type: 'FeatureCollection', features: []};
  for (let track = 0; track < 32; track++) {
    const jitter = ((track * 17 + seedOffset * 13) % 19) / 10 - 0.9;
    const interval = 10 + ((track + seedOffset) % 5);
    for (let fix = 0; fix < 12; fix++) {
      tracks.features.push({
        id: `fix-${track}-${fix}`,
        geometry: {
          type: 'Point',
          coordinates: project(Math.floor(fix / 3) * 40 + jitter, (track % 5) - 2)
        },
        properties: {
          track_id: `vehicle-${track}`,
          timestamp: Date.UTC(2024, 0, 1) + fix * interval * 1000
        }
      });
    }
  }
  const zones: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [-10, 30, 150].map((minimum, index) => ({
      id: ['depot-west', 'depot-east', 'unvisited'][index],
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [minimum, -10],
            [minimum + 20, -10],
            [minimum + 20, 10],
            [minimum, 10],
            [minimum, -10]
          ].map(([horizontal, vertical]) => project(horizontal, vertical))
        ]
      },
      properties: {name: ['West depot', 'East depot', 'Unvisited depot'][index]}
    }))
  };
  return {tracks, zones};
}
