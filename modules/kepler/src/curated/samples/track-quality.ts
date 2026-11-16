// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';

export const TRACK_QUALITY_SAMPLE_LICENSE = {
  id: 'track-quality',
  license: 'MIT',
  attribution: 'vis.gl contributors',
  reference: 'THIRD-PARTY-DATA.md#curated-track-quality-synthetic-sample'
} as const;

/** Recipe-review interior outlier, a long gap, and a singleton; coordinates are planar metres. */
export function createTrackQualitySample(seedOffset = 0): CatalogFeatureCollection {
  const origin = Date.UTC(2026, 0, 1) + seedOffset * 1000;
  const shift = seedOffset * 3;
  const scale = 1 + Math.abs(seedOffset % 7) / 10;
  const observations = [
    {track: 'outlier', horizontal: 0, time: 0},
    {track: 'outlier', horizontal: 1, time: 1000},
    {track: 'outlier', horizontal: 502 * scale, time: 2000},
    {track: 'outlier', horizontal: 3, time: 3000},
    {track: 'outlier', horizontal: 4, time: 4000},
    {track: 'gap', horizontal: 0, time: 0},
    {track: 'gap', horizontal: 1, time: 120000},
    {track: 'singleton', horizontal: 10, time: 500}
  ];
  // Interleave tracks and scramble time order to exercise stable identity packing.
  return {
    type: 'FeatureCollection',
    features: [3, 6, 0, 7, 2, 5, 4, 1].map(index => ({
      id: `observation-${index}`,
      geometry: {type: 'Point', coordinates: [observations[index].horizontal + shift, 0, index]},
      properties: {
        track_id: observations[index].track,
        timestamp: origin + observations[index].time
      }
    }))
  };
}
