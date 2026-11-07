// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  decodeA5Footprint,
  decodeGeohashFootprint,
  decodeS2Footprint,
  type DGGSFamily
} from './dggs-source';
import type {ViewerDataset} from './standalone-viewer-data';

/** Decodes S2, A5 and geohash tokens into polygon features with the host's own decoders. */
export function decodeCellFeatures(
  family: DGGSFamily,
  dataset: ViewerDataset,
  tokenField: string,
  rows: Uint32Array
): any[] {
  const tokens = dataset.readStrings(tokenField);
  const features: any[] = [];
  if (!tokens) return features;
  for (const row of rows) {
    const token = tokens[row];
    if (!token) continue;
    const ring =
      family === 'geohash'
        ? decodeGeohashFootprint(token)
        : family === 's2'
          ? decodeS2Footprint(token)
          : decodeA5Footprint(token);
    if (typeof ring === 'string') continue; // invalid token: skipped, like native
    const coordinates: [number, number][] = [];
    for (let index = 0; index < ring.length; index += 2)
      coordinates.push([ring[index], ring[index + 1]]);
    features.push({
      type: 'Feature',
      geometry: {type: 'Polygon', coordinates: [coordinates]},
      properties: {__row: row}
    });
  }
  return features;
}
