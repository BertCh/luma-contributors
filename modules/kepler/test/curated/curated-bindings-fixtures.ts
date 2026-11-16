// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser-safe fixtures: keep runtime dependencies out so headless specs can import them.
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';

/** Unit squares on a `columns` by `rows` lattice, row-major, with exact shared vertices (planar coordinates). */
export function makeLattice(
  columns: number,
  rows: number,
  getProperties: (index: number, column: number, row: number) => Record<string, unknown>
): CatalogFeatureCollection {
  const features: CatalogFeature[] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++)
      features.push({
        id: `a${row * columns + column}`,
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [column, row],
              [column + 1, row],
              [column + 1, row + 1],
              [column, row + 1],
              [column, row]
            ]
          ]
        },
        properties: getProperties(row * columns + column, column, row)
      });
  return {type: 'FeatureCollection', features};
}

export function makePoints(
  horizontalCoordinates: ArrayLike<number>,
  verticalCoordinates: ArrayLike<number>,
  getProperties: (index: number) => Record<string, unknown> = () => ({})
): CatalogFeatureCollection {
  const features: CatalogFeature[] = [];
  for (let index = 0; index < horizontalCoordinates.length; index++)
    features.push({
      id: `p${index}`,
      geometry: {
        type: 'Point',
        coordinates: [horizontalCoordinates[index], verticalCoordinates[index]]
      },
      properties: getProperties(index)
    });
  return {type: 'FeatureCollection', features};
}
