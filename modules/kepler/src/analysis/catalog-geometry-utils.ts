// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogPosition} from './catalog-geojson';

/**
 * Maps every position of a nested GeoJSON coordinates value. Ordinates after the first two (z, m)
 * are carried through unchanged so 3D geometry is not flattened by planar edits.
 */
export function mapGeometryCoordinates(
  value: unknown,
  mapPosition: (position: CatalogPosition) => CatalogPosition
): unknown {
  if (!Array.isArray(value)) return value;
  if (typeof value[0] === 'number') {
    const mapped = mapPosition([value[0], value[1]]);
    return value.length > 2 ? [...mapped, ...value.slice(2)] : mapped;
  }
  return value.map(child => mapGeometryCoordinates(child, mapPosition));
}
