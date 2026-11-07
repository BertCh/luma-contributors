// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GeoArrowBuilder} from '@math.gl/geospatial/geoarrow';
import type {
  GeoArrowColumn,
  GeoArrowGeometryValue,
  GeoArrowList
} from '@math.gl/geospatial/geoarrow';
import type {CatalogPosition} from './catalog-geojson';

/** Encodes one row family of GeoJSON-like coordinates into a Float32 interleaved XY GeoArrow column. */
export function makeGeoArrowColumnFromCatalogGeometries(
  geometries: readonly GeoArrowGeometryValue[],
  encoding: 'geoarrow.multipoint' | 'geoarrow.multilinestring' | 'geoarrow.multipolygon'
): GeoArrowColumn {
  return GeoArrowBuilder.build(geometries, {
    encoding,
    dimension: 'xy',
    coordinateLayout: 'interleaved',
    coordinateType: 'float32',
    offsetType: 'int32'
  });
}

function getUint32Offsets(list: GeoArrowList): Uint32Array {
  // Builder offsets are non-negative Int32 values; reinterpret without copying.
  const offsets = list.offsets as Int32Array;
  return new Uint32Array(offsets.buffer, offsets.byteOffset, offsets.length);
}

function getInterleavedPositions(coordinates: unknown, vertexCount: number): Float32Array {
  const child = (coordinates as {child: {values: Float32Array}}).child;
  return child.values.subarray(0, vertexCount * 2);
}

/** Packed arrays of a `geoarrow.multipolygon` column: feature, polygon and ring offsets. */
export function readGeoArrowMultiPolygon(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
} {
  const features = column.chunks[0] as GeoArrowList;
  const polygons = features.child as GeoArrowList;
  const rings = polygons.child as GeoArrowList;
  const ringOffsets = getUint32Offsets(rings);
  return {
    positions: getInterleavedPositions(rings.child, ringOffsets[ringOffsets.length - 1]),
    featureOffsets: getUint32Offsets(features),
    polygonOffsets: getUint32Offsets(polygons),
    ringOffsets
  };
}

/** Packed arrays of a `geoarrow.multilinestring` column: feature and line offsets. */
export function readGeoArrowMultiLineString(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  lineOffsets: Uint32Array;
} {
  const features = column.chunks[0] as GeoArrowList;
  const lines = features.child as GeoArrowList;
  const lineOffsets = getUint32Offsets(lines);
  return {
    positions: getInterleavedPositions(lines.child, lineOffsets[lineOffsets.length - 1]),
    featureOffsets: getUint32Offsets(features),
    lineOffsets
  };
}

/** Packed arrays of a `geoarrow.multipoint` column, read straight from the GeoArrow list. */
export function readGeoArrowMultiPoint(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
} {
  const features = column.chunks[0] as GeoArrowList;
  const featureOffsets = getUint32Offsets(features);
  return {
    positions: getInterleavedPositions(features.child, featureOffsets[featureOffsets.length - 1]),
    featureOffsets
  };
}

/** Applies the optional projection and keeps the XY pair of a position. */
export function projectCatalogPosition(
  position: CatalogPosition,
  project?: (position: CatalogPosition) => [number, number]
): [number, number] {
  const [x, y] = project ? project(position) : position;
  return [x, y];
}
