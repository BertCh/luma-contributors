// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GeoArrowBuilder} from '@math.gl/arrow/geoarrow-utils';
import type {GeoArrowColumn, GeoArrowGeometryValue} from '@math.gl/arrow/geoarrow-utils';
import {makeArrowVectorFromGeoArrowColumn} from '@math.gl/arrow/geoarrow';
import {
  prepareGeometryInput,
  type CPUSpatialJoinGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
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

/** Bridges every descriptor chunk to the public Arrow-vector preparation contract, preserving source row order and empty features. */
function prepareColumn(
  column: GeoArrowColumn,
  encoding: 'geoarrow.multipoint' | 'geoarrow.multilinestring' | 'geoarrow.multipolygon'
): CPUSpatialJoinGeometry {
  const id = `catalog-${encoding}`;
  validateRequirement(
    id,
    column.encoding === encoding,
    `encoding must be ${encoding}; received ${column.encoding}`
  );
  validateRequirement(
    id,
    column.dimension === 'xy',
    `dimension must be xy; received ${column.dimension}`
  );
  const metadataEdges = column.metadata?.['edges'];
  for (const edges of [column.edges, metadataEdges])
    validateRequirement(
      id,
      edges === undefined || edges === 'planar' || edges === 'spherical',
      `edges must be planar or spherical; received ${String(edges)}`
    );
  validateRequirement(
    id,
    column.edges === undefined || metadataEdges === undefined || column.edges === metadataEdges,
    `edges declarations conflict: received ${column.edges} and metadata.edges ${String(metadataEdges)}`
  );
  const edges = column.edges ?? metadataEdges;
  validateRequirement(
    id,
    edges !== 'spherical',
    'spherical edges require projection before preparing planar geometry'
  );
  // An Arrow Vector cannot be constructed without any physical Data chunks. The canonical
  // zero-row hierarchy has one terminal offset at each level and no coordinate storage.
  if (!column.chunks.length)
    return {
      kind:
        encoding === 'geoarrow.multipoint'
          ? 'points'
          : encoding === 'geoarrow.multilinestring'
            ? 'lines'
            : 'polygons',
      positions: new Float32Array(0),
      featureOffsets: Uint32Array.of(0),
      ...(encoding === 'geoarrow.multilinestring' ? {lineOffsets: Uint32Array.of(0)} : {}),
      ...(encoding === 'geoarrow.multipolygon'
        ? {polygonOffsets: Uint32Array.of(0), ringOffsets: Uint32Array.of(0)}
        : {})
    };
  return prepareGeometryInput(makeArrowVectorFromGeoArrowColumn(column), {
    encoding: column.encoding,
    dimension: column.dimension,
    coordinateLayout: column.coordinateLayout,
    metadata: {...column.metadata, ...(edges === undefined ? {} : {edges})}
  }).geometry;
}

/** Packed arrays of a `geoarrow.multipolygon` column: feature, polygon and ring offsets. */
export function readGeoArrowMultiPolygon(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
} {
  const geometry = prepareColumn(column, 'geoarrow.multipolygon');
  if (
    geometry.kind !== 'polygons' ||
    !geometry.featureOffsets ||
    !geometry.polygonOffsets ||
    !geometry.ringOffsets
  )
    throw new Error('MultiPolygon preparation requires feature, polygon and ring offsets');
  return {
    positions: geometry.positions,
    featureOffsets: geometry.featureOffsets,
    polygonOffsets: geometry.polygonOffsets,
    ringOffsets: geometry.ringOffsets
  };
}

/** Packed arrays of a `geoarrow.multilinestring` column: feature and line offsets. */
export function readGeoArrowMultiLineString(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  lineOffsets: Uint32Array;
} {
  const geometry = prepareColumn(column, 'geoarrow.multilinestring');
  if (geometry.kind !== 'lines' || !geometry.featureOffsets || !geometry.lineOffsets)
    throw new Error('MultiLineString preparation requires feature and line offsets');
  return {
    positions: geometry.positions,
    featureOffsets: geometry.featureOffsets,
    lineOffsets: geometry.lineOffsets
  };
}

/** Packed arrays of every chunk of a `geoarrow.multipoint` column in source row order. */
export function readGeoArrowMultiPoint(column: GeoArrowColumn): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
} {
  const geometry = prepareColumn(column, 'geoarrow.multipoint');
  if (geometry.kind !== 'points' || !geometry.featureOffsets)
    throw new Error('MultiPoint preparation requires feature offsets');
  return {positions: geometry.positions, featureOffsets: geometry.featureOffsets};
}

/** Applies the optional projection and keeps the XY pair of a position. */
export function projectCatalogPosition(
  position: CatalogPosition,
  project?: (position: CatalogPosition) => [number, number]
): [number, number] {
  const [x, y] = project ? project(position) : position;
  if (!Number.isFinite(Math.fround(x)) || !Number.isFinite(Math.fround(y))) {
    throw new Error(
      'Geometry input coordinates must be finite after projection and float32 conversion; correct the source coordinates before retrying'
    );
  }
  return [x, y];
}
