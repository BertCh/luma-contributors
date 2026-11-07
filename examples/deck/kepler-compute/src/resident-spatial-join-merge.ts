// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, Vector} from 'apache-arrow';
import type {PreparedPolygonTopologySource} from '@luma.gl/engine/arrow';

type Position = number[];
type PolygonCoordinates = Position[][];
export type SpatialJoinPolygonGeometry =
  | {type: 'Polygon'; coordinates: PolygonCoordinates}
  | {type: 'MultiPolygon'; coordinates: PolygonCoordinates[]};

/** Total UTF-16 characters of merged GeoJSON labels that one merge may retain (64 MiB). */
export const MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_LABEL_LENGTH = 33554432;

/**
 * Reads the original-coordinate GeoJSON geometry of one retained polygon row. Rows with one part
 * are Polygons and rows with several parts are MultiPolygons; ring longitude and latitude are copied unchanged.
 */
export function readSpatialJoinPolygonGeometry(
  source: PreparedPolygonTopologySource,
  rowIndex: number
): SpatialJoinPolygonGeometry {
  const batch = source.batches.find(
    candidate =>
      rowIndex >= candidate.sourceRowOffset &&
      rowIndex < candidate.sourceRowOffset + candidate.rowCount
  );
  if (!batch) throw new Error('Spatial Join polygon row is unavailable');
  const parts = readList(batch.polygons, rowIndex - batch.sourceRowOffset);
  const polygons: PolygonCoordinates[] = [];
  for (let partIndex = 0; parts && partIndex < parts.length; partIndex++) {
    const rings = readList(parts, partIndex);
    const polygon: PolygonCoordinates = [];
    for (let ringIndex = 0; rings && ringIndex < rings.length; ringIndex++) {
      const vertices = readList(rings, ringIndex);
      const ring: Position[] = [];
      for (let vertexIndex = 0; vertices && vertexIndex < vertices.length; vertexIndex++) {
        const coordinate = vertices.get(vertexIndex);
        if (!(coordinate instanceof Vector)) throw new Error('Invalid polygon vertex');
        // Retained vertices are padded to three components; GeoJSON merge keeps longitude and latitude.
        ring.push([Number(coordinate.get(0)), Number(coordinate.get(1))]);
      }
      polygon.push(ring);
    }
    polygons.push(polygon);
  }
  return polygons.length === 1
    ? {type: 'Polygon', coordinates: polygons[0]}
    : {type: 'MultiPolygon', coordinates: polygons};
}

/**
 * Native mergeGeometries over polygon rows in matched order: none give null, one keeps its own
 * geometry, several Polygons or several MultiPolygons become one MultiPolygon, and a mix becomes a
 * GeometryCollection. The result is a GeoJSON Feature string.
 */
export function mergeSpatialJoinGeometries(
  geometries: readonly SpatialJoinPolygonGeometry[]
): string | null {
  if (!geometries.length) return null;
  let geometry: object;
  if (geometries.length === 1) {
    geometry = geometries[0];
  } else if (geometries.every(candidate => candidate.type === 'Polygon')) {
    geometry = {type: 'MultiPolygon', coordinates: geometries.map(({coordinates}) => coordinates)};
  } else if (geometries.every(candidate => candidate.type === 'MultiPolygon')) {
    geometry = {
      type: 'MultiPolygon',
      coordinates: geometries.flatMap(({coordinates}) => coordinates as PolygonCoordinates[])
    };
  } else {
    geometry = {type: 'GeometryCollection', geometries};
  }
  return JSON.stringify({type: 'Feature', geometry, properties: {}});
}

function readList(vector: Vector, rowIndex: number): Vector | null {
  if (!DataType.isList(vector.type)) throw new Error('Invalid polygon list type');
  const value = vector.get(rowIndex);
  if (value === null) return null;
  if (!(value instanceof Vector)) throw new Error('Invalid polygon list value');
  return value;
}
