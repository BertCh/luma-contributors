// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createGPULocalFrame} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {GeoArrowColumn, GeoArrowGeometryValue} from '@math.gl/geospatial/geoarrow';
import {
  makeGeoArrowColumnFromCatalogGeometries,
  projectCatalogPosition,
  readGeoArrowMultiLineString,
  readGeoArrowMultiPoint,
  readGeoArrowMultiPolygon
} from './catalog-geoarrow';
import {
  getLineParts,
  getPointPositions,
  getPolygonParts,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {makeLocalFrame} from './catalog-local-frame';

/** Maps a longitude/latitude position to planar frame coordinates, or leaves planar input unchanged. */
export type CatalogFrame = {
  readonly kind: 'identity' | 'azimuthal-equidistant';
  project(position: CatalogPosition): [number, number];
  unproject(x: number, y: number): [number, number];
};

export const CATALOG_IDENTITY_FRAME: CatalogFrame = {
  kind: 'identity',
  project: position => [position[0], position[1]],
  unproject: (x, y) => [x, y]
};

/**
 * Spherical azimuthal-equidistant frame centered on a longitude/latitude, in meters on the mean
 * Earth radius. Distances from the center are exact; away from it scale distorts, so it suits
 * extents of a few hundred kilometers.
 */
export function createCatalogAzimuthalFrame(
  centerLongitude: number,
  centerLatitude: number
): CatalogFrame {
  const gpuFrame = createGPULocalFrame({origin: [centerLongitude, centerLatitude]});
  return {
    kind: 'azimuthal-equidistant',
    project: position => gpuFrame.project(position[0], position[1]),
    unproject: (x, y) => gpuFrame.unproject(x, y)
  };
}

export function getCatalogBounds(
  collections: readonly CatalogFeatureCollection[]
): [number, number, number, number] | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const collection of collections)
    for (const feature of collection.features) {
      const positions: CatalogPosition[] = [
        ...getPointPositions(feature.geometry),
        ...getLineParts(feature.geometry).flat(),
        ...getPolygonParts(feature.geometry).flat(2)
      ];
      for (const [x, y] of positions) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
}

/** Frame for the declared coordinate system: planar input is used as is, degrees use a local frame. */
export function createCatalogFrame(
  coordinateSystem: 'EPSG:4326' | 'planar',
  collections: readonly CatalogFeatureCollection[]
): CatalogFrame {
  if (coordinateSystem === 'planar') return CATALOG_IDENTITY_FRAME;
  // The shared local frame picks an antimeridian aware origin at the data extent center.
  const positions: CatalogPosition[] = [];
  for (const collection of collections)
    for (const feature of collection.features)
      positions.push(
        ...getPointPositions(feature.geometry),
        ...getLineParts(feature.geometry).flat(),
        ...getPolygonParts(feature.geometry).flat(2)
      );
  const gpuFrame = makeLocalFrame(coordinateSystem, positions).gpuFrame!;
  return {
    kind: 'azimuthal-equidistant',
    project: position => gpuFrame.project(position[0], position[1]),
    unproject: (x, y) => gpuFrame.unproject(x, y)
  };
}

export type PackedPolygons = {
  /** Source GeoArrow column the arrays below view; usable with upstream GeoArrow/CPUVector adapters. */
  geoarrow: GeoArrowColumn;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Source feature index of every packed feature. */
  sourceIndices: number[];
  vertexCount: number;
  ringCount: number;
};

export type PackedLines = {
  /** Source GeoArrow column the arrays below view; usable with upstream GeoArrow/CPUVector adapters. */
  geoarrow: GeoArrowColumn;
  positions: Float32Array;
  /** `lineCount + 1` vertex offsets; one entry per line part. */
  lineOffsets: Uint32Array;
  /** Features own consecutive lines: `featureCount + 1` line offsets. */
  featureOffsets: Uint32Array;
  /** Source feature index of every packed feature. */
  sourceIndices: number[];
  /** Source feature index of every line. */
  lineSourceIndices: number[];
  /** Part index within its feature of every line. */
  linePartIndices: number[];
  vertexCount: number;
};

export type PackedPoints = {
  /** Source GeoArrow column the arrays below view; usable with upstream GeoArrow/CPUVector adapters. */
  geoarrow: GeoArrowColumn;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  sourceIndices: number[];
  vertexCount: number;
};

function dropClosingPosition(ring: readonly CatalogPosition[]): readonly CatalogPosition[] {
  const first = ring[0];
  const last = ring[ring.length - 1];
  return ring.length > 1 && first[0] === last[0] && first[1] === last[1] ? ring.slice(0, -1) : ring;
}

type CatalogProject = (position: CatalogPosition) => [number, number];

/**
 * Packs features that hold polygon parts into a `geoarrow.multipolygon` column and exposes its
 * four-level layout; others are skipped. `closing: 'drop'` removes each ring's closing vertex.
 */
export function packCatalogPolygons(
  features: readonly CatalogFeature[],
  options: {project?: CatalogProject; closing: 'keep' | 'drop'}
): PackedPolygons {
  const rows: GeoArrowGeometryValue[] = [];
  const sourceIndices: number[] = [];
  features.forEach((feature, sourceIndex) => {
    const polygons = getPolygonParts(feature.geometry);
    if (!polygons.length) return;
    rows.push({
      type: 'MultiPolygon',
      coordinates: polygons.map(rings =>
        rings.map(ring =>
          (options.closing === 'drop' ? dropClosingPosition(ring) : ring).map(position =>
            projectCatalogPosition(position, options.project)
          )
        )
      )
    });
    sourceIndices.push(sourceIndex);
  });
  const geoarrow = makeGeoArrowColumnFromCatalogGeometries(rows, 'geoarrow.multipolygon');
  const packed = readGeoArrowMultiPolygon(geoarrow);
  return {
    ...packed,
    geoarrow,
    sourceIndices,
    vertexCount: packed.positions.length / 2,
    ringCount: packed.ringOffsets.length - 1
  };
}

/** Packs features that hold line parts into a `geoarrow.multilinestring` column; every part is one line. */
export function packCatalogLines(
  features: readonly CatalogFeature[],
  options: {project?: CatalogProject} = {}
): PackedLines {
  const rows: GeoArrowGeometryValue[] = [];
  const sourceIndices: number[] = [];
  const lineSourceIndices: number[] = [];
  const linePartIndices: number[] = [];
  features.forEach((feature, sourceIndex) => {
    const parts = getLineParts(feature.geometry);
    if (!parts.length) return;
    rows.push({
      type: 'MultiLineString',
      coordinates: parts.map(part =>
        part.map(position => projectCatalogPosition(position, options.project))
      )
    });
    parts.forEach((_, partIndex) => {
      lineSourceIndices.push(sourceIndex);
      linePartIndices.push(partIndex);
    });
    sourceIndices.push(sourceIndex);
  });
  const geoarrow = makeGeoArrowColumnFromCatalogGeometries(rows, 'geoarrow.multilinestring');
  const packed = readGeoArrowMultiLineString(geoarrow);
  return {
    ...packed,
    geoarrow,
    sourceIndices,
    lineSourceIndices,
    linePartIndices,
    vertexCount: packed.positions.length / 2
  };
}

/** Packs features that hold points into a `geoarrow.multipoint` column; multipoints own consecutive points. */
export function packCatalogPoints(
  features: readonly CatalogFeature[],
  options: {project?: CatalogProject} = {}
): PackedPoints {
  const rows: GeoArrowGeometryValue[] = [];
  const sourceIndices: number[] = [];
  features.forEach((feature, sourceIndex) => {
    const points = getPointPositions(feature.geometry);
    if (!points.length) return;
    rows.push({
      type: 'MultiPoint',
      coordinates: points.map(position => projectCatalogPosition(position, options.project))
    });
    sourceIndices.push(sourceIndex);
  });
  const geoarrow = makeGeoArrowColumnFromCatalogGeometries(rows, 'geoarrow.multipoint');
  const packed = readGeoArrowMultiPoint(geoarrow);
  return {...packed, geoarrow, sourceIndices, vertexCount: packed.positions.length / 2};
}

/** Closes a ring in place of a missing closing vertex. */
export function closeCatalogRing(ring: CatalogPosition[]): CatalogPosition[] {
  const first = ring[0];
  const last = ring[ring.length - 1];
  return ring.length && (first[0] !== last[0] || first[1] !== last[1]) ? [...ring, first] : ring;
}

/** Signed shoelace area of a ring (counterclockwise positive). */
export function getCatalogRingArea(ring: readonly CatalogPosition[]): number {
  let doubled = 0;
  for (let index = 0; index < ring.length; index++) {
    const next = ring[(index + 1) % ring.length];
    doubled += ring[index][0] * next[1] - next[0] * ring[index][1];
  }
  return doubled / 2;
}

/**
 * Reads polygons from packed GeoArrow arrays. Rings are closed and unprojected; polygons whose
 * shell has fewer than three vertices are dropped, as are degenerate holes.
 */
export function readCatalogPolygons(
  positions: ArrayLike<number>,
  ringOffsets: ArrayLike<number>,
  polygonOffsets: ArrayLike<number>,
  polygonRange: readonly [number, number],
  unproject: (x: number, y: number) => [number, number] = (x, y) => [x, y]
): CatalogPosition[][][] {
  const polygons: CatalogPosition[][][] = [];
  for (let polygon = polygonRange[0]; polygon < polygonRange[1]; polygon++) {
    const rings: CatalogPosition[][] = [];
    for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
      const vertices: CatalogPosition[] = [];
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++)
        vertices.push(unproject(positions[2 * vertex], positions[2 * vertex + 1]));
      const dropped = dropClosingPosition(vertices) as CatalogPosition[];
      if (dropped.length >= 3) rings.push(closeCatalogRing([...dropped]));
    }
    if (rings.length) polygons.push(rings);
  }
  return polygons;
}

/** Area of shells minus holes, in the units of the ring coordinates. */
export function getCatalogPolygonArea(rings: readonly (readonly CatalogPosition[])[]): number {
  return rings.reduce(
    (sum, ring, index) => sum + (index === 0 ? 1 : -1) * Math.abs(getCatalogRingArea(ring)),
    0
  );
}

/** Whether a point lies in a polygon (shell minus holes) by ray casting. */
export function isCatalogPointInPolygon(
  point: CatalogPosition,
  rings: readonly (readonly CatalogPosition[])[]
): boolean {
  const insideRing = (ring: readonly CatalogPosition[]): boolean => {
    let inside = false;
    for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
      const [ax, ay] = ring[previous];
      const [bx, by] = ring[index];
      if (
        ay > point[1] !== by > point[1] &&
        point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax
      )
        inside = !inside;
    }
    return inside;
  };
  return insideRing(rings[0]) && !rings.slice(1).some(insideRing);
}

/** Axis-aligned rectangle of a four-corner ring, or null when the ring is anything else. */
export function getCatalogRectangle(
  ring: readonly CatalogPosition[]
): {minX: number; minY: number; maxX: number; maxY: number} | null {
  const corners = dropClosingPosition(ring);
  if (corners.length !== 4) return null;
  const xs = [...new Set(corners.map(corner => corner[0]))];
  const ys = [...new Set(corners.map(corner => corner[1]))];
  if (xs.length !== 2 || ys.length !== 2) return null;
  // Four distinct corners with two x values and two y values: every edge is axis aligned.
  const distinct = new Set(corners.map(corner => `${corner[0]}|${corner[1]}`));
  if (distinct.size !== 4) return null;
  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs),
    maxY: Math.max(...ys)
  };
}
