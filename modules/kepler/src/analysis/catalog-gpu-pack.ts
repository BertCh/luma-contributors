// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createGPULocalFrame} from '@luma-contributors/analysis/analysis-core';
import type {GeoArrowColumn, GeoArrowGeometryValue} from '@math.gl/arrow/geoarrow-utils';
import {
  makeGeoArrowColumnFromCatalogGeometries,
  projectCatalogPosition,
  readGeoArrowMultiLineString,
  readGeoArrowMultiPoint,
  readGeoArrowMultiPolygon
} from './catalog-geoarrow';
import {
  iterateCatalogPositions,
  getLineParts,
  validateCatalogGeometryPositions,
  measureRingSignedArea,
  getPointPositions,
  getPolygonParts,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {CatalogGpuUnsupportedError} from './catalog-gpu-session';
import {makeLocalFrame} from './catalog-local-frame';

/** Maps a longitude/latitude position to planar frame coordinates, or translates planar input to a local origin. */
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
  for (const [x, y] of iterateCatalogPositions(collections)) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
}

/** Planar frames prefer exact float32 coordinate round trips; degrees use a local metric frame. */
export function createCatalogFrame(
  coordinateSystem: 'EPSG:4326' | 'planar',
  collections: readonly CatalogFeatureCollection[]
): CatalogFrame {
  if (coordinateSystem === 'planar') {
    const positions = Array.from(iterateCatalogPositions(collections));
    const centered = makeLocalFrame('planar', positions, {shift: 'center'});
    // Try the shared center first, then raw coordinates and the data minimum. Never merge
    // distinct positions merely because shifting moved them into a coarser float32 bin.
    const candidates = [
      centered,
      {forward: CATALOG_IDENTITY_FRAME.project, inverse: CATALOG_IDENTITY_FRAME.unproject},
      makeLocalFrame('planar', positions, {shift: 'minimum'})
    ];
    let usableFrame: CatalogFrame | undefined;
    for (const frame of candidates) {
      const originals = new Map<string, string>();
      let usable = true;
      let roundTrips = true;
      for (const position of positions) {
        const [x, y] = frame.forward(position);
        const packedX = Math.fround(x),
          packedY = Math.fround(y);
        const key = `${packedX}|${packedY}`,
          original = `${position[0]}|${position[1]}`;
        if (
          !Number.isFinite(packedX) ||
          !Number.isFinite(packedY) ||
          (originals.has(key) && originals.get(key) !== original)
        ) {
          usable = false;
          break;
        }
        const restored = frame.inverse(packedX, packedY);
        if (restored[0] !== position[0] || restored[1] !== position[1]) roundTrips = false;
        originals.set(key, original);
      }
      if (usable) {
        const candidate: CatalogFrame = {
          kind: 'identity',
          project: frame.forward,
          unproject: frame.inverse
        };
        // A noncollapsing frame can still distort an edge. Prefer exact round trips when a
        // candidate provides them, retaining the existing finite, noncollapsing fallback.
        if (roundTrips) return candidate;
        usableFrame ??= candidate;
      }
    }
    if (usableFrame) return usableFrame;
    throw new CatalogGpuUnsupportedError(
      'Planar geometry input has distinct coordinates that coincide after float32 packing; use the CPU engine or split the extent'
    );
  }
  // The shared local frame picks an antimeridian aware origin at the data extent center.
  const positions = Array.from(iterateCatalogPositions(collections));
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
    validateCatalogGeometryPositions(feature.geometry);
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
    validateCatalogGeometryPositions(feature.geometry);
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
    validateCatalogGeometryPositions(feature.geometry);
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
  return measureRingSignedArea(ring, 'planar');
}

/**
 * Reads polygons from packed GeoArrow arrays. Rings are closed and unprojected; polygons whose
 * shell is nonfinite, has fewer than three distinct corners or has zero planar area are dropped,
 * as are unusable holes. Nested offsets must stay within the readable prefix. Byte inputs
 * are interpreted as float32 XY and uint32 offsets, including their byte offsets.
 */
export function readCatalogPolygons(
  positions: ArrayLike<number> | ArrayBuffer,
  ringOffsets: ArrayLike<number> | ArrayBuffer,
  polygonOffsets: ArrayLike<number> | ArrayBuffer,
  polygonRange: readonly [number, number],
  unproject: (x: number, y: number) => [number, number] = (x, y) => [x, y]
): CatalogPosition[][][] {
  const readWords = (
    values: ArrayLike<number> | ArrayBuffer,
    floating: boolean
  ): ArrayLike<number> => {
    if (!(values instanceof Uint8Array) && !(values instanceof ArrayBuffer)) return values;
    const bytes = values instanceof Uint8Array ? values : new Uint8Array(values);
    if (bytes.byteLength % 4)
      throw new Error('polygon-readback: byte length must contain whole 32-bit elements');
    const aligned = bytes.byteOffset % 4 ? bytes.slice() : bytes;
    return floating
      ? new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4)
      : new Uint32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
  };
  positions = readWords(positions, true);
  ringOffsets = readWords(ringOffsets, false);
  polygonOffsets = readWords(polygonOffsets, false);
  const offset = (values: ArrayLike<number>, index: number, maximum: number): number => {
    const value = values[index];
    if (!Number.isInteger(value) || value < 0 || value > maximum)
      throw new Error('polygon-readback: offset exceeds readable prefix');
    return value;
  };
  if (
    positions.length % 2 ||
    !Number.isInteger(polygonRange[0]) ||
    !Number.isInteger(polygonRange[1]) ||
    polygonRange[0] < 0 ||
    polygonRange[1] < polygonRange[0] ||
    polygonRange[1] >= polygonOffsets.length
  )
    throw new Error('polygon-readback: invalid XY length or polygon offset range');
  const polygons: CatalogPosition[][][] = [];
  for (let polygon = polygonRange[0]; polygon < polygonRange[1]; polygon++) {
    const rings: CatalogPosition[][] = [];
    const firstRing = offset(polygonOffsets, polygon, ringOffsets.length - 1);
    const endRing = offset(polygonOffsets, polygon + 1, ringOffsets.length - 1);
    if (endRing < firstRing) throw new Error('polygon-readback: decreasing polygon offset');
    for (let ring = firstRing; ring < endRing; ring++) {
      const vertices: CatalogPosition[] = [];
      const firstVertex = offset(ringOffsets, ring, positions.length / 2);
      const endVertex = offset(ringOffsets, ring + 1, positions.length / 2);
      if (endVertex < firstVertex) throw new Error('polygon-readback: decreasing ring offset');
      for (let vertex = firstVertex; vertex < endVertex; vertex++)
        vertices.push(unproject(positions[2 * vertex], positions[2 * vertex + 1]));
      const dropped = dropClosingPosition(vertices) as CatalogPosition[];
      const distinct = new Set(dropped.map(([x, y]) => `${x}|${y}`));
      const usable =
        distinct.size >= 3 &&
        dropped.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)) &&
        Number.isFinite(getCatalogRingArea(dropped)) &&
        getCatalogRingArea(dropped) !== 0;
      if (!usable && ring === firstRing) break;
      if (usable) rings.push(closeCatalogRing([...dropped]));
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
  for (let index = 0; index < corners.length; index++) {
    const next = corners[(index + 1) % corners.length];
    if (corners[index][0] !== next[0] && corners[index][1] !== next[1]) return null;
  }
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
