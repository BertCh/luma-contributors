// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, Field, FixedSizeList, Float64, List, Vector, makeData} from 'apache-arrow';
import type {Data} from 'apache-arrow';
import type {
  GeometryMetadata,
  PreparedGeometryBatch,
  PreparedKeplerGeometrySource
} from './geometry-gpu-vectors';

/** Native GeoArrow encodings handled without materializing per-row JavaScript objects. */
export type NativeGeoArrowEncoding =
  | 'geoarrow.point'
  | 'geoarrow.multipoint'
  | 'geoarrow.linestring'
  | 'geoarrow.multilinestring'
  | 'geoarrow.polygon'
  | 'geoarrow.multipolygon';

export type NumericValues = ArrayLike<number>;
type Kind = 'point' | 'path' | 'polygon';

export type EncodingLayout = {
  geometryType: string;
  kind: Kind;
  /** Number of nested List levels above the coordinate array. */
  listDepth: number;
  /** Offset level that maps a row to its parts (polygons); null means one part per row. */
  partLevel: number | null;
  /** Offset level that maps a part to its rings/paths; null means one ring per part. */
  ringLevel: number | null;
  /** Offset level that maps a ring/path to its vertices; null means one vertex per ring. */
  vertexLevel: number | null;
};

export const ENCODING_LAYOUTS: Record<NativeGeoArrowEncoding, EncodingLayout> = {
  'geoarrow.point': {
    geometryType: 'Point',
    kind: 'point',
    listDepth: 0,
    partLevel: null,
    ringLevel: null,
    vertexLevel: null
  },
  'geoarrow.multipoint': {
    geometryType: 'MultiPoint',
    kind: 'point',
    listDepth: 1,
    partLevel: null,
    ringLevel: null,
    vertexLevel: 0
  },
  'geoarrow.linestring': {
    geometryType: 'LineString',
    kind: 'path',
    listDepth: 1,
    partLevel: null,
    ringLevel: null,
    vertexLevel: 0
  },
  'geoarrow.multilinestring': {
    geometryType: 'MultiLineString',
    kind: 'path',
    listDepth: 2,
    partLevel: null,
    ringLevel: 0,
    vertexLevel: 1
  },
  'geoarrow.polygon': {
    geometryType: 'Polygon',
    kind: 'polygon',
    listDepth: 2,
    partLevel: null,
    ringLevel: 0,
    vertexLevel: 1
  },
  'geoarrow.multipolygon': {
    geometryType: 'MultiPolygon',
    kind: 'polygon',
    listDepth: 3,
    partLevel: 0,
    ringLevel: 1,
    vertexLevel: 2
  }
};

/** Reads vertex `index` component `c` as `values[index * stride + offset]`; arrays may be shared. */
export type CoordinateReader = {
  xValues: NumericValues;
  yValues: NumericValues;
  zValues: NumericValues | null;
  stride: number;
  xOffset: number;
  yOffset: number;
  zOffset: number;
};

export type NativeChunk = {
  data: Data;
  offsets: Int32Array[];
  coordinates: CoordinateReader;
  /** Validity bitmap of the top-level array only; nested arrays are required to be non-null. */
  validity: Uint8Array | null;
  validityOffset: number;
};

export type NativeGeoArrowOptions = {
  encoding: NativeGeoArrowEncoding;
  dimension: 'xy' | 'xyz';
  crs?: unknown;
  fp64?: boolean;
};

export function isNativeGeoArrowEncoding(encoding: unknown): encoding is NativeGeoArrowEncoding {
  return typeof encoding === 'string' && encoding in ENCODING_LAYOUTS;
}

/**
 * Prepares native point/line/polygon GeoArrow (interleaved or separated coordinates) by reading
 * the Arrow offset and coordinate buffers directly. Mirrors the semantics of the generic
 * GeoJSON-style route (finite float32 coordinates, ring closing, invalid row rejection,
 * `geoarrow-mean-center` representatives) without allocating per-row or per-vertex objects.
 * Returns null when the buffers need the generic route (nested nulls, unexpected layout).
 */
export function prepareNativeGeoArrowSource(
  vector: Vector,
  options: NativeGeoArrowOptions
): PreparedKeplerGeometrySource | null {
  const layout = ENCODING_LAYOUTS[options.encoding];
  const chunks: NativeChunk[] = [];
  for (const data of vector.data) {
    const chunk = readNativeChunk(data, layout, options.dimension);
    if (!chunk) return null;
    chunks.push(chunk);
  }

  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const featureTypes = {point: false, line: false, polygon: false};
  let hasZ = false;
  let hasPointAltitude = false;
  let hasValidRow = false;
  let sourceRowOffset = 0;
  const batches: PreparedGeometryBatch[] = [];
  for (const [chunkIndex, chunk] of chunks.entries()) {
    const rowCount = chunk.data.length;
    const prepared = prepareNativeChunk(chunk, layout, rowCount, sourceRowOffset, options.fp64);
    for (const invalid of prepared.invalidRows) invalidRows.push(invalid);
    hasZ ||= prepared.hasZ;
    hasPointAltitude ||= prepared.hasPointAltitude;
    hasValidRow ||= prepared.hasValidRow;
    featureTypes.point ||= prepared.batch.points.sourceRows.length > 0;
    featureTypes.line ||= prepared.batch.paths.sourceRows.length > 0;
    featureTypes.polygon ||= prepared.polygonCount > 0;
    bounds[0] = Math.min(bounds[0], prepared.bounds[0]);
    bounds[1] = Math.min(bounds[1], prepared.bounds[1]);
    bounds[2] = Math.max(bounds[2], prepared.bounds[2]);
    bounds[3] = Math.max(bounds[3], prepared.bounds[3]);
    batches.push({...prepared.batch, id: `geometry-batch-${chunkIndex}`});
    sourceRowOffset += rowCount;
  }
  const metadata: GeometryMetadata = {
    rowCount: sourceRowOffset,
    bounds: Number.isFinite(bounds[0]) ? bounds : null,
    crs: options.crs,
    hasZ,
    hasPointAltitude,
    sourceStyleProperties: [],
    fixedRadius: false,
    invalidStyleProperties: [],
    geometryTypes: hasValidRow ? [layout.geometryType] : [],
    featureTypes,
    invalidRows
  };
  const source: PreparedKeplerGeometrySource = {
    metadata,
    batches,
    representativeRecipe: 'geoarrow-mean-center'
  };
  return finalizeGeometryLowParts(source, options.fp64);
}

/**
 * Drops fp64 low streams when every source coordinate is already Float32-representable and
 * flags the source as fp64 otherwise.
 */
export function finalizeGeometryLowParts(
  source: PreparedKeplerGeometrySource,
  fp64: boolean | undefined
): PreparedKeplerGeometrySource {
  if (!fp64) return source;
  const needsLowParts = source.batches.some(batch =>
    [batch.paths.positionsLow, batch.polygonOutlines.positionsLow].some(low =>
      low?.some(value => value !== 0)
    )
  );
  if (!needsLowParts)
    for (const batch of source.batches) {
      delete batch.paths.positionsLow;
      delete batch.polygonOutlines.positionsLow;
    }
  return needsLowParts ? {...source, fp64: true} : source;
}

export function readNativeChunk(
  chunkData: Data,
  layout: EncodingLayout,
  dimension: 'xy' | 'xyz'
): NativeChunk | null {
  const offsets: Int32Array[] = [];
  let data = chunkData;
  for (let depth = 0; depth < layout.listDepth; depth++) {
    if (!DataType.isList(data.type) || (depth > 0 && data.nullCount > 0)) return null;
    const valueOffsets = data.valueOffsets;
    if (!(valueOffsets instanceof Int32Array) || valueOffsets.length < data.length + 1) return null;
    offsets.push(valueOffsets);
    data = data.children[0];
  }
  // Nested coordinate arrays cannot carry nulls; a top-level point array may.
  if (layout.listDepth > 0 && data.nullCount > 0) return null;
  const dimensionSize = dimension === 'xyz' ? 3 : 2;
  let coordinates: CoordinateReader;
  if (DataType.isFixedSizeList(data.type)) {
    const child = data.children[0];
    if (data.type.listSize !== dimensionSize || !DataType.isFloat(child.type)) return null;
    // Null point slots leave null child values behind; their validity bit already excludes them.
    if (layout.listDepth > 0 && child.nullCount > 0) return null;
    // Sliced nested coordinate children would need an element offset the reader does not apply.
    if (layout.listDepth > 0 && (data.offset !== 0 || child.offset !== 0)) return null;
    const values = child.values as NumericValues;
    coordinates = {
      xValues: values,
      yValues: values,
      zValues: dimensionSize === 3 ? values : null,
      stride: dimensionSize,
      xOffset: 0,
      yOffset: 1,
      zOffset: 2
    };
  } else if (DataType.isStruct(data.type)) {
    const children: Record<string, Data> = {};
    data.type.children.forEach((field, index) => {
      children[field.name] = data.children[index];
    });
    const x = children['x'];
    const y = children['y'];
    const z = children['z'];
    if (!x || !y || (dimensionSize === 3 && !z)) return null;
    for (const child of dimensionSize === 3 ? [x, y, z] : [x, y])
      if (!DataType.isFloat(child.type) || (layout.listDepth > 0 && child.nullCount > 0))
        return null;
    coordinates = {
      xValues: x.values as NumericValues,
      yValues: y.values as NumericValues,
      zValues: dimensionSize === 3 ? (z.values as NumericValues) : null,
      stride: 1,
      xOffset: 0,
      yOffset: 0,
      zOffset: 0
    };
  } else return null;
  const hasValidity = chunkData.nullCount > 0 && chunkData.nullBitmap?.length > 0;
  return {
    data: chunkData,
    offsets,
    coordinates,
    validity: hasValidity ? chunkData.nullBitmap : null,
    validityOffset: chunkData.offset
  };
}

type PreparedNativeChunk = {
  batch: PreparedGeometryBatch;
  invalidRows: {sourceRowIndex: number; reason: string}[];
  bounds: [number, number, number, number];
  hasZ: boolean;
  hasPointAltitude: boolean;
  hasValidRow: boolean;
  polygonCount: number;
};

const FINITE_COORDINATE_REASON = 'Geometry requires finite XYZ coordinates';
const RING_VERTEX_REASON = 'Polygon ring requires at least three vertices';

function prepareNativeChunk(
  chunk: NativeChunk,
  layout: EncodingLayout,
  rowCount: number,
  sourceRowOffset: number,
  fp64: boolean | undefined
): PreparedNativeChunk {
  const {coordinates, offsets, validity, validityOffset} = chunk;
  const {xValues, yValues, zValues, stride, xOffset, yOffset, zOffset} = coordinates;
  const partOffsets = layout.partLevel === null ? null : offsets[layout.partLevel];
  const ringOffsets = layout.ringLevel === null ? null : offsets[layout.ringLevel];
  const vertexOffsets = layout.vertexLevel === null ? null : offsets[layout.vertexLevel];
  const getRangeStart = (rangeOffsets: Int32Array | null, index: number): number =>
    rangeOffsets ? rangeOffsets[index] : index;
  const getRangeEnd = (rangeOffsets: Int32Array | null, index: number): number =>
    rangeOffsets ? rangeOffsets[index + 1] : index + 1;
  const isPolygon = layout.kind === 'polygon';

  const rowStatus = new Uint8Array(rowCount); // 0 null, 1 valid, 2 invalid
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const representativePositions = new Float32Array(rowCount * 2);
  const representativeValidity = new Uint32Array(rowCount);
  let hasZ = false;
  let hasPointAltitude = false;
  let hasValidRow = false;
  let pointCount = 0;
  let pathCount = 0;
  let pathVertexCount = 0;
  let ringCount = 0;
  let ringVertexCount = 0;
  let polygonCount = 0;

  // Pass 1: validate rows and gather sizes, bounds and representative means.
  for (let row = 0; row < rowCount; row++) {
    if (validity) {
      const bit = validityOffset + row;
      if (!(validity[bit >> 3] & (1 << (bit & 7)))) continue;
    }
    let reason: string | null = null;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let sumX = 0;
    let sumY = 0;
    let meanCount = 0;
    let rowAltitude = false;
    let rowVertexCount = 0;
    let rowPathCount = 0;
    let rowPathVertexCount = 0;
    let rowRingCount = 0;
    let rowRingVertexCount = 0;
    let rowPolygonCount = 0;
    const partStart = getRangeStart(partOffsets, row);
    const partEnd = getRangeEnd(partOffsets, row);
    scan: for (let part = partStart; part < partEnd; part++) {
      const ringStart = getRangeStart(ringOffsets, part);
      const ringEnd = getRangeEnd(ringOffsets, part);
      let partRingCount = 0;
      for (let ring = ringStart; ring < ringEnd; ring++) {
        const vertexStart = getRangeStart(vertexOffsets, ring);
        const vertexEnd = getRangeEnd(vertexOffsets, ring);
        const vertexLength = vertexEnd - vertexStart;
        for (let vertex = vertexStart; vertex < vertexEnd; vertex++) {
          const base = vertex * stride;
          const x = xValues[base + xOffset];
          const y = yValues[base + yOffset];
          // fround(NaN/Infinity/overflow) is not finite, which also rejects non-finite doubles.
          if (
            !Number.isFinite(Math.fround(x)) ||
            !Number.isFinite(Math.fround(y)) ||
            (zValues && !Number.isFinite(Math.fround(zValues[base + zOffset])))
          ) {
            reason = FINITE_COORDINATE_REASON;
            break scan;
          }
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          // Polygon means exclude the first vertex of every ring.
          if (!isPolygon || vertex > vertexStart) {
            sumX += x;
            sumY += y;
            meanCount++;
          }
          if (layout.kind === 'point' && zValues && zValues[base + zOffset] !== 0)
            rowAltitude = true;
        }
        if (isPolygon) {
          if (vertexLength > 0 && vertexLength < 3) {
            reason = RING_VERTEX_REASON;
            break scan;
          }
          if (vertexLength > 0) {
            partRingCount++;
            rowRingCount++;
            rowRingVertexCount +=
              vertexLength + (isRingClosed(coordinates, vertexStart, vertexEnd) ? 0 : 1);
          }
        } else if (layout.kind === 'path') {
          if (vertexLength > 0) {
            rowPathCount++;
            rowPathVertexCount += vertexLength;
          }
        } else rowVertexCount += vertexLength;
      }
      if (partRingCount > 0) rowPolygonCount++;
    }
    if (reason) {
      rowStatus[row] = 2;
      invalidRows.push({sourceRowIndex: sourceRowOffset + row, reason});
      continue;
    }
    rowStatus[row] = 1;
    hasValidRow = true;
    if (zValues && rowVertexCount + rowPathVertexCount + rowRingVertexCount > 0) hasZ = true;
    hasPointAltitude ||= rowAltitude;
    pointCount += rowVertexCount;
    pathCount += rowPathCount;
    pathVertexCount += rowPathVertexCount;
    ringCount += rowRingCount;
    ringVertexCount += rowRingVertexCount;
    polygonCount += rowPolygonCount;
    if (minX <= maxX) {
      bounds[0] = Math.min(bounds[0], minX);
      bounds[1] = Math.min(bounds[1], minY);
      bounds[2] = Math.max(bounds[2], maxX);
      bounds[3] = Math.max(bounds[3], maxY);
    }
    if (meanCount > 0) {
      const centerX = sumX / meanCount;
      const centerY = sumY / meanCount;
      if (Number.isFinite(Math.fround(centerX)) && Number.isFinite(Math.fround(centerY))) {
        representativePositions[row * 2] = centerX;
        representativePositions[row * 2 + 1] = centerY;
        representativeValidity[row] = 1;
      }
    }
  }

  // Pass 2: write primitive tables for valid rows.
  const pointPositions = new Float32Array(pointCount * 3);
  const pointRows = new Uint32Array(pointCount);
  const pathPositions = new Float32Array(pathVertexCount * 3);
  const pathOffsets = new Int32Array(pathCount + 1);
  const pathRows = new Uint32Array(pathCount);
  const pathLow = fp64 ? new Float32Array(pathVertexCount * 2) : undefined;
  const ringPositions = new Float32Array(ringVertexCount * 3);
  const ringDoubles = new Float64Array(ringVertexCount * 3);
  const ringOffsetsOut = new Int32Array(ringCount + 1);
  const ringRows = new Uint32Array(ringCount);
  const ringSideOrientations = new Float32Array(ringCount);
  const ringLow = fp64 ? new Float32Array(ringVertexCount * 2) : undefined;
  const polygonRingOffsets = new Int32Array(polygonCount + 1);
  const rowPolygonOffsets = new Int32Array(rowCount + 1);
  let pointIndex = 0;
  let pathIndex = 0;
  let pathVertexIndex = 0;
  let ringIndex = 0;
  let ringVertexIndex = 0;
  let polygonIndex = 0;

  for (let row = 0; row < rowCount; row++) {
    rowPolygonOffsets[row] = polygonIndex;
    if (rowStatus[row] !== 1) continue;
    const sourceRow = sourceRowOffset + row;
    const partStart = getRangeStart(partOffsets, row);
    const partEnd = getRangeEnd(partOffsets, row);
    for (let part = partStart; part < partEnd; part++) {
      const ringStart = getRangeStart(ringOffsets, part);
      const ringEnd = getRangeEnd(ringOffsets, part);
      let partRingCount = 0;
      for (let ring = ringStart; ring < ringEnd; ring++) {
        const vertexStart = getRangeStart(vertexOffsets, ring);
        const vertexEnd = getRangeEnd(vertexOffsets, ring);
        const vertexLength = vertexEnd - vertexStart;
        if (layout.kind === 'point') {
          for (let vertex = vertexStart; vertex < vertexEnd; vertex++) {
            writeVertex(coordinates, vertex, pointPositions, pointIndex * 3);
            pointRows[pointIndex++] = sourceRow;
          }
        } else if (layout.kind === 'path') {
          if (vertexLength === 0) continue;
          pathOffsets[pathIndex] = pathVertexIndex;
          pathRows[pathIndex++] = sourceRow;
          for (let vertex = vertexStart; vertex < vertexEnd; vertex++) {
            writeVertex(coordinates, vertex, pathPositions, pathVertexIndex * 3);
            if (pathLow) writeLowParts(coordinates, vertex, pathLow, pathVertexIndex * 2);
            pathVertexIndex++;
          }
        } else {
          if (vertexLength === 0) continue;
          ringOffsetsOut[ringIndex] = ringVertexIndex;
          ringRows[ringIndex] = sourceRow;
          // Shell is the first non-empty ring of the polygon; later rings are holes.
          ringSideOrientations[ringIndex] = getRingSideOrientation(
            coordinates,
            vertexStart,
            vertexEnd,
            partRingCount > 0
          );
          ringIndex++;
          partRingCount++;
          for (let vertex = vertexStart; vertex < vertexEnd; vertex++) {
            writeRingVertex(vertex);
          }
          if (!isRingClosed(coordinates, vertexStart, vertexEnd)) writeRingVertex(vertexStart);
        }
      }
      if (isPolygon && partRingCount > 0) polygonRingOffsets[++polygonIndex] = ringIndex;
    }
  }
  rowPolygonOffsets[rowCount] = polygonIndex;
  pathOffsets[pathCount] = pathVertexIndex;
  ringOffsetsOut[ringCount] = ringVertexIndex;

  function writeRingVertex(vertex: number): void {
    writeVertex(coordinates, vertex, ringPositions, ringVertexIndex * 3);
    writeVertex(coordinates, vertex, ringDoubles, ringVertexIndex * 3);
    if (ringLow) writeLowParts(coordinates, vertex, ringLow, ringVertexIndex * 2);
    ringVertexIndex++;
  }

  const batch: PreparedGeometryBatch = {
    id: '',
    rowCount,
    sourceRowOffset,
    points: {positions: pointPositions, sourceRows: pointRows},
    paths: {
      positions: pathPositions,
      ...(pathLow ? {positionsLow: pathLow} : {}),
      valueOffsets: pathOffsets,
      sourceRows: pathRows
    },
    polygonOutlines: {
      positions: ringPositions,
      ...(ringLow ? {positionsLow: ringLow} : {}),
      valueOffsets: ringOffsetsOut.slice(),
      sourceRows: ringRows,
      sideOrientations: ringSideOrientations
    },
    polygons: makeMultiPolygonVector(
      rowPolygonOffsets,
      polygonRingOffsets,
      ringOffsetsOut,
      ringDoubles
    ),
    representative: {positions: representativePositions, validity: representativeValidity},
    properties: undefined
  };
  return {batch, invalidRows, bounds, hasZ, hasPointAltitude, hasValidRow, polygonCount};
}

function writeVertex(
  coordinates: CoordinateReader,
  vertex: number,
  target: Float32Array | Float64Array,
  targetOffset: number
): void {
  const base = vertex * coordinates.stride;
  target[targetOffset] = coordinates.xValues[base + coordinates.xOffset];
  target[targetOffset + 1] = coordinates.yValues[base + coordinates.yOffset];
  target[targetOffset + 2] = coordinates.zValues
    ? coordinates.zValues[base + coordinates.zOffset]
    : 0;
}

function writeLowParts(
  coordinates: CoordinateReader,
  vertex: number,
  target: Float32Array,
  targetOffset: number
): void {
  const base = vertex * coordinates.stride;
  const x = coordinates.xValues[base + coordinates.xOffset];
  const y = coordinates.yValues[base + coordinates.yOffset];
  target[targetOffset] = x - Math.fround(x);
  target[targetOffset + 1] = y - Math.fround(y);
}

/** A ring is closed when its last vertex equals its first in every stored component. */
export function isRingClosed(coordinates: CoordinateReader, start: number, end: number): boolean {
  const {xValues, yValues, zValues, stride, xOffset, yOffset, zOffset} = coordinates;
  const first = start * stride;
  const last = (end - 1) * stride;
  return (
    xValues[first + xOffset] === xValues[last + xOffset] &&
    yValues[first + yOffset] === yValues[last + yOffset] &&
    (!zValues || zValues[first + zOffset] === zValues[last + zOffset])
  );
}

/** Signed-area side orientation over the closed ring in source precision (see generic route). */
function getRingSideOrientation(
  coordinates: CoordinateReader,
  start: number,
  end: number,
  isHole: boolean
): number {
  const {xValues, yValues, stride, xOffset, yOffset} = coordinates;
  const closed = isRingClosed(coordinates, start, end);
  const length = end - start + (closed ? 0 : 1);
  if (length < 3) return isHole ? -1 : 1;
  const getX = (index: number): number =>
    xValues[(index < end - start ? start + index : start) * stride + xOffset];
  const getY = (index: number): number =>
    yValues[(index < end - start ? start + index : start) * stride + yOffset];
  const originX = getX(0);
  const originY = getY(0);
  let signedArea = 0;
  for (let index = 1; index + 1 < length; index++) {
    signedArea +=
      (getX(index) - originX) * (getY(index + 1) - originY) -
      (getX(index + 1) - originX) * (getY(index) - originY);
  }
  return (signedArea < 0 ? -1 : 1) * (isHole ? -1 : 1);
}

/** Builds the same nested multipolygon vector shape as the generic GeoJSON route. */
function makeMultiPolygonVector(
  rowPolygonOffsets: Int32Array,
  polygonRingOffsets: Int32Array,
  ringVertexOffsets: Int32Array,
  coordinateValues: Float64Array
) {
  const coordinateType = new FixedSizeList(3, new Field('xyz', new Float64(), false));
  const ringType = new List(new Field('vertices', coordinateType, false));
  const polygonType = new List(new Field('rings', ringType, false));
  const multiPolygonType = new List(new Field('polygons', polygonType, false));
  const vertexCount = coordinateValues.length / 3;
  const valueData = makeData({
    type: new Float64(),
    length: coordinateValues.length,
    data: coordinateValues
  });
  const coordinateData = makeData({
    type: coordinateType,
    length: vertexCount,
    nullCount: 0,
    child: valueData
  });
  const ringData = makeData({
    type: ringType,
    length: ringVertexOffsets.length - 1,
    nullCount: 0,
    valueOffsets: ringVertexOffsets.slice(),
    child: coordinateData
  });
  const polygonData = makeData({
    type: polygonType,
    length: polygonRingOffsets.length - 1,
    nullCount: 0,
    valueOffsets: polygonRingOffsets,
    child: ringData
  });
  const multiPolygonData = makeData({
    type: multiPolygonType,
    length: rowPolygonOffsets.length - 1,
    nullCount: 0,
    valueOffsets: rowPolygonOffsets,
    child: polygonData
  });
  return new Vector([multiPolygonData]);
}
