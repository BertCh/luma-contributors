// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, assert, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {
  Field,
  FixedSizeList,
  Float64,
  List,
  type Vector,
  vectorFromArray,
  DataType
} from 'apache-arrow';
import {getGeoArrowRowView} from '@math.gl/arrow/geoarrow-utils';
import {decodeGeometrySourceWKB, normalizeEmptyWKBPoints} from './geometry-wkb';
import {prepareGlobeGeometryParts, type KeplerGlobeGeometryOptions} from './geometry-globe';
import type {WellKnownGeometry} from '@math.gl/arrow/wkb';
import {
  GEOMETRY_PROPERTY_NAMES,
  getGeometryFeatureProperties,
  ensureGeometryProperty,
  writeGeometryProperty,
  type GeometryPropertyName,
  type PreparedGeometryProperties,
  type GeometryStyleDiagnostic
} from './geometry-properties';
import {
  makeGeoArrowColumnFromArrowVector,
  makeArrowVectorFromGeoArrowColumn,
  type MakeGeoArrowColumnFromArrowVectorOptions
} from '@luma.gl/graph/arrow';
import {convertGeoArrowVectorToInterleaved} from '@luma.gl/graph/arrow';
import {makeKeplerPolygonPositionsLow} from './polygon-positions-low';
import {nameKeplerCoordinateDimension} from './coordinate-dimension-names';
import {isNativeGeoArrowEncoding, prepareNativeGeoArrowSource} from './geometry-native-geoarrow';
import {
  allocateStorageBuffer,
  makeBatchedGPUVector,
  makePropertyGPUVectors,
  makeSourceRowResolver,
  type StorageBufferValues
} from './gpu-upload-utils';
import {convertArrowPolygonToGPUVectors, type ArrowMultiPolygonType} from '@luma.gl/engine/arrow';

export type GeometrySourceBatch = {id: string; rows: readonly unknown[]};
export type GeometryMetadata = {
  rowCount: number;
  bounds: readonly [number, number, number, number] | null;
  crs?: unknown;
  spatialReferences?: readonly {sourceRowIndex: number; srids: readonly number[]}[];
  hasZ: boolean;
  hasPointAltitude: boolean;
  sourceStyleProperties: readonly string[];
  fixedRadius: boolean;
  invalidStyleProperties: readonly GeometryStyleDiagnostic[];
  geometryTypes: readonly string[];
  featureTypes: {point: boolean; line: boolean; polygon: boolean};
  invalidRows: readonly {sourceRowIndex: number; reason: string}[];
};
type Coordinates = [number, number, number];
type Path = Coordinates[];
type Polygon = Path[];
type PrimitiveParts = {
  timedLineStrings?: boolean;
  points: Coordinates[];
  paths: Path[];
  polygons: Polygon[];
  types: Set<string>;
  hasZ: boolean;
};
/** Options for one-time GeoJSON preparation. */
export type GeometryPreparationOptions = {
  /** Explicit trip source: static LineStrings read XYZ and leave the fourth ordinate to trips. */
  timedLineStrings?: boolean;
  crs?: unknown;
  globe?: KeplerGlobeGeometryOptions;
  /**
   * Also retain a Float32 low part (`value - fround(value)`) of every path and polygon XY so
   * renderers can subtract a split viewport origin in emulated double precision. Default false.
   */
  fp64?: boolean;
};
export type PreparedGeometryPathBatch = {
  /** Optional fp64 low part of XY, two values per vertex, when prepared with `fp64`. */
  positionsLow?: Float32Array;
  /** Outward side normal multiplier for polygon rings; absent for ordinary paths. */
  sideOrientations?: Float32Array;
  positions: Float32Array;
  valueOffsets: Int32Array;
  sourceRows: Uint32Array;
};
export type PreparedGeometryBatch = {
  id: string;
  rowCount: number;
  sourceRowOffset: number;
  points: {positions: Float32Array; sourceRows: Uint32Array};
  paths: PreparedGeometryPathBatch;
  polygonOutlines: PreparedGeometryPathBatch;
  polygons: Vector<ArrowMultiPolygonType>;
  representative?: {positions: Float32Array; validity: Uint32Array};
  properties?: PreparedGeometryProperties;
};
export type PreparedKeplerGeometrySource = {
  /** True when path and polygon batches carry fp64 low parts. */
  fp64?: boolean;
  representativeRecipe?: 'json-bounds-center' | 'geoarrow-mean-center' | 'wkb-vertex-mean';
  representativeUnsupportedReason?: string;
  metadata: GeometryMetadata;
  batches: readonly PreparedGeometryBatch[];
};
export type KeplerGeometryPrimitiveMapping = {
  /** Global source-row indices per primitive, preserving every physical source batch. */
  sourceRowIds: GPUVector<'uint32'>;
  sourceRows: readonly Uint32Array[];
  resolveSourceRow: (primitiveIndex: number) => number;
};
export type KeplerGeometryGPUPaths = KeplerGeometryPrimitiveMapping & {
  /** fp64 low part of XY aligned with `getPath`, present only for fp64 preparation. */
  getPathLow?: GPUVector<VertexList<'float32x2'>>;
  /** Ring-aligned multiplier for the right normal [dy, -dx], including hole orientation. */
  sideOrientations?: GPUVector<'float32'>;
  getPath: GPUVector<VertexList<'float32x3'>>;
};
export type GeometryGPUProperties = {
  fillColor?: {values: GPUVector<'unorm8x4'>; validity: GPUVector<'uint32'>};
  lineColor?: {values: GPUVector<'unorm8x4'>; validity: GPUVector<'uint32'>};
  lineWidth?: {values: GPUVector<'float32'>; validity: GPUVector<'uint32'>};
  radius?: {values: GPUVector<'float32'>; validity: GPUVector<'uint32'>};
  elevation?: {values: GPUVector<'float32'>; validity: GPUVector<'uint32'>};
  elevationOffset?: {values: GPUVector<'float32'>; validity: GPUVector<'uint32'>};
};
export type PreparedKeplerGeometryGPUVectors = {
  properties?: GeometryGPUProperties;
  metadata: GeometryMetadata;
  batches: readonly {id: string; rowCount: number; sourceRowOffset: number}[];
  points: KeplerGeometryPrimitiveMapping & {
    positions: GPUVector<'float32x2'>;
    positions3D: GPUVector<'float32x3'>;
  };
  paths: KeplerGeometryGPUPaths;
  polygonOutlines: KeplerGeometryGPUPaths;
  polygons: {
    positions: GPUVector<VertexList<'float32x4'>>;
    /** fp64 low part of XY aligned with `positions`, present only for fp64 preparation. */
    positionsLow?: GPUVector<VertexList<'float32x2'>>;
    rowIndices: GPUVector<VertexList<'uint32'>>;
    indices: GPUVector<VertexList<'uint32'>>;
  };
  representative?: {
    kind: 'representative-point';
    positions: GPUVector<'float32x2'>;
    validity: GPUVector<'uint32'>;
    recipe: 'json-bounds-center' | 'geoarrow-mean-center' | 'wkb-vertex-mean';
    nativeRectangleBounds: boolean;
  };
  representativeUnsupportedReason?: string;
  /** Unique owned GPU allocations; expose these to the dataset registry without counting aliases. */
  buffers: readonly Buffer[];
  gpuByteLength: number;
  /** Retained CPU primitive maps and unique topology-offset arrays. */
  mappingByteLength: number;
  destroy: () => void;
};

/**
 * One-time GeoJSON preparation. Every row remains a source feature, including null/invalid rows.
 * Multi geometries and collections expand only primitive tables; they never duplicate table rows.
 * Invalid geometry rejects its complete source row and publishes a diagnostic. CRS is retained,
 * not transformed. Float32 render buffers retain XYZ; source recovery belongs to the caller.
 */
export function prepareKeplerGeometrySource(
  batches: readonly GeometrySourceBatch[],
  options: GeometryPreparationOptions = {}
): PreparedKeplerGeometrySource {
  return prepareGeometryBatches(batches, options, true);
}

function prepareGeometryBatches(
  batches: readonly GeometrySourceBatch[],
  options: GeometryPreparationOptions,
  prepareRepresentatives: boolean
): PreparedKeplerGeometrySource {
  const preparation = prepareGeometryBatchesIncremental(batches, options, prepareRepresentatives);
  for (let step = preparation.next(); ; step = preparation.next()) if (step.done) return step.value;
}

/**
 * Time-sliceable GeoJSON preparation: yields after each source batch so a host can return to the
 * event loop between batches. Running it to completion equals `prepareKeplerGeometrySource`.
 */
export function prepareKeplerGeometrySourceIncremental(
  batches: Iterable<GeometrySourceBatch>,
  options: GeometryPreparationOptions = {}
): Generator<void, PreparedKeplerGeometrySource, void> {
  return prepareGeometryBatchesIncremental(batches, options, true);
}

function* prepareGeometryBatchesIncremental(
  batches: Iterable<GeometrySourceBatch>,
  options: GeometryPreparationOptions,
  prepareRepresentatives: boolean
): Generator<void, PreparedKeplerGeometrySource, void> {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const types = new Set<string>();
  let hasZ = false;
  let hasPointAltitude = false;
  const sourceStyleProperties = new Set<string>();
  const invalidStyleProperties: GeometryStyleDiagnostic[] = [];
  let fixedRadius = false;
  let sourceRowOffset = 0;
  const featureTypes = {point: false, line: false, polygon: false};
  const result: PreparedGeometryBatch[] = [];
  for (const batch of batches) {
    const points: Coordinates[] = [];
    const pointRows: number[] = [];
    const paths: Path[] = [];
    const pathRows: number[] = [];
    const outlines: Path[] = [];
    const outlineRows: number[] = [];
    const outlineSideOrientations: number[] = [];
    const polygonRows: Polygon[][] = [];
    const properties: PreparedGeometryProperties = {};
    const representative = prepareRepresentatives
      ? {
          positions: new Float32Array(batch.rows.length * 2),
          validity: new Uint32Array(batch.rows.length)
        }
      : undefined;
    for (let rowIndex = 0; rowIndex < batch.rows.length; rowIndex++) {
      const sourceRowIndex = sourceRowOffset + rowIndex;
      const parts: PrimitiveParts = {
        timedLineStrings: options.timedLineStrings,
        points: [],
        paths: [],
        polygons: [],
        types: new Set(),
        hasZ: false
      };
      let globe: ReturnType<typeof prepareGlobeGeometryParts> | undefined;
      // Parse a serialized row once and share the document with every consumer below.
      let row: unknown;
      try {
        row = parseGeometryRow(batch.rows[rowIndex]);
        appendGeometry(row, parts, 0);
        // GlobeView variant: unwrap, subdivide and cut long edges once so they follow the sphere.
        globe = options.globe ? prepareGlobeGeometryParts(parts, options.globe) : undefined;
      } catch (error) {
        invalidRows.push({
          sourceRowIndex,
          reason: error instanceof Error ? error.message : String(error)
        });
        polygonRows.push([]);
        continue;
      }
      if (representative) {
        const center = getJSONSelectionRepresentative(row, parts);
        if (center) {
          representative.positions.set(center, rowIndex * 2);
          representative.validity[rowIndex] = 1;
        }
      }
      hasZ ||= parts.hasZ;
      hasPointAltitude ||= parts.points.some(point => point[2] !== 0);
      if (prepareRepresentatives) {
        const featureProperties = getGeometryFeatureProperties(row);
        fixedRadius ||= Boolean(featureProperties?.['radius']);
        for (const property of GEOMETRY_PROPERTY_NAMES) {
          const value = featureProperties?.[property];
          if (value === undefined) continue;
          sourceStyleProperties.add(property);
          const reason = writeGeometryProperty(
            properties,
            property,
            value,
            rowIndex,
            batch.rows.length
          );
          if (reason) invalidStyleProperties.push({sourceRowIndex, property, reason});
        }
      }
      for (const type of parts.types) types.add(type);
      featureTypes.point ||= parts.points.length > 0;
      featureTypes.line ||= parts.paths.length > 0;
      featureTypes.polygon ||= parts.polygons.length > 0;
      for (const point of parts.points) {
        points.push(point);
        pointRows.push(sourceRowIndex);
      }
      for (const path of globe?.paths ?? parts.paths) {
        paths.push(path);
        pathRows.push(sourceRowIndex);
      }
      for (const polygon of globe?.outlines ?? parts.polygons)
        for (const [ringIndex, ring] of polygon.entries()) {
          outlines.push(ring);
          outlineRows.push(sourceRowIndex);
          outlineSideOrientations.push(getRingSideOrientation(ring, ringIndex > 0));
        }
      polygonRows.push(globe?.polygons ?? parts.polygons);
      expandBounds(bounds, parts);
    }
    const coordinateType = new FixedSizeList(3, new Field('xyz', new Float64(), false));
    const ringType = new List(new Field('vertices', coordinateType, false));
    const polygonType = new List(new Field('rings', ringType, false));
    const multiPolygonType = new List(new Field('polygons', polygonType, false));
    const prepared: PreparedGeometryBatch = {
      id: batch.id,
      rowCount: batch.rows.length,
      sourceRowOffset,
      points: {
        positions: flattenCoordinates(points),
        sourceRows: Uint32Array.from(pointRows)
      },
      paths: makePathBatch(paths, pathRows, options.fp64),
      polygonOutlines: {
        ...makePathBatch(outlines, outlineRows, options.fp64),
        sideOrientations: Float32Array.from(outlineSideOrientations)
      },
      polygons: vectorFromArray(polygonRows, multiPolygonType),
      representative,
      properties: prepareRepresentatives ? properties : undefined
    };
    sourceRowOffset += batch.rows.length;
    result.push(prepared);
    yield;
  }
  // Sources whose coordinates are already Float32-representable need no low stream.
  const needsLowParts =
    options.fp64 &&
    result.some(batch =>
      [batch.paths.positionsLow, batch.polygonOutlines.positionsLow].some(low =>
        low?.some(value => value !== 0)
      )
    );
  if (options.fp64 && !needsLowParts)
    for (const batch of result) {
      delete batch.paths.positionsLow;
      delete batch.polygonOutlines.positionsLow;
    }
  // A present property channel keeps every original batch, even batches with no overrides.
  for (const batch of result) {
    for (const property of GEOMETRY_PROPERTY_NAMES) {
      if (sourceStyleProperties.has(property))
        ensureGeometryProperty(batch.properties!, property, batch.rowCount);
    }
  }
  return {
    metadata: {
      rowCount: sourceRowOffset,
      bounds: Number.isFinite(bounds[0]) ? bounds : null,
      crs: options.crs,
      hasZ,
      hasPointAltitude,
      sourceStyleProperties: [...sourceStyleProperties],
      fixedRadius,
      invalidStyleProperties,
      geometryTypes: [...types],
      featureTypes,
      invalidRows
    },
    batches: result,
    ...(needsLowParts ? {fp64: true} : {}),
    representativeRecipe: prepareRepresentatives ? 'json-bounds-center' : undefined
  };
}

export type GeometrySourceFromArrowOptions = MakeGeoArrowColumnFromArrowVectorOptions & {
  crs?: unknown;
  /** Prepare the GlobeView variant: antimeridian unwrap, 5 degree edge subdivision and polygon cuts. */
  globe?: KeplerGlobeGeometryOptions;
  /** Explicit provenance of a decoded native vector; never infer WKB semantics from a union. */
  sourceEncoding?: 'geoarrow.wkb';
  /** Retain fp64 low parts of path and polygon XY (see GeometryPreparationOptions). */
  fp64?: boolean;
  /**
   * Read native point/line/polygon GeoArrow offset and coordinate buffers directly instead of
   * decoding every row through GeoJSON-style objects. Default true; falls back automatically for
   * unions, globe preparation, WKB provenance and buffers with nested nulls.
   */
  nativeFastPath?: boolean;
};

const GEOARROW_GEOMETRY_TYPES: Record<string, string> = {
  'geoarrow.point': 'Point',
  'geoarrow.multipoint': 'MultiPoint',
  'geoarrow.linestring': 'LineString',
  'geoarrow.multilinestring': 'MultiLineString',
  'geoarrow.polygon': 'Polygon',
  'geoarrow.multipolygon': 'MultiPolygon'
};

/** Native GeoArrow and serialized WKB preparation, preserving source-mode selection recipes. */
export function prepareKeplerGeometrySourceFromArrow(
  inputVector: Vector<DataType>,
  options: GeometrySourceFromArrowOptions = {}
): PreparedKeplerGeometrySource {
  const encoding = options.encoding;
  // Unnamed 3-wide coordinates stay XYZ as in the old fork; explicit options.dimension wins.
  const vector = DataType.isBinary(inputVector.type)
    ? inputVector
    : nameKeplerCoordinateDimension(inputVector, options.dimension);
  if (encoding === 'geoarrow.wkb' || (!encoding && DataType.isBinary(vector.type))) {
    const decoded = decodeGeometrySourceWKB(vector);
    const srids = [...new Set(decoded.spatialReferences.flatMap(reference => reference.srids))];
    const crs =
      options.crs ??
      (srids.length === 1 && srids[0] !== 0
        ? `EPSG:${srids[0]}`
        : srids.length > 1
          ? {kind: 'mixed-ewkb-srids', srids}
          : undefined);
    const source = prepareGeometryBatches(
      decoded.batches,
      {crs, globe: options.globe, fp64: options.fp64},
      false
    );
    source.metadata.spatialReferences = decoded.spatialReferences;
    source.metadata.invalidRows = [...source.metadata.invalidRows, ...decoded.invalidRows].sort(
      (left, right) => left.sourceRowIndex - right.sourceRowIndex
    );
    setGeometryRepresentatives(
      source,
      decoded.batches,
      'wkb-vertex-mean',
      getWKBSelectionRepresentative
    );
    return source;
  }
  if (
    encoding === 'geoarrow.wkt' ||
    encoding === 'geoarrow.geometrycollection' ||
    encoding === 'geoarrow.box'
  )
    throw new Error('Geometry source requires native point, path, polygon or union coordinates');
  if (
    options.nativeFastPath !== false &&
    !options.globe &&
    options.sourceEncoding !== 'geoarrow.wkb'
  ) {
    const native = prepareNativeGeoArrowVector(vector, options);
    if (native) return native;
  }
  const interleaved = convertGeoArrowVectorToInterleaved(vector, {encoding});
  const column = makeGeoArrowColumnFromArrowVector(interleaved, {
    ...options,
    coordinateLayout: 'interleaved'
  });
  if (
    column.encoding !== 'geoarrow.geometry' &&
    column.dimension !== 'xy' &&
    column.dimension !== 'xyz'
  )
    throw new Error(
      'Geometry source requires XY or XYZ coordinates; measure/time coordinates need a dedicated adapter'
    );
  let rowOffset = 0;
  const decodeErrors: {sourceRowIndex: number; reason: string}[] = [];
  const batches = column.chunks.map((chunk, chunkIndex) => {
    const rows: unknown[] = [];
    for (let index = 0; index < chunk.length; index++) {
      try {
        const row = getGeoArrowRowView(column, rowOffset + index);
        const value = makeArrowVectorFromGeoArrowColumn(row.column).get(0);
        // Only union columns carry a per-row dimension; other encodings were checked above.
        if (
          value !== null &&
          column.encoding === 'geoarrow.geometry' &&
          row.column.dimension !== 'xy' &&
          row.column.dimension !== 'xyz'
        )
          throw new Error(
            'Geometry source requires XY or XYZ coordinates; measure/time coordinates need a dedicated adapter'
          );
        if (value === null) rows.push(null);
        else if (GEOARROW_GEOMETRY_TYPES[row.column.encoding]) {
          const geometry = {
            type: GEOARROW_GEOMETRY_TYPES[row.column.encoding],
            coordinates: copyNestedValues(value)
          };
          rows.push(
            options.sourceEncoding === 'geoarrow.wkb'
              ? normalizeEmptyWKBPoints(geometry as WellKnownGeometry)
              : geometry
          );
        } else throw new Error(`Unsupported native geometry encoding: ${row.column.encoding}`);
      } catch (error) {
        rows.push(null);
        decodeErrors.push({
          sourceRowIndex: rowOffset + index,
          reason: error instanceof Error ? error.message : String(error)
        });
      }
    }
    rowOffset += chunk.length;
    return {id: `geometry-batch-${chunkIndex}`, rows};
  });
  const source = prepareGeometryBatches(
    batches,
    {crs: options.crs ?? column.spatialReference, globe: options.globe, fp64: options.fp64},
    false
  );
  source.metadata.invalidRows = [...source.metadata.invalidRows, ...decodeErrors].sort(
    (left, right) => left.sourceRowIndex - right.sourceRowIndex
  );
  if (column.encoding === 'geoarrow.geometry' && options.sourceEncoding !== 'geoarrow.wkb') {
    source.representativeUnsupportedReason =
      'Native Kepler Arrow mean centers do not support dense union geometry';
    return source;
  }
  setGeometryRepresentatives(
    source,
    batches,
    options.sourceEncoding === 'geoarrow.wkb' ? 'wkb-vertex-mean' : 'geoarrow-mean-center',
    options.sourceEncoding === 'geoarrow.wkb'
      ? getWKBSelectionRepresentative
      : getArrowSelectionRepresentative
  );
  return source;
}

/** Zero-object route for homogeneous native GeoArrow; null selects the generic route. */
function prepareNativeGeoArrowVector(
  vector: Vector<DataType>,
  options: GeometrySourceFromArrowOptions
): PreparedKeplerGeometrySource | null {
  try {
    const column = makeGeoArrowColumnFromArrowVector(vector, options);
    if (
      !isNativeGeoArrowEncoding(column.encoding) ||
      (column.dimension !== 'xy' && column.dimension !== 'xyz') ||
      (column.coordinateLayout !== 'interleaved' && column.coordinateLayout !== 'separated')
    )
      return null;
    return prepareNativeGeoArrowSource(vector, {
      encoding: column.encoding,
      dimension: column.dimension,
      crs: options.crs ?? column.spatialReference,
      fp64: options.fp64
    });
  } catch {
    return null;
  }
}

function setGeometryRepresentatives(
  source: PreparedKeplerGeometrySource,
  batches: readonly GeometrySourceBatch[],
  recipe: 'geoarrow-mean-center' | 'wkb-vertex-mean',
  getCenter: (value: unknown) => [number, number] | null
): void {
  source.representativeRecipe = recipe;
  const invalidRows = new Set(source.metadata.invalidRows.map(row => row.sourceRowIndex));
  for (const [batchIndex, batch] of source.batches.entries()) {
    const representative = {
      positions: new Float32Array(batch.rowCount * 2),
      validity: new Uint32Array(batch.rowCount)
    };
    for (let rowIndex = 0; rowIndex < batch.rowCount; rowIndex++) {
      if (invalidRows.has(batch.sourceRowOffset + rowIndex)) continue;
      const center = getCenter(batches[batchIndex].rows[rowIndex]);
      if (center) {
        representative.positions.set(center, rowIndex * 2);
        representative.validity[rowIndex] = 1;
      }
    }
    batch.representative = representative;
  }
}

/** Uploads independently prepared batches and reuses the existing Arrow/math.gl tessellator. */
export function makeKeplerGeometryGPUVectors(
  device: Device,
  source: PreparedKeplerGeometrySource
): PreparedKeplerGeometryGPUVectors {
  const buffers: Buffer[] = [];
  const disposers: (() => void)[] = [];
  const allocateBuffer = (name: string, values: StorageBufferValues) => {
    const buffer = allocateStorageBuffer(device, name, values);
    buffers.push(buffer);
    return buffer;
  };
  const makePrimitiveMapping = (
    name: string,
    sourceRows: Uint32Array[]
  ): KeplerGeometryPrimitiveMapping => ({
    sourceRows,
    sourceRowIds: makeBatchedGPUVector(
      name,
      'uint32',
      sourceRows.map((rows, index) => ({
        buffer: allocateBuffer(`${name}-${index}`, rows),
        length: rows.length
      }))
    ),
    resolveSourceRow: makeSourceRowResolver(sourceRows)
  });
  const makePathVectors = (
    name: string,
    parts: PreparedGeometryPathBatch[]
  ): KeplerGeometryGPUPaths => ({
    ...(parts.some(part => part.sideOrientations)
      ? {
          sideOrientations: makeBatchedGPUVector(
            `${name}-side-orientations`,
            'float32',
            parts.map((part, index) => {
              const values =
                part.sideOrientations ?? new Float32Array(part.sourceRows.length).fill(1);
              return {
                buffer: allocateBuffer(`${name}-side-orientations-${index}`, values),
                length: values.length
              };
            })
          )
        }
      : {}),
    ...makePrimitiveMapping(
      `${name}-source-rows`,
      parts.map(part => part.sourceRows)
    ),
    ...(source.fp64 && parts.every(part => part.positionsLow)
      ? {
          getPathLow: makeBatchedGPUVector(
            `${name}-low`,
            'vertex-list<float32x2>',
            parts.map((part, index) => ({
              buffer: allocateBuffer(`${name}-low-${index}`, part.positionsLow!),
              length: part.sourceRows.length,
              valueLength: part.positionsLow!.length / 2,
              valueOffsets: part.valueOffsets
            }))
          )
        }
      : {}),
    getPath: makeBatchedGPUVector(
      name,
      'vertex-list<float32x3>',
      parts.map((part, index) => ({
        buffer: allocateBuffer(`${name}-${index}`, part.positions),
        length: part.sourceRows.length,
        valueLength: part.positions.length / 3,
        valueOffsets: part.valueOffsets
      }))
    )
  });
  try {
    const points = {
      ...makePrimitiveMapping(
        'geometry-points',
        source.batches.map(batch => batch.points.sourceRows)
      ),
      positions: makeBatchedGPUVector(
        'geometry-point-xy',
        'float32x2',
        source.batches.map((batch, index) => {
          const values = new Float32Array(batch.points.sourceRows.length * 2);
          for (let row = 0; row < batch.points.sourceRows.length; row++)
            values.set(batch.points.positions.subarray(row * 3, row * 3 + 2), row * 2);
          return {
            buffer: allocateBuffer(`geometry-point-xy-${index}`, values),
            length: batch.points.sourceRows.length
          };
        })
      ),
      positions3D: makeBatchedGPUVector(
        'geometry-point-xyz',
        'float32x3',
        source.batches.map((batch, index) => ({
          buffer: allocateBuffer(`geometry-point-xyz-${index}`, batch.points.positions),
          length: batch.points.sourceRows.length
        }))
      )
    };
    const linePaths = makePathVectors(
      'geometry-paths',
      source.batches.map(batch => batch.paths)
    );
    const polygonOutlines = makePathVectors(
      'geometry-outlines',
      source.batches.map(batch => batch.polygonOutlines)
    );
    const polygonBatches = source.batches.map((batch, index) => {
      const prepared = convertArrowPolygonToGPUVectors(
        device,
        {polygons: batch.polygons},
        {
          id: `geometry-polygons-${index}`,
          rowIndexOffset: batch.sourceRowOffset,
          sourceBatchIndex: index
        }
      );
      disposers.push(prepared.destroy);
      // Published polygon conversion has no fp64 residual output; derive it locally.
      let positionsLow: GPUData<VertexList<'float32x2'>> | undefined;
      if (source.fp64) {
        const lowValues = makeKeplerPolygonPositionsLow(
          batch.polygons,
          prepared.tessellation.vertexCount
        );
        positionsLow = new GPUData({
          format: 'vertex-list<float32x2>',
          buffer: allocateBuffer(`geometry-polygons-low-${index}`, lowValues),
          length: prepared.positions.data[0].length,
          valueLength: prepared.tessellation.vertexCount,
          valueOffsets: prepared.vertexValueOffsets,
          ownsBuffer: false
        });
      }
      for (const vector of [prepared.positions, prepared.rowIndices, prepared.indices]) {
        for (const chunk of vector.data) {
          if (!(chunk.buffer instanceof Buffer))
            throw new Error('Geometry upload requires fixed buffers');
          buffers.push(chunk.buffer);
        }
      }
      return {
        positions: makePolygonChunk(prepared.positions, prepared.vertexValueOffsets),
        positionsLow,
        rowIndices: makePolygonChunk(prepared.rowIndices, prepared.vertexValueOffsets),
        indices: makePolygonChunk(prepared.indices, prepared.indexValueOffsets)
      };
    });
    const representative = source.representativeRecipe
      ? {
          kind: 'representative-point' as const,
          recipe: source.representativeRecipe,
          // These descriptors feed native GeoJsonLayer for both JSON and Arrow source modes.
          nativeRectangleBounds: true,
          positions: makeBatchedGPUVector(
            'geometry-selection-representatives',
            'float32x2',
            source.batches.map((batch, index) => ({
              buffer: allocateBuffer(
                `geometry-representative-${index}`,
                batch.representative!.positions
              ),
              length: batch.rowCount
            }))
          ),
          validity: makeBatchedGPUVector(
            'geometry-selection-validity',
            'uint32',
            source.batches.map((batch, index) => ({
              buffer: allocateBuffer(
                `geometry-representative-validity-${index}`,
                batch.representative!.validity
              ),
              length: batch.rowCount
            }))
          )
        }
      : undefined;
    const uploadProperty = <Format extends 'float32' | 'unorm8x4'>(
      name: GeometryPropertyName,
      format: Format
    ) =>
      source.metadata.sourceStyleProperties.includes(name)
        ? makePropertyGPUVectors({
            namePrefix: 'geometry',
            name,
            format,
            batches: source.batches.map(batch => ({
              rowCount: batch.rowCount,
              channel: batch.properties![name]!
            })),
            allocateBuffer
          })
        : undefined;
    const properties: GeometryGPUProperties | undefined = source.metadata.sourceStyleProperties
      .length
      ? {
          fillColor: uploadProperty('fillColor', 'unorm8x4'),
          lineColor: uploadProperty('lineColor', 'unorm8x4'),
          lineWidth: uploadProperty('lineWidth', 'float32'),
          radius: uploadProperty('radius', 'float32'),
          elevation: uploadProperty('elevation', 'float32'),
          elevationOffset: uploadProperty('elevationOffset', 'float32')
        }
      : undefined;
    let destroyed = false;
    return {
      properties,
      representative,
      representativeUnsupportedReason: source.representativeUnsupportedReason,
      metadata: source.metadata,
      batches: source.batches.map(({id, rowCount, sourceRowOffset}) => ({
        id,
        rowCount,
        sourceRowOffset
      })),
      points,
      paths: linePaths,
      polygonOutlines,
      polygons: {
        positions: new GPUVector({
          type: 'data',
          name: 'geometry-polygons',
          format: 'vertex-list<float32x4>',
          ownsData: false,
          data: polygonBatches.map(batch => batch.positions)
        }),
        ...(source.fp64 && polygonBatches.every(batch => batch.positionsLow)
          ? {
              positionsLow: new GPUVector({
                type: 'data',
                name: 'geometry-polygons-low',
                format: 'vertex-list<float32x2>',
                ownsData: false,
                data: polygonBatches.map(batch => batch.positionsLow!)
              })
            }
          : {}),
        rowIndices: new GPUVector({
          type: 'data',
          name: 'geometry-polygon-rows',
          format: 'vertex-list<uint32>',
          ownsData: false,
          data: polygonBatches.map(batch => batch.rowIndices)
        }),
        indices: new GPUVector({
          type: 'data',
          name: 'geometry-polygon-indices',
          format: 'vertex-list<uint32>',
          ownsData: false,
          data: polygonBatches.map(batch => batch.indices)
        })
      },
      buffers,
      gpuByteLength: buffers.reduce((total, buffer) => total + buffer.byteLength, 0),
      mappingByteLength:
        [points, linePaths, polygonOutlines].reduce(
          (total, part) => total + part.sourceRows.reduce((sum, rows) => sum + rows.byteLength, 0),
          0
        ) +
        source.batches.reduce(
          (total, batch) =>
            total +
            batch.paths.valueOffsets.byteLength +
            batch.polygonOutlines.valueOffsets.byteLength,
          0
        ) +
        polygonBatches.reduce(
          (total, batch) =>
            total +
            (batch.positions.valueOffsets?.byteLength ?? 0) +
            (batch.indices.valueOffsets?.byteLength ?? 0),
          0
        ),
      destroy: () => {
        if (destroyed) return;
        destroyed = true;
        for (const dispose of disposers) dispose();
        for (const buffer of buffers) if (!buffer.destroyed) buffer.destroy();
      }
    };
  } catch (error) {
    for (const dispose of disposers) dispose();
    for (const buffer of buffers) if (!buffer.destroyed) buffer.destroy();
    throw error;
  }
}

/** Keeps variable-list metadata at the GPUData boundary, including empty source rows. */
function makePolygonChunk<Format extends VertexList<'float32x4' | 'float32x2' | 'uint32'>>(
  vector: GPUVector<Format>,
  offsets: Int32Array
): GPUData<Format> {
  // The tessellator emits one chunk per prepared batch; a second chunk would be silently dropped.
  assert(vector.data.length === 1);
  const chunk = vector.data[0];
  return new GPUData({
    format: vector.format,
    buffer: chunk.buffer,
    length: chunk.length,
    valueLength: chunk.valueLength,
    valueOffsets: offsets,
    ownsBuffer: false
  });
}

/** Use original XY topology, including an implicit closing edge. Translation reduces cancellation. */
function getRingSideOrientation(ring: Path, isHole: boolean): number {
  if (ring.length < 3) return isHole ? -1 : 1;
  const origin = ring[0];
  let signedArea = 0;
  for (let index = 1; index + 1 < ring.length; index++) {
    const current = ring[index];
    const next = ring[index + 1];
    signedArea +=
      (current[0] - origin[0]) * (next[1] - origin[1]) -
      (next[0] - origin[0]) * (current[1] - origin[1]);
  }
  return (signedArea < 0 ? -1 : 1) * (isHole ? -1 : 1);
}

function makePathBatch(
  paths: Path[],
  sourceRows: number[],
  fp64 = false
): PreparedGeometryPathBatch {
  const valueOffsets = new Int32Array(paths.length + 1);
  let count = 0;
  for (let index = 0; index < paths.length; index++) {
    valueOffsets[index] = count;
    count += paths[index].length;
  }
  valueOffsets[paths.length] = count;
  const positions = new Float32Array(count * 3);
  let vertexIndex = 0;
  for (const path of paths)
    for (const vertex of path) {
      positions.set(vertex, vertexIndex * 3);
      vertexIndex++;
    }
  return {
    positions,
    ...(fp64 ? {positionsLow: makePathPositionsLow(paths)} : {}),
    valueOffsets,
    sourceRows: Uint32Array.from(sourceRows)
  };
}
/** Flattens XYZ coordinates into one Float32Array without building an intermediate array. */
function flattenCoordinates(coordinates: Coordinates[]): Float32Array {
  const positions = new Float32Array(coordinates.length * 3);
  for (let index = 0; index < coordinates.length; index++)
    positions.set(coordinates[index], index * 3);
  return positions;
}
/** Grows XY bounds over every vertex of the primitives, without spreading large arrays. */
function expandBounds(bounds: number[], parts: PrimitiveParts): void {
  const expand = (vertex: Coordinates): void => {
    bounds[0] = Math.min(bounds[0], vertex[0]);
    bounds[1] = Math.min(bounds[1], vertex[1]);
    bounds[2] = Math.max(bounds[2], vertex[0]);
    bounds[3] = Math.max(bounds[3], vertex[1]);
  };
  for (const point of parts.points) expand(point);
  for (const path of parts.paths) for (const vertex of path) expand(vertex);
  for (const polygon of parts.polygons)
    for (const ring of polygon) for (const vertex of ring) expand(vertex);
}
function makePathPositionsLow(paths: Path[]): Float32Array {
  const vertexCount = paths.reduce((total, path) => total + path.length, 0);
  const positionsLow = new Float32Array(vertexCount * 2);
  let vertexIndex = 0;
  for (const path of paths)
    for (const vertex of path) {
      positionsLow[vertexIndex * 2] = vertex[0] - Math.fround(vertex[0]);
      positionsLow[vertexIndex * 2 + 1] = vertex[1] - Math.fround(vertex[1]);
      vertexIndex++;
    }
  return positionsLow;
}
function copyNestedValues(value: unknown): unknown {
  if (value && typeof value === 'object' && Symbol.iterator in value)
    return Array.from(value as Iterable<unknown>, copyNestedValues);
  return value;
}
/** Parses a serialized top level row once; objects pass through unchanged. */
function parseGeometryRow(value: unknown): unknown {
  return typeof value === 'string' ? JSON.parse(value) : value;
}
function readCoordinate(raw: unknown, parts: PrimitiveParts, allowMeasure = false): Coordinates {
  if (
    !Array.isArray(raw) ||
    raw.length < 2 ||
    raw.length > (allowMeasure ? 4 : 3) ||
    !raw
      .slice(0, Math.min(3, raw.length))
      .every(
        value =>
          typeof value === 'number' && Number.isFinite(value) && Number.isFinite(Math.fround(value))
      )
  )
    throw new Error('Geometry requires finite XYZ coordinates');
  parts.hasZ ||= raw.length > 2;
  return [raw[0], raw[1], raw[2] ?? 0];
}
function readCoordinateList(raw: unknown): unknown[] {
  if (!Array.isArray(raw)) throw new Error('Geometry coordinates require an array');
  return raw;
}
function readPath(raw: unknown, parts: PrimitiveParts, allowMeasure = false): Path {
  return readCoordinateList(raw).map(coordinate => readCoordinate(coordinate, parts, allowMeasure));
}
function readPolygon(raw: unknown, parts: PrimitiveParts): Polygon {
  return readCoordinateList(raw)
    .map(ring => {
      const points = readPath(ring, parts);
      if (points.length && points.length < 3)
        throw new Error('Polygon ring requires at least three vertices');
      if (
        points.length &&
        !points[0].every((value, index) => value === points[points.length - 1][index])
      )
        points.push([...points[0]]);
      return points;
    })
    .filter(ring => ring.length > 0);
}
function appendGeometry(value: unknown, parts: PrimitiveParts, depth: number): void {
  if (value === null || value === undefined) return;
  if (depth > 32) throw new Error('Geometry nesting exceeds supported depth');
  if (typeof value === 'string') {
    appendGeometry(JSON.parse(value), parts, depth + 1);
    return;
  }
  if (typeof value !== 'object' || !('type' in value))
    throw new Error('Expected GeoJSON geometry or feature');
  const geometry = value as {
    type: string;
    geometry?: unknown;
    features?: unknown[];
    geometries?: unknown[];
    coordinates?: unknown;
    properties?: Record<string, unknown>;
  };
  if (geometry.type === 'Feature') {
    appendGeometry(geometry.geometry, parts, depth + 1);
    return;
  }
  if (geometry.type === 'FeatureCollection' || geometry.type === 'GeometryCollection') {
    const children =
      geometry.type === 'FeatureCollection' ? geometry.features : geometry.geometries;
    if (!Array.isArray(children)) throw new Error('Geometry collection requires an array');
    for (const child of children) appendGeometry(child, parts, depth + 1);
    return;
  }
  parts.types.add(geometry.type);
  // Loops, not `push(...spread)`: spreading large multi geometries overflows the argument limit.
  switch (geometry.type) {
    case 'Point':
      if (readCoordinateList(geometry.coordinates).length)
        parts.points.push(readCoordinate(geometry.coordinates, parts));
      break;
    case 'MultiPoint':
      for (const coordinate of readCoordinateList(geometry.coordinates))
        parts.points.push(readCoordinate(coordinate, parts));
      break;
    case 'LineString': {
      // Kepler's timed LineStrings use XYZ + time. The static GeoJSON layer reads only XYZ;
      // the trip adapter separately validates and retains the fourth ordinate for animation.
      const points = readPath(geometry.coordinates, parts, parts.timedLineStrings);
      if (points.length) parts.paths.push(points);
      break;
    }
    case 'MultiLineString':
      for (const rawPath of readCoordinateList(geometry.coordinates)) {
        const points = readPath(rawPath, parts, parts.timedLineStrings);
        if (points.length) parts.paths.push(points);
      }
      break;
    case 'Polygon': {
      const rings = readPolygon(geometry.coordinates, parts);
      if (rings.length) parts.polygons.push(rings);
      break;
    }
    case 'MultiPolygon':
      for (const rawPolygon of readCoordinateList(geometry.coordinates)) {
        const rings = readPolygon(rawPolygon, parts);
        if (rings.length) parts.polygons.push(rings);
      }
      break;
    default:
      throw new Error(`Unsupported GeoJSON geometry type: ${geometry.type}`);
  }
}

/** Mirrors the pinned Kepler JSON normalization plus Turf center contract, not area centroids.
 * Only the normalized root Feature bbox overrides coordinate bounds; child geometry bboxes do not.
 * Kepler normalizes a FeatureCollection to its first feature before calculating its representative.
 */
function getJSONSelectionRepresentative(
  raw: unknown,
  rowParts?: PrimitiveParts
): [number, number] | null {
  let value = raw;
  try {
    if (typeof value === 'string') value = JSON.parse(value);
    if (!value || typeof value !== 'object' || !('type' in value)) return null;
    let feature = value as {type: string; bbox?: unknown; features?: unknown[]; geometry?: unknown};
    if (feature.type === 'FeatureCollection') {
      const first = feature.features?.[0];
      if (!first || typeof first !== 'object' || !('type' in first)) return null;
      feature = first as typeof feature;
    }
    const isFeature = feature.type === 'Feature';
    const geometry = isFeature ? feature.geometry : feature;
    if (!geometry || typeof geometry !== 'object' || !('type' in geometry)) return null;
    if (
      ![
        'Point',
        'MultiPoint',
        'LineString',
        'MultiLineString',
        'Polygon',
        'MultiPolygon',
        'GeometryCollection'
      ].includes(String(geometry.type))
    )
      return null;
    let bounds: number[];
    if (isFeature && feature.bbox !== undefined && feature.bbox !== null) {
      if (
        !Array.isArray(feature.bbox) ||
        feature.bbox.length < 4 ||
        !feature.bbox
          .slice(0, 4)
          .every(value => typeof value === 'number' && Number.isFinite(value))
      )
        return null;
      // Turf center uses the first four entries even when the provided bbox contains Z bounds.
      bounds = feature.bbox;
    } else {
      // The row primitives already cover the whole root, except a FeatureCollection, which
      // Kepler normalizes to its first feature.
      let parts = rowParts;
      if (!parts || value.type === 'FeatureCollection') {
        parts = {
          points: [],
          paths: [],
          polygons: [],
          types: new Set(),
          hasZ: false,
          timedLineStrings: rowParts?.timedLineStrings
        };
        appendGeometry(geometry, parts, 0);
      }
      bounds = [Infinity, Infinity, -Infinity, -Infinity];
      expandBounds(bounds, parts);
    }
    const center: [number, number] = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2];
    return center.every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))
      ? center
      : null;
  } catch {
    return null;
  }
}

/** Pinned loaders.gl 4.5.4 Arrow recipe: vertex-weighted XY means, with the first coordinate
 * of every polygon ring excluded unconditionally. Compute before render preparation closes rings.
 */
function getArrowSelectionRepresentative(value: unknown): [number, number] | null {
  if (!value || typeof value !== 'object' || !('type' in value) || !('coordinates' in value))
    return null;
  const geometry = value as {type: string; coordinates: unknown[]};
  let vertices: unknown[];
  switch (geometry.type) {
    case 'Point':
      vertices = [geometry.coordinates];
      break;
    case 'MultiPoint':
    case 'LineString':
      vertices = geometry.coordinates;
      break;
    case 'MultiLineString':
      vertices = geometry.coordinates.flat();
      break;
    case 'Polygon':
      vertices = geometry.coordinates.flatMap(ring => (Array.isArray(ring) ? ring.slice(1) : []));
      break;
    case 'MultiPolygon':
      vertices = geometry.coordinates.flatMap(polygon =>
        Array.isArray(polygon)
          ? polygon.flatMap(ring => (Array.isArray(ring) ? ring.slice(1) : []))
          : []
      );
      break;
    default:
      return null;
  }
  if (!vertices.length) return null;
  let longitudeSum = 0;
  let latitudeSum = 0;
  for (const vertex of vertices) {
    if (!Array.isArray(vertex) || !Number.isFinite(vertex[0]) || !Number.isFinite(vertex[1]))
      return null;
    longitudeSum += vertex[0];
    latitudeSum += vertex[1];
  }
  const center: [number, number] = [longitudeSum / vertices.length, latitudeSum / vertices.length];
  return center.every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))
    ? center
    : null;
}

/** Pinned WKB GeoJsonLayer recipe averages all finite XY vertices, including ring closures. */
function getWKBSelectionRepresentative(value: unknown): [number, number] | null {
  let longitudeSum = 0;
  let latitudeSum = 0;
  let count = 0;
  const visitCoordinates = (coordinates: unknown): void => {
    if (!Array.isArray(coordinates)) return;
    if (
      coordinates.length >= 2 &&
      typeof coordinates[0] === 'number' &&
      typeof coordinates[1] === 'number' &&
      Number.isFinite(coordinates[0]) &&
      Number.isFinite(coordinates[1])
    ) {
      longitudeSum += coordinates[0];
      latitudeSum += coordinates[1];
      count++;
    } else {
      for (const child of coordinates) visitCoordinates(child);
    }
  };
  const visitGeometry = (geometry: unknown): void => {
    if (!geometry || typeof geometry !== 'object') return;
    if (
      'type' in geometry &&
      geometry.type === 'GeometryCollection' &&
      'geometries' in geometry &&
      Array.isArray(geometry.geometries)
    ) {
      for (const child of geometry.geometries) visitGeometry(child);
    } else if ('coordinates' in geometry) visitCoordinates(geometry.coordinates);
  };
  visitGeometry(value);
  if (!count) return null;
  const center: [number, number] = [longitudeSum / count, latitudeSum / count];
  return center.every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))
    ? center
    : null;
}
