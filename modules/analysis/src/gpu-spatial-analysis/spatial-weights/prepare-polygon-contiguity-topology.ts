// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  computeContiguityWeightsOnCPU,
  type CPUContiguityWeightsResult
} from './cpu-contiguity-weights-operation';
import type {GPUContiguityCriterion} from './gpu-contiguity-weights';
import {prepareGeometryInput} from '../spatial-join/prepare-geometry-input';
import {DataType, Vector} from 'apache-arrow';

/** Bounded CPU ingestion and optional all-pairs CPU adjacency route. */
export const POLYGON_CONTIGUITY_MAXIMUM_ROWS = 4096;
/** Bounds temporary decoding and keeps integer equality tokens exact in Float32. */
export const POLYGON_CONTIGUITY_MAXIMUM_VERTICES = 65536;
/** Conservative allowance for temporary row arrays, exact-coordinate maps and Arrow row views. */
export const POLYGON_CONTIGUITY_PREPARATION_BYTES = 24 * 1024 * 1024;
const MAXIMUM_MANIFEST_BYTES = 1024 * 1024;

export type PreparedPolygonTopologySource = {
  metadata: {rowCount: number; invalidRows: readonly {sourceRowIndex: number; reason: string}[]};
  /** Native Polygon or MultiPolygon Arrow lists; XY[Z…] originals define equality. */
  batches: readonly {id: string; rowCount: number; sourceRowOffset: number; polygons: Vector}[];
};

export type PolygonContiguityDiagnostic = {
  sourceRowIndex: number;
  reason: 'source-invalid' | 'invalid-topology';
};

export type PreparedPolygonContiguityTopology = {
  /** Private equality tokens [vertexId, 0]; these are NOT renderable or metric coordinates. */
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  geometryMask: Uint32Array;
  batches: readonly {id: string; rowCount: number; sourceRowOffset: number}[];
  invalidRows: readonly PolygonContiguityDiagnostic[];
  rowCount: number;
  vertexCount: number;
  uniqueVertexCount: number;
  /** Exact typed-array bytes plus 64 bytes/batch, UTF-16 batch IDs and 64 bytes/diagnostic. */
  cpuByteLength: number;
  topologyEncoding: 'source-xy-key-ids';
  equality: 'exact-source-xy';
  ignoresZ: true;
};

type Coordinate = readonly [number, number];
type PolygonRow = Coordinate[][][];

/**
 * Explicitly pack the polygon parts of each original source row for GPU equality-based adjacency.
 *
 * Original Float64 XY values are interned before Float32 upload. Integer equality tokens therefore
 * preserve source-coordinate equality, including across source batches, without Float32 coordinate
 * collapse. Queen compares tokens; rook compares complete endpoint pairs. No snapping, noding,
 * antimeridian normalization or geometric repair is performed. Z is ignored. Holes and polygon
 * parts from GeometryCollections participate. The source is borrowed and is not retained.
 *
 * Each row is validated before any output is committed. Invalid rows retain empty feature ranges;
 * empty/nonpolygon rows have mask 0 without being classified as malformed. Rings close implicitly.
 */
export function preparePolygonContiguityTopology(
  source: PreparedPolygonTopologySource
): PreparedPolygonContiguityTopology {
  const rowCount = source.metadata.rowCount;
  const id = 'preparePolygonContiguityTopology';
  validateCapacity(id, 'source row capacity', rowCount, 0, POLYGON_CONTIGUITY_MAXIMUM_ROWS);
  validateMaximumCount(
    id,
    'source batch capacity',
    source.batches.length,
    POLYGON_CONTIGUITY_MAXIMUM_ROWS
  );
  const batches: {id: string; rowCount: number; sourceRowOffset: number}[] = [];
  let expectedOffset = 0;
  let manifestBytes = 0;
  for (const batch of source.batches) {
    if (
      !Number.isSafeInteger(batch.rowCount) ||
      batch.rowCount < 0 ||
      batch.sourceRowOffset !== expectedOffset ||
      batch.polygons.length !== batch.rowCount
    )
      throw new Error('preparePolygonContiguityTopology requires contiguous source batch rows');
    expectedOffset += batch.rowCount;
    if (expectedOffset > rowCount)
      throw new Error('preparePolygonContiguityTopology source row count mismatch');
    manifestBytes += 64 + batch.id.length * 2;
    if (manifestBytes > MAXIMUM_MANIFEST_BYTES)
      throw new Error('preparePolygonContiguityTopology source batch metadata exceeds capacity');
    batches.push({id: batch.id, rowCount: batch.rowCount, sourceRowOffset: batch.sourceRowOffset});
  }
  if (expectedOffset !== rowCount)
    throw new Error('preparePolygonContiguityTopology source row count mismatch');
  if (source.metadata.invalidRows.length > rowCount)
    throw new Error('preparePolygonContiguityTopology source diagnostics exceed row capacity');
  const sourceInvalidRows = new Set<number>();
  for (const diagnostic of source.metadata.invalidRows) {
    if (
      !Number.isSafeInteger(diagnostic.sourceRowIndex) ||
      diagnostic.sourceRowIndex < 0 ||
      diagnostic.sourceRowIndex >= rowCount
    )
      throw new Error('preparePolygonContiguityTopology source diagnostic row is invalid');
    sourceInvalidRows.add(diagnostic.sourceRowIndex);
  }

  // Vertex identifiers are stored as Float32 tokens, exact only up to 2^24; raising the vertex
  // cap past that needs Uint32 tokens.
  validateMaximumCount(
    id,
    'Float32 equality-token vertex capacity',
    POLYGON_CONTIGUITY_MAXIMUM_VERTICES,
    2 ** 24
  );
  const positionValues: number[] = [];
  const ringOffsetValues = [0];
  const polygonOffsetValues = [0];
  const featureOffsets = new Uint32Array(rowCount + 1);
  const geometryMask = new Uint32Array(rowCount);
  const invalidRows: PolygonContiguityDiagnostic[] = [];
  const coordinateIds = new Map<number, Map<number, number>>();
  let uniqueVertexCount = 0;
  // Charge every decoded occurrence, including closing vertices and rejected rows, so malformed
  // geometry cannot bypass the preparation budget by being discarded after a large temporary read.
  let inspectedVertices = 0;
  const accountVertices = (count: number) => {
    if (
      !Number.isSafeInteger(count) ||
      count < 0 ||
      count > POLYGON_CONTIGUITY_MAXIMUM_VERTICES - inspectedVertices
    )
      throw new PolygonContiguityCapacityError();
    inspectedVertices += count;
  };
  for (const batch of source.batches) {
    for (let localRow = 0; localRow < batch.rowCount; localRow++) {
      const sourceRowIndex = batch.sourceRowOffset + localRow;
      let row: PolygonRow = [];
      if (sourceInvalidRows.has(sourceRowIndex)) {
        invalidRows.push({sourceRowIndex, reason: 'source-invalid'});
      } else {
        try {
          row = readPolygonRow(batch.polygons, localRow, accountVertices);
        } catch (error) {
          if (error instanceof PolygonContiguityCapacityError) throw error;
          invalidRows.push({sourceRowIndex, reason: 'invalid-topology'});
        }
      }
      if (row.length) {
        geometryMask[sourceRowIndex] = 1;
        for (const polygon of row) {
          for (const ring of polygon) {
            for (const [coordinateX, coordinateY] of ring) {
              let column = coordinateIds.get(coordinateX);
              if (!column) {
                column = new Map<number, number>();
                coordinateIds.set(coordinateX, column);
              }
              let identifier = column.get(coordinateY);
              if (identifier === undefined) {
                identifier = uniqueVertexCount++;
                column.set(coordinateY, identifier);
              }
              positionValues.push(identifier, 0);
            }
            ringOffsetValues.push(positionValues.length / 2);
          }
          polygonOffsetValues.push(ringOffsetValues.length - 1);
        }
      }
      featureOffsets[sourceRowIndex + 1] = polygonOffsetValues.length - 1;
    }
  }
  const positions = new Float32Array(positionValues);
  const ringOffsets = new Uint32Array(ringOffsetValues);
  const polygonOffsets = new Uint32Array(polygonOffsetValues);
  const cpuByteLength =
    positions.byteLength +
    ringOffsets.byteLength +
    polygonOffsets.byteLength +
    featureOffsets.byteLength +
    geometryMask.byteLength +
    manifestBytes +
    invalidRows.length * 64;
  return {
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    geometryMask,
    batches,
    invalidRows,
    rowCount,
    vertexCount: positions.length / 2,
    uniqueVertexCount,
    cpuByteLength,
    topologyEncoding: 'source-xy-key-ids',
    equality: 'exact-source-xy',
    ignoresZ: true
  };
}

class PolygonContiguityCapacityError extends Error {
  constructor() {
    super(
      `preparePolygonContiguityTopology decoded vertex capacity must be at most ${POLYGON_CONTIGUITY_MAXIMUM_VERTICES}`
    );
  }
}

function readPolygonRow(
  vector: Vector,
  rowIndex: number,
  accountVertices: (count: number) => void
): PolygonRow {
  // Normalize the hierarchy through the shared schema adapter. Equality is still read from
  // the original Arrow coordinates below, never its rounded metric Float32 positions.
  let coordinateType = vector.type;
  let depth = 0;
  while (DataType.isList(coordinateType)) {
    depth++;
    coordinateType = coordinateType.children[0].type;
  }
  const parts = readList(vector, rowIndex);
  if (parts === null || parts.length === 0) return [];
  if (parts.length > POLYGON_CONTIGUITY_MAXIMUM_VERTICES)
    throw new PolygonContiguityCapacityError();
  const row: PolygonRow = [];
  const partCount = depth === 2 ? 1 : parts.length;
  for (let partIndex = 0; partIndex < partCount; partIndex++) {
    const rings = depth === 2 ? parts : readList(parts, partIndex);
    if (!rings || !rings.length) throw new Error('Invalid polygon part');
    if (rings.length > POLYGON_CONTIGUITY_MAXIMUM_VERTICES)
      throw new PolygonContiguityCapacityError();
    const polygon: Coordinate[][] = [];
    const ringCount = rings.length;
    for (let ringIndex = 0; ringIndex < ringCount; ringIndex++) {
      const vertices = readList(rings, ringIndex);
      if (!vertices) throw new Error('Invalid polygon ring');
      accountVertices(vertices.length);
      const separated = DataType.isStruct(vertices.type);
      if (!separated && (!DataType.isFixedSizeList(vertices.type) || vertices.type.listSize < 2))
        throw new Error('Invalid polygon coordinate type');
      const ring: Coordinate[] = [];
      for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
        const coordinate = vertices.get(vertexIndex);
        if (!separated && (!(coordinate instanceof Vector) || coordinate.length < 2))
          throw new Error('Invalid polygon coordinate');
        const coordinateX: unknown = separated
          ? (coordinate as Record<string, unknown> | null)?.['x']
          : (coordinate as Vector).get(0);
        const coordinateY: unknown = separated
          ? (coordinate as Record<string, unknown> | null)?.['y']
          : (coordinate as Vector).get(1);
        if (
          typeof coordinateX !== 'number' ||
          typeof coordinateY !== 'number' ||
          !Number.isFinite(coordinateX) ||
          !Number.isFinite(coordinateY)
        )
          throw new Error('Nonfinite polygon coordinate');
        ring.push([coordinateX, coordinateY]);
      }
      while (ring.length > 1 && equalCoordinate(ring[0], ring[ring.length - 1])) ring.pop();
      const distinct = new Map<number, Set<number>>();
      let distinctCount = 0;
      for (const [coordinateX, coordinateY] of ring) {
        let column = distinct.get(coordinateX);
        if (!column) {
          column = new Set<number>();
          distinct.set(coordinateX, column);
        }
        if (!column.has(coordinateY)) {
          column.add(coordinateY);
          distinctCount++;
        }
        if (distinctCount === 3) break;
      }
      if (distinctCount < 3) throw new Error('Polygon ring needs three distinct XY vertices');
      polygon.push(ring);
    }
    row.push(polygon);
  }
  // Account and validate original vertices before the shared adapter allocates metric arrays.
  let normalized: ReturnType<typeof prepareGeometryInput> | undefined;
  try {
    normalized =
      (DataType.isFixedSizeList(coordinateType) && coordinateType.listSize === 2) ||
      DataType.isStruct(coordinateType)
        ? prepareGeometryInput(vector.slice(rowIndex, rowIndex + 1), {
            precision: 'allow-loss',
            encoding: depth === 2 ? 'geoarrow.polygon' : 'geoarrow.multipolygon'
          })
        : undefined;
  } catch (error) {
    // Metric adaptation is optional for equality-only data outside float32's finite range.
    if (
      !(error instanceof Error) ||
      !error.message.includes('positions must remain finite float32 XY')
    )
      throw error;
  }
  if (normalized && normalized.geometry.kind !== 'polygons')
    throw new Error('Polygon encoding required');
  if (normalized) {
    const geometry = normalized.geometry;
    const featureOffsets = geometry.featureOffsets;
    const polygonOffsets = geometry.polygonOffsets;
    validateRequirement(
      'preparePolygonContiguityTopology',
      featureOffsets !== undefined && featureOffsets.length >= 2 && polygonOffsets !== undefined,
      'normalized Polygon geometry requires featureOffsets and polygonOffsets'
    );
    const originalRings = row.flat();
    return Array.from({length: featureOffsets[1]}, (_, part) =>
      originalRings.slice(polygonOffsets[part], polygonOffsets[part + 1])
    );
  }
  return row;
}

/** Arrow slices already slice valueOffsets; data.offset applies to validity, not offset indexing. */
function readList(vector: Vector, rowIndex: number): Vector | null {
  if (!DataType.isList(vector.type)) throw new Error('Invalid polygon list type');
  let localIndex = rowIndex;
  for (const data of vector.data) {
    if (localIndex >= data.length) {
      localIndex -= data.length;
      continue;
    }
    if (!data.getValid(localIndex)) return null;
    const start = data.valueOffsets[localIndex];
    const end = data.valueOffsets[localIndex + 1];
    if (
      !Number.isSafeInteger(start) ||
      !Number.isSafeInteger(end) ||
      start < 0 ||
      end < start ||
      !data.children[0] ||
      end > data.children[0].length
    )
      throw new Error('Invalid polygon list offsets');
    const value = vector.get(rowIndex);
    if (!(value instanceof Vector)) throw new Error('Invalid polygon list value');
    return value;
  }
  throw new Error('Invalid polygon row index');
}

function equalCoordinate(first: Coordinate, second: Coordinate): boolean {
  return first[0] === second[0] && first[1] === second[1];
}

/**
 * Optional CPU execution over the same exact-source equality tokens used by GPUContiguityWeights.
 * Always uses zero snap tolerance: tokens are identities, never metric coordinates. The CPU twin
 * has no candidate-pair overflow and reports an exact requiredCount; final capacity still truncates.
 * Empty and invalid rows remain isolated and source aligned via geometryMask.
 */
export function computePolygonContiguityOnCPU(
  topology: PreparedPolygonContiguityTopology,
  options: {criterion: GPUContiguityCriterion; capacity: number}
): CPUContiguityWeightsResult {
  validateRequirement(
    'computePolygonContiguityOnCPU',
    topology.topologyEncoding === 'source-xy-key-ids' && topology.equality === 'exact-source-xy',
    'topology must use exact-source-xy equality tokens'
  );
  return computeContiguityWeightsOnCPU(topology, {...options, snapTolerance: 0});
}
