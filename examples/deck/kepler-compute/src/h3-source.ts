// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  cellToBoundary,
  cellToLatLng,
  getHexagonEdgeLengthAvg,
  getIcosahedronFaces,
  getResolution,
  gridDistance,
  isPentagon,
  isValidCell,
  latLngToCell
} from 'h3-js';

/** Pinned Kepler 3.3 accessor semantics: h3-typed fields are raw hex; other fields try BigInt. */
export type H3SourceAccessor = 'native-h3' | 'decimal-or-hex';
export type H3SourceBatch = {id: string; values: readonly unknown[]};
export type H3SourceOptions = {
  accessor?: H3SourceAccessor;
  maximumRowCount?: number;
  maximumVertexCount?: number;
};
export type H3SourceDiagnostic = {
  sourceRowIndex: number;
  code: 'number-identity' | 'invalid-type' | 'uint64-range' | 'invalid-cell' | 'accessor-mismatch';
  reason: string;
};
export type PreparedH3Batch = {
  id: string;
  rowCount: number;
  sourceRowOffset: number;
  /** Original string/bigint spelling, including invalid values; no Number conversion. */
  sourceIds: readonly (string | bigint | null)[];
  canonicalIds: readonly (string | null)[];
  cellIds: BigUint64Array;
  /** Source-row aligned longitude, latitude. Invalid rows use zero plus validity=0. */
  centers: Float64Array;
  validity: Uint32Array;
  resolutions: Uint32Array;
  pentagons: Uint32Array;
  /** Canonical h3-js closed boundaries, unscaled and with original longitude wrapping. */
  boundaryPositions: Float64Array;
  boundaryOffsets: Uint32Array;
  vertexSourceRows: Uint32Array;
  primitiveSourceRows: Uint32Array;
};
export type H3SourceMetadata = {
  resolution: number;
  resolutions: readonly number[];
  hasMultipleRes: boolean;
  hasPentagon: boolean;
  hasFaceCrossing: boolean;
  hasDatelineCrossing: boolean;
};
export type PreparedH3Source = {
  batches: readonly PreparedH3Batch[];
  rowCount: number;
  validRowCount: number;
  nullRowCount: number;
  vertexCount: number;
  invalidRows: readonly H3SourceDiagnostic[];
  metadata: H3SourceMetadata;
  /** Spatial Float32 diagnostics, independent of exact H3 identity and source validity. */
  precision: {
    maximumCoordinateErrorDegrees: number;
    maximumCenterErrorDegrees: number;
    collapsedRows: readonly {sourceRowIndex: number; reason: string}[];
  };
  typedStorageByteLength: number;
  /** UTF8 bytes for original and canonical strings, plus 8 per retained bigint; excludes JS object overhead. */
  identityByteLength: number;
};

/**
 * Initial source preparation only. Canonical boundaries are not Deck's viewport column mesh.
 * Exact identity is separate from spatial precision: canonical coordinates remain Float64 here,
 * while precision diagnoses the current Float32 center/boundary upload. Consumers must reject
 * the whole source when collapsedRows is nonempty, before publishing or allocating geometry;
 * those valid H3 rows are never reclassified as invalid IDs or silently dropped.
 * Noncollapsing rounding is admitted and measured in degrees, not claimed to match fp64 Deck.
 * Split high/low geographic coordinates remain a future adapter/renderer requirement.
 */
export function prepareH3Source(
  batches: readonly H3SourceBatch[],
  options: H3SourceOptions = {}
): PreparedH3Source {
  const accessor = options.accessor ?? 'decimal-or-hex';
  if (accessor !== 'native-h3' && accessor !== 'decimal-or-hex')
    throw new Error('Unknown H3 source accessor');
  const maximumRowCount = options.maximumRowCount ?? 0xffffffff;
  const maximumVertexCount = options.maximumVertexCount ?? 0xffffffff;
  for (const limit of [maximumRowCount, maximumVertexCount])
    if (!Number.isSafeInteger(limit) || limit < 0 || limit > 0xffffffff)
      throw new Error('H3 preparation limits must fit uint32');
  let rowCount = 0;
  let validRowCount = 0;
  let nullRowCount = 0;
  let vertexCount = 0;
  let identityByteLength = 0;
  let maximumCoordinateErrorDegrees = 0;
  let maximumCenterErrorDegrees = 0;
  const collapsedRows: {sourceRowIndex: number; reason: string}[] = [];
  let hasPentagon = false;
  let hasFaceCrossing = false;
  let hasDatelineCrossing = false;
  const resolutions = new Set<number>();
  const batchIds = new Set<string>();
  const invalidRows: H3SourceDiagnostic[] = [];
  const encoder = new TextEncoder();
  const preparedBatches = batches.map(batch => {
    if (!batch.id || batchIds.has(batch.id)) throw new Error('H3 source batch IDs must be unique');
    batchIds.add(batch.id);
    const sourceRowOffset = rowCount;
    rowCount += batch.values.length;
    if (rowCount > maximumRowCount) throw new Error('H3 source exceeds its row limit');
    const sourceIds: (string | bigint | null)[] = [];
    const canonicalIds: (string | null)[] = new Array(batch.values.length).fill(null);
    const cellIds = new BigUint64Array(batch.values.length);
    const centers = new Float64Array(batch.values.length * 2);
    const validity = new Uint32Array(batch.values.length);
    const rowResolutions = new Uint32Array(batch.values.length);
    const pentagons = new Uint32Array(batch.values.length);
    const boundaryPositions: number[] = [];
    const boundaryOffsets = new Uint32Array(batch.values.length + 1);
    const vertexSourceRows: number[] = [];
    const primitiveSourceRows: number[] = [];
    for (let rowIndex = 0; rowIndex < batch.values.length; rowIndex++) {
      const value = batch.values[rowIndex];
      const sourceRowIndex = sourceRowOffset + rowIndex;
      sourceIds.push(typeof value === 'string' || typeof value === 'bigint' ? value : null);
      identityByteLength +=
        typeof value === 'string'
          ? encoder.encode(value).byteLength
          : typeof value === 'bigint'
            ? 8
            : 0;
      boundaryOffsets[rowIndex] = boundaryPositions.length / 2;
      if (value === null || value === undefined) {
        nullRowCount++;
        continue;
      }
      const resolved = resolveH3Identity(value, accessor);
      if ('code' in resolved) {
        invalidRows.push({sourceRowIndex, ...resolved});
        continue;
      }
      const cellId = resolved.cellId;
      const boundary = cellToBoundary(cellId, true);
      if (vertexCount + boundary.length > maximumVertexCount)
        throw new Error('H3 source exceeds its boundary vertex limit');
      const [latitude, longitude] = cellToLatLng(cellId);
      const centerError = Math.max(
        Math.abs(Math.fround(longitude) - longitude),
        Math.abs(Math.fround(latitude) - latitude)
      );
      maximumCenterErrorDegrees = Math.max(maximumCenterErrorDegrees, centerError);
      const precision = getBoundaryPrecision(boundary);
      maximumCoordinateErrorDegrees = Math.max(
        maximumCoordinateErrorDegrees,
        centerError,
        precision.maximumErrorDegrees
      );
      if (precision.reason) collapsedRows.push({sourceRowIndex, reason: precision.reason});
      const resolution = getResolution(cellId);
      const pentagon = isPentagon(cellId);
      hasFaceCrossing ||= getIcosahedronFaces(cellId).length > 1;
      hasPentagon ||= pentagon;
      resolutions.add(resolution);
      canonicalIds[rowIndex] = cellId;
      identityByteLength += encoder.encode(cellId).byteLength;
      cellIds[rowIndex] = resolved.integer;
      centers.set([longitude, latitude], rowIndex * 2);
      validity[rowIndex] = 1;
      rowResolutions[rowIndex] = resolution;
      pentagons[rowIndex] = Number(pentagon);
      primitiveSourceRows.push(sourceRowIndex);
      for (let vertexIndex = 0; vertexIndex < boundary.length; vertexIndex++) {
        const position = boundary[vertexIndex];
        if (vertexIndex && Math.abs(position[0] - boundary[vertexIndex - 1][0]) > 180)
          hasDatelineCrossing = true;
        boundaryPositions.push(position[0], position[1]);
        vertexSourceRows.push(sourceRowIndex);
      }
      vertexCount += boundary.length;
      validRowCount++;
    }
    boundaryOffsets[batch.values.length] = boundaryPositions.length / 2;
    return {
      id: batch.id,
      rowCount: batch.values.length,
      sourceRowOffset,
      sourceIds,
      canonicalIds,
      cellIds,
      centers,
      validity,
      resolutions: rowResolutions,
      pentagons,
      boundaryPositions: Float64Array.from(boundaryPositions),
      boundaryOffsets,
      vertexSourceRows: Uint32Array.from(vertexSourceRows),
      primitiveSourceRows: Uint32Array.from(primitiveSourceRows)
    };
  });
  const resolutionValues = [...resolutions];
  return {
    batches: preparedBatches,
    rowCount,
    validRowCount,
    nullRowCount,
    vertexCount,
    invalidRows,
    metadata: {
      resolution: resolutionValues[0] ?? -1,
      resolutions: resolutionValues,
      hasMultipleRes: resolutions.size > 1,
      hasPentagon,
      hasFaceCrossing,
      hasDatelineCrossing
    },
    precision: {maximumCoordinateErrorDegrees, maximumCenterErrorDegrees, collapsedRows},
    typedStorageByteLength: preparedBatches.reduce(
      (total, batch) =>
        total +
        batch.cellIds.byteLength +
        batch.centers.byteLength +
        batch.validity.byteLength +
        batch.resolutions.byteLength +
        batch.pentagons.byteLength +
        batch.boundaryPositions.byteLength +
        batch.boundaryOffsets.byteLength +
        batch.vertexSourceRows.byteLength +
        batch.primitiveSourceRows.byteLength,
      0
    ),
    identityByteLength
  };
}

/** Inspect the exact normalized coordinates uploaded by the current canonical polygon adapter. */
function getBoundaryPrecision(boundary: number[][]): {
  maximumErrorDegrees: number;
  reason?: string;
} {
  const reference = boundary[0][0];
  const normalized = boundary.map(([longitude, latitude]) => [
    normalizeLongitude(longitude, reference),
    latitude
  ]);
  const quantized = normalized.map(position => position.map(Math.fround));
  let maximumErrorDegrees = 0;
  let collapsedEdge = false;
  for (let index = 0; index < normalized.length; index++) {
    maximumErrorDegrees = Math.max(
      maximumErrorDegrees,
      Math.abs(normalized[index][0] - quantized[index][0]),
      Math.abs(normalized[index][1] - quantized[index][1])
    );
    if (
      index > 0 &&
      (normalized[index][0] !== normalized[index - 1][0] ||
        normalized[index][1] !== normalized[index - 1][1]) &&
      quantized[index][0] === quantized[index - 1][0] &&
      quantized[index][1] === quantized[index - 1][1]
    )
      collapsedEdge = true;
  }
  const originalArea = getSignedBoundaryArea(normalized);
  const quantizedArea = getSignedBoundaryArea(quantized);
  const distinctVertices = new Set(quantized.map(position => `${position[0]},${position[1]}`)).size;
  const reason = collapsedEdge
    ? 'A valid H3 boundary edge collapses after Float32 longitude/latitude storage.'
    : distinctVertices < 3 || (originalArea !== 0 && quantizedArea === 0)
      ? 'A valid H3 boundary becomes degenerate after Float32 longitude/latitude storage.'
      : originalArea * quantizedArea < 0
        ? 'Float32 longitude/latitude storage reverses the H3 boundary orientation.'
        : undefined;
  return {maximumErrorDegrees, reason};
}
function getSignedBoundaryArea(positions: number[][]): number {
  const origin = positions[0];
  let area = 0;
  for (let index = 1; index + 1 < positions.length; index++) {
    const current = positions[index];
    const next = positions[index + 1];
    area +=
      (current[0] - origin[0]) * (next[1] - origin[1]) -
      (next[0] - origin[0]) * (current[1] - origin[1]);
  }
  return area;
}

/** Arrow Uint64 .get() remains bigint; slicing and physical empty chunks are not packed. */
export function prepareH3SourceFromArrow(
  vector: arrow.Vector,
  options: H3SourceOptions & {batchIds?: readonly string[]} = {}
): PreparedH3Source {
  if (options.batchIds && options.batchIds.length !== vector.data.length)
    throw new Error('H3 batch IDs must match Arrow chunks');
  return prepareH3Source(
    vector.data.map((chunk, index) => {
      const values = new arrow.Vector([chunk]);
      return {
        id: options.batchIds?.[index] ?? `h3-batch-${index}`,
        values: Array.from({length: values.length}, (_, rowIndex) => values.get(rowIndex))
      };
    }),
    options
  );
}

function resolveH3Identity(
  value: unknown,
  accessor: H3SourceAccessor
): {cellId: string; integer: bigint} | Omit<H3SourceDiagnostic, 'sourceRowIndex'> {
  if (typeof value === 'number')
    return {
      code: 'number-identity',
      reason:
        'H3 identities must remain exact strings or Uint64 bigint values, never JavaScript Numbers'
    };
  if (typeof value !== 'string' && typeof value !== 'bigint')
    return {code: 'invalid-type', reason: 'H3 identity requires a string or Uint64 bigint'};
  let integer: bigint;
  if (typeof value === 'string' && isValidCell(value)) {
    integer = BigInt(`0x${value}`);
  } else {
    if (accessor === 'native-h3')
      return {
        code: 'accessor-mismatch',
        reason: 'Native h3 fields require a valid hexadecimal cell string'
      };
    try {
      integer = BigInt(value);
    } catch {
      return {
        code: 'invalid-cell',
        reason: 'H3 identity is neither a valid hexadecimal cell nor an exact integer string'
      };
    }
  }
  if (integer < 0n || integer > 0xffffffffffffffffn)
    return {code: 'uint64-range', reason: 'H3 identity is outside the Uint64 range'};
  const cellId = integer.toString(16);
  return isValidCell(cellId)
    ? {cellId, integer}
    : {code: 'invalid-cell', reason: 'H3 identity is not a valid H3 cell'};
}

/** Supply CURRENT selected-row metadata for interaction; source-wide metadata is only the initial state. */
export function getH3RenderMode(
  metadata: Pick<H3SourceMetadata, 'resolution' | 'hasMultipleRes' | 'hasPentagon'>,
  options: {highPrecision?: boolean | 'auto'; viewportResolution?: number} = {}
): 'canonical-polygons' | 'viewport-columns' {
  const precision = options.highPrecision ?? 'auto';
  const polygons =
    precision === 'auto'
      ? Boolean(options.viewportResolution) ||
        metadata.hasMultipleRes ||
        metadata.hasPentagon ||
        (metadata.resolution >= 0 && metadata.resolution <= 5)
      : precision;
  return polygons ? 'canonical-polygons' : 'viewport-columns';
}

/** Initial canonical polygon preparation; native high-precision coverage normalization is explicit. */
export function getH3Boundary(
  source: PreparedH3Source,
  batchIndex: number,
  rowIndex: number,
  options: {coverage?: number} = {}
): [number, number][] {
  const batch = source.batches[batchIndex];
  if (
    !batch ||
    !Number.isSafeInteger(rowIndex) ||
    rowIndex < 0 ||
    rowIndex >= batch.rowCount ||
    !batch.validity[rowIndex]
  )
    return [];
  const coverage = options.coverage ?? 1;
  if (!Number.isFinite(coverage)) throw new Error('H3 coverage must be finite');
  const start = batch.boundaryOffsets[rowIndex];
  const end = batch.boundaryOffsets[rowIndex + 1];
  const centerLongitude = batch.centers[rowIndex * 2];
  const centerLatitude = batch.centers[rowIndex * 2 + 1];
  const reference = coverage === 1 ? batch.boundaryPositions[start * 2] : centerLongitude;
  const vertices: [number, number][] = [];
  for (let index = start; index < end; index++) {
    const longitude = normalizeLongitude(batch.boundaryPositions[index * 2], reference);
    const latitude = batch.boundaryPositions[index * 2 + 1];
    vertices.push(
      coverage === 1
        ? [longitude, latitude]
        : [
            centerLongitude + (longitude - centerLongitude) * coverage,
            centerLatitude + (latitude - centerLatitude) * coverage
          ]
    );
  }
  return vertices;
}

export type H3Viewport = {
  longitude: number;
  latitude: number;
  projectFlat(position: number[]): number[];
  distanceScales: {unitsPerMeter: readonly number[]};
};
export type PreparedH3ViewportMesh = {
  centerCellId: string;
  resolution: number;
  /** Closed native meter-space XY perimeter, not necessarily a regular hexagon. */
  positions: Float64Array;
};
/**
 * O(1) viewport metadata update. Matches pinned @deck.gl/geo-layers H3HexagonLayer._updateVertices:
 * center-cell cache, ten-kilometer reuse rule, longitude normalization and common-space meters.
 * Canonical source polygons are independent of this viewport-specific low-precision footprint.
 */
export function prepareH3ViewportMesh(options: {
  resolution: number;
  viewport: H3Viewport;
  previous?: PreparedH3ViewportMesh;
  centerHexagon?: string;
}): PreparedH3ViewportMesh | undefined {
  const {resolution, viewport, previous} = options;
  if (resolution < 0) return undefined;
  const centerCellId =
    options.centerHexagon ?? latLngToCell(viewport.latitude, viewport.longitude, resolution);
  if (!isValidCell(centerCellId)) throw new Error('H3 viewport center must be a valid cell');
  if (previous?.centerCellId === centerCellId) return previous;
  if (previous) {
    try {
      if (
        gridDistance(previous.centerCellId, centerCellId) *
          getHexagonEdgeLengthAvg(resolution, 'km') <
        10
      )
        return previous;
    } catch {
      /* Native gridDistance can fail across pentagons or different resolutions. Rebuild in that case. */
    }
  }
  const [latitude, longitude] = cellToLatLng(centerCellId);
  const center = viewport.projectFlat([longitude, latitude]);
  const boundary = cellToBoundary(centerCellId, true);
  const reference = boundary[0][0];
  const positions = new Float64Array(boundary.length * 2);
  for (const [index, position] of boundary.entries()) {
    const projected = viewport.projectFlat([
      normalizeLongitude(position[0], reference),
      position[1]
    ]);
    positions[index * 2] = (projected[0] - center[0]) / viewport.distanceScales.unitsPerMeter[0];
    positions[index * 2 + 1] =
      (projected[1] - center[1]) / viewport.distanceScales.unitsPerMeter[1];
  }
  return {centerCellId, resolution, positions};
}
function normalizeLongitude(longitude: number, reference: number): number {
  const delta = longitude - reference;
  return delta > 180 ? longitude - 360 : delta < -180 ? longitude + 360 : longitude;
}
