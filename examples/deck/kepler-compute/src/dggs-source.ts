// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {cellToBoundary, getResolution, hexToU64} from 'a5-js';
// s2-geometry is the pinned Kepler S2 dependency; it ships no type declarations.
// @ts-expect-error untyped CommonJS package
import {S2} from 's2-geometry';

/**
 * Discrete global grid cell families published through the resident polygon path. Cell IDs are
 * decoded exactly once, at publication, from string or Uint64 spellings. JavaScript numbers are
 * rejected because they cannot carry 64-bit identities. The original ID column is retained
 * unchanged beside a derived geoarrow.wkb footprint column that keeps source-row order and batches.
 */
export type DGGSFamily = 's2' | 'geohash' | 'a5';
export const DGGS_FOOTPRINT_COLUMN = 'dggs_footprint';

export type DGGSDiagnostic = {sourceRowIndex: number; reason: string};
export type DGGSFootprintSummary = {
  family: DGGSFamily;
  validRowCount: number;
  nullRowCount: number;
  invalidRows: readonly DGGSDiagnostic[];
  datelineCrossingRowCount: number;
};

const GEOHASH_ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz';
const MAXIMUM_S2_LEVEL = 30;
const MAXIMUM_A5_RESOLUTION = 30;
const MAXIMUM_S2_EDGE_SEGMENTS = 100;
const UINT64_MAXIMUM = (1n << 64n) - 1n;

export function stripQuotes(token: string): string {
  const first = token[0];
  return token.length >= 2 && (first === "'" || first === '"') && token.at(-1) === first
    ? token.slice(1, -1)
    : token;
}

/** Closed [longitude, latitude] ring, or a reason string for invalid input. */
export function decodeGeohashFootprint(value: string): Float64Array | string {
  const geohash = stripQuotes(value).toLowerCase();
  if (geohash.length < 1 || geohash.length > 12) return 'Geohash length must be 1 to 12';
  let minLatitude = -90;
  let maxLatitude = 90;
  let minLongitude = -180;
  let maxLongitude = 180;
  let isLongitude = true;
  for (const character of geohash) {
    const code = GEOHASH_ALPHABET.indexOf(character);
    if (code < 0) return 'Geohash contains a character outside base32';
    for (let bits = 4; bits >= 0; bits--) {
      const bit = (code >> bits) & 1;
      if (isLongitude) {
        const middle = (maxLongitude + minLongitude) / 2;
        if (bit) minLongitude = middle;
        else maxLongitude = middle;
      } else {
        const middle = (maxLatitude + minLatitude) / 2;
        if (bit) minLatitude = middle;
        else maxLatitude = middle;
      }
      isLongitude = !isLongitude;
    }
  }
  return Float64Array.of(
    minLongitude,
    minLatitude,
    maxLongitude,
    minLatitude,
    maxLongitude,
    maxLatitude,
    minLongitude,
    maxLatitude,
    minLongitude,
    minLatitude
  );
}

/** Exact Uint64 S2 cell ID from a Kepler hex token (zero padded to 16 digits) or a bigint. */
export function readS2CellId(value: string | bigint): bigint | string {
  let identifier: bigint;
  if (typeof value === 'bigint') identifier = value;
  else {
    const token = stripQuotes(value);
    if (!/^[0-9a-fA-F]{1,16}$/.test(token)) return 'S2 token must be 1 to 16 hexadecimal digits';
    identifier = BigInt(`0x${token.padEnd(16, '0')}`);
  }
  if (identifier < 1n || identifier > UINT64_MAXIMUM) return 'S2 cell ID is outside Uint64';
  const face = Number(identifier >> 61n);
  const leastSignificantBit = identifier & -identifier;
  // Valid cells have the marker bit on an even position: level = 30 - position / 2.
  if (face > 5 || (leastSignificantBit & 0x1555555555555555n) === 0n) return 'Invalid S2 cell ID';
  return identifier;
}

export function decodeS2Footprint(value: string | bigint): Float64Array | string {
  const identifier = readS2CellId(value);
  if (typeof identifier === 'string') return identifier;
  // Hilbert quadkey digits come straight from the exact BigInt, without a decimal Long round trip.
  const level =
    MAXIMUM_S2_LEVEL - Math.floor((identifier & -identifier).toString(2).length / 2 - 0.5);
  const digits = (identifier >> BigInt(61 - 2 * level)) & ((1n << BigInt(2 * level)) - 1n);
  const position = level ? digits.toString(4).padStart(level, '0') : '';
  const faceText = String(identifier >> 61n);
  const key = `${faceText}/${position}`;
  // The pinned library cannot build level 0 from an empty quadkey.
  const cell: {face: number; ij: [number, number]} =
    level === 0 ? {face: Number(faceText), ij: [0, 0]} : S2.S2Cell.FromHilbertQuadKey(key);
  const offsets = [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0]
  ];
  // Same curved-edge subdivision as the pinned Deck S2Layer.
  const segments = Math.max(1, Math.ceil(MAXIMUM_S2_EDGE_SEGMENTS * Math.pow(2, -level)));
  const ring = new Float64Array(4 * segments * 2 + 2);
  let index = 0;
  let previousLongitude = 0;
  for (let side = 0; side < 4; side++) {
    const offset = [...offsets[side]];
    const stepI = (offsets[side + 1][0] - offset[0]) / segments;
    const stepJ = (offsets[side + 1][1] - offset[1]) / segments;
    for (let segment = 0; segment < segments; segment++) {
      offset[0] += stepI;
      offset[1] += stepJ;
      const spherical = S2.STToUV(S2.IJToST(cell.ij, level, offset));
      const latLng = S2.XYZToLatLng(S2.FaceUVToXYZ(cell.face, spherical));
      let longitude: number = latLng.lng;
      if (Math.abs(latLng.lat) > 89.999) longitude = previousLongitude;
      const delta = longitude - previousLongitude;
      longitude += delta > 180 ? -360 : delta < -180 ? 360 : 0;
      ring[index++] = longitude;
      ring[index++] = latLng.lat;
      previousLongitude = longitude;
    }
  }
  ring[index++] = ring[0];
  ring[index] = ring[1];
  return ring;
}

export function decodeA5Footprint(value: string | bigint): Float64Array | string {
  let cell: bigint;
  try {
    if (typeof value === 'bigint') cell = value;
    else {
      const token = stripQuotes(value).trim();
      if (!/^[0-9a-fA-F]+$/.test(token)) return 'A5 token must be hexadecimal';
      cell = hexToU64(token);
    }
    if (cell < 0n || cell > UINT64_MAXIMUM) return 'A5 cell ID is outside Uint64';
    const resolution = getResolution(cell);
    if (resolution < 0 || resolution > MAXIMUM_A5_RESOLUTION) return 'Invalid A5 resolution';
    const boundary = cellToBoundary(cell, {closedRing: true, segments: 'auto'});
    const ring = new Float64Array(boundary.length * 2);
    let previousLongitude = Number(boundary[0][0]);
    boundary.forEach(([degreesLongitude, degreesLatitude], vertex) => {
      const longitude = Number(degreesLongitude);
      const latitude = Number(degreesLatitude);
      // Keep consecutive longitudes continuous across the antimeridian.
      const delta = longitude - previousLongitude;
      const continuous = longitude + (delta > 180 ? -360 : delta < -180 ? 360 : 0);
      ring[vertex * 2] = continuous;
      ring[vertex * 2 + 1] = latitude;
      previousLongitude = continuous;
    });
    return ring;
  } catch {
    return 'Invalid A5 cell ID';
  }
}

function decodeFootprint(family: DGGSFamily, value: string | bigint): Float64Array | string {
  if (family === 'geohash')
    return typeof value === 'string' ? decodeGeohashFootprint(value) : 'Geohash must be a string';
  return family === 's2' ? decodeS2Footprint(value) : decodeA5Footprint(value);
}

function toCounterClockwise(ring: Float64Array): Float64Array {
  let area = 0;
  for (let index = 0; index + 3 < ring.length; index += 2)
    area += ring[index] * ring[index + 3] - ring[index + 2] * ring[index + 1];
  if (area >= 0) return ring;
  const reversed = new Float64Array(ring.length);
  for (let vertex = 0; vertex < ring.length / 2; vertex++) {
    const source = ring.length / 2 - 1 - vertex;
    reversed[vertex * 2] = ring[source * 2];
    reversed[vertex * 2 + 1] = ring[source * 2 + 1];
  }
  return reversed;
}

/** Little-endian ISO WKB Polygon with one exterior ring. */
export function writePolygonWKB(ring: Float64Array): Uint8Array {
  const vertexCount = ring.length / 2;
  const bytes = new Uint8Array(1 + 4 + 4 + 4 + ring.length * 8);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, 1);
  view.setUint32(1, 3, true);
  view.setUint32(5, 1, true);
  view.setUint32(9, vertexCount, true);
  for (let index = 0; index < ring.length; index++)
    view.setFloat64(13 + index * 8, ring[index], true);
  return bytes;
}

/**
 * Append the derived footprint column to every original record batch. Invalid IDs, nulls and
 * numbers-as-IDs: JavaScript numbers throw (they may have lost precision); other invalid or null
 * IDs keep their source row and have no footprint. A source with no valid cells is rejected.
 */
export function appendDGGSFootprintColumn(
  table: arrow.Table,
  cellColumn: string,
  family: DGGSFamily,
  footprintColumn = DGGS_FOOTPRINT_COLUMN
): {table: arrow.Table; summary: DGGSFootprintSummary} {
  if (!['s2', 'geohash', 'a5'].includes(family)) throw new Error('Unknown cell index family');
  if (!table.getChild(cellColumn)) throw new Error('Cell ID column is missing from the source');
  if (table.getChild(footprintColumn)) throw new Error(`Column ${footprintColumn} already exists`);
  const invalidRows: DGGSDiagnostic[] = [];
  let validRowCount = 0;
  let nullRowCount = 0;
  let datelineCrossingRowCount = 0;
  let sourceRowIndex = 0;
  const field = new arrow.Field(
    footprintColumn,
    new arrow.Binary(),
    true,
    new Map([
      ['ARROW:extension:name', 'geoarrow.wkb'],
      ['ARROW:extension:metadata', JSON.stringify({crs: 'OGC:CRS84'})]
    ])
  );
  const schema = new arrow.Schema([...table.schema.fields, field], table.schema.metadata);
  const batches = table.batches.map(batch => {
    const column = batch.getChild(cellColumn)!;
    const builder = arrow.makeBuilder({type: new arrow.Binary(), nullValues: [null, undefined]});
    for (let row = 0; row < batch.numRows; row++, sourceRowIndex++) {
      const value: unknown = column.get(row);
      if (value === null || value === undefined) {
        nullRowCount++;
        builder.append(null);
        continue;
      }
      if (typeof value === 'number')
        throw new Error(
          `Cell IDs in row ${sourceRowIndex} are JavaScript numbers and may have lost precision; use strings or Uint64`
        );
      const ring =
        typeof value === 'string' || typeof value === 'bigint'
          ? decodeFootprint(family, value)
          : 'Unsupported cell ID type';
      if (typeof ring === 'string') {
        invalidRows.push({sourceRowIndex, reason: ring});
        builder.append(null);
        continue;
      }
      validRowCount++;
      let minimumLongitude = Infinity;
      let maximumLongitude = -Infinity;
      for (let index = 0; index < ring.length; index += 2) {
        minimumLongitude = Math.min(minimumLongitude, ring[index]);
        maximumLongitude = Math.max(maximumLongitude, ring[index]);
      }
      if (minimumLongitude < -180 || maximumLongitude > 180) datelineCrossingRowCount++;
      builder.append(writePolygonWKB(toCounterClockwise(ring)));
    }
    const footprint = builder.finish().flush();
    const data = arrow.makeData({
      type: new arrow.Struct(schema.fields),
      length: batch.numRows,
      nullCount: 0,
      children: [...batch.data.children, footprint]
    });
    return new arrow.RecordBatch(schema, data);
  });
  if (!validRowCount) throw new Error('The cell ID column contains no valid cells');
  return {
    table: new arrow.Table(schema, batches),
    summary: {family, validRowCount, nullRowCount, invalidRows, datelineCrossingRowCount}
  };
}
