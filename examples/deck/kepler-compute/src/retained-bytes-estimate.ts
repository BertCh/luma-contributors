// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

export type ColumnRestoreEstimate = {
  name: string;
  typeName: string;
  /** Retained CPU bytes: every Arrow buffer of the column (values, offsets, validity, children, dictionary). */
  cpuBytes: number;
  /** Expected resident GPU bytes after the analytics routing and the Float32 policy. */
  gpuBytes: number;
};

export type RestoreBytesEstimate = {
  cpuBytes: number;
  gpuBytes: number;
  totalBytes: number;
  perColumn: ColumnRestoreEstimate[];
};

const GEOMETRY_NAME_PATTERN = /(^|_)(wkt|wkb|geom|geometry|the_geom)($|_)/i;
/** Average WKT characters per `x y,` vertex; a deliberately cheap upper-bound heuristic. */
const WKT_CHARACTERS_PER_VERTEX = 12;
/** Average WKB bytes per vertex (two Float64 coordinates). */
const WKB_BYTES_PER_VERTEX = 16;
/** Planar Float32 x/y per vertex plus ring/polygon/feature offsets and mask per row. */
const GEOMETRY_GPU_BYTES_PER_VERTEX = 8;
const GEOMETRY_GPU_BYTES_PER_ROW = 16;
/** Int64/Uint64/Decimal default to exact decimal text (category); labels are about 24 bytes each. */
const DECIMAL_TEXT_BYTES_PER_ROW = 24;

/**
 * Per-type restore accounting that replaces the crude `2 x batch.byteLength` estimate. CPU bytes are
 * the Arrow buffers that stay retained (see `importArrowDataset`); GPU bytes follow the Arrow routing
 * (`source-type-routes.ts`): Float64 and every other numeric narrows to one 32-bit word per row under
 * the Float32 policy, Bool widens to Uint32, Timestamp/Date become two 32-bit words, string columns
 * upload a 32-bit category index (their labels are retained CPU), geometry uploads planar Float32
 * positions plus offsets, and Binary/List/Struct columns are CPU only. Nullable GPU columns add a
 * one-bit-per-row validity mask rounded up to 32-bit words.
 */
export function estimateRestoreBytes(table: arrow.Table): RestoreBytesEstimate {
  const rowCount = table.numRows;
  const validityBytes = Math.ceil(rowCount / 32) * 4;
  const perColumn: ColumnRestoreEstimate[] = [];
  for (const field of table.schema.fields) {
    const column = table.getChild(field.name);
    const type = field.type;
    const seen = new Map<ArrayBufferLike, Set<string>>();
    let cpuBytes = 0;
    let stringDataBytes = 0;
    for (const chunk of column?.data ?? []) {
      cpuBytes += measureData(chunk, seen);
      stringDataBytes += measureStringData(chunk);
    }
    let gpuBytes = 0;
    let extraCpuBytes = 0;
    const isGeometryName = GEOMETRY_NAME_PATTERN.test(field.name);
    if (arrow.DataType.isDictionary(type)) {
      gpuBytes = rowCount * 4;
    } else if (arrow.DataType.isUtf8(type) || arrow.DataType.isLargeUtf8(type)) {
      if (isGeometryName) {
        gpuBytes =
          Math.ceil(stringDataBytes / WKT_CHARACTERS_PER_VERTEX) * GEOMETRY_GPU_BYTES_PER_VERTEX +
          rowCount * GEOMETRY_GPU_BYTES_PER_ROW;
      } else {
        gpuBytes = rowCount * 4;
        // The category dictionary holds at most one label per row (upper bound: all distinct).
        extraCpuBytes = stringDataBytes;
      }
    } else if (
      arrow.DataType.isBinary(type) ||
      arrow.DataType.isLargeBinary(type) ||
      arrow.DataType.isFixedSizeBinary(type)
    ) {
      if (isGeometryName) {
        const dataBytes = Math.max(0, cpuBytes - (rowCount + 1) * 4);
        gpuBytes =
          Math.ceil(dataBytes / WKB_BYTES_PER_VERTEX) * GEOMETRY_GPU_BYTES_PER_VERTEX +
          rowCount * GEOMETRY_GPU_BYTES_PER_ROW;
      }
    } else if (arrow.DataType.isTimestamp(type) || arrow.DataType.isDate(type)) {
      gpuBytes = rowCount * 8;
    } else if (arrow.DataType.isInt(type) && type.bitWidth === 64) {
      gpuBytes = rowCount * 4;
      extraCpuBytes = rowCount * DECIMAL_TEXT_BYTES_PER_ROW;
    } else if (arrow.DataType.isDecimal(type)) {
      gpuBytes = rowCount * 4;
      extraCpuBytes = rowCount * DECIMAL_TEXT_BYTES_PER_ROW;
    } else if (
      arrow.DataType.isInt(type) ||
      arrow.DataType.isFloat(type) ||
      arrow.DataType.isBool(type)
    ) {
      // Float64 narrows to Float32; Float16, Bool and <=32-bit integers occupy one 32-bit word.
      gpuBytes = rowCount * 4;
    }
    if (gpuBytes > 0 && (column?.nullCount ?? 0) > 0) gpuBytes += validityBytes;
    perColumn.push({
      name: field.name,
      typeName: String(type),
      cpuBytes: cpuBytes + extraCpuBytes,
      gpuBytes
    });
  }
  const cpuBytes = perColumn.reduce((total, column) => total + column.cpuBytes, 0);
  const gpuBytes = perColumn.reduce((total, column) => total + column.gpuBytes, 0);
  return {cpuBytes, gpuBytes, totalBytes: cpuBytes + gpuBytes, perColumn};
}

/** Sums the distinct buffers of one Arrow Data node tree (validity, values, offsets, children, dictionary). */
function measureData(data: arrow.Data, seen: Map<ArrayBufferLike, Set<string>>): number {
  let total = 0;
  for (const buffer of [data.values, data.valueOffsets, data.nullBitmap, data.typeIds]) {
    if (!buffer) continue;
    const key = `${buffer.byteOffset}:${buffer.byteLength}`;
    const ranges = seen.get(buffer.buffer) ?? new Set<string>();
    seen.set(buffer.buffer, ranges);
    if (ranges.has(key)) continue;
    ranges.add(key);
    total += buffer.byteLength;
  }
  for (const child of data.children) total += measureData(child, seen);
  for (const dictionaryData of data.dictionary?.data ?? [])
    total += measureData(dictionaryData, seen);
  return total;
}

function measureStringData(data: arrow.Data): number {
  if (!arrow.DataType.isUtf8(data.type) && !arrow.DataType.isLargeUtf8(data.type)) return 0;
  return data.values?.byteLength ?? 0;
}
