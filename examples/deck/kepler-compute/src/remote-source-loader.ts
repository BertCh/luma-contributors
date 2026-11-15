// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {CoordinateRoles} from './point-position-source';
import type {SourceCandidate} from './source-workspace-panel';
import {routeArrowTable} from './source-type-routes';

/**
 * Remote and local file ingestion into an Arrow table without any GPU upload.
 * Decoders reused from the installed dependency tree: `d3-dsv` (CSV), `parquet-wasm` (Parquet,
 * including GeoParquet WKB geometry through `@loaders.gl/arrow-geometry`) and `apache-arrow`
 * (IPC file and stream). GeoJSON is parsed with `JSON.parse` and flattened here.
 */
export type RemoteSourceFormat = 'csv' | 'geojson' | 'parquet' | 'arrow';

export type RemoteLoadProgress = {loadedBytes: number; totalBytes: number | null};

export type RemoteLoadOptions = {
  signal?: AbortSignal;
  maximumBytes?: number;
  onProgress?: (progress: RemoteLoadProgress) => void;
  fetchFunction?: typeof fetch;
};

export type SourceTypeRoutingOptions = {
  /** Retain IPC source types for explicit preparation instead of converting wide integers to labels. */
  preserveArrowTypes?: boolean;
};

export const REMOTE_SOURCE_MAXIMUM_BYTES = 128 * 1024 * 1024;
export const REMOTE_SOURCE_FILE_ACCEPT = '.arrow,.feather,.ipc,.csv,.tsv,.geojson,.json,.parquet';

const EXTENSION_FORMATS: Readonly<Record<string, RemoteSourceFormat>> = {
  csv: 'csv',
  tsv: 'csv',
  txt: 'csv',
  geojson: 'geojson',
  json: 'geojson',
  parquet: 'parquet',
  geoparquet: 'parquet',
  arrow: 'arrow',
  feather: 'arrow',
  ipc: 'arrow'
};
const CONTENT_TYPE_FORMATS: readonly [RegExp, RemoteSourceFormat][] = [
  [/geo\+json|json/i, 'geojson'],
  [/csv|tab-separated/i, 'csv'],
  [/parquet/i, 'parquet'],
  [/arrow|feather/i, 'arrow']
];

function startsWith(bytes: Uint8Array, text: string): boolean {
  for (let index = 0; index < text.length; index++)
    if (bytes[index] !== text.charCodeAt(index)) return false;
  return true;
}

/** Magic bytes first, then file extension, then content type, then the first text character. */
export function sniffRemoteSourceFormat(
  name: string,
  contentType: string | null,
  head: Uint8Array
): RemoteSourceFormat {
  if (startsWith(head, 'PAR1')) return 'parquet';
  if (startsWith(head, 'ARROW1') || (head[0] === 0xff && head[1] === 0xff && head[2] === 0xff))
    return 'arrow';
  const extension = name.split(/[?#]/)[0].split('.').pop()?.toLowerCase() ?? '';
  const byExtension = EXTENSION_FORMATS[extension];
  if (byExtension && extension !== 'txt') return byExtension;
  const byContentType = CONTENT_TYPE_FORMATS.find(([pattern]) => pattern.test(contentType ?? ''));
  if (byContentType) return byContentType[1];
  const text = new TextDecoder().decode(head.subarray(0, 256)).trimStart();
  return text.startsWith('{') || text.startsWith('[') ? 'geojson' : 'csv';
}

/** Streams a response into one buffer, enforcing the size cap and reporting progress. */
export async function fetchRemoteBytes(
  url: string,
  options: RemoteLoadOptions = {}
): Promise<{bytes: Uint8Array; contentType: string | null}> {
  const maximumBytes = options.maximumBytes ?? REMOTE_SOURCE_MAXIMUM_BYTES;
  const response = await (options.fetchFunction ?? fetch)(url, {signal: options.signal});
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status} for ${url}`);
  const declared = Number(response.headers.get('content-length'));
  const totalBytes = Number.isFinite(declared) && declared > 0 ? declared : null;
  if (totalBytes !== null && totalBytes > maximumBytes)
    throw new Error(`Remote file is ${totalBytes} bytes, above the ${maximumBytes} byte limit.`);
  const contentType = response.headers.get('content-type');
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > maximumBytes) throw new Error('Remote file exceeds the size limit.');
    options.onProgress?.({loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength});
    return {bytes, contentType};
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  try {
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      loadedBytes += value.byteLength;
      if (loadedBytes > maximumBytes) throw new Error('Remote file exceeds the size limit.');
      chunks.push(value);
      options.onProgress?.({loadedBytes, totalBytes});
    }
  } catch (failure) {
    await reader.cancel().catch(() => undefined);
    throw failure;
  }
  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return {bytes, contentType};
}

function isMissingText(value: string | undefined): boolean {
  return value === undefined || value === '';
}

/** Column type inference from text: Bool, Int32, Float64, otherwise Utf8; empty cells are null. */
function makeColumnFromText(values: readonly (string | undefined)[]): arrow.Vector {
  let isBoolean = true;
  let isInteger = true;
  let isNumber = true;
  let nonMissing = 0;
  for (const value of values) {
    if (isMissingText(value)) continue;
    nonMissing++;
    const text = value!.trim();
    if (isBoolean && text !== 'true' && text !== 'false') isBoolean = false;
    if (isNumber) {
      const number = Number(text);
      if (text === '' || !Number.isFinite(number)) isNumber = false;
      else if (
        isInteger &&
        !(/^-?\d+$/.test(text) && number >= -(2 ** 31) && number <= 2 ** 31 - 1)
      )
        isInteger = false;
    }
  }
  if (nonMissing === 0)
    return arrow.vectorFromArray(
      values.map(() => null),
      new arrow.Utf8()
    );
  if (isBoolean)
    return arrow.vectorFromArray(
      values.map(value => (isMissingText(value) ? null : value!.trim() === 'true')),
      new arrow.Bool()
    );
  if (isNumber)
    return arrow.vectorFromArray(
      values.map(value => (isMissingText(value) ? null : Number(value))),
      isInteger ? new arrow.Int32() : new arrow.Float64()
    );
  return arrow.vectorFromArray(
    values.map(value => (isMissingText(value) ? null : value!)),
    new arrow.Utf8()
  );
}

function makeUniqueName(used: Set<string>, base: string): string {
  const root = base || 'column';
  let name = root;
  for (let suffix = 2; used.has(name); suffix++) name = `${root}_${suffix}`;
  used.add(name);
  return name;
}

export function detectDelimiter(text: string): string {
  const firstLine = text.slice(0, text.indexOf('\n') >= 0 ? text.indexOf('\n') : 4096);
  let best = ',';
  let bestCount = 0;
  for (const delimiter of [',', ';', '\t', '|']) {
    const count = firstLine.split(delimiter).length - 1;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return best;
}

export async function decodeCsvToArrowTable(bytes: Uint8Array): Promise<arrow.Table> {
  const {dsvFormat} = await import('d3-dsv');
  const text = new TextDecoder().decode(bytes).replace(/^﻿/, '');
  return makeArrowTableFromCsvRows(dsvFormat(detectDelimiter(text)).parseRows(text));
}

/** Shared by the whole-buffer and the incremental (resumable) CSV paths: header row plus data rows. */
export function makeArrowTableFromCsvRows(rows: readonly string[][]): arrow.Table {
  if (!rows.length) throw new Error('CSV has no header row');
  const used = new Set<string>();
  const names = rows[0].map((name, index) =>
    makeUniqueName(used, name.trim() || `column_${index}`)
  );
  const columns: Record<string, arrow.Vector> = {};
  names.forEach((name, columnIndex) => {
    columns[name] = makeColumnFromText(
      rows.slice(1).map(row => (columnIndex < row.length ? row[columnIndex] : undefined))
    );
  });
  return new arrow.Table(columns);
}

type JsonRecord = Record<string, unknown>;

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function makeColumnFromJsonValues(values: readonly unknown[]): arrow.Vector {
  let isBoolean = true;
  let isNumber = true;
  let nonMissing = 0;
  for (const value of values) {
    if (value === null || value === undefined) continue;
    nonMissing++;
    if (typeof value !== 'boolean') isBoolean = false;
    if (typeof value !== 'number' || !Number.isFinite(value)) isNumber = false;
  }
  const missing = (value: unknown) => value === null || value === undefined;
  if (nonMissing && isBoolean)
    return arrow.vectorFromArray(
      values.map(value => (missing(value) ? null : (value as boolean))),
      new arrow.Bool()
    );
  if (nonMissing && isNumber)
    return arrow.vectorFromArray(
      values.map(value => (missing(value) ? null : (value as number))),
      new arrow.Float64()
    );
  return arrow.vectorFromArray(
    values.map(value =>
      missing(value) ? null : typeof value === 'object' ? JSON.stringify(value) : String(value)
    ),
    new arrow.Utf8()
  );
}

/**
 * FeatureCollection, Feature, bare geometry or an array of plain records. Features produce a
 * `geometry` GeoJSON text column plus one column per property key (nested values become JSON).
 */
export function decodeGeoJsonToArrowTable(bytes: Uint8Array): arrow.Table {
  const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes));
  let features: JsonRecord[];
  let hasGeometry = true;
  if (Array.isArray(parsed)) {
    hasGeometry = false;
    features = parsed.filter(isJsonRecord).map(record => ({properties: record}));
  } else if (isJsonRecord(parsed) && parsed['type'] === 'FeatureCollection')
    features = (parsed['features'] as unknown[]).filter(isJsonRecord);
  else if (isJsonRecord(parsed) && parsed['type'] === 'Feature') features = [parsed];
  else if (isJsonRecord(parsed) && typeof parsed['type'] === 'string')
    features = [{type: 'Feature', geometry: parsed, properties: {}}];
  else throw new Error('JSON is not GeoJSON or an array of records');
  const used = new Set<string>();
  const geometryName = hasGeometry ? makeUniqueName(used, 'geometry') : '';
  const propertyKeys: string[] = [];
  const keyNames = new Map<string, string>();
  for (const feature of features) {
    const properties = isJsonRecord(feature['properties']) ? feature['properties'] : {};
    for (const key of Object.keys(properties))
      if (!keyNames.has(key)) {
        keyNames.set(key, makeUniqueName(used, key));
        propertyKeys.push(key);
      }
  }
  const columns: Record<string, arrow.Vector> = {};
  if (hasGeometry)
    columns[geometryName] = arrow.vectorFromArray(
      features.map(feature => (feature['geometry'] ? JSON.stringify(feature['geometry']) : null)),
      new arrow.Utf8()
    );
  for (const key of propertyKeys)
    columns[keyNames.get(key)!] = makeColumnFromJsonValues(
      features.map(feature =>
        isJsonRecord(feature['properties']) ? feature['properties'][key] : undefined
      )
    );
  return new arrow.Table(columns);
}

export async function decodeParquetToArrowTable(bytes: Uint8Array): Promise<arrow.Table> {
  const parquet = await import('parquet-wasm/esm');
  const wasmUrl = (await import('parquet-wasm/esm/parquet_wasm_bg.wasm?url')).default;
  await parquet.default(wasmUrl);
  const wasmTable = parquet.readParquet(bytes);
  return arrow.tableFromIPC(wasmTable.intoIPCStream());
}

/** GeoParquet "geo" file metadata names the WKB geometry columns. */
function readGeoParquetColumns(table: arrow.Table): string[] {
  const text = table.schema.metadata.get('geo');
  if (!text) return [];
  try {
    const geo = JSON.parse(text) as {columns?: Record<string, {encoding?: string}>};
    return Object.entries(geo.columns ?? {})
      .filter(([, column]) => !column.encoding || column.encoding === 'WKB')
      .map(([name]) => name);
  } catch {
    return [];
  }
}

export async function decodeRemoteBytesToArrowTable(
  bytes: Uint8Array,
  format: RemoteSourceFormat
): Promise<arrow.Table> {
  switch (format) {
    case 'csv':
      return decodeCsvToArrowTable(bytes);
    case 'geojson':
      return decodeGeoJsonToArrowTable(bytes);
    case 'parquet':
      return decodeParquetToArrowTable(bytes);
    case 'arrow':
      return arrow.tableFromIPC(bytes);
  }
}

const LATITUDE_NAME = /^(lat|latitude|y|lat_deg)$/i;
const LONGITUDE_NAME = /^(lon|lng|long|longitude|x|lon_deg)$/i;

function suggestCoordinates(table: arrow.Table): CoordinateRoles | undefined {
  const numeric = table.schema.fields.filter(
    field => arrow.DataType.isFloat(field.type) || arrow.DataType.isInt(field.type)
  );
  const latitude = numeric.find(field => LATITUDE_NAME.test(field.name));
  const longitude = numeric.find(field => LONGITUDE_NAME.test(field.name));
  return latitude && longitude ? {longitude: longitude.name, latitude: latitude.name} : undefined;
}

/** Decodes any supported bytes, routes unsupported Arrow types, and suggests map roles. */
export async function makeSourceCandidateFromBytes(
  bytes: Uint8Array,
  label: string,
  contentType: string | null = null,
  options: SourceTypeRoutingOptions = {}
): Promise<SourceCandidate> {
  const format = sniffRemoteSourceFormat(label, contentType, bytes);
  return makeSourceCandidateFromTable(
    await decodeRemoteBytesToArrowTable(bytes, format),
    label,
    format,
    options
  );
}

/** Routing and role suggestions for an already decoded table (shared with the resumable CSV path). */
export function makeSourceCandidateFromTable(
  decoded: arrow.Table,
  label: string,
  format: RemoteSourceFormat,
  options: SourceTypeRoutingOptions = {}
): SourceCandidate {
  // Explicit IPC preparation can retain Int64/Uint64 and let the analytics adapter enforce the
  // chosen Float32 error budget. The default import still keeps every digit as a category label.
  const routed =
    format === 'arrow' && options.preserveArrowTypes
      ? {table: decoded, routes: []}
      : routeArrowTable(decoded, {wkbGeometryColumns: readGeoParquetColumns(decoded)});
  const table = routed.table;
  const geometryName =
    format === 'geojson'
      ? table.schema.fields.find(field => field.name === 'geometry')?.name
      : routed.routes.find(route => route.kind === 'derived-geometry' && route.isDerived)?.name;
  const coordinates = geometryName ? undefined : suggestCoordinates(table);
  return {
    table,
    originalTable: decoded,
    label,
    isCurrent: () => true,
    ...(geometryName ? {suggestedGeometryColumn: geometryName} : {}),
    ...(coordinates ? {suggestedCoordinates: coordinates} : {})
  };
}

export function getLabelFromUrl(url: string): string {
  try {
    const path = new URL(url, 'http://localhost').pathname;
    return decodeURIComponent(path.split('/').filter(Boolean).pop() ?? url);
  } catch {
    return url;
  }
}

/** Downloads a URL and decodes it into a source candidate; nothing touches the GPU. */
export async function loadRemoteSource(
  url: string,
  options: RemoteLoadOptions = {}
): Promise<SourceCandidate> {
  const {bytes, contentType} = await fetchRemoteBytes(url, options);
  return makeSourceCandidateFromBytes(bytes, getLabelFromUrl(url), contentType);
}

/** Local file counterpart used by the file input; same decoders and routing as URLs. */
export async function loadLocalSource(
  file: File,
  options: SourceTypeRoutingOptions = {}
): Promise<SourceCandidate> {
  return makeSourceCandidateFromBytes(
    new Uint8Array(await file.arrayBuffer()),
    file.name,
    file.type || null,
    options
  );
}
