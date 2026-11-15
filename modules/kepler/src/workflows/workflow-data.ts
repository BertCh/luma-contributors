// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  countFeatureVertices,
  type CatalogColumnType,
  type CatalogFeatureCollection,
  type CatalogGeometry,
  type CatalogOutputTable
} from '../analysis/catalog-geojson';
import {WorkflowError, type WorkflowData, type WorkflowPortKind} from './workflow-types';

export type WorkflowColumnSummary = {
  name: string;
  type: 'string' | 'float64' | 'int32' | 'boolean';
  rowCount: number;
  nullCount: number;
  /** Numeric columns: finite values only. */
  minimum?: number;
  maximum?: number;
  mean?: number;
  nonFiniteCount?: number;
  /** String and boolean columns: distinct values, counted up to `distinctLimit`. */
  distinctCount?: number;
  distinctLimitReached?: boolean;
  /** Most frequent values of a string or boolean column, at most eight. */
  topValues?: Array<{value: string; count: number}>;
};

// Workflow data is immutable after hand-off; derived features follow the object's lifetime.
const FEATURE_CACHE = new WeakMap<WorkflowData, CatalogFeatureCollection>();

// The contract has no column-order field. Retain first-seen order while the column object lives;
// copying the table and reusing its columns preserves this metadata, reconstructing columns does not.
const COLUMN_ORDER_CACHE = new WeakMap<CatalogOutputTable['columns'], readonly string[]>();

function readWorkflowColumnNames(table: CatalogOutputTable): readonly string[] {
  return COLUMN_ORDER_CACHE.get(table.columns) ?? Object.keys(table.columns);
}

// Only bare geometries with numeric coordinates qualify: nested or escaped keys use JSON.parse.
const SIMPLE_GEOMETRY_PREFIX =
  /^[ \t\r\n]*\{[ \t\r\n]*"type"[ \t\r\n]*:[ \t\r\n]*"(Point|MultiPoint|LineString|MultiLineString|Polygon|MultiPolygon)"[ \t\r\n]*,[ \t\r\n]*"coordinates"[ \t\r\n]*:[ \t\r\n]*/;

function hasNumericGeometrySuffix(text: string, start: number): boolean {
  // Sticky matching rejects gaps, illegal JSON whitespace and partial numeric tokens.
  const tokens = /[ \t\r\n]*(\[|\]|,|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/y;
  tokens.lastIndex = start;
  const states: Array<'first' | 'value' | 'separator'> = [];
  for (;;) {
    const match = tokens.exec(text);
    if (!match) return false;
    const token = match[1];
    const last = states.length - 1;
    const state = states[last];
    if (token === '[') {
      if (state === 'separator') return false;
      if (states.length > 0) states[last] = 'separator';
      states.push('first');
    } else if (token === ']') {
      if (states.length === 0 || state === 'value') return false;
      states.pop();
      if (states.length === 0) return /^[ \t\r\n]*\}[ \t\r\n]*$/.test(text.slice(tokens.lastIndex));
    } else if (token === ',') {
      if (state !== 'separator') return false;
      states[last] = 'value';
    } else {
      if (states.length === 0 || state === 'separator') return false;
      states[last] = 'separator';
    }
  }
}

function readGeometryType(value: string | number | boolean): unknown {
  if (typeof value !== 'string') return undefined;
  const match = SIMPLE_GEOMETRY_PREFIX.exec(value);
  if (match && hasNumericGeometrySuffix(value, match[0].length)) return match[1];
  try {
    const geometry: unknown = JSON.parse(value);
    return geometry && typeof geometry === 'object' && 'type' in geometry
      ? geometry.type
      : undefined;
  } catch {
    // Kind inference is conservative; feature conversion reports malformed geometry with its row.
    return undefined;
  }
}

/**
 * What the rows hold: one geometry kind, `features` for mixed or unknown kinds. A zero-row table
 * declaring a geometry column is `features`; absent geometry or nonempty all-null geometry is
 * `table`.
 */
export function getWorkflowDataKind(table: CatalogOutputTable): WorkflowPortKind {
  if (table.geometryColumn !== undefined && table.rowCount === 0) return 'features';
  const values =
    table.geometryColumn !== undefined && Object.hasOwn(table.columns, table.geometryColumn)
      ? table.columns[table.geometryColumn]
      : undefined;
  if (!values) return 'table';
  let kind: WorkflowPortKind = 'table';
  for (const value of values) {
    if (value === null) continue;
    const geometryType = readGeometryType(value);
    const nextKind =
      geometryType === 'Point' || geometryType === 'MultiPoint'
        ? 'point'
        : geometryType === 'LineString' || geometryType === 'MultiLineString'
          ? 'line'
          : geometryType === 'Polygon' || geometryType === 'MultiPolygon'
            ? 'polygon'
            : 'features';
    if (nextKind === 'features' || (kind !== 'table' && kind !== nextKind)) return 'features';
    kind = nextKind;
  }
  return kind;
}

/** Wraps a table; the kind is read from its geometry column. */
export function makeWorkflowData(
  table: CatalogOutputTable,
  options: {identityColumn?: string} = {}
): WorkflowData {
  return {table, kind: getWorkflowDataKind(table), ...options};
}

type ColumnValue = string | number | boolean | null;

function inferColumnType(values: readonly unknown[]): CatalogColumnType {
  const presentValues = values.filter(value => value !== null && value !== undefined);
  if (presentValues.length === 0) return 'string';
  if (presentValues.every(value => typeof value === 'number')) {
    return presentValues.every(
      value =>
        Number.isInteger(value) &&
        (value as number) >= -2147483648 &&
        (value as number) <= 2147483647
    )
      ? 'int32'
      : 'float64';
  }
  if (presentValues.every(value => typeof value === 'boolean')) return 'boolean';
  return 'string';
}

function makeColumnValue(value: unknown, type: CatalogColumnType): ColumnValue {
  if (value === null || value === undefined) return null;
  if (type !== 'string') return value as number | boolean;
  if (typeof value === 'string') return value;
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/**
 * Table form of a feature collection: one row per feature, a `geometry` GeoJSON-text column, a
 * `feature_id` string column when any feature has an id, and one column per scalar property
 * (declared type from the first non-null value; mixed columns become strings).
 */
export function makeWorkflowDataFromFeatures(
  collection: CatalogFeatureCollection,
  options: {id: string; label: string}
): WorkflowData {
  const propertyNames = new Set<string>();
  for (const feature of collection.features) {
    for (const name of Object.keys(feature.properties ?? {})) propertyNames.add(name);
  }
  const columns: CatalogOutputTable['columns'] = Object.create(null);
  const columnTypes: CatalogOutputTable['columnTypes'] = Object.create(null);
  const columnNames: string[] = [];
  for (const propertyName of propertyNames) {
    let columnName = propertyName;
    if (columnName === 'geometry' || columnName === 'feature_id') {
      do {
        columnName += '_property';
      } while (propertyNames.has(columnName) || Object.hasOwn(columns, columnName));
    }
    const values = collection.features.map(feature =>
      feature.properties && Object.hasOwn(feature.properties, propertyName)
        ? (feature.properties[propertyName] ?? null)
        : null
    );
    const type = inferColumnType(values);
    columns[columnName] = values.map(value => makeColumnValue(value, type));
    columnTypes[columnName] = type;
    columnNames.push(columnName);
  }
  columns['geometry'] = collection.features.map(feature =>
    feature.geometry === null ? null : JSON.stringify(feature.geometry)
  );
  columnTypes['geometry'] = 'string';
  columnNames.push('geometry');
  const hasIdentity = collection.features.some(
    feature => feature.id !== null && feature.id !== undefined
  );
  if (hasIdentity) {
    // Preserve numeric GeoJSON ids as values even though the identity column is declared string.
    columns['feature_id'] = collection.features.map(feature => feature.id ?? null);
    columnTypes['feature_id'] = 'string';
    columnNames.push('feature_id');
  }
  COLUMN_ORDER_CACHE.set(columns, columnNames);
  return makeWorkflowData(
    {
      ...options,
      rowCount: collection.features.length,
      columns,
      columnTypes,
      geometryColumn: 'geometry'
    },
    hasIdentity ? {identityColumn: 'feature_id'} : {}
  );
}

/**
 * Feature form of a table, memoised per data object. Feature id is the identity column value
 * (row index when there is none); every other non-geometry column becomes a property. Throws a
 * `WorkflowError('geometry-required')` when the table has no geometry column.
 */
export function getWorkflowFeatures(data: WorkflowData): CatalogFeatureCollection {
  const cached = FEATURE_CACHE.get(data);
  if (cached) return cached;
  const {table} = data;
  const geometryColumn = table.geometryColumn;
  const geometries =
    geometryColumn !== undefined && Object.hasOwn(table.columns, geometryColumn)
      ? table.columns[geometryColumn]
      : undefined;
  if (!geometries) {
    throw new WorkflowError(
      'geometry-required',
      `${table.label}: Geometry requires a geometry column.`
    );
  }
  const identity =
    data.identityColumn !== undefined && Object.hasOwn(table.columns, data.identityColumn)
      ? table.columns[data.identityColumn]
      : undefined;
  const propertyNames = readWorkflowColumnNames(table).filter(
    name => name !== geometryColumn && !(identity && name === data.identityColumn)
  );
  const features = Array.from({length: table.rowCount}, (_, index) => {
    let geometry: CatalogGeometry = null;
    const value = geometries[index];
    if (value !== null) {
      try {
        const parsed: unknown = typeof value === 'string' ? JSON.parse(value) : undefined;
        if (
          parsed !== null &&
          (typeof parsed !== 'object' ||
            !parsed ||
            !('type' in parsed) ||
            typeof parsed.type !== 'string')
        ) {
          throw new Error('Expected a bare GeoJSON geometry');
        }
        geometry = parsed as CatalogGeometry;
      } catch (cause) {
        throw new WorkflowError(
          'geometry-required',
          `${table.label}: Geometry column ${geometryColumn} at row ${index} must contain valid GeoJSON geometry text.`,
          {cause}
        );
      }
    }
    const properties = Object.fromEntries(
      propertyNames.map(name => [name, table.columns[name][index]])
    );
    return {
      type: 'Feature' as const,
      id: identity ? (identity[index] as string | number | null) : index,
      properties,
      geometry
    };
  });
  const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
  FEATURE_CACHE.set(data, collection);
  return collection;
}

export function countWorkflowVertices(data: WorkflowData): number {
  if (data.table.geometryColumn === undefined) return 0;
  function countGeometry(geometry: CatalogGeometry): number {
    if (
      geometry &&
      geometry.type === 'GeometryCollection' &&
      'geometries' in geometry &&
      Array.isArray(geometry.geometries)
    ) {
      return geometry.geometries.reduce(
        (count: number, child: CatalogGeometry) => count + countGeometry(child),
        0
      );
    }
    return countFeatureVertices({geometry});
  }
  return getWorkflowFeatures(data).features.reduce(
    (count, feature) => count + countGeometry(feature.geometry),
    0
  );
}

/** Column names without the geometry column, in table order. */
export function getWorkflowColumnNames(data: WorkflowData): string[] {
  return readWorkflowColumnNames(data.table).filter(name => name !== data.table.geometryColumn);
}

/** Exact binary accumulation avoids both intermediate overflow and loss of tiny residuals. */
class WorkflowNumericAccumulator {
  // Every finite double is an integer multiple of 2^-1074, including subnormal values.
  private units = 0n;
  private readonly buffer = new DataView(new ArrayBuffer(8));

  add(value: number): void {
    this.buffer.setFloat64(0, value);
    const bits = this.buffer.getBigUint64(0);
    const exponent = Number((bits >> 52n) & 0x7ffn);
    const fraction = bits & 0xfffffffffffffn;
    const significand = exponent === 0 ? fraction : fraction | 0x10000000000000n;
    const units = significand << BigInt(Math.max(0, exponent - 1));
    this.units += bits >> 63n ? -units : units;
  }

  mean(count: number): number {
    const negative = this.units < 0n;
    const magnitude = negative ? -this.units : this.units;
    const divisor = BigInt(count);
    const shift = Math.max(0, (magnitude / divisor).toString(2).length - 53);
    const scaledDivisor = divisor << BigInt(shift);
    let significand = magnitude / scaledDivisor;
    const remainder = magnitude % scaledDivisor;
    // Round the exact rational mean once, to nearest with ties to an even significand.
    if (
      remainder * 2n > scaledDivisor ||
      (remainder * 2n === scaledDivisor && significand % 2n === 1n)
    ) {
      significand++;
    }
    const mean = Number(significand) * 2 ** (shift - 1074);
    return negative ? -mean : mean;
  }
}

export function summarizeWorkflowColumn(
  data: WorkflowData,
  columnName: string,
  options: {distinctLimit?: number} = {}
): WorkflowColumnSummary {
  const {table} = data;
  if (!Object.hasOwn(table.columns, columnName)) {
    throw new WorkflowError(
      'unknown-parameter',
      `${table.label}: Column ${columnName} must name an existing column.`
    );
  }
  const values = table.columns[columnName];
  const type = table.columnTypes[columnName];
  const summary: WorkflowColumnSummary = {
    name: columnName,
    type,
    rowCount: table.rowCount,
    nullCount: 0
  };
  if (type === 'int32' || type === 'float64') {
    let count = 0;
    const accumulator = new WorkflowNumericAccumulator();
    summary.nonFiniteCount = 0;
    for (const value of values) {
      if (value === null) summary.nullCount++;
      else if (typeof value !== 'number' || !Number.isFinite(value)) summary.nonFiniteCount++;
      else {
        count++;
        summary.minimum = Math.min(summary.minimum ?? value, value);
        summary.maximum = Math.max(summary.maximum ?? value, value);
        accumulator.add(value);
      }
    }
    if (count > 0) summary.mean = accumulator.mean(count);
  } else {
    const distinctLimit = Math.max(0, Math.floor(options.distinctLimit ?? 1000));
    const counts = new Map<string, number>();
    for (const value of values) {
      if (value === null) summary.nullCount++;
      else {
        const text = String(value);
        counts.set(text, (counts.get(text) ?? 0) + 1);
      }
    }
    summary.distinctCount = Math.min(counts.size, distinctLimit);
    summary.distinctLimitReached = counts.size > distinctLimit;
    summary.topValues = [...counts]
      .sort((first, second) => second[1] - first[1])
      .slice(0, 8)
      .map(([value, count]) => ({value, count}));
  }
  return summary;
}

/** Four independently mixed 32-bit lanes; updates consume UTF-16 code units without buffering. */
class WorkflowHasher {
  private first = 0x9e3779b9;
  private second = 0x243f6a88;
  private third = 0xb7e15162;
  private fourth = 0xdeadbeef;
  private length = 0;

  update(text: string): void {
    for (let index = 0; index < text.length; index++) {
      const code = text.charCodeAt(index);
      this.first = Math.imul(this.first ^ code, 0x85ebca6b);
      this.second = Math.imul(this.second ^ code, 0xc2b2ae35);
      this.third = Math.imul(this.third ^ code, 0x27d4eb2f);
      this.fourth = Math.imul(this.fourth ^ code, 0x165667b1);
    }
    this.length += text.length;
  }

  finish(): string {
    return [this.first, this.second, this.third, this.fourth]
      .map(lane => {
        let value = lane ^ this.length;
        value = Math.imul(value ^ (value >>> 16), 0x85ebca6b);
        value = Math.imul(value ^ (value >>> 13), 0xc2b2ae35);
        return ((value ^ (value >>> 16)) >>> 0).toString(16).padStart(8, '0');
      })
      .join('');
  }
}

function hashCanonicalValue(hasher: WorkflowHasher, value: unknown): void {
  if (value === null) hasher.update('null;');
  else if (typeof value === 'string') hasher.update(`s${value.length}:${value}`);
  else if (typeof value === 'number')
    hasher.update(`n${Object.is(value, -0) ? '-0' : String(value)};`);
  else if (typeof value === 'boolean') hasher.update(value ? 'true;' : 'false;');
  else if (typeof value === 'undefined') hasher.update('undefined;');
  else if (Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    hasher.update('[');
    for (const element of value as Iterable<unknown>) hashCanonicalValue(hasher, element);
    hasher.update(']');
  } else if (typeof value === 'object') {
    hasher.update('{');
    for (const key of Object.keys(value).sort()) {
      const member = (value as Record<string, unknown>)[key];
      if (member === undefined) continue;
      hashCanonicalValue(hasher, key);
      hashCanonicalValue(hasher, member);
    }
    hasher.update('}');
  } else {
    throw new WorkflowError(
      'malformed',
      `Workflow value: ${typeof value} is not a supported JSON value.`
    );
  }
}

/** 128-bit non-cryptographic hash of UTF-16 code units, as 32 hexadecimal characters. */
export function hashWorkflowText(text: string): string {
  const hasher = new WorkflowHasher();
  hasher.update(text);
  return hasher.finish();
}

/** Hash of a JSON value in canonical form (object keys sorted, arrays in order). */
export function hashWorkflowValue(value: unknown): string {
  const hasher = new WorkflowHasher();
  hashCanonicalValue(hasher, value);
  return hasher.finish();
}

/** Content hash of a data object (column names, types and values, geometry column, kind). */
export function hashWorkflowData(data: WorkflowData): string {
  const hasher = new WorkflowHasher();
  hashCanonicalValue(hasher, [
    data.kind,
    data.identityColumn,
    data.table.geometryColumn,
    data.table.rowCount
  ]);
  for (const name of readWorkflowColumnNames(data.table)) {
    hashCanonicalValue(hasher, name);
    hashCanonicalValue(hasher, data.table.columnTypes[name]);
    hashCanonicalValue(hasher, data.table.columns[name]);
  }
  return hasher.finish();
}

/** GeoJSON FeatureCollection text of a data object that has geometry. */
export function makeWorkflowGeoJSONText(data: WorkflowData): string {
  return JSON.stringify(getWorkflowFeatures(data));
}

/** RFC 4180 CSV text; the geometry column is written as its GeoJSON text. */
export function makeWorkflowCSVText(data: WorkflowData): string {
  function guardFormula(value: string): string {
    return /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  }
  function quoteField(value: string): string {
    return /[,"\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
  }
  const names = readWorkflowColumnNames(data.table);
  const rows = [names.map(name => quoteField(guardFormula(name))).join(',')];
  for (let index = 0; index < data.table.rowCount; index++) {
    rows.push(
      names
        .map(name => {
          const value = data.table.columns[name][index];
          let text = value === null ? '' : String(value);
          if (
            data.table.columnTypes[name] === 'string' &&
            typeof value === 'string' &&
            /^[=+\-@\t\r]/.test(text)
          )
            text = guardFormula(text);
          return quoteField(text);
        })
        .join(',')
    );
  }
  return rows.join('\n');
}
