// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  convertWKBToGeometry,
  convertWKTToGeometry
} from '@loaders.gl/core/arrow-geometry/geometry-codecs';

/**
 * Type routing for the Arrow path. The GPU analytics adapter accepts Utf8/Dictionary<Utf8>,
 * Bool, integers of at most 32 bits, Float32, Float64 (under an explicit policy), Timestamp and
 * Date. Every other Arrow type either converts here to an adapter-supported equivalent without
 * losing the original (the original column stays in the table) or is kept as visible text that
 * is excluded from GPU upload with a stated reason and user action.
 */
export type SourceTypeRouteKind =
  | 'gpu-exact'
  | 'gpu-dictionary'
  | 'gpu-time'
  | 'retained-original'
  | 'derived-geometry'
  | 'converted';

/** How `makeRoutedArrowTable` rewrites the column; 'none' leaves it unchanged. */
export type SourceTypeConversion =
  | 'none'
  | 'decimal-text'
  | 'wide-integer-float64'
  | 'float32'
  | 'utf8'
  | 'json-text'
  | 'hex-text'
  | 'time-seconds'
  | 'duration-seconds'
  | 'wkt-geojson'
  | 'wkb-geojson';

export type SourceTypeRoute = {
  name: string;
  kind: SourceTypeRouteKind;
  sourceType: string;
  targetType: string;
  reason: string;
  userAction: string;
  conversion: SourceTypeConversion;
  /** Present for a WKT source column: the derived GeoJSON geometry text column. */
  derivedColumn?: string;
  /** True for a derived column added by the routing (it has no source column). */
  isDerived?: boolean;
};

export type SourceTypeRouteOptions = {
  /**
   * Int64/Uint64/Decimal default to exact decimal text (a category column). 'float64' is an
   * explicit opt-in that keeps numbers only when every value is an exact JavaScript integer.
   */
  wideIntegerPolicy?: 'decimal-category' | 'float64';
  /** Signed Int64 columns the caller declares as time are left for the adapter time path. */
  timeColumns?: Readonly<Record<string, unknown>>;
  /** Binary columns holding WKB geometry (for example GeoParquet "geo" metadata columns). */
  wkbGeometryColumns?: readonly string[];
};

export type RoutedArrowTable = {
  table: arrow.Table;
  routes: readonly SourceTypeRoute[];
};

export const SOURCE_ROUTE_METADATA_KEY = 'luma.sourceRoute';
export const SOURCE_TYPE_METADATA_KEY = 'luma.sourceType';

const WKT_PATTERN =
  /^\s*(?:SRID=\d+;)?\s*(?:POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*(?:Z|M|ZM)?\s*(?:\(|EMPTY)/i;
const WKT_SAMPLE_LIMIT = 32;

/** Column kinds that never reach scalar GPU normalization. */
export function isExcludedRoute(kind: SourceTypeRouteKind): boolean {
  return kind === 'retained-original' || kind === 'derived-geometry';
}

function isWideInteger(type: arrow.DataType): boolean {
  return arrow.DataType.isInt(type) && type.bitWidth === 64;
}

function isWktColumn(vector: arrow.Vector): boolean {
  let sampled = 0;
  for (let row = 0; row < vector.length && sampled < WKT_SAMPLE_LIMIT; row++) {
    const value: unknown = vector.get(row);
    if (value === null || value === undefined || value === '') continue;
    if (typeof value !== 'string' || !WKT_PATTERN.test(value)) return false;
    sampled++;
  }
  return sampled > 0;
}

function canWideIntegersBeNumbers(vector: arrow.Vector): boolean {
  for (let row = 0; row < vector.length; row++) {
    const value: unknown = vector.get(row);
    if (value === null || value === undefined) continue;
    const number = Number(value);
    if (!Number.isSafeInteger(number)) return false;
  }
  return true;
}

function makeUniqueColumnName(names: Set<string>, base: string): string {
  let candidate = base;
  for (let suffix = 2; names.has(candidate); suffix++) candidate = `${base}_${suffix}`;
  names.add(candidate);
  return candidate;
}

/** Decides a route for every column of an Arrow table, plus derived WKT geometry columns. */
export function makeSourceTypeRoutes(
  table: arrow.Table,
  options: SourceTypeRouteOptions = {}
): SourceTypeRoute[] {
  const routes: SourceTypeRoute[] = [];
  const usedNames = new Set(table.schema.fields.map(field => field.name));
  for (const field of table.schema.fields) {
    const type = field.type;
    const sourceType = String(type);
    const base = {name: field.name, sourceType};
    const vector = table.getChild(field.name)!;
    const route = (
      kind: SourceTypeRouteKind,
      targetType: string,
      reason: string,
      userAction: string,
      conversion: SourceTypeConversion = 'none',
      extra: Partial<SourceTypeRoute> = {}
    ): SourceTypeRoute => ({
      ...base,
      kind,
      targetType,
      reason,
      userAction,
      conversion,
      ...extra
    });

    const isTextCategory =
      arrow.DataType.isUtf8(type) ||
      (arrow.DataType.isDictionary(type) && arrow.DataType.isUtf8(type.dictionary));
    const existing = readFieldRoute(field);
    if (existing) {
      // Already routed: keep the earlier decision so routing is idempotent.
      routes.push({
        ...base,
        sourceType: existing.sourceType,
        kind: existing.kind,
        targetType: sourceType,
        reason: existing.route.slice(existing.kind.length + 2),
        userAction: 'None',
        conversion: 'none'
      });
    } else if (
      options.wkbGeometryColumns?.includes(field.name) &&
      (arrow.DataType.isBinary(type) || arrow.DataType.isLargeBinary(type))
    ) {
      const derivedColumn = makeUniqueColumnName(usedNames, `${field.name}_geojson`);
      routes.push(
        route(
          'derived-geometry',
          `utf8 GeoJSON in "${derivedColumn}"`,
          'Values are WKB geometry; GeoJSON geometry is derived with @loaders.gl WKB decoding',
          'Choose the derived GeoJSON column as the geometry column; the WKB bytes stay in the original table',
          'none',
          {derivedColumn}
        )
      );
      routes.push({
        name: derivedColumn,
        sourceType: 'utf8',
        kind: 'derived-geometry',
        targetType: 'utf8 GeoJSON geometry',
        reason: `Derived from WKB column "${field.name}"; rows that fail to decode are null`,
        userAction: 'Use as the geometry column',
        conversion: 'wkb-geojson',
        isDerived: true
      });
    } else if (isTextCategory && isWktColumn(vector)) {
      const derivedColumn = makeUniqueColumnName(usedNames, `${field.name}_geojson`);
      routes.push(
        route(
          'derived-geometry',
          `utf8 GeoJSON in "${derivedColumn}"`,
          'Values are WKT text; GeoJSON geometry is derived with @loaders.gl WKT parsing',
          'Choose the derived GeoJSON column as the geometry column; the WKT text stays in the original table',
          'none',
          {derivedColumn}
        )
      );
      routes.push({
        name: derivedColumn,
        sourceType: 'utf8',
        kind: 'derived-geometry',
        targetType: 'utf8 GeoJSON geometry',
        reason: `Derived from WKT column "${field.name}"; rows that fail to parse are null`,
        userAction: 'Use as the geometry column',
        conversion: 'wkt-geojson',
        isDerived: true
      });
    } else if (isTextCategory) {
      routes.push(
        route(
          'gpu-dictionary',
          'dictionary<utf8>',
          'Stable category codes; labels append across batches',
          'None'
        )
      );
    } else if (arrow.DataType.isDictionary(type)) {
      routes.push(
        route(
          'converted',
          'utf8 category',
          `Dictionary values of type ${type.dictionary} are rendered as category labels`,
          'None; the original dictionary column is replaced by its labels',
          'utf8'
        )
      );
    } else if (arrow.DataType.isLargeUtf8(type)) {
      routes.push(
        route(
          'converted',
          'utf8',
          'LargeUtf8 is copied to Utf8 (offsets must stay below 2 GiB)',
          'None',
          'utf8'
        )
      );
    } else if (arrow.DataType.isBool(type)) {
      routes.push(
        route(
          'gpu-exact',
          'uint32',
          field.nullable
            ? 'Boolean widens to Uint32 0/1; nulls are preserved as a validity mask, not as false'
            : 'Boolean widens to Uint32 0/1',
          'None'
        )
      );
    } else if (isWideInteger(type)) {
      if (options.timeColumns?.[field.name] && type.isSigned)
        routes.push(
          route(
            'gpu-time',
            'int64 time words',
            'Declared time unit; lossless low/high words',
            'None'
          )
        );
      else if (options.wideIntegerPolicy === 'float64' && canWideIntegersBeNumbers(vector))
        routes.push(
          route(
            'converted',
            'float64',
            'Opt-in numeric policy: every value is an exact integer below 2^53',
            'Float64 then follows the precision policy',
            'wide-integer-float64'
          )
        );
      else
        routes.push(
          route(
            'converted',
            'utf8 decimal category',
            options.wideIntegerPolicy === 'float64'
              ? 'Some values exceed 2^53, so numeric conversion would round; exact decimal text kept'
              : 'Every digit is kept as an exact decimal category label (no Float32/Float64 rounding)',
            'Opt in to wideIntegerPolicy "float64" for numeric use, or declare the column as time',
            'decimal-text'
          )
        );
    } else if (arrow.DataType.isInt(type)) {
      routes.push(
        route(
          'gpu-exact',
          type.isSigned ? 'int32' : 'uint32',
          'Narrow integers widen exactly',
          'None'
        )
      );
    } else if (arrow.DataType.isFloat(type)) {
      if (type.precision === arrow.Precision.HALF)
        routes.push(
          route('converted', 'float32', 'Float16 widens exactly to Float32', 'None', 'float32')
        );
      else if (type.precision === arrow.Precision.SINGLE)
        routes.push(route('gpu-exact', 'float32', 'Stored as Float32', 'None'));
      else
        routes.push(
          route(
            'converted',
            'float32 (policy)',
            'Float64 rounds to Float32 under the explicit, measured precision policy',
            'Change the Float64 policy to exact to reject rounding',
            'none'
          )
        );
    } else if (arrow.DataType.isDecimal(type)) {
      routes.push(
        options.wideIntegerPolicy === 'float64'
          ? route(
              'converted',
              'float64',
              `Decimal(${type.precision},${type.scale}) converted to Float64 by explicit opt-in; may round`,
              'Float64 then follows the precision policy',
              'wide-integer-float64'
            )
          : route(
              'converted',
              'utf8 decimal category',
              `Decimal(${type.precision},${type.scale}) kept as exact decimal text`,
              'Opt in to wideIntegerPolicy "float64" for approximate numeric use',
              'decimal-text'
            )
      );
    } else if (arrow.DataType.isTimestamp(type) || arrow.DataType.isDate(type)) {
      routes.push(
        route(
          'gpu-time',
          'int64 time words',
          arrow.DataType.isDate(type)
            ? 'Date32 days and Date64 milliseconds are normalized to exact millisecond words'
            : 'Exact lossless time words in the declared unit',
          'None'
        )
      );
    } else if (arrow.DataType.isTime(type)) {
      routes.push(
        route(
          'converted',
          'float64 seconds',
          'Time of day converted to seconds since midnight; the original column is kept',
          'Float64 follows the precision policy',
          'time-seconds'
        )
      );
    } else if (arrow.DataType.isDuration(type)) {
      routes.push(
        route(
          'converted',
          'float64 seconds',
          'Duration converted to seconds; the original column is kept',
          'Float64 follows the precision policy',
          'duration-seconds'
        )
      );
    } else if (arrow.DataType.isNull(type)) {
      routes.push(
        route('retained-original', 'none', 'Null type has no values to upload', 'None', 'none')
      );
    } else if (
      arrow.DataType.isBinary(type) ||
      arrow.DataType.isLargeBinary(type) ||
      arrow.DataType.isFixedSizeBinary(type)
    ) {
      routes.push(
        route(
          'retained-original',
          'utf8 hex text',
          'Binary values are shown as hex text and are not uploaded to the GPU',
          'Decode the bytes (for example WKB geometry) before loading',
          'hex-text'
        )
      );
    } else {
      // List, FixedSizeList, Struct, Map, Union, Interval and any future type.
      routes.push(
        route(
          'retained-original',
          'utf8 JSON text',
          `${sourceType} is kept as JSON text and is not uploaded to the GPU`,
          'Flatten the nested field into scalar columns before loading',
          'json-text'
        )
      );
    }
  }
  return routes;
}

function jsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === 'bigint') return value.toString();
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray)
    return Array.from(value, byte => byte.toString(16).padStart(2, '0')).join('');
  return value;
}

function formatJsonText(value: unknown): string {
  return JSON.stringify(value, jsonReplacer) ?? 'null';
}

function formatHexText(value: unknown): string {
  return value instanceof Uint8Array ? (jsonReplacer('', value) as string) : formatJsonText(value);
}

/** Exact decimal text of an unscaled integer string with a decimal scale. */
export function formatScaledDecimal(unscaled: string, scale: number): string {
  if (scale <= 0) return scale === 0 ? unscaled : `${unscaled}${'0'.repeat(-scale)}`;
  const negative = unscaled.startsWith('-');
  const digits = (negative ? unscaled.slice(1) : unscaled).padStart(scale + 1, '0');
  return `${negative ? '-' : ''}${digits.slice(0, -scale)}.${digits.slice(-scale)}`;
}

function getTimeUnitDivisor(unit: arrow.TimeUnit): number {
  return [1, 1e3, 1e6, 1e9][unit] ?? 1;
}

function formatWktAsGeoJson(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const geometry = convertWKTToGeometry(value, {wkt: {shape: 'geojson-geometry', crs: false}});
    return geometry ? JSON.stringify(geometry) : null;
  } catch {
    return null;
  }
}

function getConvertedType(conversion: SourceTypeConversion): arrow.DataType {
  switch (conversion) {
    case 'float32':
      return new arrow.Float32();
    case 'wide-integer-float64':
    case 'time-seconds':
    case 'duration-seconds':
      return new arrow.Float64();
    default:
      return new arrow.Utf8();
  }
}

function formatWkbAsGeoJson(value: unknown): string | null {
  if (!(value instanceof Uint8Array)) return null;
  try {
    const geometry = convertWKBToGeometry(value);
    return geometry ? JSON.stringify(geometry) : null;
  } catch {
    return null;
  }
}

function makeConvertedValue(
  conversion: SourceTypeConversion,
  value: unknown,
  type: arrow.DataType
): unknown {
  switch (conversion) {
    case 'decimal-text':
      return arrow.DataType.isDecimal(type)
        ? formatScaledDecimal(String(value), type.scale)
        : String(value);
    case 'wide-integer-float64':
      return arrow.DataType.isDecimal(type)
        ? Number(formatScaledDecimal(String(value), type.scale))
        : Number(value);
    case 'utf8':
      return String(value);
    case 'json-text':
      return formatJsonText(value);
    case 'hex-text':
      return formatHexText(value);
    case 'time-seconds':
      return Number(value) / getTimeUnitDivisor((type as arrow.Time).unit);
    case 'duration-seconds':
      return Number(value) / getTimeUnitDivisor((type as arrow.Duration).unit);
    case 'wkt-geojson':
      return formatWktAsGeoJson(value);
    case 'wkb-geojson':
      return formatWkbAsGeoJson(value);
    default:
      return value;
  }
}

function makeConvertedData(
  conversion: SourceTypeConversion,
  vector: arrow.Vector,
  type: arrow.DataType
): arrow.Data {
  const values: unknown[] = new Array(vector.length);
  for (let row = 0; row < vector.length; row++) {
    const value: unknown = vector.get(row);
    values[row] =
      value === null || value === undefined ? null : makeConvertedValue(conversion, value, type);
  }
  return arrow.vectorFromArray(values, getConvertedType(conversion)).data[0];
}

function makeRouteMetadata(
  field: arrow.Field,
  route: SourceTypeRoute | undefined,
  originalType: string
): Map<string, string> {
  const metadata = new Map(field.metadata);
  if (route) {
    metadata.set(SOURCE_ROUTE_METADATA_KEY, `${route.kind}: ${route.reason}`);
    metadata.set(SOURCE_TYPE_METADATA_KEY, originalType);
    metadata.set('luma.sourceRouteKind', route.kind);
  }
  return metadata;
}

/**
 * Applies routes without losing any original column. Converted columns replace the unsupported
 * column in place (same name, adapter-supported type); retained-original columns become JSON/hex
 * text in place; derived GeoJSON geometry columns are appended. Every column is annotated with
 * route and original type field metadata so previews can show the decision. Batch boundaries
 * are preserved.
 */
export function makeRoutedArrowTable(
  table: arrow.Table,
  routes: readonly SourceTypeRoute[]
): RoutedArrowTable {
  const routeByName = new Map(routes.map(route => [route.name, route]));
  const outputFields: arrow.Field[] = [];
  const outputChunks: arrow.Data[][] = [];
  const batchCount = table.batches.length;
  const addColumn = (
    name: string,
    type: arrow.DataType,
    nullable: boolean,
    metadata: Map<string, string>,
    chunks: arrow.Data[]
  ) => {
    outputFields.push(new arrow.Field(name, type, nullable, metadata));
    outputChunks.push(chunks);
  };
  table.schema.fields.forEach((field, fieldIndex) => {
    const route = routeByName.get(field.name);
    const originalType = String(field.type);
    const sourceVectors = table.batches.map(batch => batch.getChildAt(fieldIndex)!);
    const addConverted = (
      conversionRoute: SourceTypeRoute,
      name: string,
      routeForMetadata: SourceTypeRoute | undefined
    ) => {
      const type = getConvertedType(conversionRoute.conversion);
      addColumn(
        name,
        type,
        true,
        makeRouteMetadata(new arrow.Field(name, type, true), routeForMetadata, originalType),
        sourceVectors.map(vector =>
          makeConvertedData(conversionRoute.conversion, vector, field.type)
        )
      );
    };
    if (
      route &&
      route.conversion !== 'none' &&
      route.conversion !== 'wkt-geojson' &&
      route.conversion !== 'wkb-geojson'
    )
      addConverted(route, field.name, route);
    else
      addColumn(
        field.name,
        field.type,
        field.nullable,
        makeRouteMetadata(field, route, originalType),
        sourceVectors.map(vector => vector.data[0])
      );
    if (route?.derivedColumn) {
      const derivedRoute = routeByName.get(route.derivedColumn)!;
      addConverted(derivedRoute, route.derivedColumn, derivedRoute);
    }
  });
  const schema = new arrow.Schema(outputFields, new Map(table.schema.metadata));
  const batches = Array.from(
    {length: batchCount},
    (_, batchIndex) =>
      new arrow.RecordBatch(
        schema,
        arrow.makeData({
          type: new arrow.Struct(outputFields),
          length: table.batches[batchIndex].numRows,
          children: outputChunks.map(chunks => chunks[batchIndex])
        })
      )
  );
  return {table: new arrow.Table(schema, batches), routes};
}

/** Reads the route annotation written by `makeRoutedArrowTable`, if any. */
export function readFieldRoute(
  field: arrow.Field
): {route: string; kind: SourceTypeRouteKind; sourceType: string} | undefined {
  const route = field.metadata.get(SOURCE_ROUTE_METADATA_KEY);
  const kind = field.metadata.get('luma.sourceRouteKind') as SourceTypeRouteKind | undefined;
  if (!route || !kind) return undefined;
  return {
    route,
    kind,
    sourceType: field.metadata.get(SOURCE_TYPE_METADATA_KEY) ?? String(field.type)
  };
}

/** Routes and rewrites in one step. */
export function routeArrowTable(
  table: arrow.Table,
  options: SourceTypeRouteOptions = {}
): RoutedArrowTable {
  return makeRoutedArrowTable(table, makeSourceTypeRoutes(table, options));
}
