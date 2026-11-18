// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  WorkflowError,
  hashWorkflowValue,
  makeWorkflowData,
  type CatalogOutputTable,
  type WorkflowData,
  type WorkflowDrawing,
  type WorkflowDisplayStyle,
  type WorkflowSqlRequest,
  type WorkflowDatasetProfile,
  type WorkflowDatasetRequest
} from '@luma-contributors/kepler';
import {analysisDatasetContext, guessAnalysisPointColumns} from '../analysis-dataset-context';
import {makeArrowTableFromCatalogTable} from '../analysis-catalog-table';
import {readOriginalColumnValues} from '../original-column-values';

/** Structural boundary shared by native Kepler tables and the Arrow readback adapter. */
export type NativeWorkflowDataset = {
  id: string;
  label: string;
  fields: readonly {name: string; type: string}[];
  dataContainer: {numRows(): number; valueAt(row: number, column: number): unknown};
  filteredIndex?: readonly number[];
  dataRevision?: number;
  /** Host-computed key of polygon-targeted layers and their geometry/coordinate roles. */
  selectionRevision?: string;
};

/** Invalid geometry is retained as null, never removed from the population. */
export type NativeWorkflowData = WorkflowData & {
  invalidGeometryCount: number;
  sourceColumnAliases?: Readonly<Record<string, string>>;
};

type Geometry = {type: string; coordinates?: unknown; geometries?: Geometry[]};

function readGeometry(value: unknown): Geometry | null {
  try {
    const parsed: unknown = typeof value === 'string' ? JSON.parse(value) : value;
    if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) return null;
    if (parsed.type === 'Feature')
      return readGeometry('geometry' in parsed ? parsed.geometry : null);
    const geometry = parsed as Geometry;
    if (geometry.type === 'GeometryCollection') {
      if (!Array.isArray(geometry.geometries)) return null;
      const geometries = geometry.geometries.map(readGeometry);
      return geometries.every(child => child !== null)
        ? {type: 'GeometryCollection', geometries: geometries as Geometry[]}
        : null;
    }
    const depths: Record<string, number> = {
      Point: 1,
      MultiPoint: 2,
      LineString: 2,
      MultiLineString: 3,
      Polygon: 3,
      MultiPolygon: 4
    };
    function checkCoordinates(coordinates: unknown, depth: number): boolean {
      return (
        Array.isArray(coordinates) &&
        (depth === 1
          ? coordinates.length >= 2 &&
            coordinates.every(
              coordinate => typeof coordinate === 'number' && Number.isFinite(coordinate)
            )
          : coordinates.every(child => checkCoordinates(child, depth - 1)))
      );
    }
    const depth = Object.hasOwn(depths, geometry.type) ? depths[geometry.type] : undefined;
    return depth && checkCoordinates(geometry.coordinates, depth) ? geometry : null;
  } catch {
    // Malformed source cells become null geometries, counted by the reader.
    return null;
  }
}

export function profileNativeDataset(
  dataset: NativeWorkflowDataset,
  resident: boolean
): WorkflowDatasetProfile {
  if (resident)
    // workbench-host caches this object by native dataset identity. Resolve shared facts when
    // read so later statistics, revision invalidation and removal remain visible through that cache.
    return {
      id: dataset.id,
      get label() {
        return analysisDatasetContext.get(dataset.id)?.profile.label ?? dataset.label;
      },
      get rowCount() {
        return (
          analysisDatasetContext.get(dataset.id)?.profile.rowCount ??
          dataset.dataContainer.numRows()
        );
      },
      resident,
      get fields() {
        return dataset.fields.map(field => ({...field}));
      },
      get geometryFields() {
        return [...(analysisDatasetContext.get(dataset.id)?.profile.geometryColumns ?? [])];
      },
      get pointColumns() {
        const pointColumns = analysisDatasetContext.get(dataset.id)?.profile.pointColumns;
        return pointColumns ? {...pointColumns} : undefined;
      }
    };
  const geometryFields = dataset.fields
    .filter((_field, column) => {
      let observed = 0;
      for (let row = 0; row < dataset.dataContainer.numRows() && observed < 16; row++) {
        const value = dataset.dataContainer.valueAt(row, column);
        if (value === null || value === undefined) continue;
        observed++;
        if (readGeometry(value)) return true;
      }
      return false;
    })
    .map(field => field.name);
  const pointColumns = guessAnalysisPointColumns(dataset.fields);
  return {
    id: dataset.id,
    label: dataset.label,
    rowCount: dataset.dataContainer.numRows(),
    resident,
    fields: dataset.fields.map(field => ({...field})),
    geometryFields,
    ...(pointColumns ? {pointColumns} : {})
  };
}

function getUniqueName(name: string, usedNames: Set<string>): string {
  let candidate = name;
  let suffix = 2;
  while (usedNames.has(candidate)) candidate = `${name}_${suffix++}`;
  usedNames.add(candidate);
  return candidate;
}

function validateRowCapacity(label: string, count: number, maximumRows: number): void {
  if (count > maximumRows)
    throw new WorkflowError(
      'capacity-exceeded',
      `${label}: ${count} rows exceed Maximum rows (${maximumRows}); filter the dataset or raise Maximum rows.`
    );
}

/** The workflow identity column is signed Int32, including after resident extraction. */
export function validateWorkflowRowIdentity(row: number, label: string): void {
  if (!Number.isInteger(row) || row < 0 || row > 2147483647)
    throw new WorkflowError(
      'capacity-exceeded',
      `${label}: Row index ${row} exceeds the Int32 identity limit (2147483647).`
    );
}

export function readNativeDatasetData(
  dataset: NativeWorkflowDataset,
  request: WorkflowDatasetRequest,
  rowIndices?: readonly number[]
): NativeWorkflowData {
  const rowCount = rowIndices?.length ?? dataset.dataContainer.numRows();
  validateRowCapacity(dataset.label, rowCount, request.maximumRows);
  const fieldNames = new Set<string>();
  for (const field of dataset.fields) {
    if (fieldNames.has(field.name))
      throw new WorkflowError(
        'dataset-read-failed',
        `${dataset.label}: Field name ${field.name} is duplicated; rename the fields to unique names before reading the dataset.`
      );
    fieldNames.add(field.name);
  }
  const geometry = request.geometry;
  function findField(name: string, label: string): number {
    const column = dataset.fields.findIndex(field => field.name === name);
    if (column < 0)
      throw new WorkflowError(
        'parameter-out-of-range',
        `${dataset.label}: ${label} must name an existing field; got ${name}.`
      );
    return column;
  }
  const geometryIndex =
    geometry.mode === 'geojson-field' ? findField(geometry.field, 'Geometry field') : -1;
  const longitudeIndex =
    geometry.mode === 'point-columns' ? findField(geometry.longitude, 'Longitude column') : -1;
  const latitudeIndex =
    geometry.mode === 'point-columns' ? findField(geometry.latitude, 'Latitude column') : -1;
  const rows = rowIndices ?? Array.from({length: rowCount}, (_, index) => index);
  for (const row of rows) {
    if (!Number.isInteger(row) || row < 0 || row >= dataset.dataContainer.numRows())
      throw new WorkflowError(
        'parameter-out-of-range',
        `${dataset.label}: Row index must be between 0 and ${dataset.dataContainer.numRows() - 1}; got ${row}.`
      );
    validateWorkflowRowIdentity(row, dataset.label);
  }
  const columns: CatalogOutputTable['columns'] = Object.create(null);
  const columnTypes: CatalogOutputTable['columnTypes'] = Object.create(null);
  const sourceColumnAliases: Record<string, string> = Object.create(null);
  if (geometry.mode === 'geojson-field') sourceColumnAliases[geometry.field] = 'geometry';
  const usedNames = new Set(dataset.fields.map(field => field.name));
  usedNames.add('row_index');
  if (geometry.mode !== 'none') usedNames.add('geometry');
  dataset.fields.forEach((field, column) => {
    if (column === geometryIndex) return;
    const name =
      field.name === 'row_index' || (field.name === 'geometry' && geometry.mode !== 'none')
        ? getUniqueName(`${field.name}_source`, usedNames)
        : field.name;
    sourceColumnAliases[field.name] = name;
    const values = rows.map(row => dataset.dataContainer.valueAt(row, column));
    const numeric = field.type === 'integer' || field.type === 'real';
    const type =
      field.type === 'integer' &&
      values.every(
        value =>
          value == null ||
          (typeof value === 'number' &&
            Number.isInteger(value) &&
            value >= -2147483648 &&
            value <= 2147483647)
      )
        ? 'int32'
        : numeric
          ? 'float64'
          : field.type === 'boolean'
            ? 'boolean'
            : 'string';
    columnTypes[name] = type;
    columns[name] = values.map(value => {
      if (value === null || value === undefined) return null;
      if (numeric) return typeof value === 'number' ? value : null;
      if (type === 'boolean') return typeof value === 'boolean' ? value : null;
      if (field.type === 'timestamp' && (typeof value === 'string' || typeof value === 'number'))
        return value;
      return typeof value === 'string'
        ? value
        : typeof value === 'object'
          ? JSON.stringify(value)
          : String(value);
    });
  });
  let invalidGeometryCount = 0;
  if (geometry.mode !== 'none') {
    columns['geometry'] = rows.map(row => {
      let value: Geometry | null;
      if (geometry.mode === 'geojson-field') {
        value = readGeometry(dataset.dataContainer.valueAt(row, geometryIndex));
      } else {
        const longitude = dataset.dataContainer.valueAt(row, longitudeIndex);
        const latitude = dataset.dataContainer.valueAt(row, latitudeIndex);
        value =
          typeof longitude === 'number' &&
          Number.isFinite(longitude) &&
          typeof latitude === 'number' &&
          Number.isFinite(latitude)
            ? {type: 'Point', coordinates: [longitude, latitude]}
            : null;
      }
      if (!value) invalidGeometryCount++;
      return value ? JSON.stringify(value) : null;
    });
    columnTypes['geometry'] = 'string';
  }
  columns['row_index'] = [...rows];
  columnTypes['row_index'] = 'int32';
  return {
    ...makeWorkflowData(
      {
        id: dataset.id,
        label: dataset.label,
        rowCount,
        columns,
        columnTypes,
        ...(geometry.mode === 'none' ? {} : {geometryColumn: 'geometry'})
      },
      {identityColumn: 'row_index'}
    ),
    invalidGeometryCount,
    ...(Object.entries(sourceColumnAliases).some(([source, converted]) => source !== converted)
      ? {sourceColumnAliases}
      : {})
  };
}

export function makeWorkflowDataFromArrowTable(
  table: arrow.Table,
  request: WorkflowDatasetRequest,
  label: string
): NativeWorkflowData {
  const vectors = table.schema.fields.map((_field, index) => table.getChildAt(index));
  const data = readNativeDatasetData(
    {
      id: request.datasetId,
      label,
      fields: table.schema.fields.map(field => ({
        name: field.name,
        type: arrow.DataType.isInt(field.type)
          ? 'integer'
          : arrow.DataType.isFloat(field.type)
            ? 'real'
            : arrow.DataType.isBool(field.type)
              ? 'boolean'
              : arrow.DataType.isTimestamp(field.type)
                ? 'timestamp'
                : 'string'
      })),
      dataContainer: {
        numRows: () => table.numRows,
        valueAt: (row, column) => {
          const value: unknown = vectors[column]?.get(row);
          if (typeof value !== 'bigint') return value;
          const converted = Number(value);
          if (!Number.isFinite(converted) || BigInt(converted) !== value)
            throw new WorkflowError(
              'dataset-read-failed',
              `${label}: Column ${table.schema.fields[column].name} at row ${row} cannot represent integer ${value} exactly as a JavaScript number.`
            );
          return converted;
        }
      }
    },
    request
  );
  // Extraction records original source-row identities in schema metadata, not compact row offsets.
  const extraction = table.schema.metadata.get('luma:row-extraction');
  if (extraction) {
    const metadata = JSON.parse(extraction) as {sourceRowColumn?: string};
    const identity = metadata.sourceRowColumn && table.getChild(metadata.sourceRowColumn);
    if (identity)
      data.table.columns['row_index'] = Array.from({length: table.numRows}, (_, row) => {
        const sourceRow = Number(identity.get(row));
        validateWorkflowRowIdentity(sourceRow, label);
        return sourceRow;
      });
  }
  return data;
}

/** Typed Arrow exchange preserving every column, including geometry text and special names. */
export function makeWorkflowArrowTable(table: CatalogOutputTable): arrow.Table {
  if (!Object.hasOwn(table.columnTypes, '__proto__')) return makeArrowTableFromCatalogTable(table);
  // The shared converter uses an ordinary object for vectors. Alias its keys, then restore the
  // original schema with Arrow's public constructors without assigning a __proto__ property.
  const names = Object.keys(table.columnTypes);
  const aliases = names.map((_name, index) => `workflow_column_${index}`);
  const converted = makeArrowTableFromCatalogTable({
    ...table,
    columns: Object.fromEntries(names.map((name, index) => [aliases[index], table.columns[name]])),
    columnTypes: Object.fromEntries(
      names.map((name, index) => [aliases[index], table.columnTypes[name]])
    )
  });
  const fields = converted.schema.fields.map((field, index) => field.clone({name: names[index]}));
  const schema = new arrow.Schema(fields, converted.schema.metadata);
  return new arrow.Table(
    schema,
    converted.batches.map(
      batch => new arrow.RecordBatch(schema, batch.data.clone(new arrow.Struct(fields)))
    )
  );
}

export function makeWorkflowPublication(
  data: WorkflowData,
  label: string
): {
  table: arrow.Table;
  label: string;
  geometryColumn?: string;
  coordinates?: {longitude: string; latitude: string};
} {
  const geometryColumn = data.table.geometryColumn;
  if (data.kind !== 'point' || !geometryColumn)
    return {
      table: makeWorkflowArrowTable(data.table),
      label,
      ...(data.kind !== 'table' && geometryColumn ? {geometryColumn} : {})
    };
  const columns = {...data.table.columns};
  const columnTypes = {...data.table.columnTypes};
  delete columns[geometryColumn];
  delete columnTypes[geometryColumn];
  const usedNames = new Set(Object.keys(columns));
  const longitude = getUniqueName('longitude', usedNames);
  const latitude = getUniqueName('latitude', usedNames);
  const positions = data.table.columns[geometryColumn].map(value => {
    const geometry = readGeometry(value);
    const coordinates = geometry?.coordinates;
    return geometry?.type === 'MultiPoint' && Array.isArray(coordinates)
      ? coordinates[0]
      : coordinates;
  });
  columns[longitude] = positions.map(position =>
    Array.isArray(position) ? (position[0] as number) : null
  );
  columns[latitude] = positions.map(position =>
    Array.isArray(position) ? (position[1] as number) : null
  );
  columnTypes[longitude] = 'float64';
  columnTypes[latitude] = 'float64';
  return {
    table: makeWorkflowArrowTable({...data.table, columns, columnTypes}),
    label,
    coordinates: {longitude, latitude}
  };
}

export function getNativeRevisionKey(
  dataset: NativeWorkflowDataset,
  filters: readonly unknown[],
  population: WorkflowDatasetRequest['population']
): string {
  const applicableFilters =
    population === 'selected'
      ? filters.filter(filter => {
          if (!filter || typeof filter !== 'object' || !('dataId' in filter)) return false;
          return Array.isArray(filter.dataId)
            ? filter.dataId.includes(dataset.id)
            : filter.dataId === dataset.id;
        })
      : [];
  return hashWorkflowValue([
    dataset.id,
    dataset.dataRevision,
    dataset.dataContainer.numRows(),
    dataset.fields.map(field => [field.name, field.type]),
    ...(population === 'selected'
      ? [applicableFilters, dataset.filteredIndex, dataset.selectionRevision]
      : [])
  ]);
}

/** Bounds of finite XY positions; tables and geometry without a finite position have no bounds. */
export function getWorkflowDataBounds(data: WorkflowData): [number, number, number, number] | null {
  const geometryColumn = data.table.geometryColumn;
  if (data.kind === 'table' || geometryColumn === undefined) return null;
  let bounds: [number, number, number, number] | null = null;
  function visitCoordinates(coordinates: unknown): void {
    if (!Array.isArray(coordinates)) return;
    if (typeof coordinates[0] === 'number' && typeof coordinates[1] === 'number') {
      const [longitude, latitude] = coordinates;
      if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
      if (!bounds) bounds = [longitude, latitude, longitude, latitude];
      else {
        bounds[0] = Math.min(bounds[0], longitude);
        bounds[1] = Math.min(bounds[1], latitude);
        bounds[2] = Math.max(bounds[2], longitude);
        bounds[3] = Math.max(bounds[3], latitude);
      }
      return;
    }
    for (const child of coordinates) visitCoordinates(child);
  }
  function visitGeometry(geometry: unknown): void {
    if (!geometry || typeof geometry !== 'object' || !('type' in geometry)) return;
    switch (geometry.type) {
      case 'Feature':
        if ('geometry' in geometry) visitGeometry(geometry.geometry);
        break;
      case 'GeometryCollection':
        if ('geometries' in geometry && Array.isArray(geometry.geometries))
          for (const child of geometry.geometries) visitGeometry(child);
        break;
      case 'Point':
      case 'MultiPoint':
      case 'LineString':
      case 'MultiLineString':
      case 'Polygon':
      case 'MultiPolygon':
        if ('coordinates' in geometry) visitCoordinates(geometry.coordinates);
        break;
    }
  }
  for (const value of data.table.columns[geometryColumn] ?? []) {
    if (typeof value !== 'string') continue;
    try {
      visitGeometry(JSON.parse(value));
    } catch {
      // A malformed cell has no usable coordinates; other geometry rows can still frame the map.
    }
  }
  return bounds;
}

// ColorBrewer palettes shipped by Kepler: sequential Blues, diverging RdBu, qualitative Set2.
export const WORKFLOW_DISPLAY_RANGE_NAMES = Object.freeze({
  sequential: 'Blues',
  diverging: 'RdBu',
  categorical: 'Set2'
});

/** Resolve Arrow scalar types to Kepler's colour-channel vocabulary. Geometry is checked by the host. */
export function chooseWorkflowDisplayChannel(
  schema: Pick<arrow.Schema, 'fields'>,
  style: WorkflowDisplayStyle | undefined
): {
  field: {name: string; type: 'integer' | 'real' | 'string' | 'boolean'};
  scale: 'quantile' | 'quantize' | 'ordinal';
  rangeName: string;
} | null {
  if (!style?.colorColumn) return null;
  const column = schema.fields.find(field => field.name === style.colorColumn);
  if (!column) return null;
  const columnType = arrow.DataType.isDictionary(column.type)
    ? column.type.dictionary
    : column.type;
  const type = arrow.DataType.isInt(columnType)
    ? 'integer'
    : arrow.DataType.isFloat(columnType)
      ? 'real'
      : arrow.DataType.isBool(columnType)
        ? 'boolean'
        : arrow.DataType.isUtf8(columnType)
          ? 'string'
          : null;
  if (!type) return null;
  const scale =
    type === 'string' || type === 'boolean' ? 'ordinal' : (style.colorScale ?? 'quantile');
  const palette = scale === 'ordinal' ? 'categorical' : (style.palette ?? 'sequential');
  return {
    field: {name: column.name, type},
    scale,
    rangeName: WORKFLOW_DISPLAY_RANGE_NAMES[palette]
  };
}

export type WorkflowDrawingFeature = {id?: unknown; properties?: unknown; geometry?: unknown};

type DrawingRow = WorkflowDrawing & {geometry: Geometry};

function readWorkflowDrawingRows(features: readonly WorkflowDrawingFeature[]): DrawingRow[] {
  const counts = new Map<string, number>();
  const identities = new Set<string>();
  const rows: DrawingRow[] = [];
  features.forEach((feature, index) => {
    const geometry = readGeometry(feature.geometry);
    const kind =
      geometry?.type === 'Point' || geometry?.type === 'MultiPoint'
        ? 'point'
        : geometry?.type === 'LineString' || geometry?.type === 'MultiLineString'
          ? 'line'
          : geometry?.type === 'Polygon' || geometry?.type === 'MultiPolygon'
            ? 'polygon'
            : null;
    if (!kind || !geometry) return;
    const properties =
      feature.properties && typeof feature.properties === 'object'
        ? (feature.properties as Record<string, unknown>)
        : {};
    const shape =
      kind === 'polygon' && properties['shape'] === 'Rectangle'
        ? 'Rectangle'
        : kind === 'polygon'
          ? 'Polygon'
          : kind === 'line'
            ? 'Line'
            : 'Point';
    const count = (counts.get(shape) ?? 0) + 1;
    counts.set(shape, count);
    const id =
      typeof feature.id === 'string' || typeof feature.id === 'number'
        ? String(feature.id)
        : `drawing-${index + 1}`;
    if (identities.has(id))
      throw new WorkflowError(
        'dataset-read-failed',
        `Drawn shapes: Shape id "${id}" must be unique.`
      );
    identities.add(id);
    const name = properties['name'];
    rows.push({
      id,
      kind,
      geometry,
      label: typeof name === 'string' && name.trim() ? name : `${shape} ${count}`
    });
  });
  return rows;
}

/** Shape labels and identities in editor order; named shapes still count in their kind's sequence. */
export function profileWorkflowDrawings(
  features: readonly WorkflowDrawingFeature[]
): WorkflowDrawing[] {
  return readWorkflowDrawingRows(features).map(({id, label, kind}) => ({id, label, kind}));
}

/** A snapshot of editor shapes; fallback ids use the original editor position. */
export function makeWorkflowDataFromDrawings(
  features: readonly WorkflowDrawingFeature[]
): WorkflowData {
  const rows = readWorkflowDrawingRows(features);
  return makeWorkflowData(
    {
      id: 'drawings',
      label: 'Drawn shapes',
      rowCount: rows.length,
      columns: {
        drawing_id: rows.map(row => row.id),
        label: rows.map(row => row.label),
        geometry: rows.map(row => JSON.stringify(row.geometry))
      },
      columnTypes: {drawing_id: 'string', label: 'string', geometry: 'string'},
      geometryColumn: 'geometry'
    },
    {identityColumn: 'drawing_id'}
  );
}

/**
 * SQL scalars use portable workflow types; wide integers retain exact text when numbers are unsafe.
 * Plain and dictionary timestamps become containing-millisecond ISO text within Date's range.
 */
export function makeWorkflowDataFromSqlTable(
  table: arrow.Table,
  request: WorkflowSqlRequest
): WorkflowData {
  validateRowCapacity('SQL', table.numRows, request.maximumRows);
  const fields = table.schema.fields.map((field, column) => {
    const vector = table.getChildAt(column);
    const type = field.type;
    let valueType = type;
    while (arrow.DataType.isDictionary(valueType)) valueType = valueType.dictionary;
    const timestamp = arrow.DataType.isTimestamp(valueType);
    const values: unknown[] = Array.from({length: table.numRows}, (_, row) => {
      if (!vector || !timestamp) return vector?.get(row);
      try {
        return readOriginalColumnValues(vector, [row])[0];
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw new WorkflowError(
          'operation-failed',
          `SQL: Column "${field.name}" contains a date outside the supported ISO date range.`
        );
      }
    });
    const wideInteger = arrow.DataType.isInt(type) && type.bitWidth === 64;
    const safeInteger =
      wideInteger && values.every(value => value == null || Number.isSafeInteger(Number(value)));
    const numeric = arrow.DataType.isFloat(type) || arrow.DataType.isDecimal(type) || safeInteger;
    return {
      name: field.name,
      type: numeric
        ? 'real'
        : arrow.DataType.isInt(type) && !wideInteger
          ? 'integer'
          : arrow.DataType.isBool(type)
            ? 'boolean'
            : 'string',
      values: values.map(value => {
        if (value == null) return null;
        if (arrow.DataType.isDecimal(type)) {
          // Arrow's public Decimal vector returns the unscaled signed integer as a BigNum.
          return Number(String(value)) * 10 ** -type.scale;
        }
        if (arrow.DataType.isDate(type) || timestamp) {
          const milliseconds = value instanceof Date ? value.getTime() : Number(value);
          if (!Number.isFinite(milliseconds) || Math.abs(milliseconds) > 8640000000000000)
            throw new WorkflowError(
              'operation-failed',
              `SQL: Column "${field.name}" contains a date outside the supported ISO date range.`
            );
          return new Date(milliseconds).toISOString();
        }
        if (wideInteger) return safeInteger ? Number(value) : String(value);
        if (numeric || arrow.DataType.isInt(type) || arrow.DataType.isBool(type)) return value;
        return String(value);
      })
    };
  });
  const fieldNames = new Set<string>();
  for (const field of fields) {
    if (fieldNames.has(field.name))
      throw new WorkflowError(
        'operation-failed',
        `SQL: Column "${field.name}" is duplicated; use unique aliases for the result columns.`
      );
    fieldNames.add(field.name);
  }
  // Native extraction reserves row_index. Temporary aliases let SQL retain every selected name.
  const converted = readNativeDatasetData(
    {
      id: 'workflow-sql',
      label: 'SQL',
      fields: fields.map((field, index) => ({...field, name: `sql_column_${index}`})),
      dataContainer: {
        numRows: () => table.numRows,
        valueAt: (row, column) => fields[column].values[row]
      }
    },
    {
      datasetId: 'workflow-sql',
      geometry: {mode: 'none'},
      population: 'all',
      maximumRows: request.maximumRows
    }
  );
  const identityColumn = getUniqueName('row_index', fieldNames);
  const columns = Object.fromEntries(
    fields.map((field, index) => [field.name, converted.table.columns[`sql_column_${index}`]])
  );
  const columnTypes = Object.fromEntries(
    fields.map((field, index) => [field.name, converted.table.columnTypes[`sql_column_${index}`]])
  );
  columns[identityColumn] = converted.table.columns['row_index'];
  columnTypes[identityColumn] = 'int32';
  const data = makeWorkflowData({...converted.table, columns, columnTypes}, {identityColumn});
  const names = new Set(
    Object.values(request.tables).flatMap(input =>
      input.table.geometryColumn === undefined ? [] : [input.table.geometryColumn]
    )
  );
  for (const field of fields) {
    if (!names.has(field.name)) continue;
    const geometries = field.values.map(value => (value == null ? null : readGeometry(value)));
    if (geometries.some((geometry, row) => field.values[row] != null && geometry === null))
      continue;
    data.table.columns[field.name] = geometries.map(geometry =>
      geometry ? JSON.stringify(geometry) : null
    );
    data.table.columnTypes[field.name] = 'string';
    return makeWorkflowData(
      {...data.table, geometryColumn: field.name},
      {identityColumn: data.identityColumn}
    );
  }
  return data;
}
