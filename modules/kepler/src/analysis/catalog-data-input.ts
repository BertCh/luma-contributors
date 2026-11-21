// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import type {CatalogFeatureCollection, CatalogOutputTable} from './catalog-geojson';
import type {CatalogRasterInput} from './catalog-raster-input';

/** Scalar rows such as origin/destination H3 keys and weights, never synthetic geometry. */
export type CatalogTableInput = Readonly<{
  type: 'table';
  table: CatalogOutputTable;
  identityColumn?: string;
}>;

export type CatalogDataInput = CatalogFeatureCollection | CatalogTableInput | CatalogRasterInput;

export function isCatalogFeatureInput(input: CatalogDataInput): input is CatalogFeatureCollection {
  return input.type === 'FeatureCollection';
}

/** Shape validation only; role semantics and value checks belong to the consuming recipe. */
export function validateCatalogTableInput(
  id: string,
  input: CatalogTableInput,
  maximumRows: number
): void {
  const {table} = input;
  validateCapacity(id, 'maximumRows', maximumRows);
  validateCapacity(id, 'table.rowCount', table.rowCount, 0, maximumRows);
  validateRequirement(
    id,
    table.geometryColumn === undefined,
    'table.geometryColumn must be absent for a geometry-free input'
  );
  const columnNames = Object.keys(table.columns);
  validateRequirement(
    id,
    columnNames.length === Object.keys(table.columnTypes).length,
    'table.columns and table.columnTypes must name the same columns'
  );
  for (const name of columnNames) {
    validateRequirement(
      id,
      Object.hasOwn(table.columnTypes, name) &&
        ['string', 'float64', 'int32', 'boolean'].includes(table.columnTypes[name]),
      `table.columnTypes.${name} must declare a supported scalar type`
    );
    validateRequirement(
      id,
      table.columns[name].length === table.rowCount,
      `table.columns.${name} length must equal table.rowCount`
    );
  }
  validateRequirement(
    id,
    input.identityColumn === undefined || Object.hasOwn(table.columns, input.identityColumn),
    'identityColumn must name an existing table column'
  );
}

/** Preserves exact string identifiers and column arrays, including nulls. */
export function makeCatalogTableInput(
  table: CatalogOutputTable,
  options: {maximumRows: number; identityColumn?: string}
): CatalogTableInput {
  const input: CatalogTableInput = {
    type: 'table',
    table,
    ...(options.identityColumn !== undefined ? {identityColumn: options.identityColumn} : {})
  };
  validateCatalogTableInput(table.id, input, options.maximumRows);
  return input;
}

/** Only genuine feature collections participate in legacy coordinate/geometry checks. */
export function getCatalogFeatureInputs(
  inputs: Readonly<Record<string, CatalogDataInput | undefined>>
): Record<string, CatalogFeatureCollection> {
  return Object.fromEntries(
    Object.entries(inputs).filter((entry): entry is [string, CatalogFeatureCollection] =>
      Boolean(entry[1] && isCatalogFeatureInput(entry[1]))
    )
  );
}

/** Parse a bounded scalar-table snapshot without generating or renaming source columns. */
export function parseCatalogTableInput(text: string, maximumRows: number): CatalogTableInput {
  const id = 'catalog-table-input';
  const parsed: unknown = JSON.parse(text, (_name, value: unknown) => {
    if (
      typeof value === 'object' &&
      value !== null &&
      'catalogNumber' in value &&
      Object.keys(value).length === 1
    ) {
      if (value.catalogNumber === '-0') return -0;
      if (value.catalogNumber === 'NaN') return NaN;
      if (value.catalogNumber === 'Infinity') return Infinity;
      if (value.catalogNumber === '-Infinity') return -Infinity;
    }
    return value;
  });
  validateRequirement(
    id,
    typeof parsed === 'object' && parsed !== null,
    'input must be a table object'
  );
  const input = parsed as Record<string, unknown>;
  validateRequirement(
    id,
    input['type'] === 'table' && typeof input['table'] === 'object' && input['table'] !== null,
    'input.type must be table and input.table must be an object'
  );
  const raw = input['table'] as Record<string, unknown>;
  validateRequirement(
    id,
    typeof raw['id'] === 'string' &&
      typeof raw['label'] === 'string' &&
      typeof raw['rowCount'] === 'number',
    'table must declare id, label and numeric rowCount'
  );
  validateRequirement(
    id,
    raw['geometryColumn'] === undefined && raw['raster'] === undefined,
    'table must be geometry-free'
  );
  validateRequirement(
    id,
    typeof raw['columns'] === 'object' &&
      raw['columns'] !== null &&
      typeof raw['columnTypes'] === 'object' &&
      raw['columnTypes'] !== null,
    'table must declare columns and columnTypes'
  );
  const rawColumns = raw['columns'] as Record<string, unknown>;
  const rawTypes = raw['columnTypes'] as Record<string, unknown>;
  validateRequirement(
    id,
    Object.keys(rawColumns).length === Object.keys(rawTypes).length,
    'columns and columnTypes must name the same columns'
  );
  const table: CatalogOutputTable = {
    id: raw['id'],
    label: raw['label'],
    rowCount: raw['rowCount'],
    columns: Object.create(null),
    columnTypes: Object.create(null)
  };
  for (const [name, values] of Object.entries(rawColumns)) {
    validateRequirement(id, Array.isArray(values), `columns.${name} must be an array`);
    const type = rawTypes[name];
    validateRequirement(
      id,
      type === 'string' || type === 'float64' || type === 'int32' || type === 'boolean',
      `columnTypes.${name} must be a supported scalar type`
    );
    table.columnTypes[name] = type;
    Object.defineProperty(table.columns, name, {
      enumerable: true,
      configurable: true,
      writable: true,
      value: values.map((value: unknown) => {
        validateRequirement(
          id,
          value === null ||
            typeof value === 'string' ||
            typeof value === 'number' ||
            typeof value === 'boolean',
          `columns.${name} must contain scalar values`
        );
        return value;
      })
    });
  }
  validateRequirement(
    id,
    input['identityColumn'] === undefined || typeof input['identityColumn'] === 'string',
    'identityColumn must be a string'
  );
  return makeCatalogTableInput(table, {
    maximumRows,
    ...(typeof input['identityColumn'] === 'string'
      ? {identityColumn: input['identityColumn']}
      : {})
  });
}

/** Preserve nonfinite missing measurements and signed zero in scalar-table snapshots. */
export function serializeCatalogTableInput(input: CatalogTableInput): string {
  return JSON.stringify(input, (_name, value: unknown) => {
    if (typeof value !== 'number') return value;
    return !Number.isFinite(value) || Object.is(value, -0)
      ? {catalogNumber: Object.is(value, -0) ? '-0' : String(value)}
      : value;
  });
}
