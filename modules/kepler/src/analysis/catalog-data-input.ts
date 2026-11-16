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
