// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {appendFileSync} from 'node:fs';
import {expect} from 'vitest';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, runCatalogOperation} from '../../src/analysis/catalog-bindings';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';

export {makeLattice, makePoints} from './curated-bindings-fixtures';

/** Runs a registered binding through the catalog entry point, on the CPU engine. */
export function runCuratedBinding(
  bindingId: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): CatalogRunResult {
  const binding = getCatalogBinding(bindingId);
  if (!binding) throw new Error(`runCuratedBinding: no registered binding ${bindingId}`);
  return runCatalogOperation(binding, parameters, inputs);
}

export function getResultTable(result: CatalogRunResult, tableId: string): CatalogOutputTable {
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table)
    throw new Error(
      `getResultTable: no table ${tableId}; has ${result.tables.map(item => item.id).join(', ')}`
    );
  return table;
}

export function getNumbers(table: CatalogOutputTable, column: string): number[] {
  return table.columns[column].map(value => (value === null ? Number.NaN : Number(value)));
}

/** Largest relative difference, `|a - b| / max(|b|, floor)`, over two lists of equal length. */
export function getMaximumRelativeError(
  actual: readonly number[],
  expected: readonly number[],
  floor = 1e-300
): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  actual.forEach((value, index) => {
    worst = Math.max(
      worst,
      Math.abs(value - expected[index]) / Math.max(Math.abs(expected[index]), floor)
    );
  });
  return worst;
}

/** Margins are appended to the file named by CURATED_MARGINS_FILE when it is set, as in curated-analyses.node.spec.ts. */
export function recordBindingMargin(name: string, value: number): void {
  const file = process.env['CURATED_MARGINS_FILE'];
  if (file) appendFileSync(file, `${name}: ${value}\n`);
}
