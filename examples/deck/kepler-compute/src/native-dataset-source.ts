// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {makeArrowColumnFromNativeField, type NativeColumnRoute} from './native-column-arrow';
import {timeToUnixMilli} from './native-time-converter';
import type {KeplerTable} from '@kepler.gl/table';

export type NativeDatasetSource = {
  table: arrow.Table;
  sourceId: string;
  sourceRevision: number;
  label: string;
  representation: 'borrowed-arrow' | 'materialized-native-rows';
  /** Arrow representation chosen for each materialized native field (empty when borrowed). */
  columnRoutes: Readonly<Record<string, NativeColumnRoute>>;
  isCurrent(): boolean;
};

/** Explicit ingestion only: borrow native Arrow batches, or materialize retained CPU rows once. */
export function makeArrowSourceFromNativeDataset(
  dataset: Pick<KeplerTable, 'id' | 'label' | 'dataRevision' | 'fields' | 'dataContainer'>,
  limits: {maximumRows: number; maximumCells: number}
): NativeDatasetSource {
  const container = dataset.dataContainer;
  const rowCount = container.numRows();
  const names = dataset.fields.map(field => field.name);
  const sourceRevision = dataset.dataRevision;
  if (
    !Number.isSafeInteger(limits.maximumRows) ||
    limits.maximumRows < 0 ||
    !Number.isSafeInteger(limits.maximumCells) ||
    limits.maximumCells < 0
  )
    throw new Error('Invalid native source materialization limits');
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 0 ||
    rowCount > limits.maximumRows ||
    rowCount * names.length > limits.maximumCells
  )
    throw new Error('Native source exceeds the explicit import capacity');
  if (new Set(names).size !== names.length)
    throw new Error('Native source field names must be unique');
  const nativeTable = container.getTable?.();
  const fieldNames = JSON.stringify(names);
  const isCurrent = () =>
    dataset.dataRevision === sourceRevision &&
    dataset.dataContainer === container &&
    container.numRows() === rowCount &&
    JSON.stringify(dataset.fields.map(field => field.name)) === fieldNames;
  if (isArrowTable(nativeTable)) {
    if (
      nativeTable.numRows !== rowCount ||
      JSON.stringify(nativeTable.schema.fields.map(field => field.name)) !== fieldNames
    )
      throw new Error('Native Arrow schema is not synchronized with its visible fields');
    return {
      table: nativeTable,
      sourceId: dataset.id,
      sourceRevision,
      label: dataset.label,
      representation: 'borrowed-arrow',
      columnRoutes: {},
      isCurrent
    };
  }
  const columns: Record<string, arrow.Vector> = Object.create(null);
  const columnRoutes: Record<string, NativeColumnRoute> = Object.create(null);
  for (const [columnIndex, field] of dataset.fields.entries()) {
    const values = Array.from({length: rowCount}, (_, rowIndex): unknown =>
      container.valueAt(rowIndex, columnIndex)
    );
    const {vector, route} = makeArrowColumnFromNativeField(field, values, timeToUnixMilli);
    columns[field.name] = vector;
    columnRoutes[field.name] = route;
  }
  const table = new arrow.Table(columns);
  if (!isCurrent()) throw new Error('Native source changed during materialization');
  return {
    table,
    sourceId: dataset.id,
    sourceRevision,
    label: dataset.label,
    representation: 'materialized-native-rows',
    columnRoutes,
    isCurrent
  };
}

/** Kepler and DuckDB can carry equivalent Arrow tables from another installed Arrow instance. */
function isArrowTable(value: unknown): value is arrow.Table {
  return Boolean(
    value &&
      typeof value === 'object' &&
      'batches' in value &&
      Array.isArray(value.batches) &&
      'schema' in value &&
      value.schema &&
      typeof value.schema === 'object' &&
      'fields' in value.schema &&
      Array.isArray(value.schema.fields) &&
      'getChild' in value &&
      typeof value.getChild === 'function'
  );
}
