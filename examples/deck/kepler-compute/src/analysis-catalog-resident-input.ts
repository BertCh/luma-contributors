// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '@luma-contributors/kepler';
import type {ComputeRuntime} from './compute-runtime';
import type {ResidentTableField, ResidentTableProvider} from './resident-data-table';
import {
  MAXIMUM_CATALOG_NATIVE_ROWS,
  makeCatalogPointColumns,
  readCatalogFeaturesFromNativeDataset
} from './analysis-catalog-input';

/** Metadata and borrowed public column reader; never inspect the resident Kepler dataContainer. */
export type CatalogResidentDataset = {
  id: string;
  label: string;
  rowCount: number;
  geometryColumn?: string;
  provider: Pick<
    ResidentTableProvider,
    | 'fields'
    | 'originalFields'
    | 'readColumnValues'
    | 'getRevision'
    | 'readPage'
    | 'maximumPageSize'
  >;
  isCurrent(): boolean;
};

export function makeCatalogResidentDatasets(
  runtime: ComputeRuntime,
  getRuntime: () => ComputeRuntime | null
): CatalogResidentDataset[] {
  return [
    {
      id: runtime.explorer.dataset.id,
      label: 'Resident observations',
      rowCount: runtime.explorer.dataset.value.source.numRows,
      provider: runtime.nativeTable,
      isCurrent: () => getRuntime() === runtime && runtime.explorer.isCurrent
    },
    ...[...runtime.publications.values()].map(published => ({
      id: published.dataset.id,
      label: published.label,
      rowCount: published.dataset.value.source.numRows,
      geometryColumn: published.geometryColumn,
      provider: published.consumer.table,
      isCurrent: () =>
        getRuntime() === runtime &&
        runtime.publications.get(published.dataset.id) === published &&
        published.consumer.explorer.isCurrent
    }))
  ];
}

export function getCatalogResidentFields(
  dataset: CatalogResidentDataset
): readonly ResidentTableField[] {
  return dataset.provider.originalFields ?? dataset.provider.fields;
}

export function getCatalogResidentGeometryFields(dataset: CatalogResidentDataset): string[] {
  const fields = getCatalogResidentFields(dataset);
  return [
    ...(dataset.geometryColumn && fields.some(field => field.name === dataset.geometryColumn)
      ? [dataset.geometryColumn]
      : []),
    ...(fields.length >= 2 ? [makeCatalogPointColumns('', '')] : [])
  ];
}

export function getCatalogResidentDatasetLabel(dataset: CatalogResidentDataset): string {
  return `${dataset.label} (resident, ${dataset.rowCount.toLocaleString()} rows)`;
}

/** Complete source population in source-row order, bounded before any asynchronous column read. */
export async function readCatalogFeaturesFromResidentDataset(
  dataset: CatalogResidentDataset,
  geometryField: string,
  maximumRows = MAXIMUM_CATALOG_NATIVE_ROWS
): Promise<CatalogFeatureCollection> {
  const rowCount = dataset.rowCount;
  const limit = Math.min(maximumRows, MAXIMUM_CATALOG_NATIVE_ROWS);
  if (!Number.isSafeInteger(rowCount) || rowCount < 0 || rowCount > limit)
    throw new Error(`Resident dataset exceeds catalog capacity (${limit} rows)`);
  if (!dataset.provider.readColumnValues)
    throw new Error('Resident dataset has no original column readback');
  const revision = dataset.provider.getRevision();
  const requireCurrent = () => {
    if (!dataset.isCurrent() || dataset.provider.getRevision() !== revision)
      throw new Error('Resident dataset changed during catalog input readback');
  };
  requireCurrent();
  const fields = getCatalogResidentFields(dataset);
  const columns: unknown[][] = [];
  // Read sequentially: the provider owns revision-gated paging and lazy resident materialization.
  for (const field of fields) {
    if (field.derived && dataset.provider.originalFields) {
      // originalFields may rename a derived column after a source-name collision. The original
      // view owns that alias map; readColumnValues accepts only the underlying names.
      const values: unknown[] = [];
      if (
        !Number.isSafeInteger(dataset.provider.maximumPageSize) ||
        dataset.provider.maximumPageSize < 1
      )
        throw new Error('Resident provider has an invalid page size');
      for (let offset = 0; offset < rowCount; ) {
        const pageSize = Math.min(dataset.provider.maximumPageSize, rowCount - offset);
        const page = await dataset.provider.readPage({
          offset,
          limit: pageSize,
          population: 'all',
          columns: [field.name],
          view: 'original'
        });
        requireCurrent();
        if (
          !page ||
          page.totalCount !== rowCount ||
          page.sourceRowIndices.length !== pageSize ||
          page.columns[field.name]?.length !== pageSize
        )
          throw new Error('Resident column readback is incomplete or stale');
        for (let row = 0; row < pageSize; row++) {
          if (page.sourceRowIndices[row] !== offset + row)
            throw new Error('Resident original-view rows are not in source order');
          values.push(
            page.validity[field.name]?.[row] === 0 ? null : page.columns[field.name][row]
          );
        }
        offset += pageSize;
      }
      columns.push(values);
      continue;
    }
    const result = await dataset.provider.readColumnValues({column: field.name, population: 'all'});
    requireCurrent();
    if (!result || result.values.length !== rowCount)
      throw new Error('Resident column readback is incomplete or stale');
    columns.push(
      result.values.map(value => {
        if (typeof value !== 'bigint') return value;
        const numeric = Number(value);
        if (!Number.isSafeInteger(numeric))
          throw new Error(`Resident column ${field.name} exceeds exact catalog numeric range`);
        return numeric;
      })
    );
  }
  requireCurrent();
  return readCatalogFeaturesFromNativeDataset(
    {
      fields,
      dataContainer: {numRows: () => rowCount, valueAt: (row, column) => columns[column][row]}
    },
    geometryField
  );
}
