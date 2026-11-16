// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOriginalFieldType} from './original-column-values';
import type {CatalogFeatureCollection} from '@luma-contributors/kepler';
import type {ComputeRuntime} from './compute-runtime';
import type {PublishedSourceRuntime} from './source-publication';
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
  geometryKind?: 'point' | 'line' | 'polygon';
  /** Original schema and publication roles; inspecting these never reads a source cell. */
  fieldTypes?: Readonly<Record<string, string>>;
  coordinates?: {longitude: string; latitude: string};
  timeColumns?: readonly string[];
  valueColumn?: string;
  bounds?: readonly [number, number, number, number];
  revision?: number;
  /** Snapshot identity protects same-id, same-revision replacements. */
  owner?: object;
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
      ...getResidentProfileMetadata(runtime.explorer.dataset),
      coordinates: {longitude: 'longitude', latitude: 'latitude'},
      timeColumns: ['timestamp'],
      valueColumn: runtime.analysis?.settings.column ?? 'value',
      isCurrent: () => getRuntime() === runtime && runtime.explorer.isCurrent
    },
    ...[...runtime.publications.values()].map(published => ({
      id: published.dataset.id,
      label: published.label,
      rowCount: published.dataset.value.source.numRows,
      // Grouped trips retain observation rows, not GeoJSON in their trip-ID column.
      geometryColumn: published.tripOptions?.grouping ? undefined : published.geometryColumn,
      geometryKind: published.tripOptions?.grouping ? 'point' : getPublishedGeometryKind(published),
      ...getResidentProfileMetadata(published.dataset),
      coordinates:
        published.coordinates ??
        (published.tripOptions?.grouping
          ? {
              longitude: published.tripOptions.grouping.longitudeColumn,
              latitude: published.tripOptions.grouping.latitudeColumn
            }
          : undefined),
      timeColumns: Object.keys(published.dataset.value.prepared?.timeColumns ?? {}),
      valueColumn: published.analysis?.settings.column ?? published.coordinates?.flow?.count,
      bounds: published.bounds ?? undefined,
      provider: published.consumer.table,
      isCurrent: () =>
        getRuntime() === runtime &&
        runtime.publications.get(published.dataset.id) === published &&
        published.consumer.explorer.isCurrent
    }))
  ];
}

function getPublishedGeometryKind(published: PublishedSourceRuntime) {
  const families = published.geometry?.geometry.metadata.featureTypes;
  if (!families) return published.polygonSource ? ('polygon' as const) : undefined;
  const present = (['point', 'line', 'polygon'] as const).filter(kind => families[kind]);
  return present.length === 1 ? present[0] : undefined;
}

function getResidentProfileMetadata(dataset: ComputeRuntime['explorer']['dataset']) {
  return {
    revision: dataset.revision,
    owner: dataset,
    fieldTypes: Object.fromEntries(
      (dataset.value.source.schema?.fields ?? []).map(field => [
        field.name,
        getOriginalFieldType(field.type)
      ])
    )
  };
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
  const pointColumns =
    geometryField === makeCatalogPointColumns('', '') ? dataset.coordinates : undefined;
  const resolvedField = pointColumns
    ? makeCatalogPointColumns(pointColumns.longitude, pointColumns.latitude)
    : geometryField;
  const columns = await readCatalogColumnsFromResidentDataset(dataset, maximumRows);
  const fields = getCatalogResidentFields(dataset);
  return readCatalogFeaturesFromNativeDataset(
    {
      fields,
      dataContainer: {
        numRows: () => dataset.rowCount,
        valueAt: (row, column) => columns[column][row]
      }
    },
    resolvedField
  );
}

export const MAXIMUM_CATALOG_RESIDENT_READBACK_BYTES = 64 * 1024 * 1024;

/** Shared bounded original-column path for catalog execution and AX profiling. */
export async function readCatalogColumnsFromResidentDataset(
  dataset: CatalogResidentDataset,
  maximumRows = MAXIMUM_CATALOG_NATIVE_ROWS
): Promise<unknown[][]> {
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
  let readbackBytes = 0;
  const admitValues = (values: unknown[]) => {
    // JSON cells are strings in the original provider. Bound rich/list values as well.
    for (const value of values) {
      readbackBytes +=
        typeof value === 'string'
          ? value.length * 2
          : value && typeof value === 'object'
            ? JSON.stringify(value).length * 2
            : 8;
      if (readbackBytes > MAXIMUM_CATALOG_RESIDENT_READBACK_BYTES)
        throw new Error(
          `Resident original readback exceeds ${MAXIMUM_CATALOG_RESIDENT_READBACK_BYTES} bytes`
        );
    }
    columns.push(values);
  };
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
      admitValues(values);
      continue;
    }
    const result = await dataset.provider.readColumnValues({column: field.name, population: 'all'});
    requireCurrent();
    if (!result || result.values.length !== rowCount)
      throw new Error('Resident column readback is incomplete or stale');
    admitValues(
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
  return columns;
}
