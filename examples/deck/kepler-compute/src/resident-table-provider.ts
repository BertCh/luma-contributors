// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUKeplerTableInspector} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {
  formatOriginalTableCell,
  MAXIMUM_ORIGINAL_TABLE_CELLS,
  MAXIMUM_ORIGINAL_TABLE_CHARACTERS
} from './original-table-values';
export {formatOriginalTableValue} from './original-table-values';
import {
  createColumnStatisticsReader,
  type ResidentGPUColumnStatistics
} from './resident-column-statistics';
import {createOriginalRichSorter, isOriginalSortType} from './original-table-rich-sort';
import {ResidentTableGPUSorter} from './resident-table-gpu-sorter';
import {classifyTableSortType} from './resident-table-sort-keys';
import {ResidentTableWideStatistics} from './resident-table-wide-statistics';
import type {ComputeRuntime} from './compute-runtime';
import type {
  ResidentTableField,
  ResidentTablePage,
  ResidentTableProvider
} from './resident-data-table';

/** Own alongside the runtime; destroy before its analysis/explorer borrowed buffers. */
export function createResidentTableProvider(runtime: ComputeRuntime): ResidentTableProvider & {
  destroy(): Promise<void>;
} {
  const inspector = new GPUKeplerTableInspector({
    session: runtime.session,
    explorer: runtime.explorer,
    sortMemoryBudgetBytes: 64 * 1024 * 1024,
    getDerivedColumns: () => runtime.analysis.getTableColumns()
  });
  const provider: ResidentTableProvider = {
    datasetId: runtime.explorer.dataset.id,
    sourceRevision: runtime.explorer.dataset.revision,
    maximumPageSize: inspector.maximumPageSize,
    get fields() {
      return inspector.columns.map(name => ({
        name,
        label:
          name === 'analysis_z_score'
            ? 'Analysis: z-score'
            : name === 'analysis_class'
              ? 'Analysis: class'
              : name,
        sortable: true,
        derived: Boolean(runtime.analysis.getTableColumns()?.columns[name])
      }));
    },
    getRevision: () =>
      `${runtime.explorer.currentRevision}:${runtime.analysis.getTableColumns()?.revision ?? 'off'}`,
    subscribe(listener) {
      const unsubscribeSelection = runtime.subscribe(listener);
      const unsubscribeAnalysis = runtime.analysis.subscribe(listener);
      const unsubscribeInvalidation = runtime.subscribeInvalidation(listener);
      return () => {
        unsubscribeSelection();
        unsubscribeAnalysis();
        unsubscribeInvalidation();
      };
    },
    readPage: request => inspector.readPage(request),
    readRows: (sourceRows, columns) => inspector.readRows(sourceRows, columns)
  };
  const gpuSorter = new ResidentTableGPUSorter({
    session: runtime.session,
    explorer: runtime.explorer,
    memoryBudgetBytes: 512 * 1024 * 1024
  });
  const wideStatistics = new ResidentTableWideStatistics({
    session: runtime.session,
    explorer: runtime.explorer,
    memoryBudgetBytes: 512 * 1024 * 1024
  });
  const gpuStatistics: ResidentGPUColumnStatistics = {
    plan: column => inspector.planColumnStatistics(column),
    read: request => inspector.readColumnStatistics(request)
  };
  return Object.assign(
    withOriginalSourceTable(
      provider,
      runtime.explorer.dataset.value.source,
      () => runtime.explorer.isCurrent,
      runtime.explorer.dataset.value.resident,
      gpuStatistics,
      gpuSorter,
      wideStatistics
    ),
    {
      destroy: async () => (
        await gpuSorter.destroy(), await wideStatistics.destroy(), inspector.destroy()
      )
    }
  );
}

/** Float32, Int32 and Uint32 sort as prepared GPU scalars with no row bound (see chooseTableSortRoute). */
function isGPUScalarSortType(type: arrow.DataType): boolean {
  return (
    (arrow.DataType.isFloat(type) && type.precision === arrow.Precision.SINGLE) ||
    (arrow.DataType.isInt(type) && type.bitWidth === 32)
  );
}

/** Retained Arrow cells are read only for the bounded page of GPU-selected source row IDs. */
export function withOriginalSourceTable(
  provider: ResidentTableProvider,
  source: arrow.Table,
  isCurrent: () => boolean,
  resident?: {ensureRows(rows: readonly number[]): Promise<void>},
  gpuStatistics?: ResidentGPUColumnStatistics,
  gpuSorter?: ResidentTableGPUSorter,
  wideStatistics?: ResidentTableWideStatistics
): ResidentTableProvider {
  const sourceFields = new Map(source.schema.fields.map(field => [field.name, field]));
  // GPU-only resident datasets keep no Arrow words for the exact 64-bit/Float64 GPU sort; their
  // prepared Float32 column still sorts on the GPU (approximate: values round to Float32).
  const usesPreparedScalarSort = (type: arrow.DataType) =>
    isGPUScalarSortType(type) ||
    (Boolean(resident) &&
      ['wide-int64', 'wide-uint64', 'float64'].includes(classifyTableSortType(type)));
  const richSorter = createOriginalRichSorter({source, provider, isCurrent, resident});
  const columnReader = createColumnStatisticsReader({
    source,
    provider,
    isCurrent,
    readPopulationRows: population => richSorter.readPopulationRows(population),
    gpu: gpuStatistics,
    wide: wideStatistics,
    getPopulationRows: async population => {
      if (population === 'all') return source.numRows;
      const identityColumn = provider.fields.find(field => !field.derived)?.name;
      const page =
        identityColumn &&
        (await provider.readPage({
          offset: 0,
          limit: 1,
          population,
          columns: [identityColumn],
          view: 'prepared'
        }));
      return page ? page.totalCount : source.numRows;
    }
  });
  let requestRevision = 0;
  const originalFields = () => {
    const fields: ResidentTableField[] = source.schema.fields.map(field => {
      const prepared = provider.fields.find(
        candidate => candidate.name === field.name && !candidate.derived
      );
      const gpuSortable = Boolean(prepared?.sortable && usesPreparedScalarSort(field.type));
      // Other ordered types sort exactly through the bounded asynchronous original sorter.
      const sortable =
        gpuSortable || (Boolean(provider.readRows) && isOriginalSortType(field.type));
      return {
        name: field.name,
        label: `${field.name} (${field.type.toString()}, original)`,
        sortable,
        ...(sortable
          ? {}
          : {
              sortUnavailableReason:
                'This source type has no total order (binary, list, struct or map); sort a derived or scalar field instead.'
            })
      };
    });
    const usedNames = new Set(sourceFields.keys());
    const derivedNames = new Map<string, string>();
    for (const field of provider.fields) {
      if (!field.derived && sourceFields.has(field.name)) continue;
      let name = field.name;
      while (usedNames.has(name)) name += '_analysis';
      usedNames.add(name);
      derivedNames.set(name, field.name);
      fields.push({...field, name, derived: true});
    }
    return {fields, derivedNames};
  };
  return {
    datasetId: provider.datasetId,
    sourceRevision: provider.sourceRevision,
    maximumPageSize: provider.maximumPageSize,
    get fields() {
      return provider.fields;
    },
    get originalFields() {
      return originalFields().fields;
    },
    getRevision: () => provider.getRevision(),
    subscribe: listener => provider.subscribe(listener),
    readRows: provider.readRows && ((rows, columns) => provider.readRows!(rows, columns)),
    readColumnStatistics: request => columnReader.readColumnStatistics(request),
    readColumnValues: request => columnReader.readColumnValues(request),
    async readPage(request) {
      const capturedRevision = ++requestRevision;
      const revision = provider.getRevision();
      if (!isCurrent()) return null;
      if (request.view !== 'original') return provider.readPage(request);
      const {fields, derivedNames} = originalFields();
      if (
        !Number.isSafeInteger(request.offset) ||
        request.offset < 0 ||
        !Number.isInteger(request.limit) ||
        request.limit < 1 ||
        request.limit > provider.maximumPageSize
      )
        throw new Error('Invalid bounded original-source page');
      if (
        !request.columns.length ||
        new Set(request.columns).size !== request.columns.length ||
        request.columns.some(name => !fields.some(field => field.name === name))
      )
        throw new Error('Unknown or duplicate original-source column');
      if (
        request.sort &&
        !fields.some(field => field.name === request.sort!.column && field.sortable)
      )
        throw new Error('Exact original-value sorting is unavailable for this field');
      const columns = [...request.columns];
      const limit = request.limit;
      if (columns.length * limit > MAXIMUM_ORIGINAL_TABLE_CELLS)
        throw new Error(
          'Original table page has too many cells; hide columns or reduce rows per page'
        );
      const requestedDerivedNames = columns.flatMap(name =>
        derivedNames.has(name) ? [derivedNames.get(name)!] : []
      );
      const sortField = request.sort
        ? fields.find(field => field.name === request.sort!.column)
        : undefined;
      const richSort = Boolean(
        request.sort &&
          sortField &&
          !sortField.derived &&
          !(
            usesPreparedScalarSort(sourceFields.get(sortField.name)!.type) &&
            provider.fields.some(
              candidate => candidate.name === sortField.name && candidate.sortable
            )
          )
      );
      // Dictionary strings and 64-bit/Float64 columns above the CPU bound sort on the GPU.
      let populationRows = source.numRows;
      if (richSort && gpuSorter && provider.readRows && request.population === 'selected') {
        const identityColumn = provider.fields.find(field => !field.derived)?.name;
        const countPage =
          identityColumn &&
          (await provider.readPage({
            offset: 0,
            limit: 1,
            population: 'selected',
            columns: [identityColumn],
            view: 'prepared'
          }));
        if (countPage) populationRows = countPage.totalCount;
      }
      const gpuRoute =
        richSort && gpuSorter && provider.readRows
          ? gpuSorter.planSort(request.sort!.column, populationRows)
          : undefined;
      let gpuPage: ResidentTablePage | null | undefined;
      if (gpuRoute?.route === 'gpu') {
        const sorted = await gpuSorter!.readSortedRows({
          column: request.sort!.column,
          direction: request.sort!.direction,
          population: request.population,
          offset: request.offset,
          limit: request.limit
        });
        const sortedPage =
          sorted && (await provider.readRows!(Array.from(sorted.rows), requestedDerivedNames));
        gpuPage = sortedPage ? {...sortedPage, totalCount: sorted!.totalCount} : null;
      }
      const page =
        gpuPage !== undefined
          ? gpuPage
          : richSort
            ? await richSorter.readSortedPage(request, request.sort!, requestedDerivedNames)
            : await provider.readPage({
                ...request,
                view: 'prepared',
                columns: requestedDerivedNames,
                sort: request.sort
                  ? {
                      ...request.sort,
                      column: derivedNames.get(request.sort.column) ?? request.sort.column
                    }
                  : undefined
              });
      if (
        !page ||
        capturedRevision !== requestRevision ||
        revision !== provider.getRevision() ||
        !isCurrent() ||
        page.datasetRevision !== provider.sourceRevision
      )
        return null;
      if (page.sourceRowIndices.length > limit)
        throw new Error('Source page exceeds its requested bound');
      // Resident sources are lazy: page only the GPU-selected rows into the CPU view.
      if (resident) {
        await resident.ensureRows(Array.from(page.sourceRowIndices));
        if (
          capturedRevision !== requestRevision ||
          revision !== provider.getRevision() ||
          !isCurrent()
        )
          return null;
      }
      const result: ResidentTablePage = {
        ...page,
        columns: Object.create(null),
        validity: Object.create(null),
        truncatedCells: 0
      };
      const charactersPerCell = Math.floor(
        MAXIMUM_ORIGINAL_TABLE_CHARACTERS /
          Math.max(1, page.sourceRowIndices.length * columns.length)
      );
      for (const name of columns) {
        const field = sourceFields.get(name);
        if (!field) {
          const originalName = derivedNames.get(name)!;
          result.columns[name] = page.columns[originalName];
          if (page.validity[originalName]) result.validity[name] = page.validity[originalName];
          continue;
        }
        const vector = source.getChild(name)!;
        const values: string[] = [];
        const validity = new Uint32Array(page.sourceRowIndices.length);
        for (const [index, sourceRow] of page.sourceRowIndices.entries()) {
          if (sourceRow >= source.numRows)
            throw new Error('Source row identity is outside retained Arrow data');
          if (vector.isValid(sourceRow)) {
            const cell = formatOriginalTableCell(vector, sourceRow, charactersPerCell);
            values.push(cell.text);
            if (cell.truncated) result.truncatedCells!++;
            validity[index] = 1;
          } else values.push('');
        }
        result.columns[name] = values;
        result.validity[name] = validity;
      }
      return capturedRevision === requestRevision &&
        revision === provider.getRevision() &&
        isCurrent()
        ? result
        : null;
    }
  };
}
