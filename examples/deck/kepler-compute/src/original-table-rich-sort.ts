// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import {
  MAXIMUM_ORIGINAL_SORT_ROWS,
  isOriginalSortType,
  sortOriginalRows,
  type OriginalSortDirection
} from './original-table-sort';
import type {
  ResidentTablePage,
  ResidentTablePageRequest,
  ResidentTableProvider
} from './resident-data-table';

import {chooseTableSortRoute} from './resident-table-population-routes';

export {isOriginalSortType};

/** Stated reason for a refused CPU ordering of a population above the bound. */
function describeLargePopulation(populationRows: number): string {
  return chooseTableSortRoute({
    orderKind: 'other-ordered',
    hasPreparedGPUColumn: false,
    populationRows
  }).reason;
}

/**
 * Exact asynchronous sorting of original Arrow columns that the GPU scalar sorter cannot order
 * (strings, dictionaries, timestamps, Int64/Uint64, Float64, dates). Row identities for the current
 * population come back in bounded pages, the lazy source is materialized only for those rows, the
 * order is computed once per revision and cached; every page then reads just its own rows.
 */
export function createOriginalRichSorter(props: {
  source: arrow.Table;
  provider: ResidentTableProvider;
  isCurrent(): boolean;
  resident?: {ensureRows(rows: readonly number[]): Promise<void>};
}) {
  const {source, provider, resident} = props;
  let cached: {key: string; rows: Promise<Uint32Array | null>} | undefined;
  const readIdentities = async (
    population: 'all' | 'selected',
    revision: string
  ): Promise<Uint32Array | null> => {
    const identityColumn = provider.fields.find(field => !field.derived)?.name;
    if (population === 'all' || !identityColumn) {
      if (source.numRows > MAXIMUM_ORIGINAL_SORT_ROWS)
        throw new Error(describeLargePopulation(source.numRows)); // narrow the selection
      return Uint32Array.from({length: source.numRows}, (_, index) => index);
    }
    let identities: Uint32Array | undefined;
    let filled = 0;
    let selectionRevision: number | undefined;
    for (;;) {
      const page = await provider.readPage({
        offset: filled,
        limit: provider.maximumPageSize,
        population,
        columns: [identityColumn],
        view: 'prepared'
      });
      if (!page || provider.getRevision() !== revision || !props.isCurrent()) return null;
      if (selectionRevision === undefined) selectionRevision = page.revision;
      else if (page.revision !== selectionRevision) return null;
      if (page.totalCount > MAXIMUM_ORIGINAL_SORT_ROWS)
        throw new Error(describeLargePopulation(page.totalCount)); // narrow the selection
      identities ??= new Uint32Array(page.totalCount);
      identities.set(page.sourceRowIndices, filled);
      filled += page.sourceRowIndices.length;
      if (!page.sourceRowIndices.length || filled >= page.totalCount) break;
    }
    return identities ?? new Uint32Array(0);
  };
  /** Identities of the population with their lazy CPU cells paged in, or null when superseded. */
  const readMaterializedRows = async (
    population: 'all' | 'selected',
    revision: string
  ): Promise<Uint32Array | null> => {
    const identities = await readIdentities(population, revision);
    if (!identities) return null;
    if (resident) {
      const step = provider.maximumPageSize;
      for (let start = 0; start < identities.length; start += step) {
        await resident.ensureRows(Array.from(identities.subarray(start, start + step)));
        if (provider.getRevision() !== revision || !props.isCurrent()) return null;
      }
    }
    return identities;
  };
  const computeRows = async (
    population: 'all' | 'selected',
    column: string,
    direction: OriginalSortDirection,
    revision: string
  ): Promise<Uint32Array | null> => {
    const identities = await readMaterializedRows(population, revision);
    return identities && sortOriginalRows(source.getChild(column)!, identities, direction);
  };
  return {
    /** Bounded source rows of the population in source order, materialized for original reads. */
    readPopulationRows(population: 'all' | 'selected'): Promise<Uint32Array | null> {
      return readMaterializedRows(population, provider.getRevision());
    },
    /** Exact sorted source-row order for the population, or null when superseded. */
    readSortedRows(
      population: 'all' | 'selected',
      column: string,
      direction: OriginalSortDirection
    ): Promise<Uint32Array | null> {
      const revision = provider.getRevision();
      const key = `${revision}:${population}:${column}:${direction}`;
      if (cached?.key !== key) {
        const rows = computeRows(population, column, direction, revision);
        cached = {key, rows};
        // A failure or superseded read must not poison the cache for a retry.
        void rows.then(
          result => {
            if (!result && cached?.key === key) cached = undefined;
          },
          () => {
            if (cached?.key === key) cached = undefined;
          }
        );
      }
      return cached.rows;
    },
    /** One page of the exact order, with derived columns gathered for exactly those rows. */
    async readSortedPage(
      request: ResidentTablePageRequest,
      sort: {column: string; direction: OriginalSortDirection},
      derivedColumns: readonly string[]
    ): Promise<ResidentTablePage | null> {
      const rows = await this.readSortedRows(request.population, sort.column, sort.direction);
      if (!rows || !provider.readRows) return null;
      const slice = Array.from(rows.subarray(request.offset, request.offset + request.limit));
      const page = await provider.readRows(slice, derivedColumns);
      return page ? {...page, totalCount: rows.length} : null;
    }
  };
}
