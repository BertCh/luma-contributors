// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useRef, useState, useSyncExternalStore, type ComponentProps} from 'react';
import {DataTableModalFactory} from '@kepler.gl/components';
import type {KeplerTablePage, KeplerTablePageRequest} from '@luma-contributors/kepler';
import type {ResidentColumnStatistics, ResidentColumnValues} from './resident-column-statistics';
import {ResidentTableColumnTools} from './resident-table-column-tools';
import {ResidentTableColumnEdits, makeResidentColumnFormatter} from './resident-table-column-edits';
import {
  applyResidentColumnEdits,
  formatResidentCell,
  residentColumnEditStore,
  type ResidentColumnDisplayType,
  type ResidentColumnEdits
} from './resident-column-edits';
import {
  changeResidentColumnTypeNative,
  deleteResidentColumnNative,
  renameResidentColumnNative,
  type ResidentNativeColumnEditHost,
  type ResidentNativeEditOutcome
} from './resident-column-native-edit';
import {
  applyResidentColumnDelete,
  planResidentColumnDelete,
  type ResidentColumnEditHost
} from './resident-column-edit-propagation';
import {ResidentConsumerNotice} from './resident-consumer-notice';
import './resident-data-table.css';

export type ResidentTableField = {
  name: string;
  label?: string;
  sortable: boolean;
  derived?: boolean;
  sortUnavailableReason?: string;
  format?: (value: number) => string;
};
export type ResidentTablePageRequest = KeplerTablePageRequest & {view?: 'prepared' | 'original'};
export type ResidentTablePage = Omit<KeplerTablePage, 'columns'> & {
  columns: Record<string, ArrayLike<number | string>>;
  truncatedCells?: number;
};
export type ResidentTableProvider = {
  datasetId: string;
  sourceRevision: number;
  fields: readonly ResidentTableField[];
  originalFields?: readonly ResidentTableField[];
  maximumPageSize: number;
  /** Changes whenever source, selection or computed columns change. */
  getRevision(): string;
  subscribe(listener: () => void): () => void;
  readPage(request: ResidentTablePageRequest): Promise<ResidentTablePage | null>;
  /** Explicit source rows in the caller's order (exact rich-field sorting); totalCount is their count. */
  readRows?(
    sourceRows: readonly number[],
    columns: readonly string[]
  ): Promise<ResidentTablePage | null>;
} & ResidentTableProviderColumnReaders;
export type ResidentTableColumnRequest = {column: string; population: 'all' | 'selected'};
export type ResidentTableProviderColumnReaders = {
  /** Native Kepler column statistics over the bounded current population. */
  readColumnStatistics?(
    request: ResidentTableColumnRequest
  ): Promise<ResidentColumnStatistics | null>;
  /** Exact column values over the bounded current population, for copying. */
  readColumnValues?(request: ResidentTableColumnRequest): Promise<ResidentColumnValues | null>;
};
/** Host actions behind the column operations (native filter creation needs the Kepler dispatchers). */
export type ResidentTableColumnActions = {
  createFilter?(column: string): void;
  /** Dispatches Kepler's setColumnDisplayFormat for the dataset field; null clears the format. */
  setDisplayFormat?(column: string, format: string | null): void;
  /** Real Kepler rename (layers, filters, tooltips and channels follow); refused with a reason. */
  renameColumn?(
    column: string,
    newName: string,
    edits: ResidentColumnEdits
  ): ResidentNativeEditOutcome;
  /** Real Kepler type conversion of the whole column; refused with a reason. */
  changeColumnType?(
    column: string,
    type: ResidentColumnDisplayType,
    edits: ResidentColumnEdits
  ): ResidentNativeEditOutcome;
  /** Real Kepler deleteTableColumn; refused with a reason so the caller can hide the column instead. */
  deleteColumnNative?(column: string, edits: ResidentColumnEdits): ResidentNativeEditOutcome;
  /** Propagates a delete to filters and tooltips; refuses while a layer still uses the column. */
  deleteColumn?(
    column: string
  ): {ok: true; removedFilters: number; tooltipUpdated: boolean} | {ok: false; reason: string};
};
export type ResidentDataTableOptions = {
  /** Host dispatchers that propagate column edits to Kepler filters and tooltips. */
  columnEditHost?: ResidentColumnEditHost;
  nativeColumnEditHost?: ResidentNativeColumnEditHost;
  isResidentDataset(datasetId: string): boolean;
  resolveProvider(datasetId: string): ResidentTableProvider | undefined;
};

/** Replace the modal before its filtered view calls Kepler's synchronous CPU materialization. */
export function createResidentDataTableModalFactory(options: ResidentDataTableOptions) {
  function ResidentDataTableModalFactory(
    ...dependencies: Parameters<typeof DataTableModalFactory>
  ) {
    const NativeModal = DataTableModalFactory(...dependencies);
    return function ResidentDataTableModal(props: ComponentProps<typeof NativeModal>) {
      const datasetId = props.dataId ?? '';
      if (!options.isResidentDataset(datasetId)) return <NativeModal {...props} />;
      const provider = options.resolveProvider(datasetId);
      return (
        <section className="resident-table-modal" aria-label="Resident data table">
          <nav aria-label="Table datasets">
            {Object.values(props.datasets).map(dataset => (
              <button
                key={dataset.id}
                type="button"
                aria-pressed={dataset.id === datasetId}
                onClick={() => props.showDatasetTable(dataset.id)}
              >
                {dataset.label}
              </button>
            ))}
          </nav>
          {provider ? (
            <ResidentDataTable
              key={`${datasetId}:${provider.sourceRevision}`}
              provider={provider}
              columnActions={getResidentTableColumnActions(
                props,
                datasetId,
                options.columnEditHost,
                options.nativeColumnEditHost
              )}
            />
          ) : (
            <p role="status">
              The resident dataset is being prepared. Close and reopen the table when it is ready.
            </p>
          )}
        </section>
      );
    };
  }
  ResidentDataTableModalFactory.deps = DataTableModalFactory.deps;
  return ResidentDataTableModalFactory;
}

/** Native modal props carry the Kepler dispatchers; only filter creation is forwarded. */
function getResidentTableColumnActions(
  props: unknown,
  datasetId: string,
  editHost?: ResidentColumnEditHost,
  nativeHost?: ResidentNativeColumnEditHost
): ResidentTableColumnActions {
  const {
    createFilterFromColumn,
    uiStateActions,
    setColumnDisplayFormat,
    renameTableColumn,
    changeTableColumnType,
    deleteTableColumn
  } = props as {
    deleteTableColumn?(dataId: string, column: string): void;
    renameTableColumn?(dataId: string, column: string, newName: string): void;
    changeTableColumnType?(dataId: string, column: string, type: string): void;
    createFilterFromColumn?(dataId: string, column: string): void;
    setColumnDisplayFormat?(dataId: string, formats: Record<string, string | null>): void;
    uiStateActions?: {toggleSidePanel(id: string): void; toggleModal(id: string | null): void};
  };
  return {
    ...(createFilterFromColumn
      ? {
          createFilter(column: string) {
            createFilterFromColumn(datasetId, column);
            uiStateActions?.toggleSidePanel('filter');
            uiStateActions?.toggleModal(null);
          }
        }
      : {}),
    ...(setColumnDisplayFormat
      ? {
          setDisplayFormat: (column: string, format: string | null) =>
            setColumnDisplayFormat(datasetId, {[column]: format})
        }
      : {}),
    ...(nativeHost && renameTableColumn
      ? {
          renameColumn: (column: string, newName: string, edits: ResidentColumnEdits) =>
            renameResidentColumnNative(
              {...nativeHost, renameTableColumn},
              datasetId,
              edits,
              column,
              newName
            )
        }
      : {}),
    ...(nativeHost && changeTableColumnType
      ? {
          changeColumnType: (
            column: string,
            type: ResidentColumnDisplayType,
            edits: ResidentColumnEdits
          ) =>
            changeResidentColumnTypeNative(
              {...nativeHost, changeTableColumnType},
              datasetId,
              edits,
              column,
              type
            )
        }
      : {}),
    ...(nativeHost && deleteTableColumn
      ? {
          deleteColumnNative: (column: string, edits: ResidentColumnEdits) =>
            deleteResidentColumnNative(
              {
                ...nativeHost,
                deleteTableColumn,
                getDeleteRefusal: (id, name) => {
                  const state = editHost?.getVisState();
                  if (!state) return null;
                  const plan = planResidentColumnDelete(state, id, name);
                  return plan.ok ? null : plan.reason;
                }
              },
              datasetId,
              edits,
              column
            )
        }
      : {}),
    ...(editHost
      ? {deleteColumn: (column: string) => applyResidentColumnDelete(editHost, datasetId, column)}
      : {})
  };
}

export function ResidentDataTable({
  provider,
  columnActions
}: {
  provider: ResidentTableProvider;
  columnActions?: ResidentTableColumnActions;
}) {
  const identifier = useId();
  const columnEdits = useSyncExternalStore(residentColumnEditStore.subscribe, () =>
    residentColumnEditStore.get(provider.datasetId)
  );
  const [revision, setRevision] = useState(provider.getRevision());
  const [view, setView] = useState<'prepared' | 'original'>('prepared');
  const [population, setPopulation] = useState<'all' | 'selected'>('selected');
  const [offset, setOffset] = useState(0);
  const [pageSize, setPageSize] = useState(Math.min(50, provider.maximumPageSize));
  const [sort, setSort] = useState<KeplerTablePageRequest['sort']>();
  const [hidden, setHidden] = useState<readonly string[]>([]);
  const [pinned, setPinned] = useState<readonly string[]>([]);
  const [wrapCells, setWrapCells] = useState(false);
  const [page, setPage] = useState<ResidentTablePage | null>(null);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState('');
  // A null page means the revision moved or the GPU selection is still computing: a stale state
  // that retries with backoff instead of an error (NATIVEUI5).
  const [staleReads, setStaleReads] = useState(0);
  const staleReadsReference = useRef(0);
  const [retry, setRetry] = useState(0);
  const allFields =
    view === 'original' ? (provider.originalFields ?? provider.fields) : provider.fields;
  const fields = applyResidentColumnEdits(allFields, columnEdits);
  const shownFields = fields.filter(field => !hidden.includes(field.name));
  // Pinned columns lead, in pin order, and stay sticky beside the source-row column.
  const visibleFields = [
    ...pinned.flatMap(name => shownFields.filter(field => field.name === name)),
    ...shownFields.filter(field => !pinned.includes(field.name))
  ];
  const tableReference = useRef<HTMLTableElement>(null);
  const columnKey = JSON.stringify(visibleFields.map(field => field.name));
  useEffect(() => {
    if (sort && !fields.some(field => field.name === sort.column && field.sortable))
      setSort(undefined);
  }, [provider, revision, sort, view]);
  useEffect(
    () =>
      provider.subscribe(() => {
        setRevision(provider.getRevision());
        setRetry(value => value + 1);
      }),
    [provider]
  );
  useEffect(() => {
    setOffset(0);
  }, [revision, population, pageSize, sort?.column, sort?.direction, view]);
  useEffect(() => {
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    const startedRevision = provider.getRevision();
    setPage(null);
    setPending(true);
    setError('');
    void provider
      .readPage({offset, limit: pageSize, population, sort, columns: JSON.parse(columnKey), view})
      .then(result => {
        if (cancelled || provider.getRevision() !== startedRevision) return;
        setPage(result);
        setPending(false);
        if (result) {
          staleReadsReference.current = 0;
          setStaleReads(0);
        }
        if (!result) {
          staleReadsReference.current += 1;
          setStaleReads(staleReadsReference.current);
          if (staleReadsReference.current <= 5)
            retryTimer = setTimeout(
              () => setRetry(value => value + 1),
              Math.min(4000, 250 * 2 ** staleReadsReference.current)
            );
        } else if (offset > 0 && offset >= result.totalCount) setOffset(0);
      })
      .catch(failure => {
        if (cancelled || provider.getRevision() !== startedRevision) return;
        setPending(false);
        setError(failure instanceof Error ? failure.message : String(failure));
      });
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
    };
  }, [
    provider,
    revision,
    population,
    offset,
    pageSize,
    sort?.column,
    sort?.direction,
    columnKey,
    view,
    retry
  ]);
  useEffect(() => {
    // Sticky offsets are measured so pinned columns stack after the source-row column.
    const table = tableReference.current;
    if (!table) return;
    let left = table.querySelector<HTMLElement>('thead th:first-child')?.offsetWidth ?? 0;
    for (const header of table.querySelectorAll<HTMLElement>('thead th[data-pinned]')) {
      for (const cell of table.querySelectorAll<HTMLElement>(
        `tr > :nth-child(${(header as HTMLTableCellElement).cellIndex + 1})`
      ))
        cell.style.left = `${left}px`;
      left += header.offsetWidth;
    }
  }, [page, pinned, wrapCells, columnKey]);
  const changeSort = (field: ResidentTableField) => {
    if (!field.sortable) return;
    setSort(previous =>
      previous?.column !== field.name
        ? {column: field.name, direction: 'ascending'}
        : previous.direction === 'ascending'
          ? {column: field.name, direction: 'descending'}
          : undefined
    );
  };
  return (
    <>
      <h2>Data table</h2>
      <div className="resident-table-controls">
        {provider.originalFields && (
          <>
            <label htmlFor={`${identifier}-view`}>Values</label>
            <select
              id={`${identifier}-view`}
              value={view}
              onChange={event => {
                setView(event.target.value === 'original' ? 'original' : 'prepared');
                setSort(undefined);
                setHidden([]);
                setPinned([]);
                setOffset(0);
              }}
            >
              <option value="prepared">Prepared and derived</option>
              <option value="original">Original source and derived</option>
            </select>
          </>
        )}
        <label htmlFor={`${identifier}-population`}>Rows</label>
        <select
          id={`${identifier}-population`}
          value={population}
          onChange={event => setPopulation(event.target.value === 'all' ? 'all' : 'selected')}
        >
          <option value="selected">Current selection</option>
          <option value="all">All source rows</option>
        </select>
        <label htmlFor={`${identifier}-page-size`}>Rows per page</label>
        <select
          id={`${identifier}-page-size`}
          value={pageSize}
          onChange={event => setPageSize(Number(event.target.value))}
        >
          {[
            ...new Set([
              Math.min(25, provider.maximumPageSize),
              Math.min(50, provider.maximumPageSize),
              Math.min(100, provider.maximumPageSize)
            ])
          ].map(size => (
            <option key={size} value={size}>
              {size}
            </option>
          ))}
        </select>
        <button type="button" onClick={() => setRetry(value => value + 1)}>
          Refresh
        </button>
        <label className="resident-table-wrap-control">
          <input
            type="checkbox"
            checked={wrapCells}
            onChange={event => setWrapCells(event.target.checked)}
          />
          Wrap cell previews
        </label>
      </div>
      <details>
        <summary>Columns</summary>
        {fields.map(field => (
          <label key={field.name} className="resident-table-column">
            <input
              type="checkbox"
              checked={!hidden.includes(field.name)}
              disabled={visibleFields.length === 1 && !hidden.includes(field.name)}
              onChange={event =>
                setHidden(previous =>
                  event.target.checked
                    ? previous.filter(name => name !== field.name)
                    : [...previous, field.name]
                )
              }
            />
            {field.label ?? field.name}
          </label>
        ))}
      </details>
      <ResidentTableColumnEdits
        fields={allFields}
        edits={columnEdits}
        onChange={next => residentColumnEditStore.set(provider.datasetId, next)}
        onCreateFilter={columnActions?.createFilter}
        actions={columnActions}
      />
      <ResidentTableColumnTools
        provider={provider}
        fields={visibleFields}
        population={population}
        revision={revision}
        pinned={pinned}
        onTogglePin={name =>
          setPinned(previous =>
            previous.includes(name) ? previous.filter(item => item !== name) : [...previous, name]
          )
        }
      />
      <p className="resident-table-status" role="status">
        {pending
          ? 'Reading current rows…'
          : page
            ? `${page.totalCount.toLocaleString()} ${population === 'selected' ? 'selected' : 'source'} rows · source revision ${page.datasetRevision}`
            : 'No current page'}
      </p>
      {error && <ResidentConsumerNotice consumer="table" status="error" reason={error} />}
      {!pending && !error && !page && (
        <ResidentConsumerNotice
          consumer="table"
          status={staleReads > 5 ? 'error' : 'stale'}
          reason={
            staleReads > 5
              ? 'The dataset changed or its selection did not finish computing. Change the view to retry.'
              : 'The dataset or its selection changed while reading; refreshing the rows.'
          }
        />
      )}
      {!pending && !error && page && page.totalCount === 0 && (
        <ResidentConsumerNotice
          consumer="table"
          status="empty"
          reason={
            population === 'selected'
              ? 'No rows pass the active filters and selection.'
              : 'The dataset has no rows.'
          }
        />
      )}
      {Boolean(page?.truncatedCells) && (
        <p role="status">
          {page!.truncatedCells} {page!.truncatedCells === 1 ? 'cell' : 'cells'} shortened for
          display. Export original Arrow for complete values.
        </p>
      )}
      <div
        className={`resident-table-scroll${wrapCells ? ' resident-table-wrap' : ''}`}
        tabIndex={0}
        role="region"
        aria-label="Scrollable data rows"
        aria-busy={pending}
      >
        <table ref={tableReference} aria-rowcount={page ? page.totalCount + 1 : undefined}>
          <caption>
            {sort
              ? `${sort.column} · ${sort.direction}; missing values last; equal values retain source order.`
              : 'Source row order'}
          </caption>
          <thead>
            <tr>
              <th scope="col">Source row</th>
              {visibleFields.map(field => (
                <th
                  key={field.name}
                  scope="col"
                  data-pinned={pinned.includes(field.name) || undefined}
                  aria-sort={sort?.column === field.name ? sort.direction : 'none'}
                >
                  <button
                    type="button"
                    disabled={!field.sortable}
                    title={field.sortUnavailableReason}
                    onClick={() => changeSort(field)}
                  >
                    {field.label ?? field.name}
                    {!field.sortable && <span> (sort unavailable)</span>}
                    {sort?.column === field.name
                      ? sort.direction === 'ascending'
                        ? ' ↑'
                        : ' ↓'
                      : ''}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page &&
              Array.from(page.sourceRowIndices, (sourceRow, index) => (
                <tr key={sourceRow}>
                  <th scope="row">{sourceRow}</th>
                  {visibleFields.map(field => (
                    <td
                      key={field.name}
                      data-pinned={pinned.includes(field.name) || undefined}
                      title={String(page.columns[field.name]?.[index] ?? '')}
                    >
                      {page.validity[field.name]?.[index] === 0
                        ? 'null'
                        : formatResidentTableCell(
                            page.columns[field.name]?.[index],
                            field,
                            columnEdits
                          )}
                    </td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
        {page?.totalCount === 0 && <p>No rows match the current selection.</p>}
      </div>
      <nav className="resident-table-controls" aria-label="Table pages">
        <button
          type="button"
          disabled={pending || offset === 0}
          onClick={() => setOffset(Math.max(0, offset - pageSize))}
        >
          Previous page
        </button>
        <span>
          {page
            ? `${page.totalCount ? offset + 1 : 0}–${offset + page.sourceRowIndices.length} of ${page.totalCount.toLocaleString()}`
            : '—'}
        </span>
        <button
          type="button"
          disabled={pending || !page || offset + pageSize >= page.totalCount}
          onClick={() => setOffset(offset + pageSize)}
        >
          Next page
        </button>
      </nav>
      <p className="resident-table-note">
        {view === 'original'
          ? 'Original values come from retained Arrow rows using only this page of source IDs. Timestamps retain their source units and timezone; strings, dates, timestamps and 64-bit integers sort exactly on the CPU for at most 262,144 rows. Derived values are computed results. Rich cells are bounded previews; export original Arrow for complete values.'
          : 'Only requested rows return to the CPU. Sorting runs on the GPU; it keeps source buffers and row identities intact.'}
      </p>
    </>
  );
}

function formatResidentTableCell(
  value: number | string | undefined,
  field: ResidentTableField,
  edits: ReturnType<typeof residentColumnEditStore.get>
): string {
  const type = edits.types[field.name];
  const format = edits.formats[field.name];
  if (!type && !format) return formatTableValue(value, field.format);
  return formatResidentCell(value, type, makeResidentColumnFormatter(type, format)).text;
}

function formatTableValue(
  value: number | string | undefined,
  format?: (value: number) => string
): string {
  if (value === undefined) return 'unavailable';
  if (typeof value === 'string') return value;
  return format ? format(value) : String(Number(value.toPrecision(7)));
}
