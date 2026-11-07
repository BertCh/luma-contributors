// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {ResidentColumnStatistics} from './resident-column-statistics';
import type {ResidentTableField, ResidentTableProvider} from './resident-data-table';

type NativeBin = {x0?: number; x1?: number; count?: number};
type NativeStats = Record<string, unknown> & {
  bins?: NativeBin[];
  quantiles?: {label: string; value: number | null}[];
};

const formatStatistic = (value: unknown): string =>
  typeof value === 'number'
    ? Number.isInteger(value)
      ? value.toLocaleString()
      : String(Number(value.toPrecision(7)))
    : value === null || value === undefined
      ? 'n/a'
      : String(value);

/** Pure row list for a Kepler column-statistics result (also used by the specs). */
export function makeColumnStatisticRows(
  result: ResidentColumnStatistics,
  isTimestamp = result.fieldType === 'timestamp'
): [label: string, value: string][] {
  const stats = result.stats as NativeStats;
  const rows: [string, string][] = [
    ['Rows', result.rowCount.toLocaleString()],
    [
      'Nulls',
      `${result.nullCount.toLocaleString()} (${formatStatistic(
        result.rowCount ? (100 * result.nullCount) / result.rowCount : 0
      )}%)`
    ]
  ];
  if (result.source)
    rows.push([
      'Computed on',
      result.source === 'gpu'
        ? 'GPU profile (interior quantiles estimated from the histogram)'
        : 'CPU over the bounded population'
    ]);
  if (stats['type'] === 'numeric') {
    rows.push(
      ['Mean', formatStatistic(stats['mean'])],
      ['Std. dev.', formatStatistic(stats['std'])]
    );
    for (const quantile of stats.quantiles ?? [])
      rows.push([
        quantile.label === '50%' ? 'Median (50%)' : quantile.label,
        formatStatistic(quantile.value)
      ]);
  } else if (stats['type'] === 'categorical') {
    rows.push(['Unique values', formatStatistic(stats['uniqueValues'])]);
  } else if (stats['type'] === 'time') {
    const show = (value: unknown) =>
      isTimestamp && typeof value === 'number'
        ? new Date(value).toISOString()
        : formatStatistic(value);
    rows.push(['Minimum', show(stats['min'])], ['Maximum', show(stats['max'])]);
  }
  return rows;
}

/** Pure clipboard text: one value per line, missing cells empty. */
export function makeColumnClipboardText(values: readonly unknown[]): string {
  return values
    .map(value => (value === null || value === undefined ? '' : String(value)))
    .join('\n');
}

/**
 * Per-column tools beside the native sort and hide controls: pin, copy values and Kepler column
 * statistics. Statistics and copies read only the current population within the table row bound.
 */
export function ResidentTableColumnTools(props: {
  provider: ResidentTableProvider;
  fields: readonly ResidentTableField[];
  population: 'all' | 'selected';
  revision: string;
  pinned: readonly string[];
  onTogglePin(name: string): void;
}) {
  const {provider, fields, population, revision, pinned} = props;
  const [column, setColumn] = useState(fields[0]?.name ?? '');
  const [status, setStatus] = useState('');
  const [statistics, setStatistics] = useState<ResidentColumnStatistics | null>(null);
  const [busy, setBusy] = useState(false);
  const selected = fields.find(field => field.name === column) ?? fields[0];
  useEffect(() => {
    // A new revision, population or column makes earlier results stale.
    setStatistics(null);
    setStatus('');
  }, [revision, population, selected?.name]);
  if (!selected) return null;
  const run = async (action: () => Promise<string | void>) => {
    setBusy(true);
    setStatus('Reading current rows…');
    const startedRevision = provider.getRevision();
    try {
      const message = await action();
      setStatus(
        provider.getRevision() === startedRevision
          ? (message ?? '')
          : 'The data changed while this finished; run it again.'
      );
    } catch (failure) {
      setStatus(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  };
  const request = {column: selected.name, population};
  return (
    <details className="resident-table-column-tools">
      <summary>Column tools</summary>
      <div className="resident-table-controls">
        <label htmlFor="resident-table-tool-column">Column</label>
        <select
          id="resident-table-tool-column"
          value={selected.name}
          onChange={event => setColumn(event.target.value)}
        >
          {fields.map(field => (
            <option key={field.name} value={field.name}>
              {field.label ?? field.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          aria-pressed={pinned.includes(selected.name)}
          onClick={() => props.onTogglePin(selected.name)}
        >
          {pinned.includes(selected.name) ? 'Unpin column' : 'Pin column'}
        </button>
        <button
          type="button"
          disabled={busy || !provider.readColumnValues}
          onClick={() =>
            void run(async () => {
              const result = await provider.readColumnValues!(request);
              if (!result) return 'The data changed; copy again.';
              await navigator.clipboard.writeText(makeColumnClipboardText(result.values));
              return `Copied ${result.values.length.toLocaleString()} values.`;
            })
          }
        >
          Copy column
        </button>
        <button
          type="button"
          disabled={busy || !provider.readColumnStatistics}
          onClick={() =>
            void run(async () => {
              const result = await provider.readColumnStatistics!(request);
              setStatistics(result);
              return result ? '' : 'The data changed; compute again.';
            })
          }
        >
          Column statistics
        </button>
      </div>
      <p role="status" className="resident-table-status">
        {status}
      </p>
      {statistics && statistics.column === selected.name && (
        <table className="resident-table-statistics">
          <caption>
            {statistics.column} · {statistics.population === 'selected' ? 'selection' : 'all rows'}
          </caption>
          <tbody>
            {makeColumnStatisticRows(statistics).map(([label, value]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </details>
  );
}
