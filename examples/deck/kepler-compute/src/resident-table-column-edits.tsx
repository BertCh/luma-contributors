// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {getFieldFormatLabels, getFormatter} from '@kepler.gl/utils';
import {
  RESIDENT_COLUMN_DISPLAY_TYPES,
  changeResidentColumnType,
  deleteResidentColumn,
  getResidentColumnDisplayName,
  getVisibleResidentColumns,
  renameResidentColumn,
  restoreResidentColumn,
  setResidentColumnFormat,
  type ResidentColumnDisplayType,
  type ResidentColumnEditResult,
  type ResidentColumnEdits
} from './resident-column-edits';
import type {ResidentTableColumnActions, ResidentTableField} from './resident-data-table';

/** Kepler format lookups for one display type; the first entry is always "None". */
export function getResidentColumnFormatOptions(type: ResidentColumnDisplayType | undefined) {
  return getFieldFormatLabels(type ?? 'real').filter(option => option.format);
}

/** Resolve a stored Kepler format string to a cell formatter (used by the data table). */
export function makeResidentColumnFormatter(
  type: ResidentColumnDisplayType | undefined,
  format: string | undefined
): ((value: number | Date) => string) | undefined {
  if (!format) return undefined;
  const formatter = getFormatter(format);
  return value => String(formatter(value as never));
}

/**
 * Native column operations for resident datasets: rename, change type, format, create filter and
 * delete. They change how the table shows the column (display metadata); source Arrow data, GPU
 * buffers, layers and filters are never rewritten.
 */
export function ResidentTableColumnEdits(props: {
  /** Every source and derived column, before edits. */
  fields: readonly ResidentTableField[];
  edits: ResidentColumnEdits;
  onChange(edits: ResidentColumnEdits): void;
  /** Present when the host can create a native filter from a dataset column. */
  onCreateFilter?(column: string): void;
  /** Host propagation of format and delete to Kepler state (setColumnDisplayFormat, filters, tooltip). */
  actions?: ResidentTableColumnActions;
}) {
  const {fields, edits} = props;
  const names = fields.map(field => field.name);
  const visible = getVisibleResidentColumns(names, edits);
  const [column, setColumn] = useState('');
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState('');
  const selected = visible.includes(column) ? column : (visible[0] ?? '');
  const field = fields.find(candidate => candidate.name === selected);
  const type = edits.types[selected];
  const apply = (result: ResidentColumnEditResult, message: string) => {
    if (result.ok) {
      props.onChange(result.edits);
      setStatus(message);
    } else setStatus(result.reason);
  };
  if (!selected || !field) return null;
  const formatOptions = getResidentColumnFormatOptions(type);
  return (
    <details className="resident-table-column-edits">
      <summary>Column operations</summary>
      <div className="resident-table-controls">
        <label htmlFor="resident-table-edit-column">Column</label>
        <select
          id="resident-table-edit-column"
          value={selected}
          onChange={event => {
            setColumn(event.target.value);
            setDraft('');
            setStatus('');
          }}
        >
          {visible.map(name => (
            <option key={name} value={name}>
              {getResidentColumnDisplayName(edits, name)}
            </option>
          ))}
        </select>
        <label htmlFor="resident-table-edit-name">Rename</label>
        <input
          id="resident-table-edit-name"
          value={draft}
          placeholder={getResidentColumnDisplayName(edits, selected)}
          onChange={event => setDraft(event.target.value)}
        />
        <button
          type="button"
          disabled={!draft.trim()}
          onClick={() => {
            const native = props.actions?.renameColumn?.(selected, draft, edits);
            if (native?.ok) {
              apply({ok: true, edits: native.edits}, native.message);
              setColumn(draft.trim());
            } else
              apply(
                renameResidentColumn(edits, names, selected, draft),
                native
                  ? `Alias only (${native.reason})`
                  : 'Column renamed (display alias; this host cannot rename natively).'
              );
            setDraft('');
          }}
        >
          Rename column
        </button>
        <label htmlFor="resident-table-edit-type">Display type</label>
        <select
          id="resident-table-edit-type"
          value={type ?? ''}
          onChange={event => {
            const nextType = (event.target.value || null) as ResidentColumnDisplayType | null;
            const native = nextType
              ? props.actions?.changeColumnType?.(selected, nextType, edits)
              : undefined;
            if (native?.ok) return apply({ok: true, edits: native.edits}, native.message);
            apply(
              {ok: true, edits: changeResidentColumnType(edits, selected, nextType)},
              native
                ? `Display conversion only (${native.reason})`
                : 'Display type changed. Invalid cells show "invalid"; source values are unchanged.'
            );
          }}
        >
          <option value="">Source type</option>
          {RESIDENT_COLUMN_DISPLAY_TYPES.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <label htmlFor="resident-table-edit-format">Display format</label>
        <select
          id="resident-table-edit-format"
          value={edits.formats[selected] ?? ''}
          onChange={event => {
            props.actions?.setDisplayFormat?.(selected, event.target.value || null);
            apply(
              {
                ok: true,
                edits: setResidentColumnFormat(edits, selected, event.target.value || null)
              },
              props.actions?.setDisplayFormat
                ? 'Display format changed and set on the Kepler dataset field.'
                : 'Display format changed.'
            );
          }}
        >
          <option value="">None</option>
          {formatOptions.map(option => (
            <option key={option.id} value={option.format ?? ''}>
              {option.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          disabled={!props.onCreateFilter || field.derived}
          title={
            field.derived
              ? 'Computed analysis columns are not dataset fields, so no native filter can be created.'
              : props.onCreateFilter
                ? undefined
                : 'The host did not provide filter creation.'
          }
          onClick={() => props.onCreateFilter?.(selected)}
        >
          Create filter from column
        </button>
        <button
          type="button"
          onClick={() => {
            const native = props.actions?.deleteColumnNative?.(selected, edits);
            if (native?.ok) return apply({ok: true, edits: native.edits}, native.message);
            const result = deleteResidentColumn(edits, names, selected);
            if (!result.ok) return apply(result, '');
            const propagated = props.actions?.deleteColumn?.(selected);
            if (propagated && !propagated.ok) return setStatus(propagated.reason);
            apply(
              result,
              propagated
                ? `${native ? `Hidden from the table only (${native.reason}). ` : ''}Column removed from this table and exports; ${propagated.removedFilters} filter(s) removed${propagated.tooltipUpdated ? ', tooltip field dropped' : ''}.`
                : 'Column removed from this table.'
            );
          }}
        >
          Delete column
        </button>
      </div>
      {edits.deleted.length > 0 && (
        <p className="resident-table-note">
          Removed from the table:{' '}
          {edits.deleted.map(name => (
            <button
              key={name}
              type="button"
              onClick={() => {
                props.onChange(restoreResidentColumn(edits, name));
                setStatus('Column restored.');
              }}
            >
              Restore {getResidentColumnDisplayName(edits, name)}
            </button>
          ))}
        </p>
      )}
      <p role="status" className="resident-table-status">
        {status}
      </p>
      <p className="resident-table-note">
        Rename and type change are applied to the Kepler dataset (layers, filters, tooltips and
        channels follow) and the resident dataset is re-prepared; when the dataset cannot be edited
        in place they fall back to a display alias or display conversion with the reason shown.
        Aliases and deleted columns are applied to table and export headers. Delete removes
        dependent filters and the tooltip field, and is refused while a layer uses the column.
        Display format is applied to the Kepler dataset field.
      </p>
    </details>
  );
}
