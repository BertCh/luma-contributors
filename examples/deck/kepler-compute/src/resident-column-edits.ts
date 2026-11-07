// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentTableField} from './resident-data-table';

/**
 * Native Kepler's table menu renames, retypes, formats and deletes columns by rewriting the dataset
 * (new Arrow data, remapped layer and filter references). Resident datasets own immutable GPU buffers,
 * so these edits are kept here as per-dataset DISPLAY metadata and never mutate Arrow data, GPU
 * buffers, layers or filters. Column identity stays the source name; only the table shows the edit.
 */
export type ResidentColumnDisplayType = 'integer' | 'real' | 'string' | 'timestamp';
export const RESIDENT_COLUMN_DISPLAY_TYPES: readonly ResidentColumnDisplayType[] = [
  'integer',
  'real',
  'string',
  'timestamp'
];

export type ResidentColumnEdits = {
  /** Source column name to the display name. */
  names: Readonly<Record<string, string>>;
  /** Source names hidden from the table (native "delete column" removes them from the view). */
  deleted: readonly string[];
  /** Source column name to the display type (changes cell conversion and the available formats). */
  types: Readonly<Record<string, ResidentColumnDisplayType>>;
  /** Source column name to a Kepler format string (d3-format or time format); absent means none. */
  formats: Readonly<Record<string, string>>;
};

export type ResidentColumnEditResult =
  | {ok: true; edits: ResidentColumnEdits}
  | {ok: false; reason: string};

export const EMPTY_RESIDENT_COLUMN_EDITS: ResidentColumnEdits = {
  names: {},
  deleted: [],
  types: {},
  formats: {}
};

export function getResidentColumnDisplayName(edits: ResidentColumnEdits, name: string): string {
  return edits.names[name] ?? name;
}

/** Names that remain after deletions, in source order. */
export function getVisibleResidentColumns(
  columns: readonly string[],
  edits: ResidentColumnEdits
): string[] {
  return columns.filter(name => !edits.deleted.includes(name));
}

export function renameResidentColumn(
  edits: ResidentColumnEdits,
  columns: readonly string[],
  name: string,
  newName: string
): ResidentColumnEditResult {
  const trimmed = newName.trim();
  if (!columns.includes(name)) return {ok: false, reason: 'Unknown column.'};
  if (!trimmed) return {ok: false, reason: 'A column name cannot be empty.'};
  const taken = getVisibleResidentColumns(columns, edits).some(
    other => other !== name && getResidentColumnDisplayName(edits, other) === trimmed
  );
  if (taken) return {ok: false, reason: `Another column is already named "${trimmed}".`};
  const names = {...edits.names};
  if (trimmed === name) delete names[name];
  else names[name] = trimmed;
  return {ok: true, edits: {...edits, names}};
}

/** The last remaining column is kept, as in native Kepler. */
export function deleteResidentColumn(
  edits: ResidentColumnEdits,
  columns: readonly string[],
  name: string
): ResidentColumnEditResult {
  if (!columns.includes(name)) return {ok: false, reason: 'Unknown column.'};
  if (edits.deleted.includes(name)) return {ok: true, edits};
  if (getVisibleResidentColumns(columns, edits).length <= 1)
    return {ok: false, reason: 'The last remaining column cannot be deleted.'};
  return {ok: true, edits: {...edits, deleted: [...edits.deleted, name]}};
}

export function restoreResidentColumn(
  edits: ResidentColumnEdits,
  name: string
): ResidentColumnEdits {
  return {...edits, deleted: edits.deleted.filter(candidate => candidate !== name)};
}

/** Changing the type drops a format that belonged to the previous type. */
export function changeResidentColumnType(
  edits: ResidentColumnEdits,
  name: string,
  type: ResidentColumnDisplayType | null
): ResidentColumnEdits {
  const types = {...edits.types};
  const formats = {...edits.formats};
  if (type) types[name] = type;
  else delete types[name];
  delete formats[name];
  return {...edits, types, formats};
}

export function setResidentColumnFormat(
  edits: ResidentColumnEdits,
  name: string,
  format: string | null
): ResidentColumnEdits {
  const formats = {...edits.formats};
  if (format) formats[name] = format;
  else delete formats[name];
  return {...edits, formats};
}

/** Fields without deleted columns, with the display name as label. Identity (name) is unchanged. */
export function applyResidentColumnEdits(
  fields: readonly ResidentTableField[],
  edits: ResidentColumnEdits
): ResidentTableField[] {
  return fields
    .filter(field => !edits.deleted.includes(field.name))
    .map(field =>
      edits.names[field.name] ? {...field, label: edits.names[field.name]} : {...field}
    );
}

export type ResidentCellDisplay = {text: string; invalid: boolean};

/** Display-only conversion of one cell (page values are numbers or already formatted strings). */
export function convertResidentCell(
  value: number | string | undefined,
  type: ResidentColumnDisplayType
): ResidentCellDisplay {
  if (value === undefined) return {text: 'unavailable', invalid: false};
  if (type === 'string') return {text: String(value), invalid: false};
  if (type === 'real' || type === 'integer') {
    const number =
      typeof value === 'number' ? value : value.trim() === '' ? NaN : Number(value.trim());
    if (!Number.isFinite(number)) return {text: 'invalid', invalid: true};
    if (type === 'integer' && typeof value === 'string' && !/^[+-]?\d+$/.test(value.trim()))
      return {text: 'invalid', invalid: true};
    return {text: String(type === 'integer' ? Math.trunc(number) : number), invalid: false};
  }
  const milliseconds =
    typeof value === 'number'
      ? value
      : /^-?\d+(\.\d+)?$/.test(value.trim())
        ? Number(value)
        : Date.parse(value);
  const date = new Date(milliseconds);
  return Number.isNaN(date.getTime())
    ? {text: 'invalid', invalid: true}
    : {text: date.toISOString(), invalid: false};
}

/**
 * Cell text after type conversion and the optional Kepler format. Formats apply to numbers and,
 * for timestamps, to the instant in milliseconds; the formatter is injected (Kepler getFormatter).
 */
export function formatResidentCell(
  value: number | string | undefined,
  type: ResidentColumnDisplayType | undefined,
  format: ((value: number | Date) => string) | undefined
): ResidentCellDisplay {
  if (value === undefined) return {text: 'unavailable', invalid: false};
  if (!type && !format)
    return {
      text: typeof value === 'string' ? value : String(Number(value.toPrecision(7))),
      invalid: false
    };
  const converted = convertResidentCell(
    value,
    type ?? (typeof value === 'number' ? 'real' : 'string')
  );
  if (converted.invalid || !format) return converted;
  if (type === 'timestamp') return {text: format(new Date(converted.text)), invalid: false};
  const number = Number(converted.text);
  return Number.isFinite(number) ? {text: format(number), invalid: false} : converted;
}

/** Small observable store so edits survive closing and reopening the table modal. */
export function createResidentColumnEditStore() {
  const edits = new Map<string, ResidentColumnEdits>();
  const listeners = new Set<() => void>();
  return {
    get: (datasetId: string): ResidentColumnEdits =>
      edits.get(datasetId) ?? EMPTY_RESIDENT_COLUMN_EDITS,
    set(datasetId: string, next: ResidentColumnEdits) {
      edits.set(datasetId, next);
      for (const listener of listeners) listener();
    },
    clear(datasetId: string) {
      edits.delete(datasetId);
      for (const listener of listeners) listener();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    }
  };
}
export const residentColumnEditStore = createResidentColumnEditStore();
