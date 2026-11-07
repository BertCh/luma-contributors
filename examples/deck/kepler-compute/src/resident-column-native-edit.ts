// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentColumnDisplayType, ResidentColumnEdits} from './resident-column-edits';

/**
 * Real column rename and type change for resident datasets.
 *
 * Kepler's renameTableColumn and changeTableColumnType rewrite the dataset schema and reconcile
 * every layer column, visual channel field, filter, tooltip field and chart that named the column.
 * Resident GPU buffers are keyed by column name, so after Kepler commits the edit the resident
 * dataset is re-prepared from the edited Kepler dataset under the same id (the same replace flow as
 * a reload), which re-uploads the columns. When the Kepler dataset cannot be edited in place (no
 * Arrow-backed container, or a dataset larger than automatic preparation admits), the edit falls
 * back to the display alias with the reason stated; nothing is changed natively in that case.
 */
export type ResidentNativeDatasetSnapshot = {
  type?: string;
  fields: readonly {name: string; type: string}[];
  dataContainer?: {numRows(): number; getColumn?: unknown};
};

export type ResidentNativeColumnEditHost = {
  getDataset(datasetId: string): ResidentNativeDatasetSnapshot | undefined;
  /** Kepler renameTableColumn / changeTableColumnType dispatchers, already bound to the instance. */
  renameTableColumn?(datasetId: string, name: string, newName: string): void;
  changeTableColumnType?(datasetId: string, name: string, type: string): void;
  /** Kepler deleteTableColumn: drops the column and the filters, tooltip fields and layers needing it. */
  deleteTableColumn?(datasetId: string, name: string): void;
  /** Optional pre-check; returns a refusal reason (for example a layer still uses the column). */
  getDeleteRefusal?(datasetId: string, name: string): string | null;
  /** Re-prepares the resident dataset from the (edited) Kepler dataset under the same id. */
  requestRepublish?(datasetId: string): void;
  /** Rows above which automatic preparation refuses to materialize a native dataset. */
  maximumRows: number;
};

export type ResidentNativeEditOutcome =
  | {ok: true; mode: 'native'; edits: ResidentColumnEdits; message: string}
  | {ok: false; reason: string};

/** Why the dataset cannot be edited natively, or null when it can. */
export function getResidentNativeEditBlocker(
  host: ResidentNativeColumnEditHost,
  datasetId: string
): string | null {
  const dataset = host.getDataset(datasetId);
  if (!dataset) return 'The Kepler dataset is not available.';
  if (dataset.type) return 'Derived and computed datasets cannot be edited in place.';
  if (!dataset.dataContainer || typeof dataset.dataContainer.getColumn !== 'function')
    return 'The Kepler dataset is not Arrow-backed, so its columns cannot be projected in place.';
  if (dataset.dataContainer.numRows() > host.maximumRows)
    return `The dataset has more than ${host.maximumRows.toLocaleString('en-US')} rows, above automatic preparation.`;
  return null;
}

/** Moves display edits (alias, display type, format, deleted flag) from the old name to the new. */
export function moveResidentColumnEdits(
  edits: ResidentColumnEdits,
  name: string,
  newName: string
): ResidentColumnEdits {
  const move = <T>(record: Readonly<Record<string, T>>) => {
    const next = {...record};
    if (name in next) {
      next[newName] = next[name];
      delete next[name];
    }
    return next;
  };
  const names = move(edits.names);
  delete names[newName];
  return {
    names,
    types: move(edits.types),
    formats: move(edits.formats),
    deleted: edits.deleted.map(column => (column === name ? newName : column))
  };
}

export function renameResidentColumnNative(
  host: ResidentNativeColumnEditHost,
  datasetId: string,
  edits: ResidentColumnEdits,
  name: string,
  newName: string
): ResidentNativeEditOutcome {
  const trimmed = newName.trim();
  if (!host.renameTableColumn) return {ok: false, reason: 'Kepler rename is not available.'};
  const blocker = getResidentNativeEditBlocker(host, datasetId);
  if (blocker) return {ok: false, reason: blocker};
  if (!trimmed || trimmed === name) return {ok: false, reason: 'Enter a different column name.'};
  if (host.getDataset(datasetId)!.fields.some(field => field.name === trimmed))
    return {ok: false, reason: `Another column is already named "${trimmed}".`};
  host.renameTableColumn(datasetId, name, trimmed);
  // Kepler commits synchronously; a refused edit leaves the old field in place.
  if (!host.getDataset(datasetId)?.fields.some(field => field.name === trimmed))
    return {ok: false, reason: 'Kepler did not rename the column.'};
  host.requestRepublish?.(datasetId);
  return {
    ok: true,
    mode: 'native',
    edits: moveResidentColumnEdits(edits, name, trimmed),
    message: `Column renamed to "${trimmed}" in Kepler; layers, filters, tooltips and channels follow it and the resident dataset is being re-prepared.`
  };
}

export function changeResidentColumnTypeNative(
  host: ResidentNativeColumnEditHost,
  datasetId: string,
  edits: ResidentColumnEdits,
  name: string,
  type: ResidentColumnDisplayType
): ResidentNativeEditOutcome {
  if (!host.changeTableColumnType)
    return {ok: false, reason: 'Kepler type change is not available.'};
  const blocker = getResidentNativeEditBlocker(host, datasetId);
  if (blocker) return {ok: false, reason: blocker};
  const before = host.getDataset(datasetId)!.fields.find(field => field.name === name);
  if (!before) return {ok: false, reason: 'Unknown column.'};
  if (before.type === type) return {ok: false, reason: `The column is already ${type}.`};
  host.changeTableColumnType(datasetId, name, type);
  // A failed conversion leaves the type unchanged and Kepler reports the offending value.
  if (host.getDataset(datasetId)?.fields.find(field => field.name === name)?.type !== type)
    return {
      ok: false,
      reason: `Kepler could not convert "${name}" to ${type}; no values were changed (see the notification).`
    };
  host.requestRepublish?.(datasetId);
  const types = {...edits.types};
  const formats = {...edits.formats};
  delete types[name];
  delete formats[name];
  return {
    ok: true,
    mode: 'native',
    edits: {...edits, types, formats},
    message: `Column "${name}" converted to ${type} in Kepler; the resident dataset is being re-prepared.`
  };
}

/** Drops the display edits of a column that no longer exists in the dataset. */
export function dropResidentColumnEdits(
  edits: ResidentColumnEdits,
  name: string
): ResidentColumnEdits {
  const omit = <T>(record: Readonly<Record<string, T>>) => {
    const next = {...record};
    delete next[name];
    return next;
  };
  return {
    names: omit(edits.names),
    types: omit(edits.types),
    formats: omit(edits.formats),
    deleted: edits.deleted.filter(column => column !== name)
  };
}

export function deleteResidentColumnNative(
  host: ResidentNativeColumnEditHost,
  datasetId: string,
  edits: ResidentColumnEdits,
  name: string
): ResidentNativeEditOutcome {
  if (!host.deleteTableColumn) return {ok: false, reason: 'Kepler column delete is not available.'};
  const blocker = getResidentNativeEditBlocker(host, datasetId);
  if (blocker) return {ok: false, reason: blocker};
  const dataset = host.getDataset(datasetId)!;
  if (!dataset.fields.some(field => field.name === name))
    return {ok: false, reason: 'Unknown column.'};
  if (dataset.fields.length <= 1)
    return {ok: false, reason: 'Kepler keeps the last remaining column.'};
  const refusal = host.getDeleteRefusal?.(datasetId, name);
  if (refusal) return {ok: false, reason: refusal};
  host.deleteTableColumn(datasetId, name);
  // Kepler commits synchronously; a refused delete leaves the field in place.
  if (host.getDataset(datasetId)?.fields.some(field => field.name === name))
    return {ok: false, reason: 'Kepler did not delete the column.'};
  host.requestRepublish?.(datasetId);
  return {
    ok: true,
    mode: 'native',
    edits: dropResidentColumnEdits(edits, name),
    message: `Column "${name}" deleted from the Kepler dataset; dependent filters and tooltip fields were dropped and the resident dataset is being re-prepared.`
  };
}
