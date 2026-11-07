// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewerFilterSpec} from './standalone-viewer-config';

/** Pure CPU row filtering with Kepler's range, select, multiSelect and timeRange semantics. */

export type ViewerColumnReaders = {
  readNumbers(fieldName: string): ArrayLike<number> | null;
  readStrings(fieldName: string): ArrayLike<string | null> | null;
  rowCount: number;
};

/** True when the saved filter value constrains rows (Kepler ignores empty values). */
export function isActiveFilter(filter: ViewerFilterSpec): boolean {
  if (filter.type === 'polygon') return false;
  if (filter.type === 'multiSelect') return Array.isArray(filter.value) && filter.value.length > 0;
  if (filter.type === 'select') return typeof filter.value === 'boolean';
  return Array.isArray(filter.value) && filter.value.length === 2;
}

/** Row indices of a dataset that pass every active filter targeting it. */
export function selectFilteredRows(
  datasetId: string,
  filters: readonly ViewerFilterSpec[],
  readers: ViewerColumnReaders
): Uint32Array {
  const mask = new Uint8Array(readers.rowCount).fill(1);
  for (const filter of filters) {
    const fieldIndex = filter.datasetIds.indexOf(datasetId);
    if (fieldIndex < 0 || !isActiveFilter(filter)) continue;
    const fieldName = filter.fieldNames[fieldIndex] ?? filter.fieldNames[0];
    if (filter.type === 'multiSelect') {
      const allowed = new Set(filter.value as string[]);
      const values = readers.readStrings(fieldName);
      if (!values) continue;
      for (let row = 0; row < mask.length; row++) {
        const value = values[row];
        if (value === null || !allowed.has(value)) mask[row] = 0;
      }
    } else if (filter.type === 'select') {
      const values = readers.readNumbers(fieldName);
      if (!values) continue;
      const wanted = filter.value ? 1 : 0;
      for (let row = 0; row < mask.length; row++) if (values[row] !== wanted) mask[row] = 0;
    } else {
      const values = readers.readNumbers(fieldName);
      if (!values) continue;
      const [minimum, maximum] = filter.value as [number, number];
      for (let row = 0; row < mask.length; row++) {
        const value = values[row];
        if (!(value >= minimum && value <= maximum)) mask[row] = 0;
      }
    }
  }
  let count = 0;
  for (let row = 0; row < mask.length; row++) count += mask[row];
  const rows = new Uint32Array(count);
  let cursor = 0;
  for (let row = 0; row < mask.length; row++) if (mask[row]) rows[cursor++] = row;
  return rows;
}
