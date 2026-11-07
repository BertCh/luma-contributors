// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import type {ViewerColumnReaders} from './standalone-viewer-filters';

/** Cached column readers over one decoded Arrow table (CPU only, no GPU residency). */
export type ViewerDataset = ViewerColumnReaders & {
  datasetId: string;
  fieldNames: string[];
  readValue(fieldName: string, row: number): unknown;
};

export function createViewerDataset(datasetId: string, table: arrow.Table): ViewerDataset {
  const numbers = new Map<string, Float64Array | null>();
  const strings = new Map<string, (string | null)[] | null>();
  return {
    datasetId,
    rowCount: table.numRows,
    fieldNames: table.schema.fields.map(field => field.name),
    readNumbers(fieldName) {
      if (!numbers.has(fieldName)) {
        const vector = table.getChild(fieldName);
        if (!vector) numbers.set(fieldName, null);
        else {
          const values = new Float64Array(table.numRows);
          for (let row = 0; row < values.length; row++) {
            const value = vector.get(row) as unknown;
            // Arrow JS already normalises Timestamp and Date to epoch milliseconds.
            values[row] =
              value === null || value === undefined
                ? NaN
                : value instanceof Date
                  ? value.getTime()
                  : typeof value === 'boolean'
                    ? Number(value)
                    : Number(value);
          }
          numbers.set(fieldName, values);
        }
      }
      return numbers.get(fieldName)!;
    },
    readStrings(fieldName) {
      if (!strings.has(fieldName)) {
        const vector = table.getChild(fieldName);
        if (!vector) strings.set(fieldName, null);
        else {
          const values: (string | null)[] = new Array(table.numRows);
          for (let row = 0; row < values.length; row++) {
            const value = vector.get(row) as unknown;
            values[row] = value === null || value === undefined ? null : String(value);
          }
          strings.set(fieldName, values);
        }
      }
      return strings.get(fieldName)!;
    },
    readValue(fieldName, row) {
      return table.getChild(fieldName)?.get(row);
    }
  };
}
