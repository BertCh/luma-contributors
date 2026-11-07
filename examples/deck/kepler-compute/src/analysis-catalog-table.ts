// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {CatalogOutputTable} from '@luma-contributors/kepler';

/** Typed Arrow conversion; column types come from the binding's declared output schema. */
export function makeArrowTableFromCatalogTable(table: CatalogOutputTable): arrow.Table {
  const columns: Record<string, arrow.Vector> = {};
  for (const [name, type] of Object.entries(table.columnTypes)) {
    const values = table.columns[name];
    columns[name] =
      type === 'string'
        ? arrow.vectorFromArray(
            values.map(value => (value === null ? null : String(value))),
            new arrow.Utf8()
          )
        : type === 'int32'
          ? arrow.vectorFromArray(
              values.map(value => (value === null ? null : Number(value))),
              new arrow.Int32()
            )
          : type === 'boolean'
            ? arrow.vectorFromArray(
                values.map(value => (value === null ? null : Boolean(value))),
                new arrow.Bool()
              )
            : arrow.vectorFromArray(
                values.map(value => (value === null ? null : Number(value))),
                new arrow.Float64()
              );
  }
  return new arrow.Table(columns);
}
