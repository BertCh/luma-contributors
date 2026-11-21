// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '../../src/analysis-core/graph-support/arrow-analytics-normalization';

// Column names that collide with Object.prototype members. Computed keys define own properties,
// including `__proto__`, so these literals never reach the prototype setter.
const PROTOTYPE_NAMES = ['constructor', 'toString', '__proto__', 'hasOwnProperty'];

function makeInt64Table(name: string) {
  const vector = arrow.makeVector({
    type: new arrow.Int64(),
    data: new BigInt64Array([42n]),
    length: 1
  } as never);
  return new arrow.Table({[name]: vector});
}

function makeUtf8Table(name: string) {
  return new arrow.Table({[name]: arrow.vectorFromArray(['Oak'], new arrow.Utf8())});
}

describe('Arrow preparation reads per-column options as own properties only', () => {
  test.each(PROTOTYPE_NAMES)('DPC-104: empty timeColumns keeps Int64 "%s" scalar', name => {
    const table = makeInt64Table(name);
    expect(table.schema.fields.map(field => field.name)).toEqual([name]);
    for (const timeColumns of [{}, Object.create(null)]) {
      const prepared = prepareArrowTableForGPUAnalytics(table, {
        float64: {mode: 'exact'},
        timeColumns,
        dictionaries: {}
      });
      expect(prepared.table.schema.fields.map(field => field.name)).toEqual([name]);
      expect(Object.keys(prepared.timeColumns)).toEqual([]);
      expect(prepared.table.getChildAt(0)?.get(0)).toBe(42);
    }
  });

  test.each(PROTOTYPE_NAMES)('DPC-104: an explicit own unit still declares "%s" as time', name => {
    const prepared = prepareArrowTableForGPUAnalytics(makeInt64Table(name), {
      timeColumns: {[name]: 'microsecond'}
    });
    expect(prepared.table.schema.fields).toEqual([]);
    expect(Object.keys(prepared.timeColumns)).toEqual([name]);
    expect(prepared.timeColumns[name].unit).toBe('microsecond');
    expect(Array.from(prepared.timeColumns[name].chunks[0])).toEqual([42, 0]);
  });

  test.each(PROTOTYPE_NAMES)('DPC-115: empty dictionaries accept UTF-8 "%s"', name => {
    const table = makeUtf8Table(name);
    for (const dictionaries of [{}, Object.create(null)]) {
      const prepared = prepareArrowTableForGPUAnalytics(table, {dictionaries, timeColumns: {}});
      expect(prepared.dictionaries[name]).toEqual({values: ['Oak'], ordered: false});
      expect(prepared.table.getChildAt(0)?.get(0)).toBe('Oak');
    }
  });

  test.each(PROTOTYPE_NAMES)('DPC-115: an explicit own dictionary for "%s" is honoured', name => {
    const prepared = prepareArrowTableForGPUAnalytics(makeUtf8Table(name), {
      dictionaries: {[name]: {values: ['Elm'], ordered: false}}
    });
    expect(prepared.dictionaries[name]).toEqual({values: ['Elm', 'Oak'], ordered: false});
  });
});
