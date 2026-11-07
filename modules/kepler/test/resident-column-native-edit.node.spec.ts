// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, test} from 'vitest';
import {
  changeResidentColumnTypeNative,
  deleteResidentColumnNative,
  moveResidentColumnEdits,
  renameResidentColumnNative,
  type ResidentNativeColumnEditHost
} from '../../../examples/deck/kepler-compute/src/resident-column-native-edit';
import {
  applyColumnEditsToArrowTable,
  wrapPageSourceWithColumnEdits
} from '../../../examples/deck/kepler-compute/src/resident-export-column-edits';
import type {ResidentColumnEdits} from '../../../examples/deck/kepler-compute/src/resident-column-edits';

const EDITS: ResidentColumnEdits = {
  names: {a: 'Alpha'},
  deleted: ['c'],
  types: {a: 'real'},
  formats: {a: ',.2f'}
};

function makeHost(overrides: Partial<ResidentNativeColumnEditHost> = {}) {
  let fields = [
    {name: 'a', type: 'real'},
    {name: 'b', type: 'integer'}
  ];
  const calls: string[] = [];
  const host: ResidentNativeColumnEditHost = {
    maximumRows: 100,
    getDataset: () => ({fields, dataContainer: {numRows: () => 10, getColumn: () => undefined}}),
    renameTableColumn: (_id, name, newName) => {
      calls.push(`rename:${name}:${newName}`);
      fields = fields.map(field => (field.name === name ? {...field, name: newName} : field));
    },
    changeTableColumnType: (_id, name, type) => {
      calls.push(`type:${name}:${type}`);
      fields = fields.map(field => (field.name === name ? {...field, type} : field));
    },
    requestRepublish: id => calls.push(`republish:${id}`),
    ...overrides
  };
  return {host, calls};
}

describe('resident native column edits', () => {
  test('rename dispatches Kepler, moves display edits and republishes', () => {
    const {host, calls} = makeHost();
    const outcome = renameResidentColumnNative(host, 'd', EDITS, 'a', 'alpha2');
    expect(outcome.ok).toBe(true);
    expect(calls).toEqual(['rename:a:alpha2', 'republish:d']);
    if (outcome.ok) {
      expect(outcome.edits.names).toEqual({});
      expect(outcome.edits.types).toEqual({alpha2: 'real'});
      expect(outcome.edits.formats).toEqual({alpha2: ',.2f'});
    }
  });

  test('rename is refused for duplicates and non-Arrow datasets without dispatching', () => {
    const {host, calls} = makeHost();
    expect(renameResidentColumnNative(host, 'd', EDITS, 'a', 'b').ok).toBe(false);
    const lazy = makeHost({getDataset: () => ({fields: [], dataContainer: {numRows: () => 1}})});
    expect(renameResidentColumnNative(lazy.host, 'd', EDITS, 'a', 'z').ok).toBe(false);
    expect(calls).toEqual([]);
    expect(lazy.calls).toEqual([]);
  });

  test('type change reports a refused conversion and clears stale format', () => {
    const {host} = makeHost({changeTableColumnType: () => undefined});
    expect(changeResidentColumnTypeNative(host, 'd', EDITS, 'a', 'string').ok).toBe(false);
    const accepted = makeHost();
    const outcome = changeResidentColumnTypeNative(accepted.host, 'd', EDITS, 'a', 'string');
    expect(outcome.ok).toBe(true);
    if (outcome.ok) expect(outcome.edits.formats).toEqual({});
  });

  test('moveResidentColumnEdits renames deleted entries', () => {
    expect(moveResidentColumnEdits(EDITS, 'c', 'z').deleted).toEqual(['z']);
  });

  test('exports omit deleted columns and use aliases', async () => {
    const table = new arrow.Table({
      a: arrow.vectorFromArray([1, 2], new arrow.Float64()),
      b: arrow.vectorFromArray([3, 4], new arrow.Float64()),
      c: arrow.vectorFromArray([5, 6], new arrow.Float64())
    });
    const edited = applyColumnEditsToArrowTable(table, EDITS);
    expect(edited.schema.fields.map(field => field.name)).toEqual(['Alpha', 'b']);
    const wrapped = wrapPageSourceWithColumnEdits(
      {getTotalRows: () => 2, readPage: async () => table},
      () => EDITS
    );
    expect((await wrapped.readPage(0, 2, undefined)).schema.fields.map(f => f.name)).toEqual([
      'Alpha',
      'b'
    ]);
  });
});

describe('resident native column delete', () => {
  test('dispatches Kepler delete, drops display edits and republishes', () => {
    const calls: string[] = [];
    let fields = [
      {name: 'a', type: 'real'},
      {name: 'b', type: 'integer'}
    ];
    const host: ResidentNativeColumnEditHost = {
      maximumRows: 100,
      getDataset: () => ({fields, dataContainer: {numRows: () => 10, getColumn: () => undefined}}),
      deleteTableColumn: (_id, name) => {
        calls.push(`delete:${name}`);
        fields = fields.filter(field => field.name !== name);
      },
      requestRepublish: id => calls.push(`republish:${id}`)
    };
    const outcome = deleteResidentColumnNative(host, 'd', EDITS, 'a');
    expect(outcome.ok).toBe(true);
    expect(calls).toEqual(['delete:a', 'republish:d']);
    if (outcome.ok) {
      expect(outcome.edits.names).toEqual({});
      expect(outcome.edits.types).toEqual({});
      expect(outcome.edits.deleted).toEqual(['c']);
    }
  });

  test('refuses with a reason so the caller can hide the column', () => {
    const host: ResidentNativeColumnEditHost = {
      maximumRows: 100,
      getDataset: () => ({
        fields: [
          {name: 'a', type: 'real'},
          {name: 'b', type: 'real'}
        ],
        dataContainer: {numRows: () => 10, getColumn: () => undefined}
      }),
      deleteTableColumn: () => {
        throw new Error('must not run');
      },
      getDeleteRefusal: () => 'A layer uses it.'
    };
    expect(deleteResidentColumnNative(host, 'd', EDITS, 'a')).toEqual({
      ok: false,
      reason: 'A layer uses it.'
    });
    expect(
      deleteResidentColumnNative(
        {
          ...host,
          getDeleteRefusal: undefined,
          getDataset: () => ({fields: [{name: 'a', type: 'real'}]})
        },
        'd',
        EDITS,
        'a'
      ).ok
    ).toBe(false);
  });
});
