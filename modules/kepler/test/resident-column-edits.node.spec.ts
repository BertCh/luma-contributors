// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  EMPTY_RESIDENT_COLUMN_EDITS,
  applyResidentColumnEdits,
  changeResidentColumnType,
  convertResidentCell,
  createResidentColumnEditStore,
  deleteResidentColumn,
  formatResidentCell,
  renameResidentColumn,
  restoreResidentColumn,
  setResidentColumnFormat
} from '../../../examples/deck/kepler-compute/src/resident-column-edits';

const columns = ['a', 'b', 'c'];
const fields = columns.map(name => ({name, sortable: true}));

function unwrap(result: ReturnType<typeof renameResidentColumn>) {
  if (!result.ok) throw new Error(result.reason);
  return result.edits;
}

test('rename changes the label but not the identity, and rejects duplicates and empties', () => {
  const edits = unwrap(renameResidentColumn(EMPTY_RESIDENT_COLUMN_EDITS, columns, 'a', ' Alpha '));
  expect(applyResidentColumnEdits(fields, edits).map(field => [field.name, field.label])).toEqual([
    ['a', 'Alpha'],
    ['b', undefined],
    ['c', undefined]
  ]);
  expect(renameResidentColumn(edits, columns, 'b', 'Alpha')).toEqual({
    ok: false,
    reason: 'Another column is already named "Alpha".'
  });
  expect(renameResidentColumn(edits, columns, 'b', '  ').ok).toBe(false);
  // Renaming back to the source name clears the override.
  expect(unwrap(renameResidentColumn(edits, columns, 'a', 'a')).names).toEqual({});
});

test('delete hides a column, keeps the last one, and restore undoes it', () => {
  let edits = unwrap(deleteResidentColumn(EMPTY_RESIDENT_COLUMN_EDITS, columns, 'a'));
  edits = unwrap(deleteResidentColumn(edits, columns, 'b'));
  expect(applyResidentColumnEdits(fields, edits).map(field => field.name)).toEqual(['c']);
  expect(deleteResidentColumn(edits, columns, 'c').ok).toBe(false);
  edits = restoreResidentColumn(edits, 'a');
  expect(applyResidentColumnEdits(fields, edits).map(field => field.name)).toEqual(['a', 'c']);
});

test('changing the type drops the previous type format; formats set and clear', () => {
  let edits = setResidentColumnFormat(EMPTY_RESIDENT_COLUMN_EDITS, 'a', ',');
  expect(edits.formats).toEqual({a: ','});
  edits = changeResidentColumnType(edits, 'a', 'integer');
  expect(edits.types).toEqual({a: 'integer'});
  expect(edits.formats).toEqual({});
  expect(changeResidentColumnType(edits, 'a', null).types).toEqual({});
  expect(
    setResidentColumnFormat(setResidentColumnFormat(edits, 'a', '.2f'), 'a', null).formats
  ).toEqual({});
});

test('cell conversion reports invalid values instead of guessing', () => {
  expect(convertResidentCell('12', 'integer')).toEqual({text: '12', invalid: false});
  expect(convertResidentCell('12.5', 'integer')).toEqual({text: 'invalid', invalid: true});
  expect(convertResidentCell(12.9, 'integer')).toEqual({text: '12', invalid: false});
  expect(convertResidentCell('x', 'real').invalid).toBe(true);
  expect(convertResidentCell('', 'real').invalid).toBe(true);
  expect(convertResidentCell(5, 'string')).toEqual({text: '5', invalid: false});
  expect(convertResidentCell(0, 'timestamp').text).toBe('1970-01-01T00:00:00.000Z');
  expect(convertResidentCell('2020-01-02T03:04:05Z', 'timestamp').text).toBe(
    '2020-01-02T03:04:05.000Z'
  );
  expect(convertResidentCell('nope', 'timestamp').invalid).toBe(true);
});

test('format applies after conversion with an injected formatter', () => {
  const formatter = (value: number | Date) =>
    `<${value instanceof Date ? value.getUTCFullYear() : value}>`;
  expect(formatResidentCell(1234.5, undefined, undefined).text).toBe('1234.5');
  expect(formatResidentCell('7', 'integer', formatter).text).toBe('<7>');
  expect(formatResidentCell(Date.UTC(2021, 0, 1), 'timestamp', formatter).text).toBe('<2021>');
  expect(formatResidentCell('x', 'real', formatter)).toEqual({text: 'invalid', invalid: true});
});

test('the store notifies subscribers and survives per dataset', () => {
  const store = createResidentColumnEditStore();
  let calls = 0;
  const unsubscribe = store.subscribe(() => calls++);
  expect(store.get('d')).toBe(EMPTY_RESIDENT_COLUMN_EDITS);
  store.set('d', setResidentColumnFormat(EMPTY_RESIDENT_COLUMN_EDITS, 'a', ','));
  expect(store.get('d').formats).toEqual({a: ','});
  expect(store.get('other')).toBe(EMPTY_RESIDENT_COLUMN_EDITS);
  store.clear('d');
  unsubscribe();
  store.set('d', EMPTY_RESIDENT_COLUMN_EDITS);
  expect(calls).toBe(2);
});
