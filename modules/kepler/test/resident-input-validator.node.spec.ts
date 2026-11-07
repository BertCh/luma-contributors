// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  createResidentInputValidator,
  markResidentArrowInput
} from '../../../examples/deck/kepler-compute/src/resident-input-validator';

function makeInput() {
  const source = new arrow.Table({
    identifier: arrow.makeVector(new Uint32Array([4294967293, 4294967294, 4294967295])),
    timestamp: arrow.vectorFromArray(
      [new Date('2026-01-01'), null, new Date('2026-01-02')],
      new arrow.TimestampMillisecond('UTC')
    ),
    label: arrow.vectorFromArray(['8928308280fffff', '8928308280fffff', null], new arrow.Utf8())
  });
  return {
    arrowTable: source,
    fields: [
      {name: 'identifier', type: 'integer', format: ''},
      {name: 'timestamp', type: 'timestamp', format: 'x'},
      {name: 'label', type: 'string', format: ''}
    ],
    cols: source.schema.fields.map(field => source.getChild(field.name)!),
    rows: []
  };
}

test('published Arrow metadata preserves full Uint32, exact time roles and H3-looking string categories without inference', () => {
  const input = makeInput();
  const reads = input.cols.map(column => vi.spyOn(column, 'get'));
  const nativeValidator = vi.fn(() => {
    throw new Error('Native sampling must not run for prepared Arrow');
  });
  const validate = createResidentInputValidator(nativeValidator);
  const marked = markResidentArrowInput(input);
  for (const payload of [marked, {...marked}]) {
    const result = validate(payload)!;
    expect(result.fields.map(field => [field.type, field.analyzerType, field.format])).toEqual([
      ['integer', 'INT', ''],
      ['timestamp', 'DATETIME', 'x'],
      ['string', 'STRING', '']
    ]);
    expect(result.cols).toBe(input.cols);
    expect(result.rows).toBe(input.rows);
    expect(result.fields.map(field => field.name)).toEqual(input.fields.map(field => field.name));
  }
  for (const read of reads) {
    expect(read).not.toHaveBeenCalled();
    read.mockRestore();
  }
  expect(nativeValidator).not.toHaveBeenCalled();
});

test('ordinary native ingestion keeps its validator and malformed explicit schemas reject before publication', () => {
  const input = makeInput();
  const nativeValidator = vi.fn(() => null);
  const validate = createResidentInputValidator(nativeValidator);
  expect(validate(input)).toBeNull();
  expect(nativeValidator).toHaveBeenCalledExactlyOnceWith(input);
  const marked = markResidentArrowInput(input);
  expect(() => validate({...marked, fields: [...marked.fields].reverse()})).toThrow(
    'source schema'
  );
  expect(nativeValidator).toHaveBeenCalledTimes(1);
});
