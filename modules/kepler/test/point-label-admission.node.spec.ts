// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {createRequire} from 'node:module';
import * as arrow from 'apache-arrow';
import {
  prepareResidentPointLabels,
  prepareStreamedPointLabels
} from '../../../examples/deck/kepler-compute/src/point-label-admission';
import {
  makePointLabelFormatter,
  setPointLabelFormatterFactory
} from '../../../examples/deck/kepler-compute/src/label-format';

const exampleRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
// The utils barrel loads deck.gl 9; the same native formatter's module is deck-free.
const native = createRequire(exampleRequire.resolve('@kepler.gl/utils'))('./data-utils');

describe('resident point label admission', () => {
  it.each([
    'plain',
    'dictionary'
  ])('formats pre-epoch %s timestamps consistently in resident and streamed labels', storage => {
    const timestamps = arrow.makeVector(
      arrow.makeData({
        type: new arrow.TimestampNanosecond('UTC'),
        data: new BigInt64Array([-1n, 0n]),
        length: 2
      })
    );
    const source = new arrow.Table({
      label:
        storage === 'plain'
          ? timestamps
          : arrow.makeVector(
              arrow.makeData({
                type: new arrow.Dictionary(timestamps.type, new arrow.Int32()),
                data: new Int32Array([0, 1]),
                length: 2,
                dictionary: timestamps
              })
            )
    });
    setPointLabelFormatterFactory((format, field) => native.getFormatter(format, field));
    try {
      const format = makePointLabelFormatter({
        field: {name: 'label', type: 'timestamp', displayFormat: 'YYYY-MM-DD'}
      })!;
      expect(format(-1)).toBe('1969-12-31');
      expect(format(0)).toBe('1970-01-01');
      const batches = [{id: 'time', rowCount: 2}];
      const resident = prepareResidentPointLabels(source, batches, 'label', format).chunks[0];
      const residentText = [0, 1].map(row =>
        String.fromCodePoint(
          ...resident.codePoints.subarray(resident.rowOffsets[row], resident.rowOffsets[row + 1])
        )
      );
      expect.soft(residentText).toEqual(['1969-12-31', '1970-01-01']);
      const streamed = prepareStreamedPointLabels(source, batches, 'label', format)[0];
      expect.soft([0, 1].map(row => streamed.getText(row))).toEqual(['1969-12-31', '1970-01-01']);
    } finally {
      setPointLabelFormatterFactory(null);
    }
  });
  it('preserves empty chunks, null rows, Unicode, and exact integer scalar text', () => {
    const first = new arrow.Table({
      label: arrow.vectorFromArray(['unused', 'Å🙂', null], new arrow.Utf8())
    }).slice(1);
    const empty = new arrow.Table({label: arrow.vectorFromArray([], new arrow.Utf8())});
    const last = new arrow.Table({label: arrow.vectorFromArray(['last'], new arrow.Utf8())});
    const source = first.concat(empty, last);
    const prepared = prepareResidentPointLabels(
      source,
      [
        {id: 'first', rowCount: 2},
        {id: 'empty', rowCount: 0},
        {id: 'last', rowCount: 1}
      ],
      'label'
    );
    expect(prepared.chunks.map(chunk => [chunk.id, chunk.rowCount, chunk.rowIndexBase])).toEqual([
      ['first', 2, 0],
      ['empty', 0, 2],
      ['last', 1, 2]
    ]);
    expect([...prepared.chunks[0].rowOffsets]).toEqual([0, 2, 2]);
    expect([...prepared.chunks[0].codePoints]).toEqual([197, 128578]);
    const integers = new arrow.Table({
      label: arrow.vectorFromArray([9007199254740993n], new arrow.Int64())
    });
    const integerLabels = prepareResidentPointLabels(
      integers,
      [{id: 'integer', rowCount: 1}],
      'label'
    );
    expect(String.fromCodePoint(...integerLabels.chunks[0].codePoints)).toBe('9007199254740993');
  });

  it('rejects oversized UTF-8 before Arrow decodes a scalar cell', () => {
    const source = new arrow.Table({
      label: arrow.vectorFromArray(['x'.repeat(2048)], new arrow.Utf8())
    });
    const column = source.batches[0].getChild('label')!;
    const get = vi.spyOn(column, 'get');
    vi.spyOn(source.batches[0], 'getChild').mockReturnValue(column);
    expect(() =>
      prepareResidentPointLabels(source, [{id: 'oversized', rowCount: 1}], 'label')
    ).toThrow('UTF-8');
    expect(get).not.toHaveBeenCalled();
  });

  it('accepts timestamp columns and rejects incomplete batch identity without coercion', () => {
    const source = new arrow.Table({
      label: arrow.vectorFromArray([new Date(0)], new arrow.TimestampMillisecond())
    });
    expect(() =>
      prepareResidentPointLabels(source, [{id: 'time', rowCount: 1}], 'label')
    ).not.toThrow();
    const strings = new arrow.Table({label: arrow.vectorFromArray(['label'], new arrow.Utf8())});
    expect(() =>
      prepareResidentPointLabels(strings, [{id: 'wrong', rowCount: 2}], 'label')
    ).toThrow('source rows');
  });
});
