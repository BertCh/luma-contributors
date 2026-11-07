// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareResidentPointLabels} from '../../../examples/deck/kepler-compute/src/point-label-admission';

describe('resident point label admission', () => {
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
