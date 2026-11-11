// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  assertNotResidentSource,
  selectRangeDomain
} from '../../../examples/deck/kepler-compute/src/resident-source-guards';
import {
  estimateResidentReadbackCpuByteLength,
  isNonFiniteFloatBits,
  makeResidentReadbackTable,
  MAXIMUM_RESIDENT_READBACK_ROW_COUNT,
  shouldReadBackResidentResult,
  type ResidentReadbackColumn
} from '../../../examples/deck/kepler-compute/src/resident-result-readback';

const floatWords = (values: number[]) => new Uint32Array(new Float32Array(values).buffer);

describe('size-gated resident result publication', () => {
  test('reads back at or below the cap and stays resident above it', () => {
    expect(MAXIMUM_RESIDENT_READBACK_ROW_COUNT).toBe(262144);
    expect(shouldReadBackResidentResult(1)).toBe(true);
    expect(shouldReadBackResidentResult(262144)).toBe(true);
    expect(shouldReadBackResidentResult(262145)).toBe(false);
  });

  // Skipped: resident-result-readback.ts needs prepareArrowTableForGPUAnalytics, which published
  // @luma.gl/graph/arrow 10.0.0-alpha.10 does not export.
  test.skip('builds one Arrow batch per GPU chunk with nulls, dictionaries and metadata', () => {
    const columns: ResidentReadbackColumn[] = [
      {
        name: 'label',
        format: 'uint32',
        dictionary: ['a', 'b'],
        chunks: [
          {values: Uint32Array.from([1, 0]), validity: Uint32Array.from([1, 0])},
          {values: Uint32Array.from([0]), validity: Uint32Array.from([1])}
        ]
      },
      {
        name: 'score',
        format: 'float32',
        chunks: [
          {values: floatWords([0.5, 9]), validity: Uint32Array.from([1, 0])},
          {values: floatWords([2]), validity: Uint32Array.from([1])}
        ]
      },
      {
        name: 'delta',
        format: 'sint32',
        chunks: [
          {
            values: new Uint32Array(Int32Array.from([-3, 4]).buffer),
            validity: Uint32Array.from([1, 1])
          },
          {values: new Uint32Array(Int32Array.from([5]).buffer), validity: Uint32Array.from([1])}
        ]
      }
    ];
    const result = makeResidentReadbackTable({columns, metadata: {'luma.gl.join': '{}'}});
    expect(result.table.numRows).toBe(3);
    expect(result.table.batches.map(batch => batch.numRows)).toEqual([2, 1]);
    expect(result.prepared.sourceBatches.map(batch => batch.sourceRowCount)).toEqual([2, 1]);
    expect(result.table.schema.metadata.get('luma.gl.join')).toBe('{}');
    expect(result.table.getChild('label')!.toJSON()).toEqual(['b', null, 'a']);
    expect(result.table.getChild('score')!.toJSON()).toEqual([0.5, null, 2]);
    expect(result.table.getChild('delta')!.toJSON()).toEqual([-3, 4, 5]);
    expect(result.prepared.dictionaries.label.values).toEqual(['a', 'b']);
  });

  // Skipped: resident-result-readback.ts needs prepareArrowTableForGPUAnalytics, which published
  // @luma.gl/graph/arrow 10.0.0-alpha.10 does not export.
  test.skip('refuses invalid validity, dictionary codes and non-finite aggregates', () => {
    const build = (column: ResidentReadbackColumn) =>
      makeResidentReadbackTable({columns: [column], metadata: {}});
    expect(() =>
      build({
        name: 'x',
        format: 'uint32',
        chunks: [{values: Uint32Array.from([1]), validity: Uint32Array.from([2])}]
      })
    ).toThrow();
    expect(() =>
      build({
        name: 'x',
        format: 'uint32',
        dictionary: ['a'],
        chunks: [{values: Uint32Array.from([1]), validity: Uint32Array.from([1])}]
      })
    ).toThrow();
    // An out-of-range code on an invalid row is a null, not an error.
    expect(() =>
      build({
        name: 'x',
        format: 'uint32',
        dictionary: ['a'],
        chunks: [{values: Uint32Array.from([7]), validity: Uint32Array.from([0])}]
      })
    ).not.toThrow();
    for (const bad of [NaN, Infinity, -Infinity])
      expect(() =>
        build({
          name: 'x',
          format: 'float32',
          refuseNonFinite: true,
          chunks: [{values: floatWords([bad]), validity: Uint32Array.from([1])}]
        })
      ).toThrow();
    // Invalid rows and unflagged columns are not refused.
    expect(() =>
      build({
        name: 'x',
        format: 'float32',
        refuseNonFinite: true,
        chunks: [{values: floatWords([NaN]), validity: Uint32Array.from([0])}]
      })
    ).not.toThrow();
    expect(isNonFiniteFloatBits(floatWords([NaN])[0])).toBe(true);
    expect(isNonFiniteFloatBits(floatWords([1e38])[0])).toBe(false);
  });

  test('estimates a CPU reservation that grows with rows and dictionary text', () => {
    const small = estimateResidentReadbackCpuByteLength([{}], 10);
    expect(estimateResidentReadbackCpuByteLength([{}], 20)).toBeGreaterThan(small);
    expect(estimateResidentReadbackCpuByteLength([{dictionary: ['abcd']}], 10)).toBeGreaterThan(
      small
    );
  });

  test('source guards apply only to GPU-resident datasets', () => {
    const readBackValue = {};
    const residentValue = {resident: {domains: {score: [1, 2] as [number, number]}}};
    expect(() => assertNotResidentSource(readBackValue, 'export')).not.toThrow();
    expect(() => assertNotResidentSource(residentValue, 'export')).toThrow();
    let scanned = 0;
    const scan = () => {
      scanned++;
      return [0, 9] as [number, number];
    };
    expect(selectRangeDomain(readBackValue, 'score', scan)).toEqual([0, 9]);
    expect(scanned).toBe(1);
    expect(selectRangeDomain(residentValue, 'score', scan)).toEqual([1, 2]);
    expect(scanned).toBe(1);
  });
});
