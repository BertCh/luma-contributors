// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {latLngToCell} from 'h3-js';
import {preparePublishedH3} from '../../../examples/deck/kepler-compute/src/source-publication';

describe('H3 publication admission', () => {
  it('retains exact IDs, missing and invalid rows, and original batch identities', () => {
    const identity = BigInt(`0x${latLngToCell(40.74, -74.01, 7)}`);
    const table = new arrow.Table({
      cell: arrow.vectorFromArray([null, identity], new arrow.Uint64())
    }).concat(new arrow.Table({cell: arrow.vectorFromArray([0n, identity], new arrow.Uint64())}));
    const prepared = preparePublishedH3(table, 'cell', ['source-one', 'source-two']);
    expect(prepared.batches.map(batch => batch.id)).toEqual(['source-one', 'source-two']);
    expect(prepared.batches.map(batch => batch.rowCount)).toEqual([2, 2]);
    expect(prepared.batches[0].sourceIds[1]).toBe(identity);
    expect(prepared.validRowCount).toBe(2);
    expect(prepared.nullRowCount).toBe(1);
    expect(prepared.invalidRows.map(row => row.sourceRowIndex)).toEqual([2]);
  });

  it('rejects Number identity input before upload', () => {
    const table = arrow.tableFromArrays({
      cell: new Float64Array([Number(BigInt('0x872a1072dffffff'))])
    });
    expect(() => preparePublishedH3(table, 'cell')).toThrow();
  });

  it('admits valid cells whose Float32 geometry collapses (fp64 hi/lo column path)', () => {
    const table = new arrow.Table({
      cell: arrow.vectorFromArray([BigInt('0x8fdbb16cc2e02d9')], new arrow.Uint64())
    });
    expect(preparePublishedH3(table, 'cell').validRowCount).toBe(1);
  });
});
