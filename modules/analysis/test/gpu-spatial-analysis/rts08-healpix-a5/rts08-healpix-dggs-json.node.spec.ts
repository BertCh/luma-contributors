// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createHealpixDGGSJSON} from '../../../src/gpu-spatial-analysis/healpix-cells/index';

describe('createHealpixDGGSJSON', () => {
  it('writes the required DGGS-JSON members and a dense data array in nested order', () => {
    // Order 2 cells below the order 1 ancestor 3: nested indexes 12..15.
    const document = createHealpixDGGSJSON({
      order: 2,
      ancestorOrder: 1,
      ancestor: 3n,
      cells: [14n, 12n],
      properties: {
        count: {title: 'rows', values: [5, 2]},
        mean: {unit: 'C', values: [Number.NaN, 7.5]}
      }
    });
    // Required by dggs-json.json: dggrs, zoneId, depths, values.
    expect(document.dggrs).toBe('https://www.opengis.net/def/dggrs/OGC/1.0/HEALPix');
    expect(document.zoneId).toBe('healpix-nested:1:3');
    expect(document.depths).toEqual([1]);
    expect(document.values['count']).toEqual([
      {depth: 1, shape: {count: 4, subZones: 4}, data: [2, null, 5, null]}
    ]);
    expect(document.values['mean'][0].data).toEqual([7.5, null, null, null]);
    expect(document.schema.properties['mean']['x-ogc-unit']).toBe('C');
    expect(() => JSON.parse(JSON.stringify(document))).not.toThrow();
  });

  it('reads the UNIQ scheme and rejects cells outside the ancestor', () => {
    const uniq = createHealpixDGGSJSON({
      order: 1,
      scheme: 'uniq',
      ancestor: 1n,
      cells: [16n + 5n],
      properties: {n: {values: [1]}}
    });
    expect(uniq.values['n'][0].data).toEqual([null, 1, null, null]);
    expect(uniq.values['n'][0].shape.subZones).toBe(4 ** 1);
  });

  it('throws on a foreign cell, a bad ancestor and too deep a document', () => {
    expect(() =>
      createHealpixDGGSJSON({order: 2, ancestorOrder: 1, ancestor: 3n, cells: [0n], properties: {}})
    ).toThrow(/outside the ancestral zone/);
    expect(() =>
      createHealpixDGGSJSON({order: 1, ancestor: 12n, cells: [], properties: {}})
    ).toThrow(/ancestor is not a nested index/);
    expect(() => createHealpixDGGSJSON({order: 12, cells: [], properties: {}})).toThrow(
      /order - ancestorOrder/
    );
  });
});
