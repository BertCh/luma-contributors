// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {aggregateCellsOnCPU} from './cell-aggregation-oracle';

// Test-side host reference (there is no production CPU contributor twin).
// Resolution-2 tiles (0,0),(1,0),(2,0) have Morton prefixes 0000,0001,0100.
// A,A,B,C,C,C contributes [2,1,3]; dividing tile coordinates by two gives
// parent counts [3,3]; the world contains all six. These are hand cardinalities.
const keys = [
  0x4820ffffffffffffn,
  0x4820ffffffffffffn,
  0x4821ffffffffffffn,
  0x4824ffffffffffffn,
  0x4824ffffffffffffn,
  0x4824ffffffffffffn
];

for (const frame of [
  {name: 'all', mask: [1, 1, 1, 1, 1, 1], counts: [[2, 1, 3], [3, 3], [6]]},
  {name: 'one', mask: [0, 0, 0, 1, 1, 1], counts: [[3], [3], [3]]},
  {name: 'empty', mask: [0, 0, 0, 0, 0, 0], counts: [[], [], []]}
]) {
  it(`host cell reference follows independent tile cardinalities (${frame.name})`, () => {
    for (const [level, resolution] of [2, 1, 0].entries()) {
      const result = aggregateCellsOnCPU({
        family: 'quadbin',
        resolution,
        keys,
        mask: Uint32Array.from(frame.mask),
        sumScale: 65536
      });
      expect(result.map(row => row.count)).toEqual(frame.counts[level]);
      expect(result.length).toBe(frame.counts[level].length);
    }
  });
}
