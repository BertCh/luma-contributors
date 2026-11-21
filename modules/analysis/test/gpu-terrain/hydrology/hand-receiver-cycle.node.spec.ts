// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {executeHANDPublication} from './hand-publication-fixture';

/**
 * Transcription of the in-place pointer-doubling rounds of `createTerrainFlowPointerNodes`
 * (`pointer[c] = pointer[pointer[c]]`, cells visited in `order`) until a round changes nothing.
 */
function runInPlaceDoubling(initial: number[], order: number[]): number[] {
  const pointer = [...initial];
  for (let changed = true; changed; ) {
    changed = false;
    for (const cell of order) {
      const next = pointer[pointer[cell]];
      if (next !== pointer[cell]) {
        pointer[cell] = next;
        changed = true;
      }
    }
  }
  return pointer;
}

it('in-place doubling collapses a two-cell receiver cycle to a self-pointer (RT-123)', () => {
  // 2 x 1 ESRI directions [1 (east), 16 (west)]: cell 0 drains to 1 and cell 1 to 0, so the
  // initial pointers are the receivers [1, 0]. Neither cell is a stop or lacks a receiver.
  expect(runInPlaceDoubling([1, 0], [0, 1])).toEqual([0, 0]);
  expect(runInPlaceDoubling([1, 0], [1, 0])).toEqual([1, 1]);
});

for (const [directionEncoding, directions] of [
  ['esri', [1, 16]],
  ['whitebox', [2, 32]],
  ['taudem', [1, 5]]
] as const) {
  it(`HAND rejects collapsed receiver cycles in production publication: ${directionEncoding} (MRG-05)`, () => {
    for (const pointer of [
      [0, 0],
      [1, 1]
    ]) {
      const result = executeHANDPublication({
        directionEncoding,
        directions: [...directions],
        pointer,
        streams: [0, 0]
      });
      // A cycle reaches no first stream, regardless of which self-pointer doubling produces.
      expect(result).toEqual({
        drainage: [0xfffffffe, 0xfffffffe],
        heights: [NaN, NaN],
        converged: 0
      });
    }
  });
}
