// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {chunkLines} from './line-chunk-oracle';
import {createFlatLines} from './line-segmentize-oracle';

it('the line-chunk oracle retains rows between shared and terminal cuts', () => {
  const lines = createFlatLines([
    [
      [0, 0],
      [0, 0],
      [2, 0],
      [2, 0],
      [4, 0],
      [4, 0]
    ]
  ]);

  // Measures are the x coordinates. The cuts at 0, 2 and 4 use source rows 0, 2 and 5.
  expect(chunkLines(lines, {mode: 'chunk', chunkLength: 2, lineCapacity: 2})).toEqual({
    pieceTotal: 2,
    lineOffsets: [0, 3, 7],
    sourceLineIndices: [0, 0],
    sourceRows: [0, 1, 1, 2, 3, 4, 4],
    measures: [0, 0, 2, 2, 2, 4, 4],
    positions: [
      [0, 0],
      [0, 0],
      [2, 0],
      [2, 0],
      [2, 0],
      [4, 0],
      [4, 0]
    ]
  });
});

it('the line-chunk oracle selects the final moving row at the published terminal measure', () => {
  const lines = createFlatLines([
    [
      [0, 0],
      [1, 0],
      [1, 2 ** -25]
    ]
  ]);
  // The true last measure is 1 + 2^-25, which rounds to 1. The cut at 1 uses row 2.
  expect(
    chunkLines(lines, {mode: 'substring', startMeasure: 0, endMeasure: 1, lineCapacity: 1})
  ).toMatchObject({
    lineOffsets: [0, 3],
    sourceRows: [0, 1, 1],
    positions: [
      [0, 0],
      [1, 0],
      [1, 2 ** -25]
    ]
  });
});
