// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {FontAtlas} from '@luma.gl/engine/text';
import {
  STREAMED_LABEL_LIMITS,
  collectVisibleRows,
  layoutVisibleLabels,
  measureLabelBounds,
  normalizeLabelText,
  scanLabelVocabulary,
  type StreamedLabelTextChunk
} from '../../../examples/deck/kepler-compute/src/label-streamed';

const noYield = async () => undefined;
const characters = ['A', 'B', 'C', 'D', '1', '2', ' '];
const fontAtlas: FontAtlas = {
  baselineOffset: 5,
  lineHeight: 64,
  width: 128,
  height: 64,
  pages: [],
  xOffset: 0,
  yOffsetMin: 0,
  yOffsetMax: 0,
  renderSettings: {mode: 'bitmap', threshold: 0, smoothing: 0},
  mapping: Object.fromEntries(
    characters.map((character, index) => [
      character,
      {
        x: index * 12,
        y: 0,
        width: 9,
        height: 14,
        anchorX: 4,
        anchorY: 10,
        advance: 10,
        layoutOffsetX: 0,
        layoutOffsetY: -10
      }
    ])
  )
};

function makeChunks(texts: string[][]): StreamedLabelTextChunk[] {
  return texts.map((rows, index) => ({
    id: `chunk-${index}`,
    rowCount: rows.length,
    getText: rowIndex => rows[rowIndex]
  }));
}

test('streamed labels normalize cells without throwing on tabs, CRLF or long text', () => {
  expect(normalizeLabelText('a\tb\r\nc')).toBe('a b\nc');
  expect(normalizeLabelText(null)).toBe('');
  expect(normalizeLabelText('x'.repeat(300))).toHaveLength(
    STREAMED_LABEL_LIMITS.maximumCharactersPerLabel
  );
});

test('vocabulary scan keeps the most frequent characters across windows and chunks', async () => {
  const chunks = makeChunks([
    ['AAAA', 'AB', 'C'],
    ['AAB', 'ZZ\nZ']
  ]);
  const vocabulary = await scanLabelVocabulary(chunks, {
    windowRows: 2,
    maximumVocabulary: 2,
    yieldControl: noYield
  });
  expect(vocabulary.characterSet).toEqual(['A', 'Z']);
  expect(vocabulary.distinctCount).toBe(4);
});

test('bounds are measured per window and rows outside the vocabulary get zero bounds', async () => {
  const chunks = makeChunks([['AB', 'A', 'AZ'], ['ABCD']]);
  const written: [number, number, number[]][] = [];
  const result = await measureLabelBounds(
    chunks,
    fontAtlas,
    new Set(['A', 'B', 'C', 'D']),
    (chunkIndex, firstRow, bounds) => written.push([chunkIndex, firstRow, Array.from(bounds)]),
    {windowRows: 2, yieldControl: noYield}
  );
  expect(result.hiddenRows).toBe(1);
  expect(written.map(([chunkIndex, firstRow]) => [chunkIndex, firstRow])).toEqual([
    [0, 0],
    [0, 2],
    [1, 0]
  ]);
  // Width is the advance sum, height the line height; the hidden row stays zero.
  expect(written[0][2]).toEqual([20, 64, 0, 0, 10, 64, 0, 0]);
  expect(written[1][2]).toEqual([0, 0, 0, 0]);
  expect(written[2][2]).toEqual([40, 64, 0, 0]);
});

test('visible rows are collected per chunk from the global words', () => {
  // Chunk A has 40 rows and chunk B 10: rows 3, 33 of A and 2 of B (global 42) are visible.
  const words = new Uint32Array(2);
  for (const global of [3, 33, 42]) words[global >>> 5] |= 1 << (global & 31);
  const visible = collectVisibleRows(words, [{rowCount: 40}, {rowCount: 10}]);
  expect(Array.from(visible[0])).toEqual([3, 33]);
  expect(Array.from(visible[1])).toEqual([2]);
});

test('only visible rows get glyph instances, keyed by chunk-local row and bounded by the glyph budget', () => {
  const texts = Array.from({length: 1000}, (_value, index) => (index % 2 ? 'AB' : 'CD1'));
  const [chunk] = makeChunks([texts]);
  const full = layoutVisibleLabels(chunk, Uint32Array.from([10, 501, 999]), fontAtlas, 100);
  expect(full.rowCount).toBe(3);
  expect(full.glyphCount).toBe(3 + 2 + 2);
  expect(Array.from(full.glyphRows).filter((_value, index) => index % 2 === 0)).toEqual([
    10, 10, 10, 501, 501, 999, 999
  ]);
  // The glyph budget keeps the leading rows and drops the rest.
  const limited = layoutVisibleLabels(chunk, Uint32Array.from([10, 501, 999]), fontAtlas, 5);
  expect(limited.rowCount).toBe(2);
  expect(limited.glyphCount).toBe(5);
  // A million-row source costs only the visible rows.
  const huge: StreamedLabelTextChunk = {id: 'huge', rowCount: 2_000_000, getText: () => 'AB'};
  const sparse = layoutVisibleLabels(huge, Uint32Array.from([0, 1_999_999]), fontAtlas, 1000);
  expect(sparse.glyphCount).toBe(4);
  expect(sparse.glyphLayouts.length).toBe(4 * 8);
});
