// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {FontAtlas} from '@luma.gl/engine/text';
import {
  preparePointLabelLayout,
  preparePointLabelSource,
  type PointLabelScalar
} from './label-source';

/**
 * Streamed point labels for GPU-collision sources above the 100,000 row glyph cap.
 *
 * Instead of retaining glyph instances for every row, the source is read in bounded windows:
 * pass 1 counts characters to choose one atlas vocabulary (reused for every window and every
 * viewport), pass 2 measures per-row label bounds with that atlas (16 bytes per row on the GPU, no
 * glyph instances), GPU collision then selects visible rows, and only those rows are laid out as
 * glyph instances. The glyph cost therefore follows the screen, not the row count.
 */
export const STREAMED_LABEL_LIMITS = Object.freeze({
  maximumRows: 0xfffffe,
  windowRows: 16384,
  /** One atlas vocabulary, below the 512 character atlas budget. */
  maximumVocabulary: 511,
  maximumCharactersPerLabel: 256,
  /** Glyph instances kept for the visible rows of one placement (rows beyond it are dropped). */
  maximumVisibleGlyphs: 250000,
  collisionCellSize: 4
});

export type StreamedLabelTextChunk = {
  id: string;
  rowCount: number;
  /** Display text of one row (already formatted); read once per pass and for visible rows. */
  getText(rowIndex: number): string;
};

export function yieldToEventLoop(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/** Single-line admission of one cell: CRLF reads as LF, tabs become spaces, long cells truncate. */
export function normalizeLabelText(
  value: PointLabelScalar,
  maximumCharacters = STREAMED_LABEL_LIMITS.maximumCharactersPerLabel
): string {
  if (value === null || value === undefined) return '';
  const text = String(value);
  let output = '';
  let count = 0;
  for (const character of text) {
    if (character === '\r') continue;
    if (count >= maximumCharacters) break;
    output += character === '\t' ? ' ' : character;
    count++;
  }
  return output;
}

/** Counts characters in windows and keeps the most frequent ones as the shared atlas vocabulary. */
export async function scanLabelVocabulary(
  chunks: readonly StreamedLabelTextChunk[],
  options: {
    windowRows?: number;
    maximumVocabulary?: number;
    yieldControl?: () => Promise<void>;
    isCurrent?: () => boolean;
  } = {}
): Promise<{characterSet: string[]; distinctCount: number}> {
  const windowRows = options.windowRows ?? STREAMED_LABEL_LIMITS.windowRows;
  const maximumVocabulary = options.maximumVocabulary ?? STREAMED_LABEL_LIMITS.maximumVocabulary;
  const yieldControl = options.yieldControl ?? yieldToEventLoop;
  const counts = new Map<string, number>();
  for (const chunk of chunks) {
    for (let start = 0; start < chunk.rowCount; start += windowRows) {
      if (options.isCurrent && !options.isCurrent())
        throw new Error('Streamed label source is stale.');
      const end = Math.min(chunk.rowCount, start + windowRows);
      for (let rowIndex = start; rowIndex < end; rowIndex++) {
        for (const character of chunk.getText(rowIndex)) {
          if (character === '\n') continue; // Line breaks never enter the atlas.
          counts.set(character, (counts.get(character) ?? 0) + 1);
        }
      }
      await yieldControl();
    }
  }
  const ranked = [...counts.entries()].sort(
    (left, right) => right[1] - left[1] || (left[0] < right[0] ? -1 : 1)
  );
  return {
    characterSet: ranked.slice(0, maximumVocabulary).map(([character]) => character),
    distinctCount: ranked.length
  };
}

/**
 * Measures per-row label bounds (widest line advance, paragraph height, in atlas pixels) window by
 * window and hands each window to `writeBounds`. Rows with a character outside the vocabulary get
 * zero bounds, which collision treats as hidden; their number is returned.
 */
export async function measureLabelBounds(
  chunks: readonly StreamedLabelTextChunk[],
  fontAtlas: FontAtlas,
  characterSet: ReadonlySet<string>,
  writeBounds: (chunkIndex: number, firstRow: number, bounds: Float32Array) => void,
  options: {
    windowRows?: number;
    yieldControl?: () => Promise<void>;
    isCurrent?: () => boolean;
  } = {}
): Promise<{hiddenRows: number}> {
  const windowRows = options.windowRows ?? STREAMED_LABEL_LIMITS.windowRows;
  const yieldControl = options.yieldControl ?? yieldToEventLoop;
  let hiddenRows = 0;
  for (const [chunkIndex, chunk] of chunks.entries()) {
    for (let start = 0; start < chunk.rowCount; start += windowRows) {
      if (options.isCurrent && !options.isCurrent())
        throw new Error('Streamed label source is stale.');
      const end = Math.min(chunk.rowCount, start + windowRows);
      const texts: string[] = [];
      for (let rowIndex = start; rowIndex < end; rowIndex++) {
        let text = chunk.getText(rowIndex);
        for (const character of text) {
          if (character !== '\n' && !characterSet.has(character)) {
            text = '';
            hiddenRows++;
            break;
          }
        }
        texts.push(text);
      }
      const source = preparePointLabelSource({
        fieldName: 'streamed-window',
        chunks: [{id: chunk.id, values: texts}],
        maxCharactersPerLabel: STREAMED_LABEL_LIMITS.maximumCharactersPerLabel,
        maxGlyphs: texts.length * STREAMED_LABEL_LIMITS.maximumCharactersPerLabel,
        maxSourceBytes: 1 << 30
      });
      writeBounds(chunkIndex, start, preparePointLabelLayout(source, fontAtlas)[0].rowBounds);
      await yieldControl();
    }
  }
  return {hiddenRows};
}

/** Rows (chunk-local indices) whose bit is set in the global visibility words, per chunk. */
export function collectVisibleRows(
  words: Uint32Array,
  chunks: readonly {rowCount: number}[]
): Uint32Array[] {
  const result: Uint32Array[] = [];
  let base = 0;
  for (const chunk of chunks) {
    const rows: number[] = [];
    for (let rowIndex = 0; rowIndex < chunk.rowCount; ) {
      const globalRow = base + rowIndex;
      const word = words[globalRow >>> 5] ?? 0;
      if (word === 0) {
        rowIndex += 32 - (globalRow & 31);
        continue;
      }
      if ((word >>> (globalRow & 31)) & 1) rows.push(rowIndex);
      rowIndex++;
    }
    result.push(Uint32Array.from(rows));
    base += chunk.rowCount;
  }
  return result;
}

export type VisibleLabelLayout = {
  glyphCount: number;
  glyphLayouts: Float32Array;
  glyphRows: Uint32Array;
  /** Rows that received glyphs (the rest were dropped by the glyph budget). */
  rowCount: number;
};

/**
 * Lays out only the given rows as glyph instances. `glyphRows` keep chunk-local row indices so the
 * batch still aligns with the chunk's position vector. Stops at `maximumGlyphs` (rows are kept in
 * ascending order, later rows dropped).
 */
export function layoutVisibleLabels(
  chunk: StreamedLabelTextChunk,
  visibleRows: Uint32Array,
  fontAtlas: FontAtlas,
  maximumGlyphs: number
): VisibleLabelLayout {
  const codePoints: number[] = [];
  const rowOffsets: number[] = [0];
  const keptRows: number[] = [];
  for (const rowIndex of visibleRows) {
    const text = chunk.getText(rowIndex);
    let glyphs = 0;
    for (const _character of text) glyphs++;
    if (codePoints.length + glyphs > maximumGlyphs) break;
    for (const character of text) codePoints.push(character.codePointAt(0)!);
    rowOffsets.push(codePoints.length);
    keptRows.push(rowIndex);
  }
  const layout = preparePointLabelLayout(
    {
      chunks: [
        {
          id: chunk.id,
          rowIndexBase: 0,
          rowCount: keptRows.length,
          rowOffsets: Uint32Array.from(rowOffsets),
          codePoints: Uint32Array.from(codePoints)
        }
      ]
    },
    fontAtlas
  )[0];
  const glyphRows = layout.glyphRows;
  for (let glyphIndex = 0; glyphIndex < layout.glyphCount; glyphIndex++)
    glyphRows[glyphIndex * 2] = keptRows[glyphRows[glyphIndex * 2]];
  return {
    glyphCount: layout.glyphCount,
    glyphLayouts: layout.glyphLayouts,
    glyphRows,
    rowCount: keptRows.length
  };
}
