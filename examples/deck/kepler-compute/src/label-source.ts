// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {walkTextGlyphs, type FontAtlas} from '@luma.gl/engine/text';

const LINE_FEED = 10;

/** Values already admitted by the scalar-source adapter; null labels occupy rows but no glyphs. */
export type PointLabelScalar = string | number | boolean | bigint | null | undefined;
export type PreparedPointLabelSource = {
  readonly fieldName: string;
  readonly chunks: readonly {
    readonly id: string;
    readonly rowIndexBase: number;
    readonly rowCount: number;
    readonly rowOffsets: Uint32Array;
    readonly codePoints: Uint32Array;
  }[];
  readonly characterSet: readonly string[];
  readonly rowCount: number;
  readonly glyphCount: number;
  /** Retained typed arrays and character strings, excluding JavaScript object overhead. */
  readonly byteLength: number;
};

/** Explicit field admission. Ordinary String(value), no formatting, shaping or repeated row accessor. */
export function preparePointLabelSource(options: {
  fieldName: string;
  chunks: readonly {id: string; values: readonly PointLabelScalar[]}[];
  maxCharactersPerLabel: number;
  maxGlyphs: number;
  maxSourceBytes: number;
}): PreparedPointLabelSource {
  const {maxCharactersPerLabel, maxGlyphs, maxSourceBytes} = options;
  for (const maximum of [maxCharactersPerLabel, maxGlyphs, maxSourceBytes]) {
    if (!Number.isSafeInteger(maximum) || maximum < 0)
      throw new Error('Invalid label source budget');
  }
  let rowCount = 0;
  let glyphCount = 0;
  let byteLength = 0;
  const characters = new Set<string>();
  const chunkIds = new Set<string>();
  const chunks = options.chunks.map(chunk => {
    if (chunkIds.has(chunk.id)) throw new Error('Duplicate label source chunk ID');
    chunkIds.add(chunk.id);
    byteLength += (chunk.values.length + 1) * 4;
    if (byteLength > maxSourceBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Label source exceeds its admitted byte budget'
      });
    const rowOffsets = new Uint32Array(chunk.values.length + 1);
    const codePoints: number[] = [];
    for (const [rowIndex, value] of chunk.values.entries()) {
      if (
        value !== null &&
        value !== undefined &&
        !['string', 'number', 'boolean', 'bigint'].includes(typeof value)
      ) {
        throw new Error('Point labels require ordinary scalar values');
      }
      const text = value === null || value === undefined ? '' : String(value);
      let characterCount = 0;
      for (const character of text) {
        if (character === '\r') continue; // CRLF is read as LF.
        if (character === '\t') {
          throw new Error('Point labels currently require one line without tabs');
        }
        characterCount++;
        glyphCount++;
        byteLength += 4;
        // Line breaks are zero-size glyphs and never enter the font atlas.
        if (character !== '\n' && !characters.has(character)) {
          characters.add(character);
          byteLength += character.length * 2;
        }
        if (
          characterCount > maxCharactersPerLabel ||
          glyphCount > maxGlyphs ||
          byteLength > maxSourceBytes ||
          glyphCount > 0xffffffff
        ) {
          throw new KeplerAllocationError('memory-budget', {
            message: 'Point labels exceed their admitted character or byte budget'
          });
        }
        codePoints.push(character.codePointAt(0)!);
      }
      rowOffsets[rowIndex + 1] = codePoints.length;
    }
    const prepared = {
      id: chunk.id,
      rowIndexBase: rowCount,
      rowCount: chunk.values.length,
      rowOffsets,
      codePoints: new Uint32Array(codePoints)
    };
    rowCount += chunk.values.length;
    if (!Number.isSafeInteger(rowCount) || rowCount > 0xffffffff)
      throw new Error('Label row identity exceeds uint32');
    return prepared;
  });
  return {
    fieldName: options.fieldName,
    chunks,
    characterSet: [...characters],
    rowCount,
    glyphCount,
    byteLength
  };
}

/** Float-preserving native single-line metrics; anchor/baseline shifts remain shader controls. */
export function preparePointLabelLayout(
  source: Pick<PreparedPointLabelSource, 'chunks'>,
  fontAtlas: FontAtlas
) {
  return source.chunks.map(chunk => {
    const glyphCount = chunk.codePoints.length;
    const glyphLayouts = new Float32Array(glyphCount * 8);
    const glyphRows = new Uint32Array(glyphCount * 2);
    // Per row: widest line advance and paragraph height, for backgrounds. Empty rows stay zero.
    const rowBounds = new Float32Array(chunk.rowCount * 4);
    for (let rowIndex = 0; rowIndex < chunk.rowCount; rowIndex++) {
      const rowStart = chunk.rowOffsets[rowIndex];
      const rowEnd = chunk.rowOffsets[rowIndex + 1];
      let lineCount = 1;
      for (let index = rowStart; index < rowEnd; index++)
        if (chunk.codePoints[index] === LINE_FEED) lineCount++;
      // Anchors use each line's advance; baseline alignment uses the whole paragraph height.
      const paragraphHeight = fontAtlas.lineHeight * lineCount;
      let lineIndex = 0;
      let lineStart = rowStart;
      for (let lineEnd = rowStart; lineEnd <= rowEnd; lineEnd++) {
        if (lineEnd < rowEnd && chunk.codePoints[lineEnd] !== LINE_FEED) continue;
        let glyphIndex = lineStart;
        const lineOffsetY = lineIndex * fontAtlas.lineHeight;
        const metrics = walkTextGlyphs(
          visit => {
            for (let index = lineStart; index < lineEnd; index++) visit(chunk.codePoints[index]);
          },
          {fontAtlas, textAnchor: 'start'},
          glyph => {
            if (!glyph.frame)
              throw new Error('Prepared label atlas is missing an admitted character');
            glyphLayouts.set(
              [
                glyph.offsetX,
                glyph.offsetY + lineOffsetY,
                0,
                paragraphHeight,
                glyph.frame.x,
                glyph.frame.y,
                glyph.frame.width,
                glyph.frame.height
              ],
              glyphIndex * 8
            );
            glyphRows.set([rowIndex, glyph.atlasPage], glyphIndex * 2);
            glyphIndex++;
          }
        );
        for (let index = lineStart; index < lineEnd; index++)
          glyphLayouts[index * 8 + 2] = metrics.advance;
        if (rowEnd > rowStart) {
          rowBounds[rowIndex * 4] = Math.max(rowBounds[rowIndex * 4], metrics.advance);
          rowBounds[rowIndex * 4 + 1] = paragraphHeight;
        }
        if (lineEnd < rowEnd) {
          // The line-feed glyph is a degenerate zero-size quad.
          glyphRows.set([rowIndex, 0], lineEnd * 2);
        }
        lineIndex++;
        lineStart = lineEnd + 1;
      }
    }
    return {
      id: chunk.id,
      rowCount: chunk.rowCount,
      rowIndexBase: chunk.rowIndexBase,
      glyphCount,
      glyphLayouts,
      glyphRows,
      rowBounds
    };
  });
}
