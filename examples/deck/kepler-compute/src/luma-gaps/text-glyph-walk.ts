// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCharacterAtlasPage,
  getCharacterLayoutOffset,
  getTextKerningOffset,
  type Character,
  type FontAtlas
} from '@luma.gl/engine/text';

const DEFAULT_MISSING_CHARACTER_WIDTH = 32;

export type KeplerGlyphWalkOptions = {
  fontAtlas: FontAtlas;
  lineHeight?: number;
  textAnchor?: 'start' | 'middle' | 'end';
  alignmentBaseline?: 'center' | 'top' | 'bottom';
  missingCharacterWidth?: number;
  characterSet?: Set<string>;
};

export type KeplerGlyphVisit = {
  character: string;
  codePoint: number;
  frame: Character | undefined;
  atlasPage: number;
  offsetX: number;
  offsetY: number;
};

export type KeplerGlyphRowMetrics = {
  advance: number;
  glyphCount: number;
  anchorOffsetX: number;
  baselineOffsetY: number;
  bounds: {min: [number, number]; max: [number, number]};
};

/**
 * Walks one line of label text with the CPU atlas layout semantics of luma.gl's text layout.
 * Published @luma.gl/engine/text does not export its glyph walker, so this domain helper composes
 * the public font atlas, kerning and layout-offset functions. Glyph callbacks receive unaligned
 * offsets; apply the returned anchor and baseline offsets when writing final positions.
 */
export function walkKeplerLabelGlyphs(
  visitCodePoints: (visitCodePoint: (codePoint: number) => void) => void,
  options: KeplerGlyphWalkOptions,
  visitGlyph?: (glyph: KeplerGlyphVisit) => void
): KeplerGlyphRowMetrics {
  const {fontAtlas} = options;
  const lineHeight = options.lineHeight ?? fontAtlas.lineHeight;
  const missingCharacterWidth = options.missingCharacterWidth ?? DEFAULT_MISSING_CHARACTER_WIDTH;
  let advance = 0;
  let glyphCount = 0;
  let previousCodePoint: number | undefined;
  let minimumX = Number.POSITIVE_INFINITY;
  let minimumY = Number.POSITIVE_INFINITY;
  let maximumX = Number.NEGATIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;

  visitCodePoints(codePoint => {
    const character = String.fromCodePoint(codePoint);
    const frame = fontAtlas.mapping[character];
    options.characterSet?.add(character);
    advance += getTextKerningOffset(fontAtlas.kerning, previousCodePoint, codePoint);
    const [layoutOffsetX, layoutOffsetY] = frame ? getCharacterLayoutOffset(frame) : [0, 0];
    const offsetX = advance + layoutOffsetX;
    const offsetY = fontAtlas.baselineOffset + lineHeight / 2 + layoutOffsetY;
    if (frame) {
      minimumX = Math.min(minimumX, offsetX);
      minimumY = Math.min(minimumY, offsetY);
      maximumX = Math.max(maximumX, offsetX + frame.width);
      maximumY = Math.max(maximumY, offsetY + frame.height);
    }
    visitGlyph?.({
      character,
      codePoint,
      frame,
      atlasPage: frame ? getCharacterAtlasPage(frame) : 0,
      offsetX,
      offsetY
    });
    advance += frame?.advance ?? missingCharacterWidth;
    previousCodePoint = codePoint;
    glyphCount++;
  });

  const anchorOffsetX =
    options.textAnchor === 'middle' ? -advance / 2 : options.textAnchor === 'end' ? -advance : 0;
  const baselineOffsetY =
    options.alignmentBaseline === 'top'
      ? lineHeight / 2
      : options.alignmentBaseline === 'bottom'
        ? -lineHeight / 2
        : 0;
  return {
    advance,
    glyphCount,
    anchorOffsetX,
    baselineOffsetY,
    bounds: Number.isFinite(minimumX)
      ? {
          min: [minimumX + anchorOffsetX, minimumY + baselineOffsetY],
          max: [maximumX + anchorOffsetX, maximumY + baselineOffsetY]
        }
      : {min: [0, 0], max: [0, 0]}
  };
}
