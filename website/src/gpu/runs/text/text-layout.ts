// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU side of picture text: measures labels with the generated font atlas, sizes them for the
// width the picture is shown at, drops or thins labels that would overlap, and writes one GPU
// instance per glyph, swatch and rule. Nothing here touches the GPU, so it runs in any test.

import {CARD_DISPLAY_WIDTH} from './display';
import {FONT_METRICS} from './font-metrics';

// ---------------------------------------------------------------------------------------------
// Declarative label API

/** Where a point is given: drawing-world units (y up), or 0 to 1 across the canvas (y down). */
export type TextSpace = 'world' | 'canvas';

/** Which point of the label's box sits at `at`. */
export type TextAnchor =
  | 'center'
  | 'left'
  | 'right'
  | 'top'
  | 'bottom'
  | 'top-left'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-right';

/**
 * Size and color of a label. `tick` is axis numbers and matrix ticks (muted), `label` is bar,
 * class and legend names, `title` is an axis or panel title, `value` a number worth reading.
 */
export type TextRole = 'tick' | 'label' | 'title' | 'value';

export type TextTone = 'ink' | 'muted';

export type RunLabel = {
  text: string;
  /** Position in `space`: world units by default (degrees for geographic runs). */
  at: readonly [number, number];
  space?: TextSpace;
  /** Default `center`. `left` puts the label's left edge at `at`. */
  anchor?: TextAnchor;
  /** Nudge in text pixels (CSS pixels on a card), x right, y up. */
  offset?: readonly [number, number];
  /** Default `label`. */
  role?: TextRole;
  /** Default by role: `muted` for ticks, `ink` otherwise. */
  tone?: TextTone;
  /** A colored square before the text: a categorical index (theme colors) or RGB 0 to 1. */
  swatch?: number | readonly [number, number, number];
  /** Degrees counter-clockwise. 90 reads bottom to top (a y-axis title). */
  angle?: 0 | 90 | -90;
  /** Higher survives an overlap. Default: title 3, value 2, label 1, tick 0. */
  priority?: number;
  /**
   * Labels of one group are thinned evenly (every second, every third...) until none overlap,
   * instead of being dropped one by one. Use one group per axis.
   */
  group?: string;
};

/** A straight mark in the ink or muted color: an axis line, a tick, a reference rule. */
export type RunRule = {
  from: readonly [number, number];
  to: readonly [number, number];
  space?: TextSpace;
  /** Nudges of the two ends in text pixels (x right, y up): tick marks of a fixed length. */
  fromOffset?: readonly [number, number];
  toOffset?: readonly [number, number];
  /** Default `muted`. `background` makes a backing panel (with `width` as its height). */
  tone?: TextTone | 'background';
  /** Width in text pixels. Default 1. */
  width?: number;
  /** 0 to 1. Default 1. */
  opacity?: number;
  /** Dashed, with this dash length in text pixels. */
  dash?: number;
};

/** Labels and rules drawn together, over the picture. */
export type RunText = {
  labels?: readonly RunLabel[];
  rules?: readonly RunRule[];
};

// ---------------------------------------------------------------------------------------------
// Sizing

export {CARD_DISPLAY_WIDTH};

const ROLE_STYLE: Record<
  TextRole,
  {size: number; tone: TextTone; weight: number; priority: number}
> = {
  tick: {size: 9.5, tone: 'muted', weight: 0.12, priority: 0},
  label: {size: 10, tone: 'ink', weight: 0.15, priority: 1},
  title: {size: 10.5, tone: 'ink', weight: 0.4, priority: 3},
  value: {size: 10, tone: 'ink', weight: 0.35, priority: 2}
};

/**
 * Device pixels per text pixel. Text pixels are CSS pixels on a card; on a wider picture (the
 * function-page hero) text grows more slowly than the picture, up to 1.45 times.
 *
 * `displayWidth` is the CSS width the picture is shown at. A live canvas is shown at its own CSS
 * width; a poster is rendered large and shown on a card, so it passes `CARD_DISPLAY_WIDTH` and its
 * labels then have the proportions of the live card.
 */
export function getTextScale(deviceWidth: number, displayWidth: number): number {
  const devicePixelRatio = deviceWidth / Math.max(displayWidth, 1);
  const growth = Math.min(1.45, Math.max(0.85, (displayWidth / CARD_DISPLAY_WIDTH) ** 0.45));
  return devicePixelRatio * growth;
}

// ---------------------------------------------------------------------------------------------
// Font

type Glyph = {
  atlasX: number;
  atlasY: number;
  atlasWidth: number;
  atlasHeight: number;
  /** Quad left edge and top edge relative to the pen on the baseline, em (top is above). */
  quadLeft: number;
  quadTop: number;
  advance: number;
};

const GLYPHS = new Map<number, Glyph>();
for (const row of FONT_METRICS.glyphs) {
  GLYPHS.set(row[0], {
    atlasX: row[1],
    atlasY: row[2],
    atlasWidth: row[3],
    atlasHeight: row[4],
    quadLeft: row[5],
    quadTop: row[6],
    advance: row[7]
  });
}
const KERNING = new Map<number, number>();
for (const [first, second, adjustment] of FONT_METRICS.kerning) {
  KERNING.set(first * 0x10000 + second, adjustment);
}

/** Characters the font lacks, written with ones it has. */
const SUBSTITUTES: Record<string, string> = {
  '·': '⋅',
  µ: 'μ',
  '`': '′',
  '²': '^2',
  '³': '^3',
  '≠': '!=',
  ' ': ' ',
  ' ': ' '
};
const FALLBACK = '?'.codePointAt(0) as number;

function getGlyphCodes(text: string): number[] {
  const codes: number[] = [];
  for (const character of text) {
    const substitute = SUBSTITUTES[character];
    for (const part of substitute ?? character) {
      const code = part.codePointAt(0) as number;
      codes.push(GLYPHS.has(code) ? code : FALLBACK);
    }
  }
  return codes;
}

/** Width of `text` in em. */
export function measureText(text: string): number {
  const codes = getGlyphCodes(text);
  let width = 0;
  codes.forEach((code, index) => {
    width += (GLYPHS.get(code) as Glyph).advance;
    if (index + 1 < codes.length) width += KERNING.get(code * 0x10000 + codes[index + 1]) ?? 0;
  });
  return width;
}

/** Font size of a role in text pixels. */
export function getRoleSize(role: TextRole = 'label'): number {
  return ROLE_STYLE[role].size;
}

/** Width of a label's swatch and the gap after it, in text pixels, as the layout draws them. */
export function getSwatchWidth(role: TextRole = 'label'): number {
  const size = ROLE_STYLE[role].size;
  return getSwatchSize(size) + 0.4 * size;
}

function getSwatchSize(fontSize: number): number {
  return Math.round(FONT_METRICS.capHeight * fontSize * 1.25);
}

/** Width of `text` in text pixels when drawn in `role`. */
export function getTextWidth(text: string, role: TextRole = 'label'): number {
  return measureText(text) * ROLE_STYLE[role].size;
}

// ---------------------------------------------------------------------------------------------
// Instances

/** Floats per GPU instance: five vec4f (see text-layer.ts). */
export const INSTANCE_FLOATS = 20;
const KIND_GLYPH = 0;
const KIND_RECT = 1;

export type TextPalette = {
  ink: readonly [number, number, number];
  muted: readonly [number, number, number];
  background: readonly [number, number, number];
  categories: readonly (readonly [number, number, number])[];
};

/** How the picture is mapped to the canvas: world to canvas pixels (y down). */
export type TextViewport = {
  width: number;
  height: number;
  toPixel(x: number, y: number): readonly [number, number];
};

export type TextLayout = {
  instances: Float32Array;
  instanceCount: number;
  /** Labels that were drawn, in placement order (for tests and descriptions). */
  placed: string[];
  /** Labels that were left out because they overlapped or did not fit. */
  dropped: string[];
};

type Box = {left: number; top: number; right: number; bottom: number};

type Candidate = {
  label: RunLabel;
  index: number;
  priority: number;
  /** Pixel pivot (snapped), the box in canvas pixels, and what to emit if it is kept. */
  pivot: [number, number];
  box: Box;
  emit: (writer: InstanceWriter) => void;
};

class InstanceWriter {
  values: Float32Array = new Float32Array(INSTANCE_FLOATS * 64);
  count = 0;

  push(
    pivot: readonly [number, number],
    box: readonly [number, number, number, number],
    atlas: readonly [number, number, number, number],
    angle: number,
    kind: number,
    weight: number,
    color: readonly [number, number, number, number]
  ): void {
    if ((this.count + 1) * INSTANCE_FLOATS > this.values.length) {
      const grown = new Float32Array(this.values.length * 2);
      grown.set(this.values);
      this.values = grown;
    }
    const offset = this.count * INSTANCE_FLOATS;
    this.values.set(
      [pivot[0], pivot[1], 0, 0, ...box, ...atlas, angle, kind, weight, 0, ...color],
      offset
    );
    this.count++;
  }
}

function getPivotPixel(viewport: TextViewport, at: readonly [number, number], space?: TextSpace) {
  if (space === 'canvas') return [at[0] * viewport.width, at[1] * viewport.height] as const;
  return viewport.toPixel(at[0], at[1]);
}

/** Fraction of the box width and height (y down) that sits left of and above the pivot. */
function getAnchorFractions(anchor: TextAnchor): [number, number] {
  switch (anchor) {
    case 'left':
      return [0, 0.5];
    case 'right':
      return [1, 0.5];
    case 'top':
      return [0.5, 0];
    case 'bottom':
      return [0.5, 1];
    case 'top-left':
      return [0, 0];
    case 'top-right':
      return [1, 0];
    case 'bottom-left':
      return [0, 1];
    case 'bottom-right':
      return [1, 1];
    default:
      return [0.5, 0.5];
  }
}

/** Rotates a y-down pixel offset by `angle` radians counter-clockwise as seen on screen. */
function rotate(x: number, y: number, angle: number): [number, number] {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [x * cosine + y * sine, -x * sine + y * cosine];
}

function overlaps(first: Box, second: Box, padding: number): boolean {
  return (
    first.left < second.right + padding &&
    second.left < first.right + padding &&
    first.top < second.bottom + padding &&
    second.top < first.bottom + padding
  );
}

function createCandidate(
  label: RunLabel,
  index: number,
  viewport: TextViewport,
  scale: number,
  palette: TextPalette,
  alpha: number
): Candidate | null {
  if (!label.text && label.swatch === undefined) return null;
  const role = ROLE_STYLE[label.role ?? 'label'];
  const tone = label.tone ?? role.tone;
  const color = tone === 'muted' ? palette.muted : palette.ink;
  const fontSize = role.size * scale;
  const codes = getGlyphCodes(label.text);
  // The box spans cap height above the baseline and a little below it, so digits and capitals
  // center visually; descenders may reach past it.
  const capHeight = FONT_METRICS.capHeight * fontSize;
  const below = 0.22 * fontSize;
  const swatchSize = label.swatch === undefined ? 0 : getSwatchSize(fontSize);
  const swatchGap = label.swatch === undefined || !label.text ? 0 : 0.4 * fontSize;
  const textWidth = measureText(label.text) * fontSize;
  const width = swatchSize + swatchGap + textWidth;
  const height = Math.max(capHeight + below, swatchSize);
  const angle = ((label.angle ?? 0) * Math.PI) / 180;
  const [fractionX, fractionY] = getAnchorFractions(label.anchor ?? 'center');
  // Box in the label's own frame (y down), relative to the pivot.
  const frameLeft = -fractionX * width;
  const frameTop = -fractionY * height;
  const baseline = frameTop + (height - capHeight - below) / 2 + capHeight;
  const [pivotX, pivotY] = getPivotPixel(viewport, label.at, label.space);
  if (!Number.isFinite(pivotX) || !Number.isFinite(pivotY)) return null;
  const offset = label.offset ?? [0, 0];
  let pivot: [number, number] = [pivotX + offset[0] * scale, pivotY - offset[1] * scale];
  // Canvas-space box: the corners of the rotated frame box.
  const corners = [
    rotate(frameLeft, frameTop, angle),
    rotate(frameLeft + width, frameTop, angle),
    rotate(frameLeft, frameTop + height, angle),
    rotate(frameLeft + width, frameTop + height, angle)
  ];
  const box = (): Box => ({
    left: pivot[0] + Math.min(...corners.map(corner => corner[0])),
    right: pivot[0] + Math.max(...corners.map(corner => corner[0])),
    top: pivot[1] + Math.min(...corners.map(corner => corner[1])),
    bottom: pivot[1] + Math.max(...corners.map(corner => corner[1]))
  });
  // Keep the label on the canvas: slide it in when it pokes out by less than its own size.
  const margin = 1.5 * scale;
  let bounds = box();
  const shiftX =
    bounds.left < margin
      ? margin - bounds.left
      : bounds.right > viewport.width - margin
        ? viewport.width - margin - bounds.right
        : 0;
  const shiftY =
    bounds.top < margin
      ? margin - bounds.top
      : bounds.bottom > viewport.height - margin
        ? viewport.height - margin - bounds.bottom
        : 0;
  if (
    Math.abs(shiftX) > bounds.right - bounds.left ||
    Math.abs(shiftY) > bounds.bottom - bounds.top
  ) {
    return null;
  }
  pivot = [Math.round(pivot[0] + shiftX), Math.round(pivot[1] + shiftY)];
  bounds = box();
  if (bounds.right - bounds.left > viewport.width || bounds.bottom - bounds.top > viewport.height) {
    return null;
  }
  const ink = [color[0], color[1], color[2], alpha] as const;
  const weight = role.weight;
  const emit = (writer: InstanceWriter) => {
    let penX = frameLeft;
    if (label.swatch !== undefined) {
      const swatchColor =
        typeof label.swatch === 'number'
          ? palette.categories[
              ((label.swatch % palette.categories.length) + palette.categories.length) %
                palette.categories.length
            ]
          : label.swatch;
      const swatchTop = Math.round(frameTop + (height - swatchSize) / 2);
      writer.push(
        pivot,
        [Math.round(penX), swatchTop, swatchSize, swatchSize],
        [0, 0, 0, 0],
        angle,
        KIND_RECT,
        0,
        [swatchColor[0], swatchColor[1], swatchColor[2], alpha]
      );
      penX += swatchSize + swatchGap;
    }
    const roundedBaseline = Math.round(baseline);
    const texelSize = fontSize / FONT_METRICS.em;
    codes.forEach((code, position) => {
      const glyph = GLYPHS.get(code) as Glyph;
      if (code !== 32) {
        writer.push(
          pivot,
          [
            penX + glyph.quadLeft * fontSize,
            roundedBaseline - glyph.quadTop * fontSize,
            glyph.atlasWidth * texelSize,
            glyph.atlasHeight * texelSize
          ],
          [glyph.atlasX, glyph.atlasY, glyph.atlasWidth, glyph.atlasHeight],
          angle,
          KIND_GLYPH,
          weight,
          ink
        );
      }
      penX += glyph.advance * fontSize;
      if (position + 1 < codes.length) {
        penX += (KERNING.get(code * 0x10000 + codes[position + 1]) ?? 0) * fontSize;
      }
    });
  };
  return {
    label,
    index,
    priority: label.priority ?? role.priority,
    pivot,
    box: bounds,
    emit
  };
}

function emitRule(
  rule: RunRule,
  viewport: TextViewport,
  scale: number,
  palette: TextPalette,
  alpha: number,
  writer: InstanceWriter
): void {
  const nudge = (point: readonly [number, number], offset?: readonly [number, number]) =>
    offset ? ([point[0] + offset[0] * scale, point[1] - offset[1] * scale] as const) : point;
  const from = nudge(getPivotPixel(viewport, rule.from, rule.space), rule.fromOffset);
  const to = nudge(getPivotPixel(viewport, rule.to, rule.space), rule.toOffset);
  if (![...from, ...to].every(Number.isFinite)) return;
  const tone = rule.tone ?? 'muted';
  const color =
    tone === 'ink' ? palette.ink : tone === 'background' ? palette.background : palette.muted;
  const width = Math.max(1, Math.round((rule.width ?? 1) * scale));
  const deltaX = to[0] - from[0];
  const deltaY = to[1] - from[1];
  const length = Math.hypot(deltaX, deltaY);
  if (length < 0.5) return;
  // Axis-aligned rules of odd width sit on pixel centers, so they stay one crisp pixel wide.
  const snap = (value: number) => (width % 2 === 1 ? Math.floor(value) + 0.5 : Math.round(value));
  const isHorizontal = Math.abs(deltaY) < 1e-6;
  const isVertical = Math.abs(deltaX) < 1e-6;
  const start: [number, number] = [
    isVertical ? snap(from[0]) : from[0],
    isHorizontal ? snap(from[1]) : from[1]
  ];
  const angle = Math.atan2(-deltaY, deltaX);
  const ink = [color[0], color[1], color[2], alpha * (rule.opacity ?? 1)] as const;
  const dash = rule.dash ? rule.dash * scale : 0;
  const step = dash > 0 ? dash * 2 : length;
  for (let along = 0; along < length; along += step) {
    const piece = Math.min(dash > 0 ? dash : length, length - along);
    writer.push(start, [along, -width / 2, piece, width], [0, 0, 0, 0], angle, KIND_RECT, 0, ink);
  }
}

/**
 * Places `text` on a `viewport.width` x `viewport.height` canvas: rules first, then labels by
 * priority (ties keep their order), each group thinned evenly until it does not overlap itself, and
 * every label that would overlap one already placed or not fit on the canvas left out.
 */
export function layoutText(
  text: RunText,
  viewport: TextViewport,
  scale: number,
  palette: TextPalette,
  alpha = 1
): TextLayout {
  const writer = new InstanceWriter();
  for (const rule of text.rules ?? []) emitRule(rule, viewport, scale, palette, alpha, writer);
  const candidates: Candidate[] = [];
  const dropped: string[] = [];
  (text.labels ?? []).forEach((label, index) => {
    const candidate = createCandidate(label, index, viewport, scale, palette, alpha);
    if (candidate) candidates.push(candidate);
    else dropped.push(label.text);
  });
  const padding = 2 * scale;
  // Even thinning per group: the smallest stride whose members do not overlap one another.
  const groups = new Map<string, Candidate[]>();
  for (const candidate of candidates) {
    const group = candidate.label.group;
    if (!group) continue;
    const members = groups.get(group) ?? [];
    members.push(candidate);
    groups.set(group, members);
  }
  const thinned = new Set<Candidate>();
  for (const members of groups.values()) {
    for (let stride = 1; stride <= members.length; stride++) {
      const kept = members.filter((_, position) => position % stride === 0);
      const isClear = kept.every((first, position) =>
        kept.slice(position + 1).every(second => !overlaps(first.box, second.box, padding))
      );
      if (isClear) {
        for (const member of members) if (!kept.includes(member)) thinned.add(member);
        break;
      }
    }
  }
  const order = candidates
    .filter(candidate => {
      if (!thinned.has(candidate)) return true;
      dropped.push(candidate.label.text);
      return false;
    })
    .sort((first, second) => second.priority - first.priority || first.index - second.index);
  const placedBoxes: Box[] = [];
  const placed: string[] = [];
  for (const candidate of order) {
    if (placedBoxes.some(box => overlaps(box, candidate.box, padding))) {
      dropped.push(candidate.label.text);
      continue;
    }
    placedBoxes.push(candidate.box);
    placed.push(candidate.label.text);
    candidate.emit(writer);
  }
  // Every halo first, then every ink: a halo never covers a neighbouring glyph (an en dash after a
  // zero) or a label's own rule. The fourth style component selects the pass (1 halo, 2 ink).
  const count = writer.count;
  const instances = new Float32Array(count * 2 * INSTANCE_FLOATS);
  instances.set(writer.values.subarray(0, count * INSTANCE_FLOATS), 0);
  instances.set(writer.values.subarray(0, count * INSTANCE_FLOATS), count * INSTANCE_FLOATS);
  for (let index = 0; index < count * 2; index++) {
    instances[index * INSTANCE_FLOATS + 15] = index < count ? 1 : 2;
  }
  return {instances, instanceCount: count * 2, placed, dropped};
}
