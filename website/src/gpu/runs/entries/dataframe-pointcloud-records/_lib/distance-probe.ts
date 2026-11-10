// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The distance axis of the chart-beside-map runs (variogram, correlogram and Ripley families): the
// pointer's x across the card is a position on the chart's distance axis, and the map beside it shows
// the pairs or rings that belong to that distance. Everything here is render-only CPU work.

import type {RunBuffer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';
import {clamp} from '../../../interaction';

export type PlotRectangle = readonly [number, number, number, number];
export type DrawingWorld = {width: number; height: number};

/** The chart rectangle the group's chart-beside-map runs share (fractions of the view). */
export const DISTANCE_PLOT: PlotRectangle = [0.58, 0.12, 0.96, 0.88];

/** Pointer x across the card (inner 90%) to a fraction of the chart's distance axis. */
export function getAxisFraction(unitX: number): number {
  return clamp((unitX - 0.05) / 0.9, 0, 1);
}

/** The pointer position (across the card) that sits at `fraction` of the distance axis. */
export function getAxisPointer(fraction: number): number {
  return 0.05 + fraction * 0.9;
}

/** Lag or radius bin under a fraction of the axis. */
export function getBinIndex(fraction: number, binCount: number): number {
  return Math.min(binCount - 1, Math.floor(fraction * binCount));
}

/** A box around one chart column from the bottom to the top of the plot rectangle. */
export function getBinBox(
  rectangle: PlotRectangle,
  world: DrawingWorld,
  binIndex: number,
  binCount: number
): RunGlyph {
  const left = rectangle[0] + ((rectangle[2] - rectangle[0]) * binIndex) / binCount;
  const right = rectangle[0] + ((rectangle[2] - rectangle[0]) * (binIndex + 1)) / binCount;
  return {
    shape: 'box',
    minimum: [left * world.width, rectangle[1] * world.height],
    maximum: [right * world.width, rectangle[3] * world.height]
  };
}

/** World position of chart value `value` on an axis `range` drawn into `rectangle`. */
export function getChartWorldY(
  rectangle: PlotRectangle,
  world: DrawingWorld,
  range: readonly [number, number],
  value: number
): number {
  const fraction = clamp((value - range[0]) / (range[1] - range[0]), 0, 1);
  return (rectangle[1] + fraction * (rectangle[3] - rectangle[1])) * world.height;
}

/** Horizontal rules across the plot at chart values (a zero line, a sill, a significance level). */
export function createRuleSegments(
  rectangle: PlotRectangle,
  world: DrawingWorld,
  range: readonly [number, number],
  levels: readonly number[]
): Float32Array {
  const segments = new Float32Array(levels.length * 4);
  levels.forEach((level, index) => {
    const y = getChartWorldY(rectangle, world, range, level);
    segments.set([rectangle[0] * world.width, y, rectangle[2] * world.width, y], index * 4);
  });
  return segments;
}

/** A polyline of `count` points of a theoretical curve, as consecutive segments for a lines layer. */
export function createCurveSegments(
  rectangle: PlotRectangle,
  world: DrawingWorld,
  range: readonly [number, number],
  values: ArrayLike<number>
): Float32Array {
  const count = values.length;
  const segments = new Float32Array(Math.max(count - 1, 0) * 4);
  const getX = (index: number) =>
    (rectangle[0] + ((rectangle[2] - rectangle[0]) * (index + 0.5)) / count) * world.width;
  for (let index = 0; index < count - 1; index++) {
    segments.set(
      [
        getX(index),
        getChartWorldY(rectangle, world, range, values[index]),
        getX(index + 1),
        getChartWorldY(rectangle, world, range, values[index + 1])
      ],
      index * 4
    );
  }
  return segments;
}

export type PairProbe = {
  /** Pairs with a distance in [minimum, maximum), chosen evenly; unused rows are zero-length. */
  find(minimum: number, maximum: number): {segments: Float32Array; total: number; shown: number};
};

/** Looks up the point pairs of one distance bin. Results are cached for the last bin. */
export function createPairProbe(positions: Float32Array, limit: number): PairProbe {
  const count = positions.length >> 1;
  let cachedKey = '';
  let cached = {segments: new Float32Array(limit * 4), total: 0, shown: 0};
  return {
    find(minimum, maximum) {
      const key = `${minimum.toFixed(5)}:${maximum.toFixed(5)}`;
      if (key === cachedKey) return cached;
      const minimumSquared = minimum * minimum;
      const maximumSquared = maximum * maximum;
      const matches: number[] = [];
      for (let first = 0; first < count; first++) {
        for (let second = first + 1; second < count; second++) {
          const deltaX = positions[first * 2] - positions[second * 2];
          const deltaY = positions[first * 2 + 1] - positions[second * 2 + 1];
          const squared = deltaX * deltaX + deltaY * deltaY;
          if (squared >= minimumSquared && squared < maximumSquared) matches.push(first, second);
        }
      }
      const total = matches.length >> 1;
      const segments = new Float32Array(limit * 4);
      const shown = Math.min(limit, total);
      for (let row = 0; row < shown; row++) {
        const pair = Math.floor((row * total) / shown);
        const first = matches[pair * 2];
        const second = matches[pair * 2 + 1];
        segments.set(
          [
            positions[first * 2],
            positions[first * 2 + 1],
            positions[second * 2],
            positions[second * 2 + 1]
          ],
          row * 4
        );
      }
      cachedKey = key;
      cached = {segments, total, shown};
      return cached;
    }
  };
}

/** Evenly spread row indices, for the events a probe draws rings around. */
export function pickSpreadRows(count: number, wanted: number): number[] {
  return Array.from({length: wanted}, (_, index) => Math.floor(((index + 0.5) * count) / wanted));
}

/**
 * The anchors' rings of radius `radius` and the rows of `positions` within `radius` of any anchor
 (and at least `innerRadius` away), written into `highlight` (x, y per row; unused rows repeat the first anchor).
 */
export function collectInsideRings(
  positions: Float32Array,
  anchors: readonly number[],
  radius: number,
  highlight: Float32Array,
  innerRadius = 0
): number {
  const radiusSquared = radius * radius;
  const innerSquared = innerRadius * innerRadius;
  const capacity = highlight.length >> 1;
  let found = 0;
  const anchorSet = new Set(anchors);
  for (let row = 0; row < positions.length >> 1 && found < capacity; row++) {
    if (anchorSet.has(row)) continue;
    for (const anchor of anchors) {
      const deltaX = positions[row * 2] - positions[anchor * 2];
      const deltaY = positions[row * 2 + 1] - positions[anchor * 2 + 1];
      const squared = deltaX * deltaX + deltaY * deltaY;
      if (squared <= radiusSquared && squared >= innerSquared) {
        highlight[found * 2] = positions[row * 2];
        highlight[found * 2 + 1] = positions[row * 2 + 1];
        found++;
        break;
      }
    }
  }
  return found;
}

/** Number of highlighted rows a ring probe can show. */
export const RING_HIGHLIGHT_LIMIT = 300;

/**
 * Rings of `radius` around the anchor rows, and the other rows inside any ring written to
 * `highlight`. Returns the ring glyphs and the number of highlighted rows (for the dots layer).
 */
export function drawRingProbe(
  positions: Float32Array,
  anchors: readonly number[],
  radius: number,
  highlight: RunBuffer
): {glyphs: RunGlyph[]; found: number} {
  const rows = new Float32Array(RING_HIGHLIGHT_LIMIT * 2).fill(-10);
  const found = collectInsideRings(positions, anchors, radius, rows);
  highlight.write(rows);
  return {
    found,
    glyphs: anchors.map(anchor => ({
      shape: 'ring' as const,
      at: [positions[anchor * 2], positions[anchor * 2 + 1]] as const,
      radius
    }))
  };
}

/** The `count` argument for a dots layer that shows `found` highlighted rows (never zero). */
export function getHighlightCount(focus: number | readonly number[] | undefined): number {
  return Math.max(1, typeof focus === 'number' ? focus : 0);
}
