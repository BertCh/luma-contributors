// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseViewerColor} from './standalone-viewer-config';
import type {ColorTriple, ViewerColorScale} from './standalone-viewer-scales';

/**
 * Kepler custom palettes: `colorRange.colorMap` is a list of `[value, color]` entries. For the
 * `custom` scale the values are ascending upper bounds (the last is ignored); for `customOrdinal`
 * they are a category string, a list of category strings, or null for every other value.
 */
type ColorMapEntry = [unknown, unknown];

export function readColorMap(colorRange: unknown): ColorMapEntry[] | null {
  const colorMap = (colorRange as {colorMap?: unknown} | null)?.colorMap;
  return Array.isArray(colorMap) && colorMap.length
    ? colorMap.filter((entry): entry is ColorMapEntry => Array.isArray(entry) && entry.length >= 2)
    : null;
}

export function isCustomColorScale(scaleType: string | null, colorRange: unknown): boolean {
  return (
    (scaleType === 'custom' || scaleType === 'customOrdinal') && readColorMap(colorRange) !== null
  );
}

/** Threshold scale over the colorMap upper bounds (d3 scaleThreshold: value < bound[i] -> color i). */
export function createCustomColorScale(colorMap: ColorMapEntry[]): ViewerColorScale {
  const colors = colorMap.map(entry => parseViewerColor(entry[1]));
  const thresholds = colorMap.slice(0, -1).map(entry => Number(entry[0]));
  return {
    kind: 'continuous',
    domain: thresholds,
    categories: [],
    colors,
    getColor(value) {
      let index = 0;
      while (index < thresholds.length && value >= thresholds[index]) index++;
      return colors[Math.min(index, colors.length - 1)];
    },
    getCategoryColor: () => colors[0]
  };
}

/** Category lookup; entries with a null value become the fallback color. */
export function createCustomOrdinalColorScale(colorMap: ColorMapEntry[]): ViewerColorScale {
  const lookup = new Map<string, ColorTriple>();
  const categories: string[] = [];
  let fallback: ColorTriple | null = null;
  const colors = colorMap.map(entry => parseViewerColor(entry[1]));
  colorMap.forEach((entry, index) => {
    const labels = Array.isArray(entry[0]) ? entry[0] : entry[0] === null ? null : [entry[0]];
    if (!labels) fallback = colors[index];
    else
      for (const label of labels) {
        lookup.set(String(label), colors[index]);
        categories.push(String(label));
      }
  });
  const otherColor: ColorTriple = fallback ?? [100, 100, 100];
  return {
    kind: 'ordinal',
    domain: [],
    categories,
    colors,
    getColor: () => otherColor,
    getCategoryColor: label => lookup.get(label) ?? otherColor
  };
}
