// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/** CPU visual-channel scales for the offline viewer (Kepler quantize, quantile, ordinal, linear, sqrt, log). */

export type ColorTriple = [number, number, number];

export type ViewerColorScale = {
  kind: 'continuous' | 'ordinal';
  /** Numeric domain endpoints or class thresholds for continuous scales. */
  domain: number[];
  /** Category labels for ordinal scales. */
  categories: string[];
  colors: ColorTriple[];
  getColor(value: number): ColorTriple;
  getCategoryColor(label: string): ColorTriple;
};

function getFiniteExtent(values: ArrayLike<number>): [number, number] {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return minimum <= maximum ? [minimum, maximum] : [0, 1];
}

/** Quantile cut points (colors.length - 1 thresholds) of the finite values. */
export function getQuantileThresholds(values: ArrayLike<number>, classCount: number): number[] {
  const sorted = Float64Array.from(Array.from(values).filter(Number.isFinite)).sort();
  const thresholds: number[] = [];
  for (let step = 1; step < classCount; step++) {
    const position = (sorted.length - 1) * (step / classCount);
    const lower = Math.floor(position);
    const fraction = position - lower;
    thresholds.push(
      sorted.length
        ? sorted[lower] +
            (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction
        : 0
    );
  }
  return thresholds;
}

/** Quantize/quantile/linear-gradient color scale over finite numeric values. */
export function createContinuousColorScale(
  scaleType: string | null,
  colors: ColorTriple[],
  values: ArrayLike<number>
): ViewerColorScale {
  const palette = colors.length ? colors : [[30, 150, 190] as ColorTriple];
  const [minimum, maximum] = getFiniteExtent(Array.from(values).filter(Number.isFinite));
  const classCount = palette.length;
  let thresholds: number[];
  if (scaleType === 'quantile') thresholds = getQuantileThresholds(values, classCount);
  else {
    const width = (maximum - minimum) / classCount;
    thresholds = Array.from({length: classCount - 1}, (_, index) => minimum + width * (index + 1));
  }
  const useGradient = scaleType === 'linear' || scaleType === 'sqrt' || scaleType === 'log';
  const transform = (value: number) =>
    scaleType === 'sqrt'
      ? Math.sqrt(Math.max(value, 0))
      : scaleType === 'log'
        ? Math.log(Math.max(value, 1e-5))
        : value;
  const lower = transform(minimum);
  const span = transform(maximum) - lower || 1;
  return {
    kind: 'continuous',
    domain: useGradient ? [minimum, maximum] : [minimum, ...thresholds, maximum],
    categories: [],
    colors: palette,
    getColor(value) {
      if (useGradient) {
        const position = clamp((transform(value) - lower) / span, 0, 1) * (classCount - 1);
        const index = Math.floor(position);
        const next = palette[Math.min(index + 1, classCount - 1)];
        const fraction = position - index;
        const start = palette[index];
        return [0, 1, 2].map(channel =>
          Math.round(start[channel] + (next[channel] - start[channel]) * fraction)
        ) as ColorTriple;
      }
      let index = 0;
      while (index < thresholds.length && value >= thresholds[index]) index++;
      return palette[Math.min(index, classCount - 1)];
    },
    getCategoryColor: () => palette[0]
  };
}

/** Ordinal scale: sorted unique labels (Kepler order) cycle through the palette. */
export function createOrdinalColorScale(
  labels: ArrayLike<string | null>,
  colors: ColorTriple[]
): ViewerColorScale {
  const palette = colors.length ? colors : [[30, 150, 190] as ColorTriple];
  const categories = Array.from(
    new Set(Array.from(labels).filter((label): label is string => label !== null))
  ).sort();
  const lookup = new Map(
    categories.map((label, index) => [label, palette[index % palette.length]])
  );
  return {
    kind: 'ordinal',
    domain: [],
    categories,
    colors: palette,
    getColor: () => palette[0],
    getCategoryColor: label => lookup.get(label) ?? palette[0]
  };
}

/** Linear, sqrt or log mapping of a numeric domain into a size range. */
export function createSizeScale(
  scaleType: string | null,
  values: ArrayLike<number>,
  range: readonly [number, number]
): {domain: [number, number]; getSize(value: number): number} {
  const [minimum, maximum] = getFiniteExtent(Array.from(values).filter(Number.isFinite));
  const transform = (value: number) =>
    scaleType === 'sqrt'
      ? Math.sqrt(Math.max(value, 0))
      : scaleType === 'log'
        ? Math.log(Math.max(value, 1e-5))
        : value;
  const lower = transform(minimum);
  const span = transform(maximum) - lower || 1;
  return {
    domain: [minimum, maximum],
    getSize(value) {
      const fraction = clamp((transform(value) - lower) / span, 0, 1);
      return range[0] + (range[1] - range[0]) * fraction;
    }
  };
}
