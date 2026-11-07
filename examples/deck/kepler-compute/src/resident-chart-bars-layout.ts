// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Above this many bars the SVG chart is replaced by a canvas renderer. */
export const CANVAS_BAR_THRESHOLD = 512;

export type BarColumn = {
  /** Pixel column index. */
  column: number;
  /** Largest bar value in the column (bars drawn from the baseline, so max dominates). */
  value: number;
  /** Sum of row counts of the covered bins. */
  count: number;
  /** First and last bin index covered, inclusive. */
  firstIndex: number;
  lastIndex: number;
};

export type BarLayout = {
  columns: BarColumn[];
  /** Bin indexes covered per pixel column (at least 1 when bars are wider than a pixel). */
  binsPerColumn: number;
  /** Horizontal pixels per bin, when at least one pixel wide. */
  barWidth: number;
  binCount: number;
  plotWidth: number;
};

export function shouldUseCanvasBars(binCount: number): boolean {
  return binCount > CANVAS_BAR_THRESHOLD;
}

/**
 * Layout bins across plotWidth pixels. Wide bars map one-to-one. Bars narrower than a pixel are
 * aggregated per pixel column: value by absolute maximum (so spikes never vanish), count by sum.
 */
export function makeBarLayout(
  values: ArrayLike<number>,
  counts: ArrayLike<number>,
  plotWidth: number
): BarLayout {
  const binCount = values.length;
  const barWidth = binCount > 0 ? plotWidth / binCount : 0;
  const columns: BarColumn[] = [];
  if (binCount === 0 || plotWidth <= 0)
    return {columns, binsPerColumn: 1, barWidth, binCount, plotWidth};
  if (barWidth >= 1) {
    for (let index = 0; index < binCount; index++)
      columns.push({
        column: Math.floor(index * barWidth),
        value: values[index],
        count: counts[index],
        firstIndex: index,
        lastIndex: index
      });
    return {columns, binsPerColumn: 1, barWidth, binCount, plotWidth};
  }
  const columnCount = Math.max(1, Math.floor(plotWidth));
  const binsPerColumn = binCount / columnCount;
  for (let column = 0; column < columnCount; column++) {
    const firstIndex = Math.floor(column * binsPerColumn);
    const nextIndex =
      column === columnCount - 1 ? binCount : Math.floor((column + 1) * binsPerColumn);
    const lastIndex = Math.max(firstIndex, nextIndex - 1);
    let value = 0;
    let count = 0;
    for (let index = firstIndex; index <= lastIndex; index++) {
      if (Math.abs(values[index]) > Math.abs(value)) value = values[index];
      count += counts[index];
    }
    columns.push({column, value, count, firstIndex, lastIndex});
  }
  return {columns, binsPerColumn, barWidth, binCount, plotWidth};
}

/** Map a pixel x (relative to the plot's left edge) to a bin index, or -1 outside the plot. */
export function getBarIndexAtPixel(layout: BarLayout, pixelX: number): number {
  if (!(pixelX >= 0) || pixelX >= layout.plotWidth || layout.binCount === 0) return -1;
  return Math.min(layout.binCount - 1, Math.floor((pixelX / layout.plotWidth) * layout.binCount));
}

/** Inclusive bin range under a pixel x, for hit-testing aggregated columns. */
export function getBarRangeAtPixel(
  layout: BarLayout,
  pixelX: number
): readonly [number, number] | undefined {
  const first = getBarIndexAtPixel(layout, pixelX);
  if (first < 0) return undefined;
  if (layout.barWidth >= 1) return [first, first];
  const column = layout.columns[Math.min(layout.columns.length - 1, Math.floor(pixelX))];
  return column ? [column.firstIndex, column.lastIndex] : [first, first];
}

/** Bin with the largest absolute value inside a range, used so tooltips match the drawn bar. */
export function getDominantBarIndex(
  values: ArrayLike<number>,
  range: readonly [number, number]
): number {
  let best = range[0];
  for (let index = range[0]; index <= range[1]; index++)
    if (Math.abs(values[index]) > Math.abs(values[best])) best = index;
  return best;
}
