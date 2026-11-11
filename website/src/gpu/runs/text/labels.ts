// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Ready-made text for the pictures runs draw most: chart axes, bar labels, a class legend and
// matrix ticks. Each returns a RunText (labels and rules) anchored in the same world units as the
// layers it labels, with defaults that do not overlap (ticks thin evenly, titles win).

import {
  getSwatchWidth,
  getTextWidth,
  type RunLabel,
  type RunRule,
  type RunText
} from './text-layout';

/** x0, y0, x1, y1 in world units. */
export type WorldRectangle = readonly [number, number, number, number];

/** The `chart` layer's default plot rectangle, as fractions of the world (define-run.ts). */
export const DEFAULT_CHART_FRACTIONS = [0.07, 0.14, 0.95, 0.88] as const;

/** The plot rectangle of a `chart` layer in world units, from its `rectangle` fractions. */
export function getChartRectangle(
  world: {width: number; height: number},
  fractions: readonly [number, number, number, number] = DEFAULT_CHART_FRACTIONS
): WorldRectangle {
  return [
    fractions[0] * world.width,
    fractions[1] * world.height,
    fractions[2] * world.width,
    fractions[3] * world.height
  ];
}

/** About `count` round tick values (steps of 1, 2, 2.5 or 5 times a power of ten) in the range. */
export function getNiceTicks(minimum: number, maximum: number, count = 4): number[] {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return [];
  if (maximum < minimum) [minimum, maximum] = [maximum, minimum];
  if (maximum === minimum) return [minimum];
  const rough = (maximum - minimum) / Math.max(count, 1);
  const power = 10 ** Math.floor(Math.log10(rough));
  const step =
    [1, 2, 2.5, 5, 10].map(factor => factor * power).find(value => value >= rough) ?? 10 * power;
  const ticks: number[] = [];
  const first = Math.ceil(minimum / step - 1e-9) * step;
  for (let value = first; value <= maximum + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
  }
  return ticks;
}

/**
 * A number for a tick: as many decimals as the spacing between ticks needs, a true minus sign,
 * and k or M for thousands and millions.
 */
export function formatTick(value: number, step = 0): string {
  const magnitude = Math.abs(value);
  const sign = value < 0 ? '−' : '';
  if (magnitude >= 1e6 && (step === 0 || step >= 1e5)) {
    return `${sign}${trimZeros((magnitude / 1e6).toFixed(step >= 1e6 ? 0 : 1))}M`;
  }
  if (magnitude >= 1e4 && (step === 0 || step >= 100)) {
    return `${sign}${trimZeros((magnitude / 1e3).toFixed(step >= 1e3 ? 0 : 1))}k`;
  }
  const decimals =
    step > 0
      ? Math.max(0, Math.min(6, Math.ceil(-Math.log10(step) - 1e-9) + (isHalfStep(step) ? 1 : 0)))
      : 2;
  return `${sign}${step > 0 ? magnitude.toFixed(decimals) : trimZeros(magnitude.toFixed(decimals))}`;
}

function isHalfStep(step: number): boolean {
  const scaled = step / 10 ** Math.floor(Math.log10(step));
  return Math.abs(scaled - 2.5) < 1e-6;
}

function trimZeros(text: string): string {
  return text.includes('.') ? text.replace(/\.?0+$/, '') : text;
}

function getStep(ticks: readonly number[]): number {
  return ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0;
}

/** Joins several RunTexts into one. */
export function mergeText(...parts: (RunText | null | undefined | false)[]): RunText {
  const labels: RunLabel[] = [];
  const rules: RunRule[] = [];
  for (const part of parts) {
    if (!part) continue;
    labels.push(...(part.labels ?? []));
    rules.push(...(part.rules ?? []));
  }
  return {labels, rules};
}

// ---------------------------------------------------------------------------------------------
// Chart axes

export type ValueAxis = {
  /** The data range mapped across the plot (the chart layer's `range` or `xRange`). */
  range: readonly [number, number];
  /** Tick values, or about how many. Default about 4. */
  ticks?: readonly number[] | number;
  format?: (value: number, step: number) => string;
  /** Short title with its unit, for example `r (km)` or `K(r)`. */
  title?: string;
  /** Draw gridlines across the plot at each tick (y axis default true, x default false). */
  grid?: boolean;
};

export type CategoryAxis = {
  /** One name per bar, in buffer order: bar i is centered at (i + 0.5) / count. */
  categories: readonly string[];
  title?: string;
};

export type ChartAxesOptions = {
  /** The plot rectangle in world units; see getChartRectangle. */
  rectangle: WorldRectangle;
  x?: ValueAxis | CategoryAxis;
  y?: ValueAxis;
  /** Reference values on y drawn as a dashed ink rule with a label: `{value: 0, label: 'CSR'}`. */
  references?: readonly {value: number; label?: string}[];
};

function isCategoryAxis(axis: ValueAxis | CategoryAxis): axis is CategoryAxis {
  return 'categories' in axis;
}

function getTickValues(axis: ValueAxis): number[] {
  if (Array.isArray(axis.ticks)) return [...(axis.ticks as readonly number[])];
  return getNiceTicks(
    axis.range[0],
    axis.range[1],
    typeof axis.ticks === 'number' ? axis.ticks : 4
  );
}

function toPlot(
  value: number,
  range: readonly [number, number],
  start: number,
  end: number
): number {
  const span = range[1] - range[0];
  return start + ((value - range[0]) / (Math.abs(span) < 1e-12 ? 1e-12 : span)) * (end - start);
}

/** Chart axes; draw `grid` as a text layer before the data and the rest after it. */
export type ChartAxes = RunText & {grid: RunText};

/**
 * Axes for a `chart` layer drawn in `rectangle`: an x baseline with ticks and labels below it,
 * y values written inside the plot just above faint gridlines (no left margin needed), the y title
 * on the top gridline after its value, and the x title at the right end of the x labels.
 */
export function createChartAxes({rectangle, x, y, references}: ChartAxesOptions): ChartAxes {
  const [x0, y0, x1, y1] = rectangle;
  const labels: RunLabel[] = [];
  const rules: RunRule[] = [{from: [x0, y0], to: [x1, y0], tone: 'muted'}];
  const grid: RunRule[] = [];
  if (x) {
    if (isCategoryAxis(x)) {
      const count = Math.max(x.categories.length, 1);
      x.categories.forEach((name, index) => {
        labels.push({
          text: name,
          at: [x0 + ((index + 0.5) / count) * (x1 - x0), y0],
          anchor: 'top',
          offset: [0, -3],
          role: 'tick',
          group: 'x-categories'
        });
      });
    } else {
      const ticks = getTickValues(x);
      const step = getStep(ticks);
      const format = x.format ?? formatTick;
      for (const value of ticks) {
        const position = toPlot(value, x.range, x0, x1);
        if (position < x0 - 1e-9 || position > x1 + 1e-9) continue;
        rules.push({from: [position, y0], to: [position, y0], toOffset: [0, -3], tone: 'muted'});
        if (x.grid) grid.push({from: [position, y0], to: [position, y1], opacity: 0.3});
        labels.push({
          text: format(value, step),
          at: [position, y0],
          anchor: 'top',
          offset: [0, -4],
          role: 'tick',
          group: 'x-ticks'
        });
      }
    }
    if (x.title) {
      // Category names fill the row under the bars: the title goes in the plot's top-right corner.
      const isCategories = isCategoryAxis(x);
      labels.push({
        text: x.title,
        at: [x1, isCategories ? y1 : y0],
        anchor: isCategories ? 'bottom-right' : 'top-right',
        offset: isCategories ? [0, 1] : [0, -4],
        role: 'title',
        priority: 3
      });
    }
  }
  if (y) {
    const ticks = getTickValues(y);
    const step = getStep(ticks);
    const format = y.format ?? formatTick;
    let top: {position: number; text: string} | null = null;
    for (const value of ticks) {
      const position = toPlot(value, y.range, y0, y1);
      if (position < y0 - 1e-9 || position > y1 + 1e-9) continue;
      if ((y.grid ?? true) && Math.abs(position - y0) > 1e-9) {
        grid.push({from: [x0, position], to: [x1, position], opacity: 0.3});
      }
      const text = format(value, step);
      labels.push({
        text,
        at: [x0, position],
        anchor: 'bottom-left',
        offset: [1, 1],
        role: 'tick',
        group: 'y-ticks'
      });
      if (!top || position > top.position) top = {position, text};
    }
    if (y.title) {
      // On the top gridline, after its value; above the plot when there is no tick.
      labels.push({
        text: y.title,
        at: [x0, top ? top.position : y1],
        anchor: 'bottom-left',
        offset: top ? [1 + getTextWidth(top.text, 'tick') + 4, 1] : [0, 5],
        role: 'title',
        priority: 3
      });
    }
    for (const reference of references ?? []) {
      const position = toPlot(reference.value, y.range, y0, y1);
      if (position < y0 - 1e-9 || position > y1 + 1e-9) continue;
      rules.push({from: [x0, position], to: [x1, position], tone: 'ink', dash: 3, opacity: 0.8});
      if (reference.label) {
        labels.push({
          text: reference.label,
          at: [x1, position],
          anchor: 'bottom-right',
          offset: [0, 1],
          role: 'label',
          priority: 2
        });
      }
    }
  }
  return {labels, rules, grid: {rules: grid}};
}

// ---------------------------------------------------------------------------------------------
// Bars

export type BarLabelsOptions = {
  /** The plot rectangle in world units (the same as the chart layer's). */
  rectangle: WorldRectangle;
  /** CPU copies of the bar values (from `peek`, `extra` or the inputs). */
  values: ArrayLike<number>;
  /** The chart layer's y range. */
  range: readonly [number, number];
  /**
   * The value bars grow from. Default `range[0]`, which is how the `chart` layer draws bars (a bar
   * runs from the bottom of the range to its value). Pass 0 for bars drawn from a zero line: values
   * below it are labelled under the bar's end instead of above it.
   */
  baseline?: number;
  format?: (value: number) => string;
  /** Only label these bars (for example the hovered one and the largest). Default all. */
  indices?: readonly number[];
};

/** Value labels just above each bar's top (inside the plot top when a bar is full height). */
export function createBarValueLabels({
  rectangle,
  values,
  range,
  format = value => formatTick(value),
  indices,
  baseline = range[0]
}: BarLabelsOptions): RunText {
  const [x0, y0, x1, y1] = rectangle;
  const count = Math.max(values.length, 1);
  const chosen = indices ?? Array.from({length: values.length}, (_, index) => index);
  const labels: RunLabel[] = [];
  for (const index of chosen) {
    const value = values[index];
    if (!Number.isFinite(value) || Math.abs(value) >= 1e20) continue;
    const end = Math.min(y1, Math.max(y0, toPlot(value, range, y0, y1)));
    const isBelow = value < baseline;
    labels.push({
      text: format(value),
      at: [x0 + ((index + 0.5) / count) * (x1 - x0), end],
      anchor: isBelow ? 'top' : 'bottom',
      offset: [0, isBelow ? -2 : 2],
      role: 'value',
      group: 'bar-values'
    });
  }
  return {labels};
}

// ---------------------------------------------------------------------------------------------
// Legend

export type LegendEntry = {
  label: string;
  /** Categorical index (theme colors, the same index the uint32 layer draws) or RGB 0 to 1. */
  swatch: number | readonly [number, number, number];
};

export type LegendOptions = {
  entries: readonly LegendEntry[];
  title?: string;
  /** Default `top-right`. */
  corner?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  /** Distance from the canvas edges in text pixels. Default 6. */
  inset?: number;
  /** Draw a background panel behind the legend. Default true. */
  panel?: boolean;
};

const LEGEND_LINE = 13;

/** A class legend in a canvas corner: one swatch and name per line, optional title, on a panel. */
export function createLegend({
  entries,
  title,
  corner = 'top-right',
  inset = 6,
  panel = true
}: LegendOptions): RunText {
  const swatchWidth = getSwatchWidth('label');
  const width = Math.max(
    title ? getTextWidth(title, 'title') : 0,
    ...entries.map(entry => swatchWidth + getTextWidth(entry.label, 'label'))
  );
  const lineCount = entries.length + (title ? 1 : 0);
  const height = lineCount * LEGEND_LINE;
  const isRight = corner.endsWith('right');
  const isBottom = corner.startsWith('bottom');
  const at: readonly [number, number] = [isRight ? 1 : 0, isBottom ? 1 : 0];
  // Offsets of the legend's top-left corner from the canvas corner, in text pixels (y up).
  const left = isRight ? -(inset + width) : inset;
  const top = isBottom ? inset + height : -inset;
  const labels: RunLabel[] = [];
  const rules: RunRule[] = [];
  if (panel) {
    const padding = 4;
    rules.push({
      from: at,
      to: at,
      space: 'canvas',
      fromOffset: [left - padding, top - height / 2],
      toOffset: [left + width + padding, top - height / 2],
      width: height + 2 * padding,
      tone: 'background',
      opacity: 0.78
    });
  }
  let line = 0;
  if (title) {
    labels.push({
      text: title,
      at,
      space: 'canvas',
      anchor: 'left',
      offset: [left, top - LEGEND_LINE / 2],
      role: 'title',
      priority: 5
    });
    line++;
  }
  for (const entry of entries) {
    labels.push({
      text: entry.label,
      at,
      space: 'canvas',
      anchor: 'left',
      offset: [left, top - (line + 0.5) * LEGEND_LINE],
      swatch: entry.swatch,
      role: 'label',
      priority: 4
    });
    line++;
  }
  return {labels, rules};
}

// ---------------------------------------------------------------------------------------------
// Matrix

export type MatrixTicksOptions = {
  /** The matrix in world units (row 0 at the top), as given to createMatrixMesh. */
  rectangle: WorldRectangle;
  rows: readonly string[];
  columns: readonly string[];
  /** Where column names go. Default `top`. */
  columnSide?: 'top' | 'bottom';
  /** Write column names bottom to top, for long names. */
  rotateColumns?: boolean;
  /** For example `from` and `to`. */
  rowTitle?: string;
  columnTitle?: string;
};

/** Row names left of a matrix and column names above (or below) it, thinned evenly when crowded. */
export function createMatrixTicks({
  rectangle,
  rows,
  columns,
  columnSide = 'top',
  rotateColumns = false,
  rowTitle,
  columnTitle
}: MatrixTicksOptions): RunText {
  const [x0, y0, x1, y1] = rectangle;
  const labels: RunLabel[] = [];
  rows.forEach((name, row) => {
    labels.push({
      text: name,
      at: [x0, y1 - ((row + 0.5) / rows.length) * (y1 - y0)],
      anchor: 'right',
      offset: [-3, 0],
      role: 'tick',
      group: 'matrix-rows'
    });
  });
  const isTop = columnSide === 'top';
  columns.forEach((name, column) => {
    const x = x0 + ((column + 0.5) / columns.length) * (x1 - x0);
    labels.push({
      text: name,
      at: [x, isTop ? y1 : y0],
      anchor: rotateColumns ? (isTop ? 'left' : 'right') : isTop ? 'bottom' : 'top',
      angle: rotateColumns ? 90 : 0,
      offset: [0, isTop ? 3 : -3],
      role: 'tick',
      group: 'matrix-columns'
    });
  });
  if (rowTitle) {
    labels.push({
      text: rowTitle,
      at: [x0, y1],
      anchor: 'bottom-right',
      offset: [-3, 3],
      role: 'title',
      priority: 3
    });
  }
  if (columnTitle) {
    labels.push({
      text: columnTitle,
      at: [(x0 + x1) / 2, isTop ? y1 : y0],
      anchor: isTop ? 'bottom' : 'top',
      offset: [0, isTop ? 15 : -15],
      role: 'title',
      priority: 3
    });
  }
  return {labels};
}
