// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Series charts beside the map: plot rectangles in the chart area, dot and line layers, and the
// glyphs (frames, rules, year ticks) that frame them. Charts are scatter plots over real time, so
// irregular acquisitions sit where they were taken.

import type {DemoTheme} from '../../../../demos/kit';
import type {ChartRunLayer, RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';
import {CHART_FRACTION} from './stacks';

export type Rectangle = readonly [number, number, number, number];
type Point = [number, number];
type World = {width: number; height: number};
export type Rgba = readonly [number, number, number, number];

/** Plot rectangles, as fractions of the view, stacked from the top of the chart area. */
export function getChartRectangles(count: number): Rectangle[] {
  const left = 0.035;
  const right = CHART_FRACTION - 0.025;
  const gap = 0.12;
  const top = 0.94;
  const bottom = 0.1;
  const height = (top - bottom - gap * (count - 1)) / count;
  return Array.from({length: count}, (_, index) => {
    const upper = top - index * (height + gap);
    return [left, upper - height, right, upper] as const;
  });
}

export function getColor(color: readonly number[], alpha = 1): Rgba {
  return [color[0], color[1], color[2], alpha];
}

/** Dots at (`xValues`, `values`) in a plot rectangle. */
export function getDotLayer(
  values: RunBuffer,
  xValues: RunBuffer,
  rectangle: Rectangle,
  range: readonly [number, number],
  xRange: readonly [number, number],
  color: Rgba,
  radius = 2.6
): ChartRunLayer {
  return {
    layer: 'chart',
    kind: 'scatter',
    values,
    xValues,
    rectangle,
    range,
    xRange,
    color,
    radius
  };
}

/** A polyline through evenly spaced values. */
export function getLineLayer(
  values: RunBuffer,
  rectangle: Rectangle,
  range: readonly [number, number],
  color: Rgba,
  radius = 1.4
): ChartRunLayer {
  return {layer: 'chart', kind: 'line', values, rectangle, range, color, radius};
}

/** Bars over a plot rectangle. */
export function getBarLayer(
  values: RunBuffer,
  rectangle: Rectangle,
  range: readonly [number, number],
  color: Rgba
): ChartRunLayer {
  return {layer: 'chart', kind: 'bars', values, rectangle, range, color};
}

/** World position of a chart value. */
export function getChartPoint(
  world: World,
  rectangle: Rectangle,
  xRange: readonly [number, number],
  range: readonly [number, number],
  x: number,
  y: number
): Point {
  const unitX = (x - xRange[0]) / (xRange[1] - xRange[0]);
  const unitY = (y - range[0]) / (range[1] - range[0]);
  return [
    world.width * (rectangle[0] + Math.min(1, Math.max(0, unitX)) * (rectangle[2] - rectangle[0])),
    world.height * (rectangle[1] + Math.min(1, Math.max(0, unitY)) * (rectangle[3] - rectangle[1]))
  ];
}

export function getFrameGlyph(world: World, rectangle: Rectangle): RunGlyph {
  return {
    shape: 'box',
    minimum: [world.width * rectangle[0], world.height * rectangle[1]],
    maximum: [world.width * rectangle[2], world.height * rectangle[3]]
  };
}

/** A vertical rule at chart x, full height of the plot. */
export function getVerticalRule(
  world: World,
  rectangle: Rectangle,
  xRange: readonly [number, number],
  x: number,
  dashed = false
): RunGlyph {
  const unitX = Math.min(1, Math.max(0, (x - xRange[0]) / (xRange[1] - xRange[0])));
  const worldX = world.width * (rectangle[0] + unitX * (rectangle[2] - rectangle[0]));
  return {
    shape: 'line',
    from: [worldX, world.height * rectangle[1]],
    to: [worldX, world.height * rectangle[3]],
    dashed
  };
}

/** A horizontal rule at chart value y, full width of the plot. */
export function getHorizontalRule(
  world: World,
  rectangle: Rectangle,
  range: readonly [number, number],
  y: number,
  dashed = true
): RunGlyph {
  const unitY = Math.min(1, Math.max(0, (y - range[0]) / (range[1] - range[0])));
  const worldY = world.height * (rectangle[1] + unitY * (rectangle[3] - rectangle[1]));
  return {
    shape: 'line',
    from: [world.width * rectangle[0], worldY],
    to: [world.width * rectangle[2], worldY],
    dashed
  };
}

/**
 * Short ticks at 1 January of each calendar year inside the plot. `midpointSeconds` is the
 * midpoint of the first and last acquisition, the origin of centred fractional years.
 */
export function getYearTicks(
  world: World,
  rectangle: Rectangle,
  xRange: readonly [number, number],
  midpointSeconds: number
): RunGlyph[] {
  const ticks: RunGlyph[] = [];
  const midpointYear = new Date(midpointSeconds * 1000).getUTCFullYear();
  for (let year = midpointYear - 12; year <= midpointYear + 12; year++) {
    const centred = (Date.UTC(year, 0, 1) / 1000 - midpointSeconds) / 31556952;
    if (centred < xRange[0] || centred > xRange[1]) continue;
    const unitX = (centred - xRange[0]) / (xRange[1] - xRange[0]);
    const worldX = world.width * (rectangle[0] + unitX * (rectangle[2] - rectangle[0]));
    ticks.push({
      shape: 'line',
      from: [worldX, world.height * rectangle[1]],
      to: [worldX, world.height * (rectangle[1] + 0.03)]
    });
  }
  return ticks;
}

/** Midpoint of the first and last time, in seconds. */
export function getMidpointSeconds(times: BigInt64Array): number {
  return Number(times[0] + (times[times.length - 1] - times[0]) / 2n);
}

/**
 * Writes the finite points first and repeats the last of them to fill the buffers, so a scatter
 * layer of fixed length draws only real points. Returns how many points are real.
 */
export function writeCompactPoints(
  xBuffer: RunBuffer,
  yBuffer: RunBuffer,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  fallbackY?: number
): number {
  const capacity = xBuffer.length;
  const outputX = new Float32Array(capacity);
  const outputY = new Float32Array(capacity);
  let count = 0;
  for (let index = 0; index < xs.length && count < capacity; index++) {
    if (!Number.isFinite(ys[index]) || !Number.isFinite(xs[index])) continue;
    outputX[count] = xs[index];
    outputY[count] = ys[index];
    count++;
  }
  const lastX = count > 0 ? outputX[count - 1] : -1e9;
  const lastY = count > 0 ? outputY[count - 1] : (fallbackY ?? -1e9);
  for (let index = count; index < capacity; index++) {
    outputX[index] = lastX;
    outputY[index] = lastY;
  }
  xBuffer.write(outputX);
  yBuffer.write(outputY);
  return count;
}

/** The map layer: a raster over the whole world with the chart columns clipped away. */
export function getMapLayer(
  values: RunBuffer,
  grid: {columns: number; rows: number},
  mapLeft: number,
  options: Partial<Extract<RunLayer, {layer: 'raster'}>> = {}
): RunLayer {
  return {
    layer: 'raster',
    values,
    columns: grid.columns,
    rows: grid.rows,
    smoothing: false,
    clip: {side: 'right', x: mapLeft},
    ...options
  };
}

export function getAccent(theme: DemoTheme, index: number): Rgba {
  return getColor(theme.categories[index % theme.categories.length], 0.95);
}

export function getInk(theme: DemoTheme, alpha = 0.9): Rgba {
  return getColor(theme.ink, alpha);
}

export function getMutedInk(theme: DemoTheme, alpha = 0.7): Rgba {
  return getColor(theme.muted, alpha);
}

/** The x and y buffers of one dotted series. */
export type Dots = {x: RunBuffer; y: RunBuffer};

export function allocateDots(builder: RunBuilder, id: string, capacity: number): Dots {
  return {
    x: builder.allocate(`${id}-x`, 'float32', capacity),
    y: builder.allocate(`${id}-y`, 'float32', capacity)
  };
}

/** Writes the finite points of a series; see {@link writeCompactPoints}. */
export function writeDots(
  dots: Dots,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  fallbackY?: number
): number {
  return writeCompactPoints(dots.x, dots.y, xs, ys, fallbackY);
}

export function getDotsLayer(
  dots: Dots,
  rectangle: Rectangle,
  range: readonly [number, number],
  xRange: readonly [number, number],
  color: Rgba,
  radius = 4
): ChartRunLayer {
  return getDotLayer(dots.y, dots.x, rectangle, range, xRange, color, radius);
}
