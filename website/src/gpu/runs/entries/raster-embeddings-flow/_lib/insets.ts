// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Chart insets for the embedding runs: a backing panel (theme background at 0.88, the shared
// helper of `space-time-models`) under bars colored one per class or component.

import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import type {DemoTheme} from '../../../../demos/kit';
import {
  createInsetBacking,
  getInsetBackingLayer,
  type InsetBacking
} from '../../space-time-models/_lib/inset';

/** Fractions of the view: x0, y0, x1, y1 with y up. */
export type InsetFractions = readonly [number, number, number, number];

export const BAR_INSET: InsetFractions = [0.03, 0.05, 0.22, 0.28];

export type ColoredBars = {backing: InsetBacking; bars: RunBuffer[]; rectangle: InsetFractions};

/** One single-value bar buffer per class, so each bar takes its class color. */
export function createColoredBars(
  builder: RunBuilder,
  id: string,
  count: number,
  worldWidth: number,
  rectangle: InsetFractions = BAR_INSET
): ColoredBars {
  return {
    backing: createInsetBacking(
      builder,
      `${id}-backing`,
      getPaddedRectangle(rectangle, worldWidth)
    ),
    bars: Array.from({length: count}, (_, bar) =>
      builder.allocate(`${id}-bar-${bar}`, 'float32', 1)
    ),
    rectangle
  };
}

export function writeColoredBars(bars: ColoredBars, values: ArrayLike<number>): void {
  bars.bars.forEach((buffer, bar) => buffer.write(Float32Array.of(values[bar] ?? 0)));
}

export function getColoredBarLayers(
  bars: ColoredBars,
  theme: DemoTheme,
  range: readonly [number, number] = [0, 1]
): RunLayer[] {
  const [x0, y0, x1, y1] = bars.rectangle;
  const slot = (x1 - x0) / bars.bars.length;
  return [
    getInsetBackingLayer(bars.backing, theme),
    ...bars.bars.map(
      (values, bar): RunLayer => ({
        layer: 'chart',
        kind: 'bars',
        values,
        range,
        color: [...theme.categories[bar % theme.categories.length].slice(0, 3), 1] as [
          number,
          number,
          number,
          number
        ],
        rectangle: [x0 + bar * slot, y0, x0 + (bar + 1) * slot, y1]
      })
    )
  ];
}

/** The inset rectangle in world units with a small margin, for the backing panel. */
export function getPaddedRectangle(rectangle: InsetFractions, worldWidth: number) {
  const margin = 0.015;
  return {
    x0: (rectangle[0] - margin) * worldWidth,
    y0: rectangle[1] - margin,
    x1: (rectangle[2] + margin) * worldWidth,
    y1: rectangle[3] + margin
  };
}
