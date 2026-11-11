// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The reference distribution of a permutation test as a histogram, with the observed statistic
// marked on it. The histogram is binned on the CPU from the GPU-written shuffled statistics, over a
// range that includes the observed value, and written into a render-only buffer.

import type {RunBuffer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';
import {
  formatTick,
  getChartRectangle,
  getNiceTicks,
  type RunLabel,
  type RunRule,
  type RunText
} from '../../../text';

export type ChartRectangle = readonly [number, number, number, number];

/** What the latest binning of a histogram buffer covers, for the text that labels it. */
type ReferenceExtent = {
  minimum: number;
  maximum: number;
  mean: number;
  observed: number;
  pseudoP?: number;
};

const extents = new WeakMap<RunBuffer<'float32'>, ReferenceExtent>();

export type ReferenceSummary = {
  mean: number;
  /** Shuffled statistics at least as large as the observed one. */
  greaterCount: number;
  sampleCount: number;
};

/**
 * Bins `samples[start .. start + sampleCount)` into `histogram` (heights scaled so the tallest bin
 * is 1) and returns the glyphs of the chart: baseline, observed marker and the shuffled mean.
 */
export function drawReferenceHistogram(options: {
  histogram: RunBuffer<'float32'>;
  samples: ArrayLike<number> | null;
  start?: number;
  sampleCount: number;
  observed: number;
  rectangle: ChartRectangle;
  world: {width: number; height: number};
  /** The pseudo p-value the operation wrote, shown beside the observed value. */
  pseudoP?: number;
}): {glyphs: RunGlyph[]; summary: ReferenceSummary | null} {
  const {histogram, samples, observed, rectangle, world, sampleCount} = options;
  if (!samples || !Number.isFinite(observed)) return {glyphs: [], summary: null};
  const start = options.start ?? 0;
  let minimum = observed;
  let maximum = observed;
  let total = 0;
  let used = 0;
  let greaterCount = 0;
  for (let index = 0; index < sampleCount; index++) {
    const sample = samples[start + index];
    if (!Number.isFinite(sample)) continue;
    minimum = Math.min(minimum, sample);
    maximum = Math.max(maximum, sample);
    total += sample;
    used++;
    if (sample >= observed) greaterCount++;
  }
  if (used === 0) return {glyphs: [], summary: null};
  const padding = Math.max((maximum - minimum) * 0.05, 1e-9);
  minimum -= padding;
  maximum += padding;
  const binCount = histogram.length;
  const counts = new Float32Array(binCount);
  for (let index = 0; index < sampleCount; index++) {
    const sample = samples[start + index];
    if (!Number.isFinite(sample)) continue;
    const bin = Math.min(
      binCount - 1,
      Math.floor(((sample - minimum) / (maximum - minimum)) * binCount)
    );
    counts[bin]++;
  }
  const tallest = Math.max(...counts);
  histogram.write(counts.map(count => count / tallest));
  extents.set(histogram, {
    minimum,
    maximum,
    mean: total / used,
    observed,
    pseudoP: options.pseudoP
  });
  const [left, bottom, right, top] = rectangle;
  const getX = (value: number) =>
    (left + ((value - minimum) / (maximum - minimum)) * (right - left)) * world.width;
  const mean = total / used;
  const glyphs: RunGlyph[] = [
    {
      shape: 'line',
      from: [left * world.width, bottom * world.height],
      to: [right * world.width, bottom * world.height]
    },
    {
      shape: 'line',
      from: [getX(mean), bottom * world.height],
      to: [getX(mean), top * world.height],
      dashed: true
    },
    {
      shape: 'line',
      from: [getX(observed), bottom * world.height],
      to: [getX(observed), (top + 0.03) * world.height]
    }
  ];
  return {glyphs, summary: {mean, greaterCount, sampleCount: used}};
}

/**
 * Text for the reference histogram, for `draw` (call it with the same `histogram` and `rectangle`
 * as `drawReferenceHistogram`): a title naming the shuffled statistic, a few round values of the
 * statistic along the baseline, and the observed value with its pseudo p-value at the solid line.
 * Empty until the first binning.
 */
export function createReferenceHistogramText(options: {
  histogram: RunBuffer<'float32'>;
  rectangle: ChartRectangle;
  world: {width: number; height: number};
  /** For example `Knox count`: the title reads `shuffled <statistic>`. */
  statistic: string;
  decimals: number;
}): RunText {
  const extent = extents.get(options.histogram);
  if (!extent) return {};
  const {minimum, maximum, mean, observed, pseudoP} = extent;
  const [x0, y0, x1, y1] = getChartRectangle(options.world, options.rectangle);
  const getX = (value: number) => x0 + ((value - minimum) / (maximum - minimum)) * (x1 - x0);
  const labels: RunLabel[] = [];
  const rules: RunRule[] = [];
  const ticks = getNiceTicks(minimum, maximum, 3);
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0;
  for (const value of ticks) {
    // The dashed mean line would run through the number.
    if (Math.abs(value - mean) < 0.05 * (maximum - minimum)) continue;
    labels.push({
      text: formatTick(value, step),
      at: [getX(value), y0],
      anchor: 'bottom',
      offset: [0, 1],
      role: 'tick',
      tone: 'ink',
      group: 'statistic-ticks'
    });
  }
  labels.push({
    text: `shuffled ${options.statistic}`,
    at: [x0, y1],
    anchor: 'bottom-left',
    offset: [0, 4],
    role: 'title',
    priority: 3
  });
  const text = `observed ${observed.toFixed(options.decimals)}`;
  labels.push({
    text: pseudoP === undefined ? text : `${text} · p ${pseudoP.toFixed(3)}`,
    at: [getX(observed), y1 + 0.03 * options.world.height],
    anchor: 'bottom',
    offset: [0, 2],
    role: 'value',
    priority: 4
  });
  return {labels, rules};
}
