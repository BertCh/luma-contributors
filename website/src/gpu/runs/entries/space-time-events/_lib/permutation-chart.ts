// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Local copy of spatial-statistics/shared/permutation-chart.ts. The reference distribution of a permutation test as a histogram, with the observed statistic
// marked on it. The histogram is binned on the CPU from the GPU-written shuffled statistics, over a
// range that includes the observed value, and written into a render-only buffer.

import type {RunBuffer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';

export type ChartRectangle = readonly [number, number, number, number];

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
