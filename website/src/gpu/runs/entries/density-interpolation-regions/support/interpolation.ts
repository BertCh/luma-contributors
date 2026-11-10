// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs shared by the interpolation runs: scattered samples and a query grid.

import {createUniformPoints, type PointsInput, type Random, type World} from '../../../inputs';
import type {RunBuffer, RunLayer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';
import {createRunGrid, type RunGrid} from './grid';

export type InterpolationInputs = {
  grid: RunGrid;
  samples: PointsInput;
  /** Observation rows [value, errorVariance, 0, 1] of the samples. */
  observations: Float32Array;
};

/**
 * Rank-stretches measurements so they fill 0.05 to 0.95. The rain gauges are skewed toward dry
 * stations, so the plain ramp would show one hotspot on a dark field.
 */
export function stretchValues(values: Float32Array): Float32Array {
  const order = Array.from(values.keys()).sort((first, second) => values[first] - values[second]);
  const stretched = new Float32Array(values.length);
  order.forEach((row, rank) => {
    stretched[row] = 0.05 + (0.9 * rank) / Math.max(1, values.length - 1);
  });
  return stretched;
}

/** Rain-gauge samples with stretched values. */
export function createStretchedSamples(
  random: Random,
  world: World,
  sampleCount: number,
  noise = 0.05
): PointsInput {
  const samples = createUniformPoints(random, world, sampleCount, noise);
  return {...samples, values: stretchValues(samples.values)};
}

export function createInterpolationInputs(
  random: () => number,
  world: {width: number; height: number},
  rows = 40,
  sampleCount = 60
): InterpolationInputs {
  const samples = createStretchedSamples(random, world, sampleCount);
  const observations = new Float32Array(sampleCount * 4);
  samples.values.forEach((value, index) => {
    observations.set([value, 0.01, 0, 1], index * 4);
  });
  return {grid: createRunGrid(world, rows), samples, observations};
}

/** Sample dots colored by their value on the surface's ramp, with a dark ring so they stay visible. */
export function drawSampleDots(
  positions: RunBuffer<'float32x2'>,
  values: RunBuffer<'float32'>,
  count?: number
): RunLayer[] {
  return [
    {
      layer: 'dots',
      context: true,
      positions,
      count,
      radius: 5,
      color: [0.04, 0.04, 0.06, 0.9]
    },
    {layer: 'dots', positions, values, count, radius: 3.4, range: [0, 1]}
  ];
}

/** Indices of the `count` samples nearest to (x, y), nearest first. */
export function findNearestSamples(
  positions: Float32Array,
  x: number,
  y: number,
  count: number
): number[] {
  return Array.from({length: positions.length >> 1}, (_, row) => row)
    .sort(
      (first, second) =>
        Math.hypot(positions[first * 2] - x, positions[first * 2 + 1] - y) -
        Math.hypot(positions[second * 2] - x, positions[second * 2 + 1] - y)
    )
    .slice(0, count);
}

/** Spokes from a grid cell to the samples a local interpolator fitted for it, as glyphs. */
export function createNeighborhoodGlyphs(
  positions: Float32Array,
  at: readonly [number, number],
  neighborCount: number,
  isDashed = false
): RunGlyph[] {
  const glyphs: RunGlyph[] = [];
  for (const row of findNearestSamples(positions, at[0], at[1], Math.min(neighborCount, 30))) {
    const to = [positions[row * 2], positions[row * 2 + 1]] as const;
    glyphs.push({shape: 'line', from: at, to, dashed: isDashed}, {shape: 'dot', at: to});
  }
  return glyphs;
}

/** Convex hull of the first `count` samples as segments (x0, y0, x1, y1), padded with NaN rows to `capacity`. */
export function createHullSegments(
  positions: Float32Array,
  count: number,
  capacity: number
): Float32Array {
  const points = Array.from({length: count}, (_, row) => [
    positions[row * 2],
    positions[row * 2 + 1]
  ]);
  points.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const build = (ordered: number[][]) => {
    const chain: number[][] = [];
    for (const point of ordered) {
      while (
        chain.length >= 2 &&
        cross(chain[chain.length - 2], chain[chain.length - 1], point) <= 0
      ) {
        chain.pop();
      }
      chain.push(point);
    }
    chain.pop();
    return chain;
  };
  const hull = [...build(points), ...build([...points].reverse())];
  const segments = new Float32Array(capacity * 4).fill(Number.NaN);
  hull.forEach((point, index) => {
    const next = hull[(index + 1) % hull.length];
    segments.set([point[0], point[1], next[0], next[1]], index * 4);
  });
  return segments;
}
