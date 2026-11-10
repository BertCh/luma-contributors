// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Binary64 CPU reference for the STX-06 change point kernels: the unpruned optimal partitioning
// dynamic program with the segment costs written from the published `changepoint` definitions
// (Killick and Eckley 2014), independent of the kernel's standardized and deviance forms. The
// external R fixture (changepoint is GPL and R-only) is pending; ruptures' Pelt covers mean, slope
// and mean-variance in `fixtures/stx06-change-points.json`.

import type {GPULocationChangePointCost} from '../../../src/gpu-spatial-analysis/change-points/index';

/** One segmentation: change points as first slices of the later segments, and the penalized objective. */
export type ChangePointReference = {changePoints: number[]; optimalCost: number};

const VARIANCE_FLOOR = 1e-11;

function getSegmentCost(
  cost: GPULocationChangePointCost,
  series: number[],
  positions: number[],
  start: number,
  end: number,
  overallMean: number
): number {
  const length = end - start;
  let total = 0;
  let squares = 0;
  for (let row = start; row < end; row++) {
    total += series[row];
    squares += series[row] * series[row];
  }
  switch (cost) {
    case 'mean':
      return squares - (total * total) / length;
    case 'variance': {
      let deviations = 0;
      for (let row = start; row < end; row++) {
        deviations += (series[row] - overallMean) ** 2;
      }
      const variance = deviations <= 0 ? VARIANCE_FLOOR : deviations / length;
      return length * (Math.log(2 * Math.PI) + Math.log(variance) + 1);
    }
    case 'mean-variance': {
      const variance = (squares - (total * total) / length) / length;
      const floored = variance <= 0 ? VARIANCE_FLOOR : variance;
      return length * (Math.log(2 * Math.PI) + Math.log(floored) + 1);
    }
    case 'poisson':
      return total === 0 ? 0 : 2 * total * (Math.log(length) - Math.log(total));
    case 'slope': {
      let meanPosition = 0;
      for (let row = start; row < end; row++) meanPosition += positions[row];
      meanPosition /= length;
      const meanValue = total / length;
      let covariance = 0;
      let positionVariance = 0;
      let valueVariance = 0;
      for (let row = start; row < end; row++) {
        const position = positions[row] - meanPosition;
        const value = series[row] - meanValue;
        covariance += position * value;
        positionVariance += position * position;
        valueVariance += value * value;
      }
      return positionVariance > 0
        ? valueVariance - (covariance * covariance) / positionVariance
        : valueVariance;
    }
  }
}

/** Unpruned optimal partitioning over the finite values of `values` (NaN is skipped). */
export function computeChangePointReference(
  cost: GPULocationChangePointCost,
  values: ArrayLike<number>,
  penalty: number,
  minimumSegmentLength: number
): ChangePointReference {
  const series: number[] = [];
  const positions: number[] = [];
  for (let slice = 0; slice < values.length; slice++) {
    if (Number.isFinite(values[slice])) {
      series.push(values[slice]);
      positions.push(slice);
    }
  }
  const length = series.length;
  const overallMean = series.reduce((sum, value) => sum + value, 0) / (length || 1);
  const best = new Float64Array(length + 1).fill(Infinity);
  const previous = new Int32Array(length + 1).fill(-1);
  best[0] = -penalty;
  for (let end = minimumSegmentLength; end <= length; end++) {
    for (let start = 0; start <= end - minimumSegmentLength; start++) {
      if ((start > 0 && start < minimumSegmentLength) || !Number.isFinite(best[start])) {
        continue;
      }
      const candidate =
        best[start] + getSegmentCost(cost, series, positions, start, end, overallMean) + penalty;
      if (candidate < best[end]) {
        best[end] = candidate;
        previous[end] = start;
      }
    }
  }
  const changePoints: number[] = [];
  for (let cursor = length; previous[cursor] > 0; cursor = previous[cursor]) {
    changePoints.unshift(positions[previous[cursor]]);
  }
  return {changePoints, optimalCost: best[length]};
}

/** Pettitt's K, the first slice after the change, and the asymptotic p-value, over finite values. */
export function computePettittReference(values: ArrayLike<number>): {
  statistic: number;
  changeSlice: number;
  pValue: number;
} {
  const series: number[] = [];
  const positions: number[] = [];
  for (let slice = 0; slice < values.length; slice++) {
    if (Number.isFinite(values[slice])) {
      series.push(values[slice]);
      positions.push(slice);
    }
  }
  const length = series.length;
  let running = 0;
  let statistic = 0;
  let largestRow = -1;
  for (let row = 0; row + 1 < length; row++) {
    for (let other = 0; other < length; other++) {
      running += Math.sign(series[other] - series[row]);
    }
    if (Math.abs(running) > statistic) {
      statistic = Math.abs(running);
      largestRow = row;
    }
  }
  return {
    statistic,
    changeSlice: largestRow >= 0 ? positions[largestRow + 1] : 0xffffffff,
    pValue:
      length >= 2
        ? Math.min(1, 2 * Math.exp((-6 * statistic * statistic) / (length ** 3 + length ** 2)))
        : 1
  };
}

/** A deterministic scene: mean steps, variance change, line break, counts and a gappy series. */
export function createChangePointScene(sliceCount: number): {
  name: string;
  cost: GPULocationChangePointCost;
  penalty: number;
  minimumSegmentLength: number;
  values: Float32Array;
}[] {
  const noise = (index: number) => {
    const hashed = Math.sin((index + 1) * 12.9898) * 43758.5453;
    return 2 * (hashed - Math.floor(hashed)) - 1;
  };
  const mean = Float32Array.from({length: sliceCount}, (_, slice) => {
    const level = slice < sliceCount / 3 ? 0 : slice < (2 * sliceCount) / 3 ? 8 : -5;
    return Math.fround(level + noise(slice));
  });
  const variance = Float32Array.from({length: sliceCount}, (_, slice) =>
    Math.fround((slice < sliceCount / 2 ? 0.2 : 4) * noise(slice + 100))
  );
  const meanVariance = Float32Array.from({length: sliceCount}, (_, slice) =>
    Math.fround(
      (slice < sliceCount / 2 ? 0.3 : 3) * noise(slice + 200) + (slice < sliceCount / 2 ? 1 : 5)
    )
  );
  const slope = Float32Array.from({length: sliceCount}, (_, slice) => {
    const bend = Math.floor(sliceCount / 2);
    const trend = slice < bend ? 0.8 * slice : 0.8 * bend - 1.2 * (slice - bend);
    return Math.fround(trend + 0.3 * noise(slice + 300));
  });
  const counts = Float32Array.from({length: sliceCount}, (_, slice) =>
    Math.max(
      0,
      Math.round(
        (slice < sliceCount / 3 ? 3 : slice < (2 * sliceCount) / 3 ? 20 : 8) +
          2 * noise(slice + 400)
      )
    )
  );
  const gappy = Float32Array.from(mean, (value, slice) => (slice % 7 === 3 ? Number.NaN : value));
  return [
    {name: 'mean', cost: 'mean', penalty: 12, minimumSegmentLength: 2, values: mean},
    {name: 'variance', cost: 'variance', penalty: 8, minimumSegmentLength: 2, values: variance},
    {
      name: 'mean-variance',
      cost: 'mean-variance',
      penalty: 8,
      minimumSegmentLength: 2,
      values: meanVariance
    },
    {name: 'slope', cost: 'slope', penalty: 4, minimumSegmentLength: 3, values: slope},
    {name: 'poisson', cost: 'poisson', penalty: 7, minimumSegmentLength: 1, values: counts},
    {name: 'mean with gaps', cost: 'mean', penalty: 12, minimumSegmentLength: 2, values: gappy}
  ];
}
