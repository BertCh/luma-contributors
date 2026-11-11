// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getTheilSenSlopeWGSL} from '../../src/analysis-core/kernels/wgsl-theil-sen-median';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getTheilSenSlope against the definition (math review finding core-numerics-11: no spec named
// this kernel): the median of all pairwise slopes, found here by sorting every slope in float64.
// The kernel selects it in a bounded heap without sorting, so the two share no code path.

const CAPACITY = 16;

function getReferenceSlope(series: number[]): number {
  const slopes: number[] = [];
  for (let first = 0; first < series.length; first++) {
    for (let second = first + 1; second < series.length; second++) {
      slopes.push((series[second] - series[first]) / (second - first));
    }
  }
  if (slopes.length === 0) {
    return 0;
  }
  slopes.sort((left, right) => left - right);
  const middle = Math.floor(slopes.length / 2);
  return slopes.length % 2 === 1 ? slopes[middle] : 0.5 * (slopes[middle - 1] + slopes[middle]);
}

it('the Theil-Sen slope is the median of all pairwise slopes for odd and even pair counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(77);
  const cases: number[][] = [
    // Hand cases. Empty and single series: 0. Two points: the one slope.
    [],
    [4],
    [1, 4],
    // Three points 0, 1, 5: slopes 1, 2.5, 4, median 2.5.
    [0, 1, 5],
    // Four points 0, 2, 3, 9: slopes 2, 1.5, 3, 1, 3.5, 6; sorted 1, 1.5, 2, 3, 3.5, 6; median 2.5.
    [0, 2, 3, 9],
    // A straight line with one outlier keeps the line's slope: 3.
    [0, 3, 6, 9, 100, 15, 18, 21],
    // Constant series and a descending one.
    [7, 7, 7, 7, 7],
    [10, 8, 6, 4, 2, 0]
  ];
  for (let index = 0; index < 120; index++) {
    const count = 2 + (index % (CAPACITY - 1));
    const kind = index % 4;
    cases.push(
      Array.from({length: count}, (_, step) =>
        kind === 0
          ? Math.round(40 * random() - 20)
          : kind === 1
            ? 0.5 * step + Math.round(6 * random() - 3)
            : kind === 2
              ? Math.round(3 * random())
              : 1000 * (random() - 0.5)
      )
    );
  }
  const series = new Float32Array(cases.length * CAPACITY);
  cases.forEach((values, index) => series.set(values, index * CAPACITY));
  const words = await runWGSLModuleKernel(device, {
    id: 'theil-sen-median',
    inputs: {series, counts: Uint32Array.from(cases, values => values.length)},
    outputLength: cases.length,
    invocationCount: cases.length,
    declarations: getTheilSenSlopeWGSL(CAPACITY),
    body: `var values: array<f32, ${CAPACITY}>;
  for (var step = 0u; step < ${CAPACITY}u; step++) {
    values[step] = series[seriesOffset + index * ${CAPACITY}u + step];
  }
  output[outputOffset + index] = bitcast<u32>(getTheilSenSlope(&values, counts[countsOffset + index]));`
  });
  const slopes = getFloatWords(words);
  expect(Array.from(slopes.slice(0, 8))).toEqual([0, 0, 3, 2.5, 2.5, 3, 0, -2]);
  const mismatches: string[] = [];
  cases.forEach((values, index) => {
    // The reference uses the float32 series the kernel reads; a slope is one float32 division.
    const stored = Array.from(series.slice(index * CAPACITY, index * CAPACITY + values.length));
    const reference = getReferenceSlope(stored);
    const scale = Math.max(1, ...stored.map(Math.abs));
    if (!(Math.abs(slopes[index] - reference) <= 4e-7 * scale)) {
      mismatches.push(
        `case ${index} (${values.length} points): ${slopes[index]} against ${reference}`
      );
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
});
