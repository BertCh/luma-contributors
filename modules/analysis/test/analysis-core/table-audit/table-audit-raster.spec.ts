// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterBFASTMonitor,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterTemporal,
  GPU_RASTER_BFAST_MONITOR_NO_BREAK,
  MAXIMUM_RASTER_EMBEDDING_NEIGHBOR_COUNT,
  getGPURasterBFASTMonitorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {GPUPointCloudNeighborhoods} from '@luma-contributors/analysis/pointcloud-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../../gpu-raster/raster-time-series/raster-time-series-test-utils';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';

/** Controls (16, 32), the window (33 to about 250) and the largest sizes. */
const TABLE_SIZES = [16, 32, 33, 36, 40, 48, 64, 96, 128, 200, 256];
const NO_SLICE = 0xffffffff;

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

/** Deterministic pseudo-random numbers in [0, 1), so that the cases are reproducible. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------------
// Site 1: k nearest neighbour classifier. neighborCount is limited to 16 and classCount to 32, so
// both tables are below the 33 entry window; this runs the controls at the real limits.

function getKNearestExpected(
  metric: 'euclidean' | 'cosine',
  pixel: number[],
  training: number[][],
  labels: number[],
  neighborCount: number,
  classCount: number
) {
  const distances = training.map((sample, index) => {
    let distance: number;
    if (metric === 'euclidean') {
      distance = Math.sqrt(
        sample.reduce((sum, value, axis) => sum + (value - pixel[axis]) ** 2, 0)
      );
    } else {
      const dot = sample.reduce((sum, value, axis) => sum + value * pixel[axis], 0);
      const normA = Math.sqrt(sample.reduce((sum, value) => sum + value * value, 0));
      const normB = Math.sqrt(pixel.reduce((sum, value) => sum + value * value, 0));
      distance = 1 - dot / (normA * normB);
    }
    return {index, distance};
  });
  distances.sort((first, second) => first.distance - second.distance || first.index - second.index);
  const gap = distances[neighborCount].distance - distances[neighborCount - 1].distance;
  const votes = new Array<number>(classCount).fill(0);
  for (const entry of distances.slice(0, neighborCount)) votes[labels[entry.index]]++;
  const winner = votes.indexOf(Math.max(...votes));
  return {votes, winner, gap};
}

it('table audit site 1: k nearest neighbour classifier at its allowed sizes', async () => {
  const device = await getDevice();
  const wrongSizes: string[] = [];
  const dimensions = 4;
  const pixelCount = 24;
  const trainingCount = 60;
  for (const metric of ['euclidean', 'cosine'] as const) {
    for (const classCount of [3, 32]) {
      for (const neighborCount of [1, 8, 16, 17]) {
        if (neighborCount > MAXIMUM_RASTER_EMBEDDING_NEIGHBOR_COUNT) {
          // Size 17 is not allowed; assert the contributor refuses it rather than running it.
          const harness = createRasterTimeSeriesHarness(device);
          let refused = false;
          try {
            new GPURasterEmbeddingKNearestClassifier({
              pixelCount,
              dimensions,
              vectors: harness.floats(new Float32Array(pixelCount * dimensions)),
              trainingVectors: harness.floats(new Float32Array(trainingCount * dimensions)),
              trainingLabels: harness.integers(new Uint32Array(trainingCount)),
              trainingCount,
              classCount,
              neighborCount,
              output: {labels: harness.output('labels', 'uint32', pixelCount)}
            });
          } catch {
            refused = true;
          }
          if (!refused) wrongSizes.push(`neighborCount ${neighborCount} was not refused`);
          continue;
        }
        const random = createRandom(7 + neighborCount * 31 + classCount);
        const pixels = Array.from({length: pixelCount}, () =>
          Array.from({length: dimensions}, () => 0.1 + random())
        );
        // Sample 0 is far from every pixel (opposite orthant) and of a class that no near sample
        // gets, so a table of zeros (every neighbour is sample 0) would name that class.
        const farClass = classCount - 1;
        const training = [
          Array.from({length: dimensions}, () => -5),
          ...Array.from({length: trainingCount - 1}, () =>
            Array.from({length: dimensions}, () => 0.1 + random())
          )
        ];
        const labels = [
          farClass,
          ...Array.from({length: trainingCount - 1}, () => Math.floor(random() * (classCount - 1)))
        ];
        const expected = pixels.map(pixel =>
          getKNearestExpected(metric, pixel, training, labels, neighborCount, classCount)
        );
        const harness = createRasterTimeSeriesHarness(device);
        const result = await harness.run(
          new GPURasterEmbeddingKNearestClassifier({
            pixelCount,
            dimensions,
            metric,
            vectors: harness.floats(pixels.flat()),
            trainingVectors: harness.floats(training.flat()),
            trainingLabels: harness.integers(labels),
            trainingCount,
            classCount,
            neighborCount,
            output: {
              labels: harness.output('labels', 'uint32', pixelCount),
              confidence: harness.output('confidence', 'float32', pixelCount),
              probabilities: harness.output('probabilities', 'float32', pixelCount * classCount)
            }
          })
        );
        let wrongLabels = 0;
        let wrongProbabilities = 0;
        let ambiguous = 0;
        expected.forEach((entry, pixel) => {
          // Skip a pixel whose k-th and (k+1)-th distances are too close for float32 to order.
          if (entry.gap < 1e-4) {
            ambiguous++;
            return;
          }
          if (result['labels'][pixel] !== entry.winner) wrongLabels++;
          entry.votes.forEach((count, label) => {
            const actual = result['probabilities'][pixel * classCount + label];
            if (Math.abs(actual - count / neighborCount) > 1e-6) wrongProbabilities++;
          });
          const confidence = Math.max(...entry.votes) / neighborCount;
          if (Math.abs(result['confidence'][pixel] - confidence) > 1e-6) wrongProbabilities++;
        });
        if (ambiguous === pixelCount)
          wrongSizes.push(`${metric} k ${neighborCount}: no usable pixel`);
        if (wrongLabels > 0 || wrongProbabilities > 0) {
          wrongSizes.push(
            `${metric} classes ${classCount} k ${neighborCount}: ${wrongLabels} wrong labels, ${wrongProbabilities} wrong shares`
          );
        }
      }
    }
  }
  expect(wrongSizes).toEqual([]);
}, 180000);

// ---------------------------------------------------------------------------------------------
// Site 2: temporal composites. dates: array<u32, sliceCount>.

function getPermutation(sliceCount: number, cellIndex: number): number[] {
  const multipliers = [7, 11, 13, 17, 19, 23, 29];
  const multiplier = multipliers.filter(candidate => sliceCount % candidate !== 0)[cellIndex % 3];
  // The minimum of cell 0 is at slice 37 (mod sliceCount); other cells put it elsewhere.
  const target = [37, 5, sliceCount - 1][cellIndex % 3] % sliceCount;
  return Array.from({length: sliceCount}, (_, slice) => {
    return (((slice - target + sliceCount) % sliceCount) * multiplier) % sliceCount;
  });
}

it('table audit site 2: temporal composites choose the right slice', async () => {
  const device = await getDevice();
  const wrongSizes: string[] = [];
  for (const sliceCount of TABLE_SIZES) {
    const cellCount = 3;
    const values: number[] = [];
    const validity: number[] = [];
    for (let cell = 0; cell < cellCount; cell++) {
      const permutation = getPermutation(sliceCount, cell);
      for (let slice = 0; slice < sliceCount; slice++) {
        let value = permutation[slice] * 1.5 - 20;
        let valid = 1;
        if (cell === 1 && [0, 5, sliceCount - 1].includes(slice)) value = Number.NaN;
        if (cell === 2 && [1, 2, 3, 9].includes(slice)) valid = 0;
        values.push(value);
        validity.push(valid);
      }
    }
    const reference = (cell: number, operation: 'minimum' | 'maximum' | 'median') => {
      const observations: {value: number; slice: number}[] = [];
      for (let slice = 0; slice < sliceCount; slice++) {
        const source = cell * sliceCount + slice;
        if (Number.isFinite(values[source]) && validity[source] !== 0) {
          observations.push({value: values[source], slice});
        }
      }
      observations.sort(
        (first, second) => first.value - second.value || first.slice - second.slice
      );
      const count = observations.length;
      if (operation === 'minimum')
        return {value: observations[0].value, slice: observations[0].slice, count};
      if (operation === 'maximum') {
        const last = observations[count - 1];
        return {value: last.value, slice: last.slice, count};
      }
      if (count % 2 === 1) {
        const middle = observations[(count - 1) / 2];
        return {value: middle.value, slice: middle.slice, count};
      }
      return {
        value: (observations[count / 2 - 1].value + observations[count / 2].value) / 2,
        slice: NO_SLICE,
        count
      };
    };
    for (const operation of ['minimum', 'maximum', 'median'] as const) {
      const harness = createRasterTimeSeriesHarness(device);
      const result = await harness.run(
        new GPURasterTemporal({
          cellCount,
          sliceCount,
          values: harness.floats(values),
          mask: harness.integers(validity),
          operation,
          output: {
            values: harness.output('values', 'float32', cellCount),
            counts: harness.output('counts', 'uint32', cellCount),
            chosenSlice: harness.output('chosenSlice', 'uint32', cellCount),
            status: harness.output('status', 'uint32', cellCount)
          }
        })
      );
      for (let cell = 0; cell < cellCount; cell++) {
        const expected = reference(cell, operation);
        const actual = {
          value: result['values'][cell],
          slice: result['chosenSlice'][cell],
          count: result['counts'][cell]
        };
        if (
          actual.value !== expected.value ||
          actual.slice !== expected.slice ||
          actual.count !== expected.count
        ) {
          wrongSizes.push(
            `sliceCount ${sliceCount} ${operation} cell ${cell}: got ${JSON.stringify(actual)} expected ${JSON.stringify(expected)}`
          );
        }
      }
    }
  }
  expect(wrongSizes).toEqual([]);
}, 180000);

// ---------------------------------------------------------------------------------------------
// Site 3: point cloud neighbourhoods. nearestIds: array<u32, capacity>.

it('table audit site 3: point cloud neighbourhoods keep the nearest capacity points', async () => {
  const device = await getDevice();
  const wrongSizes: string[] = [];
  // A 7 x 7 x 6 lattice (many equal distances, so ties exercise the lower-id rule), an isolated
  // point and a small cluster far away. Squared distances are integers, exact in float32.
  const points: number[][] = [];
  for (let index = 0; index < 294; index++) {
    points.push([index % 7, Math.floor(index / 7) % 7, Math.floor(index / 49)]);
  }
  points.push([100, 0, 0]);
  for (let index = 0; index < 10; index++)
    points.push([50 + (index % 2), 50 + Math.floor(index / 2), 50]);
  const radius = 3;
  const brute = points.map((position, query) => {
    const support: {index: number; distance: number}[] = [];
    points.forEach((candidate, index) => {
      if (index === query) return;
      const distance = candidate.reduce(
        (sum, value, axis) => sum + (value - position[axis]) ** 2,
        0
      );
      if (distance <= radius * radius) support.push({index, distance});
    });
    support.sort((first, second) => first.distance - second.distance || first.index - second.index);
    return support;
  });
  for (const capacity of TABLE_SIZES.filter(size => size <= 64)) {
    for (const strategy of ['serial', 'cooperative'] as const) {
      const harness = createOptimizationHarness(device);
      const positions = harness.input('float32x3', new Float32Array(points.flat()));
      const indices = harness.output('uint32', points.length * capacity);
      const squaredDistances = harness.output('float32', points.length * capacity);
      const counts = harness.output('uint32', points.length);
      const status = harness.output('uint32', points.length);
      const read = await harness.run(
        new GPUPointCloudNeighborhoods({
          id: `table-audit-${strategy}-${capacity}`,
          positions,
          radius,
          capacity,
          strategy,
          output: {indices, squaredDistances, counts, status}
        })
      );
      const actualIndices = read(indices);
      const actualDistances = read(squaredDistances);
      const actualCounts = read(counts);
      let wrongQueries = 0;
      let firstWrong = '';
      brute.forEach((support, query) => {
        let wrong = actualCounts[query] !== support.length;
        for (let slot = 0; slot < capacity && !wrong; slot++) {
          const entry = support[slot];
          const index = actualIndices[query * capacity + slot];
          const distance = actualDistances[query * capacity + slot];
          if (entry ? index !== entry.index || distance !== entry.distance : index !== NO_SLICE) {
            wrong = true;
          }
        }
        if (wrong) {
          wrongQueries++;
          if (!firstWrong) firstWrong = `query ${query}`;
        }
      });
      if (wrongQueries > 0) {
        wrongSizes.push(
          `capacity ${capacity} ${strategy}: ${wrongQueries} wrong queries, first ${firstWrong}`
        );
      }
    }
  }
  expect(wrongSizes).toEqual([]);
}, 180000);

// ---------------------------------------------------------------------------------------------
// Site 4: BFAST Monitor. slicesUsed: array<u32, slices>.

it('table audit site 4: BFAST Monitor reports the original slice of the break', async () => {
  const device = await getDevice();
  const wrongSizes: string[] = [];
  const context: GPUTemporalContext = {
    unit: 'second',
    epoch: 0n,
    encoding: 'int64-words',
    axis: {kind: 'irregular'}
  };
  for (const sliceCount of TABLE_SIZES) {
    const historySliceCount = sliceCount <= 16 ? 10 : sliceCount <= 32 ? 14 : 24;
    const breakSlice = historySliceCount + 3;
    // Missing observations before and after the history boundary, so the break slice differs from
    // the position among the valid observations.
    const missing = [2, 5, breakSlice - 2];
    const values = Array.from({length: sliceCount}, (_, slice) => {
      if (missing.includes(slice)) return Number.NaN;
      return 10 + (slice % 2 === 0 ? 0.1 : -0.1) + (slice >= breakSlice ? 100 : 0);
    });
    const harness = createRasterTimeSeriesHarness(device);
    const times = BigInt64Array.from({length: sliceCount}, (_, slice) =>
      BigInt(1_000_000_000 + slice * 2_592_000)
    );
    const result = await harness.run(
      new GPURasterBFASTMonitor({
        cellCount: 1,
        sliceCount,
        values: harness.floats(values),
        parameters: harness.floats(getGPURasterBFASTMonitorParameterValues({criticalValue: 5})),
        dateAxis: {acquisitionTimes: harness.words(times), temporalContext: context},
        historySliceCount,
        harmonicCount: 1,
        includeTrend: false,
        output: {
          coefficients: harness.output('coefficients', 'float32', 3),
          scores: harness.output('scores', 'float32', 2),
          cellSummary: harness.output('summary', 'uint32', 3)
        }
      })
    );
    const expectedHistoryCount = historySliceCount - 2;
    const [actualBreak, status, historyCount] = result['summary'];
    if (actualBreak !== breakSlice || status !== 0 || historyCount !== expectedHistoryCount) {
      wrongSizes.push(
        `sliceCount ${sliceCount}: break ${actualBreak === GPU_RASTER_BFAST_MONITOR_NO_BREAK ? 'none' : actualBreak} (expected ${breakSlice}), status ${status}, history count ${historyCount} (expected ${expectedHistoryCount})`
      );
    }
  }
  expect(wrongSizes).toEqual([]);
}, 180000);
