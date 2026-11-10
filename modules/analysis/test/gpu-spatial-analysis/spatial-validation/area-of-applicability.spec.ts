// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUAreaOfApplicability,
  GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR,
  GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS
} from '../../../src/gpu-spatial-analysis/spatial-validation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createRandom(seed: number) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return {uniform, normal};
}

type Case = {
  rows: number;
  queryRows: number;
  columns: number;
  training: number[];
  query: number[];
  weights?: number[];
  mask?: number[];
  folds?: {testFolds: number[]; trainingMasks: number[]; foldCount: number};
};

async function runGPU(testCase: Case) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let next = 0;
  const input = (values: number[], format: GPUVectorFormat) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(values) : new Float32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length);
  };
  const output = (length: number, format: GPUVectorFormat) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const trainDissimilarity = output(testCase.rows, 'float32');
  const queryDissimilarity = output(testCase.queryRows, 'float32');
  const applicable = output(testCase.queryRows, 'uint32');
  const summary = output(6, 'float32');
  const status = output(1, 'uint32');
  graph.add(
    new GPUAreaOfApplicability({
      training: input(testCase.training, 'float32') as never,
      query: input(testCase.query, 'float32') as never,
      predictorCount: testCase.columns,
      weights: testCase.weights && (input(testCase.weights, 'float32') as never),
      mask: testCase.mask && (input(testCase.mask, 'uint32') as never),
      crossValidation: testCase.folds && {
        folds: input(testCase.folds.testFolds, 'uint32') as never,
        training: input(testCase.folds.trainingMasks, 'uint32') as never,
        foldCount: testCase.folds.foldCount
      },
      output: {
        trainDissimilarity: trainDissimilarity.view as never,
        queryDissimilarity: queryDissimilarity.view as never,
        applicable: applicable.view as never,
        summary: summary.view as never,
        status: status.view as never
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      trainDissimilarity: await trainDissimilarity.read(),
      queryDissimilarity: await queryDissimilarity.read(),
      applicable: await applicable.read(),
      summary: await summary.read(),
      status: (await status.read())[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** Binary64 reference written from the CAST definition, sharing no code with the kernels. */
function referenceAOA(testCase: Case) {
  const {rows, columns} = testCase;
  const at = (matrix: number[], row: number, column: number) => matrix[row * columns + column];
  const usable = Array.from(
    {length: rows},
    (_, row) =>
      (testCase.mask ? testCase.mask[row] !== 0 : true) &&
      Array.from({length: columns}, (_, column) => at(testCase.training, row, column)).every(
        Number.isFinite
      )
  );
  const usableRows = usable.flatMap((flag, row) => (flag ? [row] : []));
  const center: number[] = [];
  const spread: number[] = [];
  for (let column = 0; column < columns; column++) {
    const values = usableRows.map(row => at(testCase.training, row, column));
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    center.push(mean);
    spread.push(
      Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1))
    );
  }
  const weight = (column: number) => (testCase.weights ? testCase.weights[column] : 1);
  const scale = (value: number, column: number) =>
    ((value - center[column]) / spread[column]) * weight(column);
  const scaledTraining = (row: number) =>
    Array.from({length: columns}, (_, column) => scale(at(testCase.training, row, column), column));
  const distance = (left: number[], right: number[]) =>
    Math.sqrt(left.reduce((sum, value, column) => sum + (value - right[column]) ** 2, 0));
  const scaledRows = new Map(usableRows.map(row => [row, scaledTraining(row)]));
  const averages: number[] = [];
  const nearest: number[] = Array.from({length: rows}, () => NaN);
  for (const row of usableRows) {
    const others = usableRows.filter(other => other !== row);
    averages.push(
      others.reduce(
        (sum, other) => sum + distance(scaledRows.get(row)!, scaledRows.get(other)!),
        0
      ) / others.length
    );
    let candidates = others;
    if (testCase.folds) {
      const fold = testCase.folds.testFolds[row];
      candidates =
        fold < testCase.folds.foldCount
          ? others.filter(other => testCase.folds!.trainingMasks[fold * rows + other] !== 0)
          : [];
    }
    if (candidates.length > 0) {
      nearest[row] = Math.min(
        ...candidates.map(other => distance(scaledRows.get(row)!, scaledRows.get(other)!))
      );
    }
  }
  const meanDistance = averages.reduce((sum, value) => sum + value, 0) / averages.length;
  const trainDissimilarity = nearest.map(value => value / meanDistance);
  const scored = trainDissimilarity.filter(Number.isFinite).sort((left, right) => left - right);
  const quantile = (probability: number) => {
    const position = (scored.length - 1) * probability;
    const lower = Math.floor(position);
    return (
      scored[lower] +
      (position - lower) * ((scored[Math.min(lower + 1, scored.length - 1)] ?? 0) - scored[lower])
    );
  };
  const quartiles = [quantile(0.25), quantile(0.75)];
  const threshold = Math.min(
    quartiles[1] + 1.5 * (quartiles[1] - quartiles[0]),
    scored[scored.length - 1]
  );
  const queryDissimilarity = Array.from({length: testCase.queryRows}, (_, query) => {
    const raw = Array.from({length: columns}, (_, column) => at(testCase.query, query, column));
    if (!raw.every(Number.isFinite)) return NaN;
    const point = raw.map((value, column) => scale(value, column));
    return Math.min(...usableRows.map(row => distance(point, scaledRows.get(row)!))) / meanDistance;
  });
  return {meanDistance, trainDissimilarity, threshold, quartiles, queryDissimilarity, scored};
}

function buildCase(seed: number, withFolds: boolean, options: Partial<Case> = {}): Case {
  const random = createRandom(seed);
  const rows = 70;
  const queryRows = 50;
  const columns = 3;
  const training = Array.from({length: rows * columns}, (_, index) =>
    Math.fround(random.normal() * (1 + (index % columns)) + (index % columns) * 2)
  );
  // Queries are spread over twice the training range, so a good share lies outside the area.
  const query = Array.from({length: queryRows * columns}, (_, index) =>
    Math.fround(random.normal() * 2.5 * (1 + (index % columns)) + (index % columns) * 2)
  );
  query[4] = NaN;
  const foldCount = 4;
  const testFolds = Array.from({length: rows}, () =>
    Math.floor(random.uniform() * (foldCount + 1))
  );
  // A fold index of foldCount marks a row held out in no fold. Buffered training masks drop some rows.
  const trainingMasks = Array.from({length: foldCount * rows}, (_, index) => {
    const fold = Math.floor(index / rows);
    const row = index % rows;
    return testFolds[row] !== fold && random.uniform() > 0.15 ? 1 : 0;
  });
  return {
    rows,
    queryRows,
    columns,
    training,
    query,
    folds: withFolds ? {testFolds, trainingMasks, foldCount} : undefined,
    ...options
  };
}

function expectMatches(testCase: Case, actual: Awaited<ReturnType<typeof runGPU>>) {
  const expected = referenceAOA(testCase);
  const close = (left: number, right: number, name: string) => {
    if (Number.isNaN(right)) expect(left, name).toBeNaN();
    else
      expect(Math.abs(left - right), `${name} ${left} vs ${right}`).toBeLessThan(
        2e-4 * Math.max(1, Math.abs(right))
      );
  };
  expect(actual.status).toBe(0);
  close(actual.summary[0], expected.meanDistance, 'meanDistance');
  close(actual.summary[1], expected.threshold, 'threshold');
  close(actual.summary[4], expected.quartiles[0], 'first quartile');
  close(actual.summary[5], expected.quartiles[1], 'third quartile');
  close(actual.summary[3], expected.scored.length, 'scored rows');
  expected.trainDissimilarity.forEach((value, row) =>
    close(actual.trainDissimilarity[row], value, `train ${row}`)
  );
  let outside = 0;
  let checked = 0;
  expected.queryDissimilarity.forEach((value, query) => {
    close(actual.queryDissimilarity[query], value, `query ${query}`);
    if (!Number.isFinite(value)) {
      expect(actual.applicable[query]).toBe(0);
      return;
    }
    // A float32 index within the tolerance of the threshold may fall on either side.
    if (Math.abs(value - expected.threshold) < 2e-4 * expected.threshold) return;
    checked++;
    expect(actual.applicable[query], `applicable ${query}`).toBe(
      value <= expected.threshold ? 1 : 0
    );
    if (value > expected.threshold) outside++;
  });
  // The data are built so that both outcomes occur.
  expect(outside).toBeGreaterThan(3);
  expect(checked - outside).toBeGreaterThan(3);
}

it('matches the CAST definition with cross-validation folds, buffered masks and unscored rows', async () => {
  const testCase = buildCase(1, true);
  expectMatches(testCase, await runGPU(testCase));
});

it('matches the CAST definition without folds, with weights and a row mask', async () => {
  const random = createRandom(5);
  const base = buildCase(2, false);
  const testCase: Case = {
    ...base,
    weights: [1, 0.5, 2],
    mask: Array.from({length: base.rows}, () => (random.uniform() < 0.85 ? 1 : 0))
  };
  testCase.training[7] = NaN;
  expectMatches(testCase, await runGPU(testCase));
});

it('restricting neighbours to the training rows of each fold never lowers the index and sometimes raises it', async () => {
  const withFolds = buildCase(3, true);
  const folded = await runGPU(withFolds);
  const free = await runGPU({...withFolds, folds: undefined});
  // The mean distance does not depend on the folds, so the indices compare directly.
  expect(Math.abs(folded.summary[0] - free.summary[0])).toBeLessThan(1e-5);
  let raised = 0;
  referenceAOA(withFolds).trainDissimilarity.forEach((value, row) => {
    if (!Number.isFinite(value)) return;
    expect(folded.trainDissimilarity[row]).toBeGreaterThanOrEqual(
      free.trainDissimilarity[row] * (1 - 1e-5)
    );
    if (folded.trainDissimilarity[row] > free.trainDissimilarity[row] * 1.01) raised++;
  });
  expect(raised).toBeGreaterThan(5);
});

it('reports too few rows and constant predictors', async () => {
  const few = await runGPU({
    rows: 4,
    queryRows: 1,
    columns: 1,
    training: [1, 2, 3, 4],
    query: [2],
    mask: [1, 0, 0, 0]
  });
  expect(few.status & GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS).toBe(
    GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS
  );
  expect(few.queryDissimilarity[0]).toBeNaN();
  expect(few.applicable[0]).toBe(0);
  const constant = await runGPU({
    rows: 3,
    queryRows: 1,
    columns: 2,
    training: [1, 5, 2, 5, 3, 5],
    query: [2, 5]
  });
  expect(constant.status & GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR).toBe(
    GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR
  );
  expect(constant.applicable[0]).toBe(0);
});

it('worked example: four collinear points', async () => {
  // Points 0, 1, 2, 10 in one column. The index is invariant to the standardization, so unscaled distances serve.
  const result = await runGPU({
    rows: 4,
    queryRows: 2,
    columns: 1,
    training: [0, 1, 2, 10],
    query: [1.5, 20]
  });
  // Unscaled: nearest 1, 1, 1, 8; mean distances (1+2+10)/3, (1+1+9)/3, (2+1+8)/3, (10+9+8)/3.
  const averages = [13 / 3, 11 / 3, 11 / 3, 9];
  const meanDistance = averages.reduce((sum, value) => sum + value, 0) / 4;
  const indices = [1, 1, 1, 8].map(value => value / meanDistance);
  // Type 7 quartiles of the sorted indices [a, a, a, 8a]: Q1 = a, Q3 = a + 0.25 (8a - a) = 2.75a; threshold = min(2.75a + 1.5 x 1.75a, 8a).
  const unit = 1 / meanDistance;
  const threshold = Math.min(2.75 * unit + 1.5 * 1.75 * unit, 8 * unit);
  expect(threshold).toBeCloseTo(5.375 * unit, 12);
  indices.forEach((value, row) => expect(result.trainDissimilarity[row]).toBeCloseTo(value, 5));
  expect(result.summary[1]).toBeCloseTo(threshold, 5);
  // Query 1.5 is 0.5 from the nearest point; query 20 is 10 away: indices 0.5 unit and 10 unit.
  expect(result.queryDissimilarity[0]).toBeCloseTo(0.5 * unit, 5);
  expect(result.queryDissimilarity[1]).toBeCloseTo(10 * unit, 5);
  expect(result.applicable).toEqual([1, 0]);
});

it('worked example: the threshold is capped at the largest training index', async () => {
  // Points 0, 1, 10, 12 have nearest distances 1, 1, 2, 2 and mean distances 23/3, 7, 7, 25/3,
  // so meanDistance is 7.5 unscaled (the index does not depend on the standardization). Type 7 quartiles of [1, 1, 2, 2] / 7.5 are 1 and 2 (over 7.5), so
  // Q3 + 1.5 IQR = 3.5 / 7.5 exceeds the largest index 2 / 7.5, which becomes the threshold.
  const result = await runGPU({
    rows: 4,
    queryRows: 3,
    columns: 1,
    training: [0, 1, 10, 12],
    query: [15, 11, 13]
  });
  expect(result.summary[1]).toBeCloseTo(2 / 7.5, 6);
  expect(result.queryDissimilarity[0]).toBeCloseTo(3 / 7.5, 6);
  // Distance 3 gives 0.4 > 0.2667 (outside); without the cap the threshold would be 0.4667 (inside).
  expect(result.applicable).toEqual([0, 1, 1]);
});
