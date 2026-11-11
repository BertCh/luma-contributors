// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review, interpolation-density-03. Ordinary co-kriging has one unbiasedness
// constraint per variable. In a neighbourhood with no sample of a secondary variable that
// constraint is `0 = 0`: it used to stay in the system as a zero row and fail the solve
// (status 2). Without it the system is the ordinary kriging system of the target variable, which
// is what gstat predicts there. The expected values are a float64 solve of that ordinary kriging
// system, written here from the textbook equations (Cressie 1993, section 3.2).

const SILL = 2;
const RANGE = 10;
// Loadings L with B = L L^T = [[1, 0.6], [0.6, 1]]: the target variable has unit loading.
const LOADINGS = [1, 0, 0.6, 0.8] as const;

const TARGET_SAMPLES = [
  {position: [0, 0], value: 3.1},
  {position: [1, 0.2], value: 2.7},
  {position: [-0.7, 1.1], value: 3.6},
  {position: [0.4, -1.3], value: 2.2},
  {position: [1.8, 1.5], value: 3.9},
  {position: [-1.6, -0.9], value: 2.9}
];
const SECONDARY_SAMPLES = [
  {position: [100, 100], value: 11},
  {position: [101, 100.5], value: 12},
  {position: [100.2, 101.3], value: 13}
];

function getCovariance(first: readonly number[], second: readonly number[]): number {
  return SILL * Math.exp((-3 * Math.hypot(first[0] - second[0], first[1] - second[1])) / RANGE);
}

/** Solves `matrix * x = right` in float64 by Gaussian elimination with partial pivoting. */
function solveLinearSystem(matrix: number[][], right: number[]): number[] {
  const size = right.length;
  const rows = matrix.map((row, index) => [...row, right[index]]);
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivotRow][column])) {
        pivotRow = row;
      }
    }
    [rows[column], rows[pivotRow]] = [rows[pivotRow], rows[column]];
    for (let row = column + 1; row < size; row++) {
      const factor = rows[row][column] / rows[column][column];
      for (let entry = column; entry <= size; entry++) {
        rows[row][entry] -= factor * rows[column][entry];
      }
    }
  }
  const solution = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row--) {
    let value = rows[row][size];
    for (let column = row + 1; column < size; column++) {
      value -= rows[row][column] * solution[column];
    }
    solution[row] = value / rows[row][row];
  }
  return solution;
}

/** Ordinary kriging prediction and variance at `query` from the target samples alone. */
function getOrdinaryKriging(query: readonly number[]): {prediction: number; variance: number} {
  const count = TARGET_SAMPLES.length;
  const matrix = TARGET_SAMPLES.map(first => [
    ...TARGET_SAMPLES.map(second => getCovariance(first.position, second.position)),
    1
  ]);
  matrix.push([...new Array<number>(count).fill(1), 0]);
  const cross = TARGET_SAMPLES.map(sample => getCovariance(sample.position, query));
  const solution = solveLinearSystem(matrix, [...cross, 1]);
  let prediction = 0;
  let variance = SILL - solution[count];
  for (let row = 0; row < count; row++) {
    prediction += solution[row] * TARGET_SAMPLES[row].value;
    variance -= solution[row] * cross[row];
  }
  return {prediction, variance};
}

async function runCoKriging(queries: number[][], searchRadius: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const samples = [
    ...TARGET_SAMPLES.map(sample => ({...sample, variable: 0})),
    ...SECONDARY_SAMPLES.map(sample => ({...sample, variable: 1}))
  ];
  const positionsBuffer = createInputBuffer(
    device,
    Float32Array.from(samples.flatMap(sample => sample.position))
  );
  const observationsBuffer = createInputBuffer(
    device,
    Float32Array.from(samples.flatMap(sample => [sample.value, 0, sample.variable, 1]))
  );
  const supportsBuffer = createInputBuffer(device, Float32Array.from(queries.flat()));
  const valuesBuffer = createOutputBuffer(device, queries.length * 4);
  const statusBuffer = createOutputBuffer(device, queries.length);
  const graph = new GPUCommandGraph(device, {id: 'co-kriging-missing-variable'});
  graph.add(
    new GPULocalKriging({
      positions: importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        samples.length
      ),
      observations: importGraphBuffer(
        graph,
        'observations',
        observationsBuffer,
        'float32x4',
        samples.length
      ),
      supports: importGraphBuffer(graph, 'supports', supportsBuffer, 'float32x2', queries.length),
      mode: 'co-kriging',
      coKrigingMean: 'ordinary',
      variableCount: 2,
      targetVariable: 0,
      covariance: {model: 'exponential', sill: SILL, range: RANGE, nugget: 0},
      coregionalization: LOADINGS,
      neighborCount: 8,
      searchRadius,
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32x4', queries.length),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', queries.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(valuesBuffer, queries.length * 4),
    status: await readUint32(statusBuffer, queries.length)
  };
  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    observationsBuffer,
    supportsBuffer,
    valuesBuffer,
    statusBuffer
  ]) {
    buffer.destroy();
  }
  return result;
}

it('ordinary co-kriging with no secondary sample in the neighbourhood is ordinary kriging of the target', async () => {
  const queries = [
    [0.3, 0.2],
    [-0.5, 0.4],
    [1.2, -0.6]
  ];
  // Search radius 20: all six target samples, none of the secondary samples near (100, 100).
  const {values, status} = await runCoKriging(queries, 20);
  expect(status).toEqual([0, 0, 0]);
  for (const [index, query] of queries.entries()) {
    const expected = getOrdinaryKriging(query);
    expect(values[index * 4]).toBeCloseTo(expected.prediction, 4);
    expect(values[index * 4 + 1]).toBeCloseTo(expected.variance, 4);
    expect(values[index * 4 + 3]).toBe(TARGET_SAMPLES.length);
  }
});

it('ordinary co-kriging still fails where the target variable has no sample in the neighbourhood', async () => {
  // Only secondary samples within the radius: the target constraint `sum of no weights = 1`
  // cannot hold, so there is no unbiased predictor.
  const {values, status} = await runCoKriging([[100.4, 100.3]], 20);
  expect(status).toEqual([2]);
  expect(Number.isNaN(values[0])).toBe(true);
  expect(Number.isNaN(values[1])).toBe(true);
});

// Review follow-up (fr-models-05). The cases above have two variables and target 0, so the table
// of kept constraints is the identity. The cases below make it a real remapping: three variables
// with the middle one absent and the target last (kept constraints [0, 2], target in compact
// column 1), and a neighbourhood with fewer samples than variables.

type CoKrigingSample = {position: number[]; value: number; variable: number};

// Loadings L (row-major 3 x 3, lower triangular); the coregionalization matrix is B = L L^T.
const THREE_VARIABLE_LOADINGS = [1, 0, 0, 0.5, 0.9, 0, -0.4, 0.3, 0.8];

function getCoregionalization(loadings: readonly number[], variableCount: number): number[][] {
  return Array.from({length: variableCount}, (_, first) =>
    Array.from({length: variableCount}, (__, second) => {
      let value = 0;
      for (let component = 0; component < variableCount; component++) {
        value +=
          loadings[first * variableCount + component] *
          loadings[second * variableCount + component];
      }
      return value;
    })
  );
}

/**
 * Ordinary co-kriging in float64 with one unbiasedness constraint for each variable in
 * `constrainedVariables` (Wackernagel 2003, chapter 24): `K w + F m = k0`, `F' w = f0`, with
 * `K_ij = B[v_i][v_j] C(h_ij)`, `F` the indicators of the constrained variables and `f0` the
 * indicator of the target.
 */
function getOrdinaryCoKriging(
  samples: CoKrigingSample[],
  query: number[],
  coregionalization: number[][],
  targetVariable: number,
  constrainedVariables: number[]
): {prediction: number; variance: number} {
  const count = samples.length;
  const size = count + constrainedVariables.length;
  const matrix = Array.from({length: size}, () => new Array<number>(size).fill(0));
  const right = new Array<number>(size).fill(0);
  for (let row = 0; row < count; row++) {
    for (let column = 0; column < count; column++) {
      matrix[row][column] =
        coregionalization[samples[row].variable][samples[column].variable] *
        getCovariance(samples[row].position, samples[column].position);
    }
    right[row] =
      coregionalization[samples[row].variable][targetVariable] *
      getCovariance(samples[row].position, query);
    for (const [constraint, variable] of constrainedVariables.entries()) {
      const indicator = samples[row].variable === variable ? 1 : 0;
      matrix[row][count + constraint] = indicator;
      matrix[count + constraint][row] = indicator;
    }
  }
  for (const [constraint, variable] of constrainedVariables.entries()) {
    right[count + constraint] = variable === targetVariable ? 1 : 0;
  }
  const solution = solveLinearSystem(matrix, right);
  let prediction = 0;
  let variance = coregionalization[targetVariable][targetVariable] * SILL;
  for (let row = 0; row < count; row++) {
    prediction += solution[row] * samples[row].value;
    variance -= solution[row] * right[row];
  }
  for (let constraint = 0; constraint < constrainedVariables.length; constraint++) {
    variance -= solution[count + constraint] * right[count + constraint];
  }
  return {prediction, variance};
}

async function runGeneralCoKriging(props: {
  samples: CoKrigingSample[];
  queries: number[][];
  variableCount: number;
  targetVariable: number;
  loadings: readonly number[];
  searchRadius: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {samples, queries} = props;
  const positionsBuffer = createInputBuffer(
    device,
    Float32Array.from(samples.flatMap(sample => sample.position))
  );
  const observationsBuffer = createInputBuffer(
    device,
    Float32Array.from(samples.flatMap(sample => [sample.value, 0, sample.variable, 1]))
  );
  const supportsBuffer = createInputBuffer(device, Float32Array.from(queries.flat()));
  const valuesBuffer = createOutputBuffer(device, queries.length * 4);
  const statusBuffer = createOutputBuffer(device, queries.length);
  const graph = new GPUCommandGraph(device, {id: 'co-kriging-constraint-remapping'});
  graph.add(
    new GPULocalKriging({
      positions: importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        samples.length
      ),
      observations: importGraphBuffer(
        graph,
        'observations',
        observationsBuffer,
        'float32x4',
        samples.length
      ),
      supports: importGraphBuffer(graph, 'supports', supportsBuffer, 'float32x2', queries.length),
      mode: 'co-kriging',
      coKrigingMean: 'ordinary',
      variableCount: props.variableCount,
      targetVariable: props.targetVariable,
      // The constructor asks for `covariance` as well; the structures below are what is used.
      covariance: {model: 'exponential', sill: SILL, range: RANGE, nugget: 0},
      coregionalizationModels: [
        {
          covariance: {model: 'exponential', sill: SILL, range: RANGE, nugget: 0},
          loadings: props.loadings
        }
      ],
      neighborCount: 16,
      searchRadius: props.searchRadius,
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32x4', queries.length),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', queries.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(valuesBuffer, queries.length * 4),
    status: await readUint32(statusBuffer, queries.length)
  };
  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    observationsBuffer,
    supportsBuffer,
    valuesBuffer,
    statusBuffer
  ]) {
    buffer.destroy();
  }
  return result;
}

it('ordinary co-kriging remaps the kept constraints: three variables, the middle one absent, target last', async () => {
  // Near the queries: four samples of variable 0 (secondary) and five of variable 2 (target).
  // Variable 1 only has samples near (100, 100), outside the search radius.
  const nearSamples: CoKrigingSample[] = [
    {position: [0.2, 0.1], value: 4.2, variable: 0},
    {position: [1.3, -0.4], value: 3.1, variable: 0},
    {position: [-0.9, 0.8], value: 5.0, variable: 0},
    {position: [0.5, 1.6], value: 3.7, variable: 0},
    {position: [0, 0], value: -1.2, variable: 2},
    {position: [1, 0.2], value: -0.3, variable: 2},
    {position: [-0.7, 1.1], value: -2.1, variable: 2},
    {position: [0.4, -1.3], value: 0.4, variable: 2},
    {position: [1.8, 1.5], value: -0.8, variable: 2}
  ];
  const farSamples: CoKrigingSample[] = [
    {position: [100, 100], value: 11, variable: 1},
    {position: [101, 100.5], value: 12, variable: 1},
    {position: [100.2, 101.3], value: 13, variable: 1}
  ];
  const queries = [
    [0.3, 0.2],
    [-0.5, 0.4],
    [1.2, -0.6]
  ];
  // Interleave the far samples so the retained rows are not the leading rows of the buffers.
  const samples = [
    farSamples[0],
    ...nearSamples.slice(0, 5),
    farSamples[1],
    ...nearSamples.slice(5),
    farSamples[2]
  ];
  const {values, status} = await runGeneralCoKriging({
    samples,
    queries,
    variableCount: 3,
    targetVariable: 2,
    loadings: THREE_VARIABLE_LOADINGS,
    searchRadius: 20
  });
  expect(status).toEqual([0, 0, 0]);
  const coregionalization = getCoregionalization(THREE_VARIABLE_LOADINGS, 3);
  for (const [index, query] of queries.entries()) {
    const expected = getOrdinaryCoKriging(nearSamples, query, coregionalization, 2, [0, 2]);
    expect(values[index * 4]).toBeCloseTo(expected.prediction, 4);
    expect(values[index * 4 + 1]).toBeCloseTo(expected.variance, 4);
    expect(values[index * 4 + 3]).toBe(nearSamples.length);
    // The secondary variable matters: the target alone gives a different prediction.
    const targetOnly = getOrdinaryCoKriging(
      nearSamples.filter(sample => sample.variable === 2),
      query,
      coregionalization,
      2,
      [2]
    );
    expect(Math.abs(expected.prediction - targetOnly.prediction)).toBeGreaterThan(1e-2);
  }
});

it('ordinary co-kriging predicts from fewer samples than variables when the target is present', async () => {
  // Three variables and two retained samples, both of the target (variable 1): one constraint
  // is kept, so the system is 3 x 3. Before the constraints were dropped this was status 1
  // (fewer samples than variables). Expected: ordinary kriging of the target from two samples.
  const nearSamples: CoKrigingSample[] = [
    {position: [0, 0], value: 2.5, variable: 1},
    {position: [2, 1], value: 4.0, variable: 1}
  ];
  const samples: CoKrigingSample[] = [
    {position: [100, 100], value: 9, variable: 0},
    ...nearSamples,
    {position: [101, 100], value: 7, variable: 2}
  ];
  const queries = [
    [0.5, 0.5],
    [100.2, 100.1]
  ];
  const {values, status} = await runGeneralCoKriging({
    samples,
    queries,
    variableCount: 3,
    targetVariable: 1,
    loadings: THREE_VARIABLE_LOADINGS,
    searchRadius: 20
  });
  const coregionalization = getCoregionalization(THREE_VARIABLE_LOADINGS, 3);
  const expected = getOrdinaryCoKriging(nearSamples, queries[0], coregionalization, 1, [1]);
  expect(status[0]).toBe(0);
  expect(values[0]).toBeCloseTo(expected.prediction, 4);
  expect(values[1]).toBeCloseTo(expected.variance, 4);
  expect(values[3]).toBe(2);
  // Second query: two secondary samples (variables 0 and 2), no target. Three constraints are
  // kept (target, 0, 2) for two samples: status 1, insufficient data.
  expect(status[1]).toBe(1);
  expect(Number.isNaN(values[4])).toBe(true);
});
