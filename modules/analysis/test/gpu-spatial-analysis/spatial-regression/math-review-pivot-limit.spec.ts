// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext,
  getNormalDistribution
} from './math-review-regression-test-utils';

/**
 * regression-10: probit, SUR and the panel lag estimator tested only the positivity of a Cholesky
 * pivot, so two predictors that agree to five digits returned status 0 with a meaningless
 * solution. They now apply the relative pivot limit of `GPUOrdinaryLeastSquares` (a factor pivot
 * of 1e-3 on the unit-diagonal matrix: a squared cosine of 1 - 1e-6 with the earlier columns).
 *
 * `spread` is the relative size of the independent part of the second predictor, so the squared
 * cosine of the two centered columns is `1 - spread^2`. The limit is pinned from both sides
 * (fr-statistics-05): 7e-4 (`1 - 4.9e-7`) must be singular, where the old positivity test had a
 * pivot of about 5e-7 and returned status 0, and 2e-3 (`1 - 4e-6`) must fit. 1e-5 and 0.1 are
 * the far sides.
 */
function createPredictorPair(rows: number, spread: number, seed: number): Float32Array {
  const normal = createNormalRandom(seed);
  const predictors = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row++) {
    const value = normal();
    predictors[row * 2] = value;
    predictors[row * 2 + 1] = value + spread * normal();
  }
  return predictors;
}

async function runProbit(spread: number) {
  const rows = 2000;
  const predictors = createPredictorPair(rows, spread, 41);
  const uniform = createRandom(42);
  const response = Float32Array.from({length: rows}, (_, row) =>
    uniform() < getNormalDistribution(0.2 + 0.8 * predictors[row * 2]) ? 1 : 0
  );
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const status = output('uint32', 1);
    const coefficients = output('float32', 3);
    graph.add(
      new GPUProbitRegression({
        predictors: input('float32', predictors),
        predictorCount: 2,
        response: input('float32', response),
        output: {
          coefficients: coefficients.view,
          standardErrors: output('float32', 3).view,
          fitted: output('float32', rows).view,
          marginalEffects: output('float32', 2).view,
          summary: output('float32', 4).view,
          status: status.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, 3))
    };
  } finally {
    context.destroy();
  }
}

async function runSeeminglyUnrelated(spread: number) {
  const rows = 400;
  const normal = createNormalRandom(52);
  // Equation-major, then row-major: both equations use the same near-collinear pair.
  const pair = createPredictorPair(rows, spread, 51);
  const predictors = new Float32Array(2 * rows * 2);
  predictors.set(pair, 0);
  predictors.set(pair, rows * 2);
  const responses = new Float32Array(2 * rows);
  for (let row = 0; row < rows; row++) {
    const shared = normal();
    responses[row] = 1 + pair[row * 2] + 0.3 * (shared + normal());
    responses[rows + row] = -1 + 0.5 * pair[row * 2] + 0.3 * (shared + normal());
  }
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const status = output('uint32', 1);
    const coefficients = output('float32', 6);
    graph.add(
      new GPUSeeminglyUnrelatedRegression({
        predictors: input('float32', predictors),
        responses: input('float32', responses),
        equationCount: 2,
        predictorCount: 2,
        rowCount: rows,
        output: {
          coefficients: coefficients.view,
          covariance: output('float32', 36).view,
          residualCovariance: output('float32', 4).view,
          residuals: output('float32', 2 * rows).view,
          status: status.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, 6))
    };
  } finally {
    context.destroy();
  }
}

async function runPanelLag(spread: number) {
  const entityCount = 12;
  const periodCount = 12;
  const rows = entityCount * periodCount;
  const predictors = createPredictorPair(rows, spread, 61);
  const normal = createNormalRandom(62);
  const effects = Array.from({length: entityCount}, () => 0.5 * normal());
  const response = Float32Array.from(
    {length: rows},
    (_, row) => 1 + predictors[row * 2] + effects[row % entityCount] + 0.5 * normal()
  );
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let entity = 0; entity < entityCount; entity++) {
    neighbors.push(
      ...[(entity + entityCount - 1) % entityCount, (entity + 1) % entityCount].sort(
        (first, second) => first - second
      )
    );
    weights.push(0.5, 0.5);
    offsets.push(neighbors.length);
  }
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const status = output('uint32', 1);
    const coefficients = output('float32', 5);
    graph.add(
      new GPUSpatialPanelLagMaximumLikelihood({
        weights: {
          offsets: input('uint32', offsets),
          neighbors: input('uint32', neighbors),
          weights: input('float32', weights)
        },
        periodCount,
        predictorCount: 2,
        predictors: input('float32', predictors),
        response: input('float32', response),
        maximumIterations: 128,
        tolerance: 2e-5,
        output: {
          coefficients: coefficients.view,
          covariance: output('float32', 16).view,
          summary: output('float32', 6).view,
          status: status.view,
          fitted: output('float32', rows).view,
          residuals: output('float32', rows).view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, 5))
    };
  } finally {
    context.destroy();
  }
}

async function runErrorMaximumLikelihood(spread: number) {
  const rows = 400;
  const predictors = createPredictorPair(rows, spread, 71);
  const normal = createNormalRandom(72);
  const response = Float32Array.from(
    {length: rows},
    (_, row) => 1 + predictors[row * 2] + 0.5 * normal()
  );
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < rows; row++) {
    neighbors.push(
      ...[(row + rows - 1) % rows, (row + 1) % rows].sort((first, second) => first - second)
    );
    weights.push(0.5, 0.5);
    offsets.push(neighbors.length);
  }
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const status = output('uint32', 1);
    const table = output('float32', 16);
    graph.add(
      new GPUSpatialErrorMaximumLikelihood({
        weights: {
          offsets: input('uint32', offsets),
          neighbors: input('uint32', neighbors),
          weights: input('float32', weights)
        },
        predictors: input('float32', predictors),
        predictorCount: 2,
        response: input('float32', response),
        output: {
          table: table.view,
          summary: output('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).view,
          status: {code: status.view}
        }
      })
    );
    context.run();
    const tableValues = await readFloat32(table.buffer, 16);
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: [tableValues[0], tableValues[4], tableValues[8]]
    };
  } finally {
    context.destroy();
  }
}

for (const [name, run, singularStatus] of [
  ['probit', runProbit, 2],
  ['seemingly unrelated regression', runSeeminglyUnrelated, 2],
  ['panel lag maximum likelihood', runPanelLag, 2],
  // The maximum-likelihood family numbers singular as 1.
  ['spatial error maximum likelihood', runErrorMaximumLikelihood, 1]
] as const) {
  for (const spread of [1e-5, 7e-4])
    it(`${name} reports predictors with a squared cosine of 1 - ${(spread * spread).toExponential(1)} as singular (regression-10)`, async () => {
      const result = await run(spread);
      expect(result.status, `coefficients ${result.coefficients}`).toBe(singularStatus);
    });

  for (const spread of [2e-3, 0.1])
    it(`${name} still fits predictors with a squared cosine of 1 - ${(spread * spread).toExponential(1)}`, async () => {
      const result = await run(spread);
      expect(result.status).toBe(0);
      expect(result.coefficients.slice(0, 3).every(Number.isFinite)).toBe(true);
    });
}
