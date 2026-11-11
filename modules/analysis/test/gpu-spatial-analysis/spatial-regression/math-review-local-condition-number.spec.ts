// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer, readFloat32} from '../../utils/gpu-contributor-test-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext
} from './math-review-regression-test-utils';

const ROW_COUNT = 150;
const BANDWIDTH = 4;

function createScene(center: number, halfRange: number) {
  const random = createRandom(1111);
  const normal = createNormalRandom(1112);
  const positions = new Float32Array(ROW_COUNT * 2);
  const predictor = new Float32Array(ROW_COUNT);
  const response = new Float32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    positions[row * 2] = 10 * random();
    positions[row * 2 + 1] = 10 * random();
    predictor[row] = center + (2 * random() - 1) * halfRange;
    response[row] = 1 + (2 * (predictor[row] - center)) / halfRange + 0.2 * normal();
  }
  return {positions, predictor, response};
}

/**
 * Float64 mgwr `local_collinearity` condition number of the design `[1, x]` at every row: with
 * the columns `a = w` and `b = w x` (kernel weights `w`, not their square roots), the
 * column-normalized design has singular values `sqrt(1 +- c)`, `c` the cosine of `a` and `b`, so
 * the condition number is `sqrt((1 + |c|) / (1 - |c|))` in closed form.
 */
function getReferenceConditionNumbers(scene: ReturnType<typeof createScene>): number[] {
  return Array.from({length: ROW_COUNT}, (_, focal) => {
    let squaredOnes = 0;
    let squaredPredictor = 0;
    let product = 0;
    for (let row = 0; row < ROW_COUNT; row++) {
      const deltaX = scene.positions[row * 2] - scene.positions[focal * 2];
      const deltaY = scene.positions[row * 2 + 1] - scene.positions[focal * 2 + 1];
      const weight = Math.exp(
        (-0.5 * (deltaX * deltaX + deltaY * deltaY)) / (BANDWIDTH * BANDWIDTH)
      );
      squaredOnes += weight * weight;
      squaredPredictor += weight * weight * scene.predictor[row] * scene.predictor[row];
      product += weight * weight * scene.predictor[row];
    }
    // 1 - c from the weighted variance, which float64 holds for these ratios:
    // 1 - c^2 = (S0 S2 - S1^2) / (S0 S2).
    const cosine = Math.abs(product) / Math.sqrt(squaredOnes * squaredPredictor);
    const complement =
      (squaredOnes * squaredPredictor - product * product) /
      (squaredOnes * squaredPredictor) /
      (1 + cosine);
    return Math.sqrt((1 + cosine) / complement);
  });
}

async function runLocalConditionNumbers(scene: ReturnType<typeof createScene>) {
  const context = await createRegressionTestContext();
  const {graph, input, output, device} = context;
  const positionBuffer = createInputBuffer(device, scene.positions);
  const parameters = new GPUParameterBuffer(device, {
    id: 'condition-number-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(1)
  });
  try {
    const conditionNumbers = output('float32', ROW_COUNT);
    graph.add(
      new GPUGeographicallyWeightedRegression({
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', ROW_COUNT),
        predictors: input('float32', scene.predictor),
        predictorCount: 1,
        response: input('float32', scene.response),
        parameters: parameters.importToGraph(graph),
        maximumBandwidthCount: 1,
        refinement: false,
        output: {
          coefficients: output('float32', ROW_COUNT * 2).view,
          selectedBandwidth: output('float32', 2).view,
          localConditionNumber: conditionNumbers.view
        }
      })
    );
    parameters.write(
      getGPUGeographicallyWeightedRegressionParameterValues(
        {kernel: 'gaussian', bandwidths: [BANDWIDTH], bandwidthMode: 'fixed'},
        1
      )
    );
    context.run();
    return Array.from(await readFloat32(conditionNumbers.buffer, ROW_COUNT));
  } finally {
    context.destroy();
    positionBuffer.destroy();
    parameters.destroy();
  }
}

for (const [name, center, halfRange, tolerance] of [
  ['an order-1 predictor (control)', 2, 1, 2e-3],
  ['an elevation (1500 +- 3)', 1500, 3, 5e-3],
  ['a UTM northing (5.2e6 +- 5000)', 5.2e6, 5000, 2e-2]
] as const)
  it(`GWR local condition number of ${name} matches the float64 closed form (regression-11)`, async () => {
    const scene = createScene(center, halfRange);
    const reference = getReferenceConditionNumbers(scene);
    const actual = await runLocalConditionNumbers(scene);
    let largestError = 0;
    let worstRow = 0;
    reference.forEach((value, row) => {
      const error = Math.abs(actual[row] / value - 1);
      if (!(error <= largestError)) {
        largestError = error;
        worstRow = row;
      }
    });
    expect(
      largestError,
      `row ${worstRow}: ${actual[worstRow]} vs ${reference[worstRow]} (largest relative error ${largestError})`
    ).toBeLessThan(tolerance);
  });
