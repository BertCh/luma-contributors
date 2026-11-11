// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer, readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  createRandom,
  createRegressionTestContext,
  solveDense
} from './math-review-regression-test-utils';

const ROW_COUNT = 200;
const BANDWIDTHS = [0.6, 2, 8];

/** Bernoulli responses with a steep logit: the linear predictor reaches about +-18. */
function createScene() {
  const random = createRandom(707);
  const positions = new Float32Array(ROW_COUNT * 2);
  const predictor = new Float32Array(ROW_COUNT);
  const response = new Float32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    positions[row * 2] = random();
    positions[row * 2 + 1] = random();
    predictor[row] = 12 * random() - 6;
    response[row] = random() < 1 / (1 + Math.exp(-(0.5 + 3 * predictor[row]))) ? 1 : 0;
  }
  return {positions, predictor, response};
}

/**
 * Float64 leave-one-out local logistic regression with Gaussian weights `exp(-0.5 (d / h)^2)`:
 * Newton iterations at every focal row with the focal row removed, then the Bernoulli deviance
 * `2 log(1 + exp(-+eta))` of the focal response at its predicted linear predictor.
 */
function getLeaveOneOutDeviance(scene: ReturnType<typeof createScene>, bandwidth: number) {
  let total = 0;
  let largestLinear = 0;
  for (let focal = 0; focal < ROW_COUNT; focal++) {
    const weights: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      const deltaX = scene.positions[row * 2] - scene.positions[focal * 2];
      const deltaY = scene.positions[row * 2 + 1] - scene.positions[focal * 2 + 1];
      weights.push(
        row === focal
          ? 0
          : Math.exp((-0.5 * (deltaX * deltaX + deltaY * deltaY)) / (bandwidth * bandwidth))
      );
    }
    let coefficients = [0, 0];
    for (let iteration = 0; iteration < 200; iteration++) {
      const gradient = [0, 0];
      const information = [
        [0, 0],
        [0, 0]
      ];
      for (let row = 0; row < ROW_COUNT; row++) {
        const design = [1, scene.predictor[row]];
        const linear = coefficients[0] + coefficients[1] * scene.predictor[row];
        const mean = 1 / (1 + Math.exp(-linear));
        for (let first = 0; first < 2; first++) {
          gradient[first] += weights[row] * design[first] * (scene.response[row] - mean);
          for (let second = 0; second < 2; second++)
            information[first][second] +=
              weights[row] * mean * (1 - mean) * design[first] * design[second];
        }
      }
      const step = solveDense(information, gradient);
      coefficients = coefficients.map((value, column) => value + step[column]);
      if (Math.max(Math.abs(step[0]), Math.abs(step[1])) < 1e-12) break;
    }
    const linear = coefficients[0] + coefficients[1] * scene.predictor[focal];
    largestLinear = Math.max(largestLinear, Math.abs(linear));
    const opposed = scene.response[focal] === 1 ? -linear : linear;
    total += 2 * (Math.max(opposed, 0) + Math.log(1 + Math.exp(-Math.abs(opposed))));
  }
  return {score: total / ROW_COUNT, largestLinear};
}

it('logistic bandwidth selection scores confident leave-one-out predictions (regression-07)', async () => {
  const scene = createScene();
  const references = BANDWIDTHS.map(bandwidth => getLeaveOneOutDeviance(scene, bandwidth));
  // Every candidate has a leave-one-out linear predictor beyond 17, where the float32 fitted mean
  // is exactly 1: the former kernel rejected that row and voided the candidate.
  for (const reference of references) expect(reference.largestLinear).toBeGreaterThan(17);
  const expectedBandwidth =
    BANDWIDTHS[
      references.reduce(
        (best, reference, index) => (reference.score < references[best].score ? index : best),
        0
      )
    ];

  const context = await createRegressionTestContext();
  const {graph, input, output, device} = context;
  const positionBuffer = createInputBuffer(device, scene.positions);
  try {
    const scores = output('float32', BANDWIDTHS.length);
    const bandwidth = output('float32', 1);
    const status = output('uint32', 1);
    const fitted = output('float32', ROW_COUNT);
    const codes = output('uint32', ROW_COUNT);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'logistic',
        predictorCount: 1,
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', ROW_COUNT),
        predictors: input('float32', scene.predictor),
        response: input('float32', scene.response),
        kernel: 'gaussian',
        parameters: input('float32', [1, 1e-4]),
        maximumIterations: 128,
        bandwidthSelection: {
          bandwidths: BANDWIDTHS,
          output: {scores: scores.view, bandwidth: bandwidth.view, status: status.view}
        },
        output: {
          coefficients: output('float32', ROW_COUNT * 2).view,
          fitted: fitted.view,
          status: {codes: codes.view, iterationCount: output('uint32', ROW_COUNT).view}
        }
      })
    );
    context.run();
    const scoreValues = Array.from(await readFloat32(scores.buffer, BANDWIDTHS.length));
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    references.forEach((reference, index) =>
      expect(
        Math.abs(scoreValues[index] / reference.score - 1),
        `bandwidth ${BANDWIDTHS[index]}: ${scoreValues} vs ${references.map(value => value.score)}`
      ).toBeLessThan(2e-3)
    );
    expect((await readFloat32(bandwidth.buffer, 1))[0]).toBe(expectedBandwidth);
    // The selected fit still publishes fitted means in [0, 1].
    const fittedValues = Array.from(await readFloat32(fitted.buffer, ROW_COUNT));
    expect(Array.from(await readUint32(codes.buffer, ROW_COUNT)).every(code => code === 0)).toBe(
      true
    );
    expect(fittedValues.every(value => value >= 0 && value <= 1)).toBe(true);
    expect(fittedValues.some(value => value === 1)).toBe(true);
  } finally {
    context.destroy();
    positionBuffer.destroy();
  }
});
