// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer, readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext,
  solveDense
} from './math-review-regression-test-utils';

type Scene = {
  positions: Float32Array;
  times: Float32Array;
  predictor: Float32Array;
  response: Float32Array;
};

function createScene(rows: number, seed: number): Scene {
  const random = createRandom(seed);
  const normal = createNormalRandom(seed + 1);
  const positions = new Float32Array(rows * 2);
  const times = new Float32Array(rows);
  const predictor = new Float32Array(rows);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = 10 * random();
    positions[row * 2 + 1] = 10 * random();
    times[row] = 5 * random();
    predictor[row] = normal();
    response[row] =
      1 +
      0.1 * positions[row * 2] +
      (0.5 + 0.1 * positions[row * 2 + 1]) * predictor[row] +
      0.3 * normal();
  }
  return {positions, times, predictor, response};
}

async function runGTWR(scene: Scene, bandwidth: number, timeScale: number) {
  const context = await createRegressionTestContext();
  const {graph, input, output, device} = context;
  const rows = scene.response.length;
  const positionBuffer = createInputBuffer(device, scene.positions);
  try {
    const summary = output('float32', 12);
    const localStatus = output('uint32', rows);
    const hatDiagonal = output('float32', rows);
    graph.add(
      new GPUGeographicallyTemporallyWeightedRegression({
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rows),
        times: input('float32', scene.times),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'float32-relative',
          axis: {kind: 'irregular'}
        },
        predictors: input('float32', scene.predictor),
        predictorCount: 1,
        response: input('float32', scene.response),
        kernel: 'gaussian',
        parameters: input('float32', getGPUGTWRParameterValues({bandwidth, timeScale})),
        output: {
          coefficients: output('float32', rows * 2).view,
          localStatus: localStatus.view,
          fitted: output('float32', rows).view,
          hatDiagonal: hatDiagonal.view,
          summary: summary.view
        }
      })
    );
    context.run();
    return {
      summary: Array.from(await readFloat32(summary.buffer, 12)),
      localStatus: Array.from(await readUint32(localStatus.buffer, rows)),
      hatDiagonal: Array.from(await readFloat32(hatDiagonal.buffer, rows))
    };
  } finally {
    context.destroy();
    positionBuffer.destroy();
  }
}

/**
 * Float64 fixed-bandwidth Gaussian GWR (mgwr): local weighted least squares at every row with
 * weights `exp(-0.5 (d / h)^2)`, the hat diagonal `x_i' (X'WX)^-1 x_i w_ii`, and mgwr's
 * `aicc = -2 llf + 2 n (tr(S) + 1) / (n - tr(S) - 2)` with
 * `llf = -n/2 (ln(2 pi RSS / n) + 1)`.
 */
function fitGaussianGWR(scene: Scene, bandwidth: number) {
  const rows = scene.response.length;
  let residualSquares = 0;
  let trace = 0;
  for (let focal = 0; focal < rows; focal++) {
    const normal = [
      [0, 0],
      [0, 0]
    ];
    const right = [0, 0];
    for (let row = 0; row < rows; row++) {
      const deltaX = scene.positions[row * 2] - scene.positions[focal * 2];
      const deltaY = scene.positions[row * 2 + 1] - scene.positions[focal * 2 + 1];
      const weight = Math.exp(
        (-0.5 * (deltaX * deltaX + deltaY * deltaY)) / (bandwidth * bandwidth)
      );
      const design = [1, scene.predictor[row]];
      for (let first = 0; first < 2; first++) {
        right[first] += weight * design[first] * scene.response[row];
        for (let second = 0; second < 2; second++)
          normal[first][second] += weight * design[first] * design[second];
      }
    }
    const coefficients = solveDense(normal, right);
    const focalDesign = [1, scene.predictor[focal]];
    const solved = solveDense(normal, focalDesign);
    trace += focalDesign[0] * solved[0] + focalDesign[1] * solved[1];
    const residual =
      scene.response[focal] - coefficients[0] - coefficients[1] * scene.predictor[focal];
    residualSquares += residual * residual;
  }
  const logLikelihood = -0.5 * rows * (Math.log((2 * Math.PI * residualSquares) / rows) + 1);
  return {
    trace,
    logLikelihood,
    corrected: -2 * logLikelihood + (2 * rows * (trace + 1)) / (rows - trace - 2),
    // mgtwr's own aicc, the value this summary slot used to hold.
    mgtwrCorrected: -2 * logLikelihood + (2 * (trace + 1) * (rows - 1)) / (rows - trace - 1)
  };
}

it('GTWR with timeScale 0 reports the GWR (mgwr) AICc of the same model (regression-06)', async () => {
  const scene = createScene(120, 606);
  // A small bandwidth gives a large hat trace, where the two formulas are furthest apart.
  const bandwidth = 0.9;
  const reference = fitGaussianGWR(scene, bandwidth);
  expect(reference.trace).toBeGreaterThan(30);
  expect(reference.corrected - reference.mgtwrCorrected).toBeGreaterThan(1);
  const result = await runGTWR(scene, bandwidth, 0);
  expect(result.localStatus.every(code => code === 0)).toBe(true);
  expect(Math.abs(result.summary[1] - reference.trace)).toBeLessThan(2e-3 * reference.trace);
  expect(Math.abs(result.summary[10] - reference.logLikelihood)).toBeLessThan(
    2e-3 * Math.abs(reference.logLikelihood)
  );
  expect(
    Math.abs(result.summary[2] - reference.corrected),
    `AICc ${result.summary[2]} vs mgwr ${reference.corrected} (mgtwr ${reference.mgtwrCorrected})`
  ).toBeLessThan(0.25 * (reference.corrected - reference.mgtwrCorrected));
});

it('GTWR voids the AICc when n - tr(S) - 2 <= 0 (regression-05)', async () => {
  const scene = createScene(40, 505);
  // A bandwidth far below the row spacing: every local fit nearly interpolates its own row, so
  // the hat trace approaches the row count and the correction is undefined. The former formula
  // published a large negative correction here, which wins any smallest-AICc profile search.
  const result = await runGTWR(scene, 0.25, 0);
  const fitted = result.localStatus.filter(code => code === 0).length;
  const trace = result.summary[1];
  expect(result.summary[4]).toBe(fitted);
  expect(fitted).toBeGreaterThan(10);
  expect(fitted - trace - 2).toBeLessThanOrEqual(0);
  expect(Number.isNaN(result.summary[2])).toBe(true);
  // The other summary values stay defined.
  expect(Number.isFinite(result.summary[0])).toBe(true);
  expect(Number.isFinite(result.summary[10])).toBe(true);

  // A sensible bandwidth on the same rows keeps a finite AICc.
  const regular = await runGTWR(scene, 3, 0);
  expect(Number.isFinite(regular.summary[2])).toBe(true);
});
