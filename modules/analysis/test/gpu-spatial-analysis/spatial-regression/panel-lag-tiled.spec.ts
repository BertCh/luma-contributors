// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {computeSpatialPanelLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext, solveDense} from './math-review-regression-test-utils';
import fixtures from './spatial-panel-lag-maximum-likelihood-operation-fixtures.json';

function createRingScene(entities: number, periodCount: number) {
  const offsets = new Uint32Array(entities + 1);
  const neighbors = new Uint32Array(entities * 2);
  const weights = new Float32Array(entities * 2).fill(0.5);
  for (let entity = 0; entity < entities; entity++) {
    const pair = [(entity + entities - 1) % entities, (entity + 1) % entities].sort(
      (first, second) => first - second
    );
    offsets[entity] = entity * 2;
    neighbors.set(pair, entity * 2);
  }
  offsets[entities] = entities * 2;
  const predictors = new Float32Array(entities * periodCount);
  const response = new Float32Array(entities * periodCount);
  const matrix = Array.from({length: entities}, (_, row) =>
    Array.from(
      {length: entities},
      (_, column) =>
        Number(row === column) -
        (column === (row + 1) % entities || column === (row + entities - 1) % entities ? 0.1 : 0)
    )
  );
  for (let period = 0; period < periodCount; period++) {
    const signal = Array.from({length: entities}, (_, entity) => {
      const row = period * entities + entity;
      predictors[row] = Math.sin(row * 0.17) + Math.cos(period * 0.41);
      return 1.5 * predictors[row] + Math.sin(entity * 0.21) + 0.5 * Math.cos(row * 0.31);
    });
    const solved = solveDense(matrix, signal);
    response.set(solved, period * entities);
  }
  return {
    weights: {offsets, neighbors, weights},
    predictors,
    response,
    eigenvalues: Float32Array.from({length: entities}, (_, index) =>
      Math.cos((2 * Math.PI * index) / entities)
    )
  };
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 0.005) {
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index]).toBeNaN();
    else
      expect(Math.abs(actual[index] - expected[index]), `slot ${index}`).toBeLessThan(
        tolerance * Math.max(1, Math.abs(expected[index]))
      );
  }
}

for (const [entities, periods, packed] of [
  [4, 4, true],
  [4, 65, false],
  [40, 8, false]
] as const) {
  it(`tiled exact unit panel ${entities}x${periods}, packed=${packed}, matches the float64 CPU oracle`, {
    timeout: 120000
  }, async () => {
    const scene = createRingScene(entities, periods);
    const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(scene, {
      periodCount: periods,
      predictorCount: 1,
      effects: 'unit'
    });
    expect(Array.from(expected.status)).toEqual([0]);
    const context = await createRegressionTestContext();
    try {
      const rows = entities * periods;
      const coefficients = context.output('float32', packed ? rows + 4 : 4);
      const covariance = context.output('float32', 9);
      const summary = context.output('float32', 6);
      const status = context.output('uint32', 1);
      const fitted = context.output('float32', rows);
      const residuals = context.output('float32', rows);
      const coefficientView = packed
        ? context.graph.createDataView(coefficients.view.buffer, {format: 'float32', length: 4})
        : coefficients.view;
      const fittedView = packed
        ? context.graph.createDataView(coefficients.view.buffer, {
            format: 'float32',
            byteOffset: 16,
            length: rows
          })
        : fitted.view;
      context.graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          periodCount: periods,
          predictorCount: 1,
          effects: 'unit',
          eigenvalues: context.input('float32', scene.eigenvalues),
          predictors: context.input('float32', scene.predictors),
          response: context.input('float32', scene.response),
          weights: {
            offsets: context.input('uint32', scene.weights.offsets),
            neighbors: context.input('uint32', scene.weights.neighbors),
            weights: context.input('float32', scene.weights.weights)
          },
          output: {
            coefficients: coefficientView,
            covariance: covariance.view,
            summary: summary.view,
            status: status.view,
            fitted: fittedView,
            residuals: residuals.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const packedValues = await readFloat32(coefficients.buffer, packed ? rows + 4 : 4);
      expectClose(packedValues.slice(0, 4), expected.coefficients);
      expectClose(await readFloat32(covariance.buffer, 9), expected.covariance);
      expectClose((await readFloat32(summary.buffer, 6)).slice(0, 2), expected.summary.slice(0, 2));
      expectClose(
        packed ? packedValues.slice(4) : await readFloat32(fitted.buffer, rows),
        expected.fitted
      );
      expectClose(await readFloat32(residuals.buffer, rows), expected.residuals);
    } finally {
      context.destroy();
    }
  });
}

for (const effects of ['random', 'unit', 'hausman'] as const) {
  it(`reference panel ${effects} keeps its CPU oracle after shared LU adoption`, async () => {
    const fixture = fixtures[0];
    const input = {
      predictors: Float32Array.from(fixture.predictors),
      response: Float32Array.from(fixture.response),
      weights: {
        offsets: Uint32Array.from(fixture.weights.offsets),
        neighbors: Uint32Array.from(fixture.weights.neighbors),
        weights: Float32Array.from(fixture.weights.weights)
      }
    };
    const options = {
      periodCount: fixture.periodCount,
      predictorCount: fixture.predictorCount,
      effects: effects === 'unit' ? ('unit' as const) : ('random' as const),
      hausman: effects === 'hausman'
    };
    const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, options);
    const context = await createRegressionTestContext();
    try {
      const rows = fixture.response.length;
      const coefficientCount = fixture.predictorCount + 3;
      const coefficients = context.output('float32', coefficientCount);
      const covariance = context.output('float32', (coefficientCount - 1) ** 2);
      const summary = context.output('float32', 6);
      const status = context.output('uint32', 1);
      const fitted = context.output('float32', rows);
      const residuals = context.output('float32', rows);
      const hausman = context.output('float32', 3);
      const hausmanStatus = context.output('uint32', 1);
      context.graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          ...options,
          fitMode: 'reference',
          hausman:
            effects === 'hausman' ? {tests: hausman.view, status: hausmanStatus.view} : undefined,
          predictors: context.input('float32', input.predictors),
          response: context.input('float32', input.response),
          weights: {
            offsets: context.input('uint32', input.weights.offsets),
            neighbors: context.input('uint32', input.weights.neighbors),
            weights: context.input('float32', input.weights.weights)
          },
          output: {
            coefficients: coefficients.view,
            covariance: covariance.view,
            summary: summary.view,
            status: status.view,
            fitted: fitted.view,
            residuals: residuals.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual(Array.from(expected.status));
      expectClose(await readFloat32(coefficients.buffer, coefficientCount), expected.coefficients);
      expectClose(
        await readFloat32(covariance.buffer, (coefficientCount - 1) ** 2),
        expected.covariance
      );
      expectClose(await readFloat32(fitted.buffer, rows), expected.fitted);
      expectClose(await readFloat32(residuals.buffer, rows), expected.residuals);
      if (expected.hausman) {
        expect(Array.from(await readUint32(hausmanStatus.buffer, 1))).toEqual(
          Array.from(expected.hausman.status)
        );
        expectClose(await readFloat32(hausman.buffer, 3), expected.hausman.tests, 0.02);
      }
    } finally {
      context.destroy();
    }
  });
}
