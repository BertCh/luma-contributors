// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  buildVecchiaConditioningSets,
  buildVecchiaPredictionNeighborhoods
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createExperimentGaussian,
  createExperimentRandom,
  getReferenceCorrelation,
  simulateReferenceVecchiaField,
  type ReferenceVecchiaCovariance
} from './vecchia-likelihood-reference';

/** Simple-kriging mean and variance at `query` from `rows` by dense Cholesky in binary64. */
function solveReferencePrediction(
  positions: ArrayLike<number>,
  values: ArrayLike<number>,
  rows: readonly number[],
  query: readonly [number, number],
  covariance: ReferenceVecchiaCovariance
): {mean: number; variance: number} {
  const {model, range, sill, nugget = 0} = covariance;
  const covarianceAt = (first: number, second: number): number => {
    const offsetX = positions[first * 2] - positions[second * 2];
    const offsetY = positions[first * 2 + 1] - positions[second * 2 + 1];
    return (
      sill * getReferenceCorrelation(model, Math.hypot(offsetX, offsetY) / range) +
      (first === second ? nugget : 0)
    );
  };
  const size = rows.length;
  const lower = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      lower[row * size + column] = covarianceAt(rows[row], rows[column]);
    }
  }
  for (let column = 0; column < size; column++) {
    let diagonal = lower[column * size + column];
    for (let index = 0; index < column; index++) {
      diagonal -= lower[column * size + index] ** 2;
    }
    const pivot = Math.sqrt(diagonal);
    lower[column * size + column] = pivot;
    for (let row = column + 1; row < size; row++) {
      let value = lower[row * size + column];
      for (let index = 0; index < column; index++) {
        value -= lower[row * size + index] * lower[column * size + index];
      }
      lower[row * size + column] = value / pivot;
    }
  }
  const cross = rows.map(
    row =>
      sill *
      getReferenceCorrelation(
        model,
        Math.hypot(positions[row * 2] - query[0], positions[row * 2 + 1] - query[1]) / range
      )
  );
  const forward = Float64Array.from(cross);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < row; column++) {
      forward[row] -= lower[row * size + column] * forward[column];
    }
    forward[row] /= lower[row * size + row];
  }
  let variance = sill + nugget;
  for (let row = 0; row < size; row++) {
    variance -= forward[row] * forward[row];
  }
  const weights = Float64Array.from(forward);
  for (let row = size - 1; row >= 0; row--) {
    for (let column = row + 1; column < size; column++) {
      weights[row] -= lower[column * size + row] * weights[column];
    }
    weights[row] /= lower[row * size + row];
  }
  let mean = 0;
  for (let row = 0; row < size; row++) {
    mean += weights[row] * values[rows[row]];
  }
  return {mean, variance};
}

describe('Vecchia prediction', () => {
  it('equals the local kriging solve on the nearest data neighbours, with no floor', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createExperimentRandom(1234);
    const rowCount = 2000;
    const queryCount = 150;
    const positions = Float64Array.from({length: rowCount * 2}, () => (random() - 0.5) * 100000);
    const queries = Float64Array.from({length: queryCount * 2}, () => (random() - 0.5) * 90000);
    const covariances: ReferenceVecchiaCovariance[] = [
      {model: 'exponential', range: 30000, sill: 1.2, nugget: 0},
      {model: 'matern-3/2', range: 22000, sill: 1, nugget: 0}
    ];
    const conditioning = buildVecchiaConditioningSets({positions, neighborCount: 20});
    const values = simulateReferenceVecchiaField(
      createExperimentGaussian(random),
      positions,
      conditioning.offsets,
      conditioning.neighbors,
      covariances[0]
    );
    const prediction = buildVecchiaPredictionNeighborhoods({positions, queries, neighborCount: 20});

    const graph = new GPUCommandGraph(device);
    const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
    const observationRows = new Float32Array(rowCount * 4);
    for (let row = 0; row < rowCount; row++) {
      observationRows[row * 4] = values[row];
      observationRows[row * 4 + 3] = 1;
    }
    const observations = createInputBuffer(device, observationRows);
    const supports = createInputBuffer(device, Float32Array.from(queries));
    const offsets = createInputBuffer(device, prediction.offsets);
    const neighbors = createInputBuffer(device, prediction.neighbors);
    const parameters = createInputBuffer(
      device,
      getGPULocalKrigingParameterValues({covariance: covariances[0]})
    );
    const results = createOutputBuffer(device, queryCount * 4);
    const status = createOutputBuffer(device, queryCount);
    graph.add(
      new GPULocalKriging({
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
        observations: importGraphBuffer(graph, 'observations', observations, 'float32x4', rowCount),
        supports: importGraphBuffer(graph, 'supports', supports, 'float32x2', queryCount),
        mode: 'simple',
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
        neighborCount: 20,
        neighborhoods: importGraphBuffer(
          graph,
          'neighbors',
          neighbors,
          'uint32',
          prediction.neighbors.length
        ),
        neighborhoodOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', queryCount + 1),
        output: {
          values: importGraphBuffer(graph, 'values', results, 'float32x4', queryCount),
          status: importGraphBuffer(graph, 'status', status, 'uint32', queryCount)
        }
      })
    );
    const compiled = graph.compile();
    for (const covariance of covariances) {
      parameters.write(getGPULocalKrigingParameterValues({covariance}));
      submitGraph(device, compiled, undefined);
      const output = await readFloat32(results, queryCount * 4);
      const statuses = await readUint32(status, queryCount);
      let unreliable = 0;
      let largestMeanError = 0;
      let largestVarianceError = 0;
      for (let query = 0; query < queryCount; query++) {
        const begin = prediction.offsets[query];
        const end = prediction.offsets[query + 1];
        const expected = solveReferencePrediction(
          positions,
          values,
          Array.from(prediction.neighbors.subarray(begin, end)),
          [queries[query * 2], queries[query * 2 + 1]],
          covariance
        );
        if (statuses[query] !== 0) {
          // Prediction keeps its own rule (no nugget floor): status 2 flags an unreliable solve for
          // a smooth covariance with no nugget, and the exponential must not produce one here.
          expect(covariance.model).toBe('matern-3/2');
          expect(statuses[query]).toBe(2);
          unreliable++;
          continue;
        }
        largestMeanError = Math.max(largestMeanError, Math.abs(output[query * 4] - expected.mean));
        largestVarianceError = Math.max(
          largestVarianceError,
          Math.abs(output[query * 4 + 1] - expected.variance)
        );
      }
      console.log(
        `VECCHIA_PREDICTION ${covariance.model} nugget=${covariance.nugget ?? 0} unreliable=${unreliable} largestMeanError=${largestMeanError} largestVarianceError=${largestVarianceError}`
      );
      expect(largestMeanError).toBeLessThan(covariance.model === 'exponential' ? 5e-3 : 2e-2);
      expect(largestVarianceError).toBeLessThan(covariance.model === 'exponential' ? 5e-3 : 2e-2);
    }
    compiled.destroy();
    for (const buffer of [
      positionsBuffer,
      observations,
      supports,
      offsets,
      neighbors,
      parameters,
      results,
      status
    ]) {
      buffer.destroy();
    }
  });

  it('neighbourhoods match a brute-force nearest-data scan', () => {
    const random = createExperimentRandom(99);
    const positions = Float64Array.from({length: 2000}, () => (random() - 0.5) * 1000);
    const queries = Float64Array.from({length: 200}, () => (random() - 0.5) * 1000);
    const {offsets, neighbors} = buildVecchiaPredictionNeighborhoods({
      positions,
      queries,
      neighborCount: 7
    });
    for (let query = 0; query < 100; query++) {
      const nearest = Array.from({length: 1000}, (_, row) => ({
        row,
        distance:
          (positions[row * 2] - queries[query * 2]) ** 2 +
          (positions[row * 2 + 1] - queries[query * 2 + 1]) ** 2
      }))
        .sort((first, second) => first.distance - second.distance || first.row - second.row)
        .slice(0, 7)
        .map(entry => entry.row);
      expect(Array.from(neighbors.subarray(offsets[query], offsets[query + 1]))).toEqual(nearest);
    }
  });
});
