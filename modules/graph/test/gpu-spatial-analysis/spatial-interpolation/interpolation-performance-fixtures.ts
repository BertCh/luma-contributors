// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVariogramModelFit} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram-model-fit';
import type {
  InterpolationBenchmarkCase,
  InterpolationBenchmarkOutput
} from './interpolation-performance-harness';

export const INTERPOLATION_BENCHMARK_VERSION = 1;
export const INTERPOLATION_BENCHMARK_SEED = 0x494e5445;
const COVARIANCE = {model: 'exponential' as const, range: 1, sill: 1, nugget: 0.05};

function makePositions(count: number): Float32Array {
  const columns = count === 32 ? 8 : 4;
  const rows = count / columns;
  return Float32Array.from(
    Array.from({length: count}, (_, row) => [
      (row % columns) / (columns - 1),
      Math.floor(row / columns) / (rows - 1)
    ]).flat()
  );
}
function makeQueries(count: number): Float32Array {
  return Float32Array.from(
    Array.from({length: count}, (_, row) => [
      0.14 + (0.72 * ((row * 7) % 17)) / 16,
      0.14 + (0.72 * ((row * 11 + 3) % 19)) / 18
    ]).flat()
  );
}
function meanValue(horizontal: number, vertical: number, bilinear = false): number {
  return 2 + 0.5 * horizontal - 0.25 * vertical + (bilinear ? 0.125 * horizontal * vertical : 0);
}
function closeTo(actual: number, expected: number, tolerance = 1e-4): void {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    tolerance * Math.max(1, Math.abs(expected))
  );
}
function checkPredictions(
  actual: InterpolationBenchmarkOutput,
  queries: Float32Array,
  updated: boolean,
  sampleCount: number,
  bilinear = false
): void {
  expect(actual['status'].every(status => status === 0)).toBe(true);
  for (let query = 0; query < queries.length / 2; query++) {
    closeTo(
      actual['result'][query * 4],
      meanValue(queries[query * 2], queries[query * 2 + 1], bilinear) + (updated ? 3 : 0)
    );
    expect(actual['result'].slice(query * 4, query * 4 + 4).every(Number.isFinite)).toBe(true);
    if (actual['weights']) {
      closeTo(
        actual['weights']
          .slice(query * sampleCount, (query + 1) * sampleCount)
          .reduce((sum, weight) => sum + weight, 0),
        1
      );
    }
  }
}

/** Independent f64 elimination oracle for conditional means and marginal variances. */
export function solveBenchmarkSystem(
  inputMatrix: readonly number[][],
  inputRight: readonly number[]
): number[] {
  const matrix = inputMatrix.map((row, index) => [...row, inputRight[index]]);
  const count = matrix.length;
  for (let column = 0; column < count; column++) {
    let pivot = column;
    for (let row = column + 1; row < count; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    if (Math.abs(matrix[column][column]) < 1e-12) {
      throw new Error('Singular benchmark reference');
    }
    const diagonal = matrix[column][column];
    for (let entry = column; entry <= count; entry++) {
      matrix[column][entry] /= diagonal;
    }
    for (let row = 0; row < count; row++) {
      if (row === column) {
        continue;
      }
      const factor = matrix[row][column];
      for (let entry = column; entry <= count; entry++) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
    }
  }
  return matrix.map(row => row[count]);
}
function computeSimulationMoments(
  positions: Float32Array,
  queries: Float32Array,
  values: Float32Array,
  updated: boolean
): number[] {
  const count = positions.length / 2;
  const covariance = (
    firstHorizontal: number,
    firstVertical: number,
    secondHorizontal: number,
    secondVertical: number
  ) =>
    Math.exp(-3 * Math.hypot(firstHorizontal - secondHorizontal, firstVertical - secondVertical)) +
    (firstHorizontal === secondHorizontal && firstVertical === secondVertical ? 0.05 : 0);
  const matrix = Array.from({length: count}, (_, row) =>
    Array.from(
      {length: count},
      (_, column) =>
        covariance(
          positions[row * 2],
          positions[row * 2 + 1],
          positions[column * 2],
          positions[column * 2 + 1]
        ) + (row === column ? 0.1 : 0)
    )
  );
  return Array.from({length: queries.length / 2}, (_, query) => {
    const cross = Array.from({length: count}, (_, row) =>
      covariance(
        positions[row * 2],
        positions[row * 2 + 1],
        queries[query * 2],
        queries[query * 2 + 1]
      )
    );
    const weights = solveBenchmarkSystem(matrix, cross);
    return [
      2 +
        weights.reduce(
          (sum, weight, row) => sum + weight * (values[row] + (updated ? 3 : 0) - 2),
          0
        ),
      1.05 - weights.reduce((sum, weight, row) => sum + weight * cross[row], 0)
    ];
  }).flat();
}

export function createInterpolationBenchmarkCases(): InterpolationBenchmarkCase[] {
  const cases: InterpolationBenchmarkCase[] = [];
  for (const large of [false, true]) {
    const sampleCount = large ? 16 : 8;
    const queryCount = large ? 128 : 4;
    const positions = makePositions(sampleCount);
    const queries = makeQueries(queryCount);
    const affineValues = Float32Array.from(
      Array.from({length: sampleCount}, (_, row) =>
        meanValue(positions[row * 2], positions[row * 2 + 1])
      )
    );
    const updatedAffineValues = affineValues.map(value => value + 3);
    const supportCount = large ? 16 : 1;
    const supports = Float32Array.from(
      Array.from({length: queryCount}, (_, query) =>
        Array.from({length: supportCount}, (_, support) => [
          queries[query * 2] + (large ? ((support % 4) - 1.5) * 0.01 : 0),
          queries[query * 2 + 1] + (large ? (Math.floor(support / 4) - 1.5) * 0.01 : 0)
        ]).flat()
      ).flat()
    );
    const observations = Float32Array.from(
      Array.from({length: sampleCount}, (_, row) => [
        meanValue(positions[row * 2], positions[row * 2 + 1], true),
        0.1,
        0,
        1
      ]).flat()
    );
    const updatedObservations = observations.map(
      (value, index) => value + (index % 4 === 0 ? 3 : 0)
    );
    const sampleDrift = Float32Array.from(
      Array.from({length: sampleCount}, (_, row) => [
        1,
        positions[row * 2],
        positions[row * 2 + 1],
        positions[row * 2] * positions[row * 2 + 1]
      ]).flat()
    );
    const queryDrift = Float32Array.from(
      Array.from({length: queryCount}, (_, query) => [
        1,
        queries[query * 2],
        queries[query * 2 + 1],
        queries[query * 2] * queries[query * 2 + 1]
      ]).flat()
    );
    cases.push({
      name: `local-kriging-${sampleCount}-${queryCount}`,
      shape: {
        samples: sampleCount,
        queries: queryCount,
        neighbors: sampleCount,
        driftColumns: 4,
        supportCount,
        storageBindings: 8
      },
      correctness:
        'Bilinear external-drift reproduction, weight partition, finite variance and successful status, before and after +3 observation update.',
      construct(scope) {
        scope.graph.add(
          new GPULocalKriging({
            positions: scope.input('positions', positions, 'float32x2'),
            observations: scope.input(
              'observations',
              observations,
              'float32x4',
              updatedObservations
            ),
            supports: scope.input('supports', supports, 'float32x2'),
            supportCount,
            neighborCount: sampleCount,
            mode: 'external-drift',
            driftCount: 4,
            sampleDrift: scope.input('sample-drift', sampleDrift, 'float32x4'),
            queryDrift: scope.input('query-drift', queryDrift, 'float32x4'),
            covariance: COVARIANCE,
            output: {
              values: scope.output('result', queryCount, 'float32x4'),
              status: scope.output('status', queryCount, 'uint32'),
              weights: scope.output('weights', queryCount * sampleCount, 'float32')
            }
          })
        );
        return (actual, updated) => {
          checkPredictions(actual, queries, updated, sampleCount, true);
          for (let query = 0; query < queryCount; query++) {
            expect(actual['result'][query * 4 + 1]).toBeGreaterThanOrEqual(0);
            expect(actual['result'][query * 4 + 3]).toBe(sampleCount);
          }
        };
      }
    });
    cases.push({
      name: `local-rbf-${sampleCount}-${queryCount}`,
      shape: {
        samples: sampleCount,
        queries: queryCount,
        neighbors: sampleCount,
        polynomialDegree: 1,
        smoothing: 0.05
      },
      correctness:
        'Affine reproduction of thin-plate spline with smoothing and finite pivot diagnostics; +3 input update.',
      construct(scope) {
        scope.graph.add(
          new GPULocalRBF({
            positions: scope.input('positions', positions, 'float32x2'),
            values: scope.input('values', affineValues, 'float32', updatedAffineValues),
            queries: scope.input('queries', queries, 'float32x2'),
            neighborCount: sampleCount,
            smoothing: 0.05,
            output: {
              values: scope.output('result', queryCount, 'float32x4'),
              status: scope.output('status', queryCount, 'uint32')
            }
          })
        );
        return (actual, updated) => {
          checkPredictions(actual, queries, updated, sampleCount);
          for (let query = 0; query < queryCount; query++) {
            expect(actual['result'][query * 4 + 2]).toBe(sampleCount);
          }
        };
      }
    });
    const naturalSampleCount = large ? 32 : 8;
    const naturalPositions = makePositions(naturalSampleCount);
    const naturalValues = Float32Array.from(
      Array.from({length: naturalSampleCount}, (_, row) =>
        meanValue(naturalPositions[row * 2], naturalPositions[row * 2 + 1])
      )
    );
    cases.push({
      name: `sibson-${naturalSampleCount}-${queryCount}`,
      shape: {samples: naturalSampleCount, queries: queryCount, maximumVertexCount: 64},
      correctness:
        'Affine precision, nonnegative weights, partition of unity, all statuses successful, +3 input update.',
      construct(scope) {
        scope.graph.add(
          new GPUNaturalNeighbor({
            positions: scope.input('positions', naturalPositions, 'float32x2'),
            values: scope.input(
              'values',
              naturalValues,
              'float32',
              naturalValues.map(value => value + 3)
            ),
            queries: scope.input('queries', queries, 'float32x2'),
            output: {
              values: scope.output('result', queryCount, 'float32x4'),
              status: scope.output('status', queryCount, 'uint32'),
              weights: scope.output('weights', queryCount * naturalSampleCount, 'float32')
            }
          })
        );
        return (actual, updated) => {
          checkPredictions(actual, queries, updated, naturalSampleCount);
          expect(actual['weights'].every(weight => weight >= 0)).toBe(true);
        };
      }
    });
    const simulationQueryCount = large ? 16 : 4;
    const simulationQueries = makeQueries(simulationQueryCount);
    const replicateCount = large ? 128 : 16;
    const simulationObservations = Float32Array.from(
      Array.from({length: sampleCount}, (_, row) => [affineValues[row], 0.1, 0, 1]).flat()
    );
    const initialMoments = computeSimulationMoments(
      positions,
      simulationQueries,
      affineValues,
      false
    );
    const updatedMoments = computeSimulationMoments(
      positions,
      simulationQueries,
      affineValues,
      true
    );
    cases.push({
      name: `conditional-simulation-${sampleCount}-${simulationQueryCount}-${replicateCount}`,
      shape: {
        observations: sampleCount,
        queries: simulationQueryCount,
        replicates: replicateCount,
        seed: INTERPOLATION_BENCHMARK_SEED
      },
      correctness:
        'Independent f64 conditional moments; finite joint draws; unchanged-input stream replay; observation update shifts each draw by the independently computed mean shift.',
      construct(scope) {
        scope.graph.add(
          new GPUConditionalGaussianSimulation({
            positions: scope.input('positions', positions, 'float32x2'),
            observations: scope.input(
              'observations',
              simulationObservations,
              'float32x4',
              simulationObservations.map((value, index) => value + (index % 4 === 0 ? 3 : 0))
            ),
            queries: scope.input('queries', simulationQueries, 'float32x2'),
            mean: 2,
            covariance: COVARIANCE,
            seed: INTERPOLATION_BENCHMARK_SEED,
            replicateCount,
            output: {
              values: scope.output('values', replicateCount * simulationQueryCount, 'float32'),
              moments: scope.output('moments', simulationQueryCount, 'float32x2'),
              status: scope.output('status', replicateCount, 'uint32')
            }
          })
        );
        let initialDraws: number[] | undefined;
        return (actual, updated) => {
          expect(actual['status'].every(status => status === 0)).toBe(true);
          const moments = updated ? updatedMoments : initialMoments;
          moments.forEach((expected, index) => closeTo(actual['moments'][index], expected));
          expect(actual['values'].every(Number.isFinite)).toBe(true);
          if (!initialDraws) {
            initialDraws = [...actual['values']];
            return;
          }
          actual['values'].forEach((value, index) => {
            const query = index % simulationQueryCount;
            closeTo(
              value,
              initialDraws![index] +
                (updated ? updatedMoments[query * 2] - initialMoments[query * 2] : 0)
            );
          });
        };
      }
    });
    const binCount = large ? 256 : 20;
    const distances = Float32Array.from(
      Array.from({length: binCount}, (_, row) => 0.05 + (4 * row) / (binCount - 1))
    );
    const semivariances = distances.map(
      distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
    );
    const pairCounts = Uint32Array.from(Array.from({length: binCount}, (_, row) => 10 + row));
    cases.push({
      name: `variogram-fit-${binCount}`,
      shape: {
        bins: binCount,
        modelCandidates: 3,
        rangeSteps: 200,
        refinementIterations: 32,
        weighting: 'pairs'
      },
      correctness:
        'Admissible candidate fits and explicit convergence/boundary status; generating exponential parameters recovered before and after x2 semivariance update.',
      construct(scope) {
        scope.graph.add(
          new GPUVariogramModelFit({
            distances: scope.input('distances', distances, 'float32'),
            semivariances: scope.input(
              'semivariances',
              semivariances,
              'float32',
              semivariances.map(value => value * 2)
            ),
            pairCounts: scope.input('pair-counts', pairCounts, 'uint32'),
            weighting: 'pairs',
            output: {
              parameters: scope.output('parameters', 3, 'float32x4'),
              diagnostics: scope.output('diagnostics', 3, 'float32x4'),
              status: scope.output('status', 3, 'uint32')
            }
          })
        );
        return (actual, updated) => {
          expect(actual['status'].every(status => status === 0 || status === 3)).toBe(true);
          expect(actual['parameters'].every(Number.isFinite)).toBe(true);
          for (let model = 0; model < 3; model++) {
            expect(actual['parameters'][model * 4]).toBeGreaterThanOrEqual(0);
            expect(actual['parameters'][model * 4 + 1]).toBeGreaterThanOrEqual(0);
            expect(actual['parameters'][model * 4 + 2]).toBeGreaterThan(0);
            expect(actual['diagnostics'][model * 4]).toBe(binCount);
          }
          closeTo(actual['parameters'][4], 0.2 * (updated ? 2 : 1), 0.003);
          closeTo(actual['parameters'][5], 1.5 * (updated ? 2 : 1), 0.003);
          closeTo(actual['parameters'][6], 2.2, 0.003);
        };
      }
    });
  }
  return cases;
}
