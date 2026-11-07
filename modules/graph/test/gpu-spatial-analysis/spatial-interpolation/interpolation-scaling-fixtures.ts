// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect} from 'vitest';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import type {InterpolationBenchmarkCase} from './interpolation-performance-harness';
import {solveBenchmarkSystem} from './interpolation-performance-fixtures';
import {calculateConditionalSimulationReference} from './conditional-simulation-reference';

/** Fixture topology is constructed outside GPU timing; runtime accepts caller-owned resident topology. */
function gridTopology(side: number) {
  const positions = Float32Array.from(
    Array.from({length: side * side}, (_, row) => [
      (row % side) / (side - 1),
      Math.floor(row / side) / (side - 1)
    ]).flat()
  );
  const triangles: number[][] = [];
  for (let vertical = 0; vertical < side - 1; vertical++) {
    for (let horizontal = 0; horizontal < side - 1; horizontal++) {
      const first = vertical * side + horizontal;
      triangles.push(
        [first, first + 1, first + side + 1, 0xffffffff, 0xffffffff, 0xffffffff],
        [first, first + side + 1, first + side, 0xffffffff, 0xffffffff, 0xffffffff]
      );
    }
  }
  const edges = new Map<string, [number, number]>();
  triangles.forEach((triangle, row) => {
    for (let edge = 0; edge < 3; edge++) {
      const first = triangle[edge];
      const second = triangle[(edge + 1) % 3];
      const key = `${Math.min(first, second)}:${Math.max(first, second)}`;
      const other = edges.get(key);
      if (other) {
        triangle[edge + 3] = other[0];
        triangles[other[0]][other[1] + 3] = row;
      } else {
        edges.set(key, [row, edge]);
      }
    }
  });
  return {positions, triangles: Uint32Array.from(triangles.flat())};
}
function queryRows(count: number) {
  return Float32Array.from(
    Array.from({length: count}, (_, row) => [
      0.13 + (0.71 * ((row * 17) % 101)) / 101,
      0.17 + (0.67 * ((row * 29) % 103)) / 103
    ]).flat()
  );
}
const response = (horizontal: number, vertical: number) => 2 + 0.5 * horizontal - 0.25 * vertical;

export function createInterpolationScalingCases(): InterpolationBenchmarkCase[] {
  const cases: InterpolationBenchmarkCase[] = [];
  for (const side of [32, 128, 256]) {
    const topology = gridTopology(side);
    const queryCount = 128;
    const queries = queryRows(queryCount);
    const seeds = Uint32Array.from(
      Array.from({length: queryCount}, (_, row) => {
        const horizontal = queries[row * 2] * (side - 1);
        const vertical = queries[row * 2 + 1] * (side - 1);
        return (
          (Math.floor(vertical) * (side - 1) + Math.floor(horizontal)) * 2 +
          (horizontal % 1 < vertical % 1 ? 1 : 0)
        );
      })
    );
    const data = Float32Array.from(
      Array.from({length: side * side}, (_, row) =>
        response(topology.positions[row * 2], topology.positions[row * 2 + 1])
      )
    );
    cases.push({
      name: `sibson-indexed-${side * side}-sites`,
      shape: {
        sampleCount: side * side,
        queryCount,
        maximumNeighborCount: 64,
        maximumVertexCount: 64,
        triangleCount: topology.triangles.length / 6
      },
      correctness:
        'GPU cavity result and changed values reproduce affine field; supplied indexed topology outside timings',
      construct(scope) {
        scope.graph.add(
          new GPUNaturalNeighbor({
            positions: scope.input('positions', topology.positions, 'float32x2'),
            values: scope.input(
              'values',
              data,
              'float32',
              Float32Array.from(data, value => value + 3)
            ),
            queries: scope.input('queries', queries, 'float32x2'),
            triangles: scope.input('triangles', topology.triangles, 'uint32'),
            queryTriangles: scope.input('seeds', seeds, 'uint32'),
            maximumNeighborCount: 64,
            output: {
              values: scope.output('values-output', queryCount, 'float32x4'),
              status: scope.output('status', queryCount, 'uint32')
            }
          })
        );
        return (actual, updated) => {
          expect(actual.status.every(value => value === 0)).toBe(true);
          for (let query = 0; query < queryCount; query++) {
            expect(actual['values-output'][query * 4]).toBeCloseTo(
              response(queries[query * 2], queries[query * 2 + 1]) + (updated ? 3 : 0),
              3
            );
          }
        };
      }
    });
  }
  for (const side of [32, 128]) {
    const positions = gridTopology(side).positions;
    const queryCount = 128;
    const queries = queryRows(queryCount);
    const data = Float32Array.from(
      Array.from({length: side * side}, (_, row) =>
        response(positions[row * 2], positions[row * 2 + 1])
      )
    );
    cases.push({
      name: `rbf-resident-grid-${side * side}-sites`,
      shape: {sampleCount: side * side, queryCount, neighborCount: 32, degree: 2, gridSide: 32},
      correctness:
        'Includes GPU grid construction/CSR neighbor search plus degree2 smoothed solve; affine reproduction and updates',
      construct(scope) {
        const positionsView = scope.input('positions', positions, 'float32x2');
        const queriesView = scope.input('queries', queries, 'float32x2');
        const offsets = scope.output('offsets', queryCount + 1, 'uint32');
        const neighbors = scope.output('neighbors', queryCount * 32, 'uint32');
        scope.graph.add(
          new GPUNeighborSearch({
            positions: positionsView,
            queryPositions: queriesView,
            mode: 'knn',
            k: 32,
            gridSize: [32, 32],
            parameters: scope.input(
              'parameters',
              getGPUNeighborSearchParameterValues({bounds: [0, 0, 1, 1]}),
              'float32'
            ),
            output: {
              offsets,
              neighbors,
              weights: scope.output('neighborWeights', queryCount * 32, 'float32'),
              status: {overflow: scope.output('overflow', 1, 'uint32')}
            }
          })
        );
        scope.graph.add(
          new GPULocalRBF({
            positions: positionsView,
            queries: queriesView,
            values: scope.input(
              'values',
              data,
              'float32',
              Float32Array.from(data, value => value + 3)
            ),
            neighborhoods: neighbors,
            neighborhoodOffsets: offsets,
            neighborCount: 32,
            degree: 2,
            kernel: 'thin-plate-spline',
            smoothing: 0.1,
            output: {
              values: scope.output('values-output', queryCount, 'float32x4'),
              status: scope.output('status', queryCount, 'uint32')
            }
          })
        );
        return (actual, updated) => {
          expect(actual.overflow).toEqual([0]);
          expect(actual.status.every(value => value === 0)).toBe(true);
          for (let query = 0; query < queryCount; query++) {
            expect(actual['values-output'][query * 4]).toBeCloseTo(
              response(queries[query * 2], queries[query * 2 + 1]) + (updated ? 3 : 0),
              3
            );
          }
        };
      }
    });
  }
  for (const count of [32, 128]) {
    const positions = Float32Array.from(
      Array.from({length: count}, (_, row) => [
        (row % 16) / 15,
        Math.floor(row / 16) / Math.max(count / 16 - 1, 1)
      ]).flat()
    );
    const queries = queryRows(count);
    const reference = calculateConditionalSimulationReference(
      positions,
      Float32Array.from(Array.from({length: count}, () => [2, 0.1, 0, 1]).flat()),
      queries,
      {model: 'exponential', range: 1, sill: 1},
      2
    );
    const covariance = (first: number, second: number) =>
      Math.exp(
        -3 *
          Math.hypot(
            positions[first * 2] - positions[second * 2],
            positions[first * 2 + 1] - positions[second * 2 + 1]
          )
      );
    const matrix = Array.from({length: count}, (_, row) =>
      Array.from(
        {length: count},
        (_, column) => covariance(row, column) + (row === column ? 0.1 : 0)
      )
    );
    const coefficients = solveBenchmarkSystem(matrix, Array(count).fill(1));
    const updatedMeans = Array.from(
      {length: count},
      (_, query) =>
        2 +
        coefficients.reduce(
          (sum, value, row) =>
            sum +
            value *
              Math.exp(
                -3 *
                  Math.hypot(
                    positions[row * 2] - queries[query * 2],
                    positions[row * 2 + 1] - queries[query * 2 + 1]
                  )
              ),
          0
        )
    );
    cases.push({
      name: `simulation-shared-factor-${count}-plus-${count}`,
      shape: {
        sampleCount: count,
        queryCount: count,
        replicateCount: 128,
        jointMatrixCount: count * 2
      },
      correctness:
        'Shared factor once per encode; independent f64 conditional means and Schur-complement variances; changed observation replay',
      construct(scope) {
        scope.graph.add(
          new GPUConditionalGaussianSimulation({
            positions: scope.input('positions', positions, 'float32x2'),
            observations: scope.input(
              'observations',
              Float32Array.from(Array.from({length: count}, () => [2, 0.1, 0, 1]).flat()),
              'float32x4',
              Float32Array.from(Array.from({length: count}, () => [3, 0.1, 0, 1]).flat())
            ),
            queries: scope.input('queries', queries, 'float32x2'),
            covariance: {model: 'exponential', range: 1, sill: 1},
            mean: 2,
            seed: 19147,
            replicateCount: 128,
            output: {
              values: scope.output('values', count * 128, 'float32'),
              moments: scope.output('moments', count, 'float32x2'),
              status: scope.output('status', 128, 'uint32')
            }
          })
        );
        return (actual, updated) => {
          expect(actual.status.every(value => value === 0)).toBe(true);
          expect(actual.values.every(Number.isFinite)).toBe(true);
          for (let query = 0; query < count; query++) {
            expect(actual.moments[query * 2]).toBeCloseTo(updated ? updatedMeans[query] : 2, 3);
            expect(actual.moments[query * 2 + 1]).toBeGreaterThanOrEqual(0);
            expect(actual.moments[query * 2 + 1]).toBeCloseTo(
              reference.covariance[query][query],
              4
            );
          }
        };
      }
    });
  }
  return cases;
}
