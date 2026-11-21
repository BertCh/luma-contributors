// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeGlobalSpatialStatisticsOracle} from './global-spatial-statistics-oracle';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function completeWeights(rows: number, weight: (row: number, neighbor: number) => number) {
  const neighbors = Array.from({length: rows}, (_, row) =>
    Array.from({length: rows}, (_, neighbor) => neighbor).filter(neighbor => neighbor !== row)
  );
  return {
    offsets: Uint32Array.from({length: rows + 1}, (_, row) => row * (rows - 1)),
    neighbors: Uint32Array.from(neighbors.flat()),
    weights: Float32Array.from(
      neighbors.flatMap((entries, row) => entries.map(neighbor => weight(row, neighbor)))
    )
  };
}

it('W5-17 Geary randomization variance is invariant under rescaling tiny perturbed weights', () => {
  // Independent exact Fraction enumeration: for x=0..8, M2=60 and delta=2^-23,
  // each ordered endpoint pair (a,b) has C = [72 + 8*delta*(a-b)^2/120]/[72+delta].
  // Enumerate the 72 pairs (each represents 7! permutations), center at their exact mean,
  // and average squared deviations: 28/9119789275624243225 = 3.0702463789201337e-18.
  for (const scale of [1, 2 ** -60]) {
    const result = computeGlobalSpatialStatisticsOracle({
      values: Float32Array.from({length: 9}, (_, row) => row),
      weights: completeWeights(
        9,
        (row, neighbor) => scale * (row === 0 && neighbor === 1 ? 1 + 2 ** -23 : 1)
      )
    });
    const actual = result.geary.varianceRandomization;
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeGreaterThan(0);
    expect(Math.abs(actual / 3.0702463789201337e-18 - 1)).toBeLessThan(1e-7);
  }
});

it('W5-17 Geary publishes a positive quadratic numerator even when C minus one rounds to minus one', () => {
  // Labels [0,0,1,1] have M2=1. Eight directed cross-label edges weigh epsilon=2^-60;
  // S0=2+10*epsilon, so C=3*8*epsilon/(2*S0)=5.204170427930421e-18.
  const result = computeGlobalSpatialStatisticsOracle({
    values: Float32Array.of(0, 0, 1, 1),
    weights: completeWeights(4, (row, neighbor) => (row < 2 && neighbor < 2 ? 1 : 2 ** -60))
  });
  const actual = result.geary.statistic;
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBeGreaterThan(0);
  expect(Math.abs(actual / 5.204170427930421e-18 - 1)).toBeLessThan(1e-12);
});

for (const mode of ['tiled', 'reference'] as const)
  it(`W5-17 ${mode} Geary kernels parse and fit eight storage bindings`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `view-${serial++}`, format, length);
      const sources = getGPUCommandNodeKernelSources(
        new GPUGlobalSpatialStatistics({
          mode,
          statistics: ['geary'],
          values: view('float32', 9),
          weights: {
            offsets: view('uint32', 10),
            neighbors: view('uint32', 72),
            weights: view('float32', 72)
          },
          output: {summary: view('float32', 52)}
        }).getCommandNodes(graph)
      );
      for (const kernel of sources) {
        const reflection = new WgslReflect(kernel.source);
        const used = new Set(
          reflection.entry.compute.flatMap(entry => entry.resources.map(resource => resource.name))
        );
        expect(
          reflection.storage.every(binding => used.has(binding.name)),
          kernel.id
        ).toBe(true);
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      }
    } finally {
      device.destroy();
    }
  });

it('B2 complete uniform weights publish exactly C=1 despite uneven labels', () => {
  for (const rows of [5, 8]) {
    const result = computeGlobalSpatialStatisticsOracle({
      values: Float32Array.from({length: rows}, (_, row) => Number(row === rows - 1)),
      weights: completeWeights(rows, () => 1)
    });
    // Every permutation has Q=2*n*M2 and S0=n*(n-1), hence C=1 exactly.
    expect(result.geary.statistic).toBe(1);
    expect(result.geary.varianceRandomization).toBe(0);
    expect(result.geary.varianceNormality).toBe(0);
    expect(result.geary.zRandomization).toBeNaN();
  }
});

it('B3 Geary on complete weights of scale 2^-112 matches exact enumeration', () => {
  // Exact Fraction enumeration of all 4! permutations
  // (.typecheck/r19/scratch/spatial-stats-w5-local-global-and-rate-statistics/geary-b3-reference.py):
  // C = 12/13, randomization variance 1/169, z = -1. The GPU's float32 preliminary C is 0/0
  // here; geary-small-weights.spec.ts pins that the stable pass still runs.
  const result = computeGlobalSpatialStatisticsOracle({
    values: Float32Array.of(1, 1, 1, 1 + 2 ** -23),
    weights: completeWeights(
      4,
      (row, neighbor) => (row === 0 && neighbor === 1 ? 2 : 1) * 2 ** -112
    )
  });
  for (const [actual, expected] of [
    [result.geary.statistic, 12 / 13],
    [result.geary.varianceRandomization, 1 / 169],
    [result.geary.zRandomization, -1]
  ] as const) {
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-9);
  }
});
