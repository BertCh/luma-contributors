// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUSpatialPearson,
  GPUMultivariateLocalGeary,
  GPUMultivariateLocalGearyPermutationTest
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {computeLocalMoranOracle, computeHotSpotOracle} from './spatial-autocorrelation-oracle';
import {computeGlobalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {computeGlobalSpatialStatisticsOracle} from '../global-spatial-statistics/global-spatial-statistics-oracle';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {computeQuantileLISAReference} from '../recipes/quantile-lisa-reference';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  return {device, graph, view};
}

it('review2 R10 sparse conditional twins read the selected population only linearly', () => {
  for (const compute of [computeLocalMoranOracle, computeHotSpotOracle]) {
    const rows = 1000;
    const original = Float32Array.from({length: rows}, (_, row) => row % 7);
    let reads = 0;
    const values = new Proxy(original, {
      get(target, key) {
        if (typeof key === 'string' && /^\d+$/.test(key)) reads++;
        const value = Reflect.get(target, key, target);
        return typeof value === 'function' ? value.bind(target) : value;
      }
    });
    compute({
      values,
      selfWeight: 0,
      weights: {
        offsets: Uint32Array.from({length: rows + 1}, (_, row) => row),
        neighbors: Uint32Array.from({length: rows}, (_, row) => (row + 1) % rows),
        weights: new Float32Array(rows).fill(1)
      }
    });
    // Direct indexed reads count work independently of timing and machine load.
    expect(reads).toBeLessThan(100 * rows);
  }
  const input = {
    values: Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0, 99),
    mask: Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 1, 0),
    weights: {
      offsets: Uint32Array.from({length: 10}, (_, row) => row),
      neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 0, 0),
      weights: new Float32Array(9).fill(1)
    }
  };
  for (const result of [
    computeLocalMoranOracle(input),
    computeHotSpotOracle({...input, selfWeight: 0})
  ]) {
    expect(result.zScores[0]).toBeNaN();
    expect(result.pValues[0]).toBeNaN();
  }
  expect(computeLocalMoranOracle(input).localI[0]).toBe(-7 / 8);
});

it('review2 R8 scaled invariant and near-invariant complete graphs retain their true variance', () => {
  for (const perturbed of [false, true]) {
    const rows = perturbed ? 9 : 8,
      weights = new Float32Array(rows * (rows - 1)).fill(perturbed ? 1 : 0.1);
    if (perturbed) weights[0] = 1 + 2 ** -23;
    const result = computeGlobalSpatialStatisticsOracle({
      values: Float32Array.from({length: rows}, (_, row) => row),
      weights: {
        offsets: Uint32Array.from({length: rows + 1}, (_, row) => row * (rows - 1)),
        neighbors: Uint32Array.from(
          Array.from({length: rows}, (_, row) =>
            Array.from({length: rows}, (_, other) => other).filter(other => other !== row)
          ).flat()
        ),
        weights
      }
    });
    if (!perturbed) {
      expect.soft(result.geary.varianceNormality).toBe(0);
      expect.soft(result.geary.zNormality).toBeNaN();
      expect.soft(result.geary.pNormality).toBeNaN();
      expect.soft(result.geary.varianceRandomization).toBe(0);
      expect.soft(result.geary.zRandomization).toBeNaN();
      expect.soft(result.geary.pRandomization).toBeNaN();
    } else {
      expect(Number.isFinite(result.geary.varianceRandomization)).toBe(true);
      expect(
        Math.abs(result.geary.varianceRandomization / 3.0702463789201337e-18 - 1)
      ).toBeLessThan(1e-8);
      expect(result.geary.zRandomization).toBeCloseTo(-0.8819171037, 8);
      expect(result.geary.pRandomization).toBeCloseTo(0.37782163710006383, 7);
      expect(Number.isFinite(result.geary.varianceNormality)).toBe(true);
      expect(Math.abs(result.geary.varianceNormality / 3.8378079736501665e-18 - 1)).toBeLessThan(
        1e-8
      );
      expect(result.geary.zNormality).toBeCloseTo(-0.7888106377466154, 8);
      expect(result.geary.pNormality).toBeCloseTo(0.4302226911321914, 7);
    }
  }
});

it('review2 R8 stable pair passes lower within binding limits in both modes and layouts', () => {
  const {device, graph, view} = setup();
  try {
    for (const mode of ['reference', 'tiled'] as const)
      for (const chunked of [false, true]) {
        const csr = (rows: number) => ({
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', rows * 7),
          weights: view('float32', rows * 7)
        });
        const weights = chunked
          ? {
              rowCount: 8,
              chunks: [
                {rowStart: 0, rowCount: 3, weights: csr(3)},
                {rowStart: 3, rowCount: 5, weights: csr(5)}
              ]
            }
          : csr(8);
        const nodes = new GPUGlobalSpatialStatistics({
          id: `review-geary-${mode}-${chunked}`,
          mode,
          weights,
          values: view('float32', 8),
          statistics: ['geary'],
          output: {summary: view('float32', 52)}
        }).getCommandNodes(graph);
        expect(nodes.some(node => node.id.endsWith('-geary-deviations-finalize'))).toBe(true);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
          const functions = Array.from(
            kernel.source.matchAll(/\bfn\s+(\w+)\s*\(/g),
            match => match[1]
          );
          expect(new Set(functions).size, kernel.id).toBe(functions.length);
        }
      }
  } finally {
    device.destroy();
  }
});

it('review2 R5a constant labels have zero squared Gamma but undefined Geary scale', () => {
  const values = new Float32Array(8).fill(2);
  const weights = {
    offsets: Uint32Array.from({length: 9}, (_, row) => row * 7),
    neighbors: Uint32Array.from(
      Array.from({length: 8}, (_, row) =>
        Array.from({length: 8}, (_, other) => other).filter(other => other !== row)
      ).flat()
    ),
    weights: new Float32Array(56).fill(1)
  };
  expect(Array.from(values, value => (value - 2) ** 2).reduce((sum, value) => sum + value, 0)).toBe(
    0
  );
  expect(
    computeGlobalPermutationOracle({
      values,
      weights,
      statistic: 'geary',
      seed: 0,
      permutationCount: 9
    }).observed
  ).toBeNaN();
});

it('review2 R7 tiled estimates cover observed row tiles and reject 16385 single-row tiles', () => {
  const {device, graph, view} = setup();
  const props = (rows: number) => ({
    strategy: 'tiled' as const,
    variableCount: 1,
    permutationCount: 1,
    rowTileSize: 1,
    permutationTileSize: 1,
    values: view('float32', rows),
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows),
      weights: view('float32', rows)
    },
    output: {
      statistic: view('float32', rows),
      pseudoPValues: view('float32', rows),
      classes: view('uint32', rows)
    }
  });
  try {
    const accepted = new GPUMultivariateLocalGearyPermutationTest(props(320));
    expect
      .soft(accepted.getCommandNodes(graph).length)
      .toBeLessThanOrEqual(accepted.estimatedCommandNodeCount);
    expect(() => new GPUMultivariateLocalGearyPermutationTest(props(16385))).toThrow(
      /estimated command node count/
    );
  } finally {
    device.destroy();
  }
});

it('review2 R4 equal weights over all donors make conditional inference undefined', () => {
  const input = {
    values: Float32Array.from([0, 1, 2, 3, 4, 5]),
    rowStandardized: true,
    weights: {
      offsets: Uint32Array.from({length: 7}, (_, row) => row * 5),
      neighbors: Uint32Array.from(
        Array.from({length: 6}, (_, row) =>
          Array.from({length: 6}, (_, other) => other).filter(other => other !== row)
        ).flat()
      ),
      weights: new Float32Array(30).fill(0.2)
    }
  };
  for (const result of [
    computeLocalMoranOracle(input),
    computeHotSpotOracle({...input, selfWeight: 0})
  ]) {
    expect(result.zScores.every(Number.isNaN)).toBe(true);
    expect(result.pValues.every(Number.isNaN)).toBe(true);
  }
  expect(computeLocalMoranOracle(input).localI[0]).toBeCloseTo(-5 / 14, 14);
});

it('review2 R1 balanced four-neighbor LOSH must preserve addition errors', () => {
  const weight = Math.fround(0.1);
  let sum = 0;
  for (const term of [1, weight, -1, -weight]) sum = Math.fround(sum + term);
  expect(sum).toBe(2.2351741790771484e-8);
  const {device, graph, view} = setup();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPULocalSpatialHeteroscedasticity({
        values: view('float32', 8),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 11),
          weights: view('float32', 11)
        },
        output: {statistic: view('float32', 8), status: view('uint32', 8)}
      }).getCommandNodes(graph)
    );
    expect(
      sources
        .find(source => source.id.endsWith('-residuals'))!
        .source.includes('getAdditionRoundingError(weighted, term, next)')
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R1 harmonic LOSH field must not normalize weighted-mean rounding noise', () => {
  const weight = Math.fround(0.9);
  const oldMean = Math.fround(
    Math.fround(Math.fround(weight * 2) + Math.fround(weight * 4)) / Math.fround(2 * weight)
  );
  expect(oldMean).toBe(2.999999761581421);
  expect(Math.fround(weight * (2 - 3)) + Math.fround(weight * (4 - 3))).toBe(0);
  const {device, graph, view} = setup();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPULocalSpatialHeteroscedasticity({
        values: view('float32', 8),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 9),
          weights: view('float32', 9)
        },
        output: {statistic: view('float32', 8), status: view('uint32', 8)}
      }).getCommandNodes(graph)
    );
    expect(sources.find(source => source.id.endsWith('-residuals'))!.source).toContain(
      'getRoundedDifference(values[valuesOffset + neighbor], focus, runtimeZero)'
    );
  } finally {
    device.destroy();
  }
});

it('review2 R3a inference distances retain distinct original donors after moment centering', () => {
  const distance = 2 ** -26;
  expect(Math.fround(distance - 1)).toBe(-1);
  expect(Math.fround(0 - 1)).toBe(-1);
  for (const strategy of ['exact', 'tiled'] as const) {
    const {device, graph, view} = setup();
    try {
      const sources = getGPUCommandNodeKernelSources(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          variableCount: 1,
          permutationCount: 19,
          tail: 'esda-min-tail',
          values: view('float32', 8),
          weights: {
            offsets: view('uint32', 9),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          output: {
            statistic: view('float32', 8),
            pseudoPValues: view('float32', 8),
            classes: view('uint32', 8)
          }
        }).getCommandNodes(graph)
      );
      expect(
        sources.some(source => source.source.includes('getRoundedDifference(standardized['))
      ).toBe(true);
    } finally {
      device.destroy();
    }
  }
});

it('R9 integer Hazen ranks survive quantile selection without a probability round trip', () => {
  const reference = computeQuantileLISAReference(
    Float32Array.from({length: 25}, (_, row) => row),
    10,
    4
  );
  expect(reference.boundaries).toEqual([7, 9.5]);
  expect(Array.from(reference.membership).flatMap((member, row) => (member ? [row] : []))).toEqual([
    7, 8, 9
  ]);
  const rank = Math.fround(Math.fround(25 * Math.fround(3 / 10)) - 0.5);
  expect(Math.fround(Math.fround(rank / 24) * 24)).toBe(7.000000476837158);
  const {device, graph, view} = setup();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUQuantileLISA({
        alternative: 'geoda-legacy',
        values: view('float32', 25),
        mask: view('uint32', 25),
        quantileParameters: view('uint32', 2),
        parameters: view('uint32', 4),
        maximumPermutations: 9,
        weights: {
          offsets: view('uint32', 26),
          neighbors: view('uint32', 25),
          weights: view('float32', 25)
        },
        output: {
          membership: view('uint32', 25),
          boundaries: view('float32', 2),
          validCount: view('uint32', 1),
          classes: view('uint32', 25),
          invalidQuantile: view('uint32', 1),
          exceedances: view('uint32', 25),
          pseudoPValues: view('float32', 25),
          status: {overflow: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    );
    expect(sources.find(source => source.id.endsWith('-boundary-parameters'))!.source).toContain(
      'divideUint64ByUint32'
    );
    for (const source of sources)
      expect(getStorageBindingCount(source.source), source.id).toBeLessThanOrEqual(8);
  } finally {
    device.destroy();
  }
});

it('R8 complete unit graphs have exactly zero Geary null variance at eight and nine rows', () => {
  for (const rows of [8, 9]) {
    const result = computeGlobalSpatialStatisticsOracle({
      values: Float32Array.from({length: rows}, (_, row) => row),
      weights: {
        offsets: Uint32Array.from({length: rows + 1}, (_, row) => row * (rows - 1)),
        neighbors: Uint32Array.from(
          Array.from({length: rows}, (_, row) =>
            Array.from({length: rows}, (_, neighbor) => neighbor).filter(
              neighbor => neighbor !== row
            )
          ).flat()
        ),
        weights: new Float32Array(rows * (rows - 1)).fill(1)
      }
    });
    expect(result.geary.statistic).toBe(1);
    expect(result.geary.varianceRandomization).toBe(0);
    expect(result.geary.zRandomization).toBeNaN();
    expect(result.geary.pRandomization).toBeNaN();
  }
});

it('R7 tiled multivariate inference enforces the command-node budget at construction', () => {
  const {device, graph, view} = setup();
  try {
    const props = {
      strategy: 'tiled' as const,
      variableCount: 1,
      permutationCount: 129,
      rowTileSize: 1,
      permutationTileSize: 1,
      values: view('float32', 256),
      weights: {
        offsets: view('uint32', 257),
        neighbors: view('uint32', 256),
        weights: view('float32', 256)
      },
      output: {
        statistic: view('float32', 256),
        pseudoPValues: view('float32', 256),
        classes: view('uint32', 256)
      }
    };
    expect(() => new GPUMultivariateLocalGearyPermutationTest(props)).toThrow(
      /estimated command node count/
    );
    const accepted = new GPUMultivariateLocalGearyPermutationTest({
      ...props,
      rowTileSize: 16,
      permutationTileSize: 16
    });
    expect(Reflect.get(accepted, 'estimatedCommandNodeCount')).toBeGreaterThanOrEqual(16 * 9);
    expect(accepted.getCommandNodes(graph).length).toBeLessThanOrEqual(
      Reflect.get(accepted, 'estimatedCommandNodeCount')
    );
  } finally {
    device.destroy();
  }
});

it('R6 all-tied draws must determine direction from zero differences', () => {
  let average = 0;
  for (let replicate = 0; replicate < 10; replicate++)
    average = Math.fround(average + Math.fround(4 / 10));
  expect(average).toBe(4.000000476837158);
  expect(4 < average).toBe(true);
  for (const strategy of ['exact', 'tiled'] as const) {
    const {device, graph, view} = setup();
    try {
      const sources = getGPUCommandNodeKernelSources(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          permutationCount: 10,
          variableCount: 1,
          values: view('float32', 8),
          weights: {
            offsets: view('uint32', 9),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          output: {
            statistic: view('float32', 8),
            pseudoPValues: view('float32', 8),
            classes: view('uint32', 8)
          }
        }).getCommandNodes(graph)
      );
      expect(sources.some(source => source.source.includes('differenceAverage'))).toBe(true);
    } finally {
      device.destroy();
    }
  }
});

it('R5 masked Gamma and restandardized Geary can order the same labels oppositely', () => {
  const gamma: number[] = [],
    geary: number[] = [];
  for (const labels of [
    [0, 1, 2, 4, 99],
    [0, 4, 1, 2, 99]
  ]) {
    gamma.push(
      (0.25 * (labels[0] - labels[1]) ** 2 +
        (labels[1] - labels[2]) ** 2 +
        (labels[2] - labels[3]) ** 2 +
        (labels[3] - labels[0]) ** 2) /
        3.25
    );
    geary.push(
      computeGlobalPermutationOracle({
        values: Float32Array.from(labels),
        mask: Uint32Array.of(1, 1, 1, 1, 0),
        weights: {
          offsets: Uint32Array.of(0, 2, 3, 4, 5, 5),
          neighbors: Uint32Array.of(1, 4, 2, 3, 0),
          weights: Float32Array.of(0.25, 0.75, 1, 1, 1)
        },
        rowStandardized: true,
        statistic: 'geary',
        seed: 1,
        permutationCount: 1
      }).observed
    );
  }
  expect(gamma).toEqual([85 / 13, 72 / 13]);
  for (const [index, expected] of [33 / 35, 9 / 7].entries()) {
    expect(Number.isFinite(geary[index])).toBe(true);
    expect(Math.abs(geary[index] - expected)).toBeLessThan(2e-7);
  }
  expect(gamma[0]).toBeGreaterThan(gamma[1]);
  expect(geary[0]).toBeLessThan(geary[1]);
});

it('R4 a constant leave-one-out donor population has undefined z and p', () => {
  const input = {
    values: Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0, 0),
    weights: {
      offsets: Uint32Array.from({length: 10}, (_, row) => row),
      neighbors: Uint32Array.from({length: 9}, (_, row) => (row + 1) % 9),
      weights: new Float32Array(9).fill(1)
    }
  };
  for (const result of [
    computeLocalMoranOracle(input),
    computeHotSpotOracle({...input, selfWeight: 0})
  ]) {
    expect(result.zScores[0]).toBeNaN();
    expect(result.pValues[0]).toBeNaN();
  }
  expect(computeLocalMoranOracle(input).localI[0]).toBeCloseTo(-8 / 9, 14);
});

it('R3 multivariate moments center into stored differences before running updates', () => {
  const distance = 2 ** -23;
  let mean = 0,
    squares = 0;
  for (let row = 0; row < 8; row++) {
    const value = 1 + (row < 4 ? 0 : distance);
    const delta = Math.fround(value - mean);
    mean = Math.fround(mean + Math.fround(delta / (row + 1)));
    squares = Math.fround(squares + Math.fround(delta * Math.fround(value - mean)));
  }
  expect(squares / 8).toBe((distance * distance) / 2);
  const {device, graph, view} = setup();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUMultivariateLocalGeary({
        variableCount: 1,
        values: view('float32', 8),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8)
        },
        output: {statistic: view('float32', 8), status: view('uint32', 8)}
      }).getCommandNodes(graph)
    );
    expect(sources.some(source => source.id.endsWith('-center-values'))).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R2 Pearson refuses exactly constant eligible marginals despite rounded weighted means', () => {
  const values = new Float32Array(256);
  for (let row = 0; row < 8; row++)
    values[row] = Math.fround(Math.fround(0.1) * (row === 7 ? 2 : 1));
  for (let stride = 128; stride; stride >>= 1)
    for (let row = 0; row < stride; row++)
      values[row] = Math.fround(values[row] + values[row + stride]);
  const mean = Math.fround(values[0] / 9);
  expect(mean).toBe(0.09999999403953552);
  expect(Math.fround(0.1) - mean).toBeGreaterThan(0);
  const {device, graph, view} = setup();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUSpatialPearson({
        values: view('float32', 8),
        secondValues: view('float32', 8),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8)
        },
        output: {statistic: view('float32', 1)}
      }).getCommandNodes(graph)
    );
    expect(sources.find(source => source.id.endsWith('-statistic'))!.source).toContain(
      'maximumX[0] > minimumX[0] && maximumY[0] > minimumY[0]'
    );
  } finally {
    device.destroy();
  }
});
