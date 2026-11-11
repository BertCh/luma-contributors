// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUCompositeScore,
  type GPUCompositeScoreProps
} from '../../../src/dataframe-analysis/composite-indicators/gpu-composite-score';
import {
  getGPUCompositeScoreParameterValues,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
} from '../../../src/dataframe-analysis/composite-indicators/composite-score-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {computeCompositeScoreOnCPU} from './composite-score-oracle';

/**
 * Largest eigenpair of a symmetric positive semi-definite matrix by a long float64 power
 * iteration from a start with unequal components. A different algorithm from the Jacobi
 * decomposition of the twin, used here only on matrices whose first eigenvalue is well
 * separated, where 5,000 steps converge to rounding.
 */
function getLargestEigenpairByPowerIteration(
  matrix: Float64Array,
  d: number
): {value: number; vector: Float64Array} {
  let vector = Float64Array.from({length: d}, (_, index) => 1 + 0.37 * index);
  let value = 0;
  for (let iteration = 0; iteration < 5000; iteration++) {
    const product = new Float64Array(d);
    for (let row = 0; row < d; row++) {
      for (let column = 0; column < d; column++) {
        product[row] += matrix[row * d + column] * vector[column];
      }
    }
    value = Math.hypot(...product) / Math.hypot(...vector);
    const norm = Math.hypot(...product);
    vector = product.map(component => component / norm);
  }
  return {value, vector};
}

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCompositeScoreProps> = {}
): GPUCompositeScoreProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    indicators: view('float32', 30),
    indicatorCount: 3,
    parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
    output: {score: view('float32', 10)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCompositeScoreProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCompositeScore(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUCompositeScoreParameterValues packs modes, weights and directions', () => {
  const values = getGPUCompositeScoreParameterValues({
    scaler: 'rank',
    aggregation: 'weighted-geometric-mean',
    weights: [2, 0.5],
    directions: [1, -3],
    epsilon: 0.25
  });
  expect(values.length).toBe(GPU_COMPOSITE_SCORE_PARAMETER_LENGTH);
  expect(Array.from(values.subarray(0, 6))).toEqual([2, 1, 0.25, 0, 2, 0.5]);
  expect(values[6]).toBe(0);
  expect(Array.from(values.subarray(20, 23))).toEqual([1, -1, 1]);
  expect(() => getGPUCompositeScoreParameterValues({weights: new Array(17).fill(1)})).toThrow(
    /at most 16/
  );
  expect(() => getGPUCompositeScoreParameterValues({weights: [NaN]})).toThrow(/finite/);
  // B-26: a weighted geometric mean needs w >= 0 (the kernel used to drop negative weights silently,
  // while the weighted sum penalizes them); directions flip an indicator instead.
  expect(() =>
    getGPUCompositeScoreParameterValues({
      aggregation: 'weighted-geometric-mean',
      weights: [1, -0.5]
    })
  ).toThrow(/non-negative/);
  expect(
    getGPUCompositeScoreParameterValues({aggregation: 'weighted-sum', weights: [1, -0.5]})[5]
  ).toBe(-0.5);
  expect(() => getGPUCompositeScoreParameterValues({weights: [1], epsilon: 0})).toThrow(/epsilon/);
  expect(() => getGPUCompositeScoreParameterValues({weights: [1]}, new Float32Array(4))).toThrow(
    /hold/
  );
});

it('GPUCompositeScore validates props', () => {
  expectThrows(() => ({indicatorCount: 0}), /indicatorCount/);
  expectThrows(() => ({indicatorCount: 17}), /indicatorCount/);
  expectThrows(() => ({indicatorCount: 4}), /multiple/);
  expectThrows(
    graph => ({parameters: createTransientView(graph, `p-${serial++}`, 'float32', 4)}),
    /parameters/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, `m-${serial++}`, 'uint32', 9)}),
    /mask length/
  );
  expectThrows(
    graph => ({output: {score: createTransientView(graph, `s-${serial++}`, 'float32', 9)}}),
    /score/
  );
  expectThrows(
    graph => ({
      output: {
        score: createTransientView(graph, `s-${serial++}`, 'float32', 10),
        loadings: createTransientView(graph, `l-${serial++}`, 'float32', 3)
      }
    }),
    /enablePrincipalComponent/
  );
  expectThrows(
    graph => ({
      output: {
        score: createTransientView(graph, `s-${serial++}`, 'float32', 10),
        scaled: createTransientView(graph, `c-${serial++}`, 'float32', 29)
      }
    }),
    /scaled/
  );
});

it('GPUCompositeScore emits deterministic node IDs and compiles optional passes only when enabled', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const basic = new GPUCompositeScore(createProps(graph, {id: 'basic'}));
  const basicIds = basic.getCommandNodes(graph).map(node => node.id);
  expect(basicIds).toEqual([
    'basic-validate',
    'basic-tile-sums',
    'basic-means',
    'basic-tile-moments',
    'basic-moments',
    'basic-score'
  ]);
  const full = new GPUCompositeScore(
    createProps(graph, {id: 'full', enableRank: true, enablePrincipalComponent: true})
  );
  const fullIds = full.getCommandNodes(graph).map(node => node.id);
  expect(fullIds).toContain('full-principal-component');
  expect(fullIds).toContain('full-rank-keys-0');
  expect(fullIds).toContain('full-rank-scatter-2');
  expect(fullIds[fullIds.length - 1]).toBe('full-score');
  expect(new Set(fullIds).size).toBe(fullIds.length);
  device.destroy();
});

it('GPUCompositeScore rank pipelines do not grow with the indicator count', () => {
  const build = (indicatorCount: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const score = new GPUCompositeScore(
      createProps(graph, {
        id: `rank${indicatorCount}`,
        enableRank: true,
        indicatorCount,
        indicators: createTransientView(
          graph,
          `rank-indicators-${serial++}`,
          'float32',
          10 * indicatorCount
        )
      })
    );
    const nodes = score.getCommandNodes(graph);
    device.destroy();
    return nodes;
  };
  const two = build(2);
  const three = build(3);
  expect(getGPUCommandNodePipelineKeys(two).size).toBe(getGPUCommandNodePipelineKeys(three).size);
  const ids = three.map(node => node.id);
  expect(ids).toContain('rank3-rank-reset');
  for (const column of [0, 1, 2]) {
    expect(ids).toContain(`rank3-rank-advance-${column}`);
  }
  for (const {id, source} of getGPUCommandNodeKernelSources(three)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(source, id).not.toContain('RANK_COLUMN');
  }
});

it('GPUCompositeScore second-stage reductions run in 256-lane workgroups', () => {
  // 300 * 256 + 7 rows give 301 tiles (the strided loop runs twice); 100 rows give one tile.
  for (const rowCount of [300 * 256 + 7, 100]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const id = `two-stage-${rowCount}`;
    const score = new GPUCompositeScore(
      createProps(graph, {
        id,
        enablePrincipalComponent: true,
        indicators: createTransientView(graph, `ind-${serial++}`, 'float32', rowCount * 3),
        output: {score: createTransientView(graph, `out-${serial++}`, 'float32', rowCount)}
      })
    );
    const sources = getGPUCommandNodeKernelSources(score.getCommandNodes(graph));
    for (const variant of ['means', 'moments']) {
      const kernel = sources.find(({id: kernelId}) => kernelId.includes(`${id}-${variant}`));
      expect(kernel, variant).toBeDefined();
      const source = kernel!.source;
      expect(() => new WgslReflect(source), variant).not.toThrow();
      expect(source, variant).toContain('@workgroup_size(256)');
      expect(source, variant).toContain('var<workgroup>');
      expect(source, variant).toContain('workgroupBarrier()');
      // No invocation walks every tile: the loop starts at the lane and strides by 256.
      expect(source, variant).not.toContain('tile = 0u; tile < TILE_COUNT');
    }
    device.destroy();
  }
});

it('computeCompositeScoreOnCPU first component matches an independent power iteration', () => {
  const rows = 200;
  const d = 4;
  const indicators = new Float32Array(rows * d);
  let state = 7;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296 - 0.5;
  };
  for (let row = 0; row < rows; row++) {
    const latent = random();
    indicators[row * d] = latent + 0.1 * random();
    indicators[row * d + 1] = 2 * latent + 0.3 * random();
    indicators[row * d + 2] = -latent + 0.5 * random();
    indicators[row * d + 3] = random();
  }
  const result = computeCompositeScoreOnCPU({
    indicators,
    indicatorCount: d,
    settings: {aggregation: 'principal-component', weights: []}
  });
  // Correlation matrix of the raw columns for the power-iteration reference.
  const statistics = result.columnStatistics;
  const correlation = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      let sum = 0;
      for (let row = 0; row < rows; row++) {
        sum +=
          ((indicators[row * d + a] - statistics[a * 4 + 2]) / statistics[a * 4 + 3]) *
          ((indicators[row * d + b] - statistics[b * 4 + 2]) / statistics[b * 4 + 3]);
      }
      correlation[a * d + b] = sum / rows;
    }
  }
  const reference = getLargestEigenpairByPowerIteration(correlation, d);
  expect(result.principalComponent[0]).toBeCloseTo(reference.value, 8);
  const sign = Math.sign(reference.vector.reduce((sum, value) => sum + value, 0));
  for (let c = 0; c < d; c++) {
    expect(result.loadings[c]).toBeCloseTo(sign * reference.vector[c], 6);
  }
  expect(result.principalComponent[1]).toBeCloseTo(reference.value / d, 8);
});

it('computeCompositeScoreOnCPU returns the largest eigenpair when indicators are negatively correlated', () => {
  // Two indicators: the correlation matrix [[1, r], [r, 1]] has eigenvalues 1 + r and 1 - r with
  // eigenvectors (1, 1) and (1, -1). Ten rows of zeros and ones, five ones in each column, four
  // rows agreeing and six disagreeing: the mean product is 0.2, the covariance 0.2 - 0.25 = -0.05,
  // each variance 0.25, so r = -0.2. The first component is (1, -1) / sqrt(2), eigenvalue 1.2.
  const pair = Float32Array.from([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]);
  const pairResult = computeCompositeScoreOnCPU({
    indicators: pair,
    indicatorCount: 2,
    settings: {aggregation: 'principal-component', weights: []}
  });
  expect(pairResult.principalComponent[0]).toBeCloseTo(1.2, 10);
  expect(pairResult.principalComponent[1]).toBeCloseTo(0.6, 10);
  expect(pairResult.loadings[0]).toBeCloseTo(Math.SQRT1_2, 10);
  expect(pairResult.loadings[1]).toBeCloseTo(-Math.SQRT1_2, 10);

  // Three indicators, one negative pair and an independent column, against the power iteration.
  const rows = 300;
  const d = 3;
  const indicators = new Float32Array(rows * d);
  let state = 19;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296 - 0.5;
  };
  for (let row = 0; row < rows; row++) {
    const latent = random();
    indicators[row * d] = latent + 0.2 * random();
    indicators[row * d + 1] = -latent + 0.2 * random();
    indicators[row * d + 2] = random();
  }
  const result = computeCompositeScoreOnCPU({
    indicators,
    indicatorCount: d,
    settings: {aggregation: 'principal-component', weights: []}
  });
  const statistics = result.columnStatistics;
  const correlation = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      let sum = 0;
      for (let row = 0; row < rows; row++) {
        sum +=
          ((indicators[row * d + a] - statistics[a * 4 + 2]) / statistics[a * 4 + 3]) *
          ((indicators[row * d + b] - statistics[b * 4 + 2]) / statistics[b * 4 + 3]);
      }
      correlation[a * d + b] = sum / rows;
    }
  }
  const reference = getLargestEigenpairByPowerIteration(correlation, d);
  expect(reference.value).toBeGreaterThan(1.8);
  expect(result.principalComponent[0]).toBeCloseTo(reference.value, 8);
  // The eigenvector is defined up to sign; compare after aligning on the first loading.
  const sign = Math.sign(reference.vector[0]) * Math.sign(result.loadings[0]);
  for (let c = 0; c < d; c++) {
    expect(result.loadings[c]).toBeCloseTo(sign * reference.vector[c], 6);
  }
});

it('computeCompositeScoreOnCPU resolves weak correlations and close eigenvalues', () => {
  // Two balanced columns of -1 and +1 with `agreeing` rows of equal sign: r is exactly
  // (2 * agreeing - rows) / rows, and the first component is (1, sign(r)) / sqrt(2) with
  // eigenvalue 1 + |r|, however small |r| is.
  for (const [rows, agreeing, correlation] of [
    [400, 198, -0.01],
    [4000, 1998, -0.001],
    [400, 202, 0.01]
  ] as const) {
    const pair = new Float32Array(rows * 2);
    for (let row = 0; row < rows; row++) {
      const first = row % 2 === 0 ? 1 : -1;
      pair[row * 2] = first;
      pair[row * 2 + 1] = row < agreeing ? first : -first;
    }
    const result = computeCompositeScoreOnCPU({
      indicators: pair,
      indicatorCount: 2,
      settings: {aggregation: 'principal-component', weights: []}
    });
    expect(result.principalComponent[0], `r = ${correlation}`).toBeCloseTo(
      1 + Math.abs(correlation),
      12
    );
    expect(result.loadings[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(result.loadings[1]).toBeCloseTo(Math.sign(correlation) * Math.SQRT1_2, 12);
  }
  // Three columns with correlation matrix [[1, 0.0008, 0], [0.0008, 1, 0], [0, 0, 1]]:
  // eigenvalues 1.0008, 1 and 0.9992, first component (1, 1, 0) / sqrt(2).
  const triple = new Float32Array(10000 * 3);
  let row = 0;
  for (const [first, second, count] of [
    [1, 1, 1251],
    [-1, -1, 1251],
    [1, -1, 1249],
    [-1, 1, 1249]
  ] as const) {
    for (const third of [1, -1]) {
      for (let repeat = 0; repeat < count; repeat++) {
        triple.set([first, second, third], row * 3);
        row++;
      }
    }
  }
  const tripleResult = computeCompositeScoreOnCPU({
    indicators: triple,
    indicatorCount: 3,
    settings: {aggregation: 'principal-component', weights: []}
  });
  expect(tripleResult.principalComponent[0]).toBeCloseTo(1.0008, 12);
  expect(tripleResult.loadings[0]).toBeCloseTo(Math.SQRT1_2, 10);
  expect(tripleResult.loadings[1]).toBeCloseTo(Math.SQRT1_2, 10);
  expect(tripleResult.loadings[2]).toBeCloseTo(0, 10);
});

it('computeCompositeScoreOnCPU averages tied ranks and honours directions', () => {
  const result = computeCompositeScoreOnCPU({
    indicators: Float32Array.from([1, 10, 2, 20, 2, 30, 3, 40, -0, 50, 0, 60]),
    indicatorCount: 2,
    settings: {scaler: 'rank', weights: [1, 1], directions: [1, -1]}
  });
  // Column 0 sorted: -0, 0, 1, 2, 2, 3 -> ranks 0.5, 0.5, 2, 3.5, 3.5, 5 over n - 1 = 5.
  expect(Array.from({length: 6}, (_, row) => result.scaled[row * 2])).toEqual([
    0.4, 0.7, 0.7, 1, 0.1, 0.1
  ]);
  // Column 1 is reversed: 1 - rank / 5.
  expect(result.scaled[1]).toBe(1);
  expect(result.scaled[11]).toBe(0);
});
