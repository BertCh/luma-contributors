// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCompositeScore} from '../../../src/dataframe-analysis/composite-indicators/gpu-composite-score';
import {
  getGPUCompositeScoreParameterValues,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  type GPUCompositeScoreSettings
} from '../../../src/dataframe-analysis/composite-indicators/composite-score-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeCompositeScoreOnCPU,
  type CompositeScoreOracleResult
} from './composite-score-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Scene = {indicators: Float32Array; indicatorCount: number; mask?: Uint32Array};

type GPUResult = {
  score: number[];
  scaled: number[];
  columnStatistics: number[];
  loadings: number[];
  principalComponentSummary: number[];
};

type Fixture = {
  run(settings: GPUCompositeScoreSettings): Promise<GPUResult>;
  readonly rebuildCount: number;
  destroy(): void;
};

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Correlated indicators with integer-rounded ties, a few NaN rows and a masked block. */
function createScene(rows: number, indicatorCount: number, seed: number): Scene {
  const random = createRandom(seed);
  const indicators = new Float32Array(rows * indicatorCount);
  const mask = new Uint32Array(rows).fill(1);
  for (let row = 0; row < rows; row++) {
    const latent = random() * 10;
    for (let column = 0; column < indicatorCount; column++) {
      const noise = random() * (column + 1);
      const value = column % 2 === 0 ? latent + noise : 20 - latent + noise;
      // Even columns are rounded so the rank scaler sees many ties.
      indicators[row * indicatorCount + column] = column % 2 === 0 ? Math.round(value) : value;
    }
    if (row % 97 === 5) {
      indicators[row * indicatorCount + (row % indicatorCount)] = NaN;
    }
    if (row % 50 === 7) {
      mask[row] = 0;
    }
  }
  return {indicators, indicatorCount, mask};
}

function createFixture(device: Device, scene: Scene): Fixture {
  const {indicatorCount} = scene;
  const rows = scene.indicators.length / indicatorCount;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'composite-parameters',
    format: 'float32',
    length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
  });
  const outputs = {
    score: track(createOutputBuffer(device, rows)),
    scaled: track(createOutputBuffer(device, rows * indicatorCount)),
    columnStatistics: track(createOutputBuffer(device, indicatorCount * 4)),
    loadings: track(createOutputBuffer(device, indicatorCount)),
    principalComponentSummary: track(createOutputBuffer(device, 3))
  };
  const graph = new GPUCommandGraph(device, {id: 'composite-graph'});
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  graph.add(
    new GPUCompositeScore({
      id: 'composite',
      indicators: importGraphBuffer(
        graph,
        'indicators',
        track(createInputBuffer(device, scene.indicators)),
        'float32',
        rows * indicatorCount
      ),
      indicatorCount,
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      enableRank: true,
      enablePrincipalComponent: true,
      output: {
        score: float('score', rows),
        scaled: float('scaled', rows * indicatorCount),
        columnStatistics: float('columnStatistics', indicatorCount * 4),
        loadings: float('loadings', indicatorCount),
        principalComponentSummary: float('principalComponentSummary', 3)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(getGPUCompositeScoreParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        score: await readFloat32(outputs.score, rows),
        scaled: await readFloat32(outputs.scaled, rows * indicatorCount),
        columnStatistics: await readFloat32(outputs.columnStatistics, indicatorCount * 4),
        loadings: await readFloat32(outputs.loadings, indicatorCount),
        principalComponentSummary: await readFloat32(outputs.principalComponentSummary, 3)
      };
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label}[${index}] should be NaN, got ${got}`).toBe(true);
    } else {
      const scale = Math.max(1, Math.abs(want));
      expect(Math.abs(got - want), `${label}[${index}] ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function compare(result: GPUResult, oracle: CompositeScoreOracleResult, label: string): void {
  expectClose(result.columnStatistics, oracle.columnStatistics, 1e-4, `${label} statistics`);
  expectClose(result.scaled, oracle.scaled, 1e-4, `${label} scaled`);
  expectClose(result.score, oracle.score, 1e-4, `${label} score`);
}

const SETTINGS: GPUCompositeScoreSettings[] = [
  {scaler: 'min-max', aggregation: 'weighted-sum', weights: [1, 2, 0.5, 3, 1]},
  {scaler: 'z-score', aggregation: 'weighted-sum', weights: [1, -1, 2, 0, 1], directions: [1, -1]},
  {scaler: 'rank', aggregation: 'weighted-sum', weights: [1, 1, 1, 1, 1], directions: [-1, 1, -1]},
  {
    scaler: 'min-max',
    aggregation: 'weighted-geometric-mean',
    weights: [1, 2, 0, 3, 1],
    epsilon: 0.01
  },
  {
    scaler: 'rank',
    aggregation: 'weighted-geometric-mean',
    weights: [2, 1, 1, 1, 0.5],
    epsilon: 0.05
  },
  {aggregation: 'principal-component', weights: [], directions: [1, -1, 1, -1, 1]}
];

it('GPUCompositeScore matches the CPU oracle across scalers and aggregations without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(3000, 5, 11);
  const fixture = createFixture(device, scene);
  try {
    for (const settings of SETTINGS) {
      const label = `${settings.scaler ?? 'pca'}/${settings.aggregation}`;
      const result = await fixture.run(settings);
      const oracle = computeCompositeScoreOnCPU({...scene, settings});
      compare(result, oracle, label);
      if (settings.aggregation === 'principal-component') {
        expectClose(result.loadings, oracle.loadings, 1e-4, 'loadings');
        expectClose(
          result.principalComponentSummary.slice(0, 2),
          oracle.principalComponent,
          1e-4,
          'principal component'
        );
        expect(result.principalComponentSummary[2]).toBeLessThan(1e-3);
      }
    }
    expect(fixture.rebuildCount).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUCompositeScore averages tied ranks and the output is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(1500, 3, 3);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUCompositeScoreSettings = {scaler: 'rank', weights: [1, 1, 1]};
    const first = await fixture.run(settings);
    const oracle = computeCompositeScoreOnCPU({...scene, settings});
    // Ranks are exact integers; only the final f32 division (2.5 ULP in WGSL) may round.
    expectClose(first.scaled, oracle.scaled, 1e-6, 'ranks');
    const zScore = await fixture.run({scaler: 'z-score', weights: [1, 2, 3]});
    const again = await fixture.run({scaler: 'z-score', weights: [1, 2, 3]});
    expect(again.score).toEqual(zScore.score);
    expect(again.columnStatistics).toEqual(zScore.columnStatistics);
  } finally {
    fixture.destroy();
  }
});

it('GPUCompositeScore returns the major component of two negatively correlated indicators', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two indicators that move against each other, both left at direction +1 (income and poverty).
  // The correlation matrix [[1, r], [r, 1]] has the closed-form eigenpairs (1 + r, (1, 1)) and
  // (1 - r, (1, -1)); with r < 0 the first principal component is the second pair.
  //
  // The design is exact in float32. Column 0 takes 10 or 14 (mean 12, deviation 2), column 1
  // takes 3 or 9 (mean 6, deviation 3). 180 rows agree in sign and 220 disagree, so the
  // covariance is (180 - 220) * 6 / 400 = -0.6 and r = -0.6 / (2 * 3) = -0.1.
  const rows = 400;
  const indicators = new Float32Array(rows * 2);
  const expectedScore = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    // Per 40 rows: 9 rows (+, +), 9 rows (-, -), 11 rows (+, -), 11 rows (-, +).
    const slot = row % 40;
    const firstSign = slot < 9 || (slot >= 18 && slot < 29) ? 1 : -1;
    const secondSign = slot < 9 || slot >= 29 ? 1 : -1;
    indicators[row * 2] = 12 + 2 * firstSign;
    indicators[row * 2 + 1] = 6 + 3 * secondSign;
    expectedScore[row] = (firstSign - secondSign) * Math.SQRT1_2;
  }
  const fixture = createFixture(device, {indicators, indicatorCount: 2});
  try {
    const result = await fixture.run({aggregation: 'principal-component', weights: []});
    expectClose(result.columnStatistics, [10, 14, 12, 2, 3, 9, 6, 3], 1e-6, 'statistics');
    const [eigenvalue, explainedVarianceRatio, residual] = result.principalComponentSummary;
    expect(eigenvalue).toBeCloseTo(1.1, 5);
    expect(explainedVarianceRatio).toBeCloseTo(0.55, 5);
    expect(residual).toBeLessThan(1e-4);
    // Zero-sum loadings: the first of the tied largest loadings is positive.
    expect(result.loadings[0]).toBeCloseTo(Math.SQRT1_2, 4);
    expect(result.loadings[1]).toBeCloseTo(-Math.SQRT1_2, 4);
    expectClose(result.score, expectedScore, 1e-4, 'score');
  } finally {
    fixture.destroy();
  }
});

/**
 * Two columns of -1 and +1, each balanced, with `agreeing` rows of equal sign: every sum is an
 * exact integer in float32, the deviations are 1 and r = (2 * agreeing - rows) / rows exactly.
 */
function createSignPair(rows: number, agreeing: number): Float32Array {
  const indicators = new Float32Array(rows * 2);
  const halfAgreeing = agreeing / 2;
  const halfDisagreeing = (rows - agreeing) / 2;
  for (let row = 0; row < rows; row++) {
    // (+, +) and (-, -) for the agreeing rows, then (+, -) and (-, +).
    let first = 1;
    let second = 1;
    if (row < halfAgreeing) {
      first = 1;
      second = 1;
    } else if (row < agreeing) {
      first = -1;
      second = -1;
    } else if (row < agreeing + halfDisagreeing) {
      first = 1;
      second = -1;
    } else {
      first = -1;
      second = 1;
    }
    indicators[row * 2] = first;
    indicators[row * 2 + 1] = second;
  }
  return indicators;
}

it('GPUCompositeScore gives the exact eigenvector of two weakly correlated indicators', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // [[1, r], [r, 1]] has eigenpairs (1 + r, (1, 1) / sqrt(2)) and (1 - r, (1, -1) / sqrt(2)) for
  // every r, however small. A power iteration does not separate them when |r| is small: 64 steps
  // from an axis gave (0.87, -0.49) at r = -0.01 and (0.998, -0.064) at r = -0.001.
  for (const [rows, agreeing, correlation] of [
    [400, 198, -0.01],
    [4000, 1998, -0.001],
    [400, 202, 0.01]
  ] as const) {
    const fixture = createFixture(device, {
      indicators: createSignPair(rows, agreeing),
      indicatorCount: 2
    });
    try {
      const result = await fixture.run({aggregation: 'principal-component', weights: []});
      const label = `r = ${correlation}`;
      const [eigenvalue, explainedVarianceRatio, residual] = result.principalComponentSummary;
      expect(eigenvalue, label).toBeCloseTo(1 + Math.abs(correlation), 6);
      expect(explainedVarianceRatio, label).toBeCloseTo((1 + Math.abs(correlation)) / 2, 6);
      expect(residual, label).toBeLessThan(1e-6);
      expect(result.loadings[0], label).toBeCloseTo(Math.SQRT1_2, 6);
      expect(result.loadings[1], label).toBeCloseTo(Math.sign(correlation) * Math.SQRT1_2, 6);
    } finally {
      fixture.destroy();
    }
  }
});

it('GPUCompositeScore separates two leading eigenvalues that are 8e-4 apart', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Three balanced columns of -1 and +1 over 10,000 rows. Columns 0 and 1 agree in 5,004 rows
  // (r = 0.0008) and column 2 is uncorrelated with both: the correlation matrix is
  // [[1, 0.0008, 0], [0.0008, 1, 0], [0, 0, 1]] with eigenvalues 1.0008, 1 and 0.9992, and the
  // first component is (1, 1, 0) / sqrt(2). Every moment is an exact integer in float32.
  const rows = 10000;
  const indicators = new Float32Array(rows * 3);
  let row = 0;
  for (const [first, second, count] of [
    [1, 1, 1251],
    [-1, -1, 1251],
    [1, -1, 1249],
    [-1, 1, 1249]
  ] as const) {
    for (const third of [1, -1]) {
      for (let repeat = 0; repeat < count; repeat++) {
        indicators.set([first, second, third], row * 3);
        row++;
      }
    }
  }
  expect(row).toBe(rows);
  const fixture = createFixture(device, {indicators, indicatorCount: 3});
  try {
    const result = await fixture.run({aggregation: 'principal-component', weights: []});
    const [eigenvalue, explainedVarianceRatio, residual] = result.principalComponentSummary;
    expect(eigenvalue).toBeCloseTo(1.0008, 6);
    expect(explainedVarianceRatio).toBeCloseTo(1.0008 / 3, 6);
    expect(residual).toBeLessThan(1e-5);
    expectClose(result.loadings, [Math.SQRT1_2, Math.SQRT1_2, 0], 1e-4, 'loadings');
  } finally {
    fixture.destroy();
  }
});

it('GPUCompositeScore writes NaN for every row when nothing is included', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    indicators: Float32Array.from([1, 2, 3, 4]),
    indicatorCount: 2,
    mask: Uint32Array.from([0, 0])
  };
  const fixture = createFixture(device, scene);
  try {
    const result = await fixture.run({weights: [1, 1]});
    expect(result.score.every(Number.isNaN)).toBe(true);
    expect(result.columnStatistics.every(Number.isNaN)).toBe(true);
  } finally {
    fixture.destroy();
  }
});
