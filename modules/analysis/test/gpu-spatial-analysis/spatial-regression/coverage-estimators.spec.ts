// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {PhiloxStream} from '../../../src/analysis-core/kernels/permutation-random';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {GPUSpatialCrossValidatedOLS} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';

import reference from '../../oracles/fixtures/statistics-coverage.json';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-4) {
  expected.forEach((value, index) =>
    Number.isNaN(value)
      ? expect(actual[index]).toBeNaN()
      : expect(
          Math.abs(actual[index] - value),
          `element ${index}: ${actual[index]} vs ${value}`
        ).toBeLessThan(tolerance)
  );
}

for (const scenario of ['reference', 'exhausted', 'constant'] as const)
  it(`probit likelihood and marginal effects: ${scenario}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const coefficients = output('float32', 3);
      const errors = output('float32', 3);
      const fitted = output('float32', 40);
      const effects = output('float32', 2);
      const summary = output('float32', 4);
      const status = output('uint32', 1);
      graph.add(
        new GPUProbitRegression({
          predictors: input('float32', reference.probit.predictors),
          predictorCount: 2,
          response: input(
            'float32',
            scenario === 'constant' ? Array(40).fill(1) : reference.probit.response
          ),
          maximumIterations: scenario === 'exhausted' ? 1 : 64,
          output: {
            coefficients: coefficients.view,
            standardErrors: errors.view,
            fitted: fitted.view,
            marginalEffects: effects.view,
            summary: summary.view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([
        scenario === 'reference' ? 0 : scenario === 'exhausted' ? 3 : 4
      ]);
      if (scenario === 'reference') {
        close(await readFloat32(coefficients.buffer, 3), reference.probit.coefficients, 1e-3);
        close(await readFloat32(errors.buffer, 3), reference.probit.standardErrors, 1e-3);
        close(await readFloat32(effects.buffer, 2), reference.probit.effects, 1e-3);
        close(await readFloat32(fitted.buffer, 40), reference.probit.fitted, 1e-3);
        expect((await readFloat32(summary.buffer, 4))[1]).toBeCloseTo(
          reference.probit.logLikelihood,
          3
        );
      } else expect(Array.from(await readFloat32(errors.buffer, 3)).every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });
it('restricted regimes match independent dense nested-model Chow statistic and covariance', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', 3);
    const covariance = output('float32', 9);
    const chow = output('float32', 3);
    const status = output('uint32', 1);
    const fitted = output('float32', 40);
    graph.add(
      new GPURestrictedRegimeRegression({
        predictors: input('float32', reference.regimes.predictors),
        predictorCount: 1,
        response: input('float32', reference.regimes.response),
        regimes: input('uint32', reference.regimes.regimes),
        regimeCount: 2,
        common: [false, true],
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          chow: chow.view,
          status: status.view,
          fitted: fitted.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    close(await readFloat32(coefficients.buffer, 3), reference.regimes.coefficients, 1e-3);
    close(await readFloat32(covariance.buffer, 9), reference.regimes.covariance, 1e-3);
    close(await readFloat32(chow.buffer, 3), reference.regimes.chow, 1e-3);
  } finally {
    context.destroy();
  }
});
it('SUR matches independent dense block GLS including cross-equation covariance', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', 4);
    const covariance = output('float32', 16);
    const residualCovariance = output('float32', 4);
    const residuals = output('float32', 80);
    const status = output('uint32', 1);
    graph.add(
      new GPUSeeminglyUnrelatedRegression({
        predictors: input('float32', reference.sur.predictors),
        responses: input('float32', reference.sur.responses),
        equationCount: 2,
        predictorCount: 1,
        rowCount: 40,
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          residualCovariance: residualCovariance.view,
          residuals: residuals.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    close(await readFloat32(coefficients.buffer, 4), reference.sur.coefficients, 1e-3);
    close(await readFloat32(covariance.buffer, 16), reference.sur.covariance, 1e-3);
    close(await readFloat32(residualCovariance.buffer, 4), reference.sur.residualCovariance, 1e-3);
    close(await readFloat32(residuals.buffer, 80), reference.sur.residuals, 1e-3);
  } finally {
    context.destroy();
  }
});
it('unit FE error ML matches direct dense profiled likelihood', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const expected = reference.panelError;
  try {
    const table = output('float32', 12);
    const summary = output('float32', 22);
    const status = output('uint32', 1);
    graph.add(
      new GPUSpatialPanelErrorMaximumLikelihood({
        predictors: input('float32', expected.predictors),
        predictorCount: 1,
        response: input('float32', expected.response),
        periodCount: expected.periods,
        weights: {
          offsets: input('uint32', expected.weights.offsets),
          neighbors: input('uint32', expected.weights.neighbors),
          weights: input('float32', expected.weights.weights)
        },
        output: {table: table.view, summary: summary.view, status: {code: status.view}}
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    const result = await readFloat32(table.buffer, 12);
    close([result[4], result[8]], [expected.coefficient, expected.parameter], 3e-3);
    expect(result[0]).toBeNaN();
    const metrics = await readFloat32(summary.buffer, 22);
    close([metrics[2], metrics[3]], [expected.variance, expected.logLikelihood], 3e-3);
    expect(metrics[4]).toBeNaN();
  } finally {
    context.destroy();
  }
});
for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const)
  it(`interaction ${constraint} calibrated grid and conserved margins match NumPy IPF`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const predictions = output('float32', 9);
      const summary = output('float32', 4);
      const status = output('uint32', 1);
      const candidates = output('float32', 16);
      const originResiduals = output('float32', 3);
      const destinationResiduals = output('float32', 3);
      graph.add(
        new GPUSpatialInteraction({
          origins: input('uint32', [0, 0, 0, 1, 1, 1, 2, 2, 2]),
          destinations: input('uint32', [0, 1, 2, 0, 1, 2, 0, 1, 2]),
          originCount: 3,
          destinationCount: 3,
          flows: input('float32', reference.interaction.flows),
          covariate: input('float32', reference.interaction.cost),
          coefficients: reference.interaction.coefficients,
          constraint,
          output: {
            predictions: predictions.view,
            summary: summary.view,
            status: status.view,
            candidates: candidates.view,
            originResiduals: originResiduals.view,
            destinationResiduals: destinationResiduals.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const expected = reference.interaction.models[constraint];
      close(await readFloat32(predictions.buffer, 9), expected.predictions, 3e-4);
      const result = await readFloat32(summary.buffer, 4);
      close([result[0], result[1]], [expected.coefficient, expected.deviance], 3e-4);
      if (constraint === 'origin' || constraint === 'doubly')
        close(await readFloat32(originResiduals.buffer, 3), [0, 0, 0], 3e-4);
      if (constraint === 'destination' || constraint === 'doubly')
        close(await readFloat32(destinationResiduals.buffer, 3), [0, 0, 0], 3e-4);
    } finally {
      context.destroy();
    }
  });
it('multivariate Geary replays whole-row draws, including cross-variable dependence', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const values = [
    [1, 2],
    [3, 1],
    [2, 5],
    [7, 4]
  ];
  const permutations = 19;
  try {
    const statistic = output('float32', 4);
    const pValues = output('float32', 4);
    const classes = output('uint32', 4);
    const simulations = output('float32', 4 * permutations);
    graph.add(
      new GPUMultivariateLocalGearyPermutationTest({
        values: input('float32', values.flat()),
        variableCount: 2,
        maximumPermutations: permutations,
        seed: [19, 23],
        weights: {
          offsets: input('uint32', [0, 2, 4, 6, 8]),
          neighbors: input('uint32', [1, 2, 0, 3, 0, 3, 1, 2]),
          weights: input('float32', [0.25, 0.75, 0.4, 0.6, 0.2, 0.8, 0.3, 0.7])
        },
        output: {
          statistic: statistic.view,
          pValues: pValues.view,
          classes: classes.view,
          simulations: simulations.view
        }
      })
    );
    context.run();
    const means = [0, 1].map(column => values.reduce((sum, row) => sum + row[column], 0) / 4);
    const deviations = [0, 1].map(column =>
      Math.sqrt(values.reduce((sum, row) => sum + (row[column] - means[column]) ** 2, 0) / 4)
    );
    const standardized = values.map(row =>
      row.map((value, column) => (value - means[column]) / deviations[column])
    );
    const weights = [
      [0.25, 0.75],
      [0.4, 0.6],
      [0.2, 0.8],
      [0.3, 0.7]
    ];
    const neighbors = [
      [1, 2],
      [0, 3],
      [0, 3],
      [1, 2]
    ];
    const evaluate = (row: number, selected: number[]) =>
      selected.reduce(
        (sum, other, slot) =>
          sum +
          (weights[row][slot] *
            standardized[row].reduce(
              (distance, value, column) => distance + (value - standardized[other][column]) ** 2,
              0
            )) /
            2,
        0
      );
    const expected = values.flatMap((_, row) =>
      Array.from({length: permutations}, (_, replicate) => {
        const candidates = [0, 1, 2, 3].filter(other => other !== row);
        const stream = new PhiloxStream(
          [19, 23],
          row,
          replicate + 1,
          PHILOX_PURPOSES.localPermutationTest.id
        );
        const selected = [];
        for (let slot = 0; slot < 2; slot++) {
          const choice = slot + stream.nextBelow(candidates.length - slot);
          selected.push(candidates[choice]);
          [candidates[choice], candidates[slot]] = [candidates[slot], candidates[choice]];
        }
        return evaluate(row, selected);
      })
    );
    close(await readFloat32(simulations.buffer, expected.length), expected, 2e-5);
    close(
      await readFloat32(statistic.buffer, 4),
      neighbors.map((selected, row) => evaluate(row, selected)),
      2e-5
    );
  } finally {
    context.destroy();
  }
});
it('local collinearity returns weighted VIF and identifies singular neighborhoods', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  try {
    const inflation = output('float32', 8);
    const condition = output('float32', 4);
    const status = output('uint32', 4);
    graph.add(
      new GPULocalCollinearity({
        predictors: input('float32', [0, 0, 1, 2, 2, 1, 3, 3]),
        predictorCount: 2,
        weights: {
          offsets: input('uint32', [0, 4, 8, 8, 10]),
          neighbors: input('uint32', [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
          weights: input('float32', Array(10).fill(1))
        },
        output: {
          varianceInflationFactors: inflation.view,
          conditionNumbers: condition.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 4))).toEqual([0, 0, 1, 1]);
    close(
      (await readFloat32(inflation.buffer, 8)).slice(0, 4),
      [25 / 9, 25 / 9, 25 / 9, 25 / 9],
      2e-4
    );
    close((await readFloat32(condition.buffer, 4)).slice(0, 2), [3, 3], 2e-4);
  } finally {
    context.destroy();
  }
});

for (const capacity of [2, 8])
  it(`fuzzy polygon distance preserves positive-tolerance provenance and required count: ${capacity}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const offsets = output('uint32', 5);
      const neighbors = output('uint32', capacity);
      const weights = output('float32', capacity);
      const added = output('uint32', capacity);
      const status = output('uint32', 4);
      const overflow = output('uint32', 1);
      const required = output('uint32', 1);
      graph.add(
        new GPUFuzzyContiguity({
          polygons: {
            kind: 'polygons',
            positions: input(
              'float32x2',
              [
                0, 0, 1, 0, 1, 1, 0, 1, 1.05, 0, 2.05, 0, 2.05, 1, 1.05, 1, 0.2, 0.2, 0.4, 0.2, 0.4,
                0.4, 0.2, 0.4
              ]
            ),
            featureOffsets: input('uint32', [0, 1, 2, 3, 3]),
            polygonOffsets: input('uint32', [0, 1, 2, 3]),
            ringOffsets: input('uint32', [0, 4, 8, 12])
          },
          tolerance: 0.1,
          output: {
            offsets: offsets.view,
            neighbors: neighbors.view,
            weights: weights.view,
            added: added.view,
            featureStatus: status.view,
            status: {overflow: overflow.view, requiredCount: required.view}
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 4))).toEqual([0, 0, 0, 1]);
      expect(Array.from(await readUint32(required.buffer, 1))).toEqual([4]);
      expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([capacity < 4 ? 1 : 0]);
      expect(Array.from(await readUint32(neighbors.buffer, Math.min(capacity, 4)))).toEqual(
        [1, 2, 0, 0].slice(0, capacity)
      );
      expect(Array.from(await readUint32(added.buffer, Math.min(capacity, 4)))).toEqual(
        [1, 0, 1, 0].slice(0, capacity)
      );
    } finally {
      context.destroy();
    }
  });
it('cross-validated OLS is unchanged by perturbing its held-out responses', async () => {
  const rows = 24;
  const predictors = Array.from({length: rows}, (_, row) => row / 4);
  const baseline = predictors.map((value, row) => 1 + 2 * value + Math.sin(row));
  async function run(response: number[]) {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const folds = output('uint32', rows);
      const training = output('uint32', rows * 3);
      const testing = output('uint32', rows * 3);
      const counts = output('uint32', 9);
      const foldStatus = output('uint32', 3);
      const coefficients = output('float32', 6);
      const predictions = output('float32', rows * 3);
      const status = output('uint32', 3);
      const metrics = output('float32', 18);
      const metricStatus = output('uint32', 3);
      graph.add(
        new GPUSpatialCrossValidatedOLS({
          folds: {
            positions: input(
              'float32x2',
              predictors.flatMap(value => [value, 0])
            ),
            blocks: input(
              'uint32',
              Array.from({length: rows}, (_, row) => row)
            ),
            foldCount: 3,
            output: {
              folds: folds.view,
              training: training.view,
              testing: testing.view,
              counts: counts.view,
              status: foldStatus.view
            }
          },
          predictors: input('float32', predictors),
          predictorCount: 1,
          response: input('float32', response),
          output: {
            coefficients: coefficients.view,
            predictions: predictions.view,
            fitStatus: status.view,
            metrics: metrics.view,
            metricStatus: metricStatus.view
          }
        })
      );
      context.run();
      return {
        folds: await readUint32(folds.buffer, rows),
        coefficients: await readFloat32(coefficients.buffer, 6),
        metrics: await readFloat32(metrics.buffer, 18),
        status: await readUint32(status.buffer, 3)
      };
    } finally {
      context.destroy();
    }
  }
  const first = await run(baseline);
  const selected = first.folds[0];
  const second = await run(
    baseline.map((value, row) => value + (first.folds[row] === selected ? 1000 : 0))
  );
  expect(Array.from(first.status)).toEqual([0, 0, 0]);
  close(
    second.coefficients.slice(selected * 2, selected * 2 + 2),
    Array.from(first.coefficients.slice(selected * 2, selected * 2 + 2)),
    1e-5
  );
  expect(second.metrics[selected * 6 + 2]).toBeGreaterThan(900000);
  expect(first.metrics[selected * 6 + 2]).toBeLessThan(2);
});

/** Deterministic (x, y) probit sample from a latent index with coefficients `[intercept, slope]`. */
function createProbitSample(rows: number, coefficients: [number, number], seed: number) {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
  const gaussian = () => {
    let sum = 0;
    for (let term = 0; term < 12; term++) sum += next();
    return sum - 6;
  };
  const predictors: number[] = [];
  const response: number[] = [];
  for (let row = 0; row < rows; row++) {
    const x = gaussian();
    predictors.push(x);
    response.push(coefficients[0] + coefficients[1] * x + gaussian() > 0 ? 1 : 0);
  }
  return {predictors, response};
}

function getNormalCdf(value: number): number {
  // Abramowitz-Stegun 7.1.26 erf in f64 (|error| < 1.5e-7), enough for a 1e-3 oracle.
  const t = 1 / (1 + 0.3275911 * Math.abs(value) * Math.SQRT1_2);
  const poly =
    t *
    (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-0.5 * value * value);
  return 0.5 * (1 + Math.sign(value) * erf);
}

/** Float64 Fisher-scoring probit with intercept; returns coefficients and log likelihood. */
function fitProbitOnCPU(predictors: number[], response: number[]) {
  let intercept = 0;
  let slope = 0;
  let logLikelihood = 0;
  for (let iteration = 0; iteration < 50; iteration++) {
    let gradient = [0, 0];
    let information = [0, 0, 0];
    logLikelihood = 0;
    predictors.forEach((x, row) => {
      const linear = intercept + slope * x;
      const fitted = Math.min(Math.max(getNormalCdf(linear), 1e-12), 1 - 1e-12);
      const density = Math.exp(-0.5 * linear * linear) / Math.sqrt(2 * Math.PI);
      const outcome = response[row];
      logLikelihood += outcome * Math.log(fitted) + (1 - outcome) * Math.log(1 - fitted);
      const score = ((outcome - fitted) * density) / (fitted * (1 - fitted));
      const weight = (density * density) / (fitted * (1 - fitted));
      gradient = [gradient[0] + score, gradient[1] + score * x];
      information = [
        information[0] + weight,
        information[1] + weight * x,
        information[2] + weight * x * x
      ];
    });
    const determinant = information[0] * information[2] - information[1] ** 2;
    intercept += (information[2] * gradient[0] - information[1] * gradient[1]) / determinant;
    slope += (information[0] * gradient[1] - information[1] * gradient[0]) / determinant;
  }
  return {coefficients: [intercept, slope], logLikelihood};
}

for (const scenario of [
  {
    name: 'large row count',
    rows: 30000,
    coefficients: [0.3, 0.8] as [number, number],
    tolerance: 2e-2
  },
  // A strong slope whose maximum likelihood estimate keeps every |eta| below 12 (the float64
  // maximum has max |eta| = 9.96). The former scene used a slope of 3.5, whose maximum has
  // max |eta| = 12.06 on a correctly classified row: it is checked against statsmodels in
  // math-review-probit-bound.spec.ts, where the fit now returns that maximum (regression-01).
  {name: 'strong slope', rows: 400, coefficients: [0.2, 3.0] as [number, number], tolerance: 2e-2}
])
  it(`probit fit is stable and not misreported as separated: ${scenario.name}`, async () => {
    const sample = createProbitSample(scenario.rows, scenario.coefficients, 12345);
    const expected = fitProbitOnCPU(sample.predictors, sample.response);
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const coefficients = output('float32', 2);
      const errors = output('float32', 2);
      const fitted = output('float32', scenario.rows);
      const effects = output('float32', 1);
      const summary = output('float32', 4);
      const status = output('uint32', 1);
      graph.add(
        new GPUProbitRegression({
          predictors: input('float32', sample.predictors),
          predictorCount: 1,
          response: input('float32', sample.response),
          output: {
            coefficients: coefficients.view,
            standardErrors: errors.view,
            fitted: fitted.view,
            marginalEffects: effects.view,
            summary: summary.view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      close(await readFloat32(coefficients.buffer, 2), expected.coefficients, scenario.tolerance);
      const logLikelihood = (await readFloat32(summary.buffer, 4))[1];
      expect(Math.abs(logLikelihood / expected.logLikelihood - 1)).toBeLessThan(1e-4);
    } finally {
      context.destroy();
    }
  });
