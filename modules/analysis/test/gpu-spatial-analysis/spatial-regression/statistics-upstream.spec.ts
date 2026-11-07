// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  GPUInstrumentalVariablesCovariance,
  GPUProbitRegression,
  GPURestrictedRegimeRegression,
  GPULocalCollinearity,
  GPUSeeminglyUnrelatedRegression,
  GPUSpatialPanelErrorMaximumLikelihood,
  GPUMultivariateLocalGearyPermutationTest,
  GPURankMarkov,
  GPUFuzzyContiguity,
  GPUIslandAttachment,
  GPUSpatialInteraction,
  GPUSpatialCrossValidation
} from '../../../src/gpu-spatial-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from '../../oracles/fixtures/statistics-upstream.json';
import blockReference from '../../oracles/fixtures/statistics-blockcv.json';

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

function uploadWeights(
  input: Awaited<ReturnType<typeof setup>>['input'],
  weights: {offsets: number[]; neighbors: number[]; weights: number[]}
) {
  return {
    offsets: input('uint32', weights.offsets),
    neighbors: input('uint32', weights.neighbors),
    weights: input('float32', weights.weights)
  };
}

for (const method of ['HC0', 'HC1', 'HAC'] as const)
  for (const overflow of method === 'HAC' ? [undefined, 0, 1] : [undefined])
    it(`direct spreg TSLS ${method} covariance (overflow=${overflow})`, async () => {
      const context = await setup();
      const {graph, input, output} = context;
      const data = reference.instrumentalVariables;
      try {
        const covariance = output('float32', 9);
        const status = output('uint32', 1);
        graph.add(
          new GPUInstrumentalVariablesCovariance({
            design: input('float32', data.design),
            coefficientCount: 2,
            instruments: input('float32', data.instruments),
            instrumentCount: 2,
            residuals: input('float32', data.residuals),
            method,
            ...(method === 'HAC'
              ? {
                  kernel: {
                    ...uploadWeights(input, data.kernel),
                    ...(overflow !== undefined
                      ? {status: {overflow: input('uint32', [overflow])}}
                      : {})
                  }
                }
              : {}),
            output: {covariance: covariance.view, status: status.view}
          })
        );
        context.run();
        expect(Array.from(await readUint32(status.buffer))).toEqual([overflow ? 4 : 0]);
        if (overflow) {
          expect(Array.from(await readFloat32(covariance.buffer)).every(Number.isNaN)).toBe(true);
        } else {
          close(await readFloat32(covariance.buffer), data.covariance[method], 0.002);
        }
      } finally {
        context.destroy();
      }
    });

for (const overflow of [undefined, 0, 1])
  it(`direct spreg Probit likelihood, average effects and spatial diagnostics (overflow=${overflow})`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.probit;
    try {
      const coefficients = output('float32', 3),
        standardErrors = output('float32', 3),
        fitted = output('float32', 40),
        effects = output('float32', 2),
        summary = output('float32', 4),
        status = output('uint32', 1),
        tests = output('float32', 6),
        diagnosticStatus = output('uint32', 1);
      graph.add(
        new GPUProbitRegression({
          predictors: input('float32', data.predictors),
          predictorCount: 2,
          response: input('float32', data.response),
          spatialDiagnostics: {
            weights: {
              ...uploadWeights(input, data.weights),
              ...(overflow !== undefined ? {status: {overflow: input('uint32', [overflow])}} : {})
            },
            output: {tests: tests.view, status: diagnosticStatus.view}
          },
          output: {
            coefficients: coefficients.view,
            standardErrors: standardErrors.view,
            fitted: fitted.view,
            marginalEffects: effects.view,
            summary: summary.view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
      expect(Array.from(await readUint32(diagnosticStatus.buffer))).toEqual([overflow ? 2 : 0]);
      close(await readFloat32(coefficients.buffer), data.coefficients);
      close(await readFloat32(effects.buffer), data.effects);
      if (overflow) {
        expect(Array.from(await readFloat32(tests.buffer)).every(Number.isNaN)).toBe(true);
      } else {
        close(await readFloat32(tests.buffer), data.diagnostics, 0.002);
      }
      expect((await readFloat32(summary.buffer))[1]).toBeCloseTo(data.logLikelihood, 3);
    } finally {
      context.destroy();
    }
  });

it('direct spreg restricted regime covariance and coefficient/joint Chow stability tails', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.regimes;
  try {
    const coefficients = output('float32', 3),
      covariance = output('float32', 9),
      fitted = output('float32', 40),
      chow = output('float32', 3),
      stability = output('float32', 9),
      status = output('uint32', 1);
    graph.add(
      new GPURestrictedRegimeRegression({
        predictors: input('float32', data.predictors),
        predictorCount: 1,
        response: input('float32', data.response),
        regimes: input('uint32', data.regimes),
        regimeCount: 2,
        common: [false, true],
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          fitted: fitted.view,
          chow: chow.view,
          stability: stability.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
    close(await readFloat32(coefficients.buffer), data.coefficients);
    close(await readFloat32(covariance.buffer), data.covariance);
    const values = await readFloat32(stability.buffer);
    close(values.slice(0, 3), [data.stability[0], 1, data.stability[1]], 0.003);
    expect(values[3]).toBeNaN();
    close(values.slice(6), [data.joint[0], 1, data.joint[1]], 0.003);
  } finally {
    context.destroy();
  }
});

it('direct mgwr local collinearity includes intercept and variance decomposition', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.collinearity;
  try {
    const inflation = output('float32', 48),
      correlationCondition = output('float32', 24),
      condition = output('float32', 24),
      decomposition = output('float32', 72),
      status = output('uint32', 24);
    graph.add(
      new GPULocalCollinearity({
        predictors: input('float32', data.predictors),
        predictorCount: 2,
        weights: uploadWeights(input, data.weights),
        output: {
          varianceInflationFactors: inflation.view,
          conditionNumbers: correlationCondition.view,
          designConditionNumbers: condition.view,
          varianceDecomposition: decomposition.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer))).toEqual(Array(24).fill(0));
    close(await readFloat32(inflation.buffer), data.varianceInflationFactors);
    close(await readFloat32(condition.buffer), data.designConditionNumbers, 0.001);
    close(await readFloat32(decomposition.buffer), data.varianceDecomposition, 0.001);
  } finally {
    context.destroy();
  }
});

for (const mode of ['ordinary', 'spatial'] as const)
  it(`direct spreg ${mode} joint SUR`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.sur;
    try {
      const coefficients = output('float32', 4),
        covariance = output('float32', 16),
        sigma = output('float32', 4),
        residuals = output('float32', 80),
        status = output('uint32', 1),
        parameters = output('float32', 2),
        parameterStatus = output('uint32', 2);
      graph.add(
        new GPUSeeminglyUnrelatedRegression({
          predictors: input('float32', data.predictors),
          responses: input('float32', data.responses),
          equationCount: 2,
          predictorCount: 1,
          rowCount: 40,
          ...(mode === 'spatial'
            ? {
                spatial: {
                  weights: uploadWeights(input, data.weights),
                  output: {parameters: parameters.view, status: parameterStatus.view}
                }
              }
            : {}),
          output: {
            coefficients: coefficients.view,
            covariance: covariance.view,
            residualCovariance: sigma.view,
            residuals: residuals.view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
      close(await readFloat32(coefficients.buffer), data[mode].coefficients, 0.003);
      close(await readFloat32(covariance.buffer), data[mode].covariance, 0.003);
      close(await readFloat32(sigma.buffer), data[mode].residualCovariance, 0.003);
      if (mode === 'spatial') {
        expect(Array.from(await readUint32(parameterStatus.buffer))).toEqual([0, 0]);
        close(await readFloat32(parameters.buffer), data.spatial.parameters, 0.003);
        close(await readFloat32(residuals.buffer), data.spatial.residuals, 0.004);
      }
    } finally {
      context.destroy();
    }
  });

for (const overflow of [undefined, 0, 1])
  it(`direct spreg Panel_FE_Error selected likelihood and coefficients (overflow=${overflow})`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.panelError;
    try {
      const table = output('float32', 12),
        summary = output('float32', 22),
        status = output('uint32', 1),
        tests = output('float32', 6),
        diagnosticStatus = output('uint32', 1);
      graph.add(
        new GPUSpatialPanelErrorMaximumLikelihood({
          weights: {
            ...uploadWeights(input, data.weights),
            ...(overflow !== undefined ? {status: {overflow: input('uint32', [overflow])}} : {})
          },
          predictors: input('float32', data.predictors),
          predictorCount: 1,
          response: input('float32', data.response),
          periodCount: data.periods,
          maximumPower: 128,
          diagnostics: {tests: tests.view, status: diagnosticStatus.view},
          output: {table: table.view, summary: summary.view, status: {code: status.view}}
        })
      );
      context.run();
      if (overflow) {
        expect((await readUint32(status.buffer))[0]).not.toBe(0);
        expect(Array.from(await readUint32(diagnosticStatus.buffer))).toEqual([1]);
        expect(Array.from(await readFloat32(tests.buffer)).every(Number.isNaN)).toBe(true);
        return;
      }
      expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
      const values = await readFloat32(table.buffer);
      expect(values[4]).toBeCloseTo(data.coefficient, 3);
      expect(values[8]).toBeCloseTo(data.parameter, 3);
      expect(Array.from(await readUint32(diagnosticStatus.buffer))).toEqual([0]);
      close(await readFloat32(tests.buffer), data.diagnostics, 0.003);
    } finally {
      context.destroy();
    }
  });

for (const scenario of ['reference', 'weights-overflow', 'variance-overflow'])
  it(`direct esda multivariate Geary and joint replay (${scenario})`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.multivariate;
    try {
      const statistic = output('float32', 12),
        probabilities = output('float32', 12),
        classes = output('uint32', 12),
        simulations = output('float32', 12 * 19);
      graph.add(
        new GPUMultivariateLocalGearyPermutationTest({
          weights: {
            ...uploadWeights(input, data.weights),
            status: {overflow: input('uint32', [scenario === 'weights-overflow' ? 1 : 0])}
          },
          values: input(
            'float32',
            data.values.map(value => value * (scenario === 'variance-overflow' ? 1e20 : 1))
          ),
          variableCount: 2,
          maximumPermutations: 19,
          tail: 'esda-min-tail',
          output: {
            statistic: statistic.view,
            pValues: probabilities.view,
            classes: classes.view,
            simulations: simulations.view
          }
        })
      );
      context.run();
      const observed = await readFloat32(statistic.buffer),
        replays = await readFloat32(simulations.buffer),
        actual = await readFloat32(probabilities.buffer);
      if (scenario !== 'reference') {
        expect(Array.from(observed).every(Number.isNaN)).toBe(true);
        expect(Array.from(actual).every(Number.isNaN)).toBe(true);
        expect(Array.from(replays).every(Number.isNaN)).toBe(true);
        expect(Array.from(await readUint32(classes.buffer))).toEqual(Array(12).fill(0xffffffff));
        return;
      }
      close(observed, data.statistic);
      for (let row = 0; row < 12; row++) {
        const greater = Array.from(replays.slice(row * 19, (row + 1) * 19)).filter(
          value => value >= observed[row]
        ).length;
        expect(actual[row]).toBeCloseTo((Math.min(greater, 19 - greater) + 1) / 20, 6);
      }
    } finally {
      context.destroy();
    }
  });

for (const mode of ['full-rank', 'geographic-rank'] as const)
  it(`direct giddy ${mode} ties and transitions`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.rank;
    try {
      const ranks = output('uint32', 18),
        counts = output('uint32', 36),
        probabilities = output('float32', 36);
      graph.add(
        new GPURankMarkov({
          values: input('float32', data.values),
          periodCount: 3,
          mode,
          output: {ranks: ranks.view, counts: counts.view, probabilities: probabilities.view}
        })
      );
      context.run();
      expect(Array.from(await readUint32(ranks.buffer))).toEqual(data.ranks);
      expect(Array.from(await readUint32(counts.buffer))).toEqual(
        mode === 'full-rank' ? data.fullCounts : data.geographicCounts
      );
    } finally {
      context.destroy();
    }
  });

for (const mode of ['fuzzy', 'islands'] as const)
  it(`direct libpysal ${mode} adjacency`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const offsets = output('uint32', 5),
        neighbors = output('uint32', 12),
        weights = output('float32', 12),
        added = output('uint32', 12),
        overflow = output('uint32', 1),
        required = output('uint32', 1);
      const common = {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: weights.view,
        added: added.view,
        status: {overflow: overflow.view, requiredCount: required.view}
      };
      if (mode === 'fuzzy')
        graph.add(
          new GPUFuzzyContiguity({
            polygons: {
              kind: 'polygons',
              positions: input('float32x2', reference.fuzzy.positions),
              featureOffsets: input('uint32', [0, 1, 2, 3, 4]),
              polygonOffsets: input('uint32', [0, 1, 2, 3, 4]),
              ringOffsets: input('uint32', [0, 4, 8, 12, 16])
            },
            tolerance: 0.1,
            output: {...common, featureStatus: output('uint32', 4).view}
          })
        );
      else
        graph.add(
          new GPUIslandAttachment({
            weights: uploadWeights(input, reference.islands.initial),
            positions: input('float32x2', reference.islands.positions),
            output: {...common, attachment: output('uint32', 4).view}
          })
        );
      context.run();
      const expected = mode === 'fuzzy' ? reference.fuzzy.weights : reference.islands.attached;
      expect(Array.from(await readUint32(overflow.buffer))).toEqual([0]);
      expect(Array.from(await readUint32(offsets.buffer))).toEqual(expected.offsets);
      expect(
        Array.from(await readUint32(neighbors.buffer)).slice(0, expected.neighbors.length)
      ).toEqual(expected.neighbors);
    } finally {
      context.destroy();
    }
  });

for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const)
  it(`direct spint continuous ${constraint} calibration`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.interaction;
    try {
      const predictions = output('float32', 20),
        summary = output('float32', 4),
        status = output('uint32', 1),
        candidates = output('float32', 8),
        calibration = output('float32', 4),
        originResiduals = output('float32', 4),
        destinationResiduals = output('float32', 5);
      graph.add(
        new GPUSpatialInteraction({
          origins: input('uint32', data.origins),
          destinations: input('uint32', data.destinations),
          originCount: 4,
          destinationCount: 5,
          flows: input('float32', data.flows),
          covariate: input('float32', data.cost),
          constraint,
          coefficients: [-3, 1],
          calibration: 'bounded',
          maximumCalibrationIterations: 32,
          coefficientTolerance: 1e-5,
          tolerance: 1e-6,
          output: {
            predictions: predictions.view,
            summary: summary.view,
            status: status.view,
            candidates: candidates.view,
            calibration: calibration.view,
            originResiduals: originResiduals.view,
            destinationResiduals: destinationResiduals.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
      const values = await readFloat32(summary.buffer);
      expect(values[0]).toBeCloseTo(data.models[constraint].coefficient, 2);
      expect(values[1]).toBeCloseTo(data.models[constraint].deviance, 2);
      close(await readFloat32(predictions.buffer), data.models[constraint].predictions, 0.035);
      expect((await readFloat32(calibration.buffer))[3]).toBe(1);
    } finally {
      context.destroy();
    }
  });

for (const family of ['poisson', 'logistic'] as const)
  it(`direct mgwr ${family} local inference and influence`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.generalizedGWR[family];
    try {
      const coefficients = output('float32', 72),
        fitted = output('float32', 24),
        codes = output('uint32', 24),
        iterationCount = output('uint32', 24),
        standardErrors = output('float32', 72),
        influence = output('float32', 24);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          family,
          predictorCount: 2,
          positions: input('float32x2', data.positions),
          predictors: input('float32', data.predictors),
          response: input('float32', data.response),
          parameters: input('float32', [3, 1e-6]),
          maximumIterations: 128,
          inference: {standardErrors: standardErrors.view, influence: influence.view},
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: iterationCount.view}
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(codes.buffer))).toEqual(Array(24).fill(0));
      close(await readFloat32(coefficients.buffer), data.coefficients, 0.002);
      close(await readFloat32(standardErrors.buffer), data.standardErrors, 0.002);
      close(await readFloat32(influence.buffer), data.influence, 0.002);
    } finally {
      context.destroy();
    }
  });

it('generalized bandwidth selection never uses outer-heldout responses and predicts those queries', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.generalizedGWR.poisson;
  try {
    const results = [];
    for (const heldoutResponse of [0, 100000]) {
      const response = data.response.map((value, index) =>
        index % 4 === 0 ? heldoutResponse : value
      );
      const coefficients = output('float32', 72),
        fitted = output('float32', 24),
        codes = output('uint32', 24),
        iterationCount = output('uint32', 24),
        scores = output('float32', 3),
        bandwidth = output('float32', 1),
        status = output('uint32', 1);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          id: `masked-selection-${heldoutResponse}`,
          family: 'poisson',
          predictorCount: 2,
          positions: input('float32x2', data.positions),
          predictors: input('float32', data.predictors),
          response: input('float32', response),
          mask: input(
            'uint32',
            response.map((_, index) => (index % 4 === 0 ? 0 : 1))
          ),
          parameters: input('float32', [1, 1e-6]),
          maximumIterations: 128,
          predictExcluded: true,
          bandwidthSelection: {
            bandwidths: [1.5, 3, 6],
            output: {scores: scores.view, bandwidth: bandwidth.view, status: status.view}
          },
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: iterationCount.view}
          }
        })
      );
      results.push({coefficients, fitted, codes, scores, bandwidth, status});
    }
    context.run();
    for (const result of results) {
      expect(Array.from(await readUint32(result.status.buffer))).toEqual([0]);
      expect(Array.from(await readUint32(result.codes.buffer))).toEqual(Array(24).fill(0));
    }
    close(
      await readFloat32(results[0].scores.buffer),
      reference.generalizedGWR.selection.scores,
      0.003
    );
    expect((await readFloat32(results[0].bandwidth.buffer))[0]).toEqual(
      reference.generalizedGWR.selection.bandwidth
    );
    close(
      await readFloat32(results[0].scores.buffer),
      Array.from(await readFloat32(results[1].scores.buffer)),
      1e-6
    );
    close(
      await readFloat32(results[0].coefficients.buffer),
      Array.from(await readFloat32(results[1].coefficients.buffer)),
      1e-6
    );
    close(
      await readFloat32(results[0].fitted.buffer),
      Array.from(await readFloat32(results[1].fitted.buffer)),
      1e-6
    );
  } finally {
    context.destroy();
  }
});

it('bounded flow calibration exposes exhausted search with finite provisional fit', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.interaction;
  try {
    const predictions = output('float32', 20),
      summary = output('float32', 4),
      status = output('uint32', 1),
      calibration = output('float32', 4),
      candidates = output('float32', 8);
    graph.add(
      new GPUSpatialInteraction({
        origins: input('uint32', data.origins),
        destinations: input('uint32', data.destinations),
        originCount: 4,
        destinationCount: 5,
        flows: input('float32', data.flows),
        covariate: input('float32', data.cost),
        constraint: 'doubly',
        coefficients: [-3, 1],
        calibration: 'bounded',
        maximumCalibrationIterations: 1,
        coefficientTolerance: 1e-8,
        maximumIterations: 1024,
        output: {
          predictions: predictions.view,
          summary: summary.view,
          status: status.view,
          calibration: calibration.view,
          candidates: candidates.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer))).toEqual([4]);
    const interval = await readFloat32(calibration.buffer);
    expect(interval[3]).toEqual(0);
    expect(interval[2]).toBeGreaterThan(1);
    expect((await readFloat32(predictions.buffer)).every(Number.isFinite)).toBe(true);
  } finally {
    context.destroy();
  }
});

it('spatial SUR rejects unstandardized weights without reporting a valid equation parameter', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.sur;
  try {
    const coefficients = output('float32', 4),
      covariance = output('float32', 16),
      residualCovariance = output('float32', 4),
      residuals = output('float32', 80),
      status = output('uint32', 1),
      parameters = output('float32', 2),
      spatialStatus = output('uint32', 2);
    graph.add(
      new GPUSeeminglyUnrelatedRegression({
        rowCount: 40,
        equationCount: 2,
        predictorCount: 1,
        predictors: input('float32', data.predictors),
        responses: input('float32', data.responses),
        spatial: {
          weights: uploadWeights(input, {
            ...data.weights,
            weights: data.weights.weights.map(value => value * 2)
          }),
          output: {parameters: parameters.view, status: spatialStatus.view}
        },
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
    expect(Array.from(await readUint32(spatialStatus.buffer))).toEqual([4, 4]);
    expect((await readFloat32(parameters.buffer)).every(Number.isNaN)).toBe(true);
    expect((await readUint32(status.buffer))[0]).not.toEqual(0);
  } finally {
    context.destroy();
  }
});

for (const mode of ['spatial', 'buffer'] as const)
  it(`direct blockCV ${mode} fold membership and training separation`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = blockReference[mode];
    try {
      const folds = output('uint32', 12),
        training = output('uint32', 12 * data.foldCount),
        testing = output('uint32', 12 * data.foldCount),
        counts = output('uint32', 3 * data.foldCount),
        status = output('uint32', data.foldCount);
      graph.add(
        new GPUSpatialCrossValidation({
          positions: input('float32x2', data.positions),
          blocks: input('uint32', data.blocks),
          foldCount: data.foldCount,
          seed: data.seed,
          bufferDistance: data.bufferDistance,
          output: {
            folds: folds.view,
            training: training.view,
            testing: testing.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer))).toEqual(Array(data.foldCount).fill(0));
      expect(Array.from(await readUint32(training.buffer))).toEqual(data.training);
      expect(Array.from(await readUint32(testing.buffer))).toEqual(data.testing);
    } finally {
      context.destroy();
    }
  });
