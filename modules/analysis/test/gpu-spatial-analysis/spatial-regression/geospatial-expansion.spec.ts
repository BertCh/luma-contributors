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
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {
  GPUCrossVariogram,
  GPUSpaceTimeVariogram
} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import reference from '../../oracles/fixtures/geospatial-expansion-statistics.json';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';

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

it('LOSH and multivariate Geary agree with complete-case neighborhood references', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const weights = {
    offsets: input('uint32', [0, 1, 3, 5, 6]),
    neighbors: input('uint32', [1, 0, 2, 1, 3, 2]),
    weights: input('float32', [1, 1, 1, 1, 1, 1])
  };
  const statistic = output('float32', 4);
  const status = output('uint32', 4);
  const multivariate = output('float32', 4);
  const multivariateStatus = output('uint32', 4);
  graph.add(
    new GPULocalSpatialHeteroscedasticity({
      weights,
      values: input('float32', [1, 2, 4, 8]),
      output: {statistic: statistic.view, status: status.view}
    })
  );
  graph.add(
    new GPUMultivariateLocalGeary({
      weights,
      values: input('float32', [1, 1, 2, 4, 4, 2, 8, 7]),
      variableCount: 2,
      output: {statistic: multivariate.view, status: multivariateStatus.view}
    })
  );
  context.run();
  close(await readFloat32(statistic.buffer, 4), [
    0.25 / 4.5625,
    1 / 4.5625,
    8.125 / 4.5625,
    1 / 4.5625
  ]);
  const first = [1, 2, 4, 8];
  const second = [1, 4, 2, 7];
  const adjacency = [[1], [0, 2], [1, 3], [2]];
  const variance = (values: number[]) => {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  };
  const expected = adjacency.map(
    (neighbors, row) =>
      neighbors.reduce(
        (sum, neighbor) =>
          sum +
          ((first[row] - first[neighbor]) ** 2 / variance(first) +
            (second[row] - second[neighbor]) ** 2 / variance(second)) /
            2,
        0
      ) / neighbors.length
  );
  close(await readFloat32(multivariate.buffer, 4), expected);
  expect(Array.from(await readUint32(status.buffer, 4))).toEqual([0, 0, 0, 0]);
  expect(Array.from(await readUint32(multivariateStatus.buffer, 4))).toEqual([0, 0, 0, 0]);
  context.destroy();
});

it('cross and space-time variograms retain negative covariance and exact boundary counts', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const positions = input('float32x2', [0, 0, 1, 0, 2, 0]);
  const values = input('float32', [1, 3, 2]);
  const secondValues = input('float32', [4, 0, 2]);
  const parameters = input('float32', [3, 3]);
  const cross = output('float32', 3);
  const counts = output('uint32', 3);
  const status = output('uint32', 3);
  graph.add(
    new GPUCrossVariogram({
      positions,
      values,
      secondValues,
      parameters,
      lagCount: 3,
      output: {semivariances: cross.view, pairCounts: counts.view, status: status.view}
    })
  );
  const temporal = output('float32', 16);
  const temporalCounts = output('uint32', 16);
  const temporalStatus = output('uint32', 16);
  graph.add(
    new GPUSpaceTimeVariogram({
      positions,
      values,
      secondValues,
      // Bounds cover the three stations; maximum distance 3 and maximum time lag 3.
      parameters: input('float32', [0, 0, 2, 0, 3, 3, 0, 0]),
      gridSize: [2, 2],
      lagCount: 3,
      times: input('float32', [0, 1, 3]),
      temporalLagCount: 3,
      output: {
        semivariances: temporal.view,
        pairCounts: temporalCounts.view,
        status: temporalStatus.view
      }
    })
  );
  context.run();
  close(await readFloat32(cross.buffer, 3), [NaN, -2.5, -1]);
  expect(Array.from(await readUint32(counts.buffer, 3))).toEqual([0, 2, 1]);
  // The former 3 x 3 grid had its occupied bins at 4, 5 and 8 (spatial class s, temporal class t,
  // index s * 3 + t). With a zero-lag class on each axis, bin (s, t) moves to (s + 1, t + 1) of a
  // 4 x 4 grid: indices 10, 11 and 15. The values -4, -1 and -1 are unchanged.
  close(await readFloat32(temporal.buffer, 16), [
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    -4,
    -1,
    NaN,
    NaN,
    NaN,
    -1
  ]);
  expect(Array.from(await readUint32(temporalCounts.buffer, 16))).toEqual([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1
  ]);
  context.destroy();
});

it('marked point statistics cover cross-type K/L, supplied intensity and product marks', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const positions = input('float32x2', [1, 1, 2, 1, 3, 1]);
  const parameters = input('float32', [0, 0, 4, 4, 1.5, 0]);
  const marks = input('float32', [1, 2, 3]);
  const homogeneous = output('float32', 4);
  const inhomogeneous = output('float32', 4);
  const cross = output('float32', 4);
  for (const [name, target, extra] of [
    ['homogeneous', homogeneous, {}],
    ['inhomogeneous', inhomogeneous, {intensities: input('float32', [3 / 16, 3 / 16, 3 / 16])}],
    ['cross', cross, {types: input('uint32', [0, 1, 0]), typePair: [0, 1] as const}]
  ] as const) {
    graph.add(
      new GPUMarkedPointStatistics({
        id: name,
        positions,
        parameters,
        marks,
        radiusCount: 1,
        edgeCorrection: 'none',
        ...extra,
        output: {
          statistics: target.view,
          pairCounts: output('uint32', 1).view,
          status: output('uint32', 1).view
        }
      })
    );
  }
  context.run();
  // Three points, four ordered pairs within 1.5, area 16. Same-type K without intensities uses
  // n (n - 1) = 6 (GPURipley, spatstat Kest): 16 * 4 / 6 = 32 / 3.
  close(await readFloat32(homogeneous.buffer, 4), [
    32 / 3,
    Math.sqrt(32 / 3 / Math.PI),
    32 / 3 / (Math.PI * 2.25),
    1
  ]);
  // With intensities 3 / 16 the estimator is sum 1 / (lambda_i lambda_j area) = 4 * 256 / 144,
  // the n^2 form, which is (n - 1) / n of the homogeneous one.
  close(await readFloat32(inhomogeneous.buffer, 4), [
    64 / 9,
    Math.sqrt(64 / 9 / Math.PI),
    64 / 9 / (Math.PI * 2.25),
    1
  ]);
  close(await readFloat32(cross.buffer, 4), [
    16,
    Math.sqrt(16 / Math.PI),
    16 / (Math.PI * 2.25),
    1
  ]);
  context.destroy();
});

it('Poisson and logistic local IRLS converge and expose forced iteration limits', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const positions = input('float32x2', new Array(12).fill(0));
  const predictors = input('float32', [-1, 0, 1, -1, 0, 1]);
  const parameters = input('float32', [1, 1e-5]);
  const poisson = output('float32', 12);
  const poissonStatus = output('uint32', 6);
  const logistic = output('float32', 12);
  const logisticStatus = output('uint32', 6);
  const limitStatus = output('uint32', 6);
  for (const [name, family, response, coefficients, status, maximumIterations] of [
    ['poisson', 'poisson', [1, 2, 4, 1, 2, 4], poisson, poissonStatus, 32],
    ['logistic', 'logistic', [0, 0, 0, 1, 1, 1], logistic, logisticStatus, 32],
    ['limit', 'poisson', [1, 2, 4, 1, 2, 4], output('float32', 12), limitStatus, 1]
  ] as const)
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        id: name,
        family,
        positions,
        predictors,
        response: input('float32', [...response]),
        predictorCount: 1,
        parameters,
        maximumIterations,
        output: {
          coefficients: coefficients.view,
          status: {codes: status.view, iterationCount: output('uint32', 6).view},
          fitted: output('float32', 6).view
        }
      })
    );
  context.run();
  close(await readFloat32(poisson.buffer, 12), new Array(12).fill(Math.log(2)), 1e-3);
  close(await readFloat32(logistic.buffer, 12), new Array(12).fill(0));
  expect(Array.from(await readUint32(poissonStatus.buffer, 6))).toEqual(new Array(6).fill(0));
  expect(Array.from(await readUint32(logisticStatus.buffer, 6))).toEqual(new Array(6).fill(0));
  expect(Array.from(await readUint32(limitStatus.buffer, 6))).toEqual(new Array(6).fill(3));
  context.destroy();
});

it('multiscale backfitting converges on orthogonal predictors and reports incomplete sweeps', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const positions = input('float32x2', [0, 0, 0, 0, 0, 0]);
  const predictors = input('float32', [-1, 0, 1]);
  const response = input('float32', [1, 3, 5]);
  const parameters = input('float32', [1e-5, 1, 2]);
  const coefficients = output('float32', 6);
  const status = output('uint32', 3);
  const limited = output('uint32', 3);
  for (const [name, maximumIterations, result, code] of [
    ['full', 4, coefficients, status],
    ['limited', 1, output('float32', 6), limited]
  ] as const)
    graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        id: name,
        positions,
        predictors,
        response,
        parameters,
        predictorCount: 1,
        maximumIterations,
        output: {
          coefficients: result.view,
          status: {codes: code.view, iterationCount: output('uint32', 1).view},
          fitted: output('float32', 3).view
        }
      })
    );
  context.run();
  close(await readFloat32(coefficients.buffer, 6), [3, 2, 3, 2, 3, 2]);
  expect(Array.from(await readUint32(status.buffer, 3))).toEqual([0, 0, 0]);
  // The global least squares start is already the fixed point of identical locations.
  expect(Array.from(await readUint32(limited.buffer, 3))).toEqual([0, 0, 0]);
  context.destroy();
});

it('multiscale backfitting starts from the global fit and reports incomplete sweeps', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  // Spread locations with wide bandwidths: local surfaces differ from the global fit, so one
  // sweep cannot reach the tolerance while many can.
  const positions = input('float32x2', [0, 0, 1, 0, 2, 0, 3, 0, 4, 0]);
  const predictors = input('float32', [-2, -1, 0, 1, 2]);
  const response = input('float32', [0, 3, 1, 6, 5]);
  const parameters = input('float32', [1e-6, 2, 3]);
  const limited = output('uint32', 5);
  const full = output('uint32', 5);
  for (const [name, maximumIterations, code] of [
    ['limited', 1, limited],
    ['full', 64, full]
  ] as const)
    graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        id: name,
        positions,
        predictors,
        response,
        parameters,
        predictorCount: 1,
        maximumIterations,
        output: {
          coefficients: output('float32', 10).view,
          status: {codes: code.view, iterationCount: output('uint32', 1).view},
          fitted: output('float32', 5).view
        }
      })
    );
  context.run();
  expect(Array.from(await readUint32(limited.buffer, 5))).toEqual([3, 3, 3, 3, 3]);
  expect(Array.from(await readUint32(full.buffer, 5))).toEqual([0, 0, 0, 0, 0]);
  context.destroy();
});

it('spatial regimes recover distinct slopes and retain empty-regime status', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const coefficients = output('float32', 6);
  const fitted = output('float32', 6);
  const status = output('uint32', 3);
  const counts = output('uint32', 3);
  graph.add(
    new GPUSpatialRegimeRegression({
      predictors: input('float32', [0, 1, 2, 0, 1, 2]),
      response: input('float32', [1, 3, 5, 4, 3, 2]),
      regimes: input('uint32', [0, 0, 0, 1, 1, 1]),
      predictorCount: 1,
      regimeCount: 3,
      output: {
        coefficients: coefficients.view,
        fitted: fitted.view,
        status: status.view,
        counts: counts.view
      }
    })
  );
  context.run();
  close(await readFloat32(coefficients.buffer, 6), [1, 2, 4, -1, NaN, NaN]);
  close(await readFloat32(fitted.buffer, 6), [1, 3, 5, 4, 3, 2]);
  expect(Array.from(await readUint32(status.buffer, 3))).toEqual([0, 0, 1]);
  context.destroy();
});

it('pooled spatial panel equals direct 2SLS on period-block weights', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const offsets = [0, 2, 4, 6, 8];
  const neighbors = [1, 3, 0, 2, 1, 3, 0, 2];
  const edgeWeights = new Array(8).fill(0.5);
  const weights = {
    offsets: input('uint32', offsets),
    neighbors: input('uint32', neighbors),
    weights: input('float32', edgeWeights)
  };
  const predictors = input('float32', [1, 4, 2, 8, 3, 9, 4, 1]);
  const response = input('float32', [2, 7, 3, 5, 6, 2, 9, 8]);
  const panel = output('float32', 12);
  const direct = output('float32', 12);
  const panelStatus = output('uint32', 1);
  const directStatus = output('uint32', 1);
  graph.add(
    new GPUSpatialPanelRegression({
      weights,
      periodCount: 2,
      predictorCount: 1,
      predictors,
      response,
      output: {
        table: panel.view,
        summary: output('float32', 32).view,
        status: {code: panelStatus.view}
      }
    })
  );
  graph.add(
    new GPUSpatialTwoStageLeastSquares({
      weights: {
        offsets: input('uint32', [...offsets.slice(0, 4), ...offsets.map(value => value + 8)]),
        neighbors: input('uint32', [...neighbors, ...neighbors.map(value => value + 4)]),
        weights: input('float32', [...edgeWeights, ...edgeWeights])
      },
      predictorCount: 1,
      predictors,
      response,
      output: {
        table: direct.view,
        summary: output('float32', 32).view,
        status: {code: directStatus.view}
      }
    })
  );
  context.run();
  expect(Array.from(await readUint32(panelStatus.buffer, 1))).toEqual([0]);
  expect(Array.from(await readUint32(directStatus.buffer, 1))).toEqual([0]);
  close(
    await readFloat32(panel.buffer, 12),
    Array.from(await readFloat32(direct.buffer, 12)),
    1e-3
  );
  context.destroy();
});

it('statistical expansion matches independent esda, mgwr, statsmodels and spreg fixtures', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const rows = 16;
  const positions = input('float32x2', reference.positions);
  const predictors = input('float32', reference.predictors);
  const response = input('float32', reference.response);
  const weights = {
    offsets: input('uint32', reference.local.offsets),
    neighbors: input('uint32', reference.local.neighbors),
    weights: input('float32', reference.local.weights)
  };
  const losh = output('float32', rows);
  const geary = output('float32', rows);
  graph.add(
    new GPULocalSpatialHeteroscedasticity({
      weights,
      values: response,
      output: {statistic: losh.view, status: output('uint32', rows).view}
    })
  );
  graph.add(
    new GPUMultivariateLocalGeary({
      weights,
      values: input('float32', reference.local.multivariateValues),
      variableCount: 2,
      output: {statistic: geary.view, status: output('uint32', rows).view}
    })
  );
  const cross = output('float32', 3);
  const spaceTime = output('float32', 16);
  const pairInputs = {
    positions,
    values: response,
    secondValues: input('float32', reference.pairs.secondValues),
    parameters: input('float32', reference.pairs.parameters),
    lagCount: 3
  };
  graph.add(
    new GPUCrossVariogram({
      ...pairInputs,
      output: {
        semivariances: cross.view,
        pairCounts: output('uint32', 3).view,
        status: output('uint32', 3).view
      }
    })
  );
  graph.add(
    new GPUSpaceTimeVariogram({
      ...pairInputs,
      times: input('float32', reference.pairs.times),
      // The 4 x 4 lattice spans [0, 3]; the maximum lags are the fixture's [3.3, 4.5].
      parameters: input('float32', [0, 0, 3, 3, ...reference.pairs.parameters, 0, 0]),
      gridSize: [4, 4],
      temporalLagCount: 3,
      output: {
        semivariances: spaceTime.view,
        pairCounts: output('uint32', 16).view,
        status: output('uint32', 16).view
      }
    })
  );
  const marked = output('float32', 12);
  const markedCounts = output('uint32', 3);
  graph.add(
    new GPUMarkedPointStatistics({
      positions,
      types: input('uint32', reference.marked.types),
      typePair: [0, 1],
      marks: input('float32', reference.marked.marks),
      intensities: input('float32', reference.marked.intensities),
      parameters: input('float32', reference.marked.parameters),
      radiusCount: 3,
      output: {
        statistics: marked.view,
        pairCounts: markedCounts.view,
        status: output('uint32', 3).view
      }
    })
  );
  const generalized = reference.generalized.models.map(model => {
    const coefficients = output('float32', rows * 2);
    const status = output('uint32', rows);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        id: `reference-${model.family}`,
        family: model.family === 'poisson' ? 'poisson' : 'logistic',
        positions,
        predictors,
        response: input('float32', model.response),
        offset: model.offset ? input('float32', model.offset) : undefined,
        mask: input('uint32', reference.generalized.mask),
        predictorCount: 1,
        parameters: input('float32', [reference.generalized.bandwidth, 1e-5]),
        maximumIterations: 64,
        output: {
          coefficients: coefficients.view,
          status: {codes: status.view, iterationCount: output('uint32', rows).view},
          fitted: output('float32', rows).view
        }
      })
    );
    return {coefficients, status, model};
  });
  const multiscale = output('float32', rows * 2);
  const multiscaleStatus = output('uint32', rows);
  const standardErrors = output('float32', rows * 2);
  graph.add(
    new GPUMultiscaleGeographicallyWeightedRegression({
      positions,
      predictors,
      response,
      predictorCount: 1,
      parameters: input('float32', [1e-5, ...reference.multiscale.bandwidths]),
      noiseVariance: input('float32', [0.25]),
      maximumIterations: 32,
      output: {
        coefficients: multiscale.view,
        fitted: output('float32', rows).view,
        status: {codes: multiscaleStatus.view, iterationCount: output('uint32', 1).view},
        conditionalStandardErrors: standardErrors.view
      }
    })
  );
  const gridCoefficients = output('float32', rows * 2);
  const gridBandwidths = output('float32', 2);
  const gridStatus = output('uint32', rows);
  graph.add(
    new GPUMultiscaleGeographicallyWeightedRegression({
      id: 'grid-backfit',
      positions,
      predictors,
      response,
      predictorCount: 1,
      parameters: input('float32', [1e-5, ...reference.multiscale.bandwidths]),
      bandwidthCandidates: input('float32', reference.multiscale.gridCandidates),
      bandwidthCandidateCount: 3,
      maximumIterations: 32,
      output: {
        coefficients: gridCoefficients.view,
        fitted: output('float32', rows).view,
        status: {codes: gridStatus.view, iterationCount: output('uint32', 1).view},
        selectedBandwidths: gridBandwidths.view
      }
    })
  );
  const maskedMultiscale = output('float32', rows * 2);
  const maskedMultiscaleStatus = output('uint32', rows);
  graph.add(
    new GPUMultiscaleGeographicallyWeightedRegression({
      id: 'masked-backfit',
      positions,
      predictors,
      response,
      predictorCount: 1,
      parameters: input('float32', [1e-5, ...reference.multiscale.bandwidths]),
      mask: input('uint32', reference.generalized.mask),
      maximumIterations: 32,
      output: {
        coefficients: maskedMultiscale.view,
        fitted: output('float32', rows).view,
        status: {codes: maskedMultiscaleStatus.view, iterationCount: output('uint32', 1).view}
      }
    })
  );
  const panel = output('float32', 12);
  const panelStatus = output('uint32', 1);
  graph.add(
    new GPUSpatialPanelRegression({
      weights,
      periodCount: 2,
      predictorCount: 1,
      predictors: input('float32', reference.panel.predictors),
      response: input('float32', reference.panel.response),
      output: {
        table: panel.view,
        summary: output('float32', 32).view,
        status: {code: panelStatus.view}
      }
    })
  );
  const regimes = output('float32', 4);
  graph.add(
    new GPUSpatialRegimeRegression({
      predictors,
      response,
      predictorCount: 1,
      regimes: input('uint32', reference.regimes.regimes),
      regimeCount: 2,
      output: {
        coefficients: regimes.view,
        fitted: output('float32', rows).view,
        status: output('uint32', 2).view,
        counts: output('uint32', 2).view
      }
    })
  );
  const recipeLosh = output('float32', rows);
  const recipeStatus = output('uint32', rows);
  const recipe = addLocalVariabilityRecipe(graph, {
    weights,
    values: response,
    parameters: input('float32', Array.from(getGPUSpatialAutocorrelationParameterValues({}))),
    outputs: {losh: recipeLosh.view, loshStatus: recipeStatus.view}
  });
  expect(recipe.inference.losh).toBe('descriptive-only');
  context.run();
  close(await readFloat32(losh.buffer, rows), reference.local.losh);
  close(await readFloat32(geary.buffer, rows), reference.local.geary);
  close(
    await readFloat32(cross.buffer, 3),
    reference.pairs.cross.map(value => value ?? NaN)
  );
  // The fixture is in the former 3 x 3 layout. No two stations coincide and no two times are
  // equal, so the zero-lag classes are empty and bin (s, t) is bin (s + 1, t + 1) of the 4 x 4 grid.
  const expectedSpaceTime = new Array<number>(16).fill(NaN);
  reference.pairs.spaceTime.forEach((value, bin) => {
    expectedSpaceTime[(Math.floor(bin / 3) + 1) * 4 + (bin % 3) + 1] = value ?? NaN;
  });
  close(await readFloat32(spaceTime.buffer, 16), expectedSpaceTime);
  close(
    await readFloat32(marked.buffer, 12),
    reference.marked.statistics,
    reference.provenance.tolerance
  );
  expect(Array.from(await readUint32(markedCounts.buffer, 3))).toEqual(reference.marked.counts);
  close(
    await readFloat32(maskedMultiscale.buffer, rows * 2),
    reference.multiscale.maskedCoefficients.map(value => value ?? NaN),
    reference.provenance.tolerance
  );
  expect(Array.from(await readUint32(maskedMultiscaleStatus.buffer, rows))).toEqual(
    reference.generalized.mask.map(value => (value ? 0 : 1))
  );
  for (const {coefficients, status, model} of generalized) {
    close(
      await readFloat32(coefficients.buffer, rows * 2),
      model.coefficients.map(value => value ?? NaN),
      reference.provenance.tolerance
    );
    expect(Array.from(await readUint32(status.buffer, rows))).toEqual(
      reference.generalized.mask.map(value => (value ? 0 : 1))
    );
  }
  expect(Array.from(await readUint32(multiscaleStatus.buffer, rows))).toEqual(
    new Array(rows).fill(0)
  );
  close(
    await readFloat32(multiscale.buffer, rows * 2),
    reference.multiscale.coefficients,
    reference.provenance.tolerance
  );
  close(
    await readFloat32(standardErrors.buffer, rows * 2),
    reference.multiscale.conditionalStandardErrors
  );
  expect(Array.from(await readUint32(gridStatus.buffer, rows))).toEqual(new Array(rows).fill(0));
  close(
    await readFloat32(gridCoefficients.buffer, rows * 2),
    reference.multiscale.gridCoefficients,
    reference.provenance.tolerance
  );
  close(await readFloat32(gridBandwidths.buffer, 2), reference.multiscale.gridBandwidths);
  expect(Array.from(await readUint32(panelStatus.buffer, 1))).toEqual([0]);
  const panelTable = await readFloat32(panel.buffer, 12);
  close(
    Float32Array.from([panelTable[0], panelTable[4], panelTable[8]]),
    reference.panel.coefficients,
    reference.provenance.tolerance
  );
  close(await readFloat32(regimes.buffer, 4), reference.regimes.coefficients);
  close(await readFloat32(recipeLosh.buffer, rows), reference.local.losh);
  expect(Array.from(await readUint32(recipeStatus.buffer, rows))).toEqual(new Array(rows).fill(0));
  context.destroy();
});

it('upstream overflow, singular fits and constant-variable failures remain explicit', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const rows = 4;
  const weights = {
    offsets: input('uint32', [0, 1, 2, 3, 4]),
    neighbors: input('uint32', [1, 2, 3, 0]),
    weights: input('float32', [1, 1, 1, 1]),
    status: {overflow: input('uint32', [1])}
  };
  const loshStatus = output('uint32', rows);
  const gearyStatus = output('uint32', rows);
  const constantStatus = output('uint32', rows);
  graph.add(
    new GPULocalSpatialHeteroscedasticity({
      weights,
      values: input('float32', [1, 2, 3, 4]),
      output: {statistic: output('float32', rows).view, status: loshStatus.view}
    })
  );
  graph.add(
    new GPUMultivariateLocalGeary({
      weights,
      values: input('float32', [1, 2, 3, 4]),
      variableCount: 1,
      output: {statistic: output('float32', rows).view, status: gearyStatus.view}
    })
  );
  graph.add(
    new GPUMultivariateLocalGeary({
      id: 'constant-geary',
      weights: {...weights, status: undefined},
      values: input('float32', [1, 1, 1, 1]),
      variableCount: 1,
      output: {statistic: output('float32', rows).view, status: constantStatus.view}
    })
  );
  const positions = input('float32x2', new Array(rows * 2).fill(0));
  const response = input('float32', [1, 2, 3, 4]);
  const parameters = input('float32', [1, 1e-5, 1]);
  const generalizedStatus = output('uint32', rows);
  const multiscaleStatus = output('uint32', rows);
  graph.add(
    new GPUGeneralizedGeographicallyWeightedRegression({
      family: 'poisson',
      positions,
      predictors: input('float32', [1, 1, 1, 1]),
      response,
      predictorCount: 1,
      parameters,
      output: {
        coefficients: output('float32', 8).view,
        fitted: output('float32', rows).view,
        status: {codes: generalizedStatus.view, iterationCount: output('uint32', rows).view}
      }
    })
  );
  graph.add(
    new GPUMultiscaleGeographicallyWeightedRegression({
      positions,
      predictors: input('float32', [0, 0, 0, 0]),
      response,
      predictorCount: 1,
      parameters,
      maximumIterations: 2,
      output: {
        coefficients: output('float32', 8).view,
        fitted: output('float32', rows).view,
        status: {codes: multiscaleStatus.view, iterationCount: output('uint32', 1).view}
      }
    })
  );
  context.run();
  expect(Array.from(await readUint32(loshStatus.buffer, rows))).toEqual([3, 3, 3, 3]);
  expect(Array.from(await readUint32(gearyStatus.buffer, rows))).toEqual([3, 3, 3, 3]);
  expect(Array.from(await readUint32(constantStatus.buffer, rows))).toEqual([2, 2, 2, 2]);
  expect(Array.from(await readUint32(generalizedStatus.buffer, rows))).toEqual([2, 2, 2, 2]);
  expect(Array.from(await readUint32(multiscaleStatus.buffer, rows))).toEqual([2, 2, 2, 2]);
  context.destroy();
});
