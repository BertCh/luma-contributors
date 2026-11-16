// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {GPUEnsembleScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type ViewFactory = <Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2'>(
  format: Format,
  length: number
) => GraphDataView<Format>;

function createWeights(view: ViewFactory, entities = 2, includeOverflow = true) {
  return {
    offsets: view('uint32', entities + 1),
    neighbors: view('uint32', entities),
    weights: view('float32', entities),
    status: includeOverflow ? {overflow: view('uint32', 1)} : undefined
  };
}

function createLikelihoodOutput(view: ViewFactory, rows: number) {
  return {
    table: view('float32', 12),
    summary: view('float32', 22),
    residuals: view('float32', rows),
    powerTraceStandardErrors: view('float32', 2),
    status: {code: view('uint32', 1), converged: view('uint32', 1)}
  };
}

const factories: [string, (view: ViewFactory) => GPUCommandNodeProducer][] = [
  [
    'OLS',
    view =>
      new GPUOrdinaryLeastSquares({
        predictorCount: 1,
        predictors: view('float32', 4),
        response: view('float32', 4),
        mask: view('uint32', 4),
        parameters: view('float32', 1),
        output: {
          coefficients: view('float32', 2),
          standardErrors: view('float32', 2),
          tStatistics: view('float32', 2),
          summary: view('float32', 22),
          residuals: view('float32', 4),
          fitted: view('float32', 4),
          status: {code: view('uint32', 1)}
        }
      })
  ],
  ...[false, true].map(
    spatial =>
      [
        `SUR spatial=${spatial}`,
        (view: ViewFactory) =>
          new GPUSeeminglyUnrelatedRegression({
            rowCount: 4,
            equationCount: 2,
            predictorCount: 1,
            predictors: view('float32', 8),
            responses: view('float32', 8),
            ...(spatial
              ? {
                  spatial: {
                    weights: createWeights(view, 4),
                    output: {parameters: view('float32', 2), status: view('uint32', 2)}
                  }
                }
              : {mask: view('uint32', 4), initialResiduals: view('float32', 8)}),
            output: {
              coefficients: view('float32', 4),
              covariance: view('float32', 16),
              residualCovariance: view('float32', 4),
              residuals: view('float32', 8),
              status: view('uint32', 1)
            }
          })
      ] as [string, (view: ViewFactory) => GPUCommandNodeProducer]
  ),
  [
    'error ML',
    view =>
      new GPUSpatialErrorMaximumLikelihood({
        predictorCount: 1,
        maximumPower: 2,
        traceMode: 'stochastic',
        predictors: view('float32', 2),
        response: view('float32', 2),
        weights: createWeights(view, 2, false),
        logDeterminant: 'exact',
        eigenvalues: view('float32', 2),
        traceParameters: view('uint32', 1),
        output: createLikelihoodOutput(view, 2)
      })
  ],
  [
    'panel-error ML',
    view =>
      new GPUSpatialPanelErrorMaximumLikelihood({
        periodCount: 2,
        predictorCount: 1,
        maximumPower: 2,
        traceMode: 'stochastic',
        predictors: view('float32', 4),
        response: view('float32', 4),
        weights: createWeights(view),
        logDeterminant: 'exact',
        eigenvalues: view('float32', 4),
        traceParameters: view('uint32', 1),
        output: createLikelihoodOutput(view, 4),
        diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
      })
  ],
  ...[false, true].map(
    tiled =>
      [
        `panel-lag tiled=${tiled}`,
        (view: ViewFactory) =>
          new GPUSpatialPanelLagMaximumLikelihood({
            periodCount: 2,
            predictorCount: 1,
            predictors: view('float32', 4),
            response: view('float32', 4),
            weights: createWeights(view),
            ...(tiled
              ? {effects: 'unit' as const, eigenvalues: view('float32', 2)}
              : {hausman: {tests: view('float32', 3), status: view('uint32', 1)}}),
            output: {
              coefficients: view('float32', 4),
              covariance: view('float32', 9),
              summary: view('float32', 6),
              status: view('uint32', 1),
              fitted: view('float32', 4),
              residuals: view('float32', 4)
            }
          })
      ] as [string, (view: ViewFactory) => GPUCommandNodeProducer]
  ),
  [
    'nested validation',
    view =>
      new GPUNestedInterpolationValidation({
        positions: view('float32x2', 4),
        response: view('float32', 4),
        training: view('uint32', 8),
        testing: view('uint32', 8),
        innerBlocks: view('uint32', 4),
        foldCount: 2,
        innerFoldCount: 2,
        neighborCount: 4,
        candidates: [{mode: 'rbf', epsilon: 1}],
        output: {
          selected: view('uint32', 2),
          scores: view('float32', 6),
          predictions: view('float32', 8),
          metrics: view('float32', 12),
          status: view('uint32', 2)
        }
      })
  ],
  [
    'ensemble scores',
    view =>
      new GPUEnsembleScores({
        memberCount: 2,
        dimension: 2,
        ensemble: view('float32', 4),
        observations: view('float32', 2),
        variogramPairs: view('uint32x2', 1),
        variogramWeights: view('float32', 1),
        output: {
          crps: view('float32', 2),
          energyScore: view('float32', 1),
          variogramScore: view('float32', 1)
        }
      })
  ]
];

for (const [name, createContributor] of factories) {
  it(`${name} preflights STORAGE usage of every bound caller view without mutating the graph`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const views: GraphDataView[] = [];
    const view: ViewFactory = (format, length) => {
      const result = createTransientView(
        graph,
        `storage-preflight-${views.length}`,
        format,
        length
      );
      views.push(result);
      return result;
    };
    try {
      const contributor = createContributor(view);
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const addNode = vi.spyOn(graph, 'add');
      for (const callerView of views) {
        const originalUsage = callerView.buffer.usage;
        Object.defineProperty(callerView.buffer, 'usage', {value: Buffer.COPY_DST});
        expect(() => contributor.getCommandNodes(graph), callerView.buffer.id).toThrow(
          /usage must include Buffer.STORAGE/
        );
        expect(allocate).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
        expect(addNode).not.toHaveBeenCalled();
        Object.defineProperty(callerView.buffer, 'usage', {value: originalUsage});
      }
    } finally {
      device.destroy();
    }
  });
}

for (const metadata of ['foreign-non-storage', 'aliases-output'] as const) {
  for (const chunked of [false, true]) {
    it(`error likelihood ignores unused weights distance metadata: ${metadata}, chunked=${chunked}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const otherGraph = new GPUCommandGraph(device);
      let serial = 0;
      const view: ViewFactory = (format, length) =>
        createTransientView(graph, `unused-distance-${serial++}`, format, length);
      try {
        const output = createLikelihoodOutput(view, 2);
        const distances =
          metadata === 'aliases-output'
            ? output.residuals
            : createTransientView(otherGraph, 'foreign-distances', 'float32', 2);
        if (metadata === 'foreign-non-storage')
          Object.defineProperty(distances.buffer, 'usage', {value: Buffer.COPY_DST});
        const weights = {...createWeights(view, 2, false), distances};
        const contributor = new GPUSpatialErrorMaximumLikelihood({
          predictorCount: 1,
          maximumPower: 2,
          predictors: view('float32', 2),
          response: view('float32', 2),
          weights: chunked ? {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]} : weights,
          output
        });
        expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      } finally {
        device.destroy();
      }
    });
  }
}

for (const tiled of [false, true]) {
  it(`panel-lag ignores unused distance metadata aliasing an output, tiled=${tiled}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view: ViewFactory = (format, length) =>
      createTransientView(graph, `panel-metadata-${serial++}`, format, length);
    try {
      const coefficients = view('float32', 4);
      const distances = graph.createDataView(coefficients.buffer, {format: 'float32', length: 2});
      const contributor = new GPUSpatialPanelLagMaximumLikelihood({
        periodCount: 2,
        predictorCount: 1,
        predictors: view('float32', 4),
        response: view('float32', 4),
        weights: {...createWeights(view), distances},
        ...(tiled ? {effects: 'unit' as const, eigenvalues: view('float32', 2)} : {}),
        output: {
          coefficients,
          covariance: view('float32', 9),
          summary: view('float32', 6),
          status: view('uint32', 1),
          fitted: view('float32', 4),
          residuals: view('float32', 4)
        }
      });
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}
