// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {GPUSpatialCrossValidatedOLS} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPURankMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';

import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';

import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';

it('statistics coverage kernels parse and preserve default storage binding limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `coverage-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 25),
    neighbors: view('uint32', 96),
    weights: view('float32', 96),
    status: {overflow: view('uint32', 1)}
  };
  const contributors = [
    new GPUSpatialCrossValidation({
      positions: view('float32x2', 24),
      blocks: view('uint32', 24),
      mask: view('uint32', 24),
      foldCount: 3,
      bufferDistance: 2,
      output: {
        folds: view('uint32', 24),
        training: view('uint32', 72),
        testing: view('uint32', 72),
        counts: view('uint32', 9),
        status: view('uint32', 3)
      }
    })
  ];
  const nodes = contributors.flatMap(contributor => [...contributor.getCommandNodes(graph)]);
  for (const method of ['HC0', 'HC1', 'HAC'] as const) {
    nodes.push(
      ...new GPUInstrumentalVariablesCovariance({
        id: `covariance-${method}`,
        design: view('float32', 48),
        coefficientCount: 2,
        instruments: view('float32', 72),
        instrumentCount: 3,
        residuals: view('float32', 24),
        method,
        ...(method === 'HAC' ? {kernel: weights} : {}),
        output: {covariance: view('float32', 9), status: view('uint32', 1)}
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...new GPUSpatialTwoStageLeastSquares({
        id: `two-stage-${method}`,
        weights,
        predictors: view('float32', 24),
        predictorCount: 1,
        response: view('float32', 24),
        robust: {
          method,
          ...(method === 'HAC' ? {kernel: weights} : {}),
          output: {covariance: view('float32', 9), status: view('uint32', 1)}
        },
        output: {
          table: view('float32', 12),
          summary: view('float32', 16),
          status: {code: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    );
  }
  for (const mode of ['full-rank', 'geographic-rank'] as const)
    nodes.push(
      ...new GPURankMarkov({
        id: `markov-${mode}`,
        mode,
        values: view('float32', 24),
        periodCount: 3,
        output: {
          counts: view('uint32', 64),
          probabilities: view('float32', 64),
          ranks: view('uint32', 24),
          mobility: view('float32', 2)
        }
      }).getCommandNodes(graph)
    );
  nodes.push(
    ...new GPUProbitRegression({
      spatialDiagnostics: {weights, output: {tests: view('float32', 6), status: view('uint32', 1)}},
      predictors: view('float32', 48),
      predictorCount: 2,
      response: view('float32', 24),
      mask: view('uint32', 24),
      output: {
        coefficients: view('float32', 3),
        standardErrors: view('float32', 3),
        fitted: view('float32', 24),
        marginalEffects: view('float32', 2),
        summary: view('float32', 4),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPULocalCollinearity({
      weights,
      predictors: view('float32', 48),
      predictorCount: 2,
      mask: view('uint32', 24),
      output: {
        varianceInflationFactors: view('float32', 48),
        designConditionNumbers: view('float32', 24),
        varianceDecomposition: view('float32', 72),
        conditionNumbers: view('float32', 24),
        status: view('uint32', 24)
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUMultivariateLocalGearyPermutationTest({
      weights,
      values: view('float32', 48),
      variableCount: 2,
      maximumPermutations: 19,
      correction: 'bonferroni',
      mask: view('uint32', 24),
      output: {
        statistic: view('float32', 24),
        pValues: view('float32', 24),
        classes: view('uint32', 24)
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUIslandAttachment({
      weights,
      positions: view('float32x2', 24),
      output: {
        offsets: view('uint32', 25),
        neighbors: view('uint32', 128),
        weights: view('float32', 128),
        attachment: view('uint32', 24),
        added: view('uint32', 128),
        status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
      }
    }).getCommandNodes(graph)
  );
  for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const)
    nodes.push(
      ...new GPUSpatialInteraction({
        id: `interaction-${constraint}`,
        origins: view('uint32', 24),
        destinations: view('uint32', 24),
        originCount: 4,
        destinationCount: 6,
        flows: view('float32', 24),
        covariate: view('float32', 24),
        pairState: view('uint32', 24),
        constraint,
        coefficients: [-1, 1],
        calibration: 'bounded',
        maximumCalibrationIterations: 2,
        output: {
          predictions: view('float32', 24),
          summary: view('float32', 4),
          status: view('uint32', 1),
          candidates: view('float32', 8),
          calibration: view('float32', 4),
          originResiduals: view('float32', 4),
          destinationResiduals: view('float32', 6)
        }
      }).getCommandNodes(graph)
    );
  nodes.push(
    ...new GPURestrictedRegimeRegression({
      predictors: view('float32', 24),
      predictorCount: 1,
      response: view('float32', 24),
      regimes: view('uint32', 24),
      regimeCount: 2,
      common: [false, true],
      mask: view('uint32', 24),
      output: {
        coefficients: view('float32', 3),
        covariance: view('float32', 9),
        chow: view('float32', 3),
        stability: view('float32', 9),
        fitted: view('float32', 24),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUSeeminglyUnrelatedRegression({
      spatial: {weights, output: {parameters: view('float32', 2), status: view('uint32', 2)}},
      predictors: view('float32', 48),
      responses: view('float32', 48),
      equationCount: 2,
      predictorCount: 1,
      rowCount: 24,
      output: {
        coefficients: view('float32', 4),
        covariance: view('float32', 16),
        residualCovariance: view('float32', 4),
        residuals: view('float32', 48),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUSpatialPanelErrorMaximumLikelihood({
      weights,
      predictors: view('float32', 48),
      predictorCount: 1,
      response: view('float32', 48),
      periodCount: 2,
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUFuzzyContiguity({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 16),
        featureOffsets: view('uint32', 5),
        polygonOffsets: view('uint32', 5),
        ringOffsets: view('uint32', 5)
      },
      tolerance: 0.1,
      output: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 12),
        weights: view('float32', 12),
        distances: view('float32', 12),
        featureStatus: view('uint32', 4),
        added: view('uint32', 12),
        status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUSpatialCrossValidatedOLS({
      folds: {
        positions: view('float32x2', 24),
        blocks: view('uint32', 24),
        foldCount: 3,
        output: {
          folds: view('uint32', 24),
          training: view('uint32', 72),
          testing: view('uint32', 72),
          counts: view('uint32', 9),
          status: view('uint32', 3)
        }
      },
      predictors: view('float32', 24),
      predictorCount: 1,
      response: view('float32', 24),
      output: {
        coefficients: view('float32', 6),
        predictions: view('float32', 72),
        fitStatus: view('uint32', 3),
        metrics: view('float32', 18),
        metricStatus: view('uint32', 3)
      }
    }).getCommandNodes(graph)
  );
  for (const source of getGPUCommandNodeKernelSources(nodes)) {
    expect(new WgslReflect(source.source).storage.length).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
  }
  device.destroy();
});

it('local collinearity rejects aliased writable outputs through graph validation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `alias-${serial++}`, format, length);
  const shared = view('float32', 4);
  try {
    expect(() => {
      graph.add(
        new GPULocalCollinearity({
          predictors: view('float32', 4),
          predictorCount: 1,
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', 12),
            weights: view('float32', 12)
          },
          output: {
            varianceInflationFactors: shared,
            conditionNumbers: shared,
            status: view('uint32', 4)
          }
        })
      );
      graph.compile();
    }).toThrow();
  } finally {
    device.destroy();
  }
});
