// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {S2Uploads, S2CaptureSpec} from './s2-lens-run';
import {createStatisticsLensInputs} from './statistics-coverage-inputs';
import {runExample as runExample0} from './fuzzy-contiguity.example';
import {runExample as runExample1} from './rank-markov.example';
import {runExample as runExample2} from './spatial-panel-error-maximum-likelihood.example';
import {runExample as runExample3} from './restricted-regime-regression.example';
import {runExample as runExample4} from './instrumental-variables-covariance.example';
import {runExample as runExample5} from './island-attachment.example';
import {runExample as runExample6} from './spatial-interaction.example';
import {runExample as runExample7} from './multivariate-local-geary-permutation-test.example';
import {runExample as runExample8} from './seemingly-unrelated-regression.example';
import {runExample as runExample9} from './spatial-cross-validation.example';
import {runExample as runExample10} from './spatial-cross-validated-ols.example';
import {runExample as runExample11} from './probit-regression.example';
import {runExample as runPanelLagExample} from './spatial-panel-lag-maximum-likelihood.example';
import {runExample as runExample12} from './local-collinearity.example';

/** Build the uploaded synthetic inputs and execute the public example declaration. */
export async function buildStatisticsCoverageScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const {input, output, weights, captures} = createStatisticsLensInputs(element, uploads);
  switch (element) {
    case 'GPUSpatialPanelLagMaximumLikelihood':
      await runPanelLagExample(graph, {
        weights,
        periodCount: 2,
        predictorCount: 1,
        predictors: input('predictors', 'float32', 48),
        response: input('response', 'float32', 48),
        maximumIterations: 128,
        tolerance: 1e-4,
        hausman: {
          tests: output('hausman', 'float32', 3),
          status: output('hausmanStatus', 'uint32', 1)
        },
        output: {
          coefficients: output('coefficients', 'float32', 4),
          covariance: output('covariance', 'float32', 9),
          summary: output('summary', 'float32', 6),
          status: output('status', 'uint32', 1),
          fitted: output('fitted', 'float32', 48),
          residuals: output('residuals', 'float32', 48)
        }
      });
      break;
    case 'GPUFuzzyContiguity':
      await runExample0(graph, {
        polygons: {
          kind: 'polygons',
          positions: input('polygons-positions', 'float32x2', 16),
          featureOffsets: input('polygons-featureOffsets', 'uint32', 5),
          polygonOffsets: input('polygons-polygonOffsets', 'uint32', 5),
          ringOffsets: input('polygons-ringOffsets', 'uint32', 5)
        },
        tolerance: 0.1,
        output: {
          offsets: output('offsets', 'uint32', 5),
          neighbors: output('neighbors', 'uint32', 12),
          weights: output('weights', 'float32', 12),
          distances: output('distances', 'float32', 12),
          featureStatus: output('featureStatus', 'uint32', 4),
          added: output('added', 'uint32', 12),
          status: {
            overflow: output('status-overflow', 'uint32', 1),
            requiredCount: output('status-requiredCount', 'uint32', 1)
          }
        }
      });
      break;
    case 'GPURankMarkov':
      await runExample1(graph, {
        mode: 'geographic-rank',
        values: input('values', 'float32', 24),
        periodCount: 3,
        output: {
          counts: output('counts', 'uint32', 64),
          probabilities: output('probabilities', 'float32', 64),
          ranks: output('ranks', 'uint32', 24),
          mobility: output('mobility', 'float32', 2)
        }
      });
      break;
    case 'GPUSpatialPanelErrorMaximumLikelihood':
      await runExample2(graph, {
        weights,
        predictors: input('predictors', 'float32', 48),
        predictorCount: 1,
        response: input('response', 'float32', 48),
        periodCount: 2,
        diagnostics: {
          tests: output('diagnosticTests', 'float32', 6),
          status: output('diagnosticStatus', 'uint32', 1)
        },
        output: {
          table: output('table', 'float32', 12),
          summary: output('summary', 'float32', 22),
          status: {code: output('status-code', 'uint32', 1)}
        }
      });
      break;
    case 'GPURestrictedRegimeRegression':
      await runExample3(graph, {
        predictors: input('predictors', 'float32', 24),
        predictorCount: 1,
        response: input('response', 'float32', 24),
        regimes: input('regimes', 'uint32', 24),
        regimeCount: 2,
        common: [false, true],
        mask: input('mask', 'uint32', 24),
        output: {
          coefficients: output('coefficients', 'float32', 3),
          covariance: output('covariance', 'float32', 9),
          chow: output('chow', 'float32', 3),
          stability: output('stability', 'float32', 9),
          fitted: output('fitted', 'float32', 24),
          status: output('status', 'uint32', 1)
        }
      });
      break;
    case 'GPUInstrumentalVariablesCovariance':
      await runExample4(graph, {
        design: input('design', 'float32', 48),
        coefficientCount: 2,
        instruments: input('instruments', 'float32', 72),
        instrumentCount: 3,
        residuals: input('residuals', 'float32', 24),
        method: 'HAC',
        kernel: weights,
        output: {
          covariance: output('covariance', 'float32', 9),
          status: {code: output('status', 'uint32', 1)}
        }
      });
      break;
    case 'GPUIslandAttachment':
      await runExample5(graph, {
        weights,
        positions: input('positions', 'float32x2', 24),
        output: {
          offsets: output('offsets', 'uint32', 25),
          neighbors: output('neighbors', 'uint32', 128),
          weights: output('weights', 'float32', 128),
          attachment: output('attachment', 'uint32', 24),
          added: output('added', 'uint32', 128),
          status: {
            overflow: output('status-overflow', 'uint32', 1),
            requiredCount: output('status-requiredCount', 'uint32', 1)
          }
        }
      });
      break;
    case 'GPUSpatialInteraction':
      await runExample6(graph, {
        origins: input('origins', 'uint32', 24),
        destinations: input('destinations', 'uint32', 24),
        originCount: 4,
        destinationCount: 6,
        flows: input('flows', 'float32', 24),
        covariate: input('covariate', 'float32', 24),
        pairState: input('pairState', 'uint32', 24),
        constraint: 'doubly',
        coefficients: [-1, 1],
        calibration: 'bounded',
        output: {
          predictions: output('predictions', 'float32', 24),
          summary: output('summary', 'float32', 4),
          status: output('status', 'uint32', 1),
          candidates: output('candidates', 'float32', 8),
          calibration: output('calibration', 'float32', 4),
          originResiduals: output('originResiduals', 'float32', 4),
          destinationResiduals: output('destinationResiduals', 'float32', 6)
        }
      });
      break;
    case 'GPUMultivariateLocalGearyPermutationTest':
      await runExample7(graph, {
        weights,
        values: input('values', 'float32', 48),
        variableCount: 2,
        permutationCount: 19,
        correction: 'bonferroni',
        tail: 'esda-min-tail',
        mask: input('mask', 'uint32', 24),
        output: {
          statistic: output('statistic', 'float32', 24),
          adjustedPValues: output('pValues', 'float32', 24),
          classes: output('classes', 'uint32', 24)
        }
      });
      break;
    case 'GPUSeeminglyUnrelatedRegression':
      await runExample8(graph, {
        predictors: input('predictors', 'float32', 48),
        responses: input('responses', 'float32', 48),
        equationCount: 2,
        predictorCount: 1,
        rowCount: 24,
        spatial: {
          weights,
          output: {
            parameters: output('spatialParameters', 'float32', 2),
            status: output('spatialStatus', 'uint32', 2)
          }
        },
        output: {
          coefficients: output('coefficients', 'float32', 4),
          covariance: output('covariance', 'float32', 16),
          residualCovariance: output('residualCovariance', 'float32', 4),
          residuals: output('residuals', 'float32', 48),
          status: output('status', 'uint32', 1)
        }
      });
      break;
    case 'GPUSpatialCrossValidation':
      await runExample9(graph, {
        positions: input('positions', 'float32x2', 24),
        blocks: input('blocks', 'uint32', 24),
        mask: input('mask', 'uint32', 24),
        foldCount: 3,
        seed: 17,
        bufferDistance: 0.25,
        output: {
          folds: output('folds', 'uint32', 24),
          training: output('training', 'uint32', 72),
          testing: output('testing', 'uint32', 72),
          counts: output('counts', 'uint32', 9),
          status: output('status', 'uint32', 3)
        }
      });
      break;
    case 'GPUSpatialCrossValidatedOLS':
      await runExample10(graph, {
        folds: {
          positions: input('folds-positions', 'float32x2', 24),
          blocks: input('folds-blocks', 'uint32', 24),
          foldCount: 3,
          seed: 17,
          output: {
            folds: output('folds-folds', 'uint32', 24),
            training: output('folds-training', 'uint32', 72),
            testing: output('folds-testing', 'uint32', 72),
            counts: output('folds-counts', 'uint32', 9),
            status: output('folds-status', 'uint32', 3)
          }
        },
        predictors: input('predictors', 'float32', 24),
        predictorCount: 1,
        response: input('response', 'float32', 24),
        output: {
          coefficients: output('coefficients', 'float32', 6),
          predictions: output('predictions', 'float32', 72),
          fitStatus: output('fitStatus', 'uint32', 3),
          metrics: output('metrics', 'float32', 18),
          metricStatus: output('metricStatus', 'uint32', 3)
        }
      });
      break;
    case 'GPUProbitRegression':
      await runExample11(graph, {
        predictors: input('predictors', 'float32', 48),
        predictorCount: 2,
        response: input('response', 'float32', 24),
        mask: input('mask', 'uint32', 24),
        spatialDiagnostics: {
          weights,
          output: {
            tests: output('diagnosticTests', 'float32', 6),
            status: output('diagnosticStatus', 'uint32', 1)
          }
        },
        output: {
          coefficients: output('coefficients', 'float32', 3),
          standardErrors: output('standardErrors', 'float32', 3),
          fitted: output('fitted', 'float32', 24),
          marginalEffects: output('marginalEffects', 'float32', 2),
          summary: output('summary', 'float32', 4),
          status: output('status', 'uint32', 1)
        }
      });
      break;
    case 'GPULocalCollinearity':
      await runExample12(graph, {
        weights,
        predictors: input('predictors', 'float32', 48),
        predictorCount: 2,
        mask: input('mask', 'uint32', 24),
        output: {
          varianceInflationFactors: output('varianceInflationFactors', 'float32', 48),
          conditionNumbers: output('conditionNumbers', 'float32', 24),
          designConditionNumbers: output('designConditionNumbers', 'float32', 24),
          varianceDecomposition: output('varianceDecomposition', 'float32', 72),
          status: output('status', 'uint32', 24)
        }
      });
      break;
    default:
      throw new Error(`Unknown statistics example: ${element}`);
  }
  return captures;
}
