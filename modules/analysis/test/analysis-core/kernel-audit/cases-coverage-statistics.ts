// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as spatial from '../../../src/gpu-spatial-analysis/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

function folds({view}: KernelAuditContext) {
  return {
    positions: view('float32x2', 65536),
    blocks: view('uint32', 65536),
    mask: view('uint32', 65536),
    foldCount: 64,
    seed: 0xffffffff,
    bufferDistance: 2,
    output: {
      folds: view('uint32', 65536),
      training: view('uint32', 65536 * 64),
      testing: view('uint32', 65536 * 64),
      counts: view('uint32', 192),
      status: view('uint32', 64)
    }
  };
}

/** Maximum declared dimensions and all optional outputs; shader compilation only. */
export const COVERAGE_STATISTICS_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUMultiscaleGeographicallyWeightedRegression',
    name: 'maximum-full-backfit-inference',
    build({graph, view}) {
      graph.add(
        new spatial.GPUMultiscaleGeographicallyWeightedRegression({
          positions: view('float32x2', 64),
          predictors: view('float32', 64 * 7),
          predictorCount: 7,
          response: view('float32', 64),
          parameters: view('float32', 9),
          mask: view('uint32', 64),
          maximumIterations: 128,
          bandwidthCandidates: view('float32', 8 * 32),
          bandwidthCandidateCount: 32,
          inference: {
            varianceConvention: 'smoother-df',
            output: {
              standardErrors: view('float32', 64 * 8),
              effectiveParameters: view('float32', 8),
              summary: view('float32', 4),
              status: view('uint32', 1)
            }
          },
          output: {
            coefficients: view('float32', 64 * 8),
            fitted: view('float32', 64),
            status: {codes: view('uint32', 64), iterationCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialPanelLagMaximumLikelihood',
    name: 'maximum-random-unit-hausman',
    build({graph, view, weights}) {
      graph.add(
        new spatial.GPUSpatialPanelLagMaximumLikelihood({
          weights: weights(32, 31),
          periodCount: 32,
          predictorCount: 7,
          predictors: view('float32', 1024 * 7),
          response: view('float32', 1024),
          maximumIterations: 128,
          maximumSearchIterations: 64,
          hausman: {tests: view('float32', 3), status: view('uint32', 1)},
          output: {
            coefficients: view('float32', 10),
            covariance: view('float32', 81),
            summary: view('float32', 6),
            status: view('uint32', 1),
            fitted: view('float32', 1024),
            residuals: view('float32', 1024)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeneralizedGeographicallyWeightedRegression',
    name: 'maximum-selection-inference',
    build({graph, view}) {
      graph.add(
        new spatial.GPUGeneralizedGeographicallyWeightedRegression({
          family: 'poisson',
          positions: view('float32x2', 16384),
          predictors: view('float32', 16384 * 11),
          predictorCount: 11,
          response: view('float32', 16384),
          parameters: view('float32', 2),
          mask: view('uint32', 16384),
          offset: view('float32', 16384),
          maximumIterations: 128,
          bandwidthMode: 'adaptive',
          maximumNeighborCount: 128,
          predictExcluded: true,
          bandwidthSelection: {
            bandwidths: Array.from({length: 32}, (_, index) => index + 97),
            output: {
              scores: view('float32', 32),
              bandwidth: view('float32', 1),
              status: view('uint32', 1)
            }
          },
          inference: {
            standardErrors: view('float32', 16384 * 12),
            influence: view('float32', 16384)
          },
          output: {
            coefficients: view('float32', 16384 * 12),
            fitted: view('float32', 16384),
            status: {codes: view('uint32', 16384), iterationCount: view('uint32', 16384)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSeeminglyUnrelatedRegression',
    name: 'maximum-spatial-error-gm',
    build({graph, view, weights}) {
      graph.add(
        new spatial.GPUSeeminglyUnrelatedRegression({
          equationCount: 4,
          predictorCount: 3,
          rowCount: 65536,
          predictors: view('float32', 65536 * 12),
          responses: view('float32', 65536 * 4),
          spatial: {
            weights: weights(65536),
            output: {parameters: view('float32', 4), status: view('uint32', 4)}
          },
          output: {
            coefficients: view('float32', 16),
            covariance: view('float32', 256),
            residualCovariance: view('float32', 16),
            residuals: view('float32', 65536 * 4),
            status: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialInteraction',
    name: 'maximum-bounded-doubly',
    build({graph, view}) {
      graph.add(
        new spatial.GPUSpatialInteraction({
          origins: view('uint32', 4096),
          destinations: view('uint32', 4096),
          originCount: 128,
          destinationCount: 128,
          flows: view('float32', 4096),
          covariate: view('float32', 4096),
          pairState: view('uint32', 4096),
          constraint: 'doubly',
          coefficients: [-3, 1],
          calibration: 'bounded',
          maximumCalibrationIterations: 64,
          maximumIterations: 4096,
          output: {
            predictions: view('float32', 4096),
            summary: view('float32', 4),
            status: view('uint32', 1),
            calibration: view('float32', 4),
            candidates: view('float32', 8),
            originResiduals: view('float32', 128),
            destinationResiduals: view('float32', 128)
          }
        })
      );
    }
  },
  ...(['HC0', 'HC1', 'HAC'] as const).map(method => ({
    exportName: 'GPUInstrumentalVariablesCovariance',
    name: `maximum-${method}`,
    build({graph, view, weights}: KernelAuditContext) {
      graph.add(
        new spatial.GPUInstrumentalVariablesCovariance({
          design: view('float32', 65536 * 15),
          coefficientCount: 15,
          instruments: view('float32', 65536 * 16),
          instrumentCount: 16,
          residuals: view('float32', 65536),
          method,
          ...(method === 'HAC' ? {kernel: weights(65536)} : {}),
          output: {covariance: view('float32', 256), status: view('uint32', 1)}
        })
      );
    }
  })),
  ...([2, 8] as const).map(regimeCount => ({
    exportName: 'GPURestrictedRegimeRegression',
    name: `maximum-${regimeCount}-regimes`,
    build({graph, view}: KernelAuditContext) {
      const predictorCount = 16 / regimeCount - 1;
      const common = Array.from({length: predictorCount + 1}, (_, index) => index === 0);
      const coefficients = 1 + predictorCount * regimeCount;
      graph.add(
        new spatial.GPURestrictedRegimeRegression({
          predictors: view('float32', 65536 * predictorCount),
          predictorCount,
          response: view('float32', 65536),
          regimes: view('uint32', 65536),
          regimeCount,
          common,
          mask: view('uint32', 65536),
          output: {
            coefficients: view('float32', coefficients),
            covariance: view('float32', coefficients ** 2),
            chow: view('float32', 3),
            stability: view('float32', (predictorCount + 2) * 3),
            fitted: view('float32', 65536),
            status: view('uint32', 1)
          }
        })
      );
    }
  })),
  ...(['exact', 'stochastic'] as const).map(traceMode => ({
    exportName: 'GPUSpatialPanelErrorMaximumLikelihood',
    name: `maximum-${traceMode}`,
    build({graph, view, weights}: KernelAuditContext) {
      const rows = traceMode === 'exact' ? 4096 : 65536;
      graph.add(
        new spatial.GPUSpatialPanelErrorMaximumLikelihood({
          weights: weights(rows / 1024),
          periodCount: 1024,
          diagnostics: {tests: view('float32', 6), status: view('uint32', 1)},
          predictors: view('float32', rows * 15),
          predictorCount: 15,
          response: view('float32', rows),
          maximumPower: 256,
          traceMode,
          traceProbeCount: 64,
          traceParameters: view('uint32', 1),
          tileRowCount: 64,
          output: {
            table: view('float32', 68),
            summary: view('float32', 22),
            residuals: view('float32', rows),
            status: {code: view('uint32', 1), converged: view('uint32', 1)}
          }
        })
      );
    }
  })),
  ...([2, 4] as const).map(equationCount => ({
    exportName: 'GPUSeeminglyUnrelatedRegression',
    name: `maximum-${equationCount}-equations`,
    build({graph, view}: KernelAuditContext) {
      const predictorCount = 16 / equationCount - 1;
      graph.add(
        new spatial.GPUSeeminglyUnrelatedRegression({
          equationCount,
          predictorCount,
          rowCount: 65536,
          predictors: view('float32', 65536 * predictorCount * equationCount),
          responses: view('float32', 65536 * equationCount),
          mask: view('uint32', 65536),
          output: {
            coefficients: view('float32', 16),
            covariance: view('float32', 256),
            residualCovariance: view('float32', equationCount ** 2),
            residuals: view('float32', 65536 * equationCount),
            status: view('uint32', 1)
          }
        })
      );
    }
  })),
  {
    exportName: 'GPUProbitRegression',
    name: 'maximum-masked-fisher-scoring',
    build({graph, view, weights}) {
      graph.add(
        new spatial.GPUProbitRegression({
          predictors: view('float32', 65536 * 11),
          predictorCount: 11,
          response: view('float32', 65536),
          mask: view('uint32', 65536),
          spatialDiagnostics: {
            weights: weights(65536),
            output: {tests: view('float32', 6), status: view('uint32', 1)}
          },
          maximumIterations: 256,
          tolerance: 1e-6,
          output: {
            coefficients: view('float32', 12),
            standardErrors: view('float32', 12),
            fitted: view('float32', 65536),
            marginalEffects: view('float32', 11),
            summary: view('float32', 4),
            status: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULocalCollinearity',
    name: 'maximum-binding-limited-masked',
    build({graph, view, weights, maximumRows}) {
      const rows = maximumRows(52);
      graph.add(
        new spatial.GPULocalCollinearity({
          weights: weights(rows),
          predictors: view('float32', rows * 12),
          predictorCount: 12,
          mask: view('uint32', rows),
          output: {
            varianceInflationFactors: view('float32', rows * 12),
            conditionNumbers: view('float32', rows),
            designConditionNumbers: view('float32', rows),
            varianceDecomposition: view('float32', rows * 13),
            status: view('uint32', rows)
          }
        })
      );
    }
  },
  ...(['none', 'bonferroni'] as const).map(correction => ({
    exportName: 'GPUMultivariateLocalGearyPermutationTest',
    name: `maximum-${correction}-replay`,
    build({graph, view, weights}: KernelAuditContext) {
      graph.add(
        new spatial.GPUMultivariateLocalGearyPermutationTest({
          weights: weights(256, 255),
          values: view('float32', 256 * 16),
          variableCount: 16,
          maximumPermutations: 4096,
          correction,
          tail: correction === 'none' ? 'esda-min-tail' : 'equal-tail',
          seed: [0xffffffff, 0xffffffff],
          mask: view('uint32', 256),
          output: {
            statistic: view('float32', 256),
            pValues: view('float32', 256),
            classes: view('uint32', 256),
            simulations: view('float32', 256 * 4096)
          }
        })
      );
    }
  })),
  ...(['full-rank', 'geographic-rank'] as const).map(mode => ({
    exportName: 'GPURankMarkov',
    name: `maximum-${mode}-diagnostics`,
    build({graph, view}: KernelAuditContext) {
      graph.add(
        new spatial.GPURankMarkov({
          mode,
          tieOrder: mode === 'full-rank' ? 'giddy' : 'row-id',
          values: view('float32', 64 * 65536),
          periodCount: 65536,
          mask: view('uint32', 64),
          output: {
            counts: view('uint32', 4096),
            probabilities: view('float32', 4096),
            ranks: view('uint32', 64 * 65536),
            mobility: view('float32', 65535),
            rowTotals: view('uint32', 64),
            ignored: view('uint32', 1),
            steadyStates: view('float32', 64),
            stateSparsity: view('uint32', 2),
            status: {converged: view('uint32', 1)}
          }
        })
      );
    }
  })),
  {
    exportName: 'GPUFuzzyContiguity',
    name: 'maximum-polygons-and-vertices',
    build({graph, view}) {
      graph.add(
        new spatial.GPUFuzzyContiguity({
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 65536),
            featureOffsets: view('uint32', 65537),
            polygonOffsets: view('uint32', 16385),
            ringOffsets: view('uint32', 16385)
          },
          tolerance: 1,
          output: {
            offsets: view('uint32', 65537),
            neighbors: view('uint32', 65536 * 16),
            weights: view('float32', 65536 * 16),
            distances: view('float32', 65536 * 16),
            added: view('uint32', 65536 * 16),
            featureStatus: view('uint32', 65536),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  ...([false, true] as const).map(reciprocal => ({
    exportName: 'GPUIslandAttachment',
    name: `maximum-reciprocal-${reciprocal}`,
    build({graph, view, weights}: KernelAuditContext) {
      graph.add(
        new spatial.GPUIslandAttachment({
          weights: weights(65535),
          positions: view('float32x2', 65535),
          reciprocal,
          output: {
            offsets: view('uint32', 65536),
            neighbors: view('uint32', 65535 * 10),
            weights: view('float32', 65535 * 10),
            attachment: view('uint32', 65535),
            added: view('uint32', 65535 * 10),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          }
        })
      );
    }
  })),
  ...(['scan', 'grid'] as const).map(nearestSearch => ({
    exportName: 'GPUIslandAttachment',
    name: `maximum-nearest-search-${nearestSearch}`,
    build({graph, view, weights}: KernelAuditContext) {
      graph.add(
        new spatial.GPUIslandAttachment({
          weights: weights(65535),
          positions: view('float32x2', 65535),
          nearestSearch,
          output: {
            offsets: view('uint32', 65536),
            neighbors: view('uint32', 65535 * 10),
            weights: view('float32', 65535 * 10),
            attachment: view('uint32', 65535),
            added: view('uint32', 65535 * 10),
            status: {
              overflow: view('uint32', 1),
              requiredCount: view('uint32', 1),
              uncertainCount: view('uint32', 1)
            }
          }
        })
      );
    }
  })),
  {
    exportName: 'GPUSpatialCrossValidation',
    name: 'maximum-buffered-masked-folds',
    build(context) {
      context.graph.add(new spatial.GPUSpatialCrossValidation(folds(context)));
    }
  },
  {
    exportName: 'GPUSpatialCrossValidatedOLS',
    name: 'maximum-training-only-fits',
    build(context) {
      const {graph, view} = context;
      graph.add(
        new spatial.GPUSpatialCrossValidatedOLS({
          folds: folds(context),
          predictors: view('float32', 65536 * 15),
          predictorCount: 15,
          response: view('float32', 65536),
          output: {
            coefficients: view('float32', 64 * 16),
            predictions: view('float32', 64 * 65536),
            fitStatus: view('uint32', 64),
            metrics: view('float32', 64 * 6),
            metricStatus: view('uint32', 64)
          }
        })
      );
    }
  },
  ...(['unconstrained', 'origin', 'destination', 'doubly'] as const).map(constraint => ({
    exportName: 'GPUSpatialInteraction',
    name: `maximum-${constraint}`,
    build({graph, view}: KernelAuditContext) {
      graph.add(
        new spatial.GPUSpatialInteraction({
          origins: view('uint32', 4096),
          destinations: view('uint32', 4096),
          originCount: 128,
          destinationCount: 128,
          flows: view('float32', 4096),
          covariate: view('float32', 4096),
          pairState: view('uint32', 4096),
          constraint,
          coefficients: Array.from({length: 128}, (_, index) => index / 128),
          maximumIterations: 4096,
          output: {
            predictions: view('float32', 4096),
            summary: view('float32', 4),
            status: view('uint32', 1),
            candidates: view('float32', 128 * 4),
            originResiduals: view('float32', 128),
            destinationResiduals: view('float32', 128)
          }
        })
      );
    }
  }))
];
