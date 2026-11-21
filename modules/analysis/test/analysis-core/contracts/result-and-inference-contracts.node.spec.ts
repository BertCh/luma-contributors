// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, expectTypeOf, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createGPURegressionModelResult,
  validateGPURecipeStatusPort,
  validateGPUStatusPort,
  type GPUAnalyticAlternative,
  type GPUBoundedResultStatusPort,
  type GPUCompletenessStatusPort,
  type GPUInferenceAlternative,
  type GPUInferenceMetadata,
  type GPUNetworkTopology,
  type GPUNodedNetworkTopologyPort,
  type GPUQuantileLISAAlternative,
  type GPURecipeStageStatusPort,
  type GPURegressionModelResult,
  type GPURegressionModelResultInput,
  type GPURegressionRegularization,
  type GPUStatusScalar,
  type GPUUncertaintyStatusPort
} from '@luma-contributors/analysis/analysis-core';
import type {GPUNetworkNodingOutput} from '../../../src/gpu-network/network-noding/gpu-network-noding';
import {
  getGPUGeographicallyTemporallyWeightedRegressionModelResult,
  type GPUGeographicallyTemporallyWeightedRegressionModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import {
  getGPUSpatialPanelErrorMaximumLikelihoodModelResult,
  type GPUSpatialPanelErrorMaximumLikelihoodModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-panel-error-maximum-likelihood-model-result';

function createView<Format extends 'uint32' | 'float32'>(format: Format, length = 1) {
  return {format, length} as GraphDataView<Format>;
}

describe('result and inference contracts', () => {
  it('distinguishes remaining unresolved work from classifications resolved by fallback (STA-08)', () => {
    const fallbackCount = createView('uint32');
    const unresolvedCount = createView('uint32');
    const status: GPUUncertaintyStatusPort = {uncertainCount: fallbackCount, unresolvedCount};
    const stage: GPURecipeStageStatusPort = {stage: 'classification', status};
    expect(stage.status.uncertainCount).toBe(fallbackCount);
    expect(stage.status.unresolvedCount).toBe(unresolvedCount);
    expectTypeOf<GPUUncertaintyStatusPort['unresolvedCount']>().toEqualTypeOf<
      GPUStatusScalar | undefined
    >();
  });

  it('represents optional completeness of a result and of a named stage (STA-12)', () => {
    const complete = createView('uint32');
    const certificate: GPUCompletenessStatusPort = {complete};
    const bounded: GPUBoundedResultStatusPort = {
      ...certificate,
      count: createView('uint32'),
      overflow: createView('uint32')
    };
    const stage: GPURecipeStageStatusPort = {stage: 'classification', status: {complete}};
    expect(bounded.complete).toBe(complete);
    expect(stage.status.complete).toBe(complete);
    expect(() => validateGPURecipeStatusPort('fixture', {stages: [stage]})).not.toThrow();
    expect(() =>
      validateGPURecipeStatusPort('fixture', {
        stages: [
          {
            stage: 'classification',
            status: {complete: createView('float32') as unknown as GPUStatusScalar}
          }
        ]
      })
    ).toThrow('fixture stage classification complete must contain one uint32 row');
    const legacy: GPUCompletenessStatusPort = {};
    expect(legacy.complete).toBeUndefined();
  });

  it.each([0, 2])('rejects a certificate with %i rows (review scalar note)', length => {
    expect(() =>
      validateGPUStatusPort('fixture', {complete: createView('uint32', length)})
    ).toThrow('fixture complete must contain one uint32 row');
  });

  it.each([0, 2])('rejects a stage code with %i rows (review scalar note)', length => {
    expect(() =>
      validateGPURecipeStatusPort('fixture', {
        stages: [
          {
            stage: 'build',
            status: {},
            code: createView('uint32', length)
          }
        ]
      })
    ).toThrow('fixture stage build code must contain one uint32 row');
  });

  it('retains the live node count and every supplied build loss view by identity (CMP-08)', () => {
    expectTypeOf<GPUNetworkNodingOutput>().toExtend<GPUNodedNetworkTopologyPort>();
    const topology: GPUNetworkTopology = {
      offsets: createView('uint32', 4),
      neighbors: createView('uint32', 2),
      weights: createView('float32', 2)
    };
    const source = {
      topology,
      nodes: {status: {count: createView('uint32')}},
      pieces: {
        status: {
          count: createView('uint32'),
          overflow: createView('uint32'),
          candidateOverflow: createView('uint32'),
          vertexCount: createView('uint32'),
          requiredVertexCount: createView('uint32')
        }
      },
      status: {overflow: createView('uint32'), uncertainCount: createView('uint32')}
    };
    const port: GPUNodedNetworkTopologyPort = source;
    expect(port.topology).toBe(topology);
    expect(port.nodes.status.count).toBe(source.nodes.status.count);
    expect(port.status?.overflow).toBe(source.status.overflow);
    expect(port.status?.uncertainCount).toBe(source.status.uncertainCount);
    expect(port.pieces.status.count).toBe(source.pieces.status.count);
    expect(port.pieces.status.overflow).toBe(source.pieces.status.overflow);
    expect(port.pieces.status.candidateOverflow).toBe(source.pieces.status.candidateOverflow);
    expect(port.pieces.status.vertexCount).toBe(source.pieces.status.vertexCount);
    expect(port.pieces.status.requiredVertexCount).toBe(source.pieces.status.requiredVertexCount);
    const withoutOptionalStatus: GPUNodedNetworkTopologyPort = {
      topology,
      nodes: source.nodes,
      pieces: source.pieces
    };
    expect(withoutOptionalStatus.status).toBeUndefined();
    expectTypeOf<{
      topology: GPUNetworkTopology;
      nodes: GPUNetworkNodingOutput['nodes'];
    }>().not.toExtend<GPUNodedNetworkTopologyPort>();
    expectTypeOf<
      GPUNodedNetworkTopologyPort['pieces']['status']['count']
    >().toEqualTypeOf<GPUStatusScalar>();
    expectTypeOf<
      GPUNodedNetworkTopologyPort['pieces']['status']['overflow']
    >().toEqualTypeOf<GPUStatusScalar>();
    // Source words retain noding's legacy meanings; they are not a canonical fallback status.
    expectTypeOf<keyof NonNullable<GPUNodedNetworkTopologyPort['status']>>().toEqualTypeOf<
      'uncertainCount' | 'overflow'
    >();
  });

  // Hand fixture: n=4, k=2, RSS=8 => df=2, RSS/df=4, RSS/n=2.
  // For df=2, two-sided t p = 1 - abs(t)/sqrt(t*t+2); t=6 and t=3 give the literals below.
  // Metadata must not reinterpret the coefficient table, likelihood, or solver code.
  const ordinary: GPURegressionModelResultInput = {
    kind: 'ordinary-least-squares',
    coefficients: [
      {
        name: 'CONSTANT',
        role: 'intercept',
        estimate: 3,
        standardError: 0.5,
        statistic: 6,
        statisticKind: 't',
        pValue: 0.0266714732154248
      },
      {
        name: 'x0',
        role: 'predictor',
        estimate: 1.5,
        standardError: 0.5,
        statistic: 3,
        statisticKind: 't',
        pValue: 0.0954659662667091
      }
    ],
    parameterCount: 2,
    rowCount: 4,
    residualSumOfSquares: 8,
    standardErrorSigmaSquared: 'degrees-of-freedom',
    statusCode: 0,
    logLikelihood: -6
  };

  it('carries ridge and approximate inference metadata without changing fitted quantities (UST-06)', () => {
    const regularization = {kind: 'ridge' as const, lambda: 5};
    const result = createGPURegressionModelResult({
      ...ordinary,
      regularization,
      inferenceMethod: 'ridge-approximate'
    });
    expect(result.regularization).toEqual({kind: 'ridge', lambda: 5});
    expect(result.inferenceMethod).toBe('ridge-approximate');
    regularization.lambda = 9;
    expect(result.regularization?.lambda).toBe(5);
    expect(result.coefficients).toEqual([
      {
        name: 'CONSTANT',
        role: 'intercept',
        estimate: 3,
        standardError: 0.5,
        statistic: 6,
        statisticKind: 't',
        pValue: 0.0266714732154248
      },
      {
        name: 'x0',
        role: 'predictor',
        estimate: 1.5,
        standardError: 0.5,
        statistic: 3,
        statisticKind: 't',
        pValue: 0.0954659662667091
      }
    ]);
    expect(result.degreesOfFreedom).toBe(2);
    expect(Number.isFinite(result.sigmaSquared)).toBe(true);
    expect(result.sigmaSquared).toBe(4);
    expect(result.sigmaSquaredMaximumLikelihood).toBe(2);
    expect(result.logLikelihood).toBe(-6);
    expect(result.status).toEqual({code: 0, ok: true, converged: null, illConditioned: false});
  });

  it('preserves ordinary results without introducing metadata fields (UST-06)', () => {
    const result = createGPURegressionModelResult(ordinary);
    expect(result).not.toHaveProperty('regularization');
    expect(result).not.toHaveProperty('inferenceMethod');
    expect(result.inference.alternative).toBe('two-sided');
    expect(result.inference.permutationCount).toBe(0);
  });

  it.each([
    {kind: 'lasso', lambda: -2},
    {kind: 'lasso', lambda: 2},
    {kind: 'ridge', lambda: -2},
    {kind: 'ridge', lambda: NaN},
    {kind: 'ridge', lambda: Infinity},
    {kind: 'ridge', lambda: '2'}
  ])('rejects invalid regularization metadata $kind / $lambda (review S1)', regularization => {
    expect(() =>
      createGPURegressionModelResult({
        ...ordinary,
        regularization: regularization as unknown as GPURegressionRegularization
      })
    ).toThrow(/createGPURegressionModelResult regularization\.(kind|lambda)/);
  });

  it.each([0, 2])('accepts a finite non-negative ridge penalty %i (review S1)', lambda => {
    expect(
      createGPURegressionModelResult({...ordinary, regularization: {kind: 'ridge', lambda}})
        .regularization
    ).toEqual({kind: 'ridge', lambda});
  });

  it('requires an explicit absorbed-intercept parameter count for panel errors (review B5)', () => {
    expect(() =>
      createGPURegressionModelResult({
        ...ordinary,
        kind: 'spatial-panel-error-maximum-likelihood',
        parameterCount: undefined
      })
    ).toThrow(/parameterCount.*absorbed/);
  });

  it('matches the panel reader when finite likelihood has unavailable criteria (review B5)', () => {
    // Three entities * four periods, one predictor and lambda: k=3+1+1=5, df=12-5=7.
    // ML variance 2 => RSS=24; ll=-18 is retained but panel criteria are unavailable.
    const summary = new Float32Array(22).fill(NaN);
    summary[0] = 12;
    summary[2] = 2;
    summary[3] = -18;
    const reader = getGPUSpatialPanelErrorMaximumLikelihoodModelResult(
      {
        table: Float32Array.of(NaN, NaN, NaN, NaN, 2, NaN, NaN, NaN, 0.5, NaN, NaN, NaN),
        summary,
        status: Uint32Array.of(0)
      },
      {entityCount: 3}
    );
    const built = createGPURegressionModelResult({
      kind: 'spatial-panel-error-maximum-likelihood',
      coefficients: reader.coefficients,
      rowCount: 12,
      parameterCount: 5,
      residualSumOfSquares: 24,
      logLikelihood: -18,
      standardErrorSigmaSquared: 'maximum-likelihood',
      statusCode: 0
    });
    for (const result of [reader, built]) {
      expect(result.parameterCount).toBe(5);
      expect(result.degreesOfFreedom).toBe(7);
      expect(Number.isFinite(result.sigmaSquared)).toBe(true);
      expect(result.sigmaSquared).toBe(3.4285714285714284);
      expect(result.logLikelihood).toBe(-18);
      expect(result.akaikeInformationCriterion).toBeNull();
      expect(result.schwarzCriterion).toBeNull();
      expect(result.correctedAkaikeInformationCriterion).toBeNull();
    }
  });

  it('counts the error variance alongside GTWR tr(S) for information criteria (review B5)', () => {
    // n=12, tr(S)=2, ll=-18 => AIC=2*(2+1)+36=42;
    // Schwarz=3*log(12)+36=43.454719949364; Hurvich AICc=36+2*12*3/(12-2-2)=45.
    const result = createGPURegressionModelResult({
      ...ordinary,
      kind: 'geographically-temporally-weighted-regression',
      coefficients: [],
      rowCount: 12,
      parameterCount: 2,
      logLikelihood: -18
    });
    expect(result.akaikeInformationCriterion).toBe(42);
    expect(Number.isFinite(result.schwarzCriterion)).toBe(true);
    expect(result.schwarzCriterion).toBeCloseTo(43.454719949364, 10);
    expect(result.correctedAkaikeInformationCriterion).toBe(45);
  });

  it('accepts the delivered GTWR reader result as the shared result without adaptation (W2-04)', () => {
    expectTypeOf<GPUGeographicallyTemporallyWeightedRegressionModelResult>().toExtend<GPURegressionModelResult>();
    // Hand fixture: RSS=8, n=10, tr(S)=2 => df=8, sigmaSquared=1.
    // Gaussian ll=-5*(log(2*pi*.8)+1); AICc=-2*ll+2*10*3/(10-2-2).
    const result = getGPUGeographicallyTemporallyWeightedRegressionModelResult({
      summary: Float32Array.of(
        8,
        2,
        36.147335150951356,
        0.75,
        10,
        NaN,
        1,
        1,
        NaN,
        NaN,
        -13.073667575475678,
        11
      )
    });
    const common: GPURegressionModelResult = result;
    expect(common).toBe(result);
    expect(common.kind).toBe('geographically-temporally-weighted-regression');
    expect(common.degreesOfFreedom).toBe(8);
    expect(Number.isFinite(common.sigmaSquared)).toBe(true);
    expect(common.sigmaSquared).toBe(1);
    expect(common.status.code).toBe(0);
  });

  it('accepts the delivered panel-error reader and counts absorbed intercepts (W2-04)', () => {
    expectTypeOf<GPUSpatialPanelErrorMaximumLikelihoodModelResult>().toExtend<GPURegressionModelResult>();
    // Hand readback: two entities, three periods, one slope, lambda => k=2+1+1=4,
    // df=6-4=2. ML sigmaSquared=2 means filtered RSS=12 and descriptive sigmaSquared=6.
    const summary = new Float32Array(22).fill(NaN);
    summary[0] = 6;
    summary[2] = 2;
    summary[3] = -6;
    const result = getGPUSpatialPanelErrorMaximumLikelihoodModelResult(
      {
        table: Float32Array.of(NaN, NaN, NaN, NaN, 2, NaN, NaN, NaN, 0.5, NaN, NaN, NaN),
        summary,
        status: Uint32Array.of(0)
      },
      {entityCount: 2}
    );
    const common: GPURegressionModelResult = result;
    expect(common).toBe(result);
    expect(common.kind).toBe('spatial-panel-error-maximum-likelihood');
    expect(common.parameterCount).toBe(4);
    expect(common.degreesOfFreedom).toBe(2);
    expect(Number.isFinite(common.sigmaSquared)).toBe(true);
    expect(common.sigmaSquared).toBe(6);
    expect(common.logLikelihood).toBe(-6);
    expect(common.akaikeInformationCriterion).toBeNull();
    expect(common.status.code).toBe(0);
  });

  it('records the Quantile LISA legacy rule separately from inclusive tails (DEF-01)', () => {
    expectTypeOf<GPUQuantileLISAAlternative>().toEqualTypeOf<
      'greater' | 'two-sided' | 'geoda-legacy'
    >();
    const alternatives: GPUQuantileLISAAlternative[] = ['greater', 'two-sided', 'geoda-legacy'];
    const metadata: GPUInferenceMetadata<GPUQuantileLISAAlternative>[] = alternatives.map(
      alternative => ({
        alternative,
        permutationCount: 9,
        multipleTesting: 'none',
        testedRowsDefinition: 'rows with at least one neighbor'
      })
    );
    expect(metadata.map(value => value.alternative)).toEqual([
      'greater',
      'two-sided',
      'geoda-legacy'
    ]);
    expectTypeOf<GPUInferenceAlternative>().toEqualTypeOf<
      'two-sided' | 'greater' | 'lesser' | 'folded' | 'directed'
    >();
  });

  it('separates sidedness from the permutation p-value construction (NUM-05)', () => {
    expectTypeOf<GPUInferenceMetadata['pValueConstruction']>().toEqualTypeOf<
      'upper-tail-count' | 'doubled-minimum-tail' | undefined
    >();
    const legacy: GPUInferenceMetadata = {
      alternative: 'two-sided',
      permutationCount: 19,
      multipleTesting: 'none',
      testedRowsDefinition: 'one maximum statistic'
    };
    const maximumStatistic: GPUInferenceMetadata = {
      ...legacy,
      pValueConstruction: 'upper-tail-count'
    };
    const doubledTail: GPUInferenceMetadata = {
      ...legacy,
      pValueConstruction: 'doubled-minimum-tail'
    };
    expect(maximumStatistic.alternative).toBe('two-sided');
    expect(maximumStatistic.pValueConstruction).toBe('upper-tail-count');
    expect(doubledTail.pValueConstruction).toBe('doubled-minimum-tail');
    expect(legacy.pValueConstruction).toBeUndefined();
  });

  it('exports an analytic subset while preserving permutation-only alternatives (NAM-10)', () => {
    expectTypeOf<GPUAnalyticAlternative>().toEqualTypeOf<'two-sided' | 'greater' | 'lesser'>();
    expectTypeOf<GPUAnalyticAlternative>().toExtend<GPUInferenceAlternative>();
    const permutationAlternatives: GPUInferenceAlternative[] = ['folded', 'directed'];
    expect(permutationAlternatives).toEqual(['folded', 'directed']);
  });
});
