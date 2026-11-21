// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, expectTypeOf, it} from 'vitest';
import type {
  GPURegressionModelResult,
  GPURegressionModelStatus
} from '@luma-contributors/analysis/analysis-core';
import * as network from '@luma-contributors/analysis/gpu-network';
import * as spatial from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPUDecodedRegressionStatus,
  GPUGeographicallyTemporallyWeightedRegressionReadback,
  GPUGeographicallyTemporallyWeightedRegressionModelResult,
  GPUSpatialMaximumLikelihoodReadback,
  GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions,
  GPUSpatialPanelErrorMaximumLikelihoodModelResult
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import statisticsCoverage from '../oracles/fixtures/statistics-coverage.json';

it('publishes the network K-function event limit through the package entry (NET-110)', () => {
  // floor(n*(n-1)/2) must fit uint32; 92682 is the existing constructor boundary.
  expect(network.GPU_NETWORK_K_FUNCTION_MAXIMUM_EVENT_COUNT).toBe(92682);
});

it('publishes model-specific named status fields alongside existing strings and constants (SSR-112)', () => {
  expectTypeOf<GPURegressionModelStatus['name']>().toEqualTypeOf<string | undefined>();
  expectTypeOf<GPURegressionModelStatus['reason']>().toEqualTypeOf<string | undefined>();
  expectTypeOf<GPURegressionModelStatus['excluded']>().toEqualTypeOf<boolean | undefined>();
  expectTypeOf<GPURegressionModelStatus['failed']>().toEqualTypeOf<boolean | undefined>();
  expectTypeOf<GPUDecodedRegressionStatus>().toExtend<
    Omit<GPURegressionModelStatus, 'illConditioned'>
  >();
  const decoders: readonly ((code: number) => GPUDecodedRegressionStatus)[] = [
    spatial.decodeGPUGeographicallyWeightedRegressionStatusFields,
    spatial.decodeGPUGeographicallyTemporallyWeightedRegressionStatusFields,
    spatial.decodeGPUMultiscaleGeographicallyTemporallyWeightedRegressionStatusFields,
    spatial.decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields,
    spatial.decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields,
    spatial.decodeGPUOrdinaryLeastSquaresStatusFields,
    spatial.decodeGPUSpatialLagMaximumLikelihoodStatusFields,
    spatial.decodeGPUSpatialErrorMaximumLikelihoodStatusFields,
    spatial.decodeGPUSpatialErrorGMStatusFields,
    spatial.decodeGPUSpatialTwoStageLeastSquaresStatusFields,
    spatial.decodeGPUSpatialRegressionDiagnosticsStatusFields,
    spatial.decodeGPUInstrumentalVariablesCovarianceStatusFields,
    spatial.decodeGPUSeeminglyUnrelatedRegressionStatusFields,
    spatial.decodeGPUSpatialSeeminglyUnrelatedRegressionStatusFields,
    spatial.decodeGPUProbitRegressionStatusFields,
    spatial.decodeGPUSpatialPanelRegressionStatusFields,
    spatial.decodeGPUSpatialPanelErrorMaximumLikelihoodStatusFields,
    spatial.decodeGPUSpatialPanelLagMaximumLikelihoodStatusFields,
    spatial.decodeGPUSpatialPanelErrorDiagnosticsStatusFields,
    spatial.decodeGPUSpatialPanelHausmanStatusFields,
    spatial.decodeGPUProbitSpatialDiagnosticsStatusFields,
    spatial.decodeGPURestrictedRegimeRegressionStatusFields,
    spatial.decodeGPUSpatialRegimeRegressionStatusFields,
    spatial.decodeGPUMultiscaleGeographicallyWeightedRegressionInferenceStatusFields
  ];
  for (const decoder of decoders) {
    expect(typeof decoder).toBe('function');
    expect(decoder(0)).toMatchObject({code: 0, ok: true, excluded: false, failed: false});
  }
  const geographicallyWeightedStatus: GPUDecodedRegressionStatus =
    spatial.decodeGPUGeographicallyWeightedRegressionStatusFields(1);
  expect(geographicallyWeightedStatus).toEqual({
    code: 1,
    name: 'singular',
    ok: false,
    excluded: false,
    failed: true,
    converged: null
  });
  expect(spatial.decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields(1)).toEqual({
    code: 1,
    name: 'excluded',
    ok: false,
    excluded: true,
    failed: false,
    converged: false
  });
  expect(spatial.decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields(5).name).toBe(
    'degenerate-response'
  );
  expect(spatial.decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields(6).name).toBe(
    'separation'
  );
  // These are different emitted model words, not a shared numeric status enum.
  expect(spatial.decodeGPUGeographicallyWeightedRegressionStatusFields(3).name).toBe('unknown');
  expect(spatial.decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields(3).name).toBe(
    'iteration-limit'
  );
  expect(spatial.decodeGPUSpatialErrorMaximumLikelihoodStatusFields(3).name).toBe('unusable-fit');
  expect(spatial.decodeGPUSpatialRegressionDiagnosticsStatusFields(3).name).toBe(
    'degenerate-or-incomplete-weights'
  );
  expect(spatial.decodeGPUSeeminglyUnrelatedRegressionStatusFields(3).name).toBe(
    'singular-error-covariance'
  );
  expect(spatial.decodeGPUProbitRegressionStatusFields(3).name).toBe('not-converged');
  expect(spatial.decodeGPUGeographicallyWeightedRegressionStatus(1)).toBe('singular');
  expect(spatial.decodeGPUMultiscaleGeographicallyWeightedRegressionStatus(1)).toBe('excluded');
  expect(spatial.GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR).toBe(1);
  expect(spatial.GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED).toBe(1);
});

it('publishes the GTWR summary reader without changing the GWR selection contract (SSR-117)', () => {
  expect(typeof spatial.getGPUGeographicallyTemporallyWeightedRegressionModelResult).toBe(
    'function'
  );
  // /usr/bin/python3: ll=-5*(log(2*pi*.8)+1)=-13.073667575475678;
  // AICc=-2*ll + 2*10*(2+1)/(10-2-2)=36.147335150951356.
  // RSS=8, tr(S)=2, n=10; dense/indexed candidate-selection slot 5 stays NaN.
  const summary = Float32Array.of(
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
  );
  // Independent Gaussian likelihood and AICc: -n/2*(log(2*pi*RSS/n)+1),
  // -2*ll + 2*n*(tr(S)+1)/(n-tr(S)-2). A valid producer summary must agree.
  expect(Number.isFinite(summary[10])).toBe(true);
  expect(summary[10]).toBeCloseTo(-13.073667575478, 5);
  expect(summary[2]).toBeCloseTo(36.147335150951356, 5);

  expect(spatial.getGPUGeographicallyWeightedRegressionModelResult({summary}).status.ok).toBe(
    false
  );
  const readback: GPUGeographicallyTemporallyWeightedRegressionReadback = {summary};
  const result: GPUGeographicallyTemporallyWeightedRegressionModelResult =
    spatial.getGPUGeographicallyTemporallyWeightedRegressionModelResult(readback);
  expectTypeOf<GPUGeographicallyTemporallyWeightedRegressionModelResult>().toExtend<GPURegressionModelResult>();
  const commonResult: GPURegressionModelResult = result;
  expect(commonResult).toBe(result);
  expect(result.kind).toBe('geographically-temporally-weighted-regression');
  expect(result.status.ok).toBe(true);
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBe(Math.fround(-13.073667575475678));
  // RSS/(n-tr(S)) = 8/8 = 1; AIC=-2*ll+2*(tr(S)+1).
  expect(Number.isFinite(result.sigmaSquared)).toBe(true);
  expect(result.sigmaSquared).toBe(1);
  expect(result.akaikeInformationCriterion).toBe(-2 * Math.fround(-13.073667575475678) + 6);
  expect(result.correctedAkaikeInformationCriterion).toBe(Math.fround(36.147335150951356));
  expect(Number.isNaN(summary[5])).toBe(true);
});

it('publishes panel-error absorbed effects and preserves its independent likelihood convention (SSR-119)', () => {
  expect(typeof spatial.getGPUSpatialPanelErrorMaximumLikelihoodModelResult).toBe('function');
  // Independent NumPy panel fixture in test/oracles/fixtures/statistics-coverage.json:
  // six entities * eight periods, one slope; Gaussian NT likelihood = -56.584351089654.
  // This tests the public readback seam, not the numerical estimator or spreg's logll.
  const fixture = statisticsCoverage['panelError'];
  expect(fixture.logLikelihood).toBe(-56.584351089654);
  const summary = new Float32Array(22).fill(NaN);
  summary[0] = 48;
  summary[2] = fixture.variance;
  summary[3] = fixture.logLikelihood;
  const readback: GPUSpatialMaximumLikelihoodReadback = {
    table: Float32Array.of(
      NaN,
      NaN,
      NaN,
      NaN,
      fixture.coefficient,
      NaN,
      NaN,
      NaN,
      fixture.parameter,
      NaN,
      NaN,
      NaN
    ),
    summary,
    status: Uint32Array.of(0)
  };
  const options: GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions = {entityCount: 6};
  const result: GPUSpatialPanelErrorMaximumLikelihoodModelResult =
    spatial.getGPUSpatialPanelErrorMaximumLikelihoodModelResult(readback, options);
  expectTypeOf<GPUSpatialPanelErrorMaximumLikelihoodModelResult>().toExtend<GPURegressionModelResult>();
  const commonResult: GPURegressionModelResult = result;
  expect(commonResult).toBe(result);
  expect(result.kind).toBe('spatial-panel-error-maximum-likelihood');
  expect(result.absorbedEntityCount).toBe(6);
  // Six absorbed intercepts + slope + lambda => k=8, df=48-8=40.
  expect(result.parameterCount).toBe(8);
  expect(result.degreesOfFreedom).toBe(40);
  expect(result.coefficients).toHaveLength(3);
  expect(result.coefficients[0].estimate).toBeNaN();
  expect(Number.isFinite(result.logLikelihood)).toBe(true);
  expect(result.logLikelihood).toBe(Math.fround(-56.584351089654));
  expect(result.akaikeInformationCriterion).toBeNull();
  expect(result.schwarzCriterion).toBeNull();
  expect(result.correctedAkaikeInformationCriterion).toBeNull();
  const crossSectionalAkaike =
    spatial.getGPUSpatialErrorMaximumLikelihoodModelResult(readback).akaikeInformationCriterion;
  expect(Number.isFinite(crossSectionalAkaike)).toBe(true);
  expect(crossSectionalAkaike).toBe(4 - 2 * Math.fround(-56.584351089654));
});

it('keeps the six GWR result bindings behind a whole-leaf barrel export (review B1)', () => {
  const barrel = readFileSync('modules/analysis/src/gpu-spatial-analysis/index.ts', 'utf8');
  const leaf = './spatial-regression/geographically-weighted-regression-model-result';
  expect(barrel).toContain(`export * from '${leaf}';`);
  expect(
    [...barrel.matchAll(/export\s+(?:type\s+)?\{[^}]*\}\s+from\s+'([^']*)'/g)].map(
      match => match[1]
    )
  ).not.toContain(leaf);
});
