// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as decoders from '../../../src/gpu-spatial-analysis/spatial-regression/regression-status-decoders';

// Literal tables audited against each model's emitted words, not a shared numeric vocabulary.
const tables = [
  ['GeographicallyWeightedRegression', ['ok', 'singular', 'excluded']],
  ['GeographicallyTemporallyWeightedRegression', ['ok', 'singular', 'excluded']],
  ['MultiscaleGeographicallyTemporallyWeightedRegression', ['ok', 'singular', 'excluded']],
  [
    'GeneralizedGeographicallyWeightedRegression',
    [
      'converged',
      'excluded',
      'singular',
      'iteration-limit',
      'numerical-failure',
      'degenerate-response',
      'separation'
    ]
  ],
  [
    'MultiscaleGeographicallyWeightedRegression',
    ['converged', 'excluded', 'singular', 'iteration-limit', 'numerical-failure']
  ],
  ['OrdinaryLeastSquares', ['ok', 'singular', 'too-few-rows']],
  ['SpatialLagMaximumLikelihood', ['ok', 'singular', 'too-few-rows', 'unusable-fit']],
  ['SpatialErrorMaximumLikelihood', ['ok', 'singular', 'too-few-rows', 'unusable-fit']],
  ['SpatialErrorGM', ['ok', 'singular', 'too-few-rows', 'unusable-fit']],
  ['SpatialTwoStageLeastSquares', ['ok', 'singular-or-nonfinite', 'too-few-rows', 'unusable-fit']],
  [
    'SpatialRegressionDiagnostics',
    [
      'ok',
      'singular',
      'too-few-rows',
      'degenerate-or-incomplete-weights',
      'nonfinite-or-nonpositive-residual-variance'
    ]
  ],
  [
    'InstrumentalVariablesCovariance',
    [
      'ok',
      'singular-projection',
      'insufficient-rows',
      'nonfinite-input-or-result',
      'invalid-kernel',
      'invalid-leverage',
      'invalid-clusters',
      'not-positive-semidefinite'
    ]
  ],
  [
    'SeeminglyUnrelatedRegression',
    [
      'ok',
      'insufficient-rows',
      'singular-design',
      'singular-error-covariance',
      'not-converged',
      'invalid-parameters'
    ]
  ],
  [
    'SpatialSeeminglyUnrelatedRegression',
    ['ok', 'singular', 'too-few-rows', 'unusable-fit', 'inadmissible-weights']
  ],
  [
    'ProbitRegression',
    ['converged', 'insufficient-rows', 'singular', 'not-converged', 'separated-or-invalid-response']
  ],
  ['SpatialPanelRegression', ['ok', 'singular-or-nonfinite', 'too-few-rows', 'unusable-fit']],
  ['SpatialPanelErrorMaximumLikelihood', ['ok', 'singular', 'too-few-rows', 'unusable-fit']],
  [
    'SpatialPanelLagMaximumLikelihood',
    ['converged', 'invalid-or-incomplete-inputs', 'singular-or-invalid-variance', 'not-converged']
  ],
  ['SpatialPanelErrorDiagnostics', ['ok', 'unsuccessful-fit-or-incomplete-inputs', 'degenerate']],
  ['SpatialPanelHausman', ['ok', 'unsuccessful-fit', 'invalid-covariance-difference']],
  [
    'ProbitSpatialDiagnostics',
    ['ok', 'unsuccessful-fit', 'invalid-or-incomplete-weights', 'degenerate']
  ],
  [
    'RestrictedRegimeRegression',
    [
      'ok',
      'insufficient-rows',
      'restricted-singular-or-nonfinite',
      'unrestricted-singular-or-nonfinite'
    ]
  ],
  ['SpatialRegimeRegression', ['ok', 'insufficient-rows', 'singular-or-nonfinite']],
  [
    'MultiscaleGeographicallyWeightedRegressionInference',
    ['ok', 'unsuccessful-fit', 'invalid-inference']
  ]
] as const;

for (const [model, names] of tables) {
  it(`decodes the complete ${model} table and unknown words`, () => {
    const decode = decoders[`decodeGPU${model}StatusFields`];
    names.forEach((name, code) => {
      const excluded = name === 'excluded';
      const retainedIterate =
        (model === 'SeeminglyUnrelatedRegression' && code === 4) ||
        (code === 3 &&
          [
            'GeneralizedGeographicallyWeightedRegression',
            'MultiscaleGeographicallyWeightedRegression',
            'ProbitRegression'
          ].includes(model));
      expect(decode(code)).toMatchObject({
        code,
        name,
        ok: code === 0,
        excluded,
        failed: code !== 0 && !excluded && !retainedIterate
      });
    });
    for (const code of [-1, names.length, 999, 0.5, NaN, Infinity]) {
      expect(decode(code)).toMatchObject({
        code,
        name: 'unknown',
        ok: false,
        excluded: false,
        failed: true
      });
    }
  });
}

it('keeps the ambiguous meanings of codes 1 and 3 model-specific', () => {
  expect(decoders.decodeGPUGeographicallyWeightedRegressionStatusFields(1).name).toBe('singular');
  expect(decoders.decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields(1).excluded).toBe(
    true
  );
  expect(decoders.decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields(3).name).toBe(
    'iteration-limit'
  );
  expect(decoders.decodeGPUSpatialLagMaximumLikelihoodStatusFields(3).name).toBe('unusable-fit');
  expect(decoders.decodeGPUProbitRegressionStatusFields(3).name).toBe('not-converged');
  // Matches the device fixtures in generalized-gwr-selection.spec.ts (constant boundary and separation).
  expect(decoders.decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields(5).name).toBe(
    'degenerate-response'
  );
  expect(decoders.decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields(6).name).toBe(
    'separation'
  );
});

// Emitter fixtures pin what a status word permits us to say, not a unique diagnosis.
function getRingWeights(rows: number) {
  return {
    offsets: Uint32Array.from({length: rows + 1}, (_, row) => row),
    neighbors: Uint32Array.from({length: rows}, (_, row) => (row + 1) % rows),
    weights: new Float32Array(rows).fill(1)
  };
}

it('does not diagnose probit separation from a word also emitted for a nonbinary response', async () => {
  const {computeProbitRegressionOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation'
  );
  for (const response of [Float32Array.of(0, 1, 0.5, 1, 0), Float32Array.of(0, 0, 0, 1, 1)]) {
    const result = computeProbitRegressionOnCPU(
      {predictors: Float32Array.of(-2, -1, 0, 1, 2), response},
      {predictorCount: 1}
    );
    // The first response is invalid; the second is strictly separated by eta=x-0.5.
    expect(result.status[0]).toBe(4);
    expect(decoders.decodeGPUProbitRegressionStatusFields(result.status[0]).name).toBe(
      'separated-or-invalid-response'
    );
  }
});

it('includes zero residual variance in diagnostics code 4 for a finite perfect fit', async () => {
  const {computeSpatialRegressionDiagnosticsOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation'
  );
  const result = computeSpatialRegressionDiagnosticsOnCPU(
    {
      weights: getRingWeights(6),
      predictors: Float32Array.of(0, 1, 2, 3, 4, 5),
      response: Float32Array.of(2, 5, 8, 11, 14, 17),
      residuals: new Float32Array(6)
    },
    {predictorCount: 1}
  );
  // y=2+3x is full-rank with exactly zero RSS, so residual tests are undefined.
  expect(result.status[0]).toBe(4);
  expect(decoders.decodeGPUSpatialRegressionDiagnosticsStatusFields(result.status[0]).name).toBe(
    'nonfinite-or-nonpositive-residual-variance'
  );
});

it('includes nonpositive innovation variance in panel-lag code 2', async () => {
  const {computeSpatialPanelLagMaximumLikelihoodOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation'
  );
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      weights: getRingWeights(4),
      predictors: Float32Array.of(1, 2, 3, 4, 3, 1, 4, 2, 2, 4, 1, 3),
      response: new Float32Array(12)
    },
    {predictorCount: 1, periodCount: 3, effects: 'unit'}
  );
  // Within x has nonzero variation, but the all-zero response has innovation variance zero.
  expect(result.status[0]).toBe(2);
  expect(Array.from(result.summary)).toEqual(new Array(6).fill(NaN));
  expect(
    decoders.decodeGPUSpatialPanelLagMaximumLikelihoodStatusFields(result.status[0]).name
  ).toBe('singular-or-invalid-variance');
});

it('includes a nonfinite early solve failure in 2SLS and inherited panel status 1', async () => {
  const {computeSpatialTwoStageLeastSquaresOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation'
  );
  const input = {
    weights: getRingWeights(7),
    predictors: Float32Array.of(0, 1, 2, 4, 3, -1, -2),
    response: Float32Array.of(1, 4, 2, 5, 8, 0, -1)
  };
  expect(computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 1}).status[0]).toBe(0);
  input.response[2] = NaN;
  const result = computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 1});
  expect(result.status[0]).toBe(1);
  expect(decoders.decodeGPUSpatialTwoStageLeastSquaresStatusFields(result.status[0]).name).toBe(
    'singular-or-nonfinite'
  );
  expect(decoders.decodeGPUSpatialPanelRegressionStatusFields(result.status[0]).name).toBe(
    'singular-or-nonfinite'
  );
});

it('does not narrow nonfinite-result words to input failures or rank failures', () => {
  // GPU IV covariance finish sets 3 when the scaled covariance overflows (finite inputs allowed).
  expect(decoders.decodeGPUInstrumentalVariablesCovarianceStatusFields(3).name).toBe(
    'nonfinite-input-or-result'
  );
  // Restricted-regime finish sets 2/3 for nonfinite restricted/unrestricted residual sums,
  // even after successful solves, as well as for the respective rank failures.
  expect(decoders.decodeGPURestrictedRegimeRegressionStatusFields(2).name).toBe(
    'restricted-singular-or-nonfinite'
  );
  expect(decoders.decodeGPURestrictedRegimeRegressionStatusFields(3).name).toBe(
    'unrestricted-singular-or-nonfinite'
  );
});

it('decodes staged SUR and spatial-lag probit emitter constants', async () => {
  const {GPU_SUR_STATUS_INVALID_PARAMETERS} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/seemingly-unrelated-regression-parameters'
  );
  const {GPU_PROBIT_SPATIAL_LAG_STATUS} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/probit-spatial-lag-nodes'
  );
  expect(
    decoders.decodeGPUSeeminglyUnrelatedRegressionStatusFields(GPU_SUR_STATUS_INVALID_PARAMETERS)
  ).toMatchObject({name: 'invalid-parameters', failed: true});
  const expected = [
    [GPU_PROBIT_SPATIAL_LAG_STATUS.OK, 'ok'],
    [GPU_PROBIT_SPATIAL_LAG_STATUS.MODEL_NOT_CONVERGED, 'model-not-converged'],
    [GPU_PROBIT_SPATIAL_LAG_STATUS.SINGULAR_FIRST_STAGE, 'singular-first-stage'],
    [GPU_PROBIT_SPATIAL_LAG_STATUS.SINGULAR_SECOND_STAGE, 'singular-second-stage'],
    [GPU_PROBIT_SPATIAL_LAG_STATUS.RHO_OUT_OF_RANGE, 'rho-out-of-range'],
    [GPU_PROBIT_SPATIAL_LAG_STATUS.INVALID_WEIGHTS, 'invalid-weights']
  ] as const;
  for (const [code, name] of expected) {
    expect(decoders.decodeGPUProbitSpatialLagStatusFields(code)).toMatchObject({
      name,
      ok: code === 0
    });
  }
});
