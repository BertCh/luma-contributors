// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Named interpretation of one estimator-specific word; unknown words never imply success. */
export type GPUDecodedRegressionStatus = {
  code: number;
  name: string;
  ok: boolean;
  excluded: boolean;
  /** Failed fit/test; excludes exclusions and unfinished fits whose model preserves usable iterates. */
  failed: boolean;
  /** Null for non-iterative status vocabularies. */
  converged: boolean | null;
};

function decodeStatus(
  code: number,
  names: readonly string[],
  iterative = false,
  retainedIterateCode?: number
): GPUDecodedRegressionStatus {
  const name = Number.isInteger(code) && code >= 0 && code < names.length ? names[code] : 'unknown';
  const ok = code === 0;
  const excluded = name === 'excluded';
  return {
    code,
    name,
    ok,
    excluded,
    failed: !ok && !excluded && code !== retainedIterateCode,
    converged: iterative ? ok : null
  };
}

// Each table describes emitted words, not a diagnosis inferred from their numeric value.
const LOCAL_FIT = ['ok', 'singular', 'excluded'] as const;
const LINEAR_FIT = ['ok', 'singular', 'too-few-rows', 'unusable-fit'] as const;
const TWO_STAGE_FIT = ['ok', 'singular-or-nonfinite', 'too-few-rows', 'unusable-fit'] as const;
const BACKFIT = [
  'converged',
  'excluded',
  'singular',
  'iteration-limit',
  'numerical-failure'
] as const;

/** Local GWR row; singular includes ill-conditioning or an invalid bandwidth, not selection failure. */
export function decodeGPUGeographicallyWeightedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LOCAL_FIT);
}

/** Local GTWR row for either dense or indexed traversal; not the summary status. */
export function decodeGPUGeographicallyTemporallyWeightedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LOCAL_FIT);
}

/** Local MGTWR row; code 0 does not certify backfit or bandwidth-search convergence. */
export function decodeGPUMultiscaleGeographicallyTemporallyWeightedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LOCAL_FIT);
}

/** GGWR row: 5 is a constant boundary response, 6 a witnessed complete/quasi separation. */
export function decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [...BACKFIT, 'degenerate-response', 'separation'], true, 3);
}

/** Coupled MGWR backfit outcome on each included row; 3 retains a finite last iterate. Search exhaustion is separate. */
export function decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, BACKFIT, true, 3);
}

/** OLS model status. */
export function decodeGPUOrdinaryLeastSquaresStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, ['ok', 'singular', 'too-few-rows']);
}

/** Lag ML model: code 3 covers nonfinite/incomplete inputs, inadmissible intervals, or a rejected optimum; it cannot identify which. */
export function decodeGPUSpatialLagMaximumLikelihoodStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LINEAR_FIT);
}

/** Error ML model: code 3 covers nonfinite/incomplete inputs, moments or inadmissible parameter intervals. */
export function decodeGPUSpatialErrorMaximumLikelihoodStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LINEAR_FIT);
}

/** Error GM model: code 3 also includes zero OLS residuals; it does not isolate incomplete weights. */
export function decodeGPUSpatialErrorGMStatusFields(code: number): GPUDecodedRegressionStatus {
  return decodeStatus(code, LINEAR_FIT);
}

/** Spatial 2SLS: code 1 includes early nonfinite solve failures and rank failures; 3 covers other nonfinite inputs/moments or incomplete weights. */
export function decodeGPUSpatialTwoStageLeastSquaresStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, TWO_STAGE_FIT);
}

/** Cross-section tests: 3 combines degenerate/incomplete weights; 4 includes nonpositive residual variance even with finite inputs. */
export function decodeGPUSpatialRegressionDiagnosticsStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [
    'ok',
    'singular',
    'too-few-rows',
    'degenerate-or-incomplete-weights',
    'nonfinite-or-nonpositive-residual-variance'
  ]);
}

/** IV covariance status, independently of the model fit; 3 includes computed covariance overflow with finite inputs. */
export function decodeGPUInstrumentalVariablesCovarianceStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [
    'ok',
    'singular-projection',
    'insufficient-rows',
    'nonfinite-input-or-result',
    'invalid-kernel',
    'invalid-leverage',
    'invalid-clusters',
    'not-positive-semidefinite'
  ]);
}

/** Ordinary SUR system status (output.status), not the per-equation spatial status. */
export function decodeGPUSeeminglyUnrelatedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(
    code,
    [
      'ok',
      'insufficient-rows',
      'singular-design',
      'singular-error-covariance',
      'not-converged',
      'invalid-parameters'
    ],
    false,
    4
  );
}

/** Spatial SUR per-equation parameter status (spatial.output.status): inherits error-GM words 0-3, adds 4 for inadmissible weights. */
export function decodeGPUSpatialSeeminglyUnrelatedRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [...LINEAR_FIT, 'inadmissible-weights']);
}

/** Probit fit: 3 preserves an unfinished iterate; 4 combines separation/boundary response and invalid nonbinary responses. */
export function decodeGPUProbitRegressionStatusFields(code: number): GPUDecodedRegressionStatus {
  return decodeStatus(
    code,
    [
      'converged',
      'insufficient-rows',
      'singular',
      'not-converged',
      'separated-or-invalid-response'
    ],
    true,
    3
  );
}

/** Pooled/unit fixed-effects panel 2SLS: underlying 2SLS status plus 2 for nonpositive residual df. */
export function decodeGPUSpatialPanelRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, TWO_STAGE_FIT);
}

/** Panel-error model status inherited from nested error ML; not its LM diagnostics status. */
export function decodeGPUSpatialPanelErrorMaximumLikelihoodStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, LINEAR_FIT);
}

/** Panel lag: 2 combines singularity and nonpositive/nonfinite innovation variance; 3 combines exhaustion and lower-bound phi (inspect boundary bits). */
export function decodeGPUSpatialPanelLagMaximumLikelihoodStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(
    code,
    ['converged', 'invalid-or-incomplete-inputs', 'singular-or-invalid-variance', 'not-converged'],
    true
  );
}

/** Panel-error LM diagnostic status; 1 combines unsuccessful projection/fit and incomplete inputs. */
export function decodeGPUSpatialPanelErrorDiagnosticsStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, ['ok', 'unsuccessful-fit-or-incomplete-inputs', 'degenerate']);
}

/** Panel random-versus-unit Hausman test status. */
export function decodeGPUSpatialPanelHausmanStatusFields(code: number): GPUDecodedRegressionStatus {
  return decodeStatus(code, ['ok', 'unsuccessful-fit', 'invalid-covariance-difference']);
}

/** Probit spatial specification tests, independently of the probit fit status. */
export function decodeGPUProbitSpatialDiagnosticsStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [
    'ok',
    'unsuccessful-fit',
    'invalid-or-incomplete-weights',
    'degenerate'
  ]);
}

/** Restricted regime/stability status: 2 and 3 include nonfinite restricted/unrestricted residual sums after successful solves. */
export function decodeGPURestrictedRegimeRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, [
    'ok',
    'insufficient-rows',
    'restricted-singular-or-nonfinite',
    'unrestricted-singular-or-nonfinite'
  ]);
}

/** Fully varying regime status, one word per regime; 2 does not distinguish singular from nonfinite. */
export function decodeGPUSpatialRegimeRegressionStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, ['ok', 'insufficient-rows', 'singular-or-nonfinite']);
}

/** Full MGWR inference: 2 combines nonpositive residual df/variance and nonfinite smoother output. */
export function decodeGPUMultiscaleGeographicallyWeightedRegressionInferenceStatusFields(
  code: number
): GPUDecodedRegressionStatus {
  return decodeStatus(code, ['ok', 'unsuccessful-fit', 'invalid-inference']);
}

/** Optional latent spatial-lag probit estimate; independent of model and diagnostics words. */
export function decodeGPUProbitSpatialLagStatusFields(code: number): GPUDecodedRegressionStatus {
  return decodeStatus(code, [
    'ok',
    'model-not-converged',
    'singular-first-stage',
    'singular-second-stage',
    'rho-out-of-range',
    'invalid-weights'
  ]);
}
