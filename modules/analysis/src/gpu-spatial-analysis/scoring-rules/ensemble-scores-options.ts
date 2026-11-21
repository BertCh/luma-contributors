// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Estimator of the member distribution's CRPS and energy score. */
export type EnsembleScoresEstimator = 'fair' | 'empirical';

export type EnsembleScoresEstimatorOptions = {
  /** Defaults to the empirical distribution; fair requires at least two members. */
  estimator?: EnsembleScoresEstimator;
};

/** Shared by the contributor, operation forms and CPU evaluator. */
export function resolveEnsembleScoresEstimator(
  id: string,
  options: EnsembleScoresEstimatorOptions
): EnsembleScoresEstimator {
  const {estimator} = options;
  validateRequirement(
    id,
    estimator === undefined || estimator === 'fair' || estimator === 'empirical',
    `estimator must be fair or empirical; received ${String(estimator)}`
  );
  validateRequirement(
    id,
    !('fair' in options),
    "fair was removed. Use estimator: 'fair' or estimator: 'empirical'"
  );
  return estimator ?? 'empirical';
}

export type EnsembleScore = 'crps' | 'energyScore' | 'variogramScore';
export type EnsembleScoresOutputSelection = Partial<Record<EnsembleScore, string>>;

/** Validate the enumerable selection once; descriptors and bindings use the same captured pairs. */
export function captureEnsembleScoreOutputs<Value>(
  id: string,
  output: Partial<Record<EnsembleScore, Value>>
): readonly (readonly [EnsembleScore, Value])[] {
  const selected: (readonly [EnsembleScore, Value])[] = [];
  for (const port of Object.keys(output)) {
    validateRequirement(
      id,
      port === 'crps' || port === 'energyScore' || port === 'variogramScore',
      `output.${port} is unknown; accepted outputs are crps, energyScore and variogramScore`
    );
  }
  // Public declared fields can be prototype/non-enumerable accessors. Read each once, while
  // still rejecting unknown enumerable names from reusable object/dictionary selections.
  for (const port of ['crps', 'energyScore', 'variogramScore'] as const) {
    const value = output[port];
    if (value !== undefined) selected.push([port, value]);
  }
  validateRequirement(id, selected.length > 0, 'output needs crps, energyScore or variogramScore');
  return selected;
}
