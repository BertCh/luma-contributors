// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {VariogramModelType, VariogramModelWeighting} from './variogram-model';

/** @internal Validate model/weighting choices before either form considers bin eligibility. */
export function validateVariogramModelFitChoices(
  id: string,
  models: readonly VariogramModelType[],
  weighting: VariogramModelWeighting
): void {
  const supportedModels = ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2'];
  for (const model of models) {
    validateRequirement(
      id,
      supportedModels.includes(model),
      `invalid model or bin topology: models member ${model} must be one of ${supportedModels.join(', ')}`
    );
  }
  const supportedWeightings = ['pairs', 'none', 'cressie'];
  validateRequirement(
    id,
    supportedWeightings.includes(weighting),
    `invalid model or bin topology: weighting ${weighting} must be one of ${supportedWeightings.join(', ')}`
  );
}
