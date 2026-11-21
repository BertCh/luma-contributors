// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPURadiationFlowModelNormalization} from './gpu-radiation-flow-model';

/** @internal Resolve omitted/null JavaScript options consistently with the resident model. */
export function resolveRadiationFlowModelNormalization(
  id: string,
  value: unknown
): GPURadiationFlowModelNormalization {
  const normalization = value ?? 'none';
  validateRequirement(
    id,
    normalization === 'none' || normalization === 'finite-size',
    `normalization must be 'none' or 'finite-size'; received ${String(value)}`
  );
  return normalization;
}
