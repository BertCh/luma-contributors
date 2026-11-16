// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Runtime [selfWeight, atkinsonInequalityAversion], each representable as a valid float32. */
export const GPU_SEGREGATION_PARAMETER_LENGTH = 2;
export type GPUSegregationSettings = {selfWeight?: number; atkinsonInequalityAversion?: number};

/** Packs per-frame settings; numeric props on the contributor supply identical uploaded defaults. */
export function getGPUSegregationParameterValues(
  settings: GPUSegregationSettings = {},
  id = 'segregation-parameters'
): Float32Array {
  const selfWeight = Math.fround(settings.selfWeight ?? 1);
  const atkinsonInequalityAversion = Math.fround(settings.atkinsonInequalityAversion ?? 0.5);
  validateRequirement(
    id,
    Number.isFinite(selfWeight) && selfWeight >= 0,
    'selfWeight must round to a finite nonnegative float32'
  );
  validateRequirement(
    id,
    atkinsonInequalityAversion > 0 && atkinsonInequalityAversion < 1,
    'atkinsonInequalityAversion must round to a float32 in (0, 1)'
  );
  return Float32Array.of(selfWeight, atkinsonInequalityAversion);
}
