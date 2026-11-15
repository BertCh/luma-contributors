// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a LOSH residual-power parameter buffer. */
export const GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH = 1;

/** Per-frame descriptive LOSH settings. */
export type GPULocalSpatialHeteroscedasticitySettings = {
  /** Absolute-deviation power: 1 for absolute deviation, 2 for variance. Defaults to 2. */
  residualPower?: 1 | 2;
};

/** Packs `[residualPower]` for `residualPower: parameterBuffer.importToGraph(graph)`. */
export function getGPULocalSpatialHeteroscedasticityParameterValues(
  settings: GPULocalSpatialHeteroscedasticitySettings = {},
  target: Float32Array = new Float32Array(GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPULocalSpatialHeteroscedasticityParameterValues';
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH
  );
  validateCapacity(id, 'residualPower', settings.residualPower ?? 2, 1, 2);
  target[0] = settings.residualPower ?? 2;
  return target;
}
