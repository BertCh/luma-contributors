// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Four words align the per-frame time scale to a GPU parameter buffer. */
export const GPU_HDBSCAN_PARAMETER_LENGTH = 4;

/** Per-frame settings of space-time GPUHDBSCAN. */
export type GPUHDBSCANSettings = {
  /** Spatial position units per temporal-context tick. Finite and non-negative; zero ignores time distances. */
  timeScale: number;
};

/** Packs [timeScale, 0, 0, 0] for a caller-owned GPUParameterBuffer, with float32 validation. */
export function getGPUHDBSCANParameterValues(
  settings: GPUHDBSCANSettings,
  target: Float32Array = new Float32Array(GPU_HDBSCAN_PARAMETER_LENGTH)
): Float32Array {
  const id = 'GPUHDBSCAN';
  const timeScale = Math.fround(settings.timeScale);
  validateRequirement(
    id,
    Number.isFinite(timeScale) && timeScale >= 0 && settings.timeScale >= 0,
    'timeScale must be finite and in [0, float32 maximum]'
  );
  validateCapacity(id, 'target length', target.length, GPU_HDBSCAN_PARAMETER_LENGTH);
  target.set([timeScale, 0, 0, 0]);
  return target;
}
