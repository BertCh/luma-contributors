// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a hot-spot self-weight parameter buffer. */
export const GPU_HOT_SPOT_ANALYSIS_PARAMETER_LENGTH = 1;

/** Smallest normal float32: GPUs may flush smaller positive weights to zero and select Gi. */
const MINIMUM_POSITIVE_HOT_SPOT_SELF_WEIGHT = 2 ** -126;

/** Checks both numeric defaults and packed settings without losing the Gi/Gi* selector. @internal */
export function validateHotSpotSelfWeight(id: string, selfWeight: number): void {
  validateFiniteNumber(id, 'selfWeight', selfWeight, 0);
  validateRequirement(
    id,
    selfWeight === 0 || Math.fround(selfWeight) >= MINIMUM_POSITIVE_HOT_SPOT_SELF_WEIGHT,
    `selfWeight must be 0 (Gi) or round to a finite normal float32 >= ${MINIMUM_POSITIVE_HOT_SPOT_SELF_WEIGHT} (Gi*)`
  );
}

/** Per-frame focal weight for Gi/Gi*; the spatial-autocorrelation parameters remain separate. */
export type GPUHotSpotAnalysisSettings = {
  /**
   * Non-negative focal weight. Zero selects Gi, a positive value selects Gi*. Positive weights
   * must round to a normal float32 (at least 2^-126); smaller values are rejected rather than
   * silently switching statistics. Defaults to 1.
   */
  selfWeight?: number;
};

/**
 * Packs `[selfWeight]` for `selfWeight: parameterBuffer.importToGraph(graph)`.
 * @throws If the weight is negative, nonfinite in float32, or positive but rounds below the
 * smallest normal float32. A failed validation leaves the target unchanged.
 */
export function getGPUHotSpotAnalysisParameterValues(
  settings: GPUHotSpotAnalysisSettings = {},
  target: Float32Array = new Float32Array(GPU_HOT_SPOT_ANALYSIS_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUHotSpotAnalysisParameterValues';
  validateCapacity(id, 'target.length', target.length, GPU_HOT_SPOT_ANALYSIS_PARAMETER_LENGTH);
  validateHotSpotSelfWeight(id, settings.selfWeight ?? 1);
  target[0] = settings.selfWeight ?? 1;
  return target;
}
