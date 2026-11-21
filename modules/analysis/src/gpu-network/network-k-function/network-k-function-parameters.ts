// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of uint32 elements in a {@link GPUNetworkKFunction} parameter view. */
export const GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUNetworkKFunction}. */
export type GPUNetworkKFunctionSettings = {
  /** Random seed of the simulated patterns, an integer in `[0, 2^53)`. */
  seed: number;
  /**
   * Number of simulated patterns evaluated, at most the compile-time `simulationCount`. The
   * remaining patterns are skipped (their K values and counts are 0).
   */
  activeSimulations: number;
};

/**
 * Packs per-frame {@link GPUNetworkKFunction} settings.
 *
 * Layout (uint32): `[seedLow, seedHigh, activeSimulations, 0]`.
 *
 * @param settings Seed and active simulation count.
 * @param target Optional destination of at least 4 elements.
 * @param simulationCount Optional compiled maximum, checked before writing.
 * @param id Instance ID for validation diagnostics.
 * @throws If the target is too short or a value is invalid.
 */
export function getGPUNetworkKFunctionParameterValues(
  settings: GPUNetworkKFunctionSettings,
  target: Uint32Array = new Uint32Array(GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
  simulationCount: number = 0xffffffff,
  id: string = 'GPUNetworkKFunction'
): Uint32Array {
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH,
    Number.MAX_SAFE_INTEGER
  );
  const {seed, activeSimulations} = settings;
  validateCapacity(id, 'seed', seed, 0, Number.MAX_SAFE_INTEGER);
  validateCapacity(id, 'simulationCount', simulationCount, 0, 0xffffffff);
  validateCapacity(id, 'activeSimulations', activeSimulations, 0, simulationCount);
  target[0] = seed % 2 ** 32;
  target[1] = Math.floor(seed / 2 ** 32);
  target[2] = activeSimulations;
  target[3] = 0;
  return target;
}
