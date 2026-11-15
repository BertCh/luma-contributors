// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Three uint32 words preserve both Philox key words and the float32 significance bits exactly. */
export const GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH = 3;

/** Per-frame settings of the exact and tiled multivariate conditional randomization. */
export type GPUMultivariateLocalGearyPermutationTestSettings = {
  /** Two unsigned 32-bit Philox key words, including all high bits. Defaults to [0, 0]. */
  seed?: readonly [number, number];
  /** Significance threshold in (0, 1), compared as float32. Defaults to 0.05. */
  significance?: number;
};

/** Validates the legacy numeric defaults and the public packer's settings identically. @internal */
export function validateMultivariateGearySettings(
  id: string,
  settings: GPUMultivariateLocalGearyPermutationTestSettings
): void {
  const seed = settings.seed ?? [0, 0];
  validateRequirement(id, seed.length === 2, 'seed must contain exactly 2 uint32 words');
  validateCapacity(id, 'seed[0]', seed[0], 0, 0xffffffff);
  validateCapacity(id, 'seed[1]', seed[1], 0, 0xffffffff);
  const significance = settings.significance ?? 0.05;
  validateRequirement(id, significance > 0 && significance < 1, 'significance must lie in (0, 1)');
}

/**
 * Packs [seedLow, seedHigh, float32 bits of significance] into a uint32 GPUParameterBuffer.
 * Failed validation leaves target unchanged; words beyond the three parameters are preserved.
 */
export function getGPUMultivariateLocalGearyPermutationTestParameterValues(
  settings: GPUMultivariateLocalGearyPermutationTestSettings = {},
  target: Uint32Array = new Uint32Array(
    GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH
  )
): Uint32Array {
  const id = 'getGPUMultivariateLocalGearyPermutationTestParameterValues';
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH
  );
  validateMultivariateGearySettings(id, settings);
  const seed = settings.seed ?? [0, 0];
  const significanceBits = new Uint32Array(
    new Float32Array([settings.significance ?? 0.05]).buffer
  )[0];
  target.set([seed[0], seed[1], significanceBits]);
  return target;
}
