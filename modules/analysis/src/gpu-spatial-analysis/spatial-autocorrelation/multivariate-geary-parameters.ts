// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {getPermutationSeedKey} from '../permutation-inference/permutation-parameters';

/** Three uint32 words preserve both Philox key words and the float32 significance bits exactly. */
export const GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH = 3;

/** Per-frame settings of the exact and tiled multivariate conditional randomization. */
export type GPUMultivariateLocalGearyPermutationTestSettings = {
  /** Nonnegative safe-integer seed or two uint32 Philox words, preserving every high bit. Defaults to [0, 0]. */
  seed?: number | readonly [number, number];
  /** Significance threshold in (0, 1), compared as float32. Defaults to 0.05. */
  significanceLevel?: number;
  /** Fixed number of draws, resolved at construction, not a per-frame parameter. Defaults to 999. */
  permutationCount?: number;
};

/** Checks old prop keys before construction copies the canonical ones. @internal */
export function validateMultivariateGearyOptionNames(id: string, settings: object): void {
  validateRequirement(
    id,
    !('significance' in settings),
    'significance was removed. Use significanceLevel'
  );
  validateRequirement(
    id,
    !('maximumPermutations' in settings),
    'maximumPermutations was removed. Use permutationCount'
  );
}

/** Validates defaults and the public packer's settings identically. @internal */
export function validateMultivariateGearySettings(
  id: string,
  settings: GPUMultivariateLocalGearyPermutationTestSettings
): {seed: readonly [number, number]; significanceLevel: number; permutationCount: number} {
  validateMultivariateGearyOptionNames(id, settings);
  if (typeof settings.seed === 'number') {
    validateRequirement(
      id,
      Number.isSafeInteger(settings.seed) && settings.seed >= 0,
      `seed must be a nonnegative safe integer or 2 uint32 words; received ${settings.seed}`
    );
  }
  const seed =
    typeof settings.seed === 'number'
      ? getPermutationSeedKey(settings.seed)
      : (settings.seed ?? [0, 0]);
  validateRequirement(id, seed.length === 2, 'seed must contain exactly 2 uint32 words');
  validateCapacity(id, 'seed[0]', seed[0], 0, 0xffffffff);
  validateCapacity(id, 'seed[1]', seed[1], 0, 0xffffffff);
  validateCapacity(id, 'permutationCount', settings.permutationCount ?? 999, 1);
  const significance = settings.significanceLevel ?? 0.05;
  validateRequirement(
    id,
    Math.fround(significance) > 0 && Math.fround(significance) < 1,
    `significanceLevel must remain in (0, 1) after float32 packing; received ${significance}`
  );
  return {
    seed: [seed[0], seed[1]],
    significanceLevel: significance,
    permutationCount: settings.permutationCount ?? 999
  };
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
  const {seed, significanceLevel} = validateMultivariateGearySettings(id, settings);
  const significanceBits = new Uint32Array(new Float32Array([significanceLevel]).buffer)[0];
  target.set([seed[0], seed[1], significanceBits]);
  return target;
}
