// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateFiniteNumber, validateRequirement} from './gpu-contributor-utils';

/** learningRate, beta1, beta2, epsilon, tolerance, skipZeroGradient. */
export const GPU_ADAM_STEP_PARAMETER_LENGTH = 6;
export type GPUAdamStepSettings = {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  /** Relative loss plateau threshold; zero disables early convergence. */
  tolerance?: number;
  skipZeroGradient?: boolean;
};

/** Packs per-frame values for a GPUParameterBuffer (float32). */
export function getGPUAdamStepParameterValues(settings: GPUAdamStepSettings = {}): Float32Array {
  const values = new Float32Array([
    settings.learningRate ?? 0.001,
    settings.beta1 ?? 0.9,
    settings.beta2 ?? 0.999,
    settings.epsilon ?? 1e-8,
    settings.tolerance ?? 0,
    settings.skipZeroGradient ? 1 : 0
  ]);
  const names = ['learningRate', 'beta1', 'beta2', 'epsilon', 'tolerance'];
  for (const [index, name] of names.entries())
    validateFiniteNumber('adam-step', name, values[index]);
  validateRequirement('adam-step', values[0] > 0, 'learningRate must be positive in float32');
  validateRequirement(
    'adam-step',
    values[1] >= 0 && values[1] < 1,
    'beta1 must be in [0, 1) in float32'
  );
  validateRequirement(
    'adam-step',
    values[2] >= 0 && values[2] < 1,
    'beta2 must be in [0, 1) in float32'
  );
  // Keep the denominator above the normal float32 floor even on flush-to-zero devices.
  validateRequirement('adam-step', values[3] >= 1e-30, 'epsilon must be at least 1e-30 in float32');
  validateRequirement('adam-step', values[4] >= 0, 'tolerance must be nonnegative');
  return values;
}
