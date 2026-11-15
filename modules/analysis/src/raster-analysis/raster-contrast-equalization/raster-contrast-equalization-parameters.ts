// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateParameterTarget} from '../raster-band-utils';

/** Smallest normal float32: domain operands and widths must survive GPU denormal flushing.
 * Also floors positive clip fractions away from the zero sentinel without changing threshold one.
 */
const MINIMUM_NORMAL_FLOAT32 = 2 ** -126;

/** Float32 slots: input minimum/maximum, reference minimum/maximum, clip limit, three reserved. */
export const GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH = 8;

export type GPURasterContrastEqualizationSettings = {
  /** Fixed input domain, clamped before binning. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126). */
  inputDomain?: readonly [number, number];
  /** Reference domain for histogram matching. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126). */
  referenceDomain?: readonly [number, number];
  /** Fraction of valid tile samples per bin, in [0, 1]. Default 0.01; zero disables clipping. Tiny positive values pack to at least 2^-126, preserving threshold one. */
  clipLimit?: number;
};

/**
 * Packs per-frame settings for GPURasterContrastEqualization into a GPUParameterBuffer.
 * Domain endpoints must round to zero or normal finite float32 values; widths must be finite
 * normal float32 values at least 2^-126, so GPU denormal flushing cannot erase the divisor.
 * @param settings Domains and normalized clip limit.
 * @param target Optional destination of at least eight float32 values.
 */
export function getGPURasterContrastEqualizationParameterValues(
  settings: GPURasterContrastEqualizationSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterContrastEqualizationParameterValues';
  validateParameterTarget(id, target, GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH);
  const inputDomain = settings.inputDomain ?? [0, 1];
  const referenceDomain = settings.referenceDomain ?? [0, 1];
  for (const [name, domain] of [
    ['inputDomain', inputDomain],
    ['referenceDomain', referenceDomain]
  ] as const) {
    const minimum = Math.fround(domain[0]);
    const maximum = Math.fround(domain[1]);
    const width = Math.fround(maximum - minimum);
    validateRequirement(
      id,
      domain.length === 2 &&
        [minimum, maximum].every(
          value =>
            Number.isFinite(value) && (value === 0 || Math.abs(value) >= MINIMUM_NORMAL_FLOAT32)
        ) &&
        Number.isFinite(width) &&
        width >= MINIMUM_NORMAL_FLOAT32,
      `${name} must have two finite increasing zero or normal float32 endpoints and a finite normal float32 range (at least ${MINIMUM_NORMAL_FLOAT32})`
    );
  }
  const clipLimit = settings.clipLimit ?? 0.01;
  validateFiniteNumber(id, 'clipLimit', clipLimit, 0, 1);
  const packedClipLimit = clipLimit > 0 ? Math.max(clipLimit, MINIMUM_NORMAL_FLOAT32) : 0;
  target.set([...inputDomain, ...referenceDomain, packedClipLimit, 0, 0, 0]);
  return target;
}
