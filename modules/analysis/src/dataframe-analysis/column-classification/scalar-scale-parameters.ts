// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

const ID = 'GPUScalarScale';

export const GPU_SCALAR_SCALE_PARAMETER_LENGTH = 4;
/** Point scale searches its ordered domain without changing source category codes. */
export const GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT = 256;
export const GPU_SCALAR_SCALE_CODES = {linear: 0, sqrt: 1, log: 2, identity: 3, point: 4} as const;
export type GPUScalarScaleType = keyof typeof GPU_SCALAR_SCALE_CODES;
export type GPUScalarScaleSettings = {
  scale: GPUScalarScaleType;
  /** Numeric modes require exactly two entries. Point accepts zero through the capacity. */
  domainCount: number;
  /** A finite value written for nulls, unknown categories and invalid numerical results. */
  nullValue?: number;
  /** Reverse the GPU domain as the output range of the native fixed (identity) scale only. */
  invert?: boolean;
};

/** Packs [scaleCode, domainCount, nullValue, invertIdentity], suitable for replay without compilation. */
export function getGPUScalarScaleParameterValues(
  settings: GPUScalarScaleSettings,
  target: Float32Array = new Float32Array(GPU_SCALAR_SCALE_PARAMETER_LENGTH)
): Float32Array {
  // The replay target must hold the complete parameter layout.
  validateRequirement(
    ID,
    target.length >= GPU_SCALAR_SCALE_PARAMETER_LENGTH,
    `parameter target must hold at least ${GPU_SCALAR_SCALE_PARAMETER_LENGTH} values`
  );
  // Only implemented native scalar modes are accepted.
  validateRequirement(
    ID,
    Object.hasOwn(GPU_SCALAR_SCALE_CODES, settings.scale),
    'scale must be linear, sqrt, log, identity or point'
  );
  const domainCount = settings.domainCount;
  // Numeric modes require two endpoints; point domains may be empty.
  validateRequirement(
    ID,
    Number.isInteger(domainCount) &&
      domainCount >= 0 &&
      domainCount <= GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT &&
      (settings.scale === 'point' || domainCount === 2),
    `domainCount must be an integer from 0 to ${GPU_SCALAR_SCALE_MAXIMUM_DOMAIN_COUNT}, and exactly 2 unless scale is point`
  );
  const nullValue = settings.nullValue ?? 0;
  // The fallback must remain finite when stored in the output format.
  validateRequirement(
    ID,
    Number.isFinite(Math.fround(nullValue)),
    'nullValue must be finite as float32'
  );
  // Fixed scales alone derive their output range from the GPU domain.
  validateRequirement(
    ID,
    !settings.invert || settings.scale === 'identity',
    'invert is accepted only for the identity scale'
  );
  target[0] = GPU_SCALAR_SCALE_CODES[settings.scale];
  target[1] = domainCount;
  target[2] = nullValue;
  target[3] = Number(settings.invert ?? false);
  return target;
}

export type GPUScalarScaleControls = {
  scale: GPUScalarScaleType;
  /** Point domain is ordered by desired category position, not sorted numerically. */
  domain: readonly number[];
  /** Two output endpoints; unused by identity. Defaults to [0, 1]. */
  range?: readonly [number, number];
  nullValue?: number;
  /** Reverse the GPU domain as the output range of the native fixed (identity) scale only. */
  invert?: boolean;
};

/**
 * Prepares host metadata only, never source rows. Domain/range conversion must be exact f32;
 * callers must explicitly normalize more precise controls before using this helper.
 * Returned domain storage contains only active entries, including zero for an empty point
 * domain. The caller allocates a graph domain view with maximumDomainCount capacity (at least
 * two), then uploads these entries into its prefix; remaining entries are ignored.
 * Numeric domains may descend; point domains must be unique after normalizing signed zero.
 * GPU-derived controls must satisfy the same contract; their producer guarantees point-domain
 * uniqueness, while the mapping kernel checks finite values and control counts.
 */
export function prepareGPUScalarScaleControls(settings: GPUScalarScaleControls): {
  parameters: Float32Array;
  domain: Float32Array;
  range: Float32Array;
  /** Live entry counts; domainCount is distinct from allocation capacity. */
  counts: {domainCount: number; rangeCount: number; parameterCount: number};
  /** Graph view formats, named for the consuming GPUScalarScale fields. */
  formats: {domain: 'float32'; range: 'float32'; parameters: 'float32'};
  /** Minimum domain allocation; use as maximumDomainCount for a minimal graph. */
  minimumDomainCapacity: number;
} {
  const parameters = getGPUScalarScaleParameterValues({
    scale: settings.scale,
    domainCount: settings.domain.length,
    nullValue: settings.nullValue,
    invert: settings.invert
  });
  const domain = Float32Array.from(settings.domain);
  const range = Float32Array.from(settings.range ?? [0, 1]);
  for (const values of [settings.domain, settings.range ?? [0, 1]]) {
    // More precise controls require an explicit caller normalization policy.
    validateRequirement(
      ID,
      values.every(value => Number.isFinite(value) && Math.fround(value) === value),
      'domain and range entries must be finite and exactly representable as float32'
    );
  }
  // Piecewise ranges are outside this contributor's supported contract.
  validateRequirement(ID, range.length === 2, 'range must hold exactly 2 entries');
  // Duplicate point categories would make the requested ordinal positions ambiguous.
  validateRequirement(
    ID,
    settings.scale !== 'point' || new Set(settings.domain).size === settings.domain.length,
    'point domain entries must be unique'
  );
  // Native log domains require nonzero endpoints with the same sign.
  validateRequirement(
    ID,
    settings.scale !== 'log' ||
      (domain[0] > 0 && domain[1] > 0) ||
      (domain[0] < 0 && domain[1] < 0),
    'log domain endpoints must be nonzero with the same sign'
  );
  return {
    parameters,
    domain,
    range,
    counts: {
      domainCount: domain.length,
      rangeCount: range.length,
      parameterCount: parameters.length
    },
    formats: {domain: 'float32', range: 'float32', parameters: 'float32'},
    minimumDomainCapacity: Math.max(2, domain.length)
  };
}
