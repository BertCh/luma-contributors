// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';

/** Capability overrides applied by {@link makeCapabilityDevice}. */
export type CapabilityOverrides = {
  /** Hides the `subgroups` feature so shaders take the portable fallback path. */
  disableSubgroups?: boolean;
  /** Lowers the per-dimension dispatch limit so small inputs exercise 2D and 3D layouts. */
  maxComputeWorkgroupsPerDimension?: number;
};

/**
 * Runs `callback` while `device` reports overridden capabilities, then restores the originals.
 * The real device object is patched in place because graphs reject buffers owned by wrappers.
 * Callers must not run overlapping overrides against the same device.
 */
export async function withDeviceCapabilities<T>(
  device: Device,
  overrides: CapabilityOverrides,
  callback: () => Promise<T>
): Promise<T> {
  const originalLimits = Object.getOwnPropertyDescriptor(device, 'limits');
  const originalFeatures = Object.getOwnPropertyDescriptor(device, 'features');
  const patch = (property: string, value: unknown) =>
    Object.defineProperty(device, property, {value, configurable: true, writable: true});
  if (overrides.maxComputeWorkgroupsPerDimension !== undefined) {
    const limits = Object.create(device.limits) as Device['limits'];
    Object.defineProperty(limits, 'maxComputeWorkgroupsPerDimension', {
      value: overrides.maxComputeWorkgroupsPerDimension
    });
    patch('limits', limits);
  }
  if (overrides.disableSubgroups) {
    patch('features', new Set([...device.features].filter(feature => feature !== 'subgroups')));
  }
  try {
    return await callback();
  } finally {
    for (const [property, descriptor] of [
      ['limits', originalLimits],
      ['features', originalFeatures]
    ] as const) {
      if (descriptor) Object.defineProperty(device, property, descriptor);
      else delete (device as unknown as Record<string, unknown>)[property];
    }
  }
}

/** Deterministic pseudo-random uint32 values for reproducible fixtures. */
export function makePseudoRandomUint32(length: number, seed: number = 1): Uint32Array {
  const values = new Uint32Array(length);
  let state = seed >>> 0 || 1;
  for (let index = 0; index < length; index++) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    values[index] = state;
  }
  return values;
}
