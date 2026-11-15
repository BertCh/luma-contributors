// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WebGPU limits the host requests when it creates its device.
 *
 * The default limit of eight storage buffers per shader stage is too low for the crossfilter
 * controller, which binds one mask per filtered dimension plus its output in a single compute
 * pass. The host therefore asks the adapter for its maximum storage buffer count (and never less
 * than the default), and the resident explorer reports a clear error when a dataset still needs
 * more than the device offers.
 */
export type HostRequiredLimits = {maxStorageBuffersPerShaderStage?: number};

let hostRequiredLimits: HostRequiredLimits = {};

/** Reads the adapter limits once, before the first device is created. Failures keep defaults. */
export async function resolveHostRequiredLimits(): Promise<HostRequiredLimits> {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    const adapterLimit = adapter?.limits.maxStorageBuffersPerShaderStage;
    if (typeof adapterLimit === 'number' && adapterLimit > 8) {
      hostRequiredLimits = {maxStorageBuffersPerShaderStage: adapterLimit};
    }
  } catch {
    hostRequiredLimits = {};
  }
  return hostRequiredLimits;
}

/** The limits resolved by resolveHostRequiredLimits, or an empty object before it has run. */
export function getHostRequiredLimits(): HostRequiredLimits {
  return hostRequiredLimits;
}
