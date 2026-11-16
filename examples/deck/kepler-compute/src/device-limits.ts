// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getAnalysisRequiredLimits} from '@luma-contributors/analysis/analysis-core';

/**
 * WebGPU limits the host requests when it creates its device.
 *
 * The default limit of eight storage buffers per shader stage is too low for the crossfilter
 * controller, which binds one mask per filtered dimension plus its output in a single compute
 * pass. The host asks for the adapter's maximum storage buffer count and binding/buffer sizes,
 * so large resident datasets can use the available capacity. The resident explorer reports a
 * clear error when a dataset still needs more than the device offers.
 */
export type HostRequiredLimits = Record<string, number>;

let hostRequiredLimits: HostRequiredLimits = {};

/** Reads the adapter limits once, before the first device is created. Failures keep defaults. */
export async function resolveHostRequiredLimits(): Promise<HostRequiredLimits> {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    hostRequiredLimits = getAnalysisRequiredLimits(adapter?.limits, {
      requestedStorageBuffersPerShaderStage: adapter?.limits.maxStorageBuffersPerShaderStage
    });
  } catch {
    hostRequiredLimits = {};
  }
  return hostRequiredLimits;
}

/** The limits resolved by resolveHostRequiredLimits, or an empty object before it has run. */
export function getHostRequiredLimits(): HostRequiredLimits {
  return hostRequiredLimits;
}
