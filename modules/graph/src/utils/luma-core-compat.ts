// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding, ComputePass, Device} from '@luma.gl/core';

/**
 * Compatibility helpers for @luma.gl/core APIs that this library uses but that are not
 * available in every supported luma.gl release. Each helper prefers the native API.
 */

/** Stable per-group keys that let backends reuse bind groups across dispatches. */
export type BindGroupCacheKeys = Partial<Record<number, object>>;

type ComputePassSetBindingsWithOptions = (
  bindings: Record<string, Binding>,
  options?: {_bindGroupCacheKeys?: BindGroupCacheKeys}
) => void;

/**
 * Set compute pass bindings, forwarding bind-group cache keys to luma.gl releases that
 * accept them. Releases with a single-argument `setBindings` ignore the extra argument.
 */
export function setComputePassBindings(
  computePass: ComputePass,
  bindings: Record<string, Binding>,
  bindGroupCacheKeys?: BindGroupCacheKeys
): void {
  const setBindings = computePass.setBindings as ComputePassSetBindingsWithOptions;
  if (bindGroupCacheKeys) {
    setBindings.call(computePass, bindings, {_bindGroupCacheKeys: bindGroupCacheKeys});
  } else {
    setBindings.call(computePass, bindings);
  }
}

type DeviceWithSubmittedWorkWait = Device & {waitForSubmittedWork?: () => Promise<void>};

/** Resolve when all GPU work submitted so far has completed. */
export async function waitForSubmittedWork(device: Device): Promise<void> {
  const deviceWithWait = device as DeviceWithSubmittedWorkWait;
  if (typeof deviceWithWait.waitForSubmittedWork === 'function') {
    await deviceWithWait.waitForSubmittedWork();
    return;
  }
  const fence = device.createFence();
  try {
    await fence.signaled;
  } finally {
    fence.destroy();
  }
}

/**
 * True when an error reports a failed GPU allocation: luma.gl's `OutOfMemoryError`
 * (matched by name so every luma.gl release is supported) or a WebGPU `GPUOutOfMemoryError`.
 */
export function isOutOfMemoryError(error: unknown): boolean {
  if (typeof GPUOutOfMemoryError !== 'undefined' && error instanceof GPUOutOfMemoryError) {
    return true;
  }
  return error instanceof Error && error.name === 'OutOfMemoryError';
}
