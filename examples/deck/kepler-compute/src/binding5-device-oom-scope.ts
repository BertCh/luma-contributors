// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';

/**
 * Cause attached to a KeplerAllocationError when WebGPU reports a device out-of-memory. Published
 * luma.gl exports no out-of-memory error class, so this domain error carries the name that
 * isKeplerAllocationError matches in modules/kepler/src/runtime/dataset-registry.ts.
 */
export class KeplerDeviceOutOfMemoryError extends Error {
  override readonly name = 'OutOfMemoryError';
}

/** The subset of a WebGPU GPUDevice used to observe out-of-memory errors. */
type ErrorScopeHandle = {
  pushErrorScope(filter: 'out-of-memory'): void;
  popErrorScope(): Promise<{message?: string} | null>;
};

/**
 * Runs an allocation step inside a WebGPU `out-of-memory` error scope so a device out-of-memory,
 * which WebGPU reports asynchronously instead of throwing, becomes a thrown KeplerAllocationError
 * that restore can roll back on. luma's `Device.pushErrorScope` is a no-op outside debug mode, so
 * the raw device handle is used. Devices without error scopes (WebGL, null) run the step unwrapped.
 * Scopes are device-wide: an unrelated allocation that fails while the step runs is also caught.
 */
export async function runWithOutOfMemoryScope<Result>(
  device: {handle?: unknown},
  step: () => Promise<Result> | Result
): Promise<Result> {
  const handle = device.handle as Partial<ErrorScopeHandle> | undefined;
  if (typeof handle?.pushErrorScope !== 'function' || typeof handle.popErrorScope !== 'function') {
    return step();
  }
  handle.pushErrorScope('out-of-memory');
  let result: Result;
  try {
    result = await step();
  } catch (error) {
    // Always pop to keep the scope stack balanced; the original failure wins.
    await handle.popErrorScope().catch(() => null);
    throw error;
  }
  const outOfMemory = await handle.popErrorScope();
  if (outOfMemory) {
    throw new KeplerAllocationError('device-out-of-memory', {
      cause: new KeplerDeviceOutOfMemoryError(`Device out of memory: ${outOfMemory.message ?? ''}`)
    });
  }
  return result;
}
