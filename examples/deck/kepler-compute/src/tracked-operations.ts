// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {trackOperation, type OperationHandle} from './operation-tracker';

/** Operations shorter than this never appear in the memory manager, so quick chart reads do not flicker. */
export const TRACKED_OPERATION_DELAY_MILLISECONDS = 250;

export type DeferredOperationHandle = Pick<OperationHandle, 'update' | 'finish'>;

/** Registers the operation only if it is still running after `delayMilliseconds`. */
export function trackDeferredOperation(
  label: string,
  cancel: () => void,
  delayMilliseconds = TRACKED_OPERATION_DELAY_MILLISECONDS
): DeferredOperationHandle {
  let handle: OperationHandle | undefined;
  let finished = false;
  let pending: {fraction?: number; detail?: string} = {};
  const timer = setTimeout(() => {
    if (!finished) {
      handle = trackOperation(label, cancel);
      handle.update(pending);
    }
  }, delayMilliseconds);
  return {
    update(patch) {
      pending = {...pending, ...patch};
      handle?.update(patch);
    },
    finish() {
      finished = true;
      clearTimeout(timer);
      handle?.finish();
      handle = undefined;
    }
  };
}

/**
 * Runs `run` as a tracked, cancellable operation. The signal given to `run` aborts when the memory
 * manager cancels the operation or when `externalSignal` aborts.
 */
export async function runTrackedOperation<T>(
  label: string,
  externalSignal: AbortSignal | undefined,
  run: (signal: AbortSignal, handle: DeferredOperationHandle) => Promise<T>,
  delayMilliseconds = TRACKED_OPERATION_DELAY_MILLISECONDS
): Promise<T> {
  const controller = new AbortController();
  const forward = () => controller.abort();
  if (externalSignal?.aborted) controller.abort();
  externalSignal?.addEventListener('abort', forward, {once: true});
  const handle = trackDeferredOperation(label, forward, delayMilliseconds);
  try {
    return await run(controller.signal, handle);
  } finally {
    handle.finish();
    externalSignal?.removeEventListener('abort', forward);
  }
}
