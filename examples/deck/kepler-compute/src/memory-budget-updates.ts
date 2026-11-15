// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {subscribeOperations} from './operation-tracker';

/** Defers panel updates from render-time reservations and cancels delivery after unmount/rebind. */
export function subscribeMemoryBudgetUpdates(
  registry: {subscribeMemory(listener: () => void): () => void} | undefined,
  listener: () => void
): () => void {
  let active = true;
  let pending = false;
  const refresh = () => {
    if (!active || pending) return;
    pending = true;
    // MapContainer prepares resident layers synchronously during render. Registry accounting
    // stays immediate; only React's notification waits until that render has returned.
    queueMicrotask(() => {
      pending = false;
      if (active) listener();
    });
  };
  const stopOperations = subscribeOperations(refresh);
  const stopRegistry = registry?.subscribeMemory(refresh);
  return () => {
    active = false;
    stopOperations();
    stopRegistry?.();
  };
}
