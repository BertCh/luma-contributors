// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type TrackedOperation = {
  id: number;
  label: string;
  /** Fraction in [0, 1]; undefined while the total is unknown. */
  fraction?: number;
  detail?: string;
  cancel(): void;
};

export type OperationHandle = {
  update(patch: {fraction?: number; detail?: string}): void;
  finish(): void;
};

const operations = new Map<number, TrackedOperation>();
const listeners = new Set<() => void>();
let sequence = 0;
let version = 0;

function notify(): void {
  version++;
  for (const listener of listeners) listener();
}

/** Registers a long-running host operation so the memory panel can show progress and cancel it. */
export function trackOperation(label: string, cancel: () => void): OperationHandle {
  const id = ++sequence;
  operations.set(id, {id, label, cancel});
  notify();
  return {
    update(patch) {
      const current = operations.get(id);
      if (!current) return;
      operations.set(id, {...current, ...patch});
      notify();
    },
    finish() {
      if (operations.delete(id)) notify();
    }
  };
}

export function listTrackedOperations(): readonly TrackedOperation[] {
  return [...operations.values()];
}

export function subscribeOperations(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getOperationVersion(): number {
  return version;
}
