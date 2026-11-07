// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const MEMORY_BUDGET_STORAGE_KEY = 'luma-kepler-memory-budget-bytes';

type BudgetStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

function getDefaultStorage(): BudgetStorage | undefined {
  try {
    return typeof localStorage === 'undefined' ? undefined : localStorage;
  } catch {
    // Blocked site data: the budget simply is not remembered on this viewer.
    return undefined;
  }
}

/** Per-viewer convenience; the saved bundle is the authoritative copy. Never throws. */
export function readStoredMemoryBudget(storage = getDefaultStorage()): number | undefined {
  try {
    const value = Number(storage?.getItem(MEMORY_BUDGET_STORAGE_KEY));
    return Number.isSafeInteger(value) && value >= 1 ? value : undefined;
  } catch {
    return undefined;
  }
}

export function writeStoredMemoryBudget(
  memoryBudgetBytes: number | undefined,
  storage = getDefaultStorage()
): void {
  try {
    if (memoryBudgetBytes === undefined) storage?.removeItem(MEMORY_BUDGET_STORAGE_KEY);
    else storage?.setItem(MEMORY_BUDGET_STORAGE_KEY, String(memoryBudgetBytes));
  } catch {
    // Storage may be full or blocked; the in-memory budget is unaffected.
  }
}
