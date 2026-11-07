// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Structural view of a dataset value that may be a GPU-resident (lazy) result. */
export type ResidentAwareDatasetValue = {
  resident?: {
    domains: Readonly<Record<string, readonly [number, number] | null>>;
  };
};

/**
 * Synchronous consumers cannot page GPU rows, so over-cap (GPU-resident) results are refused there
 * with a pointer to the asynchronous materialization path (resident-materialization.ts). Export,
 * DuckDB, sandbox bundle and point labels materialize asynchronously and no longer use this guard.
 */
export function assertNotResidentSource(value: ResidentAwareDatasetValue, operation: string): void {
  // Resident results cannot be read synchronously; use materializeResidentRows() first
  if (value.resident)
    throw new Error(`Resident results ${operation} requires asynchronous materialization`);
}

/** Resident datasets use GPU-reduced domains and never scan the lazy CPU view. */
export function selectRangeDomain(
  value: ResidentAwareDatasetValue,
  name: string,
  scanDomain: () => [number, number]
): [number, number] {
  if (!value.resident) return scanDomain();
  const domain = value.resident.domains[name];
  return domain ? [domain[0], domain[1]] : [0, 0];
}
