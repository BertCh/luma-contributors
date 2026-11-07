// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kepler table methods that synchronously rewrite or scan CPU rows. On a resident table the GPU
 * source is authoritative, so each of them is replaced by an explicit refusal that reports a
 * diagnostic instead of letting the host row copy diverge silently from resident buffers.
 */
export const RESIDENT_ROW_MUTATION_METHODS = [
  'appendRows',
  'upsertRows',
  'removeRows',
  'findRowIndexesByFieldValues'
] as const;

export type ResidentRowMutationMethod = (typeof RESIDENT_ROW_MUTATION_METHODS)[number];

export type ResidentSynchronousGuardReport = {
  readonly method: ResidentRowMutationMethod;
  readonly reason: string;
};

export function formatResidentGuardReason(method: ResidentRowMutationMethod): string {
  return method === 'findRowIndexesByFieldValues'
    ? 'Resident tables do not scan rows on the CPU; use an asynchronous bounded table lookup'
    : `Resident tables cannot apply ${method} synchronously; republish the source explicitly`;
}

/**
 * Replaces synchronous row mutation and scan methods with refusals. Methods the table does not
 * define (older Kepler builds) are left alone.
 */
export function installResidentSynchronousGuards<Table extends object>(
  table: Table,
  onBlocked: (report: ResidentSynchronousGuardReport) => void
): Table {
  for (const method of RESIDENT_ROW_MUTATION_METHODS) {
    if (typeof (table as Record<string, unknown>)[method] !== 'function') continue;
    Object.defineProperty(table, method, {
      configurable: true,
      enumerable: true,
      writable: true,
      value: () => {
        const reason = formatResidentGuardReason(method);
        onBlocked({method, reason});
        // Resident tables refuse synchronous row access: republish explicitly or read asynchronously.
        throw new Error(reason);
      }
    });
  }
  return table;
}
