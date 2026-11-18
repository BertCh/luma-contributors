// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Deliberately staged documentation: remove each entry when its GPU capture lands. */
export const PENDING_LENS_RECORDS: Readonly<Record<string, string>> = {};

export function collectPendingRecordFailures(
  scenarioIds: ReadonlySet<string>,
  pending: Readonly<Record<string, string>>,
  hasRecord: (scenarioId: string) => boolean
): string[] {
  const failures: string[] = [];
  for (const scenarioId of scenarioIds) {
    if (!hasRecord(scenarioId) && !pending[scenarioId]?.trim()) {
      failures.push(`${scenarioId}: missing record without a documented pending entry`);
    }
  }
  for (const [scenarioId, reason] of Object.entries(pending)) {
    if (!scenarioIds.has(scenarioId))
      failures.push(`${scenarioId}: pending entry has no directive`);
    if (!reason.trim()) failures.push(`${scenarioId}: pending entry needs a reason`);
    if (hasRecord(scenarioId)) failures.push(`${scenarioId}: captured; remove pending entry`);
  }
  return failures.sort();
}
