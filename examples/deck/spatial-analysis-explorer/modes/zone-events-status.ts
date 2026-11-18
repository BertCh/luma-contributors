// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Panel suffix for final and earlier event losses. */
export function formatZoneEventsStatus(
  overflow: number,
  candidateOverflow: number,
  trackOverflow: number,
  scratchOverflow: number
): string {
  const causes: string[] = [];
  if (overflow) causes.push('final event capacity');
  if (trackOverflow) causes.push('per-track event capacity');
  if (scratchOverflow) causes.push('candidate scratch');
  if (candidateOverflow && !trackOverflow && !scratchOverflow)
    causes.push('earlier candidate loss');
  return causes.length ? `OVERFLOW (${causes.join(', ')})` : 'no overflow';
}
