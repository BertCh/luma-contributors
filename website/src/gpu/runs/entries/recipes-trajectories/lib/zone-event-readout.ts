// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Status readbacks can be pending; absent evidence never certifies completeness. */
export function formatZoneEventReadout(
  zone: string,
  crossings: number | undefined,
  flags: {
    eventOverflow?: number;
    eventCandidateOverflow?: number;
    scratchCandidateOverflow?: number;
  },
  dwell?: number | null
): string {
  const crossingState =
    flags.eventOverflow === undefined || flags.eventCandidateOverflow === undefined
      ? ' (status pending)'
      : flags.eventOverflow || flags.eventCandidateOverflow
        ? ' (incomplete)'
        : '';
  const dwellState =
    flags.scratchCandidateOverflow === undefined
      ? ' (status pending)'
      : flags.scratchCandidateOverflow
        ? ' (incomplete)'
        : '';
  const crossingText =
    crossings === undefined || !Number.isFinite(crossings)
      ? 'crossings pending'
      : `${crossings} crossings${crossingState}`;
  const dwellText =
    dwell === undefined
      ? ''
      : dwell === null || !Number.isFinite(dwell)
        ? ' · dwell unavailable'
        : ` · dwell ${dwell} s${dwellState}`;
  return `zone ${zone} · ${crossingText}${dwellText}`;
}
