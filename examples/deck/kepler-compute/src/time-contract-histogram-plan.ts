// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  buildZoneOffsetTable,
  getLocalBucketEdges,
  parseTimeZoneSpecification,
  type LocalBucketUnit
} from './time-contract-zone';

const LOCAL_INTERVAL = /^1-(day|hour|minute)$/;
const BUCKET_MILLISECONDS = {day: 86_400_000, hour: 3_600_000, minute: 60_000};
/** Edge scans stop here; the native 16,384 bin cap is checked by the caller of this planner. */
const MAXIMUM_EDGE_COUNT = 16_385;

/** Returns the local bucket unit when the interval is a zone-aware one (1-day, 1-hour, 1-minute). */
export function getZonedBucketUnit(interval: string): LocalBucketUnit | null {
  return (LOCAL_INTERVAL.exec(interval)?.[1] as LocalBucketUnit | undefined) ?? null;
}

/**
 * UTC millisecond edges of local-wall-clock buckets that enclose the native domain, as the native
 * planner encloses it with UTC floor/ceiling edges: the first edge is at or before domain[0] and the
 * last at or after domain[1]. A repeated local bucket (fall back) contributes one edge per
 * occurrence, a skipped one (gap) none, so 23 h and 25 h days appear naturally.
 */
export function makeZonedTimeHistogramEdges(
  domain: readonly [number, number],
  unit: LocalBucketUnit,
  zone: string
): number[] {
  const width = BUCKET_MILLISECONDS[unit];
  const margin = BigInt(width * 2);
  const start = BigInt(domain[0]) - margin - 86_400_000n;
  const end = BigInt(domain[1]) + margin + 86_400_000n;
  if ((end - start) / BigInt(width) > BigInt(MAXIMUM_EDGE_COUNT))
    throw new Error('Time histogram exceeds 16384 bins; choose a coarser interval');
  const table = buildZoneOffsetTable(parseTimeZoneSpecification(zone), start, end);
  const edges = getLocalBucketEdges(table, unit, start, end).map(Number);
  let first = 0;
  while (first + 1 < edges.length && edges[first + 1] <= domain[0]) first++;
  let last = first;
  while (last < edges.length - 1 && edges[last] < domain[1]) last++;
  if (edges[last] < domain[1])
    throw new Error('Time histogram calendar boundary exceeds Date range');
  const thresholds = edges.slice(first, last + 1);
  // A domain on a single edge still needs one closed bin.
  if (thresholds.length < 2 && last + 1 < edges.length) thresholds.push(edges[last + 1]);
  return thresholds;
}
