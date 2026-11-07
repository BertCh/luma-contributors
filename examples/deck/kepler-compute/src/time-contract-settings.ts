// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {IntervalBoundaryClosure} from './time-contract-interval';
import {DEFAULT_INTERVAL_BOUNDARY_CLOSURE} from './time-contract-interval';

/**
 * Host-level time contract settings shared by the histogram planner, the interval selection and the
 * exact playback driver. Native Kepler applies a timezone only to labels, so UTC (the default)
 * leaves every native behavior unchanged; a zone here is a documented extension that re-buckets
 * day/hour/minute histograms by local wall-clock edges.
 */
export type TimeContractSettings = {
  /** 'UTC', a fixed offset such as '+05:30', or an IANA name such as 'America/New_York'. */
  timeZone: string;
  intervalClosure: IntervalBoundaryClosure;
};

let settings: TimeContractSettings = {
  timeZone: 'UTC',
  intervalClosure: DEFAULT_INTERVAL_BOUNDARY_CLOSURE
};
let revision = 0;
const listeners = new Set<() => void>();

export function getTimeContractSettings(): TimeContractSettings {
  return settings;
}

export function getTimeContractRevision(): number {
  return revision;
}

export function setTimeContractSettings(next: Partial<TimeContractSettings>): void {
  settings = {...settings, ...next};
  revision++;
  for (const listener of listeners) listener();
}

export function subscribeTimeContractSettings(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export type TimeIntervalDeclaration = {endColumn: string; closure?: IntervalBoundaryClosure};

const intervalDeclarations = new Map<string, TimeIntervalDeclaration>();

/**
 * Declares that rows of a dataset are intervals [startColumn, endColumn). The next publication of
 * that dataset binds the start field's native time control and filters to the contract overlap
 * predicate and accepts native filters carrying that endName. Must precede publication.
 */
export function declareTimeInterval(
  datasetId: string,
  startColumn: string,
  declaration: TimeIntervalDeclaration
): void {
  intervalDeclarations.set(`${datasetId}\u0000${startColumn}`, declaration);
  revision++;
}

export function clearTimeIntervalDeclarations(): void {
  intervalDeclarations.clear();
  revision++;
}

export function getTimeIntervalDeclaration(
  datasetId: string,
  startColumn: string
): TimeIntervalDeclaration | undefined {
  return intervalDeclarations.get(`${datasetId}\u0000${startColumn}`);
}
