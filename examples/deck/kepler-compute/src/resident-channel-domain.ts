// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentTableProvider} from './resident-data-table';

type DomainEntry =
  | {status: 'pending'; promise: Promise<void>}
  | {status: 'ready'; domain: readonly [number, number]}
  | {status: 'failed'; reason: string};

export type ResidentChannelDomainReaderOptions = {
  resolveProvider(datasetId: string): ResidentTableProvider | undefined;
  /**
   * Domain-mask scope of the dataset. When present, domains follow dynamic filters (like native
   * filteredIndexForDomain); the dataset-wide statistics are only the fallback for non-range fields.
   */
  resolveScope?(datasetId: string):
    | {
        readFilteredDomain(field: string): readonly [number, number] | null;
        readSummary(): Promise<boolean>;
        readonly isSummaryCurrent: boolean;
        readonly currentRevision: number;
        hasDomainField?(field: string): boolean;
      }
    | undefined;
  /** Called after an asynchronous domain read settles so legends can render again. */
  notify(): void;
};

/**
 * Global size/radius/height domains from the existing column statistics route (GPU profile for
 * Float32 columns, bounded CPU path otherwise). Native Kepler domains cover the whole dataset, so
 * the population is always `all`. Entries are keyed by provider revision so any source, selection
 * or computed-column change reads again; a read never blocks rendering.
 */
export function createResidentChannelDomainReader(options: ResidentChannelDomainReaderOptions) {
  const entries = new Map<string, DomainEntry>();

  function getEntry(datasetId: string, column: string): DomainEntry | null {
    const provider = options.resolveProvider(datasetId);
    if (!provider?.readColumnStatistics) return null;
    const key = `${datasetId}\u0000${column}\u0000${provider.sourceRevision}`;
    const existing = entries.get(key);
    if (existing) return existing;
    const promise = provider
      .readColumnStatistics({column, population: 'all'})
      .then(result => {
        const quantiles =
          result?.stats && 'quantiles' in result.stats ? result.stats.quantiles : [];
        const minimum = quantiles.find(quantile => quantile.label === 'Min')?.value;
        const maximum = quantiles.find(quantile => quantile.label === 'Max')?.value;
        entries.set(
          key,
          typeof minimum === 'number' && typeof maximum === 'number'
            ? {status: 'ready', domain: [minimum, maximum]}
            : {status: 'failed', reason: 'The field has no finite values.'}
        );
      })
      .catch(error => {
        entries.set(key, {
          status: 'failed',
          reason: error instanceof Error ? error.message : String(error)
        });
      })
      .then(() => options.notify());
    const entry: DomainEntry = {status: 'pending', promise};
    entries.set(key, entry);
    return entry;
  }

  const scopeAttempts = new Map<string, {revision: number; time: number; settled: boolean}>();

  /** Filter-reactive domain from the domain mask; undefined when the scope cannot serve the field. */
  function getScopedDomain(
    datasetId: string,
    column: string
  ): {domain: readonly number[] | null; settled: boolean} | undefined {
    const scope = options.resolveScope?.(datasetId);
    if (!scope || scope.hasDomainField?.(column) === false) return undefined;
    if (scope.isSummaryCurrent) return {domain: scope.readFilteredDomain(column), settled: true};
    const attempt = scopeAttempts.get(datasetId);
    const now = Date.now();
    // One summary read per scope revision; retry later if the host had not submitted its frame yet.
    if (
      !attempt ||
      (attempt.revision !== scope.currentRevision && attempt.settled) ||
      (attempt.settled && now - attempt.time > 500)
    ) {
      const next = {revision: scope.currentRevision, time: now, settled: false};
      scopeAttempts.set(datasetId, next);
      void scope
        .readSummary()
        .catch(() => false)
        .then(() => {
          next.settled = true;
          options.notify();
        });
    }
    return {domain: null, settled: false};
  }

  return {
    /** Finite [minimum, maximum] once read; null while pending or unavailable. */
    getDomain(datasetId: string, column: string): readonly number[] | null {
      const scoped = getScopedDomain(datasetId, column);
      if (scoped) return scoped.domain;
      const entry = getEntry(datasetId, column);
      return entry?.status === 'ready' ? entry.domain : null;
    },
    /** Reason the domain cannot be shown, or null while pending or ready. */
    getFailure(datasetId: string, column: string): string | null {
      const scoped = getScopedDomain(datasetId, column);
      if (scoped)
        return scoped.settled && !scoped.domain ? 'No rows pass the active filters.' : null;
      const entry = getEntry(datasetId, column);
      return entry?.status === 'failed' ? entry.reason : null;
    },
    clear(): void {
      entries.clear();
    }
  };
}
