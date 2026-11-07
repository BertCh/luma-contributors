// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type FieldPreparationOutcome =
  | {status: 'requested'}
  | {status: 'pending'}
  | {status: 'already-attempted'; reason: string}
  | {status: 'unavailable'; reason: string};

export type FieldPreparationHost = {
  /** Names of the columns the native source dataset of this ID has, or undefined when it has none. */
  getSourceFieldNames(datasetId: string): readonly string[] | undefined;
  /** Columns the last preparation excluded from GPU upload, with the reason. */
  getExcludedFields(datasetId: string): readonly {name: string; reason: string}[];
  /** Re-prepare the dataset from its current native table under the same ID. */
  requestRepublish(datasetId: string): void;
};

/**
 * Prepares a column on demand when a native filter or dynamic domain reads one that the resident
 * copy lacks. The resident dataset is re-prepared from its native table once per dataset; columns
 * that preparation already excluded (unsupported type, Float32 overflow) or that the source does
 * not have are reported with a reason instead of looping.
 */
export class FieldPreparationCoordinator {
  private host?: FieldPreparationHost;
  private readonly attempted = new Map<string, Set<string>>();
  private readonly requested = new Set<string>();
  private readonly outcomes = new Map<string, FieldPreparationOutcome>();
  private requestCount = 0;

  setHost(host: FieldPreparationHost | undefined): void {
    this.host = host;
  }

  /** Plain-data view for debug state: republish count, attempted fields and last outcomes. */
  snapshot(): {
    republishCount: number;
    attempted: Record<string, string[]>;
    outcomes: Record<string, FieldPreparationOutcome>;
  } {
    return {
      republishCount: this.requestCount,
      attempted: Object.fromEntries(
        [...this.attempted].map(([datasetId, fields]) => [datasetId, [...fields]])
      ),
      outcomes: Object.fromEntries(this.outcomes)
    };
  }

  request(datasetId: string, field: string): FieldPreparationOutcome {
    const outcome = this.decide(datasetId, field);
    this.outcomes.set(`${datasetId}:${field}`, outcome);
    return outcome;
  }

  private decide(datasetId: string, field: string): FieldPreparationOutcome {
    const host = this.host;
    if (!host) return {status: 'unavailable', reason: 'Automatic preparation is not available'};
    const sourceFields = host.getSourceFieldNames(datasetId);
    if (!sourceFields)
      return {status: 'unavailable', reason: `Dataset ${datasetId} has no native source table`};
    if (!sourceFields.includes(field))
      return {status: 'unavailable', reason: `Field "${field}" is not a column of ${datasetId}`};
    const excluded = host.getExcludedFields(datasetId).find(entry => entry.name === field);
    if (excluded)
      return {
        status: 'already-attempted',
        reason: `Field "${field}" was excluded from GPU upload: ${excluded.reason}`
      };
    const attempts = this.attempted.get(datasetId) ?? new Set<string>();
    if (attempts.has(field))
      return {
        status: 'already-attempted',
        reason: `Field "${field}" is still not prepared after re-preparing ${datasetId}`
      };
    if (this.requested.has(datasetId)) return {status: 'pending'};
    attempts.add(field);
    this.attempted.set(datasetId, attempts);
    this.requested.add(datasetId);
    this.requestCount++;
    host.requestRepublish(datasetId);
    return {status: 'requested'};
  }

  /** Call when a republish of the dataset settled so a different field may request another one. */
  markSettled(datasetId: string): void {
    this.requested.delete(datasetId);
  }

  forget(datasetId: string): void {
    for (const key of [...this.outcomes.keys()])
      if (key.startsWith(`${datasetId}:`)) this.outcomes.delete(key);
    this.attempted.delete(datasetId);
    this.requested.delete(datasetId);
  }
}

export const fieldPreparationCoordinator = new FieldPreparationCoordinator();
