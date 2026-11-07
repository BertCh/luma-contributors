// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type DatasetOperationKind = 'replace' | 'remove';

export type DatasetOperationContext = {
  /** True once a later operation on the same dataset was queued; stop and release partial work. */
  isSuperseded(): boolean;
};

type Entry = {generation: number; pending: number; tail: Promise<void>};

/**
 * Serializes replacement and removal per dataset ID. Operations on one dataset never overlap, so a
 * removal cannot race a half-finished replacement; a failed operation never blocks the next one;
 * and a queued replacement that a later operation superseded is skipped without running. Different
 * datasets run independently. Removal always runs, even if superseded by a later replacement,
 * because the later replacement starts from a clean slot.
 */
export class DatasetOperationQueue {
  private readonly entries = new Map<string, Entry>();

  run<Result>(
    datasetId: string,
    kind: DatasetOperationKind,
    operation: (context: DatasetOperationContext) => Promise<Result>
  ): Promise<Result | undefined> {
    const entry = this.entries.get(datasetId) ?? {
      generation: 0,
      pending: 0,
      tail: Promise.resolve()
    };
    const generation = ++entry.generation;
    this.entries.set(datasetId, entry);
    const execute = async (): Promise<Result | undefined> => {
      if (kind === 'replace' && entry.generation !== generation) return undefined;
      return operation({isSuperseded: () => entry.generation !== generation});
    };
    // An idle dataset starts immediately; only operations queued behind another can be superseded.
    const result = entry.pending === 0 ? execute() : entry.tail.then(execute);
    entry.pending++;
    entry.tail = result.then(
      () => undefined,
      () => undefined
    );
    void entry.tail.then(() => {
      entry.pending--;
      if (this.entries.get(datasetId) === entry && entry.pending === 0)
        this.entries.delete(datasetId);
    });
    return result;
  }

  /** Dataset IDs with an operation queued or running. */
  get busyDatasetIds(): string[] {
    return [...this.entries.keys()];
  }

  /** Resolves when every queued operation, including those queued meanwhile, has settled. */
  async whenIdle(): Promise<void> {
    while (this.entries.size)
      await Promise.all([...this.entries.values()].map(entry => entry.tail));
  }
}
