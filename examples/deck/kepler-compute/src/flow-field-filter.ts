// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';

/** Reads the resident row mask chunk by chunk (one bounded readback per filter revision). */
export async function readSelectionWords(mask: GPUVector<'uint32'>): Promise<Uint32Array> {
  const words = new Uint32Array(mask.length);
  let offset = 0;
  for (const chunk of mask.data) {
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    words.set(new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length), offset);
    offset += chunk.length;
  }
  return words;
}

export type FlowFieldSelectionContext = {
  mask: GPUVector<'uint32'>;
  /** Changes whenever the resident row mask changes. */
  getRevision(): number;
  requestUpdate(): void;
  memoryRegistry?: KeplerDatasetRegistry;
};

type SelectionEntry = {
  mask: GPUVector<'uint32'>;
  requestedRevision: number;
  appliedRevision: number | null;
  words?: Uint32Array;
  reservation?: KeplerMemoryReservation;
  failure?: string;
  requestedGeneration: number;
  appliedGeneration: number;
  appliedMask?: GPUVector<'uint32'>;
  pending?: Promise<void>;
};

/**
 * Per-layer cache of the CPU copy of the resident filter mask used to build the field grid.
 * The grid is rebuilt from the previous mask until the new one has been read back, so a filter
 * change never blanks the particles; the CPU words are reserved against the dataset budget.
 */
export class FlowFieldSelectionCache {
  private readonly entries = new Map<string, SelectionEntry>();
  private nextGeneration = 0;

  /** Returns the applied words, the key fragment identifying them, and any failure reason. */
  resolve(
    layerId: string,
    context: FlowFieldSelectionContext | undefined
  ): {
    words: Uint32Array | undefined;
    key: string;
    failure: string | undefined;
    appliedRevision: number | null;
    appliedMask: GPUVector<'uint32'> | undefined;
    appliedGeneration: number;
  } {
    if (!context) {
      this.release(layerId);
      return {
        words: undefined,
        key: 'unfiltered',
        failure: undefined,
        appliedRevision: null,
        appliedMask: undefined,
        appliedGeneration: 0
      };
    }
    const revision = context.getRevision();
    let entry = this.entries.get(layerId);
    if (!entry) {
      entry = {
        mask: context.mask,
        requestedRevision: -1,
        appliedRevision: null,
        requestedGeneration: 0,
        appliedGeneration: 0
      };
      this.entries.set(layerId, entry);
    }
    if (entry.requestedRevision !== revision || entry.mask !== context.mask) {
      entry.requestedRevision = revision;
      entry.mask = context.mask;
      entry.failure = undefined;
      const generation = ++this.nextGeneration;
      entry.requestedGeneration = generation;
      const loadingEntry = entry;
      entry.pending = this.load(entry, revision, context, generation).finally(() => {
        if (loadingEntry.requestedGeneration === generation) loadingEntry.pending = undefined;
      });
    }
    return {
      words: entry.words,
      key: `filter:${entry.appliedRevision ?? 'pending'}`,
      failure: entry.failure,
      appliedRevision: entry.appliedRevision,
      appliedMask: entry.appliedMask,
      appliedGeneration: entry.appliedGeneration
    };
  }

  /** Analysis must await the current readback; rendering may continue using the previous words. */
  async waitForCurrent(
    layerId: string,
    context: FlowFieldSelectionContext | undefined,
    timeoutMilliseconds = 10000
  ): Promise<void> {
    if (!context) {
      this.resolve(layerId, undefined);
      return;
    }
    const deadline = performance.now() + timeoutMilliseconds;
    while (true) {
      this.resolve(layerId, context);
      const entry = this.entries.get(layerId)!;
      if (entry.pending) {
        const remaining = deadline - performance.now();
        if (remaining <= 0)
          throw new Error('Flow selection did not settle before the analysis timeout');
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          await Promise.race([
            entry.pending,
            new Promise<never>((_resolve, reject) => {
              timer = setTimeout(
                () =>
                  reject(new Error('Flow selection did not settle before the analysis timeout')),
                remaining
              );
            })
          ]);
        } finally {
          clearTimeout(timer);
        }
      }
      if (this.entries.get(layerId) !== entry)
        throw new Error('Flow selection was removed while awaiting analysis');
      if (entry.mask !== context.mask)
        throw new Error('Flow selection mask changed while awaiting analysis');
      if (entry.pending || entry.requestedRevision !== context.getRevision()) continue;
      if (entry.failure) throw new Error(`Flow selection mask load failed: ${entry.failure}`);
      if (entry.appliedRevision !== context.getRevision() || entry.appliedMask !== context.mask)
        throw new Error('Flow selection has no current mask for analysis');
      return;
    }
  }

  private async load(
    entry: SelectionEntry,
    revision: number,
    context: FlowFieldSelectionContext,
    generation: number
  ): Promise<void> {
    const mask = context.mask;
    try {
      const byteLength = mask.length * 4;
      if (!entry.reservation)
        entry.reservation = context.memoryRegistry?.reserveMemory(byteLength, 'cpu', 'scratch');
      else if (entry.reservation.byteLength !== byteLength) entry.reservation.resize(byteLength);
      const words = await readSelectionWords(mask);
      // A newer revision was requested while reading; its own load will publish.
      if (entry.requestedGeneration !== generation) return;
      entry.words = words;
      entry.appliedRevision = revision;
      entry.appliedMask = mask;
      entry.appliedGeneration = generation;
      entry.failure = undefined;
    } catch (error) {
      if (entry.requestedGeneration !== generation) return;
      entry.failure = error instanceof Error ? error.message : String(error);
    }
    context.requestUpdate();
  }

  release(layerId: string): void {
    const entry = this.entries.get(layerId);
    if (entry) entry.requestedGeneration++;
    entry?.reservation?.release();
    this.entries.delete(layerId);
  }

  destroy(): void {
    for (const layerId of [...this.entries.keys()]) this.release(layerId);
  }
}

/** Layer ids to release: cached state whose layer no longer exists in the Kepler state. */
export function getRemovedLayerIds(
  cachedLayerIds: Iterable<string>,
  activeLayerIds: ReadonlySet<string>
): string[] {
  return [...cachedLayerIds].filter(layerId => !activeLayerIds.has(layerId));
}
