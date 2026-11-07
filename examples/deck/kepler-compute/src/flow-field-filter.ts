// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma-contributors/graph/gpu-data';
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
};

/**
 * Per-layer cache of the CPU copy of the resident filter mask used to build the field grid.
 * The grid is rebuilt from the previous mask until the new one has been read back, so a filter
 * change never blanks the particles; the CPU words are reserved against the dataset budget.
 */
export class FlowFieldSelectionCache {
  private readonly entries = new Map<string, SelectionEntry>();

  /** Returns the applied words, the key fragment identifying them, and any failure reason. */
  resolve(
    layerId: string,
    context: FlowFieldSelectionContext | undefined
  ): {words: Uint32Array | undefined; key: string; failure: string | undefined} {
    if (!context) {
      this.release(layerId);
      return {words: undefined, key: 'unfiltered', failure: undefined};
    }
    const revision = context.getRevision();
    let entry = this.entries.get(layerId);
    if (!entry) {
      entry = {mask: context.mask, requestedRevision: -1, appliedRevision: null};
      this.entries.set(layerId, entry);
    }
    if (entry.requestedRevision !== revision || entry.mask !== context.mask) {
      entry.requestedRevision = revision;
      entry.mask = context.mask;
      void this.load(entry, revision, context);
    }
    return {
      words: entry.words,
      key: `filter:${entry.appliedRevision ?? 'pending'}`,
      failure: entry.failure
    };
  }

  private async load(
    entry: SelectionEntry,
    revision: number,
    context: FlowFieldSelectionContext
  ): Promise<void> {
    try {
      const byteLength = context.mask.length * 4;
      if (!entry.reservation)
        entry.reservation = context.memoryRegistry?.reserveMemory(byteLength, 'cpu', 'scratch');
      else if (entry.reservation.byteLength !== byteLength) entry.reservation.resize(byteLength);
      const words = await readSelectionWords(context.mask);
      // A newer revision was requested while reading; its own load will publish.
      if (entry.requestedRevision !== revision) return;
      entry.words = words;
      entry.appliedRevision = revision;
      entry.failure = undefined;
    } catch (error) {
      if (entry.requestedRevision !== revision) return;
      entry.failure = error instanceof Error ? error.message : String(error);
      entry.appliedRevision = revision;
    }
    context.requestUpdate();
  }

  release(layerId: string): void {
    const entry = this.entries.get(layerId);
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
