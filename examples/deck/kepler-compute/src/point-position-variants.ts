// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PreparedArrowAnalyticsTable} from '@luma-contributors/analysis/analysis-core';
import type {
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';

import {
  estimateTabularPointPositionBytes,
  makeGPUVectorsFromTabularPointPositions,
  prepareTabularPointPositions,
  type TabularPointGPUVectors
} from './point-position-source';

export type PointPositionVariantRequest = {
  datasetId: string;
  datasetRevision: number;
  prepared: PreparedArrowAnalyticsTable;
  longitude: string;
  latitude: string;
  altitude?: string;
};

export type PointPositionVariant = {
  /** Monotonic id, usable as a selector geometry revision because a variant never mutates. */
  generation: number;
  vectors: TabularPointGPUVectors;
  /** Lazily attached projection runtime for grid/hexagon layers on this pair; destroyed with the variant. */
  aggregation?: {destroy(completion: Promise<unknown>): Promise<unknown>};
};

export type PointPositionVariantResult =
  | {status: 'ready'; variant: PointPositionVariant}
  | {status: 'unavailable'; reason: string};

type VariantEntry = {
  key: string;
  datasetId: string;
  variant: PointPositionVariant;
  layerIds: Set<string>;
  lease: KeplerDatasetLease;
};

/** Stable cache key: a variant is one immutable upload of one coordinate pair of one dataset revision. */
export function getPointPositionVariantKey(request: PointPositionVariantRequest): string {
  return [
    request.datasetId,
    request.datasetRevision,
    request.longitude,
    request.latitude,
    request.altitude ?? ''
  ].join('\u0000');
}

/**
 * Additional resident positions for point layers whose lng/lat columns differ from the dataset's
 * primary published positions. Variants are prepared on demand through the same tabular position
 * path as publication (so memory is reserved in the session registry as 'source' GPU bytes plus
 * temporary 'scratch' CPU bytes), shared by every layer with the same pair, and destroyed once
 * no layer references them and the GPU has finished the frames that used them.
 */
export class PointPositionVariantRuntime {
  private readonly entries = new Map<string, VariantEntry>();
  private readonly layerKeys = new Map<string, string>();
  private readonly failures = new Map<string, string>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly errors: unknown[] = [];
  private generation = 0;
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void
  ) {}

  /** GPU bytes currently held by variants (also reserved in the session memory registry). */
  get gpuByteLength(): number {
    let total = 0;
    for (const entry of this.entries.values()) total += entry.variant.vectors.gpuByteLength;
    return total;
  }

  /** The live variant a layer currently references (after `resolve`), if any. */
  getVariant(layerId: string): PointPositionVariant | undefined {
    const key = this.layerKeys.get(layerId);
    return key ? this.entries.get(key)?.variant : undefined;
  }

  get variantCount(): number {
    return this.entries.size;
  }

  /** Returns the cached variant for this layer, preparing it on first use. */
  resolve(layerId: string, request: PointPositionVariantRequest): PointPositionVariantResult {
    if (this.destroyed || this.session.deviceLost)
      return {status: 'unavailable', reason: 'Point position variants are unavailable.'};
    const key = getPointPositionVariantKey(request);
    const previousKey = this.layerKeys.get(layerId);
    if (previousKey && previousKey !== key) this.entries.get(previousKey)?.layerIds.delete(layerId);
    const existing = this.entries.get(key);
    if (existing && existing.lease.isValid()) {
      existing.layerIds.add(layerId);
      this.layerKeys.set(layerId, key);
      return {status: 'ready', variant: existing.variant};
    }
    const failure = this.failures.get(key);
    if (failure) return {status: 'unavailable', reason: failure};
    let scratch: KeplerMemoryReservation | undefined;
    let lease: KeplerDatasetLease | undefined;
    let vectors: TabularPointGPUVectors | undefined;
    try {
      const estimate = estimateTabularPointPositionBytes(
        request.prepared.sourceBatches.map(batch => batch.sourceRowCount),
        request.altitude !== undefined
      );
      scratch = this.session.registry.reserveMemory(estimate.cpuByteLength, 'cpu', 'scratch');
      lease = this.session.registry.acquireDataset(request.datasetId);
      vectors = makeGPUVectorsFromTabularPointPositions(
        this.session.device,
        prepareTabularPointPositions(request.prepared, {
          longitude: request.longitude,
          latitude: request.latitude,
          ...(request.altitude === undefined ? {} : {altitude: request.altitude})
        }),
        {memoryRegistry: this.session.registry}
      );
      this.entries.set(key, {
        key,
        datasetId: request.datasetId,
        variant: {generation: ++this.generation, vectors},
        layerIds: new Set([layerId]),
        lease
      });
      this.layerKeys.set(layerId, key);
      this.requestUpdate();
      return {status: 'ready', variant: this.entries.get(key)!.variant};
    } catch (error) {
      if (lease) this.track(lease.release());
      const reason = `Positions for ${request.longitude}/${request.latitude} cannot be prepared: ${
        error instanceof Error ? error.message : String(error)
      }`;
      this.failures.set(key, reason);
      return {status: 'unavailable', reason};
    } finally {
      scratch?.release();
    }
  }

  /** Drops the layer's reference (e.g. its mapping now matches the primary positions). */
  releaseLayer(layerId: string): void {
    const key = this.layerKeys.get(layerId);
    if (!key) return;
    this.layerKeys.delete(layerId);
    this.entries.get(key)?.layerIds.delete(layerId);
  }

  get hasUnreferencedVariants(): boolean {
    for (const entry of this.entries.values()) if (!entry.layerIds.size) return true;
    return false;
  }

  /** Keeps references only for existing layers, then destroys unreferenced variants after `completion`. */
  async retainLayers(ids: readonly string[], completion: PromiseLike<unknown>): Promise<void> {
    const retained = new Set(ids);
    // Sub-keys (`<layerId>\u0001<role>`) belong to the layer before the separator.
    for (const layerId of [...this.layerKeys.keys()])
      if (!retained.has(layerId.split('\u0001')[0])) this.releaseLayer(layerId);
    for (const [key, entry] of this.entries)
      if (!entry.layerIds.size) {
        this.entries.delete(key);
        this.retire(entry, completion);
      }
    // Failures are cached per (dataset revision, column pair); bounded so remapping can retry.
    if (this.failures.size > 32) this.failures.clear();
    await this.drain();
  }

  async retireDataset(datasetId: string, completion: PromiseLike<unknown>): Promise<void> {
    for (const [key, entry] of this.entries)
      if (entry.datasetId === datasetId) {
        this.entries.delete(key);
        for (const layerId of entry.layerIds) this.layerKeys.delete(layerId);
        this.retire(entry, completion);
      }
    await this.drain();
  }

  async destroy(completion: PromiseLike<unknown>): Promise<void> {
    this.destroyed = true;
    await this.retainLayers([], completion);
  }

  private retire(entry: VariantEntry, completion: PromiseLike<unknown>): void {
    this.track(
      (async () => {
        try {
          await entry.variant.aggregation?.destroy(Promise.resolve(completion));
          await entry.variant.vectors.destroy(completion);
        } finally {
          await entry.lease.release();
        }
      })()
    );
  }

  private track(promise: Promise<void>): void {
    this.retirements.add(promise);
    void promise
      .catch(error => this.errors.push(error))
      .finally(() => this.retirements.delete(promise));
  }

  private async drain(): Promise<void> {
    await Promise.allSettled([...this.retirements]);
    if (this.errors.length)
      throw new AggregateError(this.errors.splice(0), 'Point position variant cleanup failed');
  }
}
