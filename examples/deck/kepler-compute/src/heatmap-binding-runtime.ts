// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder} from '@luma.gl/core';
import {KeplerResidentStyle} from '@luma-contributors/kepler';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerDatasetSnapshot,
  KeplerMemoryReservation,
  GPUResidentScalarFormat,
  KeplerResidentStylePlan,
  KeplerResidentStyleLegend
} from '@luma-contributors/kepler';
import {
  getResidentHeatmapUnsupportedReason,
  RESIDENT_HEATMAP_WORKING_MEMORY_BYTES
} from './resident-heatmap-layer';
import type {ResidentHeatmapBinding} from './resident-heatmap-layers';
import {isNativeHeatmapWeightScale} from './heatmap-weight-scales';

type HeatmapOwner = {
  datasetId: string;
  revision: number;
  lease: KeplerDatasetLease;
  reservation: KeplerMemoryReservation;
  deviceLost: PromiseLike<unknown>;
  style?: KeplerResidentStyle;
  legend?: KeplerResidentStyleLegend;
  densityRevision: number;
  selectionRevision: number;
  selectionMask: GPUVector<'uint32'>;
};

/** Accounts Deck-owned heatmap textures and keeps borrowed sources alive through view retirement. */
export class HeatmapBindingRuntime {
  private readonly entries = new Map<string, HeatmapOwner>();
  private readonly retirements = new Set<Promise<void>>();
  private destroyed = false;
  private readonly failures: unknown[] = [];
  private nextRevision = 0;
  private readonly listeners = new Set<() => void>();
  constructor(
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void
  ) {}

  resolve(options: {
    layerId: string;
    dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
    positions: GPUVector<'float32x2'>;
    columns: {lat: string; lng: string};
    selectionMask: GPUVector<'uint32'>;
    revision: number;
    weightColumn?: string;
    weightScale?: string | number | symbol;
    domainMask?: GPUVector<'uint32'>;
    domainRevision: number;
  }): {binding?: ResidentHeatmapBinding; reason?: string} {
    if (this.destroyed || this.session.registry.getDataset(options.dataset.id) !== options.dataset)
      return {reason: 'The heatmap source is stale or closed.'};
    const unavailable = getResidentHeatmapUnsupportedReason(this.session.device);
    if (unavailable) return {reason: unavailable};
    const weights = options.weightColumn
      ? options.dataset.value.columns[options.weightColumn]
      : undefined;
    if (options.weightColumn && (!weights || !isFloat32Vector(weights)))
      return {reason: 'Resident heatmap weights require a prepared Float32 numeric field.'};
    const scale = options.weightScale ?? 'linear';
    if (options.weightColumn && !isNativeHeatmapWeightScale(scale))
      return {
        reason:
          'Resident heatmap weights support native linear, square-root and logarithmic scales.'
      };
    const plan: KeplerResidentStylePlan | undefined =
      options.weightColumn && isNativeHeatmapWeightScale(scale)
        ? {
            id: `${options.layerId}:weight`,
            source: {datasetId: options.dataset.id, revision: options.dataset.revision},
            column: options.weightColumn,
            kind: 'scalar',
            scale,
            range: [0, 1],
            nullValue: 0,
            domain: options.domainMask
              ? {kind: 'selection', revision: options.domainRevision}
              : {kind: 'source'}
          }
        : undefined;
    let owner = this.entries.get(options.layerId);
    if (
      owner &&
      (owner.datasetId !== options.dataset.id ||
        owner.revision !== options.dataset.revision ||
        Boolean(owner.style) !== Boolean(plan) ||
        (plan && !owner.style?.canUpdate(plan, options.domainMask)))
    ) {
      const fence = this.session.device.createFence();
      this.entries.delete(options.layerId);
      void this.retire(owner, fence.signaled).then(
        () => fence.destroy(),
        () => fence.destroy()
      );
      owner = undefined;
    }
    if (!owner) {
      const reservation = this.session.registry.reserveMemory(
        RESIDENT_HEATMAP_WORKING_MEMORY_BYTES,
        'gpu',
        'scratch'
      );
      let style: KeplerResidentStyle | undefined;
      try {
        if (plan)
          style = new KeplerResidentStyle({
            session: this.session,
            plan,
            selectionMask: options.domainMask,
            workingMemoryBudgetBytes:
              4 * 1024 * 1024 +
              options.dataset.rowCount * 128 +
              options.dataset.batches.length * 1024,
            maximumLegendBytes: 16
          });
        const lease = this.session.registry.acquireDataset(options.dataset.id);
        owner = {
          datasetId: options.dataset.id,
          revision: options.dataset.revision,
          lease,
          reservation,
          deviceLost: this.session.device.lost,
          style,
          densityRevision: ++this.nextRevision,
          selectionRevision: options.revision,
          selectionMask: options.selectionMask
        };
        this.entries.set(options.layerId, owner);
        this.requestUpdate();
      } catch (error) {
        void style?.destroy(Promise.resolve());
        reservation.release();
        throw error;
      }
    }
    if (!owner.lease.isValid()) return {reason: 'The heatmap source lease expired.'};
    const previousStyleRevision = owner.style?.currentRevision;
    if (plan) owner.style?.update(plan);
    if (
      owner.selectionRevision !== options.revision ||
      owner.selectionMask !== options.selectionMask ||
      previousStyleRevision !== owner.style?.currentRevision
    ) {
      owner.selectionRevision = options.revision;
      owner.selectionMask = options.selectionMask;
      owner.densityRevision = ++this.nextRevision;
      this.requestUpdate();
    }
    if (owner.style) {
      if (owner.legend?.revision !== owner.style.currentRevision) return {};
      if (owner.legend.domainStatus === 'invalid-log-domain')
        return {
          reason: 'The native logarithmic heatmap weight domain crosses zero or ends at zero.'
        };
    }
    return {
      binding: {
        positions: options.positions,
        columns: options.columns,
        selectionMask: options.selectionMask,
        revision: owner.densityRevision,
        weightColumn: options.weightColumn,
        weights: owner.style?.values,
        weightValidity: owner.style?.validity
      }
    };
  }

  encode(commandEncoder: CommandEncoder): void {
    for (const owner of this.entries.values()) owner.style?.encode(commandEncoder);
  }

  async readSummary(): Promise<void> {
    await Promise.all(
      [...this.entries.values()].map(async owner => {
        const style = owner.style;
        if (!style || owner.legend?.revision === style.currentRevision) return;
        const legend = await style.readLegend();
        if (
          !legend ||
          this.destroyed ||
          ![...this.entries.values()].includes(owner) ||
          legend.revision !== style.currentRevision
        )
          return;
        owner.legend = legend;
        this.requestUpdate();
        for (const listener of this.listeners) listener();
      })
    );
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getDiagnostics(layerId: string) {
    const owner = this.entries.get(layerId);
    if (!owner || !owner.lease.isValid()) return undefined;
    return {
      revision: owner.densityRevision,
      selectionRevision: owner.selectionRevision,
      styleRevision: owner.style?.currentRevision,
      plan: owner.style?.currentPlan,
      legend: owner.legend?.revision === owner.style?.currentRevision ? owner.legend : undefined,
      weightBufferIds: owner.style?.values?.data.map(chunk => chunk.buffer.id),
      weightChunks: owner.style?.values?.data.map(chunk => chunk.length),
      readbackBytes: owner.style?.readbackByteLength ?? 0,
      executionCounts: owner.style?.executionCounts
    };
  }

  retainLayers(layerIds: readonly string[], completion: PromiseLike<unknown>): Promise<void> {
    const retained = new Set(layerIds);
    return this.retireMatching(layerId => !retained.has(layerId), completion);
  }
  retireDataset(datasetId: string, completion: PromiseLike<unknown>): Promise<void> {
    return this.retireMatching((_layerId, owner) => owner.datasetId === datasetId, completion);
  }
  async destroy(completion: PromiseLike<unknown>): Promise<void> {
    this.destroyed = true;
    this.listeners.clear();
    const remaining = this.retireMatching(() => true, completion);
    await Promise.allSettled([remaining, ...this.retirements]);
    if (this.failures.length)
      throw new AggregateError(this.failures, 'Heatmap resource retirement failed');
  }
  private retireMatching(
    predicate: (layerId: string, owner: HeatmapOwner) => boolean,
    completion: PromiseLike<unknown>
  ): Promise<void> {
    const pending: Promise<void>[] = [];
    for (const [layerId, owner] of this.entries)
      if (predicate(layerId, owner)) {
        this.entries.delete(layerId);
        pending.push(this.retire(owner, completion));
      }
    return Promise.allSettled(pending).then(outcomes => {
      const failures = outcomes.flatMap(outcome =>
        outcome.status === 'rejected' ? [outcome.reason] : []
      );
      if (failures.length) throw new AggregateError(failures, 'Heatmap resource retirement failed');
    });
  }
  private retire(owner: HeatmapOwner, completion: PromiseLike<unknown>): Promise<void> {
    const pending = Promise.race([Promise.resolve(completion), owner.deviceLost])
      .finally(async () => {
        try {
          try {
            await owner.style?.destroy(Promise.resolve());
          } finally {
            await owner.lease.release();
          }
        } finally {
          owner.reservation.release();
        }
      })
      .then(() => undefined);
    this.retirements.add(pending);
    void pending.then(
      () => this.retirements.delete(pending),
      error => {
        this.retirements.delete(pending);
        this.failures.push(error);
      }
    );
    return pending;
  }
}

function isFloat32Vector(value: GPUVector<GPUResidentScalarFormat>): value is GPUVector<'float32'> {
  return value.format === 'float32';
}
