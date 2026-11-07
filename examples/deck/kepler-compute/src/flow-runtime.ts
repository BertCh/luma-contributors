// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import type {FlowAggregateSource} from './flow-gpu-aggregation';
import {
  makeFlowDataFromPairs,
  prepareFlowSourceFromAggregates,
  type FlowLayerData,
  type FlowRoles,
  type PreparedFlowSource
} from './flow-source';

export type ResidentFlowBinding = {
  coordinates: {longitude: string; latitude: string};
  roles: FlowRoles;
  bounds: [number, number, number, number] | null;
  /** Current flows for native FlowmapLayer; its identity changes only when selection changes. */
  getLayerData(): FlowLayerData;
  /** Idempotent per mask and selection revision; schedules one bounded readback and render update. */
  update(selectionMask: GPUVector<'uint32'>): void;
};

type FlowRuntimeProps = {
  coordinates: {longitude: string; latitude: string};
  roles: FlowRoles;
  requestUpdate(): void;
  /** Selection revision that changes whenever the resident row mask changes. */
  getRevision(): number;
  memoryRegistry?: KeplerDatasetRegistry;
  /** GPU device for the resident aggregation; required unless `createAggregator` is supplied. */
  device?: Device;
  /** Replaces the GPU aggregation, for example with the CPU reference in Node tests. */
  createAggregator?(table: arrow.Table, props: FlowRuntimeProps): Promise<FlowAggregateSource>;
};

/**
 * Native Flow aggregates flows, location totals, clustering and top-N display inside the pinned
 * `@flowmap.gl` provider from CPU flow rows. This runtime keeps that library as the clustering
 * authority and the native FlowmapLayer as the renderer (WGSL FlowmapGPULayer over the same provider), but feeds both aggregates: the retained
 * table is streamed once into GPU vectors, locations are deduplicated and origin-destination pairs
 * summed on the GPU, and each selection revision reads back only the unique pairs (names and exact
 * coordinates are resolved on the CPU for the unique locations only). There is no row cap; the
 * limits apply to unique locations and aggregated pairs. See `flow-gpu-aggregation.ts` for the
 * deliberate differences from native (float32 location keys, one datum per aggregated pair).
 */
export class FlowRuntime {
  readonly source: PreparedFlowSource;
  readonly binding: ResidentFlowBinding;
  private layerData: FlowLayerData;
  private requested?: {mask: GPUVector<'uint32'>; revision: number};
  private generation = 0;
  private destroyed = false;
  private readonly reservation?: KeplerMemoryReservation;
  /** Readbacks of the creation pass are included; per selection revision there is one. */
  get readbackCount(): number {
    return this.aggregator.readbackCount;
  }
  get readbackBytes(): number {
    return this.aggregator.readbackBytes;
  }

  static async create(table: arrow.Table, props: FlowRuntimeProps): Promise<FlowRuntime> {
    const aggregator = await (props.createAggregator ?? createGpuAggregator)(table, props);
    try {
      // Native clusters with unfiltered weights, so the creation pass aggregates every row.
      const unfilteredPairs = await aggregator.aggregatePairs();
      const source = await prepareFlowSourceFromAggregates(
        table,
        props.coordinates,
        props.roles,
        aggregator,
        unfilteredPairs
      );
      return new FlowRuntime(source, aggregator, props);
    } catch (error) {
      aggregator.destroy();
      throw error;
    }
  }

  private constructor(
    source: PreparedFlowSource,
    private readonly aggregator: FlowAggregateSource,
    private readonly props: FlowRuntimeProps
  ) {
    this.source = source;
    this.reservation = props.memoryRegistry?.reserveMemory(this.source.byteLength, 'cpu', 'source');
    this.layerData = {
      locations: this.source.locations,
      clusterLevels: this.source.clusterLevels,
      flows: []
    };
    this.binding = {
      coordinates: props.coordinates,
      roles: props.roles,
      bounds: this.source.bounds,
      getLayerData: () => this.layerData,
      update: mask => this.update(mask, this.props.getRevision())
    };
  }

  private update(mask: GPUVector<'uint32'>, revision: number): void {
    if (this.destroyed) return;
    if (this.requested?.mask === mask && this.requested.revision === revision) return;
    this.requested = {mask, revision};
    const generation = ++this.generation;
    void this.aggregator
      .aggregatePairs(mask)
      .then(pairs => {
        if (this.destroyed || generation !== this.generation) return;
        this.layerData = makeFlowDataFromPairs(this.source, pairs);
        this.props.requestUpdate();
      })
      .catch(() => {
        // A failed or rejected aggregation keeps the previous flows; the next revision retries.
        if (generation === this.generation) this.requested = undefined;
      });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.generation++;
    this.aggregator.destroy();
    this.reservation?.release();
  }
}

async function createGpuAggregator(
  table: arrow.Table,
  props: FlowRuntimeProps
): Promise<FlowAggregateSource> {
  // Loaded lazily so Node tests that inject an aggregator never import the GPU modules.
  const {createFlowGpuAggregator} = await import('./flow-gpu-pipeline');
  // Assertion: a Flow layer needs the dataset device; pass `device` in FlowRuntimeProps.
  if (!props.device) throw new Error('Flow GPU aggregation requires a device');
  return createFlowGpuAggregator(props.device, table, props);
}
