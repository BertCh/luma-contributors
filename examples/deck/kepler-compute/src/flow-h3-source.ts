// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {Device} from '@luma.gl/core';
import type {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import {prepareH3SourceFromArrow, type PreparedH3Source} from './h3-source';
import {FlowRuntime, type ResidentFlowBinding} from './flow-runtime';
import type {FlowRoles} from './flow-source';

/** Synthetic Float64 center columns appended to the retained table for H3 endpoint flows. */
export const FLOW_H3_COLUMNS = {
  sourceLongitude: '__flowH3SourceLongitude',
  sourceLatitude: '__flowH3SourceLatitude',
  targetLongitude: '__flowH3TargetLongitude',
  targetLatitude: '__flowH3TargetLatitude'
} as const;

export type FlowH3Centers = {
  longitudes: Float64Array;
  latitudes: Float64Array;
  invalidCount: number;
};

/**
 * Native H3 flow endpoints are the cell centers (`getPositionFromHexValue`); invalid or null
 * cells become NaN so the row is never a flow, as in native.
 */
export function getFlowH3Centers(source: PreparedH3Source): FlowH3Centers {
  const longitudes = new Float64Array(source.rowCount).fill(Number.NaN);
  const latitudes = new Float64Array(source.rowCount).fill(Number.NaN);
  let invalidCount = 0;
  for (const batch of source.batches)
    for (let row = 0; row < batch.rowCount; row++) {
      if (!batch.validity[row]) {
        invalidCount++;
        continue;
      }
      longitudes[batch.sourceRowOffset + row] = batch.centers[row * 2];
      latitudes[batch.sourceRowOffset + row] = batch.centers[row * 2 + 1];
    }
  return {longitudes, latitudes, invalidCount};
}

/** Appends synthetic center columns so the lat/lng flow pipeline serves H3 endpoints unchanged. */
export function makeFlowH3Table(
  table: arrow.Table,
  sourceColumn: string,
  targetColumn: string
): {table: arrow.Table; invalidCount: number} | {reason: string} {
  const sourceVector = table.getChild(sourceColumn);
  const targetVector = table.getChild(targetColumn);
  if (!sourceVector || !targetVector)
    return {reason: 'A flow H3 column is missing from the retained source.'};
  const sourceCenters = getFlowH3Centers(prepareH3SourceFromArrow(sourceVector));
  const targetCenters = getFlowH3Centers(prepareH3SourceFromArrow(targetVector));
  return {
    table: table.assign(
      arrow.tableFromArrays({
        [FLOW_H3_COLUMNS.sourceLongitude]: sourceCenters.longitudes,
        [FLOW_H3_COLUMNS.sourceLatitude]: sourceCenters.latitudes,
        [FLOW_H3_COLUMNS.targetLongitude]: targetCenters.longitudes,
        [FLOW_H3_COLUMNS.targetLatitude]: targetCenters.latitudes
      })
    ),
    invalidCount: sourceCenters.invalidCount + targetCenters.invalidCount
  };
}

export type FlowH3Resolution = {
  binding?: ResidentFlowBinding;
  reason: string | null;
};

type FlowH3Context = {
  device: Device;
  table: arrow.Table | undefined;
  getRevision(): number;
  requestUpdate(): void;
  memoryRegistry?: KeplerDatasetRegistry;
};

/** Per-layer cache of H3-endpoint flow runtimes, built asynchronously from the retained table. */
export class FlowH3Bindings {
  private readonly entries = new Map<
    string,
    {
      key: string;
      table: arrow.Table;
      state: 'pending' | 'ready' | 'failed';
      runtime?: FlowRuntime;
      reason?: string;
    }
  >();

  resolve(
    layerId: string,
    columns: {[name: string]: {value?: string | null} | undefined},
    context: FlowH3Context
  ): FlowH3Resolution {
    const sourceColumn = columns['sourceH3']?.value;
    const targetColumn = columns['targetH3']?.value;
    if (!sourceColumn || !targetColumn)
      return {reason: 'Select source and target H3 columns for the flow layer.'};
    if (!context.table) return {reason: 'No retained source table is bound to this flow dataset.'};
    const roles = {
      count: columns['count']?.value ?? undefined,
      sourceName: columns['sourceName']?.value ?? undefined,
      targetName: columns['targetName']?.value ?? undefined
    };
    const key = [sourceColumn, targetColumn, roles.count, roles.sourceName, roles.targetName].join(
      '|'
    );
    let entry = this.entries.get(layerId);
    if (!entry || entry.key !== key || entry.table !== context.table) {
      entry?.runtime?.destroy();
      const created = {
        key,
        table: context.table,
        state: 'pending' as 'pending' | 'ready' | 'failed',
        runtime: undefined as FlowRuntime | undefined,
        reason: undefined as string | undefined
      };
      entry = created;
      this.entries.set(layerId, created);
      void this.create(created, sourceColumn, targetColumn, roles, context);
    }
    return entry.state === 'ready'
      ? {binding: entry.runtime?.binding, reason: null}
      : {
          reason:
            entry.state === 'failed'
              ? (entry.reason ?? 'H3 flow preparation failed.')
              : 'Preparing H3 flow endpoints.'
        };
  }

  private async create(
    entry: {
      state: 'pending' | 'ready' | 'failed';
      runtime?: FlowRuntime;
      reason?: string;
      table: arrow.Table;
    },
    sourceColumn: string,
    targetColumn: string,
    roles: Pick<FlowRoles, 'count' | 'sourceName' | 'targetName'>,
    context: FlowH3Context
  ): Promise<void> {
    try {
      const derived = makeFlowH3Table(entry.table, sourceColumn, targetColumn);
      if ('reason' in derived) {
        entry.state = 'failed';
        entry.reason = derived.reason;
        return;
      }
      entry.runtime = await FlowRuntime.create(derived.table, {
        coordinates: {
          longitude: FLOW_H3_COLUMNS.sourceLongitude,
          latitude: FLOW_H3_COLUMNS.sourceLatitude
        },
        roles: {
          targetLongitude: FLOW_H3_COLUMNS.targetLongitude,
          targetLatitude: FLOW_H3_COLUMNS.targetLatitude,
          ...roles
        },
        requestUpdate: context.requestUpdate,
        getRevision: context.getRevision,
        memoryRegistry: context.memoryRegistry,
        device: context.device
      });
      entry.state = 'ready';
    } catch (error) {
      entry.state = 'failed';
      entry.reason = error instanceof Error ? error.message : String(error);
    }
    context.requestUpdate();
  }

  getStates(): Record<string, {state: string; reason?: string; flowCount?: number}> {
    return Object.fromEntries(
      [...this.entries].map(([layerId, entry]) => [
        layerId,
        {
          state: entry.state,
          reason: entry.reason,
          flowCount: entry.runtime?.binding.getLayerData().flows.length
        }
      ])
    );
  }

  release(layerId: string): void {
    this.entries.get(layerId)?.runtime?.destroy();
    this.entries.delete(layerId);
  }

  /** Releases runtimes of layers that were removed from the Kepler state. */
  retainLayers(activeLayerIds: ReadonlySet<string>): void {
    for (const layerId of [...this.entries.keys()])
      if (!activeLayerIds.has(layerId)) this.release(layerId);
  }

  destroy(): void {
    for (const layerId of [...this.entries.keys()]) this.release(layerId);
  }
}
