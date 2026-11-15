// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRasterLiveHost} from './raster-live-host';
import {flowFieldBindings} from './flow-field-bindings';
import {appState} from './app-state';
import {getDeckLayers} from './deck-internals';
import {executeTerrainResidentPublication} from './terrain-resident-publication';
import {store} from './app-store';
import {getActiveRuntime} from './native-config';
import {publishSource, attachPublishedDataset} from './publication';

export const rasterLiveHost = createRasterLiveHost({
  getLayers: () =>
    (store.getState().keplerGl['compute']?.visState?.layers ?? []) as unknown as readonly {
      id: string;
      type: string;
    }[],
  getDevice: () => appState.runtime?.device,
  listFlowGrids: () =>
    Object.entries(flowFieldBindings.getStats())
      .filter(([, stats]) => stats.mode === 'UV' || stats.mode === 'SPEED_DIR')
      .map(([id]) => ({id, label: id})),
  readFlowGrid: layerId => flowFieldBindings.readGridForAnalysis(layerId),
  getMemoryRegistry: () => appState.runtime?.session.registry,
  getDeckLayers: () => getDeckLayers(appState.runtime?.deck),
  listPolygonDatasets: () =>
    [...(appState.runtime?.publications ?? [])]
      .filter(([, published]) => published.geometryColumn)
      .map(([id]) => ({
        id,
        label:
          (
            store.getState().keplerGl['compute']?.visState?.datasets as
              | Record<string, {label?: string}>
              | undefined
          )?.[id]?.label ?? id
      })),
  getPolygonSource: datasetId => {
    const published = appState.runtime?.publications.get(datasetId);
    return published?.geometryColumn
      ? {table: published.dataset.value.source, geometryColumn: published.geometryColumn}
      : undefined;
  },
  getDerivedGrid: () => {
    const analysis = appState.runtime?.analysis;
    return analysis
      ? {
          isAvailable: () => analysis.hasInterpolationGrid(),
          read: async () => {
            const grid = await analysis.readInterpolationGrid();
            return {...grid, id: `${grid.sourceId}-${grid.operation}`};
          }
        }
      : undefined;
  },
  publishTable: async ({table, label, geometryColumn, normalization, isCurrent}) => {
    const owner = getActiveRuntime();
    await publishSource(
      {table, label, isCurrent: () => appState.runtime === owner && (isCurrent?.() ?? true)},
      normalization,
      undefined,
      geometryColumn
    );
  },
  publishResidentTerrain: async options => {
    const owner = getActiveRuntime();
    const result = await executeTerrainResidentPublication({
      session: owner.session,
      source: options.source,
      operation: options.operation,
      bandIndex: options.bandIndex,
      zFactor: options.zFactor,
      azimuthDegrees: options.azimuthDegrees,
      altitudeDegrees: options.altitudeDegrees,
      resultId: `terrain-${options.operation}-${Date.now().toString(36)}`,
      isCurrent: () => appState.runtime === owner
    });
    try {
      await attachPublishedDataset(
        owner,
        result.dataset,
        options.label,
        // Coordinate roles scan CPU source cells that resident results lack (assertNotResidentSource),
        // so the cells publish as a GPU-resident table with longitude/latitude columns and no map layer.
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        () => appState.runtime === owner && result.isCurrent()
      );
      result.transfer();
    } catch (error) {
      await result.destroy();
      throw error;
    }
    return {
      rowCount: result.rowCount,
      contributors: result.contributors,
      readbackBytes: result.readbackBytes
    };
  }
});

store.subscribe(() => rasterLiveHost.syncLayers());
