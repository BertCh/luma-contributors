// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {setFilter, receiveMapConfig, addDataToMap, removeDataset} from '@kepler.gl/actions';
import {DATASET_ID, SAMPLE_FIELDS, createSampleData} from './sample-data';
import {type KeplerJSONValue, type KeplerSandboxSource} from '@luma-contributors/kepler';
import {assertSingleViewConfiguration} from './resident-action-guards';
import KeplerGlSchema from '@kepler.gl/schemas';
import {appState} from './app-state';
import {getResidentJoinDerivedDataset} from './resident-join-lineage';
import {getResidentSuitabilityDerivedDataset} from './resident-suitability-lineage';
import {getStagedNativeDatasets, clearStagedNativeSources} from './bundle-native-sources';
import {ComputeRuntime} from './compute-runtime';
import {readSandboxFixture} from './sandbox-state';
import {store} from './app-store';
import {SUPPORTED_VIEW_COUNT} from './split-view';
import {
  beginPublication,
  getGroupByGeneration,
  waitForNativeDataset,
  finishPublication
} from './publication';
import {
  plotDiagnostics,
  effect,
  destroyClusterRuntimes,
  residentDatasetIds,
  residentEviction,
  notify
} from './app-services';
import {initializeDevice} from './device-init';
import {cancelSampleMapLoads, whenSampleMapMutationsSettled} from './sample-map-lifecycle';
import {getSampleMapProvenance, SAMPLE_MAP_METADATA_KEY} from './sample-map-provenance';

export function createResidentFilterAction(
  field: 'value' | 'category' | 'time' | 'timestamp',
  property: Parameters<typeof setFilter>[1],
  value: Parameters<typeof setFilter>[2]
) {
  const filters = store.getState().keplerGl['compute']?.visState?.filters ?? [];
  const index = filters.findIndex(
    filter =>
      filter.id === `resident-${field}` &&
      filter.name?.[0] === field &&
      filter.dataId.includes(DATASET_ID)
  );
  return index < 0 ? {type: 'LUMA_MISSING_FILTER'} : setFilter(index, property, value);
}

export function setNativeCategories(categories: number[]): void {
  // Kepler's empty multi-select means bypass; an inverted full set means select no rows.
  store.dispatch(createResidentFilterAction('category', 'inverted', categories.length === 0));
  store.dispatch(
    createResidentFilterAction('category', 'value', categories.length ? categories : [0, 1, 2, 3])
  );
}

export async function applyNativeConfig(config: KeplerJSONValue): Promise<void> {
  cancelSampleMapLoads();
  await whenSampleMapMutationsSettled();
  assertSingleViewConfiguration(config, SUPPORTED_VIEW_COUNT);
  if (
    !config ||
    typeof config !== 'object' ||
    Array.isArray(config) ||
    !('version' in config) ||
    !('config' in config)
  )
    throw new Error('Expected a saved Kepler configuration');
  const parsed = KeplerGlSchema.parseSavedConfig({
    version: config['version'],
    config: config['config']
  });
  if (!parsed) throw new Error('Kepler could not parse the saved configuration');
  const activeRuntime = getActiveRuntime();
  const publications = [...activeRuntime.publications.values()];
  const scopeBindings = [
    activeRuntime.tableBinding,
    ...publications.map(published => published.scopes)
  ];
  for (const binding of scopeBindings) binding.beginLayerRegistration();
  for (const published of publications) beginPublication(published.dataset.id);
  try {
    plotDiagnostics.clear();
    store.dispatch(receiveMapConfig(parsed, {keepExistingConfig: false, autoCreateLayers: false}));
    store.dispatch(
      addDataToMap({
        datasets: [
          {
            info: {id: DATASET_ID, label: 'Resident observations'},
            data: {fields: SAMPLE_FIELDS, rows: appState.source.rows}
          },
          ...publications.map(published => ({
            info: {id: published.dataset.id, label: published.label},
            data: published.nativeData,
            ...(getSampleMapProvenance(published.dataset.source?.metadata)
              ? {
                  metadata: {
                    [SAMPLE_MAP_METADATA_KEY]: getSampleMapProvenance(
                      published.dataset.source?.metadata
                    )
                  }
                }
              : {}),
            ...(published.groupBy
              ? {
                  metadata: {
                    lumaGroupByGeneration: getGroupByGeneration(published.dataset),
                    derivedDataset: {
                      type: 'groupBy',
                      sourceDataIds: [published.groupBy.source.datasetId],
                      operationId: published.groupBy.operationId
                    }
                  }
                }
              : published.join
                ? {
                    metadata: {
                      lumaGroupByGeneration: getGroupByGeneration(published.dataset),
                      derivedDataset: getResidentJoinDerivedDataset(published.join)
                    }
                  }
                : published.suitability
                  ? {
                      metadata: {
                        lumaGroupByGeneration: getGroupByGeneration(published.dataset),
                        derivedDataset: getResidentSuitabilityDerivedDataset(published.suitability)
                      }
                    }
                  : {})
          })),
          // Native-only originals restored from a portable bundle bind to the replayed config here.
          ...getStagedNativeDatasets()
        ],
        options: {
          centerMap: false,
          autoCreateLayers: false,
          autoCreateTooltips: false,
          keepExistingConfig: true
        }
      })
    );
    await Promise.all(
      [DATASET_ID, ...publications.map(published => published.dataset.id)].map(datasetId =>
        waitForNativeDataset(datasetId, activeRuntime)
      )
    );
    // Kepler can skip filterTable when the restored source has no targeted filters.
    // Commit an explicit empty plan as well, rather than retaining the previous source mask.
    const restoredState = store.getState().keplerGl['compute']?.visState;
    if (restoredState)
      for (const datasetId of [DATASET_ID, ...publications.map(published => published.dataset.id)])
        restoredState.datasets[datasetId]?.filterTable(restoredState.filters, restoredState.layers);
    activeRuntime.requestUpdate();
  } finally {
    clearStagedNativeSources();
    for (const published of publications) finishPublication(published.dataset.id);
    const failures: unknown[] = [];
    for (const binding of scopeBindings) {
      try {
        binding.endLayerRegistration();
      } catch (error) {
        failures.push(error);
      }
    }
    activeRuntime.requestUpdate();
    store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
    if (failures.length) throw new AggregateError(failures, 'Native layer registration failed');
  }
}

export function getActiveRuntime(): ComputeRuntime {
  if (!appState.runtime) throw new Error('Resident runtime is not ready');
  return appState.runtime;
}

export async function reimportSource(descriptor: KeplerSandboxSource): Promise<ComputeRuntime> {
  cancelSampleMapLoads();
  const fixture = readSandboxFixture(descriptor);
  const previous = getActiveRuntime();
  effect.runtime = null;
  destroyClusterRuntimes();
  appState.runtime = null;
  // A native table stores its resident binding. Recreate it with the new explorer rather than
  // copying a same-ID table whose source lease has just been closed.
  store.dispatch(removeDataset(DATASET_ID));
  for (const datasetId of previous.publications.keys()) {
    store.dispatch(removeDataset(datasetId));
    residentDatasetIds.delete(datasetId);
    residentEviction.forget(datasetId);
  }
  appState.latestSummary = null;

  previous.deck.setProps({layers: []});
  previous.deck.redraw('Resident source replacement');
  notify();
  await previous.destroy();
  appState.rowCount = fixture.rowCount;
  appState.source = createSampleData(appState.rowCount);
  initializeDevice(previous.deck, previous.device);
  return getActiveRuntime();
}
