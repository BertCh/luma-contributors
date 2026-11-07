// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createLocalCloudProvider} from './local-cloud-provider';
import {ResidentClusterRuntime} from './cluster-runtime';
import {parseAutomaticPolicyConfig} from './automatic-recovery-policy';
import {appState} from './app-state';
import {ResidentResultRecipes} from './resident-result-recipes';
import {ResidentEvictionHost, collectVisibleDatasetIds} from './resident-eviction-host';
import {type KeplerMemoryReservation} from '@luma-contributors/kepler';
import {DATASET_ID} from './sample-data';
import {createMemoryManagerHost} from './memory-manager-host';
import {wrapTo, addNotification, toggleModal} from '@kepler.gl/actions';
import {createNotification} from '@kepler.gl/utils';
import {getDeckLayers} from './deck-internals';
import {setSandboxHostSettingsProvider} from './sandbox-host-settings';
import {KeplerComputeEffect} from './compute-runtime';
import {makeBasemapPicker} from './basemap-picker';
import {makeBasemapContext} from './basemap-context';
import {installNativeAccessibilityNames} from './native-accessibility-names';
import {installBasemapStyleUrlInput} from './basemap-style-url-input';
import {VectorTileResidentRenderer} from './vector-tile-resident-layers';
import {VectorTileLiveController} from './vector-tile-live-controller';
import {makeVectorTileBufferFactory} from './vector-tile-live-gpu';
import {renderVectorTileScopeStatus} from './vector-tile-live-status';
import {createWebGPUEffectsPipeline} from './effects-wgsl-effects';
import {createEffectsFilter} from './effects-webgpu-policy';
import {createDeviceRecovery, getEvictionRebuildSource, rebuildEvictedDataset} from './recovery';
import {store} from './app-store';
import {publicationsInProgress, removePublishedDataset} from './publication';

export const LOCAL_CLOUD_PROVIDERS = [createLocalCloudProvider()];

/** Exact native supercluster adapters (worker-built) keyed by their borrowed point vector. */
export const clusterRuntimes = new Map<unknown, ResidentClusterRuntime>();

export function destroyClusterRuntimes(): void {
  for (const clusterRuntime of clusterRuntimes.values()) clusterRuntime.destroy();
  clusterRuntimes.clear();
}

export const statusListeners = new Set<() => void>();

export const abandonedDeviceGenerations = new Set<number>();

export const AUTOMATIC_POLICY = parseAutomaticPolicyConfig(globalThis.location?.search ?? '');

appState.deviceRecovery = createDeviceRecovery();

/** Lineage of GPU-resident results, so device recovery and eviction can recompute them. */
export const residentResultRecipes = new ResidentResultRecipes();

export const residentEviction = new ResidentEvictionHost<KeplerMemoryReservation>({
  enabled: AUTOMATIC_POLICY.autoEvict,
  getVisibleDatasetIds() {
    const visState = store.getState().keplerGl['compute']?.visState;
    return collectVisibleDatasetIds({
      layers: (visState?.layers ?? []).map(layer => ({
        isVisible: Boolean(layer.config.isVisible),
        dataId: layer.config.dataId
      })),
      filters: visState?.filters ?? [],
      pinnedIds: [DATASET_ID, ...publicationsInProgress.keys()],
      lineage: residentResultRecipes.getLineage()
    });
  },
  canRebuild: datasetId => getEvictionRebuildSource(datasetId) !== undefined,
  evict(datasetId) {
    const owner = appState.runtime;
    if (!owner) return;
    // Registry removal releases owned bytes now; publication teardown (styles, fences) follows.
    owner.session.registry.removeDataset(datasetId);
    void owner.removePublication(datasetId).catch(reportError);
  },
  rebuild: datasetId => rebuildEvictedDataset(datasetId)
});

residentEviction.subscribe(() => notify());

/** Native memory manager: registry accounting, budget changes, unload/evict and Kepler notices. */
export const memoryManager = createMemoryManagerHost({
  getRegistry: () => appState.runtime?.session.registry,
  getLayers: () =>
    (store.getState().keplerGl['compute']?.visState?.layers ?? []).map(layer => ({
      id: layer.id,
      label: String(layer.config.label ?? layer.id),
      dataId: layer.config.dataId ?? null,
      isVisible: Boolean(layer.config.isVisible)
    })),
  getDatasetLabels: () =>
    Object.fromEntries(
      Object.entries(store.getState().keplerGl['compute']?.visState?.datasets ?? {}).map(
        ([datasetId, dataset]) => [
          datasetId,
          String((dataset as {label?: string}).label ?? datasetId)
        ]
      )
    ),
  getPinnedIds: () => new Set([DATASET_ID, ...publicationsInProgress.keys()]),
  getEvictedIds: () => residentEviction.status.evictedIds,
  isEvictionEnabled: residentEviction.isEnabled,
  canEvict: datasetId =>
    residentEviction.isEnabled &&
    getEvictionRebuildSource(datasetId) !== undefined &&
    !residentEviction.isEvicted(datasetId),
  evictDataset: datasetId => residentEviction.evictDataset(datasetId),
  unloadDataset: datasetId => void removePublishedDataset(datasetId).catch(reportError),
  emitNotice: notice =>
    void store.dispatch(
      wrapTo(
        'compute',
        addNotification(
          createNotification({
            id: notice.key,
            message: notice.message,
            type: notice.type,
            topic: 'global'
          })
        )
      )
    ),
  openManager: () => void store.dispatch(wrapTo('compute', toggleModal('residentMemory'))),
  getDeckLayers: () => getDeckLayers(appState.runtime?.deck)
});

// The chosen budget is saved with the workspace bundle and remembered per viewer in localStorage.
setSandboxHostSettingsProvider(memoryManager.settings);

export function notify(): void {
  for (const listener of statusListeners) listener();
}

export function reportError(error: unknown): void {
  memoryManager.reportError(error);
  appState.currentError = error instanceof Error ? error.message : String(error);
  // biome-ignore lint/suspicious/noConsole: preserve the diagnostic stack alongside the visible error
  console.error(error);
  notify();
}

export const effect = new KeplerComputeEffect(reportError);

// Optional external basemap style: failures explain themselves and keep the local background.
// Validated styles are also added to Kepler's native map-style picker (basemap-picker.ts).
export const basemapPicker = makeBasemapPicker({
  dispatch: action => store.dispatch(action as never),
  instanceId: 'compute'
});

export const basemapContext = makeBasemapContext({
  fetchStyle: basemapPicker.fetchStyle,
  onChange: state => {
    if (state.status === 'failed' && state.errorMessage) reportError(new Error(state.errorMessage));
  }
});

installNativeAccessibilityNames();

installBasemapStyleUrlInput({
  addStyle: url => basemapPicker.addStyle(basemapContext, url).then(result => result.state)
});

// Native vectorTile (MVT) layers: working-set admission, stable ids and resident tile batches.
export const vectorTileResident = new VectorTileResidentRenderer({
  getDevice: () =>
    appState.runtime && !appState.runtime.session.deviceLost ? appState.runtime.device : null,
  getReserveMemory: () =>
    appState.runtime
      ? (byteLength, memoryKind, category) =>
          appState.runtime!.session.registry.reserveMemory(byteLength, memoryKind, category)
      : null,
  onChange: () => queueMicrotask(notify)
});

export const vectorTileLive: VectorTileLiveController = new VectorTileLiveController({
  residentRenderer: vectorTileResident,
  getReserveMemory: () =>
    appState.runtime
      ? (byteLength, memoryKind, category) =>
          appState.runtime!.session.registry.reserveMemory(byteLength, memoryKind, category)
      : null,
  getCreateBuffer: () =>
    appState.runtime ? makeVectorTileBufferFactory(appState.runtime.device) : null,
  onChange: () => renderVectorTileScopeStatus(vectorTileLive)
});

// Kepler's GLSL-only shadow and fog effects are skipped on WebGPU with a visible reason.
const reportEffectMessage = (message: string) =>
  queueMicrotask(() => reportError(new Error(message)));

// Distance fog, surface fog and lighting have WGSL replacements; shadow maps and unknown GLSL effects are skipped.
export const webGPUEffectsPipeline = createWebGPUEffectsPipeline(
  reportEffectMessage,
  createEffectsFilter(reportEffectMessage)
);

export const filterWebGPUEffects = webGPUEffectsPipeline.filter;

export const layerDiagnostics = new Map<string, string>();

export const roleSetDiagnosticLayerIds = new Set<string>();

export const styleDiagnostics = new Map<string, string>();

export const plotDiagnostics = new Map<string, string>();

export const residentDatasetIds = new Set([DATASET_ID]);
