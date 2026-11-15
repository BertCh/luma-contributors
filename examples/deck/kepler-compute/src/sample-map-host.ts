// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addDataToMap,
  updateDatasetProps,
  removeDataset,
  toggleModal,
  wrapTo
} from '@kepler.gl/actions';
import KeplerGlSchema from '@kepler.gl/schemas';
import {store} from './app-store';
import {appState} from './app-state';
import {notify} from './app-services';
import {DATASET_ID} from './sample-data';
import {fetchSampleMap, fetchSampleMaps} from './sample-map-loader';
import {
  describeSampleFallback,
  isSampleRecord,
  parseStartupSample,
  type SampleMap
} from './sample-map-catalog';
import type {SampleMapLimitation} from './sample-map-support';
import {nativeAutoPrepare} from './native-hosts';
import {removePublishedDataset, isResidentDataset} from './publication';
import {SampleRemoteServiceError} from './sample-map-remote.mjs';
import {
  beginSampleMapRequest,
  beginSampleMapMutation,
  REMOTE_SAMPLE_RECOVERY_NOTICE
} from './sample-map-lifecycle';
import {commitNativeSampleMap} from './sample-map-native-transaction';

type SamplePayload = Parameters<typeof addDataToMap>[0];
type SampleDataset = Exclude<SamplePayload['datasets'], unknown[]>;
type Request = ReturnType<typeof beginSampleMapRequest>;

export const sampleMapStatus: {
  loadingId?: string;
  loaded?: SampleMap;
  notice: string;
  error: string;
  limitations: SampleMapLimitation[];
  remoteServiceFailure?: {code: string; url: string; status: number};
  loadFailure?: {message: string; stack?: string};
  rollbackFailure?: {message: string; stack?: string};
} = {notice: '', error: '', limitations: []};

let hasStarted = false;
let loadTail: Promise<unknown> = Promise.resolve();

function getInstance() {
  return store.getState().keplerGl['compute'];
}

function captureDataset(datasetId: string): SampleDataset | undefined {
  const table = getInstance()?.visState?.datasets[datasetId];
  const published = appState.runtime?.publications.get(datasetId);
  if (!table)
    return published
      ? {
          info: {id: datasetId, label: published.label},
          data: published.nativeData,
          metadata: {...published.dataset.source?.metadata}
        }
      : undefined;
  const data = published?.nativeData ?? {
    fields: table.fields.map(field => ({...field})),
    rows: table.type
      ? []
      : Array.from({length: table.dataContainer.numRows()}, (_, rowIndex) =>
          table.fields.map((_, columnIndex) => table.dataContainer.valueAt(rowIndex, columnIndex))
        )
  };
  return {
    info: {
      id: table.id,
      label: table.label,
      color: table.color,
      ...(table.type ? {type: table.type} : {})
    },
    data,
    metadata: {...table.metadata}
  };
}

async function executeSampleMap(
  sample: SampleMap,
  startup: boolean,
  request: Request
): Promise<void> {
  if (!request.isCurrent()) return;
  const owner = appState.runtime;
  const ownerController = new AbortController();
  const isHealthy = () =>
    Boolean(
      owner &&
        appState.runtime === owner &&
        !owner.session.deviceLost &&
        !owner.device.isLost &&
        appState.deviceRecovery.status.phase === 'ready'
    );
  const assertOwner = () => {
    request.signal.throwIfAborted();
    if (!isHealthy())
      throw new Error('The GPU workspace changed while the sample was loading. Please retry.');
  };
  const monitor = () => {
    if (!isHealthy()) {
      const reason = new Error('The GPU workspace changed while the sample was loading.');
      ownerController.abort(reason);
      request.abort(reason);
    }
  };
  const interval = setInterval(monitor, 50);
  const unsubscribe = appState.deviceRecovery.subscribe(monitor);
  sampleMapStatus.loadingId = sample.id;
  sampleMapStatus.error = '';
  sampleMapStatus.remoteServiceFailure = undefined;
  sampleMapStatus.loadFailure = undefined;
  sampleMapStatus.rollbackFailure = undefined;
  notify();
  let previousConfig: ReturnType<typeof KeplerGlSchema.getConfigToSave>;
  let backup: SampleDataset | undefined;
  let backupDatasets: SampleDataset[] = [];
  let datasetId: string | undefined;
  let wasResident = false;
  let finishMutation: (() => void) | undefined;
  try {
    assertOwner();
    const {limitations, ...payload} = await fetchSampleMap(sample, fetch, request.signal);
    assertOwner();
    const datasets = Array.isArray(payload.datasets) ? payload.datasets : [payload.datasets];
    if (datasets.length !== 1 || !datasets[0]?.info.id)
      throw new Error('A sample must provide exactly one dataset.');
    datasetId = datasets[0].info.id;
    previousConfig = KeplerGlSchema.getConfigToSave(getInstance()!);
    backupDatasets = [
      ...new Set([
        ...Object.keys(getInstance()?.visState?.datasets ?? {}),
        ...owner!.publications.keys()
      ])
    ].flatMap(retainedDatasetId => {
      const captured = captureDataset(retainedDatasetId);
      return captured ? [captured] : [];
    });
    backup = backupDatasets.find(dataset => dataset.info.id === datasetId);
    wasResident = isResidentDataset(datasetId);
    if (sample.dataUrl && wasResident && nativeAutoPrepare.enabled === false)
      throw new Error(
        'Reloading a resident sample requires automatic preparation to replace its GPU data. Enable automatic preparation before retrying.'
      );
    // Capture the healthy presentation before Kepler applies the incoming config asynchronously.
    appState.deviceRecovery.checkpoint(owner!);
    finishMutation = beginSampleMapMutation();
    await commitNativeSampleMap(
      onFinish => {
        assertOwner();
        // A resident table may use an Arrow container even when the sample is CSV/GeoJSON.
        // Create a fresh native table rather than feeding row snapshots to that old container.
        if (backup) store.dispatch(wrapTo('compute', removeDataset(datasetId!)));
        assertOwner();
        // Kepler's keepExistingConfig:false resets native datasets as well as presentation.
        // Re-register retained sources in the same transaction so their tables and metadata survive.
        const action = {
          ...addDataToMap({
            ...payload,
            datasets: [
              ...datasets,
              ...backupDatasets.filter(dataset => dataset.info.id !== datasetId)
            ]
          }),
          onFinish
        };
        store.dispatch(wrapTo('compute', action));
      },
      AbortSignal.any([request.signal, AbortSignal.timeout(60000)]),
      isHealthy
    );
    assertOwner();
    if (!getInstance()?.visState?.datasets[datasetId])
      throw new Error(`Kepler did not register sample dataset ${datasetId}.`);
    assertOwner();
    store.dispatch(
      wrapTo(
        'compute',
        updateDatasetProps(datasetId, {label: sample.label, metadata: datasets[0]!.metadata})
      )
    );
    if (sample.dataUrl && nativeAutoPrepare.enabled !== false) {
      // Includes reloads of resident IDs: this uses publishSource's dataset operation queue.
      await nativeAutoPrepare.republish(
        datasetId,
        AbortSignal.any([request.signal, AbortSignal.timeout(180000)])
      );
      assertOwner();
    }
    const saved: unknown = payload.config;
    if (isSampleRecord(saved) && isSampleRecord(saved['config'])) {
      const visState = saved['config']['visState'];
      const expected =
        isSampleRecord(visState) && Array.isArray(visState['layers']) ? visState['layers'] : [];
      const current = getInstance()?.visState?.layers ?? [];
      const missing = expected.filter(
        layer => isSampleRecord(layer) && !current.some(candidate => candidate.id === layer['id'])
      );
      if (missing.length)
        throw new Error(
          `Kepler did not restore sample layer(s): ${missing.map(layer => (isSampleRecord(layer) ? String(layer['id']) : '?')).join(', ')}.`
        );
    }
    assertOwner();
    if (startup) store.dispatch(wrapTo('compute', removeDataset(DATASET_ID)));
    assertOwner();
    store.dispatch(wrapTo('compute', toggleModal(null)));
    assertOwner();
    sampleMapStatus.loaded = sample;
    sampleMapStatus.limitations = limitations;
    sampleMapStatus.notice = !sample.dataUrl
      ? REMOTE_SAMPLE_RECOVERY_NOTICE
      : nativeAutoPrepare.enabled === false
        ? 'Automatic preparation is disabled. Open Add Data → Prepare data to publish this sample to the GPU.'
        : '';
  } catch (failure) {
    let error = failure;
    const describeFailure = (value: unknown) => ({
      message: String(value),
      stack: value instanceof Error ? value.stack : undefined
    });
    if (request.isCurrent()) sampleMapStatus.loadFailure = describeFailure(failure);
    // New requests are serialized behind this rollback. Never mutate a replacement/lost workspace.
    if (finishMutation && isHealthy()) {
      try {
        if (datasetId && !backup) await removePublishedDataset(datasetId);
        if (!isHealthy()) throw new Error('Workspace changed before sample rollback.');
        // Remove the changed table so rollback cannot reuse its mutated data container.
        if (datasetId && backup) store.dispatch(wrapTo('compute', removeDataset(datasetId)));
        const rollbackSignal = AbortSignal.any([
          ownerController.signal,
          AbortSignal.timeout(60000)
        ]);
        await commitNativeSampleMap(
          onFinish => {
            if (!isHealthy()) throw new Error('Workspace changed before sample rollback.');
            const action = {
              ...addDataToMap({
                datasets: backupDatasets,
                config: previousConfig!,
                options: {centerMap: false, keepExistingConfig: false, autoCreateLayers: false}
              }),
              onFinish
            };
            store.dispatch(wrapTo('compute', action));
          },
          rollbackSignal,
          isHealthy
        );
        if (!isHealthy()) throw new Error('Workspace changed during sample rollback.');
        if (datasetId && wasResident) await nativeAutoPrepare.republish(datasetId, rollbackSignal);
        if (!isHealthy()) throw new Error('Workspace changed after sample rollback.');
      } catch (rollbackError) {
        if (request.isCurrent()) sampleMapStatus.rollbackFailure = describeFailure(rollbackError);
        error = new AggregateError(
          [failure, rollbackError],
          `Sample load failed: ${String(failure)}; restoring the previous presentation failed: ${String(rollbackError)}`,
          {cause: failure}
        );
      }
    }
    if (request.isCurrent() && isHealthy()) {
      if (error instanceof SampleRemoteServiceError)
        sampleMapStatus.remoteServiceFailure = {
          code: error.code,
          url: error.url,
          status: error.status
        };
      sampleMapStatus.error = error instanceof Error ? error.message : String(error);
      // This notice is only true after a successful rollback (or failure before any mutation).
      if (startup && !(error instanceof AggregateError))
        sampleMapStatus.notice = describeSampleFallback(sample.id, error);
      else if (startup) sampleMapStatus.notice = '';
    }
    throw error;
  } finally {
    finishMutation?.();
    clearInterval(interval);
    unsubscribe();
    if (sampleMapStatus.loadingId === sample.id) sampleMapStatus.loadingId = undefined;
    if (isHealthy()) appState.deviceRecovery.checkpoint(owner!);
    notify();
  }
}

function enqueueSampleMap(sample: SampleMap, startup: boolean, request: Request): Promise<void> {
  const result = loadTail.then(() => executeSampleMap(sample, startup, request));
  loadTail = result.catch(() => {});
  return result;
}

/** Explicit loads invalidate pending startup/catalog work before waiting for the previous rollback. */
export function loadSampleMap(sample: SampleMap): Promise<void> {
  return enqueueSampleMap(sample, false, beginSampleMapRequest());
}

/** Startup owns a cancellable generation from the catalog request through final publication. */
export async function initializeStartupSample(): Promise<void> {
  if (hasStarted) return;
  hasStarted = true;
  const {sampleId} = parseStartupSample(location.search);
  if (sampleId === 'synthetic') return;
  const request = beginSampleMapRequest();
  const owner = appState.runtime;
  const monitor = () => {
    if (
      appState.runtime !== owner ||
      owner?.session.deviceLost ||
      owner?.device.isLost ||
      appState.deviceRecovery.status.phase !== 'ready'
    )
      request.abort(new Error('Workspace changed during startup.'));
  };
  const interval = setInterval(monitor, 50);
  sampleMapStatus.notice = `Loading Kepler sample “${sampleId}”…`;
  notify();
  try {
    const samples = await fetchSampleMaps(fetch, request.signal);
    monitor();
    if (!request.isCurrent()) return;
    const sample = samples.find(entry => entry.id === sampleId);
    if (!sample) throw new Error('No sample with this ID is listed in the public catalog.');
    await enqueueSampleMap(sample, true, request);
  } catch (error) {
    if (request.isCurrent() && !sampleMapStatus.error) {
      sampleMapStatus.notice = describeSampleFallback(sampleId, error);
      notify();
    }
  } finally {
    clearInterval(interval);
  }
}
