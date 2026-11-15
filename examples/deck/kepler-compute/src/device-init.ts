// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {appState} from './app-state';
import {createOrUpdateFilter, addDataToMap} from '@kepler.gl/actions';
import {DATASET_ID, SAMPLE_FIELDS} from './sample-data';
import {Deck} from '@deck.gl/core';
import {type Device} from '@luma.gl/core';
import {SINGLE_VIEW_REASON} from './resident-action-guards';
import {ComputeRuntime} from './compute-runtime';
import {
  recordUploadMeasure,
  recordSummaryObserved,
  recordSubmittedFrame
} from './acceptance-instrumentation';
import {getKeplerComputeTableState} from './kepler-table-bridge';
import {registerAnalysisTooltipProvider} from './analysis-tooltip-provider';
import {readPointPosition} from './analysis-neighbor-tooltip';
import {SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS} from './time-runtime';
import {store} from './app-store';
import {createResidentFilterAction} from './native-config';
import {
  abandonedDeviceGenerations,
  notify,
  reportError,
  effect,
  destroyClusterRuntimes,
  plotDiagnostics,
  styleDiagnostics,
  residentEviction,
  memoryManager
} from './app-services';
import {createDeviceRecovery, invalidateLostRuntime, retireRecoveryRuntime} from './recovery';
import {attachPublishedDataset} from './publication';
import {initializeStartupSample} from './sample-map-host';
import {cancelSampleMapLoads} from './sample-map-lifecycle';

export function setTimestampRange(minimum: number, maximum: number): void {
  appState.runtime?.time.validateRange(minimum, maximum);
  store.dispatch(
    createOrUpdateFilter('resident-timestamp', DATASET_ID, 'timestamp', [minimum, maximum])
  );
  store.dispatch(
    createResidentFilterAction('timestamp', 'plotType', {type: 'histogram', interval: '1-hour'})
  );
}

export function initializeDevice(
  deck: Deck,
  device: Device,
  generation = appState.deviceGeneration
): void {
  if (generation !== appState.deviceGeneration || abandonedDeviceGenerations.has(generation)) {
    deck.finalize();
    return;
  }
  const pendingInitialization =
    appState.pendingDeviceInitialization?.generation === generation
      ? appState.pendingDeviceInitialization
      : null;
  if (pendingInitialization?.signal.aborted) {
    deck.finalize();
    pendingInitialization.reject(pendingInitialization.signal.reason);
    return;
  }
  if (appState.runtime && appState.runtime.deck !== deck && !pendingInitialization) {
    deck.finalize();
    appState.workflowNotice = SINGLE_VIEW_REASON;
    notify();
    return;
  }
  if (appState.runtime?.device === device) return;
  cancelSampleMapLoads();
  if (!pendingInitialization) appState.deviceRecovery = createDeviceRecovery();
  if (device.type !== 'webgpu') {
    reportError(new Error('This resident example requires WebGPU.'));
    return;
  }
  if (appState.runtime) void appState.runtime.destroy().catch(reportError);
  effect.runtime = null;
  destroyClusterRuntimes();
  appState.runtime = null;
  plotDiagnostics.clear();
  const runtimeUploadStartMilliseconds = performance.now();
  try {
    appState.runtime = new ComputeRuntime(device, deck, appState.source, {
      onSelectionChange() {
        appState.unsupportedSelection = false;
        appState.currentError = '';
      },
      onPlotUnsupported(filterId, reason) {
        if (reason) plotDiagnostics.set(filterId, reason);
        else plotDiagnostics.delete(filterId);
        queueMicrotask(notify);
      },
      onStyleUnsupported(layerId, reason) {
        if (reason) styleDiagnostics.set(layerId, reason);
        else styleDiagnostics.delete(layerId);
        queueMicrotask(notify);
      },
      onUnsupported(reason) {
        appState.unsupportedSelection = true;
        reportError(reason);
        queueMicrotask(() => {
          if (appState.runtime?.deck === deck) store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
        });
        deck.redraw('Unsupported resident selection');
      }
    });
    recordUploadMeasure(
      'runtime-source',
      runtimeUploadStartMilliseconds,
      appState.runtime.uploadBytes ?? 0
    );
    effect.runtime = pendingInitialization ? null : appState.runtime;
    appState.runtime.styles.subscribe(() => {
      store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
      notify();
    });
    const activeRuntime = appState.runtime;
    appState.runtime.heatmaps.subscribe(() => {
      if (appState.runtime !== activeRuntime) return;
      store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
      notify();
    });
    appState.runtime.labels.subscribe(() => {
      if (appState.runtime !== activeRuntime) return;
      store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
      notify();
    });
    appState.runtime.getPublicationBridge = datasetId =>
      getKeplerComputeTableState(
        store.getState().keplerGl['compute']?.visState?.datasets[datasetId]
      );
    // Histograms published into a table bridge reach native plots only through the filter-plot seam.
    // Native plots of a published dataset are rebuilt only by LUMA_PUBLISH_SUMMARY. Refresh once
    // per ready bridge revision while a filter on the dataset still shows a pending plot; a filter
    // edit can mark its plot pending at the new revision before the histograms arrive, so the plot
    // revision alone cannot tell whether the refresh has happened.
    const refreshedRevisions = new Map<string, number>();
    appState.runtime.onPublicationHistograms = datasetId => {
      if (appState.runtime !== activeRuntime) return;
      const visState = store.getState().keplerGl['compute']?.visState;
      const bridge = getKeplerComputeTableState(visState?.datasets[datasetId]);
      if (bridge?.status !== 'ready' || refreshedRevisions.get(datasetId) === bridge.revision)
        return;
      const hasPendingPlot = visState?.filters.some(filter => {
        const ids = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
        const plot = (filter as {lumaComputePlot?: {status: string}}).lumaComputePlot;
        return ids.includes(datasetId) && plot?.status === 'pending';
      });
      if (!hasPendingPlot) return;
      refreshedRevisions.set(datasetId, bridge.revision);
      store.dispatch({type: 'LUMA_PUBLISH_SUMMARY'});
    };
    let publishedSqlRevision = 0;
    appState.runtime.duckdb.subscribe(() => {
      notify();
      const imported = activeRuntime.duckdb.importedDataset;
      if (
        !imported ||
        imported.revision === publishedSqlRevision ||
        appState.runtime !== activeRuntime
      )
        return;
      publishedSqlRevision = imported.revision;
      void attachPublishedDataset(activeRuntime, imported, 'SQL result').catch(reportError);
    });
    registerAnalysisTooltipProvider(appState.runtime.analysis, {
      readPosition: sourceRowIndex => readPointPosition(activeRuntime.positions, sourceRowIndex)
    });
    appState.runtime.analysis.subscribe(() => store.dispatch({type: 'LUMA_RENDER_ANALYSIS'}));
    appState.runtime.subscribe(summary => {
      if (
        appState.runtime !== activeRuntime ||
        activeRuntime.session.deviceLost ||
        appState.deviceRecovery.status.phase !== 'ready'
      )
        return;
      appState.latestSummary = summary;
      if (summary.revision === activeRuntime.explorer.currentRevision) recordSummaryObserved();
      const table = store.getState().keplerGl['compute']?.visState?.datasets[DATASET_ID];
      const bridge = getKeplerComputeTableState(table);
      if (bridge) appState.runtime?.tableBinding.publishHistograms(bridge);
      const timeHistogram = summary.histograms['timestamp'];
      if (bridge && timeHistogram)
        bridge.setHistogram(
          'timestamp',
          Array.from(timeHistogram, (count, index) => ({
            x0: SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS + index * 3600000,
            x1: SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS + (index + 1) * 3600000,
            count,
            indexes: []
          })),
          bridge.revision
        );
      const histogram = summary.histograms['value'];
      if (bridge && histogram) {
        bridge.setHistogram(
          'value',
          // The native range plot's own edges (d3 ticks), not an equal-width relabeling of its counts.
          appState.runtime?.tableBinding.getFilterHistogram('resident-value') ??
            Array.from(histogram, (count, index) => ({
              x0: (index * 100) / histogram.length,
              x1: ((index + 1) * 100) / histogram.length,
              count,
              indexes: []
            })),
          bridge.revision
        );
        store.dispatch({type: 'LUMA_PUBLISH_SUMMARY'});
      }
    });
    void device.lost.then(information => {
      if (appState.runtime !== activeRuntime) return;
      const reason = `GPU device lost: ${information.message}`;
      if (appState.deviceRecovery.status.phase === 'rebuilding') {
        invalidateLostRuntime(activeRuntime, reason);
        void appState.deviceRecovery.cancelRecovery().catch(reportError);
      } else if (!appState.deviceRecovery.markLost(activeRuntime, reason, Promise.resolve())) {
        invalidateLostRuntime(activeRuntime, reason);
      }
    });
    store.dispatch(
      addDataToMap({
        datasets: {
          info: {id: DATASET_ID, label: 'Resident observations'},
          data: {fields: SAMPLE_FIELDS, rows: appState.source.rows}
        },
        options: {centerMap: false},
        config: {
          version: 'v1',
          config: {
            visState: {
              layers: [
                {
                  id: 'resident-points',
                  type: 'point',
                  config: {
                    dataId: DATASET_ID,
                    label: 'GPU observations',
                    color: [86, 203, 249],
                    columns: {lat: 'latitude', lng: 'longitude'},
                    isVisible: true,
                    visConfig: {radius: 10, opacity: 0.8, filled: true}
                  }
                }
              ],
              interactionConfig: {
                tooltip: {
                  enabled: true,
                  fieldsToShow: {},
                  compareMode: false,
                  compareType: 'absolute'
                }
              },
              filters: [
                {
                  id: 'resident-value',
                  dataId: [DATASET_ID],
                  name: ['value'],
                  type: 'range',
                  value: [0, 100]
                },
                {
                  id: 'resident-category',
                  dataId: [DATASET_ID],
                  name: ['category'],
                  type: 'multiSelect',
                  value: [0, 1, 2, 3]
                },
                {
                  id: 'resident-time',
                  dataId: [DATASET_ID],
                  name: ['time'],
                  type: 'range',
                  value: [0, 1439]
                }
              ],
              layerBlending: 'normal',
              splitMaps: []
            },
            mapState: {latitude: 40.755, longitude: -73.985, zoom: 11, pitch: 0, bearing: 0},
            mapStyle: {styleType: 'compute-background'}
          }
        }
      })
    );
    residentEviction.attach(appState.runtime.session.registry);
    memoryManager.bindRegistry();
    appState.runtime.requestUpdate();
    if (pendingInitialization) pendingInitialization.resolve(appState.runtime);
    else appState.deviceRecovery.checkpoint(appState.runtime);
    notify();
    if (!pendingInitialization) void initializeStartupSample();
  } catch (error) {
    if (pendingInitialization) {
      pendingInitialization.reject(error);
      const failed = appState.runtime;
      if (failed?.device === device) void retireRecoveryRuntime(failed).catch(reportError);
    }
    reportError(error);
  }
}

export function recordSubmittedPerformanceFrame(owner: ComputeRuntime): void {
  const encoded = new Map<string, {sourceRevision: number; explorerRevision: number}>();
  for (const datasetId of [owner.explorer.dataset.id, ...owner.publications.keys()]) {
    const revision = owner.getEncodedSourceRevision(datasetId);
    if (revision) encoded.set(datasetId, {...revision});
  }
  // Deck invokes onAfterRender before its synchronous submit. This microtask observes that
  // submission boundary; it does not measure GPU completion or physical presentation.
  queueMicrotask(() => {
    if (appState.runtime !== owner || !owner.isReadyForRender) return;
    const timestamp = performance.now();
    recordSubmittedFrame();
    appState.submittedPerformanceFrames = {
      owner,
      datasets: new Map(
        [...encoded].map(([datasetId, revision]) => [
          datasetId,
          {...revision, timestamp, stage: 'submitted'}
        ])
      )
    };
  });
}
