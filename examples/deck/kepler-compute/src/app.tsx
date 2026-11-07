// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ComputeRuntime, type ExplorerSummary} from './compute-runtime';
import React, {useState, useEffect} from 'react';
import {ArrowStreamPanel} from './arrow-stream-panel';
import {STREAM_ORIGIN_METADATA_KEY} from './remote-stream-source';
import {appState} from './app-state';
import {
  removeDataset,
  wrapTo,
  addNotification,
  updateMap,
  setSelectedFeature,
  setPolygonFilterLayer,
  layerVisConfigChange,
  setSuitabilityConfig,
  setJoinConfig
} from '@kepler.gl/actions';
import {type SourceCandidate, SourceWorkspacePanel} from './source-workspace-panel';
import {DATASET_ID} from './sample-data';
import {bindLocalCloudHost} from './local-cloud-provider';
import {createLocalCloudHost} from './local-cloud-host';
import KeplerGlSchema from '@kepler.gl/schemas';
import {createNotification} from '@kepler.gl/utils';
import {NativeWorkflows, NATIVE_WORKFLOW_MESSAGES} from './native-workflows';
import {MemoryManagerPanel} from './memory-manager-panel';
import {NativeAutoPreparePanel} from './native-auto-prepare-panel';
import {ResidentDrawingPanel, type ResidentDrawingPanelProps} from './resident-drawing-panel';
import {
  toDrawnFeatures,
  toDrawnFeature,
  toNativeFeature,
  toSelectedNativeFeature
} from './main-native-feature';
import {extractSelectedRows} from './resident-drawing-actions';
import {RasterToolsPanel} from './raster-tools-panel';
import {BitmapToolsPanel} from './bitmap-tools-panel';
import {
  SuitabilityPolicyPanel,
  listSuitabilityPolicyRecords
} from './analysis4-suitability-policy-panel';
import {JoinPolicyPanel, listJoinPolicyRecords} from './analysis4-join-policy-panel';
import {TimeContractPanel} from './time-contract-panel';
import {makeTimeContractPanelProps} from './time-contract-host';
import {SpatialJoinCrsPanel, listSpatialJoinCrsRecords} from './analysis4-spatial-join-crs-panel';
import {MemoryBudgetPanel} from './memory-budget-panel';
import {categoryCoverageRegistry} from './binding3-filter-runtime';
import {RemoteRefetchSection} from './remote-refetch-host';
import {AnalysisPanel} from './analysis-panel';
import {AnalysisCatalogPanel} from './analysis-catalog-panel';
import {NativeSqlExplorer} from './native-sql-explorer';
import {makeNativeSqlGpuExecutor, getResidentSqlGpuRuntime} from './resident-sql-gpu-runtime';
import {SandboxPanel} from './sandbox-panel';
import {AssistantPanel} from './assistant-panel';
import {SandboxCapabilities} from './sandbox-capabilities-panel';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {
  attachPublishedDataset,
  isResidentDataset,
  publishSource,
  removePublishedDataset
} from './publication';
import {store, assistantHost} from './app-store';
import {
  residentDatasetIds,
  statusListeners,
  memoryManager,
  notify,
  layerDiagnostics,
  styleDiagnostics,
  plotDiagnostics,
  reportError,
  residentEviction,
  LOCAL_CLOUD_PROVIDERS
} from './app-services';
import {applyNativeConfig, reimportSource, getActiveRuntime} from './native-config';
import {restoreBundle, formatAutomaticRecoveryStatus, formatEvictionStatus} from './recovery';
import {nativeAutoPrepare} from './native-hosts';
import {rasterLiveHost} from './main-raster-live';
import {KeplerGl, mapStyles} from './map-container';

function ResidentStreamSection({owner}: {owner: ComputeRuntime}): React.JSX.Element {
  return (
    <ArrowStreamPanel
      session={owner.session}
      remoteSources={Array.from(owner.publications.values(), published => ({
        label: published.label,
        url: (
          published.dataset.source?.metadata?.[STREAM_ORIGIN_METADATA_KEY] as
            | {url: string}
            | undefined
        )?.url
      })).filter((source): source is {label: string; url: string} => Boolean(source.url))}
      getCompletionFence={() => {
        if (owner.device.isLost || owner.session.deviceLost) return Promise.resolve();
        const fence = owner.device.createFence();
        return Promise.race([fence.signaled, owner.device.lost]).finally(() => fence.destroy());
      }}
      onPublishCompleted={async (dataset, label, coordinates) => {
        if (
          appState.runtime !== owner ||
          owner.session.registry.getDataset(dataset.id) !== dataset ||
          dataset.population?.kind !== 'stream' ||
          dataset.population.state !== 'complete'
        )
          throw new Error('Only the current completed stream can be published.');
        try {
          await attachPublishedDataset(owner, dataset, label, coordinates);
        } catch (error) {
          if (appState.runtime === owner) store.dispatch(removeDataset(dataset.id));
          residentDatasetIds.delete(dataset.id);
          await owner.removePublication(dataset.id);
          throw error;
        }
      }}
    />
  );
}

export function App(): React.JSX.Element {
  const [size, setSize] = useState({width: innerWidth, height: innerHeight});
  const [summary, setSummary] = useState<ExplorerSummary | null>(null);
  const [stateRevision, setStateRevision] = useState(0);
  const [sandboxRuntime, setSandboxRuntime] = useState<ComputeRuntime | null>(appState.runtime);
  const [sourceCandidate, setSourceCandidate] = useState<SourceCandidate | null>(null);
  const [selectedAnalysisDatasetId, setSelectedAnalysisDatasetId] = useState(DATASET_ID);
  useEffect(() => {
    if (sourceCandidate && !sourceCandidate.isCurrent()) setSourceCandidate(null);
  }, [sourceCandidate, stateRevision]);
  // Kepler's native Save/Load map dialogs use the local cloud provider through the portable bundle path.
  useEffect(() => {
    if (!sandboxRuntime) return undefined;
    bindLocalCloudHost(
      createLocalCloudHost(
        {
          runtime: sandboxRuntime,
          getNativeConfig: () =>
            KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute']),
          applyNativeConfig,
          reimportSource,
          restoreBundle
        },
        message =>
          void store.dispatch(
            wrapTo(
              'compute',
              addNotification(createNotification({message, type: 'warning', topic: 'global'}))
            )
          )
      )
    );
    return () => bindLocalCloudHost(null);
  }, [sandboxRuntime]);

  useEffect(() => {
    const onResize = () => setSize({width: innerWidth, height: innerHeight});
    const onStatus = () => {
      if (appState.runtime) setSandboxRuntime(appState.runtime);
      if (
        !appState.runtime ||
        appState.runtime.session.deviceLost ||
        appState.deviceRecovery.status.phase !== 'ready'
      ) {
        setSummary(null);
        setSourceCandidate(null);
      }
      setStateRevision(value => value + 1);
    };
    window.addEventListener('resize', onResize);
    statusListeners.add(onStatus);
    return () => {
      window.removeEventListener('resize', onResize);
      statusListeners.delete(onStatus);
    };
  }, []);
  useEffect(
    () =>
      appState.runtime?.subscribe(value => {
        appState.latestSummary = value;
        setSummary(value);
      }),
    [appState.runtime]
  );
  useEffect(() => store.subscribe(() => setStateRevision(value => value + 1)), []);

  const recovery = appState.deviceRecovery.status;
  return (
    <NativeWorkflows.Provider
      value={{
        unavailable: recovery.phase !== 'ready',
        memory: <MemoryManagerPanel source={memoryManager.source} variant="compact" />,
        memoryManager: <MemoryManagerPanel source={memoryManager.source} variant="full" />,
        status: (
          <div data-testid="native-compute-status" data-revision={stateRevision}>
            <NativeAutoPreparePanel controller={nativeAutoPrepare} />
            <ResidentDrawingPanel
              features={toDrawnFeatures(
                store.getState().keplerGl['compute']?.visState?.editor?.features
              )}
              selectedFeature={toDrawnFeature(
                store.getState().keplerGl['compute']?.visState?.editor?.selectedFeature
              )}
              filters={store.getState().keplerGl['compute']?.visState?.filters ?? []}
              layers={(store.getState().keplerGl['compute']?.visState?.layers ?? []).map(layer => ({
                id: layer.id,
                type: layer.type ?? '',
                dataId: layer.config.dataId ?? '',
                label: layer.config.label
              }))}
              zoom={store.getState().keplerGl['compute']?.mapState?.zoom ?? 0}
              annotations={
                (store.getState().keplerGl['compute']?.visState?.annotations ??
                  []) as unknown as NonNullable<ResidentDrawingPanelProps['annotations']>
              }
              viewportSize={{
                width: store.getState().keplerGl['compute']?.mapState?.width ?? 800,
                height: store.getState().keplerGl['compute']?.mapState?.height ?? 600
              }}
              onFocusReplayStep={step => {
                store.dispatch(
                  wrapTo(
                    'compute',
                    updateMap(
                      {
                        longitude: step.longitude,
                        latitude: step.latitude,
                        zoom: step.zoom
                      },
                      0
                    )
                  )
                );
                const feature = store
                  .getState()
                  .keplerGl['compute']?.visState?.editor?.features.find(
                    candidate => String(candidate.id) === step.id
                  );
                if (step.kind === 'feature' && feature)
                  store.dispatch(wrapTo('compute', setSelectedFeature(toNativeFeature(feature))));
              }}
              readPointerCoordinate={() => {
                const coordinate =
                  store.getState().keplerGl['compute']?.visState?.mousePos?.coordinate;
                return coordinate ? [coordinate[0], coordinate[1]] : null;
              }}
              onSelectFeature={feature =>
                store.dispatch(
                  wrapTo('compute', setSelectedFeature(toSelectedNativeFeature(feature)))
                )
              }
              onCreateLayer={({table, label, geometryColumn}) => {
                const activeRuntime = appState.runtime;
                setSourceCandidate({
                  table: table as SourceCandidate['table'],
                  label,
                  isCurrent: () => appState.runtime === activeRuntime,
                  suggestedGeometryColumn: geometryColumn
                });
              }}
              onUseAsFilter={(feature, layerIds) => {
                // The native updater turns the sketch into one polygon filter and toggles each layer.
                const layers = store.getState().keplerGl['compute']?.visState?.layers ?? [];
                for (const layerId of layerIds) {
                  const layer = layers.find(candidate => candidate.id === layerId);
                  const current =
                    store
                      .getState()
                      .keplerGl['compute']?.visState?.editor?.features.find(
                        candidate => candidate.id === feature.id
                      ) ?? feature;
                  if (layer)
                    store.dispatch(
                      wrapTo('compute', setPolygonFilterLayer(layer, toNativeFeature(current)))
                    );
                }
              }}
              onExtractRows={async datasetId => {
                const activeRuntime = appState.runtime;
                if (!activeRuntime) throw new Error('No active workspace.');
                const explorer =
                  datasetId === DATASET_ID
                    ? activeRuntime.explorer
                    : activeRuntime.publications.get(datasetId)?.consumer.explorer;
                if (!explorer) throw new Error('The dataset is not resident.');
                const state = await extractSelectedRows(activeRuntime.session, explorer, {
                  maximumRows: 1_000_000,
                  maximumOutputBytes: 256 * 1024 * 1024
                });
                if (state.status === 'ready')
                  setSourceCandidate({
                    table: state.value.table,
                    label: `Extracted rows (${state.value.selectedCount})`,
                    isCurrent: () => appState.runtime === activeRuntime && explorer.isCurrent
                  });
                return state;
              }}
            />
            <RasterToolsPanel host={rasterLiveHost} {...rasterLiveHost.getPanelModel()} />
            <BitmapToolsPanel
              bitmapLayers={rasterLiveHost.getPanelModel().bitmapLayers}
              getGeoreference={layerId =>
                rasterLiveHost.bitmap.getGeoreferences().find(entry => entry.layerId === layerId)
              }
              attachGeoTiff={(layerId, bytes) =>
                rasterLiveHost.bitmap.attachGeoTiff(layerId, bytes)
              }
              applyVisConfig={(layerId, patch) => {
                const layer = store
                  .getState()
                  .keplerGl['compute']?.visState?.layers.find(
                    candidate => candidate.id === layerId
                  );
                if (layer) store.dispatch(wrapTo('compute', layerVisConfigChange(layer, patch)));
              }}
            />
            <SuitabilityPolicyPanel
              suitabilities={listSuitabilityPolicyRecords(
                store.getState().keplerGl['compute']?.visState?.suitabilities
              )}
              setPolicy={(id, missingValues) =>
                store.dispatch(
                  wrapTo('compute', setSuitabilityConfig(id, {missingValues} as never))
                )
              }
              setCategoryScores={(id, categoryScores) =>
                store.dispatch(
                  wrapTo('compute', setSuitabilityConfig(id, {categoryScores} as never))
                )
              }
            />
            <JoinPolicyPanel
              joins={listJoinPolicyRecords(store.getState().keplerGl['compute']?.visState?.joins)}
              setPolicy={(id, change) =>
                store.dispatch(wrapTo('compute', setJoinConfig(id, change as never)))
              }
            />
            <TimeContractPanel {...makeTimeContractPanelProps(store)} />
            <SpatialJoinCrsPanel
              joins={listSpatialJoinCrsRecords(
                store.getState().keplerGl['compute']?.visState?.joins
              )}
              setCrs={(id, change) =>
                store.dispatch(wrapTo('compute', setJoinConfig(id, change as never)))
              }
            />
            <MemoryBudgetPanel runtime={appState.runtime} />
            {appState.workflowNotice ? (
              <div role="alert" data-testid="resident-workflow-notice">
                <p>{appState.workflowNotice}</p>
                <button
                  type="button"
                  onClick={() => {
                    appState.workflowNotice = '';
                    notify();
                  }}
                >
                  Dismiss
                </button>
              </div>
            ) : null}
            {appState.currentError || layerDiagnostics.size || styleDiagnostics.size ? (
              <p role="alert" className="error">
                {appState.currentError ||
                  [...layerDiagnostics.values(), ...styleDiagnostics.values()].join('\n')}
              </p>
            ) : null}
            {categoryCoverageRegistry.list().length ? (
              <p role="status" className="hint" data-testid="category-coverage-warning">
                {categoryCoverageRegistry
                  .list()
                  .map(warning => warning.message)
                  .join('\n')}
              </p>
            ) : null}
            {plotDiagnostics.size ? (
              <p role="status" className="hint" data-testid="plot-diagnostics">
                {[...plotDiagnostics.values()].join('\n')}
              </p>
            ) : null}
            {recovery.phase !== 'ready' || recovery.reason ? (
              <section aria-label="GPU device recovery">
                <h2>GPU device recovery</h2>
                <p>
                  {recovery.phase === 'rebuilding'
                    ? 'Creating a replacement device and restoring the workspace…'
                    : recovery.reason}
                </p>
                <p>
                  {recovery.hasCheckpoint
                    ? `${recovery.sourceCount} source snapshots remain in memory for this page. Recovery restores their configuration and analysis recipes; source rows are not saved to storage or uploaded.`
                    : 'No complete healthy checkpoint is available. Recovery cannot reconstruct this workspace.'}
                </p>
                {appState.automaticRecovery?.status.enabled ? (
                  <p role="status" data-testid="automatic-recovery-status">
                    {formatAutomaticRecoveryStatus(appState.automaticRecovery.status)}
                  </p>
                ) : null}
                <button
                  disabled={
                    !recovery.hasCheckpoint ||
                    (recovery.phase !== 'lost' && recovery.phase !== 'failed')
                  }
                  onClick={() => void appState.deviceRecovery.recover().catch(reportError)}
                >
                  Recover GPU workspace
                </button>
                {recovery.phase === 'rebuilding' ? (
                  <button
                    onClick={() => void appState.deviceRecovery.cancelRecovery().catch(reportError)}
                  >
                    Cancel recovery
                  </button>
                ) : null}
              </section>
            ) : null}
            {appState.runtime && !appState.runtime.session.deviceLost ? (
              <RemoteRefetchSection
                getSources={() =>
                  [...getActiveRuntime().publications.values()].map(published => ({
                    datasetId: published.dataset.id,
                    label: published.label,
                    stream: {
                      origin: published.dataset.source?.metadata?.[STREAM_ORIGIN_METADATA_KEY] as
                        | {kind: 'remote'; url: string}
                        | undefined
                    }
                  }))
                }
                getExistingIds={() =>
                  new Set(
                    Object.keys(store.getState().keplerGl['compute']?.visState?.datasets ?? {})
                  )
                }
                dependencies={{
                  getSession: () => getActiveRuntime().session,
                  getCompletionFence: () => {
                    const owner = getActiveRuntime();
                    const fence = owner.device.createFence();
                    return Promise.race([fence.signaled, owner.device.lost]).finally(() =>
                      fence.destroy()
                    );
                  },
                  attach: async (dataset, label) => {
                    const owner = getActiveRuntime();
                    await attachPublishedDataset(owner, dataset, label);
                  }
                }}
              />
            ) : null}
            {residentEviction.isEnabled ? (
              <section aria-label="GPU memory eviction">
                <p role="status" data-testid="automatic-eviction-status">
                  {formatEvictionStatus(residentEviction.status)}
                </p>
                {residentEviction.status.failedIds.length ? (
                  <button type="button" onClick={() => residentEviction.retryFailed()}>
                    Retry table rebuild
                  </button>
                ) : null}
              </section>
            ) : null}
            <p className="count" data-testid="selected-count">
              {recovery.phase !== 'ready'
                ? 'Device unavailable'
                : summary
                  ? summary.selectedCount.toLocaleString()
                  : 'Preparing…'}
              <small> selected observations</small>
            </p>
          </div>
        ),
        analysis: (
          <>
            {appState.runtime && !appState.runtime.session.deviceLost ? (
              <AnalysisPanel
                analysis={appState.runtime.analysis}
                selectedDatasetId={selectedAnalysisDatasetId}
                onSelectDataset={setSelectedAnalysisDatasetId}
                sources={[
                  {
                    id: DATASET_ID,
                    label: 'Resident observations',
                    getAnalysis: () => getActiveRuntime().analysis
                  },
                  ...[...appState.runtime.publications.values()].map(published => ({
                    id: published.dataset.id,
                    label: published.label,
                    polygonAnalysisUnavailableReason: published.polygonAnalysisUnavailableReason,
                    getAnalysis: () => published.getAnalysis()
                  }))
                ]}
              />
            ) : null}
            {appState.runtime && !appState.runtime.session.deviceLost ? (
              <AnalysisCatalogPanel
                device={appState.runtime.session.device}
                datasets={Object.values(
                  store.getState().keplerGl['compute']?.visState?.datasets ?? {}
                ).filter(dataset => !isResidentDataset(dataset.id))}
                onPublish={(table, label, geometryColumn) => {
                  const activeRuntime = appState.runtime;
                  setSourceCandidate({
                    table,
                    label,
                    isCurrent: () => appState.runtime === activeRuntime,
                    suggestedGeometryColumn: geometryColumn
                  });
                }}
              />
            ) : null}
          </>
        ),
        sql: (
          <>
            {appState.runtime ? (
              <NativeSqlExplorer
                runtime={appState.runtime.duckdb}
                gpu={makeNativeSqlGpuExecutor(
                  getResidentSqlGpuRuntime(appState.runtime.session),
                  execution =>
                    attachPublishedDataset(
                      appState.runtime as NonNullable<typeof appState.runtime>,
                      execution.dataset,
                      'SQL result (GPU)'
                    )
                )}
                onPrepareResult={result => {
                  const activeRuntime = appState.runtime;
                  setSourceCandidate({
                    table: result.table,
                    label: 'SQL result',
                    isCurrent: () =>
                      appState.runtime === activeRuntime &&
                      Boolean(activeRuntime?.duckdb.isResultCurrent(result))
                  });
                }}
              />
            ) : null}
          </>
        ),
        sources: (
          <>
            {appState.runtime ? (
              <SourceWorkspacePanel
                candidate={sourceCandidate}
                datasets={Object.values(
                  store.getState().keplerGl['compute']?.visState?.datasets ?? {}
                ).filter(dataset => !isResidentDataset(dataset.id))}
                resolveNative={datasetId =>
                  store.getState().keplerGl['compute']?.visState?.datasets[datasetId]
                }
                publications={[...appState.runtime.publications.values()].map(published => ({
                  id: published.dataset.id,
                  label: published.label,
                  rowCount: published.dataset.rowCount,
                  selectedCount: appState.runtime?.session.deviceLost
                    ? undefined
                    : published.consumer.summary?.selectedCount,
                  error: appState.runtime?.session.deviceLost
                    ? 'GPU device unavailable; source retained for explicit recovery.'
                    : published.error
                }))}
                onPublish={publishSource}
                onRemove={removePublishedDataset}
              />
            ) : null}
            {appState.runtime ? <ResidentStreamSection owner={appState.runtime} /> : null}
          </>
        ),
        exportData: (
          <>
            {sandboxRuntime ? (
              <SandboxPanel
                mode="export"
                runtime={sandboxRuntime}
                getNativeConfig={() =>
                  KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute'])
                }
                applyNativeConfig={applyNativeConfig}
                reimportSource={reimportSource}
                restoreBundle={restoreBundle}
              />
            ) : null}
          </>
        ),
        saveWorkspace: (
          <>
            {sandboxRuntime ? (
              <SandboxPanel
                mode="workspace"
                runtime={sandboxRuntime}
                getNativeConfig={() =>
                  KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute'])
                }
                applyNativeConfig={applyNativeConfig}
                reimportSource={reimportSource}
                restoreBundle={restoreBundle}
              />
            ) : null}
          </>
        ),
        help: (
          <>
            <AssistantPanel host={assistantHost} />
            <SandboxCapabilities />
          </>
        )
      }}
    >
      <KeplerGl
        id="compute"
        width={size.width}
        height={size.height}
        mapboxApiAccessToken=""
        appName="Kepler"
        cloudProviders={LOCAL_CLOUD_PROVIDERS}
        localeMessages={NATIVE_WORKFLOW_MESSAGES}
        mapStyles={mapStyles}
        mapStylesReplaceDefault
        deckGlProps={{
          deviceProps: {
            type: 'webgpu',
            adapters: [webgpuAdapter],
            // ?maxLimits=1 requests the adapter maximum storage binding and buffer sizes, which
            // raises the neighbor-slot and large-column capacity checks (default limits are 128 MiB).
            ...(new URLSearchParams(window.location.search).get('maxLimits') === '1'
              ? {featureLevel: 'max' as const}
              : {})
          }
        }}
        RTLTextPlugin={false}
      />
    </NativeWorkflows.Provider>
  );
}
