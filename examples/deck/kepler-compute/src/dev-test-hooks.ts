// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {installAcceptanceInstrumentation} from './acceptance-instrumentation';
import {appState} from './app-state';
import {VectorTileResidentRenderer} from './vector-tile-resident-layers';
import {VectorTileLiveController} from './vector-tile-live-controller';
import {BASEMAP_ATTRIBUTION_ELEMENT_ID} from './basemap-attribution';
import {getTile3DDiagnostics, pickTile3D, getDeviceResourceCounts} from './tile3d-diagnostics';
import {getDeckInternals, getDeckLayers} from './deck-internals';
import {flattenDeckLayers} from './main-globe-layers';
import {
  wrapTo,
  addEffect,
  updateEffect,
  removeEffect,
  addDataToMap,
  updateMap,
  setFeatures,
  addChart,
  layerConfigChange,
  layerTextLabelChange,
  layerVisConfigChange,
  layerVisualChannelConfigChange,
  setLayerAnimationTime,
  createOrUpdateFilter,
  setFilter,
  setFilterPlot,
  toggleMapControl,
  setPolygonFilterAllLayers,
  removeFilter,
  toggleLayerForMap,
  toggleSplitMap,
  toggleSplitMapViewport,
  onLayerHover,
  deleteFeature,
  duplicateLayer,
  removeLayer,
  reorderLayer
} from '@kepler.gl/actions';
import {getFlowmapStats} from './flowmap-gpu-layer';
import {categoryCoverageRegistry} from './binding3-filter-runtime';
import {fieldPreparationCoordinator} from './binding3-field-preparation';
import {getPointLabelResourceEntries, getRenderedLayerInfo} from './main-layer-info';
import {DATASET_ID} from './sample-data';
import KeplerGlSchema from '@kepler.gl/schemas';
import {toNativeFeature} from './main-native-feature';
import {peekGeometryColumnVariantRuntime} from './geometry-column-variants';
import {
  getActiveRuntime,
  createResidentFilterAction,
  setNativeCategories,
  applyNativeConfig,
  reimportSource
} from './native-config';
import {rasterLiveHost} from './main-raster-live';
import {assistantHost, store} from './app-store';
import {
  vectorTileLive,
  vectorTileResident,
  basemapContext,
  basemapPicker,
  webGPUEffectsPipeline,
  residentEviction,
  layerDiagnostics,
  styleDiagnostics,
  plotDiagnostics,
  clusterRuntimes
} from './app-services';
import {flowFieldBindings, flowH3Bindings} from './layers';
import {getEvictionRebuildSource} from './recovery';
import {setTimestampRange} from './device-init';
import {readNativeTooltip, pickPublishedPosition} from './selection';

if (import.meta.env.DEV) {
  installAcceptanceInstrumentation(() => {
    const owner = getActiveRuntime();
    return {
      device: owner.device,
      memoryUsage: owner.session.registry.memoryUsage,
      uploadBytes: owner.uploadBytes,
      readbackBytes: owner.explorer.readbackByteLength
    };
  });
  Object.assign(window, {
    __keplerComputeTest: {
      ...rasterLiveHost.testHooks,
      reserveTaggedMemory: (
        byteLength: number,
        category: 'source' | 'output' | 'scratch' | 'readback'
      ) => getActiveRuntime().session.registry.reserveMemory(byteLength, 'cpu', category),
      injectRestoreFailure: (index: number) => {
        appState.injectedRestoreFailureIndex = index;
      },
      runAssistantOperation: (operation: unknown) => assistantHost.runOperation(true, operation),
      getAssistantContext: () => assistantHost.getContext(),
      /** Kepler field types of a dataset (for example `timestamp` after CAST ... AS TIMESTAMP). */
      getDatasetFieldTypes: (datasetId: string) =>
        Object.fromEntries(
          (
            (
              store.getState().keplerGl['compute']?.visState?.datasets?.[datasetId] as
                | {fields?: {name: string; type: string}[]}
                | undefined
            )?.fields ?? []
          ).map(field => [field.name, field.type])
        ),
      getVectorTileState: () => vectorTileLive.getState(),
      getVectorTileResidentState: () => vectorTileResident.getState(),
      getVectorTileDevice: () =>
        appState.runtime && !appState.runtime.session.deviceLost ? appState.runtime.device : null,
      readVectorTileFilterMasks: (layerId: string) => vectorTileResident.readFilterMasks(layerId),
      readVectorTileColors: (layerId: string) => vectorTileResident.readColors(layerId),
      setVectorTileStyleFromNativeLayer: (
        layerId: string,
        layer: Parameters<VectorTileResidentRenderer['setStyleFromNativeLayer']>[1]
      ) => vectorTileResident.setStyleFromNativeLayer(layerId, layer),
      setVectorTileFilters: (
        layerId: string,
        filters: Parameters<VectorTileLiveController['setFilters']>[1]
      ) => vectorTileLive.setFilters(layerId, filters),
      decideVectorTileAnalysis: (
        layerId: string,
        request: Parameters<VectorTileLiveController['decideAnalysis']>[1]
      ) => vectorTileLive.decideAnalysis(layerId, request),
      setBasemapStyleUrl: (url: string | null) => basemapContext.setStyleUrl(url),
      getBasemapState: () => basemapContext.getState(),
      addBasemapStyle: (url: string, label?: string) =>
        basemapPicker.addStyle(basemapContext, url, label).then(result => ({
          ...result,
          activeStyleType: store.getState().keplerGl['compute']?.mapStyle?.styleType,
          pickerStyleIds: Object.keys(
            store.getState().keplerGl['compute']?.mapStyle?.mapStyles ?? {}
          ),
          attributionText:
            document.getElementById(BASEMAP_ATTRIBUTION_ELEMENT_ID)?.textContent ?? null
        })),
      getBasemapAttributionText: () =>
        document.getElementById(BASEMAP_ATTRIBUTION_ELEMENT_ID)?.textContent ?? null,
      getEffectReplacements: () => webGPUEffectsPipeline.replacer.getReplacements(),
      getTile3DDiagnostics: () => getTile3DDiagnostics(appState.runtime?.deck as never),
      /** Dev probe: Deck's effect list and the shadow/lighting module props of each layer model. */
      getDeckEffectProbe: () => {
        const deck = getDeckInternals(appState.runtime?.deck);
        return {
          effects: (deck?.effectManager?.effects ?? []).map(item => item.constructor.name),
          layers: (deck?.layerManager?.getLayers() ?? []).map(layer => ({
            id: layer.id,
            models: (layer.getModels?.() ?? []).map(model => {
              const inputs = (
                model as {
                  shaderInputs?: {
                    modules?: Record<string, unknown>;
                    moduleUniforms?: Record<string, Record<string, unknown>>;
                  };
                }
              ).shaderInputs;
              const summarize = (name: string) =>
                JSON.stringify(inputs?.moduleUniforms?.[name] ?? null, (key, value) =>
                  key !== '' && typeof value === 'object' && value && !Array.isArray(value)
                    ? '[object]'
                    : value
                );
              return {
                modules: Object.keys(inputs?.modules ?? {}),
                shadow: summarize('shadow').slice(0, 600),
                lighting: summarize('lighting').slice(0, 800)
              };
            })
          }))
        };
      },
      getDeckLayerIds: () => getDeckLayers(appState.runtime?.deck).map(layer => layer.id),
      /** Ids of the layers last handed to Deck as props (before Deck instantiates them). */
      getDeckPropLayerIds: () =>
        flattenDeckLayers(getDeckInternals(appState.runtime?.deck)?.props?.layers).map(
          layer => layer.id
        ),
      pickTile3D: (position?: {x: number; y: number; radius?: number}) =>
        pickTile3D(appState.runtime?.deck as never, position),
      getDeviceResourceCounts: () => getDeviceResourceCounts(appState.runtime?.device as never),
      addNativeEffect: (type: string) =>
        store.dispatch(wrapTo('compute', addEffect({type} as never))),
      updateNativeEffect: (id: string, parameters: Record<string, unknown>) =>
        store.dispatch(wrapTo('compute', updateEffect(id, {parameters} as never))),
      removeNativeEffect: (id: string) => store.dispatch(wrapTo('compute', removeEffect(id))),
      getNativeEffects: () =>
        store.getState().keplerGl['compute']?.visState?.effects?.map(item => ({
          id: item.id,
          type: item.type,
          isEnabled: item.isEnabled
        })),
      addNativeTile3DDataset: (datasetId: string, url: string, label = '3D tiles') =>
        store.dispatch(
          addDataToMap({
            datasets: {
              info: {id: datasetId, label, type: 'tile-3d', format: 'rows'},
              data: {fields: [], rows: []},
              metadata: {tile3dUrl: url},
              disableDataOperation: true
            } as never,
            options: {centerMap: false, autoCreateLayers: true, keepExistingConfig: true}
          })
        ),
      setNativeMapView: (view: {
        longitude: number;
        latitude: number;
        zoom: number;
        pitch?: number;
      }) =>
        store.dispatch(wrapTo('compute', updateMap({pitch: 0, bearing: 0, ...view} as never, 0))),
      getFlowFieldStats: () => flowFieldBindings.getStats(),
      readFlowFieldGridForDebug: async (layerId: string) => {
        const texels = await flowFieldBindings.readGridForDebug(layerId);
        return texels ? Array.from(texels) : null;
      },
      setFlowFieldGridMode: (gridMode: 'auto' | 'cpu') => {
        flowFieldBindings.gridMode = gridMode;
      },
      getFlowH3States: () => flowH3Bindings.getStates(),
      getFlowmapStats: () => getFlowmapStats(),
      /** Native layer metadata (bounds, flow location count) for browser verification. */
      getNativeLayerMeta: (layerId: string) => {
        const meta = store
          .getState()
          .keplerGl['compute']?.visState?.layers.find(layer => layer.id === layerId)?.meta as
          | {bounds?: number[] | null; locations?: unknown[]}
          | undefined;
        return meta
          ? {bounds: meta.bounds ?? null, locationCount: meta.locations?.length ?? 0}
          : null;
      },
      /** Kepler dataset metadata and computed layer data of the first layer of a type (tile3d diagnostics). */
      getNativeLayerDataSummary: (layerType: string) => {
        const visState = store.getState().keplerGl['compute']?.visState;
        const layerIndex = visState?.layers.findIndex(layer => layer.type === layerType) ?? -1;
        const layer = visState?.layers[layerIndex];
        const dataId = layer?.config.dataId;
        return {
          layerIndex,
          dataId,
          datasetType: dataId ? visState?.datasets[dataId]?.type : null,
          datasetMetadata: dataId ? (visState?.datasets[dataId]?.metadata ?? null) : null,
          layerData: visState?.layerData[layerIndex] ?? null,
          shouldRender: layer ? layer.shouldRenderLayer(visState?.layerData[layerIndex]) : null,
          layerOrder: visState?.layerOrder ?? [],
          layerId: layer?.id ?? null,
          overlayType: layer?.overlayType ?? null,
          renderProbe: (() => {
            try {
              const rendered = layer?.renderLayer({
                data: visState?.layerData[layerIndex],
                idx: layerIndex,
                interactionConfig: visState?.interactionConfig,
                mapState: {},
                visible: true,
                dataset: dataId ? visState?.datasets[dataId] : undefined
              } as never) as {id: string}[] | undefined;
              return rendered?.map(item => item.id) ?? null;
            } catch (error) {
              return String(error);
            }
          })()
        };
      },
      setFlowFieldUnitPolicy: (policy: 'visual' | 'physical') => {
        flowFieldBindings.unitPolicy = policy;
      },
      runFlowGpuCheck: async (columns: import('./flow-gpu-check').FlowGpuCheckColumns) =>
        (await import('./flow-gpu-check')).runFlowGpuCheck(getActiveRuntime().device, columns),
      setPerformanceDiagnosticsEnabled: (enabled: boolean) =>
        getActiveRuntime().setPerformanceDiagnosticsEnabled(enabled),
      getPerformanceSnapshot: (datasetId: string) => {
        const owner = getActiveRuntime();
        const published = owner.publications.get(datasetId);
        const explorer =
          datasetId === owner.explorer.dataset.id ? owner.explorer : published?.consumer.explorer;
        if (!explorer?.isCurrent) throw new Error('Performance source is stale or unavailable.');
        const summary = published ? published.consumer.summary : owner.latestSummary;
        const frame =
          appState.submittedPerformanceFrames?.owner === owner
            ? appState.submittedPerformanceFrames.datasets.get(datasetId)
            : undefined;
        return {
          now: performance.now(),
          sourceRevision: explorer.dataset.revision,
          explorerRevision: explorer.currentRevision,
          summary:
            summary?.revision === explorer.currentRevision
              ? {revision: summary.revision, selectedCount: summary.selectedCount}
              : null,
          frame: frame?.sourceRevision === explorer.dataset.revision ? frame : null,
          ...owner.getPerformanceDiagnostics(datasetId),
          readbackBytes: {explorer: explorer.readbackByteLength},
          memoryUsage: owner.session.registry.memoryUsage,
          deviceInfo: owner.device.info
        };
      },
      getState: () => ({
        ready: Boolean(appState.runtime) && appState.deviceRecovery.status.phase === 'ready',
        recovery: appState.deviceRecovery.status,
        automaticRecovery: appState.automaticRecovery?.status,
        automaticEviction: residentEviction.status,
        deviceGeneration: appState.deviceGeneration,
        deviceId: appState.runtime?.device.id,
        error:
          appState.currentError ||
          [...layerDiagnostics.values(), ...styleDiagnostics.values()].join('\n'),
        selectedCount: appState.latestSummary?.selectedCount,
        revision: appState.latestSummary?.revision,
        frameCount: appState.runtime?.frameCount,
        uploadBytes: appState.runtime?.uploadBytes,
        nativeFilters: store.getState().keplerGl['compute']?.visState?.filters,
        nativeNotifications: store.getState().keplerGl['compute']?.uiState?.notifications,
        animation: store.getState().keplerGl['compute']?.visState?.animationConfig,
        nativeLayers: store.getState().keplerGl['compute']?.visState?.layers.map(layer => ({
          id: layer.id,
          type: layer.type,
          dataId: layer.config.dataId,
          visible: layer.config.isVisible,
          valid: layer.isValid,
          error: layer.errorMessage
        })),
        publications: [...(appState.runtime?.publications.values() ?? [])].map(published => ({
          id: published.dataset.id,
          sourceRevision: published.dataset.revision,
          sourceUploadBytes: published.dataset.value.uploadedBufferBytes,
          sourceBufferIds: [
            ...Object.values(published.dataset.value.columns).flatMap(column =>
              column.data.map(chunk => chunk.buffer.id)
            ),
            ...Object.values(published.dataset.value.timeColumns).flatMap(column =>
              [...column.timestamps.data, ...column.validity.data].map(chunk => chunk.buffer.id)
            )
          ],
          globeGeometry: published.globeGeometryStatus ?? null,
          rowCount: published.dataset.rowCount,
          sourceBatches: published.dataset.value.prepared.sourceBatches.map(
            batch => batch.sourceRowCount
          ),
          selectedCount: published.consumer.summary?.selectedCount,
          scopeRevision: published.scopes.currentRevision,
          scopePending: published.scopes.isLayerRegistrationPending,
          aggregation: published.aggregation
            ? {
                projectedBufferIds: published.aggregation.positions.data.map(
                  chunk => chunk.buffer.id
                ),
                projectedChunks: published.aggregation.positions.data.map(chunk => chunk.length),
                uploadBytes: published.aggregation.uploadBytes,
                readbackBytes: published.aggregation.readbackByteLength,
                contract: 'mercator-centroid-meters-float32'
              }
            : undefined,
          error: published.error,
          geometry: published.geometry
            ? {
                sourceUploadBytes: published.geometry.geometry.gpuByteLength,
                sourceBufferIds: published.geometry.geometry.buffers.map(buffer => buffer.id),
                sourceBatches: published.dataset.batches.map(batch => batch.rowCount)
              }
            : undefined,
          h3: published.h3
            ? {
                validRowCount: published.h3.source.validRowCount,
                invalidRows: published.h3.source.invalidRows,
                precision: published.h3.source.precision,
                sourceUploadBytes: published.h3.gpuByteLength,
                sourceBufferIds: [
                  ...[
                    ...published.h3.centers.data,
                    ...published.h3.validity.data,
                    ...published.h3.resolutions.data,
                    ...published.h3.pentagons.data,
                    ...published.h3.primitiveSourceRows.data,
                    ...published.h3.outlineVertexRows.data
                  ].map(chunk => chunk.buffer.id),
                  ...published.h3.geometry.buffers.map(buffer => buffer.id)
                ]
              }
            : undefined,
          trips: published.tripGeometry
            ? {
                tripCount: published.tripGeometry.source.tripCount,
                domain: published.tripGeometry.source.animationDomainMilliseconds,
                origin: published.tripGeometry.source.originMilliseconds,
                actualTimeError: published.tripGeometry.source.actualTimeErrorMilliseconds,
                sourceUploadBytes: published.tripGeometry.gpuByteLength,
                sourceBufferIds: published.tripGeometry.getPath.data.map(chunk => chunk.buffer.id)
              }
            : undefined
        })),
        plotDiagnostics: Object.fromEntries(plotDiagnostics),
        categoryCoverage: categoryCoverageRegistry.list(),
        fieldPreparation: fieldPreparationCoordinator.snapshot(),
        unsupportedSelection: appState.unsupportedSelection
      }),
      resourceIdentity: () => ({
        device: appState.runtime?.device,
        labelBuffers: Object.fromEntries(
          getPointLabelResourceEntries(appState.runtime?.deck.props.layers)
        ),
        aggregationBuffers: Object.fromEntries(
          [...(appState.runtime?.publications.values() ?? [])].map(published => [
            published.dataset.id,
            published.aggregation?.positions.data.map(chunk => chunk.buffer) ?? []
          ])
        ),
        geometryBuffers: Object.fromEntries(
          [...(appState.runtime?.publications.values() ?? [])].map(published => [
            published.dataset.id,
            published.geometry?.geometry.buffers ?? []
          ])
        ),
        sourceBuffers: Object.fromEntries(
          [
            appState.runtime?.explorer.dataset,
            ...[...(appState.runtime?.publications.values() ?? [])].map(
              published => published.dataset
            )
          ].flatMap(dataset =>
            dataset
              ? [
                  [
                    dataset.id,
                    Object.values(dataset.value.columns).flatMap(column =>
                      column.data.map(chunk => chunk.buffer)
                    )
                  ]
                ]
              : []
          )
        )
      }),
      loseDevice: () => {
        const activeRuntime = getActiveRuntime();
        if (!appState.deviceRecovery.checkpoint(activeRuntime))
          throw new Error('A healthy recovery checkpoint is required');
        activeRuntime.device.destroy();
      },
      /** Test-only: show or hide a native layer, which controls eviction eligibility. */
      setDrawnFeatures: (features: unknown[]) =>
        store.dispatch(wrapTo('compute', setFeatures(features as never))),
      addNativeChart: (chart: unknown) =>
        store.dispatch(wrapTo('compute', addChart(chart as never))),
      setLayerVisible: (layerId: string, isVisible: boolean) => {
        const layer = store
          .getState()
          .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === layerId);
        if (!layer) throw new Error('A current native layer is required.');
        store.dispatch(wrapTo('compute', layerConfigChange(layer, {isVisible})));
      },
      /** Test-only: why a dataset is or is not evictable right now. */
      getEvictionDiagnostics: () => ({
        footprints: getActiveRuntime().session.registry.getDatasetFootprints(),
        visibleIds: [...residentEviction['props'].getVisibleDatasetIds()],
        rebuildable: [...getActiveRuntime().publications.keys()].map(id => ({
          id,
          canRebuild: getEvictionRebuildSource(id) !== undefined,
          currentRevision: getActiveRuntime().session.registry.getDataset(id)?.revision,
          checkpointRevision: appState.deviceRecovery.checkpointSnapshot?.bundle.state.sources.find(
            candidate => candidate.dataset.datasetId === id
          )?.dataset.revision
        }))
      }),
      /** Test-only: rebuild an evicted dataset and wait for it, as a table view would. */
      ensureDatasetResident: (datasetId: string) => residentEviction.ensureResident(datasetId),
      /** Test-only: request more GPU memory than is available so the eviction policy must run. */
      applyMemoryPressure: () => {
        const registry = getActiveRuntime().session.registry;
        const reservation = registry.reserveMemory(
          registry.memoryUsage.availableBytes + 1,
          'gpu',
          'scratch'
        );
        reservation.release();
      },
      setRange: (minimum: number, maximum = 100) =>
        store.dispatch(createResidentFilterAction('value', 'value', [minimum, maximum])),
      setPointLabelStyle: (
        layerId: string,
        style: {size?: number; anchor?: string; alignment?: string}
      ) => {
        for (const [property, value] of Object.entries(style)) {
          const layer = store
            .getState()
            .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === layerId);
          if (!layer || layer.type !== 'point' || !layer.config.textLabel[0])
            throw new Error('A current native point label is required.');
          if (!['size', 'anchor', 'alignment'].includes(property))
            throw new Error('Unsupported point label style diagnostic action.');
          store.dispatch(wrapTo('compute', layerTextLabelChange(layer, 0, property, value)));
        }
      },
      setPointMarkerFill: (layerId: string, filled: boolean) => {
        const layer = store
          .getState()
          .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === layerId);
        if (!layer || layer.type !== 'point')
          throw new Error('A current native point is required.');
        store.dispatch(wrapTo('compute', layerVisConfigChange(layer, {filled})));
      },
      /** Native Trips marker orientation channels are not part of saved Kepler configurations. */
      setTripMarkerFields: (
        layerId: string,
        fields: {roll?: string; pitch?: string; yaw?: string; color?: string}
      ) => {
        const channels = {
          roll: 'adjustRoll',
          pitch: 'adjustPitch',
          yaw: 'adjustYaw',
          color: 'color'
        } as const;
        for (const [name, channel] of Object.entries(channels)) {
          const fieldName = fields[name as keyof typeof channels];
          const visState = store.getState().keplerGl['compute']?.visState;
          const layer = visState?.layers.find(candidate => candidate.id === layerId);
          if (!layer || layer.type !== 'trip')
            throw new Error('A current native trip is required.');
          const field = fieldName
            ? visState?.datasets[layer.config.dataId!]?.fields.find(
                candidate => candidate.name === fieldName
              )
            : null;
          if (fieldName && !field) throw new Error(`Unknown trip field ${fieldName}.`);
          store.dispatch(
            wrapTo(
              'compute',
              layerVisualChannelConfigChange(layer, {[`${name}Field`]: field ?? null}, channel)
            )
          );
        }
      },
      setAnimationTime: (value: number) =>
        store.dispatch(wrapTo('compute', setLayerAnimationTime(value))),
      setStreamedPointWeights: (
        datasetId: string,
        options: {forceStreamed?: boolean; maximumSlotsPerChunk?: number}
      ) =>
        getActiveRuntime()
          .publications.get(datasetId)
          ?.getAnalysis()
          .setStreamedWeightsOptions(options),
      setPublishedRange: (datasetId: string, field: string, minimum: number, maximum: number) =>
        store.dispatch(
          createOrUpdateFilter(`published-${datasetId}-${field}`, datasetId, field, [
            minimum,
            maximum
          ])
        ),
      setPublishedRangeValue: (
        datasetId: string,
        field: string,
        minimum: number,
        maximum: number
      ) => {
        const filters = store.getState().keplerGl['compute']?.visState?.filters ?? [];
        const identifier = `published-${datasetId}-${field}`;
        const matches = filters.flatMap((filter, index) =>
          filter.id === identifier ? [{filter, index}] : []
        );
        const match = matches[0];
        if (
          matches.length !== 1 ||
          !match ||
          match.filter.type !== 'range' ||
          match.filter.dataId.length !== 1 ||
          match.filter.dataId[0] !== datasetId ||
          match.filter.name.length !== 1 ||
          match.filter.name[0] !== field
        )
          throw new Error(
            'A unique matching native range filter must exist before a value update.'
          );
        return store.dispatch(
          wrapTo('compute', setFilter(match.index, 'value', [minimum, maximum]))
        );
      },
      setFilterPlot: (filterId: string, newProp: Record<string, unknown>) => {
        const filters = store.getState().keplerGl['compute']?.visState?.filters ?? [];
        const index = filters.findIndex(filter => filter.id === filterId);
        if (index < 0) throw new Error(`Unknown native filter ${filterId}.`);
        return store.dispatch(wrapTo('compute', setFilterPlot(index, newProp)));
      },
      setCategories: (categories: number[]) => {
        store.dispatch(createResidentFilterAction('category', 'inverted', false));
        store.dispatch(createResidentFilterAction('category', 'value', categories));
      },
      setCheckedCategories: setNativeCategories,
      openNativeCharts: () => store.dispatch(toggleMapControl('chart', 0)),
      addPolygon: (coordinates: number[][][]) =>
        store.dispatch(
          setPolygonFilterAllLayers({
            type: 'Feature',
            id: 'resident-test-polygon',
            properties: {},
            geometry: {type: 'Polygon', coordinates}
          })
        ),
      removeFilter: (index: number) => store.dispatch(removeFilter(index)),
      addNativeCpuDataset: (
        datasetId: string,
        fields: {name: string; type: string; format: string; analyzerType: string}[],
        rows: unknown[][]
      ) =>
        store.dispatch(
          addDataToMap({
            datasets: {info: {id: datasetId, label: datasetId}, data: {fields, rows}},
            options: {
              centerMap: false,
              autoCreateLayers: false,
              autoCreateTooltips: false,
              keepExistingConfig: true
            }
          })
        ),
      /** Adds Kepler's image overlay dataset (type 'bitmap') and lets Kepler create its layer. */
      addNativeBitmapDataset: (
        datasetId: string,
        imageUrl: string,
        bounds: [number, number, number, number]
      ) =>
        store.dispatch(
          addDataToMap({
            datasets: {
              info: {id: datasetId, label: datasetId, type: 'bitmap'},
              data: {fields: [], rows: []},
              metadata: {imageUrl, bounds}
            } as never,
            options: {centerMap: false, autoCreateTooltips: false, keepExistingConfig: true}
          })
        ),
      getNativeGpuFilterRange: (datasetId: string) =>
        store.getState().keplerGl['compute']?.visState?.datasets[datasetId]?.gpuFilter?.filterRange,
      getNativeFilteredCount: (datasetId: string) =>
        store.getState().keplerGl['compute']?.visState?.datasets[datasetId]?.filteredIndex.length,
      setTimestamp: setTimestampRange,
      timestampDomain: () => appState.runtime?.time.domainMilliseconds,
      setTime: (minimum: number, maximum: number) =>
        store.dispatch(createResidentFilterAction('time', 'value', [minimum, maximum])),
      readSummary: () => {
        const activeRuntime = appState.runtime;
        return activeRuntime?.latestSummary?.revision === activeRuntime?.explorer.currentRevision &&
          activeRuntime
          ? activeRuntime.latestSummary
          : null;
      },
      readPage: (offset = 0, limit = 10) => appState.runtime?.explorer.readPage(offset, limit),
      analysis: () => appState.runtime?.analysis,
      analysisForDataset: (datasetId: string) =>
        datasetId === DATASET_ID
          ? appState.runtime?.analysis
          : appState.runtime?.publications.get(datasetId)?.analysis,
      getH3ViewportMesh: (layerId: string) => appState.runtime?.styles.getH3ViewportMesh(layerId),
      getH3Metadata: (layerId: string) => appState.runtime?.styles.getH3Metadata(layerId),
      readPublishedAggregate: (datasetId: string, layerId: string, cellId: number) =>
        readNativeTooltip(datasetId, layerId, cellId, []),
      setMapViewport: (viewport: Parameters<typeof updateMap>[0]) =>
        store.dispatch(wrapTo('compute', updateMap(viewport, 0))),
      toggleLayerForMap: (mapIndex: number, layerId: string) =>
        store.dispatch(wrapTo('compute', toggleLayerForMap(mapIndex, layerId))),
      toggleSplitMap: (mapIndex: number) =>
        store.dispatch(wrapTo('compute', toggleSplitMap(mapIndex))),
      toggleSplitMapViewport: (payload: Parameters<typeof toggleSplitMapViewport>[0]) =>
        store.dispatch(wrapTo('compute', toggleSplitMapViewport(payload))),
      clearHoverInfo: () => store.dispatch(wrapTo('compute', onLayerHover(null, 0))),
      getHoverInfo: () => {
        const info = store.getState().keplerGl['compute']?.visState?.hoverInfo;
        return info
          ? {mapIndex: info.mapIndex, layerId: info.layer?.id ?? null, index: info.index}
          : null;
      },
      projectPositionInView: (
        viewId: string | null,
        longitude: number,
        latitude: number,
        altitude = 0
      ) => {
        const viewport = appState.runtime?.deck
          .getViewports()
          .find(candidate => viewId === null || candidate.id === viewId);
        const position = viewport?.project([longitude, latitude, altitude]);
        return viewport && position ? [viewport.x + position[0], viewport.y + position[1]] : null;
      },
      unprojectPositionInView: (viewId: string | null, x: number, y: number) => {
        const viewport = appState.runtime?.deck
          .getViewports()
          .find(candidate => viewId === null || candidate.id === viewId);
        return viewport?.unproject([x - viewport.x, y - viewport.y]) ?? null;
      },
      projectSourceRowInView: (viewId: string, index: number) => {
        const viewport = appState.runtime?.deck
          .getViewports()
          .find(candidate => candidate.id === viewId);
        const position = viewport?.project([
          appState.source.longitude[index],
          appState.source.latitude[index],
          0
        ]);
        return viewport && position ? [viewport.x + position[0], viewport.y + position[1]] : null;
      },
      getStyleLegend: (
        layerId: string,
        channel:
          | 'fillColor'
          | 'strokeColor'
          | 'size'
          | 'pointSize'
          | 'targetColor'
          | 'height'
          | 'elevationOffset'
      ) => appState.runtime?.styles.getLegend(layerId, channel),
      saveNativeConfig: () => KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute']),
      applyNativeConfig,
      reimportSource,
      renderedLayers: () => getRenderedLayerInfo(appState.runtime?.deck.props.layers),
      getHeatmapState: (layerId: string) => appState.runtime?.heatmaps.getDiagnostics(layerId),
      projectPosition: (longitude: number, latitude: number) =>
        appState.runtime?.deck.getViewports()[0]?.project([longitude, latitude, 0]),
      nativeEditor: () => store.getState().keplerGl['compute']?.visState?.editor,
      deleteNativeEditorFeatures: () => {
        const features = store.getState().keplerGl['compute']?.visState?.editor?.features ?? [];
        for (const feature of features)
          store.dispatch(wrapTo('compute', deleteFeature(toNativeFeature(feature))));
      },
      duplicateNativeLayer: (layerId: string) =>
        store.dispatch(wrapTo('compute', duplicateLayer(layerId))),
      removeNativeLayer: (layerId: string) =>
        store.dispatch(wrapTo('compute', removeLayer(layerId))),
      reorderNativeLayers: (order: string[]) =>
        store.dispatch(wrapTo('compute', reorderLayer(order as never))),
      getNativeLayerOrder: () => store.getState().keplerGl['compute']?.visState?.layerOrder,
      setNativePointColumns: (layerId: string, longitude: string, latitude: string) => {
        const layer = store
          .getState()
          .keplerGl['compute']?.visState?.layers.find(entry => entry.id === layerId);
        if (!layer) throw new Error(`Unknown layer ${layerId}`);
        const columns = layer.config.columns as Record<
          string,
          {value: string | null; fieldIdx: number}
        >;
        store.dispatch(
          wrapTo(
            'compute',
            layerConfigChange(layer, {
              columns: {
                ...columns,
                lng: {...columns['lng'], value: longitude},
                lat: {...columns['lat'], value: latitude}
              }
            } as never)
          )
        );
      },
      getGeometryVariantState: () => {
        const geometryVariants =
          appState.runtime && peekGeometryColumnVariantRuntime(appState.runtime.session);
        return {
          variantCount: geometryVariants?.variantCount ?? 0,
          gpuByteLength: geometryVariants?.gpuByteLength ?? 0,
          memoryUsage: appState.runtime?.session.registry.memoryUsage
        };
      },
      getPointVariantState: () => ({
        variantCount: appState.runtime?.pointPositionVariants.variantCount ?? 0,
        gpuByteLength: appState.runtime?.pointPositionVariants.gpuByteLength ?? 0,
        memoryUsage: appState.runtime?.session.registry.memoryUsage
      }),
      projectSourceRow: (index: number) =>
        appState.runtime?.deck
          .getViewports()[0]
          ?.project([appState.source.longitude[index], appState.source.latitude[index], 0]),
      pickSourceRow: async (index: number) => {
        const activeRuntime = appState.runtime;
        const position = activeRuntime?.deck
          .getViewports()[0]
          ?.project([appState.source.longitude[index], appState.source.latitude[index], 0]);
        if (!activeRuntime || !position) return null;
        const picked = await activeRuntime.deck.pickObjectAsync({
          x: position[0],
          y: position[1],
          radius: 4,
          layerIds: getRenderedLayerInfo(activeRuntime.deck.props.layers)
            .filter(layer => layer.id === 'resident-points')
            .map(layer => layer.renderId)
        });
        return picked
          ? {
              index: picked.index,
              layerId: getRenderedLayerInfo(picked.layer)[0]?.id,
              renderId: picked.layer?.id
            }
          : null;
      },
      pickPublishedPosition,
      pickTripPosition: pickPublishedPosition,
      readClusters: async (layerId: string, memberLimit: number) => {
        const binding =
          appState.runtime && clusterRuntimes.get(appState.runtime.positions)?.getBinding(layerId);
        if (!appState.runtime || !binding) return null;
        const readFloats = async (vector: {
          data: {
            buffer: {readAsync(byteOffset: number, byteLength: number): Promise<Uint8Array>};
            byteOffset: number;
            length: number;
            byteStride: number;
          }[];
        }) => {
          const [chunk] = vector.data;
          const bytes = await chunk.buffer.readAsync(
            chunk.byteOffset,
            chunk.length * chunk.byteStride
          );
          return Array.from(
            new Float32Array(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
            )
          );
        };
        const clusters = [];
        for (let index = 0; index < binding.clusterCount; index++) {
          const read = await clusterRuntimes
            .get(appState.runtime.positions)
            ?.getMembers(layerId, index, memberLimit);
          clusters.push({value: read?.value, total: read?.members.total, rows: read?.members.rows});
        }
        const mapState = store.getState().keplerGl['compute']?.mapState;
        return {
          clusterCount: binding.clusterCount,
          centers: await readFloats(binding.centers),
          view: {
            longitude: mapState?.longitude,
            latitude: mapState?.latitude,
            zoom: mapState?.zoom,
            width: mapState?.width,
            height: mapState?.height
          },
          clusters
        };
      },
      /** Pages the requested rows of a published dataset in (lazy for resident results). */
      readPublishedRows: async (datasetId: string, rowIndices: number[]) => {
        const published = getActiveRuntime().publications.get(datasetId);
        if (!published) throw new Error('Unknown published dataset.');
        const {resident, source: table} = published.dataset.value;
        await resident?.ensureRows(rowIndices);
        return {
          resident: Boolean(resident),
          rowCount: table.numRows,
          loadedRowCount: resident?.loadedRowCount,
          readbackBytes: resident?.readbackByteLength,
          rows: rowIndices.map(rowIndex => {
            const row = table.get(rowIndex)?.toJSON() ?? null;
            return row
              ? Object.fromEntries(
                  Object.entries(row).map(([name, value]) => [
                    name,
                    typeof value === 'bigint' ? Number(value) : value
                  ])
                )
              : null;
          })
        };
      },
      source: () => appState.source.rows
    }
  });
}
