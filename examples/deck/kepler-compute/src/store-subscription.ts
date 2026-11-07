// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderBasemapAttribution, getActiveBasemapAttributionSegments} from './basemap-attribution';
import {appState} from './app-state';
import {type KeplerSelectionLayer} from '@luma-contributors/kepler';
import {
  resolvePublishedGeometryVariant,
  makeGeometryVariantSelector,
  peekGeometryColumnVariantRuntime
} from './geometry-column-variants';
import {isPolygonFootprintLayerType, getFootprintLayerColumn} from './cell-layer-types';
import {DATASET_ID} from './sample-data';
import {planPointLayerSelectorWithVariants} from './point-variant-layer-source';
import {describeUnavailableRoleSet} from './binding3-role-sets';
import {getResidentChartTimeNullEpochFilters} from './resident-chart-plan';
import {store} from './app-store';
import {nativeAutoPrepare} from './native-hosts';
import {groupByRequests, joinRequests, suitabilityRequests} from './resident-operations';
import {scheduleRecoveryCheckpoint} from './recovery';
import {makeNativeMapViewport} from './split-view';
import {flowFieldBindings, flowH3Bindings, omitRetiredLayers} from './layers';
import {
  notify,
  roleSetDiagnosticLayerIds,
  layerDiagnostics,
  reportError,
  plotDiagnostics,
  clusterRuntimes,
  styleDiagnostics
} from './app-services';
import {publicationsInProgress, isResidentDataset} from './publication';

store.subscribe(() => {
  // Attribution of the style Kepler currently shows (picker entry or native custom style).
  renderBasemapAttribution(
    getActiveBasemapAttributionSegments(store.getState().keplerGl['compute']?.mapStyle)
  );
  nativeAutoPrepare.evaluate();
  groupByRequests.reconcile();
  joinRequests.reconcile();
  suitabilityRequests.reconcile();
  scheduleRecoveryCheckpoint();
  const instance = store.getState().keplerGl['compute'];
  appState.nativeMapZoom = instance?.mapState?.zoom;
  if (instance?.mapState && instance.mapState !== appState.previousNativeMapState) {
    appState.previousNativeMapState = instance.mapState;
    appState.nativeMapViewport = makeNativeMapViewport(instance.mapState);
  }
  const state = instance?.visState;
  if (!state) return;
  // Layer removal releases flow runtimes (GPU particles, derived H3 centers, filter words).
  const activeLayerIds = new Set<string>(state.layers.map(layer => layer.id));
  flowFieldBindings.retainLayers(activeLayerIds);
  flowH3Bindings.retainLayers(activeLayerIds);
  if (appState.runtime) {
    const activeRuntime = appState.runtime;
    const fixtureLayers: KeplerSelectionLayer[] = [];
    const publishedLayers = new Map<string, KeplerSelectionLayer[]>();
    for (const layer of state.layers) {
      const datasetId = layer.config.dataId;
      if (!datasetId) continue;
      const published = activeRuntime.publications.get(datasetId);
      if (published) {
        const geometryVariant = resolvePublishedGeometryVariant(
          activeRuntime.session,
          () => queueMicrotask(notify),
          published,
          layer
        );
        if (geometryVariant.status !== 'primary') {
          const variantSelector =
            geometryVariant.status === 'variant'
              ? makeGeometryVariantSelector(layer, geometryVariant.variant)
              : undefined;
          if (variantSelector) {
            const selectors = publishedLayers.get(datasetId) ?? [];
            selectors.push(variantSelector);
            publishedLayers.set(datasetId, selectors);
          }
          continue;
        }
      }
      if (
        published?.h3 &&
        layer.type === 'hexagonId' &&
        layer.config.columns['hex_id']?.value === published.h3Column
      ) {
        const selectors = publishedLayers.get(datasetId) ?? [];
        selectors.push({
          id: layer.id,
          visible: layer.config.isVisible,
          geometryRevision: published.dataset.revision,
          geometry: {
            kind: 'point',
            positions: published.h3.centers,
            validity: published.h3.validity
          }
        });
        publishedLayers.set(datasetId, selectors);
        continue;
      }
      const representative = published?.representativeGeometry;
      if (
        published &&
        representative &&
        isPolygonFootprintLayerType(layer.type) &&
        getFootprintLayerColumn(layer) === (published.cellIndex?.column ?? published.geometryColumn)
      ) {
        const selectors = publishedLayers.get(datasetId) ?? [];
        selectors.push({
          id: layer.id,
          visible: layer.config.isVisible,
          geometryRevision: published.dataset.revision,
          geometry: representative
        });
        publishedLayers.set(datasetId, selectors);
        continue;
      }
      if (
        published?.representativeUnsupportedReason &&
        isPolygonFootprintLayerType(layer.type) &&
        getFootprintLayerColumn(layer) === (published.cellIndex?.column ?? published.geometryColumn)
      ) {
        // Pinned native converter yields no centroid for unlabelled dense unions: never inside.
        const selectors = publishedLayers.get(datasetId) ?? [];
        selectors.push({
          id: layer.id,
          visible: layer.config.isVisible,
          geometryRevision: published.dataset.revision,
          geometry: {kind: 'no-representative'}
        });
        publishedLayers.set(datasetId, selectors);
        continue;
      }
      const positions = datasetId === DATASET_ID ? activeRuntime.positions : published?.positions;
      if (!positions) continue;
      const columns = layer.config.columns;
      const longitude = datasetId === DATASET_ID ? 'longitude' : published?.coordinates?.longitude;
      const latitude = datasetId === DATASET_ID ? 'latitude' : published?.coordinates?.latitude;
      const revision = published?.dataset.revision ?? activeRuntime.explorer.dataset.revision;
      let selector: KeplerSelectionLayer | undefined;
      const variantDataset = published?.dataset ?? activeRuntime.explorer.dataset;
      const pointPlan = planPointLayerSelectorWithVariants(
        activeRuntime.pointPositionVariants,
        layer,
        {
          id: variantDataset.id,
          revision: variantDataset.revision,
          prepared: variantDataset.value.prepared
        },
        {longitude, latitude, positions, revision}
      );
      if (pointPlan.status === 'selector') selector = pointPlan.selector;
      // A coordinate pair that cannot be prepared leaves the layer without a selector: say why.
      if (pointPlan.status === 'mismatch') {
        roleSetDiagnosticLayerIds.add(layer.id);
        layerDiagnostics.set(layer.id, describeUnavailableRoleSet(layer.id, pointPlan.reason));
      } else if (roleSetDiagnosticLayerIds.delete(layer.id)) layerDiagnostics.delete(layer.id);
      if (
        datasetId === DATASET_ID &&
        (layer.type === 'line' || layer.type === 'arc') &&
        columns['lng0']?.value === longitude &&
        columns['lat0']?.value === latitude &&
        columns['lng1']?.value === 'targetLongitude' &&
        columns['lat1']?.value === 'targetLatitude'
      )
        selector = {
          id: layer.id,
          visible: layer.config.isVisible,
          geometryRevision: revision,
          geometry: {
            kind: 'endpoint-pair',
            sourcePositions: positions,
            targetPositions: activeRuntime.targetPositions
          }
        };
      if (selector) {
        if (datasetId === DATASET_ID) fixtureLayers.push(selector);
        else {
          const selectors = publishedLayers.get(datasetId) ?? [];
          selectors.push(selector);
          publishedLayers.set(datasetId, selectors);
        }
      }
    }
    const geometryVariants = peekGeometryColumnVariantRuntime(activeRuntime.session);
    if (geometryVariants?.hasUnreferencedVariants && !activeRuntime.session.deviceLost) {
      const geometryFence = activeRuntime.device.createFence();
      void geometryVariants
        .retainLayers(
          state.layers.map(layer => layer.id),
          geometryFence.signaled
        )
        .catch(reportError)
        .finally(() => geometryFence.destroy());
    }
    if (
      activeRuntime.pointPositionVariants.hasUnreferencedVariants &&
      !activeRuntime.session.deviceLost
    ) {
      // Variants left without layers (remapped or hidden pairs) are destroyed after in-flight frames.
      const variantFence = activeRuntime.device.createFence();
      void activeRuntime.pointPositionVariants
        .retainLayers(
          state.layers.map(layer => layer.id),
          variantFence.signaled
        )
        .catch(reportError)
        .finally(() => variantFence.destroy());
    }
    try {
      activeRuntime.tableBinding.setChartTimeNullEpochFilters(
        getResidentChartTimeNullEpochFilters(
          state.charts ?? [],
          DATASET_ID,
          Object.keys(activeRuntime.explorer.dataset.value.timeColumns)
        )
      );
      activeRuntime.tableBinding.setLayers(fixtureLayers, state.layers);
      for (const published of activeRuntime.publications.values()) {
        published.scopes.setChartTimeNullEpochFilters(
          getResidentChartTimeNullEpochFilters(
            state.charts ?? [],
            published.dataset.id,
            Object.keys(published.dataset.value.timeColumns)
          )
        );
        published.scopes.setLayers(publishedLayers.get(published.dataset.id) ?? [], state.layers);
      }
    } catch (error) {
      queueMicrotask(() => reportError(error));
    }
  }
  if (!appState.cleanupScheduled) {
    appState.cleanupScheduled = true;
    queueMicrotask(() => {
      appState.cleanupScheduled = false;
      const current = store.getState().keplerGl['compute']?.visState;
      const activeRuntime = appState.runtime;
      if (!current || !activeRuntime) return;
      const filterIds = new Set(current.filters.map(filter => filter.id));
      for (const filterId of plotDiagnostics.keys())
        if (!filterIds.has(filterId)) plotDiagnostics.delete(filterId);
      for (const datasetId of activeRuntime.publications.keys()) {
        if (!current.datasets[datasetId] && !publicationsInProgress.has(datasetId)) {
          void activeRuntime.removePublication(datasetId).then(notify).catch(reportError);
        }
      }
    });
  }
  const topology = JSON.stringify(
    state.layers
      .filter(layer => isResidentDataset(layer.config.dataId ?? ''))
      .map(layer => ({
        id: layer.id,
        dataId: layer.config.dataId,
        type: layer.type,
        columns: layer.config.columns
      }))
  );
  const currentLayerIds = new Set(state.layers.map(layer => layer.id));
  const currentHeatmapIds = new Set(
    state.layers
      .filter(layer => layer.type === 'heatmap' && layer.config.isVisible)
      .map(layer => layer.id)
  );
  const retiredHeatmapIds = [...appState.previousHeatmapIds].filter(
    layerId => !currentHeatmapIds.has(layerId)
  );
  appState.previousHeatmapIds = currentHeatmapIds;
  const removedLayerIds = [...appState.previousLayerIds].filter(
    layerId => !currentLayerIds.has(layerId)
  );
  if ((removedLayerIds.length || retiredHeatmapIds.length) && appState.runtime) {
    const activeRuntime = appState.runtime;
    // Detach render consumers before any native dataset/style cleanup fences release inputs.
    activeRuntime.deck.setProps({
      layers: omitRetiredLayers(activeRuntime.deck.props.layers ?? [], [
        ...removedLayerIds,
        ...retiredHeatmapIds
      ])
    });
    queueMicrotask(() => {
      if (appState.runtime !== activeRuntime) return;
      const fence = activeRuntime.device.createFence();
      const retainedLayerIds =
        store.getState().keplerGl['compute']?.visState?.layers.map(layer => layer.id) ?? [];
      activeRuntime.styles.retainLayers(retainedLayerIds, fence.signaled);
      void activeRuntime.labels.retainLayers(retainedLayerIds, fence.signaled).catch(reportError);
      void activeRuntime.pointAltitudes
        .retainLayers(retainedLayerIds, fence.signaled)
        .catch(reportError);
      void activeRuntime.pointPositionVariants
        .retainLayers(retainedLayerIds, fence.signaled)
        .catch(reportError);
      void peekGeometryColumnVariantRuntime(activeRuntime.session)
        ?.retainLayers(retainedLayerIds, fence.signaled)
        .catch(reportError);
      void activeRuntime.heatmaps
        .retainLayers([...currentHeatmapIds], fence.signaled)
        .catch(reportError);
      void fence.signaled.finally(() => fence.destroy());
    });
  }
  for (const layerId of appState.previousLayerIds) {
    if (!currentLayerIds.has(layerId)) {
      appState.runtime?.aggregation.remove(layerId);
      for (const clusterRuntime of clusterRuntimes.values()) clusterRuntime.remove(layerId);
      for (const published of appState.runtime?.publications.values() ?? [])
        published.aggregation?.remove(layerId);
      layerDiagnostics.delete(layerId);
      styleDiagnostics.delete(layerId);
    }
  }
  appState.previousLayerIds = currentLayerIds;
  if (topology === appState.previousLayerTopology) return;
  appState.previousLayerTopology = topology;
  // Layer-scoped polygon masks depend on layer identity/order for every resident dataset, not
  // only the primary one: a deleted or duplicated layer on a published dataset must re-plan too.
  const polygonDatasetIds = [
    ...new Set(
      state.filters
        .filter(filter => filter.type === 'polygon')
        .flatMap(filter => filter.dataId)
        .filter(isResidentDataset)
    )
  ];
  if (!polygonDatasetIds.length) return;
  if (polygonDatasetIds.includes(DATASET_ID)) appState.unsupportedSelection = true;
  const activeRuntime = appState.runtime;
  queueMicrotask(() => {
    if (!activeRuntime || appState.runtime !== activeRuntime) return;
    const current = store.getState().keplerGl['compute']?.visState;
    for (const datasetId of polygonDatasetIds)
      current?.datasets[datasetId]?.filterTable(current.filters, current.layers);
    store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
  });
});
