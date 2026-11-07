// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ComputeRuntime} from './compute-runtime';
import {WebMercatorViewport, Layer, type LayersList} from '@deck.gl/core';
import {appState} from './app-state';
import {withResidentResultEvents} from './resident-result-events';
import {wrapTo, onLayerHover, layerSetIsValid} from '@kepler.gl/actions';
import {FlowFieldBindings} from './flow-field-bindings';
import {FlowH3Bindings} from './flow-h3-source';
import {TripMarkerModelCache} from './trip-marker-model';
import {createResidentLayerClasses} from './resident-layers';
import {DATASET_ID} from './sample-data';
import {
  resolvePublishedGeometryVariant,
  makeGeometryVariantOverrides
} from './geometry-column-variants';
import {
  makeConnectionVariantOverride,
  resolveVariantPositions,
  resolveAggregationVariant
} from './point-variant-extended-sources';
import {makePointVariantBindingOverride} from './point-variant-layer-source';
import {
  createPreparedLayerClasses,
  PREPARE_DATA_MESSAGE,
  AUTO_PREPARE_FAILURE_PREFIX
} from './prepared-layer-classes';
import {createResidentHeatmapLayerClasses} from './resident-heatmap-layers';
import {createResidentAggregationLayerClasses} from './resident-aggregation-layers';
import {ResidentClusterRuntime, type ResidentClusterField} from './cluster-runtime';
import {aggregate} from '@kepler.gl/utils';
import {store} from './app-store';
import {notify, residentEviction, layerDiagnostics, clusterRuntimes} from './app-services';
import {isResidentDataset} from './publication';
import {nativeAutoPrepare} from './native-hosts';

function getH3Viewport(activeRuntime: ComputeRuntime): WebMercatorViewport | undefined {
  if (appState.nativeMapViewport) return appState.nativeMapViewport;
  const viewport = activeRuntime.deck.getViewports()[0];
  return viewport instanceof WebMercatorViewport ? viewport : undefined;
}

export function getResidentAnalysisLayers(): Layer[] {
  const owner = appState.runtime;
  if (!owner) return [];
  return owner.getAnalysisResultLayers().map(layer =>
    withResidentResultEvents(layer, {
      isCurrent: () => appState.runtime === owner && owner.isReadyForRender,
      clearNativeHover: () => {
        if (store.getState().keplerGl['compute']?.visState?.hoverInfo)
          store.dispatch(wrapTo('compute', onLayerHover(null, 0)));
      }
    })
  );
}

export const flowFieldBindings = new FlowFieldBindings();

export const flowH3Bindings = new FlowH3Bindings();

const tripMarkerModels = new TripMarkerModelCache(() => queueMicrotask(notify));

const primitiveLayerClasses = createResidentLayerClasses({
  isResidentDataset,
  resolveTripMarker: layer => tripMarkerModels.resolve(layer.config.visConfig),
  resolveFlowH3: layer => {
    const activeRuntime = appState.runtime;
    const published = layer.config.dataId
      ? activeRuntime?.publications.get(layer.config.dataId)
      : undefined;
    if (!activeRuntime || !published || activeRuntime.session.deviceLost) return undefined;
    return flowH3Bindings.resolve(layer.id, layer.config.columns, {
      device: activeRuntime.session.device,
      table: published.dataset.value.source,
      getRevision: () => published.scopes.currentRevision,
      requestUpdate: () => queueMicrotask(notify),
      memoryRegistry: activeRuntime.session.registry
    });
  },
  resolveFlowField: (layer, selectionMask) => {
    const activeRuntime = appState.runtime;
    const published = layer.config.dataId
      ? activeRuntime?.publications.get(layer.config.dataId)
      : undefined;
    // Flow fields are not styled layers: read the published layer mask directly, not through styles.
    selectionMask = published
      ? published.scopes.registeredLayerIds.includes(layer.id)
        ? published.scopes.getLayerMask(layer.id)
        : published.renderBinding?.selectionMask
      : undefined;
    if (!activeRuntime || activeRuntime.session.deviceLost) return undefined;
    return flowFieldBindings.resolve(layer.id, layer.config as never, {
      device: activeRuntime.session.device,
      table: published?.dataset.value.source,
      requestUpdate: () => queueMicrotask(notify),
      // Layer filters narrow the field grid like native filtered rows.
      selection:
        selectionMask && published
          ? {
              mask: selectionMask,
              getRevision: () => published.scopes.currentRevision,
              requestUpdate: () => queueMicrotask(notify),
              memoryRegistry: activeRuntime.session.registry
            }
          : undefined
    });
  },
  resolveDataset(datasetId, layer) {
    const activeRuntime = appState.runtime;
    // A hidden layer must not rebuild its evicted table, or eviction would thrash immediately.
    if (layer.config.isVisible) {
      residentEviction.touch(datasetId);
      residentEviction.requestResident(datasetId);
    }
    if (
      activeRuntime &&
      (activeRuntime.isLayerRegistrationPending || appState.deviceRecovery.status.phase !== 'ready')
    ) {
      const pendingBinding = activeRuntime.publications.get(datasetId)?.renderBinding;
      return pendingBinding ? {...pendingBinding, enabled: false} : undefined;
    }
    if (datasetId !== DATASET_ID) {
      const published = activeRuntime?.publications.get(datasetId);
      const binding = published?.renderBinding;
      if (!activeRuntime || !published || !binding) return undefined;
      const sourceMask = published.scopes.registeredLayerIds.includes(layer.id)
        ? published.scopes.getLayerMask(layer.id)
        : binding.selectionMask;
      const altitude = published.coordinates?.altitude;
      // Layers bound to a geometry/H3/cell column other than the published one use an on-demand variant.
      const geometryVariant = resolvePublishedGeometryVariant(
        activeRuntime.session,
        () => queueMicrotask(notify),
        published,
        layer
      );
      if (geometryVariant.status === 'unavailable')
        return {...binding, enabled: false, unsupportedReason: geometryVariant.reason};
      const geometryOverride =
        geometryVariant.status === 'variant'
          ? makeGeometryVariantOverrides(geometryVariant.variant, layer.id, published.scopes)
          : undefined;
      const activeAltitude =
        (layer.type === 'point' || layer.type === 'icon' || (layer.type as string) === '3D') &&
        altitude &&
        'altitude' in layer.config.columns &&
        layer.config.columns.altitude?.value === altitude &&
        published.pointPositions?.positions3D;
      const selectionMask = activeRuntime.pointAltitudes.resolve(
        datasetId,
        layer.id,
        sourceMask,
        activeAltitude ? published.pointPositions?.altitudeValidity : undefined
      );
      return activeRuntime.styles.resolve(
        layer,
        {
          ...binding,
          ...(activeAltitude && binding.point
            ? {
                point: {
                  positions: activeAltitude,
                  columns: {...binding.point.columns, altitude}
                }
              }
            : {}),
          selectionMask,
          ...geometryOverride?.binding,
          ...makeConnectionVariantOverride(
            activeRuntime.pointPositionVariants,
            layer,
            {
              id: datasetId,
              revision: published.dataset.revision,
              prepared: published.dataset.value.prepared
            },
            binding.connections?.columns
          ),
          ...makePointVariantBindingOverride(
            activeRuntime.pointPositionVariants,
            layer,
            {
              id: datasetId,
              revision: published.dataset.revision,
              prepared: published.dataset.value.prepared
            },
            {
              longitude: published.coordinates?.longitude,
              latitude: published.coordinates?.latitude
            },
            validity =>
              activeRuntime.pointAltitudes.resolve(datasetId, layer.id, sourceMask, validity)
          )
        },
        {
          h3: published.h3,
          h3Column: published.h3Column,
          h3MetadataMask:
            published.h3 && published.scopes.registeredLayerIds.includes(layer.id)
              ? published.scopes.getLayerTopologyMask(layer.id)
              : undefined,
          h3MetadataRevision: published.scopes.currentRevision + 1,
          h3Viewport: getH3Viewport(activeRuntime),
          // Deck's GlobeViewport carries `resolution`; its geometry unwraps, subdivides and cuts long edges.
          geometry:
            ((appState.nativeMapViewport as {resolution?: number} | undefined)?.resolution
              ? published.getGlobeGeometry(appState.nativeMapZoom)
              : published.releaseGlobeGeometry()) ?? published.geometry?.geometry,
          trips: published.tripGeometry,
          tripColumn: published.trip ? published.geometryColumn : undefined,
          geometryColumn: published.geometryColumn,
          domainMask: published.scopes.domainExplorer.selectionMask,
          domainRevision: published.scopes.currentRevision + 1,
          zoom: appState.nativeMapZoom,
          ...geometryOverride?.options
        }
      );
    }
    if (datasetId !== DATASET_ID || !activeRuntime) return undefined;
    return activeRuntime.styles.resolve(
      layer,
      {
        point: {positions: activeRuntime.positions, columns: {lat: 'latitude', lng: 'longitude'}},
        connections: {
          sourcePositions: activeRuntime.positions,
          targetPositions: activeRuntime.targetPositions,
          columns: {
            lat0: 'latitude',
            lng0: 'longitude',
            lat1: 'targetLatitude',
            lng1: 'targetLongitude'
          }
        },
        selectionMask: activeRuntime.tableBinding.registeredLayerIds.includes(layer.id)
          ? activeRuntime.tableBinding.getLayerMask(layer.id)
          : activeRuntime.explorer.selectionMask,
        enabled: !appState.unsupportedSelection,
        bounds: [-74.05, 40.69, -73.9, 40.84],
        colorOverride: activeRuntime.analysis.active ? activeRuntime.analysis.colors : undefined,
        ...makePointVariantBindingOverride(
          activeRuntime.pointPositionVariants,
          layer,
          {
            id: DATASET_ID,
            revision: activeRuntime.explorer.dataset.revision,
            prepared: activeRuntime.explorer.dataset.value.prepared
          },
          {longitude: 'longitude', latitude: 'latitude'},
          validity =>
            activeRuntime.pointAltitudes.resolve(
              DATASET_ID,
              layer.id,
              activeRuntime.tableBinding.registeredLayerIds.includes(layer.id)
                ? activeRuntime.tableBinding.getLayerMask(layer.id)
                : activeRuntime.explorer.selectionMask,
              validity
            )
        )
      },
      {
        domainMask: activeRuntime.tableBinding.domainExplorer.selectionMask,
        domainRevision: activeRuntime.tableBinding.currentRevision + 1,
        zoom: appState.nativeMapZoom,
        dictionaries: {category: [0, 1, 2, 3]}
      }
    );
  },
  resolvePointNeighborSource(layer) {
    const activeRuntime = appState.runtime;
    const dataset =
      layer.config.dataId === DATASET_ID
        ? activeRuntime?.explorer.dataset
        : activeRuntime?.publications.get(layer.config.dataId)?.dataset;
    return dataset?.value.source;
  },
  resolvePointLabels(layer, binding, renderOptions, context) {
    const activeRuntime = appState.runtime;
    const dataset =
      layer.config.dataId === DATASET_ID
        ? activeRuntime?.explorer.dataset
        : activeRuntime?.publications.get(layer.config.dataId)?.dataset;
    const geojsonAnchors = layer.type === 'geojson' ? binding.geojson?.labelAnchors : undefined;
    const labelPositions = binding.point?.positions ?? geojsonAnchors?.positions;
    if (!activeRuntime || !dataset || !labelPositions) {
      return layer.type === 'geojson' && layer.config.textLabel?.some(label => label.field)
        ? {layers: [], reason: 'GeoJSON labels need resident representative anchors.'}
        : {layers: []};
    }
    const viewport = getH3Viewport(activeRuntime);
    const longitude = renderOptions.mapState?.longitude ?? viewport?.longitude;
    const latitude = renderOptions.mapState?.latitude ?? viewport?.latitude;
    const zoom = renderOptions.mapState?.zoom ?? viewport?.zoom;
    if (longitude === undefined || latitude === undefined || zoom === undefined)
      return {layers: [], reason: 'Point labels are waiting for the native map viewport.'};
    return activeRuntime.labels.resolve({
      dataset,
      layerId: layer.id,
      // Icon and point configurations type a missing label background as null.
      labels: (layer.config.textLabel ?? []).map(label => ({
        ...label,
        backgroundColor: label.backgroundColor ?? undefined
      })),
      positions: labelPositions,
      anchors: geojsonAnchors
        ? {kind: 'gpu', positions: geojsonAnchors.positions, validity: geojsonAnchors.validity}
        : binding.point
          ? {
              kind: 'columns',
              longitude: binding.point.columns.lng,
              latitude: binding.point.columns.lat
            }
          : undefined,
      selectionMask: binding.selectionMask,
      radius: context.radius,
      radiusScale: context.radiusScale,
      visible: context.visible,
      mapState: {
        longitude,
        latitude,
        zoom,
        scale: renderOptions.mapState?.scale,
        globe: renderOptions.mapState?.globe,
        width: renderOptions.mapState?.width,
        height: renderOptions.mapState?.height,
        pitch: renderOptions.mapState?.pitch,
        bearing: renderOptions.mapState?.bearing
      }
    });
  },
  onRenderError(layer) {
    const activeRuntime = appState.runtime;
    const configuration = layer.config;
    queueMicrotask(() => {
      if (!activeRuntime || appState.runtime !== activeRuntime) return;
      const currentLayer = store
        .getState()
        .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === layer.id);
      // A queued failure from a retired render must not disable a replacement native layer.
      if (currentLayer !== layer || currentLayer.config !== configuration) return;
      store.dispatch(wrapTo('compute', layerSetIsValid(currentLayer, false)));
    });
  },
  onUnsupported(layerId, reason) {
    if (reason) layerDiagnostics.set(layerId, reason);
    else layerDiagnostics.delete(layerId);
    queueMicrotask(notify);
  }
});

export const residentLayerClasses = createPreparedLayerClasses(
  {
    ...primitiveLayerClasses,
    ...createResidentHeatmapLayerClasses({
      isResidentDataset,
      resolveHeatmap(datasetId, layer) {
        const activeRuntime = appState.runtime;
        if (
          !activeRuntime ||
          activeRuntime.isLayerRegistrationPending ||
          appState.deviceRecovery.status.phase !== 'ready'
        )
          return {};
        const published = activeRuntime.publications.get(datasetId);
        const fixture = datasetId === DATASET_ID;
        const positions = fixture ? activeRuntime.positions : published?.positions;
        const sourceDataset = fixture ? activeRuntime.explorer.dataset : published?.dataset;
        const scopes = fixture ? activeRuntime.tableBinding : published?.scopes;
        const coordinates = fixture
          ? {longitude: 'longitude', latitude: 'latitude'}
          : published?.coordinates;
        if (
          !positions ||
          !sourceDataset ||
          !scopes ||
          !coordinates ||
          !scopes.registeredLayerIds.includes(layer.id)
        )
          return {reason: 'Resident heatmaps require registered published point coordinate roles.'};
        if ((fixture && appState.unsupportedSelection) || published?.error)
          return {reason: published?.error ?? 'The resident selection is unavailable.'};
        const heatmapVariant = resolveVariantPositions(
          activeRuntime.pointPositionVariants,
          layer,
          {
            id: sourceDataset.id,
            revision: sourceDataset.revision,
            prepared: sourceDataset.value.prepared
          },
          coordinates
        );
        if (heatmapVariant.status === 'unavailable') return {reason: heatmapVariant.reason};
        try {
          return activeRuntime.heatmaps.resolve({
            layerId: layer.id,
            dataset: sourceDataset,
            positions: heatmapVariant.status === 'variant' ? heatmapVariant.positions : positions,
            columns:
              heatmapVariant.status === 'variant'
                ? heatmapVariant.columns
                : {lat: coordinates.latitude, lng: coordinates.longitude},
            selectionMask: scopes.getLayerMask(layer.id),
            revision: scopes.currentRevision,
            weightColumn: layer.config.weightField?.name,
            weightScale: layer.config.weightScale,
            domainMask: scopes.domainExplorer.selectionMask,
            domainRevision: scopes.currentRevision + 1
          });
        } catch (error) {
          return {reason: error instanceof Error ? error.message : String(error)};
        }
      },
      getBounds: datasetId =>
        datasetId === DATASET_ID
          ? appState.runtime?.aggregation.projection.geographicBounds
          : appState.runtime?.publications.get(datasetId)?.bounds,
      onUnsupported(layerId, reason) {
        if (reason) layerDiagnostics.set(layerId, reason);
        else layerDiagnostics.delete(layerId);
        queueMicrotask(notify);
      },
      onRenderError(layer, error) {
        const activeRuntime = appState.runtime;
        const configuration = layer.config;
        layerDiagnostics.set(layer.id, error.message);
        queueMicrotask(() => {
          if (!activeRuntime || appState.runtime !== activeRuntime) return;
          const current = store
            .getState()
            .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === layer.id);
          if (current !== layer || current.config !== configuration) return;
          store.dispatch(wrapTo('compute', layerSetIsValid(current, false)));
        });
      }
    }),
    ...createResidentAggregationLayerClasses({
      isResidentDataset,
      resolveAggregation(datasetId, layer) {
        if (
          appState.runtime?.isLayerRegistrationPending ||
          appState.deviceRecovery.status.phase !== 'ready'
        )
          return {};
        if (!appState.runtime) return {reason: 'The resident selection is unavailable.'};
        if (datasetId !== DATASET_ID) {
          const published = appState.runtime.publications.get(datasetId);
          if (!published || published.error || !published.consumer.explorer.isCurrent)
            return {reason: published?.error ?? 'The published source selection is unavailable.'};
          if (!published.scopes.registeredLayerIds.includes(layer.id))
            return {reason: 'The published aggregate point scope is not registered.'};
          try {
            const variantAggregation = resolveAggregationVariant(
              appState.runtime.pointPositionVariants,
              {
                device: appState.runtime.device,
                registry: appState.runtime.session.registry,
                requestUpdate: notify
              },
              layer as never,
              {
                id: datasetId,
                revision: published.dataset.revision,
                prepared: published.dataset.value.prepared,
                rowCount: published.dataset.rowCount,
                batchRowCounts: published.dataset.batches.map(batch => batch.rowCount),
                columns: Object.fromEntries(
                  Object.entries(published.dataset.value.columns).map(([name, values]) => [
                    name,
                    {values, validity: published.dataset.value.analytics.validity[name]}
                  ])
                )
              },
              {
                longitude: published.coordinates?.longitude,
                latitude: published.coordinates?.latitude
              },
              published.scopes.getLayerMask(layer.id),
              published.scopes.currentRevision
            );
            if (variantAggregation) return variantAggregation;
            return published
              .getAggregation()
              .resolve(
                layer,
                published.scopes.getLayerMask(layer.id),
                published.scopes.currentRevision,
                published.scopes.getLayerTopologyMask(layer.id)
              );
          } catch (error) {
            return {reason: error instanceof Error ? error.message : String(error)};
          }
        }
        if (appState.unsupportedSelection)
          return {reason: 'The resident selection is unavailable.'};
        const fixtureMask = appState.runtime.tableBinding.registeredLayerIds.includes(layer.id)
          ? appState.runtime.tableBinding.getLayerMask(layer.id)
          : appState.runtime.explorer.selectionMask;
        const fixtureValues = appState.runtime.explorer.dataset.value;
        const fixtureVariant = resolveAggregationVariant(
          appState.runtime.pointPositionVariants,
          {
            device: appState.runtime.device,
            registry: appState.runtime.session.registry,
            requestUpdate: notify
          },
          layer as never,
          {
            id: DATASET_ID,
            revision: appState.runtime.explorer.dataset.revision,
            prepared: fixtureValues.prepared,
            rowCount: appState.runtime.explorer.dataset.rowCount,
            batchRowCounts: appState.runtime.explorer.dataset.batches.map(batch => batch.rowCount),
            columns: Object.fromEntries(
              Object.entries(fixtureValues.columns).map(([name, values]) => [
                name,
                {values, validity: fixtureValues.analytics.validity[name]}
              ])
            )
          },
          {longitude: 'longitude', latitude: 'latitude'},
          fixtureMask,
          appState.runtime.tableBinding.currentRevision
        );
        if (fixtureVariant) return fixtureVariant;
        const registeredFixtureLayer = appState.runtime.tableBinding.registeredLayerIds.includes(
          layer.id
        );
        return appState.runtime.aggregation.resolve(
          layer,
          registeredFixtureLayer
            ? appState.runtime.tableBinding.getLayerMask(layer.id)
            : appState.runtime.explorer.selectionMask,
          appState.runtime.tableBinding.currentRevision,
          registeredFixtureLayer
            ? appState.runtime.tableBinding.getLayerTopologyMask(layer.id)
            : undefined
        );
      },
      resolveCluster(datasetId, layer, request) {
        const activeRuntime = appState.runtime;
        if (
          !activeRuntime ||
          activeRuntime.isLayerRegistrationPending ||
          appState.deviceRecovery.status.phase !== 'ready'
        )
          return {};
        const published =
          datasetId === DATASET_ID ? undefined : activeRuntime.publications.get(datasetId);
        const primaryPositions =
          datasetId === DATASET_ID ? activeRuntime.positions : published?.positions;
        if (
          !primaryPositions ||
          (published && (published.error || !published.consumer.explorer.isCurrent))
        )
          return {reason: 'The resident cluster source is unavailable.'};
        const clusterDataset = published?.dataset ?? activeRuntime.explorer.dataset;
        const clusterVariant = resolveVariantPositions(
          activeRuntime.pointPositionVariants,
          layer,
          {
            id: clusterDataset.id,
            revision: clusterDataset.revision,
            prepared: clusterDataset.value.prepared
          },
          datasetId === DATASET_ID
            ? {longitude: 'longitude', latitude: 'latitude'}
            : {
                longitude: published?.coordinates?.longitude,
                latitude: published?.coordinates?.latitude
              }
        );
        if (clusterVariant.status === 'unavailable') return {reason: clusterVariant.reason};
        const positions =
          clusterVariant.status === 'variant' ? clusterVariant.positions : primaryPositions;
        // Drop CPU-readback runtimes whose source vectors are no longer current.
        const currentSources = new Set<unknown>([activeRuntime.positions, positions]);
        for (const entry of activeRuntime.publications.values())
          currentSources.add(entry.positions);
        for (const [source, clusterRuntime] of clusterRuntimes)
          if (!currentSources.has(source)) {
            clusterRuntime.destroy();
            clusterRuntimes.delete(source);
          }
        let clusterRuntime = clusterRuntimes.get(positions);
        if (!clusterRuntime) {
          clusterRuntime = new ResidentClusterRuntime({
            device: activeRuntime.session.device,
            registry: activeRuntime.session.registry,
            positions,
            requestUpdate: notify
          });
          clusterRuntimes.set(positions, clusterRuntime);
        }
        let field: ResidentClusterField | undefined;
        if (request.colorField) {
          const {name, aggregation} = request.colorField;
          const source = (published?.dataset ?? activeRuntime.explorer.dataset).value;
          const values = source.columns[name];
          if (!values) return {reason: 'Resident clusters need a numeric resident color field.'};
          field = {
            name,
            values,
            validity: source.analytics.validity[name],
            aggregate: entries => aggregate(entries, aggregation) as number
          };
        }
        const scopes = published ? published.scopes : activeRuntime.tableBinding;
        const scoped = scopes.registeredLayerIds.includes(layer.id);
        return clusterRuntime.resolve({
          layerId: layer.id,
          sourceRevision: published?.dataset.revision ?? activeRuntime.explorer.dataset.revision,
          mask: scoped
            ? scopes.getLayerMask(layer.id)
            : (published?.consumer.explorer ?? activeRuntime.explorer).selectionMask,
          maskRevision: scopes.currentRevision,
          settings: request.settings,
          view: request.view,
          field
        });
      },
      getBounds: datasetId =>
        datasetId === DATASET_ID
          ? appState.runtime?.aggregation.projection.geographicBounds
          : appState.runtime?.publications.get(datasetId)?.bounds,
      onUnsupported(layerId, reason) {
        if (reason) layerDiagnostics.set(layerId, reason);
        else layerDiagnostics.delete(layerId);
        queueMicrotask(notify);
      }
    })
  },
  {
    isPrepared: isResidentDataset,
    isPending: datasetId => nativeAutoPrepare.isPending(datasetId),
    getFailure: datasetId => nativeAutoPrepare.getFailure(datasetId),
    onPreparationRequired(_layerId, reason) {
      if (reason) appState.workflowNotice = reason;
      else if (
        appState.workflowNotice === PREPARE_DATA_MESSAGE ||
        appState.workflowNotice.startsWith(AUTO_PREPARE_FAILURE_PREFIX)
      )
        appState.workflowNotice = '';
      queueMicrotask(notify);
    }
  }
);

export function omitRetiredLayers(layers: LayersList, retiredIds: readonly string[]): LayersList {
  const retained: LayersList = [];
  for (const layer of layers) {
    if (Array.isArray(layer)) retained.push(omitRetiredLayers(layer, retiredIds));
    else if (
      layer instanceof Layer &&
      !retiredIds.some(id => layer.id === id || layer.id.startsWith(`${id}-`))
    )
      retained.push(layer);
  }
  return retained;
}
