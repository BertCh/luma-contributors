// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type ResidentGeometryReader} from './resident-select-geometry-control';
import {appState} from './app-state';
import {isPolygonFootprintLayerType, getFootprintLayerColumn} from './cell-layer-types';
import {readResidentSelectedGeometry} from './resident-select-geometry';
import {DATASET_ID} from './sample-data';
import {readOriginalTooltipValue} from './original-table-values';
import {createResidentBrushController} from './resident-brush-controller';
import {type GPUVector} from '@luma-contributors/graph/gpu-data';
import {getRenderedLayerInfo} from './main-layer-info';
import {store} from './app-store';
import {clusterRuntimes} from './app-services';

export const readSelectedGeometry: ResidentGeometryReader = async identity => {
  const activeRuntime = appState.runtime;
  const published = activeRuntime?.publications.get(identity.datasetId);
  const layer = store
    .getState()
    .keplerGl['compute']?.visState?.layers.find(candidate => candidate.id === identity.layerId);
  if (
    !activeRuntime ||
    activeRuntime.isLayerRegistrationPending ||
    !published ||
    !layer ||
    layer.config.dataId !== identity.datasetId ||
    !published.consumer.explorer.isCurrent ||
    (!isPolygonFootprintLayerType(layer.type) && layer.type !== 'hexagonId')
  )
    return {status: 'stale', reason: 'The clicked resident geometry is no longer current.'};
  const column = layer.type === 'hexagonId' ? published.h3Column : published.geometryColumn;
  if (!column)
    return {status: 'unsupported', reason: 'The selected source has no published geometry column.'};
  if (
    (layer.type === 'hexagonId'
      ? layer.config.columns['hex_id']?.value
      : getFootprintLayerColumn(layer)) !==
    (layer.type === 'hexagonId' ? column : (published.cellIndex?.column ?? column))
  )
    return {
      status: 'unsupported',
      reason: 'The native layer geometry role no longer matches the published source.'
    };
  const selectionRevision = published.scopes.currentRevision;
  const result = await readResidentSelectedGeometry({
    registry: activeRuntime.session.registry,
    datasetId: identity.datasetId,
    revision: published.dataset.revision,
    sourceRowIndex: identity.sourceRowIndex,
    kind: layer.type === 'hexagonId' ? 'hexagonId' : 'geojson',
    column,
    featureId: identity.featureId,
    maximumPayloadBytes: 1024 * 1024,
    maximumVertices: 10000,
    h3Accessor: 'decimal-or-hex'
  });
  if (result.status !== 'ready') return result;
  return {
    ...result,
    isCurrent: () =>
      result.isCurrent() &&
      appState.runtime === activeRuntime &&
      activeRuntime.publications.get(identity.datasetId) === published &&
      published.scopes.currentRevision === selectionRevision &&
      store
        .getState()
        .keplerGl['compute']?.visState?.layers.find(
          candidate => candidate.id === identity.layerId
        ) === layer
  };
};

/**
 * The hovered popover, the click-pinned popover and the compare-mode pinned read all start together,
 * but the resident explorer admits only a few concurrent reads and answers the rest with null
 * ("row no longer available"). Tooltip reads therefore run one at a time.
 */
export function readNativeTooltip(
  ...request: Parameters<typeof readNativeTooltipNow>
): ReturnType<typeof readNativeTooltipNow> {
  const result = appState.tooltipReadQueue.then(() => readNativeTooltipNow(...request));
  appState.tooltipReadQueue = result.catch(() => undefined);
  return result;
}

/** Native tooltips read one current source row or aggregate cell, never the CPU table facade. */
async function readNativeTooltipNow(
  datasetId: string,
  layerId: string,
  sourceRowIndex: number,
  fieldNames: readonly string[]
): Promise<Record<string, string | number | boolean | null> | null> {
  const activeRuntime = appState.runtime;
  if (!activeRuntime || activeRuntime.session.deviceLost) return null;
  const layer = store
    .getState()
    .keplerGl['compute']?.visState?.layers.find(layer => layer.id === layerId);
  if (layer?.type === 'cluster') {
    // The picking index of a cluster is its position in the current cluster list.
    const positions =
      datasetId === DATASET_ID
        ? activeRuntime.positions
        : activeRuntime.publications.get(datasetId)?.positions;
    const read = await clusterRuntimes.get(positions)?.getMembers(layerId, sourceRowIndex, 8);
    if (!read) return null;
    const {members, value} = read;
    return {
      Cluster: sourceRowIndex,
      Points: members.total,
      Value: value,
      Members: members.rows.join(', ') + (members.truncated ? ', ...' : '')
    };
  }
  if (layer?.type === 'grid' || layer?.type === 'hexagon') {
    const published = activeRuntime.publications.get(datasetId);
    const aggregation =
      datasetId === DATASET_ID ? activeRuntime.aggregation : published?.aggregation;
    const selection = published?.scopes ?? activeRuntime.tableBinding;
    const selectionRevision = selection.currentRevision;
    const cell = await aggregation?.readCell(layerId, sourceRowIndex);
    if (
      !cell ||
      appState.runtime !== activeRuntime ||
      selection.currentRevision !== selectionRevision ||
      (published && !published.consumer.explorer.isCurrent)
    )
      return null;
    return {
      Cell: cell.cellId,
      Observations: cell.count,
      [cell.statistic]: cell.label ?? cell.value
    };
  }
  const published = activeRuntime.publications.get(datasetId);
  const explorer = datasetId === DATASET_ID ? activeRuntime.explorer : published?.consumer.explorer;
  if (!explorer) return null;
  // Resolve one identity through the resident inspector. Format only requested retained source
  // cells with the same bounded strings/nested-value handling as the native original-data table.
  const page = await explorer.readRows([sourceRowIndex], []);
  if (!page || appState.runtime !== activeRuntime || !explorer.isCurrent) return null;
  await explorer.dataset.value.resident?.ensureRows([sourceRowIndex]);
  if (appState.runtime !== activeRuntime || !explorer.isCurrent) return null;
  const table = explorer.dataset.value.source;
  return {
    Row: sourceRowIndex,
    ...Object.fromEntries(
      fieldNames.slice(0, 8).flatMap(name => {
        const column = table.getChild(name);
        return column ? [[name, readOriginalTooltipValue(column, sourceRowIndex)]] : [];
      })
    )
  };
}

/** GPU brush masks over every resident point dataset, linked into each dataset's selection scopes. */
export const residentBrushController = createResidentBrushController({
  store,
  instanceId: 'compute',
  getSources: () => {
    if (!appState.runtime || appState.runtime.device.isLost) return [];
    const device = appState.runtime.device;
    const primary = appState.runtime;
    return [
      {
        datasetId: DATASET_ID,
        device,
        positions: primary.positions,
        linkBrushMask: mask => primary.tableBinding.setBrushMask(mask)
      },
      ...[...primary.publications].flatMap(([publishedId, published]) =>
        published.positions
          ? [
              {
                datasetId: publishedId,
                device,
                positions: published.positions,
                linkBrushMask: (mask: GPUVector<'uint32'> | null) =>
                  published.scopes.setBrushMask(mask)
              }
            ]
          : []
      )
    ];
  }
});

export async function pickPublishedPosition(
  datasetId: string,
  longitude: number,
  latitude: number,
  altitude = 0,
  viewId?: string
) {
  const activeRuntime = appState.runtime;
  const published = activeRuntime?.publications.get(datasetId);
  const viewport = viewId
    ? activeRuntime?.deck.getViewports().find(candidate => candidate.id === viewId)
    : activeRuntime?.deck.getViewports()[0];
  const projected = viewport?.project([longitude, latitude, altitude]);
  // Deck picks in canvas coordinates; a split view's projection is relative to its rectangle.
  const position = viewport && projected && [viewport.x + projected[0], viewport.y + projected[1]];
  if (!activeRuntime || activeRuntime.session.deviceLost || !published || !position) return null;
  const nativeIds = new Set(
    store
      .getState()
      .keplerGl['compute']?.visState?.layers.filter(layer => layer.config.dataId === datasetId)
      .map(layer => layer.id)
  );
  const picked = await activeRuntime.deck.pickObjectAsync({
    x: position[0],
    y: position[1],
    radius: 4,
    layerIds: getRenderedLayerInfo(activeRuntime.deck.props.layers)
      .filter(layer => nativeIds.has(layer.id))
      .map(layer => layer.renderId)
  });
  if (
    !picked ||
    picked.index < 0 ||
    appState.runtime !== activeRuntime ||
    activeRuntime.publications.get(datasetId) !== published
  )
    return null;
  const properties = picked.layer?.props;
  const sourceRowIndex =
    properties &&
    'nativeSourceRowIndex' in properties &&
    typeof properties.nativeSourceRowIndex === 'function'
      ? properties.nativeSourceRowIndex(picked.index)
      : picked.index;
  return {
    sourceRowIndex,
    primitiveIndex: picked.index,
    layerId: getRenderedLayerInfo(picked.layer)[0]?.id
  };
}
