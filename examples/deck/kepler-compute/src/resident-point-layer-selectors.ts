// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerSelectionLayer} from '@luma-contributors/kepler';

/** Native layer types whose polygon selector is a point test over the dataset lng/lat positions. */
const POINT_SELECTOR_LAYER_TYPES: ReadonlySet<string> = new Set([
  'point',
  'icon',
  'grid',
  'cluster',
  'heatmap',
  'hexagon'
]);

export type PointSelectorLayer = {
  id: string;
  type: string | null;
  config: {
    isVisible?: boolean;
    dataId?: string | null;
    columns: Record<string, {value?: string | null} | undefined>;
  };
};

export type ResidentPositionSource = {
  longitude: string | undefined;
  latitude: string | undefined;
  positions: GPUVector<'float32x2'>;
  revision: number;
};

export type PointSelectorPlan =
  | {status: 'selector'; selector: KeplerSelectionLayer}
  | {status: 'mismatch'; layerId: string; reason: string}
  | {status: 'not-point'};

/**
 * One selector per native layer id over the dataset's shared resident positions, so duplicated
 * point layers each own a layer mask while sharing the resident table and filters. A layer mapped
 * to a different lng/lat pair has no resident positions here and is reported instead of silently
 * borrowing another layer's coordinates.
 */
export function planPointLayerSelector(
  layer: PointSelectorLayer,
  source: ResidentPositionSource
): PointSelectorPlan {
  if (!layer.type || !POINT_SELECTOR_LAYER_TYPES.has(layer.type)) return {status: 'not-point'};
  const longitude = layer.config.columns['lng']?.value;
  const latitude = layer.config.columns['lat']?.value;
  if (longitude !== source.longitude || latitude !== source.latitude) {
    return {
      status: 'mismatch',
      layerId: layer.id,
      reason: `Layer ${layer.id} maps ${longitude ?? '?'}/${latitude ?? '?'} but the resident positions use ${source.longitude ?? '?'}/${source.latitude ?? '?'}; publish that coordinate pair as its own dataset.`
    };
  }
  return {
    status: 'selector',
    selector: {
      id: layer.id,
      visible: layer.config.isVisible,
      geometryRevision: source.revision,
      geometry: {kind: 'point', positions: source.positions}
    }
  };
}

/** Layers of one dataset that a polygon filter can target with an independent mask. */
export function listPointSelectorLayerIds(
  layers: readonly PointSelectorLayer[],
  datasetId: string,
  source: ResidentPositionSource
): string[] {
  return layers
    .filter(layer => layer.config.dataId === datasetId)
    .filter(layer => planPointLayerSelector(layer, source).status === 'selector')
    .map(layer => layer.id);
}
