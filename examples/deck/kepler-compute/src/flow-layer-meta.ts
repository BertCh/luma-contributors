// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ClusterLevel} from '@flowmap.gl/data';
import type {FlowLocation} from './flow-source';

/** Layer metadata the native Flow layer publishes through `updateMeta`. */
export type FlowLayerMeta = {
  bounds: [number, number, number, number] | null;
  locations: FlowLocation[];
  clusterLevels: ClusterLevel[];
};

type FlowMetaSource = {
  bounds: [number, number, number, number] | null;
  getLayerData(): {locations: FlowLocation[]; clusterLevels: ClusterLevel[]};
};

/**
 * Native metadata for LAT_LNG and H3 flows. H3 endpoints are cell centers, so the layer bounds
 * come from the derived center runtime and never from the dataset's (possibly absent) point
 * bounds; `fallbackBounds` applies only while the runtime has no finite endpoint.
 */
export function getFlowLayerMeta(
  flow: FlowMetaSource,
  fallbackBounds?: [number, number, number, number] | null
): FlowLayerMeta {
  const layerData = flow.getLayerData();
  return {
    bounds: flow.bounds ?? fallbackBounds ?? null,
    locations: layerData.locations,
    clusterLevels: layerData.clusterLevels
  };
}

/** Whether publishing `next` would change what native consumers see (identity, not deep, compare). */
export function hasFlowLayerMetaChanged(
  current: Partial<FlowLayerMeta> | undefined,
  next: FlowLayerMeta
): boolean {
  return (
    current?.locations !== next.locations ||
    current?.clusterLevels !== next.clusterLevels ||
    current?.bounds?.join(',') !== next.bounds?.join(',')
  );
}
