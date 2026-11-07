// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PointSelectorLayer} from './resident-point-layer-selectors';

export type CoordinateRoleSet = {longitude: string; latitude: string; altitude?: string};

export type LayerRoleSetBinding = {
  /** Stable key of the coordinate role set: one prepared position variant (or the primary) per key. */
  key: string;
  roles: CoordinateRoleSet;
  /** True when the role set equals the dataset's primary published coordinates. */
  primary: boolean;
  layerIds: string[];
};

export function getCoordinateRoleSetKey(roles: CoordinateRoleSet): string {
  return `${roles.longitude}\u0000${roles.latitude}\u0000${roles.altitude ?? ''}`;
}

/**
 * Groups the layers of one dataset by the lng/lat(/altitude) columns they map. Layers in the same
 * group share one prepared position variant and register their own selector over it; the primary
 * group reuses the published positions. Layers without both roles are not grouped.
 */
export function groupLayersByCoordinateRoleSet(
  layers: readonly PointSelectorLayer[],
  datasetId: string,
  primary: {longitude?: string; latitude?: string}
): LayerRoleSetBinding[] {
  const groups = new Map<string, LayerRoleSetBinding>();
  for (const layer of layers) {
    if (layer.config.dataId !== datasetId) continue;
    const longitude = layer.config.columns['lng']?.value;
    const latitude = layer.config.columns['lat']?.value;
    if (!longitude || !latitude) continue;
    const altitude = layer.config.columns['altitude']?.value || undefined;
    const roles = {longitude, latitude, ...(altitude ? {altitude} : {})};
    // Same rule as resolvePointVariantSource: lng/lat decide, altitude only selects the XYZ vector.
    const isPrimary = longitude === primary.longitude && latitude === primary.latitude;
    const key = getCoordinateRoleSetKey(roles);
    const group = groups.get(key) ?? {key, roles, primary: isPrimary, layerIds: []};
    group.layerIds.push(layer.id);
    groups.set(key, group);
  }
  return [...groups.values()];
}

/** One readable diagnostic per layer that has no prepared position variant, for the layer notice. */
export function describeUnavailableRoleSet(layerId: string, reason: string): string {
  return reason.includes(layerId) ? reason : `Layer ${layerId}: ${reason}`;
}
