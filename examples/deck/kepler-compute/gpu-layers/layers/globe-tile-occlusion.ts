// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Far-side occlusion policy for basemap tile layers in Deck's GlobeView.
 *
 * Deck `TileLayer`, `MVTLayer` and `TerrainLayer` sublayers draw through the planet in GlobeView
 * because GlobeView has no occluder. A depth disk (`GPUGlobeDepthDiskLayer`) drawn first lets the
 * tiles depth-test against it. Basemaps drawn by MapLibre cull their own far side.
 */

/** Deck's globe common-space radius. */
export const GLOBE_COMMON_RADIUS = 256;

const GLOBE_TILE_LAYER_NAMES = /^(TileLayer|MVTLayer|TerrainLayer|GPUTileLayer)$/;

export type GlobeDepthDiskGeometry = {
  /** Distance of the disk plane from the globe center, toward the camera. */
  shift: number;
  /** Disk radius: the globe's silhouette cross-section at the tangent-cone plane. */
  radius: number;
};

/** Returns the silhouette disk for a camera at `cameraDistance`, or null inside the sphere. */
export function getGlobeDepthDiskGeometry(cameraDistance: number): GlobeDepthDiskGeometry | null {
  if (!(cameraDistance > GLOBE_COMMON_RADIUS)) return null;
  const ratio = GLOBE_COMMON_RADIUS / cameraDistance;
  return {
    shift: GLOBE_COMMON_RADIUS * ratio,
    radius: GLOBE_COMMON_RADIUS * Math.sqrt(1 - ratio * ratio)
  };
}

type TileLayerCandidate = {
  constructor?: {layerName?: string};
  props?: {tile?: unknown};
  parent?: TileLayerCandidate | null;
} | null;

/**
 * True for Deck tile layers and their tile sublayers. Sublayers are recognized structurally (the
 * TileLayer passes `props.tile` to each, and the parent chain reaches a tile layer class), never by
 * id or by generic classes such as BitmapLayer that user layers also use.
 */
export function isGlobeTileLayer(layer: unknown): boolean {
  let candidate = layer as TileLayerCandidate | undefined;
  if (!candidate) return false;
  if (candidate.props?.tile) return true;
  while (candidate) {
    if (GLOBE_TILE_LAYER_NAMES.test(candidate.constructor?.layerName ?? '')) return true;
    candidate = candidate.parent;
  }
  return false;
}

export type GlobeTileParameters = {
  depthCompare?: string;
  depthWriteEnabled?: boolean;
  [name: string]: unknown;
};

/** Tiles must depth-test the disk so far-side tiles fail against it; user-set depth parameters win. */
export function getGlobeTileParameters(parameters: GlobeTileParameters = {}): GlobeTileParameters {
  return {
    ...parameters,
    depthCompare: parameters.depthCompare ?? 'less-equal',
    depthWriteEnabled: parameters.depthWriteEnabled ?? true
  };
}

/**
 * Returns layer ids in draw order with the depth disk guaranteed ahead of the first tile layer.
 * Layers keep their relative order; the disk is inserted when missing and moved when late.
 */
export function orderGlobeTileLayers<LayerType extends {id: string}>(
  layers: LayerType[],
  depthDiskId: string,
  isTileLayer: (layer: LayerType) => boolean = isGlobeTileLayer
): {layers: LayerType[]; needsDepthDisk: boolean} {
  const firstTileIndex = layers.findIndex(isTileLayer);
  if (firstTileIndex < 0) return {layers, needsDepthDisk: false};
  const diskIndex = layers.findIndex(layer => layer.id === depthDiskId);
  if (diskIndex >= 0 && diskIndex < firstTileIndex) return {layers, needsDepthDisk: false};
  if (diskIndex < 0) return {layers, needsDepthDisk: true};
  const disk = layers[diskIndex];
  const rest = layers.filter((_, index) => index !== diskIndex);
  const insertAt = rest.findIndex(isTileLayer);
  return {
    layers: [...rest.slice(0, insertAt), disk, ...rest.slice(insertAt)],
    needsDepthDisk: false
  };
}

/** Orders layers so the depth disk precedes the first tile layer, creating the disk when absent. */
export function insertGlobeDepthDisk<LayerType extends {id: string}>(
  layers: LayerType[],
  depthDiskId: string,
  createDepthDisk: () => LayerType,
  isTileLayer: (layer: LayerType) => boolean = isGlobeTileLayer
): LayerType[] {
  const ordered = orderGlobeTileLayers(layers, depthDiskId, isTileLayer);
  if (!ordered.needsDepthDisk) return ordered.layers;
  const firstTileIndex = ordered.layers.findIndex(isTileLayer);
  return [
    ...ordered.layers.slice(0, firstTileIndex),
    createDepthDisk(),
    ...ordered.layers.slice(firstTileIndex)
  ];
}
