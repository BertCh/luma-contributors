// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Layer, _GlobeView as GlobeView} from '@deck.gl/core';
import {
  GPUGlobeDepthDiskLayer,
  getGlobeTileParameters,
  insertGlobeDepthDisk,
  isGlobeTileLayer
} from '@deck.gl-community/gpu-layers';

export function flattenDeckLayers(layers: unknown): Layer[] {
  return (Array.isArray(layers) ? layers.flat(Infinity) : layers ? [layers] : []).filter(
    (layer): layer is Layer => Boolean(layer) && typeof layer === 'object'
  );
}
export function getRootLayerId(layer: Layer): string {
  let root = layer;
  while (root.parent) root = root.parent as Layer;
  return root.id;
}
/**
 * Swipe compare: both maps are full size and each is visible only on its side of the divider, so
 * each view gets its side's rectangle plus a projection that keeps the full-size map's geography.
 */

export function orderGlobeBasemapTiles<LayerType extends {id: string}>(
  layers: LayerType[]
): LayerType[] {
  const isTile = (layer: LayerType) =>
    layer.id.startsWith('globe-basemap') && layer.id.includes('-resident-tile-');
  const tiles = layers.filter(isTile);
  if (!tiles.length) return layers;
  const rest = layers.filter(layer => !isTile(layer));
  let insertIndex = 0;
  // Only the host's own globe layers anchor the insertion point; a data layer whose id merely
  // starts with `globe-` must stay above the basemap tiles.
  rest.forEach((layer, index) => {
    if (/^globe-(background|basemap|depth-disk)/.test(layer.id)) insertIndex = index + 1;
  });
  return [...rest.slice(0, insertIndex), ...tiles, ...rest.slice(insertIndex)];
}
/** Draws the depth disk before the first Deck tile layer so far-side tiles fail the depth test. */
export function orderGlobeDepthDisk<LayerType extends {id: string}>(
  layers: LayerType[],
  views: unknown
): LayerType[] {
  const hasGlobeView = [views].flat().some(view => view instanceof GlobeView);
  return hasGlobeView
    ? insertGlobeDepthDisk(
        layers,
        'globe-depth-disk-layer',
        () => new GPUGlobeDepthDiskLayer({id: 'globe-depth-disk-layer'}) as unknown as LayerType
      )
    : layers;
}
/**
 * Kepler's globe atmosphere and halo layers are GLSL `SimpleMeshLayer` shaders, which invalidate
 * every WebGPU pipeline they join, so the host omits these purely decorative layers. Resident
 * layers horizon-cull themselves (`gpuGlobeIsHidden`), and the GLSL depth disk that hides far-side
 * basemap tiles is replaced by the WGSL `GPUGlobeDepthDiskLayer`.
 */
export const GLSL_GLOBE_LAYER_IDS = /^(atmosphere(-sky|-huge-halo)?)$/;
export type LegacyGlobeParameters = {
  cull?: boolean;
  depthTest?: boolean;
  depthMask?: boolean;
  [name: string]: unknown;
};
/** Kepler's globe layers use WebGL parameter names, which WebGPU rejects mid render pass. */
export function toWebGPUParameters(parameters: LegacyGlobeParameters): Record<string, unknown> {
  const {cull, depthTest, depthMask, ...rest} = parameters;
  return {
    ...rest,
    // WebGPU's flipped clip-space Y reverses triangle winding, so Kepler's back-face cull of the
    // globe surface would remove the near hemisphere; the horizon is clipped per layer instead.
    ...(cull === undefined ? {} : {cullMode: 'none'}),
    ...(depthTest === undefined ? {} : {depthCompare: depthTest ? 'less-equal' : 'always'}),
    ...(depthMask === undefined ? {} : {depthWriteEnabled: depthMask})
  };
}
export function toWebGPUGlobeLayer<LayerType>(layer: LayerType): LayerType {
  const depthDisk = layer as {id?: unknown; props?: {getColor?: unknown}};
  if (depthDisk?.id === 'globe-depth-disk-layer') {
    const color = depthDisk.props?.getColor;
    return new GPUGlobeDepthDiskLayer({
      id: depthDisk.id,
      ...(Array.isArray(color) ? {fillColor: color as [number, number, number]} : {})
    }) as LayerType;
  }
  const candidate = layer as {props?: {parameters?: LegacyGlobeParameters}; clone?: Function};
  const parameters = candidate?.props?.parameters;
  // Deck tile layers (none by default; the basemap is MapLibre) must depth-test against the disk.
  if (isGlobeTileLayer(layer) && typeof candidate.clone === 'function') {
    return candidate.clone({
      parameters: getGlobeTileParameters(toWebGPUParameters(parameters ?? {}))
    }) as LayerType;
  }
  return parameters &&
    typeof candidate.clone === 'function' &&
    ('cull' in parameters || 'depthTest' in parameters || 'depthMask' in parameters)
    ? (candidate.clone({parameters: toWebGPUParameters(parameters)}) as LayerType)
    : layer;
}
/**
 * Kepler's GlobeView sets `clear: true, clearColor: false` to keep the WebGL picking buffer
 * transparent. On WebGPU that view clear leaves every layer invisible, and the canvas is already
 * transparent without it, so the host rebuilds the view without those flags.
 */
export const lastGlobeViews: Record<string, {source: unknown; stripped: unknown}> = {};
/** `slot` keeps the primary and the hidden second Deck's views from evicting each other. */
export function withoutGlobeViewClear(views: unknown, slot = 'primary'): unknown {
  const view = views as {
    props?: Record<string, unknown>;
    constructor: new (props: object) => unknown;
  };
  if (!view?.props || Array.isArray(views) || view.props['clearColor'] !== false) return views;
  if (lastGlobeViews[slot]?.source !== views) {
    const {clear: _clear, clearColor: _clearColor, ...props} = view.props;
    lastGlobeViews[slot] = {source: views, stripped: new view.constructor(props)};
  }
  return lastGlobeViews[slot].stripped;
}
export function isGlslGlobeLayer(layer: unknown): boolean {
  return (
    Boolean(layer) &&
    typeof layer === 'object' &&
    typeof (layer as {id?: unknown}).id === 'string' &&
    GLSL_GLOBE_LAYER_IDS.test((layer as {id: string}).id)
  );
}
