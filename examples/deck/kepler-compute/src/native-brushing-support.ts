// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Audit of the pinned Kepler 3.3 `@kepler.gl/layers` brushing support against the resident
 * renderers. Native layers attach Deck's `BrushingExtension` and pass `getBrushingExtensionProps`
 * (radius = brush size in kilometers times 1000). Layers absent from this table (geojson, trip,
 * heatmap, cluster, grid, hexagon, s2, a5, geohash, flow, flow field, raster, 3D tiles) never
 * attach the extension natively, so resident parity means they must not brush either.
 */
export type NativeBrushingTarget = 'source' | 'source_target';

export type NativeBrushingEntry = {
  /** Kepler layer type key. */
  layerType: 'point' | 'icon' | 'hexagonId' | 'arc' | 'line';
  /** Native `brushingTarget`: the position tested against the mouse. */
  target: NativeBrushingTarget;
  /** Native sublayers that carry the brush (label layers are brushed with their anchor). */
  nativeSublayers: readonly string[];
  /** Resident GPU layers that evaluate the brush in their shader (render-only, mask untouched). */
  residentLayers: readonly string[];
};

export const NATIVE_BRUSHING_AUDIT: readonly NativeBrushingEntry[] = [
  {
    layerType: 'point',
    target: 'source',
    nativeSublayers: ['ScatterplotLayer', 'text labels'],
    residentLayers: ['GPUScatterplotLayer', 'GPUPointLabelLayer']
  },
  {
    layerType: 'icon',
    target: 'source',
    nativeSublayers: ['SvgIconLayer', 'text labels'],
    residentLayers: ['GPUMeshIconLayer (inherits GPUScatterplotLayer)', 'GPUPointLabelLayer']
  },
  {
    layerType: 'hexagonId',
    target: 'source',
    nativeSublayers: ['H3HexagonLayer'],
    residentLayers: ['GPUMeshColumnLayer', 'GPUSolidPolygonLayer', 'GPUPathLayer']
  },
  {
    layerType: 'arc',
    target: 'source_target',
    nativeSublayers: ['ArcLayer', 'target arrow head'],
    residentLayers: ['GPUArcLayer', 'GPUArrowHeadLayer']
  },
  {
    layerType: 'line',
    target: 'source_target',
    nativeSublayers: ['LineLayer / ArcLayer (globe)', 'target arrow head'],
    residentLayers: ['GPULineLayer', 'GPUArcLayer', 'GPUArrowHeadLayer']
  }
];

export function getNativeBrushingEntry(layerType: string): NativeBrushingEntry | undefined {
  return NATIVE_BRUSHING_AUDIT.find(entry => entry.layerType === layerType);
}

export type BrushInteractionConfig = {enabled?: boolean; config?: {size?: number}} | undefined;

/** Native `getBrushingExtensionProps` radius contract: kilometers to meters, off when disabled. */
export function getBrushingRadiusMeters(brush: BrushInteractionConfig): number {
  const size = brush?.config?.size;
  return typeof size === 'number' && Number.isFinite(size) && size > 0 ? size * 1000 : 0;
}
