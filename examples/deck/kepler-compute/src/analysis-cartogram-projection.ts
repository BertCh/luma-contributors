// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, WebMercatorViewport, type Viewport} from '@deck.gl/core';
import type {GPUScatterplotLayerProps} from '@deck.gl-community/gpu-layers';

/** EPSG:3857 plane units to Deck's 512-unit common world; not ground-meter distances. */
export const CARTOGRAM_MERCATOR_COMMON_SCALE = 512 / (2 * Math.PI * 6378137);

export type CartogramMercatorLayerProps = Pick<
  GPUScatterplotLayerProps,
  | 'coordinateSystem'
  | 'coordinateOrigin'
  | 'modelMatrix'
  | 'radiusUnits'
  | 'radiusScale'
  | 'radiusMinPixels'
  | 'radiusMaxPixels'
  | 'billboard'
>;

/**
 * Borrow EPSG:3857 XY positions and radii directly. Deck applies the model scale before the
 * Cartesian origin; radiusScale is separate because modelMatrix does not scale circle radii.
 * Ground-plane circles retain the solver's projected geometry when the map is tilted.
 */
export function getCartogramMercatorLayerProps(): CartogramMercatorLayerProps {
  const scale = CARTOGRAM_MERCATOR_COMMON_SCALE;
  return {
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    coordinateOrigin: [256, 256, 0],
    modelMatrix: [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1],
    radiusUnits: 'common',
    radiusScale: scale,
    radiusMinPixels: 0,
    radiusMaxPixels: Number.MAX_SAFE_INTEGER,
    billboard: false
  };
}

/** Apply this per viewport: Cartesian common coordinates do not describe a globe surface. */
export function isCartogramMercatorViewport(viewport: Viewport): boolean {
  return viewport instanceof WebMercatorViewport;
}
