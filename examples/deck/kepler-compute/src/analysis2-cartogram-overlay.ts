// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUScatterplotLayer, type GPUScatterplotLayerProps} from '../gpu-layers/index';
import {
  getDeclaredProjectedCartogramLayerProps,
  planDeclaredProjectedCartogram,
  type DeclaredProjectedCartogramOptions
} from './analysis-cartogram-declared-crs';
import {getCartogramMercatorUnavailableReason} from './cartogram-mercator-layer';

export type CartogramOverlayDecision =
  | {kind: 'web-mercator'}
  | {kind: 'declared-plane'; relativeError: number}
  | {kind: 'table-only'; reason: string};

/**
 * Chooses how a Dorling result is displayed: EPSG:3857 results use the Web Mercator overlay; any
 * other projected CRS needs a declared unit size and geographic anchor and a small enough extent.
 */
export function planCartogramOverlay(
  crs: string,
  resultBounds: readonly [number, number, number, number] | undefined,
  declaration?: DeclaredProjectedCartogramOptions
): CartogramOverlayDecision {
  if (getCartogramMercatorUnavailableReason(crs) === undefined) return {kind: 'web-mercator'};
  if (!declaration || !resultBounds)
    return {
      kind: 'table-only',
      reason: 'Declare the CRS unit size and a geographic anchor to draw this projected CRS.'
    };
  const plan = planDeclaredProjectedCartogram(declaration, {
    minimumX: resultBounds[0],
    minimumY: resultBounds[1],
    maximumX: resultBounds[2],
    maximumY: resultBounds[3]
  });
  return plan.ok
    ? {kind: 'declared-plane', relativeError: plan.relativeError}
    : {kind: 'table-only', reason: plan.reason ?? 'Declared plane overlay is unavailable.'};
}

/**
 * Display-only Dorling circles in a declared local tangent plane. Unlike the EPSG:3857 layer it
 * is a METER_OFFSETS layer, so it draws in every viewport; it is not yet created by the cartogram
 * runtime (see backlog ANALYSIS2-3-6).
 */
export class CartogramDeclaredPlaneLayer extends GPUScatterplotLayer<GPUScatterplotLayerProps> {
  static override layerName = 'CartogramDeclaredPlaneLayer';

  constructor(
    declaration: DeclaredProjectedCartogramOptions,
    ...props: Partial<GPUScatterplotLayerProps>[]
  ) {
    const declared = getDeclaredProjectedCartogramLayerProps(declaration);
    // The helper returns a plain 16-number array; Deck types it as a fixed-length matrix.
    super(...props, {
      ...declared,
      modelMatrix: declared.modelMatrix as unknown as GPUScatterplotLayerProps['modelMatrix']
    });
  }
}
