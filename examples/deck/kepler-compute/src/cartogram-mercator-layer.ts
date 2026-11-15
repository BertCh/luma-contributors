// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GetPickingInfoParams, PickingInfo} from '@deck.gl/core';
import {GPUScatterplotLayer, type GPUScatterplotLayerProps} from '../gpu-layers/index';
import type {RenderPass} from '@luma.gl/core';
import {
  getCartogramMercatorLayerProps,
  isCartogramMercatorViewport,
  type CartogramMercatorLayerProps as CartogramProjectionProps
} from './analysis-cartogram-projection';

/** Borrowed, source-aligned vectors; the projection and radius unit conversion are fixed. */
export type CartogramMercatorLayerProps = Omit<
  GPUScatterplotLayerProps,
  keyof CartogramProjectionProps
>;

/** Check before creating a runtime: this layer does not convert other projected CRSs. */
export function getCartogramMercatorUnavailableReason(crs: string): string | undefined {
  return crs === 'EPSG:3857'
    ? undefined
    : 'The cartogram overlay requires projected EPSG:3857 coordinates in meters.';
}

/**
 * Display-only Dorling circles in the EPSG:3857 plane. All position, radius, color and selection
 * vectors remain borrowed; the layer neither reads rows nor replaces source map geometry.
 * The draw gate also applies to Deck's picking pass, including multi-view maps.
 */
export class CartogramMercatorLayer extends GPUScatterplotLayer<CartogramMercatorLayerProps> {
  static override layerName = 'CartogramMercatorLayer';

  constructor(...props: Partial<CartogramMercatorLayerProps>[]) {
    super(...props, getCartogramMercatorLayerProps());
  }

  override draw(options: {renderPass: RenderPass}): void {
    if (!isCartogramMercatorViewport(this.context.viewport)) return;
    super.draw(options);
  }

  override getPickingInfo(params: GetPickingInfoParams): PickingInfo {
    if (!isCartogramMercatorViewport(params.info.viewport ?? this.context.viewport)) {
      params.info.picked = false;
      params.info.index = -1;
      params.info.object = undefined;
      return params.info;
    }
    return super.getPickingInfo(params);
  }
}
