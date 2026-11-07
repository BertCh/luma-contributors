// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Makes Kepler's 3D tile layers pickable outside annotation mode.
 *
 * Kepler sets `pickable` on its Tile3D deck layer only while annotations are placed. The host
 * clones the top-level tileset layer with `pickable: true` so hover and click reach the tileset's
 * sublayers (Deck sublayers inherit `pickable` from their parent props).
 */

type PickableLayerLike = {
  constructor: {name: string};
  props?: Record<string, unknown>;
  clone?: (props: Record<string, unknown>) => unknown;
};

const TILE3D_OWNER_PATTERN = /Tile3DLayer/;

/** True for the top-level tileset owner layer (not the scenegraph/mesh sublayers it creates). */
export function isTile3DOwnerLayer(layer: unknown): layer is PickableLayerLike {
  return Boolean(
    layer &&
      typeof layer === 'object' &&
      TILE3D_OWNER_PATTERN.test((layer as PickableLayerLike).constructor.name)
  );
}

/** Returns a pickable clone of a non-pickable Tile3D owner layer, otherwise the layer itself. */
export function makeTile3DPickable<LayerT>(layer: LayerT): LayerT {
  if (!isTile3DOwnerLayer(layer) || layer.props?.['pickable'] || !layer.clone) return layer;
  return layer.clone({pickable: true}) as LayerT;
}
