// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';

/**
 * Creates a deck.gl layer whose `data` is a row count and whose accessors take the row index.
 * deck.gl types accessors over data objects, so these props cannot satisfy the layer's declared
 * props; the cast lives here once instead of at every layer.
 */
export function createIndexedLayer<LayerType extends Layer>(
  LayerClass: new (props: never) => LayerType,
  props: object
): LayerType {
  return new LayerClass(props as never);
}
