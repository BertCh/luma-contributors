// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, LayerProps} from '@deck.gl/core';

/** Independent analysis overlays consume picks without impersonating native Kepler layers. */
export function withResidentResultEvents(
  layer: Layer,
  options: {isCurrent(): boolean; clearNativeHover(): void}
): Layer {
  const wrap =
    (callback: LayerProps['onHover']): NonNullable<LayerProps['onHover']> =>
    (information, event) => {
      if (!options.isCurrent()) return true;
      const handled = callback?.(information, event);
      if (handled === true) options.clearNativeHover();
      return handled;
    };
  return layer.clone({
    onHover: wrap(layer.props.onHover),
    onClick: wrap(layer.props.onClick)
  });
}
