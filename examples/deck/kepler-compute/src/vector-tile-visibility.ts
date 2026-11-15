// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Tileset2D} from '@deck.gl/geo-layers';
import type {Viewport} from '@deck.gl/core';

/**
 * Report native refinement visibility after every traversal, including partially loaded views.
 * onViewportLoad only reports selected tiles after loading completes; cached parent/child fallbacks
 * can be visible without being selected. Preserve the owner's custom tileset implementation.
 */
export function createVectorTileVisibilityTilesetClass(
  BaseTileset: typeof Tileset2D,
  onVisibleTiles: (tileKeys: string[]) => void
): typeof Tileset2D {
  return class ResidentVectorTileTileset<
    DataType = unknown,
    ViewStateType = Viewport
  > extends BaseTileset<DataType, ViewStateType> {
    override update(...updateArguments: Parameters<Tileset2D['update']>): number {
      const frameNumber = super.update(...updateArguments);
      onVisibleTiles(
        this.tiles
          .filter(tile => tile.isVisible)
          .map(tile => {
            const {x, y, z} = tile.index;
            return `${z}/${x}/${y}`;
          })
      );
      return frameNumber;
    }
  };
}
