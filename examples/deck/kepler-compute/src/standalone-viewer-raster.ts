// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {BitmapLayer} from '@deck.gl/layers';
import {TileLayer} from '@deck.gl/geo-layers';

/**
 * Raster layers: Kepler raster tiles come from tile servers and the bundle embeds no pixels, so the
 * layer is drawn only when a `{z}/{x}/{y}` template is found in the saved configuration AND the page
 * is online. Otherwise the layer is listed as not drawn with that reason.
 */
const TEMPLATE_PATTERN = /^https?:\/\/.*\{z\}.*\{x\}.*\{y\}/;

export function findRasterTileTemplate(root: unknown): string | null {
  const queue: unknown[] = [root];
  let visited = 0;
  while (queue.length && visited++ < 20_000) {
    const value = queue.shift();
    if (typeof value === 'string' && TEMPLATE_PATTERN.test(value)) return value;
    if (value && typeof value === 'object') queue.push(...Object.values(value));
  }
  return null;
}

export function isViewerOnline(): boolean {
  return typeof navigator === 'undefined' || navigator.onLine !== false;
}

export function buildRasterLayer(options: {
  id: string;
  opacity: number;
  template: string;
}): TileLayer | null {
  return new TileLayer({
    id: options.id,
    data: options.template,
    opacity: options.opacity,
    pickable: false,
    minZoom: 0,
    maxZoom: 19,
    tileSize: 256,
    renderSubLayers: (props: any) => {
      const {west, south, east, north} = props.tile.bbox;
      return new BitmapLayer(props, {
        data: undefined,
        image: props.data,
        bounds: [west, south, east, north]
      });
    }
  });
}
