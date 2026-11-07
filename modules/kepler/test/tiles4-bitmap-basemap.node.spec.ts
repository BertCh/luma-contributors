// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getStyleAttributionSegments,
  parseAttributionSegments
} from '../../../examples/deck/kepler-compute/src/basemap-attribution';
import {makeBitmapBoundsPatch} from '../../../examples/deck/kepler-compute/src/bitmap-georeference';
import {BitmapLiveBinding} from '../../../examples/deck/kepler-compute/src/raster-live-bitmap';

describe('bitmap bounds patch', () => {
  it('normalizes and validates edited bounds', () => {
    expect(makeBitmapBoundsPatch([10, 5, -10, -5], 0.5)).toEqual({
      boundsWest: -10,
      boundsSouth: -5,
      boundsEast: 10,
      boundsNorth: 5,
      opacity: 0.5
    });
    expect(() => makeBitmapBoundsPatch([0, 0, 0, 10])).toThrow();
    expect(() => makeBitmapBoundsPatch([-200, 0, 10, 10])).toThrow();
  });
  it('prunes state of removed bitmap layers', () => {
    const layers = [{id: 'b1', type: 'bitmap', config: {visConfig: {}}}];
    const binding = new BitmapLiveBinding({getLayers: () => layers});
    expect(binding.pruneRemovedLayers()).toBe(0);
    layers.length = 0;
    expect(binding.pruneRemovedLayers()).toBe(0);
  });
});

describe('attribution links', () => {
  it('keeps http(s) anchors and drops other markup and schemes', () => {
    expect(
      parseAttributionSegments(
        '&copy; <a href="https://osm.org/c">OSM</a> <b>x</b> <a href="javascript:alert(1)">bad</a>'
      )
    ).toEqual([{text: '©'}, {text: 'OSM', href: 'https://osm.org/c'}, {text: 'x'}, {text: 'bad'}]);
  });
  it('dedupes source attributions', () => {
    const style = {
      sources: {a: {attribution: '<a href="https://e.org">E</a>'}, b: {attribution: 'E'}}
    };
    expect(getStyleAttributionSegments(style)).toHaveLength(1);
  });
});
