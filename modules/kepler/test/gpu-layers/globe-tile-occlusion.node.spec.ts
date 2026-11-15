// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GLOBE_COMMON_RADIUS,
  getGlobeDepthDiskGeometry,
  getGlobeTileParameters,
  insertGlobeDepthDisk,
  isGlobeTileLayer,
  orderGlobeTileLayers
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/globe-tile-occlusion';

describe('globe tile occlusion', () => {
  it('sizes the depth disk to the silhouette cross-section', () => {
    expect(getGlobeDepthDiskGeometry(GLOBE_COMMON_RADIUS)).toBeNull();
    expect(getGlobeDepthDiskGeometry(100)).toBeNull();
    const disk = getGlobeDepthDiskGeometry(512)!;
    expect(disk.shift).toBeCloseTo(128);
    expect(disk.radius).toBeCloseTo(256 * Math.sqrt(0.75));
    const far = getGlobeDepthDiskGeometry(1e6)!;
    expect(far.shift).toBeLessThan(1);
    expect(far.radius).toBeCloseTo(GLOBE_COMMON_RADIUS, 2);
  });

  it('recognizes tile layers by class or tile sublayer structure, not by name or id', () => {
    class TileLayer {
      static layerName = 'TileLayer';
      id = 'basemap';
    }
    class BitmapLayer {
      static layerName = 'BitmapLayer';
      id = 'user-bitmap';
      parent: unknown = null;
      props: {tile?: unknown} = {};
    }
    const tileLayer = new TileLayer();
    const sublayer = new BitmapLayer();
    sublayer.props = {tile: {index: {x: 1, y: 2, z: 3}}};
    const nested = new BitmapLayer();
    nested.parent = tileLayer;
    expect(isGlobeTileLayer(tileLayer)).toBe(true);
    expect(isGlobeTileLayer(sublayer)).toBe(true);
    expect(isGlobeTileLayer(nested)).toBe(true);
    expect(isGlobeTileLayer(new BitmapLayer())).toBe(false);
    expect(isGlobeTileLayer({id: 'basemap-3-1-2-bitmap'})).toBe(false);
    expect(isGlobeTileLayer({id: 'scatterplot'})).toBe(false);
    expect(isGlobeTileLayer(null)).toBe(false);
  });

  it('defaults depth testing against the disk but keeps user choices', () => {
    expect(getGlobeTileParameters({blend: true})).toEqual({
      blend: true,
      depthCompare: 'less-equal',
      depthWriteEnabled: true
    });
    expect(getGlobeTileParameters({depthCompare: 'always', depthWriteEnabled: false})).toEqual({
      depthCompare: 'always',
      depthWriteEnabled: false
    });
  });

  it('puts the depth disk before the first tile layer', () => {
    const disk = {id: 'disk'};
    const tile = {id: 'map', props: {tile: {}}};
    const points = {id: 'points'};
    expect(orderGlobeTileLayers([points], 'disk').needsDepthDisk).toBe(false);
    expect(orderGlobeTileLayers([tile, points], 'disk').needsDepthDisk).toBe(true);
    expect(orderGlobeTileLayers([disk, tile], 'disk').layers).toEqual([disk, tile]);
    expect(orderGlobeTileLayers([points, tile, disk], 'disk').layers).toEqual([points, disk, tile]);
  });

  it('inserts a created depth disk ahead of the first tile layer only when tiles exist', () => {
    const tile = {id: 'map', props: {tile: {}}};
    const points = {id: 'points'};
    const create = () => ({id: 'disk'});
    expect(insertGlobeDepthDisk([points], 'disk', create)).toEqual([points]);
    expect(insertGlobeDepthDisk([points, tile], 'disk', create)).toEqual([
      points,
      {id: 'disk'},
      tile
    ]);
    expect(insertGlobeDepthDisk([tile, {id: 'disk'}], 'disk', create)).toEqual([
      {id: 'disk'},
      tile
    ]);
  });
});
