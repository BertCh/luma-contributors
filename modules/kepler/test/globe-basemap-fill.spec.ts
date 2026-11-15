// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CompositeLayer, Deck, _GlobeView as GlobeView, type Layer} from '@deck.gl/core';
import {MVTLayer} from '@deck.gl/geo-layers';
import {Buffer, Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {GPUGlobeDepthDiskLayer} from '../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-globe-depth-disk-layer';
import {GlobeBasemapPolygonLayer} from '../../../examples/deck/kepler-compute/src/globe-basemap-polygon-layer';
import {renderGlobeBasemapTile} from '../../../examples/deck/kepler-compute/src/globe-basemap-tiles';

const SIZE = 256;
const WATER_COLOR = [17, 35, 48, 255] as const;

function getLeafLayers(layer: Layer): Layer[] {
  return layer instanceof CompositeLayer ? layer.getSubLayers().flatMap(getLeafLayers) : [layer];
}

test('a decoded z0 globe MVT fills both hemispheres through WGSL without a tile server', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const parent = document.createElement('div');
  document.body.append(parent);
  const color = device.createTexture({
    width: SIZE,
    height: SIZE,
    format: 'rgba8unorm',
    usage: Texture.RENDER | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: SIZE,
    height: SIZE,
    colorAttachments: [color],
    depthStencilAttachment: 'depth24plus'
  });
  const layout = color.computeMemoryLayout({width: SIZE, height: SIZE});
  const readback = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  const water = [
    {
      type: 'Feature',
      properties: {layerName: 'water'},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
          ]
        ]
      }
    }
  ];
  const tiles = new MVTLayer({
    id: 'globe-basemap-mvt-layer-fixture',
    data: 'memory://water/{z}/{x}/{y}.mvt',
    minZoom: 0,
    maxZoom: 0,
    zoomOffset: -2,
    // Match the native basemap extent: zoom 1.2 + offset -2 selects below minZoom. Without
    // bounds deck deliberately selects no tiles; with them it overviews the bounded z0 tile.
    extent: [-180, -85.051129, 180, 85.051129],
    binary: false,
    // Exercise native MVT's local-coordinate transform/ClipExtension and real polygon init, while
    // removing network/worker availability from this pixel regression.
    fetch: async () => ({shape: 'geojson-table', features: water}),
    renderSubLayers: renderGlobeBasemapTile as never,
    getFillColor: WATER_COLOR,
    getLineWidth: 0,
    getPointRadius: 0,
    parameters: {depthCompare: 'less-equal', depthWriteEnabled: false, cullMode: 'none'}
  });
  let error: Error | null = null;
  let frames = 0;
  const deck = new Deck({
    device,
    parent,
    width: SIZE,
    height: SIZE,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    views: new GlobeView({id: 'main', resolution: 72}),
    initialViewState: {longitude: -30, latitude: 20, zoom: 1.2},
    layers: [
      new GPUGlobeDepthDiskLayer({id: 'globe-depth-disk-layer', fillColor: [9, 16, 29, 255]}),
      tiles
    ],
    onError: next => {
      error = next;
    },
    onAfterRender: () => {
      frames++;
    }
  });
  const waitUntil = async (predicate: () => boolean) => {
    const deadline = performance.now() + 15000;
    while (!predicate() && performance.now() < deadline) {
      if (error) throw error;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (error) throw error;
    expect(predicate(), 'globe MVT frame ready').toBe(true);
  };
  const countWater = (pixels: Uint8Array, start: number, end: number) => {
    let count = 0;
    for (let row = start; row < end; row++) {
      for (let column = start; column < end; column++) {
        const offset = row * layout.bytesPerRow + column * 4;
        if (WATER_COLOR.every((value, channel) => Math.abs(pixels[offset + channel] - value) <= 1))
          count++;
      }
    }
    return count;
  };
  try {
    await waitUntil(() => {
      const fills = getLeafLayers(tiles).filter(layer => layer instanceof GlobeBasemapPolygonLayer);
      return (
        deck.isInitialized &&
        tiles.isLoaded &&
        fills.length === 1 &&
        fills[0].getModels().length > 0 &&
        fills[0].getModels().every(model => model.pipeline.linkStatus === 'success')
      );
    });
    for (const [longitude, latitude] of [
      [-30, 20],
      [150, -20]
    ]) {
      const previous = frames;
      deck.setProps({viewState: {longitude, latitude, zoom: 1.2}});
      deck.redraw('globe basemap fill');
      await waitUntil(() => frames > previous);
      color.readBuffer({width: SIZE, height: SIZE}, readback);
      const pixels = await readback.readAsync();
      expect(
        countWater(pixels, SIZE / 2 - 16, SIZE / 2 + 16),
        'the center contains water fill, not just outlines'
      ).toBe(32 * 32);
      expect(countWater(pixels, 0, 16), 'tiles do not fill outside the globe disc').toBe(0);
      if (error) throw error;
    }
  } finally {
    deck.finalize();
    readback.destroy();
    framebuffer.destroy();
    color.destroy();
    parent.remove();
  }
}, 60000);
