// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  AmbientLight,
  DirectionalLight,
  Deck,
  LightingEffect,
  MapView,
  _GlobeView as GlobeView
} from '@deck.gl/core';
import {TileLayer} from '@deck.gl/geo-layers';
import {Buffer, Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {RasterWgslLayer} from '../../../examples/deck/kepler-compute/src/raster-wgsl-layer';
import {planRasterRender} from '../../../examples/deck/kepler-compute/src/raster-wgsl-shader';
import {WgslFogEffect} from '../../../examples/deck/kepler-compute/src/effects-wgsl-effects';
import {RasterWgslWrapper} from '../../../examples/deck/kepler-compute/src/raster-wgsl-wrap';

const SIZE = 128;

test('native raster image descriptors, colormaps and terrain draw in WGSL on planar and globe views', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const parent = document.createElement('div');
  document.body.append(parent);
  const canvas = new OffscreenCanvas(2, 2);
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#00ff00';
  context.fillRect(0, 0, 2, 2);
  const image = canvas.transferToImageBitmap();
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
  let error: Error | null = null;
  let frames = 0;
  // Exercise deck patch hunk 6 through a real depth-reading post effect. Without the scene
  // attachment the fog skips every frame, even when the explicit output framebuffer has depth.
  const fog = new WgslFogEffect('distance', {
    id: 'raster-depth-parity',
    constructor: {name: 'DistanceFogEffect'},
    props: {density: 0}
  });
  const deck = new Deck<MapView | GlobeView>({
    device,
    parent,
    width: SIZE,
    height: SIZE,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    views: new MapView({id: 'main'}),
    initialViewState: {longitude: 0, latitude: 0, zoom: 1},
    // Native Phong's ambient coefficient is 0.35. Three units of ambient keep the synthetic
    // green above the same ink threshold as the unlit bitmap while still exercising lighting.
    effects: [new LightingEffect({ambientLight: new AmbientLight({intensity: 3})}), fog],
    layers: [],
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
    expect(predicate(), 'raster frame completed').toBe(true);
  };
  const makeLayer = (numeric: boolean, terrain = false) =>
    new RasterWgslLayer({
      id: 'raster-parity',
      coordinateSystem: 'lnglat',
      bounds: [-45, -45, 45, 45],
      images: numeric
        ? {
            imageBands: [{data: new Float32Array(4).fill(50), width: 2, height: 2}],
            imageColormap: {data: image}
          }
        : {imageRgba: {data: image}},
      plan: planRasterRender({
        bandCombination: 'single',
        imageBandCount: numeric ? 1 : 0,
        hasColormap: numeric,
        hasRgba: !numeric,
        hasMask: false,
        minPixelValue: 0,
        maxPixelValue: 100
      })!,
      ...(terrain
        ? {
            mesh: {
              attributes: {
                POSITION: {
                  value: new Float32Array([-45, -45, 0, -45, 45, 1000, 45, 45, 1000, 45, -45, 0])
                },
                TEXCOORD_0: {value: new Float32Array([0, 1, 0, 0, 1, 0, 1, 1])}
              },
              indices: {value: new Uint16Array([0, 1, 2, 0, 2, 3])}
            }
          }
        : {})
    });
  const countGreen = async () => {
    color.readBuffer({width: SIZE, height: SIZE}, readback);
    const pixels = await readback.readAsync();
    let count = 0;
    for (let row = 0; row < SIZE; row++)
      for (let column = 0; column < SIZE; column++) {
        const offset = row * layout.bytesPerRow + column * 4;
        if (pixels[offset] < 25 && pixels[offset + 1] > 150 && pixels[offset + 2] < 25) count++;
      }
    return count;
  };
  try {
    for (const [globe, numeric, terrain] of [
      [false, false, false],
      [true, false, false],
      [true, true, false],
      [false, true, true]
    ]) {
      const layer = makeLayer(numeric, terrain);
      deck.setProps({
        views: globe ? new GlobeView({id: 'main', resolution: 72}) : new MapView({id: 'main'}),
        viewState: {longitude: 0, latitude: 0, zoom: 1},
        layers: [layer]
      });
      await waitUntil(
        () =>
          deck.isInitialized &&
          layer.isLoaded &&
          layer.getModels().every(model => model.pipeline.linkStatus === 'success')
      );
      const before = frames;
      deck.redraw('raster parity');
      await waitUntil(() => frames > before);
      expect(fog.renderedFrameCount).toBeGreaterThan(0);
      expect(fog.skippedFrameCount, 'scene renderbuffer has the hunk 6 depth attachment').toBe(0);
      expect(
        await countGreen(),
        `globe=${globe}, numeric=${numeric}, terrain=${terrain}`
      ).toBeGreaterThan(100);
      if (globe) {
        const previous = frames;
        deck.setProps({viewState: {longitude: 180, latitude: 0, zoom: 1}});
        await waitUntil(() => frames > previous);
        expect(await countGreen(), 'far-side raster is hidden').toBe(0);
      }
    }
    // Exercise native RasterMeshLayer interception with quantized-mesh-shaped data (no normals),
    // under directional lighting. Manually setting flatShading would hide the integration bug.
    const wrapper = new RasterWgslWrapper({
      ensureTileBandsResident() {},
      peekTileBandTextures: () => undefined
    } as never);
    const brightness: number[] = [];
    deck.setProps({
      effects: [
        new LightingEffect({
          ambientLight: new AmbientLight({intensity: 0}),
          sun: new DirectionalLight({direction: [1, 0, -1], intensity: 1})
        })
      ],
      views: new MapView({id: 'main'}),
      viewState: {longitude: 0, latitude: 0, zoom: 16}
    });
    for (const elevation of [0, 100]) {
      const nativeMesh = {
        attributes: {
          POSITION: {
            value: new Float32Array([
              -0.0005,
              -0.0005,
              0,
              0.0005,
              -0.0005,
              elevation,
              0.0005,
              0.0005,
              elevation,
              -0.0005,
              0.0005,
              0
            ])
          },
          TEXCOORD_0: {value: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1])}
        },
        indices: {value: new Uint16Array([0, 1, 2, 0, 2, 3])}
      };
      const nativeProps = {
        id: 'native-terrain',
        mesh: nativeMesh,
        coordinateSystem: 'lnglat',
        opacity: 1,
        // A constructed RasterMeshLayer inherits visible:true from Deck. The structural fixture
        // must carry it too: the wrapper copies visibility, and explicit undefined hides a layer.
        visible: true,
        images: {imageRgba: {data: new Uint8Array([255, 255, 255, 255]), width: 1, height: 1}}
      };
      // The native layer is intercepted before initialization, just as in the host; the fixture
      // needs only its public props/clone shape, avoiding obsolete GLSL initialization entirely.
      const tileLayer = wrapper.wrapLayer(
        new TileLayer({
          id: 'terrain-tile',
          data: '',
          renderSubLayers: () =>
            ({
              id: nativeProps.id,
              constructor: {layerName: 'RasterMeshLayer'},
              props: nativeProps,
              clone() {
                return this;
              }
            }) as never
        })
      );
      const terrainLayer = tileLayer.props.renderSubLayers({
        tile: {bbox: {west: -0.0005, south: -0.0005, east: 0.0005, north: 0.0005}}
      } as never) as RasterWgslLayer;
      expect(terrainLayer).toBeInstanceOf(RasterWgslLayer);
      expect(terrainLayer.props.flatShading).toBeUndefined();
      expect(terrainLayer.props.visible, 'native fixture retains Deck visibility').toBe(true);
      deck.setProps({layers: [terrainLayer]});
      await waitUntil(
        () =>
          terrainLayer.isLoaded &&
          terrainLayer.getModels().length > 0 &&
          terrainLayer.getModels().every(model => model.pipeline.linkStatus === 'success')
      );
      for (const model of terrainLayer.getModels()) {
        expect(model.parameters.cullMode ?? 'none', 'terrain triangles are not culled').toBe(
          'none'
        );
        for (const shader of new Set([model.pipeline.vs, model.pipeline.fs])) {
          if (!shader) continue;
          const messages = await shader.getCompilationInfo();
          expect(
            messages.filter(message => message.type === 'error'),
            `${model.id} shader diagnostics`
          ).toEqual([]);
        }
      }
      const previous = frames;
      deck.redraw('native terrain lighting');
      await waitUntil(() => frames > previous);
      for (const model of terrainLayer.getModels()) {
        expect(model.pipeline.linkStatus, `${model.id} pipeline after the lighting draw`).toBe(
          'success'
        );
      }
      color.readBuffer({width: SIZE, height: SIZE}, readback);
      const pixels = await readback.readAsync();
      let total = 0;
      for (let row = SIZE / 2 - 8; row < SIZE / 2 + 8; row++) {
        for (let column = SIZE / 2 - 8; column < SIZE / 2 + 8; column++) {
          const offset = row * layout.bytesPerRow + column * 4;
          expect(pixels[offset + 3], 'terrain covers the sampled center').toBe(255);
          total += pixels[offset];
        }
      }
      brightness.push(total / (16 * 16));
    }
    expect(
      brightness[1] - brightness[0],
      'sloping terrain faces the sun and is brighter than level terrain'
    ).toBeGreaterThan(25);
  } finally {
    deck.finalize();
    readback.destroy();
    framebuffer.destroy();
    color.destroy();
    image.close();
    parent.remove();
  }
}, 60000);
