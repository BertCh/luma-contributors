// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AmbientLight, DirectionalLight, Deck, MapView, type Layer} from '@deck.gl/core';
import {
  GPUMeshColumnLayer,
  WGSLShadowLightingEffect
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const SIZE = 128;

function makeRegularFootprint(device: Device, radius: number, count: number): GPUData<'float32x2'> {
  const values = [0, 0];
  for (let index = 0; index < count; index++) {
    const angle = (index / count) * Math.PI * 2;
    values.push(Math.cos(angle) * radius, Math.sin(angle) * radius);
  }
  return new GPUData({
    format: 'float32x2',
    length: count,
    byteOffset: 8,
    ownsBuffer: false,
    buffer: device.createBuffer({
      data: new Float32Array(values),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    })
  });
}

function makeVector<Format extends 'float32x2' | 'float32' | 'unorm8x4' | 'uint32'>(
  device: Device,
  format: Format,
  row: number[]
): GPUVector<Format> {
  const data =
    format === 'uint32'
      ? new Uint32Array(row)
      : format === 'unorm8x4'
        ? new Uint8Array(row)
        : new Float32Array(row);
  return new GPUVector({
    type: 'data',
    format,
    ownsData: false,
    data: [
      new GPUData({
        format,
        length: 1,
        ownsBuffer: false,
        buffer: device.createBuffer({
          data,
          usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    ]
  });
}

test('WGSL shadow map darkens ground behind a tall column and stands down with no light', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const parent = document.createElement('div');
  document.body.append(parent);
  const texture = device.createTexture({
    width: SIZE,
    height: SIZE,
    format: 'rgba8unorm',
    usage: Texture.RENDER | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: SIZE,
    height: SIZE,
    colorAttachments: [texture],
    depthStencilAttachment: 'depth24plus'
  });
  let deck: Deck<MapView> | undefined;
  let error: Error | null = null;
  let frames = 0;
  let restore = () => {};
  const groundFootprint = makeRegularFootprint(device, 40, 6);
  const towerFootprint = makeRegularFootprint(device, 4, 6);
  const position = makeVector(device, 'float32x2', [0, 0]);
  const white = makeVector(device, 'unorm8x4', [220, 220, 220, 255]);
  const mask = makeVector(device, 'uint32', [1]);
  const groundHeight = makeVector(device, 'float32', [0]);
  const towerHeight = makeVector(device, 'float32', [30]);
  const coverage = makeVector(device, 'float32', [1]);
  try {
    deck = new Deck({
      device,
      parent,
      width: SIZE,
      height: SIZE,
      useDevicePixels: false,
      _framebuffer: framebuffer,
      views: new MapView({id: 'main'}),
      initialViewState: {longitude: 0, latitude: 0, zoom: 17, pitch: 45, bearing: 0},
      layers: [],
      onError: next => {
        error = next;
      },
      onAfterRender: () => {
        frames++;
      }
    });
    const makeLayers = (): Layer[] => [
      new GPUMeshColumnLayer({
        id: 'ground',
        vertices: groundFootprint,
        edgeDistance: 40 * Math.cos(Math.PI / 6),
        getPosition: position,
        getFillColor: white,
        getElevation: groundHeight,
        getCoverage: coverage,
        selectionMask: mask,
        extruded: true,
        material: {ambient: 0.6, diffuse: 0.6, shininess: 0, specularColor: [0, 0, 0]}
      }),
      new GPUMeshColumnLayer({
        id: 'tower',
        vertices: towerFootprint,
        edgeDistance: 4 * Math.cos(Math.PI / 6),
        getPosition: position,
        getFillColor: white,
        getElevation: towerHeight,
        getCoverage: coverage,
        selectionMask: mask,
        extruded: true,
        material: {ambient: 0.6, diffuse: 0.6, shininess: 0, specularColor: [0, 0, 0]}
      })
    ];
    const render = async (effect: WGSLShadowLightingEffect | null) => {
      const layers = makeLayers();
      deck!.setProps({layers, effects: effect ? [effect as never] : []});
      const deadline = performance.now() + 10000;
      while (
        performance.now() < deadline &&
        !(
          deck?.isInitialized &&
          layers.every(
            layer =>
              layer.state &&
              layer.getModels().every(model => model.pipeline.linkStatus === 'success')
          )
        )
      ) {
        if (error) throw error;
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      if (error) throw error;
      for (let frame = 0; frame < 2; frame++) {
        const previous = frames;
        deck!.redraw('shadow oracle');
        while (frames <= previous && performance.now() < deadline) {
          if (error) throw error;
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
      }
      if (error) throw error;
      return readPixels(device, texture);
    };
    const lightingProps = () => ({
      ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 1}),
      sun: new DirectionalLight({color: [255, 255, 255], intensity: 1, direction: [-1, 0, -1]})
    });
    const lit = await render(new WGSLShadowLightingEffect(lightingProps(), {lights: []}));
    const shadowEffect = new WGSLShadowLightingEffect(lightingProps(), {
      lights: [{direction: [-1, 0, -1]}],
      shadowColor: [0, 0, 0, 0.8]
    });
    shadowEffect.id = 'shadow-light';
    const shadowed = await render(shadowEffect);
    expect(shadowEffect.isShadowActive, shadowEffect.skipReason ?? '').toBe(true);
    expect(shadowEffect.participantLayerCount).toBe(2);
    let darker = 0;
    let brighter = 0;
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
      const before = lit[pixel * 4] + lit[pixel * 4 + 1] + lit[pixel * 4 + 2];
      const after = shadowed[pixel * 4] + shadowed[pixel * 4 + 1] + shadowed[pixel * 4 + 2];
      if (after < before - 60) darker++;
      else if (after > before + 30) brighter++;
    }
    const map = shadowEffect.getShadowMaps()[0];
    const mapBytes = await readTextureBytes(device, map);
    let nonWhite = 0;
    for (let index = 0; index < mapBytes.length; index += 4)
      if (mapBytes[index] !== 255 || mapBytes[index + 1] !== 255) nonWhite++;
    expect(nonWhite).toBeGreaterThan(100);
    expect(darker).toBeGreaterThan(40);
    expect(brighter).toBe(0);
    const noLight = new WGSLShadowLightingEffect(lightingProps(), {lights: []});
    noLight.id = 'no-light';
    const unshadowed = await render(noLight);
    expect(noLight.isShadowActive).toBe(false);
    let differing = 0;
    for (let index = 0; index < unshadowed.length; index++)
      if (Math.abs(unshadowed[index] - lit[index]) > 1) differing++;
    expect(differing).toBe(0);
    deck.finalize();
    deck = undefined;
  } finally {
    deck?.finalize();
    restore();
    framebuffer.destroy();
    texture.destroy();
    parent.remove();
  }
}, 60_000);

async function readPixels(device: Device, texture: Texture): Promise<Uint8Array> {
  const layout = texture.computeMemoryLayout({width: SIZE, height: SIZE});
  const buffer = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: SIZE, height: SIZE}, buffer);
    const bytes = await buffer.readAsync();
    const pixels = new Uint8Array(SIZE * SIZE * 4);
    for (let row = 0; row < SIZE; row++)
      pixels.set(
        bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + SIZE * 4),
        row * SIZE * 4
      );
    return pixels;
  } finally {
    buffer.destroy();
  }
}

async function readTextureBytes(device: Device, texture: Texture): Promise<Uint8Array> {
  const layout = texture.computeMemoryLayout({width: texture.width, height: texture.height});
  const buffer = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: texture.width, height: texture.height}, buffer);
    return new Uint8Array(await buffer.readAsync());
  } finally {
    buffer.destroy();
  }
}
