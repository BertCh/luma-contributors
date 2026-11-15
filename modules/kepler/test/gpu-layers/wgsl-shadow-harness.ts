// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AmbientLight, DirectionalLight, Deck, MapView, type Layer} from '@deck.gl/core';
import {WGSLShadowLightingEffect} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';

export const SHADOW_SIZE = 160;

export function makeShadowFootprint(
  device: Device,
  radius: number,
  count = 6
): GPUData<'float32x2'> {
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

export function makeShadowVector<Format extends 'float32x2' | 'float32' | 'unorm8x4' | 'uint32'>(
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
  const perRow = format === 'float32x2' ? 2 : format === 'unorm8x4' ? 4 : 1;
  return new GPUVector({
    type: 'data',
    format,
    ownsData: false,
    data: [
      new GPUData({
        format,
        length: row.length / perRow,
        ownsBuffer: false,
        buffer: device.createBuffer({
          data,
          usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    ]
  });
}

export type ShadowComparison = {
  darker: number;
  brighter: number;
  participants: number;
  shadowMaps: number;
  skipReason: string | null;
  active: boolean;
  /** Pixels that differ between the lit frame and a frame from a no-light effect. */
  noLightDifference: number;
  /** Texels of the first light's shadow map that hold geometry depth (not the cleared white). */
  shadowMapCoverage: number;
  lit: Uint8Array;
  shadowed: Uint8Array;
};

/** Renders layers with and without a shadow-casting light and compares the frames. */
export async function compareShadowFrames(
  device: Device,
  makeLayers: () => Layer[],
  options: {
    viewState?: Record<string, unknown>;
    lightDirection?: [number, number, number];
  } = {}
): Promise<ShadowComparison> {
  const size = SHADOW_SIZE;
  const direction = options.lightDirection ?? [-1, 0, -1];
  const parent = document.createElement('div');
  document.body.append(parent);
  const texture = device.createTexture({
    width: size,
    height: size,
    format: 'rgba8unorm',
    usage: Texture.RENDER | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: size,
    height: size,
    colorAttachments: [texture],
    depthStencilAttachment: 'depth24plus'
  });
  let error: Error | null = null;
  let frames = 0;
  let restore = () => {};
  const deck = new Deck({
    device,
    parent,
    width: size,
    height: size,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    views: new MapView({id: 'main'}),
    initialViewState: options.viewState ?? {
      longitude: 0,
      latitude: 0,
      zoom: 17,
      pitch: 45,
      bearing: 0
    },
    layers: [],
    onError: next => {
      error = next;
    },
    onAfterRender: () => {
      frames++;
    }
  });
  const lightingProps = () => ({
    ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 1}),
    sun: new DirectionalLight({color: [255, 255, 255], intensity: 1, direction})
  });
  const render = async (effect: WGSLShadowLightingEffect): Promise<Uint8Array> => {
    const layers = makeLayers();
    deck.setProps({layers, effects: [effect as never]});
    const deadline = performance.now() + 15000;
    while (
      performance.now() < deadline &&
      !(
        deck.isInitialized &&
        layers.every(
          layer =>
            layer.state && layer.getModels().every(model => model.pipeline.linkStatus === 'success')
        )
      )
    ) {
      if (error) throw error;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (error) throw error;
    for (let frame = 0; frame < 2; frame++) {
      const previous = frames;
      deck.redraw('shadow harness');
      while (frames <= previous && performance.now() < deadline) {
        if (error) throw error;
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
    }
    if (error) throw error;
    const layout = texture.computeMemoryLayout({width: size, height: size});
    const buffer = device.createBuffer({
      byteLength: layout.byteLength,
      usage: Buffer.COPY_DST | Buffer.MAP_READ
    });
    try {
      texture.readBuffer({width: size, height: size}, buffer);
      const bytes = await buffer.readAsync();
      const pixels = new Uint8Array(size * size * 4);
      for (let row = 0; row < size; row++)
        pixels.set(
          bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + size * 4),
          row * size * 4
        );
      return pixels;
    } finally {
      buffer.destroy();
    }
  };
  try {
    const litEffect = new WGSLShadowLightingEffect(lightingProps(), {lights: []});
    litEffect.id = 'lit-light';
    const lit = await render(litEffect);
    const effect = new WGSLShadowLightingEffect(lightingProps(), {
      lights: [{direction}],
      shadowColor: [0, 0, 0, 0.8]
    });
    effect.id = 'shadow-light';
    const shadowed = await render(effect);
    const snapshot = {
      active: effect.isShadowActive,
      skipReason: effect.skipReason,
      participants: effect.participantLayerCount,
      shadowMaps: effect.getShadowMaps().length
    };
    const shadowMap = effect.getShadowMaps()[0];
    let shadowMapCoverage = 0;
    if (shadowMap) {
      const mapLayout = shadowMap.computeMemoryLayout({
        width: shadowMap.width,
        height: shadowMap.height
      });
      const mapBuffer = device.createBuffer({
        byteLength: mapLayout.byteLength,
        usage: Buffer.COPY_DST | Buffer.MAP_READ
      });
      shadowMap.readBuffer({width: shadowMap.width, height: shadowMap.height}, mapBuffer);
      const mapBytes = new Uint8Array(await mapBuffer.readAsync());
      for (let row = 0; row < shadowMap.height; row++)
        for (let column = 0; column < shadowMap.width; column++) {
          const offset = row * mapLayout.bytesPerRow + column * 4;
          if (mapBytes[offset] !== 255 || mapBytes[offset + 1] !== 255) shadowMapCoverage++;
        }
      mapBuffer.destroy();
    }
    let darker = 0;
    let brighter = 0;
    for (let pixel = 0; pixel < size * size; pixel++) {
      const before = lit[pixel * 4] + lit[pixel * 4 + 1] + lit[pixel * 4 + 2];
      const after = shadowed[pixel * 4] + shadowed[pixel * 4 + 1] + shadowed[pixel * 4 + 2];
      if (after < before - 60) darker++;
      else if (after > before + 30) brighter++;
    }
    const noLightEffect = new WGSLShadowLightingEffect(lightingProps(), {lights: []});
    noLightEffect.id = 'no-light';
    const unshadowed = await render(noLightEffect);
    let noLightDifference = 0;
    for (let index = 0; index < unshadowed.length; index++)
      if (Math.abs(unshadowed[index] - lit[index]) > 1) noLightDifference++;
    return {...snapshot, shadowMapCoverage, darker, brighter, noLightDifference, lit, shadowed};
  } finally {
    deck.finalize();
    restore();
    framebuffer.destroy();
    texture.destroy();
    parent.remove();
  }
}
