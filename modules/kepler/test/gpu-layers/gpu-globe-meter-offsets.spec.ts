// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, _GlobeView as GlobeView, type Layer} from '@deck.gl/core';
import {ScatterplotLayer} from '@deck.gl/layers';
import {GPUScatterplotLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const SIZE = 128;
// The origin sits on the antimeridian: raw offsets read as degrees would put [0, 0, 0] on the far side.
const ORIGIN: [number, number, number] = [180, 0, 0];

/**
 * Meter-offset data on a GlobeView is horizon-culled in common space: a point under the camera at
 * longitude 180 draws, the same point seen from the opposite side of the planet does not.
 */
test('resident scatterplot culls meter offsets by their projected position on a globe', async () => {
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
  const buffers: Buffer[] = [];
  let error: Error | null = null;
  let frames = 0;
  const deck = new Deck({
    device,
    parent,
    width: SIZE,
    height: SIZE,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    // Deck 10 counts earth intervals; 72 preserves deck 9's five-degree tessellation.
    views: new GlobeView({id: 'main', resolution: 72}),
    initialViewState: {longitude: 180, latitude: 0, zoom: 0},
    layers: [],
    onError: next => {
      error = next;
    },
    onAfterRender: () => {
      frames++;
    }
  });
  try {
    const render = async (layer: Layer, longitude: number) => {
      deck.setProps({
        layers: [layer],
        viewState: {longitude, latitude: 0, zoom: 0}
      } as never);
      await waitUntil(
        () =>
          Boolean(
            deck.isInitialized &&
              layer.state &&
              layer.getModels().every(model => model.pipeline.linkStatus === 'success')
          ),
        () => error
      );
      const previous = frames;
      deck.redraw('globe');
      await waitUntil(
        () => frames > previous,
        () => error
      );
      return painted(await readPixels(device, texture));
    };
    const positions = new GPUVector({
      name: 'globe-meter-offsets',
      type: 'data',
      format: 'float32x2',
      ownsData: false,
      data: [
        new GPUData({
          format: 'float32x2',
          length: 1,
          ownsBuffer: false,
          buffer: (() => {
            const buffer = device.createBuffer({
              data: new Float32Array([0, 0]),
              usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_DST
            });
            buffers.push(buffer);
            return buffer;
          })()
        })
      ]
    }) as GPUVector<'float32x2'>;
    const shared = {
      coordinateSystem: 'meter-offsets',
      coordinateOrigin: ORIGIN,
      getFillColor: [220, 60, 40, 255],
      billboard: true,
      radiusUnits: 'pixels',
      getRadius: 10
    } as const;
    const nativeVisible = await render(
      new ScatterplotLayer({
        ...shared,
        id: 'native',
        data: [[0, 0, 0]],
        getPosition: point => point as [number, number, number]
      }),
      180
    );
    expect(nativeVisible, 'native meter-offset point under the camera draws').toBeGreaterThan(20);
    const residentVisible = await render(
      new GPUScatterplotLayer({...shared, id: 'resident', getPosition: positions}),
      180
    );
    expect(residentVisible, 'resident point under the camera draws').toBeGreaterThan(20);
    const residentFar = await render(
      new GPUScatterplotLayer({...shared, id: 'resident-far', getPosition: positions}),
      0
    );
    expect(residentFar, 'resident point on the far side is culled').toBe(0);
  } finally {
    deck.finalize();
    framebuffer.destroy();
    texture.destroy();
    buffers.forEach(buffer => buffer.destroy());
    parent.remove();
  }
}, 60_000);

function painted(pixels: Uint8Array): number {
  let count = 0;
  for (let pixel = 0; pixel < SIZE * SIZE; pixel++) if (pixels[pixel * 4 + 3]) count++;
  return count;
}

async function waitUntil(predicate: () => boolean, getError: () => Error | null): Promise<void> {
  const deadline = performance.now() + 10_000;
  while (!predicate() && performance.now() < deadline) {
    if (getError()) throw getError();
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completed the requested globe frame').toBe(true);
}

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
