// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrthographicView} from '@deck.gl/core';
import {GPUMeshIconLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const VIEWPORT_SIZE = 64;

for (const positionFormat of ['float32x2', 'float32x3'] as const) {
  test(`mesh icons fill exact triangles and holes with chunked masks, sizing and picking (${positionFormat})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU device unavailable');
    const positions = makeVector(
      device,
      positionFormat,
      positionFormat === 'float32x2'
        ? [
            [-12, 0],
            [12, 0]
          ]
        : [
            [-12, 0, 0],
            [12, 0, 0]
          ]
    );
    const colors = makeVector(device, 'unorm8x4', [
      [255, 0, 0, 255],
      [0, 255, 0, 255]
    ]);
    const radii = makeVector(device, 'float32', [[10], [6]]);
    const selectionMask = makeVector(device, 'uint32', [[1], [1]]);
    // Four tessellated rectangles leave an actual central hole. The prefix tests borrowed offsets.
    const ring = makeMesh(
      device,
      [
        ...rectangle(-1, -1, 1, -0.35),
        ...rectangle(-1, 0.35, 1, 1),
        ...rectangle(-1, -0.35, -0.35, 0.35),
        ...rectangle(0.35, -0.35, 1, 0.35)
      ],
      12
    );
    const solid = makeMesh(device, rectangle(-1, -1, 1, 1));
    const borrowedBuffers = [
      ...positions.data,
      ...colors.data,
      ...radii.data,
      ...selectionMask.data,
      ring,
      solid
    ].map(data => data.buffer as Buffer);
    const readSpies = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const parent = document.createElement('div');
    document.body.append(parent);
    const texture = device.createTexture({
      width: VIEWPORT_SIZE,
      height: VIEWPORT_SIZE,
      format: 'rgba8unorm',
      usage: Texture.RENDER | Texture.COPY_SRC
    });
    const framebuffer = device.createFramebuffer({
      width: VIEWPORT_SIZE,
      height: VIEWPORT_SIZE,
      colorAttachments: [texture],
      depthStencilAttachment: 'depth24plus'
    });
    let activeLayer = new GPUMeshIconLayer({
      id: 'mesh-icons',
      iconGeometry: ring,
      getPosition: positions,
      getRadius: radii,
      getFillColor: colors,
      selectionMask,
      radiusUnits: 'pixels',
      antialiasing: false,
      coordinateSystem: 'cartesian',
      billboard: true,
      pickable: true
    });
    let deck: Deck<OrthographicView> | undefined;
    let error: Error | null = null;
    let frameCount = 0;
    let restoreShaderAssembler = () => {};
    try {
      deck = new Deck({
        device,
        parent,
        width: VIEWPORT_SIZE,
        height: VIEWPORT_SIZE,
        useDevicePixels: false,
        _framebuffer: framebuffer,
        views: new OrthographicView({id: 'main'}),
        initialViewState: {target: [0, 0], zoom: 0},
        layers: [activeLayer],
        onError: next => {
          error = next;
        },
        onAfterRender: () => {
          frameCount++;
        }
      });
      const redraw = async () => {
        await waitUntil(
          () =>
            Boolean(
              deck?.isInitialized &&
                activeLayer.state &&
                activeLayer.getModels()[0]?.pipeline.linkStatus === 'success'
            ),
          () => error
        );
        const previous = frameCount;
        deck!.redraw('mesh test');
        await waitUntil(
          () => frameCount > previous,
          () => error
        );
        return readPixels(device, texture);
      };
      let pixels = await redraw();
      const model = activeLayer.getModels()[0];
      expect(pixels[(40 * VIEWPORT_SIZE + 28) * 4]).toBe(255); // Outside a radius-10 circle.
      expect(pixels[(37 * VIEWPORT_SIZE + 49) * 4 + 1]).toBe(255);
      expect(await deck.pickObjectAsync({x: 20, y: 32})).toBeNull();
      expect(await deck.pickObjectAsync({x: 44, y: 32})).toBeNull();
      expect((await deck.pickObjectAsync({x: 28, y: 40}))?.index).toBe(0);
      expect(await deck.pickObjectAsync({x: 49, y: 37})).toHaveProperty('gpuVector', {
        rowIndex: 1,
        batchIndex: 1,
        batchRowIndex: 0
      });
      (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([0]));
      pixels = await redraw();
      expect(pixels[(37 * VIEWPORT_SIZE + 49) * 4 + 1]).toBe(0);
      expect(await deck.pickObjectAsync({x: 49, y: 37})).toBeNull();
      expect(activeLayer.getModels()[0]).toBe(model);
      activeLayer = activeLayer.clone({radiusScale: 0.5, billboard: false});
      deck.setProps({layers: [activeLayer]});
      pixels = await redraw();
      expect(activeLayer.getModels()[0]).toBe(model);
      expect(pixels[(40 * VIEWPORT_SIZE + 28) * 4]).toBe(0);
      expect((await deck.pickObjectAsync({x: 24, y: 36}))?.index).toBe(0);
      activeLayer = activeLayer.clone({iconGeometry: solid});
      deck.setProps({layers: [activeLayer]});
      pixels = await redraw();
      expect(activeLayer.getModels()[0]).not.toBe(model);
      expect(pixels[(32 * VIEWPORT_SIZE + 20) * 4]).toBe(255);
      expect((await deck.pickObjectAsync({x: 20, y: 32}))?.index).toBe(0);
      for (const spy of readSpies) expect(spy).not.toHaveBeenCalled();
      deck.finalize();
      deck = undefined;
      for (const buffer of borrowedBuffers) expect(buffer.destroyed).toBe(false);
    } finally {
      deck?.finalize();
      restoreShaderAssembler();
      readSpies.forEach(spy => spy.mockRestore());
      framebuffer.destroy();
      texture.destroy();
      borrowedBuffers.forEach(buffer => buffer.destroy());
      parent.remove();
    }
  }, 30_000);
}

function rectangle(left: number, bottom: number, right: number, top: number): number[] {
  return [
    left,
    bottom,
    0,
    right,
    bottom,
    0,
    left,
    top,
    0,
    left,
    top,
    0,
    right,
    bottom,
    0,
    right,
    top,
    0
  ];
}
function makeMesh(device: Device, values: number[], byteOffset = 0): GPUData<'float32x3'> {
  const data = new Float32Array(values.length + byteOffset / 4);
  data.set(values, byteOffset / 4);
  return new GPUData({
    format: 'float32x3',
    length: values.length / 3,
    byteOffset,
    ownsBuffer: false,
    buffer: device.createBuffer({data, usage: Buffer.VERTEX | Buffer.COPY_DST})
  });
}
function makeVector<Format extends 'float32x2' | 'float32x3' | 'float32' | 'uint32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  rows: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    format,
    ownsData: false,
    data: rows.map(
      row =>
        new GPUData({
          format,
          length: 1,
          ownsBuffer: false,
          buffer: device.createBuffer({
            data:
              format === 'uint32'
                ? new Uint32Array(row)
                : format === 'unorm8x4'
                  ? new Uint8Array(row)
                  : new Float32Array(row),
            usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}
async function waitUntil(predicate: () => boolean, getError: () => Error | null): Promise<void> {
  const deadline = performance.now() + 10000;
  while (!predicate() && performance.now() < deadline) {
    if (getError()) throw getError();
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completes the requested mesh frame').toBe(true);
}
async function readPixels(device: Device, texture: Texture): Promise<Uint8Array> {
  const layout = texture.computeMemoryLayout({width: VIEWPORT_SIZE, height: VIEWPORT_SIZE});
  const buffer = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: VIEWPORT_SIZE, height: VIEWPORT_SIZE}, buffer);
    const bytes = await buffer.readAsync();
    const pixels = new Uint8Array(VIEWPORT_SIZE * VIEWPORT_SIZE * 4);
    for (let row = 0; row < VIEWPORT_SIZE; row++)
      pixels.set(
        bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + VIEWPORT_SIZE * 4),
        row * VIEWPORT_SIZE * 4
      );
    return pixels;
  } finally {
    buffer.destroy();
  }
}
