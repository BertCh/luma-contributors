// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrthographicView} from '@deck.gl/core';
import {GPUScatterplotLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const VIEWPORT_SIZE = 64;

for (const vectorStyles of [false, true]) {
  test(`scatterplot centered outlines preserve chunks, coverage and native picking (vectors=${vectorStyles})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU device unavailable');
    const positions = makeVector(device, 'float32x2', [
      [-12, 0],
      [12, 0]
    ]);
    const colors = makeVector(device, 'unorm8x4', [
      [255, 0, 0, 255],
      [0, 255, 0, 255]
    ]);
    const widths = makeVector(device, 'float32', [[4], [6]]);
    const selectionMask = makeVector(device, 'uint32', [[1], [1]]);
    const borrowedBuffers = [
      ...positions.data,
      ...colors.data,
      ...widths.data,
      ...selectionMask.data
    ].map(chunk => chunk.buffer as Buffer);
    const readSpies = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const layer = new GPUScatterplotLayer({
      id: 'outlined-points',
      coordinateSystem: 'cartesian',
      getPosition: positions,
      getRadius: 7,
      radiusUnits: 'pixels',
      getFillColor: [0, 0, 255, 255],
      stroked: true,
      filled: false,
      getLineColor: vectorStyles ? colors : [255, 0, 0, 255],
      getLineWidth: vectorStyles ? widths : 4,
      lineWidthUnits: 'pixels',
      antialiasing: false,
      selectionMask,
      pickable: true
    });
    const parent = document.createElement('div');
    document.body.append(parent);
    const colorTexture = device.createTexture({
      width: VIEWPORT_SIZE,
      height: VIEWPORT_SIZE,
      format: 'rgba8unorm',
      usage: Texture.RENDER | Texture.COPY_SRC
    });
    const framebuffer = device.createFramebuffer({
      width: VIEWPORT_SIZE,
      height: VIEWPORT_SIZE,
      colorAttachments: [colorTexture],
      depthStencilAttachment: 'depth24plus'
    });
    let deck: Deck<OrthographicView> | null = null;
    let deckError: Error | null = null;
    let restoreShaderAssembler = () => {};
    let renderedFrames = 0;
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
        layers: [layer],
        onError: error => {
          deckError = error;
        },
        onAfterRender: () => {
          renderedFrames++;
        }
      });
      await waitUntil(
        () =>
          Boolean(
            deck?.isInitialized &&
              layer.state &&
              layer.getModels()[0]?.pipeline.linkStatus === 'success'
          ),
        () => deckError
      );
      const model = layer.getModels()[0];
      let activeLayer = layer;
      const redraw = async () => {
        const previousFrames = renderedFrames;
        deck!.redraw('outline changed');
        await waitUntil(
          () => renderedFrames > previousFrames,
          () => deckError
        );
        return readPixels(device, colorTexture);
      };
      let pixels = await redraw();
      for (const [rowIndex, center] of [20, 44].entries()) {
        const channel = vectorStyles ? rowIndex : 0;
        expect(pixels[(32 * VIEWPORT_SIZE + center) * 4 + channel]).toBe(0);
        expect(await deck.pickObjectAsync({x: center, y: 32})).toBeNull();
        // Radius=7, outline centered: distance=8 is inside the outer half of the stroke.
        expect(pixels[(32 * VIEWPORT_SIZE + center + 8) * 4 + channel]).toBe(255);
        const picked = await deck.pickObjectAsync({x: center + 8, y: 32});
        expect(picked?.index).toBe(rowIndex);
        expect(picked).toHaveProperty('gpuVector', {
          rowIndex,
          batchIndex: rowIndex,
          batchRowIndex: 0
        });
        expect(await deck.pickObjectAsync({x: center + 11, y: 32})).toBeNull();
      }
      if (vectorStyles) {
        (widths.data[0].buffer as Buffer).write(new Float32Array([12]));
        pixels = await redraw();
        expect(pixels[(32 * VIEWPORT_SIZE + 22) * 4]).toBe(255);
        expect((await deck.pickObjectAsync({x: 22, y: 32}))?.index).toBe(0);
      }
      for (const [filled, stroked] of [
        [true, true],
        [true, false],
        [false, false]
      ]) {
        activeLayer = activeLayer.clone({filled, stroked});
        deck.setProps({layers: [activeLayer]});
        await waitUntil(
          () => Boolean(activeLayer.state),
          () => deckError
        );
        pixels = await redraw();
        expect(activeLayer.getModels()[0], 'style toggles retain the model').toBe(model);
        expect(pixels[(32 * VIEWPORT_SIZE + 44) * 4 + 2]).toBe(filled ? 255 : 0);
        const picked = await deck.pickObjectAsync({x: 44, y: 32});
        if (filled) expect(picked?.index).toBe(1);
        else expect(picked).toBeNull();
        if (!stroked) {
          expect(await deck.pickObjectAsync({x: 52, y: 32})).toBeNull();
        }
      }
      activeLayer = activeLayer.clone({filled: false, stroked: true});
      (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([0]));
      deck.setProps({layers: [activeLayer]});
      await waitUntil(
        () => Boolean(activeLayer.state),
        () => deckError
      );
      pixels = await redraw();
      expect([
        ...pixels.subarray((32 * VIEWPORT_SIZE + 52) * 4, (32 * VIEWPORT_SIZE + 52) * 4 + 3)
      ]).toEqual([0, 0, 0]);
      expect(await deck.pickObjectAsync({x: 52, y: 32})).toBeNull();
      for (const spy of readSpies) expect(spy).not.toHaveBeenCalled();
      deck.finalize();
      deck = null;
      for (const buffer of borrowedBuffers) expect(buffer.destroyed).toBe(false);
    } finally {
      deck?.finalize();
      restoreShaderAssembler();
      readSpies.forEach(spy => spy.mockRestore());
      framebuffer.destroy();
      colorTexture.destroy();
      borrowedBuffers.forEach(buffer => buffer.destroy());
      parent.remove();
    }
  }, 30_000);
}

function makeVector<Format extends 'float32x2' | 'uint32' | 'float32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  rows: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'outline-style',
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
  const deadline = performance.now() + 10_000;
  while (!predicate() && performance.now() < deadline) {
    const error = getError();
    if (error) throw error;
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completed the requested render').toBe(true);
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
