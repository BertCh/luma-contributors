// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrthographicView} from '@deck.gl/core';
import {GPULineLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const VIEWPORT_SIZE = 64;

for (const vectorStyles of [false, true]) {
  test(`line gradients interpolate RGBA, retain chunks and native picking (vectors=${vectorStyles})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU device unavailable');
    const sourceCoordinates = [
      [-24, -12],
      [24, 12]
    ];
    const targetCoordinates = [
      [24, -12],
      [-24, 12]
    ];
    const sourceValues = [
      [255, 0, 0, 255],
      [0, 255, 0, 255]
    ];
    const targetValues = [
      [0, 0, 255, 128],
      [255, 0, 0, 255]
    ];
    const positions = makeVector(device, 'float32x2', sourceCoordinates);
    const targets = makeVector(device, 'float32x2', targetCoordinates);
    const sourceColors = makeVector(device, 'unorm8x4', sourceValues);
    const targetColors = makeVector(device, 'unorm8x4', targetValues);
    const selectionMask = makeVector(device, 'uint32', [[1], [1]]);
    const borrowedBuffers = [
      ...positions.data,
      ...targets.data,
      ...sourceColors.data,
      ...targetColors.data,
      ...selectionMask.data
    ].map(chunk => chunk.buffer as Buffer);
    const readSpies = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const layer = new GPULineLayer({
      id: 'gradient-lines',
      coordinateSystem: 'cartesian',
      getSourcePosition: positions,
      getTargetPosition: targets,
      getColor: vectorStyles ? sourceColors : [255, 0, 0, 255],
      getTargetColor: vectorStyles ? targetColors : [0, 0, 255, 128],
      getWidth: 6,
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
        deck!.redraw('gradient changed');
        await waitUntil(
          () => renderedFrames > previousFrames,
          () => deckError
        );
        return readPixels(device, colorTexture);
      };
      const sample = (rowIndex: number, progress: number) => {
        const viewport = deck!.getViewports()[0];
        const sourcePosition = viewport.project(sourceCoordinates[rowIndex]);
        const targetPosition = viewport.project(targetCoordinates[rowIndex]);
        const horizontal = Math.floor(
          sourcePosition[0] + (targetPosition[0] - sourcePosition[0]) * progress
        );
        const vertical = Math.floor(sourcePosition[1]);
        return {
          horizontal,
          vertical,
          progress: (horizontal + 0.5 - sourcePosition[0]) / (targetPosition[0] - sourcePosition[0])
        };
      };
      const expectGradient = async (
        pixels: Uint8Array,
        targetsByRow: number[][],
        selected = [true, true]
      ) => {
        for (const rowIndex of [0, 1]) {
          for (const progress of [0.25, 0.5, 0.75]) {
            const point = sample(rowIndex, progress);
            const offset = (point.vertical * VIEWPORT_SIZE + point.horizontal) * 4;
            const sourceColor = sourceValues[vectorStyles ? rowIndex : 0];
            const targetColor = targetsByRow[rowIndex];
            const alpha = sourceColor[3] + (targetColor[3] - sourceColor[3]) * point.progress;
            for (const channel of [0, 1, 2]) {
              const straightColor =
                sourceColor[channel] +
                (targetColor[channel] - sourceColor[channel]) * point.progress;
              const expected = selected[rowIndex] ? Math.round((straightColor * alpha) / 255) : 0;
              expect(
                Math.abs(pixels[offset + channel] - expected),
                `row ${rowIndex}, progress ${progress}, channel ${channel}`
              ).toBeLessThanOrEqual(2);
            }
            const picked = await deck!.pickObjectAsync({x: point.horizontal, y: point.vertical});
            if (selected[rowIndex]) {
              expect(picked?.index).toBe(rowIndex);
              expect(picked).toHaveProperty('gpuVector', {
                rowIndex,
                batchIndex: rowIndex,
                batchRowIndex: 0
              });
            } else expect(picked).toBeNull();
          }
        }
      };
      await expectGradient(
        await redraw(),
        vectorStyles ? targetValues : [targetValues[0], targetValues[0]]
      );
      if (vectorStyles) {
        targetValues[0] = [0, 255, 0, 255];
        (targetColors.data[0].buffer as Buffer).write(new Uint8Array(targetValues[0]));
      } else {
        activeLayer = activeLayer.clone({getTargetColor: [0, 255, 0, 255]});
        deck.setProps({layers: [activeLayer]});
        await waitUntil(
          () => Boolean(activeLayer.state),
          () => deckError
        );
      }
      const changedTargets = vectorStyles
        ? targetValues
        : [
            [0, 255, 0, 255],
            [0, 255, 0, 255]
          ];
      await expectGradient(await redraw(), changedTargets);
      expect(activeLayer.getModels()[0], 'color changes reuse the resident model').toBe(model);
      (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([0]));
      await expectGradient(await redraw(), changedTargets, [true, false]);
      (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([1]));
      activeLayer = activeLayer.clone({getTargetColor: undefined});
      deck.setProps({layers: [activeLayer]});
      await waitUntil(
        () =>
          Boolean(
            activeLayer.state && activeLayer.getModels()[0]?.pipeline.linkStatus === 'success'
          ),
        () => deckError
      );
      // Omitted target uses each row's source, including source vectors, rather than a global constant.
      await expectGradient(
        await redraw(),
        vectorStyles ? sourceValues : [sourceValues[0], sourceValues[0]]
      );
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
