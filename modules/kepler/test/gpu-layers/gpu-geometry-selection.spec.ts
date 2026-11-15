// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrthographicView} from '@deck.gl/core';
import {
  GPUPathLayer,
  GPUSolidPolygonLayer
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const VIEWPORT_SIZE = 64;

for (const kind of ['path', 'polygon', 'trips'] as const) {
  test(`GPU ${kind} selection updates rendering and picking without source readback`, async () => {
    const device = await getWebGPUTestDevice();
    expect(device, 'a real WebGPU device is required').toBeTruthy();
    if (!device) throw new Error('WebGPU device unavailable');
    const geometry = makeGeometry(device);
    const selectionMask = makeVector(device, 'uint32', [[7], [0]]);
    const replacementMask = makeVector(device, 'uint32', [[1], [0]]);
    const borrowedBuffers = [
      ...geometry.paths.data,
      ...geometry.positions.data,
      ...geometry.rowIndices.data,
      ...geometry.indices.data,
      ...geometry.timestamps.data,
      ...selectionMask.data,
      ...replacementMask.data
    ].map(chunk => chunk.buffer as Buffer);
    const readSpies = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const props = {
      id: `selected-${kind}`,
      coordinateSystem: 'cartesian',
      selectionMask,
      pickable: true
    };
    let layer: GPUPathLayer | GPUSolidPolygonLayer =
      kind === 'polygon'
        ? new GPUSolidPolygonLayer({
            ...props,
            positions: geometry.positions,
            rowIndices: geometry.rowIndices,
            indices: geometry.indices,
            getFillColor: [255, 0, 0, 255]
          })
        : new GPUPathLayer({
            ...props,
            getPath: geometry.paths,
            getWidth: 8,
            getColor: [255, 0, 0, 255],
            ...(kind === 'trips'
              ? {getTimestamps: geometry.timestamps, currentTime: 10, trailLength: 20}
              : {})
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
      for (const [step, selectedIndices] of [[0], [1], [], [0], [0, 1]].entries()) {
        if (step === 1) {
          (selectionMask.data[0].buffer as Buffer).write(new Uint32Array([0]));
          (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([99]));
        }
        if (step === 2) {
          (selectionMask.data[1].buffer as Buffer).write(new Uint32Array([0]));
        }
        if (step === 3 || step === 4) {
          layer = layer.clone({selectionMask: step === 3 ? replacementMask : undefined});
          deck.setProps({layers: [layer]});
          await waitUntil(
            () => layer.getModels()[0] === model,
            () => deckError
          );
        }
        const previousFrames = renderedFrames;
        deck.redraw('resident mask changed');
        await waitUntil(
          () => renderedFrames > previousFrames,
          () => deckError
        );
        const pixels = await readPixels(device, colorTexture);
        for (const [rowIndex, horizontal] of [20, 44].entries()) {
          const offset = (32 * VIEWPORT_SIZE + horizontal) * 4;
          expect(pixels[offset], `step ${step} row ${rowIndex} red channel`).toBe(
            selectedIndices.includes(rowIndex) ? 255 : 0
          );
          const picked = await deck.pickObjectAsync({x: horizontal, y: 32});
          if (selectedIndices.includes(rowIndex)) {
            expect(picked?.index).toBe(rowIndex);
            expect(picked).toHaveProperty('gpuVector', {
              rowIndex,
              batchIndex: rowIndex,
              batchRowIndex: 0
            });
          } else {
            expect(picked).toBeNull();
          }
        }
        expect(layer.getModels()[0], 'mask changes reuse the model').toBe(model);
        expect(layer.getModels()).toHaveLength(2);
      }
      const unmaskedLayer = layer.clone({selectionMask: undefined});
      deck.setProps({layers: [unmaskedLayer]});
      await waitUntil(
        () =>
          Boolean(
            unmaskedLayer.state && unmaskedLayer.getModels()[0]?.pipeline.linkStatus === 'success'
          ),
        () => deckError
      );
      const previousFrames = renderedFrames;
      deck.redraw('selection removed');
      await waitUntil(
        () => renderedFrames > previousFrames,
        () => deckError
      );
      const unmaskedPixels = await readPixels(device, colorTexture);
      for (const [rowIndex, horizontal] of [20, 44].entries()) {
        expect(unmaskedPixels[(32 * VIEWPORT_SIZE + horizontal) * 4]).toBe(255);
        expect((await deck.pickObjectAsync({x: horizontal, y: 32}))?.index).toBe(rowIndex);
      }
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

function makeVector<Format extends 'float32x2' | 'uint32'>(
  device: Device,
  format: Format,
  rows: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'test-selection',
    format,
    ownsData: false,
    data: rows.map(
      row =>
        new GPUData({
          format,
          length: 1,
          ownsBuffer: false,
          buffer: device.createBuffer({
            data: format === 'uint32' ? new Uint32Array(row) : new Float32Array(row),
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

function makeGeometry(device: Device) {
  function makeList<Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
    format: Format,
    values: number[][],
    valueLengths: number[]
  ): GPUVector<VertexList<Format>> {
    return new GPUVector({
      type: 'data',
      name: format,
      format: `vertex-list<${format}>`,
      ownsData: false,
      data: values.map(
        (chunk, chunkIndex) =>
          new GPUData({
            format: `vertex-list<${format}>`,
            length: 1,
            valueLength: valueLengths[chunkIndex],
            valueOffsets: new Int32Array([0, valueLengths[chunkIndex]]),
            ownsBuffer: false,
            buffer: device.createBuffer({
              data: format === 'uint32' ? new Uint32Array(chunk) : new Float32Array(chunk),
              usage:
                Buffer.STORAGE |
                Buffer.VERTEX |
                Buffer.COPY_DST |
                (format === 'uint32' ? Buffer.INDEX : 0)
            })
          })
      )
    });
  }
  return {
    paths: makeList(
      'float32x2',
      [
        [-12, -6, -12, 6],
        [12, -6, 12, 6]
      ],
      [2, 2]
    ),
    timestamps: makeList(
      'float32',
      [
        [0, 10],
        [0, 10]
      ],
      [2, 2]
    ),
    positions: makeList(
      'float32x4',
      [
        [-17, -6, 0, 1, -7, -6, 0, 1, -12, 7, 0, 1],
        [7, -6, 0, 1, 17, -6, 0, 1, 12, 7, 0, 1]
      ],
      [3, 3]
    ),
    rowIndices: makeList(
      'uint32',
      [
        [0, 0, 0],
        [1, 1, 1]
      ],
      [3, 3]
    ),
    indices: makeList(
      'uint32',
      [
        [0, 1, 2],
        [0, 1, 2]
      ],
      [3, 3]
    )
  };
}
