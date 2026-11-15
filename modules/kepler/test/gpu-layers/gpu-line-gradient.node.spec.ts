// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULineLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import type {LayerContext} from '@deck.gl/core';
import type {Buffer, RenderPass} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

test('line target colors require the same physical source chunks', () => {
  const positions = makeVector('float32x2', [2, 3]);
  const targetColors = makeVector('unorm8x4', [1, 4]);
  const layer = new GPULineLayer({
    getSourcePosition: positions,
    getTargetPosition: positions,
    getTargetColor: targetColors
  });
  expect(() => layer.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow(
    'chunk 0 row counts must align'
  );
});

for (const vectorColors of [false, true]) {
  test(`line color uniforms and batches retain source identities (vectors=${vectorColors})`, () => {
    const positions = makeVector('float32x2', [2, 3]);
    const colors = makeVector('unorm8x4', [2, 3]);
    const layer = new GPULineLayer({
      getSourcePosition: positions,
      getTargetPosition: positions,
      getColor: [40, 80, 160, 200],
      getTargetColor: vectorColors ? colors : [10, 20, 30, 40]
    });
    const bytes = new Uint8Array(512);
    const bindings: Array<{offset: number; size: number}> = [];
    const drawBatches = vi.fn((_renderPass, options) => {
      options.onBatch({batchIndex: 0, rowIndexOffset: 0});
      options.onBatch({batchIndex: 1, rowIndexOffset: 2});
    });
    layer.state = {
      model: {
        drawBatches,
        setBindings: ({lineStyle}: {lineStyle: {offset: number; size: number}}) =>
          bindings.push(lineStyle)
      },
      styleBuffer: {
        byteLength: 512,
        device: {limits: {minUniformBufferOffsetAlignment: 256}},
        write: (data: Uint8Array, offset: number) => bytes.set(data, offset)
      }
    } as never;
    layer.draw({renderPass: {} as RenderPass});
    expect(drawBatches.mock.calls[0][1].vectors).toEqual({
      sourcePositions: positions,
      targetPositions: positions,
      targetColors: vectorColors ? colors : undefined
    });
    expect(bindings.map(binding => [binding.offset, binding.size])).toEqual([
      [0, 80],
      [256, 80]
    ]);
    for (const [batchIndex, binding] of bindings.entries()) {
      const uniforms = new Uint32Array(bytes.buffer, binding.offset, 20);
      expect(uniforms[10]).toBe(batchIndex === 0 ? 0 : 2);
      expect(uniforms[11]).toBe(1);
      expect(uniforms[16]).toBe(vectorColors ? 1 : 0);
      if (!vectorColors) {
        const floats = new Float32Array(bytes.buffer, binding.offset, 20);
        expect([...floats.subarray(12, 16)]).toEqual(
          [10, 20, 30, 40].map(value => Math.fround(value / 255))
        );
      }
    }
  });
}

function makeVector<Format extends 'float32x2' | 'unorm8x4'>(
  format: Format,
  lengths: number[]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    format,
    data: lengths.map(
      length =>
        new GPUData({
          format,
          length,
          buffer: {
            byteLength: length * (format === 'float32x2' ? 8 : 4),
            destroy: vi.fn()
          } as unknown as Buffer,
          ownsBuffer: false
        })
    ),
    ownsData: false
  });
}
