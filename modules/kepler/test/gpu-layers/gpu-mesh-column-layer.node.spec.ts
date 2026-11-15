// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUMeshColumnLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import type {LayerContext, PickingInfo} from '@deck.gl/core';
import type {Buffer, RenderPass} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

function chunk<Format extends 'float32x2' | 'uint32'>(
  format: Format,
  length: number
): GPUData<Format> {
  return new GPUData({
    format,
    length,
    ownsBuffer: false,
    buffer: {byteLength: length * 8, destroy: vi.fn(), readAsync: vi.fn()} as unknown as Buffer
  });
}
function fixture() {
  const vertices = chunk('float32x2', 6);
  const positions = new GPUVector({
    type: 'data',
    format: 'float32x2',
    data: [chunk('float32x2', 1), chunk('float32x2', 2)],
    ownsData: false
  });
  return {vertices, positions};
}

test('custom columns retain borrowed footprint and source chunk provenance', () => {
  const {vertices, positions} = fixture();
  const layer = new GPUMeshColumnLayer({vertices, edgeDistance: 8, getPosition: positions});
  expect(layer.getPickingInfo({info: {index: 2} as PickingInfo})).toHaveProperty('gpuVector', {
    rowIndex: 2,
    batchIndex: 1,
    batchRowIndex: 1
  });
  expect(vertices.buffer.readAsync).not.toHaveBeenCalled();
  expect(vertices.buffer.destroy).not.toHaveBeenCalled();
});

test('custom columns reserve separate batch uniforms for fill and native outline opacity', () => {
  const {vertices, positions} = fixture();
  const layer = new GPUMeshColumnLayer({
    vertices,
    edgeDistance: 8,
    getPosition: positions,
    extruded: false,
    stroked: true,
    strokeOpacity: 0.25,
    opacity: 0.8,
    coverage: 0.75,
    getCoverage: 0.5,
    getLineWidth: 3,
    lineWidthUnits: 'pixels'
  });
  const writes: Array<{offset: number; bytes: Uint8Array}> = [];
  const counts: number[] = [];
  const model = {
    setBindings: vi.fn(),
    setVertexCount: (count: number) => counts.push(count),
    drawBatches: vi.fn((_pass, options) => {
      options.onBatch({batchIndex: 0, rowIndexOffset: 0});
      options.onBatch({batchIndex: 1, rowIndexOffset: 1});
    })
  };
  layer.state = {
    model,
    defaults: [],
    styleBuffer: {
      byteLength: 1024,
      device: {limits: {minUniformBufferOffsetAlignment: 256}},
      write: (bytes: Uint8Array, offset: number) => writes.push({offset, bytes: bytes.slice()})
    }
  } as never;
  layer.context = {device: {}} as never;
  layer.draw({renderPass: {} as RenderPass});
  expect(counts).toEqual([22, 14]);
  expect(writes.map(write => write.offset)).toEqual([0, 256, 512, 768]);
  expect(writes.map(write => new Uint32Array(write.bytes.buffer)[24])).toEqual([0, 1, 0, 1]);
  expect(writes.map(write => new Uint32Array(write.bytes.buffer)[27])).toEqual([0, 0, 1, 1]);
  const floats = new Float32Array(writes[2].bytes.buffer);
  expect(floats[29]).toBeCloseTo(Math.pow(0.25, 1 / 2.2));
  expect(floats[9]).toBe(0.5);
  expect(floats[11]).toBe(0.75);
  expect(model.drawBatches.mock.calls[0][1].vectors.positions).toBe(positions);
});

test('custom columns reject invalid footprint metadata and mask chunk layouts before allocation', () => {
  const {vertices, positions} = fixture();
  const invalid = new GPUMeshColumnLayer({vertices, edgeDistance: 0, getPosition: positions});
  expect(() => invalid.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow();
  const selectionMask = new GPUVector({
    type: 'data',
    format: 'uint32',
    data: [chunk('uint32', 3)],
    ownsData: false
  });
  const mismatch = new GPUMeshColumnLayer({
    vertices,
    edgeDistance: 8,
    getPosition: positions,
    selectionMask
  });
  expect(() => mismatch.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow(
    /chunk counts/
  );
});
