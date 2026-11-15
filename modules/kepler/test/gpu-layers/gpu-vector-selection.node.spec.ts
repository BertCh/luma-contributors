// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUArcLayer,
  GPULineLayer,
  GPUScatterplotLayer
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import type {LayerContext, PickingInfo} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {describe, expect, test, vi} from 'vitest';
import {
  GPU_VECTOR_MAXIMUM_PICKABLE_ROWS,
  getGPUVectorLayerBatches,
  getGPUVectorPickingProvenance,
  validateGPUVectorPickingCapacity
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-vector-layer-utils';

describe('resident GPU layer selection', () => {
  test('accepts aligned masks without consuming or destroying their buffers', () => {
    const positions = makeVector('float32x2', [2, 3]);
    const selectionMask = makeVector('uint32', [2, 3]);
    const batches = getGPUVectorLayerBatches(
      'selection',
      {positions, selectionMask},
      {positions: ['float32x2'], selectionMask: ['uint32']}
    );
    expect(batches.map(batch => [batch.rowIndexOffset, batch.rowCount])).toEqual([
      [0, 2],
      [2, 3]
    ]);
    expect(batches[1].data['selectionMask']).toBe(selectionMask.data[1]);
    for (const chunk of selectionMask.data) {
      expect(chunk.buffer.readAsync).not.toHaveBeenCalled();
      expect(chunk.buffer.destroy).not.toHaveBeenCalled();
    }
  });

  test('rejects masks whose chunks differ from the source features', () => {
    const positions = makeVector('float32x2', [2, 3]);
    const selectionMask = makeVector('uint32', [1, 4]);
    for (const layer of [
      new GPUScatterplotLayer({getPosition: positions, selectionMask}),
      new GPULineLayer({getSourcePosition: positions, getTargetPosition: positions, selectionMask}),
      new GPUArcLayer({getSourcePosition: positions, getTargetPosition: positions, selectionMask})
    ]) {
      expect(() => layer.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow(
        'chunk 0 row counts must align'
      );
    }
  });

  test('preserves original row identity, including the second physical chunk', () => {
    const positions = makeVector('float32x2', [2, 3]);
    const selectionMask = makeVector('uint32', [2, 3]);
    for (const layer of [
      new GPUScatterplotLayer({getPosition: positions, selectionMask}),
      new GPULineLayer({getSourcePosition: positions, getTargetPosition: positions, selectionMask}),
      new GPUArcLayer({getSourcePosition: positions, getTargetPosition: positions, selectionMask})
    ]) {
      const info = layer.getPickingInfo({info: {index: 4} as PickingInfo} as never);
      expect(info.index).toBe(4);
      expect(info).toHaveProperty('gpuVector', {rowIndex: 4, batchIndex: 1, batchRowIndex: 2});
    }
    expect(getGPUVectorPickingProvenance(positions, -1)).toEqual({
      rowIndex: -1,
      batchIndex: -1,
      batchRowIndex: -1
    });
    expect(getGPUVectorPickingProvenance(positions, 5).batchIndex).toBe(-1);
  });

  test('rejects RGB24 picking overflow before allocating GPU resources', () => {
    expect(() =>
      validateGPUVectorPickingCapacity(GPU_VECTOR_MAXIMUM_PICKABLE_ROWS, true)
    ).not.toThrow();
    expect(() =>
      validateGPUVectorPickingCapacity(GPU_VECTOR_MAXIMUM_PICKABLE_ROWS + 1, true)
    ).toThrow();
    expect(() =>
      validateGPUVectorPickingCapacity(GPU_VECTOR_MAXIMUM_PICKABLE_ROWS + 1, false)
    ).not.toThrow();
    const positions = makeVector('float32x2', [GPU_VECTOR_MAXIMUM_PICKABLE_ROWS, 1]);
    for (const layer of [
      new GPUScatterplotLayer({getPosition: positions, pickable: true}),
      new GPULineLayer({
        getSourcePosition: positions,
        getTargetPosition: positions,
        pickable: true
      }),
      new GPUArcLayer({getSourcePosition: positions, getTargetPosition: positions, pickable: true})
    ]) {
      expect(() => layer.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow();
    }
  });
});

function makeVector<Format extends 'float32x2' | 'uint32'>(
  format: Format,
  lengths: number[]
): GPUVector<Format> {
  const byteStride = format === 'float32x2' ? 8 : 4;
  return new GPUVector({
    type: 'data',
    name: 'test-selection',
    format,
    data: lengths.map(
      length =>
        new GPUData({
          format,
          length,
          byteStride,
          buffer: {
            byteLength: length * byteStride,
            destroy: vi.fn(),
            readAsync: vi.fn()
          } as unknown as Buffer,
          ownsBuffer: false
        })
    ),
    ownsData: false
  });
}
