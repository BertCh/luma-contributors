// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerContext, PickingInfo} from '@deck.gl/core';
import {GPUMeshIconLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import type {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

function chunk<Format extends 'float32x2' | 'float32x3' | 'uint32'>(
  format: Format,
  length: number
): GPUData<Format> {
  return new GPUData({
    format,
    length,
    ownsBuffer: false,
    buffer: {
      byteLength: length * 16,
      destroy: vi.fn(),
      readAsync: vi.fn()
    } as unknown as Buffer
  });
}

test('mesh icons retain chunk picking provenance and borrow mesh storage', () => {
  const iconGeometry = chunk('float32x3', 6);
  const positions = new GPUVector({
    type: 'data',
    format: 'float32x2',
    data: [chunk('float32x2', 2), chunk('float32x2', 3)],
    ownsData: false
  });
  const layer = new GPUMeshIconLayer({iconGeometry, getPosition: positions});
  const pickingInfo = layer.getPickingInfo({info: {index: 4} as PickingInfo} as never);
  expect(pickingInfo).toHaveProperty('gpuVector', {rowIndex: 4, batchIndex: 1, batchRowIndex: 2});
  expect(layer.props.iconGeometry).toBe(iconGeometry);
  expect(iconGeometry.buffer.readAsync).not.toHaveBeenCalled();
  expect(iconGeometry.buffer.destroy).not.toHaveBeenCalled();
});

test('mesh icons reject incomplete triangles and source masks with incompatible chunks before allocation', () => {
  const positions = new GPUVector({
    type: 'data',
    format: 'float32x3',
    data: [chunk('float32x3', 2)],
    ownsData: false
  });
  const invalidMesh = new GPUMeshIconLayer({
    iconGeometry: chunk('float32x3', 4),
    getPosition: positions
  });
  expect(() => invalidMesh.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow();
  const selectionMask = new GPUVector({
    type: 'data',
    format: 'uint32',
    data: [chunk('uint32', 1)],
    ownsData: false
  });
  const invalidMask = new GPUMeshIconLayer({
    iconGeometry: chunk('float32x3', 6),
    getPosition: positions,
    selectionMask
  });
  expect(() => invalidMask.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow(
    /rows must match/
  );
});
