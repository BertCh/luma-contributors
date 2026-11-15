// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUMeshColumnLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import type {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

function makeVector(length: number): GPUVector<'float32x2'> {
  return new GPUVector({
    type: 'data',
    format: 'float32x2',
    ownsData: false,
    data: [
      new GPUData({
        format: 'float32x2',
        length,
        ownsBuffer: false,
        buffer: {byteLength: length * 8, destroy: vi.fn(), readAsync: vi.fn()} as unknown as Buffer
      })
    ]
  });
}

test('custom columns retain a borrowed fp64 low position vector without reading or destroying it', () => {
  const getPosition = makeVector(2);
  const getPositionLow = makeVector(2);
  const vertices = makeVector(6);
  const layer = new GPUMeshColumnLayer({vertices, edgeDistance: 8, getPosition, getPositionLow});
  expect(layer.props.getPositionLow).toBe(getPositionLow);
  expect(getPositionLow.data[0].buffer.readAsync).not.toHaveBeenCalled();
  expect(getPositionLow.data[0].buffer.destroy).not.toHaveBeenCalled();
});
