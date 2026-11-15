// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUPathLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {UNIT, type LayerContext, type PickingInfo} from '@deck.gl/core';
import {type Buffer, type RenderPass} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

function fixture() {
  const device = {};
  const paths = new GPUVector({
    type: 'data',
    format: 'vertex-list<float32x3>',
    ownsData: false,
    data: [1, 2].map(
      length =>
        new GPUData({
          format: 'vertex-list<float32x3>',
          length,
          valueLength: length * 2,
          valueOffsets: Int32Array.from({length: length + 1}, (_value, index) => index * 2),
          ownsBuffer: false,
          buffer: {
            device,
            byteLength: length * 24,
            destroy: vi.fn(),
            readAsync: vi.fn()
          } as unknown as Buffer
        })
    )
  });
  return {device, paths};
}

test('path joins keep explicit pixel/billboard defaults while enabling native miter/cap controls', () => {
  const {paths, device} = fixture();
  for (const custom of [false, true]) {
    const layer = new GPUPathLayer({
      getPath: paths,
      ...(custom
        ? ({
            billboard: false,
            capRounded: true,
            jointRounded: true,
            miterLimit: 2,
            antialiasing: true,
            timestampMode: 'trips',
            widthUnits: 'meters',
            widthScale: 3,
            widthMinPixels: 2,
            widthMaxPixels: 16
          } as const)
        : {})
    });
    layer.context = {device} as LayerContext;
    const write = vi.fn();
    layer.state = {
      batches: [],
      temporalBuffer: {write},
      defaultTimestampBuffer: null,
      boundInputs: []
    };
    layer.draw({renderPass: {} as RenderPass});
    const bytes = write.mock.calls[0][0] as Uint8Array;
    const floats = new Float32Array(bytes.buffer);
    const integers = new Uint32Array(bytes.buffer);
    expect(bytes.byteLength).toBe(64);
    expect(integers[7]).toBe(UNIT[custom ? 'meters' : 'pixels']);
    expect(Array.from(integers.slice(8, 11))).toEqual(custom ? [0, 1, 1] : [1, 0, 0]);
    expect(floats[11]).toBe(custom ? 2 : 4);
    expect(integers[12]).toBe(custom ? 1 : 0);
    expect(integers[13]).toBe(custom ? 1 : 0);
    expect(Array.from(floats.slice(4, 7))).toEqual(custom ? [3, 2, 16] : [1, 0, 1e9]);
  }
});

test('path joins retain source chunk/global picking identity and borrow coordinates', () => {
  const {paths} = fixture();
  const layer = new GPUPathLayer({getPath: paths, capRounded: true, jointRounded: true});
  expect(layer.getPickingInfo({info: {index: 2} as PickingInfo})).toHaveProperty('gpuVector', {
    rowIndex: 2,
    batchIndex: 1,
    batchRowIndex: 1
  });
  for (const chunk of paths.data) {
    expect(chunk.buffer.readAsync).not.toHaveBeenCalled();
    expect(chunk.buffer.destroy).not.toHaveBeenCalled();
  }
});
