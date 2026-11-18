// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {writeWKB} from '@math.gl/arrow/wkb';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeometrySourceFromArrow
} from '../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {GPUKeplerSpatialSelection} from '../src/selection/gpu-spatial-selection';

it('constructs WKB containment with a four-byte empty position chunk and preserves source topology', () => {
  const device = createNullWebGPUDevice();
  const point = writeWKB({type: 'Point', coordinates: [2.5, 2.5]});
  const first = arrow.vectorFromArray([null, point], new arrow.Binary());
  const empty = arrow.vectorFromArray([], new arrow.Binary());
  const last = arrow.vectorFromArray([point, new Uint8Array([1, 2])], new arrow.Binary());
  const source = prepareKeplerGeometrySourceFromArrow(
    new arrow.Vector([...first.data, ...empty.data, ...last.data]),
    {encoding: 'geoarrow.wkb'}
  );
  const geometry = makeKeplerGeometryGPUVectors(device, source);
  let selection: GPUKeplerSpatialSelection | undefined;
  try {
    const positions = geometry.representative!.positions;
    expect(positions.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
    expect(positions.data[1].buffer.byteLength).toBe(4);
    selection = new GPUKeplerSpatialSelection({
      device,
      positions,
      selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}
    });
    expect(selection.mask.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
    expect(selection.mask.length).toBe(4);
  } finally {
    selection?.destroy();
    geometry.destroy();
    device.destroy();
  }
});

it('rejects interleaved positions instead of dropping their buffer layout', () => {
  const device = createNullWebGPUDevice();
  const positions = new GPUVector({
    type: 'interleaved',
    name: 'positions',
    buffer: device.createBuffer({byteLength: 8, usage: Buffer.STORAGE}),
    format: 'float32x2',
    length: 1,
    byteStride: 8,
    attributes: [
      {attribute: 'horizontal', format: 'float32', byteOffset: 0},
      {attribute: 'vertical', format: 'float32', byteOffset: 4}
    ],
    ownsBuffer: true
  });
  let selection: GPUKeplerSpatialSelection | undefined;
  try {
    expect(() => {
      selection = new GPUKeplerSpatialSelection({
        device,
        positions,
        selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}
      });
    }).toThrow(
      'GPUCommandGraph import "kepler-spatial-selection-points" does not accept interleaved GPUVector data'
    );
  } finally {
    selection?.destroy();
    positions.destroy();
    device.destroy();
  }
});

it('rejects an interleaved borrowed output mask instead of dropping its buffer layout', () => {
  const device = createNullWebGPUDevice();
  const positions = new GPUVector({
    type: 'buffer',
    name: 'positions',
    buffer: device.createBuffer({byteLength: 8, usage: Buffer.STORAGE}),
    format: 'float32x2',
    length: 1,
    ownsBuffer: true
  });
  const outputMask = new GPUVector({
    type: 'interleaved',
    name: 'mask',
    buffer: device.createBuffer({byteLength: 4, usage: Buffer.STORAGE}),
    format: 'uint32',
    length: 1,
    byteStride: 4,
    attributes: [
      {attribute: 'lower', format: 'uint16', byteOffset: 0},
      {attribute: 'upper', format: 'uint16', byteOffset: 2}
    ],
    ownsBuffer: true
  });
  let selection: GPUKeplerSpatialSelection | undefined;
  try {
    expect(() => {
      selection = new GPUKeplerSpatialSelection({
        device,
        positions,
        outputMask,
        selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}
      });
    }).toThrow(
      'GPUCommandGraph import "kepler-spatial-selection-mask" does not accept interleaved GPUVector data'
    );
  } finally {
    selection?.destroy();
    outputMask.destroy();
    positions.destroy();
    device.destroy();
  }
});

it('rejects a negative position chunk length instead of silently omitting that chunk', () => {
  const device = createNullWebGPUDevice();
  const positions = new GPUVector({
    type: 'data',
    name: 'positions',
    format: 'float32x2',
    data: [2, -1].map(
      length =>
        new GPUData({
          buffer: device.createBuffer({
            byteLength: Math.max(4, length * 8),
            usage: Buffer.STORAGE
          }),
          format: 'float32x2',
          length,
          ownsBuffer: true
        })
    ),
    ownsData: true
  });
  let selection: GPUKeplerSpatialSelection | undefined;
  try {
    expect(positions.data.map(chunk => chunk.length)).toEqual([2, -1]);
    expect(positions.length).toBe(1);
    expect(() => {
      selection = new GPUKeplerSpatialSelection({
        device,
        positions,
        selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}
      });
    }).toThrow(/length/);
  } finally {
    selection?.destroy();
    positions.destroy();
    device.destroy();
  }
});

it.each([
  {lengths: [0, 2, 0]},
  {lengths: [0, 0, 0]}
])('preserves borrowed mask ownership with leading, trailing or entirely empty chunks $lengths', ({
  lengths
}) => {
  const device = createNullWebGPUDevice();
  const positions = new GPUVector({
    type: 'data',
    name: 'positions',
    format: 'float32x2',
    data: lengths.map(
      length =>
        new GPUData({
          buffer: device.createBuffer({
            byteLength: Math.max(4, length * 8),
            usage: Buffer.STORAGE
          }),
          format: 'float32x2',
          length,
          ownsBuffer: true
        })
    ),
    ownsData: true
  });
  const outputMask = new GPUVector({
    type: 'data',
    name: 'mask',
    format: 'uint32',
    data: lengths.map(
      length =>
        new GPUData({
          buffer: device.createBuffer({
            byteLength: Math.max(4, length * 4),
            usage: Buffer.STORAGE
          }),
          format: 'uint32',
          length,
          ownsBuffer: true
        })
    ),
    ownsData: true
  });
  let selection: GPUKeplerSpatialSelection | undefined;
  try {
    selection = new GPUKeplerSpatialSelection({
      device,
      positions,
      outputMask,
      selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}
    });
    expect(selection.mask).toBe(outputMask);
    expect(selection.mask.data.map(chunk => chunk.length)).toEqual(lengths);
    selection.destroy();
    expect(outputMask.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    expect(positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  } finally {
    selection?.destroy();
    outputMask.destroy();
    positions.destroy();
    device.destroy();
  }
});
