// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {createNullWebGPUDevice} from '../../graph/test/utils/gpu-contributor-test-utils';
import {
  GPUKeplerSpatialSelection,
  prepareKeplerSpatialSelection,
  type KeplerSpatialSelection
} from '../src/selection';

const RECTANGLE: KeplerSpatialSelection = {kind: 'rectangle', bounds: [0, 0, 4, 4]};

describe('Kepler drawn spatial selection preparation', () => {
  it('preserves shells, holes, and union parts in one feature', () => {
    const geometry = prepareKeplerSpatialSelection({
      kind: 'polygon',
      polygons: [
        [
          [
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4],
            [0, 0]
          ],
          [
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3]
          ]
        ],
        [
          [
            [8, 8],
            [10, 8],
            [10, 10],
            [8, 10]
          ]
        ]
      ]
    });
    expect(Array.from(geometry.featureOffsets)).toEqual([0, 2]);
    expect(Array.from(geometry.polygonOffsets)).toEqual([0, 2, 3]);
    expect(Array.from(geometry.ringOffsets)).toEqual([0, 4, 8, 12]);
    expect(geometry.positions.length).toBe(24);
  });

  it('makes rectangle and equivalent polygon topology compatible and rounds to Float32', () => {
    const rectangle = prepareKeplerSpatialSelection(RECTANGLE);
    const polygon = prepareKeplerSpatialSelection({
      kind: 'polygon',
      polygons: [
        [
          [
            [0.1, 0],
            [4, 0],
            [4, 4],
            [0.1, 4]
          ]
        ]
      ]
    });
    expect(polygon.topologyKey).toBe(rectangle.topologyKey);
    expect(polygon.positions[0]).toBe(Math.fround(0.1));
  });

  it('rejects empty, malformed, nonfinite and collapsed rectangle inputs', () => {
    expect(() => prepareKeplerSpatialSelection({kind: 'polygon', polygons: []})).toThrow();
    expect(() => prepareKeplerSpatialSelection({kind: 'polygon', polygons: [[]]})).toThrow();
    expect(() =>
      prepareKeplerSpatialSelection({
        kind: 'polygon',
        polygons: [
          [
            [
              [0, 0],
              [1, 0],
              [0, 0]
            ]
          ]
        ]
      })
    ).toThrow();
    expect(() =>
      prepareKeplerSpatialSelection({
        kind: 'polygon',
        polygons: [
          [
            [
              [0, 0],
              [Infinity, 0],
              [0, 1]
            ]
          ]
        ]
      })
    ).toThrow();
    expect(() =>
      prepareKeplerSpatialSelection({kind: 'rectangle', bounds: [0, 0, 0, 1]})
    ).toThrow();
    expect(() =>
      prepareKeplerSpatialSelection({kind: 'rectangle', bounds: [1e10, 0, 1e10 + 1, 1]})
    ).toThrow();
  });

  it('rejects unprepared antimeridian and pole rings only in geographic mode', () => {
    const crossing: KeplerSpatialSelection = {
      kind: 'polygon',
      polygons: [
        [
          [
            [179, 0],
            [-179, 0],
            [-179, 1],
            [179, 1]
          ]
        ]
      ]
    };
    expect(() => prepareKeplerSpatialSelection(crossing, 'longitude-latitude')).toThrow(
      /antimeridian/
    );
    expect(() => prepareKeplerSpatialSelection(crossing, 'planar')).not.toThrow();
    // Native Kepler performs no antimeridian, range or pole validation.
    expect(() => prepareKeplerSpatialSelection(crossing, 'native-linear-degrees')).not.toThrow();
    expect(() =>
      prepareKeplerSpatialSelection(
        {kind: 'rectangle', bounds: [170, 80, 200, 90]},
        'native-linear-degrees'
      )
    ).not.toThrow();
    expect(() =>
      prepareKeplerSpatialSelection(
        {
          kind: 'rectangle',
          bounds: [0, 80, 1, 90]
        },
        'longitude-latitude'
      )
    ).toThrow(/geographic preparation/);
  });
});

describe('GPUKeplerSpatialSelection ownership and topology', () => {
  it('borrows a caller destination and rejects source aliasing', () => {
    const device = createNullWebGPUDevice();
    const source = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE | Buffer.COPY_DST});
    const output = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE | Buffer.VERTEX});
    const positions = new GPUVector({
      type: 'buffer',
      name: 'positions',
      buffer: source,
      format: 'float32x2',
      length: 2,
      ownsBuffer: false
    });
    const outputMask = new GPUVector({
      type: 'buffer',
      name: 'mask',
      buffer: output,
      format: 'uint32',
      length: 2,
      ownsBuffer: false
    });
    const aliasMask = new GPUVector({
      type: 'buffer',
      name: 'alias',
      buffer: source,
      format: 'uint32',
      length: 2,
      ownsBuffer: false
    });
    expect(
      () =>
        new GPUKeplerSpatialSelection({
          device,
          positions,
          outputMask: aliasMask,
          selection: RECTANGLE
        })
    ).toThrow(/separate matching storage/);
    const selection = new GPUKeplerSpatialSelection({
      device,
      positions,
      outputMask,
      selection: RECTANGLE
    });
    expect(selection.mask).toBe(outputMask);
    selection.destroy();
    expect(output.destroyed).toBe(false);
    expect(source.destroyed).toBe(false);
    output.destroy();
    source.destroy();
    device.destroy();
  });

  it('preserves chunk lengths, borrows inputs and rejects topology changes atomically', () => {
    const device = createNullWebGPUDevice();
    const sourceBuffers = [1, 2].map(length =>
      device.createBuffer({
        byteLength: length * 8,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      })
    );
    const positions = new GPUVector({
      type: 'data',
      name: 'points',
      format: 'float32x2',
      data: sourceBuffers.map(
        (buffer, index) =>
          new GPUData({
            buffer,
            format: 'float32x2',
            length: index + 1,
            ownsBuffer: true
          })
      )
    });
    const selection = new GPUKeplerSpatialSelection({device, positions, selection: RECTANGLE});
    const maskBuffers = selection.mask.data.map(chunk => chunk.buffer);
    expect(selection.mask.data.map(chunk => chunk.length)).toEqual([1, 2]);
    expect(maskBuffers.every(buffer => (buffer.usage & Buffer.VERTEX) !== 0)).toBe(true);
    expect(selection.updateSelection({kind: 'rectangle', bounds: [2, 2, 6, 6]})).toBe(1);
    expect(() =>
      selection.updateSelection({
        kind: 'polygon',
        polygons: [
          [
            [
              [0, 0],
              [4, 0],
              [0, 4]
            ]
          ]
        ]
      })
    ).toThrow(/topology/);
    expect(selection.updateSelection(RECTANGLE)).toBe(2);
    selection.destroy();
    selection.destroy();
    expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    expect(maskBuffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(() => selection.updateSelection(RECTANGLE)).toThrow(/destroyed/);
    positions.destroy();
    device.destroy();
  });
});
