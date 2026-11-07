// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {expect, test} from 'vitest';

test('GPUData rejects negative or non-integer scalar row layouts', () => {
  const device = new NullDevice({});
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  expect(() => new GPUData({buffer, format: 'float32', length: 4})).not.toThrow();
  expect(() => new GPUData({buffer, format: 'float32', length: -1})).toThrow();
  expect(() => new GPUData({buffer, format: 'float32', length: 1.5})).toThrow();
  expect(() => new GPUData({buffer, format: 'float32', length: 2, byteOffset: -4})).toThrow();
  device.destroy();
});

test('GPUVector.transferBufferOwnership requires single-chunk vectors', () => {
  const device = new NullDevice({});
  const buffer = device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
  const first = new GPUData({buffer, format: 'float32', length: 4, ownsBuffer: true});
  const second = new GPUData({buffer, format: 'float32', length: 4, byteOffset: 16});
  const multiChunk = new GPUVector({type: 'data', name: 'multi', data: [first, second]});
  const singleChunk = new GPUVector({type: 'data', name: 'single', data: [second]});

  expect(() => multiChunk.transferBufferOwnership(singleChunk)).toThrow();
  expect(first.ownsBuffer).toBe(true);

  const owner = new GPUVector({type: 'data', name: 'owner', data: [first]});
  owner.transferBufferOwnership(singleChunk);
  expect(first.ownsBuffer).toBe(false);
  expect(second.ownsBuffer).toBe(true);
  singleChunk.data[0].destroy();
  device.destroy();
});
