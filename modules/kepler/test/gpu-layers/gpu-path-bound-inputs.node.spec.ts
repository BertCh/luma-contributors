// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {describe, expect, test} from 'vitest';
import {getGPUPathBoundInputs} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-path-layer';

function makePathVector(chunkLengths: number[]): GPUVector<'float32x2'> {
  const data = chunkLengths.map(
    length =>
      new GPUData<'float32x2'>({
        buffer: {byteLength: length * 8, destroy() {}} as unknown as Buffer,
        format: 'float32x2',
        length,
        byteStride: 8,
        rowByteLength: 8,
        ownsBuffer: false
      })
  );
  return new GPUVector({type: 'data', name: 'paths', format: 'float32x2', data, ownsData: false});
}

function isSameSignature(first: unknown[], second: unknown[]): boolean {
  return first.length === second.length && first.every((input, index) => input === second[index]);
}

describe('GPUPathLayer bound inputs', () => {
  test('inline constant color and width literals do not change the signature', () => {
    const getPath = makePathVector([2, 3]);
    const first = getGPUPathBoundInputs({id: 'path', getPath, getColor: [255, 0, 0], getWidth: 4});
    const second = getGPUPathBoundInputs({id: 'path', getPath, getColor: [255, 0, 0], getWidth: 4});
    expect(isSameSignature(first, second)).toBe(true);
  });

  test('changed constant values change the signature', () => {
    const getPath = makePathVector([2]);
    const first = getGPUPathBoundInputs({id: 'path', getPath, getColor: [255, 0, 0], getWidth: 4});
    const second = getGPUPathBoundInputs({id: 'path', getPath, getColor: [0, 0, 255], getWidth: 4});
    expect(isSameSignature(first, second)).toBe(false);
  });

  test('chunks appended in place change the signature', () => {
    const getPath = makePathVector([2]);
    const before = getGPUPathBoundInputs({id: 'path', getPath});
    const [appended] = makePathVector([3]).data;
    (getPath.data as GPUData<'float32x2'>[]).push(appended);
    expect(isSameSignature(before, getGPUPathBoundInputs({id: 'path', getPath}))).toBe(false);
  });
});
