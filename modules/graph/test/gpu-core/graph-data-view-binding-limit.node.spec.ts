// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {getViewBinding} from '../../src/gpu-core/graph-data-view-utils';
import type {GraphDataView} from '../../src/gpu-core/gpu-command-graph';

function createView(length: number): GraphDataView {
  return {
    buffer: {byteLength: 4096},
    byteOffset: 0,
    byteStride: 4,
    rowByteLength: 4,
    length
  } as unknown as GraphDataView;
}

function createBuffer(maxStorageBufferBindingSize: number): Buffer {
  return {device: {limits: {maxStorageBufferBindingSize}}} as unknown as Buffer;
}

test('getViewBinding accepts views within maxStorageBufferBindingSize', () => {
  const view = createView(64);
  expect(getViewBinding(view, () => createBuffer(256))).toMatchObject({offset: 0, size: 256});
});

test('getViewBinding rejects views larger than maxStorageBufferBindingSize', () => {
  const view = createView(128);
  expect(() => getViewBinding(view, () => createBuffer(256))).toThrow(
    /maxStorageBufferBindingSize/
  );
});
