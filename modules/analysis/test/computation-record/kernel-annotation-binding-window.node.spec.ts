// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createGPUKernelAnnotation} from '../../src/computation-record/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('C4: manual annotations retain aligned binding windows and bare buffer handles', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const storage = createTransientView(graph, 'storage', 'uint32', 128);
    for (const [byteOffset, bindingByteOffset] of [
      [8, 0],
      [264, 256]
    ]) {
      const view = graph.createDataView(storage.buffer, {format: 'uint32', byteOffset, length: 3});
      const annotation = createGPUKernelAnnotation({
        op: 'TransferMask',
        moduleId: 'transfer-mask',
        source: 'fn main() {}',
        workgroupSize: [1, 1, 1],
        workgroups: [3, 1, 1],
        bindings: [
          {name: 'view', view, access: 'read'},
          {name: 'handle', view: storage.buffer, access: 'read_write'}
        ]
      });
      // 256-byte binding alignment leaves two u32 words before the three logical words:
      // the physical window is 2*4 + 3*4 = 20 bytes at either aligned offset.
      expect(annotation.bindings).toEqual([
        {
          name: 'view',
          resource: 'storage',
          access: 'read',
          elementType: 'u32',
          elementOffset: 2,
          length: 3,
          bindingByteOffset,
          bindingByteLength: 20
        },
        {name: 'handle', resource: 'storage', access: 'read_write'}
      ]);
    }
  } finally {
    device.destroy();
  }
});
