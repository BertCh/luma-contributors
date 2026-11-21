// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GEOMETRY_VALIDITY_CLASSES_WGSL} from '../../../../../website/src/gpu/runs/entries/geometry-processing/shared/geometry-validity-classes';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('the website device class keeps uncertain predicates separate from incomplete and observed defects', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const mask = createInputBuffer(device, Uint32Array.of(256, 0, 16, 128));
  const validity = createInputBuffer(device, Uint32Array.of(0, 0, 0, 1));
  const classes = createOutputBuffer(device, 4);
  const node = createWGSLKernelNode(graph, {
    id: 'website-validity',
    operation: 'test',
    invocationCount: 4,
    bindings: [
      {
        name: 'mask',
        view: importGraphBuffer(graph, 'mask', mask, 'uint32', 4),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'validity',
        view: importGraphBuffer(graph, 'validity', validity, 'uint32', 4),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'classes',
        view: importGraphBuffer(graph, 'classes', classes, 'uint32', 4),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: GEOMETRY_VALIDITY_CLASSES_WGSL
  });
  graph.add({getCommandNodes: () => [node]});
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(classes, 4)).toEqual([6, 5, 1, 3]);
  } finally {
    compiled.destroy();
    mask.destroy();
    validity.destroy();
    classes.destroy();
  }
});
