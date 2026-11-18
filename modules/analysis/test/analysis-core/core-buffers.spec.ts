// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {sliceGraphChunkGroupView} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {GPUParameterBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readFloat32, readUint32, submitGraph} from '../../src/testing/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('submits an empty end-slice fill without binding past the buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'empty-end-values',
    format: 'uint32',
    length: 2,
    values: Uint32Array.of(11, 12)
  });
  const graph = new GPUCommandGraph(device);
  const parent = parameters.importToGraph(graph);
  const empty = sliceGraphChunkGroupView(graph, parent, 2, 0);
  graph.add(
    createFillNode(graph, {
      id: 'empty-end-fill',
      operation: 'CoreBuffers',
      view: empty,
      type: 'u32',
      value: '99u'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(parameters.buffer, 2)).toEqual([11, 12]);
  } finally {
    compiled.destroy();
    parameters.destroy();
  }
});

it('fills a float32 word using its numeric scalar representation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'scalar-float-values',
    format: 'float32',
    length: 1,
    values: Float32Array.of(0)
  });
  const graph = new GPUCommandGraph(device);
  const view = parameters.importToGraph(graph);
  expect(() =>
    createFillNode(graph, {
      id: 'wrong-scalar-fill',
      operation: 'CoreBuffers',
      view,
      type: 'u32',
      value: '1u'
    })
  ).toThrow(/type.*format/);
  graph.add(
    createFillNode(graph, {
      id: 'scalar-float-fill',
      operation: 'CoreBuffers',
      view,
      type: 'f32',
      value: '1.0'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(parameters.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    parameters.destroy();
  }
});

it.each([
  undefined,
  3,
  0
])('fills a vector prefix of %s components within sentinels', async componentCount => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const initial = Float32Array.of(91, 92, 1, 2, 3, 4, 5, 6, 7, 8, 93, 94);
  const parameters = new GPUParameterBuffer(device, {
    id: 'vector-values',
    format: 'float32',
    length: initial.length,
    values: initial
  });
  const graph = new GPUCommandGraph(device);
  const imported = parameters.importToGraph(graph);
  const vectors = graph.createDataView(imported.buffer, {
    format: 'float32x4',
    length: 2,
    byteOffset: 8
  });
  graph.add(
    createFillNode(graph, {
      id: 'vector-fill',
      operation: 'CoreBuffers',
      view: vectors,
      type: 'f32',
      value: '42.0',
      componentCount
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const expected = Array.from(initial);
    expected.fill(42, 2, 2 + (componentCount ?? 8));
    expect(await readFloat32(parameters.buffer, initial.length)).toEqual(expected);
    // The same compiled graph must refill values after a host update.
    parameters.write(initial);
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(parameters.buffer, initial.length)).toEqual(expected);
  } finally {
    compiled.destroy();
    parameters.destroy();
  }
});

it('fills a borrowed parent subview without changing adjacent allocations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const initial = Uint32Array.of(11, 12, 21, 22, 31, 32, 33, 34);
  const parameters = new GPUParameterBuffer(device, {
    id: 'chunk-values',
    format: 'uint32',
    length: initial.length,
    values: initial
  });
  const graph = new GPUCommandGraph(device);
  const imported = parameters.importToGraph(graph);
  const parent = graph.createDataView(imported.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 8
  });
  const child = sliceGraphChunkGroupView(graph, parent, 1, 1);
  graph.add(
    createFillNode(graph, {
      id: 'child-fill',
      operation: 'CoreBuffers',
      view: child,
      type: 'u32',
      value: '42u'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(parameters.buffer, initial.length)).toEqual([
      11, 12, 21, 42, 31, 32, 33, 34
    ]);
  } finally {
    compiled.destroy();
    parameters.destroy();
  }
});

it('reads inferred vector bindings as scalar components with their byte offset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'inferred-values',
    format: 'uint32',
    length: 6,
    values: Uint32Array.of(91, 1, 2, 3, 4, 92)
  });
  const graph = new GPUCommandGraph(device);
  const imported = parameters.importToGraph(graph);
  const vectors = graph.createDataView(imported.buffer, {
    format: 'uint32x2',
    length: 2,
    byteOffset: 4
  });
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'inferred-increment',
      operation: 'CoreBuffers',
      invocationCount: 4,
      bindings: [getWGSLKernelBinding('values', vectors, 'read_write')],
      body: 'values[valuesOffset + index] += 10u;'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(parameters.buffer, 6)).toEqual([91, 11, 12, 13, 14, 92]);
  } finally {
    compiled.destroy();
    parameters.destroy();
  }
});
