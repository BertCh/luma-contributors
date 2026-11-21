// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {createGraphChunkGather} from '../../src/analysis-core/kernels/graph-chunk-gather';
import {GPU_MAXIMUM_COMMAND_NODE_COUNT} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

function createVector<Format extends 'uint32' | 'uint32x2' | 'uint8' | 'uint16'>(
  format: Format,
  chunks: GraphDataView<Format>[],
  rowByteLength = chunks[0]?.rowByteLength ?? 4
): GraphVectorView<Format> {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const stride = format === 'uint32x2' ? 2 : 1;
  return new GraphVectorView({
    id: 'source',
    name: 'source',
    format,
    length,
    valueLength: length * stride,
    stride,
    byteStride: rowByteLength,
    rowByteLength,
    data: chunks
  });
}

it('plans ordered copies with byte offsets, empty chunks and a shared source buffer', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(
    graph,
    'backing',
    'uint32x2',
    8,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const first = graph.createDataView(backing.buffer, {
    format: 'uint32x2',
    byteOffset: 4,
    length: 2
  });
  const second = graph.createDataView(backing.buffer, {
    format: 'uint32x2',
    byteOffset: 28,
    length: 3
  });
  const empty = createTransientView(graph, 'empty', 'uint32x2', 0);
  const {view, nodes} = createGraphChunkGather(graph, {
    id: 'gather',
    source: createVector('uint32x2', [first, empty, second])
  });
  expect(view).toMatchObject({format: 'uint32x2', length: 5, byteStride: 8, byteOffset: 0});
  expect(view.buffer).toMatchObject({transient: true, byteLength: 40});
  expect(view.buffer.usage).toBe(Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST);
  expect(nodes).toHaveLength(2);
  const sourceBuffer = device.createBuffer({byteLength: 64});
  const destinationBuffer = device.createBuffer({byteLength: 40});
  const commandEncoder = device.createCommandEncoder();
  const copy = vi.spyOn(commandEncoder, 'copyBufferToBuffer');
  for (const node of nodes) {
    expect(node.type).toBe('copy');
    expect(node.resources).toEqual([
      {buffer: node === nodes[0] ? first : second, usage: 'copy-source'},
      {buffer: view, usage: 'copy-destination'}
    ]);
    node.compile({device}).encode({
      commandEncoder,
      parameters: undefined,
      getBuffer: input => (input === view ? destinationBuffer : sourceBuffer),
      getTexture: () => {
        throw new Error('unexpected texture');
      },
      getTextureView: () => {
        throw new Error('unexpected texture');
      },
      getExternalTexture: () => {
        throw new Error('unexpected texture');
      }
    });
  }
  expect(copy.mock.calls.map(([options]) => options)).toEqual([
    {sourceBuffer, sourceOffset: 4, destinationBuffer, destinationOffset: 0, size: 16},
    {sourceBuffer, sourceOffset: 28, destinationBuffer, destinationOffset: 16, size: 24}
  ]);
  expect(nodes.map(node => node.workload?.readByteLength)).toEqual([16, 24]);
  sourceBuffer.destroy();
  destinationBuffer.destroy();
  commandEncoder.destroy();
});

it('borrows packed input without requiring copy usage or declaring scratch', () => {
  const graph = new GPUCommandGraph(device);
  const source = createTransientView(graph, 'source', 'uint32', 10);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(createGraphChunkGather(graph, {id: 'gather', source})).toEqual({view: source, nodes: []});
  expect(allocate).not.toHaveBeenCalled();
});

it('normalizes a singleton vector and supports empty vectors without copy nodes', () => {
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(graph, 'source', 'uint32', 3, Buffer.STORAGE | Buffer.COPY_SRC);
  const singleton = createGraphChunkGather(graph, {
    id: 'singleton',
    source: createVector('uint32', [chunk])
  });
  expect(singleton.view).not.toBe(chunk);
  expect(singleton.nodes).toHaveLength(1);
  for (const [index, chunks] of [
    [],
    [createTransientView(graph, 'empty', 'uint32', 0)]
  ].entries()) {
    const empty = createGraphChunkGather(graph, {
      id: `empty-${index}`,
      source: createVector('uint32', chunks)
    });
    expect(empty.view.length).toBe(0);
    expect(empty.view.buffer.byteLength).toBe(4);
    expect(empty.nodes).toEqual([]);
  }
});

it('plans thirty million rows on a device with a larger binding limit without allocating data', () => {
  const largerDevice = createNullWebGPUDevice();
  Object.defineProperty(largerDevice.limits, 'maxStorageBufferBindingSize', {value: 240_000_000});
  const graph = new GPUCommandGraph(largerDevice);
  const chunks = [0, 1, 2].map(index =>
    createTransientView(
      graph,
      `source-${index}`,
      'uint32x2',
      10_000_000,
      Buffer.STORAGE | Buffer.COPY_SRC
    )
  );
  const result = createGraphChunkGather(graph, {
    id: 'large',
    source: createVector('uint32x2', chunks)
  });
  expect(result.view.length).toBe(30_000_000);
  expect(result.view.buffer.byteLength).toBe(240_000_000);
  expect(result.nodes).toHaveLength(3);
  largerDevice.destroy();
});

it('checks the complete binding before allocating scratch, including a borrowed view offset', () => {
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(
    graph,
    'source',
    'uint32',
    33_554_433,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const source = createVector('uint32', [chunk]);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkGather(graph, {id: 'too-large', source})).toThrow(
    'too-large source packed scratch needs 134217732 bytes in one binding (limit 134217728)'
  );
  expect(allocate).not.toHaveBeenCalled();
  const borrowed = graph.createDataView(chunk.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  expect(() => createGraphChunkGather(graph, {id: 'offset', source: borrowed})).toThrow(
    'offset source packed scratch needs 134217732 bytes'
  );
});

it('checks aligned borrowed and empty scratch bindings while copy-only sources need no storage binding', () => {
  const limitedDevice = createNullWebGPUDevice();
  Object.defineProperty(limitedDevice.limits, 'maxStorageBufferBindingSize', {value: 4});
  const graph = new GPUCommandGraph(limitedDevice);
  const backing = createTransientView(
    graph,
    'backing',
    'uint32',
    70,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const aligned = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const prefixed = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 260
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(createGraphChunkGather(graph, {id: 'aligned', source: aligned}).view).toBe(aligned);
    expect(() => createGraphChunkGather(graph, {id: 'prefix', source: prefixed})).toThrow(
      'prefix source packed scratch needs 8 bytes in one binding (limit 4)'
    );
    expect(() =>
      createGraphChunkGather(graph, {
        id: 'empty-wide',
        source: createVector('uint32x2', [], 8)
      })
    ).toThrow('empty-wide source packed scratch needs 8 bytes in one binding (limit 4)');
    expect(allocate).not.toHaveBeenCalled();
    const copied = createGraphChunkGather(graph, {
      id: 'copy-only-source',
      source: createVector('uint32', [prefixed])
    });
    expect(copied.view.buffer.byteLength).toBe(4);
    expect(copied.nodes).toHaveLength(1);
    expect(allocate).toHaveBeenCalledTimes(1);
  } finally {
    limitedDevice.destroy();
  }
});

it('rejects missing copy usage, strided input, unaligned copies and foreign graph chunks', () => {
  const graph = new GPUCommandGraph(device);
  const source = createTransientView(graph, 'source', 'uint32', 4);
  expect(() =>
    createGraphChunkGather(graph, {id: 'usage', source: createVector('uint32', [source])})
  ).toThrow('usage source nonempty chunk 0 buffer usage must include Buffer.COPY_SRC');
  const strided = graph.createDataView(source.buffer, {format: 'uint32', length: 2, byteStride: 8});
  expect(() => createGraphChunkGather(graph, {id: 'stride', source: strided})).toThrow(
    'stride source chunk 0 must be packed'
  );
  const bytes = createTransientView(graph, 'bytes', 'uint8', 2, Buffer.STORAGE | Buffer.COPY_SRC);
  expect(() =>
    createGraphChunkGather(graph, {id: 'alignment', source: createVector('uint8', [bytes])})
  ).toThrow(
    'alignment source.rowByteLength must be a positive multiple of GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT (4)'
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() => createGraphChunkGather(otherGraph, {id: 'ownership', source})).toThrow(
    'ownership views must belong to the target graph'
  );
});

it.each([
  'uint8',
  'uint16'
] as const)('rejects %s rows on empty-vector, chunked and borrowed routes before declaring scratch or nodes', format => {
  const graph = new GPUCommandGraph(device);
  const rowByteLength = format === 'uint8' ? 1 : 2;
  const chunks = [0, 1, 4, 8].map(length =>
    createTransientView(graph, `source-${length}`, format, length, Buffer.STORAGE | Buffer.COPY_SRC)
  );
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const addCopy = vi.spyOn(graph, 'addCopyPass');
  const addCompute = vi.spyOn(graph, 'addComputePass');
  const inputs = [
    ...chunks,
    ...chunks.map(chunk => createVector(format, [chunk])),
    createVector(format, [], rowByteLength)
  ];
  for (const source of inputs) {
    expect(() => createGraphChunkGather(graph, {id: 'word-rows', source})).toThrow(
      'word-rows source.rowByteLength must be a positive multiple of GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT (4)'
    );
  }
  expect(allocate).not.toHaveBeenCalled();
  expect(addCopy).not.toHaveBeenCalled();
  expect(addCompute).not.toHaveBeenCalled();
});

it('creates zero shaders and pipelines as chunk count grows', () => {
  const shader = vi.spyOn(device, 'createShader');
  const pipeline = vi.spyOn(device, 'createComputePipeline');
  const asynchronousPipeline = vi.spyOn(device, 'createComputePipelineAsync');
  for (const chunkCount of [1, 4, 257]) {
    const graph = new GPUCommandGraph(device);
    const chunk = createTransientView(
      graph,
      'source',
      'uint32',
      1,
      Buffer.STORAGE | Buffer.COPY_SRC
    );
    const {nodes} = createGraphChunkGather(graph, {
      id: 'pipeline-invariant',
      source: createVector('uint32', new Array(chunkCount).fill(chunk))
    });
    expect(nodes).toHaveLength(chunkCount);
    for (const node of nodes) {
      expect(node.type).toBe('copy');
      node.compile({device});
    }
    expect(shader).not.toHaveBeenCalled();
    expect(pipeline).not.toHaveBeenCalled();
    expect(asynchronousPipeline).not.toHaveBeenCalled();
  }
});

it('checks the copy-node budget before allocating scratch', () => {
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(graph, 'source', 'uint32', 1, Buffer.STORAGE | Buffer.COPY_SRC);
  const source = createVector('uint32', new Array(GPU_MAXIMUM_COMMAND_NODE_COUNT + 1).fill(chunk));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkGather(graph, {id: 'budget', source})).toThrow(
    `budget estimated command node count must be at most ${GPU_MAXIMUM_COMMAND_NODE_COUNT}`
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('rejects inconsistent vector totals before allocating scratch', () => {
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(graph, 'chunk', 'uint32', 2, Buffer.STORAGE | Buffer.COPY_SRC);
  const source = createVector('uint32', [chunk]);
  Object.defineProperty(source, 'length', {value: 3});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkGather(graph, {id: 'total', source})).toThrow(
    'total source.length must equal its chunk total'
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('rejects borrowed input without STORAGE and copy ranges that exceed the source buffer', () => {
  const graph = new GPUCommandGraph(device);
  const source = graph.createDataView(
    graph.createTransientBuffer({id: 'copy-only', byteLength: 8, usage: Buffer.COPY_SRC}),
    {format: 'uint32', length: 2}
  );
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkGather(graph, {id: 'borrowed', source})).toThrow(
    'borrowed source chunk 0 buffer usage must include Buffer.STORAGE'
  );
  Object.defineProperty(source, 'byteOffset', {value: 4});
  expect(() =>
    createGraphChunkGather(graph, {id: 'range', source: createVector('uint32', [source])})
  ).toThrow(
    'range source nonempty chunk 0 copy range must be four-byte aligned and fit its buffer byteLength'
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('checks packed scratch maxBufferSize before declaring any buffer, view or node', () => {
  const limitedDevice = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(limitedDevice);
  const chunk = createTransientView(graph, 'chunk', 'uint32', 3, Buffer.STORAGE | Buffer.COPY_SRC);
  const source = createVector('uint32', [chunk, chunk]);
  Object.defineProperty(limitedDevice.limits, 'maxBufferSize', {value: 16});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => createGraphChunkGather(graph, {id: 'allocation-limit', source})).toThrow(
      /allocation-limit packed scratch byteLength.*\[0, 16\]/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    limitedDevice.destroy();
  }
});

it('COMMON-W3 gathers foreign-prototype vectors including single and empty topology', () => {
  for (const lengths of [[1], [], [0], [0, 1, 0]]) {
    const graph = new GPUCommandGraph(device);
    const chunks = lengths.map((length, index) =>
      createTransientView(
        graph,
        `foreign-${index}`,
        'uint32',
        length,
        Buffer.STORAGE | Buffer.COPY_SRC
      )
    );
    const source = createVector('uint32', chunks);
    // A separately loaded entry condition has the published shape, but a different prototype.
    Object.setPrototypeOf(source, Object.prototype);
    const gathered = createGraphChunkGather(graph, {id: 'foreign-gather', source});
    expect(gathered.view).not.toBe(source);
    expect(gathered.view.length).toBe(lengths.reduce((total, length) => total + length, 0));
    expect(gathered.nodes).toHaveLength(lengths.filter(length => length > 0).length);
  }
});
