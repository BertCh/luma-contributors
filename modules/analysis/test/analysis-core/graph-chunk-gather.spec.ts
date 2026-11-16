// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  getViewBinding,
  GPUCommandGraph,
  GPUReduction,
  GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createGraphChunkGather} from '../../src/analysis-core/kernels/graph-chunk-gather';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createOutputBuffer, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

/** Checks storage binding, exact row contents and a storage-reading sum against CPU expectations. */
async function expectGatheredUint32(
  device: Device,
  graph: GPUCommandGraph,
  chunks: GraphDataView<'uint32'>[],
  expected: readonly number[]
): Promise<void> {
  const source = new GraphVectorView({
    id: 'chunks',
    name: 'chunks',
    format: 'uint32',
    length: expected.length,
    valueLength: expected.length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
  const {view, nodes} = createGraphChunkGather(graph, {id: 'gather', source});
  for (const node of nodes) graph.add(node);
  const gpuDevice = device.handle as GPUDevice;
  // This also binds the one-row backing allocation of a zero-length gathered view.
  graph.addCopyPass({
    id: 'validate-packed-storage-binding',
    resources: [{buffer: view, usage: 'storage-read'}],
    compile: () => ({
      encode: ({getBuffer}) => {
        const binding = getViewBinding(view, getBuffer);
        gpuDevice.createBindGroup({
          layout: gpuDevice.createBindGroupLayout({
            entries: [
              {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'read-only-storage'}}
            ]
          }),
          entries: [
            {
              binding: 0,
              resource: {
                buffer: binding.buffer.handle as GPUBuffer,
                offset: binding.offset,
                size: binding.size
              }
            }
          ]
        });
      }
    })
  });
  const outputBuffer = createOutputBuffer(device, expected.length);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', expected.length);
  if (expected.length > 0) {
    graph.addCopyPass({
      id: 'consume-packed',
      resources: [
        {buffer: view, usage: 'copy-source'},
        {buffer: output, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(view),
            sourceOffset: 0,
            destinationBuffer: getBuffer(output),
            destinationOffset: 0,
            size: expected.length * 4
          })
      })
    });
  }
  const sumBuffer = createOutputBuffer(device, 1);
  const sum = importGraphBuffer(graph, 'sum', sumBuffer, 'uint32', 1);
  graph.add(new GPUReduction({id: 'storage-sum', input: view, output: sum, operation: 'sum'}));
  let compiled: ReturnType<typeof graph.compile> | undefined;
  gpuDevice.pushErrorScope('validation');
  try {
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    if (expected.length > 0)
      expect(await readUint32(outputBuffer, expected.length)).toEqual(expected);
    const expectedSum = expected.reduce((total, value) => (total + value) >>> 0, 0);
    expect(await readUint32(sumBuffer, 1)).toEqual([expectedSum]);
  } finally {
    compiled?.destroy();
    outputBuffer.destroy();
    sumBuffer.destroy();
    await gpuDevice.queue.onSubmittedWorkDone();
    expect(await gpuDevice.popErrorScope()).toBeNull();
  }
}

it.each([
  'uint8',
  'uint16'
] as const)('rejects empty and borrowed %s input before any invalid storage binding reaches WebGPU', async format => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  const graph = new GPUCommandGraph(device);
  const backing = importGraphBuffer(graph, 'backing', buffer, format, 1);
  const empty = graph.createDataView(backing.buffer, {format, length: 0});
  const rowByteLength = format === 'uint8' ? 1 : 2;
  const emptyVector = (data: GraphDataView<typeof format>[]) =>
    new GraphVectorView({
      id: 'empty',
      name: 'empty',
      format,
      length: 0,
      valueLength: 0,
      stride: 1,
      byteStride: rowByteLength,
      rowByteLength,
      data
    });
  const gpuDevice = device.handle as GPUDevice;
  gpuDevice.pushErrorScope('validation');
  try {
    for (const source of [emptyVector([]), emptyVector([empty]), empty, backing]) {
      expect(() => {
        const {view} = createGraphChunkGather(graph, {id: 'reject-subword', source});
        // A regression must fail the early-rejection assertion, even if WebGPU reports the
        // invalid size asynchronously rather than throwing from createBindGroup.
        const binding = getViewBinding(view, () => buffer);
        gpuDevice.createBindGroup({
          layout: gpuDevice.createBindGroupLayout({
            entries: [
              {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'read-only-storage'}}
            ]
          }),
          entries: [
            {
              binding: 0,
              resource: {
                buffer: binding.buffer.handle as GPUBuffer,
                offset: binding.offset,
                size: binding.size
              }
            }
          ]
        });
      }).toThrow(
        'reject-subword source.rowByteLength must be a positive multiple of GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT (4)'
      );
    }
  } finally {
    expect(await gpuDevice.popErrorScope()).toBeNull();
    buffer.destroy();
  }
});

it.each([
  1, 255, 256, 257
])('copies chunks of %i rows with distinctive boundary values and reduces through storage', async chunkLength => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const arrays = [0, 1, 2].map(chunkIndex =>
    Uint32Array.from({length: chunkLength}, (_, row) => (chunkIndex + 1) * 100_000 + row * 17 + 1)
  );
  const buffers = arrays.map(values => {
    const padded = new Uint32Array(values.length + 3);
    padded.fill(0xdeadbeef);
    padded.set(values, 3);
    return device.createBuffer({data: padded, usage: Buffer.COPY_SRC});
  });
  const chunks = buffers.map((buffer, chunkIndex) => {
    const backing = importGraphBuffer(graph, `source-${chunkIndex}`, buffer, 'uint32');
    return graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: chunkLength,
      byteOffset: 12
    });
  });
  try {
    await expectGatheredUint32(
      device,
      graph,
      chunks,
      arrays.flatMap(values => Array.from(values))
    );
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

it.each([
  'no-chunks',
  'empty-chunk',
  'singleton'
] as const)('binds and reduces a %s uint32 vector', async shape => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({data: new Uint32Array([314159]), usage: Buffer.COPY_SRC});
  const backing = importGraphBuffer(graph, 'source', buffer, 'uint32', 1);
  const empty = graph.createDataView(backing.buffer, {format: 'uint32', length: 0});
  try {
    await expectGatheredUint32(
      device,
      graph,
      shape === 'singleton' ? [backing] : shape === 'empty-chunk' ? [empty] : [],
      shape === 'singleton' ? [314159] : []
    );
  } finally {
    buffer.destroy();
  }
});

it('preserves repeated shared-buffer chunks in reversed physical-offset order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    data: new Uint32Array([91, 17, 29, 43, 59, 71, 83, 97]),
    usage: Buffer.COPY_SRC
  });
  const backing = importGraphBuffer(graph, 'source', buffer, 'uint32');
  const higherOffset = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 20
  });
  const lowerOffset = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 3,
    byteOffset: 4
  });
  const empty = graph.createDataView(backing.buffer, {format: 'uint32', length: 0});
  try {
    await expectGatheredUint32(
      device,
      graph,
      [higherOffset, empty, lowerOffset, higherOffset],
      [71, 83, 17, 29, 43, 71, 83]
    );
  } finally {
    buffer.destroy();
  }
});

it.each([
  'uint32x2',
  'float32x2'
] as const)('copies %s chunks bit for bit at nonzero offsets and refreshes transient scratch on each encoding', async format => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Includes float32 NaN payloads, negative zero, subnormal and infinity bit patterns.
  const words = new Uint32Array([
    99, 0x7fc01234, 0x80000000, 1, 0x7f800000, 98, 97, 0xffffffff, 42, 0xff800000, 0x3f800000, 96
  ]);
  const sourceBuffer = device.createBuffer({
    data: words,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const outputBuffer = createOutputBuffer(device, 8);
  const graph = new GPUCommandGraph(device);
  const backing = importGraphBuffer(graph, 'source', sourceBuffer, 'uint32', words.length);
  const first = graph.createDataView(backing.buffer, {format, length: 2, byteOffset: 4});
  const empty = graph.createDataView(backing.buffer, {format, length: 0});
  const second = graph.createDataView(backing.buffer, {format, length: 2, byteOffset: 28});
  const source = new GraphVectorView({
    id: 'chunks',
    name: 'chunks',
    format,
    length: 4,
    valueLength: 8,
    stride: 2,
    byteStride: 8,
    rowByteLength: 8,
    data: [first, empty, second]
  });
  const {view, nodes} = createGraphChunkGather(graph, {id: 'gather', source});
  for (const node of nodes) graph.add(node);
  const output = importGraphBuffer(graph, 'output', outputBuffer, format, 4);
  graph.addCopyPass({
    id: 'consume-packed',
    resources: [
      {buffer: view, usage: 'copy-source'},
      {buffer: output, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(view),
          sourceOffset: view.byteOffset,
          destinationBuffer: getBuffer(output),
          destinationOffset: output.byteOffset,
          size: 32
        })
    })
  });
  const compiled = graph.compile();
  try {
    const expected = [...words.slice(1, 5), ...words.slice(7, 11)];
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer)).toEqual(expected);
    sourceBuffer.write(new Uint32Array([123456789]), 4);
    expected[0] = 123456789;
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer)).toEqual(expected);
  } finally {
    compiled.destroy();
    sourceBuffer.destroy();
    outputBuffer.destroy();
  }
});

it('gathers two million scalar rows exactly against CPU concatenation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const lengths = [1_000_003, 999_997];
  let firstRow = 0;
  const arrays = lengths.map(length => {
    const values = Uint32Array.from({length}, (_, index) => (firstRow + index) ^ 0xa5a5a5a5);
    firstRow += length;
    return values;
  });
  const buffers = arrays.map(data => device.createBuffer({data, usage: Buffer.COPY_SRC}));
  const chunks = buffers.map((buffer, index) =>
    importGraphBuffer(graph, `source-${index}`, buffer, 'uint32', lengths[index])
  );
  const source = new GraphVectorView({
    id: 'large-chunks',
    name: 'large-chunks',
    format: 'uint32',
    length: firstRow,
    valueLength: firstRow,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
  const {view, nodes} = createGraphChunkGather(graph, {id: 'large-gather', source});
  for (const node of nodes) graph.add(node);
  const outputBuffer = createOutputBuffer(device, firstRow);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', firstRow);
  graph.addCopyPass({
    id: 'consume-packed',
    resources: [
      {buffer: view, usage: 'copy-source'},
      {buffer: output, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(view),
          sourceOffset: 0,
          destinationBuffer: getBuffer(output),
          destinationOffset: 0,
          size: firstRow * 4
        })
    })
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const bytes = await outputBuffer.readAsync();
    const actual = new Uint32Array(bytes.buffer, bytes.byteOffset, firstRow);
    const expected = new Uint32Array(firstRow);
    expected.set(arrays[0], 0);
    expected.set(arrays[1], lengths[0]);
    // Compare every word without expanding millions of entries in the assertion machinery.
    expect(actual.findIndex((value, index) => value !== expected[index])).toBe(-1);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    outputBuffer.destroy();
  }
});
