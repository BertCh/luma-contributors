// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../src/analysis-core';
import {createGraphChunkGather} from '../../src/analysis-core/kernels/graph-chunk-gather';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSegmentRingAssembly} from '../../src/analysis-core/ring-assembly/gpu-segment-ring-assembly';
import {
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../src/testing/gpu-contributor-test-utils';
import {expect, it} from 'vitest';
import {
  getGPUResourceCounts,
  getGPUResourceLeaks,
  validateNoGPUResourceLeaks
} from '../../src/testing/gpu-resource-counts';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createResourceLifecycleFixture} from './gpu-resource-lifecycle.test-utils';

it('releases graph scratch, parameter uniforms and caller buffers for three core contributors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = async () => {
    const {test, rowCount, sums, order} = createResourceLifecycleFixture(device);
    try {
      test.run();
      // cos(0) = 1, sin(0) = 0 for every row; sum is an independent closed form.
      expect(await test.readFloats(sums)).toEqual([rowCount, 0]);
      expect((await test.readUints(order)).sort((left, right) => left - right)).toEqual(
        Array.from({length: rowCount}, (_, row) => row)
      );
    } finally {
      test.destroy();
    }
  };
  await run(); // Warm the shared device and pipeline caches before taking the global baseline.
  const baseline = getGPUResourceCounts();
  for (let repetition = 0; repetition < 3; repetition++) {
    await run();
    validateNoGPUResourceLeaks(`core contributors repetition ${repetition}`, baseline);
  }
});

it('tracks real textures in the luma 10 counter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const baseline = getGPUResourceCounts();
  const texture = device.createTexture({width: 1, height: 1, format: 'rgba8unorm'});
  try {
    expect(getGPUResourceLeaks(baseline)).toMatchObject({textures: 1});
  } finally {
    texture.destroy();
  }
  validateNoGPUResourceLeaks('texture', baseline);
});

it('assembles a counter-clockwise unit square with independent status and hole outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const graph = new GPUCommandGraph(device);
    const input = device.createBuffer({
      data: new Float32Array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]),
      usage: Buffer.STORAGE
    });
    buffers.push(input);
    const outputView = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const offsets = outputView('ring-offsets', 2);
    const positions = outputView('positions', 10);
    const hole = outputView('hole', 1);
    const count = outputView('count', 1);
    const overflow = outputView('overflow', 1);
    const props = {
      endpoints: importGraphBuffer(graph, 'endpoints', input, 'float32x4', 4),
      output: {
        positions: graph.createDataView(positions.view.buffer, {format: 'float32x2', length: 5}),
        ringOffsets: offsets.view,
        ringIsHole: hole.view,
        status: {count: count.view, overflow: overflow.view}
      }
    };
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...props,
          output: {...props.output, ringIsHole: count.view}
        })
    ).toThrow(/overlap/);
    graph.add(new GPUSegmentRingAssembly(props));
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(count.buffer)).toEqual([1]);
    expect(await readUint32(hole.buffer)).toEqual([0]);
    expect(await readUint32(overflow.buffer)).toEqual([0]);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('runs equivalent kernels with Unicode binding names through the public builder', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const graph = new GPUCommandGraph(device);
    for (const [bindingIndex, name] of ['réflexion', '朝焼け'].entries()) {
      const buffer = createOutputBuffer(device, 3);
      buffers.push(buffer);
      const view = importGraphBuffer(graph, `unicode-${bindingIndex}`, buffer, 'uint32', 3);
      graph.add(
        createWGSLKernelNode(graph, {
          id: `unicode-${bindingIndex}`,
          operation: 'UnicodeBindings',
          invocationCount: 3,
          bindings: [{name, view, type: 'u32', access: 'read_write'}],
          body: `${name}[${name}Offset + index] = (index + 1u) * 13u;`
        })
      );
    }
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    for (const buffer of buffers) expect(await readUint32(buffer)).toEqual([13, 26, 39]);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('gathers uneven 255/256/257-row chunks and empty chunks bit for bit on repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const graph = new GPUCommandGraph(device);
    const lengths = [0, 255, 0, 256, 257, 0];
    const patterns = [0x80000000, 0x7fc01234, 0xff800000, 1, 0xffffffff, 0x3f800000];
    const arrays = lengths.map((length, chunkIndex) => {
      const prefix = chunkIndex + 1;
      const values = new Uint32Array(prefix + Math.max(length, 1) + 1).fill(0xdeadbeef);
      for (let row = 0; row < length; row++)
        values[prefix + row] = patterns[(row + chunkIndex) % patterns.length];
      const buffer = device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const backing = importGraphBuffer(graph, `source-${chunkIndex}`, buffer, 'uint32');
      return {
        buffer,
        values,
        prefix,
        length,
        view: graph.createDataView(backing.buffer, {
          format: 'float32',
          length,
          byteOffset: prefix * 4
        })
      };
    });
    const source = createVectorView(
      'chunks',
      'float32',
      arrays.map(chunk => chunk.view)
    );
    const {view, nodes} = createGraphChunkGather(graph, {id: 'gather-boundaries', source});
    for (const node of nodes) graph.add(node);
    const outputBuffer = createOutputBuffer(device, source.length);
    buffers.push(outputBuffer);
    const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', source.length);
    graph.addCopyPass({
      id: 'consume-gather',
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
            size: source.length * 4
          })
      })
    });
    compiled = graph.compile();
    for (let repetition = 0; repetition < 3; repetition++) {
      const expected: number[] = [];
      for (const chunk of arrays) {
        for (let row = 0; row < chunk.length; row++) {
          chunk.values[chunk.prefix + row] = (chunk.values[chunk.prefix + row] ^ repetition) >>> 0;
          expected.push(chunk.values[chunk.prefix + row]);
        }
        chunk.buffer.write(chunk.values);
      }
      submitGraph(device, compiled, undefined);
      expect(await readUint32(outputBuffer)).toEqual(expected);
    }
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
