// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUGraph, GPUCPUOperation, GPUCopyOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {GPUVector, CPUVector, CPUData} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('review B2 public GPU stage rejects a resident strided input before execution', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({
    data: Float32Array.of(99, 1, 2, 99, 3, 4, 99),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const input = new GPUVector({
    type: 'buffer',
    name: 'resident-strided',
    buffer,
    format: 'float32x2',
    length: 2,
    byteOffset: 4,
    byteStride: 12,
    ownsBuffer: false
  });
  const graph = new GPUGraph();
  graph.addInput('input', gpuGraph.vector('float32x2', {length: 2}));
  graph.add(new GPUCopyOperation({input: 'input', output: 'result'}));
  const compiled = graph.compile(device, {outputs: ['result']});
  try {
    expect(() => compiled.prepare({inputs: {input}})).toThrow(
      /resident-strided.*packed 32-bit numeric chunks/
    );
  } finally {
    compiled.destroy();
    input.destroy();
    buffer.destroy();
    device.destroy();
  }
});

it('review B2 strided CPU control reaches a public callback as packed unequal chunks', async () => {
  const device = createNullWebGPUDevice();
  const storage = [Float32Array.of(99, 1, 2, 99, 99, 3, 4, 99), Float32Array.of(99, 5, 6, 99)];
  const control = new CPUVector({
    format: 'float32x2',
    data: storage.map(
      (buffer, index) =>
        new CPUData({
          buffer,
          format: 'float32x2',
          length: index === 0 ? 2 : 1,
          byteOffset: 4,
          byteStride: 16
        })
    )
  });
  const graph = new GPUGraph();
  const descriptor = gpuGraph.vector('float32x2', {length: 3, chunkLengths: [2, 1]});
  graph.addInput('control', descriptor);
  let called = false;
  graph.add(
    new GPUCPUOperation({
      id: 'control',
      inputs: {rows: 'control'},
      outputs: {result: descriptor},
      execute({inputs}) {
        const rows = inputs['rows'];
        if (typeof rows === 'number') throw new Error('Expected rows');
        called = true;
        expect(rows.chunks.map(chunk => chunk.length)).toEqual([2, 1]);
        expect(rows.chunks.map(chunk => Array.from(chunk.values))).toEqual([
          [1, 2, 3, 4],
          [5, 6]
        ]);
        expect(rows.chunks[0].values.buffer).not.toBe(storage[0].buffer);
        return {result: {chunks: rows.chunks.map(chunk => ({values: chunk.values.slice()}))}};
      }
    })
  );
  const compiled = graph.compile(device, {outputs: ['result'], outputLocations: {result: 'cpu'}});
  try {
    await compiled.run({inputs: {control}});
    expect(called).toBe(true);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});
