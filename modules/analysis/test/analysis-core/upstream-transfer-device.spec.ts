// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, GPUCPUOperation, GPUCopyOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {GPUData, GPUVector, CPUData, CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('LUMA-114 packed GPU boundary with strided CPU control preserves chunks and snapshots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const byteStride = 8;
  // Same three rows in both layouts: (1,2), (3,4), (5,6), split into unequal chunks.
  // A prefix, a tail and (in the control) two padding words per row must remain untouched.
  const storage = [
    [99, 1, 2, 3, 4, 99],
    [99, 5, 6, 99]
  ].map(values => Float32Array.from(values));
  const controlStorage = [
    Float32Array.of(99, 1, 2, 99, 99, 3, 4, 99),
    Float32Array.of(99, 5, 6, 99)
  ];
  const control = new CPUVector({
    format: 'float32x2',
    data: controlStorage.map(
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
  const buffers = storage.map(data =>
    device.createBuffer({data, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST})
  );
  const chunks = buffers.map(
    (buffer, index) =>
      new GPUData({
        buffer,
        format: 'float32x2',
        length: index === 0 ? 2 : 1,
        byteOffset: 4,
        byteStride,
        ownsBuffer: false
      })
  );
  const input = new GPUVector({
    type: 'data',
    name: 'input',
    format: 'float32x2',
    data: chunks,
    ownsData: false
  });
  const graph = new GPUGraph();
  const descriptor = gpuGraph.vector('float32x2', {length: 3, chunkLengths: [2, 1]});
  graph.addInput('input', descriptor);
  graph.addInput('control', descriptor);
  graph.add(new GPUCopyOperation({input: 'input', output: 'gpu-stage'}));
  const snapshots: Float32Array[][] = [];
  const returnedArrays: Float32Array[][] = [];
  const controlSnapshots: (Float32Array | Int32Array | Uint32Array)[][] = [];
  graph.add(
    new GPUCPUOperation({
      id: 'snapshot',
      inputs: {rows: 'input', copied: 'gpu-stage', control: 'control'},
      outputs: {middle: descriptor},
      execute({inputs}) {
        const rows = inputs['rows'];
        const copied = inputs['copied'];
        const cpuControl = inputs['control'];
        if (
          typeof rows === 'number' ||
          typeof copied === 'number' ||
          typeof cpuControl === 'number'
        )
          throw new Error('Expected vector inputs');
        const expected =
          snapshots.length === 0
            ? [
                [1, 2, 3, 4],
                [5, 6]
              ]
            : [
                [11, 12, 13, 14],
                [15, 16]
              ];
        for (const vector of [rows, copied, cpuControl]) {
          expect(vector.chunks.map(chunk => chunk.length)).toEqual([2, 1]);
          expect(vector.chunks.map(chunk => Array.from(chunk.values))).toEqual(expected);
        }
        const snapshot = rows.chunks.map(chunk => {
          expect(chunk.values).toBeInstanceOf(Float32Array);
          return chunk.values as Float32Array;
        });
        snapshots.push(snapshot);
        controlSnapshots.push(cpuControl.chunks.map(chunk => chunk.values));
        const result = snapshot.map(values => values.slice());
        returnedArrays.push(result);
        return {middle: {chunks: result.map(values => ({values}))}};
      }
    })
  );
  graph.add(new GPUCopyOperation({input: 'middle', output: 'result'}));
  const compiled = graph.compile(device, {outputs: ['result']});
  try {
    for (let iteration = 0; iteration < 2; iteration++) {
      const outputs = await compiled.run({inputs: {input, control}});
      const result = outputs['result'];
      try {
        expect(result.data.map(chunk => chunk.length)).toEqual([2, 1]);
        const readbacks = [];
        for (const chunk of result.data) {
          const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 8);
          readbacks.push(
            Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length * 2))
          );
        }
        expect(readbacks).toEqual(
          iteration === 0
            ? [
                [1, 2, 3, 4],
                [5, 6]
              ]
            : [
                [11, 12, 13, 14],
                [15, 16]
              ]
        );
      } finally {
        result.destroy();
      }
      // Once run settles, callback output arrays may be reused. They must not alias input snapshots.
      for (const values of returnedArrays[iteration]) values.fill(-100);
      for (const [chunkIndex, values] of storage.entries()) {
        const bytes = await buffers[chunkIndex].readAsync();
        expect(Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, values.length))).toEqual(
          Array.from(values)
        );
        if (iteration === 0) {
          for (let row = 0; row < chunks[chunkIndex].length; row++) {
            const offset = 1 + (row * byteStride) / 4;
            values[offset] += 10;
            values[offset + 1] += 10;
          }
          buffers[chunkIndex].write(values);
        }
      }
      // The CPU strided control is borrowed storage; only payload words change between runs.
      for (const [chunkIndex, values] of controlStorage.entries()) {
        const expected = chunkIndex === 0 ? [99, 1, 2, 99, 99, 3, 4, 99] : [99, 5, 6, 99];
        if (iteration === 1)
          for (const index of chunkIndex === 0 ? [1, 2, 5, 6] : [1, 2]) expected[index] += 10;
        expect(Array.from(values)).toEqual(expected);
        if (iteration === 0)
          for (const index of chunkIndex === 0 ? [1, 2, 5, 6] : [1, 2]) values[index] += 10;
      }
      expect(controlSnapshots[0].map(values => Array.from(values))).toEqual([
        [1, 2, 3, 4],
        [5, 6]
      ]);
      expect(snapshots[0].map(values => Array.from(values))).toEqual([
        [1, 2, 3, 4],
        [5, 6]
      ]);
    }
  } finally {
    compiled.destroy();
    input.destroy();
    for (const chunk of chunks) chunk.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
