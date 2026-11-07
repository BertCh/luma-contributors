// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUSort,
  GraphVectorView,
  type GPUSortAlgorithm,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makePseudoRandomUint32} from './gpu-primitive-test-utils';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;

it('GPUSort merges many chunks through a balanced tree and honors output topology', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const chunkLengths of [
    [10],
    [300, 5],
    [7, 0, 513, 64],
    [100, 100, 100, 100, 100],
    Array.from({length: 17}, (_, index) => 20 + index * 13)
  ]) {
    const length = chunkLengths.reduce((total, chunkLength) => total + chunkLength, 0);
    // Few distinct keys make most comparisons ties, which exercises merge stability.
    const keys = makePseudoRandomUint32(length, length + 1).map(key => key % 41);
    for (const direction of ['ascending', 'descending'] as const) {
      for (const algorithm of ['radix', 'bitonic', 'onesweep'] as const) {
        for (const outputLengths of [chunkLengths, [length], [1, length - 1]]) {
          const result = await runChunkedSort(device, keys, chunkLengths, outputLengths, {
            direction,
            algorithm
          });
          const expected = getStableSort(keys, direction);
          const label = `${chunkLengths.length} chunks ${direction} ${algorithm} outputs ${outputLengths.length}`;
          expect(result.keys, `${label} keys`).toEqual(expected.keys);
          expect(result.values, `${label} stability`).toEqual(expected.values);
        }
      }
    }
  }
}, 180_000);

it('GPUSort chunk merge respects keyBits across chunk boundaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const keys = makePseudoRandomUint32(3000, 8);
  const chunkLengths = [1000, 1000, 1000];
  const result = await runChunkedSort(device, keys, chunkLengths, chunkLengths, {
    algorithm: 'radix',
    keyBits: 10
  });
  const expected = getStableSort(
    keys.map(key => key & 0x3ff),
    'ascending'
  );
  expect(result.values).toEqual(expected.values);
  expect(result.keys).toEqual(expected.values.map(index => keys[index]));
});

function getStableSort(keys: Uint32Array, direction: 'ascending' | 'descending') {
  const indices = Array.from(keys, (_, index) => index);
  indices.sort((left, right) => {
    if (keys[left] === keys[right]) return left - right;
    return direction === 'ascending' ? keys[left] - keys[right] : keys[right] - keys[left];
  });
  return {keys: indices.map(index => keys[index]), values: indices};
}

async function runChunkedSort(
  device: Device,
  keys: Uint32Array,
  inputLengths: readonly number[],
  outputLengths: readonly number[],
  props: {direction?: 'ascending' | 'descending'; algorithm: GPUSortAlgorithm; keyBits?: number}
): Promise<{keys: number[]; values: number[]}> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputBuffers = new Map<string, Buffer[]>();
  const makeVector = (
    name: string,
    lengths: readonly number[],
    source?: Uint32Array
  ): GraphVectorView<'uint32'> => {
    outputBuffers.set(name, []);
    let offset = 0;
    const data: GraphDataView<'uint32'>[] = lengths.map((chunkLength, chunkIndex) => {
      const buffer = device.createBuffer({
        data: source ? source.slice(offset, offset + chunkLength) : undefined,
        byteLength: Math.max(chunkLength, 1) * 4,
        usage: STORAGE_USAGE
      });
      offset += chunkLength;
      buffers.push(buffer);
      outputBuffers.get(name)!.push(buffer);
      return graph.createDataView(
        graph.importBuffer(
          {id: `${name}-${chunkIndex}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format: 'uint32', length: chunkLength}
      );
    });
    const length = lengths.reduce((total, chunkLength) => total + chunkLength, 0);
    return new GraphVectorView({
      id: name,
      name,
      format: 'uint32',
      length,
      valueLength: length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data
    });
  };
  const values = Uint32Array.from(keys, (_, index) => index);
  const outputKeys = makeVector('output-keys', outputLengths);
  const outputValues = makeVector('output-values', outputLengths);
  graph.add(
    new GPUSort({
      ...props,
      keys: makeVector('keys', inputLengths, keys),
      values: makeVector('values', inputLengths, values),
      outputKeys,
      outputValues
    })
  );
  const compiled = graph.compile();
  try {
    const commandEncoder = device.createCommandEncoder({id: 'chunked-sort-test'});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const read = async (vector: GraphVectorView<'uint32'>) => {
      const result: number[] = [];
      for (const [chunkIndex, chunk] of vector.data.entries()) {
        if (chunk.length === 0) continue;
        const bytes = await outputBuffers.get(vector.name)![chunkIndex].readAsync();
        result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
      }
      return result;
    };
    return {keys: await read(outputKeys), values: await read(outputValues)};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
