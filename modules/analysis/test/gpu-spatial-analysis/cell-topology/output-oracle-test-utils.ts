// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Poisoned, guarded views at distinct offsets beyond 256 bytes, aligned to their row formats. */
export function createOutputOracleFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const sentinel = 0xa5a5a5a5;
  const allocations: {
    buffer: Buffer;
    prefix: number;
    length: number;
    words: number;
    produced: boolean;
    input: boolean;
  }[] = [];
  const views = new Map<GraphDataView, (typeof allocations)[number]>();
  function allocate<Format extends GraphDataView['format']>(
    name: string,
    format: Format,
    length: number,
    data?: Float32Array | Uint32Array,
    preserveOnReplay = false
  ) {
    const components = Number(format.split('x')[1] ?? 1);
    const prefix = Math.ceil((67 + allocations.length * 19) / components) * components;
    const words = length * components;
    const values = new Uint32Array(prefix + words + 17).fill(sentinel);
    if (data) values.set(new Uint32Array(data.buffer, data.byteOffset, data.length), prefix);
    const buffer = createOutputBuffer(device, values.length);
    buffer.write(values);
    const backing = importGraphBuffer(graph, name, buffer, 'uint32', values.length);
    const view = graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
    const allocation = {
      buffer,
      prefix,
      length: values.length,
      words,
      produced: data === undefined && !preserveOnReplay,
      input: data !== undefined
    };
    allocations.push(allocation);
    views.set(view, allocation);
    return view;
  }
  async function read(view: GraphDataView) {
    const {buffer, prefix, words} = views.get(view)!;
    // Integer vector rows contain integer words, including IDs whose bits are NaNs as f32.
    const values = await (view.format.startsWith('uint32')
      ? readUint32(buffer)
      : readFloat32(buffer));
    return values.slice(prefix, prefix + words);
  }
  async function readRows(chunks: readonly GraphDataView[]) {
    return (await Promise.all(chunks.map(read))).flat();
  }
  async function submitTwice<Result>(
    compiled: ReturnType<typeof graph.compile>,
    readResult: () => Promise<Result>
  ): Promise<Result> {
    let first: Result | undefined;
    for (let submission = 0; submission < 2; submission++) {
      for (const {buffer, prefix, words, produced} of allocations) {
        if (produced) buffer.write(new Uint32Array(words).fill(sentinel), prefix * 4);
      }
      submitGraph(device, compiled, undefined);
      const result = await readResult();
      for (const {buffer, prefix, words, length} of allocations) {
        const bits = await readUint32(buffer);
        expect(bits.slice(0, prefix)).toEqual(Array(prefix).fill(sentinel));
        expect(bits.slice(prefix + words)).toEqual(Array(length - prefix - words).fill(sentinel));
      }
      if (submission === 0) first = result;
      else expect(result).toEqual(first);
    }
    return first!;
  }
  return {
    graph,
    input: <Format extends GraphDataView['format']>(
      name: string,
      format: Format,
      data: Float32Array | Uint32Array
    ) => allocate(name, format, data.length / Number(format.split('x')[1] ?? 1), data),
    output: <Format extends GraphDataView['format']>(
      name: string,
      format: Format,
      length: number,
      options?: {preserveOnReplay?: boolean}
    ) => allocate(name, format, length, undefined, options?.preserveOnReplay),
    writeInput: (view: GraphDataView, data: Float32Array | Uint32Array) => {
      const allocation = views.get(view);
      if (!allocation?.input || data.byteLength !== allocation.words * 4)
        throw new Error('writeInput requires a registered input view and the same byte length');
      allocation.buffer.write(data, allocation.prefix * 4);
    },
    read,
    readRows,
    submitTwice,
    destroy: () => {
      for (const {buffer} of allocations) buffer.destroy();
    }
  };
}
