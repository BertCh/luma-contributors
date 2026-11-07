// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma-contributors/graph/gpu-core';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export function createRemoteSensingHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, {buffer: Buffer; length: number; integer: boolean}>();
  let serial = 0;
  const importView = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: `remote-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length});
  };
  const input = (values: number[]) =>
    importView(createInputBuffer(device, Float32Array.from(values)), 'float32', values.length);
  const mask = (values: number[]) =>
    importView(createInputBuffer(device, Uint32Array.from(values)), 'uint32', values.length);
  const output = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const scalarCount = length * (format === 'float32x2' ? 2 : 1);
    const buffer = createOutputBuffer(device, scalarCount);
    outputs.set(name, {buffer, length: scalarCount, integer: format === 'uint32'});
    return importView(buffer, format, length);
  };
  return {
    graph,
    input,
    mask,
    output,
    async run(contributor: GPUCommandNodeProducer) {
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const result: Record<string, number[]> = {};
        for (const [name, entry] of outputs)
          result[name] = await (entry.integer
            ? readUint32(entry.buffer, entry.length)
            : readFloat32(entry.buffer, entry.length));
        return result;
      } finally {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  };
}
