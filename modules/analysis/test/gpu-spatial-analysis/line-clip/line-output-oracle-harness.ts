// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** All outputs start at 93, so an unwritten zero diagnostic fails. */
export function createLineOracleHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number,
    values?: number[]
  ): GraphDataView<Format> => {
    const components = format === 'float32x2' ? 2 : 1;
    const data = values ?? new Array<number>(length * components).fill(93);
    // Every view may be read back by an oracle. Input-only buffer helpers omit
    // COPY_SRC, making WebGPU's staging-buffer copy invalid and returning zeros.
    const buffer = createOutputBuffer(device, length * components);
    buffer.write(format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    const result = importGraphBuffer(graph, `oracle-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result;
  };
  return {
    graph,
    view,
    write: (target: GraphDataView, values: Uint32Array | Float32Array) =>
      buffers.get(target)!.write(values),
    read: (output: GraphDataView, length = output.length) =>
      output.format === 'uint32'
        ? readUint32(buffers.get(output)!, length)
        : readFloat32(buffers.get(output)!, length * (output.format === 'float32x2' ? 2 : 1)),
    run: async () => {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      // Readback by the caller synchronizes the submission before destroy().
      return () => compiled.destroy();
    },
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}
