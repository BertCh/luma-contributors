// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

/** Small real-GPU harness. State buffers support upload and readback; graph compilation is deferred. */
export function createFittingTestGraph(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const bufferByView = new Map<GraphDataView, Buffer>();
  const compiledGraphs: CompiledGPUCommandGraph[] = [];
  const floats = (name: string, values: readonly number[] | Float32Array, padding = 0) => {
    const storage = new Float32Array(Math.max(1, values.length + 2 * padding)).fill(12345);
    storage.set(values, padding);
    const buffer = device.createBuffer({
      data: storage,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const wholeView = importGraphBuffer(graph, name, buffer, 'float32');
    const view = graph.createDataView(wholeView.buffer, {
      format: 'float32',
      length: values.length,
      byteOffset: padding * 4
    });
    bufferByView.set(view, buffer);
    return view;
  };
  const uints = (name: string, values: readonly number[] | Uint32Array, padding = 0) => {
    const storage = new Uint32Array(Math.max(1, values.length + 2 * padding)).fill(0xdeadbeef);
    storage.set(values, padding);
    const buffer = device.createBuffer({
      data: storage,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const wholeView = importGraphBuffer(graph, name, buffer, 'uint32');
    const view = graph.createDataView(wholeView.buffer, {
      format: 'uint32',
      length: values.length,
      byteOffset: padding * 4
    });
    bufferByView.set(view, buffer);
    return view;
  };
  return {
    graph,
    floats,
    uints,
    readFloats: async (view: ReturnType<typeof floats>) =>
      (await readFloat32(bufferByView.get(view)!)).slice(
        view.byteOffset / 4,
        view.byteOffset / 4 + view.length
      ),
    readUints: async (view: ReturnType<typeof uints>) =>
      (await readUint32(bufferByView.get(view)!)).slice(
        view.byteOffset / 4,
        view.byteOffset / 4 + view.length
      ),
    readFloatStorage: (view: ReturnType<typeof floats>) => readFloat32(bufferByView.get(view)!),
    run() {
      const compiled = graph.compile();
      compiledGraphs.push(compiled);
      submitGraph(device, compiled, undefined);
    },
    destroy() {
      for (const compiled of compiledGraphs) compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
