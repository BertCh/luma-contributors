// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Minimal shared numerical harness: compilation, submission and readback are test-owned. */
export function createOptimizationHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<GraphDataView, Buffer>();
  const inputs = new Map<GraphDataView, Buffer>();
  const input = <Format extends GPUVectorFormat>(
    format: Format,
    values: Float32Array | Uint32Array
  ): GraphDataView<Format> => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      values.byteLength / getGPUVectorFormatInfo(format).byteLength
    );
    inputs.set(view, buffer);
    return view;
  };
  const output = <Format extends GPUVectorFormat>(
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const buffer = createOutputBuffer(
      device,
      (Math.max(1, length) * getGPUVectorFormatInfo(format).byteLength) / 4
    );
    buffers.push(buffer);
    const view = importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length);
    outputs.set(view, buffer);
    return view;
  };
  return {
    input,
    output,
    rewrite(view: GraphDataView, values: Float32Array | Uint32Array) {
      inputs.get(view)!.write(values);
    },
    async run(
      producer: GPUCommandNodeProducer,
      repeat?: (read: (view: GraphDataView) => number[]) => void
    ) {
      graph.add(producer);
      const compiled = graph.compile();
      try {
        const result = new Map<GraphDataView, number[]>();
        for (let encoding = 0; encoding < (repeat ? 2 : 1); encoding++) {
          submitGraph(device, compiled, undefined);
          for (const [view, buffer] of outputs) {
            const count = (view.length * getGPUVectorFormatInfo(view.format).byteLength) / 4;
            const values = view.format.startsWith('float')
              ? await readFloat32(buffer, count)
              : await readUint32(buffer, count);
            result.set(view, Array.from(values));
          }
          if (encoding === 0 && repeat) repeat(view => result.get(view)!);
        }
        return (view: GraphDataView): number[] => result.get(view)!;
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
  };
}
