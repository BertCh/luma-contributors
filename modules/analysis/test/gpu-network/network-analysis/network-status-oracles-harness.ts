// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Test-owned allocations only: no expected-value or contributor logic. */
export function createStatusOracleHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const outputs: GraphDataView[] = [];
  const components = (format: string) =>
    format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
  const input = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    const view = importGraphBuffer(
      graph,
      `input-${buffers.size}`,
      buffer,
      format,
      values.length / components(format)
    );
    buffers.set(view, buffer);
    return view;
  };
  const output = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * components(format));
    const view = importGraphBuffer(graph, `output-${buffers.size}`, buffer, format, length);
    buffers.set(view, buffer);
    outputs.push(view);
    return view;
  };
  return {
    graph,
    input,
    output,
    rewrite(view: GraphDataView, values: Uint32Array | Float32Array) {
      buffers.get(view)!.write(values);
    },
    async run(
      producer: GPUCommandNodeProducer,
      checks: ((read: (view: GraphDataView) => number[]) => void)[]
    ) {
      graph.add(producer);
      const compiled = graph.compile();
      try {
        // Poison once: later encodings must also clear an earlier nonzero result.
        for (const view of outputs)
          buffers
            .get(view)!
            .write(
              new Uint32Array(Math.max(view.length * components(view.format), 1)).fill(0xdeadbeef)
            );
        for (const check of checks) {
          submitGraph(device, compiled, undefined);
          const result = new Map<GraphDataView, number[]>();
          for (const view of outputs)
            result.set(
              view,
              view.format.startsWith('float32')
                ? await readFloat32(buffers.get(view)!, view.length * components(view.format))
                : await readUint32(buffers.get(view)!, view.length * components(view.format))
            );
          check(view => result.get(view)!);
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers.values()) buffer.destroy();
      }
    }
  };
}
