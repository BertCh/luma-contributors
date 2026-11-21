// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Small oracle graphs use exact capacities and poison every caller-owned output. */
export function createOutputOracleFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  let compiled: ReturnType<typeof graph.compile> | undefined;
  const columns: {
    name: string;
    view: GraphDataView;
    values: Float32Array | Uint32Array;
    buffer: ReturnType<typeof createInputBuffer>;
    output: boolean;
  }[] = [];
  function column<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    output = false
  ): GraphDataView<Format> {
    const components = format === 'float32x4' ? 4 : format.endsWith('x2') ? 2 : 1;
    const buffer = output
      ? createOutputBuffer(device, values.length)
      : createInputBuffer(device, values);
    if (output) buffer.write(values);
    const view = importGraphBuffer(graph, name, buffer, format, values.length / components);
    columns.push({name, view, values, buffer, output});
    return view;
  }
  function output<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    const components = format === 'float32x4' ? 4 : format.endsWith('x2') ? 2 : 1;
    const values = format.startsWith('uint32')
      ? new Uint32Array(length * components).fill(777)
      : new Float32Array(length * components).fill(777);
    return column(name, format, values, true);
  }
  return {
    graph,
    columns,
    column,
    output,
    async runGPU() {
      compiled ??= graph.compile();
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const entry of columns.filter(entry => entry.output)) {
        result[entry.name] =
          entry.values instanceof Uint32Array
            ? await readUint32(entry.buffer, entry.values.length)
            : await readFloat32(entry.buffer, entry.values.length);
      }
      return result;
    },
    destroy() {
      compiled?.destroy();
      for (const entry of columns) entry.buffer.destroy();
    }
  };
}
