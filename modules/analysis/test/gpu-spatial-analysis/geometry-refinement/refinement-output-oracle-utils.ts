// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {OperationGraphInput, OperationGraphRows} from '../../utils/mixed-execution-test-utils';

/** Every output begins poisoned, including padding and optional scalar destinations. */
export function createRefinementOracleHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  function view<Format extends GPUGraphVectorFormat>(
    format: Format,
    length: number,
    values?: readonly number[]
  ): GraphDataView<Format> {
    const components = Number(format.match(/x(\d)$/)?.[1] ?? 1);
    const ArrayType = format.startsWith('float') ? Float32Array : Uint32Array;
    // A logically empty view still needs one row of backing: kernels bind at least rowByteLength.
    const data = new ArrayType(Math.max(components, length * components, values?.length ?? 0)).fill(
      913
    );
    if (values) data.set(values);
    // Every view can be read back by this harness. Input-only buffers omit COPY_SRC, which
    // makes readAsync's staging copy invalid on WebGPU and returns unusable zero readbacks.
    const buffer = createOutputBuffer(device, data.length);
    buffer.write(data);
    const result = importGraphBuffer(graph, `oracle-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result;
  }
  async function read(value: GraphDataView): Promise<number[]> {
    const components = (value.length * value.rowByteLength) / 4;
    return value.format.startsWith('float')
      ? readFloat32(buffers.get(value)!, components)
      : readUint32(buffers.get(value)!, components);
  }
  function write(value: GraphDataView, values: readonly number[]) {
    const ArrayType = value.format.startsWith('float') ? Float32Array : Uint32Array;
    buffers.get(value)!.write(ArrayType.from(values));
  }
  function poison(value: GraphDataView) {
    write(value, Array((value.length * value.rowByteLength) / 4).fill(913));
  }
  return {
    graph,
    view,
    read,
    write,
    poison,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

export function executeCPUOracleOperation(
  operation: {execute: (context: never) => unknown},
  inputs: Record<string, OperationGraphInput>
) {
  const vectors = Object.fromEntries(
    Object.entries(inputs).map(([name, input]) => {
      const components = Number(input.format.match(/x(\d)$/)?.[1] ?? 1);
      const length = input.values.length / components;
      return [
        name,
        {
          format: input.format,
          length,
          chunks: [{values: input.values, length}],
          pack: () => input.values
        }
      ];
    })
  );
  return operation.execute({inputs: vectors} as never) as Promise<
    Record<string, OperationGraphRows>
  >;
}

/** Lower the operation onto explicitly poisoned caller-owned output views. */
export async function executeGPUOracleOperation(
  device: Device,
  operation: {
    outputs: Record<string, {format: GPUGraphVectorFormat; length: number}>;
    createContributor: (
      resolve: <Format extends GPUGraphVectorFormat>(name: string) => GraphDataView<Format>
    ) => Parameters<GPUCommandGraph['add']>[0];
  },
  inputs: Record<string, OperationGraphInput>,
  assertRows: (rows: Record<string, number[]>) => void
) {
  const harness = createRefinementOracleHarness(device);
  const views: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(inputs)) {
    const components = Number(input.format.match(/x(\d)$/)?.[1] ?? 1);
    views[name] = harness.view(
      input.format,
      input.values.length / components,
      Array.from(input.values)
    );
  }
  for (const [name, output] of Object.entries(operation.outputs))
    views[name] = harness.view(output.format, output.length);
  harness.graph.add(
    operation.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    )
  );
  const compiled = harness.graph.compile();
  try {
    const result: Record<string, number[]> = {};
    // Assert every submission independently. Re-poisoning rejects a contributor that leaves
    // correct-but-stale rows in place instead of writing the second submission's result.
    for (let submission = 0; submission < 2; submission++) {
      if (submission > 0)
        for (const name of Object.keys(operation.outputs)) harness.poison(views[name]);
      submitGraph(device, compiled);
      for (const name of Object.keys(operation.outputs))
        result[name] = await harness.read(views[name]);
      assertRows(result);
    }
    return result;
  } finally {
    compiled.destroy();
    harness.destroy();
  }
}
