// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

/** All caller outputs start poisoned, including zero-valued status words. */
export function createOracleFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  function create<Format extends GPUVectorFormat>(format: Format, values: number[]) {
    const components = Number(format.match(/x(\d)$/)?.[1] ?? 1);
    const buffer = device.createBuffer({
      data: format.startsWith('float') ? Float32Array.from(values) : Uint32Array.from(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      view: importGraphBuffer(
        graph,
        `oracle-${buffers.length}`,
        buffer,
        format,
        values.length / components
      ),
      read: () => (format.startsWith('float') ? readFloat32(buffer) : readUint32(buffer)),
      poison: () =>
        buffer.write(
          format.startsWith('float')
            ? new Float32Array(values.length).fill(12345)
            : new Uint32Array(values.length).fill(12345)
        )
    };
  }
  return {
    graph,
    input: <Format extends GPUVectorFormat>(format: Format, values: number[]) =>
      create(format, values).view,
    output: <Format extends GPUVectorFormat>(format: Format, length: number) =>
      create(format, Array(length * Number(format.match(/x(\d)$/)?.[1] ?? 1)).fill(12345)),
    destroy: () => {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Exercise an operation's deferred contributor with caller-owned poisoned destinations. */
export function addOracleOperation(
  fixture: ReturnType<typeof createOracleFixture>,
  operation: GPUContributorOperation,
  inputs: Record<string, GraphDataView>
) {
  const outputs = Object.fromEntries(
    Object.entries(operation.outputs).map(([name, shape]) => [
      name,
      fixture.output(shape.format, shape.length)
    ])
  );
  const views = {
    ...inputs,
    ...Object.fromEntries(Object.entries(outputs).map(([name, output]) => [name, output.view]))
  };
  fixture.graph.add(operation.createContributor(name => views[name] as never));
  return outputs;
}
