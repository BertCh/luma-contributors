// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Poison every declared operation output before its contributor runs, including zero results. */
export function poisonOperationOutputs<Operation extends GPUContributorOperation>(
  operation: Operation
): Operation {
  const createContributor = operation.createContributor.bind(operation);
  operation.createContributor = getView => {
    const contributor = createContributor(getView) as {
      getCommandNodes<Parameters>(
        graph: GPUCommandGraph<Parameters>
      ): readonly GPUCommandNode<Parameters>[];
    };
    return {
      getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
        ...Object.keys(operation.outputs).map(port => {
          const view = getView(port);
          return createFillNode<Parameters>(graph, {
            id: `${operation.id}-poison-${port}`,
            operation: 'output-oracle-poison',
            view,
            type: view.format.startsWith('float32') ? 'f32' : 'u32',
            value: view.format.startsWith('float32') ? '913.0' : '913u'
          });
        }),
        ...contributor.getCommandNodes(graph)
      ]
    };
  };
  return operation;
}

/** Small caller-owned, poisoned output buffers for independent network output oracles. */
export function createOracleContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const byView = new Map<GraphDataView, Buffer>();
  const input = <Format extends 'uint32' | 'float32' | 'uint32x2' | 'float32x2'>(
    name: string,
    format: Format,
    values: number[]
  ): GraphDataView<Format> => {
    const buffer = device.createBuffer({
      data: format.startsWith('float32') ? new Float32Array(values) : new Uint32Array(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format.endsWith('x2') ? 2 : 1)
    );
    byView.set(view, buffer);
    return view;
  };
  const output = <Format extends 'uint32' | 'float32' | 'uint32x2' | 'float32x2'>(
    name: string,
    format: Format,
    length = 1
  ) => input(name, format, Array(length * (format.endsWith('x2') ? 2 : 1)).fill(913));
  const read = (view: GraphDataView) =>
    view.format.startsWith('float32')
      ? readFloat32(byView.get(view)!)
      : readUint32(byView.get(view)!);
  const write = (view: GraphDataView, values: number[]) =>
    byView
      .get(view)!
      .write(
        view.format.startsWith('float32') ? new Float32Array(values) : new Uint32Array(values)
      );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    read,
    write,
    run() {
      compiled ??= graph.compile();
      submitGraph(device, compiled);
    },
    destroy() {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
