// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export type OraclePort = {format: GPUGraphVectorFormat; length: number};
export type OracleInput = OraclePort & {values: Float32Array | Uint32Array};

/** Caller-owned buffers start poisoned on every submission, including scalar zero outputs. */
export function createPoisonedOracleGraph<Outputs extends Record<string, OraclePort>>(
  device: Device,
  inputs: Record<string, OracleInput>,
  outputs: Outputs
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Record<string, Buffer> = {};
  const descriptors = {...inputs, ...outputs};
  const views = new Map<string, GraphDataView>();
  for (const [name, port] of Object.entries(descriptors)) {
    const components = port.format.endsWith('x4') ? 4 : port.format.endsWith('x2') ? 2 : 1;
    buffers[name] =
      name in inputs
        ? createInputBuffer(device, inputs[name].values)
        : createOutputBuffer(device, Math.max(1, port.length * components));
    views.set(name, importGraphBuffer(graph, name, buffers[name], port.format, port.length));
  }
  const getView: GPUContributorOperationViewGetter = <Format extends GPUGraphVectorFormat>(
    name: string
  ): GraphDataView<Format> => {
    const view = views.get(name);
    if (!view) throw new Error(`Missing oracle port ${name}`);
    return view as GraphDataView<Format>;
  };
  return {
    graph,
    getView,
    buffers,
    async read() {
      const result: Record<string, number[]> = {};
      for (const [name, port] of Object.entries(outputs)) {
        const components = port.format.endsWith('x4') ? 4 : port.format.endsWith('x2') ? 2 : 1;
        result[name] = await (port.format.startsWith('float32') ? readFloat32 : readUint32)(
          buffers[name],
          port.length * components
        );
      }
      return result as Record<keyof Outputs, number[]>;
    },
    poison() {
      for (const name of Object.keys(outputs))
        buffers[name].write(new Uint32Array(buffers[name].byteLength / 4).fill(0x45454545));
    },
    destroy() {
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  };
}

/** Submits twice to catch stale values as well as missing first writes. */
export async function runPoisonedOracleGraph<Outputs extends Record<string, OraclePort>>(
  harness: ReturnType<typeof createPoisonedOracleGraph<Outputs>>,
  device: Device,
  beforeSubmission?: (submission: number) => void
) {
  const compiled = harness.graph.compile();
  try {
    const results = [];
    for (let submission = 0; submission < 2; submission++) {
      harness.poison();
      beforeSubmission?.(submission);
      submitGraph(device, compiled, undefined);
      results.push(await harness.read());
    }
    return results;
  } finally {
    compiled.destroy();
    harness.destroy();
  }
}
