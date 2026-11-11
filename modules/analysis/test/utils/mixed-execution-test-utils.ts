// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUGraph,
  gpuGraph,
  type GPUGraphOperation,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {CPUData, CPUVector, getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../src/analysis-core/operations/gpu-contributor-operation';

/** Packed rows of one graph value. */
export type OperationGraphRows = Float32Array | Uint32Array | Int32Array;

/** One input of {@link runOperationGraph}: packed rows and their format. */
export type OperationGraphInput = {format: GPUGraphVectorFormat; values: OperationGraphRows};

/** Properties of {@link runOperationGraph}. */
export type OperationGraphProps = {
  id?: string;
  /** Graph inputs by value name. The row count is `values.length / components`. */
  inputs: Readonly<Record<string, OperationGraphInput>>;
  /** Operations in execution order: contributor operations, CPU operations, or both. */
  operations: readonly GPUGraphOperation[];
  /** Names of the graph values to return. */
  outputs: readonly string[];
};

/** Wraps packed rows as a one-chunk `CPUVector`. */
export function makeCPUVector(format: GPUGraphVectorFormat, values: OperationGraphRows): CPUVector {
  const {components} = getGPUVectorFormatInfo(format);
  if (values.length % components !== 0) {
    throw new Error(`${format} rows need a multiple of ${components} values`);
  }
  return new CPUVector({
    format,
    data: [new CPUData({buffer: values, format, length: values.length / components})]
  });
}

/** Copies the rows of a `CPUVector` into one packed typed array of the vector's component type. */
export function getCPUVectorRows(vector: CPUVector): OperationGraphRows {
  const info = getGPUVectorFormatInfo(vector.format);
  const ArrayType = vector.format.startsWith('float32')
    ? Float32Array
    : vector.format.startsWith('uint32')
      ? Uint32Array
      : Int32Array;
  const rows = new ArrayType(vector.length * info.components);
  const bytes = new Uint8Array(rows.buffer);
  let rowOffset = 0;
  for (const chunk of vector.data) {
    for (let row = 0; row < chunk.length; row++) {
      bytes.set(
        new Uint8Array(chunk.buffer, chunk.byteOffset + row * chunk.byteStride, info.byteLength),
        (rowOffset + row) * info.byteLength
      );
    }
    rowOffset += chunk.length;
  }
  return rows;
}

/**
 * Builds a luma.gl `GPUGraph` from the inputs and operations, runs it once through the published
 * mixed executor and returns the requested graph values as packed rows read on the CPU.
 *
 * Contributor operations are lowered with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. Use it to run a
 * `GPU<Name>Operation` and its `CPU<Name>Operation` on the same inputs and compare the outputs.
 * The compiled graph and every operation that has a `destroy()` are destroyed before it returns,
 * so pass each operation instance to one call only.
 */
export async function runOperationGraph(
  device: Device,
  props: OperationGraphProps
): Promise<Record<string, OperationGraphRows>> {
  const graph = new GPUGraph({id: props.id ?? 'operation-graph'});
  const inputs: Record<string, CPUVector> = {};
  for (const [name, input] of Object.entries(props.inputs)) {
    const vector = makeCPUVector(input.format, input.values);
    graph.addInput(name, gpuGraph.vector(input.format, {length: vector.length}));
    inputs[name] = vector;
  }
  for (const operation of props.operations) {
    graph.add(operation);
  }
  const compiled = graph.compile(device, {
    outputs: props.outputs,
    outputLocations: Object.fromEntries(props.outputs.map(name => [name, 'cpu' as const])),
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  try {
    const outputs = (await compiled.run({inputs})) as Record<string, unknown>;
    const rows: Record<string, OperationGraphRows> = {};
    for (const name of props.outputs) {
      const output = outputs[name];
      if (!(output instanceof CPUVector)) {
        throw new Error(`graph value "${name}" did not come back on the CPU`);
      }
      rows[name] = getCPUVectorRows(output);
    }
    return rows;
  } finally {
    compiled.destroy();
    // A contributor operation keeps the last contributor that owns GPU resources.
    for (const operation of props.operations) {
      const owner = operation as {destroy?: unknown};
      if (typeof owner.destroy === 'function') {
        owner.destroy.call(operation);
      }
    }
  }
}
