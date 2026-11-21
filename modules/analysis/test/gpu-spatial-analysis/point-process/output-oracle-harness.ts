// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {expect} from 'vitest';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {OperationGraphInput, OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Exercise the actual operation factory/wiring with caller-owned poisoned buffers. */
export async function runPoisonedOperation(
  device: Device,
  operation: GPUContributorOperation,
  inputs: Record<string, OperationGraphInput>,
  requestedOutputs: readonly string[]
): Promise<Record<string, OperationGraphRows>> {
  expect(
    Object.values(operation.outputs)
      .map(output => output.value)
      .sort(),
    'operation output declarations'
  ).toEqual([...requestedOutputs].sort());
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const views: Record<string, GraphDataView> = {};
  const outputs: Record<string, {buffer: Buffer; length: number; float: boolean}> = {};
  for (const [port, name] of Object.entries(operation.inputs)) {
    const {format, values} = inputs[name];
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    views[port] = importGraphBuffer(
      graph,
      port,
      buffer,
      format,
      values.length / getGPUVectorFormatInfo(format).components
    );
  }
  for (const [port, output] of Object.entries(operation.outputs)) {
    const length = output.length * getGPUVectorFormatInfo(output.format).components;
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(1, length)).fill(0x42424242));
    buffers.push(buffer);
    views[port] = importGraphBuffer(graph, port, buffer, output.format, output.length);
    outputs[output.value] = {buffer, length, float: output.format.startsWith('float')};
  }
  graph.add(operation.createContributor(port => views[port] as never));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result: Record<string, OperationGraphRows> = {};
    for (const [name, {buffer, length, float}] of Object.entries(outputs)) {
      result[name] = float
        ? new Float32Array(await readFloat32(buffer, length))
        : new Uint32Array(await readUint32(buffer, length));
    }
    expect(Object.keys(result).sort()).toEqual([...requestedOutputs].sort());
    return result;
  } finally {
    compiled.destroy();
    operation.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
