// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  type GPUCommandNodeProducer,
  type GraphDataView,
  type GPUCPUVector
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {OperationGraphInput, OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {expect} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Inspect the real operation lowering on a null device, without compiling/executing GPU work. */
export function getOracleOperationNodes(
  operation: GPUContributorOperation,
  inputs: Record<string, OperationGraphInput>
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const views = new Map<string, GraphDataView>();
    for (const [port, name] of Object.entries(operation.inputs)) {
      const input = inputs[name];
      views.set(
        port,
        createTransientView(
          graph,
          port,
          input.format,
          input.values.byteLength / getGPUVectorFormatInfo(input.format).byteLength
        )
      );
    }
    for (const [port, output] of Object.entries(operation.outputs))
      views.set(port, createTransientView(graph, port, output.format, output.length));
    return (
      operation.createContributor(port => views.get(port)! as never) as GPUCommandNodeProducer
    ).getCommandNodes(graph);
  } finally {
    operation.destroy();
    device.destroy();
  }
}

/** Minimal packed CPU inputs for executing operation twins without a device. */
export function getOracleCPUVectors(
  inputs: Record<string, OperationGraphInput>,
  ports: Readonly<Record<string, string>> = Object.fromEntries(
    Object.keys(inputs).map(name => [name, name])
  )
): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(ports).map(([name, value]) => {
      const input = inputs[value];
      const length = input.values.byteLength / getGPUVectorFormatInfo(input.format).byteLength;
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
}

/** Exercise the operation's own port wiring with caller-owned, poison-filled output buffers. */
export async function runPoisonedOperation(
  device: Device,
  operation: GPUContributorOperation,
  inputs: Record<string, OperationGraphInput>,
  continuation?: {
    rewriteInputs: Record<string, OperationGraphRows>;
    expected: readonly [Record<string, readonly number[]>, Record<string, readonly number[]>];
  }
): Promise<Record<string, number[]>> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const views = new Map<string, GraphDataView>();
  const outputs = new Map<string, {buffer: Buffer; length: number; float: boolean}>();
  const inputBuffers = new Map<string, Buffer[]>();
  try {
    for (const [port, name] of Object.entries(operation.inputs)) {
      const input = inputs[name];
      const buffer = createInputBuffer(device, input.values);
      inputBuffers.set(name, [...(inputBuffers.get(name) ?? []), buffer]);
      buffers.push(buffer);
      views.set(
        port,
        importGraphBuffer(
          graph,
          `input-${port}`,
          buffer,
          input.format,
          input.values.byteLength / getGPUVectorFormatInfo(input.format).byteLength
        )
      );
    }
    for (const [port, output] of Object.entries(operation.outputs)) {
      const length = (output.length * getGPUVectorFormatInfo(output.format).byteLength) / 4;
      const buffer = createOutputBuffer(device, length);
      buffer.write(new Uint32Array(Math.max(1, length)).fill(0x4b123456));
      buffers.push(buffer);
      views.set(
        port,
        importGraphBuffer(graph, `output-${port}`, buffer, output.format, output.length)
      );
      outputs.set(output.value, {buffer, length, float: output.format.startsWith('float')});
    }
    graph.add(
      operation.createContributor(port => views.get(port)! as never) as GPUCommandNodeProducer
    );
    const compiled = graph.compile();
    try {
      // Reuse the same compiled contributor: initialization must also work on a second submission.
      let result: Record<string, number[]> = {};
      let firstResult: Record<string, number[]> | undefined;
      for (let submission = 0; submission < 2; submission++) {
        if (submission === 1 && continuation) {
          for (const [name, values] of Object.entries(continuation.rewriteInputs)) {
            expect(inputBuffers.has(name), `rewritten input ${name}`).toBe(true);
            for (const buffer of inputBuffers.get(name)!) buffer.write(values);
          }
        }
        for (const {buffer, length} of outputs.values())
          buffer.write(new Uint32Array(Math.max(1, length)).fill(0x4b123456));
        submitGraph(device, compiled, undefined);
        result = {};
        for (const [name, {buffer, length, float}] of outputs)
          result[name] = await (float ? readFloat32(buffer, length) : readUint32(buffer, length));
        if (continuation) {
          const expected = continuation.expected[submission];
          expect(Object.keys(result).sort()).toEqual(Object.keys(expected).sort());
          for (const [port, rows] of Object.entries(expected)) expectOracleRows(result[port], rows);
        } else if (firstResult) expect(result).toEqual(firstResult);
        else firstResult = result;
      }
      return result;
    } finally {
      compiled.destroy();
    }
  } finally {
    operation.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** Exact integer/padding expectations; float32 arithmetic gets an explicit absolute tolerance. */
export function expectOracleRows(
  actual: ArrayLike<number>,
  expected: readonly number[],
  tolerance = 0
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (!Number.isFinite(expected[index]))
      expect(actual[index], `row ${index}`).toBe(expected[index]);
    else {
      expect(Number.isFinite(actual[index]), `finite row ${index}`).toBe(true);
      expect(Math.abs(actual[index] - expected[index]), `row ${index}`).toBeLessThanOrEqual(
        tolerance
      );
    }
  }
}
