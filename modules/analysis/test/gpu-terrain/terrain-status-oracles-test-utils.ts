// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GPUCPUOperation
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect} from 'vitest';
import type {GPUContributorOperation} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {runOperationGraph, type OperationGraphInput} from '../utils/mixed-execution-test-utils';

export type TerrainStatusFrame = {
  inputs: Record<string, OperationGraphInput>;
  expected: Record<string, readonly number[]>;
  gpuExpected?: Record<string, readonly number[]>;
  floatExpected?: Record<string, readonly number[]>;
};
export type TerrainStatusOracle = {
  name: string;
  gpu(): GPUContributorOperation;
  cpu(): Pick<GPUCPUOperation, 'inputs' | 'execute'>;
  frames: TerrainStatusFrame[];
};

/** Executes the CPU operation itself; no expected value is taken from its result. */
export async function checkTerrainCPUStatus(oracle: TerrainStatusOracle): Promise<void> {
  const operation = oracle.cpu();
  for (const frame of oracle.frames) {
    const inputs = Object.fromEntries(
      Object.entries(operation.inputs).map(([port, name]) => {
        const input = frame.inputs[name];
        const length = input.values.length / getGPUVectorFormatInfo(input.format).components;
        return [
          port,
          {
            format: input.format,
            length,
            chunks: [{values: input.values, length}],
            pack: () => input.values
          }
        ];
      })
    );
    const output = await operation.execute({inputs});
    for (const [name, expected] of Object.entries(frame.expected)) {
      expect(output[name], name).toBeInstanceOf(Uint32Array);
      expect(Array.from(output[name] as Uint32Array), name).toEqual(expected);
    }
    for (const [name, expected] of Object.entries(frame.floatExpected ?? {})) {
      expect(output[name], name).toBeInstanceOf(Float32Array);
      expect(Array.from(output[name] as Float32Array), name).toEqual(expected);
    }
  }
}

/** Checks both the directly bound contributor (poison plus replay) and its operation wiring. */
export async function checkTerrainGPUStatus(oracle: TerrainStatusOracle): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = oracle.gpu();
  const graph = new GPUCommandGraph(device);
  const inputs = Object.fromEntries(
    Object.entries(oracle.frames[0].inputs).map(([name, input]) => [
      name,
      createInputBuffer(device, input.values)
    ])
  );
  const outputs = Object.fromEntries(
    Object.values(operation.outputs).map(output => {
      const componentCount = output.length * getGPUVectorFormatInfo(output.format).components;
      const buffer = createOutputBuffer(device, componentCount);
      buffer.write(new Uint32Array(Math.max(componentCount, 1)).fill(17));
      return [output.value, buffer];
    })
  );
  const views = Object.fromEntries([
    ...Object.entries(operation.inputs).map(([port, name]) => {
      const input = oracle.frames[0].inputs[name];
      const length = input.values.length / getGPUVectorFormatInfo(input.format).components;
      return [port, importGraphBuffer(graph, port, inputs[name], input.format, length)];
    }),
    ...Object.entries(operation.outputs).map(([port, output]) => [
      port,
      importGraphBuffer(graph, port, outputs[output.value], output.format, output.length)
    ])
  ]);
  graph.add(operation.createContributor(port => views[port] as never) as GPUCommandNodeProducer);
  const compiled = graph.compile();
  try {
    for (const frame of oracle.frames) {
      for (const [name, input] of Object.entries(frame.inputs)) inputs[name].write(input.values);
      submitGraph(device, compiled, undefined);
      const expected = {...frame.expected, ...frame.gpuExpected};
      for (const [name, values] of Object.entries(expected)) {
        expect(await readUint32(outputs[name], values.length), name).toEqual(values);
      }
      for (const [name, values] of Object.entries(frame.floatExpected ?? {})) {
        expect(await readFloat32(outputs[name], values.length), name).toEqual(values);
      }
      const result = await runOperationGraph(device, {
        inputs: frame.inputs,
        operations: [oracle.gpu()],
        outputs: [...Object.keys(expected), ...Object.keys(frame.floatExpected ?? {})]
      });
      for (const [name, values] of Object.entries(expected))
        expect(Array.from(result[name]), name).toEqual(values);
      for (const [name, values] of Object.entries(frame.floatExpected ?? {}))
        expect(Array.from(result[name]), name).toEqual(values);
    }
  } finally {
    compiled.destroy();
    operation.destroy();
    for (const buffer of [...Object.values(inputs), ...Object.values(outputs)]) buffer.destroy();
  }
}
