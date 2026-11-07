// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type QuerySet} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
type Values = Float32Array | Uint32Array;
export type Output = Record<string, number[]>;
type Write = {buffer: Buffer; values: Values};
export type BenchmarkCase = {
  name: string;
  shape: Record<string, number | string>;
  updateKind: string;
  construct: (scope: BenchmarkScope) => {
    reference: (updated: boolean) => Output;
    validate?: (actual: Output, updated: boolean) => void;
  };
};

/** Buffers allocate without data; writes are deferred so upload is a separate measured phase. */
export class BenchmarkScope {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  readonly initialWrites: Write[] = [];
  readonly updateWrites: Write[] = [];
  readonly outputs: {name: string; buffer: Buffer; format: 'float32' | 'uint32'; length: number}[] =
    [];

  constructor(
    readonly device: Device,
    name: string
  ) {
    this.graph = new GPUCommandGraph(device, {id: name});
  }

  input<Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Values,
    format: Format,
    updatedValues?: Values
  ) {
    const buffer = this.allocate(values.byteLength);
    this.initialWrites.push({buffer, values});
    if (updatedValues) this.updateWrites.push({buffer, values: updatedValues});
    return importGraphBuffer(
      this.graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }

  output<Format extends 'float32' | 'uint32' | 'float32x2'>(
    name: string,
    length: number,
    format: Format
  ) {
    const buffer = this.allocate(length * (format === 'float32x2' ? 8 : 4));
    this.outputs.push({
      name,
      buffer,
      length: length * (format === 'float32x2' ? 2 : 1),
      format: format === 'uint32' ? 'uint32' : 'float32'
    });
    return importGraphBuffer(this.graph, name, buffer, format, length);
  }

  private allocate(byteLength: number) {
    const buffer = this.device.createBuffer({
      byteLength,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    return buffer;
  }

  async read(): Promise<Output> {
    const result: Output = {};
    // Sequential maps make readback cost reproducible and avoid cross-case queue contention.
    for (const output of this.outputs) {
      result[output.name] = await (output.format === 'float32' ? readFloat32 : readUint32)(
        output.buffer,
        output.length
      );
    }
    return result;
  }

  destroy() {
    for (const buffer of this.buffers) buffer.destroy();
  }
}

function checkOutput(actual: Output, expected: Output) {
  for (const [name, values] of Object.entries(expected)) {
    expect(actual[name].length, name).toBe(values.length);
    if (
      [
        'pairCount',
        'counts',
        'chosenSlice',
        'status',
        'mapping',
        'adjacency',
        'breakIndex',
        'majority',
        'minority',
        'validity'
      ].includes(name)
    ) {
      expect(actual[name], name).toEqual(values);
      continue;
    }
    values.forEach((value, index) => {
      if (Number.isNaN(value)) expect(actual[name][index], `${name}[${index}]`).toBeNaN();
      else
        expect(Math.abs(actual[name][index] - value), `${name}[${index}]`).toBeLessThanOrEqual(
          3e-4 * Math.max(1, Math.abs(value))
        );
    });
  }
}

async function waitForQueue(device: Device) {
  // The benchmark requires a real WebGPU device, supplied by getWebGPUTestDevice.
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
}

function summarize(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    samples: values,
    minimum: sorted[0],
    median: sorted[Math.floor(sorted.length / 2)],
    maximum: sorted[sorted.length - 1]
  };
}

export async function measureRasterCase(device: Device, benchmark: BenchmarkCase) {
  await waitForQueue(device);
  const started = performance.now();
  const scope = new BenchmarkScope(device, benchmark.name);
  let compiled: CompiledGPUCommandGraph | undefined;
  let querySet: QuerySet | undefined;
  try {
    const {reference, validate} = benchmark.construct(scope);
    const setupMilliseconds = performance.now() - started;
    const upload = async (writes: Write[]) => {
      const start = performance.now();
      for (const {buffer, values} of writes) buffer.write(values);
      await waitForQueue(device);
      return performance.now() - start;
    };
    const uploadMilliseconds = await upload(scope.initialWrites);
    const compilationStart = performance.now();
    compiled = scope.graph.compile();
    const compileHostMilliseconds = performance.now() - compilationStart;
    if (device.features.has('timestamp-query'))
      querySet = device.createQuerySet({
        type: 'timestamp',
        count: compiled.stats.nodeOrder.length * 2
      });
    const run = async (profile = false) => {
      const start = performance.now();
      const encoder = device.createCommandEncoder({
        id: `${benchmark.name}-encoder`,
        ...(profile ? {timeProfilingQuerySet: querySet} : {})
      });
      const encoding = compiled!.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      await waitForQueue(device);
      const encodeSubmitCompletionMilliseconds = performance.now() - start;
      // Timestamp mapping, when supported/enabled, is outside the wall-clock execution sample.
      const timings = encoding.canReadGPUTimings ? await encoding.readTimings() : undefined;
      return {
        encodeSubmitCompletionMilliseconds,
        cpuEncodeMilliseconds: encoding.stats.cpuEncodeTimeMilliseconds,
        gpuTimestampMilliseconds: timings?.gpuTimeMilliseconds ?? null
      };
    };
    const readAndCheck = async (updated: boolean) => {
      const referenceStart = performance.now();
      const expected = reference(updated);
      const referenceMilliseconds = performance.now() - referenceStart;
      const readStart = performance.now();
      const actual = await scope.read();
      const readbackMilliseconds = performance.now() - readStart;
      checkOutput(actual, expected);
      validate?.(actual, updated);
      return {referenceMilliseconds, readbackMilliseconds};
    };
    const firstExecution = await run();
    const initialValidation = await readAndCheck(false);
    await run();
    const repeatedExecution = [];
    for (let sample = 0; sample < 5; sample++) repeatedExecution.push(await run());
    // Validate again after repetitions so repeated encodes cannot silently retain or accumulate state.
    const profiledExecution = querySet ? await run(true) : null;
    const repeatedValidation = await readAndCheck(false);
    const updateUploadMilliseconds = await upload(scope.updateWrites);
    const updatedExecution = [];
    for (let sample = 0; sample < 3; sample++) updatedExecution.push(await run());
    const updatedProfiledExecution = querySet ? await run(true) : null;
    const updatedValidation = await readAndCheck(true);
    return {
      workload: benchmark.name,
      shape: benchmark.shape,
      updateKind: benchmark.updateKind,
      phases: {
        setupMilliseconds,
        uploadMilliseconds,
        compileHostMilliseconds,
        firstExecution,
        repeatedExecution,
        profiledExecution,
        updatedProfiledExecution,
        repeatedWallClockMilliseconds: summarize(
          repeatedExecution.map(sample => sample.encodeSubmitCompletionMilliseconds)
        ),
        updateUploadMilliseconds,
        updatedExecution,
        initialValidation,
        repeatedValidation,
        updatedValidation
      },
      memory: {
        inputBytes: scope.initialWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        updateBytes: scope.updateWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        outputBytes: scope.outputs.reduce((sum, output) => sum + output.length * 4, 0),
        graphPhysicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
        graphLogicalTransientBytes: compiled.stats.logicalTransientBytes,
        driverPeakBytes: null
      },
      graphNodes: compiled.stats.nodeOrder.length,
      capabilities: compiled.capabilities,
      correctness:
        'Initial, warm, and updated numerical outputs checked against independent JavaScript/analytic references; peak and iteration diagnostics checked separately where no unique reference value exists'
    };
  } finally {
    querySet?.destroy();
    compiled?.destroy();
    scope.destroy();
  }
}
