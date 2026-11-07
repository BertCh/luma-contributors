// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

import {getBenchmarkDistribution, getBenchmarkSettings} from '../../utils/benchmark-settings';

type BenchmarkValues = Float32Array | Uint32Array;
type BenchmarkWrite = {buffer: Buffer; values: BenchmarkValues};
export type InterpolationBenchmarkOutput = Record<string, number[]>;
export type InterpolationBenchmarkCase = {
  name: string;
  shape: Record<string, number | string>;
  correctness: string;
  construct: (
    scope: InterpolationBenchmarkScope
  ) => (actual: InterpolationBenchmarkOutput, updated: boolean) => void;
};

/** Deferred uploads and separately measured outputs, matching the shared backlog timing contract. */
export class InterpolationBenchmarkScope {
  readonly graph: GPUCommandGraph;
  readonly initialWrites: BenchmarkWrite[] = [];
  readonly updateWrites: BenchmarkWrite[] = [];
  readonly outputs: {name: string; buffer: Buffer; scalarLength: number; integer: boolean}[] = [];
  private readonly buffers: Buffer[] = [];

  constructor(
    readonly device: Device,
    name: string
  ) {
    this.graph = new GPUCommandGraph(device, {id: name});
  }

  input<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    values: BenchmarkValues,
    format: Format,
    updatedValues?: BenchmarkValues
  ) {
    const buffer = this.allocate(values.byteLength);
    this.initialWrites.push({buffer, values});
    if (updatedValues) {
      this.updateWrites.push({buffer, values: updatedValues});
    }
    const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    return importGraphBuffer(this.graph, name, buffer, format, values.length / components);
  }

  output<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    length: number,
    format: Format
  ) {
    const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    const scalarLength = length * components;
    const buffer = this.allocate(scalarLength * 4);
    this.outputs.push({name, buffer, scalarLength, integer: format === 'uint32'});
    return importGraphBuffer(this.graph, name, buffer, format, length);
  }

  async read(): Promise<InterpolationBenchmarkOutput> {
    const output: InterpolationBenchmarkOutput = {};
    for (const entry of this.outputs) {
      output[entry.name] = await (entry.integer ? readUint32 : readFloat32)(
        entry.buffer,
        entry.scalarLength
      );
    }
    return output;
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }

  private allocate(byteLength: number): Buffer {
    const buffer = this.device.createBuffer({
      byteLength,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }
}

async function waitForQueue(device: Device): Promise<void> {
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
}

/** No latency threshold, speedup ratio, or claim of a cold driver is made. */
export async function measureInterpolationCase(
  device: Device,
  benchmark: InterpolationBenchmarkCase,
  settings = getBenchmarkSettings()
) {
  await waitForQueue(device);
  const setupStart = performance.now();
  const scope = new InterpolationBenchmarkScope(device, benchmark.name);
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const validate = benchmark.construct(scope);
    const setupMilliseconds = performance.now() - setupStart;
    const upload = async (writes: BenchmarkWrite[]) => {
      const started = performance.now();
      for (const write of writes) {
        write.buffer.write(write.values);
      }
      await waitForQueue(device);
      return performance.now() - started;
    };
    const uploadMilliseconds = await upload(scope.initialWrites);
    const compileStart = performance.now();
    compiled = scope.graph.compile();
    const compileHostMilliseconds = performance.now() - compileStart;
    const run = async () => {
      const started = performance.now();
      const encoding = submitGraph(device, compiled!, undefined);
      await waitForQueue(device);
      const encodeSubmitCompletionMilliseconds = performance.now() - started;
      const timings = encoding.canReadGPUTimings ? await encoding.readTimings() : undefined;
      return {
        encodeSubmitCompletionMilliseconds,
        cpuEncodeMilliseconds: encoding.stats.cpuEncodeTimeMilliseconds,
        gpuTimestampMilliseconds: timings?.gpuTimeMilliseconds ?? null
      };
    };
    const readAndCheck = async (updated: boolean) => {
      const started = performance.now();
      const output = await scope.read();
      const readbackMilliseconds = performance.now() - started;
      const validationStart = performance.now();
      validate(output, updated);
      return {
        readbackMilliseconds,
        validationMilliseconds: performance.now() - validationStart,
        correctnessStatus: 'passed',
        statusCodes: [...new Set(output['status'] ?? [])].sort()
      };
    };
    const firstExecution = await run();
    const initialValidation = await readAndCheck(false);
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await run();
    const repeatedExecution = [];
    for (let sample = 0; sample < settings.sampleCount; sample++) {
      repeatedExecution.push(await run());
    }
    const repeatedValidation = await readAndCheck(false);
    const updateUploadMilliseconds = await upload(scope.updateWrites);
    const updatedExecution = [];
    for (let sample = 0; sample < settings.updateSampleCount; sample++) {
      updatedExecution.push(await run());
    }
    const updatedValidation = await readAndCheck(true);

    return {
      workload: benchmark.name,
      shape: benchmark.shape,
      updateKind: 'changed values uploaded into the same compiled graph',
      correctness: benchmark.correctness,
      phases: {
        setupMilliseconds,
        uploadMilliseconds,
        compileHostMilliseconds,
        firstExecution,
        repeatedExecution,
        repeatedWallClockMilliseconds: getBenchmarkDistribution(
          repeatedExecution.map(sample => sample.encodeSubmitCompletionMilliseconds)
        ),
        initialValidation,
        repeatedValidation,
        updateUploadMilliseconds,
        updatedExecution,
        updatedValidation
      },
      memory: {
        inputBytes: scope.initialWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        updateBytes: scope.updateWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        outputBytes: scope.outputs.reduce((sum, output) => sum + output.scalarLength * 4, 0),
        graphPhysicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
        graphLogicalTransientBytes: compiled.stats.logicalTransientBytes,
        driverPeakBytes: null
      },
      graphNodes: compiled.stats.nodeOrder.length,
      capabilities: compiled.capabilities
    };
  } finally {
    compiled?.destroy();
    scope.destroy();
  }
}
