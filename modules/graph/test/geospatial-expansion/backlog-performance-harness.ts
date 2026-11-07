// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect} from 'vitest';
import {getBenchmarkSettings, getBenchmarkDistribution} from '../utils/benchmark-settings';
import {GPUMarkedPointStatistics} from '../../src/gpu-dataframe/pair-statistics/gpu-marked-point-statistics';
import {GPURasterWatershedSegmentation} from '../../src/gpu-raster/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/gpu-raster/raster-grid/index';
import {GPUNetworkStructure} from '../../src/gpu-network/network-structure/index';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {
  calculateNetworkReference,
  calculatePairReference,
  calculateWatershedReference,
  createNetworkWorkload,
  createPairWorkload,
  createWatershedWorkload
} from './backlog-performance-fixtures';

type Values = Float32Array | Uint32Array;
type Output = Record<string, number[]>;
type Write = {buffer: Buffer; values: Values};
type BenchmarkCase = {
  name: string;
  shape: Record<string, number | string>;
  updateKind: string;
  construct: (scope: BenchmarkScope) => {reference: (updated: boolean) => Output};
};

/** Buffers allocate without data; writes are deferred so upload is a separate measured phase. */
class BenchmarkScope {
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

  output<Format extends 'float32' | 'uint32'>(name: string, length: number, format: Format) {
    const buffer = this.allocate(length * 4);
    this.outputs.push({name, buffer, length, format});
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

export function createBacklogBenchmarkCases(): BenchmarkCase[] {
  return [
    ...[64, 512].map(count => {
      const workload = createPairWorkload(count);
      return {
        name: `marked-pairs-${count}`,
        shape: {
          points: count,
          radii: workload.radiusCount,
          directedPairs: count * (count - 1),
          edgeCorrection: 'translation'
        },
        updateKind: 'maximum-radius parameter write; same graph and positions',
        construct(scope: BenchmarkScope) {
          const parameters = (radius: number) => Float32Array.from([0, 0, 64, 64, radius, 0]);
          scope.graph.add(
            new GPUMarkedPointStatistics({
              positions: scope.input('positions', workload.positions, 'float32x2'),
              marks: scope.input('marks', workload.marks, 'float32'),
              parameters: scope.input(
                'parameters',
                parameters(workload.maximumRadius),
                'float32',
                parameters(workload.updatedMaximumRadius)
              ),
              radiusCount: workload.radiusCount,
              output: {
                statistics: scope.output('statistics', workload.radiusCount * 4, 'float32'),
                pairCounts: scope.output('pairCounts', workload.radiusCount, 'uint32'),
                status: scope.output('status', workload.radiusCount, 'uint32')
              }
            })
          );
          return {
            reference: (updated: boolean) =>
              calculatePairReference(
                workload,
                updated ? workload.updatedMaximumRadius : workload.maximumRadius
              )
          };
        }
      };
    }),
    ...[8, 24, 32, 33, 45, 46, 64].map(width => {
      const workload = createWatershedWorkload(width);
      return {
        name: `watershed-${width}`,
        shape: {
          width,
          height: width,
          cells: width * width,
          markers: 3,
          connectivity: 4,
          maximumSteps: width * width
        },
        updateKind: 'elevation input write; same markers and graph',
        construct(scope: BenchmarkScope) {
          scope.graph.add(
            new GPURasterWatershedSegmentation({
              grid: makeGPURasterMetadata({width, height: width}),
              input: {
                id: 'surface',
                format: 'float32',
                storage: {
                  kind: 'buffer',
                  values: scope.input(
                    'elevations',
                    workload.values,
                    'float32',
                    workload.updatedValues
                  )
                }
              },
              markers: scope.input('markers', workload.markers, 'uint32'),
              connectivity: 4,
              maximumSteps: width * width,
              output: {
                labels: scope.output('labels', width * width, 'uint32'),
                status: {converged: scope.output('converged', 1, 'uint32')}
              }
            })
          );
          return {
            reference: (updated: boolean) =>
              calculateWatershedReference(
                workload,
                updated ? workload.updatedValues : workload.values
              )
          };
        }
      };
    }),
    ...[16, 128].map(count => {
      const workload = createNetworkWorkload(count);
      return {
        name: `strong-components-${count}`,
        shape: {
          vertices: count,
          directedEdges: workload.neighbors.length,
          initialComponents: count / 8,
          updatedComponents: 1
        },
        updateKind:
          'one changed CSR destination closes component cycle; full neighbor buffer upload, same graph',
        construct(scope: BenchmarkScope) {
          scope.graph.add(
            new GPUNetworkStructure({
              topology: {
                offsets: scope.input('offsets', workload.offsets, 'uint32'),
                neighbors: scope.input(
                  'neighbors',
                  workload.neighbors,
                  'uint32',
                  workload.updatedNeighbors
                )
              },
              output: {strongComponents: scope.output('labels', count, 'uint32')}
            })
          );
          return {
            reference: (updated: boolean) =>
              calculateNetworkReference(
                workload.offsets,
                updated ? workload.updatedNeighbors : workload.neighbors
              )
          };
        }
      };
    })
  ];
}

function checkOutput(actual: Output, expected: Output) {
  expect(Object.keys(actual).sort()).toEqual(Object.keys(expected).sort());
  for (const [name, values] of Object.entries(expected)) {
    expect(actual[name].length, name).toBe(values.length);
    if (name !== 'statistics') {
      expect(actual[name], name).toEqual(values);
      continue;
    }
    values.forEach((value, index) => {
      if (Number.isNaN(value)) expect(actual[name][index], `${name}[${index}]`).toBeNaN();
      else {
        expect(Number.isFinite(actual[name][index]), `${name}[${index}]`).toBe(true);
        // Float32 row reductions accumulate at most 512 terms in each stage.
        expect(Math.abs(actual[name][index] - value), `${name}[${index}]`).toBeLessThanOrEqual(
          5e-5 * Math.max(1, Math.abs(value))
        );
      }
    });
  }
}

async function waitForQueue(device: Device) {
  // The benchmark requires a real WebGPU device, supplied by getWebGPUTestDevice.
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
}

function summarize(values: number[]) {
  return {samples: values, ...getBenchmarkDistribution(values)};
}

export async function measureBacklogCase(
  device: Device,
  benchmark: BenchmarkCase,
  settings = getBenchmarkSettings()
) {
  await waitForQueue(device);
  const started = performance.now();
  const scope = new BenchmarkScope(device, benchmark.name);
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const {reference} = benchmark.construct(scope);
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
    const run = async () => {
      const start = performance.now();
      const encoding = submitGraph(device, compiled!, undefined);
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
      return {referenceMilliseconds, readbackMilliseconds};
    };
    const firstExecution = await run();
    const initialValidation = await readAndCheck(false);
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await run();
    const repeatedExecution = [];
    for (let sample = 0; sample < settings.sampleCount; sample++)
      repeatedExecution.push(await run());
    // Validate again after repetitions so repeated encodes cannot silently retain or accumulate state.
    const repeatedValidation = await readAndCheck(false);
    const updateUploadMilliseconds = await upload(scope.updateWrites);
    const updatedExecution = [];
    for (let sample = 0; sample < settings.updateSampleCount; sample++)
      updatedExecution.push(await run());
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
        'full initial, repeated, and updated output agrees with independent JavaScript reference'
    };
  } finally {
    compiled?.destroy();
    scope.destroy();
  }
}
