// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUCompaction,
  GPUScan,
  GPUSort,
  type GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../utils/benchmark-settings';
import {requireTestDevice} from '../utils/require-test-device';
import {makePseudoRandomUint32} from './gpu-primitive-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;

/**
 * Opt-in comparison of the default multi-pass primitives with their alternative algorithms.
 * Run with `VITE_GPU_PRIMITIVE_BENCHMARK=true`; each line is a warm encode, submit, and queue
 * completion time in milliseconds, so it includes dispatch overhead but not graph compilation.
 */
it.skipIf(environment?.['VITE_GPU_PRIMITIVE_BENCHMARK'] !== 'true')(
  'compares scan, compaction and sort algorithms',
  {timeout: 900000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const settings = getBenchmarkSettings(environment);
    const rowCounts = [1 << 12, 1 << 16, 1 << 20, 1 << 23];
    for (const rowCount of rowCounts) {
      const values = makePseudoRandomUint32(rowCount, 1);
      const flags = values.map(value => value & 1);
      const cases: [
        string,
        (graph: GPUCommandGraph, buffers: Buffers) => GPUCommandNodeProducer
      ][] = [
        [
          'scan multi-pass',
          (graph, buffers) =>
            new GPUScan({input: view(graph, buffers.a), output: view(graph, buffers.b)})
        ],
        [
          'scan decoupled-lookback',
          (graph, buffers) =>
            new GPUScan({
              input: view(graph, buffers.a),
              output: view(graph, buffers.b),
              algorithm: 'decoupled-lookback'
            })
        ],
        ...(['scan', 'block'] as const).map(
          strategy =>
            [
              `compaction ${strategy}`,
              (graph: GPUCommandGraph, buffers: Buffers) =>
                new GPUCompaction({
                  input: view(graph, buffers.a),
                  flags: view(graph, buffers.flags),
                  output: view(graph, buffers.b),
                  count: view(graph, buffers.count, 1),
                  strategy
                })
            ] as const
        ),
        ...(
          [
            ['radix 4-bit', {algorithm: 'radix', digitBits: 4}],
            ['radix 8-bit', {algorithm: 'radix', digitBits: 8}],
            ['onesweep 8-bit', {algorithm: 'onesweep'}]
          ] as const
        ).map(
          ([name, props]) =>
            [
              `sort ${name}`,
              (graph: GPUCommandGraph, buffers: Buffers) =>
                new GPUSort({
                  ...props,
                  keys: view(graph, buffers.a),
                  values: view(graph, buffers.flags),
                  outputKeys: view(graph, buffers.b),
                  outputValues: view(graph, buffers.c)
                })
            ] as const
        )
      ];
      for (const [name, makeProducer] of cases) {
        const samples = await measure(device, values, flags, makeProducer, settings);
        console.log(
          `GPU_PRIMITIVE_BENCHMARK ${JSON.stringify({
            name,
            rowCount,
            device: device.info.renderer || device.info.vendor,
            ...getBenchmarkDistribution(samples)
          })}`
        );
      }
    }
  }
);

type Buffers = {a: Buffer; b: Buffer; c: Buffer; flags: Buffer; count: Buffer};

function view(graph: GPUCommandGraph, buffer: Buffer, length = buffer.byteLength / 4) {
  return graph.createDataView(
    graph.importBuffer(
      {id: `buffer-${buffer.id}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    ),
    {format: 'uint32', length}
  );
}

async function measure(
  device: Device,
  values: Uint32Array,
  flags: Uint32Array,
  makeProducer: (graph: GPUCommandGraph, buffers: Buffers) => GPUCommandNodeProducer,
  settings: ReturnType<typeof getBenchmarkSettings>
): Promise<number[]> {
  const buffers: Buffers = {
    a: device.createBuffer({data: values, usage: STORAGE_USAGE}),
    b: device.createBuffer({byteLength: values.byteLength, usage: STORAGE_USAGE}),
    c: device.createBuffer({byteLength: values.byteLength, usage: STORAGE_USAGE}),
    flags: device.createBuffer({data: flags, usage: STORAGE_USAGE}),
    count: device.createBuffer({byteLength: 4, usage: STORAGE_USAGE})
  };
  const graph = new GPUCommandGraph(device);
  graph.add(makeProducer(graph, buffers));
  const compiled = graph.compile();
  try {
    const samples: number[] = [];
    for (let run = 0; run < settings.warmupCount + settings.sampleCount; run++) {
      const startTime = performance.now();
      const commandEncoder = device.createCommandEncoder({id: 'primitive-benchmark'});
      compiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
      await buffers.count.readAsync();
      if (run >= settings.warmupCount) samples.push(performance.now() - startTime);
    }
    return samples;
  } finally {
    compiled.destroy();
    for (const buffer of Object.values(buffers)) buffer.destroy();
  }
}
