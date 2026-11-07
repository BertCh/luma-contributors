// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterBenchmarkCases} from './raster-performance-fixtures';
import {measureRasterCase} from './raster-performance-harness';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const runBenchmark = environment?.['VITE_RASTER_PERFORMANCE_BENCHMARK'] === 'true';

// Also isolated by browserBenchmarkTestPatterns in vitest.config.ts. See adjacent README.
it.skipIf(!runBenchmark)('raster bounded phase measurements', {timeout: 300000}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const metadata = {
    schemaVersion: 1,
    seed: 0x5eed2026,
    recordedAt: new Date().toISOString(),
    sourceRevision: environment?.['VITE_BENCHMARK_REVISION'] ?? null,
    sourceState: environment?.['VITE_BENCHMARK_SOURCE_STATE'] ?? null,
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    device: device.info,
    limits: {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
    },
    timestampQueriesAvailable: device.features.has('timestamp-query'),
    method: {
      warmupEncodesAfterFirst: 1,
      repeatedSamples: 5,
      updatedSamples: 3,
      executionClock: 'CPU encode + submit + queue completion; excludes output/timestamp readback',
      compileClock:
        'synchronous graph.compile; deferred pipeline compilation may appear in first execution',
      gpuClock:
        'separate profiled executions sum node timestamp durations when available; normal wall-clock repetitions remain unprofiled',
      uploadClock: 'buffer.write calls + queue completion, excludes allocation',
      cpuReference:
        'independent JavaScript correctness oracles; single run, not optimized library throughput',
      memory: 'graph allocated transient resources; not driver/process peak memory',
      cache:
        'shared test device; pipeline caches are not cleared; first execution is not a guaranteed cold-driver measurement',
      envelope:
        'bounded workload observations only; no latency threshold or speed advantage is asserted'
    }
  };
  console.log(`RASTER_PERFORMANCE_METADATA ${JSON.stringify(metadata)}`);
  for (const benchmark of createRasterBenchmarkCases()) {
    // Emit each completed case immediately so an interrupted run retains honest partial evidence.
    const measurement = await measureRasterCase(device, benchmark);
    console.log(`RASTER_PERFORMANCE_MEASUREMENT ${JSON.stringify(measurement)}`);
  }
});
