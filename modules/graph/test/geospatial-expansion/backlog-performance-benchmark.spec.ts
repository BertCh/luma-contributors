// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../utils/require-test-device';
import {getBenchmarkSettings} from '../utils/benchmark-settings';
import {BACKLOG_BENCHMARK_SEED, BACKLOG_BENCHMARK_VERSION} from './backlog-performance-fixtures';
import {createBacklogBenchmarkCases, measureBacklogCase} from './backlog-performance-harness';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const runBenchmark = environment?.['VITE_GEOSPATIAL_BACKLOG_BENCHMARK'] === 'true';

// Also isolated by browserBenchmarkTestPatterns in vitest.config.ts. See adjacent README.
it.skipIf(!runBenchmark)(
  'geospatial backlog bounded performance envelopes',
  {timeout: 300000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const metadata = {
      schemaVersion: BACKLOG_BENCHMARK_VERSION,
      seed: BACKLOG_BENCHMARK_SEED,
      recordedAt: new Date().toISOString(),
      sourceRevision:
        settings.sourceSnapshot?.revision ?? environment?.['VITE_BENCHMARK_REVISION'] ?? null,
      sourceSnapshot: settings.sourceSnapshot,
      sampling: settings,
      sourceState:
        settings.sourceSnapshot?.sha256 ?? environment?.['VITE_BENCHMARK_SOURCE_STATE'] ?? null,
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
        warmupEncodesAfterFirst: settings.warmupCount,
        repeatedSamples: settings.sampleCount,
        updatedSamples: settings.updateSampleCount,
        executionClock:
          'CPU encode + submit + queue completion; excludes output/timestamp readback',
        compileClock:
          'synchronous graph.compile; deferred pipeline compilation may appear in first execution',
        gpuClock: 'sum of timestamped node durations when enabled; null otherwise',
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
    console.log(`GEOSPATIAL_BACKLOG_METADATA ${JSON.stringify(metadata)}`);
    let measuredWorkloads = 0;
    for (const benchmark of createBacklogBenchmarkCases()) {
      if (!benchmark.name.includes(settings.workloadFilter)) continue;
      measuredWorkloads++;
      // Emit each completed case immediately so an interrupted run retains honest partial evidence.
      const measurement = await measureBacklogCase(device, benchmark, settings);
      console.log(`GEOSPATIAL_BACKLOG_MEASUREMENT ${JSON.stringify(measurement)}`);
    }
    if (!measuredWorkloads) throw new Error('No backlog benchmark workloads matched the filter');
  }
);
