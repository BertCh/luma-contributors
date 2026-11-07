// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInterpolationBenchmarkCases,
  INTERPOLATION_BENCHMARK_SEED,
  INTERPOLATION_BENCHMARK_VERSION
} from './interpolation-performance-fixtures';
import {measureInterpolationCase} from './interpolation-performance-harness';
import {createInterpolationScalingCases} from './interpolation-scaling-fixtures';
import {measureResidentFitComparison} from './interpolation-fit-comparison';
import {getBenchmarkSettings} from '../../utils/benchmark-settings';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const runBenchmark = environment?.['VITE_GEOSPATIAL_BACKLOG_BENCHMARK'] === 'true';

it.skipIf(!runBenchmark)(
  'interpolation bounded performance envelopes',
  {timeout: 300000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const sourceSnapshot = settings.sourceSnapshot;
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `GEOSPATIAL_INTERPOLATION_METADATA ${JSON.stringify({
        schemaVersion: INTERPOLATION_BENCHMARK_VERSION,
        seed: INTERPOLATION_BENCHMARK_SEED,
        recordedAt: new Date().toISOString(),
        sourceRevision: sourceSnapshot?.revision ?? null,
        sourceState: sourceSnapshot?.sha256 ?? null,
        sourceSnapshot,
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
            'synchronous graph.compile; deferred compilation may appear in first execution',
          uploadClock: 'buffer.write + queue completion; allocation excluded',
          gpuClock: 'timestamped nodes when enabled; null otherwise',
          validation:
            'analytic reproduction, independent f64 linear algebra, status and changed-input replay; CPU fixture generation outside measured phases',
          memory:
            'graph transient bytes and caller buffer bytes; private/function shader budgets separately audited; driver/process peak unavailable',
          cache: 'shared device; caches not cleared; first execution is not necessarily cold',
          envelope: 'bounded observations only; no latency threshold or speedup assertion'
        }
      })}`
    );
    let measuredWorkloads = 0;
    for (const benchmark of [
      ...createInterpolationBenchmarkCases(),
      ...createInterpolationScalingCases()
    ]) {
      if (!benchmark.name.includes(settings.workloadFilter)) continue;
      measuredWorkloads++;
      const measurement = await measureInterpolationCase(device, benchmark, settings);
      console.log(`GEOSPATIAL_INTERPOLATION_MEASUREMENT ${JSON.stringify(measurement)}`);
    }
    for (const binCount of [20, 256]) {
      if (!`variogram-repeated-fit-comparison-${binCount}`.includes(settings.workloadFilter))
        continue;
      measuredWorkloads++;
      console.log(
        `GEOSPATIAL_INTERPOLATION_COMPARISON ${JSON.stringify(await measureResidentFitComparison(device, binCount, settings))}`
      );
    }
    if (!measuredWorkloads)
      throw new Error(`No interpolation workload matched ${settings.workloadFilter}`);
  }
);
