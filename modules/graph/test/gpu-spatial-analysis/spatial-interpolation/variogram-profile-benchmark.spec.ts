// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {createInterpolationBenchmarkCases} from './interpolation-performance-fixtures';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {getBenchmarkSettings} from '../../utils/benchmark-settings';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
it.skipIf(environment?.['VITE_VARIOGRAM_PROFILE'] !== 'true')(
  'variogram stage attribution with explicit timestamp passes',
  {timeout: 300000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `VARIOGRAM_PROFILE_METADATA ${JSON.stringify({settings, device: device.info, timestampQueries: device.features.has('timestamp-query'), method: 'Diagnostic per-node timestamp passes; not uninstrumented end-to-end latency. Timestamp precision follows browser configuration.'})}`
    );
    if (!device.features.has('timestamp-query')) return;
    let measured = 0;
    for (const benchmark of createInterpolationBenchmarkCases()) {
      if (
        !benchmark.name.startsWith('variogram-fit-') ||
        !benchmark.name.includes(settings.workloadFilter)
      )
        continue;
      measured++;
      const scope = new InterpolationBenchmarkScope(device, `profile-${benchmark.name}`);
      const check = benchmark.construct(scope);
      const compiled = scope.graph.compile();
      const querySet = device.createQuerySet({type: 'timestamp', count: 128});
      try {
        for (const write of scope.initialWrites) write.buffer.write(write.values);
        for (let warmup = 0; warmup < settings.warmupCount; warmup++) {
          submitGraph(device, compiled, undefined);
          await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
        }
        check(await scope.read(), false);
        const samples = [];
        for (let sample = 0; sample < settings.sampleCount; sample++) {
          const encoder = device.createCommandEncoder({timeProfilingQuerySet: querySet});
          const encoding = compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          samples.push(await encoding.readTimings());
        }
        check(await scope.read(), false);
        console.log(
          `VARIOGRAM_PROFILE_MEASUREMENT ${JSON.stringify({workload: benchmark.name, samples})}`
        );
      } finally {
        querySet.destroy();
        compiled.destroy();
        scope.destroy();
      }
    }
    if (!measured) throw new Error('No variogram profiling workloads matched the filter');
  }
);
