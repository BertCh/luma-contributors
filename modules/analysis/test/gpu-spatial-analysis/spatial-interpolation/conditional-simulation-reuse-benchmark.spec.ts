// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {calculateConditionalSimulationReference} from './conditional-simulation-reference';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../../utils/benchmark-settings';
import {requireTestDevice} from '../../utils/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

type Scenario = 'draws' | 'values' | 'geometry';
const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;

async function measureSequence(
  device: Device,
  size: number,
  batchCount: number,
  scenario: Scenario,
  forcePreparation: boolean,
  settings: ReturnType<typeof getBenchmarkSettings>
) {
  const started = performance.now();
  const scope = new InterpolationBenchmarkScope(
    device,
    `reuse-${size}-${batchCount}-${scenario}-${forcePreparation}`
  );
  const positions = Float32Array.from({length: size * 2}, (_, component) =>
    component % 2 === 0 ? (Math.floor(component / 2) % 16) / 16 : Math.floor(component / 32) / 8
  );
  const queries = Float32Array.from({length: size * 2}, (_, component) =>
    component % 2 === 0
      ? ((Math.floor(component / 2) % 16) + 0.3) / 16
      : (Math.floor(component / 32) + 0.2) / 8
  );
  const observations = Float32Array.from({length: size * 4}, (_, component) =>
    component % 4 === 0
      ? 1 + Math.sin(Math.floor(component / 4))
      : component % 4 === 1
        ? 0.1
        : component % 4 === 3
          ? 1
          : 0
  );
  const originalPositions = positions.slice();
  const originalObservations = observations.slice();
  const covariance = {model: 'exponential' as const, range: 1.3, sill: 1};
  const contributor = new GPUConditionalGaussianSimulation({
    id: 'simulation',
    positions: scope.input('positions', positions, 'float32x2'),
    observations: scope.input('observations', observations, 'float32x4'),
    queries: scope.input('queries', queries, 'float32x2'),
    covariance,
    mean: 0.7,
    seed: 1307,
    replicateCount: 128,
    reusePreparation: true,
    output: {
      values: scope.output('values', size * 128, 'float32'),
      moments: scope.output('moments', size, 'float32x2'),
      status: scope.output('status', 128, 'uint32')
    }
  });
  scope.graph.add(contributor);
  const setupMilliseconds = performance.now() - started;
  const uploadStarted = performance.now();
  for (const write of scope.initialWrites) write.buffer.write(write.values);
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
  const uploadMilliseconds = performance.now() - uploadStarted;
  const compileStarted = performance.now();
  const compiled = scope.graph.compile();
  const compileMilliseconds = performance.now() - compileStarted;
  const statisticsBuffer = contributor.preparationStatistics?.buffer.defaultBuffer;
  if (!(statisticsBuffer instanceof Buffer)) throw new Error('Missing preparation counters');
  const runSequence = async () => {
    const sequenceStarted = performance.now();
    positions.set(originalPositions);
    observations.set(originalObservations);
    scope.initialWrites[0].buffer.write(positions);
    scope.initialWrites[1].buffer.write(observations);
    contributor.invalidatePreparation();
    for (let batch = 0; batch < batchCount; batch++) {
      if (scenario === 'values') {
        for (let row = 0; row < size; row++)
          observations[row * 4] = originalObservations[row * 4] + batch * 0.01;
        scope.initialWrites[1].buffer.write(observations);
        contributor.invalidateObservations();
      }
      if (scenario === 'geometry') {
        for (let row = 0; row < size; row++)
          positions[row * 2] = originalPositions[row * 2] + batch * 0.001;
        scope.initialWrites[0].buffer.write(positions);
      }
      if (forcePreparation || scenario === 'geometry') contributor.invalidatePreparation();
      contributor.setDrawParameters({replicateOffset: batch * 128});
      submitGraph(device, compiled, undefined);
      await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    }
    return performance.now() - sequenceStarted;
  };
  try {
    const firstSequenceMilliseconds = await runSequence();
    const firstUseTotalMilliseconds = performance.now() - started;
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await runSequence();
    const samples = [];
    for (let sample = 0; sample < settings.sampleCount; sample++) samples.push(await runSequence());
    const actual = await scope.read();
    expect(actual['status']).toEqual(Array(128).fill(0));
    expect(actual['values'].every(Number.isFinite)).toBe(true);
    const expected = calculateConditionalSimulationReference(
      positions,
      observations,
      queries,
      covariance,
      Math.fround(0.7)
    );
    for (let query = 0; query < size; query++) {
      expect(Math.abs(actual['moments'][query * 2] - expected.means[query])).toBeLessThanOrEqual(
        3e-4 * Math.max(1, Math.abs(expected.means[query]))
      );
      expect(
        Math.abs(actual['moments'][query * 2 + 1] - expected.covariance[query][query])
      ).toBeLessThanOrEqual(3e-4);
    }
    const counters = (await readUint32(statisticsBuffer, 7)).slice(4, 6);
    const sequences = 1 + settings.warmupCount + settings.sampleCount;
    const factorsPerSequence = forcePreparation || scenario === 'geometry' ? batchCount : 1;
    const momentsPerSequence = forcePreparation || scenario !== 'draws' ? batchCount : 1;
    expect(counters).toEqual([sequences * factorsPerSequence, sequences * momentsPerSequence]);
    const capacity = size * 2;
    return {
      measurement: {
        variant: forcePreparation ? 'forced-preparation' : 'retained-preparation',
        setupMilliseconds,
        uploadMilliseconds,
        compileMilliseconds,
        firstSequenceMilliseconds,
        firstUseTotalMilliseconds,
        samples,
        distribution: getBenchmarkDistribution(samples),
        counters,
        persistentPreparationBytes:
          (2 + capacity * capacity + capacity * 5) * 4 + size * 8 + 24 + 28,
        inputBytes: positions.byteLength + observations.byteLength + queries.byteLength,
        outputBytes: size * 128 * 4 + size * 8 + 128 * 4,
        graphPhysicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
        graphLogicalTransientBytes: compiled.stats.logicalTransientBytes,
        correctnessStatus: 'passed'
      },
      actual
    };
  } finally {
    compiled.destroy();
    contributor.destroy();
    scope.destroy();
  }
}

it.skipIf(environment?.['VITE_SIMULATION_REUSE_BENCHMARK'] !== 'true')(
  'prepared simulation complete batch sequences',
  {timeout: 900000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const retainedFirst = environment?.['VITE_SIMULATION_REUSE_RETAINED_FIRST'] === 'true';
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `SIMULATION_REUSE_METADATA ${JSON.stringify({
        settings,
        fixtureVersion: 1,
        seed: 1307,
        mean: Math.fround(0.7),
        covariance: {model: 'exponential', range: 1.3, sill: 1},
        variantOrder: retainedFirst ? 'retained-then-forced' : 'forced-then-retained',
        device: device.info,
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        limits: {
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        method:
          'Forced preparation and reuse use identical mutable-batch APIs. Sequence wall time includes input/control writes, encoding, submission and completion. Allocation, host compile and first sequence are separate. No timestamp profiling; caches not reset. Driver peak memory unknown.'
      })}`
    );
    let measured = 0;
    for (const size of [32, 128, 256, 512]) {
      for (const scenario of ['draws', 'values', 'geometry'] as const) {
        for (const batchCount of [1, 2, 10, 100]) {
          const workload = `prepared-simulation-${size}-${scenario}-${batchCount}`;
          if (!workload.includes(settings.workloadFilter)) continue;
          measured++;
          const first = await measureSequence(
            device,
            size,
            batchCount,
            scenario,
            !retainedFirst,
            settings
          );
          const second = await measureSequence(
            device,
            size,
            batchCount,
            scenario,
            retainedFirst,
            settings
          );
          const baseline = retainedFirst ? second : first;
          const retained = retainedFirst ? first : second;
          for (const name of ['values', 'moments']) {
            for (const [index, value] of baseline.actual[name].entries()) {
              expect(Math.abs(retained.actual[name][index] - value)).toBeLessThanOrEqual(
                1e-5 * Math.max(1, Math.abs(value))
              );
            }
          }
          console.log(
            `SIMULATION_REUSE_MEASUREMENT ${JSON.stringify({workload, shape: {observations: size, queries: size, replicates: 128, batchCount, scenario}, baseline: baseline.measurement, retained: retained.measurement})}`
          );
        }
      }
    }
    if (!measured) throw new Error('No prepared simulation workload matched the filter');
  }
);
