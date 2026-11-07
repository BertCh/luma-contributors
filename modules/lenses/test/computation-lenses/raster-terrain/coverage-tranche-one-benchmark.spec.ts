// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runExample as runDistribution} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one-fractional-zonal-distribution.example';
import {runExample as runFocalPairs} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one-focal-pairs.example';
import {runExample as runDasymetric} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one-dasymetric.example';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';
import {submitGraph} from '@luma-contributors/graph/testing';
import {requireTestDevice} from '@luma-contributors/graph/testing';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const RUN_BENCHMARK = environment?.['VITE_COVERAGE_TRANCHE_ONE_BENCHMARK'] === 'true';
const SEED = 0x5eed2026;
type Workload = 'focal-pairs' | 'fractional-distribution' | 'dasymetric';

async function waitForQueue(device: Device) {
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
}

/** Uses the same public examples as the lens scenes, with larger deterministic input buffers. */
async function buildBenchmark(device: Device, workload: Workload, side: number) {
  const count = side * side;
  const harness = createCoverageTrancheOneHarness(device, `${workload}-${side}`, count);
  const {session} = harness;
  const grid = {
    width: side,
    height: side,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  let state = SEED;
  const values = Array.from({length: count}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 1 + Math.floor((state / 4294967296) * 16);
  });
  const floats = (name: string, length: number) =>
    session.output(name, 'float32', {role: 'output', meaning: name}, length);
  const integers = (name: string, length: number) =>
    session.output(name, 'uint32', {role: 'intermediate', meaning: name}, length);
  if (workload === 'focal-pairs') {
    const predictor = session.input('predictor', 'float32', values, false);
    const response = session.input(
      'response',
      'float32',
      values.map(value => 2 * value + 3),
      false
    );
    await runFocalPairs(harness.graph, {
      grid,
      predictor: {id: 'predictor', format: 'float32', storage: {kind: 'buffer', values: predictor}},
      response: {id: 'response', format: 'float32', storage: {kind: 'buffer', values: response}},
      radius: 2,
      borderMode: 'nodata',
      noDataPolicy: 'ignore',
      output: {
        pairedCount: integers('pairedCount', count),
        status: integers('status', count),
        covariance: floats('covariance', count),
        correlation: floats('correlation', count),
        slope: floats('slope', count),
        intercept: floats('intercept', count),
        rSquared: floats('rSquared', count),
        residualSumSquares: floats('residualSumSquares', count)
      }
    });
    return {
      harness,
      shape: {cells: count, radius: 2, maximumWindowPairs: 25},
      update: {name: 'response', values: Float32Array.from(values, value => 2 * value + 8)},
      expected: (updated: boolean) => ({
        slope: new Array(count).fill(2),
        intercept: new Array(count).fill(updated ? 8 : 3),
        status: new Array(count).fill(0)
      })
    };
  }
  const zoneCount = 8;
  const sourceZones = Array.from({length: count}, (_, index) => index % zoneCount);
  const fractions = Array.from({length: count}, (_, index) => [1, 0.5, 0.25][index % 3]);
  const coverage = {
    zoneIds: session.input('sourceZones', 'uint32', sourceZones, false),
    cellIds: session.input(
      'cellIds',
      'uint32',
      Array.from({length: count}, (_, index) => index),
      false
    ),
    fractions: session.input('fractions', 'float32', fractions, false),
    zoneValidity: session.input('zoneValidity', 'uint32', new Array(zoneCount).fill(1), false),
    status: {
      count: session.input('count', 'uint32', [count], false),
      requiredCount: session.input('requiredCount', 'uint32', [count], false),
      overflow: session.input('overflow', 'uint32', [0], false)
    }
  };
  const sourceValues = session.input('sourceValues', 'float32', values, false);
  const input = {
    id: 'source',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: sourceValues}
  };
  if (workload === 'fractional-distribution') {
    await runDistribution(harness.graph, {
      grid,
      input,
      coverage,
      zoneCount,
      probabilities: session.input('probabilities', 'float32', [0, 0.5, 1], false),
      output: {
        means: floats('means', zoneCount),
        populationVariances: floats('populationVariances', zoneCount),
        sampleVariances: floats('sampleVariances', zoneCount),
        effectiveSampleSizes: floats('effectiveSampleSizes', zoneCount),
        weightSums: floats('weightSums', zoneCount),
        squaredWeightSums: floats('squaredWeightSums', zoneCount),
        coveredArea: floats('coveredArea', zoneCount),
        validArea: floats('validArea', zoneCount),
        quantiles: floats('quantiles', zoneCount * 3),
        quantileValidity: integers('quantileValidity', zoneCount * 3),
        validity: integers('validity', zoneCount),
        status: integers('status', zoneCount)
      }
    });
    const weights = new Array<number>(zoneCount).fill(0);
    const sums = new Array<number>(zoneCount).fill(0);
    const squaredSums = new Array<number>(zoneCount).fill(0);
    values.forEach((value, index) => {
      weights[sourceZones[index]] += fractions[index];
      sums[sourceZones[index]] += value * fractions[index];
      squaredSums[sourceZones[index]] += value * value * fractions[index];
    });
    const means = sums.map((sum, index) => sum / weights[index]);
    const variances = squaredSums.map((sum, index) => sum / weights[index] - means[index] ** 2);
    return {
      harness,
      shape: {cells: count, tuples: count, zones: zoneCount, quantiles: 3},
      update: {name: 'sourceValues', values: Float32Array.from(values, value => value + 3)},
      expected: (updated: boolean) => ({
        means: means.map(value => value + (updated ? 3 : 0)),
        populationVariances: variances,
        weightSums: weights,
        validity: new Array(zoneCount).fill(1),
        status: new Array(zoneCount).fill(0)
      })
    };
  }
  const totals = Array.from({length: zoneCount}, (_, zone) => 100 * (zone + 1));
  const targets = Array.from({length: count}, (_, index) => Math.floor(index / zoneCount) % 4);
  const result = await runDasymetric(harness.graph, {
    grid,
    source: {kind: 'coverage', coverage},
    sourceTotals: session.input('sourceTotals', 'float32', totals, false),
    targetZones: session.input('targetZones', 'uint32', targets, false),
    targetZoneCount: 4,
    ancillary: input,
    denominator: 'overlap',
    pairCapacity: count,
    outputs: {
      targetTotals: floats('targetTotals', 4),
      targetSupportWeights: floats('targetSupportWeights', 4),
      sourceEligibleWeights: floats('sourceEligibleWeights', zoneCount),
      sourceTransferredFractions: floats('sourceTransferredFractions', zoneCount),
      sourceAllocatedTotals: floats('sourceAllocatedTotals', zoneCount),
      sourceUnallocatedTotals: floats('sourceUnallocatedTotals', zoneCount),
      sourceValidity: integers('sourceValidity', zoneCount)
    }
  });
  session.capture('complete', result.status.complete, {
    role: 'intermediate',
    meaning: 'Completion evidence'
  });
  const weights = new Array<number>(zoneCount).fill(0);
  const pairWeights = Array.from({length: 4}, () => new Array<number>(zoneCount).fill(0));
  values.forEach((value, index) => {
    const weight = value * fractions[index];
    weights[sourceZones[index]] += weight;
    pairWeights[targets[index]][sourceZones[index]] += weight;
  });
  const targetTotals = pairWeights.map(weightsBySource =>
    weightsBySource.reduce(
      (sum, weight, source) => sum + (totals[source] * weight) / weights[source],
      0
    )
  );
  return {
    harness,
    shape: {cells: count, tuples: count, sourceZones: zoneCount, targetZones: 4},
    update: {name: 'sourceTotals', values: Float32Array.from(totals, total => total * 2)},
    expected: (updated: boolean) => ({
      targetTotals: targetTotals.map(total => total * (updated ? 2 : 1)),
      sourceAllocatedTotals: totals.map(total => total * (updated ? 2 : 1)),
      sourceUnallocatedTotals: new Array(zoneCount).fill(0),
      sourceValidity: new Array(zoneCount).fill(1),
      complete: [1]
    })
  };
}

async function measure(device: Device, workload: Workload, side: number) {
  await waitForQueue(device);
  const setupStart = performance.now();
  const setup = await buildBenchmark(device, workload, side);
  const setupMilliseconds = performance.now() - setupStart;
  const {harness} = setup;
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const uploadStart = performance.now();
    harness.upload();
    await waitForQueue(device);
    const uploadMilliseconds = performance.now() - uploadStart;
    const compileStart = performance.now();
    compiled = harness.graph.compile();
    const compileHostMilliseconds = performance.now() - compileStart;
    const execute = async () => {
      const start = performance.now();
      const encoding = submitGraph(device, compiled!, undefined);
      await waitForQueue(device);
      return {
        encodeSubmitQueueMilliseconds: performance.now() - start,
        cpuEncodeMilliseconds: encoding.stats.cpuEncodeTimeMilliseconds
      };
    };
    const check = async (updated: boolean) => {
      const start = performance.now();
      const actual: Record<string, number[]> = {};
      for (const name of harness.captures.keys())
        actual[name] = await harness.read(compiled!, name);
      const readbackMilliseconds = performance.now() - start;
      for (const [name, expected] of Object.entries(setup.expected(updated))) {
        expected.forEach((value, index) => {
          const obtained = actual[name][index];
          if (
            name === 'status' ||
            name === 'validity' ||
            name === 'sourceValidity' ||
            name === 'complete'
          )
            expect(obtained, `${name}[${index}]`).toBe(value);
          else
            expect(Math.abs(obtained - value), `${name}[${index}]`).toBeLessThanOrEqual(
              5e-5 * Math.max(1, Math.abs(value))
            );
        });
      }
      return readbackMilliseconds;
    };
    const firstExecution = await execute();
    const initialReadbackMilliseconds = await check(false);
    await execute();
    const repeatedExecution = [];
    for (let sample = 0; sample < 5; sample++) repeatedExecution.push(await execute());
    const repeatedReadbackMilliseconds = await check(false);
    const input = harness.inputs.get(setup.update.name)!;
    const updatedSamples = [];
    for (let sample = 0; sample < 3; sample++) {
      const updated = sample % 2 === 0;
      const start = performance.now();
      input.buffer.write(updated ? setup.update.values : input.values);
      await waitForQueue(device);
      const updateUploadMilliseconds = performance.now() - start;
      updatedSamples.push({
        updated,
        updateUploadMilliseconds,
        execution: await execute(),
        readbackMilliseconds: await check(updated)
      });
    }
    return {
      workload,
      side,
      shape: setup.shape,
      setupMilliseconds,
      uploadMilliseconds,
      compileHostMilliseconds,
      firstExecution,
      repeatedExecution,
      initialReadbackMilliseconds,
      repeatedReadbackMilliseconds,
      updatedSamples,
      changedInput: setup.update.name,
      updateBytes: setup.update.values.byteLength,
      inputBytes: [...harness.inputs.values()].reduce(
        (sum, input) => sum + input.values.byteLength,
        0
      ),
      graphNodes: compiled.stats.nodeOrder.length,
      graphPhysicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
      graphImportedBytes: compiled.stats.importedBufferBytes,
      driverPeakBytes: null,
      gpuTimestampMilliseconds: null,
      numericalChecks:
        'Independent analytic fits or CPU weighted sums checked initially, after repetitions and after each alternating update'
    };
  } finally {
    compiled?.destroy();
    harness.destroy();
  }
}

it.skipIf(!RUN_BENCHMARK)(
  'coverage tranche one bounded public-example performance',
  {timeout: 300000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `COVERAGE_TRANCHE_ONE_METADATA ${JSON.stringify({
        schemaVersion: 1,
        seed: SEED,
        recordedAt: new Date().toISOString(),
        sourceRevision: environment?.['VITE_BENCHMARK_REVISION'] ?? null,
        sourceState: environment?.['VITE_BENCHMARK_SOURCE_STATE'] ?? null,
        device: device.info,
        userAgent: navigator.userAgent,
        limits: {
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        method:
          'Setup includes allocation/declaration, upload includes queue completion, compile is synchronous host work, execution includes encode+submit+queue. One extra warmup then five repetitions; three alternating input updates. Readback and assertions outside execution samples. Capture retention enabled; caches not cleared.',
        unavailable: [
          'GPU timestamp measurements',
          'driver peak memory',
          'optimized CPU library throughput'
        ],
        scope:
          'Focal pairs, fractional distribution and prepared-coverage dasymetric; categorical and spectral throughput unmeasured. Observed bounded workloads only, no performance thresholds or speed ratios.'
      })}`
    );
    for (const workload of ['focal-pairs', 'fractional-distribution', 'dasymetric'] as const) {
      for (const side of [8, 32])
        console.log(
          `COVERAGE_TRANCHE_ONE_MEASUREMENT ${JSON.stringify(await measure(device, workload, side))}`
        );
    }
  }
);
