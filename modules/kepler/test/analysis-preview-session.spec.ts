// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {test, vi} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {AnalysisPreviewSession} from '../src/analysis/analysis-preview-session';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import type {CatalogRunResult} from '../src/analysis/catalog-binding-types';
import {KeplerDatasetRegistry, KeplerAllocationError} from '../src/runtime/dataset-registry';

function report(condition: unknown, message: string, details: unknown): asserts condition {
  if (!condition)
    throw new Error(
      `Retained point-buffer preview: ${message}\n${JSON.stringify(details, null, 2)}`
    );
}

function makeGate() {
  let resolve!: () => void;
  const promise = new Promise<void>(settle => {
    resolve = settle;
  });
  return {promise, resolve};
}

const INPUTS = {
  points: {
    type: 'FeatureCollection' as const,
    features: [
      {id: 'first', geometry: {type: 'Point' as const, coordinates: [0, 0]}},
      {id: 'second', geometry: {type: 'Point' as const, coordinates: [10, 10]}}
    ]
  }
};
const PARAMETERS = {coordinateSystem: 'planar', segmentsPerQuarter: 8, radius: 2};

test('retained catalog binding compiles once, changes radius on GPU, admits overlap and preserves failed replacement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const plan = binding.planPreview!(INPUTS, PARAMETERS);
  const peak = plan.cpuByteLength + plan.gpuByteLength;
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 3 * peak});
  const session = new AnalysisPreviewSession({device, registry, binding});
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compileAsync');
  const gpuDevice = device.handle as GPUDevice;
  const createBuffer = gpuDevice.createBuffer.bind(gpuDevice);
  let liveGpuBytes = 0;
  let actualGpuPeak = 0;
  const destroySpies: {mockRestore: () => void}[] = [];
  const allocation = vi.spyOn(gpuDevice, 'createBuffer').mockImplementation(properties => {
    const buffer = createBuffer(properties);
    const destroy = buffer.destroy.bind(buffer);
    liveGpuBytes += buffer.size;
    actualGpuPeak = Math.max(actualGpuPeak, liveGpuBytes);
    let released = false;
    destroySpies.push(
      vi.spyOn(buffer, 'destroy').mockImplementation(() => {
        if (!released) {
          released = true;
          liveGpuBytes -= buffer.size;
        }
        destroy();
      })
    );
    report(
      liveGpuBytes <= registry.memoryUsage.reservedBytes.gpu,
      'GPU allocation exceeded admission (including readback staging)',
      {liveGpuBytes, memory: registry.memoryUsage, bufferBytes: buffer.size}
    );
    return buffer;
  });
  const peaks: number[] = [];
  registry.subscribeMemory(() => peaks.push(registry.memoryUsage.totalBytes));
  let displayed: CatalogRunResult | undefined;
  const request = {
    inputs: INPUTS,
    parameters: PARAMETERS,
    revisionKey: 'one',
    isCurrent: () => true,
    commit: (result: CatalogRunResult) => {
      displayed = result;
    }
  };
  try {
    report(await session.request(request), 'initial execution did not commit', {displayed});
    const first = displayed!;
    report(
      await session.request({...request, parameters: {...PARAMETERS, radius: 3}}),
      'live radius did not commit',
      {displayed}
    );
    const second = displayed!;
    report(Number(compile.mock.calls.length) === 1, 'live radius recompiled', {
      compileCount: compile.mock.calls.length
    });
    for (const result of [first, second]) {
      const reference = runCatalogOperation(binding, result.parameters, INPUTS);
      const actualAreas = result.tables[0].columns['area'] as number[];
      const expectedAreas = reference.tables[0].columns['area'] as number[];
      report(
        actualAreas.every((value, index) => Math.abs(value - expectedAreas[index]) < 0.0001),
        'GPU output differs from the independent CPU circle oracle',
        {actualAreas, expectedAreas, parameters: result.parameters}
      );
    }
    report(
      first.tables[0].columns['radius'][0] === 2 && second.tables[0].columns['radius'][0] === 3,
      'retained output arrays were overwritten',
      {first: first.tables[0], second: second.tables[0]}
    );
    registry.setMemoryBudget(peak);
    let failure: unknown;
    try {
      await session.request({...request, revisionKey: 'two'});
    } catch (error) {
      failure = error;
    }
    report(
      failure instanceof KeplerAllocationError && displayed === second,
      'failed replacement changed display',
      {failure: String(failure), displayed}
    );
    report(Number(compile.mock.calls.length) === 1, 'rejected admission compiled resources', {
      compileCount: compile.mock.calls.length
    });
    registry.setMemoryBudget(3 * peak);
    report(
      await session.request({...request, revisionKey: 'two'}),
      'replacement did not commit',
      {}
    );
    report(Math.max(...peaks) === 2 * peak, 'overlap peak did not include both executors', {
      peaks,
      peak
    });
    report(Number(compile.mock.calls.length) === 2, 'replacement compilation count differs', {
      compileCount: compile.mock.calls.length
    });
  } finally {
    await session.destroy();
    compile.mockRestore();
    allocation.mockRestore();
    for (const destroy of destroySpies) destroy.mockRestore();
  }
  report(actualGpuPeak > 0 && liveGpuBytes === 0, 'GPU buffers were not all released', {
    actualGpuPeak,
    liveGpuBytes
  });
  report(
    registry.memoryUsage.totalBytes === 0,
    'destroy leaked reservations',
    registry.memoryUsage
  );
});

test('actual submitted GPU work survives cancellation until the completion fence settles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 8 * 1024 * 1024});
  const session = new AnalysisPreviewSession({
    device,
    registry,
    binding: getCatalogBinding('geometry-point-buffer')!
  });
  const submitted = makeGate();
  const completion = makeGate();
  const createFence = device.createFence.bind(device);
  const fence = vi.spyOn(device, 'createFence').mockImplementation(() => {
    const actual = createFence();
    submitted.resolve();
    return {
      signaled: Promise.all([actual.signaled, completion.promise]).then(() => undefined),
      destroy: () => actual.destroy()
    } as ReturnType<typeof device.createFence>;
  });
  let commits = 0;
  try {
    const execution = session.request({
      inputs: INPUTS,
      parameters: PARAMETERS,
      revisionKey: 'one',
      isCurrent: () => true,
      commit: () => {
        commits++;
      }
    });
    await submitted.promise;
    const destruction = session.destroy();
    report(
      registry.memoryUsage.totalBytes > 0 && commits === 0,
      'cancellation released in-flight memory or committed',
      {memory: registry.memoryUsage, commits}
    );
    completion.resolve();
    report((await execution) === false, 'cancelled submission was accepted', {commits});
    await destruction;
    report(
      commits === 0 && registry.memoryUsage.totalBytes === 0,
      'cancelled execution published or leaked memory',
      {memory: registry.memoryUsage, commits}
    );
  } finally {
    completion.resolve();
    await session.destroy();
    fence.mockRestore();
  }
});
