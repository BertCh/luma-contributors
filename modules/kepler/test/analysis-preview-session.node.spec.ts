// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  AnalysisPreviewSession,
  type AnalysisPreviewRequest
} from '../src/analysis/analysis-preview-session';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {
  CatalogPreviewExecutor,
  CatalogPreviewPlan,
  CatalogRunOutput
} from '../src/analysis/catalog-binding-types';
import {KeplerAllocationError, KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {createAnalysisPreviewController} from '../../../examples/deck/kepler-compute/src/analysis-preview-controller';

function makeGate() {
  let resolve!: () => void;
  const promise = new Promise<void>(settle => {
    resolve = settle;
  });
  return {promise, resolve};
}

const OUTPUT: CatalogRunOutput = {tables: [], diagnostics: [], summary: {value: 1}};

function makeFixture(budget = 100) {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: budget});
  const create = vi.fn(
    async (): Promise<CatalogPreviewExecutor> => ({
      initialize: vi.fn(async () => undefined),
      execute: vi.fn(async () => OUTPUT),
      destroy: vi.fn(async () => undefined)
    })
  );
  const binding = {
    ...getCatalogBinding('geometry-point-buffer')!,
    planPreview: vi.fn(
      (_inputs, parameters): CatalogPreviewPlan => ({
        key: String(parameters['segmentsPerQuarter']),
        gpuByteLength: 30,
        cpuByteLength: 10,
        create
      })
    )
  };
  const session = new AnalysisPreviewSession({registry, device: new NullDevice({}), binding});
  const commit = vi.fn();
  const request: AnalysisPreviewRequest = {
    inputs: {
      points: {
        type: 'FeatureCollection',
        features: [{geometry: {type: 'Point', coordinates: [0, 0]}}]
      }
    },
    parameters: {coordinateSystem: 'planar'},
    revisionKey: 'source-1',
    isCurrent: () => true,
    commit
  };
  return {registry, create, binding, session, request, commit};
}

afterEach(() => vi.restoreAllMocks());

test('admission precedes create, keeps old plus new peak and retains one compilation for live settings', async () => {
  const fixture = makeFixture();
  const reserveMemory = fixture.registry.reserveMemory.bind(fixture.registry);
  const releases: ReturnType<typeof vi.fn>[] = [];
  vi.spyOn(fixture.registry, 'reserveMemory').mockImplementation((...reservationArguments) => {
    const reservation = reserveMemory(...reservationArguments);
    const release = vi.fn(() => reservation.release());
    releases.push(release);
    return {...reservation, release};
  });
  const peaks: number[] = [];
  fixture.registry.subscribeMemory(() => peaks.push(fixture.registry.memoryUsage.totalBytes));
  const originalCreate = fixture.create.getMockImplementation()!;
  fixture.create.mockImplementation(async () => {
    expect(fixture.registry.memoryUsage.totalBytes).toBe(
      fixture.create.mock.calls.length === 1 ? 40 : 80
    );
    return originalCreate();
  });
  expect(await fixture.session.request(fixture.request)).toBe(true);
  const first = await fixture.create.mock.results[0].value;
  expect(
    await fixture.session.request({
      ...fixture.request,
      parameters: {...fixture.request.parameters, radius: 3}
    })
  ).toBe(true);
  expect(fixture.create).toHaveBeenCalledTimes(1);
  expect(first.initialize).toHaveBeenCalledOnce();
  expect(first.execute).toHaveBeenCalledTimes(2);
  expect(
    await fixture.session.request({
      ...fixture.request,
      parameters: {...fixture.request.parameters, segmentsPerQuarter: 16}
    })
  ).toBe(true);
  expect(Math.max(...peaks)).toBe(80);
  expect(first.destroy).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  await fixture.session.destroy();
  await fixture.session.destroy();
  expect((await fixture.create.mock.results[1].value).destroy).toHaveBeenCalledOnce();
  expect(releases).toHaveLength(4);
  for (const release of releases) expect(release).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test('failed overlap admission never allocates and keeps the current display and reservation', async () => {
  const fixture = makeFixture(70);
  await fixture.session.request(fixture.request);
  await expect(
    fixture.session.request({...fixture.request, revisionKey: 'source-2'})
  ).rejects.toBeInstanceOf(KeplerAllocationError);
  expect(fixture.create).toHaveBeenCalledTimes(1);
  expect(fixture.commit).toHaveBeenCalledTimes(1);
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  await fixture.session.destroy();
});

test('failed CPU admission rolls back GPU admission before create', async () => {
  const fixture = makeFixture(35);
  await expect(fixture.session.request(fixture.request)).rejects.toBeInstanceOf(
    KeplerAllocationError
  );
  expect(fixture.create).not.toHaveBeenCalled();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
  await fixture.session.destroy();
});

test('superseded compile is disposed without execution before the latest queued build starts', async () => {
  const fixture = makeFixture();
  const gate = makeGate();
  const started = makeGate();
  const executor = {execute: vi.fn(async () => OUTPUT), destroy: vi.fn(async () => undefined)};
  fixture.create.mockImplementationOnce(async () => {
    started.resolve();
    await gate.promise;
    return executor;
  });
  const first = fixture.session.request(fixture.request);
  await started.promise;
  const skipped = fixture.session.request({...fixture.request, revisionKey: 'source-2'});
  const latest = fixture.session.request({...fixture.request, revisionKey: 'source-3'});
  gate.resolve();
  expect(await first).toBe(false);
  expect(await skipped).toBe(false);
  expect(await latest).toBe(true);
  expect(executor.execute).not.toHaveBeenCalled();
  expect(executor.destroy).toHaveBeenCalledOnce();
  expect(fixture.create).toHaveBeenCalledTimes(2);
  expect(fixture.commit).toHaveBeenCalledOnce();
  await fixture.session.destroy();
});

test('cancel is passed into execution; destruction and reservation release await submitted completion', async () => {
  const fixture = makeFixture();
  const submitted = makeGate();
  const completed = makeGate();
  let executionSignal: AbortSignal | undefined;
  const executor = {
    execute: vi.fn(async (_parameters, signal: AbortSignal) => {
      executionSignal = signal;
      submitted.resolve();
      await completed.promise;
      signal.throwIfAborted();
      return OUTPUT;
    }),
    destroy: vi.fn(async () => undefined)
  };
  fixture.create.mockResolvedValueOnce(executor);
  const pending = fixture.session.request(fixture.request);
  await submitted.promise;
  const destroyed = fixture.session.destroy();
  expect(executionSignal?.aborted).toBe(true);
  expect(executor.destroy).not.toHaveBeenCalled();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  completed.resolve();
  expect(await pending).toBe(false);
  await destroyed;
  expect(fixture.commit).not.toHaveBeenCalled();
  expect(executor.destroy).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test('compile, execution and atomic commit failures preserve the previous retained display', async () => {
  const fixture = makeFixture();
  await fixture.session.request(fixture.request);
  const first = await fixture.create.mock.results[0].value;
  fixture.create.mockRejectedValueOnce(new Error('compile failed'));
  await expect(
    fixture.session.request({...fixture.request, revisionKey: 'source-2'})
  ).rejects.toThrow('compile failed');
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  const failed = {
    execute: vi.fn(async () => {
      throw new Error('run failed');
    }),
    destroy: vi.fn(async () => undefined)
  };
  fixture.create.mockResolvedValueOnce(failed);
  await expect(
    fixture.session.request({...fixture.request, revisionKey: 'source-2'})
  ).rejects.toThrow('run failed');
  expect(failed.destroy).toHaveBeenCalledOnce();
  const rejected = {execute: vi.fn(async () => OUTPUT), destroy: vi.fn(async () => undefined)};
  fixture.create.mockResolvedValueOnce(rejected);
  await expect(
    fixture.session.request({
      ...fixture.request,
      revisionKey: 'source-2',
      commit: () => {
        throw new Error('display failed');
      }
    })
  ).rejects.toThrow('display failed');
  expect(rejected.destroy).toHaveBeenCalledOnce();
  expect(first.destroy).not.toHaveBeenCalled();
  expect(fixture.commit).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  await fixture.session.destroy();
});

test.each([
  'revision',
  'runtime'
])('host controller rejects %s changes during a rebuild', async change => {
  const fixture = makeFixture();
  const gate = makeGate();
  const started = makeGate();
  let runtime = {};
  let revision = 'one';
  fixture.create.mockImplementationOnce(async () => {
    started.resolve();
    await gate.promise;
    return {execute: vi.fn(async () => OUTPUT), destroy: vi.fn(async () => undefined)};
  });
  const controller = createAnalysisPreviewController({
    session: fixture.session,
    getRuntime: () => runtime,
    getRevisionKey: () => revision
  });
  const pending = controller.request(fixture.request);
  await started.promise;
  if (change === 'revision') revision = 'two';
  else runtime = {};
  gate.resolve();
  expect(await pending).toBe(false);
  expect(fixture.commit).not.toHaveBeenCalled();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
  await controller.destroy();
});

test('catalog cancellation reaches the GPU binding and never falls back to CPU', async () => {
  const fixture = makeFixture();
  const controller = new AbortController();
  const run = vi.fn(() => OUTPUT);
  const runGpu = vi.fn(async context => {
    expect(context.signal).toBe(controller.signal);
    controller.abort();
    throw new Error('aborted GPU');
  });
  await expect(
    runCatalogOperationAsync(
      {...fixture.binding, run, runGpu},
      fixture.request.parameters,
      fixture.request.inputs,
      {device: new NullDevice({}), signal: controller.signal}
    )
  ).rejects.toMatchObject({name: 'AbortError'});
  expect(run).not.toHaveBeenCalled();
  await fixture.session.destroy();
});

test('public compiled graph is retained, with exactly-once disposal after the actual fence', async () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const gate = makeGate();
  const compiled = {
    encode: vi.fn(),
    destroy: vi.fn()
  } as unknown as CompiledGPUCommandGraph<undefined>;
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue(compiled);
  const fenceDestroy = vi.fn();
  vi.spyOn(device, 'createFence').mockReturnValue({
    signaled: gate.promise,
    destroy: fenceDestroy
  } as unknown as ReturnType<NullDevice['createFence']>);
  const session = new CatalogGpuSession(device, 'retained');
  const output = session.output('value', 'float32', 1);
  const bufferDestroy = vi.spyOn(output.buffer, 'destroy');
  session.run();
  session.run();
  const disposal = session.dispose();
  expect(compiled.destroy).not.toHaveBeenCalled();
  expect(bufferDestroy).not.toHaveBeenCalled();
  gate.resolve();
  await disposal;
  await session.dispose();
  expect(compile).toHaveBeenCalledOnce();
  expect(compiled.encode).toHaveBeenCalledTimes(2);
  expect(compiled.destroy).toHaveBeenCalledOnce();
  expect(bufferDestroy).toHaveBeenCalledOnce();
  expect(fenceDestroy).toHaveBeenCalledTimes(2);
});

test('rejected fences keep GPU resources and admission pending until device loss establishes completion', async () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const deviceLoss = makeGate();
  Object.defineProperty(device, 'lost', {value: deviceLoss.promise});
  let rejectFence!: (error: Error) => void;
  const signaled = new Promise<void>((_resolve, reject) => {
    rejectFence = reject;
  });
  const fenceDestroy = vi.fn();
  vi.spyOn(device, 'createFence').mockReturnValue({
    signaled,
    destroy: fenceDestroy
  } as unknown as ReturnType<NullDevice['createFence']>);
  const compiled = {
    encode: vi.fn(),
    destroy: vi.fn()
  } as unknown as CompiledGPUCommandGraph<undefined>;
  vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue(compiled);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 40});
  const reservation = registry.reserveMemory(40);
  const session = new CatalogGpuSession(device, 'rejected-fence');
  const output = session.output('value', 'float32', 1);
  const bufferDestroy = vi.spyOn(output.buffer, 'destroy');
  session.run();
  let completionSettled = false;
  const completion = session.waitForCompletion().then(
    () => {
      completionSettled = true;
    },
    () => {
      completionSettled = true;
    }
  );
  let disposalSettled = false;
  const disposal = session.dispose().then(() => {
    disposalSettled = true;
    reservation.release();
  });
  rejectFence(new Error('Completion could not be established'));
  // Let the rejection and every resulting completion/disposal continuation run.
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  expect(completionSettled).toBe(false);
  expect(disposalSettled).toBe(false);
  expect(compiled.destroy).not.toHaveBeenCalled();
  expect(bufferDestroy).not.toHaveBeenCalled();
  expect(fenceDestroy).not.toHaveBeenCalled();
  expect(registry.memoryUsage.totalBytes).toBe(40);
  deviceLoss.resolve();
  await Promise.all([completion, disposal]);
  await session.dispose();
  expect(completionSettled).toBe(true);
  expect(compiled.destroy).toHaveBeenCalledOnce();
  expect(bufferDestroy).toHaveBeenCalledOnce();
  expect(fenceDestroy).toHaveBeenCalledOnce();
  expect(registry.memoryUsage.totalBytes).toBe(0);
});

test.each([
  'graph',
  'buffer',
  'graph and buffer'
])('actual GPU helper retries failed %s cleanup without repeating successful releases or losing admission', async failure => {
  const fixture = makeFixture();
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  vi.spyOn(device, 'createFence').mockReturnValue({
    signaled: Promise.resolve(),
    destroy: vi.fn()
  } as unknown as ReturnType<NullDevice['createFence']>);
  const graphDestroy = vi.fn();
  const graphFailure = new Error('Graph cleanup failed');
  const bufferFailure = new Error('Buffer cleanup failed');
  if (failure.includes('graph')) {
    graphDestroy
      .mockImplementationOnce(() => {
        throw graphFailure;
      })
      .mockImplementationOnce(() => {
        throw graphFailure;
      });
  }
  const compiled = {
    encode: vi.fn(),
    destroy: graphDestroy
  } as unknown as CompiledGPUCommandGraph<undefined>;
  vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue(compiled);
  let failingBufferDestroy!: ReturnType<typeof vi.fn>;
  let independentBufferDestroy!: ReturnType<typeof vi.fn>;
  fixture.create.mockImplementationOnce(async () => {
    const gpuSession = new CatalogGpuSession(device, 'retry-real-helper');
    const failing = gpuSession.output('failing', 'float32', 1);
    const independent = gpuSession.output('independent', 'float32', 1);
    failingBufferDestroy = vi.spyOn(failing.buffer, 'destroy');
    independentBufferDestroy = vi.spyOn(independent.buffer, 'destroy');
    if (failure.includes('buffer')) {
      failingBufferDestroy
        .mockImplementationOnce(() => {
          throw bufferFailure;
        })
        .mockImplementationOnce(() => {
          throw bufferFailure;
        });
    }
    return {
      execute: async () => {
        gpuSession.run();
        await gpuSession.waitForCompletion();
        return OUTPUT;
      },
      destroy: () => gpuSession.dispose()
    };
  });
  await fixture.session.request(fixture.request);
  expect(await fixture.session.request({...fixture.request, revisionKey: 'source-2'})).toBe(true);
  expect(fixture.commit).toHaveBeenCalledTimes(2);
  expect(fixture.session.takeCleanupErrors()).toEqual([expect.any(AggregateError)]);
  expect(fixture.registry.memoryUsage.totalBytes).toBe(80);
  // Even a failed graph or buffer release must not skip independent resources.
  expect(graphDestroy).toHaveBeenCalledOnce();
  expect(failingBufferDestroy).toHaveBeenCalledOnce();
  expect(independentBufferDestroy).toHaveBeenCalledOnce();
  await expect(fixture.session.destroy()).rejects.toThrow('AnalysisPreviewSession cleanup failed');
  expect(fixture.registry.memoryUsage.totalBytes).toBe(40);
  expect(independentBufferDestroy).toHaveBeenCalledOnce();
  await fixture.session.destroy();
  await fixture.session.destroy();
  expect(graphDestroy).toHaveBeenCalledTimes(failure.includes('graph') ? 3 : 1);
  expect(failingBufferDestroy).toHaveBeenCalledTimes(failure.includes('buffer') ? 3 : 1);
  expect(independentBufferDestroy).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test('point-buffer compilation and destructor failures retain their cleanup owner and both reservations until retry succeeds', async () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535
    }
  });
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const inputs: AnalysisPreviewRequest['inputs'] = {
    points: {
      type: 'FeatureCollection',
      features: [{id: 'source', geometry: {type: 'Point', coordinates: [0, 0]}}]
    }
  };
  const parameters = {coordinateSystem: 'planar', segmentsPerQuarter: 1, radius: 2};
  const plan = binding.planPreview!(inputs, parameters);
  const peak = plan.gpuByteLength + plan.cpuByteLength;
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: peak});
  const reserveMemory = registry.reserveMemory.bind(registry);
  const releases: ReturnType<typeof vi.fn>[] = [];
  vi.spyOn(registry, 'reserveMemory').mockImplementation((...reservationArguments) => {
    const reservation = reserveMemory(...reservationArguments);
    const release = vi.fn(() => reservation.release());
    releases.push(release);
    return {...reservation, release};
  });
  const compilationFailure = new Error('Point buffer compilation failed');
  vi.spyOn(GPUCommandGraph.prototype, 'compileAsync').mockRejectedValue(compilationFailure);
  const createBuffer = device.createBuffer.bind(device);
  const bufferDestroys: ReturnType<typeof vi.fn>[] = [];
  const destructionFailure = new Error('Point buffer cleanup failed');
  vi.spyOn(device, 'createBuffer').mockImplementation(properties => {
    const buffer = createBuffer(properties);
    const destroy = vi.spyOn(buffer, 'destroy');
    if (!bufferDestroys.length) {
      destroy
        .mockImplementationOnce(() => {
          throw destructionFailure;
        })
        .mockImplementationOnce(() => {
          throw destructionFailure;
        });
    }
    bufferDestroys.push(destroy);
    return buffer;
  });
  const submit = vi.spyOn(device, 'submit');
  const commit = vi.fn();
  const session = new AnalysisPreviewSession({registry, device, binding});
  await expect(
    session.request({inputs, parameters, revisionKey: 'source-1', isCurrent: () => true, commit})
  ).rejects.toBe(compilationFailure);
  expect(submit).not.toHaveBeenCalled();
  expect(commit).not.toHaveBeenCalled();
  expect(bufferDestroys).toHaveLength(3);
  for (const destroy of bufferDestroys) expect(destroy).toHaveBeenCalledOnce();
  expect(session.takeCleanupErrors()).toEqual([expect.any(AggregateError)]);
  expect(registry.memoryUsage.reservedBytes).toEqual({
    gpu: plan.gpuByteLength,
    cpu: plan.cpuByteLength
  });
  for (const release of releases) expect(release).not.toHaveBeenCalled();
  // A failed retry must keep the construction owner available for the next destruction attempt.
  await expect(session.destroy()).rejects.toThrow('AnalysisPreviewSession cleanup failed');
  expect(registry.memoryUsage.totalBytes).toBe(peak);
  expect(bufferDestroys[0]).toHaveBeenCalledTimes(2);
  for (const destroy of bufferDestroys.slice(1)) expect(destroy).toHaveBeenCalledOnce();
  for (const release of releases) expect(release).not.toHaveBeenCalled();
  await session.destroy();
  await session.destroy();
  expect(bufferDestroys[0]).toHaveBeenCalledTimes(3);
  for (const destroy of bufferDestroys.slice(1)) expect(destroy).toHaveBeenCalledOnce();
  expect(releases).toHaveLength(2);
  for (const release of releases) expect(release).toHaveBeenCalledOnce();
  expect(registry.memoryUsage.totalBytes).toBe(0);
});

test('failed fence destruction is retried without repeating successful graph and buffer cleanup', async () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const failure = new Error('Fence cleanup failed');
  const fenceDestroy = vi
    .fn()
    .mockImplementationOnce(() => {
      throw failure;
    })
    .mockImplementationOnce(() => {
      throw failure;
    });
  vi.spyOn(device, 'createFence').mockReturnValue({
    signaled: Promise.resolve(),
    destroy: fenceDestroy
  } as unknown as ReturnType<NullDevice['createFence']>);
  const compiled = {
    encode: vi.fn(),
    destroy: vi.fn()
  } as unknown as CompiledGPUCommandGraph<undefined>;
  vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue(compiled);
  const session = new CatalogGpuSession(device, 'retry-fence-cleanup');
  const output = session.output('value', 'float32', 1);
  const bufferDestroy = vi.spyOn(output.buffer, 'destroy');
  session.run();
  await expect(session.waitForCompletion()).rejects.toBe(failure);
  await expect(session.dispose()).rejects.toThrow('CatalogGpuSession cleanup failed');
  expect(compiled.destroy).toHaveBeenCalledOnce();
  expect(bufferDestroy).toHaveBeenCalledOnce();
  await session.destroy();
  await session.destroy();
  expect(fenceDestroy).toHaveBeenCalledTimes(3);
  expect(compiled.destroy).toHaveBeenCalledOnce();
  expect(bufferDestroy).toHaveBeenCalledOnce();
});

test('pending public async compilation finishes before disposal; allocation ceiling rejects before createBuffer', async () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const gate = makeGate();
  const compiled = {destroy: vi.fn()} as unknown as CompiledGPUCommandGraph<undefined>;
  vi.spyOn(GPUCommandGraph.prototype, 'compileAsync').mockImplementation(async () => {
    await gate.promise;
    return compiled;
  });
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const session = new CatalogGpuSession(device, 'limited', 4);
  expect(() => session.output('too-large', 'float32x2', 1)).toThrow(KeplerAllocationError);
  expect(createBuffer).not.toHaveBeenCalled();
  const compilation = expect(session.compile()).rejects.toThrow('session must be active');
  const disposal = session.dispose();
  expect(compiled.destroy).not.toHaveBeenCalled();
  gate.resolve();
  await compilation;
  await disposal;
  expect(compiled.destroy).toHaveBeenCalledOnce();
});

test('cleanup failure keeps old bytes accounted without rejecting the accepted display, then retries at destruction', async () => {
  const fixture = makeFixture();
  await fixture.session.request(fixture.request);
  const first = await fixture.create.mock.results[0].value;
  vi.mocked(first.destroy).mockRejectedValueOnce(new Error('retirement failed'));
  expect(await fixture.session.request({...fixture.request, revisionKey: 'source-2'})).toBe(true);
  expect(fixture.commit).toHaveBeenCalledTimes(2);
  expect(fixture.registry.memoryUsage.totalBytes).toBe(80);
  expect(fixture.session.takeCleanupErrors()).toEqual([new Error('retirement failed')]);
  await fixture.session.destroy();
  expect(first.destroy).toHaveBeenCalledTimes(2);
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test('external cancellation before scheduling allocates nothing and unsupported bindings are explicit', async () => {
  const fixture = makeFixture();
  const controller = new AbortController();
  controller.abort();
  expect(await fixture.session.request({...fixture.request, signal: controller.signal})).toBe(
    false
  );
  expect(fixture.create).not.toHaveBeenCalled();
  expect(
    () =>
      new AnalysisPreviewSession({
        registry: fixture.registry,
        device: new NullDevice({}),
        binding: {...fixture.binding, planPreview: undefined}
      })
  ).toThrow('must support retained previews');
  await fixture.session.destroy();
});

test('a device epoch change cannot reuse this session with an old device after recovery', async () => {
  const fixture = makeFixture();
  await fixture.session.request(fixture.request);
  fixture.registry.handleDeviceLoss();
  fixture.registry.recoverDevice();
  expect(await fixture.session.request(fixture.request)).toBe(false);
  expect(fixture.create).toHaveBeenCalledTimes(1);
  expect(fixture.commit).toHaveBeenCalledTimes(1);
  await fixture.session.destroy();
});

test('a host controller stays bound to its original runtime across later requests', async () => {
  const fixture = makeFixture();
  let runtime = {};
  const controller = createAnalysisPreviewController({
    session: fixture.session,
    getRuntime: () => runtime,
    getRevisionKey: () => 'same-revision'
  });
  await controller.request(fixture.request);
  runtime = {};
  expect(await controller.request(fixture.request)).toBe(false);
  expect(fixture.commit).toHaveBeenCalledOnce();
  await controller.destroy();
});

test('allocation errors never trigger CPU fallback, even on a binding that has an oracle', async () => {
  const fixture = makeFixture();
  const run = vi.fn(() => OUTPUT);
  await expect(
    runCatalogOperationAsync(
      {
        ...fixture.binding,
        run,
        runGpu: async () => {
          throw new KeplerAllocationError('reservation');
        }
      },
      fixture.request.parameters,
      fixture.request.inputs,
      {device: new NullDevice({})}
    )
  ).rejects.toBeInstanceOf(KeplerAllocationError);
  expect(run).not.toHaveBeenCalled();
  await fixture.session.destroy();
});
