// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {afterEach, expect, test, vi} from 'vitest';

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
  vi.restoreAllMocks();
});

function createFixture(retain?: readonly string[]) {
  const device = new NullDevice({id: 'resource-plan-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  cleanups.push(() => device.destroy());
  const graph = new GPUCommandGraph(device, {id: 'resource-plan', recording: {retain}});
  const compile = vi.fn(() => ({encode: vi.fn(), destroy: vi.fn()}));
  const compileAsync = vi.fn(async () => compile());
  for (const [index, id] of ['first', 'second'].entries()) {
    const buffer = graph.createTransientBuffer({id, byteLength: 32, usage: Buffer.STORAGE});
    const texture = graph.createTransientTexture({
      id: `${id}-texture`,
      format: 'rgba8unorm',
      width: 4,
      height: 4,
      mipLevels: 2,
      usage: Texture.COPY_DST
    });
    graph.addCopyPass({
      id: `${id}-node`,
      dependsOn: index ? ['first-node'] : [],
      resources: [
        {buffer, usage: 'storage-write'},
        {texture, usage: 'copy-destination'}
      ],
      workload: {commandCount: 2, writeByteLength: 112},
      compile,
      compileAsync
    });
  }
  return {device, graph, compile, compileAsync};
}

test.each([
  false,
  true
])('dry-run matches aliased allocation and preflight (async=%s)', async asynchronous => {
  const {device, graph, compile, compileAsync} = createFixture();
  graph.importBuffer({id: 'borrowed', byteLength: 128, usage: Buffer.STORAGE});
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const createTexture = vi.spyOn(device, 'createTexture');
  const plan = graph.planResources();
  expect(createBuffer).not.toHaveBeenCalled();
  expect(createTexture).not.toHaveBeenCalled();
  expect(compile).not.toHaveBeenCalled();
  expect(compileAsync).not.toHaveBeenCalled();
  expect(plan.stats).toMatchObject({
    importedBufferBytes: 128,
    logicalTransientBytes: 64,
    physicalTransientBytes: 32,
    logicalTransientTextureBytes: 160,
    physicalTransientTextureBytes: 80,
    physicalTransientResourceBytes: 112
  });
  expect(plan.bufferAllocations).toEqual([
    {byteLength: 32, usage: Buffer.STORAGE, resourceIds: ['first', 'second']}
  ]);
  expect(plan.textureAllocations[0].resourceIds).toEqual(['first-texture', 'second-texture']);
  const compiled = asynchronous ? await graph.compileAsync() : graph.compile();
  cleanups.push(() => compiled.destroy());
  expect(compiled.stats).toEqual(plan.stats);
  expect(compiled.preflight).toEqual(plan.preflight);
  expect(createBuffer).toHaveBeenCalledTimes(1);
  expect(createBuffer.mock.calls[0][0]).toMatchObject({
    byteLength: 32,
    usage: plan.bufferAllocations[0].usage
  });
  expect(createTexture).toHaveBeenCalledTimes(1);
  expect(createTexture.mock.calls[0][0]).toMatchObject({
    format: 'rgba8unorm',
    width: 4,
    height: 4,
    mipLevels: 2
  });
});

test('retained resources and explicit observation taps affect the same allocation plan', () => {
  const {device, graph} = createFixture(['first', 'first-texture']);
  const tapped = graph.createTransientBuffer({id: 'tapped', byteLength: 16, usage: Buffer.STORAGE});
  const sink = graph.importBuffer({id: 'sink', byteLength: 16, usage: Buffer.COPY_DST});
  graph.addCopyPass({
    id: 'write-tapped',
    resources: [{buffer: tapped, usage: 'storage-write'}],
    compile: () => ({encode: vi.fn()})
  });
  graph.addObservationTap(tapped, {afterNode: 'write-tapped', sink, id: 'observe'});
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const plan = graph.planResources();
  expect(createBuffer).not.toHaveBeenCalled();
  expect(plan.retainedResourceIds).toEqual(['first', 'first-texture']);
  expect(plan.observationTapIds).toEqual(['observe']);
  expect(plan.observerCopySourceIds).toEqual(['first', 'tapped']);
  expect(plan.stats.physicalTransientTextureCount).toBe(2);
  expect(
    plan.bufferAllocations.find(allocation => allocation.resourceIds.includes('first'))?.usage
  ).toBe(Buffer.STORAGE | Buffer.COPY_SRC);
  const compiled = graph.compile();
  cleanups.push(() => compiled.destroy());
  expect(compiled.stats).toEqual(plan.stats);
  expect(compiled.preflight).toEqual(plan.preflight);
});

test('returns deeply frozen snapshots without freezing graph authoring', () => {
  const {graph} = createFixture();
  const plan = graph.planResources();
  function expectFrozen(value: unknown) {
    if (value && typeof value === 'object') {
      expect(Object.isFrozen(value)).toBe(true);
      for (const entry of Object.values(value)) expectFrozen(entry);
    }
  }
  expectFrozen(plan);
  expect(Reflect.set(plan.stats, 'physicalTransientResourceBytes', 0)).toBe(false);
  graph.createTransientBuffer({id: 'unused', byteLength: 64, usage: Buffer.STORAGE});
  expect(plan.stats.logicalTransientBytes).toBe(64);
  expect(graph.planResources().stats.logicalTransientBytes).toBe(128);
  expect(graph.planResources().stats.physicalTransientResourceBytes).toBe(112);
});

test('rejects invalid retention before allocating or compiling', () => {
  const {device, graph, compile} = createFixture(['missing']);
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const createTexture = vi.spyOn(device, 'createTexture');
  expect(() => graph.planResources()).toThrow('not a transient resource');
  expect(() => graph.compile()).toThrow('not a transient resource');
  expect(createBuffer).not.toHaveBeenCalled();
  expect(createTexture).not.toHaveBeenCalled();
  expect(compile).not.toHaveBeenCalled();
});

test.each([
  false,
  true
])('planning preserves callback failure cleanup (async=%s)', async asynchronous => {
  const {device, graph, compile} = createFixture();
  graph.addCopyPass({
    id: 'failure',
    dependsOn: ['second-node'],
    compile: () => {
      throw new Error('callback failure');
    }
  });
  graph.planResources();
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const createTexture = vi.spyOn(device, 'createTexture');
  if (asynchronous) {
    await expect(graph.compileAsync()).rejects.toThrow('callback failure');
  } else {
    expect(() => graph.compile()).toThrow('callback failure');
  }
  for (const result of createBuffer.mock.results) expect(result.value.destroyed).toBe(true);
  for (const result of createTexture.mock.results) expect(result.value.destroyed).toBe(true);
  for (const result of compile.mock.results) expect(result.value.destroy).toHaveBeenCalledTimes(1);
});

test.each([
  false,
  true
])('planning preserves allocation failure cleanup (async=%s)', async asynchronous => {
  const {device, graph, compile} = createFixture();
  graph.planResources();
  const createBuffer = vi.spyOn(device, 'createBuffer');
  vi.spyOn(device, 'createTexture').mockImplementation(() => {
    throw new Error('allocation failure');
  });
  if (asynchronous) {
    await expect(graph.compileAsync()).rejects.toThrow('allocation failure');
  } else {
    expect(() => graph.compile()).toThrow('allocation failure');
  }
  expect(createBuffer).toHaveBeenCalledTimes(1);
  expect(createBuffer.mock.results[0].value.destroyed).toBe(true);
  expect(compile).not.toHaveBeenCalled();
});
