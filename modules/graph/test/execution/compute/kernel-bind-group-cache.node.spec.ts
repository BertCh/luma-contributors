// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {Kernel} from '../../../src/execution/compute/kernel';

type BindingOptions = {_bindGroupCacheKeys?: Partial<Record<number, object>>} | undefined;

function createKernelHarness(limit = 65535): {
  kernel: Kernel;
  dispatch: (bindings: Record<string, any>) => BindingOptions;
} {
  const kernel = Object.create(Kernel.prototype) as Kernel;
  Object.assign(kernel, {
    device: {limits: {maxComputeWorkgroupsPerDimension: limit}},
    pipeline: {shaderLayout: {bindings: [{group: 0}, {group: 0}]}},
    _bindingCacheEntries: []
  });
  let lastOptions: BindingOptions;
  const computePass = {
    setPipeline: () => {},
    setBindings: (_bindings: unknown, options?: BindingOptions) => {
      lastOptions = options;
    },
    dispatch: () => {}
  };
  return {
    kernel,
    dispatch: bindings => {
      lastOptions = undefined;
      kernel.dispatch(computePass as any, {bindings, x: 1});
      return lastOptions;
    }
  };
}

test('Kernel reuses bind-group cache keys for identical buffer bindings', () => {
  const device = new NullDevice({});
  const first = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  const second = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  const {dispatch} = createKernelHarness();

  const initialKeys = dispatch({a: first, b: second})?._bindGroupCacheKeys;
  expect(initialKeys?.[0]).toBeDefined();
  expect(dispatch({a: first, b: second})?._bindGroupCacheKeys?.[0]).toBe(initialKeys![0]);
  device.destroy();
});

test('Kernel creates new cache keys when a binding buffer, offset or size changes', () => {
  const device = new NullDevice({});
  const first = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  const reallocated = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  const {dispatch} = createKernelHarness();

  const initialKey = dispatch({a: first})?._bindGroupCacheKeys?.[0];
  expect(dispatch({a: reallocated})?._bindGroupCacheKeys?.[0]).not.toBe(initialKey);
  expect(dispatch({a: {buffer: first, offset: 256, size: 256}})?._bindGroupCacheKeys?.[0]).not.toBe(
    initialKey
  );
  const rangeKey = dispatch({a: {buffer: first, offset: 256, size: 256}})?._bindGroupCacheKeys?.[0];
  expect(dispatch({a: {buffer: first, offset: 256, size: 512}})?._bindGroupCacheKeys?.[0]).not.toBe(
    rangeKey
  );
  // Returning to a previous binding set hits its cache entry again.
  expect(dispatch({a: first})?._bindGroupCacheKeys?.[0]).toBe(initialKey);
  device.destroy();
});

test('Kernel does not cache bind groups that reference destroyed buffers or textures', () => {
  const device = new NullDevice({});
  const buffer = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  const {dispatch} = createKernelHarness();

  expect(dispatch({a: buffer})?._bindGroupCacheKeys).toBeDefined();
  buffer.destroy();
  expect(dispatch({a: buffer})?._bindGroupCacheKeys).toBeUndefined();

  const texture = device.createTexture({width: 1, height: 1, format: 'rgba8unorm'});
  const liveBuffer = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  expect(dispatch({a: liveBuffer, texture})?._bindGroupCacheKeys).toBeUndefined();
  device.destroy();
});

test('Kernel respects explicit bind-group cache keys', () => {
  const device = new NullDevice({});
  const buffer = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  const {kernel} = createKernelHarness();
  const explicitKeys = {0: {}};
  let received: BindingOptions;
  kernel.dispatch(
    {
      setPipeline: () => {},
      setBindings: (_bindings: unknown, options?: BindingOptions) => {
        received = options;
      },
      dispatch: () => {}
    } as any,
    {bindings: {a: buffer}, _bindGroupCacheKeys: explicitKeys, x: 1}
  );
  expect(received?._bindGroupCacheKeys).toBe(explicitKeys);
  device.destroy();
});

test('Kernel rejects dispatches beyond maxComputeWorkgroupsPerDimension', () => {
  const {kernel} = createKernelHarness(4);
  const computePass = {setPipeline: () => {}, setBindings: () => {}, dispatch: () => {}};
  expect(() => kernel.dispatch(computePass as any, {x: 4, y: 4, z: 4})).not.toThrow();
  expect(() => kernel.dispatch(computePass as any, {x: 5})).toThrow(/maxCompute/);
  expect(() => kernel.dispatch(computePass as any, {x: 1, y: 5})).toThrow(/maxCompute/);
  expect(() => kernel.dispatch(computePass as any, {x: 1, y: 1, z: 5})).toThrow(/maxCompute/);
});
