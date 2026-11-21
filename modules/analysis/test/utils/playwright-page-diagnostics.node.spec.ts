// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EventEmitter} from 'node:events';
import {afterEach, expect, it, vi} from 'vitest';
import {
  collectPageDiagnostics,
  probeWebGPU
} from '../../../../scripts/playwright/collect-page-diagnostics.mjs';

afterEach(() => vi.unstubAllGlobals());

it('UF-12 collects console messages and page errors, and disposes only its own listeners', () => {
  const page = new EventEmitter();
  const otherListener = vi.fn();
  page.on('console', otherListener);
  const {diagnostics, dispose} = collectPageDiagnostics(page);
  const message = {
    type: () => 'error',
    text: () => 'WebGPU validation error',
    location: () => ({url: 'http://localhost/scene.js', lineNumber: 3, columnNumber: 4})
  };
  page.emit('console', message);
  const error = new TypeError('scene failed');
  page.emit('pageerror', error);
  expect(diagnostics.consoleMessages).toEqual([
    {type: 'error', text: 'WebGPU validation error', location: message.location()}
  ]);
  expect(diagnostics.pageErrors).toEqual([
    {name: 'TypeError', message: 'scene failed', stack: error.stack}
  ]);
  expect(() => JSON.stringify(diagnostics)).not.toThrow();
  dispose();
  dispose();
  page.emit('console', message);
  page.emit('pageerror', new Error('after disposal'));
  expect(diagnostics.consoleMessages).toHaveLength(1);
  expect(diagnostics.pageErrors).toHaveLength(1);
  expect(otherListener).toHaveBeenCalledTimes(2);
  expect(page.listenerCount('console')).toBe(1);
  expect(page.listenerCount('pageerror')).toBe(0);
});

const page = {evaluate: (callback: () => Promise<unknown>) => callback()};

it('UF-12 reports an absent WebGPU API and absent adapter without requesting a device', async () => {
  vi.stubGlobal('navigator', {});
  expect(await probeWebGPU(page)).toEqual({supported: false, adapter: null});
  const requestAdapter = vi.fn(async () => null);
  vi.stubGlobal('navigator', {gpu: {requestAdapter}});
  expect(await probeWebGPU(page)).toEqual({supported: true, adapter: null});
  expect(requestAdapter).toHaveBeenCalledTimes(1);
});

it('UF-12 serializes adapter information, features and prototype-backed limits', async () => {
  const limits = Object.create({maxBufferSize: 1024, maxComputeInvocationsPerWorkgroup: 64});
  const requestDevice = vi.fn();
  vi.stubGlobal('navigator', {
    gpu: {
      requestAdapter: async () => ({
        info: {
          vendor: 'fixture',
          architecture: 'test',
          device: 'literal',
          description: 'page double'
        },
        isFallbackAdapter: true,
        features: new Set(['shader-f16']),
        limits,
        requestDevice
      })
    }
  });
  expect(await probeWebGPU(page)).toEqual({
    supported: true,
    adapter: {
      info: {
        vendor: 'fixture',
        architecture: 'test',
        device: 'literal',
        description: 'page double'
      },
      isFallbackAdapter: true,
      features: ['shader-f16'],
      limits: {maxBufferSize: 1024, maxComputeInvocationsPerWorkgroup: 64}
    }
  });
  expect(requestDevice).not.toHaveBeenCalled();
});

it('B11 reads fallback status from GPUAdapterInfo before the legacy adapter property', async () => {
  vi.stubGlobal('navigator', {
    gpu: {
      requestAdapter: async () => ({
        info: {isFallbackAdapter: true},
        isFallbackAdapter: false,
        features: new Set(),
        limits: {}
      })
    }
  });
  const result = await probeWebGPU(page);
  expect(result.adapter.isFallbackAdapter).toBe(true);
});
