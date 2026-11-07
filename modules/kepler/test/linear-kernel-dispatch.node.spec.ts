// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';

const dispatches: {x: number; y?: number; z?: number}[] = [];
const sources: string[] = [];

vi.mock('@luma-contributors/graph', async importOriginal => ({
  ...(await importOriginal<object>()),
  Kernel: class {
    constructor(_device: unknown, options: {source: string}) {
      sources.push(options.source);
    }
    dispatch(_pass: unknown, options: {x: number; y?: number; z?: number}) {
      dispatches.push(options);
    }
    destroy() {}
  }
}));

vi.mock('@luma-contributors/graph/gpu-core', async importOriginal => ({
  ...(await importOriginal<object>()),
  getViewBinding: () => ({}),
  getViewElementOffset: () => 0
}));

// Node workers share a module graph; reload so analysis-scaling binds to the mocks above.
vi.resetModules();
const {addKeplerLinearKernel} = await import('../src/analysis/analysis-scaling');

function runLinearKernel(length: number, maxComputeWorkgroupsPerDimension: number) {
  dispatches.length = 0;
  sources.length = 0;
  let pass: {compile: (context: unknown) => {encode: (context: unknown) => void}} | undefined;
  const graph = {addComputePass: (descriptor: typeof pass) => (pass = descriptor)};
  const view = {format: 'uint32', byteOffset: 0, byteStride: 4, stride: 1};
  addKeplerLinearKernel(
    graph as never,
    'linear',
    length,
    [{name: 'values', view: view as never}],
    ''
  );
  const compiled = pass!.compile({device: {limits: {maxComputeWorkgroupsPerDimension}}});
  compiled.encode({computePass: {}, getBuffer: () => ({})});
  return dispatches[0];
}

test('addKeplerLinearKernel keeps one-dimensional dispatches within the device limit', () => {
  expect(runLinearKernel(64 * 100, 65535)).toMatchObject({x: 100, y: 1, z: 1});
});

test('addKeplerLinearKernel splits dispatches above 65535 workgroups', () => {
  // 5M rows need 78125 workgroups, more than the default per-dimension limit.
  const dispatch = runLinearKernel(5_000_000, 65535);
  expect(dispatch.x).toBeLessThanOrEqual(65535);
  expect(dispatch.y).toBeGreaterThan(1);
  expect(dispatch.x * dispatch.y! * dispatch.z!).toBeGreaterThanOrEqual(78125);
  expect(sources[0]).toContain('workgroupId');
});

test('addKeplerLinearKernel honors a smaller fake device limit', () => {
  const dispatch = runLinearKernel(64 * 1000, 128);
  expect(dispatch.x).toBe(128);
  expect(dispatch.x * dispatch.y! * dispatch.z!).toBeGreaterThanOrEqual(1000);
});
