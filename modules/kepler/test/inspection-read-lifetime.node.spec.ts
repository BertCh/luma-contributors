// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUResidentExplorer} from '../src/exploration/gpu-resident-explorer';
import {readAllBuffers} from '../src/exploration/read-all-buffers';

test('inspection read batches preserve the first input error and wait for every mapping, including synchronous throws', async () => {
  const failure = new Error('First mapping failed');
  let resolve!: (value: Uint8Array) => void;
  const mapping = new Promise<Uint8Array>(complete => {
    resolve = complete;
  });
  const sibling = vi.fn(() => mapping);
  let settled = false;
  const reading = readAllBuffers([
    () => {
      throw failure;
    },
    sibling
  ]);
  const result = reading.catch(error => {
    settled = true;
    return error;
  });
  await Promise.resolve();
  await Promise.resolve();
  expect(sibling).toHaveBeenCalledOnce();
  expect(settled).toBe(false);
  resolve(new Uint8Array(4));
  expect(await result).toBe(failure);
});

test('explorer destruction waits for sibling summary mappings after an early rejection', async () => {
  let resolve!: (value: Uint8Array) => void;
  const pending = new Promise<Uint8Array>(complete => {
    resolve = complete;
  });
  const destroy = vi.fn();
  // Controlled IO fixture exercises the real read/destroy methods without compiling a GPU graph.
  const explorer: GPUResidentExplorer = Object.assign(
    Object.create(GPUResidentExplorer.prototype),
    {
      selectedCount: {byteLength: 4, readAsync: () => Promise.reject(new Error('Mapping failed'))},
      histogramBuffers: new Map([['histogram', {byteLength: 4, readAsync: () => pending}]]),
      domainBuffers: new Map(),
      predicates: new Map(),
      domainPredicates: new Map(),
      domainFilters: [],
      vectors: [],
      compiled: {destroy},
      filter: {destroy: vi.fn()},
      destroyed: false,
      activeReads: 0,
      maximumConcurrentReads: 1,
      revision: 1,
      encodedRevision: 1,
      transferredBytes: 0,
      resourcesReleased: false,
      releaseWaiters: []
    }
  );
  const read = explorer.readSummary();
  const result = read.catch(error => error);
  await Promise.resolve();
  await Promise.resolve();
  expect(await explorer.readSummary()).toBeNull();
  const closing = explorer.destroyAsync();
  await Promise.resolve();
  expect(destroy).not.toHaveBeenCalled();
  resolve(new Uint8Array(4));
  expect((await result).message).toBe('Mapping failed');
  await closing;
  expect(destroy).toHaveBeenCalledOnce();
});
