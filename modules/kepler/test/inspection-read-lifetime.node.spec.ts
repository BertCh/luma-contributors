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

test.each([
  {phase: 'mapping', mappingOutcome: 'reject'},
  {phase: 'finish', mappingOutcome: 'resolve'},
  {phase: 'finish', mappingOutcome: 'reject'},
  {phase: 'submit', mappingOutcome: 'resolve'},
  {phase: 'submit', mappingOutcome: 'reject'}
])('explorer keeps admission and destruction pending after $phase failure ($mappingOutcome)', async ({
  phase,
  mappingOutcome
}) => {
  const failure = new Error(`${phase} failed`);
  let resolve!: (bytes: Uint8Array) => void;
  let reject!: (error: Error) => void;
  const pending = new Promise<Uint8Array>((complete, fail) => {
    resolve = complete;
    reject = fail;
  });
  const destroy = vi.fn();
  const destroyRing = vi.fn();
  // One staging ticket carries the whole summary; its mapping is the only GPU wait.
  const ticket = {buffer: {}, markEncoded: vi.fn(), read: vi.fn(() => pending), cancel: vi.fn()};
  // Controlled IO fixture exercises the real read/destroy methods without compiling a GPU graph.
  const explorer: GPUResidentExplorer = Object.assign(
    Object.create(GPUResidentExplorer.prototype),
    {
      device: {
        createCommandEncoder: () => ({
          copyBufferToBuffer: vi.fn(),
          finish: () => {
            if (phase === 'finish') throw failure;
          }
        }),
        submit: () => {
          if (phase === 'submit') throw failure;
        }
      },
      readbackRings: [{tryAcquire: () => ticket, destroy: destroyRing}],
      selectedCount: {byteLength: 4},
      histogramBuffers: new Map([['histogram', {byteLength: 4}]]),
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
  let readSettled = false;
  const result = read.catch(error => {
    readSettled = true;
    return error;
  });
  // Let all ready continuations run while the controlled mapping stays pending.
  await new Promise<void>(complete => setTimeout(complete, 0));
  expect(ticket.read).toHaveBeenCalledOnce();
  expect(readSettled).toBe(false);
  expect(await explorer.readSummary()).toBeNull();
  let destructionSettled = false;
  const closing = explorer.destroyAsync().then(() => {
    destructionSettled = true;
  });
  await Promise.resolve();
  expect(destructionSettled).toBe(false);
  expect(destroy).not.toHaveBeenCalled();
  expect(ticket.markEncoded).toHaveBeenCalledWith({byteOffset: 0, byteLength: 8});
  if (mappingOutcome === 'resolve') resolve(new Uint8Array(8));
  else reject(phase === 'mapping' ? failure : new Error('Cleanup mapping failed'));
  expect(await result).toBe(failure);
  await closing;
  expect(destroy).toHaveBeenCalledOnce();
  expect(destroyRing).toHaveBeenCalledOnce();
  expect(ticket.cancel).not.toHaveBeenCalled();
});
