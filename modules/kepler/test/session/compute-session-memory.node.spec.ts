// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {afterAll, beforeEach, expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../../src/session/compute-session';

const state = vi.hoisted(() => {
  // Node workers share imported modules; rebind this session import to this file's explorer mock.
  vi.resetModules();
  return {
    resident: 64,
    inspection: 32,
    construct: () => {},
    cleanup: Promise.resolve(),
    destroys: 0
  };
});
vi.mock('../../src/exploration/gpu-resident-explorer', () => ({
  GPUResidentExplorer: class {
    selectionMask = undefined;
    constructor() {
      state.construct();
    }
    get accountedResidentByteLength() {
      return state.resident;
    }
    get inspectionReservationByteLength() {
      return state.inspection;
    }
    destroy() {
      state.destroys++;
    }
    async destroyAsync() {
      await state.cleanup;
      this.destroy();
    }
  }
}));
afterAll(() => {
  // Do not leave a cached session bound to the fake explorer for subsequent shared-worker suites.
  vi.doUnmock('../../src/exploration/gpu-resident-explorer');
  vi.resetModules();
});
beforeEach(() => {
  state.resident = 64;
  state.inspection = 32;
  state.construct = () => {};
  state.cleanup = Promise.resolve();
  state.destroys = 0;
});
function fixture(budget = 1024) {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: budget});
  const dataset = session.importArrowDataset({
    id: 'data',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1]))})
  });
  return {device, session, dataset};
}

test('explorer reserves its hard cap before construction then charges its resident plan and bounded inspections', async () => {
  const {device, session} = fixture();
  state.construct = () => {
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(512);
  };
  const explorer = session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 512});
  expect(explorer.workingMemoryBudgetBytes).toBe(512);
  expect(explorer.reservedMemoryByteLength).toBe(96);
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(96);
  // A second owner can now admit its unchanged hard cap while the first retains only its charge.
  state.construct = () => {
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(608);
  };
  const second = session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 512});
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(192);
  await second.close(Promise.resolve());
  let completeFence!: () => void;
  let completeRead!: () => void;
  state.cleanup = new Promise<void>(resolve => {
    completeRead = resolve;
  });
  const closing = explorer.close(
    new Promise<void>(resolve => {
      completeFence = resolve;
    })
  );
  expect(explorer.reservedMemoryByteLength).toBe(96);
  completeFence();
  await Promise.resolve();
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(96);
  completeRead();
  await closing;
  expect(explorer.reservedMemoryByteLength).toBe(0);
  expect(explorer.workingMemoryBudgetBytes).toBe(512);
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('hard-cap admission and under-sized plan failures leave no explorer reservation or source lease', async () => {
  const {device, session, dataset} = fixture();
  const construct = vi.fn();
  state.construct = construct;
  expect(() =>
    session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 2048})
  ).toThrow(/budget/);
  expect(construct).not.toHaveBeenCalled();
  state.resident = 100;
  state.inspection = 40;
  expect(() =>
    session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 128})
  ).toThrow(/inspection/);
  expect(state.destroys).toBe(1);
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  await session.removeDataset('data', Promise.resolve());
  expect(dataset.value.columns['value'].data[0].buffer.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('throwing explorer construction releases its preallocation charge', async () => {
  const {device, session} = fixture();
  state.construct = () => {
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(512);
    throw new Error('Compiler rejected graph');
  };
  expect(() =>
    session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 512})
  ).toThrow(/Compiler/);
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('device loss releases the live charge while preserving the original explorer hard cap', async () => {
  const {device, session} = fixture();
  const explorer = session.openExplorer('data', {dimensions: [], workingMemoryBudgetBytes: 512});
  expect(explorer.reservedMemoryByteLength).toBe(96);
  const loss = session.handleDeviceLoss();
  expect(explorer.reservedMemoryByteLength).toBe(0);
  expect(explorer.workingMemoryBudgetBytes).toBe(512);
  await loss;
  await session.destroy(Promise.resolve());
  device.destroy();
});
