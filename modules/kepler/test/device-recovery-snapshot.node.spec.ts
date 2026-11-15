// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, it, vi} from 'vitest';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  captureResidentRecoverySnapshot,
  restoreResidentRecoveryState
} from '../../../examples/deck/kepler-compute/src/device-recovery';
import type {ComputeRuntime} from '../../../examples/deck/kepler-compute/src/compute-runtime';
import {saveSandbox} from '../../../examples/deck/kepler-compute/src/sandbox-state';
import {IncompleteSandboxPopulationError} from '../../../examples/deck/kepler-compute/src/sandbox-population';
import {SampleMapCheckpointDeferredError} from '../../../examples/deck/kepler-compute/src/sample-map-lifecycle';

it('recovery checkpoint detaches native metadata while retaining the original CPU table in memory', async () => {
  const context = fixture();
  try {
    const snapshot = captureResidentRecoverySnapshot(context);
    expect(snapshot.bundle.sources[0].table).toBe(context.table);
    expect(snapshot.bundle.state.nativeConfig).toEqual(context.getNativeConfig());
    expect(snapshot.bundle.state.nativeConfig).not.toBe(context.getNativeConfig());
    expect(snapshot.bundle.state.analyses).toEqual([]);
    context.nativeConfig.config.visState.filters[0].value[0] = 99;
    expect(JSON.stringify(snapshot.bundle.state.nativeConfig)).not.toContain('99');
    expect(snapshot.bundle.sources[0].table.getChild('value')!.get(0)).toBe(1);
  } finally {
    await context.cleanup();
  }
});

it('recovery checkpoint rejects device loss even when no analysis recipe exists', async () => {
  const context = fixture();
  try {
    await context.session.handleDeviceLoss();
    expect(() => captureResidentRecoverySnapshot(context)).toThrow('lost device');
  } finally {
    await context.cleanup();
  }
});

it('recovery checkpoint rejects a stale source and unrepresented native dataset', async () => {
  const context = fixture();
  try {
    context.inspector.isCurrent = false;
    expect(() => captureResidentRecoverySnapshot(context)).toThrow('stale resident dataset');
    context.inspector.isCurrent = true;
    context.nativeConfig.config.visState.filters[0].dataId = ['unretained-source'];
    expect(() => captureResidentRecoverySnapshot(context)).toThrow();
  } finally {
    await context.cleanup();
  }
});

it('actual capture identifies remote layers before generic manifest coverage failure', async () => {
  const context = fixture();
  try {
    const original = context.getNativeConfig();
    const remote = {
      version: 'v1',
      config: {visState: {layers: [{id: 'tiles', type: 'vectorTile', config: {dataId: 'remote'}}]}}
    };
    expect(() =>
      captureResidentRecoverySnapshot({...context, getNativeConfig: () => remote})
    ).toThrow(SampleMapCheckpointDeferredError);
    expect(context.getNativeConfig()).toEqual(original);
    expect(() =>
      captureResidentRecoverySnapshot({
        ...context,
        getNativeDatasets: () => [{type: 'raster-tile'}]
      })
    ).toThrow('Remote sample maps are not recoverable');
  } finally {
    await context.cleanup();
  }
});

it.each([
  'open',
  'cancelled',
  'failed'
] as const)('save and recovery reject %s streams until a completed revision exists', async state => {
  const context = fixture();
  try {
    context.inspector.dataset = context.session.registry.registerDataset({
      ...context.inspector.dataset,
      population: {kind: 'stream', streamId: 'stream', state}
    });
    expect(() => captureResidentRecoverySnapshot(context)).toThrow(
      IncompleteSandboxPopulationError
    );
    expect(() => saveSandbox({...context, applyNativeConfig() {}})).toThrow(
      IncompleteSandboxPopulationError
    );
    context.inspector.dataset = context.session.registry.registerDataset({
      ...context.inspector.dataset,
      population: {kind: 'stream', streamId: 'stream', state: 'complete'}
    });
    expect(captureResidentRecoverySnapshot(context).bundle.sources[0].table).toBe(context.table);
    expect(() => saveSandbox({...context, applyNativeConfig() {}})).not.toThrow();
  } finally {
    await context.cleanup();
  }
});

it('recovery replay validates retained sources and applies metadata to an already rebuilt runtime', async () => {
  const context = fixture();
  const replacement = fixture();
  try {
    const snapshot = captureResidentRecoverySnapshot(context);
    const applyNativeConfig = vi.fn(() => {
      replacement.inspector.currentRevision++;
    });
    const runtime = await restoreResidentRecoveryState(snapshot, {
      runtime: replacement.runtime,
      applyNativeConfig
    });
    expect(runtime).toBe(replacement.runtime);
    expect(applyNativeConfig).toHaveBeenCalledExactlyOnceWith(snapshot.bundle.state.nativeConfig);
    expect(replacement.runtime.requestUpdate).toHaveBeenCalledOnce();
    expect(snapshot.bundle.sources[0].table).toBe(context.table);
    const invalid = {...snapshot, bundle: {...snapshot.bundle, sources: []}};
    await expect(
      restoreResidentRecoveryState(invalid, {runtime: replacement.runtime, applyNativeConfig})
    ).rejects.toThrow('each declared source');
    expect(applyNativeConfig).toHaveBeenCalledTimes(1);
  } finally {
    await context.cleanup();
    await replacement.cleanup();
  }
});

it('an unfinished current registry revision defers capture while the previous preview is rebinding', async () => {
  const context = fixture();
  try {
    context.session.registry.registerDataset({
      ...context.inspector.dataset,
      population: {kind: 'stream', streamId: 'stream', state: 'open'}
    });
    context.inspector.isCurrent = false;
    expect(() => captureResidentRecoverySnapshot(context)).toThrow(
      IncompleteSandboxPopulationError
    );
  } finally {
    await context.cleanup();
  }
});

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const table = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))});
  const dataset = session.importArrowDataset({id: 'resident-observations', table});
  const inspector = {dataset, currentRevision: 1, isCurrent: true};
  const runtime = {
    device,
    session,
    explorer: inspector,
    publications: new Map(),
    analysis: {settings: {mode: 'off'}, setMode: vi.fn()},
    requestUpdate: vi.fn()
  } as unknown as ComputeRuntime;
  const nativeConfig = {
    version: 'v1',
    config: {
      visState: {
        filters: [
          {
            id: 'range',
            dataId: ['resident-observations'],
            name: ['value'],
            type: 'range',
            value: [1, 2]
          }
        ]
      }
    }
  };
  return {
    runtime,
    table,
    session,
    inspector,
    nativeConfig,
    getNativeConfig: () => nativeConfig,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}
