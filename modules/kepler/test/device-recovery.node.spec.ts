// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  ResidentDeviceRecovery,
  type ResidentDeviceRecoveryProps,
  type ResidentRecoveryContext,
  type ResidentRecoverySnapshot
} from '../../../examples/deck/kepler-compute/src/resident-device-recovery';
import {IncompleteSandboxPopulationError} from '../../../examples/deck/kepler-compute/src/sandbox-population';

type Runtime = {id: string};

it('recovery retains in-memory CPU references and waits for fenced teardown before one explicit rebuild', async () => {
  const {recovery, props, previous, replacement, snapshot, table, events} = harness();
  const fence = deferred<void>();
  const statuses: string[] = [];
  const unsubscribe = recovery.subscribe(status => {
    statuses.push(status.phase);
  });
  expect(recovery.checkpoint(previous)).toBe(true);
  expect(recovery.markLost(previous, 'Device removed', fence.promise)).toBe(true);
  expect(events).toEqual(['capture:previous', 'invalidate:previous']);
  await Promise.resolve();
  expect(props.rebuild).not.toHaveBeenCalled();
  const recovered = recovery.recover();
  expect(recovery.recover()).toBe(recovered);
  await Promise.resolve();
  expect(props.rebuild).not.toHaveBeenCalled();
  expect(recovery.status.phase).toBe('rebuilding');
  fence.resolve();
  expect(await recovered).toBe(replacement);
  expect(props.rebuild).toHaveBeenCalledExactlyOnceWith(
    snapshot,
    expect.objectContaining({attempt: 1})
  );
  expect(props.restore).toHaveBeenCalledWith(replacement, snapshot, expect.anything());
  expect(events).toEqual([
    'capture:previous',
    'invalidate:previous',
    'teardown:start:previous',
    'teardown:end:previous',
    'rebuild',
    'restore',
    'capture:replacement',
    'publish'
  ]);
  expect(snapshot.bundle.sources[0].table).toBe(table);
  expect(recovery.status).toMatchObject({
    phase: 'ready',
    attempt: 1,
    hasCheckpoint: true,
    sourceCount: 1,
    reason: null
  });
  expect(statuses).toEqual(['ready', 'lost', 'rebuilding', 'ready']);
  unsubscribe();
  await recovery.dispose();
});

it('failed device creation is surfaced and retried only by a separate explicit action', async () => {
  const {recovery, props, previous, replacement} = harness();
  props.rebuild = vi
    .fn()
    .mockRejectedValueOnce(new Error('Adapter unavailable'))
    .mockResolvedValueOnce(replacement);
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('Adapter unavailable');
  expect(recovery.status).toMatchObject({
    phase: 'failed',
    attempt: 1,
    reason: 'Adapter unavailable',
    hasCheckpoint: true
  });
  await Promise.resolve();
  expect(props.rebuild).toHaveBeenCalledTimes(1);
  expect(props.discard).not.toHaveBeenCalled();
  expect(await recovery.recover()).toBe(replacement);
  expect(props.rebuild).toHaveBeenCalledTimes(2);
  expect(props.teardown).toHaveBeenCalledTimes(1);
  await recovery.dispose();
});

it('replay failure discards the candidate, preserves the checkpoint and never publishes partial state', async () => {
  const {recovery, props, previous, replacement} = harness();
  props.restore = vi.fn(async () => {
    throw new Error('Missing recovered source');
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('Missing recovered source');
  expect(props.discard).toHaveBeenCalledExactlyOnceWith(replacement);
  expect(props.publish).not.toHaveBeenCalled();
  expect(recovery.status).toMatchObject({phase: 'failed', hasCheckpoint: true});
  await recovery.dispose();
});

it('closing during device initialization aborts and discards a late runtime without restoring or publishing it', async () => {
  const {recovery, props, previous, replacement} = harness();
  const created = deferred<Runtime>();
  const started = deferred<ResidentRecoveryContext>();
  props.rebuild = vi.fn((_snapshot, context) => {
    started.resolve(context);
    return created.promise;
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  const recovering = recovery.recover();
  const context = await started.promise;
  const disposing = recovery.dispose();
  expect(context.signal.aborted).toBe(true);
  expect(recovery.status).toMatchObject({phase: 'disposed', hasCheckpoint: false});
  created.resolve(replacement);
  await expect(recovering).rejects.toThrow('cancelled');
  await disposing;
  expect(props.discard).toHaveBeenCalledExactlyOnceWith(replacement);
  expect(props.restore).not.toHaveBeenCalled();
  expect(props.publish).not.toHaveBeenCalled();
  await expect(recovery.recover()).rejects.toThrow('disposed');
});

it('cancelling an attempt retains its checkpoint and waits for cleanup before an explicit retry', async () => {
  const {recovery, props, previous, replacement} = harness();
  const created = deferred<Runtime>();
  const started = deferred<ResidentRecoveryContext>();
  const discarded = deferred<void>();
  props.rebuild = vi.fn((_snapshot, context) => {
    started.resolve(context);
    return created.promise;
  });
  props.discard = vi.fn(() => discarded.promise);
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  const recovering = recovery.recover();
  const rejected = expect(recovering).rejects.toThrow('cancelled');
  const context = await started.promise;
  const cancelling = recovery.cancelRecovery();
  expect(context.signal.aborted).toBe(true);
  expect(recovery.recover()).toBe(recovering);
  created.resolve(replacement);
  await Promise.resolve();
  expect(recovery.status.phase).toBe('rebuilding');
  expect(props.publish).not.toHaveBeenCalled();
  discarded.resolve();
  await cancelling;
  await rejected;
  expect(recovery.status).toMatchObject({phase: 'failed', hasCheckpoint: true, attempt: 1});
  expect(props.discard).toHaveBeenCalledExactlyOnceWith(replacement);
  props.rebuild = vi.fn(async () => replacement);
  expect(await recovery.recover()).toBe(replacement);
  expect(recovery.status).toMatchObject({phase: 'ready', attempt: 2});
  await recovery.dispose();
});

it('throwing observers cannot interrupt recovery or hide teardown failures', async () => {
  const {recovery, props, previous} = harness();
  const report = vi.spyOn(console, 'error').mockImplementation(() => {});
  const observed: string[] = [];
  recovery.subscribe(() => {
    throw new Error('Broken observer');
  });
  recovery.subscribe(status => observed.push(status.phase));
  props.teardown = vi.fn(async () => {
    throw new Error('Resource release failed');
  });
  try {
    expect(recovery.checkpoint(previous)).toBe(true);
    recovery.markLost(previous, 'Device lost');
    await expect(recovery.recover()).rejects.toThrow('Resource release failed');
    expect(recovery.status.reason).toBe('Resource release failed');
    expect(observed).toEqual(['ready', 'lost', 'rebuilding', 'failed']);
    expect(report).toHaveBeenCalled();
    await expect(recovery.dispose()).rejects.toThrow('Resource release failed');
  } finally {
    report.mockRestore();
  }
});

it('loss events from retired runtimes cannot invalidate a recovered workspace', async () => {
  const {recovery, props, previous, replacement} = harness();
  recovery.checkpoint(previous);
  expect(recovery.markLost(previous, 'first loss')).toBe(true);
  expect(recovery.markLost(previous, 'duplicate loss')).toBe(false);
  await recovery.recover();
  expect(recovery.markLost(previous, 'late callback')).toBe(false);
  expect(recovery.status.phase).toBe('ready');
  expect(recovery.markLost(replacement, 'replacement loss')).toBe(true);
  expect(props.invalidate).toHaveBeenCalledTimes(2);
  expect(props.rebuild).toHaveBeenCalledTimes(1);
  await recovery.dispose();
});

it('a failed healthy checkpoint removes older state rather than silently replaying a stale workspace', async () => {
  const {recovery, props, previous} = harness();
  recovery.checkpoint(previous);
  props.capture = vi.fn(() => {
    throw new Error('Source manifest does not cover a new dataset');
  });
  expect(recovery.checkpoint(previous)).toBe(false);
  expect(recovery.status).toMatchObject({
    phase: 'ready',
    hasCheckpoint: false,
    sourceCount: 0,
    reason: 'Source manifest does not cover a new dataset'
  });
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('No healthy recovery checkpoint');
  expect(props.rebuild).not.toHaveBeenCalled();
  await recovery.dispose();
});

it('candidate checkpoint validation precedes publication and leaves the previous recovery sources available', async () => {
  const {recovery, props, previous, replacement, snapshot} = harness();
  props.capture = vi.fn(runtime => {
    if (runtime === replacement) throw new Error('Replacement lost during replay');
    return snapshot;
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('Replacement lost during replay');
  expect(props.publish).not.toHaveBeenCalled();
  expect(props.discard).toHaveBeenCalledExactlyOnceWith(replacement);
  expect(recovery.status.hasCheckpoint).toBe(true);
  await recovery.dispose();
});

it('unfinished-stream schema deferral preserves the last complete checkpoint for explicit recovery', async () => {
  const {recovery, props, previous, replacement, snapshot} = harness();
  expect(recovery.checkpoint(previous)).toBe(true);
  props.capture = vi.fn(runtime => {
    if (runtime === previous) throw new IncompleteSandboxPopulationError();
    return snapshot;
  });
  expect(recovery.checkpoint(previous)).toBe(false);
  expect(recovery.status).toMatchObject({hasCheckpoint: true, capturedAt: 100, sourceCount: 1});
  expect(recovery.status.reason).toContain('last complete recovery checkpoint is retained');
  recovery.markLost(previous, 'Device lost');
  expect(await recovery.recover()).toBe(replacement);
  expect(props.rebuild).toHaveBeenCalledWith(snapshot, expect.anything());
  await recovery.dispose();
});

it('failed teardown prevents creating another device and remains visible', async () => {
  const {recovery, props, previous} = harness();
  props.teardown = vi.fn(async () => {
    throw new Error('Old resource cleanup failed');
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('Old resource cleanup failed');
  expect(props.rebuild).not.toHaveBeenCalled();
  expect(recovery.status).toMatchObject({phase: 'failed', reason: 'Old resource cleanup failed'});
  await expect(recovery.dispose()).rejects.toThrow('Old resource cleanup failed');
});

it('a failed or cancelled synchronous publication cannot resurrect a disposed coordinator', async () => {
  const {recovery, props, previous, replacement} = harness();
  let disposal: Promise<void> | undefined;
  props.publish = vi.fn(() => {
    disposal = recovery.dispose();
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('cancelled');
  await disposal;
  expect(props.discard).toHaveBeenCalledExactlyOnceWith(replacement);
  expect(recovery.status.phase).toBe('disposed');
});

it('candidate cleanup failure remains visible and prevents overlapping replacement devices', async () => {
  const {recovery, props, previous} = harness();
  props.restore = vi.fn(async () => {
    throw new Error('Replay failed');
  });
  props.discard = vi.fn(async () => {
    throw new Error('Device disposal failed');
  });
  recovery.checkpoint(previous);
  recovery.markLost(previous, 'Device lost');
  await expect(recovery.recover()).rejects.toThrow('candidate cleanup failed');
  expect(recovery.status.reason).toBe('Recovery candidate cleanup failed');
  await expect(recovery.recover()).rejects.toThrow('candidate cleanup failed');
  expect(props.rebuild).toHaveBeenCalledTimes(1);
  await expect(recovery.dispose()).rejects.toThrow('candidate cleanup failed');
});

function harness() {
  const previous: Runtime = {id: 'previous'};
  const replacement: Runtime = {id: 'replacement'};
  const table = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))});
  const snapshot: ResidentRecoverySnapshot = {
    version: 1,
    capturedAt: 100,
    bundle: {
      state: {schemaVersion: 1, sources: [], selections: [], analyses: [], nativeConfig: null},
      sources: [{datasetId: 'private-source', revision: 1, table}]
    }
  };
  const events: string[] = [];
  const props: ResidentDeviceRecoveryProps<Runtime> = {
    capture: vi.fn(runtime => {
      events.push(`capture:${runtime.id}`);
      return snapshot;
    }),
    invalidate: vi.fn(runtime => {
      events.push(`invalidate:${runtime.id}`);
    }),
    teardown: vi.fn(async (runtime, completion) => {
      events.push(`teardown:start:${runtime.id}`);
      await completion;
      events.push(`teardown:end:${runtime.id}`);
    }),
    rebuild: vi.fn(async () => {
      events.push('rebuild');
      return replacement;
    }),
    restore: vi.fn(async () => {
      events.push('restore');
    }),
    publish: vi.fn(() => {
      events.push('publish');
    }),
    discard: vi.fn(async () => {
      events.push('discard');
    })
  };
  return {
    previous,
    replacement,
    table,
    snapshot,
    events,
    props,
    recovery: new ResidentDeviceRecovery(props)
  };
}

function deferred<Value = void>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}
