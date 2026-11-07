// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {
  AutomaticRecoveryPolicy,
  ResidentEvictionPolicy,
  getBackoffDelay,
  isTransientFetchError,
  parseAutomaticPolicyConfig,
  reserveMemoryWithEviction,
  retryWithBackoff,
  type BackoffOptions,
  type Timers
} from '../../../examples/deck/kepler-compute/src/automatic-recovery-policy';
import {ArrowStreamController} from '../../../examples/deck/kepler-compute/src/arrow-stream-controller';

const OPTIONS: BackoffOptions = {
  maximumAttempts: 3,
  initialDelayMilliseconds: 100,
  maximumDelayMilliseconds: 150,
  multiplier: 2
};

function createFakeTimers() {
  const pending = new Map<number, {callback: () => void; milliseconds: number}>();
  let next = 0;
  const delays: number[] = [];
  const timers: Timers = {
    setTimeout(callback, milliseconds) {
      pending.set(++next, {callback, milliseconds});
      delays.push(milliseconds);
      return next;
    },
    clearTimeout(handle) {
      pending.delete(handle as number);
    }
  };
  async function fire(): Promise<void> {
    for (let tick = 0; tick < 10; tick++) await Promise.resolve();
    const [id, entry] = [...pending.entries()][0];
    pending.delete(id);
    entry.callback();
    for (let tick = 0; tick < 10; tick++) await Promise.resolve();
  }
  return {timers, pending, delays, fire};
}

describe('backoff and retry', () => {
  it('caps exponential delays', () => {
    expect([1, 2, 3].map(attempt => getBackoffDelay(OPTIONS, attempt))).toEqual([100, 150, 150]);
  });

  it('retries with bounded attempts then rethrows the last error', async () => {
    const fake = createFakeTimers();
    const operation = vi.fn(async () => {
      throw new Error('down');
    });
    const result = retryWithBackoff(operation, {options: OPTIONS, timers: fake.timers});
    const settled = result.catch(error => error);
    await fake.fire();
    await fake.fire();
    expect((await settled).message).toBe('down');
    expect(operation).toHaveBeenCalledTimes(3);
    expect(fake.delays).toEqual([100, 150]);
  });

  it('succeeds after a transient failure and skips non-transient errors', async () => {
    const fake = createFakeTimers();
    let calls = 0;
    const value = retryWithBackoff(
      async () => {
        if (++calls < 2) throw Object.assign(new Error('busy'), {status: 503});
        return 'ok';
      },
      {options: OPTIONS, timers: fake.timers, shouldRetry: isTransientFetchError}
    );
    await fake.fire();
    expect(await value).toBe('ok');
    const notFound = vi.fn(async () => {
      throw Object.assign(new Error('missing'), {status: 404});
    });
    await expect(
      retryWithBackoff(notFound, {options: OPTIONS, shouldRetry: isTransientFetchError})
    ).rejects.toThrow('missing');
    expect(notFound).toHaveBeenCalledTimes(1);
  });

  it('aborts while waiting for a retry', async () => {
    const fake = createFakeTimers();
    const controller = new AbortController();
    const result = retryWithBackoff(
      async () => {
        throw new Error('down');
      },
      {options: OPTIONS, timers: fake.timers, signal: controller.signal}
    );
    const settled = result.catch(error => error);
    await Promise.resolve();
    await Promise.resolve();
    controller.abort(new Error('cancelled'));
    expect((await settled).message).toBe('cancelled');
    expect(fake.pending.size).toBe(0);
  });

  it('remote stream controller retries the request only when configured', async () => {
    const fake = createFakeTimers();
    let calls = 0;
    const fetchImplementation = (async () => {
      calls++;
      return new Response(null, {status: 503});
    }) as unknown as typeof fetch;
    const controller = new ArrowStreamController({
      session: {
        device: {lost: new Promise(() => undefined)},
        registry: {getDataset: () => undefined}
      } as never,
      id: 'retry',
      normalization: {} as never,
      maximumInputBytes: 1024,
      maximumInputChunkBytes: 1024,
      maximumQueuedBytes: 1024,
      maximumBatchRows: 10,
      getCompletionFence: () => Promise.resolve(),
      remoteUrl: 'https://example.test/data.arrow',
      fetchImplementation,
      retry: {options: OPTIONS, timers: fake.timers}
    });
    const started = controller.startRemote().catch(error => error);
    for (let tick = 0; tick < 10; tick++) await Promise.resolve();
    await fake.fire();
    await fake.fire();
    expect(String(await started)).toContain('HTTP 503');
    expect(calls).toBe(3);
  });
});

describe('automatic device recovery', () => {
  function createCoordinator(outcomes: boolean[]) {
    const state = {phase: 'lost', hasCheckpoint: true};
    const listeners = new Set<(status: typeof state) => void>();
    const emit = () => listeners.forEach(listener => listener({...state}));
    const coordinator = {
      get status() {
        return {...state};
      },
      subscribe(listener: (status: typeof state) => void) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      recover: vi.fn(async () => {
        state.phase = 'rebuilding';
        emit();
        await Promise.resolve();
        if (outcomes.shift()) {
          state.phase = 'ready';
          emit();
          return {};
        }
        state.phase = 'failed';
        emit();
        throw new Error('rebuild failed');
      })
    };
    return {coordinator, state, emit};
  }

  it('is inert when disabled', () => {
    const fake = createFakeTimers();
    const {coordinator, emit} = createCoordinator([true]);
    new AutomaticRecoveryPolicy(coordinator, {enabled: false, timers: fake.timers});
    emit();
    expect(fake.pending.size).toBe(0);
    expect(coordinator.recover).not.toHaveBeenCalled();
  });

  it('retries with backoff, then reports success and resets', async () => {
    const fake = createFakeTimers();
    const {coordinator, emit} = createCoordinator([false, true]);
    const policy = new AutomaticRecoveryPolicy(coordinator, {
      enabled: true,
      options: OPTIONS,
      timers: fake.timers,
      now: () => 1000
    });
    emit();
    expect(policy.status).toMatchObject({state: 'waiting', attempts: 0, nextAttemptAt: 1000});
    await fake.fire();
    expect(coordinator.recover).toHaveBeenCalledTimes(1);
    expect(policy.status).toMatchObject({state: 'waiting', attempts: 1, nextAttemptAt: 1100});
    await fake.fire();
    expect(coordinator.recover).toHaveBeenCalledTimes(2);
    expect(policy.status).toMatchObject({state: 'idle', attempts: 0});
  });

  it('stops after the maximum attempts and leaves explicit recovery available', async () => {
    const fake = createFakeTimers();
    const {coordinator, emit} = createCoordinator([false, false, false, true]);
    const policy = new AutomaticRecoveryPolicy(coordinator, {
      enabled: true,
      options: OPTIONS,
      timers: fake.timers
    });
    emit();
    for (let attempt = 0; attempt < 3; attempt++) await fake.fire();
    expect(coordinator.recover).toHaveBeenCalledTimes(3);
    expect(policy.status).toMatchObject({state: 'exhausted', attempts: 3});
    expect(fake.pending.size).toBe(0);
    policy.dispose();
  });

  it('does not schedule without a checkpoint', () => {
    const fake = createFakeTimers();
    const {coordinator, state, emit} = createCoordinator([true]);
    state.hasCheckpoint = false;
    new AutomaticRecoveryPolicy(coordinator, {enabled: true, timers: fake.timers});
    emit();
    expect(fake.pending.size).toBe(0);
  });
});

describe('resident eviction', () => {
  const tables = [
    {id: 'a', ownedByteLength: 100, isVisible: false},
    {id: 'b', ownedByteLength: 100, isVisible: true},
    {id: 'c', ownedByteLength: 100, isVisible: false},
    {id: 'borrowed', ownedByteLength: 0, isVisible: false}
  ];

  it('evicts least recently used hidden owned tables only', () => {
    const policy = new ResidentEvictionPolicy({enabled: true});
    ['c', 'a', 'b', 'borrowed'].forEach(id => policy.touch(id));
    expect(policy.planEviction(150, 0, tables)).toEqual(['c', 'a']);
    expect(policy.planEviction(50, 0, tables)).toEqual(['c']);
    expect(policy.planEviction(50, 80, tables)).toEqual([]);
    expect(policy.planEviction(250, 0, tables)).toBeNull();
  });

  it('is disabled by default flags', () => {
    expect(new ResidentEvictionPolicy({enabled: false}).planEviction(10, 0, tables)).toBeNull();
    expect(parseAutomaticPolicyConfig('')).toEqual({
      autoRecovery: false,
      remoteRetry: false,
      autoEvict: false
    });
    expect(parseAutomaticPolicyConfig('?automatic=recovery,evict')).toEqual({
      autoRecovery: true,
      remoteRetry: false,
      autoEvict: true
    });
    expect(parseAutomaticPolicyConfig('?automatic=all').remoteRetry).toBe(true);
  });

  it('reserves after eviction and rebuilds evicted tables once on demand', async () => {
    const policy = new ResidentEvictionPolicy({enabled: true});
    ['a', 'c'].forEach(id => policy.touch(id));
    const evict = vi.fn();
    const reserve = vi.fn((byteLength: number) => ({byteLength}));
    const reservation = reserveMemoryWithEviction({
      policy,
      byteLength: 100,
      getAvailableBytes: () => 0,
      getTables: () => tables,
      reserve,
      evict
    });
    expect(reservation).toEqual({byteLength: 100});
    expect(evict).toHaveBeenCalledExactlyOnceWith('a');
    expect(policy.isEvicted('a')).toBe(true);
    const rebuild = vi.fn(async () => undefined);
    await Promise.all([policy.ensureResident('a', rebuild), policy.ensureResident('a', rebuild)]);
    expect(rebuild).toHaveBeenCalledTimes(1);
    expect(policy.isEvicted('a')).toBe(false);
  });
});
