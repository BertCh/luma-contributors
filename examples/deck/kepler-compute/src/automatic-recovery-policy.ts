// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Opt-in automatic policies layered on explicit recovery. Every object here is pure policy: clocks,
 * timers and side effects are injected, so nothing starts without a host calling it. Defaults are
 * disabled because recovery creates a device and eviction drops resident data without a user action.
 */

export type BackoffOptions = Readonly<{
  /** Total attempts including the first; 1 disables retry. */
  maximumAttempts: number;
  initialDelayMilliseconds: number;
  maximumDelayMilliseconds: number;
  multiplier: number;
}>;

export const DEFAULT_BACKOFF_OPTIONS: BackoffOptions = {
  maximumAttempts: 3,
  initialDelayMilliseconds: 500,
  maximumDelayMilliseconds: 8000,
  multiplier: 2
};

/** Delay before the retry that follows failed attempt number `attempt` (1-based). */
export function getBackoffDelay(options: BackoffOptions, attempt: number): number {
  return Math.min(
    options.maximumDelayMilliseconds,
    options.initialDelayMilliseconds * options.multiplier ** (attempt - 1)
  );
}

export type Timers = Readonly<{
  setTimeout(callback: () => void, milliseconds: number): unknown;
  clearTimeout(handle: unknown): void;
}>;

export const SYSTEM_TIMERS: Timers = {
  setTimeout: (callback, milliseconds) => globalThis.setTimeout(callback, milliseconds),
  clearTimeout: handle => globalThis.clearTimeout(handle as number)
};

/** Retries a failing operation with bounded exponential backoff; the last error is rethrown. */
export async function retryWithBackoff<Value>(
  operation: (attempt: number) => Promise<Value>,
  props: {
    options: BackoffOptions;
    signal?: AbortSignal;
    shouldRetry?(error: unknown): boolean;
    onRetry?(error: unknown, attempt: number, delayMilliseconds: number): void;
    timers?: Timers;
  }
): Promise<Value> {
  const timers = props.timers ?? SYSTEM_TIMERS;
  for (let attempt = 1; ; attempt++) {
    props.signal?.throwIfAborted();
    try {
      return await operation(attempt);
    } catch (error) {
      if (
        attempt >= props.options.maximumAttempts ||
        props.signal?.aborted ||
        (props.shouldRetry && !props.shouldRetry(error))
      )
        throw error;
      const delayMilliseconds = getBackoffDelay(props.options, attempt);
      props.onRetry?.(error, attempt, delayMilliseconds);
      await waitForDelay(delayMilliseconds, timers, props.signal);
    }
  }
}

function waitForDelay(
  milliseconds: number,
  timers: Timers,
  signal: AbortSignal | undefined
): Promise<void> {
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      timers.clearTimeout(handle);
      reject(signal?.reason);
    };
    const handle = timers.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, milliseconds);
    signal?.addEventListener('abort', onAbort, {once: true});
  });
}

/** Network failures and HTTP 408/429/5xx are transient; other 4xx responses are not retried. */
export function isTransientFetchError(error: unknown): boolean {
  const status = (error as {status?: unknown} | undefined)?.status;
  if (typeof status !== 'number') return !(error instanceof DOMException);
  return status === 408 || status === 429 || status >= 500;
}

export type AutomaticRecoveryStatus = Readonly<{
  enabled: boolean;
  state: 'idle' | 'waiting' | 'recovering' | 'exhausted';
  /** Automatic attempts started for the current loss. */
  attempts: number;
  maximumAttempts: number;
  /** Absolute clock time of the next scheduled attempt, when waiting. */
  nextAttemptAt?: number;
}>;

/** Minimal coordinator surface, satisfied by ResidentDeviceRecovery. */
export type RecoverableCoordinator = {
  readonly status: {phase: string; hasCheckpoint: boolean};
  subscribe(listener: (status: {phase: string; hasCheckpoint: boolean}) => void): () => void;
  recover(): Promise<unknown>;
};

/**
 * Drives coordinator.recover() after loss or failure with bounded backoff. The attempt counter
 * resets when the coordinator returns to ready. Exhaustion leaves the explicit button in charge.
 */
export class AutomaticRecoveryPolicy {
  private handle: unknown;
  private attempts = 0;
  private state: AutomaticRecoveryStatus['state'] = 'idle';
  private nextAttemptAt: number | undefined;
  private unsubscribe?: () => void;
  private readonly listeners = new Set<(status: AutomaticRecoveryStatus) => void>();

  constructor(
    private readonly coordinator: RecoverableCoordinator,
    private readonly props: {
      enabled: boolean;
      options?: BackoffOptions;
      timers?: Timers;
      now?: () => number;
    }
  ) {
    if (props.enabled) this.unsubscribe = coordinator.subscribe(() => this.evaluate());
  }

  private get options(): BackoffOptions {
    return this.props.options ?? DEFAULT_BACKOFF_OPTIONS;
  }

  get status(): AutomaticRecoveryStatus {
    return {
      enabled: this.props.enabled,
      state: this.state,
      attempts: this.attempts,
      maximumAttempts: this.options.maximumAttempts,
      ...(this.nextAttemptAt === undefined ? {} : {nextAttemptAt: this.nextAttemptAt})
    };
  }

  subscribe(listener: (status: AutomaticRecoveryStatus) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  dispose(): void {
    this.unsubscribe?.();
    this.clearPending();
    this.state = 'idle';
    this.listeners.clear();
  }

  private evaluate(): void {
    const {phase, hasCheckpoint} = this.coordinator.status;
    if (phase === 'ready') {
      this.clearPending();
      this.attempts = 0;
      this.setState('idle');
    } else if (phase === 'disposed') {
      this.clearPending();
      this.setState('idle');
    } else if ((phase === 'lost' || phase === 'failed') && hasCheckpoint) {
      if (this.handle !== undefined || this.state === 'recovering') return;
      if (this.attempts >= this.options.maximumAttempts) {
        this.setState('exhausted');
        return;
      }
      const delayMilliseconds =
        this.attempts === 0 ? 0 : getBackoffDelay(this.options, this.attempts);
      this.nextAttemptAt = (this.props.now ?? Date.now)() + delayMilliseconds;
      this.setState('waiting');
      this.handle = (this.props.timers ?? SYSTEM_TIMERS).setTimeout(
        () => this.runAttempt(),
        delayMilliseconds
      );
    }
  }

  private runAttempt(): void {
    this.handle = undefined;
    this.nextAttemptAt = undefined;
    this.attempts++;
    this.setState('recovering');
    // A failed attempt moves the coordinator to 'failed', which reschedules through subscribe.
    void this.coordinator
      .recover()
      .then(
        () => undefined,
        () => undefined
      )
      .finally(() => {
        if (this.state === 'recovering') this.setState('idle');
        this.evaluate();
      });
  }

  private clearPending(): void {
    if (this.handle !== undefined) (this.props.timers ?? SYSTEM_TIMERS).clearTimeout(this.handle);
    this.handle = undefined;
    this.nextAttemptAt = undefined;
  }

  private setState(state: AutomaticRecoveryStatus['state']): void {
    this.state = state;
    for (const listener of this.listeners) listener(this.status);
  }
}

export type EvictableTable = Readonly<{
  id: string;
  /** Bytes released by evicting the table; zero for borrowed GPUData that the registry does not own. */
  ownedByteLength: number;
  /** Visible tables (bound to a layer, chart or selection) are never evicted. */
  isVisible: boolean;
}>;

/**
 * Least-recently-used eviction planning for resident tables. Evicted tables stay in a ledger so
 * the host can rebuild them from retained sources on demand. Borrowed GPUData frees nothing and is
 * never selected, so aggregate eviction cannot destroy memory it does not own.
 */
export class ResidentEvictionPolicy {
  private sequence = 0;
  private readonly lastUsed = new Map<string, number>();
  private readonly evicted = new Set<string>();

  constructor(private readonly props: {enabled: boolean}) {}

  get isEnabled(): boolean {
    return this.props.enabled;
  }

  touch(id: string): void {
    this.lastUsed.set(id, ++this.sequence);
  }

  forget(id: string): void {
    this.lastUsed.delete(id);
    this.evicted.delete(id);
  }

  /** Ids to evict, least recently used first, or null when eviction cannot free enough bytes. */
  planEviction(
    requiredBytes: number,
    availableBytes: number,
    tables: readonly EvictableTable[]
  ): string[] | null {
    if (!this.props.enabled) return null;
    let missingBytes = requiredBytes - availableBytes;
    if (missingBytes <= 0) return [];
    const victims: string[] = [];
    const candidates = tables
      .filter(table => !table.isVisible && table.ownedByteLength > 0)
      .sort((a, b) => (this.lastUsed.get(a.id) ?? 0) - (this.lastUsed.get(b.id) ?? 0));
    for (const table of candidates) {
      victims.push(table.id);
      missingBytes -= table.ownedByteLength;
      if (missingBytes <= 0) return victims;
    }
    return null;
  }

  markEvicted(id: string): void {
    this.evicted.add(id);
  }

  isEvicted(id: string): boolean {
    return this.evicted.has(id);
  }

  /** Rebuild an evicted table on first use; concurrent callers share one rebuild. */
  async ensureResident(id: string, rebuild: (id: string) => Promise<void>): Promise<void> {
    if (!this.evicted.has(id)) return;
    let pending = this.rebuilds.get(id);
    if (!pending) {
      pending = rebuild(id)
        .then(() => {
          this.evicted.delete(id);
          this.touch(id);
        })
        .finally(() => this.rebuilds.delete(id));
      this.rebuilds.set(id, pending);
    }
    await pending;
  }

  private readonly rebuilds = new Map<string, Promise<void>>();
}

/**
 * Reserve memory, evicting planned victims once when the budget would be exceeded. The registry
 * remains the authority: evict() must remove the dataset through the registry so owned resources
 * release after their fences and borrowed resources stay with their owners.
 */
export function reserveMemoryWithEviction<Reservation>(props: {
  policy: ResidentEvictionPolicy;
  byteLength: number;
  getAvailableBytes(): number;
  getTables(): readonly EvictableTable[];
  reserve(byteLength: number): Reservation;
  evict(id: string): void;
}): Reservation {
  const victims = props.policy.planEviction(
    props.byteLength,
    props.getAvailableBytes(),
    props.getTables()
  );
  for (const id of victims ?? []) {
    props.evict(id);
    props.policy.markEvicted(id);
  }
  // Throws the registry budget error when eviction is disabled or cannot free enough bytes.
  return props.reserve(props.byteLength);
}

export type AutomaticPolicyConfig = Readonly<{
  autoRecovery: boolean;
  remoteRetry: boolean;
  autoEvict: boolean;
}>;

/** All three policies are off unless named in `?automatic=recovery,retry,evict` (or `all`). */
export function parseAutomaticPolicyConfig(search: string): AutomaticPolicyConfig {
  const names = new Set((new URLSearchParams(search).get('automatic') ?? '').split(','));
  const all = names.has('all');
  return {
    autoRecovery: all || names.has('recovery'),
    remoteRetry: all || names.has('retry'),
    autoEvict: all || names.has('evict')
  };
}
