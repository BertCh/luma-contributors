// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RestoredSandboxBundle} from './sandbox-bundle';
import type {ResidentResultRecipe} from './resident-result-recipes';
import {IncompleteSandboxPopulationError} from './sandbox-population';

/** CPU references only: no GPU buffers, IPC conversion, persistent storage or network transfer. */
export type ResidentRecoverySnapshot = Readonly<{
  version: 1;
  capturedAt: number;
  /** Treat the detached metadata and borrowed immutable Arrow sources as read-only. */
  bundle: RestoredSandboxBundle;
  /**
   * Lineage of GPU-resident (over-cap) results. They have no CPU source in the bundle; the host
   * recomputes them on the recovered inputs, in dependency order, under their original dataset IDs.
   */
  derived?: readonly ResidentResultRecipe[];
}>;

export type ResidentRecoveryContext = Readonly<{signal: AbortSignal; attempt: number}>;
export type ResidentRecoveryStatus = Readonly<{
  phase: 'ready' | 'lost' | 'rebuilding' | 'failed' | 'disposed';
  attempt: number;
  hasCheckpoint: boolean;
  sourceCount: number;
  capturedAt?: number;
  reason: string | null;
}>;

export type ResidentDeviceRecoveryProps<Runtime> = {
  capture(runtime: Runtime): ResidentRecoverySnapshot;
  /** Synchronous: detach compute/render consumers, pending inspection publication and animation. */
  invalidate(runtime: Runtime, reason: string): void;
  /** The supplied final-submission/device-loss completion must precede GPU resource release. */
  teardown(runtime: Runtime, completion: PromiseLike<unknown>): Promise<void>;
  /** One explicit attempt. Honor AbortSignal, including disposing any late device-init callback. */
  rebuild(snapshot: ResidentRecoverySnapshot, context: ResidentRecoveryContext): Promise<Runtime>;
  /** Reimported sources precede native state/recipe replay; do not publish the runtime here. */
  restore(
    runtime: Runtime,
    snapshot: ResidentRecoverySnapshot,
    context: ResidentRecoveryContext
  ): Promise<void>;
  /** Synchronous activation after successful replay and a fresh healthy checkpoint. */
  publish(runtime: Runtime): void;
  /** Release a failed/cancelled candidate and its device/Deck instance without publishing it. */
  discard(runtime: Runtime): Promise<void>;
};

/**
 * Explicit host recovery state machine. Device loss never creates a device or retries a failure.
 *
 * Checkpoint supported CPU sources/native state/recipes after semantic changes while healthy.
 * markLost invalidates consumers immediately and starts fenced teardown. An explicit recover call
 * performs one rebuild/replay; duplicate clicks share that attempt. Cancellation and stale-owner
 * callbacks cannot publish a retired runtime. Failed checkpoints are cleared; an unfinished-stream
 * schema deferral explicitly retains the last complete checkpoint. CPU Arrow references remain in memory until replacement or
 * disposal; this coordinator never persists or uploads source data.
 */
export class ResidentDeviceRecovery<Runtime> {
  private active?: Runtime;
  private checkpointValue?: ResidentRecoverySnapshot;
  private phase: ResidentRecoveryStatus['phase'] = 'ready';
  private reason: string | null = null;
  private attemptNumber = 0;
  private generation = 0;
  private retirement?: Promise<void>;
  private retiringRuntime?: Runtime;
  private attempt?: Promise<Runtime>;
  private controller?: AbortController;
  private disposal?: Promise<void>;
  private cleanupFailure?: AggregateError;
  private readonly listeners = new Set<(status: ResidentRecoveryStatus) => void>();

  constructor(private readonly props: ResidentDeviceRecoveryProps<Runtime>) {}

  get status(): ResidentRecoveryStatus {
    return {
      phase: this.phase,
      attempt: this.attemptNumber,
      reason: this.reason,
      hasCheckpoint: Boolean(this.checkpointValue),
      sourceCount: this.checkpointValue?.bundle.sources.length ?? 0,
      ...(this.checkpointValue ? {capturedAt: this.checkpointValue.capturedAt} : {})
    };
  }

  /** Latest complete CPU checkpoint, also used to rebuild evicted tables on demand. */
  get checkpointSnapshot(): ResidentRecoverySnapshot | undefined {
    return this.checkpointValue;
  }

  subscribe(listener: (status: ResidentRecoveryStatus) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Busy/lost coordinators do not recapture stale handles. Failure is exposed through status. */
  checkpoint(runtime: Runtime): boolean {
    if (this.phase !== 'ready') return false;
    if (this.active !== undefined && this.active !== runtime)
      throw new Error('Recovery checkpoint belongs to another runtime');
    this.active = runtime;
    try {
      this.checkpointValue = this.props.capture(runtime);
      this.reason = null;
      this.emit();
      return true;
    } catch (error) {
      if (!(error instanceof IncompleteSandboxPopulationError)) this.checkpointValue = undefined;
      this.reason =
        error instanceof IncompleteSandboxPopulationError && this.checkpointValue
          ? `${message(error)}; the last complete recovery checkpoint is retained`
          : message(error);
      this.emit();
      return false;
    }
  }

  /** Returns false for duplicate loss events and callbacks from a previously retired runtime. */
  markLost(
    runtime: Runtime,
    reason: string,
    completion: PromiseLike<unknown> = Promise.resolve()
  ): boolean {
    if (this.phase !== 'ready' || this.active !== runtime) return false;
    this.phase = 'lost';
    this.reason = reason;
    this.generation++;
    try {
      this.props.invalidate(runtime, reason);
    } catch (error) {
      this.reason = message(error);
    }
    this.startRetirement(runtime, completion);
    this.emit();
    return true;
  }

  /** The host invokes this from an explicit recovery action; no internal automatic retry exists. */
  recover(): Promise<Runtime> {
    if (this.attempt) return this.attempt;
    if (this.phase === 'disposed')
      return Promise.reject(new Error('Recovery coordinator is disposed'));
    if (this.phase !== 'lost' && this.phase !== 'failed')
      return Promise.reject(new Error('Recovery requires a lost runtime'));
    // A failed candidate release has unknown ownership; do not stack another device on top of it.
    if (this.cleanupFailure) return Promise.reject(this.cleanupFailure);
    const snapshot = this.checkpointValue;
    if (!snapshot) {
      this.phase = 'failed';
      this.reason = this.reason ?? 'No healthy recovery checkpoint is available';
      this.emit();
      return Promise.reject(new Error('No healthy recovery checkpoint is available'));
    }
    this.phase = 'rebuilding';
    this.reason = null;
    const generation = ++this.generation;
    const controller = new AbortController();
    this.controller = controller;
    const context = {signal: controller.signal, attempt: ++this.attemptNumber};
    this.emit();
    const attempt = this.runRecovery(snapshot, context, generation).finally(() => {
      if (this.attempt === attempt) {
        this.attempt = undefined;
        this.controller = undefined;
      }
    });
    this.attempt = attempt;
    return attempt;
  }

  /** Aborts pending creation/replay; candidates returned after cancellation are discarded. */
  cancelRecovery(): Promise<void> {
    if (!this.attempt || this.phase !== 'rebuilding') return Promise.resolve();
    this.controller?.abort(new Error('Recovery cancelled'));
    // Keep the attempt registered until cleanup settles; another click cannot create a second device.
    return this.attempt.then(
      () => undefined,
      () => {
        if (this.cleanupFailure) throw this.cleanupFailure;
      }
    );
  }

  /** Permanently closes the coordinator and releases its in-memory checkpoint. */
  dispose(completion: PromiseLike<unknown> = Promise.resolve()): Promise<void> {
    if (this.disposal) return this.disposal;
    this.phase = 'disposed';
    this.generation++;
    this.controller?.abort(new Error('Recovery coordinator disposed'));
    this.checkpointValue = undefined;
    if (this.active !== undefined && this.active !== this.retiringRuntime) {
      try {
        this.props.invalidate(this.active, 'Workspace closed');
      } catch {}
      this.startRetirement(this.active, completion);
    }
    this.emit();
    this.listeners.clear();
    this.disposal = Promise.allSettled([this.retirement, this.attempt]).then(results => {
      this.active = undefined;
      this.retiringRuntime = undefined;
      // Cancellation of an in-progress rebuild is expected; teardown errors still matter.
      if (results[0].status === 'rejected') throw results[0].reason;
      if (this.cleanupFailure) throw this.cleanupFailure;
    });
    return this.disposal;
  }

  private async runRecovery(
    snapshot: ResidentRecoverySnapshot,
    context: ResidentRecoveryContext,
    generation: number
  ): Promise<Runtime> {
    let candidate: Runtime | undefined;
    let activating = false;
    try {
      await this.retirement;
      this.assertAttempt(context, generation);
      candidate = await this.props.rebuild(snapshot, context);
      this.assertAttempt(context, generation);
      await this.props.restore(candidate, snapshot, context);
      this.assertAttempt(context, generation);
      const checkpoint = this.props.capture(candidate);
      this.assertAttempt(context, generation);
      activating = true;
      this.props.publish(candidate);
      this.assertAttempt(context, generation);
      this.active = candidate;
      this.checkpointValue = checkpoint;
      this.retirement = undefined;
      this.retiringRuntime = undefined;
      this.phase = 'ready';
      this.reason = null;
      this.emit();
      return candidate;
    } catch (error) {
      this.controller?.abort(error);
      let failure = error;
      if (candidate !== undefined) {
        if (activating) {
          try {
            this.props.invalidate(candidate, message(error));
          } catch {}
        }
        try {
          await this.props.discard(candidate);
        } catch (cleanupError) {
          this.cleanupFailure = new AggregateError(
            [error, cleanupError],
            'Recovery candidate cleanup failed'
          );
          failure = this.cleanupFailure;
        }
      }
      if (this.phase !== 'disposed' && this.generation === generation) {
        this.phase = 'failed';
        this.reason = message(failure);
        this.emit();
      }
      throw failure;
    }
  }

  private startRetirement(runtime: Runtime, completion: PromiseLike<unknown>): void {
    this.retiringRuntime = runtime;
    const generation = this.generation;
    const retirement = Promise.resolve().then(() => this.props.teardown(runtime, completion));
    this.retirement = retirement;
    void retirement.catch(error => {
      if (this.phase === 'lost' && this.generation === generation) {
        this.phase = 'failed';
        this.reason = message(error);
        this.emit();
      }
    });
  }

  private assertAttempt(context: ResidentRecoveryContext, generation: number): void {
    if (context.signal.aborted || this.phase === 'disposed' || this.generation !== generation)
      throw new Error('Recovery attempt was cancelled or superseded');
  }

  private emit(): void {
    const status = this.status;
    for (const listener of this.listeners) {
      try {
        listener(status);
      } catch (error) {
        // Observers must not interrupt resource ownership transitions or hide cleanup failures.
        // biome-ignore lint/suspicious/noConsole: preserve observer diagnostics without throwing across ownership transitions
        console.error('Recovery status listener failed', error);
      }
    }
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
