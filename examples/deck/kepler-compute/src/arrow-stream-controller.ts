// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  KeplerArrowStreamPublisher,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot
} from '@luma-contributors/kepler';
import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/analysis/analysis-core';
import {formatOriginalTableCell} from './original-table-values';
import {STREAM_ORIGIN_METADATA_KEY, openRemoteStream} from './remote-stream-source';
import {openResumableRemoteStream, type ResumeEvent} from './resumable-remote-stream';
import {ResumeBatchDedupe} from './resume-batch-dedupe';
import {
  isTransientFetchError,
  retryWithBackoff,
  type BackoffOptions,
  type Timers
} from './automatic-recovery-policy';

export type ArrowStreamPhase =
  | 'idle'
  | 'opening'
  | 'paused'
  | 'reading'
  | 'exhausted'
  | 'finishing'
  | 'complete'
  | 'publishing'
  | 'published'
  | 'cancelled'
  | 'failed';
export type ArrowStreamProgress = {
  phase: ArrowStreamPhase;
  bytesRead: number;
  batches: number;
  rows: number;
  revision?: number;
  queuedBytes: number;
  /** Set only after the user confirmed closing an unfinished stream at its admitted prefix. */
  partial?: {rowCount: number; batchCount: number};
  /** Remote source URL, when the input is a fetch() body. */
  remoteUrl?: string;
  /** Latest mid-body resume event (retrying, resumed, restarted, failed) when resume is enabled. */
  resume?: ResumeEvent;
  error?: string;
};
export type ArrowStreamPreview = {
  fields: readonly string[];
  rows: readonly (readonly string[])[];
  totalRows: number;
  incomplete: boolean;
};

/** Sequential IPC stream ingestion. Never materializes a whole input file or packs source batches. */
export class ArrowStreamController {
  readonly publisher: KeplerArrowStreamPublisher;
  private reader?: arrow.RecordBatchReader;
  private inputReader?: ReadableStreamDefaultReader<Uint8Array>;
  private iterator?: AsyncIterableIterator<arrow.RecordBatch>;
  private readonly listeners = new Set<() => void>();
  private readonly abort = new AbortController();
  private readonly fetchAbort = new AbortController();
  private partial?: {rowCount: number; batchCount: number};
  private phase: ArrowStreamPhase = 'idle';
  private bytesRead = 0;
  private batches = 0;
  private readonly batchDedupe = new ResumeBatchDedupe();
  private deliveredOrdinal = 0;
  private rows = 0;
  private error?: string;
  private resumeEvent?: ResumeEvent;
  private running = false;
  private operation?: Promise<void>;
  private disposal?: Promise<void>;
  private publishing?: Promise<void>;
  private transferred = false;

  constructor(
    private readonly options: {
      session: KeplerComputeSession;
      id: string;
      normalization: PrepareArrowTableForGPUAnalyticsOptions;
      maximumInputBytes: number;
      maximumInputChunkBytes: number;
      maximumQueuedBytes: number;
      maximumBatchRows: number;
      getCompletionFence(): PromiseLike<unknown>;
      /** Recorded in the dataset source metadata and therefore in the saved source manifest. */
      remoteUrl?: string;
      fetchImplementation?: typeof fetch;
      /** Opt-in bounded retry of the request only; omitted means a single attempt. */
      retry?: {options: BackoffOptions; timers?: Timers};
      /** Opt-in (requires retry): resume the body after a mid-stream failure without duplicating bytes. */
      resume?: {maximumResumes?: number};
    }
  ) {
    if (
      ![options.maximumInputBytes, options.maximumInputChunkBytes].every(
        value => Number.isSafeInteger(value) && value > 0
      )
    )
      throw new Error('Input capacities must be positive safe integers');
    this.publisher = new KeplerArrowStreamPublisher({
      session: options.session,
      id: options.id,
      normalization: options.normalization,
      maximumQueuedBytes: options.maximumQueuedBytes,
      maximumQueuedBatches: 1,
      maximumBatchRows: options.maximumBatchRows,
      dictionaryEvolution: 'append-unordered',
      metadata: options.remoteUrl
        ? {[STREAM_ORIGIN_METADATA_KEY]: {kind: 'remote', url: options.remoteUrl}}
        : undefined,
      signal: this.abort.signal
    });
  }

  get progress(): ArrowStreamProgress {
    return {
      phase: this.phase,
      bytesRead: this.bytesRead,
      batches: this.batches,
      rows: this.rows,
      revision: this.publisher.snapshot?.revision,
      queuedBytes: this.publisher.queuedBytes,
      partial: this.partial,
      remoteUrl: this.options.remoteUrl,
      resume: this.resumeEvent,
      error: this.error
    };
  }
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Open and admit exactly the first batch, then pause for an explicitly incomplete preview. */
  start(input: ReadableStream<Uint8Array>): Promise<void> {
    if (this.phase !== 'idle') return Promise.reject(new Error('Stream already started'));
    this.phase = 'opening';
    this.inputReader = input.getReader();
    this.emit();
    return this.track(() => this.openReader());
  }
  private async openReader(): Promise<void> {
    this.reader = await arrow.RecordBatchReader.from(this.limitInput());
    if (this.abort.signal.aborted) {
      await this.reader.cancel();
      return;
    }
    await this.reader.open();
    if (this.reader.isFile())
      throw new Error(
        'Choose Arrow IPC stream format; random-access Arrow files use the existing file importer'
      );
    this.iterator = this.reader[Symbol.asyncIterator]();
    await this.readNext();
  }

  /** Fetch the configured URL. Abort/cancel/dispose cancels the network body. */
  startRemote(): Promise<void> {
    if (this.phase !== 'idle' || !this.options.remoteUrl)
      return Promise.reject(new Error('Remote stream is not configured or already started'));
    this.phase = 'opening';
    this.emit();
    return this.track(async () => {
      const open = () =>
        openRemoteStream(
          new URL(this.options.remoteUrl!),
          this.fetchAbort.signal,
          this.options.maximumInputBytes,
          this.options.fetchImplementation
        );
      const retry = this.options.retry;
      const body =
        retry && this.options.resume
          ? await openResumableRemoteStream({
              url: new URL(this.options.remoteUrl!),
              signal: this.fetchAbort.signal,
              maximumInputBytes: this.options.maximumInputBytes,
              retry,
              maximumResumes: this.options.resume.maximumResumes,
              fetchImplementation: this.options.fetchImplementation,
              onEvent: event => {
                this.resumeEvent = event;
                this.emit();
              }
            })
          : retry
            ? await retryWithBackoff(open, {
                options: retry.options,
                signal: this.fetchAbort.signal,
                shouldRetry: isTransientFetchError,
                ...(retry.timers ? {timers: retry.timers} : {})
              })
            : await open();
      if (this.abort.signal.aborted) {
        void body.cancel().catch(() => undefined);
        return;
      }
      this.inputReader = body.getReader();
      await this.openReader();
    });
  }

  /** One admitted batch per pull, including empty batches. Concurrent pulls are rejected. */
  next(): Promise<void> {
    if (this.phase !== 'paused' || this.operation)
      return Promise.reject(new Error('Pause before reading another batch'));
    this.phase = 'reading';
    this.emit();
    return this.track(() => this.readNext());
  }
  resume(): Promise<void> {
    if (this.phase !== 'paused' || this.operation)
      return Promise.reject(new Error('Stream is not paused'));
    this.running = true;
    this.phase = 'reading';
    this.emit();
    return this.track(async () => {
      while (this.running && !this.abort.signal.aborted) {
        await this.readNext();
        if (this.phase === 'exhausted') break;
        // Yield to browser input between batches so Pause/Cancel are not starved by microtasks.
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
    });
  }
  pause(): void {
    this.running = false;
    if (this.phase === 'reading' && !this.operation) this.phase = 'paused';
    this.emit();
  }
  async finish(): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    if (this.phase !== 'exhausted' || this.operation)
      throw new Error('Finish requires the end of the IPC stream');
    this.phase = 'finishing';
    this.emit();
    let completed!: KeplerDatasetSnapshot<KeplerArrowDataset>;
    await this.track(async () => {
      completed = await this.publisher.finish();
      if (!this.abort.signal.aborted) this.phase = 'complete';
    });
    if (this.abort.signal.aborted) throw new Error('Stream cancelled');
    return completed;
  }

  /**
   * Explicitly close the admitted prefix as a partial population. Requires a paused stream with at
   * least one admitted batch and the caller's confirmation; input is cancelled and the result is
   * recorded as partial (row and batch counts) rather than as a finite complete dataset.
   */
  async finishPartial(
    confirmation: 'save-incomplete-prefix'
  ): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    // Confirmation is a literal so an accidental boolean or event argument cannot persist a prefix.
    if (confirmation !== 'save-incomplete-prefix' || this.phase !== 'paused' || this.operation)
      throw new Error('Saving an incomplete prefix requires a paused stream and confirmation');
    if (!this.batches) throw new Error('A partial population needs at least one admitted batch');
    this.phase = 'finishing';
    this.emit();
    let completed!: KeplerDatasetSnapshot<KeplerArrowDataset>;
    await this.track(async () => {
      this.fetchAbort.abort();
      void this.inputReader?.cancel().catch(() => undefined);
      void Promise.resolve(this.reader?.cancel()).catch(() => undefined);
      completed = await this.publisher.finishPartial();
      if (!this.abort.signal.aborted) {
        this.partial = {rowCount: completed.rowCount, batchCount: this.batches};
        this.phase = 'complete';
      }
    });
    if (this.abort.signal.aborted) throw new Error('Stream cancelled');
    return completed;
  }

  /** Publish the already-uploaded complete revision. Successful handoff makes the host its owner. */
  publish(
    attach: (dataset: KeplerDatasetSnapshot<KeplerArrowDataset>) => Promise<void>
  ): Promise<void> {
    if (this.publishing) return this.publishing;
    if (this.phase !== 'complete')
      return Promise.reject(new Error('Finish the stream before publishing'));
    this.phase = 'publishing';
    this.emit();
    const snapshot = this.publisher.snapshot!;
    this.publishing = (async () => {
      try {
        await attach(snapshot);
        this.publisher.detachCompleted();
        this.transferred = true;
        this.phase = 'published';
      } catch (error) {
        this.error = describeError(error);
        this.phase =
          this.options.session.registry.getDataset(snapshot.id)?.revision === snapshot.revision
            ? 'complete'
            : 'failed';
        throw error;
      } finally {
        this.publishing = undefined;
        this.emit();
      }
    })();
    return this.publishing;
  }

  /** Bounded original-source preview, explicitly not a filtered GPU result or a full export. */
  preview(): ArrowStreamPreview | null {
    const snapshot = this.publisher.snapshot;
    if (!snapshot) return null;
    const source = snapshot.value.source;
    const fields = source.schema.fields.slice(0, 6).map(field => field.name);
    const rows = Array.from({length: Math.min(8, source.numRows)}, (_, row) =>
      fields.map(name => formatPreviewCell(source.getChild(name)!, row))
    );
    return {
      fields,
      rows,
      totalRows: source.numRows,
      incomplete: snapshot.population?.state !== 'complete' || Boolean(snapshot.population.partial)
    };
  }

  cancel(): void {
    if (this.transferred || this.phase === 'publishing') return;
    this.running = false;
    this.abort.abort(new Error('Stream cancelled'));
    this.fetchAbort.abort();
    this.publisher.cancel();
    this.phase = 'cancelled';
    void this.inputReader?.cancel().catch(() => undefined);
    void Promise.resolve(this.reader?.cancel()).catch(() => undefined);
    this.emit();
  }
  dispose(): Promise<void> {
    if (!this.disposal)
      this.disposal = (async () => {
        if (this.publishing) await this.publishing.catch(() => undefined);
        this.cancel();
        await this.operation?.catch(() => undefined);
        try {
          await this.reader?.cancel();
        } finally {
          try {
            if (!this.transferred) await this.publisher.dispose(this.options.getCompletionFence());
          } finally {
            this.listeners.clear();
          }
        }
      })();
    return this.disposal;
  }

  /** Call before replaying the stream from its first batch; already published batches are skipped. */
  restartDelivery(): void {
    this.deliveredOrdinal = 0;
  }

  private async readNext(): Promise<void> {
    if (this.abort.signal.aborted) return;
    const next = await this.iterator!.next();
    if (this.abort.signal.aborted) return;
    if (next.done) {
      this.running = false;
      this.phase = 'exhausted';
    } else if (
      this.batchDedupe.consider({
        ordinal: this.deliveredOrdinal++,
        rowCount: next.value.numRows
      }) === 'skip'
    ) {
      // Re-delivered batch (restarted reader): already published, so it is not counted twice.
      this.phase = this.running ? 'reading' : 'paused';
    } else {
      const snapshot = await this.publisher.enqueue(next.value);
      if (this.abort.signal.aborted) return;
      this.batches++;
      this.rows = snapshot.rowCount;
      this.phase = this.running ? 'reading' : 'paused';
    }
    this.emit();
  }
  private track(work: () => Promise<void>): Promise<void> {
    const operation = work()
      .catch(error => {
        if (!this.abort.signal.aborted) {
          this.error = describeError(error);
          this.publisher.cancel(error);
          this.phase = 'failed';
          this.running = false;
          this.fetchAbort.abort();
          void this.inputReader?.cancel().catch(() => undefined);
          void Promise.resolve(this.reader?.cancel()).catch(() => undefined);
          this.emit();
          throw error;
        }
      })
      .finally(() => {
        this.operation = undefined;
        if (!this.running && this.phase === 'reading') this.phase = 'paused';
        this.emit();
      });
    this.operation = operation;
    return operation;
  }
  private async *limitInput(): AsyncGenerator<Uint8Array> {
    let prefix = new Uint8Array(0);
    const inputReader = this.inputReader!;
    try {
      while (true) {
        const result = await inputReader.read();
        if (result.done) return;
        const chunk = result.value;
        if (this.abort.signal.aborted) return;
        if (
          chunk.byteLength > this.options.maximumInputChunkBytes ||
          this.bytesRead + chunk.byteLength > this.options.maximumInputBytes
        )
          throw new Error('IPC input exceeds the configured byte capacity');
        this.bytesRead += chunk.byteLength;
        this.emit();
        if (prefix.length < 6) {
          const bytes = new Uint8Array(Math.min(6, prefix.length + chunk.length));
          bytes.set(prefix);
          bytes.set(chunk.subarray(0, bytes.length - prefix.length), prefix.length);
          prefix = bytes;
          if (prefix.length === 6 && String.fromCharCode(...prefix) === 'ARROW1')
            throw new Error(
              'Choose Arrow IPC stream format; random-access Arrow files use the existing file importer'
            );
        }
        yield chunk;
      }
    } finally {
      await inputReader.cancel().catch(() => undefined);
      inputReader.releaseLock();
      if (this.inputReader === inputReader) this.inputReader = undefined;
    }
  }
  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
function formatPreviewCell(vector: arrow.Vector, row: number): string {
  const type = arrow.DataType.isDictionary(vector.type) ? vector.type.dictionary : vector.type;
  if (
    !(
      arrow.DataType.isUtf8(type) ||
      arrow.DataType.isLargeUtf8(type) ||
      arrow.DataType.isFloat(type) ||
      arrow.DataType.isInt(type) ||
      arrow.DataType.isBool(type) ||
      arrow.DataType.isDate(type) ||
      arrow.DataType.isTimestamp(type)
    )
  )
    return '[structured value omitted]';
  // The shared helper bounds UTF-8 byte decoding before reading a value, including dictionaries.
  return formatOriginalTableCell(vector, row, 120).text;
}
