// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {DataType, RecordBatch, Table, type Field} from 'apache-arrow';
import {
  prepareArrowTableForGPUAnalytics,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma-contributors/graph/arrow';
import type {KeplerDatasetSnapshot, KeplerMemoryReservation} from '../runtime/dataset-registry';
import {
  KeplerComputeSession,
  getPreparedArrowUploadByteLength,
  getRetainedArrowArrayBufferBytes,
  type KeplerArrowDataset
} from './compute-session';
import {
  destroyArrowStreamUpload,
  makeArrowStreamSnapshot,
  uploadArrowStreamPart,
  type ArrowStreamPart
} from './arrow-stream-snapshot';

export type KeplerArrowStreamState = 'open' | 'closing' | 'complete' | 'cancelled' | 'failed';
export type KeplerArrowStreamOptions = {
  session: KeplerComputeSession;
  id: string;
  normalization?: PrepareArrowTableForGPUAnalyticsOptions;
  /** Includes the active batch, measured as retained source ArrayBuffer capacities. */
  maximumQueuedBytes: number;
  maximumQueuedBatches: number;
  /** Bounds normalization expansion independently of compact Arrow validity/Boolean buffers. */
  maximumBatchRows: number;
  /** Ordered dictionaries always remain fixed; unordered labels may append with stable old codes. */
  dictionaryEvolution: 'fixed' | 'append-unordered';
  /** Recorded source provenance, such as a remote origin. Reserved stream keys win. */
  metadata?: Readonly<Record<string, unknown>>;
  signal?: AbortSignal;
};

type Job = {
  source: Table;
  byteLength: number;
  reservation: KeplerMemoryReservation;
  resolve: (snapshot: KeplerDatasetSnapshot<KeplerArrowDataset>) => void;
  reject: (reason: unknown) => void;
};

/** Rejected input is not retained. Await publisher.ready, then retry the same batch explicitly. */
export class KeplerStreamBackpressureError extends Error {
  constructor() {
    super('Arrow stream admission queue is full');
    this.name = 'KeplerStreamBackpressureError';
  }
}

let streamSequence = 0;

/**
 * Append-only Arrow admission. Each original record batch uploads once; immutable revision views
 * borrow shared batch allocations. Consumers lease revisions and provide fences before release.
 * No source buffers are packed, copied between GPU allocations, or read back here.
 *
 * Queue accounting conservatively counts backing capacities per admitted batch. Registered CPU
 * accounting includes source/normalized typed storage; JavaScript strings/objects and temporary
 * normalization allocations are excluded, matching ordinary session imports. Schema changes reject;
 * dictionaries follow the explicit policy. An unfinished/cancelled/failed prefix is never complete.
 */
export class KeplerArrowStreamPublisher {
  readonly id: string;
  readonly streamId: string;
  private readonly device;
  private readonly normalization: PrepareArrowTableForGPUAnalyticsOptions;
  private readonly queue: Job[] = [];
  private parts: ArrowStreamPart[] = [];
  private sourceSchema?: string;
  private current?: KeplerDatasetSnapshot<KeplerArrowDataset>;
  private stateValue: KeplerArrowStreamState = 'open';
  private failure?: unknown;
  private scheduled = false;
  private queuedBytesValue = 0;
  private queuedBatchesValue = 0;
  private sequence = 0;
  private capacity?: {promise: Promise<void>; resolve(): void};
  private completion?: {
    promise: Promise<KeplerDatasetSnapshot<KeplerArrowDataset>>;
    resolve(value: KeplerDatasetSnapshot<KeplerArrowDataset>): void;
    reject(reason: unknown): void;
  };
  private disposal?: Promise<void>;
  private partialClose = false;
  private detached = false;
  private readonly lossObserver: {notify?: () => void} = {};
  private readonly abort = () => this.cancel(this.options.signal?.reason);

  constructor(private readonly options: KeplerArrowStreamOptions) {
    this.id = options.id;
    this.streamId = `${options.id}-stream-${++streamSequence}`;
    this.device = options.session.device;
    if (!this.id || options.session.registry.getDataset(this.id))
      throw new Error('Stream requires an unused dataset id');
    for (const count of [
      options.maximumQueuedBytes,
      options.maximumQueuedBatches,
      options.maximumBatchRows
    ])
      if (!Number.isSafeInteger(count) || count < 1)
        throw new Error('Stream capacities must be positive safe integers');
    if (!['fixed', 'append-unordered'].includes(options.dictionaryEvolution))
      throw new Error('Explicit stream dictionary policy required');
    const normalization = options.normalization ?? {};
    this.normalization = {
      ...normalization,
      columns: normalization.columns && [...normalization.columns],
      float64: normalization.float64 && {...normalization.float64},
      timeColumns: normalization.timeColumns && {...normalization.timeColumns},
      dictionaries:
        normalization.dictionaries &&
        Object.fromEntries(
          Object.entries(normalization.dictionaries).map(([name, dictionary]) => [
            name,
            {...dictionary, values: [...dictionary.values]}
          ])
        )
    };
    options.signal?.addEventListener('abort', this.abort, {once: true});
    if (options.signal?.aborted) this.abort();
    const lossObserver = this.lossObserver;
    lossObserver.notify = () => this.stop('failed', new Error('Stream device lost'), false);
    void this.device.lost
      .then(
        () => lossObserver.notify?.(),
        () => lossObserver.notify?.()
      )
      .catch(() => undefined);
  }

  get state(): KeplerArrowStreamState {
    return this.stateValue;
  }
  get snapshot(): KeplerDatasetSnapshot<KeplerArrowDataset> | undefined {
    return this.current;
  }
  get queuedBytes(): number {
    return this.queuedBytesValue;
  }
  get queuedBatches(): number {
    return this.queuedBatchesValue;
  }
  /** One shared wake-up signal, not an unbounded collection of retained input waiters. */
  get ready(): Promise<void> {
    if (this.stateValue !== 'open')
      return Promise.reject(this.failure ?? new Error('Stream is closed'));
    if (!this.queuedBatchesValue) return Promise.resolve();
    if (!this.capacity) {
      let resolve!: () => void;
      const promise = new Promise<void>(complete => {
        resolve = complete;
      });
      this.capacity = {promise, resolve};
    }
    return this.capacity.promise;
  }

  /** Takes temporary ownership only after admission; caller must not mutate admitted Arrow storage. */
  enqueue(batch: RecordBatch): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    this.assertWritable();
    if (!(batch instanceof RecordBatch))
      throw new Error('Streaming admission requires one original Arrow RecordBatch');
    if (batch.numRows > this.options.maximumBatchRows)
      throw new Error('Arrow batch exceeds stream row capacity');
    const schema = schemaKey(batch);
    if (this.sourceSchema && this.sourceSchema !== schema)
      throw new Error('Arrow stream schema evolution is disabled');
    const source = new Table(batch.schema, [batch]);
    const byteLength = getRetainedArrowArrayBufferBytes(source);
    if (byteLength > this.options.maximumQueuedBytes)
      throw new KeplerAllocationError('capacity', {
        message: 'Arrow batch exceeds stream byte capacity'
      });
    if (
      this.queuedBatchesValue >= this.options.maximumQueuedBatches ||
      this.queuedBytesValue + byteLength > this.options.maximumQueuedBytes
    )
      throw new KeplerStreamBackpressureError();
    const reservation = this.options.session.registry.reserveMemory(byteLength, 'cpu', 'source');
    this.sourceSchema ??= schema;
    this.queuedBytesValue += byteLength;
    this.queuedBatchesValue++;
    const promise = new Promise<KeplerDatasetSnapshot<KeplerArrowDataset>>((resolve, reject) => {
      this.queue.push({source, byteLength, reservation, resolve, reject});
    });
    this.schedule();
    return promise;
  }

  /** Pulls at most one source batch ahead; the next iterator pull waits for publication. */
  async consume(
    source: AsyncIterable<RecordBatch> | Iterable<RecordBatch>
  ): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    try {
      for await (const batch of source) await this.enqueue(batch);
      return await this.finish();
    } catch (error) {
      if (this.stateValue === 'open' || this.stateValue === 'closing') this.stop('failed', error);
      throw error;
    }
  }

  /** Drains accepted inputs, then publishes a complete revision without allocating/uploading data. */
  finish(): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    if (this.disposal || this.detached)
      return Promise.reject(new Error('Arrow stream is disposed or detached'));
    if (this.stateValue === 'complete') return Promise.resolve(this.current!);
    if (this.stateValue === 'cancelled' || this.stateValue === 'failed')
      return Promise.reject(this.failure);
    if (!this.completion) {
      let resolve!: (snapshot: KeplerDatasetSnapshot<KeplerArrowDataset>) => void;
      let reject!: (reason: unknown) => void;
      const promise = new Promise<KeplerDatasetSnapshot<KeplerArrowDataset>>((complete, fail) => {
        resolve = complete;
        reject = fail;
      });
      this.completion = {promise, resolve, reject};
      this.stateValue = 'closing';
      this.schedule();
    }
    return this.completion.promise;
  }

  /**
   * Closes an unfinished stream at its admitted prefix after an explicit user decision. The result
   * is immutable and uploadable but its population records `partial`, so saved manifests and
   * restore never describe it as a finite complete dataset. Requires at least one admitted batch.
   */
  finishPartial(): Promise<KeplerDatasetSnapshot<KeplerArrowDataset>> {
    if (this.stateValue === 'open') this.partialClose = true;
    return this.finish();
  }

  /** Stops ingestion and retains an explicitly cancelled prefix; does not free rendered buffers. */
  cancel(reason: unknown = new Error('Arrow stream cancelled')): void {
    this.stop('cancelled', reason);
  }

  /** Transfer a completed registry revision to the host without releasing or uploading its chunks. */
  detachCompleted(): KeplerDatasetSnapshot<KeplerArrowDataset> {
    if (this.stateValue !== 'complete' || this.disposal || this.detached || !this.current)
      throw new Error('Only a completed stream can transfer ownership');
    // The host may have recovered the completed population while native attachment was pending.
    // Detaching releases only this publisher's references, never the replacement registry revision.
    const snapshot = this.current!;
    this.detached = true;
    this.lossObserver.notify = undefined;
    this.options.signal?.removeEventListener('abort', this.abort);
    this.parts = [];
    this.current = undefined;
    return snapshot;
  }

  /** Explicitly remove the retained prefix only after dependent rendering/readback work settles. */
  dispose(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.disposal) {
      this.cancel();
      this.disposal = (async () => {
        try {
          if (this.ownsCurrent()) await this.options.session.removeDataset(this.id, completion);
          else if (!this.detached) {
            if (this.options.session.deviceLost) await this.options.session.handleDeviceLoss();
            const recovery = this.options.session.registry
              .getRecoverySources()
              .find(source => source.id === this.id);
            if (recovery?.source.metadata?.['keplerStreamId'] === this.streamId)
              this.options.session.registry.removeDataset(this.id);
          }
        } finally {
          this.parts = [];
          this.current = undefined;
          this.lossObserver.notify = undefined;
          this.options.signal?.removeEventListener('abort', this.abort);
        }
      })();
    }
    return this.disposal;
  }

  private schedule(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => {
      this.scheduled = false;
      if (this.stateValue !== 'open' && this.stateValue !== 'closing') return;
      const job = this.queue.shift();
      if (job) {
        try {
          job.resolve(this.append(job));
        } catch (error) {
          job.reject(error);
          this.stop('failed', error);
        } finally {
          job.reservation.release();
          this.queuedBytesValue -= job.byteLength;
          this.queuedBatchesValue--;
          this.wakeCapacity();
        }
        if (this.queue.length || this.stateValue === 'closing') this.schedule();
      } else if (this.stateValue === 'closing') {
        try {
          if (!this.parts.length)
            throw new Error('A stream must admit at least one batch, including an empty batch');
          const snapshot = this.publish('complete');
          this.stateValue = 'complete';
          this.options.signal?.removeEventListener('abort', this.abort);
          this.completion!.resolve(snapshot);
        } catch (error) {
          this.stop('failed', error);
        }
      }
    });
  }

  private append(job: Job): KeplerDatasetSnapshot<KeplerArrowDataset> {
    this.assertOwnership();
    const previous = this.parts.at(-1)?.prepared.dictionaries ?? this.normalization.dictionaries;
    const prepared = prepareArrowTableForGPUAnalytics(job.source, {
      ...this.normalization,
      dictionaries: previous
    });
    if (
      this.options.dictionaryEvolution === 'fixed' &&
      previous &&
      Object.entries(prepared.dictionaries).some(
        ([name, dictionary]) =>
          previous[name] &&
          (dictionary.ordered !== previous[name].ordered ||
            dictionary.values.length !== previous[name].values.length ||
            dictionary.values.some((value, index) => value !== previous[name].values[index]))
      )
    )
      throw new Error('Arrow stream dictionary evolution is disabled');
    const retainedArrayBufferBytes = getRetainedArrowArrayBufferBytes(job.source, prepared);
    const extraCpu = this.options.session.registry.reserveMemory(
      Math.max(0, retainedArrayBufferBytes - job.byteLength),
      'cpu',
      'source'
    );
    let gpuReservation: KeplerMemoryReservation | undefined;
    let upload: ReturnType<typeof uploadArrowStreamPart> | undefined;
    let adopted = false;
    try {
      gpuReservation = this.options.session.registry.reserveMemory(
        getPreparedArrowUploadByteLength(prepared),
        'gpu',
        'source'
      );
      upload = uploadArrowStreamPart(this.device, job.source, prepared);
      const retainedUpload = upload;
      const identity = `${this.streamId}-batch-${this.parts.length}`;
      const part: ArrowStreamPart = {
        source: job.source,
        prepared,
        ...upload,
        retainedArrayBufferBytes,
        resources: [
          {
            id: `${identity}-cpu`,
            value: {source: job.source, prepared},
            byteLength: retainedArrayBufferBytes,
            memoryKind: 'cpu',
            ownership: 'borrowed'
          },
          {
            id: `${identity}-gpu`,
            value: upload,
            byteLength: upload.uploadedBufferBytes,
            memoryKind: 'gpu',
            ownership: 'owned',
            destroy: () => destroyArrowStreamUpload(retainedUpload)
          }
        ]
      };
      this.assertOwnership();
      if (this.stateValue !== 'open' && this.stateValue !== 'closing')
        throw this.failure ?? new Error('Arrow stream stopped during upload');
      // Synchronous adoption transfers reservations to shared resource descriptors atomically.
      job.reservation.release();
      extraCpu.release();
      gpuReservation.release();
      const snapshot = this.publish('open', [...this.parts, part]);
      this.parts.push(part);
      adopted = true;
      return snapshot;
    } finally {
      extraCpu.release();
      gpuReservation?.release();
      if (upload && !adopted) destroyArrowStreamUpload(upload);
    }
  }

  private publish(
    state: 'open' | 'complete' | 'cancelled' | 'failed',
    parts = this.parts
  ): KeplerDatasetSnapshot<KeplerArrowDataset> {
    this.assertOwnership();
    const value = makeArrowStreamSnapshot(parts);
    const views = {
      id: `${this.streamId}-views-${++this.sequence}`,
      value: value.analytics,
      byteLength: 0,
      memoryKind: 'gpu' as const,
      ownership: 'owned' as const,
      destroy: () => destroyArrowStreamUpload(value)
    };
    const partial =
      state === 'complete' && this.partialClose
        ? {rowCount: value.source.numRows, batchCount: parts.length}
        : undefined;
    const preparedResource = {
      id: `${this.streamId}-prepared-${this.sequence}`,
      value: value.prepared,
      byteLength: 0,
      memoryKind: 'cpu' as const,
      ownership: 'borrowed' as const
    };
    try {
      const snapshot = this.options.session.registry.registerDataset({
        id: this.id,
        value,
        population: {kind: 'stream', streamId: this.streamId, state, ...(partial ? {partial} : {})},
        batches: value.prepared.sourceBatches.map(batch => ({
          id: String(batch.sourceBatchIndex),
          rowCount: batch.sourceRowCount,
          sourceRowOffset: batch.sourceRowIndexOffset
        })),
        resources: [
          ...parts.flatMap(part => part.resources),
          ...(state === 'complete' ? [preparedResource] : []),
          views
        ],
        // Completed streams restore prepared chunks without renormalizing dictionary codes.
        // Unfinished populations require an explicit application reimport/rebind after loss.
        source: {
          kind: 'arrow',
          value: value.source,
          metadata: {
            ...this.options.metadata,
            keplerStreamId: this.streamId,
            ...(state === 'complete'
              ? {keplerPreparedArrow: true, keplerPreparedResourceId: preparedResource.id}
              : {}),
            streamState: state,
            normalization: this.normalization,
            dictionaryEvolution: this.options.dictionaryEvolution
          }
        }
      });
      this.current = snapshot;
      return snapshot;
    } catch (error) {
      views.destroy();
      throw error;
    }
  }

  private stop(state: 'cancelled' | 'failed', reason: unknown, publish = true): void {
    if (
      this.stateValue === 'complete' ||
      this.stateValue === 'cancelled' ||
      this.stateValue === 'failed'
    )
      return;
    this.stateValue = state;
    this.failure = reason;
    this.options.signal?.removeEventListener('abort', this.abort);
    for (const job of this.queue.splice(0)) {
      job.reservation.release();
      this.queuedBytesValue -= job.byteLength;
      this.queuedBatchesValue--;
      job.reject(reason);
    }
    if (publish && this.parts.length && this.ownsCurrent() && !this.options.session.deviceLost) {
      try {
        this.publish(state);
      } catch {
        /* External deletion/loss supersedes publication. */
      }
    }
    this.completion?.reject(reason);
    this.wakeCapacity();
  }
  private wakeCapacity(): void {
    this.capacity?.resolve();
    this.capacity = undefined;
  }
  private assertWritable(): void {
    if (this.stateValue !== 'open') throw this.failure ?? new Error('Arrow stream is closed');
    this.assertOwnership();
  }
  private ownsCurrent(): boolean {
    return Boolean(
      this.current &&
        this.options.session.registry.getDataset(this.id)?.revision === this.current.revision
    );
  }
  private assertOwnership(): void {
    if (this.options.session.deviceLost || this.options.session.device !== this.device)
      throw new Error('Arrow stream device changed');
    const registered = this.options.session.registry.getDataset(this.id);
    if (registered?.revision !== this.current?.revision)
      throw new Error('Arrow stream publication was replaced or removed');
  }
}

function schemaKey(batch: RecordBatch): string {
  const fieldKey = (field: Field): unknown => [
    field.name,
    String(field.type),
    field.nullable,
    [...field.metadata].sort(),
    DataType.isDictionary(field.type) ? [field.type.id, field.type.isOrdered] : null,
    (field.type.children ?? []).map(fieldKey)
  ];
  return JSON.stringify([[...batch.schema.metadata].sort(), batch.schema.fields.map(fieldKey)]);
}
