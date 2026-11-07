// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isOutOfMemoryError} from '@luma.gl/graph';

export type KeplerMemoryKind = 'cpu' | 'gpu';

/** Reuse the same descriptor for every alias of one allocation. */
export type KeplerDatasetResource = Readonly<{
  id: string;
  value: unknown;
  byteLength: number;
  memoryKind: KeplerMemoryKind;
  ownership: 'owned' | 'borrowed';
  destroy?: () => void;
}>;

export type KeplerSourceRow = Readonly<{
  datasetId: string;
  revision: number;
  batchId: string;
  sourceRowIndex: number;
}>;

/** Logical source batch boundaries, independent of physical GPU buffer positions. */
export type KeplerDatasetBatch = Readonly<{
  id: string;
  rowCount: number;
  sourceRowOffset?: number;
  source?: Readonly<{datasetId: string; revision: number; batchId: string}>;
}>;

/** The application retains a source or recipe that can reconstruct lost GPU resources. */
export type KeplerDatasetSource = Readonly<{
  kind: 'arrow' | 'duckdb' | 'application';
  value: unknown;
  metadata?: Readonly<Record<string, unknown>>;
}>;

/** Omitted for finite sources. A stream prefix is never a complete statistical population. */
export type KeplerDatasetPopulation = Readonly<{
  kind: 'stream';
  streamId: string;
  state: 'open' | 'complete' | 'cancelled' | 'failed';
  /**
   * Present only when a user explicitly closed the stream before end of input. The population is
   * then closed and immutable (state 'complete') but is a prefix, never a finite complete dataset.
   */
  partial?: Readonly<{rowCount: number; batchCount: number}>;
}>;

export type KeplerDatasetAccessOptions = Readonly<{
  /** Explicit bounded prefix inspection or lifecycle ownership, never inferred analysis consent. */
  allowIncomplete?: boolean;
}>;

/** `value` can contain a GPUDataFrame, GPUTable, Arrow metadata or application-specific views. */
export type KeplerDatasetDescriptor<Value = unknown> = Readonly<{
  id: string;
  value: Value;
  batches: readonly KeplerDatasetBatch[];
  resources: readonly KeplerDatasetResource[];
  source?: KeplerDatasetSource;
  population?: KeplerDatasetPopulation;
  /** Indirect results carry their identity vectors in `value`; never infer their source row. */
  rowIdentity?: 'source-order' | 'indirect';
}>;

export type KeplerDatasetSnapshot<Value = unknown> = KeplerDatasetDescriptor<Value> &
  Readonly<{revision: number; rowCount: number; deviceEpoch: number}>;

export type KeplerDatasetLease<Value = unknown> = Readonly<{
  dataset: KeplerDatasetSnapshot<Value>;
  isValid: () => boolean;
  /** A rejected completion fence still releases resources, and its error propagates. */
  release: (completion?: PromiseLike<unknown>) => Promise<void>;
}>;

/** Optional budget category for reservations; uncategorized reservations report as 'scratch'. */
export type KeplerMemoryCategory = 'source' | 'output' | 'scratch' | 'readback';

const ALLOCATION_MESSAGES = {
  'memory-budget': 'Dataset memory budget exceeded',
  capacity: 'Capacity exceeded',
  reservation: 'Reservation exceeded',
  'working-memory': 'Scalar selection exceeds its working memory reservation',
  'device-out-of-memory': 'Device out of memory'
} as const;

/** Typed allocation failure: registry budget/reservation admission or a device out-of-memory. */
export class KeplerAllocationError extends Error {
  override readonly name = 'KeplerAllocationError';
  constructor(
    readonly reason:
      | 'memory-budget'
      | 'capacity'
      | 'reservation'
      | 'working-memory'
      | 'device-out-of-memory',
    options?: {cause?: unknown; message?: string}
  ) {
    super(options?.message ?? ALLOCATION_MESSAGES[reason], options);
  }
}

/** True for KeplerAllocationError or luma OutOfMemoryError, including as a nested cause. */
export function isKeplerAllocationError(error: unknown): boolean {
  for (let depth = 0; error instanceof Error && depth < 8; depth++) {
    if (error instanceof KeplerAllocationError || isOutOfMemoryError(error)) return true;
    error = error.cause;
  }
  return false;
}

export type KeplerMemoryReservation = Readonly<{
  byteLength: number;
  memoryKind: KeplerMemoryKind;
  /** Budget category tag used by memory displays; backward compatible and optional. */
  category?: KeplerMemoryCategory;
  isActive: boolean;
  /** Admit growth before allocation; shrink only after the covered resources have been released. */
  resize: (byteLength: number) => void;
  release: () => void;
}>;

export type KeplerMemoryReserver = (
  byteLength: number,
  memoryKind: KeplerMemoryKind,
  reserve: (byteLength: number, memoryKind: KeplerMemoryKind) => KeplerMemoryReservation
) => KeplerMemoryReservation;

export type KeplerDatasetRequest = Readonly<{
  id: number;
  slot: string;
  signal: AbortSignal;
  inputs: readonly Readonly<{datasetId: string; revision: number}>[];
  deviceEpoch: number;
  /** Stops future scheduling/publication, not already submitted GPU work. */
  cancel: () => void;
}>;

export type KeplerRecoverySource = Readonly<{
  id: string;
  revision: number;
  source: KeplerDatasetSource;
  batches: readonly KeplerDatasetBatch[];
  rowIdentity: 'source-order' | 'indirect';
  population?: KeplerDatasetPopulation;
  /** CPU allocations remain retained and accounted while their source awaits re-upload. */
  resources: readonly KeplerDatasetResource[];
}>;

export type KeplerResourceCleanupError = Readonly<{resourceId: string; error: unknown}>;

type DatasetEntry = {
  dataset: KeplerDatasetSnapshot;
  leases: number;
  retired: boolean;
  disposed: boolean;
};

type ResourceEntry = {
  descriptor: KeplerDatasetResource;
  references: number;
};

type RequestEntry = {
  request: KeplerDatasetRequest;
  controller: AbortController;
  leases: readonly KeplerDatasetLease[];
  published: boolean;
  completion?: Promise<void>;
};

/**
 * Application-owned revision and resource lifecycle. No uploads, submissions or readbacks occur here.
 * A dataset owns one reference; leases keep retired revisions alive through rendering/readback fences.
 * Memory includes registered CPU/GPU allocations and reservations for uploads, scratch and readback.
 */
export class KeplerDatasetRegistry {
  private budgetBytes: number;
  private readonly datasets = new Map<string, DatasetEntry>();
  private readonly revisions = new Map<string, number>();
  private readonly resources = new Map<string, ResourceEntry>();
  private readonly resourceValues = new Map<unknown, string>();
  private readonly reservations = new Set<KeplerMemoryReservation>();
  private readonly requests = new Map<KeplerDatasetRequest, RequestEntry>();
  private readonly currentRequests = new Map<string, KeplerDatasetRequest>();
  private readonly recoverySources = new Map<string, KeplerRecoverySource>();
  private readonly cleanupErrors: KeplerResourceCleanupError[] = [];
  private allocatedBytes = {cpu: 0, gpu: 0};
  private reservedBytes = {cpu: 0, gpu: 0};
  private requestSequence = 0;
  private epoch = 1;
  private deviceLost = false;
  private destroyed = false;
  private destructionCompletion?: Promise<void>;
  private memoryReserver?: KeplerMemoryReserver;
  private pendingCategory: KeplerMemoryCategory | undefined;
  private readonly memoryListeners = new Set<() => void>();

  constructor({memoryBudgetBytes}: {memoryBudgetBytes: number}) {
    validateCount(memoryBudgetBytes);
    this.budgetBytes = memoryBudgetBytes;
  }

  get memoryBudgetBytes(): number {
    return this.budgetBytes;
  }

  /**
   * Changes the budget at runtime. Raising always succeeds; lowering below the bytes already
   * allocated or reserved is rejected so no live allocation is orphaned (evict or unload first).
   */
  setMemoryBudget(memoryBudgetBytes: number): void {
    validateCount(memoryBudgetBytes);
    // Budget below current usage; evict or unload datasets before lowering it.
    if (memoryBudgetBytes < this.memoryUsage.totalBytes) throw new Error('Budget below usage');
    this.budgetBytes = memoryBudgetBytes;
    this.notifyMemory();
  }

  get memoryUsage(): Readonly<{
    allocatedBytes: Readonly<Record<KeplerMemoryKind, number>>;
    reservedBytes: Readonly<Record<KeplerMemoryKind, number>>;
    totalBytes: number;
    availableBytes: number;
  }> {
    const totalBytes =
      this.allocatedBytes.cpu +
      this.allocatedBytes.gpu +
      this.reservedBytes.cpu +
      this.reservedBytes.gpu;
    return {
      allocatedBytes: {...this.allocatedBytes},
      reservedBytes: {...this.reservedBytes},
      totalBytes,
      availableBytes: this.budgetBytes - totalBytes
    };
  }

  /** Active reservation bytes per category (both memory kinds); untagged reservations count as scratch. */
  getReservationBreakdown(): Readonly<Record<KeplerMemoryCategory, number>> {
    const breakdown: Record<KeplerMemoryCategory, number> = {
      source: 0,
      output: 0,
      scratch: 0,
      readback: 0
    };
    for (const reservation of this.reservations)
      breakdown[reservation.category ?? 'scratch'] += reservation.byteLength;
    return breakdown;
  }

  /** Requests that are neither finished nor cancelled; each can be cancelled through its handle. */
  getActiveRequests(): readonly KeplerDatasetRequest[] {
    return Array.from(this.currentRequests.values()).filter(request => !request.signal.aborted);
  }

  /** Called after every reservation grows, shrinks, is created or released. */
  subscribeMemory(listener: () => void): () => void {
    this.memoryListeners.add(listener);
    return () => this.memoryListeners.delete(listener);
  }

  private notifyMemory(): void {
    for (const listener of this.memoryListeners) listener();
  }

  get deviceEpoch(): number {
    return this.epoch;
  }

  /**
   * Opt-in host policy (for example LRU eviction) that runs before every reservation. The reserver
   * receives the unwrapped reserve function, so it cannot recurse. Pass undefined to restore the
   * default behavior.
   */
  setMemoryReserver(reserver: KeplerMemoryReserver | undefined): void {
    this.memoryReserver = reserver;
  }

  /** Datasets and the bytes their owned resources release when removed; borrowed bytes excluded. */
  getDatasetFootprints(): readonly Readonly<{id: string; ownedByteLength: number}>[] {
    return Array.from(this.datasets.values(), ({dataset, retired}) => ({dataset, retired}))
      .filter(({retired}) => !retired)
      .map(({dataset}) => ({
        id: dataset.id,
        ownedByteLength: dataset.resources.reduce(
          (total, resource) =>
            resource.ownership === 'owned' && this.resources.get(resource.id)?.references === 1
              ? total + resource.byteLength
              : total,
          0
        )
      }));
  }

  /**
   * Per-dataset memory for display and unload decisions: bytes by memory kind, split into owned
   * (released when the dataset is removed) and borrowed (kept by another owner or alias).
   */
  getDatasetMemory(): readonly Readonly<{
    id: string;
    revision: number;
    rowCount: number;
    cpuBytes: number;
    gpuBytes: number;
    ownedBytes: number;
    borrowedBytes: number;
    sharedBytes: number;
    resourceCount: number;
  }>[] {
    return Array.from(this.datasets.values())
      .filter(({retired}) => !retired)
      .map(({dataset}) => {
        let cpuBytes = 0;
        let gpuBytes = 0;
        let ownedBytes = 0;
        let borrowedBytes = 0;
        let sharedBytes = 0;
        for (const resource of dataset.resources) {
          if (resource.memoryKind === 'cpu') cpuBytes += resource.byteLength;
          else gpuBytes += resource.byteLength;
          if (resource.ownership === 'owned') {
            ownedBytes += resource.byteLength;
            // Owned by this dataset but still referenced elsewhere, so removal will not free it.
            if ((this.resources.get(resource.id)?.references ?? 1) > 1)
              sharedBytes += resource.byteLength;
          } else borrowedBytes += resource.byteLength;
        }
        return {
          id: dataset.id,
          revision: dataset.revision,
          rowCount: dataset.rowCount,
          cpuBytes,
          gpuBytes,
          ownedBytes,
          borrowedBytes,
          sharedBytes,
          resourceCount: dataset.resources.length
        };
      });
  }

  /** Reserve before allocating. Failed registration retains the reservation for caller cleanup. */
  reserveMemory(
    byteLength: number,
    memoryKind: KeplerMemoryKind = 'gpu',
    category?: KeplerMemoryCategory
  ): KeplerMemoryReservation {
    this.pendingCategory = category;
    try {
      if (this.memoryReserver)
        return this.memoryReserver(byteLength, memoryKind, (nextByteLength, nextMemoryKind) =>
          this.reserveMemoryDirect(nextByteLength, nextMemoryKind)
        );
      return this.reserveMemoryDirect(byteLength, memoryKind);
    } finally {
      this.pendingCategory = undefined;
    }
  }

  private reserveMemoryDirect(
    byteLength: number,
    memoryKind: KeplerMemoryKind
  ): KeplerMemoryReservation {
    this.assertAvailable();
    validateCount(byteLength);
    this.assertCapacity(byteLength);
    let reservedByteLength = byteLength;
    const reservations = this.reservations;
    const category = this.pendingCategory;
    const reservation: KeplerMemoryReservation = Object.freeze({
      ...(category ? {category} : {}),
      get byteLength() {
        return reservedByteLength;
      },
      get isActive() {
        return reservations.has(reservation);
      },
      memoryKind,
      resize: (nextByteLength: number) => {
        if (!this.reservations.has(reservation)) throw new Error('Inactive memory reservation');
        validateCount(nextByteLength);
        const additionalBytes = nextByteLength - reservedByteLength;
        if (additionalBytes > 0) {
          this.assertAvailable();
          this.assertCapacity(additionalBytes);
        }
        this.reservedBytes[memoryKind] += additionalBytes;
        reservedByteLength = nextByteLength;
        this.notifyMemory();
      },
      release: () => {
        if (this.reservations.delete(reservation)) {
          this.reservedBytes[memoryKind] -= reservedByteLength;
          this.notifyMemory();
        }
      }
    });
    this.reservations.add(reservation);
    this.reservedBytes[memoryKind] += byteLength;
    this.notifyMemory();
    return reservation;
  }

  /** Atomic adoption. On failure, ownership and any reservation remain with the caller. */
  registerDataset<Value>(
    descriptor: KeplerDatasetDescriptor<Value>,
    reservation?: KeplerMemoryReservation | readonly KeplerMemoryReservation[]
  ): KeplerDatasetSnapshot<Value> {
    this.assertAvailable();
    if (!descriptor.id) throw new Error('Dataset id required');
    const population = descriptor.population;
    if (
      population &&
      (population.kind !== 'stream' ||
        !population.streamId ||
        !['open', 'complete', 'cancelled', 'failed'].includes(population.state) ||
        (population.partial &&
          (population.state !== 'complete' ||
            population.partial.batchCount !== descriptor.batches.length ||
            population.partial.rowCount !==
              descriptor.batches.reduce((total, batch) => total + batch.rowCount, 0))))
    )
      throw new Error('Invalid stream population metadata');
    const reservations: readonly KeplerMemoryReservation[] = reservation
      ? 'release' in reservation
        ? [reservation]
        : reservation
      : [];
    const reservedByKind: Record<KeplerMemoryKind, number> = {cpu: 0, gpu: 0};
    const reservedKinds = new Set<KeplerMemoryKind>();
    const seenReservations = new Set<KeplerMemoryReservation>();
    for (const candidate of reservations) {
      if (seenReservations.has(candidate)) throw new Error('Duplicate memory reservation');
      if (!this.reservations.has(candidate)) throw new Error('Inactive memory reservation');
      seenReservations.add(candidate);
      reservedKinds.add(candidate.memoryKind);
      reservedByKind[candidate.memoryKind] += candidate.byteLength;
      validateCount(reservedByKind[candidate.memoryKind]);
    }
    const batchIds = new Set<string>();
    let rowCount = 0;
    const batches = descriptor.batches.map(batch => {
      if (!batch.id || batchIds.has(batch.id)) throw new Error('Duplicate or empty batch id');
      batchIds.add(batch.id);
      validateCount(batch.rowCount);
      validateCount(batch.sourceRowOffset ?? 0);
      validateCount((batch.sourceRowOffset ?? 0) + batch.rowCount);
      if (batch.source) {
        if (!batch.source.datasetId || !batch.source.batchId)
          throw new Error('Invalid source identity');
        validateCount(batch.source.revision);
      }
      rowCount += batch.rowCount;
      validateCount(rowCount);
      return Object.freeze({...batch, source: batch.source && Object.freeze({...batch.source})});
    });
    const resources = Array.from(new Set(descriptor.resources));
    const resourceIds = new Set<string>();
    const values = new Set<unknown>();
    let addedBytes = 0;
    const addedByKind: Record<KeplerMemoryKind, number> = {cpu: 0, gpu: 0};
    for (const resource of resources) {
      validateCount(resource.byteLength);
      if (!resource.id || resourceIds.has(resource.id) || values.has(resource.value)) {
        throw new Error('Resource aliases require one descriptor');
      }
      if (resource.ownership === 'owned' && !resource.destroy) {
        throw new Error('Owned resource requires cleanup');
      }
      const existing = this.resources.get(resource.id);
      if (existing?.references === 0) throw new Error('Resource cleanup unresolved');
      const existingValueId = this.resourceValues.get(resource.value);
      if (
        (existing && existing.descriptor !== resource) ||
        (existingValueId && existingValueId !== resource.id)
      ) {
        throw new Error('Resource aliases require one descriptor');
      }
      resourceIds.add(resource.id);
      values.add(resource.value);
      if (!existing) {
        addedBytes += resource.byteLength;
        addedByKind[resource.memoryKind] += resource.byteLength;
        if (reservation && !reservedKinds.has(resource.memoryKind)) {
          throw new Error('Reservation memory kind mismatch');
        }
      }
    }
    if (
      reservation &&
      (addedByKind.cpu > reservedByKind.cpu || addedByKind.gpu > reservedByKind.gpu)
    )
      throw new KeplerAllocationError('reservation');
    this.assertCapacity(addedBytes - reservedByKind.cpu - reservedByKind.gpu);
    const revision = (this.revisions.get(descriptor.id) ?? 0) + 1;
    validateCount(revision);
    const dataset: KeplerDatasetSnapshot<Value> = Object.freeze({
      ...descriptor,
      ...(population ? {population: Object.freeze({...population})} : {}),
      batches: Object.freeze(batches),
      resources: Object.freeze(resources),
      rowCount,
      revision,
      deviceEpoch: this.epoch,
      rowIdentity: descriptor.rowIdentity ?? 'source-order'
    });
    for (const resource of resources) {
      const existing = this.resources.get(resource.id);
      if (existing) existing.references++;
      else {
        this.resources.set(resource.id, {descriptor: resource, references: 1});
        this.resourceValues.set(resource.value, resource.id);
        this.allocatedBytes[resource.memoryKind] += resource.byteLength;
      }
    }
    for (const candidate of reservations) candidate.release();
    const previous = this.datasets.get(descriptor.id);
    this.datasets.set(descriptor.id, {dataset, leases: 0, retired: false, disposed: false});
    this.revisions.set(descriptor.id, revision);
    this.releaseRecoverySource(descriptor.id);
    if (previous) this.retireDataset(previous);
    this.cancelChangedRequests(descriptor.id);
    return dataset;
  }

  getDataset<Value = unknown>(id: string): KeplerDatasetSnapshot<Value> | undefined {
    return this.datasets.get(id)?.dataset as KeplerDatasetSnapshot<Value> | undefined;
  }

  acquireDataset<Value = unknown>(
    id: string,
    options: KeplerDatasetAccessOptions = {}
  ): KeplerDatasetLease<Value> {
    this.assertAvailable();
    const entry = this.datasets.get(id);
    if (!entry) throw new Error('Unknown dataset');
    assertPopulationAdmission(entry.dataset, options);
    entry.leases++;
    let completion: Promise<void> | undefined;
    let released = false;
    return Object.freeze({
      dataset: entry.dataset as KeplerDatasetSnapshot<Value>,
      isValid: () =>
        !released &&
        !entry.disposed &&
        !this.deviceLost &&
        !this.destroyed &&
        entry.dataset.deviceEpoch === this.epoch,
      release: (fence?: PromiseLike<unknown>) => {
        if (!completion) {
          released = true;
          const finish = () => {
            entry.leases--;
            this.disposeRetiredDataset(entry);
          };
          if (fence)
            completion = Promise.resolve(fence).then(
              () => finish(),
              error => {
                finish();
                throw error;
              }
            );
          else {
            finish();
            completion = Promise.resolve();
          }
        }
        return completion;
      }
    });
  }

  /** Source-order only. Indirect result identities must be resolved from their explicit mapping. */
  resolveSourceRow(dataset: KeplerDatasetSnapshot, rowIndex: number): KeplerSourceRow {
    validateCount(rowIndex);
    if (rowIndex >= dataset.rowCount) throw new Error('Row out of range');
    if (dataset.rowIdentity === 'indirect')
      throw new Error('Indirect identity requires explicit mapping');
    let batchOffset = 0;
    for (const batch of dataset.batches) {
      if (rowIndex < batchOffset + batch.rowCount) {
        return {
          datasetId: batch.source?.datasetId ?? dataset.id,
          revision: batch.source?.revision ?? dataset.revision,
          batchId: batch.source?.batchId ?? batch.id,
          sourceRowIndex: (batch.sourceRowOffset ?? 0) + rowIndex - batchOffset
        };
      }
      batchOffset += batch.rowCount;
    }
    throw new Error('Row out of range');
  }

  removeDataset(id: string): boolean {
    const hadRecoverySource = this.recoverySources.has(id);
    this.releaseRecoverySource(id);
    const entry = this.datasets.get(id);
    if (!entry) return hadRecoverySource;
    this.datasets.delete(id);
    this.retireDataset(entry);
    this.cancelChangedRequests(id);
    return true;
  }

  beginRequest(
    slot: string,
    datasetIds: readonly string[],
    options: KeplerDatasetAccessOptions = {}
  ): KeplerDatasetRequest {
    this.assertAvailable();
    if (!slot) throw new Error('Request slot required');
    // Validate before taking leases or superseding the previous request.
    for (const id of datasetIds) {
      const entry = this.datasets.get(id);
      if (!entry) throw new Error('Unknown dataset');
      assertPopulationAdmission(entry.dataset, options);
    }
    const leases = Array.from(new Set(datasetIds)).map(id => this.acquireDataset(id, options));
    const controller = new AbortController();
    const request: KeplerDatasetRequest = Object.freeze({
      id: ++this.requestSequence,
      slot,
      signal: controller.signal,
      inputs: Object.freeze(
        leases.map(lease =>
          Object.freeze({datasetId: lease.dataset.id, revision: lease.dataset.revision})
        )
      ),
      deviceEpoch: this.epoch,
      cancel: () => controller.abort()
    });
    this.currentRequests.get(slot)?.cancel();
    this.currentRequests.set(slot, request);
    this.requests.set(request, {request, controller, leases, published: false});
    this.notifyMemory();
    return request;
  }

  /** Rejection leaves result resources/reservation with the caller. Finish the request after its fence. */
  publishResult<Value>(
    request: KeplerDatasetRequest,
    result: KeplerDatasetDescriptor<Value>,
    reservation?: KeplerMemoryReservation | readonly KeplerMemoryReservation[]
  ): KeplerDatasetSnapshot<Value> | null {
    const entry = this.requests.get(request);
    if (
      !entry ||
      entry.completion ||
      entry.published ||
      request.signal.aborted ||
      this.destroyed ||
      this.deviceLost ||
      request.deviceEpoch !== this.epoch ||
      this.currentRequests.get(request.slot) !== request
    )
      return null;
    if (
      request.inputs.some(
        input => this.datasets.get(input.datasetId)?.dataset.revision !== input.revision
      )
    )
      return null;
    // Prefix inspection permission cannot relabel an unfinished source as a complete result.
    if (
      entry.leases.some(
        lease => lease.dataset.population && lease.dataset.population.state !== 'complete'
      ) &&
      (!result.population || result.population.state === 'complete')
    )
      return null;
    const dataset = this.registerDataset(result, reservation);
    entry.published = true;
    return dataset;
  }

  /** Cancellation alone never releases input leases while submitted commands may still use them. */
  finishRequest(request: KeplerDatasetRequest, completion?: PromiseLike<unknown>): Promise<void> {
    const entry = this.requests.get(request);
    if (!entry) return Promise.resolve();
    if (!entry.completion) {
      entry.controller.abort();
      if (this.currentRequests.get(request.slot) === request)
        this.currentRequests.delete(request.slot);
      this.notifyMemory();
      entry.completion = Promise.all([
        Promise.resolve(completion),
        ...entry.leases.map(lease => lease.release(completion))
      ])
        .then(() => undefined)
        .finally(() => {
          this.requests.delete(request);
        });
    }
    return entry.completion;
  }

  /** Device-loss invalidation retains only explicit recovery sources as active application metadata. */
  handleDeviceLoss(): void {
    if (this.destroyed || this.deviceLost) return;
    this.deviceLost = true;
    this.epoch++;
    for (const entry of this.datasets.values()) {
      const dataset = entry.dataset;
      if (dataset.source) {
        this.releaseRecoverySource(dataset.id);
        const resources = dataset.resources.filter(resource => resource.memoryKind === 'cpu');
        for (const resource of resources) this.resources.get(resource.id)!.references++;
        this.recoverySources.set(dataset.id, {
          id: dataset.id,
          revision: dataset.revision,
          source: dataset.source,
          batches: dataset.batches,
          rowIdentity: dataset.rowIdentity ?? 'source-order',
          ...(dataset.population ? {population: dataset.population} : {}),
          resources
        });
      }
      this.retireDataset(entry);
    }
    this.datasets.clear();
    for (const request of this.requests.keys()) {
      request.cancel();
      void this.finishRequest(request);
    }
    for (const reservation of this.reservations) reservation.release();
  }

  getRecoverySources(): readonly KeplerRecoverySource[] {
    return Array.from(this.recoverySources.values());
  }

  /** Cleanup exceptions do not interrupt other disposals or turn successful publication into failure. */
  takeCleanupErrors(): readonly KeplerResourceCleanupError[] {
    return this.cleanupErrors.splice(0);
  }

  /** Failed destructors remain accounted until the application repairs cleanup and retries. */
  retryResourceCleanup(resourceId: string): boolean {
    const resource = this.resources.get(resourceId);
    if (!resource) return true;
    if (resource.references !== 0) return false;
    return this.disposeResource(resource.descriptor);
  }

  /** The application creates a new device and explicitly uploads each retained source afterward. */
  recoverDevice(): void {
    if (this.destroyed) throw new Error('Registry destroyed');
    this.deviceLost = false;
  }

  /** Waits for the supplied fence. External leases still own their additional completion fences. */
  destroy(completion?: PromiseLike<unknown>): Promise<void> {
    if (this.destructionCompletion) return this.destructionCompletion;
    const entries = Array.from(this.datasets.values());
    if (completion) for (const entry of entries) entry.leases++;
    this.destroyed = true;
    for (const entry of entries) this.retireDataset(entry);
    this.datasets.clear();
    for (const id of this.recoverySources.keys()) this.releaseRecoverySource(id);
    for (const reservation of this.reservations) reservation.release();
    const releaseEntries = Promise.resolve(completion).finally(() => {
      if (completion)
        for (const entry of entries) {
          entry.leases--;
          this.disposeRetiredDataset(entry);
        }
    });
    this.destructionCompletion = Promise.all([
      releaseEntries,
      ...Array.from(this.requests.keys(), request => this.finishRequest(request, completion))
    ]).then(() => undefined);
    return this.destructionCompletion;
  }

  private assertAvailable(): void {
    if (this.destroyed) throw new Error('Registry destroyed');
    if (this.deviceLost) throw new Error('Device lost');
  }

  private assertCapacity(addedBytes: number): void {
    if (this.memoryUsage.totalBytes + addedBytes > this.budgetBytes) {
      throw new KeplerAllocationError('memory-budget');
    }
  }

  private cancelChangedRequests(datasetId: string): void {
    for (const entry of this.requests.values()) {
      if (entry.request.inputs.some(input => input.datasetId === datasetId))
        entry.controller.abort();
    }
  }

  private retireDataset(entry: DatasetEntry): void {
    entry.retired = true;
    this.disposeRetiredDataset(entry);
  }

  private disposeRetiredDataset(entry: DatasetEntry): void {
    if (!entry.retired || entry.leases !== 0 || entry.disposed) return;
    entry.disposed = true;
    for (const resource of entry.dataset.resources) this.releaseResource(resource);
  }

  private releaseRecoverySource(id: string): void {
    const recovery = this.recoverySources.get(id);
    if (!recovery) return;
    this.recoverySources.delete(id);
    for (const resource of recovery.resources) this.releaseResource(resource);
  }

  private releaseResource(resource: KeplerDatasetResource): void {
    const resourceEntry = this.resources.get(resource.id)!;
    resourceEntry.references--;
    if (resourceEntry.references !== 0) return;
    this.disposeResource(resource);
  }

  private disposeResource(resource: KeplerDatasetResource): boolean {
    if (resource.ownership === 'owned') {
      try {
        resource.destroy?.();
      } catch (error) {
        this.cleanupErrors.push({resourceId: resource.id, error});
        return false;
      }
    }
    this.resources.delete(resource.id);
    this.resourceValues.delete(resource.value);
    this.allocatedBytes[resource.memoryKind] -= resource.byteLength;
    return true;
  }
}

function validateCount(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0)
    throw new Error('Expected nonnegative safe integer');
}

function assertPopulationAdmission(
  dataset: KeplerDatasetSnapshot,
  options: KeplerDatasetAccessOptions
): void {
  if (dataset.population && dataset.population.state !== 'complete' && !options.allowIncomplete)
    throw new Error('Dataset population is incomplete; explicitly opt in to prefix inspection');
}
