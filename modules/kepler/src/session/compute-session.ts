// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder, Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  makeGPUAnalyticsTableFromArrowTable,
  makeGPUTimeVectorFromPreparedArrowColumn,
  prepareArrowTableForGPUAnalytics,
  type GPUAnalyticsTableFromArrowTableResult,
  type GPUTimeVectorFromPreparedArrowColumn,
  type PreparedArrowAnalyticsTable,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma-contributors/graph/arrow';
import type {Data, Table} from 'apache-arrow';
import {
  makeAdoptedArrowAnalytics,
  type KeplerAdoptedArrowDatasetOptions
} from './adopted-arrow-dataset';
export type {KeplerAdoptedArrowDatasetOptions} from './adopted-arrow-dataset';
import {
  KeplerResidentRows,
  makeResidentAnalytics,
  makeResidentPreparedTable,
  estimateKeplerResidentRowCacheByteLength,
  type KeplerResidentDatasetOptions
} from './resident-dataset';
import {
  GPUResidentExplorer,
  type GPUResidentColumns,
  type GPUResidentExplorerProps,
  type GPUResidentPage,
  type GPUResidentScalarFormat,
  type GPUResidentSummary
} from '../exploration/gpu-resident-explorer';
import {
  KeplerAllocationError,
  KeplerDatasetRegistry,
  type KeplerDatasetLease,
  type KeplerDatasetResource,
  type KeplerDatasetDescriptor,
  type KeplerDatasetPopulation,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation
} from '../runtime/dataset-registry';

type KeplerGPUAnalytics = GPUAnalyticsTableFromArrowTableResult<
  Record<string, GPUResidentScalarFormat>
>;

/** CPU representations remain explicit for export and device recovery. */
export type KeplerArrowDataset = {
  source: Table;
  prepared: PreparedArrowAnalyticsTable;
  analytics: KeplerGPUAnalytics;
  columns: GPUResidentColumns;
  timeColumns: Readonly<Record<string, GPUTimeVectorFromPreparedArrowColumn>>;
  /** Unique retained ArrayBuffer capacities within this dataset; excludes JavaScript object overhead. */
  retainedArrayBufferBytes: number;
  /** Actual buffer byte lengths, including empty-chunk allocations. */
  uploadedBufferBytes: number;
  /**
   * Present for GPU-resident results adopted without a full readback. `source` and
   * `prepared.table` are then lazy views: cells are null until `resident.ensureRows()` pages them in.
   */
  resident?: KeplerResidentRows;
};

export type KeplerArrowDatasetOptions = {
  id: string;
  table: Table;
  normalization?: PrepareArrowTableForGPUAnalyticsOptions;
  /** Restores recorded stream provenance, such as an explicitly partial population. */
  population?: KeplerDatasetPopulation;
  /** Extra source metadata, such as a remote stream origin; reserved Kepler keys win. */
  metadata?: Readonly<Record<string, unknown>>;
};

export type KeplerSessionExplorerOptions = Omit<
  GPUResidentExplorerProps,
  'device' | 'columns' | 'validity' | 'datasetRevision' | 'sourceChunkLengths'
> & {
  /**
   * Application admission ceiling for outputs, graph scratch and bounded inspection staging.
   * Reserve the full ceiling during compilation, then retain the accounted resident plan plus
   * bounded inspection capacity. This is planned storage, not measured total device usage. Driver/pipeline allocations
   * private executable allocations and command-encoder-owned control uploads are excluded
   * from this explorer-owned accounting.
   */
  workingMemoryBudgetBytes: number;
  /** Explicit preview of the admitted prefix; complete-population analysis stays guarded. */
  allowIncompletePopulation?: boolean;
};

/**
 * Concrete Arrow -> resident dataset -> interactive exploration lifecycle.
 *
 * Sources upload once per revision. Exploration borrows them through registry leases. Retained
 * typed storage is accounted; JavaScript object/string overhead and temporary normalization
 * allocations are excluded. Shared CPU buffers in different dataset revisions are conservatively
 * counted once per revision. Applications choose separate working-set reservations for exploration.
 */
export class KeplerComputeSession {
  readonly registry: KeplerDatasetRegistry;
  private currentDevice: Device;
  private readonly explorers = new Set<KeplerSessionExplorer>();
  private resourceSequence = 0;
  private destroyed = false;
  private destroyCompletion?: Promise<void>;
  private lossCompletion?: Promise<void>;
  private lost = false;
  private resolveDeviceLoss!: () => void;
  private deviceLossSignal = new Promise<void>(resolve => {
    this.resolveDeviceLoss = resolve;
  });

  constructor(props: {device: Device; memoryBudgetBytes: number}) {
    this.currentDevice = props.device;
    this.registry = new KeplerDatasetRegistry({memoryBudgetBytes: props.memoryBudgetBytes});
    this.observeDeviceLoss(props.device);
  }

  get device(): Device {
    return this.currentDevice;
  }

  get deviceLost(): boolean {
    return this.lost;
  }

  /** Invalidates publication immediately; loss itself settles the old device's GPU work. */
  handleDeviceLoss(): Promise<void> {
    if (this.lossCompletion) return this.lossCompletion;
    this.lost = true;
    this.resolveDeviceLoss();
    this.registry.handleDeviceLoss();
    // Bounded reads may reject on loss. Their own finally handlers finish before explorer cleanup.
    this.lossCompletion = Promise.allSettled(
      [...this.explorers].map(explorer => explorer.closeAfterDeviceLoss())
    ).then(() => undefined);
    return this.lossCompletion;
  }

  /**
   * Re-upload retained Arrow sources on an application-created replacement device. Dataset revisions
   * advance; old explorers stay closed and callers explicitly recreate their filters/layers.
   * A failed upload leaves its recovery source available to restoreArrowDatasets().
   */
  async recoverDevice(
    device: Device
  ): Promise<readonly KeplerDatasetSnapshot<KeplerArrowDataset>[]> {
    if (!this.lost || this.destroyed || device === this.currentDevice || device.isLost)
      throw new Error('Recovery requires a lost session and a replacement device');
    const lossCompletion = this.lossCompletion;
    await lossCompletion;
    if (this.destroyed || !this.lost || this.lossCompletion !== lossCompletion)
      throw new Error('Recovery was superseded');
    this.currentDevice = device;
    this.registry.recoverDevice();
    this.lost = false;
    this.lossCompletion = undefined;
    this.deviceLossSignal = new Promise<void>(resolve => {
      this.resolveDeviceLoss = resolve;
    });
    this.observeDeviceLoss(device);
    return this.restoreArrowDatasets();
  }

  /** Retry pending Arrow restores without renormalizing or double-counting retained CPU buffers. */
  restoreArrowDatasets(): readonly KeplerDatasetSnapshot<KeplerArrowDataset>[] {
    this.assertActive();
    const restored: KeplerDatasetSnapshot<KeplerArrowDataset>[] = [];
    for (const recovery of this.registry.getRecoverySources()) {
      if (recovery.source.kind !== 'arrow' || !recovery.source.metadata?.['keplerPreparedArrow'])
        continue;
      const resources = recovery.resources.filter(candidate => candidate.memoryKind === 'cpu');
      const preparedResourceId = recovery.source.metadata?.['keplerPreparedResourceId'];
      const resource = preparedResourceId
        ? resources.find(candidate => candidate.id === preparedResourceId)
        : resources[0];
      if (!resource) throw new Error('Arrow recovery requires its retained prepared source');
      // Session Arrow descriptors retain these exact adapter types; foreign sources need a host loader.
      const prepared = resource.value as PreparedArrowAnalyticsTable;
      const options: KeplerArrowDatasetOptions = {
        id: recovery.id,
        table: recovery.source.value as Table,
        normalization: recovery.source.metadata?.['normalization'] as
          | PrepareArrowTableForGPUAnalyticsOptions
          | undefined
      };
      restored.push(
        this.importPreparedArrowDataset(
          options,
          prepared,
          resources,
          recovery.population,
          recovery.source.metadata
        )
      );
    }
    return restored;
  }

  /** Explicit preparation precedes GPU capacity preflight; conversion never happens on interaction. */
  importArrowDataset(
    options: KeplerArrowDatasetOptions
  ): KeplerDatasetSnapshot<KeplerArrowDataset> {
    this.assertActive();
    if (!options.id) throw new Error('Dataset id required');
    const prepared = prepareArrowTableForGPUAnalytics(options.table, options.normalization);
    return this.importPreparedArrowDataset(
      options,
      prepared,
      undefined,
      options.population,
      options.metadata
    );
  }

  /**
   * Adopt completed, caller-owned scalar outputs without uploading their retained Arrow copy.
   * Request admission is atomic. Failure leaves GPU ownership with the caller; success transfers
   * it to this dataset generation. Both reservations are consumed/released during registration.
   */
  adoptArrowDataset(
    options: KeplerAdoptedArrowDatasetOptions
  ): KeplerDatasetSnapshot<KeplerArrowDataset> | null {
    this.assertActive();
    if (
      !options.cpuReservation.isActive ||
      options.cpuReservation.memoryKind !== 'cpu' ||
      !options.gpuReservation.isActive ||
      options.gpuReservation.memoryKind !== 'gpu'
    )
      throw new Error('Adoption requires active CPU and GPU reservations');
    if (
      this.registry.getDataset(options.id) ||
      this.registry.getRecoverySources().some(source => source.id === options.id)
    )
      throw new Error('Adopted result ID is already occupied');
    const analytics = makeAdoptedArrowAnalytics(this.device, options);
    let adopted = false;
    try {
      const retainedArrayBufferBytes = getRetainedArrowArrayBufferBytes(
        options.table,
        options.prepared
      );
      const retainedStringBytes = Object.values(options.prepared.dictionaries).reduce(
        (total, dictionary) =>
          total + dictionary.values.reduce((bytes, label) => bytes + label.length * 2, 0),
        0
      );
      const uploadedBufferBytes = getUploadedBufferBytes(analytics, {});
      const retainedBytes = retainedArrayBufferBytes + retainedStringBytes;
      if (
        retainedBytes > options.cpuReservation.byteLength ||
        uploadedBufferBytes > options.gpuReservation.byteLength
      )
        throw new Error('Adopted outputs exceed their memory reservations');
      const value: KeplerArrowDataset = {
        source: options.table,
        prepared: options.prepared,
        analytics,
        columns: getResidentColumns(analytics),
        timeColumns: {},
        retainedArrayBufferBytes,
        uploadedBufferBytes
      };
      const resourceId = `kepler-adopted-${++this.resourceSequence}`;
      const dataset = this.registry.publishResult(
        options.request,
        {
          id: options.id,
          value,
          batches: options.prepared.sourceBatches.map(batch => ({
            id: String(batch.sourceBatchIndex),
            rowCount: batch.sourceRowCount,
            sourceRowOffset: batch.sourceRowIndexOffset
          })),
          source: {
            kind: 'arrow',
            value: options.table,
            metadata: {
              ...options.metadata,
              normalization: options.normalization,
              keplerPreparedArrow: true,
              keplerPreparedResourceId: `${resourceId}-cpu`
            }
          },
          resources: [
            {
              id: `${resourceId}-cpu`,
              value: options.prepared,
              byteLength: retainedBytes,
              memoryKind: 'cpu',
              ownership: 'borrowed'
            },
            {
              id: `${resourceId}-gpu`,
              value: analytics,
              byteLength: uploadedBufferBytes,
              memoryKind: 'gpu',
              ownership: 'owned',
              destroy: () => {
                analytics.table.destroy();
                for (const vector of Object.values(analytics.validity)) vector?.destroy();
                options.destroy();
              }
            }
          ]
        },
        [options.cpuReservation, options.gpuReservation]
      );
      adopted = Boolean(dataset);
      return dataset;
    } finally {
      if (!adopted) {
        analytics.table.destroy();
        for (const vector of Object.values(analytics.validity)) vector?.destroy();
      }
    }
  }

  /**
   * Adopt GPU-resident scalar columns plus a schema without reading the columns back. Rows are
   * paged in on demand through `value.resident.ensureRows()` (tooltips, table pages), bounded by
   * the CPU reservation. Failure leaves GPU ownership with the caller; success transfers it.
   * Resident datasets are not device-recoverable: no CPU source exists to re-upload.
   */
  adoptResidentDataset(
    options: KeplerResidentDatasetOptions
  ): KeplerDatasetSnapshot<KeplerArrowDataset> | null {
    this.assertActive();
    if (
      !options.cpuReservation.isActive ||
      options.cpuReservation.memoryKind !== 'cpu' ||
      !options.gpuReservation.isActive ||
      options.gpuReservation.memoryKind !== 'gpu'
    )
      throw new Error('Adoption requires active CPU and GPU reservations');
    if (
      this.registry.getDataset(options.id) ||
      this.registry.getRecoverySources().some(source => source.id === options.id)
    )
      throw new Error('Adopted result ID is already occupied');
    const resident = new KeplerResidentRows(this.device, options);
    let analytics: KeplerGPUAnalytics | undefined;
    const timeColumns: Record<string, GPUTimeVectorFromPreparedArrowColumn> = Object.create(null);
    let adopted = false;
    try {
      const prepared = makeResidentPreparedTable(
        resident,
        options.schema,
        options.timeColumns ?? {}
      );
      // Time words upload beside the scalar columns; the dataset owns and destroys them.
      for (const [name, column] of Object.entries(prepared.timeColumns)) {
        if (!options.schema.some(entry => entry.name === name))
          throw new Error('Resident time columns must name schema columns');
        timeColumns[name] = makeGPUTimeVectorFromPreparedArrowColumn(this.device, column);
      }
      analytics = makeResidentAnalytics(this.device, options, resident, prepared);
      const retainedBytes =
        estimateKeplerResidentRowCacheByteLength(
          options.schema,
          prepared.sourceBatches.map(batch => batch.sourceRowCount)
        ) +
        Object.values(prepared.timeColumns).reduce(
          (total, column) =>
            total +
            column.chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0) +
            column.validity.reduce((sum, chunk) => sum + chunk.byteLength, 0),
          0
        );
      const uploadedBufferBytes = getUploadedBufferBytes(analytics, timeColumns);
      if (
        retainedBytes > options.cpuReservation.byteLength ||
        uploadedBufferBytes > options.gpuReservation.byteLength
      )
        throw new Error('Adopted outputs exceed their memory reservations');
      const value: KeplerArrowDataset = {
        source: resident.table,
        prepared,
        analytics,
        columns: getResidentColumns(analytics),
        timeColumns,
        retainedArrayBufferBytes: retainedBytes,
        uploadedBufferBytes,
        resident
      };
      const resourceId = `kepler-resident-${++this.resourceSequence}`;
      const gpuAnalytics = analytics;
      const dataset = this.registry.publishResult(
        options.request,
        {
          id: options.id,
          value,
          batches: prepared.sourceBatches.map(batch => ({
            id: String(batch.sourceBatchIndex),
            rowCount: batch.sourceRowCount,
            sourceRowOffset: batch.sourceRowIndexOffset
          })),
          resources: [
            {
              id: `${resourceId}-cpu`,
              value: resident,
              byteLength: retainedBytes,
              memoryKind: 'cpu',
              ownership: 'owned',
              destroy: () => resident.close()
            },
            {
              id: `${resourceId}-gpu`,
              value: gpuAnalytics,
              byteLength: uploadedBufferBytes,
              memoryKind: 'gpu',
              ownership: 'owned',
              destroy: () => {
                gpuAnalytics.table.destroy();
                for (const vector of Object.values(gpuAnalytics.validity)) vector?.destroy();
                for (const column of Object.values(timeColumns)) column.destroy();
                options.destroy();
              }
            }
          ]
        },
        [options.cpuReservation, options.gpuReservation]
      );
      adopted = Boolean(dataset);
      return dataset;
    } finally {
      if (!adopted) {
        resident.close();
        analytics?.table.destroy();
        for (const vector of Object.values(analytics?.validity ?? {})) vector?.destroy();
        for (const column of Object.values(timeColumns)) column.destroy();
      }
    }
  }

  /**
   * Reduce each numeric column's finite [minimum, maximum] on the GPU and store it on the
   * resident dataset. Only the 12-byte summary per column is read back. Dictionary columns are
   * skipped; resolves null if the dataset changed or was closed while reducing.
   */
  async readResidentDomains(
    snapshot: KeplerDatasetSnapshot<KeplerArrowDataset>,
    options: {workingMemoryBudgetBytes: number}
  ): Promise<Readonly<Record<string, readonly [number, number] | null>> | null> {
    this.assertActive();
    const resident = snapshot.value.resident;
    if (!resident) throw new Error('Dataset is not GPU resident');
    const columns = Object.keys(snapshot.value.columns).filter(
      name => !snapshot.value.prepared.dictionaries[name]
    );
    if (!columns.length) {
      resident.setDomains({});
      return resident.domains;
    }
    const explorer = this.openExplorer(snapshot.id, {
      dimensions: [],
      domains: columns.map(column => ({id: column, column, dimensions: []})),
      workingMemoryBudgetBytes: options.workingMemoryBudgetBytes,
      maximumReadbackBytes: Math.max(1024, columns.length * 12 + 4)
    });
    let fence: ReturnType<Device['createFence']> | undefined;
    try {
      const encoder = this.device.createCommandEncoder();
      explorer.encode(encoder);
      this.device.submit(encoder.finish());
      fence = this.device.createFence();
      await Promise.race([fence.signaled, this.device.lost]);
      const summary = await explorer.readSummary();
      if (!summary) return null;
      resident.setDomains(summary.domains);
      return resident.domains;
    } finally {
      try {
        await explorer.close(
          fence ? Promise.race([fence.signaled, this.device.lost]) : Promise.resolve()
        );
      } finally {
        fence?.destroy();
      }
    }
  }

  private importPreparedArrowDataset(
    options: KeplerArrowDatasetOptions,
    prepared: PreparedArrowAnalyticsTable,
    retainedResources?: readonly KeplerDatasetResource[],
    population?: KeplerDatasetDescriptor['population'],
    sourceMetadata?: Readonly<Record<string, unknown>>
  ): KeplerDatasetSnapshot<KeplerArrowDataset> {
    const retainedArrayBufferBytes = getRetainedArrowArrayBufferBytes(options.table, prepared);
    const expectedUploadBytes = getPreparedArrowUploadByteLength(prepared);
    const sourceReservation = this.registry.reserveMemory(
      retainedResources ? 0 : retainedArrayBufferBytes,
      'cpu',
      'source'
    );
    let uploadReservation: KeplerMemoryReservation | undefined;
    let analytics: KeplerGPUAnalytics | undefined;
    const timeColumns: Record<string, GPUTimeVectorFromPreparedArrowColumn> = Object.create(null);
    let adopted = false;
    try {
      uploadReservation = this.registry.reserveMemory(expectedUploadBytes, 'gpu', 'source');
      // Arrow drops logical row counts on zero-field prepared batches. Select no fields from
      // the retained original instead, preserving batch identities without a synthetic column.
      analytics = prepared.table.schema.fields.length
        ? makeGPUAnalyticsTableFromArrowTable(this.device, prepared.table)
        : makeGPUAnalyticsTableFromArrowTable(this.device, options.table, {columns: []});
      for (const [name, column] of Object.entries(prepared.timeColumns)) {
        timeColumns[name] = makeGPUTimeVectorFromPreparedArrowColumn(this.device, column);
      }
      const columns = getResidentColumns(analytics);
      const uploadedBufferBytes = getUploadedBufferBytes(analytics, timeColumns);
      const value: KeplerArrowDataset = {
        source: options.table,
        prepared,
        analytics,
        columns,
        timeColumns,
        retainedArrayBufferBytes,
        uploadedBufferBytes
      };
      const resourceId = `kepler-arrow-${++this.resourceSequence}`;
      // The registry consumes the reservations on success, so capacity is never unreserved before
      // registration. An upload larger than its estimate cannot satisfy the reservation check, so
      // that rare case releases early; registration is synchronous and consumes no other capacity.
      const reservations =
        uploadedBufferBytes <= expectedUploadBytes
          ? [sourceReservation, uploadReservation]
          : undefined;
      if (!reservations) {
        sourceReservation.release();
        uploadReservation.release();
      }
      const dataset = this.registry.registerDataset(
        {
          id: options.id,
          value,
          population,
          batches: prepared.sourceBatches.map(batch => ({
            id: String(batch.sourceBatchIndex),
            rowCount: batch.sourceRowCount,
            sourceRowOffset: batch.sourceRowIndexOffset
          })),
          source: {
            kind: 'arrow',
            value: options.table,
            metadata: {
              ...sourceMetadata,
              normalization: options.normalization,
              keplerPreparedArrow: true,
              keplerPreparedResourceId: retainedResources?.find(
                resource => resource.value === prepared
              )?.id
            }
          },
          resources: [
            ...(retainedResources ?? [
              {
                id: `${resourceId}-cpu`,
                value: prepared,
                byteLength: retainedArrayBufferBytes,
                memoryKind: 'cpu',
                ownership: 'borrowed' as const
              }
            ]),
            {
              id: `${resourceId}-gpu`,
              value: analytics,
              byteLength: uploadedBufferBytes,
              memoryKind: 'gpu',
              ownership: 'owned',
              destroy: () => destroyUploadedDataset(value)
            }
          ]
        },
        reservations
      );
      adopted = true;
      return dataset;
    } finally {
      sourceReservation.release();
      uploadReservation?.release();
      if (!adopted) {
        analytics?.table.destroy();
        for (const vector of Object.values(analytics?.validity ?? {})) vector?.destroy();
        for (const column of Object.values(timeColumns)) column.destroy();
      }
    }
  }

  openExplorer(id: string, options: KeplerSessionExplorerOptions): KeplerSessionExplorer {
    this.assertActive();
    const lease = this.registry.acquireDataset<KeplerArrowDataset>(id, {
      allowIncomplete: options.allowIncompletePopulation
    });
    let reservation: KeplerMemoryReservation | undefined;
    let explorer: GPUResidentExplorer | undefined;
    try {
      reservation = this.registry.reserveMemory(options.workingMemoryBudgetBytes, 'gpu', 'scratch');
      const {workingMemoryBudgetBytes, allowIncompletePopulation, ...explorerOptions} = options;
      explorer = new GPUResidentExplorer({
        ...explorerOptions,
        device: this.device,
        columns: lease.dataset.value.columns,
        sourceChunkLengths: lease.dataset.value.prepared.sourceBatches.map(
          batch => batch.sourceRowCount
        ),
        validity: lease.dataset.value.analytics.validity,
        datasetRevision: lease.dataset.revision
      });
      const reservedMemoryByteLength =
        explorer.accountedResidentByteLength + explorer.inspectionReservationByteLength;
      if (reservedMemoryByteLength > workingMemoryBudgetBytes) {
        throw new KeplerAllocationError('working-memory', {
          message: 'Explorer storage and inspection capacity exceed working memory reservation'
        });
      }
      // Compilation can allocate before reporting its plan, so admission retains the full cap
      // through construction. Retain only known resident storage plus bounded read capacity now.
      reservation.resize(reservedMemoryByteLength);
      const handle = new KeplerSessionExplorer({
        explorer,
        workingMemoryBudgetBytes,
        allowIncompletePopulation,
        lease,
        reservation,
        isCurrent: () => this.registry.getDataset(id)?.revision === lease.dataset.revision,
        onClose: closed => this.explorers.delete(closed)
      });
      this.explorers.add(handle);
      return handle;
    } catch (error) {
      explorer?.destroy();
      reservation?.release();
      void lease.release();
      throw error;
    }
  }

  /** A fence is required because the session does not own render queue submission. */
  async removeDataset(id: string, completion: PromiseLike<unknown>): Promise<boolean> {
    this.assertActive();
    const fence = Promise.race([completion, this.deviceLossSignal]);
    const lease = this.registry.getDataset(id)
      ? this.registry.acquireDataset(id, {allowIncomplete: true})
      : undefined;
    const removed = this.registry.removeDataset(id);
    await Promise.all([
      ...[...this.explorers]
        .filter(explorer => explorer.dataset.id === id)
        .map(explorer => explorer.close(fence)),
      lease?.release(fence)
    ]);
    return removed;
  }

  /** Release explorers only after the caller's last submission and all bounded reads complete. */
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destroyCompletion) {
      this.destroyed = true;
      this.destroyCompletion = Promise.allSettled(
        [...this.explorers].map(explorer => explorer.close(completion))
      ).then(async results => {
        // A caller may render uploaded columns directly without opening an explorer lease.
        // Keep those datasets alive until the same explicit submission fence settles.
        const fence = await Promise.allSettled([Promise.race([completion, this.deviceLossSignal])]);
        await this.registry.destroy();
        const failure = results.find(result => result.status === 'rejected');
        if (failure?.status === 'rejected') throw failure.reason;
        if (fence[0].status === 'rejected') throw fence[0].reason;
      });
    }
    return this.destroyCompletion;
  }

  private assertActive(): void {
    if (this.destroyed) throw new Error('Compute session destroyed');
    if (this.lost) throw new Error('Compute session device lost');
  }

  private observeDeviceLoss(device: Device): void {
    if (device.isLost) void this.handleDeviceLoss();
    void device.lost
      .then(
        () => {
          if (this.currentDevice === device) return this.handleDeviceLoss();
          return undefined;
        },
        () => {
          if (this.currentDevice === device) return this.handleDeviceLoss();
          return undefined;
        }
      )
      .catch(() => {});
  }
}

/** An explorer lease prevents replacement/deletion from releasing in-flight source buffers. */
export class KeplerSessionExplorer {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly selectionMask: GPUVector<'uint32'>;
  readonly workingMemoryBudgetBytes: number;
  readonly allowIncompletePopulation: boolean;
  private readonly explorer: GPUResidentExplorer;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly isCurrentDataset: () => boolean;
  private readonly onClose: (handle: KeplerSessionExplorer) => void;
  private completion?: Promise<void>;
  private resolveDeviceLoss!: () => void;
  private readonly deviceLossCompletion = new Promise<void>(resolve => {
    this.resolveDeviceLoss = resolve;
  });

  constructor(props: {
    explorer: GPUResidentExplorer;
    workingMemoryBudgetBytes?: number;
    allowIncompletePopulation?: boolean;
    lease: KeplerDatasetLease<KeplerArrowDataset>;
    reservation: KeplerMemoryReservation;
    isCurrent: () => boolean;
    onClose: (handle: KeplerSessionExplorer) => void;
  }) {
    this.explorer = props.explorer;
    this.allowIncompletePopulation = props.allowIncompletePopulation ?? false;
    this.lease = props.lease;
    this.dataset = props.lease.dataset;
    this.selectionMask = props.explorer.selectionMask;
    this.reservation = props.reservation;
    this.workingMemoryBudgetBytes = props.workingMemoryBudgetBytes ?? props.reservation.byteLength;
    this.isCurrentDataset = props.isCurrent;
    this.onClose = props.onClose;
  }

  get isCurrent(): boolean {
    return !this.completion && this.lease.isValid() && this.isCurrentDataset();
  }

  /** Charged resident plan plus bounded inspection capacity, excluding borrowed source/driver memory. */
  get reservedMemoryByteLength(): number {
    return this.reservation.isActive ? this.reservation.byteLength : 0;
  }

  get currentRevision(): number {
    return this.explorer.currentRevision;
  }

  get maximumInspectionPageSize(): number {
    return this.explorer.maximumInspectionPageSize;
  }

  get readbackByteLength(): number {
    return this.explorer.readbackByteLength;
  }

  get maximumInspectionReadbackBytes(): number {
    return this.explorer.maximumInspectionReadbackBytes;
  }

  readSelectedCount(): Promise<Pick<
    GPUResidentSummary,
    'revision' | 'datasetRevision' | 'selectedCount'
  > | null> {
    return this.readCurrent(() => this.explorer.readSelectedCount());
  }

  setRange(id: string, minimum: number, maximum: number): void {
    this.assertCurrent();
    this.explorer.setRange(id, minimum, maximum);
  }

  setBounds(id: string, bounds: readonly [number, number, number, number]): void {
    this.assertCurrent();
    this.explorer.setBounds(id, bounds);
  }

  setCategories(id: string, categories: readonly number[]): void {
    this.assertCurrent();
    this.explorer.setCategories(id, categories);
  }

  setExternalActive(id: string, active: boolean): void {
    this.assertCurrent();
    this.explorer.setExternalActive(id, active);
  }

  setEmpty(id: string): void {
    this.assertCurrent();
    this.explorer.setEmpty(id);
  }

  clear(id: string): void {
    this.assertCurrent();
    this.explorer.clear(id);
  }

  clearAll(): void {
    this.assertCurrent();
    this.explorer.clearAll();
  }

  invalidate(): void {
    this.assertCurrent();
    this.explorer.invalidate();
  }

  encode(commandEncoder: CommandEncoder): number {
    this.assertCurrent();
    return this.explorer.encode(commandEncoder);
  }

  discardEncoding(): void {
    this.assertCurrent();
    this.explorer.discardEncoding();
  }

  async readSummary(): Promise<GPUResidentSummary | null> {
    return this.readCurrent(() => this.explorer.readSummary());
  }

  async readPage(
    offset: number,
    limit: number,
    columns?: readonly string[]
  ): Promise<GPUResidentPage | null> {
    return this.readCurrent(() => this.explorer.readPage(offset, limit, columns));
  }

  async readRows(
    rows: readonly number[],
    columns?: readonly string[]
  ): Promise<GPUResidentPage | null> {
    return this.readCurrent(() => this.explorer.readRows(rows, columns));
  }

  close(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.completion) {
      this.completion = Promise.race([completion, this.deviceLossCompletion])
        .finally(async () => {
          try {
            await this.explorer.destroyAsync();
          } finally {
            await this.lease.release();
            this.reservation.release();
            this.onClose(this);
          }
        })
        .then(() => undefined);
    }
    return this.completion;
  }

  /** Device loss supersedes a completion fence that the lost backend can no longer settle. */
  closeAfterDeviceLoss(): Promise<void> {
    this.resolveDeviceLoss();
    return this.close(Promise.resolve());
  }

  private assertCurrent(): void {
    if (!this.isCurrent) throw new Error('Explorer dataset revision is stale or closed');
  }

  private async readCurrent<Result>(read: () => Promise<Result>): Promise<Result | null> {
    if (!this.isCurrent) return null;
    try {
      const result = await read();
      return this.isCurrent ? result : null;
    } catch (error) {
      // Lost/retired reads cannot publish; live read failures remain visible to the caller.
      if (!this.isCurrent) return null;
      throw error;
    }
  }
}

/** Count each retained backing allocation once, including sliced buffers and dictionaries. */
export function getRetainedArrowArrayBufferBytes(
  source: Table,
  prepared?: PreparedArrowAnalyticsTable
): number {
  const buffers = new Set<ArrayBufferLike>();
  const visited = new Set<Data>();
  function visit(data: Data): void {
    if (visited.has(data)) return;
    visited.add(data);
    for (const buffer of [data.values, data.valueOffsets, data.nullBitmap, data.typeIds])
      if (buffer) buffers.add(buffer.buffer);
    for (const child of data.children ?? []) visit(child);
    for (const dictionary of data.dictionary?.data ?? []) visit(dictionary);
  }
  for (const table of prepared ? [source, prepared.table] : [source])
    for (const batch of table.batches) visit(batch.data);
  for (const time of Object.values(prepared?.timeColumns ?? {})) {
    for (const values of [...time.chunks, ...time.validity]) buffers.add(values.buffer);
  }
  return [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
}

/** Portable scalar uploads are four bytes per row; time words are eight plus four for validity. */
export function getPreparedArrowUploadByteLength(prepared: PreparedArrowAnalyticsTable): number {
  let byteLength = 0;
  for (const batch of prepared.table.batches) {
    for (const field of prepared.table.schema.fields) {
      byteLength += Math.max(4, batch.numRows * 4) * (field.nullable ? 2 : 1);
    }
  }
  for (const time of Object.values(prepared.timeColumns)) {
    for (const [index, values] of time.chunks.entries()) {
      byteLength += Math.max(8, values.byteLength) + Math.max(8, time.validity[index].byteLength);
    }
  }
  return byteLength;
}

function getResidentColumns(analytics: KeplerGPUAnalytics): GPUResidentColumns {
  const columns: Record<string, GPUVector<'float32' | 'sint32' | 'uint32'>> = Object.create(null);
  for (const [name, vector] of Object.entries(analytics.table.gpuVectors)) {
    if (vector.format !== 'float32' && vector.format !== 'sint32' && vector.format !== 'uint32') {
      throw new Error('Analytics upload returned a non-scalar column');
    }
    columns[name] = vector as GPUVector<'float32' | 'sint32' | 'uint32'>;
  }
  return columns;
}

function getUploadedBufferBytes(
  analytics: KeplerGPUAnalytics,
  times: Readonly<Record<string, GPUTimeVectorFromPreparedArrowColumn>>
): number {
  const buffers = new Set<GPUVector['data'][number]['buffer']>();
  const vectors = [
    ...Object.values(analytics.table.gpuVectors),
    ...Object.values(analytics.validity)
  ];
  for (const time of Object.values(times)) vectors.push(time.timestamps, time.validity);
  for (const vector of vectors) for (const data of vector?.data ?? []) buffers.add(data.buffer);
  return [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
}

function destroyUploadedDataset(dataset: KeplerArrowDataset): void {
  dataset.analytics.table.destroy();
  for (const vector of Object.values(dataset.analytics.validity)) vector?.destroy();
  for (const column of Object.values(dataset.timeColumns)) column.destroy();
}
