// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder} from '@luma.gl/core';
import {
  GPUKeplerTableInspector,
  type KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerSessionExplorerOptions,
  type KeplerTableDerivedSnapshot,
  type GPUResidentSummary
} from '@luma-contributors/kepler';
import type {ResidentTableProvider} from './resident-data-table';
import {withOriginalSourceTable} from './resident-table-provider';
import {ResidentTableGPUSorter} from './resident-table-gpu-sorter';
import {ResidentTableWideStatistics} from './resident-table-wide-statistics';

/**
 * Attach an already-uploaded dataset without copying or renaming its columns. The host owns
 * registration/removal and frame submission; this consumer owns its explorer and inspection state.
 */
export class PublishedDatasetRuntime {
  readonly explorer: KeplerSessionExplorer;
  readonly table: ResidentTableProvider;
  private readonly inspector: GPUKeplerTableInspector;
  private readonly gpuSorter: ResidentTableGPUSorter;
  private readonly wideStatistics: ResidentTableWideStatistics;
  private readonly session: KeplerComputeSession;
  private readonly requestUpdate: () => void;
  private readonly destroyDerived?: (completion: PromiseLike<unknown>) => Promise<void>;
  private readonly listeners = new Set<() => void>();
  private dirty = true;
  private destroyed = false;
  private destruction?: Promise<void>;
  private summaryPending?: Promise<void>;
  private summaryValue: GPUResidentSummary | null = null;
  private needsSummary = true;

  constructor(props: {
    session: KeplerComputeSession;
    datasetId: string;
    explorer: KeplerSessionExplorerOptions;
    sortMemoryBudgetBytes: number;
    requestUpdate(): void;
    getDerivedColumns?(): KeplerTableDerivedSnapshot | undefined;
    destroyDerived?(completion: PromiseLike<unknown>): Promise<void>;
  }) {
    this.session = props.session;
    this.requestUpdate = props.requestUpdate;
    this.destroyDerived = props.destroyDerived;
    this.explorer = props.session.openExplorer(props.datasetId, props.explorer);
    const constructed: {destroy(): Promise<void>}[] = [];
    try {
      this.inspector = new GPUKeplerTableInspector({
        session: props.session,
        explorer: this.explorer,
        sortMemoryBudgetBytes: props.sortMemoryBudgetBytes,
        getDerivedColumns: props.getDerivedColumns
      });
      constructed.push(this.inspector);
      this.gpuSorter = new ResidentTableGPUSorter({
        session: props.session,
        explorer: this.explorer,
        memoryBudgetBytes: props.sortMemoryBudgetBytes
      });
      constructed.push(this.gpuSorter);
      this.wideStatistics = new ResidentTableWideStatistics({
        session: props.session,
        explorer: this.explorer,
        memoryBudgetBytes: props.sortMemoryBudgetBytes
      });
    } catch (error) {
      void Promise.allSettled(constructed.map(resource => resource.destroy())).then(() =>
        this.explorer.close(Promise.resolve())
      );
      throw error;
    }
    const dataset = this.explorer.dataset;
    const inspector = this.inspector;
    const preparedProvider: ResidentTableProvider = {
      datasetId: dataset.id,
      sourceRevision: dataset.revision,
      maximumPageSize: this.inspector.maximumPageSize,
      get fields() {
        const derivedColumns = props.getDerivedColumns?.()?.columns;
        return inspector.columns.map(name => {
          const dictionary = dataset.value.prepared.dictionaries[name];
          return {
            name,
            label:
              derivedColumns && Object.hasOwn(derivedColumns, name)
                ? `Analysis: ${name
                    .replace(/^analysis_/, '')
                    .replace(/_+$/, '')
                    .replace(/z_score/, 'z-score')
                    .replaceAll('_', ' ')}`
                : name,
            sortable: true,
            derived: Boolean(derivedColumns && Object.hasOwn(derivedColumns, name)),
            ...(dictionary
              ? {format: (value: number) => dictionary.values[value] ?? String(value)}
              : {})
          };
        });
      },
      getRevision: () =>
        `${this.explorer.dataset.revision}:${this.explorer.currentRevision}:${props.getDerivedColumns?.()?.revision ?? 'off'}:${this.destroyed}`,
      subscribe: listener => this.subscribe(listener),
      readPage: request => this.inspector.readPage(request),
      readRows: (sourceRows, columns) => this.inspector.readRows(sourceRows, columns)
    };
    this.table = withOriginalSourceTable(
      preparedProvider,
      dataset.value.source,
      () =>
        !this.destroyed &&
        this.explorer.isCurrent &&
        this.session.registry.getDataset(dataset.id)?.revision === dataset.revision,
      dataset.value.resident,
      {
        plan: column => inspector.planColumnStatistics(column),
        read: request => inspector.readColumnStatistics(request)
      },
      this.gpuSorter,
      this.wideStatistics
    );
  }

  get summary(): GPUResidentSummary | null {
    return this.explorer.isCurrent && this.summaryValue?.revision === this.explorer.currentRevision
      ? this.summaryValue
      : null;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  setRange(dimension: string, minimum: number, maximum: number): void {
    this.explorer.setRange(dimension, minimum, maximum);
    this.invalidate();
  }

  setCategories(dimension: string, categories: readonly number[]): void {
    this.explorer.setCategories(dimension, categories);
    this.invalidate();
  }

  setBounds(dimension: string, bounds: readonly [number, number, number, number]): void {
    this.explorer.setBounds(dimension, bounds);
    this.invalidate();
  }

  setExternalActive(dimension: string, active: boolean): void {
    this.explorer.setExternalActive(dimension, active);
    this.invalidate();
  }

  setEmpty(dimension: string): void {
    this.explorer.setEmpty(dimension);
    this.invalidate();
  }

  clear(dimension: string): void {
    this.explorer.clear(dimension);
    this.invalidate();
  }

  /** Append to the host's encoder before rendering consumers; never submit from here. */
  encode(encoder: CommandEncoder): void {
    if (this.destroyed || !this.dirty) return;
    this.explorer.encode(encoder);
    this.dirty = false;
    this.needsSummary = true;
  }

  discardEncoding(): void {
    this.explorer.discardEncoding();
    this.invalidate();
  }

  /** Call after submission. Only the compact configured summary returns to the CPU. */
  readSummary(): Promise<void> {
    if (this.destroyed || !this.explorer.isCurrent || !this.needsSummary) return Promise.resolve();
    if (this.summaryPending) return this.summaryPending;
    this.needsSummary = false;
    const pending = this.explorer
      .readSummary()
      .then(summary => {
        if (!this.destroyed && summary && summary.revision === this.explorer.currentRevision) {
          this.summaryValue = summary;
          this.notify();
        }
      })
      .finally(() => {
        this.summaryPending = undefined;
        if (this.needsSummary && !this.destroyed) this.requestUpdate();
      });
    this.summaryPending = pending;
    return pending;
  }

  /** Release this consumer before deleting its dataset or destroying its shared session. */
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.summaryValue = null;
    this.notify();
    this.listeners.clear();
    this.destruction = (async () => {
      const outcomes = await Promise.allSettled([
        this.inspector.destroy(),
        this.gpuSorter.destroy(),
        this.wideStatistics.destroy(),
        this.summaryPending
      ]);
      const fence = this.session.device.isLost ? undefined : this.session.device.createFence();
      const completion = fence
        ? Promise.race([fence.signaled, this.session.device.lost])
        : Promise.resolve();
      try {
        outcomes.push(
          ...(await Promise.allSettled([
            Promise.resolve().then(() => this.destroyDerived?.(completion))
          ]))
        );
        outcomes.push(...(await Promise.allSettled([this.explorer.close(completion)])));
      } finally {
        fence?.destroy();
      }
      const errors = outcomes
        .filter(outcome => outcome.status === 'rejected')
        .map(outcome => outcome.reason);
      if (errors.length) throw new AggregateError(errors, 'Published dataset cleanup failed');
    })();
    return this.destruction;
  }

  invalidate(): void {
    this.dirty = true;
    this.needsSummary = true;
    this.summaryValue = null;
    this.inspector.cancel();
    this.notify();
    this.requestUpdate();
  }

  /** Derived revisions invalidate pages without changing the source selection. */
  invalidateDerived(): void {
    if (this.destroyed) return;
    this.inspector.cancel();
    this.notify();
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }
}
