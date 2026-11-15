// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {findPointFieldPairs, type KeplerTable} from '@kepler.gl/table';
import type {Layer as KeplerLayer} from '@kepler.gl/layers';
import {makeArrowSourceFromNativeDataset} from './native-dataset-source';
import {
  inferLayerRolesFromFields,
  makeAutoPreparePlan,
  type AutoPreparePlan,
  type LayerRoleInput
} from './auto-prepare-plan';
import {makeLayerInputsFromLinkedRoles, takeLinkedRoleHint} from './share-link-role-hints';
import type {NativeDatasetConfig} from './native-config-capture';
import {describeWideNumericNarrowingOnGpu} from './wide-numeric-gpu-summary';
import type {SourceCandidate} from './source-workspace-panel';

/** Rows and cells admitted to automatic CPU materialization of a native table. */
export const AUTO_PREPARE_LIMITS = Object.freeze({maximumRows: 1000000, maximumCells: 16000000});

export type AutoPrepareState = 'queued' | 'preparing' | 'ready' | 'failed';

/** What the user is told about one automatically prepared dataset. */
export type AutoPrepareReport = {
  datasetId: string;
  label: string;
  state: AutoPrepareState;
  rowCount: number;
  roleSummary: readonly string[];
  policySummary: readonly string[];
  excluded: readonly {name: string; reason: string}[];
  warnings: readonly string[];
  columnNotes: readonly string[];
  error?: string;
};

export type AutoPublishRequest = {
  datasetId: string;
  candidate: SourceCandidate;
  plan: AutoPreparePlan;
  nativeConfig: NativeDatasetConfig | undefined;
};

export type NativeAutoPrepareHost = {
  getDatasets(): Readonly<Record<string, KeplerTable>> | undefined;
  getLayers(): readonly KeplerLayer[];
  isResidentDataset(datasetId: string): boolean;
  /** False while the runtime is absent, recovering, registering layers or publishing. */
  isReady(): boolean;
  captureNativeConfig(datasetId: string): NativeDatasetConfig | undefined;
  publish(request: AutoPublishRequest): Promise<void>;
  /** Resident GPU device, when available: wide numeric narrowing is then scanned there over every row. */
  getDevice?(): Device | undefined;
  /** Layer IDs from the top of the native layer stack to the bottom. */
  getLayerOrder?(): readonly string[];
  /** Restores the stack after replaced datasets re-created their layers; unknown IDs go last. */
  setLayerOrder?(layerIds: readonly string[]): void;
};

/** Native Kepler skips rows whose coordinates are null or out of range; count them for the report. */
function countInvalidCoordinateRows(plan: AutoPreparePlan): number {
  const roles = plan.coordinates;
  if (!roles) return 0;
  const longitudes = plan.table.getChild(roles.longitude);
  const latitudes = plan.table.getChild(roles.latitude);
  if (!longitudes || !latitudes) return 0;
  let count = 0;
  for (let row = 0; row < plan.table.numRows; row++) {
    const longitude = Number(longitudes.get(row));
    const latitude = Number(latitudes.get(row));
    if (
      longitudes.get(row) === null ||
      latitudes.get(row) === null ||
      !(longitude >= -180 && longitude <= 180 && latitude >= -90 && latitude <= 90)
    )
      count++;
  }
  return count;
}

/** Keeps layers that cannot bind the prepared geometry column in the saved config, but hidden. */
function hideUnpreparedLayers(
  config: NativeDatasetConfig | undefined,
  layerIds: readonly string[]
): NativeDatasetConfig | undefined {
  if (!config || !layerIds.length) return config;
  const visState = config.config.config.visState;
  return {
    ...config,
    config: {
      ...config.config,
      config: {
        ...config.config.config,
        visState: {
          ...visState,
          layers: visState.layers.map(layer => {
            const saved = layer as {id?: string; config?: Record<string, unknown>};
            return saved.id && layerIds.includes(saved.id)
              ? {...saved, config: {...saved.config, isVisible: false}}
              : layer;
          })
        }
      }
    }
  };
}

function readLayerRoles(layer: KeplerLayer): LayerRoleInput {
  const columns: Record<string, string | null | undefined> = {};
  for (const [role, column] of Object.entries(
    (layer.config.columns ?? {}) as Record<string, {value?: string | null} | undefined>
  ))
    columns[role] = column?.value;
  return {
    id: layer.id,
    type: layer.type ?? '',
    isVisible: layer.config.isVisible,
    columnMode: (layer.config as {columnMode?: string}).columnMode,
    columns
  };
}

/**
 * Prepares every dataset that arrives through native Kepler (Load Files, saved maps, URLs) as a
 * resident dataset under the same ID, keeping its native layers, filters and tooltips. Datasets
 * are processed one at a time; failures leave the native dataset untouched and are reported.
 */
export class NativeAutoPrepare {
  readonly reports = new Map<string, AutoPrepareReport>();
  private readonly seen = new Map<string, number>();
  private readonly queue: string[] = [];
  private running = false;
  private layerOrderBeforeBatch: readonly string[] | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners = new Set<() => void>();

  constructor(
    private readonly host: NativeAutoPrepareHost,
    readonly enabled: boolean
  ) {}

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** True while a native dataset is waiting for, or undergoing, automatic preparation. */
  isPending(datasetId: string): boolean {
    const state = this.reports.get(datasetId)?.state;
    return state === 'queued' || state === 'preparing';
  }

  getFailure(datasetId: string): string | undefined {
    const report = this.reports.get(datasetId);
    return report?.state === 'failed' ? report.error : undefined;
  }

  private readonly forced = new Set<string>();
  private readonly signals = new Map<string, AbortSignal>();
  private readonly requestGenerations = new Map<string, number>();

  /** Explicit sample reloads must await the queued GPU replacement, including preparation failure. */
  republish(datasetId: string, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    if (!this.enabled) return Promise.reject(new Error('Automatic preparation is disabled.'));
    if (!this.host.getDatasets()?.[datasetId])
      return Promise.reject(new Error(`Native dataset ${datasetId} is unavailable.`));
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        unsubscribe();
        signal.removeEventListener('abort', abort);
      };
      const abort = () => {
        cleanup();
        if (this.signals.get(datasetId) === signal) {
          const queuedIndex = this.queue.indexOf(datasetId);
          if (queuedIndex >= 0) {
            this.queue.splice(queuedIndex, 1);
            this.forced.delete(datasetId);
            this.signals.delete(datasetId);
            this.update(datasetId, {state: 'failed', error: 'Sample preparation was cancelled.'});
            if (!this.queue.length && this.retry) {
              clearTimeout(this.retry);
              this.retry = null;
            }
          }
        }
        reject(signal.reason);
      };
      const unsubscribe = this.subscribe(() => {
        const report = this.reports.get(datasetId);
        if (report?.state === 'ready') {
          cleanup();
          resolve();
        } else if (report?.state === 'failed') {
          cleanup();
          reject(new Error(report.error));
        }
      });
      signal.addEventListener('abort', abort, {once: true});
      this.signals.set(datasetId, signal);
      this.requestRepublish(datasetId);
    });
  }

  /**
   * Re-prepares an already resident dataset from its (edited) Kepler dataset under the same id,
   * for example after a native column rename or type change.
   */
  requestRepublish(datasetId: string): void {
    const dataset = this.host.getDatasets()?.[datasetId];
    if (!this.enabled || !dataset) return;
    const alreadyQueued = this.queue.includes(datasetId);
    this.requestGenerations.set(datasetId, (this.requestGenerations.get(datasetId) ?? 0) + 1);
    this.forced.add(datasetId);
    this.seen.set(datasetId, dataset.dataRevision);
    this.reports.set(datasetId, {
      datasetId,
      label: dataset.label,
      state: 'queued',
      rowCount: dataset.dataContainer?.numRows() ?? 0,
      roleSummary: [],
      policySummary: [],
      excluded: [],
      warnings: [],
      columnNotes: []
    });
    if (alreadyQueued) {
      this.notify();
      return;
    }
    if (!this.queue.length && !this.running)
      this.layerOrderBeforeBatch = this.host.getLayerOrder?.() ?? null;
    this.queue.push(datasetId);
    this.notify();
    void this.drain();
  }

  /** Discover new native datasets; cheap enough to call on every store update. */
  evaluate(): void {
    if (!this.enabled) return;
    const datasets = this.host.getDatasets();
    if (!datasets) return;
    for (const [datasetId, dataset] of Object.entries(datasets)) {
      if (this.host.isResidentDataset(datasetId)) {
        this.seen.delete(datasetId);
        continue;
      }
      if (this.seen.get(datasetId) === dataset.dataRevision) continue;
      const metadata = dataset.metadata as Record<string, unknown> | undefined;
      if (
        (dataset as {type?: string}).type ||
        metadata?.['derivedDataset'] ||
        !dataset.dataContainer ||
        dataset.dataContainer.numRows() === 0
      ) {
        this.seen.set(datasetId, dataset.dataRevision);
        continue;
      }
      this.seen.set(datasetId, dataset.dataRevision);
      this.reports.set(datasetId, {
        datasetId,
        label: dataset.label,
        state: 'queued',
        rowCount: dataset.dataContainer.numRows(),
        roleSummary: [],
        policySummary: [],
        excluded: [],
        warnings: [],
        columnNotes: []
      });
      if (!this.queue.includes(datasetId)) {
        if (!this.queue.length && !this.running)
          this.layerOrderBeforeBatch = this.host.getLayerOrder?.() ?? null;
        this.queue.push(datasetId);
      }
      this.notify();
    }
    void this.drain();
  }

  private version = 0;

  getVersion(): number {
    return this.version;
  }

  private notify(): void {
    this.version++;
    for (const listener of this.listeners) listener();
  }

  private scheduleRetry(): void {
    if (this.retry) return;
    this.retry = setTimeout(() => {
      this.retry = null;
      void this.drain();
    }, 250);
  }

  private async drain(): Promise<void> {
    if (this.running || !this.queue.length) return;
    if (!this.host.isReady()) {
      this.scheduleRetry();
      return;
    }
    this.running = true;
    try {
      while (this.queue.length) {
        if (!this.host.isReady()) {
          this.scheduleRetry();
          return;
        }
        const datasetId = this.queue.shift()!;
        await this.prepareDataset(datasetId);
      }
    } finally {
      this.running = false;
      if (!this.queue.length && this.layerOrderBeforeBatch) {
        this.host.setLayerOrder?.(this.layerOrderBeforeBatch);
        this.layerOrderBeforeBatch = null;
      }
    }
  }

  private update(datasetId: string, patch: Partial<AutoPrepareReport>): void {
    const report = this.reports.get(datasetId);
    if (report) this.reports.set(datasetId, {...report, ...patch});
    this.notify();
  }

  private async prepareDataset(datasetId: string): Promise<void> {
    const generation = this.requestGenerations.get(datasetId);
    const signal = this.signals.get(datasetId);
    const update = (patch: Partial<AutoPrepareReport>) => {
      if (this.requestGenerations.get(datasetId) === generation) this.update(datasetId, patch);
    };
    const dataset = this.host.getDatasets()?.[datasetId];
    const forced = this.forced.delete(datasetId);
    if (!dataset || (this.host.isResidentDataset(datasetId) && !forced)) {
      this.reports.delete(datasetId);
      this.notify();
      return;
    }
    update({state: 'preparing', label: dataset.label});
    try {
      signal?.throwIfAborted();
      const source = makeArrowSourceFromNativeDataset(dataset, AUTO_PREPARE_LIMITS);
      const nativeLayers = this.host
        .getLayers()
        .filter(layer => layer.config.dataId === datasetId)
        .map(readLayerRoles);
      const layerInputs = nativeLayers.length
        ? nativeLayers
        : inferLayerRolesFromFields(
            dataset.fields.map(field => ({name: field.name, type: field.type})),
            fields => findPointFieldPairs(fields as never) as never
          );
      // Share-link roles override guessed layer roles for this file (applied on load).
      const linkedRoles = takeLinkedRoleHint(dataset.label);
      const linked = linkedRoles
        ? makeLayerInputsFromLinkedRoles(
            linkedRoles,
            source.table.schema.fields.map(field => field.name)
          )
        : undefined;
      const plan = makeAutoPreparePlan(source.table, [...(linked?.layers ?? []), ...layerInputs]);
      const device = this.host.getDevice?.();
      if (device) {
        // GPU scan over every row replaces the CPU narrowing lines; any failure keeps the CPU lines.
        try {
          const gpuLines = await describeWideNumericNarrowingOnGpu(device, source.table);
          if (gpuLines.length)
            plan.policySummary = [
              ...plan.policySummary.filter(line => !line.startsWith('Narrowing ')),
              ...gpuLines
            ];
        } catch {
          // Informational only.
        }
      }
      if (linked?.notes.length) plan.warnings = [...plan.warnings, ...linked.notes];
      const invalidCoordinateRows = countInvalidCoordinateRows(plan);
      if (invalidCoordinateRows)
        plan.warnings = [
          ...plan.warnings,
          `${invalidCoordinateRows} of ${plan.table.numRows} rows have null or invalid longitude/latitude; they stay in the table and are skipped on the map.`
        ];
      const columnNotes = Object.entries(source.columnRoutes).flatMap(([name, route]) =>
        route.note ? [`${name}: ${route.note}`] : []
      );
      update({
        roleSummary: plan.roleSummary,
        policySummary: plan.policySummary,
        excluded: plan.excluded,
        warnings: plan.warnings,
        columnNotes
      });
      const candidate: SourceCandidate = {
        table: plan.table,
        label: dataset.label,
        isCurrent: () =>
          !signal?.aborted &&
          this.host.getDatasets()?.[datasetId] === dataset &&
          source.isCurrent(),
        ...(plan.cellIndex ? {cellIndex: plan.cellIndex} : {})
      };
      signal?.throwIfAborted();
      await this.host.publish({
        datasetId,
        candidate,
        plan,
        nativeConfig: hideUnpreparedLayers(
          this.host.captureNativeConfig(datasetId),
          plan.unpreparedLayerIds
        )
      });
      update({state: 'ready'});
    } catch (failure) {
      update({
        state: 'failed',
        error: failure instanceof Error ? failure.message : String(failure)
      });
    } finally {
      if (this.signals.get(datasetId) === signal) this.signals.delete(datasetId);
    }
  }
}
