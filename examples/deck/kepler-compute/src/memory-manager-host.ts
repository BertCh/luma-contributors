// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  buildMemoryManagerModel,
  type MemoryManagerLayer,
  type MemoryManagerOperation,
  type MemoryManagerModel,
  type MemoryManagerRegistry
} from './memory-manager-model';
import type {MemoryManagerSource} from './memory-manager-panel';
import {measureLayerRenderBuffers, type MeasurableDeckLayer} from './memory-layer-buffers';
import {readStoredMemoryBudget, writeStoredMemoryBudget} from './memory-budget-persistence';
import {MemoryNotifier, type MemoryNotice} from './memory-notifier';
import {listTrackedOperations, subscribeOperations} from './operation-tracker';
import type {SandboxHostSettingsProvider} from './sandbox-host-settings';

export type MemoryManagerRegistryHandle = MemoryManagerRegistry & {
  getActiveRequests(): readonly Readonly<{
    id: number;
    slot: string;
    inputs: readonly Readonly<{datasetId: string}>[];
    cancel(): void;
  }>[];
  subscribeMemory(listener: () => void): () => void;
  setMemoryBudget(memoryBudgetBytes: number): void;
};

export type MemoryManagerHostProps = {
  getRegistry(): MemoryManagerRegistryHandle | undefined;
  getLayers(): readonly MemoryManagerLayer[];
  getDatasetLabels(): Readonly<Record<string, string>>;
  getPinnedIds(): ReadonlySet<string>;
  getEvictedIds(): readonly string[];
  isEvictionEnabled: boolean;
  canEvict(datasetId: string): boolean;
  evictDataset(datasetId: string): boolean;
  unloadDataset(datasetId: string): void;
  emitNotice(notice: MemoryNotice): void;
  openManager(): void;
  /** Live deck.gl layers, used to measure the buffers each layer owns. */
  getDeckLayers?(): readonly MeasurableDeckLayer[];
  /** Overrides localStorage (tests). */
  storage?: Parameters<typeof readStoredMemoryBudget>[0];
};

export type MemoryManagerHost = {
  readonly source: MemoryManagerSource;
  readonly notifier: MemoryNotifier;
  /** Rebind after the runtime (and its registry) is replaced, for example after device loss. */
  bindRegistry(): void;
  /** Host state (layers, eviction, datasets) changed. */
  refresh(): void;
  /** Maps a host error to a memory notice; returns whether it was a budget refusal. */
  reportError(error: unknown): boolean;
  /** User-chosen budget in bytes, restored from the bundle or localStorage; undefined = default. */
  readonly preferredBudgetBytes: number | undefined;
  /** Saves with the workspace bundle and applies on restore. */
  readonly settings: SandboxHostSettingsProvider;
};

const JOIN_SLOT_PREFIXES = ['resident-join:', 'resident-spatial-join:'];

export function getOperationKind(label: string): MemoryManagerOperation['kind'] {
  if (/^(spatial )?join\b/i.test(label)) return 'join';
  if (/^chart\b/i.test(label)) return 'chart';
  if (/^analysis\b/i.test(label)) return 'analysis';
  if (/(download|import|export|stream)/i.test(label)) return 'transfer';
  return 'other';
}

/** Host operations plus registry requests; joins already tracked by the host hide their duplicate request. */
export function listMemoryOperations(
  registry: Pick<MemoryManagerRegistryHandle, 'getActiveRequests'> | undefined
): MemoryManagerOperation[] {
  const tracked = listTrackedOperations().map(
    (operation): MemoryManagerOperation => ({
      key: `operation-${operation.id}`,
      kind: getOperationKind(operation.label),
      label: operation.label,
      ...(operation.detail === undefined ? {} : {detail: operation.detail}),
      ...(operation.fraction === undefined ? {} : {fraction: operation.fraction}),
      cancel: operation.cancel
    })
  );
  const hasTrackedJoin = tracked.some(operation => operation.kind === 'join');
  const requests = (registry?.getActiveRequests() ?? [])
    .filter(
      request =>
        !(hasTrackedJoin && JOIN_SLOT_PREFIXES.some(prefix => request.slot.startsWith(prefix)))
    )
    .map(
      (request): MemoryManagerOperation => ({
        key: `request-${request.id}`,
        kind: 'request',
        label: request.slot,
        detail: `inputs: ${request.inputs.map(input => input.datasetId).join(', ') || 'none'}`,
        cancel: () => request.cancel()
      })
    );
  return [...tracked, ...requests];
}

const BUDGET_REFUSAL_MESSAGE = 'Dataset memory budget exceeded';

/**
 * Connects the session registry to the native memory manager: model building, budget changes,
 * unload and evict actions, and Kepler notifications. No GPU work happens here.
 */
export function createMemoryManagerHost(props: MemoryManagerHostProps): MemoryManagerHost {
  const listeners = new Set<() => void>();
  let stopRegistry: (() => void) | undefined;
  let boundRegistry: MemoryManagerRegistryHandle | undefined;
  let preferredBudgetBytes = readStoredMemoryBudget(props.storage);
  /** Applies the remembered budget to a fresh registry; a budget below current usage is kept for later. */
  const applyPreferredBudget = (): string | null => {
    const registry = props.getRegistry();
    if (!registry || preferredBudgetBytes === undefined) return null;
    if (registry.memoryBudgetBytes === preferredBudgetBytes) return null;
    try {
      registry.setMemoryBudget(preferredBudgetBytes);
      return null;
    } catch {
      return `The saved ${Math.ceil(preferredBudgetBytes / 1024 ** 2)} MiB budget is below the ${Math.ceil(registry.memoryUsage.totalBytes / 1024 ** 2)} MiB in use; unload or evict datasets, then apply it again.`;
    }
  };
  const notifier = new MemoryNotifier({
    getUsage() {
      const registry = props.getRegistry();
      return {
        totalBytes: registry?.memoryUsage.totalBytes ?? 0,
        budgetBytes: registry?.memoryBudgetBytes ?? 0
      };
    },
    // Hysteresis/accounting stay immediate; a pressure crossing during layer preparation
    // must not dispatch into Kepler's Redux store until the current render has returned.
    emit: notice => queueMicrotask(() => props.emitNotice(notice))
  });
  const reportBudgetFailure = (message: string | null) => {
    if (message) props.emitNotice({key: 'memory-budget-not-applied', type: 'warning', message});
  };
  let refreshPending = false;
  const refresh = () => {
    if (refreshPending) return;
    refreshPending = true;
    // Render-time layer preparation can reserve memory. Notify React after that render
    // completes, and coalesce its reservations into one panel update.
    queueMicrotask(() => {
      refreshPending = false;
      for (const listener of listeners) listener();
    });
  };
  // The host lives for the page; operation changes re-render the manager.
  subscribeOperations(refresh);
  const source: MemoryManagerSource = {
    isEvictionEnabled: props.isEvictionEnabled,
    openManager: props.openManager,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getModel(): MemoryManagerModel | null {
      const registry = props.getRegistry();
      if (!registry) return null;
      return buildMemoryManagerModel({
        registry,
        layers: props.getLayers(),
        datasetLabels: props.getDatasetLabels(),
        pinnedIds: props.getPinnedIds(),
        evictedIds: props.getEvictedIds(),
        canEvict: props.canEvict,
        isEvictionEnabled: props.isEvictionEnabled,
        operations: listMemoryOperations(registry),
        layerRenderBuffers: measureLayerRenderBuffers(
          props.getDeckLayers?.() ?? [],
          props.getLayers().map(layer => layer.id)
        )
      });
    },
    unloadDataset: props.unloadDataset,
    evictDataset(datasetId) {
      const registry = props.getRegistry();
      const before = registry?.getDatasetMemory().find(row => row.id === datasetId);
      if (!props.evictDataset(datasetId) || !before) return;
      notifier.reportEviction(props.getDatasetLabels()[datasetId] ?? datasetId, before.ownedBytes);
      refresh();
    },
    setBudgetMebibytes(mebibytes) {
      const registry = props.getRegistry();
      if (!registry) return 'No active workspace.';
      try {
        registry.setMemoryBudget(mebibytes * 1024 * 1024);
        preferredBudgetBytes = mebibytes * 1024 * 1024;
        writeStoredMemoryBudget(preferredBudgetBytes, props.storage);
        return null;
      } catch {
        return `A ${mebibytes} MiB budget is below the ${Math.ceil(registry.memoryUsage.totalBytes / 1024 ** 2)} MiB in use; unload or evict datasets first.`;
      }
    }
  };
  return {
    source,
    notifier,
    get preferredBudgetBytes() {
      return preferredBudgetBytes;
    },
    settings: {
      get() {
        return preferredBudgetBytes === undefined
          ? undefined
          : {memoryBudgetBytes: preferredBudgetBytes};
      },
      apply(settings) {
        if (settings.memoryBudgetBytes === undefined) return;
        preferredBudgetBytes = settings.memoryBudgetBytes;
        writeStoredMemoryBudget(preferredBudgetBytes, props.storage);
        reportBudgetFailure(applyPreferredBudget());
        refresh();
      }
    },
    bindRegistry() {
      stopRegistry?.();
      boundRegistry = props.getRegistry();
      reportBudgetFailure(applyPreferredBudget());
      stopRegistry = boundRegistry?.subscribeMemory(() => {
        notifier.check();
        refresh();
      });
      notifier.check();
      refresh();
    },
    refresh() {
      notifier.check();
      refresh();
    },
    reportError(error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.includes(BUDGET_REFUSAL_MESSAGE)) return false;
      notifier.reportRefusal();
      return true;
    }
  };
}
