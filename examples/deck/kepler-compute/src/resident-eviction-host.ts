// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  ResidentEvictionPolicy,
  reserveMemoryWithEviction,
  type EvictableTable
} from './automatic-recovery-policy';

/** Structural subset of KeplerDatasetRegistry, so the host stays testable without a device. */
export type EvictionRegistry<Reservation> = {
  readonly memoryUsage: {readonly availableBytes: number};
  getDatasetFootprints(): readonly Readonly<{id: string; ownedByteLength: number}>[];
  setMemoryReserver(
    reserver:
      | ((
          byteLength: number,
          memoryKind: 'cpu' | 'gpu',
          reserve: (byteLength: number, memoryKind: 'cpu' | 'gpu') => Reservation
        ) => Reservation)
      | undefined
  ): void;
};

export type ResidentEvictionStatus = Readonly<{
  enabled: boolean;
  evictedIds: readonly string[];
  rebuildingIds: readonly string[];
  failedIds: readonly string[];
  evictionCount: number;
  rebuildCount: number;
  lastError?: string;
}>;

export type VisibilityInputs = Readonly<{
  layers: readonly Readonly<{isVisible?: boolean; dataId?: string | null}>[];
  /** Active filters (including chart and plot filters) target datasets through dataId arrays. */
  filters: readonly Readonly<{dataId?: readonly string[] | string | null}>[];
  /** Datasets that are never evictable: the base fixture and in-flight publications. */
  pinnedIds: Iterable<string>;
  /** Lineage of resident results: a visible result keeps its inputs resident (no evict/recompute thrash). */
  lineage?: readonly Readonly<{resultId: string; inputIds: readonly string[]}>[];
}>;

/** Datasets that a visible layer, an active filter/chart or a pin still references. */
export function collectVisibleDatasetIds(inputs: VisibilityInputs): Set<string> {
  const visible = new Set<string>(inputs.pinnedIds);
  for (const layer of inputs.layers) if (layer.isVisible && layer.dataId) visible.add(layer.dataId);
  for (const filter of inputs.filters)
    for (const id of [filter.dataId ?? []].flat()) if (id) visible.add(id);
  // Transitive closure over lineage; a pass per link is bounded by the lineage size.
  const lineage = inputs.lineage ?? [];
  for (let changed = lineage.length > 0; changed; ) {
    changed = false;
    for (const link of lineage)
      if (visible.has(link.resultId))
        for (const id of link.inputIds)
          if (!visible.has(id)) {
            visible.add(id);
            changed = true;
          }
  }
  return visible;
}

/**
 * Connects the pure eviction policy to a host. Nothing runs unless enabled: attach() installs no
 * reserver and requestResident() returns immediately, so disabled behavior is the plain registry.
 */
export class ResidentEvictionHost<Reservation> {
  private policy: ResidentEvictionPolicy;
  private registry?: EvictionRegistry<Reservation>;
  private readonly evicted = new Set<string>();
  private readonly rebuilding = new Set<string>();
  private readonly failed = new Map<string, string>();
  private readonly rebuildPromises = new Map<string, Promise<void>>();
  private evictionCount = 0;
  private rebuildCount = 0;
  private lastError?: string;
  private readonly listeners = new Set<() => void>();

  constructor(
    private readonly props: {
      enabled: boolean;
      getVisibleDatasetIds(): Set<string>;
      /** Whether a retained source can rebuild this dataset; others are never evicted. */
      canRebuild(id: string): boolean;
      /** Synchronously release the dataset's owned resources; async cleanup may follow. */
      evict(id: string): void;
      rebuild(id: string): Promise<void>;
    }
  ) {
    this.policy = new ResidentEvictionPolicy({enabled: props.enabled});
  }

  get isEnabled(): boolean {
    return this.props.enabled;
  }

  get status(): ResidentEvictionStatus {
    return {
      enabled: this.props.enabled,
      evictedIds: [...this.evicted],
      rebuildingIds: [...this.rebuilding],
      failedIds: [...this.failed.keys()],
      evictionCount: this.evictionCount,
      rebuildCount: this.rebuildCount,
      ...(this.lastError === undefined ? {} : {lastError: this.lastError})
    };
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Bind to a runtime's registry; a replacement runtime starts with a clean ledger. */
  attach(registry: EvictionRegistry<Reservation>): void {
    this.detach();
    this.policy = new ResidentEvictionPolicy({enabled: this.props.enabled});
    this.evicted.clear();
    this.rebuilding.clear();
    this.failed.clear();
    if (!this.props.enabled) return;
    this.registry = registry;
    registry.setMemoryReserver((byteLength, memoryKind, reserve) => {
      this.touchVisible();
      return reserveMemoryWithEviction({
        policy: this.policy,
        byteLength,
        getAvailableBytes: () => registry.memoryUsage.availableBytes,
        getTables: () => this.getTables(registry),
        reserve: requestedBytes => reserve(requestedBytes, memoryKind),
        evict: id => {
          this.props.evict(id);
          this.evicted.add(id);
          this.evictionCount++;
          this.emit();
        }
      });
    });
  }

  detach(): void {
    this.registry?.setMemoryReserver(undefined);
    this.registry = undefined;
  }

  touch(id: string): void {
    if (this.props.enabled) this.policy.touch(id);
  }

  touchVisible(): void {
    if (!this.props.enabled) return;
    for (const id of this.props.getVisibleDatasetIds()) this.policy.touch(id);
  }

  forget(id: string): void {
    this.policy.forget(id);
    this.failed.delete(id);
    this.rebuilding.delete(id);
    this.evicted.delete(id);
  }

  isEvicted(id: string): boolean {
    return this.props.enabled && this.policy.isEvicted(id);
  }

  /**
   * Explicit user eviction from the memory manager. Only datasets a retained source can rebuild
   * and that are not pinned; the dataset reloads on first use like a policy eviction.
   */
  evictDataset(id: string): boolean {
    if (!this.props.enabled || !this.props.canRebuild(id) || this.evicted.has(id)) return false;
    this.props.evict(id);
    this.policy.markEvicted(id);
    this.evicted.add(id);
    this.evictionCount++;
    this.emit();
    return true;
  }

  /** Called wherever a dataset is needed (layer binding, table, chart). Cheap when resident. */
  requestResident(id: string): void {
    if (!this.isEvicted(id) || this.rebuilding.has(id) || this.failed.has(id)) return;
    this.rebuilding.add(id);
    this.emit();
    const rebuild = this.policy
      .ensureResident(id, this.props.rebuild)
      .then(() => {
        this.rebuildCount++;
        this.evicted.delete(id);
      })
      .catch(error => {
        this.lastError = error instanceof Error ? error.message : String(error);
        this.failed.set(id, this.lastError);
      })
      .finally(() => {
        this.rebuilding.delete(id);
        this.rebuildPromises.delete(id);
        this.emit();
      });
    this.rebuildPromises.set(id, rebuild);
  }

  /** Resolves once the dataset is resident again; rejects when its rebuild failed. */
  async ensureResident(id: string): Promise<void> {
    if (!this.isEvicted(id)) return;
    this.requestResident(id);
    await this.rebuildPromises.get(id);
    const failure = this.failed.get(id);
    // A derived result needs its inputs; a failed input rebuild is reported by that input.
    if (failure !== undefined) throw new Error(`Rebuilding ${id} failed: ${failure}`);
  }

  /** Explicit user retry after a failed rebuild. */
  retryFailed(): void {
    const ids = [...this.failed.keys()];
    this.failed.clear();
    this.lastError = undefined;
    for (const id of ids) this.requestResident(id);
    this.emit();
  }

  private getTables(registry: EvictionRegistry<Reservation>): EvictableTable[] {
    const visible = this.props.getVisibleDatasetIds();
    return registry.getDatasetFootprints().map(({id, ownedByteLength}) => ({
      id,
      ownedByteLength: this.props.canRebuild(id) ? ownedByteLength : 0,
      isVisible: visible.has(id)
    }));
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}
