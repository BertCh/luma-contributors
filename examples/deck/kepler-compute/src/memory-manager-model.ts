// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Structural subset of KeplerDatasetRegistry so the model is testable without a device. */
export type MemoryManagerRegistry = {
  readonly memoryBudgetBytes: number;
  readonly memoryUsage: Readonly<{
    allocatedBytes: Readonly<{cpu: number; gpu: number}>;
    reservedBytes: Readonly<{cpu: number; gpu: number}>;
    totalBytes: number;
    availableBytes: number;
  }>;
  getReservationBreakdown(): Readonly<{
    source: number;
    output: number;
    scratch: number;
    readback: number;
  }>;
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
  }>[];
};

export type MemoryManagerLayer = Readonly<{
  id: string;
  label: string;
  dataId: string | null;
  isVisible: boolean;
}>;

export type MemoryManagerInputs = {
  registry: MemoryManagerRegistry;
  layers: readonly MemoryManagerLayer[];
  /** Display label per dataset id; ids without a label are shown as-is. */
  datasetLabels: Readonly<Record<string, string>>;
  /** Datasets that can never be unloaded or evicted (base fixture, in-flight publications). */
  pinnedIds: ReadonlySet<string>;
  evictedIds: readonly string[];
  /** Whether the eviction policy is on and a retained source can rebuild the dataset. */
  canEvict(datasetId: string): boolean;
  isEvictionEnabled: boolean;
  /** Running host operations and registry requests. */
  operations?: readonly MemoryManagerOperation[];
  /** Measured layer-owned render buffers by layer id; layers without an entry are unmeasured. */
  layerRenderBuffers?: ReadonlyMap<string, Readonly<{bytes: number; bufferCount: number}>>;
};

/** A tracked long-running operation shown with progress and a cancel action. */
export type MemoryManagerOperation = Readonly<{
  key: string;
  kind: 'analysis' | 'chart' | 'join' | 'request' | 'transfer' | 'other';
  label: string;
  detail?: string;
  /** Fraction in [0, 1]; undefined while the total is unknown. */
  fraction?: number;
  cancel(): void;
}>;

export type MemoryManagerDatasetRow = Readonly<{
  id: string;
  label: string;
  rowCount: number;
  cpuBytes: number;
  gpuBytes: number;
  ownedBytes: number;
  borrowedBytes: number;
  /** Bytes released by unloading the dataset now (owned and not referenced elsewhere). */
  releasableBytes: number;
  layerIds: readonly string[];
  isVisible: boolean;
  isPinned: boolean;
  isEvicted: boolean;
  canUnload: boolean;
  canEvict: boolean;
}>;

export type MemoryManagerLayerRow = Readonly<{
  id: string;
  label: string;
  dataId: string | null;
  isVisible: boolean;
  /** Full registry bytes of the dataset the layer draws; shared datasets are not divided. */
  datasetBytes: number;
  /** Measured bytes of GPU buffers the layer owns (deck.gl attributes it allocated itself). */
  renderBufferBytes: number;
  renderBufferCount: number;
  /** False when the layer has no live deck.gl layer to measure (hidden, not yet rendered). */
  isMeasured: boolean;
  sharedWithLayers: number;
}>;

export type MemoryPressure = 'ok' | 'warning' | 'critical';

export type MemoryManagerModel = Readonly<{
  budgetBytes: number;
  usedBytes: number;
  availableBytes: number;
  usedFraction: number;
  pressure: MemoryPressure;
  residentCpuBytes: number;
  residentGpuBytes: number;
  reservations: Readonly<Record<'source' | 'output' | 'scratch' | 'readback', number>>;
  /** Registered dataset bytes plus nothing else: bytes not attributable to any dataset. */
  unattributedBytes: number;
  datasets: readonly MemoryManagerDatasetRow[];
  layers: readonly MemoryManagerLayerRow[];
  /** Sum of layer-owned render buffers; these allocations are outside the registry budget. */
  layerRenderBufferBytes: number;
  operations: readonly MemoryManagerOperation[];
}>;

export const MEMORY_WARNING_FRACTION = 0.8;
export const MEMORY_CRITICAL_FRACTION = 0.95;

export function getMemoryPressure(usedFraction: number): MemoryPressure {
  if (usedFraction >= MEMORY_CRITICAL_FRACTION) return 'critical';
  return usedFraction >= MEMORY_WARNING_FRACTION ? 'warning' : 'ok';
}

export function formatMemoryBytes(byteLength: number): string {
  if (byteLength >= 1024 ** 3) return `${(byteLength / 1024 ** 3).toFixed(2)} GiB`;
  if (byteLength >= 1024 ** 2) return `${(byteLength / 1024 ** 2).toFixed(1)} MiB`;
  if (byteLength >= 1024) return `${(byteLength / 1024).toFixed(1)} KiB`;
  return `${byteLength} B`;
}

/** Builds the per-dataset and per-layer accounting shown by the memory manager. */
export function buildMemoryManagerModel(inputs: MemoryManagerInputs): MemoryManagerModel {
  const {registry, layers} = inputs;
  const usage = registry.memoryUsage;
  const budgetBytes = registry.memoryBudgetBytes;
  const evicted = new Set(inputs.evictedIds);
  const layersByDataset = new Map<string, MemoryManagerLayer[]>();
  for (const layer of layers) {
    if (!layer.dataId) continue;
    const sharing = layersByDataset.get(layer.dataId) ?? [];
    sharing.push(layer);
    layersByDataset.set(layer.dataId, sharing);
  }
  const footprints = registry.getDatasetMemory();
  const footprintById = new Map(footprints.map(footprint => [footprint.id, footprint]));
  const datasets: MemoryManagerDatasetRow[] = footprints.map(footprint => {
    const sharing = layersByDataset.get(footprint.id) ?? [];
    const isPinned = inputs.pinnedIds.has(footprint.id);
    return {
      id: footprint.id,
      label: inputs.datasetLabels[footprint.id] ?? footprint.id,
      rowCount: footprint.rowCount,
      cpuBytes: footprint.cpuBytes,
      gpuBytes: footprint.gpuBytes,
      ownedBytes: footprint.ownedBytes,
      borrowedBytes: footprint.borrowedBytes,
      releasableBytes: footprint.ownedBytes - footprint.sharedBytes,
      layerIds: sharing.map(layer => layer.id),
      isVisible: sharing.some(layer => layer.isVisible),
      isPinned,
      isEvicted: evicted.has(footprint.id),
      canUnload: !isPinned,
      canEvict: !isPinned && inputs.canEvict(footprint.id)
    };
  });
  const layerRows: MemoryManagerLayerRow[] = layers.map(layer => {
    const footprint = layer.dataId ? footprintById.get(layer.dataId) : undefined;
    const sharedWithLayers = layer.dataId ? (layersByDataset.get(layer.dataId)?.length ?? 1) : 0;
    const measured = inputs.layerRenderBuffers?.get(layer.id);
    return {
      id: layer.id,
      label: layer.label,
      dataId: layer.dataId,
      isVisible: layer.isVisible,
      datasetBytes: footprint ? footprint.cpuBytes + footprint.gpuBytes : 0,
      renderBufferBytes: measured?.bytes ?? 0,
      renderBufferCount: measured?.bufferCount ?? 0,
      isMeasured: measured !== undefined && measured.bufferCount > 0,
      sharedWithLayers
    };
  });
  const reservations = registry.getReservationBreakdown();
  const allocatedBytes = usage.allocatedBytes.cpu + usage.allocatedBytes.gpu;
  const datasetBytes = footprints.reduce(
    (total, footprint) => total + footprint.cpuBytes + footprint.gpuBytes,
    0
  );
  const usedFraction = budgetBytes > 0 ? usage.totalBytes / budgetBytes : 1;
  return {
    budgetBytes,
    usedBytes: usage.totalBytes,
    availableBytes: usage.availableBytes,
    usedFraction,
    pressure: getMemoryPressure(usedFraction),
    residentCpuBytes: usage.allocatedBytes.cpu,
    residentGpuBytes: usage.allocatedBytes.gpu,
    reservations,
    unattributedBytes: Math.max(0, allocatedBytes - datasetBytes),
    datasets,
    layers: layerRows,
    layerRenderBufferBytes: layerRows.reduce((total, row) => total + row.renderBufferBytes, 0),
    operations: inputs.operations ?? []
  };
}
