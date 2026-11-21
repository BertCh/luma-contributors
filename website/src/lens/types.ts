// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compact, JSON-safe view model of one lens record. Produced at build time by
// extract-lens-view-model.ts, consumed by the how-it-works panels and /fn/$id pages.

export type LensNodeKind = 'compute' | 'copy' | 'render' | 'other';

export type LensResourceOwnership = 'imported' | 'transient' | 'unknown';
export type LensResourceBytesKind =
  | 'logical'
  | 'lower-bound'
  | 'estimated'
  | 'unspecified'
  | 'unavailable';

export type LensViewNode = {
  id: string;
  /** Short display label. Collapsed groups show the stem with `#` for the varying number. */
  label: string;
  kind: LensNodeKind;
  /** How many recorded command graph nodes this node stands for (1 unless collapsed). */
  count: number;
  /** Domain operator names such as `GPUScan`, at most three. */
  operators: string[];
  /** Summed maximum workgroups over the members. */
  workgroups: number;
  /** Summed bytes read by the members. */
  readBytes: number;
  /** Summed bytes written by the members. */
  writeBytes: number;
  /** GPU time in milliseconds when the record has timestamps. */
  milliseconds?: number;
  /** Summed maximum invocations over the members, when the kernels report a workload. */
  invocations?: number;
  /** Members whose dispatch size comes from a GPU buffer (indirect), so a round can run zero workgroups. */
  gated?: number;
  /** Kernel variant such as `classify`; only when the graph is not collapsed. */
  variant?: string;
  /** Ids of the buffers the kernel reads; only when the graph is not collapsed. */
  reads?: string[];
  /** Ids of the buffers the kernel writes; only when the graph is not collapsed. */
  writes?: string[];
};

export type LensViewEdge = {
  from: string;
  to: string;
  /** Bytes of the distinct resources that carry the dependency. */
  bytes: number;
  bytesKind?: LensResourceBytesKind;
  /** Names of up to three carrying resources. */
  resources: string[];
  /** Number of distinct resources behind the edge. */
  resourceCount: number;
  /** Distinct hazard kinds recorded on the underlying dependencies: read-after-write and so on. */
  hazards?: ('raw' | 'war' | 'waw')[];
};

/** When one buffer is alive, in steps of the execution order (indices into `nodes`). */
export type LensResourceLifetime = {
  id: string;
  bytes: number;
  ownership: LensResourceOwnership;
  bytesKind?: LensResourceBytesKind;
  /** Step of the first kernel that writes the buffer (first user when only read). */
  first: number;
  /** Step of the last kernel that reads or writes it. */
  last: number;
  /** Transient allocation slot: buffers with the same slot alias one block of memory. */
  slot?: number;
};

export type LensResourceSummary = {
  /** False when the record omitted its resource inventory; numeric zeros then carry no evidence. */
  inventoryKnown?: boolean;
  count: number;
  totalBytes: number;
  totalBytesKind?: LensResourceBytesKind;
  importedBytes: number;
  importedBytesKind?: LensResourceBytesKind;
  transientBytes: number;
  transientBytesKind?: LensResourceBytesKind;
  unknownBytes?: number;
  unknownBytesKind?: LensResourceBytesKind;
  /** Bytes physically allocated for transients after aliasing. */
  allocatedBytes: number;
  allocatedBytesKind?: LensResourceBytesKind;
  /** Largest distinct sizes; `count` buffers share each size, `id` names one of them. */
  largest: {
    id: string;
    bytes: number;
    bytesKind?: LensResourceBytesKind;
    ownership: LensResourceOwnership;
    count: number;
  }[];
};

export type LensFormula = {id: string; ascii: string};

/** A series the run recorded per round, such as the frontier size of each iteration. */
export type LensRoundSeries = {name: string; unit?: string; values: number[]};

export type LensParameter = {name: string; label: string; value: string; unit?: string};

/** A point layer in a map frame; positions are interleaved x, y in 0..1 (y down). */
export type LensPointLayer = {
  name: string;
  positions: number[];
  /** Per point values in 0..1 on the layer's own range, when the capture has values. */
  values?: number[];
  min?: number;
  max?: number;
};

export type LensRaster = {
  name: string;
  width: number;
  height: number;
  /** Base64 of width * height bytes, 0..255 on [min, max]; row 0 on top. */
  grid: string;
  min: number;
  max: number;
  /** Colored PNG data URL: the static fallback drawn from the same bytes. */
  png: string;
};

export type LensMap = {
  title: string;
  /** Width over height of the frame. */
  aspectRatio: number;
  raster?: LensRaster;
  layers: LensPointLayer[];
};

export type LensSeries = {
  name: string;
  role: string;
  /** Up to 200 values downsampled by block mean. */
  values: number[];
  length: number;
  elementType: string;
  min: number;
  max: number;
};

export type LensViewModel = {
  /** Scenario id such as `gpu-spatial-analysis.kernel-density`. */
  id: string;
  title: string;
  pack: string;
  adapter: string;
  /** Graph form shown, normally `compiled`. */
  form: string;
  /** Recorded command graph node count before collapsing. */
  recordedNodeCount: number;
  recordedEdgeCount: number;
  collapsed: boolean;
  nodes: LensViewNode[];
  edges: LensViewEdge[];
  resources: LensResourceSummary;
  /** Largest buffers with their first write and last use; absent in older view models. */
  lifetimes?: LensResourceLifetime[];
  /** Buffers left out of `lifetimes`. */
  lifetimesOmitted?: number;
  /** Recorded dependency counts by hazard kind: read-after-write, write-after-read, write-after-write. */
  hazards: {raw: number; war: number; waw: number};
  totals: {workgroups: number; readBytes: number; writeBytes: number};
  formulas: LensFormula[];
  parameters: LensParameter[];
  map?: LensMap;
  series: LensSeries[];
  /** Per-round series, when the run recorded any. */
  rounds?: LensRoundSeries[];
  /** Recorded kernels whose dispatch is gated on the GPU. */
  gatedNodeCount?: number;
  /** Size in bytes of every buffer a node reads or writes; only when the graph is not collapsed. */
  bufferBytes?: Record<string, number>;
  bufferBytesKind?: Record<string, LensResourceBytesKind>;
};

/** Totals over every lens record, written to summary.json by generate-view-models.ts. */
export type LensSummary = {
  recordCount: number;
  packs: Record<string, number>;
  kernels: number;
  copies: number;
  gatedKernels: number;
  dependencies: {raw: number; war: number; waw: number};
  /** All recorded resource entries: buffers, textures and external textures. */
  resourceCount: number;
  /** False if any source graph omitted its inventory; count and bytes then sum only known data. */
  inventoryKnown?: boolean;
  /** Sum of observed resource sizes; precision/availability applies to the entire inventory. */
  resourceBytes: number;
  resourceBytesKind?: LensResourceBytesKind;
  snapshots: number;
  formulas: number;
  recordsWithFormulas: number;
  kernelsPerRecord: {minimum: number; median: number; maximum: number};
  smallest: {id: string; kernels: number};
  largest: {id: string; kernels: number};
  recordBytes: number;
  viewModelBytes: number;
};
