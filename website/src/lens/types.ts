// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compact, JSON-safe view model of one lens record. Produced at build time by
// extract-lens-view-model.ts, consumed by the how-it-works panels and /fn/$id pages.

export type LensNodeKind = 'compute' | 'copy' | 'render' | 'other';

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
};

export type LensViewEdge = {
  from: string;
  to: string;
  /** Bytes of the distinct resources that carry the dependency. */
  bytes: number;
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
  ownership: 'imported' | 'transient';
  /** Step of the first kernel that writes the buffer (first user when only read). */
  first: number;
  /** Step of the last kernel that reads or writes it. */
  last: number;
  /** Transient allocation slot: buffers with the same slot alias one block of memory. */
  slot?: number;
};

export type LensResourceSummary = {
  count: number;
  totalBytes: number;
  importedBytes: number;
  transientBytes: number;
  /** Bytes physically allocated for transients after aliasing. */
  allocatedBytes: number;
  /** Largest distinct sizes; `count` buffers share each size, `id` names one of them. */
  largest: {id: string; bytes: number; ownership: 'imported' | 'transient'; count: number}[];
};

export type LensFormula = {id: string; ascii: string};

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
};
