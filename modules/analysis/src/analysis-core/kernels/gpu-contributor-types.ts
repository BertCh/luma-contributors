// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';

/**
 * A packed `uint32` view that may be one chunk or an ordered vector of chunks.
 *
 * Contributors accept both forms for source-aligned masks and IDs so they preserve the caller's batch
 * and chunk boundaries instead of repacking data.
 */
export type GPUUint32Rows = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** One GPU-written unsigned scalar: a one-row `uint32` status word. */
export type GPUStatusScalar = GraphDataView<'uint32'>;

/**
 * Usable and, when known, exact required cardinality of a bounded result. Defined here because
 * `utils` sits below `analysis-core`; the analysis status contracts re-export it.
 */
export type GPUCardinalityPort = {
  /** Number of usable rows or elements written. */
  count: GPUStatusScalar;
  /**
   * Unclamped requirement at the producer's final stage, before final-output truncation.
   * A lower bound on the full requirement when candidates were lost; scan totals saturate at
   * `0xffffffff`. Exact only when neither candidate loss nor saturation occurred.
   */
  requiredCount?: GPUStatusScalar;
};

/** Final-output completeness for a capacity-bounded result. */
export type GPUOverflowStatusPort = {
  /** One when the final result is incomplete because its output capacity was too small. */
  overflow: GPUStatusScalar;
};

/**
 * Caller-owned, capacity-bounded compact result shared by every GPU contributor and geospatial
 * query (the one compact-output contract; re-exported from the analysis contracts).
 *
 * `status.count` is always clamped to `ids.length` and is therefore safe to use as an indirect
 * draw instance count. Writable views must not alias each other or contributor inputs.
 */
export type GPUCompactOutput = {
  /** Compact identities in the producer's declared ID domain; never infer source rows from IDs. */
  ids: GraphDataView<'uint32'>;
  /**
   * One-row status scalars: `count` receives `min(requiredCount, ids.length)`, `overflow` receives
   * `1` when final output capacity was too small (otherwise `0`). Optional `candidateOverflow`
   * reports earlier candidate loss, which growing `ids` cannot recover. Optional `requiredCount`
   * counts accepted rows reaching the final stage; it is a lower bound when candidates were lost
   * or the count saturated at `0xffffffff`.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort & {candidateOverflow?: GPUStatusScalar};
};
