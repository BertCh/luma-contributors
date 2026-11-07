// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUSpatialWeights} from './spatial-weights';
import {validateGPUSpatialWeights} from './spatial-weights';
import {validateCapacity} from '../../utils/gpu-contributor-utils';

/**
 * One row range of a chunked CSR: focus rows `[rowStart, rowStart + rowCount)`. `weights.offsets`
 * holds `rowCount + 1` entries that start at 0, and `weights.neighbors` holds global row IDs
 * (ascending within each row), so chunk `c` fits one storage binding whenever its own slot count
 * does, however large the whole matrix is.
 */
export type GPUSpatialWeightsChunk = {
  rowStart: number;
  rowCount: number;
  weights: GPUSpatialWeights;
};

/**
 * A spatial-weights matrix stored as consecutive row chunks, each its own CSR with its own buffers.
 * Consumers that accept it dispatch their weights-reading kernels once per chunk (writing the
 * chunk's rows) instead of binding one buffer for the whole matrix; see `GPULocalPermutationTest`.
 * The invariants of {@link GPUSpatialWeights} hold per chunk. Chunks cover `[0, rowCount)` in order
 * without gaps. Do not combine chunks into one buffer implicitly: packing is explicit.
 */
export type GPUChunkedSpatialWeights = Pick<
  GPUSpatialWeights,
  'spatialContext' | 'information' | 'status'
> & {
  /** Total focus rows across all chunks. */
  rowCount: number;
  chunks: readonly GPUSpatialWeightsChunk[];
};

export function isGPUChunkedSpatialWeights(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): weights is GPUChunkedSpatialWeights {
  return 'chunks' in weights;
}

/** The chunks of either form; whole weights are one chunk starting at row 0. */
export function getGPUSpatialWeightsChunks(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): readonly GPUSpatialWeightsChunk[] {
  return isGPUChunkedSpatialWeights(weights)
    ? weights.chunks
    : [{rowStart: 0, rowCount: weights.offsets.length - 1, weights}];
}

/** Validates the chunk layout and each chunk's CSR shape; returns the total row count. */
export function validateGPUChunkedSpatialWeights(
  id: string,
  weights: GPUChunkedSpatialWeights
): number {
  validateCapacity(id, 'weights chunk count', weights.chunks.length, 1);
  let nextRow = 0;
  weights.chunks.forEach((chunk, index) => {
    if (chunk.rowStart !== nextRow) {
      throw new Error(`${id} weights chunk ${index} must start at row ${nextRow}`);
    }
    const rowCount = validateGPUSpatialWeights(id, chunk.weights, `weights.chunks[${index}]`);
    if (rowCount !== chunk.rowCount) {
      throw new Error(`${id} weights chunk ${index} rowCount must equal its offsets length - 1`);
    }
    nextRow += chunk.rowCount;
  });
  if (nextRow !== weights.rowCount) {
    throw new Error(`${id} weights chunks must cover rowCount rows`);
  }
  return nextRow;
}

/** Row-count validation for either form of weights. */
export function validateGPUSpatialWeightsOrChunks(
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): number {
  return isGPUChunkedSpatialWeights(weights)
    ? validateGPUChunkedSpatialWeights(id, weights)
    : validateGPUSpatialWeights(id, weights);
}

/** All views of either form, for graph-membership and disjointness checks. */
export function getGPUSpatialWeightsOrChunksViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): (GraphDataView<'uint32'> | GraphDataView<'float32'> | undefined)[] {
  return getGPUSpatialWeightsChunks(weights).flatMap(chunk => [
    chunk.weights.offsets,
    chunk.weights.neighbors,
    chunk.weights.weights,
    chunk.weights.distances
  ]);
}

/**
 * Splits `rowCount` rows into consecutive chunks whose offsets fit `maximumSlotsPerChunk`, given
 * the exact per-row neighbor counts (for example from a count pass read back to the CPU). A single
 * row larger than the limit still gets its own chunk; the caller's binding check then refuses it.
 */
export function planGPUSpatialWeightsChunks(
  rowNeighborCounts: ArrayLike<number>,
  maximumSlotsPerChunk: number
): {rowStart: number; rowCount: number; slotCount: number}[] {
  validateCapacity('planGPUSpatialWeightsChunks', 'maximumSlotsPerChunk', maximumSlotsPerChunk);
  const chunks: {rowStart: number; rowCount: number; slotCount: number}[] = [];
  let rowStart = 0;
  let slotCount = 0;
  for (let row = 0; row < rowNeighborCounts.length; row++) {
    const count = rowNeighborCounts[row];
    if (row > rowStart && slotCount + count > maximumSlotsPerChunk) {
      chunks.push({rowStart, rowCount: row - rowStart, slotCount});
      rowStart = row;
      slotCount = 0;
    }
    slotCount += count;
  }
  if (rowNeighborCounts.length > rowStart) {
    chunks.push({rowStart, rowCount: rowNeighborCounts.length - rowStart, slotCount});
  }
  return chunks;
}

/**
 * One dispatch of a chunk-aware kernel: the chunk to bind plus the WGSL snippets that make a
 * whole-CSR kernel body read it. A whole CSR yields a single dispatch with no `WEIGHTS_ROW_START`,
 * so its generated WGSL is unchanged.
 */
export type GPUSpatialWeightsChunkDispatch = {
  chunkIndex: number;
  /** First focus row of the chunk. */
  rowStart: number;
  /** Focus rows of the chunk, and the invocation count of a row-indexed kernel. */
  rowCount: number;
  weights: GPUSpatialWeights;
  /** True for whole (single CSR) weights: no row offset in the generated WGSL. */
  isWhole: boolean;
  /** Node ID suffix: empty for whole weights, else `-chunk<index>`. */
  idSuffix: string;
  /** WGSL expression for the neighbor loop `rowStart` argument; undefined for whole weights. */
  rowStartWGSL: string | undefined;
  /** Declarations to add: the `WEIGHTS_ROW_START` constant for chunks, nothing for whole weights. */
  constantsWGSL: string;
};

/** Expands either form of weights into the dispatches of a chunk-aware kernel. */
export function getGPUSpatialWeightsChunkDispatches(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): GPUSpatialWeightsChunkDispatch[] {
  const isWhole = !isGPUChunkedSpatialWeights(weights);
  return getGPUSpatialWeightsChunks(weights).map((chunk, chunkIndex) => ({
    chunkIndex,
    rowStart: chunk.rowStart,
    rowCount: chunk.rowCount,
    weights: chunk.weights,
    isWhole,
    idSuffix: isWhole ? '' : `-chunk${chunkIndex}`,
    rowStartWGSL: isWhole ? undefined : 'WEIGHTS_ROW_START',
    constantsWGSL: isWhole ? '' : `const WEIGHTS_ROW_START: u32 = ${chunk.rowStart}u;`
  }));
}

/**
 * Wraps a row-indexed kernel body so that, for a chunk, `index` is the global row (the invocation
 * index plus the chunk's first row). Whole weights keep the body unchanged.
 */
export function wrapGPUSpatialWeightsChunkBody(
  dispatch: GPUSpatialWeightsChunkDispatch,
  body: string
): string {
  return dispatch.isWhole ? body : `{ let index = index + WEIGHTS_ROW_START;\n${body}\n}`;
}
