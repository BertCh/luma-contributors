// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUSpatialWeightsChunkDispatches,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsChunkDispatch
} from '../spatial-weights/gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/**
 * A weights-reading dispatch of a tile-structured kernel. Tiles are laid out per chunk (a chunk
 * never shares a tile with another chunk), so the tile partials of chunk `c` occupy
 * `[tileBase, tileBase + chunkTileCount)` of the total tile range. A whole CSR is one dispatch
 * with `tileBase` 0 and keeps its generated WGSL unchanged.
 */
export type GPUSpatialWeightsTileDispatch = GPUSpatialWeightsChunkDispatch & {
  tileBase: number;
  chunkTileCount: number;
  /** Declarations for chunk dispatches (empty for whole weights). */
  tileConstantsWGSL: string;
};

export type GPUSpatialWeightsTilePlan = {
  /** Total tiles over all chunks; the tile count of the whole CSR when not chunked. */
  tileCount: number;
  /** Total focus rows over all chunks. */
  rowCount: number;
  dispatches: GPUSpatialWeightsTileDispatch[];
};

/** Lays out the tiles (`tileRowCount` rows each) of either form of weights. */
export function getGPUSpatialWeightsTilePlan(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  tileRowCount: number
): GPUSpatialWeightsTilePlan {
  let tileBase = 0;
  let rowCount = 0;
  const dispatches = getGPUSpatialWeightsChunkDispatches(weights).map(dispatch => {
    const chunkTileCount = Math.ceil(dispatch.rowCount / tileRowCount);
    const tileDispatch: GPUSpatialWeightsTileDispatch = {
      ...dispatch,
      tileBase,
      chunkTileCount,
      tileConstantsWGSL: dispatch.isWhole
        ? ''
        : `${dispatch.constantsWGSL}
const WEIGHTS_ROW_COUNT: u32 = ${dispatch.rowCount}u;
const TILE_BASE: u32 = ${tileBase}u;
const CHUNK_TILE_COUNT: u32 = ${chunkTileCount}u;`
    };
    tileBase += chunkTileCount;
    rowCount += dispatch.rowCount;
    return tileDispatch;
  });
  return {tileCount: tileBase, rowCount, dispatches};
}

const OFFSETS_PATTERN = /\boffsets\[offsetsOffset \+ (\w+)/g;

/**
 * Rewrites a whole-CSR WGSL body that reads `offsets[offsetsOffset + <row>]` for a chunk whose
 * `offsets` start at `WEIGHTS_ROW_START`. Whole weights return the body unchanged.
 */
export function shiftWeightsChunkOffsetsWGSL(
  dispatch: GPUSpatialWeightsChunkDispatch,
  body: string
): string {
  return dispatch.isWhole
    ? body
    : body.replace(OFFSETS_PATTERN, 'offsets[offsetsOffset + $1 - WEIGHTS_ROW_START');
}

/**
 * Rewrites the body of a row-indexed kernel (`index` is the row) for a chunk dispatched with
 * `rowCount` invocations: `index` becomes the global row and the CSR offsets are chunk-local.
 */
export function wrapWeightsChunkRowBodyWGSL(
  dispatch: GPUSpatialWeightsChunkDispatch,
  body: string
): string {
  return dispatch.isWhole
    ? body
    : `{ let index = index + WEIGHTS_ROW_START;\n${shiftWeightsChunkOffsetsWGSL(dispatch, body)}\n}`;
}

/**
 * Rewrites a whole-CSR tile-kernel body for a chunk dispatch. The body has `firstRow = index *
 * TILE_ROWS` (tile-per-invocation kernels, where `index` is the tile) or `probe = index /
 * TILE_COUNT; tile = index % TILE_COUNT` (probe-by-tile kernels), `lastRow = min(firstRow +
 * TILE_ROWS, ROWS)`, and tile partials indexed by the global tile number. In a chunk dispatch the
 * tile-per-invocation `index` becomes the global tile (`TILE_BASE + index`) while the rows stay
 * those of the chunk. Whole weights return the body unchanged.
 */
export function rewriteWeightsChunkTileBodyWGSL(
  dispatch: GPUSpatialWeightsTileDispatch,
  body: string
): string {
  if (dispatch.isWhole) {
    return body;
  }
  const rowBody = shiftWeightsChunkOffsetsWGSL(dispatch, body).replaceAll(
    'min(firstRow + TILE_ROWS, ROWS)',
    'min(firstRow + TILE_ROWS, WEIGHTS_ROW_START + WEIGHTS_ROW_COUNT)'
  );
  if (rowBody.includes('let probe = index / TILE_COUNT;')) {
    return rowBody
      .replaceAll('let probe = index / TILE_COUNT;', 'let probe = index / CHUNK_TILE_COUNT;')
      .replaceAll('let tile = index % TILE_COUNT;', 'let tile = index % CHUNK_TILE_COUNT;')
      .replaceAll(
        'let firstRow = tile * TILE_ROWS;',
        'let firstRow = WEIGHTS_ROW_START + tile * TILE_ROWS;'
      )
      .replaceAll('* TILE_COUNT + tile]', '* TILE_COUNT + TILE_BASE + tile]');
  }
  return `{ let index = index + TILE_BASE;\n${rowBody.replaceAll(
    'let firstRow = index * TILE_ROWS;',
    'let firstRow = WEIGHTS_ROW_START + (index - TILE_BASE) * TILE_ROWS;'
  )}\n}`;
}
