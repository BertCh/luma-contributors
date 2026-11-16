// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Grid candidates bound generated fit stages and shader text. @internal */
export const GENERALIZED_GWR_MAXIMUM_CANDIDATE_COUNT = 32;
/** Golden-section rounds are unrolled into gated command nodes. @internal */
export const GENERALIZED_GWR_MAXIMUM_SEARCH_ITERATIONS = 64;
/** Repeated dense IRLS and AICc searches are reference work, capped at 4,096 training/query slots. @internal */
export const GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS = 4096;
/** Conservative allowance for published grid, scan and stable-sort stages (up to 16,777,216 rows). @internal */
export const GENERALIZED_GWR_GRID_NODE_ALLOWANCE = 1024;
/**
 * Twice float32 epsilon; multiplied by (design columns + 8) and the sum of term magnitudes
 * to reject signs obscured by subtraction, scaling, products and accumulation rounding.
 * This is an uncertainty bound, never an allowance for an opposing move. @internal
 */
export const GENERALIZED_GWR_SEPARATION_ERROR_UNIT = '2.384185791015625e-7';

/** Per-thread neighbor arrays bound adaptive order-statistic insertion/search. @internal */
export const GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT = 128;
/** Local IRLS executes a bounded loop in one invocation, shared with multiscale input validation. @internal */
export const LOCAL_REGRESSION_MAXIMUM_ITERATIONS = 128;
/** Grid cell offsets and sentinels reserve the final uint32 value. @internal */
export const LOCAL_REGRESSION_MAXIMUM_GRID_CELL_COUNT = 0xfffffffe;

/** Float32 score counts remain exact through 2^24 rows; uint32 grid scans cannot overflow. @internal */
export const GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT = 16777216;
/** Reuse at most 65,536 local IRLS states (14 MiB at 24 predictors), independent of row count. @internal */
export const GENERALIZED_GWR_FOCUS_BATCH_ROWS = 65536;

/** Rows accumulated in historical order by each bounded deviance-score invocation. @internal */
export const GENERALIZED_GWR_SCORE_TILE_ROWS = 256;
/** Bound serial float32 accumulation error: merge at most 256 adjacent partials per invocation. @internal */
export const GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES = 256;

/** A focal batch needs at most 256 one-dimensional groups; tail group counts stay runtime data. @internal */
export const GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE = 256;

/** Largest finite IEEE float32 literal accepted by WGSL for compile-time numeric props. @internal */
export const GENERALIZED_GWR_MAXIMUM_SHADER_FLOAT = 3.4028234663852886e38;

/** Fixed adjacent-group score order, shared by declaration, preflight and capacity checks. @internal */
export function getGeneralizedGWRScoreMergeLayout(rows: number): {
  tileCount: number;
  mergedTileCount: number;
  finalTileCount: number;
  usesParallelMerge: boolean;
} {
  const tileCount = Math.ceil(rows / GENERALIZED_GWR_SCORE_TILE_ROWS);
  const mergedTileCount = Math.ceil(tileCount / GENERALIZED_GWR_SCORE_TILE_ROWS);
  const usesParallelMerge = tileCount > GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES;
  return {
    tileCount,
    mergedTileCount,
    finalTileCount: usesParallelMerge ? mergedTileCount : tileCount,
    usesParallelMerge
  };
}
