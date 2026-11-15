// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Grid candidates bound generated fit stages and shader text. @internal */
export const GENERALIZED_GWR_MAXIMUM_CANDIDATE_COUNT = 32;
/** Golden-section rounds are unrolled into gated command nodes. @internal */
export const GENERALIZED_GWR_MAXIMUM_SEARCH_ITERATIONS = 64;
/** Repeated dense IRLS and AICc searches are reference work, capped at 4,096 training/query slots. @internal */
export const GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS = 4096;
/** Conservative allowance for published grid, scan and stable-sort stages (up to 1,048,576 rows). @internal */
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
