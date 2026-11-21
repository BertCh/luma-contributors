// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rows per global permutation pair-sum block, per CSR chunk. */
export const GLOBAL_PERMUTATION_PAIR_BLOCK_ROWS = 2048;
/** Largest total of (requested draws + observed) times row blocks on the global route. */
export const MAXIMUM_GLOBAL_PERMUTATION_PAIR_BLOCK_COUNT = Math.floor(0xffffffff / 256);
