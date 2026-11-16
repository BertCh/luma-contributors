// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Quadratic interval costs fit the reference route's 4 MiB cost-table budget. @internal */
export const NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS = 1024;
/** Recomputed exact DP avoids quadratic storage but retains O(classes * rows²) work. @internal */
export const NATURAL_BREAKS_MAXIMUM_EXACT_ROWS = 8192;
/**
 * Bounds the approximate monotone DP's O(classes * rows) partitions and worst-case fallback work.
 * At 2^18 rows, 20-bit positions also keep recentered wide moment intermediates below 2^59.
 * No scalable exact substitute exists: retain this explicit route and its current envelope.
 * @internal
 */
export const NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS = 262144;
