// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Each half-open binary search halves the remaining interval per midpoint. @internal */
export const SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS = 32;
/** All admitted rows fit unsigned shader counts; the end index itself may equal 0xffffffff. @internal */
export const SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT = 0xffffffff;

/**
 * Worst-case midpoint steps for a half-open range of rowCount rows: floor(log2(rowCount))+1
 * for a nonempty range, zero for empty. uint32 admission makes the bound at most 32.
 * This host check deliberately leaves the fingerprinted WGSL binary searches unchanged.
 * @internal
 */
export function getSortedTimeRangeMaximumSearchSteps(rowCount: number): number {
  validateCapacity(
    'sorted-time-range',
    'rowCount',
    rowCount,
    0,
    Math.min(SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT, 2 ** SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS - 1)
  );
  return rowCount === 0 ? 0 : Math.floor(Math.log2(rowCount)) + 1;
}
