// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {QUADRAT_INDEX_WGSL} from './quadrat-index-wgsl';

/**
 * @internal Left-closed lag bins with exact float32/integer edge products. Histogram capacities
 * keep counts below the shared product comparator's 65536 limit. Binary search compares exact
 * significand products, preserving edge ties without forming an extreme reciprocal.
 */
export const PAIR_LAG_INDEX_WGSL = /* wgsl */ `${QUADRAT_INDEX_WGSL}
fn getPairLagIndex(lag: f32, maximum: f32, count: u32) -> u32 {
  return getQuadratIndex(lag, maximum, count);
}`;
