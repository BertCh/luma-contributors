// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';

/** Exact independent batch addressing, including observer/pixel products above uint32. @internal */
export const TERRAIN_VIEWSHED_BATCH_START_WGSL = `${UINT64_WGSL}
fn getViewshedBatchStart(batch: u32, batchSize: u32, pixelCount: u32) -> vec2<u32> {
  if (batch <= 0xffffffffu / batchSize) {
    let pair = batch * batchSize;
    return vec2<u32>(pair / pixelCount, pair % pixelCount);
  }
  let pair = divideUint64ByUint32(multiplyUint32Wide(batch, batchSize), pixelCount);
  return vec2<u32>(pair.quotient.y, pair.remainder);
}
`;
