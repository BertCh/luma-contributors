// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateMaximumCount} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsChunks,
  type GPUSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/index';
import {getGPUSpatialWeightsTilePlan} from './chunked-weights-kernels';
import {
  REGRESSION_TILE_MAXIMUM_TILE_COUNT,
  REGRESSION_TILE_MERGE_GROUP_SIZE
} from './regression-row-tiles';

/** Resolve nested trace tiling before a likelihood contributor declares any graph resources. */
export function getRegressionTraceLayout(
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  rowCount: number
): {tileRowCount: number; tileCount: number} {
  let tileRowCount = Math.max(32, Math.ceil(rowCount / 1024));
  let tileCount = getGPUSpatialWeightsTilePlan(weights, tileRowCount).tileCount;
  const canReachOneLevel =
    getGPUSpatialWeightsChunks(weights).filter(chunk => chunk.rowCount > 0).length <=
    REGRESSION_TILE_MERGE_GROUP_SIZE;
  // Preserve every historical valid layout: widen toward the one-level merge size, stopping
  // at one tile per nonempty chunk. If chunks still require more tiles, use the hierarchical
  // merge rather than refusing them. This loop has at most ceil(log2(rowCount)) layout steps.
  while (
    canReachOneLevel &&
    tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE &&
    tileRowCount < rowCount
  ) {
    tileRowCount = Math.min(rowCount, tileRowCount * 2);
    tileCount = getGPUSpatialWeightsTilePlan(weights, tileRowCount).tileCount;
  }
  validateMaximumCount(id, 'trace tileCount', tileCount, REGRESSION_TILE_MAXIMUM_TILE_COUNT);
  return {tileRowCount, tileCount};
}
