// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateMaximumCount} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights, GPUChunkedSpatialWeights} from '../spatial-weights/index';
import {getGPUSpatialWeightsTilePlan} from './chunked-weights-kernels';

/** Same serial norm-merge bound enforced by GPUSpatialWeightsTraces. */
const MAXIMUM_TRACE_TILE_COUNT = 4096;

/** Resolve nested trace tiling before a likelihood contributor declares any graph resources. */
export function getRegressionTraceLayout(
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  rowCount: number
): {tileRowCount: number; tileCount: number} {
  let tileRowCount = Math.max(32, Math.ceil(rowCount / 1024));
  let tileCount = getGPUSpatialWeightsTilePlan(weights, tileRowCount).tileCount;
  // Each step doubles the width, stopping at one tile per nonempty chunk. The loop is over
  // layout choices, never input rows, and has at most ceil(log2(rowCount)) steps.
  while (tileCount > MAXIMUM_TRACE_TILE_COUNT && tileRowCount < rowCount) {
    tileRowCount = Math.min(rowCount, tileRowCount * 2);
    tileCount = getGPUSpatialWeightsTilePlan(weights, tileRowCount).tileCount;
  }
  validateMaximumCount(id, 'trace tileCount', tileCount, MAXIMUM_TRACE_TILE_COUNT);
  return {tileRowCount, tileCount};
}
