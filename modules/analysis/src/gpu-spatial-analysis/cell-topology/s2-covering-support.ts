// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rows one workgroup of the published `GPUScan` scans; each further level divides by it. */
const SCAN_ROWS_PER_BLOCK = 256;

/**
 * Command nodes the published `GPUScan` (10.0.0-alpha.10) adds for `rowCount` rows: one block-scan
 * node per level, and one offset node per level but the last. A test builds the real scan and
 * compares, so a changed published node count fails it. @internal
 */
export function getScanCommandNodeCount(rowCount: number): number {
  let levelCount = 1;
  for (let length = rowCount; Math.ceil(length / SCAN_ROWS_PER_BLOCK) > 1; levelCount++) {
    length = Math.ceil(length / SCAN_ROWS_PER_BLOCK);
  }
  return 2 * levelCount - 1;
}

import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';

/** One uint32 round index starts at each WebGPU-aligned 256-byte slot. */
export const S2_COVERING_ROUND_SLOT_WORDS = 64;

export function validateS2CoveringRoundParameters<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  roundCount: number
): void {
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'round parameter slot',
    4,
    'use a device supporting one storage word'
  );
  validateCapacity(
    id,
    'round parameter buffer byteLength',
    roundCount * S2_COVERING_ROUND_SLOT_WORDS * 4,
    0,
    graph.device.limits.maxBufferSize
  );
}

/** Runtime round slots preserve one classify/write pipeline for every refinement level. @internal */
export function createS2CoveringRoundParameters<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  roundCount: number
) {
  const values = new Uint32Array(roundCount * S2_COVERING_ROUND_SLOT_WORDS);
  for (let round = 0; round < roundCount; round++)
    values[round * S2_COVERING_ROUND_SLOT_WORDS] = round;
  const {parameters, nodes} = createSpatialStatisticsParameterNodes(
    graph,
    `${id}-round-parameters`,
    values
  );
  const slots: GraphDataView<'uint32'>[] = Array.from({length: roundCount}, (_, round) =>
    graph.createDataView(parameters.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: round * S2_COVERING_ROUND_SLOT_WORDS * 4
    })
  );
  return {slots, nodes};
}
