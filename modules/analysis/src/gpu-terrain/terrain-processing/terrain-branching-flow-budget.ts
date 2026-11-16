// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_GATE_INTERVAL} from '../../raster-analysis/cost-distance/raster-relaxation';

// Each shared raster gate slot starts on the WebGPU 256-byte storage-binding alignment.
const TERRAIN_BRANCHING_GATE_SLOT_BYTE_LENGTH = 256;

/** Includes the shared gate table, group gates, initialization and count/publication. @internal */
export function getTerrainBranchingFlowCommandNodeCount(
  maximumIterations: number,
  validity = false
): number {
  return (
    6 + Number(validity) + maximumIterations + Math.ceil(maximumIterations / RASTER_GATE_INTERVAL)
  );
}

/** Check before composition allocates any transient, including tile exchange buffers. @internal */
export function validateTerrainBranchingFlowScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: number,
  maximumIterations: number
): void {
  for (const [name, byteLength] of [
    ['state scratch', count * 4],
    ['iteration status scratch', 4 * 4],
    [
      'gate slot scratch',
      (Math.min(maximumIterations, RASTER_GATE_INTERVAL) + 1) *
        TERRAIN_BRANCHING_GATE_SLOT_BYTE_LENGTH
    ]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce grid dimensions or maximumIterations'
    );
  }
}
