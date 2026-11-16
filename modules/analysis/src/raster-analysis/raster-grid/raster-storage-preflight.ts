// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRasterFractionalBindingByteLength} from '../fractional-coverage/raster-fractional-tile-kernel';
import {
  RASTER_GATE_INTERVAL,
  RASTER_GATE_MAXIMUM_GROUP_SIZE
} from '../cost-distance/raster-relaxation';

/** Gate slot words use the shared relaxation table's 256-byte storage alignment. */
const RASTER_ITERATION_SLOT_BYTE_LENGTH = 256;

/**
 * Checks caller usage, containment, aligned bindings and backing allocations, then every derived
 * scratch allocation. Byte lengths describe individual bindings, not their combined footprint.
 * This declares no graph resources or nodes and imposes no buffer-identity alias restriction.
 * Callers omit optional ports that their selected route does not bind. @internal
 */
export function validateRasterStorageBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: readonly (GraphDataView | undefined)[],
  scratchByteLengths: Readonly<Record<string, number>> = {}
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view.${index}`, view]))
  );
  for (const view of views) {
    if (view) {
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        view.buffer.id,
        getViewBindingRange(view).size,
        'reduce the packed raster or output capacity',
        view.buffer.byteLength
      );
    }
  }
  for (const [name, byteLength] of Object.entries(scratchByteLengths)) {
    if (byteLength !== 0) {
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        `${name} scratch`,
        byteLength,
        'reduce the raster dimensions or output capacity'
      );
    }
  }
}

/** Actual full binding/allocation of the shared Jacobi gate table, including aligned slots. @internal */
export function getRasterIterationSlotByteLength(roundCount: number): number {
  return (
    (Math.min(RASTER_GATE_INTERVAL, RASTER_GATE_MAXIMUM_GROUP_SIZE, Math.max(roundCount, 1)) + 1) *
    RASTER_ITERATION_SLOT_BYTE_LENGTH
  );
}
