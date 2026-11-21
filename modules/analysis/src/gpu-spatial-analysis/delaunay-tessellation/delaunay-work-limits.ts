// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {clamp} from '@math.gl/core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Maximum insertion and flip rounds, shared by both execution routes. @internal */
export const MAXIMUM_STEP_COUNT = 4096;

/** State words: triangleCount, failed, remainingCount, phase, illegalCount, insertedCount,
 * stepCount, unconverged. @internal */
export const GPU_DELAUNAY_STATE_WORD_COUNT = 8;
/** Largest site axis replayed by either serial continuation kernel. @internal */
export const MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT = 2048;
/** Largest allocated triangle axis replayed by either serial continuation kernel. @internal */
export const MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT = 4104;

/** Four-word triangle records use uint32 word addresses in every topology kernel. */
const DELAUNAY_MAXIMUM_SCRATCH_TRIANGLE_COUNT = Math.floor(0xffffffff / 4);
/** Validates uint32 topology addresses without needing a graph. @internal */
export function getDelaunayScratchTriangleCapacity(
  id: string,
  pointCount: number,
  groupCount = 0
): number {
  const capacity = 2 * pointCount + Math.max(groupCount, 1) + 7;
  validateCapacity(
    id,
    'scratch triangle count',
    capacity,
    8,
    DELAUNAY_MAXIMUM_SCRATCH_TRIANGLE_COUNT
  );
  return capacity;
}

/** Preflight for tessellation and enclosing alpha-hull contributors. @internal */
export function validateDelaunayScratch(
  device: Device,
  id: string,
  pointCount: number,
  groupCount = 0
): number {
  const capacity = getDelaunayScratchTriangleCapacity(id, pointCount, groupCount);
  validateGPUStorageBindingByteLength(
    device,
    id,
    'triangle topology scratch',
    16 * capacity,
    'reduce positions rows or groups.count'
  );
  return capacity;
}

/**
 * Maximum GPU insertion/flip steps. The initial logarithmic/square-root budget is extended to
 * `2n + 32`, capped at 4096 steps. Small inputs finish in two bounded GPU continuation kernels;
 * larger inputs use indirect parallel rounds throughout. No iteration index reaches shader text.
 * The cap is not a guarantee for arbitrary Lawson chains: status reports exhaustion explicitly.
 * @internal
 */
export function getDelaunayStepCount(pointCount: number): number {
  if (pointCount < 1) {
    return 0;
  }
  const initialBudget =
    4 * Math.ceil(Math.log2(pointCount + 2)) + 32 + Math.ceil(0.5 * Math.sqrt(pointCount));
  return clamp(2 * pointCount + 32, initialBudget, MAXIMUM_STEP_COUNT);
}

/** Number of parallel steps before continuation. Both serial kernels require at most
 * MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT sites and MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT
 * allocated triangles, including empty groups. Larger axes use the same parallel rounds.
 * @internal */
export function getDelaunayParallelStepCount(pointCount: number, groupCount = 0): number {
  const limit = getDelaunayStepCount(pointCount);
  const scratchTriangleCapacity = getDelaunayScratchTriangleCapacity(
    'GPUDelaunayTessellation',
    pointCount,
    groupCount
  );
  return pointCount <= MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT &&
    scratchTriangleCapacity <= MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT
    ? Math.min(
        limit,
        4 * Math.ceil(Math.log2(pointCount + 2)) + 32 + Math.ceil(0.5 * Math.sqrt(pointCount))
      )
    : limit;
}
