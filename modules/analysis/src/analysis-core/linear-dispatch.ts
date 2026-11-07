// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getBoundedDispatchLayout} from '@luma.gl/graph/gpu-core';

/**
 * Returns the x workgroup count of a one-dimensional dispatch for kernels that index with
 * `global_invocation_id.x` only. The shared bounded layout planner validates the element count and
 * device limit, so an oversized dispatch fails instead of silently dropping elements. @internal
 */
export function getLinearDispatchCount(
  operationName: string,
  elementCount: number,
  workgroupSize: number,
  maxComputeWorkgroupsPerDimension: number
): number {
  if (elementCount === 0) return 0;
  const layout = getBoundedDispatchLayout(
    operationName,
    elementCount,
    workgroupSize,
    maxComputeWorkgroupsPerDimension
  );
  // These kernels are not 3D-aware; a layout spilling into y or z would drop elements.
  if (layout.y !== 1 || layout.z !== 1) throw new Error(`${operationName} exceeds one dimension`);
  return layout.x;
}
