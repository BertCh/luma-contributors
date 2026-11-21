// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPU_COST_DISTANCE_NONE, GPU_COST_DISTANCE_UNRESOLVED} from './gpu-cost-distance';
import type {CostDistancePathOperationProps} from './gpu-cost-distance-path-operation';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';

/** Packed output rows; only `ids[0..count)` are a published path. */
export type CPUCostDistancePathResult = {
  ids: Uint32Array;
  count: Uint32Array;
  overflow: Uint32Array;
  requiredCount: Uint32Array;
  candidateOverflow: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Follows D8 predecessor codes from one target, publishing only a walk that reaches a source.
 * Invalid walks have zero count even if their scratch prefix has been written to `ids`.
 *
 * What differs from the GPU contributor:
 * - Unwritten rows of `ids` are zero; the GPU leaves rows outside its written prefix unspecified.
 * - Nothing else: integer codes, the cell-count cycle bound and all status words match exactly.
 *
 * @throws If `backLinks` or `target` has an inconsistent length, or grid/capacity is invalid.
 */
export function computeCostDistancePathOnCPU(
  backLinks: Uint32Array,
  target: Uint32Array,
  grid: GPURasterMetadata,
  capacity: number
): CPUCostDistancePathResult {
  validatePathOptions('computeCostDistancePathOnCPU', grid, capacity);
  const cellCount = grid.width * grid.height;
  if (backLinks.length !== cellCount) {
    throw new Error('computeCostDistancePathOnCPU backLinks must contain one row per grid cell');
  }
  if (target.length !== 1) {
    throw new Error('computeCostDistancePathOnCPU target must contain one row');
  }
  const ids = new Uint32Array(capacity);
  let length = 0;
  let unresolved = 0;
  let complete = false;
  let cell = target[0];
  const columnOffsets = [1, 1, 0, -1, -1, -1, 0, 1];
  const rowOffsets = [0, 1, 1, 1, 0, -1, -1, -1];
  if (cell < cellCount && backLinks[cell] !== GPU_COST_DISTANCE_NONE) {
    for (let walkIndex = 0; walkIndex < cellCount; walkIndex++) {
      const code = backLinks[cell];
      if (code === GPU_COST_DISTANCE_UNRESOLVED) {
        unresolved = 1;
        break;
      }
      if (length < capacity) ids[length] = cell;
      length++;
      if (code === 0) {
        complete = true;
        break;
      }
      if (code > 128 || (code & (code - 1)) !== 0) break;
      const direction = Math.log2(code);
      const column = (cell % grid.width) + columnOffsets[direction];
      const row = Math.floor(cell / grid.width) + rowOffsets[direction];
      if (column < 0 || column >= grid.width || row < 0 || row >= grid.height) break;
      const next = row * grid.width + column;
      if (backLinks[next] === GPU_COST_DISTANCE_NONE) break;
      cell = next;
    }
  }
  if (!complete) length = 0;
  return {
    ids,
    count: new Uint32Array([Math.min(length, capacity)]),
    overflow: new Uint32Array([Number(length > capacity)]),
    requiredCount: new Uint32Array([length]),
    candidateOverflow: new Uint32Array([0]),
    iterationLimitReached: new Uint32Array([unresolved])
  };
}

function validatePathOptions(id: string, grid: GPURasterMetadata, capacity: number): void {
  validateCapacity(id, 'grid.width', grid.width);
  validateCapacity(id, 'grid.height', grid.height);
  validateCapacity(id, 'grid cell count', grid.width * grid.height);
  validateRequirement(
    id,
    grid.affine.length === 6 && grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite values'
  );
  validateCapacity(id, 'capacity', capacity);
}

/** CPU twin with the same graph ports and no additional GPU size restrictions. */
export class CPUCostDistancePathOperation extends GPUCPUOperation {
  constructor(props: CostDistancePathOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'backLinks',
      'target',
      'capacity',
      'output'
    ]);
    const id = props.id ?? 'cost-distance-path';
    validatePathOptions(id, props.grid, props.capacity);
    const capturedPath = snapshotConstructionFields(props.output.path, ['ids', 'status']);
    const path = Object.freeze({
      ...capturedPath,
      status: snapshotConstructionFields(capturedPath.status, [
        'count',
        'overflow',
        'requiredCount',
        'candidateOverflow'
      ])
    });
    const grid = snapshotRasterGrid(props.grid);
    const capacity = props.capacity;
    const requiredCount = path.status.requiredCount;
    const candidateOverflow = path.status.candidateOverflow;
    const iterationLimitReached = props.output.status?.iterationLimitReached;
    validateOperationGraphNames(
      id,
      {backLinks: props.backLinks, target: props.target},
      {
        ids: path.ids,
        count: path.status.count,
        overflow: path.status.overflow,
        ...(requiredCount !== undefined ? {requiredCount} : {}),
        ...(candidateOverflow !== undefined ? {candidateOverflow} : {}),
        ...(iterationLimitReached !== undefined ? {iterationLimitReached} : {})
      }
    );
    super({
      id,
      inputs: {backLinks: props.backLinks, target: props.target},
      outputs: {
        [path.ids]: gpuGraph.vector('uint32', {length: props.capacity}),
        [path.status.count]: gpuGraph.vector('uint32', {length: 1}),
        [path.status.overflow]: gpuGraph.vector('uint32', {length: 1}),
        ...(requiredCount !== undefined
          ? {[requiredCount]: gpuGraph.vector('uint32', {length: 1})}
          : {}),
        ...(candidateOverflow !== undefined
          ? {[candidateOverflow]: gpuGraph.vector('uint32', {length: 1})}
          : {}),
        ...(iterationLimitReached !== undefined
          ? {[iterationLimitReached]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const result = computeCostDistancePathOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'backLinks'),
          getCPUOperationUint32Rows(id, inputs, 'target'),
          grid,
          capacity
        );
        return {
          [path.ids]: result.ids,
          [path.status.count]: result.count,
          [path.status.overflow]: result.overflow,
          ...(requiredCount !== undefined ? {[requiredCount]: result.requiredCount} : {}),
          ...(candidateOverflow !== undefined
            ? {[candidateOverflow]: result.candidateOverflow}
            : {}),
          ...(iterationLimitReached !== undefined
            ? {[iterationLimitReached]: result.iterationLimitReached}
            : {})
        };
      }
    });
  }
}
