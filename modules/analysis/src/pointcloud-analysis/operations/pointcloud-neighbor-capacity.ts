// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';

/** Infer only an exact fixed stride; an empty row domain needs the caller's allocation choice. */
export function inferPointCloudNeighborCapacity(
  id: string,
  queryCount: number,
  slotCount: number
): number {
  validateRequirement(
    id,
    queryCount > 0,
    `capacity must be explicit for zero query rows; received ${queryCount}`
  );
  const capacity = slotCount / queryCount;
  validateCapacity(
    id,
    `capacity inferred from ${slotCount} slots / ${queryCount} query rows`,
    capacity,
    1,
    GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
  );
  return capacity;
}
