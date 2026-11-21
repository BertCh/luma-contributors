// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateRequirement} from '../kernels/gpu-contributor-utils';

/** Shared dimension-only preconditions for assembly and embedding planners. @internal */
export function validateRingAssemblyDimensions(
  id: string,
  segmentCount: number,
  ringCapacity: number,
  vertexCapacity: number
): void {
  validateRequirement(
    id,
    segmentCount >= 1,
    `needs at least one segment row; received ${segmentCount}`
  );
  validateRequirement(
    id,
    ringCapacity >= 1,
    `output.ringOffsets needs at least two rows (one ring); received ${ringCapacity + 1}`
  );
  validateRequirement(
    id,
    vertexCapacity >= 1,
    `output.positions must not be empty; received ${vertexCapacity}`
  );
}
