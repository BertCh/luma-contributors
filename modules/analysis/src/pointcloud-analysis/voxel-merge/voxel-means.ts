// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Prepare caller-owned tile means; downsampling returns representative positions instead. */
export function computePointCloudVoxelMeans(
  positions: Float32Array,
  voxelSize: number,
  gridSize: readonly [number, number, number],
  origin: readonly [number, number, number] = [0, 0, 0]
): Float32Array {
  const identifier = 'computePointCloudVoxelMeans';
  validateRequirement(
    identifier,
    positions.length % 3 === 0,
    'positions must contain complete XYZ triples'
  );
  validateRequirement(
    identifier,
    Math.fround(voxelSize) > 0 && Number.isFinite(Math.fround(voxelSize)),
    'voxelSize must be positive and finite in float32'
  );
  for (let axis = 0; axis < 3; axis++) {
    validateRequirement(
      identifier,
      Number.isSafeInteger(gridSize[axis]) && gridSize[axis] > 0,
      `gridSize[${axis}] must be a positive safe integer`
    );
    validateRequirement(
      identifier,
      Number.isFinite(Math.fround(origin[axis])),
      `origin[${axis}] must be finite in float32`
    );
  }
  const voxelCount = gridSize[0] * gridSize[1] * gridSize[2];
  validateRequirement(
    identifier,
    Number.isSafeInteger(voxelCount) && voxelCount < 0xffffffff,
    'gridSize total cell count must be a safe integer below 4294967295'
  );
  const means = new Float64Array(voxelCount * 3);
  const counts = new Uint32Array(voxelCount);
  for (let row = 0; row < positions.length / 3; row++) {
    const position = positions.subarray(row * 3, row * 3 + 3);
    const cell = Array.from(position, (value, axis) =>
      Math.floor(
        Math.fround(Math.fround(value - Math.fround(origin[axis])) / Math.fround(voxelSize))
      )
    );
    if (
      !cell.every((value, axis) => Number.isFinite(value) && value >= 0 && value < gridSize[axis])
    )
      continue;
    const voxel = (cell[2] * gridSize[1] + cell[1]) * gridSize[0] + cell[0];
    const count = ++counts[voxel];
    for (let axis = 0; axis < 3; axis++)
      means[voxel * 3 + axis] += (position[axis] - means[voxel * 3 + axis]) / count;
  }
  return Float32Array.from(means);
}
