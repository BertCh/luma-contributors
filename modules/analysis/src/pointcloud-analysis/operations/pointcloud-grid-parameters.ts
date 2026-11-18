// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Origin X/Y/Z and positive cell size; dimensions and row direction remain graph topology. */
export const GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH = 4;

export type GPUPointCloudGridSettings = {
  origin: readonly [number, number] | readonly [number, number, number];
  cellSize: number;
};

/** Pack a runtime local-grid affine for raster or 3D point-cloud contributors. */
export function getGPUPointCloudGridParameterValues(
  settings: GPUPointCloudGridSettings
): Float32Array {
  const identifier = 'gpu-pointcloud-grid';
  validateRequirement(
    identifier,
    settings.origin.length === 2 || settings.origin.length === 3,
    'origin must hold two or three coordinates'
  );
  settings.origin.forEach((value, axis) =>
    validateFiniteNumber(identifier, `origin[${axis}]`, value)
  );
  validateFiniteNumber(identifier, 'cellSize', settings.cellSize);
  validateRequirement(
    identifier,
    Math.fround(settings.cellSize) > 0,
    'cellSize must be positive in float32'
  );
  return new Float32Array([
    settings.origin[0],
    settings.origin[1],
    settings.origin[2] ?? 0,
    settings.cellSize
  ]);
}

/** Validate optional borrowed grid parameters once at contributor construction. */
export function validatePointCloudGridParameters(
  id: string,
  view?: GraphDataView<'float32'>
): void {
  if (!view) return;
  validateSinglePackedView(view, ['float32'], `${id} gridParameters`);
  validateViewLength(id, 'gridParameters', view, GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH);
}
