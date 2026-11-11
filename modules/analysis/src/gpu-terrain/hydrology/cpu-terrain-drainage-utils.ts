// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_RASTER_D8_DIRECTIONS} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {
  validateTerrainFlowDirectionEncoding,
  type GPUTerrainFlowDirectionEncoding
} from './terrain-flow-tracing';

/** D8 missing and unresolved words are shared by the drainage vector layouts. @internal */
export const TERRAIN_DRAINAGE_NONE = 0xffffffff;
export const TERRAIN_DRAINAGE_UNRESOLVED = 0xfffffffe;

/** Graph names of optional scalar status ports. @internal */
export type TerrainDrainageOperationStatus = {
  converged?: string;
  iterationLimitReached?: string;
  iterationCount?: string;
};

/** Checks static grid and encoding props with the contributor's named validators. @internal */
export function validateTerrainDrainageOptions(
  id: string,
  grid: GPURasterMetadata,
  directionEncoding?: GPUTerrainFlowDirectionEncoding
): number {
  validateCapacity(id, 'grid.width', grid.width, 1);
  validateCapacity(id, 'grid.height', grid.height, 1);
  const count = grid.width * grid.height;
  // Two reserved cell indices are required for the missing and unresolved words.
  validateMaximumCount(id, 'grid cells', count, TERRAIN_DRAINAGE_UNRESOLVED);
  validateRequirement(
    id,
    grid.affine.length === 6 && grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite numbers'
  );
  validateTerrainFlowDirectionEncoding(id, directionEncoding);
  return count;
}

/** Decodes the documented D8 encodings, retaining both sentinel words. @internal */
export function getTerrainDrainageDirection(
  word: number,
  encoding: GPUTerrainFlowDirectionEncoding = 'esri'
): number {
  if (word === 0 || word >= TERRAIN_DRAINAGE_UNRESOLVED || encoding === 'esri') return word;
  if (encoding === 'taudem') return word <= 8 ? 1 << ((9 - word) % 8) : 256;
  if (word > 128 || (word & (word - 1)) !== 0) return 256;
  return word === 1 ? 128 : word >>> 1;
}

/** Returns a receiver index, or NONE for a terminal, sentinel, bad code or tile-edge exit. @internal */
export function getTerrainDrainageReceiver(
  cell: number,
  code: number,
  width: number,
  height: number
): number {
  const direction = GPU_RASTER_D8_DIRECTIONS.find(entry => entry.code === code);
  if (!direction) return TERRAIN_DRAINAGE_NONE;
  const column = (cell % width) + direction.columnOffset;
  const row = Math.floor(cell / width) + direction.rowOffset;
  return column < 0 || row < 0 || column >= width || row >= height
    ? TERRAIN_DRAINAGE_NONE
    : row * width + column;
}
