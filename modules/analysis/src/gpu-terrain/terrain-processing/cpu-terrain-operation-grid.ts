// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {validateTerrainDrainageOptions} from '../hydrology/cpu-terrain-drainage-utils';

/** Largest packed scalar output length: graph shapes and shader offsets use uint32. @internal */
export const TERRAIN_OPERATION_MAXIMUM_WORD_COUNT = 0xffffffff;

/** Validates the grid and the row-constant ground model of terrain processing. @internal */
export function validateCPUTerrainOperationGrid(
  id: string,
  grid: GPURasterMetadata,
  cellSizeMode?: GPURasterCellSizeMode
): number {
  const count = validateTerrainDrainageOptions(id, grid);
  validateRasterCellSizeMode(id, cellSizeMode);
  validateRequirement(
    id,
    (cellSizeMode ?? 'uniform') === 'uniform' || (grid.affine[1] === 0 && grid.affine[3] === 0),
    'grid.affine must be axis-aligned for geographic and web-mercator cellSizeMode'
  );
  return count;
}

/** Float64 counterpart of getTerrainGroundMeasureWGSL, including its midpoint row. @internal */
export function getCPUTerrainGroundMeasures(
  grid: GPURasterMetadata,
  cellSizeMode: GPURasterCellSizeMode = 'uniform'
): {edgeLength(cell: number, other: number): number; cellArea(cell: number): number} {
  const georeference = getRasterGridGeoreference(grid);
  const [columnX, rowX, , columnY, rowY] = grid.affine;
  return {
    edgeLength(cell, other) {
      const columns = (other % grid.width) - (cell % grid.width);
      const cellRow = Math.floor(cell / grid.width);
      const otherRow = Math.floor(other / grid.width);
      const rows = otherRow - cellRow;
      if (cellSizeMode === 'uniform') {
        return Math.hypot(columnX * columns + rowX * rows, columnY * columns + rowY * rows);
      }
      const [groundX, groundY] = getRasterGroundCellSize(
        cellSizeMode,
        georeference,
        (cellRow + otherRow + 1) * 0.5,
        grid.height
      );
      return Math.hypot(groundX * columns, groundY * rows);
    },
    cellArea(cell) {
      if (cellSizeMode === 'uniform') return Math.abs(columnX * rowY - rowX * columnY);
      const [groundX, groundY] = getRasterGroundCellSize(
        cellSizeMode,
        georeference,
        Math.floor(cell / grid.width) + 0.5,
        grid.height
      );
      return groundX * groundY;
    }
  };
}
