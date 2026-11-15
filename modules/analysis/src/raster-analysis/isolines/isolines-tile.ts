// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Two raster-edge families must fit in uint32 without using the missing-edge sentinel. @internal */
export const ISOLINES_MAXIMUM_SAMPLE_COUNT = Math.floor(0xffffffff / 2);
// Counts and scan offsets use uint32, including up to two segments per cell and level.
const MAXIMUM_REQUIRED_SEGMENT_COUNT = 0xffffffff;

/**
 * Compile-time placement and half-open marching-cell ownership of a halo-padded raster tile.
 * All tiles of a domain use the same levels and packed domain placement parameters.
 */
export type GPUIsolinesTile = {
  /** Sample dimensions of the whole domain, each at least two. */
  domainWidth: number;
  domainHeight: number;
  /** Domain sample coordinate of local sample (0, 0), including the tile's halo. */
  columnOffset: number;
  rowOffset: number;
  /**
   * Local marching-cell lower-left coordinates in [minColumn, minRow, maxColumn, maxRow).
   * Maxima may equal grid.width - 1 / grid.height - 1. Empty rectangles emit nothing.
   * Partition domain cells exactly once; halo cells outside this rectangle are read-only.
   */
  ownedCells: readonly [number, number, number, number];
};

/** Checks tile bounds once at construction, including halo samples and the edge-ID limit. @internal */
export function validateIsolinesTile(
  id: string,
  grid: Pick<GPURasterMetadata, 'width' | 'height'>,
  tile: GPUIsolinesTile | undefined
): void {
  validateCapacity(id, 'grid.width', grid.width, 2);
  validateCapacity(id, 'grid.height', grid.height, 2);
  validateMaximumCount(id, 'grid samples', grid.width * grid.height, ISOLINES_MAXIMUM_SAMPLE_COUNT);
  if (!tile) return;
  validateCapacity(id, 'tile.domainWidth', tile.domainWidth, 2);
  validateCapacity(id, 'tile.domainHeight', tile.domainHeight, 2);
  validateMaximumCount(
    id,
    'tile domain samples',
    tile.domainWidth * tile.domainHeight,
    ISOLINES_MAXIMUM_SAMPLE_COUNT
  );
  validateCapacity(id, 'tile.columnOffset', tile.columnOffset, 0, tile.domainWidth - grid.width);
  validateCapacity(id, 'tile.rowOffset', tile.rowOffset, 0, tile.domainHeight - grid.height);
  validateRequirement(id, tile.ownedCells.length === 4, 'tile.ownedCells must contain four bounds');
  const [minColumn, minRow, maxColumn, maxRow] = tile.ownedCells;
  validateCapacity(id, 'tile.ownedCells.minColumn', minColumn, 0, grid.width - 1);
  validateCapacity(id, 'tile.ownedCells.minRow', minRow, 0, grid.height - 1);
  validateCapacity(id, 'tile.ownedCells.maxColumn', maxColumn, minColumn, grid.width - 1);
  validateCapacity(id, 'tile.ownedCells.maxRow', maxRow, minRow, grid.height - 1);
}

/** Checks the uint32 compact-count bound from the maximum level count, not per-frame values. @internal */
export function validateIsolinesMaximumSegmentCount(
  id: string,
  grid: Pick<GPURasterMetadata, 'width' | 'height'>,
  tile: GPUIsolinesTile | undefined,
  maximumLevelCount: number
): void {
  const [minColumn, minRow, maxColumn, maxRow] = tile?.ownedCells ?? [
    0,
    0,
    grid.width - 1,
    grid.height - 1
  ];
  validateMaximumCount(
    id,
    'owned cells * levels.length * 2 required segments',
    (maxColumn - minColumn) * (maxRow - minRow) * maximumLevelCount * 2,
    MAXIMUM_REQUIRED_SEGMENT_COUNT
  );
}
