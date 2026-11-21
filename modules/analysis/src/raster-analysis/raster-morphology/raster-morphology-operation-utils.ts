// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';
import {
  snapshotRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';

/** Captures documented operation fields, including inherited getters. @internal */
export function captureRasterMorphologyOperationProps<
  Props extends {
    id?: string;
    grid: GPURasterMetadata;
    marker?: string | RasterOperationBand;
    mask?: string | RasterOperationBand;
    input?: string;
    method?: 'dilation' | 'erosion';
    connectivity?: 4 | 8;
    maximumIterations?: number;
    output: {
      values: string;
      validity?: string;
      status: {converged: string; iterationLimitReached?: string; iterationCount?: string};
    };
  }
>(props: Props, operation: 'reconstruction' | 'fill-holes' = 'fill-holes'): Props {
  const captured = snapshotConstructionFields(
    props,
    operation === 'reconstruction'
      ? ['id', 'grid', 'marker', 'mask', 'method', 'connectivity', 'maximumIterations', 'output']
      : ['id', 'grid', 'input', 'connectivity', 'maximumIterations', 'output']
  );
  if (operation === 'reconstruction') {
    validateRequirement(
      captured.id ?? 'raster-reconstruction',
      captured.method === undefined ||
        captured.method === 'dilation' ||
        captured.method === 'erosion',
      'method must be dilation or erosion'
    );
  }
  const output = snapshotConstructionFields(
    captured.output,
    operation === 'reconstruction' ? ['values', 'validity', 'status'] : ['values', 'status']
  );
  return Object.freeze({
    ...captured,
    grid: snapshotRasterGrid(captured.grid),
    ...(operation === 'reconstruction' && captured.marker !== undefined
      ? {marker: captureOperationBand(captured.marker)}
      : {}),
    ...(operation === 'reconstruction' && captured.mask !== undefined
      ? {mask: captureOperationBand(captured.mask)}
      : {}),
    output: Object.freeze({
      ...output,
      status: snapshotConstructionFields(output.status, [
        'converged',
        'iterationLimitReached',
        'iterationCount'
      ])
    })
  }) as Props;
}

function captureOperationBand(input: string | RasterOperationBand): string | RasterOperationBand {
  return typeof input === 'string' ? input : snapshotRasterOperationBand(input);
}
/** Validates scalar morphology options at construction and at the pure-function boundary. @internal */
export function validateRasterMorphologyOptions(
  id: string,
  options: {grid: GPURasterMetadata; connectivity?: 4 | 8; maximumIterations?: number}
): void {
  validateCapacity(id, 'grid.width', options.grid.width);
  validateCapacity(id, 'grid.height', options.grid.height);
  validateCapacity(id, 'grid cells', options.grid.width * options.grid.height);
  validateRequirement(
    id,
    options.grid.affine.length === 6 && options.grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite values'
  );
  validateRequirement(
    id,
    options.connectivity === undefined || options.connectivity === 4 || options.connectivity === 8,
    'connectivity must be 4 or 8'
  );
  // The command-node budget is GPU-only. The round count still uses a uint32 output word.
  if (options.maximumIterations !== undefined)
    validateCapacity(id, 'maximumIterations', options.maximumIterations);
}

/** Visits the selected immediate neighbors in raster order, excluding the cell itself. @internal */
export function visitRasterMorphologyNeighbors(
  cell: number,
  width: number,
  height: number,
  connectivity: 4 | 8,
  visit: (neighbor: number) => void
): void {
  const row = Math.floor(cell / width);
  const column = cell % width;
  for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
    for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
      if (
        (rowOffset === 0 && columnOffset === 0) ||
        (connectivity === 4 && Math.abs(rowOffset) + Math.abs(columnOffset) > 1)
      )
        continue;
      const neighborRow = row + rowOffset;
      const neighborColumn = column + columnOffset;
      if (neighborRow >= 0 && neighborRow < height && neighborColumn >= 0 && neighborColumn < width)
        visit(neighborRow * width + neighborColumn);
    }
  }
}
