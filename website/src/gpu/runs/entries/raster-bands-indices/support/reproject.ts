// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Grids for the reprojection runs: a destination grid that is the source grid rotated and scaled
// about its center, so resampling into it turns the picture.

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {RasterInput} from '../../../inputs';

/** Destination grid: same size as the source, rotated by `degrees` and zoomed out by `scale`. */
export function makeRotatedGrid(
  source: RasterInput,
  degrees: number,
  scale: number
): GPURasterMetadata {
  const {columns, rows, cellSize} = source;
  const angle = (degrees * Math.PI) / 180;
  const step = cellSize * scale;
  const centerX = (columns * cellSize) / 2;
  const centerY = (rows * cellSize) / 2;
  const columnX = step * Math.cos(angle);
  const columnY = step * Math.sin(angle);
  const rowX = step * Math.sin(angle);
  const rowY = -step * Math.cos(angle);
  return {
    width: columns,
    height: rows,
    affine: [
      columnX,
      rowX,
      centerX - (columns / 2) * columnX - (rows / 2) * rowX,
      columnY,
      rowY,
      centerY - (columns / 2) * columnY - (rows / 2) * rowY
    ],
    pixelInterpretation: 'area'
  };
}

/** A source raster with a coordinate grid painted on it, so rotation and scale are visible. */
export function paintGridLines(raster: RasterInput, spacing = 8): Float32Array {
  return Float32Array.from(raster.values, (value, index) => {
    const column = index % raster.columns;
    const row = Math.floor(index / raster.columns);
    return column % spacing === 0 || row % spacing === 0 ? 1 : value * 0.7;
  });
}
