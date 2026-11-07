// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Drawing helpers for cell tables. The drawing world doubles as a longitude/latitude window in
// degrees, so Quadbin cells decoded by GPUCellGeometry (a real operation) draw directly.

import {GPUCellGeometry} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';

export type CellFill = {
  centers: RunBuffer<'float32x2'>;
  boundaries: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  triangleCount: number;
};

/** Boundary slots per cell: Quadbin cells have four corners, H3 cells up to ten with distortion. */
const BOUNDARY_STRIDE = {quadbin: 4, h3: 10} as const;
const BOUNDARY_VERTICES = {quadbin: 4, h3: 6} as const;

/** Decodes `cells` into centers and corner positions and uploads a triangle fan per cell. */
export function addCellFill(
  builder: RunBuilder,
  family: 'quadbin' | 'h3',
  cells: RunBuffer<'uint32x2'>,
  prefix = 'fill'
): CellFill {
  const stride = BOUNDARY_STRIDE[family];
  const vertexCount = BOUNDARY_VERTICES[family];
  const centers = builder.allocate(`${prefix}-centers`, 'float32x2', cells.length);
  const boundaries = builder.allocate(`${prefix}-boundaries`, 'float32x2', cells.length * stride);
  builder.graph.add(
    new GPUCellGeometry({
      id: `${prefix}-geometry`,
      family,
      cells: cells.view,
      maximumVertexCount: stride,
      output: {centers: centers.view, boundaries: boundaries.view}
    })
  );
  const trianglesPerCell = vertexCount - 2;
  const indices = new Uint32Array(cells.length * trianglesPerCell * 3);
  const owners = new Uint32Array(cells.length * trianglesPerCell);
  for (let cell = 0; cell < cells.length; cell++) {
    for (let corner = 0; corner < trianglesPerCell; corner++) {
      const triangle = cell * trianglesPerCell + corner;
      indices.set(
        [cell * stride, cell * stride + corner + 1, cell * stride + corner + 2],
        triangle * 3
      );
      owners[triangle] = cell;
    }
  }
  return {
    centers,
    boundaries,
    indices: builder.upload(`${prefix}-indices`, 'uint32', indices),
    owners: builder.upload(`${prefix}-owners`, 'uint32', owners),
    triangleCount: owners.length
  };
}
