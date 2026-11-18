// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Cells consume and return longitude/latitude. Geographic runs set their camera bounds
// to the same window as their input coordinates.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import {GPUCellGeometry, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Fixed boundary stride per cell: wide enough for H3 cells with distortion vertices. */
export const H3_BOUNDARY_STRIDE = 10;

/** Triangle fans over the first `vertexCount` boundary slots of each cell, with the owner row. */
export function createCellFanMesh(
  cellCount: number,
  stride: number,
  vertexCount: number,
  ownerStride = 1,
  ownerOffset = 0
): {indices: Uint32Array; owners: Uint32Array; triangleCount: number} {
  const trianglesPerCell = vertexCount - 2;
  const indices = new Uint32Array(cellCount * trianglesPerCell * 3);
  const owners = new Uint32Array(cellCount * trianglesPerCell);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let corner = 0; corner < trianglesPerCell; corner++) {
      const triangle = cell * trianglesPerCell + corner;
      indices[triangle * 3] = cell * stride;
      indices[triangle * 3 + 1] = cell * stride + corner + 1;
      indices[triangle * 3 + 2] = cell * stride + corner + 2;
      owners[triangle] = cell * ownerStride + ownerOffset;
    }
  }
  return {indices, owners, triangleCount: owners.length};
}

/** Degenerate triangles (a, b, b) whose wireframe is the closed boundary of each cell. */
export function createCellWireframe(
  cellCount: number,
  stride: number,
  vertexCount: number
): Uint32Array {
  const indices = new Uint32Array(cellCount * vertexCount * 3);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let corner = 0; corner < vertexCount; corner++) {
      const triangle = cell * vertexCount + corner;
      const next = (corner + 1) % vertexCount;
      indices[triangle * 3] = cell * stride + corner;
      indices[triangle * 3 + 1] = cell * stride + next;
      indices[triangle * 3 + 2] = cell * stride + next;
    }
  }
  return indices;
}

/** Adds a GPUCellGeometry node for `cells` and returns its centers and boundaries. */
export function addCellGeometry(
  builder: RunBuilder,
  family: 'h3' | 'quadbin' | 'quadkey' | 'geohash' | 's2',
  cells: RunBuffer<'uint32x2'>,
  prefix = 'draw',
  maximumVertexCount?: number
): {
  centers: RunBuffer<'float32x2'>;
  boundaries: RunBuffer<'float32x2'>;
  stride: number;
  vertexCount: number;
} {
  const stride = maximumVertexCount ?? (family === 'h3' ? H3_BOUNDARY_STRIDE : 4);
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
  return {centers, boundaries, stride, vertexCount: family === 'h3' ? 6 : 4};
}

export type CellFamily = 'h3' | 'quadbin';

/** Parameter-slider choice of the two families the runs draw. */
export function getCellFamily(familyValue: number): CellFamily {
  return familyValue === 0 ? 'h3' : 'quadbin';
}

/** H3 resolution `resolution` or the Quadbin zoom that draws cells of a similar size. */
export function getCellResolution(family: CellFamily, resolution: number): number {
  return family === 'h3' ? resolution : resolution + 5;
}

export type CellTableBuffers = {
  cells: RunBuffer<'uint32x2'>;
  counts: RunBuffer<'uint32'>;
  sums: RunBuffer<'uint32x2'>;
  sumValues: RunBuffer<'float32'>;
  minimums: RunBuffer<'float32'>;
  maximums: RunBuffer<'float32'>;
  count: RunBuffer<'uint32'>;
  overflow: RunBuffer<'uint32'>;
  candidateOverflow: RunBuffer<'uint32'>;
  requiredCount: RunBuffer<'uint32'>;
};

/** Allocates every column of a GPUCellTable with room for `capacity` cells. */
export function allocateCellTable(
  builder: RunBuilder,
  prefix: string,
  capacity: number
): CellTableBuffers {
  return {
    cells: builder.allocate(`${prefix}-cells`, 'uint32x2', capacity),
    counts: builder.allocate(`${prefix}-counts`, 'uint32', capacity),
    sums: builder.allocate(`${prefix}-sums`, 'uint32x2', capacity),
    sumValues: builder.allocate(`${prefix}-sum-values`, 'float32', capacity),
    minimums: builder.allocate(`${prefix}-minimums`, 'float32', capacity),
    maximums: builder.allocate(`${prefix}-maximums`, 'float32', capacity),
    count: builder.allocate(`${prefix}-count`, 'uint32', 1),
    overflow: builder.allocate(`${prefix}-overflow`, 'uint32', 1),
    candidateOverflow: builder.allocate(`${prefix}-candidate-overflow`, 'uint32', 1),
    requiredCount: builder.allocate(`${prefix}-required-count`, 'uint32', 1)
  };
}

/** The GPUCellTable props view of allocated table buffers. */
export function getCellTableViews(table: CellTableBuffers) {
  return {
    ...getCountTableViews(table),
    sums: table.sums.view,
    sumValues: table.sumValues.view,
    minimums: table.minimums.view,
    maximums: table.maximums.view
  };
}

/** The cells, counts and status columns only (an aggregation without point values). */
export function getCountTableViews(table: CellTableBuffers) {
  return {
    cells: table.cells.view,
    counts: table.counts.view,
    status: {
      count: table.count.view,
      overflow: table.overflow.view,
      candidateOverflow: table.candidateOverflow.view,
      requiredCount: table.requiredCount.view
    }
  };
}

export type CellFillBuffers = {
  boundaries: RunBuffer<'float32x2'>;
  centers: RunBuffer<'float32x2'>;
  /** Boundary slots per cell, and the number of them that are corners. */
  stride: number;
  vertexCount: number;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  triangleCount: number;
  /** Degenerate triangles whose wireframe is each cell's boundary (`triangles` layer). */
  wireframe: RunBuffer<'uint32'>;
  wireframeCount: number;
};

/**
 * Decodes `cells` for drawing (a real GPUCellGeometry node) and uploads the fan triangles. The
 * owner of cell `row` is `row * ownerStride + ownerOffset`, so a polygon can read its value from an
 * interleaved column such as a `float32x2` buffer.
 */
export function addCellFill(
  builder: RunBuilder,
  family: CellFamily | 's2',
  cells: RunBuffer<'uint32x2'>,
  prefix = 'fill',
  ownerStride = 1,
  ownerOffset = 0
): CellFillBuffers {
  const geometry = addCellGeometry(builder, family, cells, prefix);
  const mesh = createCellFanMesh(
    cells.length,
    geometry.stride,
    geometry.vertexCount,
    ownerStride,
    ownerOffset
  );
  const wireframe = createCellWireframe(cells.length, geometry.stride, geometry.vertexCount);
  return {
    boundaries: geometry.boundaries,
    centers: geometry.centers,
    stride: geometry.stride,
    vertexCount: geometry.vertexCount,
    indices: builder.upload(`${prefix}-indices`, 'uint32', mesh.indices),
    owners: builder.upload(`${prefix}-owners`, 'uint32', mesh.owners),
    triangleCount: mesh.triangleCount,
    wireframe: builder.upload(`${prefix}-wireframe`, 'uint32', wireframe),
    wireframeCount: wireframe.length / 3
  };
}

/** Keys `positions` with a real GPUPointToCell node (the aggregation ops take keys for H3). */
export function addPointKeys(
  builder: RunBuilder,
  family: CellFamily | 's2',
  resolution: number,
  positions: RunBuffer<'float32x2'>,
  prefix = 'keys'
): RunBuffer<'uint32x2'> {
  const cells = builder.allocate(`${prefix}-cells`, 'uint32x2', positions.length);
  builder.graph.add(
    new GPUPointToCell({
      id: `${prefix}-point-to-cell`,
      family,
      resolution,
      positions: positions.view,
      output: {cells: cells.view}
    })
  );
  return cells;
}

export type PointerCell = {
  /** The cell key of the pointer position, decoded by the same two operations as the data. */
  cells: RunBuffer<'uint32x2'>;
  centers: RunBuffer<'float32x2'>;
  boundaries: RunBuffer<'float32x2'>;
  vertexCounts: RunBuffer<'uint32'>;
  stride: number;
  vertexCount: number;
  /** Fan triangles over the one cell, for a bright fill. */
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  triangleCount: number;
};

/** The polygons layer that fills the cell under the pointer. */
export function drawPointerCell(
  pointerCell: PointerCell,
  color: readonly [number, number, number, number] = [1, 0.82, 0.3, 0.55]
) {
  return {
    layer: 'polygons' as const,
    positions: pointerCell.boundaries,
    indices: pointerCell.indices,
    owners: pointerCell.owners,
    values: pointerCell.owners,
    triangleCount: pointerCell.triangleCount,
    color
  };
}

/**
 * Keys a one-row pointer position with GPUPointToCell and decodes it with GPUCellGeometry, so the
 * cell under the cursor is answered by the same operations that key the data (no CPU cell library).
 */
export function addPointerCell(
  builder: RunBuilder,
  family: CellFamily | 's2',
  resolution: number,
  position: RunBuffer<'float32x2'>,
  prefix = 'pointer'
): PointerCell {
  const cells = addPointKeys(builder, family, resolution, position, prefix);
  const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
  const centers = builder.allocate(`${prefix}-centers`, 'float32x2', position.length);
  const boundaries = builder.allocate(
    `${prefix}-boundaries`,
    'float32x2',
    position.length * stride
  );
  const vertexCounts = builder.allocate(`${prefix}-vertex-counts`, 'uint32', position.length);
  builder.graph.add(
    new GPUCellGeometry({
      id: `${prefix}-geometry`,
      family,
      cells: cells.view,
      maximumVertexCount: stride,
      output: {
        centers: centers.view,
        boundaries: boundaries.view,
        vertexCounts: vertexCounts.view
      }
    })
  );
  const vertexCount = family === 'h3' ? 6 : 4;
  const mesh = createCellFanMesh(position.length, stride, vertexCount);
  return {
    cells,
    centers,
    boundaries,
    vertexCounts,
    stride,
    vertexCount,
    indices: builder.upload(`${prefix}-fill-indices`, 'uint32', mesh.indices),
    owners: builder.upload(`${prefix}-fill-owners`, 'uint32', mesh.owners),
    triangleCount: mesh.triangleCount
  };
}
