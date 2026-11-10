// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUGridIndexBounds,
  type GPUGridIndexPositions,
  type GPUGridIndexSize,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/**
 * WGSL `getGridUnitPosition`: maps a point onto the unit domain the published grid index takes
 * as a literal, with the per-frame lattice extended to `COLUMNS x ROWS` whole cells, so the index
 * cell of a point agrees with `getCellColumn` and `getCellRow`. Requires a `Lattice` with
 * `minimumX`, `minimumY`, `maximumX`, `maximumY`, `cellWidth`, `cellHeight` and the constants
 * `COLUMNS` and `ROWS`.
 *
 * @internal
 */
export const POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL = /* wgsl */ `
fn getGridUnitPosition(lattice: Lattice, x: f32, y: f32) -> vec2f {
  let width = max(lattice.maximumX, lattice.minimumX + lattice.cellWidth * f32(COLUMNS)) - lattice.minimumX;
  let height = max(lattice.maximumY, lattice.minimumY + lattice.cellHeight * f32(ROWS)) - lattice.minimumY;
  return vec2f((x - lattice.minimumX) / width, (y - lattice.minimumY) / height);
}`;

/** Views written by {@link createPointNeighborGridIndexNodes}. @internal */
export type PointNeighborGridIndex<Parameters> = {
  /** The index build, in execution order. */
  nodes: readonly GPUCommandNode<Parameters>[];
  /** `cellCount + 1` exclusive offsets of the cells. */
  cellOffsets: GraphDataView<'uint32'>;
  /** Row IDs grouped by cell, in unspecified order inside a cell. */
  sortedRows: GraphDataView<'uint32'>;
  /** One-row count of the rows the index accepted. */
  gridCount: GraphDataView<'uint32'>;
  /** One-row flag set when the accepted rows exceed `rowCount` (never, for a row-sized index). */
  gridOverflow: GraphDataView<'uint32'>;
};

/**
 * Builds the point grid every neighbor scan in the point family reads: `GPUNeighborSearch`,
 * `GPUSpatialClustering` and `GPUKernelDensity`. It owns only the index and its scratch. Each
 * contributor keeps its own lattice rule (radius-wide cells, epsilon-wide cells, or a fixed domain)
 * and its own walk over the cells, because those differ by what the contributor proves about a cell.
 *
 * Two dimensions, or three when a time axis is indexed as a third lattice dimension. Positions that
 * are non-finite or outside `bounds` are skipped by the index, which is how contributors exclude rows.
 *
 * @internal
 */
export function createPointNeighborGridIndexNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    /** Prefix of the transient IDs: `${id}-cell-offsets` and so on. */
    id: string;
    /** ID of the index nodes. Defaults to `${id}-grid-index`. */
    indexId?: string;
    positions: GPUGridIndexPositions;
    gridSize: GPUGridIndexSize;
    /** Defaults to the unit domain of the matching dimension. */
    bounds?: GPUGridIndexBounds;
    /** Rows the index may accept; sizes `sortedRows`. */
    rowCount: number;
  }
): PointNeighborGridIndex<Parameters> {
  const {id, positions, gridSize, rowCount} = props;
  const cellCount = gridSize.reduce((product, size) => product * size, 1);
  const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rowCount);
  const gridCount = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
  const gridOverflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
  const bounds: GPUGridIndexBounds =
    props.bounds ?? (gridSize.length === 3 ? [0, 0, 0, 1, 1, 1] : [0, 0, 1, 1]);
  return {
    nodes: new GPUGridIndex({
      id: props.indexId ?? `${id}-grid-index`,
      positions,
      gridSize,
      bounds,
      cellOffsets,
      objectIds: sortedRows,
      count: gridCount,
      overflow: gridOverflow
    }).getCommandNodes(graph),
    cellOffsets,
    sortedRows,
    gridCount,
    gridOverflow
  };
}
