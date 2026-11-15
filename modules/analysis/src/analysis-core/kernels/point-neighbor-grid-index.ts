// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GPUGroupAggregation,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUGridIndexBounds,
  type GPUGridIndexPositions,
  type GPUGridIndexSize,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

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
  /** Row IDs grouped by cell; stable input order for keyed input, unspecified for positions. */
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
 * Pair statistics can supply exact integer cell keys instead of positions: this preserves their
 * per-frame lattice (including overflowing legacy extents) and stable input-row order without a
 * second floating-point cell assignment. That path uses published aggregation, scan and sort.
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
    /** Rows the index may accept; sizes `sortedRows`. */
    rowCount: number;
  } & (
    | {
        positions: GPUGridIndexPositions;
        gridSize: GPUGridIndexSize;
        /** Defaults to the unit domain of the matching dimension. */
        bounds?: GPUGridIndexBounds;
      }
    | {
        /** Exact cell keys in [0, cellCount); cellCount is the excluded-row sentinel. */
        cellKeys: GraphDataView<'uint32'>;
        /** Ascending input row IDs, aligned with cellKeys. */
        sourceRows: GraphDataView<'uint32'>;
        cellCount: number;
      }
  )
): PointNeighborGridIndex<Parameters> {
  const {id, rowCount} = props;
  const cellCount =
    'cellKeys' in props
      ? props.cellCount
      : props.gridSize.reduce((product, size) => product * size, 1);
  const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rowCount);
  const gridCount = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
  const gridOverflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
  if ('cellKeys' in props) {
    const cellCounts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rowCount);
    return {
      nodes: [
        ...new GPUGroupAggregation({
          id: `${id}-cell-counts`,
          keys: props.cellKeys,
          output: cellCounts
        }).getCommandNodes(graph),
        ...new GPUScan({
          id: `${id}-cell-scan`,
          input: cellCounts,
          output: cellOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cell-total`,
          operation: 'PointNeighborGridIndex',
          variant: 'cell-total',
          bindings: [
            {name: 'counts', view: cellCounts, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read_write'},
            {name: 'gridCount', view: gridCount, type: 'u32', access: 'read_write'},
            {name: 'gridOverflow', view: gridOverflow, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const LAST_CELL: u32 = ${cellCount - 1}u;`,
          body: `let count = cellOffsets[cellOffsetsOffset + LAST_CELL] + counts[countsOffset + LAST_CELL];
  cellOffsets[cellOffsetsOffset + LAST_CELL + 1u] = count;
  gridCount[gridCountOffset] = count;
  gridOverflow[gridOverflowOffset] = 0u;`
        }),
        ...new GPUSort({
          id: props.indexId ?? `${id}-cell-sort`,
          keys: props.cellKeys,
          values: props.sourceRows,
          outputKeys: sortedKeys,
          outputValues: sortedRows,
          keyBits: Math.max(1, Math.ceil(Math.log2(cellCount + 1)))
        }).getCommandNodes(graph)
      ],
      cellOffsets,
      sortedRows,
      gridCount,
      gridOverflow
    };
  }
  const {positions, gridSize} = props;
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
