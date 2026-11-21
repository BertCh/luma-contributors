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
 * WGSL `getGridUnitPosition`: encodes the cell of a point, as the caller's own `getCellColumn`
 * and `getCellRow` compute it, as a position on the unit domain the published grid index takes as
 * a literal. The index therefore files every point in exactly the cell the caller's walk reads;
 * the lattice has one cell definition. (Rescaling the point instead by an extended lattice width
 * rounded `minimum + cellWidth * COLUMNS` at projected origins, so near 1e6 the index and the walk
 * could differ by a cell and a 3x3 walk missed a neighbor one epsilon away.) Requires a `Lattice`
 * and `getCellColumn(lattice, x) -> u32`, `getCellRow(lattice, y) -> u32` returning a cell below
 * the constants `COLUMNS` and `ROWS`.
 *
 * @internal
 */
export const POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL = /* wgsl */ `
// Centre of a cell of an axis of size cells on the unit domain. The published getCoordinate
// truncates unit * f32(size), a correctly rounded product; division may be less exact, so a
// centre that rounds into a neighbor (only on axes of about a million cells) moves one float per
// attempt. Exact for axes up to 2^23 cells.
fn getGridCellUnitCoordinate(cell: u32, size: u32) -> f32 {
  var unit = (f32(cell) + 0.5) / f32(size);
  for (var attempt = 0u; attempt < 4u; attempt++) {
    let mapped = u32(floor(unit * f32(size)));
    if (mapped < cell) {
      unit = bitcast<f32>(bitcast<u32>(unit) + 1u);
    } else if (mapped > cell) {
      unit = bitcast<f32>(bitcast<u32>(unit) - 1u);
    }
  }
  return unit;
}

fn getGridUnitPosition(lattice: Lattice, x: f32, y: f32) -> vec2f {
  return vec2f(
    getGridCellUnitCoordinate(getCellColumn(lattice, x), COLUMNS),
    getGridCellUnitCoordinate(getCellRow(lattice, y), ROWS)
  );
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
 * `GPUDBSCAN` and `GPUKernelDensity`. It owns only the index and its scratch. Each
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
