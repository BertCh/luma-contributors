// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createGridCellRowOrderNodes} from '../shared/grid-cell-row-order';
import {BIN_INDEX_WGSL} from '../shared/bin-index-wgsl';
import {IS_FINITE_BITS_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION
} from './geographically-weighted-regression-parameters';

/** Shared automatic GWR grid: about eight rows per cell, starting at 1024 rows. @internal */
export function getDefaultGWRGridSize(rowCount: number): readonly [number, number] | undefined {
  if (rowCount < GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT) return undefined;
  const dimension = Math.min(
    Math.max(1, Math.ceil(Math.sqrt(rowCount / 8))),
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION
  );
  return [dimension, dimension];
}

/** Rows summed by one thread of the extent pass. */
const BOUNDS_TILE_ROWS = 256;

/** Views produced by {@link createGridIndexNodes}. @internal */
export type GridIndexViews = {
  /** `cellCount + 1` slot offsets of the grid cells. */
  cellOffsets: GraphDataView<'uint32'>;
  /**
   * `rowCount + 4` words: row IDs in ascending order inside each cell, then the grid extent
   * `[minX, minY, maxX, maxY]` as float bits (kernels are at the storage-binding limit, so the
   * extent travels after the IDs).
   */
  sortedIds: GraphDataView<'uint32'>;
};

/**
 * Builds the shared GWR-family grid index over `positions`, where a non-finite position excludes
 * the row: the extent of the finite rows, a `GPUGridIndex` over that extent, and an in-cell rank
 * sort that restores ascending row order (the index scatters IDs with atomics, so in-cell order
 * varies between runs and a float sum must not depend on it).
 *
 * @internal
 */
export function createGridIndexNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    rowCount: number;
    gridSize: readonly [number, number];
  }
): {nodes: GPUCommandNode<Parameters>[]} & GridIndexViews {
  const {id, operation, positions, rowCount, gridSize} = props;
  const tileCount = Math.ceil(rowCount / BOUNDS_TILE_ROWS);
  const cellCount = gridSize[0] * gridSize[1];
  const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
  const tileBounds = transient('tile-bounds', 'float32', tileCount * 4);
  const gridBounds = transient('grid-bounds', 'float32', 4);
  const unitPositions = transient('unit-positions', 'float32x2', rowCount);
  const cellOffsets = transient('cell-offsets', 'uint32', cellCount + 1);
  const objectIds = transient('object-ids', 'uint32', rowCount);
  const sortedIds = transient('sorted-ids', 'uint32', rowCount + 4);
  const indexCount = transient('index-count', 'uint32', 1);
  const indexOverflow = transient('index-overflow', 'uint32', 1);
  const common = `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${BOUNDS_TILE_ROWS}u;
const TILE_COUNT: u32 = ${tileCount}u;
const SENTINEL: f32 = 3.0e38;
${IS_FINITE_BITS_WGSL}`;
  // The grid index scatters IDs with atomics, so in-cell order varies between runs. A stable sort
  // of the rows by cell restores ascending row order inside every cell in O(rows) work, however
  // many rows share a cell; rows the index excluded sort last.
  const cellRowOrder = createGridCellRowOrderNodes(graph, {
    id: `${id}-cell-row-order`,
    operation,
    cellOffsets,
    objectIds,
    indexCount,
    rowCount
  });
  const sortedRows = cellRowOrder.sortedRows;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-tile-bounds`,
      operation,
      variant: 'tile-bounds',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'tileBounds', view: tileBounds, type: 'f32', access: 'read_write'}
      ],
      invocationCount: tileCount,
      declarations: common,
      body: `let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var bounds = vec4f(SENTINEL, SENTINEL, -SENTINEL, -SENTINEL);
  for (var row = firstRow; row < endRow; row++) {
    let x = positions[positionsOffset + 2u * row];
    let y = positions[positionsOffset + 2u * row + 1u];
    if (isFiniteBits(x) && isFiniteBits(y)) {
      bounds = vec4f(min(bounds.x, x), min(bounds.y, y), max(bounds.z, x), max(bounds.w, y));
    }
  }
  for (var component = 0u; component < 4u; component++) {
    tileBounds[tileBoundsOffset + 4u * index + component] = bounds[component];
  }`
    }),
    // Extent of the included rows; non-finite when there are none, so the index accepts nothing.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-grid-bounds`,
      operation,
      variant: 'grid-bounds',
      bindings: [
        {name: 'tileBounds', view: tileBounds, type: 'f32', access: 'read'},
        {name: 'gridBounds', view: gridBounds, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: common,
      body: `var bounds = vec4f(SENTINEL, SENTINEL, -SENTINEL, -SENTINEL);
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let base = tileBoundsOffset + 4u * tile;
    bounds = vec4f(
      min(bounds.x, tileBounds[base]), min(bounds.y, tileBounds[base + 1u]),
      max(bounds.z, tileBounds[base + 2u]), max(bounds.w, tileBounds[base + 3u])
    );
  }
  let isEmpty = bounds.x > bounds.z || bounds.y > bounds.w;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  for (var component = 0u; component < 4u; component++) {
    gridBounds[gridBoundsOffset + component] = select(bounds[component], invalid, isEmpty);
  }`
    }),
    // The published grid index takes a literal [0, 1] domain, so positions are normalized by the
    // data-dependent extent first; non-finite or empty-extent rows become NaN and are ignored.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unit-positions`,
      operation,
      variant: 'unit-positions',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'gridBounds', view: gridBounds, type: 'f32', access: 'read'},
        {name: 'unitPositions', view: unitPositions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      declarations: `${IS_FINITE_VALUE_WGSL}${BIN_INDEX_WGSL}`,
      body: `let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let minimum = vec2f(gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 1u]);
  let maximum = vec2f(gridBounds[gridBoundsOffset + 2u], gridBounds[gridBoundsOffset + 3u]);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let extent = maximum - minimum;
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) && isFiniteValue(minimum.x) && isFiniteValue(minimum.y);
  let unit = select(vec2f(0.0), (position - minimum) / extent, extent > vec2f(0.0));
  unitPositions[unitPositionsOffset + 2u * index] = select(invalid, unit.x, included);
  unitPositions[unitPositionsOffset + 2u * index + 1u] = select(invalid, unit.y, included);`
    }),
    ...new GPUGridIndex({
      id: `${id}-grid-index`,
      positions: unitPositions,
      gridSize: [gridSize[0], gridSize[1]],
      bounds: [0, 0, 1, 1],
      cellOffsets,
      objectIds,
      count: indexCount,
      overflow: indexOverflow
    }).getCommandNodes(graph),
    ...cellRowOrder.nodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-publish-sorted-ids`,
      operation,
      variant: 'publish-sorted-ids',
      bindings: [
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'gridBounds', view: gridBounds, type: 'f32', access: 'read'},
        {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: Math.max(rowCount, 4),
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;`,
      body: `if (index < ROW_COUNT) {
    sortedIds[sortedIdsOffset + index] = sortedRows[sortedRowsOffset + index];
  }
  if (index < 4u) {
    sortedIds[sortedIdsOffset + ROW_COUNT + index] = bitcast<u32>(gridBounds[gridBoundsOffset + index]);
  }`
    })
  ];
  return {nodes, cellOffsets, sortedIds};
}

/**
 * WGSL for kernels that bind `cellOffsets` and `sortedIds` from {@link createGridIndexNodes} and
 * declare `ROW_COUNT`: cell mapping and the inclusive cell range covering a bandwidth.
 *
 * @internal
 */
export function getGridLookupWGSL(gridSize: readonly [number, number]): string {
  return /* wgsl */ `
const INDEX_WIDTH: u32 = ${gridSize[0]}u;
const INDEX_HEIGHT: u32 = ${gridSize[1]}u;

${IS_FINITE_VALUE_WGSL}${BIN_INDEX_WGSL}
// Inclusive cell range (columnLow, columnHigh, rowLow, rowHigh) covering distance < bandwidth.
// The reach is padded for f32 rounding of the box edges; rows in the padding get weight 0 or
// a weight below 1e-8 and the exact distance test in the weight rejects the rest.
fn getGridRange(origin: vec2f, bandwidth: f32) -> vec4u {
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let reach = vec2f(bandwidth * 1.0001) + abs(origin) * 2.4e-7;
  let low = clamp(origin - reach, minimum, maximum);
  let high = clamp(origin + reach, minimum, maximum);
  return vec4u(
    getBinIndex(low.x, minimum.x, maximum.x, INDEX_WIDTH),
    getBinIndex(high.x, minimum.x, maximum.x, INDEX_WIDTH),
    getBinIndex(low.y, minimum.y, maximum.y, INDEX_HEIGHT),
    getBinIndex(high.y, minimum.y, maximum.y, INDEX_HEIGHT)
  );
}
`;
}

/**
 * Block comment of the ring walk below: `selectNeighborDistances` over the grid, the same
 * `maximumK` smallest squared distances from row `i` (itself included) that a scan of every row
 * gives, found by visiting grid cells in rings around the row's cell and stopping once the
 * `maximumK`-th distance is inside the visited block (less one cell of slack for f32 cell
 * rounding). The list holds values only, so the visiting order cannot change it. Needs
 * `getPosition(row)`, `getBinIndex`, `ROW_COUNT`, `MAXIMUM_NEIGHBORS`, `SENTINEL` and the
 * grid bindings. With `positiveOnly`, coincident rows (squared distance 0, the row itself
 * included) are skipped, so the list starts at the nearest strictly positive distance.
 *
 * @internal
 */
export function getRingNeighborDistancesWGSL(
  functionName = 'selectNeighborDistances',
  positiveOnly = false
): string {
  return /* wgsl */ `
fn ${functionName}(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let cellExtent = (maximum - minimum) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
  let cellSlack = min(cellExtent.x, cellExtent.y);
  let lastColumn = i32(INDEX_WIDTH) - 1;
  let lastRow = i32(INDEX_HEIGHT) - 1;
  let centerColumn = i32(getBinIndex(clamp(origin.x, minimum.x, maximum.x), minimum.x, maximum.x, INDEX_WIDTH));
  let centerRow = i32(getBinIndex(clamp(origin.y, minimum.y, maximum.y), minimum.y, maximum.y, INDEX_HEIGHT));
  let lastRing = max(max(centerColumn, lastColumn - centerColumn), max(centerRow, lastRow - centerRow));
  for (var ring = 0; ring <= lastRing; ring++) {
    let ringColumnLow = max(centerColumn - ring, 0);
    let ringColumnHigh = min(centerColumn + ring, lastColumn);
    for (var cellRow = max(centerRow - ring, 0); cellRow <= min(centerRow + ring, lastRow); cellRow++) {
      let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
      for (var side = 0; side < 2; side++) {
        var firstColumn = ringColumnLow;
        var endColumn = ringColumnHigh;
        if (isEdgeRow) {
          if (side == 1) { continue; }
        } else {
          let column = select(centerColumn + ring, centerColumn - ring, side == 0);
          if (column < 0 || column > lastColumn) { continue; }
          firstColumn = column;
          endColumn = column;
        }
        let rowBase = u32(cellRow) * INDEX_WIDTH;
        let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
        let end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
        for (var slot = start; slot < end; slot++) {
          let row = sortedIds[sortedIdsOffset + slot];
          let delta = getPosition(row) - origin;
          let squared = dot(delta, delta);
          if (${positiveOnly ? 'squared > 0.0 && ' : ''}squared < (*list)[maximumK - 1u]) {
            var position = maximumK - 1u;
            loop {
              if (position == 0u || (*list)[position - 1u] <= squared) {
                break;
              }
              (*list)[position] = (*list)[position - 1u];
              position = position - 1u;
            }
            (*list)[position] = squared;
          }
        }
      }
    }
    if ((*list)[maximumK - 1u] < SENTINEL) {
      var gap = 3.0e38;
      if (centerColumn - ring > 0) {
        gap = min(gap, origin.x - (minimum.x + f32(centerColumn - ring) * cellExtent.x));
      }
      if (centerColumn + ring < lastColumn) {
        gap = min(gap, minimum.x + f32(centerColumn + ring + 1) * cellExtent.x - origin.x);
      }
      if (centerRow - ring > 0) {
        gap = min(gap, origin.y - (minimum.y + f32(centerRow - ring) * cellExtent.y));
      }
      if (centerRow + ring < lastRow) {
        gap = min(gap, minimum.y + f32(centerRow + ring + 1) * cellExtent.y - origin.y);
      }
      gap = gap - cellSlack;
      if (gap > 0.0 && (*list)[maximumK - 1u] < gap * gap) {
        break;
      }
    }
  }
}
`;
}
