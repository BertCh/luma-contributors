// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpaceTimeKrigingNeighborSearch} from './gpu-space-time-kriging';

/** Covariance occupies words 0..27; axis extents occupy 32..37 in private parameters. */
export const SPACE_TIME_KRIGING_INDEX_PARAMETER_LENGTH = 40;
/** Leaves the public covariance parameter layout unchanged, with room for its future padding. */
const SPACE_TIME_KRIGING_EXTENT_OFFSET = 32;
/** Three cell widths plus relative float32 slack make index pruning conservative at boundaries. */
const SPACE_TIME_KRIGING_CELL_SLACK = 3;
/** Covers coordinate arithmetic, division and distance rounding; only widens the searched cells. */
const SPACE_TIME_KRIGING_RELATIVE_SLACK = 1e-5;

/**
 * Twice sqrt(the smallest normal float32): squares below this range may flush to zero.
 * Metric pruning must not distinguish those distances when the reference metric ties at zero.
 */
const SPACE_TIME_KRIGING_MINIMUM_PRUNABLE_DISTANCE = 2 ** -62;

/** Index membership matches the predictor's observation predicate. */
const USABLE_OBSERVATION_WGSL = `${getPackedPredictionAccessors({observations: 4})}
fn isUsableObservation(row: u32) -> bool {
  let observation = read_observations(row);
  return observation.w > 0.0 && isFiniteValue(observation.x) && isFiniteValue(observation.y) && observation.y >= 0.0;
}`;

/** Shared key assignment and conservative anisotropy-scaled cell bounds. @internal */
export function getSpaceTimeKrigingGridWGSL(gridSide: number): string {
  return /* wgsl */ `
const GRID_SIDE: u32 = ${gridSide}u;
const EXTENT_OFFSET: u32 = ${SPACE_TIME_KRIGING_EXTENT_OFFSET}u;
const CELL_SLACK: f32 = ${SPACE_TIME_KRIGING_CELL_SLACK}.0;
const RELATIVE_SLACK: f32 = ${SPACE_TIME_KRIGING_RELATIVE_SLACK};
const MINIMUM_PRUNABLE_DISTANCE: f32 = ${SPACE_TIME_KRIGING_MINIMUM_PRUNABLE_DISTANCE};
fn getAxisMinimum(axis: u32) -> f32 { return params[paramsOffset + EXTENT_OFFSET + axis * 2u]; }
fn getAxisWidth(axis: u32) -> f32 { return params[paramsOffset + EXTENT_OFFSET + axis * 2u + 1u] - getAxisMinimum(axis); }
fn getAxisScale(axis: u32) -> f32 { return select(1.0, params[paramsOffset + 26u], axis == 2u); }
fn getAxisLastCell(axis: u32) -> u32 {
  let width = getAxisWidth(axis);
  return select(0u, GRID_SIDE - 1u, isFiniteValue(width) && width > 0.0 && getAxisScale(axis) > 0.0);
}
fn getAxisCell(value: f32, axis: u32) -> u32 {
  if (getAxisLastCell(axis) == 0u) { return 0u; }
  // Scaling the numerator and denominator by searchAnisotropy cancels. Keep the raw ratio
  // to avoid overflow/underflow; bounds and pruning below use the scaled metric.
  let coordinate = (value - getAxisMinimum(axis)) / getAxisWidth(axis);
  return u32(floor(clamp(coordinate, 0.0, 1.0) * f32(GRID_SIDE - 1u)));
}
fn getAxisSlack(value: f32, axis: u32) -> f32 {
  return CELL_SLACK * getAxisWidth(axis) / f32(GRID_SIDE - 1u) + RELATIVE_SLACK * max(abs(value), max(abs(getAxisMinimum(axis)), abs(getAxisMinimum(axis) + getAxisWidth(axis))));
}
fn getAxisGap(value: f32, axis: u32, firstCell: u32, lastCell: u32) -> f32 {
  if (getAxisLastCell(axis) == 0u) { return 0.0; }
  let minimum = getAxisMinimum(axis);
  let extent = getAxisWidth(axis) / f32(GRID_SIDE - 1u);
  let lower = minimum + f32(firstCell) * extent;
  let upper = minimum + f32(lastCell + 1u) * extent;
  return max(0.0, max(lower - value, value - upper) - getAxisSlack(value, axis)) * getAxisScale(axis);
}
fn getOutsideGap(value: f32, axis: u32, firstCell: u32, lastCell: u32) -> f32 {
  let extent = getAxisWidth(axis) / f32(GRID_SIDE - 1u);
  var gap = 3.402823466e38;
  if (firstCell > 0u) { gap = min(gap, value - (getAxisMinimum(axis) + f32(firstCell) * extent)); }
  if (lastCell < getAxisLastCell(axis)) { gap = min(gap, getAxisMinimum(axis) + f32(lastCell + 1u) * extent - value); }
  if (firstCell == 0u && lastCell == getAxisLastCell(axis)) { return 3.402823466e38; }
  return max(0.0, gap - getAxisSlack(value, axis)) * getAxisScale(axis);
}`;
}

/** Expanding shells stop only after every relevant quota has a strict metric certificate. */
export function getSpaceTimeKrigingGridScanWGSL(
  visitSample: string,
  search: GPUSpaceTimeKrigingNeighborSearch
): string {
  const threshold =
    search === 'octant'
      ? `var searchThreshold = 0.0;
      for (var octant = 0u; octant < 8u; octant++) {
        if (octantCounts[octant] < NEIGHBORS_PER_OCTANT) { searchThreshold = 3.402823466e38; break; }
        searchThreshold = max(searchThreshold, candidateDistance[octant * NEIGHBORS_PER_OCTANT + NEIGHBORS_PER_OCTANT - 1u]);
      }`
      : `var searchThreshold = 3.402823466e38;
      if (candidateCount == CANDIDATE_CAPACITY) { searchThreshold = candidateDistance[candidateCount - 1u]; }`;
  return /* wgsl */ `
let queryPoint = vec3<f32>(queryPosition, queryTime);
let centerCell = vec3<i32>(i32(getAxisCell(queryPoint.x, 0u)), i32(getAxisCell(queryPoint.y, 1u)), i32(getAxisCell(queryPoint.z, 2u)));
let lastCell = vec3<i32>(i32(getAxisLastCell(0u)), i32(getAxisLastCell(1u)), i32(getAxisLastCell(2u)));
let lastRing = max(max(max(centerCell.x, lastCell.x - centerCell.x), max(centerCell.y, lastCell.y - centerCell.y)), max(centerCell.z, lastCell.z - centerCell.z));
for (var ring = 0; ring <= lastRing; ring++) {
  let firstCell = max(centerCell - vec3<i32>(ring), vec3<i32>(0));
  let finalCell = min(centerCell + vec3<i32>(ring), lastCell);
  ${threshold}
  for (var cellTime = firstCell.z; cellTime <= finalCell.z; cellTime++) {
    for (var cellRow = firstCell.y; cellRow <= finalCell.y; cellRow++) {
      let edgePlane = abs(cellTime - centerCell.z) == ring || abs(cellRow - centerCell.y) == ring;
      for (var side = 0u; side < 2u; side++) {
        var firstColumn = firstCell.x;
        var finalColumn = finalCell.x;
        if (edgePlane) {
          if (side == 1u) { continue; }
        } else {
          let column = select(centerCell.x - ring, centerCell.x + ring, side == 1u);
          if (column < firstCell.x || column > finalCell.x) { continue; }
          firstColumn = column;
          finalColumn = column;
        }
        for (var cellColumn = firstColumn; cellColumn <= finalColumn; cellColumn++) {
        let gapX = getAxisGap(queryPoint.x, 0u, u32(cellColumn), u32(cellColumn));
        let gapY = getAxisGap(queryPoint.y, 1u, u32(cellRow), u32(cellRow));
        let gapTime = getAxisGap(queryPoint.z, 2u, u32(cellTime), u32(cellTime));
        if (max(max(gapX, gapY), gapTime) > max(searchThreshold * (1.0 + RELATIVE_SLACK), MINIMUM_PRUNABLE_DISTANCE)) { continue; }
        if (max(gapX, gapY) > max(searchRadius, MINIMUM_PRUNABLE_DISTANCE) || gapTime > max(maximumTimeLag * searchAnisotropy, MINIMUM_PRUNABLE_DISTANCE)) { continue; }
        let cell = u32(cellColumn) + GRID_SIDE * (u32(cellRow) + GRID_SIDE * u32(cellTime));
        let firstSlot = cellOffsets[cellOffsetsOffset + cell];
        let finalSlot = cellOffsets[cellOffsetsOffset + cell + 1u];
        for (var indexSlot = firstSlot; indexSlot < finalSlot; indexSlot++) {
          let sample = sortedRows[sortedRowsOffset + indexSlot];
          ${visitSample}
        }
        }
      }
    }
  }
  ${threshold.replaceAll('searchThreshold', 'finalThreshold')}
  let outsideGap = min(min(getOutsideGap(queryPoint.x, 0u, u32(firstCell.x), u32(finalCell.x)), getOutsideGap(queryPoint.y, 1u, u32(firstCell.y), u32(finalCell.y))), getOutsideGap(queryPoint.z, 2u, u32(firstCell.z), u32(finalCell.z)));
  if (outsideGap > max(finalThreshold * (1.0 + RELATIVE_SLACK), MINIMUM_PRUNABLE_DISTANCE)) { break; }
}`;
}

/** Builds extents and a stable keyed grid with published reduction, aggregation, scan and sort. */
export function createSpaceTimeKrigingIndexNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  geometry: GraphDataView<'float32x3'>,
  observations: GraphDataView<'float32x4'>,
  parameters: GraphDataView<'float32'>,
  gridSide: number
) {
  const rowCount = geometry.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  // Reuse the future keys as a reduction mask so invalid samples do not widen the grid bounds.
  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rowCount);
  for (let axis = 0; axis < 3; axis++) {
    const coordinates = createTransientView(graph, `${id}-axis-${axis}`, 'float32', rowCount);
    const extent = createTransientView(graph, `${id}-extent-${axis}`, 'float32', 2);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-extract-${axis}`,
        operation: 'GPUSpaceTimeKriging',
        bindings: [
          {name: 'geometry', view: geometry, type: 'f32', access: 'read'},
          {name: 'observations', view: observations, type: 'f32', access: 'read'},
          {name: 'coordinates', view: coordinates, type: 'f32', access: 'read_write'},
          {name: 'validity', view: cellKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `${IS_FINITE_VALUE_WGSL}\n${USABLE_OBSERVATION_WGSL}`,
        body: `let point = vec3<f32>(geometry[index * 3u], geometry[index * 3u + 1u], geometry[index * 3u + 2u]);
let usable = isUsableObservation(index) && isFiniteValue(point.x) && isFiniteValue(point.y) && isFiniteValue(point.z);
coordinates[index] = select(0.0, point[${axis}u], usable);
validity[index] = select(0u, 1u, usable);`
      })
    );
    nodes.push(
      ...new GPUReduction({
        id: `${id}-reduce-${axis}`,
        input: coordinates,
        mask: cellKeys,
        output: extent,
        operation: 'extent'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish-extent-${axis}`,
        operation: 'GPUSpaceTimeKriging',
        invocationCount: 2,
        bindings: [
          {name: 'extent', view: extent, type: 'f32', access: 'read'},
          {name: 'params', view: parameters, type: 'f32', access: 'read_write'}
        ],
        body: `params[${SPACE_TIME_KRIGING_EXTENT_OFFSET + axis * 2}u + index] = extent[index];`
      })
    );
  }
  const sourceRows = createTransientView(graph, `${id}-source-rows`, 'uint32', rowCount);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-keys`,
      operation: 'GPUSpaceTimeKriging',
      bindings: [
        {name: 'geometry', view: geometry, type: 'f32', access: 'read'},
        {name: 'observations', view: observations, type: 'f32', access: 'read'},
        {name: 'params', view: parameters, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'sourceRows', view: sourceRows, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      declarations: `${IS_FINITE_VALUE_WGSL}\n${USABLE_OBSERVATION_WGSL}\n${getSpaceTimeKrigingGridWGSL(gridSide)}`,
      body: `let point = vec3<f32>(geometry[index * 3u], geometry[index * 3u + 1u], geometry[index * 3u + 2u]);
sourceRows[index] = index;
var key = GRID_SIDE * GRID_SIDE * GRID_SIDE;
if (isUsableObservation(index) && isFiniteValue(point.x) && isFiniteValue(point.y) && isFiniteValue(point.z)) {
  key = getAxisCell(point.x, 0u) + GRID_SIDE * (getAxisCell(point.y, 1u) + GRID_SIDE * getAxisCell(point.z, 2u));
}
cellKeys[index] = key;`
    })
  );
  const grid = createPointNeighborGridIndexNodes(graph, {
    id: `${id}-grid`,
    rowCount,
    cellKeys,
    sourceRows,
    cellCount: gridSide ** 3
  });
  nodes.push(...grid.nodes);
  return {nodes, cellOffsets: grid.cellOffsets, sortedRows: grid.sortedRows};
}
