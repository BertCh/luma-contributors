// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPairStatisticsSegmentTotalNodes} from './pair-statistics-cell-neighbors';
import {getPairStatisticsSharedWGSL} from './pair-statistics-grid';
import type {GPUSpaceTimeVariogramProps} from './gpu-space-time-variogram';

/**
 * Compatibility fallback from the shared fixed-point estimator to legacy floating arithmetic.
 * Every legacy frame uses raw value differences: centering can erase them, and even a finite
 * fixed-point scale can quantize a small qualifying term to zero beside an unrelated outlier.
 * Pairs come from the shared cell index, merging its sorted cell ranges in input-row order
 * to retain signed cancellation independently of grid dimensions. Per-row partials and bounded trees
 * preserve the legacy arithmetic and nonfinite status without a row-sized serial finishing loop.
 * @internal
 */
export function getLegacyVariogramFallbackNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: Omit<GPUSpaceTimeVariogramProps, 'times'> &
    Pick<Partial<GPUSpaceTimeVariogramProps>, 'times'>,
  index: {
    cellOffsets: GraphDataView<'uint32'>;
    sortedRows: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const rows = props.positions.length;
  const bins = props.lagCount * props.temporalLagCount;
  const rowTable = createTransientView(graph, `${id}-legacy-rows`, 'float32', rows * 3);
  const sums = createTransientView(graph, `${id}-legacy-sums`, 'float32', rows * bins);
  const counts = createTransientView(graph, `${id}-legacy-counts`, 'uint32', rows * bins);
  const totalSums = createTransientView(graph, `${id}-legacy-total-sums`, 'float32', bins);
  const totalCounts = createTransientView(graph, `${id}-legacy-total-counts`, 'uint32', bins * 2);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-legacy-rows`,
      operation: 'GPUCrossVariogram',
      variant: 'legacy-rows',
      invocationCount: rows,
      bindings: [
        {name: 'values', view: props.values, type: 'f32', access: 'read'},
        ...(props.secondValues
          ? [
              {
                name: 'secondValues',
                view: props.secondValues,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(props.times
          ? [{name: 'times', view: props.times, type: 'f32' as const, access: 'read' as const}]
          : []),
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'rowTable', view: rowTable, type: 'f32', access: 'read_write'}
      ],
      declarations: GET_NAN_WGSL,
      body: `rowTable[rowTableOffset + index * 3u] = ${props.mask ? 'select(getNaN(), values[valuesOffset + index], mask[maskOffset + index] != 0u)' : 'values[valuesOffset + index]'};
rowTable[rowTableOffset + index * 3u + 1u] = ${props.secondValues ? 'secondValues[secondValuesOffset + index]' : 'values[valuesOffset + index]'};
rowTable[rowTableOffset + index * 3u + 2u] = ${props.times ? 'times[timesOffset + index]' : '0.0'};`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-legacy-pairs`,
      operation: 'GPUCrossVariogram',
      variant: 'legacy-pairs',
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'rowTable', view: rowTable, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'cellOffsets', view: index.cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: index.sortedRows, type: 'u32', access: 'read'},
        {name: 'sums', view: sums, type: 'f32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      declarations: `${getPairStatisticsSharedWGSL(props.gridSize, true)}\n${IS_FINITE_BITS_WGSL}
const BINS: u32 = ${bins}u;`,
      body: `var localSums: array<f32, BINS>;
var localCounts: array<u32, BINS>;
let lattice = readLattice();
let focus = index;
let focusX = positions[positionsOffset + 2u * focus];
let focusY = positions[positionsOffset + 2u * focus + 1u];
var focusKey = CELL_COUNT;
if (isPointValid(lattice, focusX, focusY) && isFiniteBits(rowTable[rowTableOffset + focus * 3u])) {
  focusKey = getCellRow(lattice, focusY) * lattice.columns + getCellColumn(lattice, focusX);
}
let maximumDistance = parameters[parametersOffset + 4u];
let maximumTime = parameters[parametersOffset + 5u];
if (lattice.valid && focusKey < CELL_COUNT) ${getLegacyVariogramNeighborLoopWGSL(
        'focusKey',
        `if (neighbor <= focus) { continue; }
let delta = vec2f(positions[positionsOffset + 2u * focus] - positions[positionsOffset + 2u * neighbor], positions[positionsOffset + 2u * focus + 1u] - positions[positionsOffset + 2u * neighbor + 1u]);
let distance = length(delta);
let time = abs(rowTable[rowTableOffset + focus * 3u + 2u] - rowTable[rowTableOffset + neighbor * 3u + 2u]);
if (!isFiniteBits(distance) || distance > maximumDistance || time > maximumTime) { continue; }
let spatialBin = min(u32(distance / maximumDistance * ${props.lagCount}.0), ${props.lagCount - 1}u);
let temporalBin = min(u32(time / maximumTime * ${props.temporalLagCount}.0), ${props.temporalLagCount - 1}u);
let bin = spatialBin * ${props.temporalLagCount}u + temporalBin;
let difference = rowTable[rowTableOffset + focus * 3u] - rowTable[rowTableOffset + neighbor * 3u];
let secondDifference = rowTable[rowTableOffset + focus * 3u + 1u] - rowTable[rowTableOffset + neighbor * 3u + 1u];
localSums[bin] += 0.5 * difference * secondDifference;
localCounts[bin] += 1u;`
      )}
for (var bin = 0u; bin < BINS; bin++) {
  sums[sumsOffset + bin * ${rows}u + focus] = localSums[bin];
  counts[countsOffset + bin * ${rows}u + focus] = localCounts[bin];
}`
    }),
    ...getPairStatisticsSegmentTotalNodes(graph, {
      id: `${id}-legacy-sums`,
      operation: 'GPUCrossVariogram',
      segmentCount: bins,
      rowCount: rows,
      input: sums,
      output: totalSums
    }),
    ...getPairStatisticsSegmentTotalNodes(graph, {
      id: `${id}-legacy-counts`,
      operation: 'GPUCrossVariogram',
      segmentCount: bins,
      rowCount: rows,
      input: counts,
      output: totalCounts,
      detectOverflow: true
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-legacy-finish`,
      operation: 'GPUCrossVariogram',
      variant: 'legacy-finish',
      invocationCount: bins,
      bindings: [
        {name: 'totalSums', view: totalSums, type: 'f32', access: 'read'},
        {name: 'totalCounts', view: totalCounts, type: 'u32', access: 'read'},
        {
          name: 'semivariances',
          view: props.output.semivariances,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'pairCounts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let count = totalCounts[totalCountsOffset + index];
let overflow = totalCounts[totalCountsOffset + ${bins}u + index] != 0u;
let value = totalSums[totalSumsOffset + index] / f32(count);
semivariances[semivariancesOffset + index] = select(getNaN(), value, count > 0u && isFiniteBits(value) && !overflow);
pairCounts[pairCountsOffset + index] = count;
status[statusOffset + index] = select(select(select(2u, 0u, isFiniteBits(value)), 1u, count == 0u), 3u, overflow);`
    })
  ];
}

/**
 * Merge at most nine sorted cell ranges, visiting candidate rows in ascending input order.
 * The shared stable sort keeps each cell in row order. A bounded merge preserves the legacy
 * default's pair-sum order without scanning rows outside the indexed neighborhood.
 */
function getLegacyVariogramNeighborLoopWGSL(focusKey: string, neighborStatement: string): string {
  return `{
  // Literal initializers avoid the GPU back-end's constant-loop integer-array store loss.
  var cursorSlots = array<u32, 9>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  var endSlots = array<u32, 9>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  var activeCellCount = 0u;
  let focusColumn = i32(${focusKey} % lattice.columns);
  let focusRow = i32(${focusKey} / lattice.columns);
  let lastCellRow = min(focusRow + 1, i32(lattice.rows) - 1);
  let lastCellColumn = min(focusColumn + 1, i32(lattice.columns) - 1);
  for (var cellRow = max(focusRow - 1, 0); cellRow <= lastCellRow; cellRow++) {
    for (var cellColumn = max(focusColumn - 1, 0); cellColumn <= lastCellColumn; cellColumn++) {
      let cell = u32(cellRow) * lattice.columns + u32(cellColumn);
      let begin = cellOffsets[cellOffsetsOffset + cell];
      let end = cellOffsets[cellOffsetsOffset + cell + 1u];
      if (begin < end) {
        cursorSlots[activeCellCount] = begin;
        endSlots[activeCellCount] = end;
        activeCellCount++;
      }
    }
  }
  loop {
    var neighbor = 0xffffffffu;
    var selectedCell = 0xffffffffu;
    for (var cellIndex = 0u; cellIndex < activeCellCount; cellIndex++) {
      let cursor = cursorSlots[cellIndex];
      if (cursor < endSlots[cellIndex]) {
        let candidate = sortedRows[sortedRowsOffset + cursor];
        if (candidate < neighbor) {
          neighbor = candidate;
          selectedCell = cellIndex;
        }
      }
    }
    if (selectedCell == 0xffffffffu) { break; }
    cursorSlots[selectedCell]++;
    ${neighborStatement}
  }
}`;
}
