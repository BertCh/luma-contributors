// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Entries one invocation of a segment-total level adds. */
const PAIR_STATISTICS_TOTAL_FAN_IN = 256;

/**
 * Per-segment totals of a segment-major table (`segment * rowCount + row`) for the grid paths of
 * the pair contributors: levels of invocations that each add at most 256 consecutive entries, in
 * a fixed order, until one total per segment remains (three levels at 2^24 rows). No invocation
 * loops over every row. float32 sums are added in that tree order; uint32 counts are exact.
 *
 * @internal
 */
export function getPairStatisticsSegmentTotalNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    segmentCount: number;
    rowCount: number;
    input: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, segmentCount} = props;
  const format = props.input.format as 'float32' | 'uint32';
  const type = format === 'float32' ? 'f32' : 'u32';
  const nodes: GPUCommandNode<Parameters>[] = [];
  let input: GraphDataView<'float32'> | GraphDataView<'uint32'> = props.input;
  let entryCount = props.rowCount;
  for (let level = 0; ; level++) {
    const groupCount = Math.ceil(entryCount / PAIR_STATISTICS_TOTAL_FAN_IN);
    const output =
      groupCount === 1
        ? props.output
        : (createTransientView(graph, `${id}-level-${level}`, format, segmentCount * groupCount) as
            | GraphDataView<'float32'>
            | GraphDataView<'uint32'>);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-level-${level}`,
        operation,
        variant: `segment-total-${type}`,
        invocationCount: segmentCount * groupCount,
        bindings: [
          {name: 'entries', view: input, type, access: 'read'},
          {name: 'totals', view: output, type, access: 'read_write'}
        ],
        body: `let segment = index / ${groupCount}u; let group = index % ${groupCount}u;
  let begin = segment * ${entryCount}u + group * ${PAIR_STATISTICS_TOTAL_FAN_IN}u;
  let end = segment * ${entryCount}u + min((group + 1u) * ${PAIR_STATISTICS_TOTAL_FAN_IN}u, ${entryCount}u);
  var total = ${type}(0);
  for (var entry = begin; entry < end; entry++) { total += entries[entriesOffset + entry]; }
  totals[totalsOffset + index] = total;`
      })
    );
    if (groupCount === 1) return nodes;
    input = output;
    entryCount = groupCount;
  }
}

/** Largest `rows * bins` of a per-row bin partial table: two float32 per entry in one binding. */
export const PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM = 2 ** 24;

/**
 * Validates an optional compile-time `gridSize` of a pair contributor that keeps a dense
 * all-pairs traversal when the grid is omitted. @internal
 */
export function validatePairStatisticsGridSize(
  id: string,
  gridSize: readonly [number, number]
): void {
  validateCapacity(id, 'gridSize[0]', gridSize[0], 1, 2 ** 24 - 1);
  validateCapacity(id, 'gridSize[1]', gridSize[1], 1, 2 ** 24 - 1);
  validateMaximumCount(id, 'gridSize[0] x gridSize[1]', gridSize[0] * gridSize[1], 2 ** 24 - 1);
}

/**
 * WGSL statements that run `neighborStatement` for every row stored in the 3 x 3 cells around
 * cell `focusKey` of the pair-statistics lattice, with `neighbor` (the row) in scope. Cells are at
 * least `maximumDistance` wide (see `getPairStatisticsSharedWGSL`), so these rows include every
 * row within `maximumDistance` of the focus. `continue` in the statement skips to the next row.
 *
 * The kernel must declare `lattice` (from `readLattice()`) and bind `cellOffsets` and `sortedRows`
 * from `getPairStatisticsInputNodes`.
 *
 * @internal
 */
export function getPairStatisticsCellNeighborLoopWGSL(
  focusKey: string,
  neighborStatement: string
): string {
  return `{
  let focusColumn = i32(${focusKey} % lattice.columns);
  let focusRow = i32(${focusKey} / lattice.columns);
  let lastCellRow = min(focusRow + 1, i32(lattice.rows) - 1);
  let lastCellColumn = min(focusColumn + 1, i32(lattice.columns) - 1);
  for (var cellRow = max(focusRow - 1, 0); cellRow <= lastCellRow; cellRow++) {
    for (var cellColumn = max(focusColumn - 1, 0); cellColumn <= lastCellColumn; cellColumn++) {
      let cell = u32(cellRow) * lattice.columns + u32(cellColumn);
      let slotEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = cellOffsets[cellOffsetsOffset + cell]; slot < slotEnd; slot++) {
        let neighbor = sortedRows[sortedRowsOffset + slot];
        ${neighborStatement}
      }
    }
  }
}`;
}
