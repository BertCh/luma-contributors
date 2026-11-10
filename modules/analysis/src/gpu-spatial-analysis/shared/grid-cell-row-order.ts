// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Result of {@link createGridCellRowOrderNodes}. @internal */
export type GridCellRowOrder<Parameters> = {
  /** Key scatter and stable sort nodes, in execution order. */
  nodes: readonly GPUCommandNode<Parameters>[];
  /**
   * `rowCount` row IDs: ascending row order inside every grid cell, cells in ascending order (so
   * slot `s` of cell `c` is at `cellOffsets[c] + s`). Rows the index did not accept follow the
   * indexed ones.
   */
  sortedRows: GraphDataView<'uint32'>;
};

/**
 * Restores ascending row order inside every cell of a published `GPUGridIndex`.
 *
 * `GPUGridIndex` scatters row IDs into cells with atomics, so the order inside a cell varies
 * between runs, and any floating-point sum or tie rule that walks a cell would then vary with it.
 * Each indexed slot writes its cell number to the key of the row it holds, and a stable radix sort
 * of the rows `0..rowCount-1` by that key yields the same order as ranking every ID against its
 * cell neighbours, in O(rows) work however many rows share a cell (the ranking costs O(m^2) for a
 * cell of m rows). Rows the index did not accept get the key `cellCount` and sort last.
 *
 * @internal
 */
export function createGridCellRowOrderNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** `cellCount + 1` slot offsets from the grid index. */
    cellOffsets: GraphDataView<'uint32'>;
    /** Row IDs in slot order from the grid index. */
    objectIds: GraphDataView<'uint32'>;
    /** Number of indexed slots from the grid index. */
    indexCount: GraphDataView<'uint32'>;
    /** Number of rows that were offered to the index. */
    rowCount: number;
  }
): GridCellRowOrder<Parameters> {
  const {id, operation, cellOffsets, objectIds, indexCount, rowCount} = props;
  const cellCount = cellOffsets.length - 1;
  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rowCount);
  const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rowCount);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rowCount);
  const declarations = `const CELL_COUNT: u32 = ${cellCount}u;
const ROW_COUNT: u32 = ${rowCount}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-initialize-cell-keys`,
      operation,
      variant: 'initialize-cell-keys',
      bindings: [
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      declarations,
      body: `cellKeys[cellKeysOffset + index] = CELL_COUNT;
  rowIds[rowIdsOffset + index] = index;`
    }),
    // Every row occupies at most one slot, so these scatter writes never collide.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-scatter-cell-keys`,
      operation,
      variant: 'scatter-cell-keys',
      bindings: [
        {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
        {name: 'objectIds', view: objectIds, type: 'u32', access: 'read'},
        {name: 'indexCount', view: indexCount, type: 'u32', access: 'read'},
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      declarations,
      body: `if (index >= min(indexCount[indexCountOffset], ROW_COUNT)) {
    return;
  }
  let row = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  cellKeys[cellKeysOffset + row] = low;`
    }),
    ...new GPUSort({
      id: `${id}-sort-rows`,
      keys: cellKeys,
      values: rowIds,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: Math.max(1, Math.ceil(Math.log2(cellCount + 1)))
    }).getCommandNodes(graph)
  ];
  return {nodes, sortedRows};
}
