// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One-line GPU gather for drawing: positions of the rows an operation selected by id.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Adds a node writing `positions[ids[row]]` per row (NaN for ids outside the table). */
export function addPositionGather(
  builder: RunBuilder,
  ids: RunBuffer<'uint32'>,
  positions: RunBuffer<'float32x2'>,
  id: string
): RunBuffer<'float32x2'> {
  const gathered = builder.allocate(id, 'float32x2', ids.length);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'position-gather',
        bindings: [
          {name: 'rowIds', view: ids.view, type: 'u32', access: 'read'},
          {name: 'rowPositions', view: positions.view, type: 'f32', access: 'read'},
          {name: 'rowOutput', view: gathered.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: ids.length,
        body: `let row = rowIds[rowIdsOffset + index];
let valid = row < ${positions.length}u;
let safeRow = min(row, ${positions.length - 1}u);
rowOutput[rowOutputOffset + index * 2u] = select(bitcast<f32>(row | 0x7fc00000u), rowPositions[rowPositionsOffset + safeRow * 2u], valid);
rowOutput[rowOutputOffset + index * 2u + 1u] = select(bitcast<f32>(row | 0x7fc00000u), rowPositions[rowPositionsOffset + safeRow * 2u + 1u], valid);`
      })
    ]
  });
  return gathered;
}

/** Adds a node copying `count` uint32 rows of `source` from row `offset` as float32, so a histogram can be drawn without its header. */
export function addSlice(
  builder: RunBuilder,
  source: RunBuffer<'uint32'>,
  offset: number,
  count: number,
  id: string
): RunBuffer<'float32'> {
  const slice = builder.allocate(id, 'float32', count);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'slice',
        bindings: [
          {name: 'sliceSource', view: source.view, type: 'u32', access: 'read'},
          {name: 'sliceTarget', view: slice.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: count,
        body: `sliceTarget[sliceTargetOffset + index] = f32(sliceSource[sliceSourceOffset + index + ${offset}u]);`
      })
    ]
  });
  return slice;
}

/** Adds a node copying the first `count` float32 rows of `source` from row `offset`, so a chart can drop empty rows or show one group. */
export function addFloatSlice(
  builder: RunBuilder,
  source: RunBuffer<'float32'>,
  count: number,
  id: string,
  offset = 0
): RunBuffer<'float32'> {
  const slice = builder.allocate(id, 'float32', count);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'float-slice',
        bindings: [
          {name: 'sliceSource', view: source.view, type: 'f32', access: 'read'},
          {name: 'sliceTarget', view: slice.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: count,
        body: `sliceTarget[sliceTargetOffset + index] = sliceSource[sliceSourceOffset + index + ${offset}u];`
      })
    ]
  });
  return slice;
}
