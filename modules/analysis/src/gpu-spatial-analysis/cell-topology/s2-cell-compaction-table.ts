// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';

type S2CompactionTableProps = {
  id: string;
  layout: {width: number; lowBit: number};
  keyLow: GraphDataView<'uint32'>;
  keyHigh: GraphDataView<'uint32'>;
  sorted: boolean;
  cells: GraphDataView<'uint32x2'>;
  count: GraphDataView<'uint32'>;
};

/**
 * Deduplicates face/Hilbert paths with published stable sorts and a flag scan. S2's marker-bit
 * encoding differs from the H3/Quadbin table encoder, so canonical IDs are rebuilt here.
 * The caller preflights all row-sized planes before declaring resources. @internal
 */
export function getS2CompactionTableNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: S2CompactionTableProps
): GPUCommandNode<Parameters>[] {
  const {id, layout} = props;
  const rows = props.keyLow.length;
  const createWords = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', rows);
  const nodes: GPUCommandNode<Parameters>[] = [];
  let keyLow = props.keyLow;
  let keyHigh = props.keyHigh;
  if (!props.sorted) {
    const sortedLow = createWords('sorted-low');
    const pairedHigh = createWords('paired-high');
    keyLow = createWords('ordered-low');
    keyHigh = createWords('ordered-high');
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-low`,
        keys: props.keyLow,
        values: props.keyHigh,
        outputKeys: sortedLow,
        outputValues: pairedHigh
      }).getCommandNodes(graph),
      ...new GPUSort({
        id: `${id}-sort-high`,
        keys: pairedHigh,
        values: sortedLow,
        outputKeys: keyHigh,
        outputValues: keyLow
      }).getCommandNodes(graph)
    );
  }
  const flags = createWords('flags');
  const positions = createWords('positions');
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-boundaries`,
      operation: 'GPUCellCompaction',
      variant: 's2-boundaries',
      invocationCount: rows,
      bindings: [
        {name: 'keyLow', view: keyLow, type: 'u32', access: 'read'},
        {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      declarations: CELL_KEY_WGSL,
      body: `let key = vec2u(keyHigh[keyHighOffset + index], keyLow[keyLowOffset + index]);
  let valid = all(cellShiftRight(key, ${layout.width}u) == vec2u(0u));
  var boundary = index == 0u;
  if (index > 0u) {
    boundary = any(key != vec2u(keyHigh[keyHighOffset + index - 1u], keyLow[keyLowOffset + index - 1u]));
  }
  flags[flagsOffset + index] = select(0u, 1u, valid && boundary);`
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: flags,
      output: positions,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-emit`,
      operation: 'GPUCellCompaction',
      variant: 's2-unique',
      invocationCount: rows,
      bindings: [
        {name: 'keyLow', view: keyLow, type: 'u32', access: 'read'},
        {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'positions', view: positions, type: 'u32', access: 'read'},
        {name: 'cells', view: props.cells, type: 'u32', access: 'read_write'},
        {name: 'count', view: props.count, type: 'u32', access: 'read_write'}
      ],
      declarations: CELL_KEY_WGSL,
      body: `if (index == 0u) {
    count[countOffset] = positions[positionsOffset + ${rows - 1}u] + flags[flagsOffset + ${rows - 1}u];
  }
  if (flags[flagsOffset + index] != 0u) {
    let compact = vec2u(keyHigh[keyHighOffset + index], keyLow[keyLowOffset + index]);
    let key = cellShiftLeft(compact, ${layout.lowBit}u) | cellShiftLeft(vec2u(0u, 1u), ${layout.lowBit - 1}u);
    let slot = positions[positionsOffset + index];
    cells[cellsOffset + 2u * slot] = key.y;
    cells[cellsOffset + 2u * slot + 1u] = key.x;
  }`
    })
  );
  return nodes;
}
