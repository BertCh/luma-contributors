// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
  name,
  view,
  type: 'u32',
  access: 'read'
});
const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
  name,
  view,
  type: 'u32',
  access: 'read_write'
});

/** Exact distinct finite float32 values per right-key segment; +/-0 are the same value. @internal */
export function createKeyJoinDistinctNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    column: GraphDataView<'float32'>;
    sortedRows: GraphDataView<'uint32'>;
    rowUnique: GraphDataView<'uint32'>;
    leftUnique: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, column, sortedRows, rowUnique, leftUnique, output} = props;
  const rows = sortedRows.length;
  const scalar = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', rows);
  const groups = scalar('groups');
  const valueKeys = scalar('value-keys');
  const indices = scalar('indices');
  const counts = scalar('counts');
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-keys`,
      operation: 'GPUKeyJoin',
      variant: 'distinct-keys',
      invocationCount: rows,
      bindings: [
        read('column', column),
        read('sortedRows', sortedRows),
        read('rowUnique', rowUnique),
        write('groups', groups),
        write('valueKeys', valueKeys),
        write('indices', indices)
      ],
      body: `let bits = column[columnOffset + sortedRows[sortedRowsOffset + index]];
  let finite = (bits & 0x7fffffffu) < 0x7f800000u;
  groups[groupsOffset + index] = select(0xffffffffu, rowUnique[rowUniqueOffset + index], finite);
  valueKeys[valueKeysOffset + index] = select(bits, 0u, (bits & 0x7fffffffu) == 0u);
  indices[indicesOffset + index] = index;`
    })
  ];
  const sorted = createStableLexicographicIndexSortNodes(graph, {
    id: `${id}-sort`,
    operation: 'GPUKeyJoin',
    indices,
    indicesAreIdentity: true,
    keys: [{view: groups}, {view: valueKeys}]
  });
  nodes.push(
    ...sorted.nodes,
    createFillNode(graph, {
      id: `${id}-clear`,
      operation: 'GPUKeyJoin',
      view: counts,
      type: 'u32',
      value: '0u'
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-heads`,
      operation: 'GPUKeyJoin',
      variant: 'distinct-heads',
      invocationCount: rows,
      bindings: [
        read('sortedGroups', sorted.sortedPrimaryKeys),
        read('sortedIndices', sorted.sortedIndices),
        read('valueKeys', valueKeys),
        {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let group = sortedGroups[sortedGroupsOffset + index];
  if (group == 0xffffffffu) { return; }
  let key = valueKeys[valueKeysOffset + sortedIndices[sortedIndicesOffset + index]];
  var head = index == 0u;
  if (index > 0u) {
    head = group != sortedGroups[sortedGroupsOffset + index - 1u] || key != valueKeys[valueKeysOffset + sortedIndices[sortedIndicesOffset + index - 1u]];
  }
  if (head) { atomicAdd(&counts[countsOffset + group], 1u); }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-finish`,
      operation: 'GPUKeyJoin',
      variant: 'distinct-finish',
      invocationCount: leftUnique.length,
      bindings: [
        read('leftUnique', leftUnique),
        read('counts', counts),
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      body: `let group = leftUnique[leftUniqueOffset + index];
  var count = 0u;
  if (group != 0xffffffffu) { count = counts[countsOffset + group]; }
  output[outputOffset + index] = f32(count);`
    })
  );
  return nodes;
}
