// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Assigns each accepted tuple its first source row without packing either zone ID. @internal */
export function createFlowLexicographicPairNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    origins: GraphDataView<'uint32'>;
    destinations: GraphDataView<'uint32'>;
    pairKeys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, origins, destinations, pairKeys} = props;
  const length = origins.length;
  const rows = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const indices = rows('indices');
  const flags = rows('flags');
  const segments = rows('segments');
  const representatives = rows('representatives');
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-iota`,
      operation: 'GPUFlowAggregation',
      bindings: [{name: 'indices', view: indices, type: 'u32', access: 'read_write'}],
      invocationCount: length,
      body: 'indices[indicesOffset + index] = index;'
    })
  ];
  const sorted = createStableLexicographicIndexSortNodes(graph, {
    id: `${id}-sort`,
    operation: 'GPUFlowAggregation',
    indices,
    indicesAreIdentity: true,
    packKeys: false,
    keys: [{view: origins}, {view: destinations}]
  });
  nodes.push(...sorted.nodes);
  const bindings = [
    {name: 'origins', view: origins, type: 'u32', access: 'read'},
    {name: 'destinations', view: destinations, type: 'u32', access: 'read'},
    {name: 'indices', view: sorted.sortedIndices, type: 'u32', access: 'read'}
  ] as const;
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-flags`,
      operation: 'GPUFlowAggregation',
      bindings: [...bindings, {name: 'flags', view: flags, type: 'u32', access: 'read_write'}],
      invocationCount: length,
      body: `let row = indices[indicesOffset + index];
  var isStart = index == 0u;
  if (index > 0u) {
    let previous = indices[indicesOffset + index - 1u];
    isStart = origins[originsOffset + previous] != origins[originsOffset + row] || destinations[destinationsOffset + previous] != destinations[destinationsOffset + row];
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: flags,
      output: segments,
      mode: 'inclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-representatives`,
      operation: 'GPUFlowAggregation',
      bindings: [
        bindings[2],
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'segments', view: segments, type: 'u32', access: 'read'},
        {name: 'representatives', view: representatives, type: 'u32', access: 'read_write'}
      ],
      invocationCount: length,
      body: `if (flags[flagsOffset + index] != 0u) {
    representatives[representativesOffset + segments[segmentsOffset + index] - 1u] = indices[indicesOffset + index];
  }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-scatter`,
      operation: 'GPUFlowAggregation',
      bindings: [
        bindings[0],
        bindings[2],
        {name: 'segments', view: segments, type: 'u32', access: 'read'},
        {name: 'representatives', view: representatives, type: 'u32', access: 'read'},
        {name: 'pairKeys', view: pairKeys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: length,
      body: `let row = indices[indicesOffset + index];
  pairKeys[pairKeysOffset + row] = select(representatives[representativesOffset + segments[segmentsOffset + index] - 1u], 0xffffffffu, origins[originsOffset + row] == 0xffffffffu);`
    })
  );
  return nodes;
}

/** Orders occupied hash slots by the original zone tuple, empty slots last. @internal */
export function createFlowLexicographicSlotNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tableValues: GraphDataView<'uint32'>;
    origins: GraphDataView<'uint32'>;
    destinations: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; slots: GraphDataView<'uint32'>} {
  const {id, tableValues, origins, destinations} = props;
  const length = tableValues.length;
  const sourceKeys = createTransientView(graph, `${id}-source-keys`, 'uint32', length);
  const targetKeys = createTransientView(graph, `${id}-target-keys`, 'uint32', length);
  const indices = createTransientView(graph, `${id}-indices`, 'uint32', length);
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-columns`,
    operation: 'GPUFlowAggregation',
    invocationCount: length,
    bindings: [
      {name: 'tableValues', view: tableValues, type: 'u32', access: 'read'},
      {name: 'origins', view: origins, type: 'u32', access: 'read'},
      {name: 'destinations', view: destinations, type: 'u32', access: 'read'},
      {name: 'sourceKeys', view: sourceKeys, type: 'u32', access: 'read_write'},
      {name: 'targetKeys', view: targetKeys, type: 'u32', access: 'read_write'},
      {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
    ],
    body: `let row = tableValues[tableValuesOffset + index];
  var source = 0xffffffffu;
  var destination = 0xffffffffu;
  if (row != 0xffffffffu) {
    source = origins[originsOffset + row];
    destination = destinations[destinationsOffset + row];
  }
  sourceKeys[sourceKeysOffset + index] = source;
  targetKeys[targetKeysOffset + index] = destination;
  indices[indicesOffset + index] = index;`
  });
  const sorted = createStableLexicographicIndexSortNodes(graph, {
    id,
    operation: 'GPUFlowAggregation',
    indices,
    indicesAreIdentity: true,
    packKeys: false,
    keys: [{view: sourceKeys}, {view: targetKeys}]
  });
  return {nodes: [node, ...sorted.nodes], slots: sorted.sortedIndices};
}
