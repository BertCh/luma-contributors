// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/**
 * Gates the owned reachability passes of one Yen search. @internal
 *
 * Reachability's unconditional passes all use createWGSLKernelNode's automatic workgroup size;
 * their workload invocation counts are exact. A regression checks the resulting dispatch triples
 * against the recorded kernel geometry. This helper is specific to those passes, not arbitrary
 * command-node producers. Frontier rounds already have GPU conditions: combine their live triples
 * with the batch predicate immediately before each round, so skipped initialization cannot leave
 * stale frontier dispatches active after an earlier encoding.
 */
export function gateKShortestPathSearchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  nodes: readonly GPUCommandNode<Parameters>[],
  activeQueries: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const shapes = nodes
    .filter(node => !node.condition)
    .map(node => {
      const invocationCount = node.workload?.maximumInvocationCount;
      validateRequirement(
        id,
        typeof invocationCount === 'number',
        'shortest-path passes must declare an exact invocation count'
      );
      return [invocationCount] as const;
    });
  const directGate = createWGSLDispatchGate(graph, {
    id: `${id}-direct-dispatch`,
    operation: 'GPUKShortestPaths',
    shapes
  });
  const result: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-direct-gate`,
      operation: 'GPUKShortestPaths',
      variant: 'search-batch-gate',
      invocationCount: 1,
      bindings: [
        {name: 'activeQueries', view: activeQueries, type: 'u32', access: 'read'},
        directGate.binding
      ],
      body: `${directGate.disarmWGSL} if (activeQueries[activeQueriesOffset] != 0u) { ${directGate.armWGSL} }`
    })
  ];
  let directSlot = 0;
  for (const node of nodes) {
    validateRequirement(
      id,
      node.type === 'compute' && node.condition?.source !== 'cpu',
      'shortest-path passes must be GPU compute nodes'
    );
    if (!node.condition) {
      const gate = directGate.getGate(directSlot++);
      result.push({
        ...node,
        condition: gate.condition,
        resources: [...(node.resources ?? []), ...gate.extraResources]
      });
      continue;
    }
    const condition = node.condition;
    const frontierDispatch = graph.createDataView(condition.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: condition.byteOffset ?? 0
    });
    const frontierGate = createWGSLDispatchGate(graph, {
      id: `${node.id}-batch-dispatch`,
      operation: 'GPUKShortestPaths',
      shapes: [[1]]
    });
    result.push(
      createWGSLKernelNode(graph, {
        id: `${node.id}-batch-gate`,
        operation: 'GPUKShortestPaths',
        variant: 'search-frontier-gate',
        invocationCount: 1,
        bindings: [
          {name: 'activeQueries', view: activeQueries, type: 'u32', access: 'read'},
          {name: 'frontierDispatch', view: frontierDispatch, type: 'u32', access: 'read'},
          frontierGate.binding
        ],
        body: `dispatch[dispatchOffset] = 0u;
        dispatch[dispatchOffset + 1u] = 1u; dispatch[dispatchOffset + 2u] = 1u;
        if (activeQueries[activeQueriesOffset] != 0u) {
          dispatch[dispatchOffset] = frontierDispatch[frontierDispatchOffset];
          dispatch[dispatchOffset + 1u] = frontierDispatch[frontierDispatchOffset + 1u];
          dispatch[dispatchOffset + 2u] = frontierDispatch[frontierDispatchOffset + 2u];
        }`
      })
    );
    const gate = frontierGate.getGate(0);
    result.push({
      ...node,
      condition: gate.condition,
      resources: [...(node.resources ?? []), ...gate.extraResources]
    });
  }
  return result;
}
