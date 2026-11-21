// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort,
  GPUOverflowStatusPort
} from '../../analysis-core/contracts/status';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT,
  NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT
} from './network-flow-limits';

/** Bounded source-to-sink decomposition of the final feasible flow, not the augmentation history.
 * Each row has nodeCount - 1 edge slots, padded with 0xffffffff; lengths holds used-slot counts.
 * volumes holds one integral volume per path. Capacity is volumes.length (1..1024).
 * Cyclic circulation is omitted. Status propagates the solver's validity and iteration limit;
 * status.count counts published paths, requiredCount counts paths before capacity truncation,
 * and overflow reports dropped paths.
 */
export type GPUNetworkFlowPathOutput = {
  edgeRows: GraphDataView<'uint32'>;
  lengths: GraphDataView<'uint32'>;
  volumes: GraphDataView<'uint32'>;
  status: {
    count: GraphDataView<'uint32'>;
    requiredCount: GraphDataView<'uint32'>;
  } & Required<GPUOverflowStatusPort> &
    Required<GPUInvalidStatusPort> &
    Required<Pick<GPUIterationStatusPort, 'converged' | 'iterationLimitReached'>>;
};

/** CPU rows matching GPUNetworkFlowPathOutput. */
export type CPUNetworkFlowPaths = {
  edgeRows: Uint32Array;
  lengths: Uint32Array;
  volumes: Uint32Array;
  status: Record<keyof GPUNetworkFlowPathOutput['status'], Uint32Array>;
};

/** The decomposition is a bounded reference pass even when the flow solver is parallel. @internal */
export function validateNetworkFlowPathShape(
  id: string,
  nodeCount: number,
  edgeCount: number,
  pathCapacity: number
): void {
  validateCapacity(
    id,
    'nodeCount with output.paths',
    nodeCount,
    2,
    NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT
  );
  validateCapacity(
    id,
    'edgeCount with output.paths',
    edgeCount,
    0,
    NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT
  );
  validateCapacity(id, 'pathCapacity', pathCapacity, 1, NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT);
}

/** @internal */
export function getNetworkFlowPathViews(
  paths?: GPUNetworkFlowPathOutput
): GraphDataView<'uint32'>[] {
  return paths
    ? [paths.edgeRows, paths.lengths, paths.volumes, ...Object.values(paths.status)]
    : [];
}

/** @internal */
export function snapshotNetworkFlowPaths<
  Paths extends {
    edgeRows: unknown;
    lengths: unknown;
    volumes: unknown;
    status: {
      count: unknown;
      requiredCount: unknown;
      overflow: unknown;
      invalidCount: unknown;
      converged: unknown;
      iterationLimitReached: unknown;
    };
  }
>(paths?: Paths, id = 'network-flow'): Paths | undefined {
  if (!paths) return undefined;
  validateRequirement(id, typeof paths === 'object', 'output.paths must be an object');
  const snapshot = snapshotConstructionFields(paths, ['edgeRows', 'lengths', 'volumes', 'status']);
  validateRequirement(
    id,
    typeof snapshot.status === 'object' && snapshot.status !== null,
    'output.paths.status is required'
  );
  const output = Object.freeze({
    ...snapshot,
    status: snapshotConstructionFields(snapshot.status, [
      'count',
      'requiredCount',
      'overflow',
      'invalidCount',
      'converged',
      'iterationLimitReached'
    ])
  });
  validateNetworkFlowPathFields(id, output);
  return output;
}

/** Check required path objects before snapshotting or dereferencing a view's length. @internal */
export function validateNetworkFlowPathFields(
  id: string,
  paths: {
    edgeRows?: unknown;
    lengths?: unknown;
    volumes?: unknown;
    status?: {
      count?: unknown;
      requiredCount?: unknown;
      overflow?: unknown;
      invalidCount?: unknown;
      converged?: unknown;
      iterationLimitReached?: unknown;
    };
  }
): void {
  validateRequirement(id, typeof paths === 'object' && paths !== null, 'output.paths is required');
  validateRequirement(
    id,
    typeof paths.status === 'object' && paths.status !== null,
    'output.paths.status is required'
  );
  for (const name of ['edgeRows', 'lengths', 'volumes'] as const)
    validateRequirement(
      id,
      paths[name] !== undefined && paths[name] !== null,
      `output.paths.${name} is required`
    );
  for (const name of [
    'count',
    'requiredCount',
    'overflow',
    'invalidCount',
    'converged',
    'iterationLimitReached'
  ] as const)
    validateRequirement(
      id,
      paths.status?.[name] !== undefined && paths.status?.[name] !== null,
      `output.paths.status.${name} is required`
    );
}

/** @internal */
export function validateNetworkFlowPaths(
  id: string,
  nodeCount: number,
  edgeCount: number,
  paths?: GPUNetworkFlowPathOutput
): void {
  if (!paths) return;
  validateNetworkFlowPathFields(id, paths);
  validateNetworkFlowPathShape(id, nodeCount, edgeCount, paths.volumes.length);
  const lengths = {
    edgeRows: paths.volumes.length * (nodeCount - 1),
    lengths: paths.volumes.length,
    volumes: paths.volumes.length,
    count: 1,
    requiredCount: 1,
    overflow: 1,
    invalidCount: 1,
    converged: 1,
    iterationLimitReached: 1
  };
  const views = {...paths, ...paths.status};
  for (const [name, length] of Object.entries(lengths)) {
    const view = views[name as keyof typeof lengths];
    const path = `output.paths.${name in paths.status ? 'status.' : ''}${name}`;
    validateRequirement(id, Boolean(view), `${path} is required`);
    validateSinglePackedView(view, ['uint32'], id);
    validateRequirement(id, view.length === length, `${path} must hold ${length} rows`);
  }
}

/** Decompose a validated solver result. BFS visits outgoing edges in ascending input row order.
 * @internal */
export function computeNetworkFlowPaths(
  nodeCount: number,
  edges: Uint32Array,
  flow: Uint32Array,
  parameters: Uint32Array,
  summary: Uint32Array,
  pathCapacity: number
): CPUNetworkFlowPaths {
  const stride = nodeCount - 1;
  const edgeRows = new Uint32Array(pathCapacity * stride).fill(0xffffffff);
  const lengths = new Uint32Array(pathCapacity);
  const volumes = new Uint32Array(pathCapacity);
  const remaining = flow.slice();
  let requiredCount = 0;
  let remainingValue = summary[2];
  const [source, sink] = parameters;
  while (summary[0] && remainingValue > 0) {
    const parents = new Int32Array(nodeCount).fill(-1);
    parents[source] = -2;
    const queue = [source];
    for (let head = 0; head < queue.length && parents[sink] === -1; head++) {
      for (let edge = 0; edge < remaining.length; edge++) {
        const target = edges[edge * 2 + 1];
        if (edges[edge * 2] === queue[head] && remaining[edge] > 0 && parents[target] === -1) {
          parents[target] = edge;
          queue.push(target);
        }
      }
    }
    // Feasibility guarantees a source-to-sink path while any net flow remains.
    if (parents[sink] < 0) throw new Error('computeNetworkFlowPaths: flow is not feasible');
    const path: number[] = [];
    let volume = remainingValue;
    for (let vertex = sink; vertex !== source; vertex = edges[parents[vertex] * 2]) {
      const edge = parents[vertex];
      path.push(edge);
      volume = Math.min(volume, remaining[edge]);
    }
    path.reverse();
    for (const edge of path) remaining[edge] -= volume;
    if (requiredCount < pathCapacity) {
      edgeRows.set(path, requiredCount * stride);
      lengths[requiredCount] = path.length;
      volumes[requiredCount] = volume;
    }
    remainingValue -= volume;
    requiredCount++;
  }
  return {
    edgeRows,
    lengths,
    volumes,
    status: {
      count: Uint32Array.of(Math.min(requiredCount, pathCapacity)),
      requiredCount: Uint32Array.of(requiredCount),
      overflow: Uint32Array.of(Number(requiredCount > pathCapacity)),
      invalidCount: Uint32Array.of(1 - summary[0]),
      converged: Uint32Array.of(summary[0] * summary[1]),
      iterationLimitReached: Uint32Array.of(summary[0] * (1 - summary[1]))
    }
  };
}

/** Two passes keep each kernel within eight storage bindings. @internal */
export function createNetworkFlowPathNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  nodeCount: number,
  edges: GraphDataView<'uint32x2'>,
  flow: GraphDataView<'uint32'>,
  parameters: GraphDataView<'uint32'>,
  summary: GraphDataView<'uint32'>,
  paths: GPUNetworkFlowPathOutput
) {
  const edgeCount = flow.length;
  const pathCapacity = paths.volumes.length;
  const requiredCount = createTransientView(graph, `${id}-path-required-count`, 'uint32', 1);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-paths`,
      operation: 'GPUNetworkFlow',
      variant: 'reference-path-decomposition',
      // contract-allow: single-invocation -- optional reference decomposition: construction enforces NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT = 128 and NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT = 1024; the edge bound also caps path capacity and extracted paths because each path saturates at least one edge.
      invocationCount: 1,
      workgroupSize: 2,
      bindings: [
        {name: 'edges', view: edges, type: 'u32', access: 'read'},
        {name: 'flow', view: flow, type: 'u32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
        {name: 'summary', view: summary, type: 'u32', access: 'read'},
        {name: 'edgeRows', view: paths.edgeRows, type: 'u32', access: 'read_write'},
        {name: 'lengths', view: paths.lengths, type: 'u32', access: 'read_write'},
        {name: 'volumes', view: paths.volumes, type: 'u32', access: 'read_write'},
        {name: 'requiredCount', view: requiredCount, type: 'u32', access: 'read_write'}
      ],
      body: `
requiredCount[requiredCountOffset] = 0u;
for (var row = 0u; row < ${pathCapacity}u; row++) {
  lengths[lengthsOffset + row] = 0u;
  volumes[volumesOffset + row] = 0u;
}
for (var slot = 0u; slot < ${paths.edgeRows.length}u; slot++) { edgeRows[edgeRowsOffset + slot] = 0xffffffffu; }
if (summary[summaryOffset] == 0u) { return; }
let source = parameters[parametersOffset];
let sink = parameters[parametersOffset + 1u];
var remainingValue = summary[summaryOffset + 2u];
// Runtime bounds prevent constant-loop integer-array initialization miscompilation on Metal.
let vertexLimit = select(0u, ${nodeCount}u, summary[summaryOffset] != 0u);
let edgeLimit = min(${edgeCount}u, arrayLength(&flow) - flowOffset);
var remaining: array<u32, ${Math.max(1, edgeCount)}>;
var parents: array<u32, ${nodeCount}>;
var queue: array<u32, ${nodeCount}>;
for (var edge = 0u; edge < edgeLimit; edge++) { remaining[edge] = flow[flowOffset + edge]; }
for (var path = 0u; path < edgeLimit && remainingValue > 0u; path++) {
  for (var vertex = 0u; vertex < vertexLimit; vertex++) { parents[vertex] = 0xffffffffu; }
  parents[source] = 0xfffffffeu;
  queue[0] = source;
  var tail = 1u;
  for (var head = 0u; head < tail && parents[sink] == 0xffffffffu; head++) {
    let sourceVertex = queue[head];
    for (var edge = 0u; edge < edgeLimit; edge++) {
      let destinationVertex = edges[edgesOffset + edge * 2u + 1u];
      if (edges[edgesOffset + edge * 2u] == sourceVertex && remaining[edge] > 0u && parents[destinationVertex] == 0xffffffffu) {
        parents[destinationVertex] = edge;
        queue[tail] = destinationVertex;
        tail++;
      }
    }
  }
  var volume = remainingValue;
  var length = 0u;
  var vertex = sink;
  loop {
    if (vertex == source) { break; }
    let edge = parents[vertex];
    volume = min(volume, remaining[edge]);
    vertex = edges[edgesOffset + edge * 2u];
    length++;
  }
  vertex = sink;
  var slot = length;
  loop {
    if (vertex == source) { break; }
    let edge = parents[vertex];
    remaining[edge] -= volume;
    slot--;
    if (path < ${pathCapacity}u) { edgeRows[edgeRowsOffset + path * ${nodeCount - 1}u + slot] = edge; }
    vertex = edges[edgesOffset + edge * 2u];
  }
  if (path < ${pathCapacity}u) {
    lengths[lengthsOffset + path] = length;
    volumes[volumesOffset + path] = volume;
  }
  remainingValue -= volume;
  requiredCount[requiredCountOffset] = path + 1u;
}`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-path-status`,
      operation: 'GPUNetworkFlow',
      variant: 'path-status',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: [
        {name: 'required', view: requiredCount, type: 'u32', access: 'read'},
        {name: 'summary', view: summary, type: 'u32', access: 'read'},
        ...Object.entries(paths.status).map(([name, view]) => ({
          name,
          view,
          type: 'u32' as const,
          access: 'read_write' as const
        }))
      ],
      body: `
count[countOffset] = min(required[requiredOffset], ${pathCapacity}u);
requiredCount[requiredCountOffset] = required[requiredOffset];
overflow[overflowOffset] = select(0u, 1u, required[requiredOffset] > ${pathCapacity}u);
invalidCount[invalidCountOffset] = 1u - summary[summaryOffset];
converged[convergedOffset] = summary[summaryOffset] * summary[summaryOffset + 1u];
iterationLimitReached[iterationLimitReachedOffset] = summary[summaryOffset] * (1u - summary[summaryOffset + 1u]);`
    })
  ];
}
