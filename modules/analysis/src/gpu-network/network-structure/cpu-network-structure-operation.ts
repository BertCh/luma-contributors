// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkStructureOperation,
  type NetworkStructureOperationProps
} from './gpu-network-structure-operation';
import {
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS,
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS
} from './gpu-network-strongly-connected-components';
import {
  GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT,
  GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT
} from './gpu-network-structure';
import {computeBudgetedNetworkStrongComponents} from './network-strong-components-cpu-budget';

const NONE = 0xffffffff;

/** Packed CSR input; reverse slots are required only when resilience results are wanted. */
export type CPUNetworkStructureInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  reverseEdgeSlots?: Uint32Array;
};
/** Output layouts are identical to GPUNetworkStructure. */
export type CPUNetworkStructureResult = {
  strongComponents: Uint32Array;
  /** Exact traversal: one iteration. Budgeted SCC: executed trim/colour/backward/finalize steps. */
  strongComponentsStatus: {
    converged: Uint32Array;
    iterationCount: Uint32Array;
    iterationLimitReached: Uint32Array;
  };
  bridges: Uint32Array;
  articulationPoints: Uint32Array;
  biconnectedComponents: Uint32Array;
};

/**
 * Finds SCCs and, with reverse slots, undirected resilience using iterative Tarjan traversals.
 * SCC labels are the lowest vertex IDs; block labels are the lowest CSR slots. Invalid neighbor
 * IDs are ignored. Without reverse slots only strongComponents is meaningful.
 *
 * What differs from the GPU contributor:
 * - Exact SCC traversal takes O(vertices + slots) time and memory rather than mutual-reachability BFS.
 * - GPU-only 1024-vertex and 65536-slot caps are dropped. Integer output layouts are unchanged.
 * - Malformed CSR offsets throw an Error on CPU; the GPU clamps offset bounds in its kernels.
 *
 * Supplying maximumIterations selects the budgeted trim/colour/backward/finalize SCC state machine.
 * Exhaustion invalidates every SCC label while leaving resilience results valid. Within each phase,
 * CPU vertex order is deterministic; GPU scheduling may change the number of propagation steps.
 * Omit the budget for an exact Tarjan traversal and status [converged=1, iterationCount=1, limit=0].
 *
 * @throws If offsets or reverseEdgeSlots have inconsistent lengths or offsets are not valid CSR.
 */
export function computeNetworkStructureOnCPU(
  input: CPUNetworkStructureInput,
  options: {maximumIterations?: number} = {}
): CPUNetworkStructureResult {
  const {offsets, neighbors, reverseEdgeSlots} = input;
  const nodeCount = offsets.length - 1;
  const edgeCount = neighbors.length;
  if (options.maximumIterations !== undefined) {
    validateCapacity(
      'computeNetworkStructureOnCPU',
      'maximumIterations',
      options.maximumIterations,
      1,
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
    );
  }
  if (
    nodeCount < 1 ||
    offsets.some(
      (offset, index) => offset > edgeCount || (index > 0 && offset < offsets[index - 1])
    )
  )
    throw new Error(
      'offsets must be monotonic CSR offsets within neighbors length, with at least two rows'
    );
  if (reverseEdgeSlots && reverseEdgeSlots.length !== edgeCount)
    throw new Error('reverseEdgeSlots length must equal neighbors length');
  const strongComponents = new Uint32Array(nodeCount);
  const discovery = new Float64Array(nodeCount).fill(-1);
  const lowLink = new Float64Array(nodeCount);
  const active = new Uint8Array(nodeCount);
  const members: number[] = [];
  let clock = 0;
  if (options.maximumIterations === undefined) {
    for (let root = 0; root < nodeCount; root++) {
      if (discovery[root] >= 0) continue;
      const frames: {vertex: number; nextEdge: number}[] = [];
      const discoverVertex = (vertex: number) => {
        discovery[vertex] = lowLink[vertex] = clock++;
        active[vertex] = 1;
        members.push(vertex);
        frames.push({vertex, nextEdge: offsets[vertex]});
      };
      discoverVertex(root);
      while (frames.length) {
        const frame = frames[frames.length - 1];
        const vertex = frame.vertex;
        if (frame.nextEdge < offsets[vertex + 1]) {
          const destination = neighbors[frame.nextEdge++];
          if (destination >= nodeCount) continue;
          if (discovery[destination] < 0) discoverVertex(destination);
          else if (active[destination])
            lowLink[vertex] = Math.min(lowLink[vertex], discovery[destination]);
        } else {
          frames.pop();
          if (frames.length) {
            const parent = frames[frames.length - 1].vertex;
            lowLink[parent] = Math.min(lowLink[parent], lowLink[vertex]);
          }
          if (lowLink[vertex] === discovery[vertex]) {
            const component: number[] = [];
            let label = vertex;
            let member: number;
            do {
              member = members.pop()!;
              active[member] = 0;
              label = Math.min(label, member);
              component.push(member);
            } while (member !== vertex);
            for (const entry of component) strongComponents[entry] = label;
          }
        }
      }
    }
  }
  const bounded =
    options.maximumIterations === undefined
      ? undefined
      : computeBudgetedNetworkStrongComponents(offsets, neighbors, options.maximumIterations);
  if (bounded) strongComponents.set(bounded.strongComponents);
  const strongComponentsStatus = bounded?.strongComponentsStatus ?? {
    converged: Uint32Array.of(1),
    iterationCount: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0)
  };
  const bridges = new Uint32Array(edgeCount);
  const articulationPoints = new Uint32Array(nodeCount);
  const biconnectedComponents = new Uint32Array(edgeCount).fill(NONE);
  if (reverseEdgeSlots) {
    discovery.fill(-1);
    const parents = new Float64Array(nodeCount).fill(-1);
    const parentEdges = new Float64Array(nodeCount).fill(-1);
    const children = new Float64Array(nodeCount);
    const nextEdges = Float64Array.from(offsets.subarray(0, nodeCount));
    const edgeStack: number[] = [];
    clock = 0;
    for (let root = 0; root < nodeCount; root++) {
      if (discovery[root] >= 0) continue;
      let vertex = root;
      discovery[vertex] = lowLink[vertex] = clock++;
      for (;;) {
        if (nextEdges[vertex] < offsets[vertex + 1]) {
          const edge = nextEdges[vertex]++;
          const destination = neighbors[edge];
          const twin = reverseEdgeSlots[edge];
          if (destination >= nodeCount || twin >= edgeCount) continue;
          if (destination === vertex) {
            biconnectedComponents[edge] = Math.min(edge, twin);
            continue;
          }
          if (parentEdges[vertex] >= 0 && edge === reverseEdgeSlots[parentEdges[vertex]]) continue;
          if (discovery[destination] < 0) {
            edgeStack.push(edge);
            children[vertex]++;
            parents[destination] = vertex;
            parentEdges[destination] = edge;
            vertex = destination;
            discovery[vertex] = lowLink[vertex] = clock++;
          } else if (discovery[destination] < discovery[vertex]) {
            edgeStack.push(edge);
            lowLink[vertex] = Math.min(lowLink[vertex], discovery[destination]);
          }
        } else {
          const parent = parents[vertex];
          if (parent < 0) {
            articulationPoints[vertex] = children[vertex] > 1 ? 1 : 0;
            break;
          }
          const treeEdge = parentEdges[vertex];
          lowLink[parent] = Math.min(lowLink[parent], lowLink[vertex]);
          if (lowLink[vertex] > discovery[parent])
            bridges[treeEdge] = bridges[reverseEdgeSlots[treeEdge]] = 1;
          if (lowLink[vertex] >= discovery[parent]) {
            if (parents[parent] >= 0) articulationPoints[parent] = 1;
            const block: number[] = [];
            let label = NONE;
            let edge: number;
            do {
              edge = edgeStack.pop()!;
              block.push(edge);
              label = Math.min(label, edge, reverseEdgeSlots[edge]);
            } while (edge !== treeEdge);
            for (const slot of block)
              biconnectedComponents[slot] = biconnectedComponents[reverseEdgeSlots[slot]] = label;
          }
          vertex = parent;
        }
      }
    }
  }
  return {
    strongComponents,
    strongComponentsStatus,
    bridges,
    articulationPoints,
    biconnectedComponents
  };
}

/**
 * CPU twin preferred for reference structure analysis; GPU work and scratch caps are dropped.
 * As on GPU, reverse slots are ignored when only strongComponents is requested.
 * With no mode or budget, SCCs always use exact Tarjan, including above the GPU exact-route caps.
 * Exact SCC status reports one Tarjan traversal. Explicit parallel mode, or explicit auto/budget
 * selecting the large-graph route, uses the phase budget and invalidates labels on exhaustion. Step counts may differ
 * with GPU vertex scheduling, but the explicit maximum always binds both forms.
 */
export class CPUNetworkStructureOperation extends GPUCPUOperation {
  constructor(props: NetworkStructureOperationProps) {
    const id = props.id ?? 'network-structure';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'edgeCount', props.edgeCount, 0);
    validateRequirement(
      id,
      props.mode === undefined || ['auto', 'parallel', 'exact'].includes(props.mode),
      'mode must be auto, parallel or exact'
    );
    if (props.strongComponentsMaximumIterations !== undefined) {
      validateCapacity(
        id,
        'strongComponentsMaximumIterations',
        props.strongComponentsMaximumIterations,
        1,
        GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
      );
    }
    validateRequirement(
      id,
      !props.strongComponentsStatus || props.output.strongComponents !== undefined,
      'strongComponentsStatus requires output.strongComponents'
    );
    validateRequirement(
      id,
      Object.values(props.output).some(value => value !== undefined),
      'output must contain at least one result'
    );
    const hasResilienceOutput =
      props.output.bridges !== undefined ||
      props.output.articulationPoints !== undefined ||
      props.output.biconnectedComponents !== undefined;
    validateRequirement(
      id,
      !hasResilienceOutput || props.reverseEdgeSlots !== undefined,
      'reverseEdgeSlots is required for resilience output'
    );
    const operation = new GPUNetworkStructureOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        const neighbors = getCPUOperationUint32Rows(id, inputs, 'neighbors');
        if (offsets.length !== props.nodeCount + 1)
          throw new Error(`${id} offsets length must equal nodeCount + 1`);
        if (neighbors.length !== props.edgeCount)
          throw new Error(`${id} neighbors length must equal edgeCount`);
        if (
          props.topology.weights !== undefined &&
          getCPUOperationFloat32Rows(id, inputs, 'weights').length !== props.edgeCount
        )
          throw new Error(`${id} weights length must equal edgeCount`);
        if (
          props.topology.nodePositions !== undefined &&
          getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2').length !==
            props.nodeCount * 2
        )
          throw new Error(`${id} nodePositions length must equal nodeCount * 2`);
        const parallelStrongComponents =
          props.output.strongComponents !== undefined &&
          (props.mode === 'parallel' ||
            ((props.mode === 'auto' || props.strongComponentsMaximumIterations !== undefined) &&
              props.mode !== 'exact' &&
              (props.nodeCount > GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT ||
                props.edgeCount > GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT)));
        const result = computeNetworkStructureOnCPU(
          {
            offsets,
            neighbors,
            reverseEdgeSlots: hasResilienceOutput
              ? getCPUOperationUint32Rows(id, inputs, 'reverseEdgeSlots')
              : undefined
          },
          {
            maximumIterations: parallelStrongComponents
              ? (props.strongComponentsMaximumIterations ??
                GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS)
              : undefined
          }
        );
        const {strongComponentsStatus, ...primaryRows} = result;
        const outputRows = {...primaryRows, ...strongComponentsStatus};
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            outputRows[port as keyof typeof outputRows]
          ])
        );
      }
    });
  }
}
