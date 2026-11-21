// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT,
  NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT,
  NETWORK_FLOW_REFERENCE_MAXIMUM_TOTAL_CAPACITY,
  NETWORK_FLOW_PARALLEL_MAXIMUM_NODE_COUNT,
  NETWORK_FLOW_PARALLEL_MAXIMUM_EDGE_COUNT,
  NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY,
  NETWORK_FLOW_MAXIMUM_UNIT_COST,
  NETWORK_FLOW_MAXIMUM_DISTANCE
} from './network-flow-limits';

const ID = 'prepareGPUNetworkFlowDemands';

/** Source and target vertex indices of one directed edge. */
export type GPUNetworkEdgeEndpoints = readonly [source: number, target: number];

/** Explicit static super-source/sink adapter. Positive demand is net inflow, negative is supply.
 * Auto resolves reference/parallel against expanded vertex/edge counts and total capacity;
 * pass the returned solver to GPUNetworkFlow, including for a small graph with wide capacities.
 * Device binding and command-budget limits still apply when constructing the contributor.
 */
export function prepareGPUNetworkFlowDemands(props: {
  nodeCount: number;
  edges: readonly GPUNetworkEdgeEndpoints[];
  capacities: ArrayLike<number>;
  demands: ArrayLike<number>;
  costs?: ArrayLike<number>;
  solver?: 'auto' | 'reference' | 'parallel';
}) {
  validateCapacity(
    ID,
    'nodeCount',
    props.nodeCount,
    1,
    NETWORK_FLOW_PARALLEL_MAXIMUM_NODE_COUNT - 2
  );
  validateRequirement(
    ID,
    props.solver === undefined || ['auto', 'reference', 'parallel'].includes(props.solver),
    'solver must be auto, reference or parallel'
  );
  validateRequirement(
    ID,
    props.capacities.length === props.edges.length,
    'capacities must hold one entry per edge'
  );
  validateRequirement(
    ID,
    props.demands.length === props.nodeCount,
    'demands must hold one entry per vertex'
  );
  validateRequirement(
    ID,
    !props.costs || props.costs.length === props.edges.length,
    'costs must hold one entry per edge'
  );
  const edges: number[] = [];
  for (let row = 0; row < props.edges.length; row++) {
    const edge = props.edges[row];
    validateRequirement(
      ID,
      Array.isArray(edge),
      `edges[${row}] must contain [source, target]; received ${edge}`
    );
    for (const [endpoint, name] of ['source', 'target'].entries()) {
      validateRequirement(
        ID,
        endpoint in edge &&
          Number.isInteger(edge[endpoint]) &&
          edge[endpoint] >= 0 &&
          edge[endpoint] < props.nodeCount,
        `edges[${row}].${name} in [source, target] must be a present integer in [0, ${props.nodeCount - 1}], received ${edge?.[endpoint]}`
      );
    }
    validateRequirement(
      ID,
      edge.length === 2,
      `edges[${row}] must contain exactly [source, target]; received length ${edge.length}`
    );
    edges.push(edge[0], edge[1]);
  }
  const capacities = Array.from(props.capacities);
  validateRequirement(
    ID,
    capacities.every(
      capacity =>
        Number.isInteger(capacity) &&
        capacity >= 0 &&
        capacity <= NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY
    ),
    `capacities must be integers in [0, ${NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY}]`
  );
  const costs = props.costs ? Array.from(props.costs) : undefined;
  validateRequirement(
    ID,
    !costs ||
      costs.every(
        cost => Number.isInteger(cost) && cost >= 0 && cost <= NETWORK_FLOW_MAXIMUM_UNIT_COST
      ),
    `costs must be integers in [0, ${NETWORK_FLOW_MAXIMUM_UNIT_COST}]`
  );
  let balance = 0;
  let required = 0;
  const source = props.nodeCount;
  const sink = source + 1;
  for (let vertex = 0; vertex < props.nodeCount; vertex++) {
    const demand = props.demands[vertex];
    validateRequirement(
      ID,
      Number.isInteger(demand) && Math.abs(demand) <= NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY,
      `demands must be integers with magnitude at most ${NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY}`
    );
    balance += demand;
    if (demand === 0) continue;
    if (demand < 0) edges.push(source, vertex);
    else {
      edges.push(vertex, sink);
      required += demand;
    }
    capacities.push(Math.abs(demand));
    costs?.push(0);
  }
  validateRequirement(ID, balance === 0, 'demands must sum to zero');
  const nodeCount = props.nodeCount + 2;
  const totalCapacity = capacities.reduce((total, value) => total + value, 0);
  const solver: 'reference' | 'parallel' =
    props.solver === 'parallel' ||
    ((props.solver ?? 'auto') === 'auto' &&
      (nodeCount > NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT ||
        capacities.length > NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT ||
        totalCapacity > NETWORK_FLOW_REFERENCE_MAXIMUM_TOTAL_CAPACITY))
      ? 'parallel'
      : 'reference';
  validateCapacity(
    ID,
    'expanded nodeCount',
    nodeCount,
    2,
    solver === 'reference'
      ? NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT
      : NETWORK_FLOW_PARALLEL_MAXIMUM_NODE_COUNT
  );
  validateMaximumCount(
    ID,
    'edges including super source and sink edges',
    capacities.length,
    solver === 'reference'
      ? NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT
      : NETWORK_FLOW_PARALLEL_MAXIMUM_EDGE_COUNT
  );
  validateMaximumCount(
    ID,
    'total capacity',
    totalCapacity,
    solver === 'reference'
      ? NETWORK_FLOW_REFERENCE_MAXIMUM_TOTAL_CAPACITY
      : NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY
  );
  const maximumCost =
    solver === 'reference'
      ? NETWORK_FLOW_MAXIMUM_UNIT_COST
      : Math.min(
          NETWORK_FLOW_MAXIMUM_UNIT_COST,
          Math.floor(NETWORK_FLOW_MAXIMUM_DISTANCE / nodeCount)
        );
  validateRequirement(
    ID,
    !costs || costs.every(cost => cost <= maximumCost),
    `costs must be at most ${maximumCost} for solver ${solver} at expanded nodeCount ${nodeCount}`
  );
  return {
    nodeCount,
    solver,
    edges: Uint32Array.from(edges),
    capacities: Uint32Array.from(capacities),
    costs: costs ? Uint32Array.from(costs) : undefined,
    parameters: Uint32Array.of(source, sink, required),
    originalEdgeCount: props.edges.length
  };
}
