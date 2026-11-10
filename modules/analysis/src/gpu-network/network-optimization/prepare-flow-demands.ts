// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

const ID = 'prepareGPUNetworkFlowDemands';
/** Largest vertex count: the super source and sink add two vertices to the 128 of `GPUNetworkFlow`. */
const MAXIMUM_NODE_COUNT = 126;
/** Largest edge count after the super source and sink edges, the limit of `GPUNetworkFlow`. */
const MAXIMUM_EDGE_COUNT = 1024;
/** Largest capacity or demand: the flow kernel keeps totals exact in 24 bits. */
const MAXIMUM_CAPACITY = 0xffffff;
/** Largest edge cost: the flow kernel rejects larger costs so its distance sums stay exact. */
const MAXIMUM_COST = 32767;

/** Explicit static super-source/sink adapter. Positive demand is net inflow, negative is supply. */
export function prepareGPUNetworkFlowDemands(props: {
  nodeCount: number;
  edges: readonly (readonly [number, number])[];
  capacities: ArrayLike<number>;
  demands: ArrayLike<number>;
  costs?: ArrayLike<number>;
}) {
  validateCapacity(ID, 'nodeCount', props.nodeCount, 1, MAXIMUM_NODE_COUNT);
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
  const edges = props.edges.flatMap(edge => {
    validateRequirement(
      ID,
      edge.every(vertex => Number.isInteger(vertex) && vertex >= 0 && vertex < props.nodeCount),
      'edges must have integer endpoints below nodeCount'
    );
    return [edge[0], edge[1]];
  });
  const capacities = Array.from(props.capacities);
  validateRequirement(
    ID,
    capacities.every(
      capacity => Number.isInteger(capacity) && capacity >= 0 && capacity <= MAXIMUM_CAPACITY
    ),
    `capacities must be integers in [0, ${MAXIMUM_CAPACITY}]`
  );
  const costs = props.costs ? Array.from(props.costs) : undefined;
  validateRequirement(
    ID,
    !costs || costs.every(cost => Number.isInteger(cost) && cost >= 0 && cost <= MAXIMUM_COST),
    `costs must be integers in [0, ${MAXIMUM_COST}]`
  );
  let balance = 0;
  let required = 0;
  const source = props.nodeCount;
  const sink = source + 1;
  for (let vertex = 0; vertex < props.nodeCount; vertex++) {
    const demand = props.demands[vertex];
    validateRequirement(
      ID,
      Number.isInteger(demand) && Math.abs(demand) <= MAXIMUM_CAPACITY,
      `demands must be integers with magnitude at most ${MAXIMUM_CAPACITY}`
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
  validateMaximumCount(
    ID,
    'edges including super source and sink edges',
    capacities.length,
    MAXIMUM_EDGE_COUNT
  );
  validateMaximumCount(
    ID,
    'total capacity',
    capacities.reduce((total, value) => total + value, 0),
    MAXIMUM_CAPACITY
  );
  return {
    nodeCount: props.nodeCount + 2,
    edges: Uint32Array.from(edges),
    capacities: Uint32Array.from(capacities),
    costs: costs ? Uint32Array.from(costs) : undefined,
    parameters: Uint32Array.of(source, sink, required),
    originalEdgeCount: props.edges.length
  };
}
