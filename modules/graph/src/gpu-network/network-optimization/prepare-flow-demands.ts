// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';

/** Explicit static super-source/sink adapter. Positive demand is net inflow, negative is supply. */
export function prepareGPUNetworkFlowDemands(props: {
  nodeCount: number;
  edges: readonly (readonly [number, number])[];
  capacities: ArrayLike<number>;
  demands: ArrayLike<number>;
  costs?: ArrayLike<number>;
}) {
  assert(Number.isInteger(props.nodeCount) && props.nodeCount > 0 && props.nodeCount <= 126);
  assert(
    props.capacities.length === props.edges.length &&
      props.demands.length === props.nodeCount &&
      (!props.costs || props.costs.length === props.edges.length)
  );
  const edges = props.edges.flatMap(edge => {
    assert(
      edge.every(vertex => Number.isInteger(vertex) && vertex >= 0 && vertex < props.nodeCount)
    );
    return [edge[0], edge[1]];
  });
  const capacities = Array.from(props.capacities);
  assert(
    capacities.every(
      capacity => Number.isInteger(capacity) && capacity >= 0 && capacity <= 0xffffff
    )
  );
  const costs = props.costs ? Array.from(props.costs) : undefined;
  assert(!costs || costs.every(cost => Number.isInteger(cost) && cost >= 0 && cost <= 32767));
  let balance = 0;
  let required = 0;
  const source = props.nodeCount;
  const sink = source + 1;
  for (let vertex = 0; vertex < props.nodeCount; vertex++) {
    const demand = props.demands[vertex];
    assert(Number.isInteger(demand) && Math.abs(demand) <= 0xffffff);
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
  assert(
    balance === 0 &&
      capacities.length <= 1024 &&
      capacities.reduce((total, value) => total + value, 0) <= 0xffffff
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
