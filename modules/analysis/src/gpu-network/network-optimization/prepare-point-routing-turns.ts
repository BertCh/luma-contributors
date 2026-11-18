// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Converts a complete, read-back line graph to point-routing pairs and separate turn penalties.
 * `weights` are encoded full arc costs (following base-edge cost plus penalty), as published by
 * GPUNetworkLineGraph. Only slots below the final CSR offset are read. Closed arcs (-1) are omitted.
 * Pass turnPenalties from the producer to preserve small penalties beside large base costs.
 * Without that column, subtraction retains only the precision of the supplied arc weights.
 * Negative or non-finite base costs close their edges; turns touching them are omitted.
 * This explicit host adapter never submits or reads GPU buffers itself. Incomplete graphs throw.
 */
export function prepareGPUNetworkPointRoutingTurns(props: {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  /** Optional separate line-graph penalty column, before adding the base cost. */
  turnPenalties?: Float32Array;
  costs: Float32Array;
  overflow: number;
}): {turns: Uint32Array; turnPenalties: Float32Array} {
  const id = 'prepareGPUNetworkPointRoutingTurns';
  const edgeCount = props.costs.length;
  validateRequirement(id, props.overflow === 0, 'overflow must be zero for a complete turn graph');
  validateRequirement(
    id,
    props.offsets.length === edgeCount + 1,
    'offsets must hold costs.length + 1 rows'
  );
  validateRequirement(
    id,
    props.weights.length === props.neighbors.length,
    'weights must hold one row per neighbor'
  );
  validateRequirement(
    id,
    !props.turnPenalties || props.turnPenalties.length === props.neighbors.length,
    'turnPenalties must hold one row per neighbor'
  );
  validateRequirement(
    id,
    props.offsets[0] === 0 &&
      props.offsets[edgeCount] <= props.neighbors.length &&
      props.offsets.every((offset, index) => index === 0 || offset >= props.offsets[index - 1]),
    'offsets must be zero-based monotonic CSR within neighbors.length'
  );
  const turns: number[] = [];
  const turnPenalties: number[] = [];
  for (let incoming = 0; incoming < edgeCount; incoming++) {
    if (!Number.isFinite(props.costs[incoming]) || props.costs[incoming] < 0) continue;
    for (let slot = props.offsets[incoming]; slot < props.offsets[incoming + 1]; slot++) {
      const weight = props.weights[slot];
      if (weight === -1) continue;
      const outgoing = props.neighbors[slot];
      const baseCost = props.costs[outgoing];
      if (outgoing < edgeCount && (!Number.isFinite(baseCost) || baseCost < 0)) continue;
      const penalty = props.turnPenalties?.[slot] ?? weight - baseCost;
      validateRequirement(
        id,
        outgoing < edgeCount &&
          Number.isFinite(baseCost) &&
          baseCost >= 0 &&
          Number.isFinite(weight) &&
          weight >= baseCost &&
          Number.isFinite(penalty) &&
          penalty >= 0,
        'active arcs must reference an edge and have finite weights at least its nonnegative base cost'
      );
      turns.push(incoming, outgoing);
      turnPenalties.push(penalty);
    }
  }
  return {turns: Uint32Array.from(turns), turnPenalties: Float32Array.from(turnPenalties)};
}
