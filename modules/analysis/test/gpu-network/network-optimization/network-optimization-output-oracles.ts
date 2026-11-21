// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {NetworkLouvainOperationProps} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import type {NetworkRationalAccessOperationProps} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import type {NetworkTimeRoutingOperationProps} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

export function createLouvainOutputOracle(
  sparse: boolean,
  bounded: boolean,
  withApproximation: boolean
) {
  // One undirected pair: singleton modularity -1/2, merged modularity 0.
  // The first sweep merges, the second stabilizes, the next level stabilizes without a move.
  // Louvain is heuristic even when its local search converges: approximation is ALWAYS 1.
  const props: NetworkLouvainOperationProps = {
    nodeCount: 2,
    maximumLevels: bounded ? 1 : 2,
    maximumIterations: bounded ? 1 : 3,
    ...(sparse
      ? {topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'}}
      : {weights: 'weights'}),
    parameters: 'parameters',
    output: {
      communities: 'communities',
      objectiveTrace: 'trace',
      summary: 'summary',
      status: withApproximation ? {approximation: 'approximation'} : undefined
    }
  };
  const inputs: Record<string, OperationGraphInput> = {
    weights: {
      format: 'float32',
      values: sparse ? Float32Array.of(2, 2) : Float32Array.of(0, 2, 2, 0)
    },
    parameters: {format: 'float32', values: Float32Array.of(1, 0)}
  };
  if (sparse) {
    inputs['offsets'] = {format: 'uint32', values: Uint32Array.of(0, 1, 2)};
    inputs['neighbors'] = {format: 'uint32', values: Uint32Array.of(1, 0)};
  }
  const expected = {
    communities: [1, 1],
    trace: bounded ? [-0.5, 0] : [-0.5, 0, 0, 0, NaN, NaN, NaN],
    summary: bounded ? [1, 0, 1, 1] : [1, 1, 2, 3]
  };
  return {props, inputs, expected};
}

export const rationalStatusPorts = [
  'converged',
  'invalidCount',
  'iterationCount',
  'iterationLimitReached'
] as const;
export const rationalOracleScenes = ['stable', 'bounded', 'invalid', 'two-cycles'] as const;
export function createRationalOutputOracle(
  scene: (typeof rationalOracleScenes)[number],
  selected: (typeof rationalStatusPorts)[number] | 'absent'
) {
  // Equal zero travel costs initialize all demand at facility 0. Supply 2 per facility.
  // Demand 4 shifts exactly 2 in one cycle: loads 2,2, gap 0. Demand 3 shifts floor(1.5)=1:
  // loads 2,1, gap 1/2, weighted score (2*1+1*.5)/3=5/6 at the one-cycle limit.
  // A zero supply is rejected before assignment, so all status words except invalidCount are 0.
  // At step .25 and demand 4, only one unit can move per cycle: [4,0]->[3,1]->[2,2].
  // This gives summary [1,1,2,0], count 2 rather than the validity word 1.
  const props: NetworkRationalAccessOperationProps = {
    originCount: 1,
    facilityCount: 2,
    costs: 'costs',
    demand: 'demand',
    supply: 'supply',
    parameters: 'parameters',
    maximumCycles: scene === 'two-cycles' ? 2 : 1,
    initialLimitCycles: 0,
    output: {
      assignments: 'assignments',
      scores: 'scores',
      unservedDemand: 'unserved',
      summary: 'summary',
      status: selected === 'absent' ? undefined : {[selected]: selected}
    }
  };
  const inputs: Record<string, OperationGraphInput> = {
    costs: {format: 'float32', values: Float32Array.of(0, 0)},
    demand: {format: 'float32', values: Float32Array.of(scene === 'bounded' ? 3 : 4)},
    supply: {format: 'float32', values: Float32Array.of(scene === 'invalid' ? 0 : 2, 2)},
    parameters: {
      format: 'float32',
      values: Float32Array.of(
        1,
        1,
        scene === 'two-cycles' ? 0.25 : 1,
        scene === 'two-cycles' ? 0.25 : 1,
        50,
        0
      )
    }
  };
  const expected =
    scene === 'invalid'
      ? {assignments: [0, 0], scores: [NaN], unserved: [4], summary: [0, 0, 0, 0]}
      : scene === 'bounded'
        ? {
            assignments: [2, 1],
            scores: [0.8333333333333334],
            unserved: [0],
            summary: [1, 0, 1, 1056964608]
          }
        : {
            assignments: [2, 2],
            scores: [1],
            unserved: [0],
            summary: [1, 1, scene === 'two-cycles' ? 2 : 1, 0]
          };
  const status = {
    converged: [scene === 'stable' || scene === 'two-cycles' ? 1 : 0],
    invalidCount: [scene === 'invalid' ? 1 : 0],
    iterationCount: [scene === 'invalid' ? 0 : scene === 'two-cycles' ? 2 : 1],
    iterationLimitReached: [scene === 'bounded' ? 1 : 0]
  };
  return {props, inputs, expected, status};
}

export function createTimeRoutingOutputOracle(
  scene: 'stable' | 'bounded' | 'invalid',
  piecewise: boolean,
  withInvalidCount: boolean
) {
  // Physical vertex IDs differ from state rows. Reverse-listed chain transitions are row 0:1->2,
  // row 1:0->1. Depart at 4: first arrival 4+2+.5*4=8, then 8+3+.25*8=13.
  // Parents are transition rows [NONE,1,0,NONE], target state row is 2 (physical vertex 30).
  // One sweep reaches only state 1. Invalid departure rejects before any state is seeded.
  // IEEE-754 bits: 13 = 0x41500000 = 1095761920; +infinity = 0x7f800000 = 2139095040.
  const props: NetworkTimeRoutingOperationProps = {
    stateCount: 4,
    transitionCount: 2,
    stateVertices: 'stateVertices',
    transitions: 'transitions',
    costs: 'costs',
    endpoints: 'endpoints',
    departure: 'departure',
    costModel: piecewise ? 'piecewise-arrival' : 'affine-duration',
    samplesPerTransition: piecewise ? 2 : undefined,
    maximumRelaxations: scene === 'bounded' ? 1 : 4,
    output: {
      arrivals: 'arrivals',
      predecessorTransitions: 'predecessors',
      summary: 'summary',
      status: withInvalidCount ? {invalidCount: 'invalidCount'} : undefined
    }
  };
  const inputs: Record<string, OperationGraphInput> = {
    stateVertices: {format: 'uint32', values: Uint32Array.of(10, 20, 30, 99)},
    transitions: {format: 'uint32x2', values: Uint32Array.of(1, 2, 0, 1)},
    costs: {
      format: 'float32x2',
      values: piecewise
        ? Float32Array.of(0, 3, 16, 23, 0, 2, 16, 26)
        : Float32Array.of(3, 0.25, 2, 0.5)
    },
    endpoints: {format: 'uint32', values: Uint32Array.of(0, 30)},
    departure: {format: 'float32', values: Float32Array.of(scene === 'invalid' ? -1 : 4)}
  };
  const expected =
    scene === 'invalid'
      ? {
          arrivals: [Infinity, Infinity, Infinity, Infinity],
          predecessors: [4294967295, 4294967295, 4294967295, 4294967295],
          summary: [0, 0, 4294967295, 2139095040]
        }
      : scene === 'bounded'
        ? {
            arrivals: [4, 8, Infinity, Infinity],
            predecessors: [4294967295, 1, 4294967295, 4294967295],
            summary: [1, 0, 4294967295, 2139095040]
          }
        : {
            arrivals: [4, 8, 13, Infinity],
            predecessors: [4294967295, 1, 0, 4294967295],
            summary: [1, 1, 2, 1095761920]
          };
  return {props, inputs, expected, invalidCount: [scene === 'invalid' ? 1 : 0]};
}
