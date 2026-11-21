// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {NetworkCostMatrixOperationProps} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

export function createCostMatrixOutputOracle(gather: boolean) {
  // Chain 0 -(2)-> 1 -(3)-> 2, isolated 3. Row zero starts at 0 with cost 1;
  // row one starts at 2 with cost .5; row two has an invalid start. Inputs are row-permuted.
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 2, 2, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(2, 3)
    },
    startNodes: Uint32Array.of(2, 0, 99),
    startCosts: Float32Array.of(0.5, 1, 0),
    startRows: Uint32Array.of(1, 0, 2),
    targetNodes: gather ? Uint32Array.of(2, 0, 3, 2, 99) : undefined
  };
  const props: NetworkCostMatrixOperationProps = {
    nodeCount: 4,
    rowCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    startCosts: 'startCosts',
    startRows: 'startRows',
    targetNodes: gather ? 'targets' : undefined,
    targetCount: gather ? 5 : undefined,
    laneCount: 2,
    maximumIterations: 8,
    localIterations: 1,
    output: {costs: 'costs'}
  };
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {format: 'uint32', values: input.topology.offsets},
    neighbors: {format: 'uint32', values: input.topology.neighbors},
    weights: {format: 'float32', values: input.topology.weights},
    starts: {format: 'uint32', values: input.startNodes},
    startCosts: {format: 'float32', values: input.startCosts},
    startRows: {format: 'uint32', values: input.startRows}
  };
  if (input.targetNodes) inputs['targets'] = {format: 'uint32', values: input.targetNodes};
  const expected = gather
    ? [
        6,
        1,
        Infinity,
        6,
        Infinity,
        0.5,
        Infinity,
        Infinity,
        0.5,
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        Infinity
      ]
    : [
        1,
        3,
        6,
        Infinity,
        Infinity,
        Infinity,
        0.5,
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        Infinity
      ];
  return {input, props, inputs, expected};
}

export function createCostMatrixSpillOracle(neighborCount: 256 | 257) {
  // One invocation scans 0's neighbors in CSR order. A 256-entry local queue holds all
  // neighbors in the control; the 257th neighbor spills to round two. Only that last
  // neighbor has a child, so the one-round target is 2 for 256 and Infinity for 257.
  const nodeCount = neighborCount + 2;
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {
      format: 'uint32',
      values: Uint32Array.from({length: nodeCount + 1}, (_, row) =>
        row === 0 ? 0 : row <= neighborCount ? neighborCount : neighborCount + 1
      )
    },
    neighbors: {
      format: 'uint32',
      values: Uint32Array.from({length: neighborCount + 1}, (_, row) => row + 1)
    },
    weights: {format: 'float32', values: new Float32Array(neighborCount + 1).fill(1)},
    starts: {format: 'uint32', values: Uint32Array.of(0)},
    targets: {format: 'uint32', values: Uint32Array.of(neighborCount + 1)}
  };
  const props: NetworkCostMatrixOperationProps = {
    nodeCount,
    rowCount: 1,
    laneCount: 1,
    maximumIterations: 1,
    localIterations: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    targetNodes: 'targets',
    targetCount: 1,
    output: {costs: 'costs', status: {converged: 'converged', iterationLimitReached: 'limited'}}
  };
  return {
    props,
    inputs,
    expected: {
      costs: [neighborCount === 256 ? 2 : Infinity],
      converged: [neighborCount === 256 ? 1 : 0],
      limited: [neighborCount === 256 ? 0 : 1]
    }
  };
}

export function createCostMatrixDefaultOracle() {
  // A 17-edge chain distinguishes exactly 16 local hops from every smaller/larger default.
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {
      format: 'uint32',
      values: Uint32Array.from({length: 19}, (_, row) => Math.min(row, 17))
    },
    neighbors: {format: 'uint32', values: Uint32Array.from({length: 17}, (_, row) => row + 1)},
    weights: {format: 'float32', values: new Float32Array(17).fill(1)},
    starts: {format: 'uint32', values: Uint32Array.of(0)}
  };
  const props: NetworkCostMatrixOperationProps = {
    nodeCount: 18,
    rowCount: 1,
    laneCount: 1,
    maximumIterations: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs', status: {converged: 'converged', iterationLimitReached: 'limited'}}
  };
  return {
    props,
    inputs,
    expected: {
      costs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, Infinity],
      converged: [0],
      limited: [1]
    }
  };
}

export function createCostMatrixRewriteOracle() {
  // One compiled one-hop chain: seed 0/cost 1 gives [1,3,Infinity], with node 1 queued.
  // Replacing only the seed with terminal 2 gives [Infinity,Infinity,1] and an empty frontier.
  const props: NetworkCostMatrixOperationProps = {
    nodeCount: 3,
    rowCount: 1,
    laneCount: 1,
    maximumIterations: 1,
    localIterations: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    startCosts: 'startCosts',
    output: {costs: 'costs', status: {converged: 'converged', iterationLimitReached: 'limited'}}
  };
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
    neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
    weights: {format: 'float32', values: Float32Array.of(2, 3)},
    starts: {format: 'uint32', values: Uint32Array.of(0)},
    startCosts: {format: 'float32', values: Float32Array.of(1)}
  };
  const first = {costs: [1, 3, Infinity], converged: [0], limited: [1]};
  const second = {costs: [Infinity, Infinity, 1], converged: [1], limited: [0]};
  return {props, inputs, first, second};
}
