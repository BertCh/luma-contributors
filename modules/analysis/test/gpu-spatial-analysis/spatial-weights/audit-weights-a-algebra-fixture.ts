// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CPUSpatialWeightsAlgebraInput,
  CPUSpatialWeightsAlgebraOptions
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';

export const AUDIT_ALGEBRA_LEFT = {
  offsets: Uint32Array.of(0, 3, 5, 7, 8, 9, 10, 10, 10),
  neighbors: Uint32Array.of(0, 1, 3, 0, 2, 3, 4, 0, 5, 4),
  weights: Float32Array.of(9, 2, 4, 3, 5, 6, 7, 8, 2, 3),
  distances: Float32Array.of(0, 10, 20, 10, 12, 14, 16, 20, 10, 10)
};
export const AUDIT_ALGEBRA_RIGHT = {
  offsets: Uint32Array.of(0, 2, 3, 4, 6, 6, 7, 8, 8),
  neighbors: Uint32Array.of(1, 2, 2, 4, 0, 5, 6, 5),
  weights: Float32Array.of(7, 11, 13, 17, 19, 23, 29, 31),
  distances: Float32Array.of(99, 98, 97, 96, 95, 94, 93, 92)
};
export const AUDIT_ALGEBRA_MASK = Uint32Array.of(1, 0, 2, 1, 1, 1, 1, 1);
export const AUDIT_ALGEBRA_GROUPS = Uint32Array.of(0, 1, 0, 1, 2, 2, 99, 3);
export const AUDIT_ALGEBRA_SELF = Float32Array.of(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4);

type OracleCase = {
  name: string;
  input: CPUSpatialWeightsAlgebraInput;
  options: CPUSpatialWeightsAlgebraOptions;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  distances?: number[];
};

/** Literal dense-set/SciPy/Kronecker expectations from audit-weights-a-algebra.oracle.py. */
export const AUDIT_ALGEBRA_CASES: OracleCase[] = [
  {
    name: 'union',
    input: {left: AUDIT_ALGEBRA_LEFT, right: AUDIT_ALGEBRA_RIGHT},
    options: {operation: 'union', weightRule: 'sum', capacity: 64, distances: true},
    offsets: [0, 4, 6, 8, 10, 11, 13, 14, 14],
    neighbors: [0, 1, 2, 3, 0, 2, 3, 4, 0, 5, 5, 4, 6, 5],
    weights: [9, 9, 11, 4, 3, 18, 6, 24, 27, 23, 2, 3, 29, 31],
    distances: [0, 10, 98, 20, 10, 12, 14, 16, 20, 94, 10, 10, 93, 92]
  },
  {
    name: 'intersection',
    input: {left: AUDIT_ALGEBRA_LEFT, right: AUDIT_ALGEBRA_RIGHT},
    options: {operation: 'intersection', weightRule: 'sum', capacity: 64, distances: true},
    offsets: [0, 1, 2, 3, 4, 4, 4, 4, 4],
    neighbors: [1, 2, 4, 0],
    weights: [9, 18, 24, 27],
    distances: [10, 12, 16, 20]
  },
  {
    name: 'difference',
    input: {left: AUDIT_ALGEBRA_LEFT, right: AUDIT_ALGEBRA_RIGHT},
    options: {operation: 'difference', capacity: 64, distances: true},
    offsets: [0, 2, 3, 4, 4, 5, 6, 6, 6],
    neighbors: [0, 3, 0, 3, 5, 4],
    weights: [9, 4, 3, 6, 2, 3],
    distances: [0, 20, 10, 14, 10, 10]
  },
  {
    name: 'symmetric-difference',
    input: {left: AUDIT_ALGEBRA_LEFT, right: AUDIT_ALGEBRA_RIGHT},
    options: {operation: 'symmetric-difference', capacity: 64, distances: true},
    offsets: [0, 3, 4, 5, 6, 7, 9, 10, 10],
    neighbors: [0, 2, 3, 0, 3, 5, 5, 4, 6, 5],
    weights: [9, 11, 4, 3, 6, 23, 2, 3, 29, 31],
    distances: [0, 98, 20, 10, 14, 94, 10, 10, 93, 92]
  },
  {
    name: 'exact2',
    input: {weights: AUDIT_ALGEBRA_LEFT},
    options: {operation: 'higherOrder', order: 2, workCapacity: 64, capacity: 64},
    offsets: [0, 1, 3, 5, 6, 6, 6, 6, 6],
    neighbors: [2, 3, 4, 0, 5, 1],
    weights: [1, 1, 1, 1, 1, 1]
  },
  {
    name: 'cumulative2',
    input: {weights: AUDIT_ALGEBRA_LEFT},
    options: {operation: 'higherOrder', order: 2, cumulative: true, workCapacity: 64, capacity: 64},
    offsets: [0, 3, 7, 11, 13, 14, 15, 15, 15],
    neighbors: [1, 2, 3, 0, 2, 3, 4, 0, 3, 4, 5, 0, 1, 5, 4],
    weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  },
  {
    name: 'subgraph',
    input: {weights: AUDIT_ALGEBRA_LEFT, mask: AUDIT_ALGEBRA_MASK},
    options: {operation: 'subgraph', capacity: 64, distances: true},
    offsets: [0, 2, 2, 4, 5, 6, 7, 7, 7],
    neighbors: [0, 3, 3, 4, 0, 5, 4],
    weights: [9, 4, 6, 7, 8, 2, 3],
    distances: [0, 20, 14, 16, 20, 10, 10]
  },
  {
    name: 'selfWeight',
    input: {weights: AUDIT_ALGEBRA_LEFT, selfWeights: AUDIT_ALGEBRA_SELF},
    options: {operation: 'selfWeight', capacity: 64, distances: true},
    offsets: [0, 3, 6, 9, 11, 13, 15, 16, 17],
    neighbors: [0, 1, 3, 0, 1, 2, 2, 3, 4, 0, 3, 4, 5, 4, 5, 6, 7],
    weights: [0.5, 2, 4, 3, 1, 5, 1.5, 6, 7, 8, 2, 2.5, 2, 3, 3, 3.5, 4],
    distances: [0, 10, 20, 10, 0, 12, 0, 14, 16, 20, 0, 0, 10, 10, 0, 0, 0]
  },
  {
    name: 'block',
    input: {groupIds: AUDIT_ALGEBRA_GROUPS},
    options: {operation: 'block', groupCount: 4, capacity: 64},
    offsets: [0, 1, 2, 3, 4, 5, 6, 6, 6],
    neighbors: [2, 3, 0, 1, 5, 4],
    weights: [1, 1, 1, 1, 1, 1]
  },
  {
    name: 'period-major',
    input: {weights: AUDIT_ALGEBRA_LEFT},
    options: {
      operation: 'spaceTime',
      periodCount: 2,
      layout: 'period-major',
      capacity: 64,
      lags: [
        {lag: 0, weight: 2},
        {lag: 1, weight: 0.5, includeSelf: true}
      ]
    },
    offsets: [0, 3, 5, 7, 8, 9, 10, 10, 10, 16, 21, 26, 29, 32, 35, 36, 37],
    neighbors: [
      0, 1, 3, 0, 2, 3, 4, 0, 5, 4, 0, 1, 3, 8, 9, 11, 0, 1, 2, 8, 10, 2, 3, 4, 11, 12, 0, 3, 8, 4,
      5, 13, 4, 5, 12, 6, 7
    ],
    weights: [
      18, 4, 8, 6, 10, 12, 14, 16, 4, 6, 0.5, 1, 2, 18, 4, 8, 1.5, 0.5, 2.5, 6, 10, 0.5, 3, 3.5, 12,
      14, 4, 0.5, 16, 0.5, 1, 4, 1.5, 0.5, 6, 0.5, 0.5
    ]
  },
  {
    name: 'cell-major',
    input: {weights: AUDIT_ALGEBRA_LEFT},
    options: {
      operation: 'spaceTime',
      periodCount: 2,
      layout: 'cell-major',
      capacity: 64,
      lags: [
        {lag: 0, weight: 2},
        {lag: 1, weight: 0.5, includeSelf: true}
      ]
    },
    offsets: [0, 3, 9, 11, 16, 18, 23, 24, 27, 28, 31, 32, 35, 35, 36, 36, 37],
    neighbors: [
      0, 2, 6, 0, 1, 2, 3, 6, 7, 0, 4, 0, 1, 2, 4, 5, 6, 8, 4, 6, 7, 8, 9, 0, 0, 1, 6, 10, 8, 10,
      11, 8, 8, 9, 10, 12, 14
    ],
    weights: [
      18, 4, 8, 0.5, 18, 1, 4, 2, 8, 6, 10, 1.5, 6, 0.5, 2.5, 10, 12, 14, 0.5, 3, 12, 3.5, 14, 16,
      4, 16, 0.5, 4, 0.5, 1, 4, 6, 1.5, 6, 0.5, 0.5, 0.5
    ]
  }
];
