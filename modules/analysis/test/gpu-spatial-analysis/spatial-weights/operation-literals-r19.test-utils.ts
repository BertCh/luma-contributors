// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Three unit squares: 0--1 share a full edge; 1 and 2 meet only at (2,1).
// These literals describe geometry/edge sets directly, without a contributor or CPU twin oracle.
export const CONTIGUITY_INPUT = {
  positions: Float32Array.of(
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    2,
    0,
    2,
    1,
    1,
    1,
    2,
    1,
    3,
    1,
    3,
    2,
    2,
    2
  ),
  ringOffsets: Uint32Array.of(0, 4, 8, 12),
  polygonOffsets: Uint32Array.of(0, 1, 2, 3),
  featureOffsets: Uint32Array.of(0, 1, 2, 3)
};
export const CONTIGUITY_EXPECTATIONS = [
  {criterion: 'rook' as const, offsets: [0, 1, 2, 2], neighbors: [1, 0]},
  {criterion: 'queen' as const, offsets: [0, 1, 3, 4], neighbors: [1, 0, 2, 1]}
];

// Directed left={0->1,0->2,2->0}; right={0->2,1->0,2->1}.
export const ALGEBRA_LEFT = {
  offsets: Uint32Array.of(0, 2, 2, 3),
  neighbors: Uint32Array.of(1, 2, 0),
  weights: Float32Array.of(1, 1, 1)
};
export const ALGEBRA_RIGHT = {
  offsets: Uint32Array.of(0, 1, 2, 3),
  neighbors: Uint32Array.of(2, 0, 1),
  weights: Float32Array.of(1, 1, 1)
};
export const ALGEBRA_EXPECTATIONS = [
  {operation: 'union' as const, offsets: [0, 2, 3, 5], neighbors: [1, 2, 0, 0, 1]},
  {operation: 'intersection' as const, offsets: [0, 1, 1, 1], neighbors: [2]},
  {operation: 'difference' as const, offsets: [0, 1, 1, 2], neighbors: [1, 0]}
];

// Row 0 has exactly two violations: out-of-range neighbor 3 (bit 2) and negative weight
// (bit 16). Rows 1 and 2 are valid, so invalidCount=1, not the number of violations (2).
export const VALIDATION_INPUT = {
  offsets: Uint32Array.of(0, 1, 2, 2),
  neighbors: Uint32Array.of(3, 0),
  weights: Float32Array.of(-1, 1)
};
export const VALIDATION_EXPECTED = {flags: [18, 0, 0], invalidCount: [1]};

export function cpuVector(values: Uint32Array | Float32Array) {
  return {
    format: values instanceof Uint32Array ? ('uint32' as const) : ('float32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

export const CONTIGUITY_PORTS = {
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  polygonOffsets: 'polygonOffsets',
  featureOffsets: 'featureOffsets'
};
export const ALGEBRA_PORTS = {
  left: {offsets: 'leftOffsets', neighbors: 'leftNeighbors', weights: 'leftWeights'},
  right: {offsets: 'rightOffsets', neighbors: 'rightNeighbors', weights: 'rightWeights'}
};
export const WEIGHTS_OUTPUT = {
  offsets: 'offsets',
  neighbors: 'neighbors',
  weights: 'weights',
  status: {
    overflow: 'overflow',
    requiredCount: 'requiredCount',
    candidateOverflow: 'candidateOverflow'
  }
};
