// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {ISLAND_ATTACHMENT_NEAREST_DECLARATIONS} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {createSpatialStatisticsCollectiveNode} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-collective-node';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-counts-wgsl';
import {createSpatialStatisticsRowNode} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-row-dispatch';
import {
  createCSRStages,
  getMergeVisitWGSL,
  getContainsFunctionWGSL,
  type SpatialWeightsMergeOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-algebra-wgsl';
import type {
  GPUSpatialWeights,
  GPUSpatialWeightsOutput
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {
  getBinaryMergeStages,
  getSelfWeightStages,
  getSubgraphStages
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-row-local-stages';
import {
  COMBINE_RULES,
  type GPUSpatialWeightsCombineRule
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {
  getSpaceTimeStages,
  type GPUSpaceTimeWeightsLayout
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-space-time-stages';
import {createStatisticsKernelNode} from '../../../src/gpu-spatial-analysis/spatial-weights/statistics-kernel-node';

/** Read storage-only transient words without requiring copy-source usage on the producer. */
export function addWeightsAuditReadback(
  graph: GPUCommandGraph,
  id: string,
  source: GraphDataView<'uint32'>,
  destination: GraphDataView<'uint32'>
): GPUCommandNode {
  const node = createWGSLKernelNode(graph, {
    id,
    operation: 'WeightsAuditReadback',
    invocationCount: source.length,
    bindings: [
      {name: 'sourceValues', view: source, type: 'u32', access: 'read'},
      {name: 'destinationValues', view: destination, type: 'u32', access: 'read_write'}
    ],
    body: 'destinationValues[destinationValuesOffset + index] = sourceValues[sourceValuesOffset + index];'
  });
  graph.add(node);
  return node;
}

/** Fixed 32-row fixture; eight groups reduce four small integers each. */
export function createCollectiveAuditNode(
  graph: GPUCommandGraph,
  values: GraphDataView<'float32'>,
  result: GraphDataView<'float32'>,
  control?: GraphDataView<'uint32'>
) {
  return createSpatialStatisticsCollectiveNode(graph, {
    id: 'collective-audit',
    operation: 'audit',
    invocationCount: 32,
    workgroupSize: 4,
    control,
    bindings: [
      {name: 'values', view: values, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'f32', access: 'read_write'}
    ],
    declarations: 'var<workgroup> partials: array<f32, 4>;',
    body: `let lane = localInvocationIndex;
partials[lane] = values[valuesOffset + index];
workgroupBarrier();
for (var stride = 2u; stride > 0u; stride /= 2u) {
 if (lane < stride) { partials[lane] += partials[lane + stride]; }
 workgroupBarrier();
}
if (lane == 0u) { result[resultOffset + workgroupIndex] = partials[0] + ${control ? 'f32(getStatisticsControl(0u))' : '0.0'}; }`
  });
}

export const FALLBACK_POSITIONS = [0, 0, 2, 0, -2, 0, 0, 3, 2, 0, 10, 10, NaN, 0, 0, 0];
export const FALLBACK_MASK = [1, 0, 1, 1, 0, 1, 0, 1];
export const FALLBACK_ALL_NEAREST = [7, 4, 0, 0, 1, 3, 0xffffffff, 0];
export const FALLBACK_SELECTED_NEAREST = [7, 0xffffffff, 0, 0, 0xffffffff, 3, 0xffffffff, 0];
export const PAIR_HIGH = [2, 1, 2, 1, 0xffffffff, 1, 0, 0xffffffff];
export const PAIR_LOW = [1, 9, 1, 2, 0xffffffff, 2, 7, 0xffffffff];
export const PAIR_ORDER = [6, 3, 5, 1, 0, 2, 4, 7];
export const PAIR_FLAGS = [1, 1, 0, 1, 1, 0, 1, 1];
export const PAIR_GROUP_INDEX = [1, 2, 2, 3, 4, 4, 5, 6];
export const PAIR_BOUNDARIES = [0, 1, 3, 4, 6, 7, 8];
export const SLOT_KEYS = [3, 0, 5, 3, 1, 5, 0, 3];
export const SLOT_ORDER = [1, 6, 4, 0, 3, 7, 2, 5];
export const SORTED_SLOT_KEYS = [0, 0, 1, 3, 3, 3, 5, 5];
export const COLUMN_OFFSETS = [0, 2, 3, 3, 6, 6];
export const STAGE_LEFT = {
  offsets: [0, 2, 3, 3, 6, 7, 7, 9, 10],
  neighbors: [1, 3, 0, 1, 2, 7, 5, 0, 7, 6],
  weights: [11, 13, 20, 41, 42, 47, 0, 70, 77, 86],
  distances: [1, 3, 0, 1, 2, 7, 5, 0, 7, 6]
};
export const ROW_LOCAL_RULE_AUDIT_CAPACITY = 8;
export const STAGE_RIGHT = {
  offsets: [0, 2, 3, 4, 6, 6, 7, 8, 9],
  neighbors: [2, 3, 2, 0, 2, 4, 4, 0, 6],
  weights: [102, 5, 112, 120, 132, 134, 154, 160, 176],
  distances: [12, 13, 12, 10, 12, 14, 14, 10, 16]
};
export const SET_EXPECTED = {
  union: {
    offsets: [0, 3, 5, 6, 10, 11, 12, 14, 15],
    neighbors: [1, 2, 3, 0, 2, 0, 1, 2, 4, 7, 5, 4, 0, 7, 6]
  },
  intersection: {offsets: [0, 1, 1, 1, 2, 2, 2, 3, 4], neighbors: [3, 2, 0, 6]},
  difference: {offsets: [0, 1, 2, 2, 4, 5, 5, 6, 6], neighbors: [1, 0, 1, 7, 5, 7]},
  'symmetric-difference': {
    offsets: [0, 2, 4, 5, 8, 9, 10, 11, 11],
    neighbors: [1, 2, 0, 2, 0, 1, 4, 7, 5, 4, 7]
  }
};
export type RowLocalAuditOperation = SpatialWeightsMergeOperation | 'selfWeight' | 'subgraph';
export const SELF_WEIGHTS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4];
export const SUBGRAPH_MASK = [1, 1, 0, 1, 1, 0, 1, 1];
export const SPACE_TIME_INPUT = {
  offsets: [0, 2, 3, 3, 5],
  neighbors: [0, 1, 0, 1, 3],
  weights: [5, 2, 3, 4, 6]
};
export const SPACE_TIME_LAGS = [
  {lag: -1, weight: 2, includeSelf: true},
  {lag: 0, weight: 0.5},
  {lag: 1, weight: 3, includeSelf: true}
];
export const SPACE_TIME_EXPECTED = {
  'period-major': {
    offsets: [0, 4, 7, 8, 12, 18, 23, 25, 31, 35, 38, 39, 43],
    neighbors: [
      0, 1, 4, 5, 0, 4, 5, 6, 1, 3, 5, 7, 0, 1, 4, 5, 8, 9, 0, 1, 4, 8, 9, 2, 10, 1, 3, 5, 7, 9, 11,
      4, 5, 8, 9, 4, 5, 8, 6, 5, 7, 9, 11
    ],
    weights: [
      2.5, 1, 2, 4, 1.5, 6, 2, 2, 2, 3, 8, 2, 3, 6, 2.5, 1, 2, 4, 9, 3, 1.5, 6, 2, 3, 2, 12, 3, 2,
      3, 8, 2, 3, 6, 2.5, 1, 9, 3, 1.5, 3, 12, 3, 2, 3
    ]
  },
  'cell-major': {
    offsets: [0, 4, 10, 14, 17, 22, 25, 26, 28, 29, 33, 39, 43],
    neighbors: [
      0, 1, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 4, 5, 0, 1, 4, 0, 1, 2, 3, 5, 1, 2, 4, 7, 6, 8, 7, 3, 4,
      9, 10, 3, 4, 5, 9, 10, 11, 4, 5, 10, 11
    ],
    weights: [
      2.5, 2, 1, 4, 3, 2.5, 2, 6, 1, 4, 3, 2.5, 6, 1, 1.5, 6, 2, 9, 1.5, 6, 3, 2, 9, 1.5, 3, 2, 3,
      2, 3, 2, 8, 3, 2, 12, 2, 8, 3, 3, 2, 12, 2, 3, 3
    ]
  }
};

export function createSpaceTimeAuditNodes(
  graph: GPUCommandGraph,
  weights: GPUSpatialWeights,
  output: GPUSpatialWeightsOutput,
  layout: GPUSpaceTimeWeightsLayout,
  scale: number
) {
  return getSpaceTimeStages(
    graph,
    {
      id: 'space-time-audit',
      operation: 'audit',
      rows: 12,
      targetOffsets: output.offsets,
      capacity: output.neighbors.length,
      overflow: output.status.overflow,
      totalNeighbors: output.status.requiredCount,
      initializesOverflow: true
    },
    {
      weights,
      output,
      cellCount: 4,
      periodCount: 3,
      layout,
      lags: SPACE_TIME_LAGS.map(lag => ({...lag, weight: lag.weight * scale}))
    }
  );
}
export const SELF_INPUT = {
  ...STAGE_LEFT,
  neighbors: [1, 3, 1, 1, 2, 7, 5, 0, 7, 6],
  distances: [1, 3, 99, 1, 2, 7, 5, 0, 7, 6]
};
const UNION_DISTANCES = [1, 12, 3, 0, 12, 10, 1, 2, 14, 7, 5, 14, 0, 7, 6];
const UNION_RULE_WEIGHTS = {
  left: [11, 102, 13, 20, 112, 120, 41, 42, 134, 47, 0, 154, 70, 77, 86],
  right: [11, 102, 5, 20, 112, 120, 41, 132, 134, 47, 0, 154, 160, 77, 176],
  sum: [11, 102, 18, 20, 112, 120, 41, 174, 134, 47, 0, 154, 230, 77, 262],
  min: [11, 102, 5, 20, 112, 120, 41, 42, 134, 47, 0, 154, 70, 77, 86],
  max: [11, 102, 13, 20, 112, 120, 41, 132, 134, 47, 0, 154, 160, 77, 176],
  product: [11, 102, 65, 20, 112, 120, 41, 5544, 134, 47, 0, 154, 11200, 77, 15136],
  binary: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
};
const SET_UNION_POSITIONS = {
  union: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  intersection: [2, 7, 12, 14],
  difference: [0, 3, 6, 9, 10, 13],
  'symmetric-difference': [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13]
};

/** Literal independent oracle tables, with only their known homogeneity applied. */
export function getRowLocalAuditExpected(
  operation: RowLocalAuditOperation,
  rule: GPUSpatialWeightsCombineRule = 'sum',
  scale = 1,
  perRowSelf = true
) {
  if (operation === 'selfWeight') {
    const selfSlots = [0, 3, 4, 7, 9, 11, 13, 16];
    return {
      offsets: [0, 3, 4, 5, 9, 11, 12, 15, 17],
      neighbors: [0, 1, 3, 1, 2, 1, 2, 3, 7, 4, 5, 5, 0, 6, 7, 6, 7],
      weights: [0.5, 11, 13, 1, 1.5, 41, 42, 2, 47, 2.5, 0, 3, 70, 3.5, 77, 86, 4].map(
        (value, index) => (!perRowSelf && selfSlots.includes(index) ? 0.5 : value) * scale
      ),
      distances: [0, 1, 3, 0, 0, 1, 2, 0, 7, 0, 5, 0, 0, 0, 7, 6, 0]
    };
  }
  if (operation === 'subgraph')
    return {
      offsets: [0, 2, 3, 3, 5, 5, 5, 7, 8],
      neighbors: [1, 3, 0, 1, 7, 0, 7, 6],
      weights: [11, 13, 20, 41, 47, 70, 77, 86].map(value => value * scale),
      distances: [1, 3, 0, 1, 7, 0, 7, 6]
    };
  const positions = SET_UNION_POSITIONS[operation];
  return {
    ...SET_EXPECTED[operation],
    weights: positions.map(
      index =>
        UNION_RULE_WEIGHTS[rule][index] *
        (rule === 'binary'
          ? 1
          : rule === 'product' && [2, 7, 12, 14].includes(index)
            ? scale * scale
            : scale)
    ),
    distances: positions.map(index => UNION_DISTANCES[index])
  };
}

export function createRowLocalAuditNodes(
  graph: GPUCommandGraph,
  left: GPUSpatialWeights,
  right: GPUSpatialWeights,
  mask: GraphDataView<'uint32'>,
  selfWeight: number | GraphDataView<'float32'>,
  output: GPUSpatialWeightsOutput,
  operation: RowLocalAuditOperation,
  rule: GPUSpatialWeightsCombineRule,
  chunk?: {rowStart: number; totalRows: number}
) {
  const base = {
    id: 'row-local-audit',
    operation: 'audit',
    rows: output.offsets.length - 1,
    targetOffsets: output.offsets,
    capacity: output.neighbors.length,
    overflow: output.status.overflow,
    totalNeighbors: output.status.requiredCount,
    initializesOverflow: true,
    chunk
  };
  if (operation === 'selfWeight')
    return getSelfWeightStages(graph, base, {weights: left, output, selfWeight});
  if (operation === 'subgraph')
    return getSubgraphStages(graph, base, {weights: left, output, mask});
  return getBinaryMergeStages(graph, base, {
    operation,
    left,
    right,
    output,
    isBinaryRule: rule === 'binary',
    combineExpression: COMBINE_RULES[rule]
  });
}

/** Directly exercise count/scan/clamp/emit stages and the generated merge visitor. */
export function createCsrStageAuditNodes(
  graph: GPUCommandGraph,
  left: GPUSpatialWeights,
  right: GPUSpatialWeights,
  output: GPUSpatialWeightsOutput,
  operation?: SpatialWeightsMergeOperation,
  initializesOverflow = true
) {
  return createCSRStages(graph, {
    id: 'csr-audit',
    operation: 'audit',
    rows: 8,
    capacity: output.neighbors.length,
    targetOffsets: output.offsets,
    overflow: output.status.overflow,
    totalNeighbors: output.status.requiredCount,
    initializesOverflow,
    bindings: [
      {name: 'leftOffsets', view: left.offsets, type: 'u32', access: 'read'},
      {name: 'leftNeighbors', view: left.neighbors, type: 'u32', access: 'read'},
      ...(operation
        ? [
            {
              name: 'rightOffsets',
              view: right.offsets,
              type: 'u32' as const,
              access: 'read' as const
            },
            {
              name: 'rightNeighbors',
              view: right.neighbors,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ],
    visit: onItem =>
      operation
        ? getMergeVisitWGSL(operation, onItem)
        : `
for (var slot = leftOffsets[leftOffsetsOffset + index]; slot < leftOffsets[leftOffsetsOffset + index + 1u]; slot++) {
 let id = leftNeighbors[leftNeighborsOffset + slot]; ${onItem}
}`,
    attributes: [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {name: 'weights', output: output.weights, type: 'f32', value: 'f32((index + 1u) * 10u + id)'}
    ]
  });
}

export function createContainsAuditNode(
  graph: GPUCommandGraph,
  keys: GraphDataView<'uint32'>,
  queries: GraphDataView<'uint32'>,
  result: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'contains-audit',
    operation: 'audit',
    invocationCount: 8,
    bindings: [
      {name: 'keys', view: keys, type: 'u32', access: 'read'},
      {name: 'queries', view: queries, type: 'u32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    declarations: getContainsFunctionWGSL('containsKey', 'keys'),
    body: 'result[resultOffset + index] = u32(containsKey(0u, 8u, queries[queriesOffset + index]));'
  });
}

export function createSortRowsAuditNodes(graph: GPUCommandGraph, output: GPUSpatialWeightsOutput) {
  return createCSRStages(graph, {
    id: 'sorted-csr-audit',
    operation: 'audit',
    rows: 8,
    capacity: 24,
    targetOffsets: output.offsets,
    overflow: output.status.overflow,
    totalNeighbors: output.status.requiredCount,
    initializesOverflow: true,
    bindings: [],
    sortRows: true,
    visit: onItem => `for (var value = 3u; value > 0u; value--) { let id = value - 1u; ${onItem} }`,
    attributes: [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {name: 'weights', output: output.weights, type: 'f32', value: '1.0'}
    ]
  });
}
export const ROW_NEIGHBORS = [11, 13, 17, 19, 23, 29, 31, 37];
export const ROW_RESULT = [99, 99, 99, 99, 319, 431, 545, 657, 771, 887, 999, 1115, 99, 99, 99];

export function createRowAuditNode(
  graph: GPUCommandGraph,
  neighbors: GraphDataView<'uint32'>,
  result: GraphDataView<'uint32'>
) {
  return createSpatialStatisticsRowNode(graph, {
    id: 'row-audit',
    operation: 'audit',
    rowBase: 3,
    invocationCount: 8,
    bindings: [
      {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    body: `let localRow = index - WEIGHTS_ROW_START;
result[resultOffset + index] = 100u * index + 10u * localRow + CAPACITY + neighbors[neighborsOffset + localRow];`
  });
}

export function createStatisticsGuardAuditNode(
  graph: GPUCommandGraph,
  result: GraphDataView<'uint32'>,
  invocationCount: number
) {
  return createStatisticsKernelNode(graph, {
    id: 'guard-audit',
    operation: 'audit',
    invocationCount,
    workgroupSize: 4,
    guardIndex: false,
    bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
    body: 'workgroupBarrier(); if (localInvocationIndex == 0u) { result[resultOffset + workgroupIndex] = workgroupIndex + 1u; }'
  });
}
export const EXACT_COUNTS = [
  16777217, 30000001, 0xffffefff, 3, 0xffffffff, 16777217, 0xffffffff, 30000001
];
export const COUNT_VALUES = [16777216, 1, 1234567, 8, -0.125, 2 ** -120, 2 ** -149, 2 ** 100];
export const ORDINARY_COUNTS = [1, 2, 3, 4, 8, 16, 256, 16777216];
export const ORDINARY_COUNT_VALUES = [1, 1, 8, -0.125, 2 ** -120, 2 ** 100, 16777216, 1];
export const ORDINARY_COUNT_BITS = [
  0x3f800000, 0x3f800000, 0, 0x3f800000, 0x3f000000, 0x40000000, 0x3f000000, 0x40000000, 0x402aaaab,
  0x41c00000, 0x3f2aaaab, 0x3ec00000, 0xbd000000, 0xbf000000, 0x3f400000, 0xc2000000, 0x02000000,
  0x05000000, 0x3f600000, 0x7d000000, 0x6f800000, 0x73800000, 0x3f700000, 0x0f800000, 0x47800000,
  0x4f800000, 0x3f7f0000, 0x37800000, 0x33800000, 0x4b800000, 0x3f7fffff, 0x4b800000
];
// Exact rational nearest/even binary32 oracle, not a double-precision proxy.
export const COUNT_RESULT_BITS = [
  0x3f7fffff, 0x57800000, 0x3f7fffff, 0x3f800000, 0x330f2a63, 0x4be4e1c0, 0x3f7fffff, 0x4be4e1c0,
  0x3996b441, 0x5996b42f, 0x3f800000, 0x45596ec4, 0x402aaaab, 0x41c00000, 0x3f2aaaab, 0x3ec00000,
  0xae000000, 0xce000000, 0x3f800000, 0xd1000000, 0x20, 0x0f800000, 0x3f7fffff, 0x7f800000, 0,
  0x05000000, 0x3f800000, 0x7f800000, 0x650f2a63, 0x7de4e1c0, 0x3f7fffff, 0x19e4e1c0
];

export function createCountArithmeticAuditNode(
  graph: GPUCommandGraph,
  counts: GraphDataView<'uint32'>,
  values: GraphDataView<'float32'>,
  result: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'count-arithmetic-audit',
    operation: 'audit',
    invocationCount: 8,
    bindings: [
      {name: 'counts', view: counts, type: 'u32', access: 'read'},
      {name: 'values', view: values, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    declarations: SPATIAL_STATISTICS_COUNTS_WGSL,
    body: `let count = counts[countsOffset + index]; let value = values[valuesOffset + index];
result[resultOffset + 4u * index] = bitcast<u32>(divideBySpatialCount(value, count));
result[resultOffset + 4u * index + 1u] = bitcast<u32>(multiplyBySpatialCount(value, count));
result[resultOffset + 4u * index + 2u] = bitcast<u32>(divideSpatialCounts(count - 1u, count));
result[resultOffset + 4u * index + 3u] = bitcast<u32>(divideSpatialCountByValue(count, value));`
  });
}

/** Exercise the exported comparator itself, separately from the cooperative fallback. */
export function createNearestComparisonAuditNode(
  graph: GPUCommandGraph,
  positions: GraphDataView<'float32x2'>,
  result: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'nearest-comparison-audit',
    operation: 'audit',
    invocationCount: 8,
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    declarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
    body: `let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
var nearest = NO_ATTACHMENT;
var best = vec2<f32>(0.0);
if (isFiniteBits(position.x) && isFiniteBits(position.y)) {
 for (var other = 0u; other < 8u; other++) {
  let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
  if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
   let difference = position * 0.5 - candidate * 0.5;
   if (isCloserDifference(difference, best, nearest == NO_ATTACHMENT)) { nearest = other; best = difference; }
  }
 }
}
result[resultOffset + index] = nearest;`
  });
}
