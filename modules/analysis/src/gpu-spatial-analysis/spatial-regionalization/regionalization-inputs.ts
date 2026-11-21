// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPairwiseSumWGSL} from '../spatial-regression/pairwise-sum-wgsl';
import {REGIONAL_MOMENT_WGSL} from './regional-moment-wgsl';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateRequirement,
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialWeightsTranspose} from '../spatial-weights/gpu-spatial-weights-transpose';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';

/** Lanes of the per-column standardization workgroup. */
const STANDARDIZE_LANES = 256;
/** Published transpose sorts use local bitonic through 256 slots, then 16 radix buckets. */
const ADJACENCY_SORT_WORKGROUP_SIZE = 256;
const ADJACENCY_SORT_BUCKET_COUNT = 16;

/** Preflight the union and nested transpose, without creating their buffers. @internal */
export function validateSymmetricAdjacencyScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GPUSpatialWeights,
  rowCount: number
): void {
  if (weights.information?.symmetric === true) return;
  const capacity = weights.neighbors.length;
  for (const [name, byteLength] of [
    ['adjacency offsets', (rowCount + 1) * 4],
    ['adjacency neighbors and weights', Math.max(1, 2 * capacity) * 4],
    [
      'transpose sort histogram',
      capacity > ADJACENCY_SORT_WORKGROUP_SIZE
        ? Math.ceil(capacity / ADJACENCY_SORT_WORKGROUP_SIZE) * ADJACENCY_SORT_BUCKET_COUNT * 4
        : 0
    ],
    ['transpose parameters', 4]
  ] as const)
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce rows or weights slot capacity'
    );
}

/**
 * Number of `uint32` words of the seed-only per-frame parameters shared by `GPUAZPRegions`,
 * `GPUWardRegions` and `GPUMaxPRegions`: `[seed]`.
 *
 * @internal
 */
export const REGIONALIZATION_SEED_PARAMETER_LENGTH = 1;

/**
 * Packs the `[seed]` parameter word of a seeded regionalization contributor.
 *
 * @param operation Contributor class name, for the message.
 * @param seed Tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0.
 * @internal
 */
export function getRegionalizationSeedParameterValues(
  operation: string,
  seed: number = 0
): Uint32Array {
  validateCapacity(operation, 'seed', seed, 0);
  return Uint32Array.of(seed);
}

/**
 * Largest `weights.neighbors` slot count of the AZP and Ward regionalizations on `device`: the
 * undirected adjacency they build holds up to two uint32 neighbors (8 bytes) per input slot in one
 * storage binding, so 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionalizationMaximumSlotCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Validates that a regionalization contributor's weights fit `maximum` slots, the value of
 * {@link getGPURegionalizationMaximumSlotCount} unless the contributor needs fewer.
 *
 * @internal
 */
export function validateRegionalizationSlotCount(
  id: string,
  weights: GPUSpatialWeights,
  maximum: number
): void {
  validateMaximumCount(id, 'weights.neighbors rows', weights.neighbors.length, maximum);
}

/**
 * Validates the per-frame `uint32` parameter view of a regionalization contributor.
 *
 * @internal
 */
export function validateRegionalizationParameters(
  id: string,
  parameters: GraphDataView,
  length: number
): void {
  validateSinglePackedView(parameters, ['uint32'], `${id} parameters`);
  if (parameters.length < length) {
    validateRequirement(
      id,
      false,
      `parameters must hold ${length} words; received parameters.length=${parameters.length}`
    );
  }
}

/**
 * Standardizes every column of a row-major attribute table to mean 0 and population standard
 * deviation 1 (divisor `n`, the `sklearn.preprocessing.StandardScaler` convention, not the sample
 * divisor `n - 1`). Statistics use the finite entries of the column only, so one missing value
 * does not turn the whole column into NaN; non-finite entries are copied to the output as they
 * are. A constant column becomes exactly zero: the column is first taken about its first finite
 * entry, so the mean of a constant column is exactly 0 and not a rounded value one last place
 * away from every entry. This is the one standardizer of AZP, Ward, max-p and the minimum
 * spanning tree.
 *
 * One workgroup per column: each lane accumulates a split sum over its strided share and a fixed
 * binary tree merges the split sums (deterministic order). Subtraction, summation and mean-division
 * residuals remain separate through variance and final scores; integer rounding-error recovery
 * and runtime bit barriers prevent fast-math regrouping.
 * Columns
 * large enough to overflow, or small enough to underflow, the float32 variance are divided by
 * a power of two near their largest finite magnitude before centering. This preserves the input
 * significands and leaves their population z-scores unchanged.
 *
 * @internal
 */
export function createStandardizeColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'>;
    scaled: GraphDataView<'float32'>;
    rowCount: number;
    columnCount: number;
  }
): GPUCommandNode<Parameters> {
  const {rowCount, columnCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'standardize',
    bindings: [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'scaled', view: props.scaled, type: 'atomic<u32>', access: 'read_write'}
    ],
    workgroupSize: STANDARDIZE_LANES,
    invocationCount: columnCount * STANDARDIZE_LANES,
    guardIndex: false,
    declarations: `const ROWS: u32 = ${rowCount}u;
const COLUMNS: u32 = ${columnCount}u;
const LANES: u32 = ${STANDARDIZE_LANES}u;
${IS_FINITE_FLOAT_WGSL}
${REGIONAL_MOMENT_WGSL}
var<workgroup> partials: array<f32, ${STANDARDIZE_LANES}>;
var<workgroup> momentPartials: array<vec2<f32>, ${STANDARDIZE_LANES}>;

fn reduceMomentLanes(lane: u32, value: vec2<f32>, zero: u32) -> vec2<f32> {
  momentPartials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      momentPartials[lane] = addRegionalMoment(momentPartials[lane], momentPartials[lane + stride], zero);
    }
    workgroupBarrier();
  }
  let total = momentPartials[0];
  workgroupBarrier();
  return total;
}

fn reduceMaximumLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  let maximum = partials[0];
  workgroupBarrier();
  return maximum;
}

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
    }
    workgroupBarrier();
  }
  let total = partials[0];
  workgroupBarrier();
  return total;
}`,
    body: `let column = index / LANES;
  let lane = localInvocationIndex;
  // Temporarily use this column's first output word as a storage-atomic runtime zero.
  // All lanes capture it before any final output is written. Atomic loads keep this opaque
  // to fast-math optimization; the rounded-difference barriers must not fold to plain algebra.
  if (lane == 0u) { atomicStore(&scaled[scaledOffset + column], 0u); }
  storageBarrier();
  let runtimeZero = atomicLoad(&scaled[scaledOffset + column]);
  storageBarrier();
  // Center before reducing: a rounded mean of a constant column can otherwise
  // create a nonzero deviation and normalize every row to +1 or -1. The origin is the first
  // finite entry; statistics skip non-finite entries, which stay as they are.
  var origin = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    let candidate = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(candidate)) { origin = candidate; break; }
  }
  var laneMaximum = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) { laneMaximum = max(laneMaximum, abs(value)); }
  }
  let maximum = reduceMaximumLanes(lane, laneMaximum);
  // Each centered deviation is at most twice the column range (4 * maximum). Reserve that
  // factor before summing squares. Division by 1 preserves the existing safe-range arithmetic.
  // A one-ULP difference among otherwise equal rows must also have a normal squared deviation
  // after division by ROWS. Use the conservative relative spacing 2^-24 for that lower bound.
  let safeMagnitude = sqrt(3.402823466e+38 / (16.0 * f32(ROWS)));
  let minimumMagnitude = sqrt(1.175494351e-38 * f32(ROWS)) * 16777216.0;
  let needsScaling = maximum > safeMagnitude || (maximum > 0.0 && maximum < minimumMagnitude);
  // Clamp the exponent to normal, finite powers of two. Divide values directly: the reciprocal
  // can be a flushed subnormal. Power-of-two division introduces no new significand rounding.
  let magnitudeExponent = clamp((bitcast<u32>(maximum) >> 23u) & 255u, 1u, 253u);
  let normalizer = select(1.0, bitcast<f32>(magnitudeExponent << 23u), needsScaling);
  origin = origin / normalizer;
  var count = 0.0;
  let laneTermCount = (ROWS + LANES - 1u - lane) / LANES;
  var sum = vec2<f32>(0.0);
  for (var item = 0u; item < laneTermCount; item++) {
    let row = lane + item * LANES;
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) {
      count += 1.0;
      sum = addRegionalMoment(sum, subtractRegionalValue(value / normalizer, origin, runtimeZero), runtimeZero);
    }
  }
  let finiteCount = max(reduceLanes(lane, count), 1.0);
  // Carry the subtraction error as well as the sum error. Correcting a mean using already
  // rounded residuals can move it past a genuine small positive residual.
  let mean = divideRegionalMoment(reduceMomentLanes(lane, sum, runtimeZero), finiteCount, runtimeZero);
  ${getPairwiseSumWGSL({
    name: 'squares',
    count: 'laneTermCount',
    setup: `let row = lane + squaresItem * LANES;
    let value = values[valuesOffset + row * COLUMNS + column];
    let centered = subtractRegionalValue(value / normalizer, origin, runtimeZero);
    let residual = addRegionalMoment(centered, -mean, runtimeZero);
    let delta = residual.x + residual.y;`,
    term: 'select(0.0, delta * delta, isFiniteFloat(value))'
  })}
  let deviation = sqrt(reduceLanes(lane, squares) / finiteCount);
  let inverse = select(0.0, 1.0 / deviation, deviation > 1e-20);
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    let centered = subtractRegionalValue(value / normalizer, origin, runtimeZero);
    let residual = addRegionalMoment(centered, -mean, runtimeZero);
    let corrected = residual.x + residual.y;
    atomicStore(&scaled[scaledOffset + row * COLUMNS + column],
      bitcast<u32>(select(value, corrected * inverse, isFiniteFloat(value))));
  }`
  });
}

/**
 * Builds the undirected union of `W` and `W'` as a CSR (each row lists the `W` neighbors followed
 * by the transposed neighbors; a link present in both directions appears twice, which every
 * consumer tolerates because they only ask whether some neighbor satisfies a condition). Neighbor
 * IDs outside `[0, rowCount)` are replaced by the row itself (a self link is never a candidate),
 * and the caller reports them separately as invalid input from the original weights.
 *
 * When `weights.information?.symmetric === true` the input is returned unchanged and no node is added.
 *
 * @internal
 */
export function createSymmetricAdjacency<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; weights: GPUSpatialWeights; rowCount: number}
): {weights: GPUSpatialWeights; nodes: GPUCommandNode<Parameters>[]} {
  const {id, operation, weights, rowCount} = props;
  if (weights.information?.symmetric === true) {
    return {weights, nodes: []};
  }
  const capacity = weights.neighbors.length;
  const transposeOffsets = createTransientView(
    graph,
    `${id}-adjacency-t-offsets`,
    'uint32',
    rowCount + 1
  );
  const transposeNeighbors = createTransientView(
    graph,
    `${id}-adjacency-t-neighbors`,
    'uint32',
    capacity
  );
  const transposeWeights = createTransientView(
    graph,
    `${id}-adjacency-t-weights`,
    'float32',
    capacity
  );
  const adjacencyOffsets = createTransientView(
    graph,
    `${id}-adjacency-offsets`,
    'uint32',
    rowCount + 1
  );
  const adjacencyNeighbors = createTransientView(
    graph,
    `${id}-adjacency-neighbors`,
    'uint32',
    2 * capacity
  );
  const adjacencyWeights = createTransientView(
    graph,
    `${id}-adjacency-weights`,
    'float32',
    2 * capacity
  );
  const declarations = `const ROWS: u32 = ${rowCount}u;
const CAPACITY: u32 = ${2 * capacity}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...new GPUSpatialWeightsTranspose({
      id: `${id}-adjacency-transpose`,
      // Only the pattern matters: drop information, status and distances.
      weights: {offsets: weights.offsets, neighbors: weights.neighbors, weights: weights.weights},
      output: {
        offsets: transposeOffsets,
        neighbors: transposeNeighbors,
        weights: transposeWeights
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-adjacency-offsets`,
      operation,
      variant: 'adjacency-offsets',
      bindings: [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'transposeOffsets', view: transposeOffsets, type: 'u32', access: 'read'},
        {name: 'adjacencyOffsets', view: adjacencyOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount + 1,
      body: 'adjacencyOffsets[adjacencyOffsetsOffset + index] = offsets[offsetsOffset + index] + transposeOffsets[transposeOffsetsOffset + index];'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-adjacency-neighbors`,
      operation,
      variant: 'adjacency-neighbors',
      bindings: [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'transposeOffsets', view: transposeOffsets, type: 'u32', access: 'read'},
        {name: 'transposeNeighbors', view: transposeNeighbors, type: 'u32', access: 'read'},
        {name: 'adjacencyOffsets', view: adjacencyOffsets, type: 'u32', access: 'read'},
        {name: 'adjacencyNeighbors', view: adjacencyNeighbors, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 2 * capacity,
      declarations: `${declarations}
// Row that owns adjacency slot \`slot\`: the first row whose range ends after it.
fn findAdjacencyRow(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (adjacencyOffsets[adjacencyOffsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}`,
      body: `var neighbor = 0u;
  if (index < adjacencyOffsets[adjacencyOffsetsOffset + ROWS]) {
    let row = findAdjacencyRow(index);
    let local = index - adjacencyOffsets[adjacencyOffsetsOffset + row];
    let ownCount = offsets[offsetsOffset + row + 1u] - offsets[offsetsOffset + row];
    if (local < ownCount) {
      neighbor = neighbors[neighborsOffset + offsets[offsetsOffset + row] + local];
      if (neighbor >= ROWS) { neighbor = row; }
    } else {
      neighbor = transposeNeighbors[transposeNeighborsOffset + transposeOffsets[transposeOffsetsOffset + row] + local - ownCount];
    }
  }
  adjacencyNeighbors[adjacencyNeighborsOffset + index] = neighbor;`
    })
  ];
  return {
    weights: describeGPUSpatialWeights(
      {offsets: adjacencyOffsets, neighbors: adjacencyNeighbors, weights: adjacencyWeights},
      {
        information: {
          rowCount,
          transform: 'unknown',
          symmetric: true
        }
      }
    ),
    nodes
  };
}
