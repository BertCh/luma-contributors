// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getGPUSpatialJoinCapacityPlan} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {ResidentJoinColumnFormat} from './resident-join-plan';
import type {ResidentSpatialJoinGpuTechnique} from './resident-spatial-join-plan';

import {clamp} from '@math.gl/core';

const WORKGROUP_SIZE = 64;

/**
 * Documented memory reservation for the join working set of one chunk: bounding-box candidates,
 * robust point/polygon rows, the polygon BVH, per-row probe state and the capacity-bounded pairs.
 * It is a reservation, not a row cap: chunks shrink to fit and every row is still processed.
 */
export const RESIDENT_SPATIAL_JOIN_WORKING_BYTES = 256 * 1024 * 1024;
/** Largest number of left points joined by one graph encoding. */
export const RESIDENT_SPATIAL_JOIN_CHUNK_ROWS = 1_048_576;
/** Candidate and pair capacity headroom applied when a run overflows and capacities grow. */
const GROWTH_FACTOR = 1.25;
const GROWTH_SLACK = 16;

export type ResidentSpatialJoinCapacities = {
  /** Left points joined per encoding; the last chunk repeats a real point as padding. */
  chunkRowCount: number;
  /** Bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Matched pairs per encoding. */
  pairCapacity: number;
};
export type ResidentSpatialJoinObservation = {
  /** Real (non-padding) rows of the chunk. */
  rowCount: number;
  /** Unclamped candidate count of the chunk. */
  candidateCount: number;
  /** Unclamped match count; a lower bound while candidates overflowed. */
  requiredCount: number;
  candidateOverflow: boolean;
};

/** First guess: about two candidates and one match per point, corrected by observation. */
export function getInitialResidentSpatialJoinCapacities(
  rowCount: number,
  rightCount: number
): ResidentSpatialJoinCapacities {
  const chunkRowCount = clamp(RESIDENT_SPATIAL_JOIN_CHUNK_ROWS, 1, rowCount);
  return {
    chunkRowCount,
    candidateCapacity: chunkRowCount * Math.min(rightCount, 2) + GROWTH_SLACK,
    pairCapacity: chunkRowCount * Math.min(rightCount, 1) + GROWTH_SLACK
  };
}

/**
 * Grows capacities from an overflowed chunk using its observed per-point densities. The result is
 * always strictly larger than the overflowed stage so retries terminate.
 */
export function growResidentSpatialJoinCapacities(
  previous: ResidentSpatialJoinCapacities,
  observation: ResidentSpatialJoinObservation
): ResidentSpatialJoinCapacities {
  const rows = Math.max(1, observation.rowCount);
  const candidateDensity = observation.candidateCount / rows;
  // Matches never exceed candidates, so candidate density bounds them while candidates overflowed.
  const matchDensity = observation.candidateOverflow
    ? candidateDensity
    : observation.requiredCount / rows;
  const grow = (density: number) =>
    Math.ceil(previous.chunkRowCount * density * GROWTH_FACTOR) + GROWTH_SLACK;
  return {
    chunkRowCount: previous.chunkRowCount,
    candidateCapacity: Math.max(previous.candidateCapacity, grow(candidateDensity)),
    pairCapacity: Math.max(previous.pairCapacity, grow(matchDensity))
  };
}

/** Estimated join working bytes for these capacities (graph transients plus pair columns). */
export function getResidentSpatialJoinWorkingBytes(
  capacities: ResidentSpatialJoinCapacities,
  rightCount: number
): number {
  const plan = getGPUSpatialJoinCapacityPlan({
    leftCount: capacities.chunkRowCount,
    rightCount,
    leftKind: 'points',
    rightKind: 'polygons',
    candidateCapacity: capacities.candidateCapacity,
    pairCapacity: capacities.pairCapacity
  });
  return plan.estimatedPeakTransientBytes + plan.pairs.estimatedOutputBytes;
}

/**
 * Halves the chunk (and capacities, which scale with it) until the estimated working set fits the
 * reservation. A single point whose candidates alone exceed the reservation cannot be joined.
 */
export function fitResidentSpatialJoinCapacities(
  capacities: ResidentSpatialJoinCapacities,
  rightCount: number,
  budgetBytes: number = RESIDENT_SPATIAL_JOIN_WORKING_BYTES
): ResidentSpatialJoinCapacities {
  let fitted = capacities;
  while (getResidentSpatialJoinWorkingBytes(fitted, rightCount) > budgetBytes) {
    if (fitted.chunkRowCount === 1)
      throw new Error(
        'Spatial Join working set exceeds its memory reservation even for one point. Use SQL for this polygon set.'
      );
    fitted = {
      chunkRowCount: Math.ceil(fitted.chunkRowCount / 2),
      candidateCapacity: Math.ceil(fitted.candidateCapacity / 2) + GROWTH_SLACK,
      pairCapacity: Math.ceil(fitted.pairCapacity / 2) + GROWTH_SLACK
    };
  }
  return fitted;
}

const TECHNIQUE_MODES: Readonly<Record<ResidentSpatialJoinGpuTechnique, number>> = {
  count: 0,
  sum: 1,
  average: 2,
  maximum: 3,
  minimum: 4
};
const SOURCE_FORMATS: Readonly<Record<ResidentJoinColumnFormat, number>> = {
  float32: 0,
  sint32: 1,
  uint32: 2
};

export type ResidentSpatialJoinAggregateProps = {
  id: string;
  technique: ResidentSpatialJoinGpuTechnique;
  /** Storage format of `sourceValues`; unused by `count`. */
  sourceFormat?: ResidentJoinColumnFormat;
  /** Matched pairs of one chunk, sorted by left row (the predicate join's `pairs`). */
  leftIds: GraphDataView<'uint32'>;
  rightIds: GraphDataView<'uint32'>;
  /** One-row total pair count word of the chunk (clamped to the pair capacity here). */
  pairCount: GraphDataView<'uint32'>;
  /** Packed right-row values and optional validity words; a one-word dummy for `count`. */
  sourceValues: GraphDataView<'uint32'>;
  sourceValidity?: GraphDataView<'uint32'>;
  /** Dummy bound when no validity exists. */
  dummy: GraphDataView<'uint32'>;
  /** One word per chunk row: u32 count, or Float32 bits of the aggregate. */
  outputValues: GraphDataView<'uint32'>;
  /** One word per chunk row: 1 when the result is defined. */
  outputValidity: GraphDataView<'uint32'>;
};

/**
 * Fused per-left-row aggregation over the join's sorted pairs: each invocation binary-searches its
 * run of pairs and reduces the matched right values in place, so no per-pair gather, null mask or
 * intermediate column is materialized. Native semantics (kepler.gl aggregateValues): count includes
 * null values; the others skip nulls and are undefined when no value is defined. Sums use
 * Neumaier-compensated Float32 accumulation and round to the Float32 output policy.
 */
export class ResidentSpatialJoinAggregate {
  readonly id: string;
  constructor(private readonly props: ResidentSpatialJoinAggregateProps) {
    this.id = props.id;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    const sourceValidity = props.sourceValidity ?? props.dummy;
    const layout = getBoundedDispatchLayout(
      'ResidentSpatialJoinAggregate',
      props.outputValues.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const source = getResidentSpatialJoinAggregateSource(props, layout);
    return [
      createGPUComputeCommandNode<Parameters>({
        id: this.id,
        workload: {
          operation: 'ResidentSpatialJoinAggregate',
          commandCount: 1,
          maximumWorkgroupCount: layout.x * layout.y * layout.z,
          maximumInvocationCount: layout.x * layout.y * layout.z * WORKGROUP_SIZE,
          readByteLength: props.leftIds.length * 8,
          writeByteLength: props.outputValues.length * 8
        },
        resources: [
          {buffer: props.leftIds, usage: 'storage-read'},
          {buffer: props.rightIds, usage: 'storage-read'},
          {buffer: props.pairCount, usage: 'storage-read'},
          {buffer: props.sourceValues, usage: 'storage-read'},
          {buffer: sourceValidity, usage: 'storage-read'},
          {buffer: props.outputValues, usage: 'storage-write'},
          {buffer: props.outputValidity, usage: 'storage-write'}
        ],
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id: this.id,
            source,
            shaderLayout: {
              bindings: [
                {name: 'outputValues', type: 'storage', group: 0, location: 0},
                {name: 'outputValidity', type: 'storage', group: 0, location: 1},
                {name: 'leftWords', type: 'read-only-storage', group: 0, location: 2},
                {name: 'rightWords', type: 'read-only-storage', group: 0, location: 3},
                {name: 'pairCountWords', type: 'read-only-storage', group: 0, location: 4},
                {name: 'sourceValueWords', type: 'read-only-storage', group: 0, location: 5},
                {name: 'sourceValidityWords', type: 'read-only-storage', group: 0, location: 6}
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) => {
              const bindings: Record<string, Binding> = {
                outputValues: getViewBinding(props.outputValues, getBuffer),
                outputValidity: getViewBinding(props.outputValidity, getBuffer),
                leftWords: getViewBinding(props.leftIds, getBuffer),
                rightWords: getViewBinding(props.rightIds, getBuffer),
                pairCountWords: getViewBinding(props.pairCount, getBuffer),
                sourceValueWords: getViewBinding(props.sourceValues, getBuffer),
                sourceValidityWords: getViewBinding(sourceValidity, getBuffer)
              };
              kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
            },
            destroy: () => kernel.destroy()
          };
        }
      })
    ];
  }
}

/** WGSL of {@link ResidentSpatialJoinAggregate}; exported for tests. */
export function getResidentSpatialJoinAggregateSource(
  props: ResidentSpatialJoinAggregateProps,
  layout: GPUBoundedDispatchLayout
): string {
  const mode = TECHNIQUE_MODES[props.technique];
  const format = SOURCE_FORMATS[props.sourceFormat ?? 'float32'];
  const hasValidity = Boolean(props.sourceValidity);
  return `const ROW_COUNT: u32 = ${props.outputValues.length}u;
const PAIR_CAPACITY: u32 = ${props.leftIds.length}u;
const MODE: u32 = ${mode}u;
const SOURCE_FORMAT: u32 = ${format}u;
const HAS_VALIDITY: bool = ${hasValidity};
const OUTPUT_BASE: u32 = ${getViewElementOffset(props.outputValues)}u;
const VALIDITY_OUTPUT_BASE: u32 = ${getViewElementOffset(props.outputValidity)}u;
const LEFT_BASE: u32 = ${getViewElementOffset(props.leftIds)}u;
const RIGHT_BASE: u32 = ${getViewElementOffset(props.rightIds)}u;
const COUNT_BASE: u32 = ${getViewElementOffset(props.pairCount)}u;
const SOURCE_BASE: u32 = ${getViewElementOffset(props.sourceValues)}u;
const SOURCE_VALIDITY_BASE: u32 = ${getViewElementOffset(props.sourceValidity ?? props.dummy)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValidity: array<u32>;
@group(0) @binding(2) var<storage, read> leftWords: array<u32>;
@group(0) @binding(3) var<storage, read> rightWords: array<u32>;
@group(0) @binding(4) var<storage, read> pairCountWords: array<u32>;
@group(0) @binding(5) var<storage, read> sourceValueWords: array<u32>;
@group(0) @binding(6) var<storage, read> sourceValidityWords: array<u32>;
fn readSource(row: u32) -> f32 {
  let word = sourceValueWords[SOURCE_BASE + row];
  if (SOURCE_FORMAT == 0u) { return bitcast<f32>(word); }
  if (SOURCE_FORMAT == 1u) { return f32(bitcast<i32>(word)); }
  return f32(word);
}
fn lowerBound(needle: u32, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (leftWords[LEFT_BASE + middle] < needle) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= ROW_COUNT) { return; }
  let pairCount = min(pairCountWords[COUNT_BASE], PAIR_CAPACITY);
  var matches = 0u;
  var defined = 0u;
  var sum = 0.0;
  var compensation = 0.0;
  var minimum = 0.0;
  var maximum = 0.0;
  var pair = lowerBound(index, pairCount);
  while (pair < pairCount && leftWords[LEFT_BASE + pair] == index) {
    matches++;
    if (MODE != 0u) {
      let row = rightWords[RIGHT_BASE + pair];
      if (!HAS_VALIDITY || sourceValidityWords[SOURCE_VALIDITY_BASE + row] != 0u) {
        let value = readSource(row);
        let total = sum + value;
        if (abs(sum) >= abs(value)) { compensation += (sum - total) + value; }
        else { compensation += (value - total) + sum; }
        sum = total;
        if (defined == 0u) { minimum = value; maximum = value; }
        else { minimum = min(minimum, value); maximum = max(maximum, value); }
        defined++;
      }
    }
    pair++;
  }
  var result = 0u;
  var valid = 0u;
  if (MODE == 0u) {
    result = matches;
    valid = 1u;
  } else if (defined > 0u) {
    var value = sum + compensation;
    if (MODE == 2u) { value = value / f32(defined); }
    if (MODE == 3u) { value = maximum; }
    if (MODE == 4u) { value = minimum; }
    result = bitcast<u32>(value);
    valid = 1u;
  }
  outputValues[OUTPUT_BASE + index] = result;
  outputValidity[VALIDITY_OUTPUT_BASE + index] = valid;
}`;
}
