// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  doGraphDataViewsOverlap,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTrajectoryPairwiseNodes} from './trajectory-pairwise-kernels';
import {createTrajectoryRestrictionNodes} from './trajectory-restriction-kernels';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

export const GPU_TRAJECTORY_RESTRICTION_STATUS = {
  interval: 1,
  singleton: 2,
  invalid: 4,
  gap: 8,
  fullTemporalSupportInside: 16,
  temporalSupport: 32,
  startOpen: 64,
  endOpen: 128,
  capacityExceeded: 256
} as const;

export type GPUTrajectoryRestrictionProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Finite relative timestamps. Strictly increasing for nonstationary segments. */
  times: GraphDataView<'float32'>;
  offsets: GraphDataView<'uint32'>;
  /** [minimumTime, maximumTime, minimumX, minimumY, maximumX, maximumY, maximumGap]; boundary inclusion is explicit. */
  parameters: GraphDataView<'float32'>;
  /** Static pairs of track IDs for temporal within-distance predicates. Parameters[2] is distance. */
  pairwiseTracks?: readonly (readonly [number, number])[];
  /** Per-pair interval capacity in pairwise mode; slots are already compact within each pair. */
  intervalsPerPair?: number;
  /** Rings are implicitly closed; coordinates follow the seven scalar parameters. Even-odd holes. */
  polygonRingVertexCounts?: readonly number[];
  timeBoundary?: 'closed' | 'open' | 'left-closed' | 'right-closed';
  geometryBoundary?: 'closed' | 'open';
  /** Fixed output slots per source observation, default one. Overflow is explicit. */
  intervalsPerSegment?: number;
  /** Merge touching intervals when their shared point is included. Requires compact outputs. */
  mergeIntervals?: boolean;
  output: {
    /** [start,end] interval slots per source observation or query pair; NaN for absent/unknown. */
    intervals: GraphDataView<'float32x2'>;
    /** [startX,startY,endX,endY] per interval slot. */
    positions: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
    /** Per track [ever, always, temporalSupportRows, unknownRows]; 0xffffffff denotes unknown relation. */
    relationships: GraphDataView<'uint32x4'>;
    compactIntervals?: GraphDataView<'float32x2'>;
    compactPositions?: GraphDataView<'float32x4'>;
    /** [first source row,last source row], preserving merged provenance ranges. */
    compactSources?: GraphDataView<'uint32x2'>;
    compactStatus?: GraphDataView<'uint32'>;
    /** [required interval count,written interval count,overflow count]. */
    compactSummary?: GraphDataView<'uint32'>;
  };
};

/**
 * Time/geometry restriction of planar linearly interpolated tracks, with explicit endpoint policy,
 * polygon rings, optional interval union and continuous pairwise within-distance predicates.
 * Positive-duration gaps above maximumGap are unknown support, never straight-line bridges.
 */
export class GPUTrajectoryRestriction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryRestrictionProps;
  constructor(props: GPUTrajectoryRestrictionProps) {
    this.id = props.id ?? 'trajectory-restriction';
    this.props = props;
    for (const view of [props.positions, props.output.intervals])
      validatePackedView(view, ['float32x2'], this.id);
    for (const view of [props.times, props.parameters])
      validatePackedView(view, ['float32'], this.id);
    for (const view of [props.offsets, props.output.status])
      validatePackedView(view, ['uint32'], this.id);
    validatePackedView(props.output.positions, ['float32x4'], this.id);
    validatePackedView(props.output.relationships, ['uint32x4'], this.id);
    const slots = props.intervalsPerSegment ?? 1;
    assert(Number.isInteger(slots) && slots > 0 && slots <= 256);
    const rings = props.polygonRingVertexCounts ?? [];
    assert(rings.every(count => Number.isInteger(count) && count >= 3));
    const vertexCount = rings.reduce((total, count) => total + count, 0);
    assert(vertexCount <= 128);
    if (props.output.compactIntervals) {
      const count = props.output.compactIntervals.length;
      assert(
        props.output.compactPositions &&
          props.output.compactSources &&
          props.output.compactStatus &&
          props.output.compactSummary
      );
      validatePackedView(props.output.compactIntervals, ['float32x2'], this.id);
      validatePackedView(props.output.compactPositions, ['float32x4'], this.id);
      validatePackedView(props.output.compactSources, ['uint32x2'], this.id);
      validatePackedView(props.output.compactStatus, ['uint32'], this.id);
      validatePackedView(props.output.compactSummary, ['uint32'], this.id);
      assert(
        props.output.compactPositions.length === count &&
          props.output.compactSources.length === count &&
          props.output.compactStatus.length === count &&
          props.output.compactSummary.length === 3
      );
    } else {
      assert(
        !props.output.compactPositions &&
          !props.output.compactSources &&
          !props.output.compactStatus &&
          !props.output.compactSummary &&
          !props.mergeIntervals
      );
    }
    if (props.pairwiseTracks) {
      assert(
        !props.polygonRingVertexCounts &&
          !props.output.compactIntervals &&
          props.intervalsPerSegment === undefined
      );
      assert(
        Number.isInteger(props.intervalsPerPair) &&
          props.intervalsPerPair! > 0 &&
          props.intervalsPerPair! <= 65536
      );
      assert(
        props.pairwiseTracks.length <= 4096 &&
          props.pairwiseTracks.every(pair =>
            pair.every(
              track => Number.isInteger(track) && track >= 0 && track < props.offsets.length - 1
            )
          )
      );
    } else {
      assert(props.intervalsPerPair === undefined);
    }
    const intervalRows = props.pairwiseTracks
      ? props.pairwiseTracks.length * props.intervalsPerPair!
      : props.positions.length * slots;
    assert(
      props.offsets.length >= 1 &&
        props.parameters.length === 7 + 2 * vertexCount &&
        props.times.length === props.positions.length &&
        props.output.intervals.length === intervalRows &&
        props.output.positions.length === intervalRows &&
        props.output.status.length === intervalRows &&
        props.output.relationships.length ===
          (props.pairwiseTracks?.length ?? props.offsets.length - 1)
    );
    // Separate passes cannot preserve independently named outputs in overlapping storage.
    const outputs = Object.values(props.output).filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
    for (const [index, output] of outputs.entries()) {
      assert(!outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)));
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.times,
      props.offsets,
      props.parameters
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.offsets,
      props.parameters,
      ...Object.values(props.output)
    ]);
    if (props.pairwiseTracks) return createTrajectoryPairwiseNodes(graph, id, props);
    if (
      props.polygonRingVertexCounts ||
      props.timeBoundary ||
      props.geometryBoundary ||
      props.intervalsPerSegment ||
      props.output.compactIntervals
    )
      return createTrajectoryRestrictionNodes(graph, id, props);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUTrajectoryRestriction',
        variant: 'closed-planar-rectangle',
        invocationCount: props.offsets.length - 1,
        workgroupSize: 32,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
          {name: 'clipped', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {
            name: 'relationships',
            view: props.output.relationships,
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `const OBSERVATION_COUNT: u32 = ${props.positions.length}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn loadPosition(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }`,
        body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u];
let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]); let maximumGap = parameters[parametersOffset + 6u];
var parametersValid = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && all(minimum <= maximum) && maximumGap > 0.0;
for (var parameter = 2u; parameter < 6u; parameter++) { parametersValid = parametersValid && finite(parameters[parametersOffset + parameter]); }
var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
var anyInside = false; var allInside = true; var supportCount = 0u; var unknownCount = 0u;
for (var row = start; row < end; row++) {
  for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * row + component] = missing; }
  for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * row + component] = missing; }
  status[statusOffset + row] = 0u;
  let next = min(row + 1u, end - 1u); let first = loadPosition(row); let second = loadPosition(next);
  let firstTime = times[timesOffset + row]; let secondTime = times[timesOffset + next]; let duration = secondTime - firstTime;
  if (!parametersValid || !finite(firstTime) || !finite(secondTime) || duration < 0.0 || !finite(first.x) || !finite(first.y) || !finite(second.x) || !finite(second.y) || (duration == 0.0 && any(first != second))) { status[statusOffset + row] = 4u; unknownCount++; continue; }
  if (secondTime < minimumTime || firstTime > maximumTime) { continue; }
  if (duration > maximumGap) {
    let unknownInterior = max(minimumTime, firstTime) < min(maximumTime, secondTime);
    status[statusOffset + row] = 8u;
    if (unknownInterior) { unknownCount++; }
    // A known endpoint remains a measured singleton even when the following gap is unsupported.
    if (firstTime >= minimumTime && firstTime <= maximumTime) {
      supportCount++;
      let inside = all(first >= minimum) && all(first <= maximum);
      allInside = allInside && inside; anyInside = anyInside || inside;
      status[statusOffset + row] |= 32u;
      if (inside) {
        status[statusOffset + row] |= 18u;
        intervals[intervalsOffset + 2u * row] = firstTime; intervals[intervalsOffset + 2u * row + 1u] = firstTime;
        clipped[clippedOffset + 4u * row] = first.x; clipped[clippedOffset + 4u * row + 1u] = first.y; clipped[clippedOffset + 4u * row + 2u] = first.x; clipped[clippedOffset + 4u * row + 3u] = first.y;
      }
    }
    continue;
  }
  supportCount++;
  var lower = 0.0; var upper = 1.0;
  if (duration > 0.0) { lower = max(0.0, (minimumTime - firstTime) / duration); upper = min(1.0, (maximumTime - firstTime) / duration); }
  let timeLower = lower; let timeUpper = upper; let direction = second - first;
  var intersects = true;
  for (var axis = 0u; axis < 2u; axis++) {
    if (direction[axis] == 0.0) { if (first[axis] < minimum[axis] || first[axis] > maximum[axis]) { intersects = false; } }
    else { let firstCrossing = (minimum[axis] - first[axis]) / direction[axis]; let secondCrossing = (maximum[axis] - first[axis]) / direction[axis]; lower = max(lower, min(firstCrossing, secondCrossing)); upper = min(upper, max(firstCrossing, secondCrossing)); }
  }
  intersects = intersects && lower <= upper;
  let fullyInside = intersects && lower == timeLower && upper == timeUpper;
  allInside = allInside && fullyInside;
  if (!intersects) { status[statusOffset + row] = 32u; continue; }
  anyInside = true;
  let clippedStart = first + direction * lower; let clippedEnd = first + direction * upper;
  intervals[intervalsOffset + 2u * row] = firstTime + duration * lower; intervals[intervalsOffset + 2u * row + 1u] = firstTime + duration * upper;
  clipped[clippedOffset + 4u * row] = clippedStart.x; clipped[clippedOffset + 4u * row + 1u] = clippedStart.y; clipped[clippedOffset + 4u * row + 2u] = clippedEnd.x; clipped[clippedOffset + 4u * row + 3u] = clippedEnd.y;
  status[statusOffset + row] = 32u | select(1u, 2u, duration == 0.0 || lower == upper) | select(0u, 16u, fullyInside);
}
var ever = select(0u, 1u, anyInside); var always = select(0u, 1u, allInside);
if (supportCount == 0u || unknownCount > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = supportCount; relationships[relationshipsOffset + 4u * index + 3u] = unknownCount;`
      })
    ];
  }
}
