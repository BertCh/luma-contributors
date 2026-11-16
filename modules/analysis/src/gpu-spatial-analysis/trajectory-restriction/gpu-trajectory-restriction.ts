// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges,
  validateTrajectoryWritableDispatches
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  doGraphDataViewsOverlap,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getEffectiveMaximumGapWGSL,
  getGPUTrajectoryTimeColumns,
  resolveGPUTrajectoryTemporalContext,
  type GPUTrajectoryTimeColumns,
  type GPUTrajectoryTimeFields
} from '../trajectory-analysis/trajectory-common';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createTrajectoryPairwiseMergeNodes} from './trajectory-restriction-pair-merge-kernels';
import {
  createRestrictionWordTimeNodes,
  createTrajectoryRestrictionNodes,
  getRestrictionTimeSource
} from './trajectory-restriction-kernels';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Bits of the per-slot `output.status` word (and of `output.compactStatus`) of
 * {@link GPUTrajectoryRestriction}. Bits 1 and 2 describe the interval a slot holds; bits 4, 8, 32
 * and 16 are set on the first slot of a source observation (or query pair); bit 256 on the first
 * slot of a source observation whose slots overflowed.
 */
export const GPU_TRAJECTORY_RESTRICTION_STATUS = {
  /** The slot holds a positive-duration interval. */
  interval: 1,
  /** The slot holds a single instant (zero duration). */
  singleton: 2,
  /** The segment or pair is unknown: invalid parameters, nonfinite or decreasing data, or a pair track out of range. */
  invalid: 4,
  /** The segment spans a gap longer than `maximumGap`: its support is unknown, never bridged. */
  gap: 8,
  /** The whole temporal support of the segment lies inside the geometry. */
  fullTemporalSupportInside: 16,
  /** The segment has temporal support inside the time window. */
  temporalSupport: 32,
  /** The slot's interval excludes its start instant (open start boundary). */
  startOpen: 64,
  /** The slot's interval excludes its end instant (open end boundary). */
  endOpen: 128,
  /** More intervals than `intervalsPerSegment` or `intervalsPerPair` slots were found; extras are dropped. */
  capacityExceeded: 256
} as const;

type GPUTrajectoryRestrictionOptions = {
  id?: string;
  /**
   * Coordinate space of `positions`, the box and the polygon rings. Only planar coordinates are
   * supported (segments are linearly interpolated in the plane); longitude-latitude, grid and
   * topology contexts are rejected at construction. With `pairs` or `pairwiseTracks` the context
   * also measures a distance, so it needs the native metric and no `local-lambert-equal-area`
   * frame. Defaults to native planar.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /**
   * Word `times` only: per-frame `[low, high]` uint32 origin. `minimumTime`, `maximumTime` and
   * output intervals are f32 time units relative to it, so keep it near the window of interest
   * (within 2^24 units keeps sample times exact). `maximumGap` stays in the timestamps' unit.
   */
  timeOrigin?: GraphDataView<'uint32'>;
  /**
   * [minimumTime, maximumTime, minimumX, minimumY, maximumX, maximumY, maximumGap]; boundary
   * inclusion is explicit. `maximumGap` is in the timestamps' unit and 0 means no limit.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time pairs of track IDs for temporal within-distance predicates (up to 4096, baked
   * into the shader: changing them recompiles). Parameters[2] is distance. Prefer `pairs`.
   */
  pairwiseTracks?: readonly (readonly [number, number])[];
  /**
   * Pairs of track IDs read from GPU views, for example `GPUTrajectoryEncounters`
   * `output.pairs.ids`, `output.pairs.partners` and `output.pairs.status.count`. The pair capacity
   * is `first.length`. Pair rows at and after `count` are inactive: NaN intervals, status 0 and
   * relationships `[0xffffffff, 0xffffffff, 0, 0]`. An active pair with a track outside the
   * track range gets status bit 4 and the same unknown relationships. Per-frame.
   */
  pairs?: {
    first: GraphDataView<'uint32'>;
    second: GraphDataView<'uint32'>;
    count?: GraphDataView<'uint32'>;
  };
  /** Per-pair interval capacity in pairwise mode; slots are already compact within each pair. */
  intervalsPerPair?: number;
  /** Rings are implicitly closed; coordinates follow the seven scalar parameters. Even-odd holes. */
  polygonRingVertexCounts?: readonly number[];
  timeBoundary?: 'closed' | 'open' | 'left-closed' | 'right-closed';
  /**
   * Whether the boundary of the box or polygon belongs to it (default `'closed'`). For a polygon a
   * point is on the boundary only when its exact orientation against an edge is zero inside the
   * edge's bounding box (the shared exact predicates, as `GPUZoneEvents`); there is no tolerance
   * band. A point where a segment crosses an edge, and a piece of a segment that runs along an
   * edge, are on the boundary by construction. Whether a segment meets an edge is decided with the
   * same exact orientation signs; float32 only places a crossing that exists.
   */
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
    /** One {@link GPU_TRAJECTORY_RESTRICTION_STATUS} bit set per interval slot; 0 for an absent slot. */
    status: GraphDataView<'uint32'>;
    /** Per track [ever, always, temporalSupportRows, unknownRows]; 0xffffffff denotes unknown relation. */
    relationships: GraphDataView<'uint32x4'>;
    compactIntervals?: GraphDataView<'float32x2'>;
    compactPositions?: GraphDataView<'float32x4'>;
    /** [first source row,last source row], preserving merged provenance ranges. */
    compactSources?: GraphDataView<'uint32x2'>;
    /** The {@link GPU_TRAJECTORY_RESTRICTION_STATUS} bits of each compact interval's first source slot. */
    compactStatus?: GraphDataView<'uint32'>;
    /** [required interval count,written interval count,overflow count]. */
    compactSummary?: GraphDataView<'uint32'>;
  };
};

/**
 * Time is either the trajectory port (`timestamps`, `trackOffsets`, as for the other trajectory
 * contributors) or the equivalent `times` and `offsets`. Finite and strictly increasing for
 * nonstationary segments. `float32`: relative time. `uint32x2`: Int64 `(low, high)` words, with
 * steps differenced exactly before the f32 conversion; requires `timeOrigin`.
 */
export type GPUTrajectoryRestrictionProps = GPUTrajectoryRestrictionOptions &
  GPUTrajectoryTimeFields;

/** {@link GPUTrajectoryRestrictionProps} with the time columns under one name. @internal */
export type GPUTrajectoryRestrictionResolvedProps = GPUTrajectoryRestrictionOptions &
  GPUTrajectoryTimeColumns;

/**
 * Time/geometry restriction of planar linearly interpolated tracks, with explicit endpoint policy,
 * polygon rings, optional interval union and continuous pairwise within-distance predicates.
 * Positive-duration gaps above a nonzero maximumGap are unknown support, never straight-line bridges.
 *
 * Pairwise mode merges the two time-sorted tracks of each pair in one pass (`O(nA + nB)` per pair,
 * not the product), and takes its pairs from GPU views (`pairs`), so an encounter list can feed it
 * without a readback. Tracks must be sorted by time. Unknown count: the number of invalid
 * segments of both tracks plus the number of overlapping segment pairs touching a gap.
 */
export class GPUTrajectoryRestriction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryRestrictionResolvedProps;
  /** Resolved coordinate context: always planar. */
  readonly spatialContext: GPUSpatialContext;
  constructor(input: GPUTrajectoryRestrictionProps) {
    this.id = input.id ?? 'trajectory-restriction';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      input.spatialContext,
      input.pairs || input.pairwiseTracks
        ? Object.freeze({...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT, measure: 'distance'} as const)
        : GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
    );
    const {times, offsets} = getGPUTrajectoryTimeColumns(this.id, input);
    resolveGPUTrajectoryTemporalContext(this.id, times, input.temporalContext);
    this.props = {...input, times, offsets};
    const {props} = this;
    for (const view of [props.positions, props.output.intervals])
      validateSinglePackedView(view, ['float32x2'], this.id);
    validateSinglePackedView(props.times, ['float32', 'uint32x2'], this.id);
    validateSinglePackedView(props.parameters, ['float32'], this.id);
    validateRequirement(
      this.id,
      (props.times.format === 'uint32x2') === Boolean(props.timeOrigin),
      'word times need timeOrigin and float times must not have one'
    );
    if (props.timeOrigin) {
      validateSinglePackedView(props.timeOrigin, ['uint32'], this.id);
      validateRequirement(
        this.id,
        props.timeOrigin.length >= 2,
        'timeOrigin must hold at least two rows'
      );
    }
    for (const view of [props.offsets, props.output.status])
      validateSinglePackedView(view, ['uint32'], this.id);
    validateSinglePackedView(props.output.positions, ['float32x4'], this.id);
    validateSinglePackedView(props.output.relationships, ['uint32x4'], this.id);
    const slots = props.intervalsPerSegment ?? 1;
    validateRequirement(
      this.id,
      Number.isInteger(slots) && slots > 0 && slots <= 256,
      'intervalsPerSegment must be an integer in [1, 256]'
    );
    const rings = props.polygonRingVertexCounts ?? [];
    validateRequirement(
      this.id,
      rings.every(count => Number.isInteger(count) && count >= 3),
      'polygon rings need at least 3 vertices each'
    );
    const vertexCount = rings.reduce((total, count) => total + count, 0);
    validateRequirement(this.id, vertexCount <= 128, 'polygon vertices must total at most 128');
    if (props.output.compactIntervals) {
      const count = props.output.compactIntervals.length;
      validateRequirement(
        this.id,
        Boolean(
          props.output.compactPositions &&
            props.output.compactSources &&
            props.output.compactStatus &&
            props.output.compactSummary
        ),
        'compact outputs need compactPositions, compactSources, compactStatus and compactSummary'
      );
      validateSinglePackedView(props.output.compactIntervals, ['float32x2'], this.id);
      validateSinglePackedView(props.output.compactPositions!, ['float32x4'], this.id);
      validateSinglePackedView(props.output.compactSources!, ['uint32x2'], this.id);
      validateSinglePackedView(props.output.compactStatus!, ['uint32'], this.id);
      validateSinglePackedView(props.output.compactSummary!, ['uint32'], this.id);
      validateRequirement(
        this.id,
        props.output.compactPositions!.length === count &&
          props.output.compactSources!.length === count &&
          props.output.compactStatus!.length === count &&
          props.output.compactSummary!.length === 3,
        'compact outputs need the length of compactIntervals and a 3-row compactSummary'
      );
    } else {
      validateRequirement(
        this.id,
        !props.output.compactPositions &&
          !props.output.compactSources &&
          !props.output.compactStatus &&
          !props.output.compactSummary &&
          !props.mergeIntervals,
        'compact outputs are absent unless compactIntervals is given, and mergeIntervals needs them'
      );
    }
    validateRequirement(
      this.id,
      !(props.pairwiseTracks && props.pairs),
      'takes pairwiseTracks or pairs'
    );
    if (props.pairs) {
      for (const view of [props.pairs.first, props.pairs.second, props.pairs.count])
        if (view) validateSinglePackedView(view, ['uint32'], this.id);
      validateRequirement(
        this.id,
        props.pairs.first.length === props.pairs.second.length,
        'pairs.first and pairs.second must have the same length'
      );
      validateRequirement(
        this.id,
        !props.pairs.count || props.pairs.count.length >= 1,
        'pairs.count must hold at least one row'
      );
    }
    if (props.pairwiseTracks || props.pairs) {
      validateRequirement(
        this.id,
        !props.polygonRingVertexCounts &&
          !props.output.compactIntervals &&
          props.intervalsPerSegment === undefined,
        'pairwise mode excludes polygons, compact outputs and intervalsPerSegment'
      );
      validateRequirement(
        this.id,
        Number.isInteger(props.intervalsPerPair) &&
          props.intervalsPerPair! > 0 &&
          props.intervalsPerPair! <= 65536,
        'intervalsPerPair must be an integer in [1, 65536]'
      );
      validateRequirement(
        this.id,
        (props.pairwiseTracks ?? []).length <= 4096 &&
          (props.pairwiseTracks ?? []).every(pair =>
            pair.every(
              track => Number.isInteger(track) && track >= 0 && track < props.offsets.length - 1
            )
          ),
        'pairwiseTracks must hold at most 4096 pairs of track indices below the track count'
      );
    } else {
      validateRequirement(
        this.id,
        props.intervalsPerPair === undefined,
        'intervalsPerPair needs pairwise mode'
      );
    }
    const pairCapacity = props.pairs?.first.length ?? props.pairwiseTracks?.length;
    const intervalRows =
      pairCapacity !== undefined
        ? pairCapacity * props.intervalsPerPair!
        : props.positions.length * slots;
    validateRequirement(
      this.id,
      props.offsets.length >= 1 &&
        props.parameters.length === 7 + 2 * vertexCount &&
        props.times.length === props.positions.length &&
        props.output.intervals.length === intervalRows &&
        props.output.positions.length === intervalRows &&
        props.output.status.length === intervalRows &&
        props.output.relationships.length === (pairCapacity ?? props.offsets.length - 1),
      'view lengths must match the observation, track, parameter and interval counts'
    );
    // Separate passes cannot preserve independently named outputs in overlapping storage.
    const outputs = Object.values(props.output).filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
    for (const [index, output] of outputs.entries()) {
      validateRequirement(
        this.id,
        !outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)),
        'output views must not overlap'
      );
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {
        intervals: props.output.intervals,
        positions: props.output.positions,
        status: props.output.status,
        relationships: props.output.relationships
      }
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.times,
      props.offsets,
      props.parameters,
      props.timeOrigin,
      props.pairs?.first,
      props.pairs?.second,
      props.pairs?.count
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.offsets,
      props.parameters,
      props.timeOrigin,
      props.pairs?.first,
      props.pairs?.second,
      props.pairs?.count,
      ...Object.values(props.output)
    ]);
    if (props.times.format === 'uint32x2')
      validateTrajectoryScratch(graph, id, {
        'relative word times': Math.max(2 * props.positions.length, 1) * 4
      });
    const wordTime =
      props.times.format === 'uint32x2'
        ? createRestrictionWordTimeNodes(graph, id, props)
        : undefined;
    const timeTable = wordTime?.timeTable;
    const prefix = wordTime?.nodes ?? [];
    if (props.pairwiseTracks || props.pairs)
      return [
        ...prefix,
        ...createTrajectoryPairwiseMergeNodes(
          graph,
          id,
          props,
          props.pairs
            ? {...props.pairs, capacity: props.pairs.first.length}
            : {staticPairs: props.pairwiseTracks, capacity: props.pairwiseTracks!.length},
          timeTable
        )
      ];
    if (
      props.polygonRingVertexCounts ||
      props.timeBoundary ||
      props.geometryBoundary ||
      props.intervalsPerSegment ||
      props.output.compactIntervals
    )
      return [...prefix, ...createTrajectoryRestrictionNodes(graph, id, props, timeTable)];
    const time = getRestrictionTimeSource(props, timeTable);
    return [
      ...prefix,
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUTrajectoryRestriction',
        variant: `closed-planar-rectangle${time.suffix}`,
        invocationCount: props.offsets.length - 1,
        workgroupSize: 32,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          ...time.bindings,
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
${time.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn loadPosition(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }`,
        body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u];
let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]); let maximumGap = ${getEffectiveMaximumGapWGSL('parameters[parametersOffset + 6u]')};
var parametersValid = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && all(minimum <= maximum) && maximumGap > 0.0;
for (var parameter = 2u; parameter < 6u; parameter++) { parametersValid = parametersValid && finite(parameters[parametersOffset + parameter]); }
var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
var anyInside = false; var allInside = true; var supportCount = 0u; var unknownCount = 0u;
for (var row = start; row < end; row++) {
  for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * row + component] = missing; }
  for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * row + component] = missing; }
  status[statusOffset + row] = 0u;
  let next = min(row + 1u, end - 1u); let first = loadPosition(row); let second = loadPosition(next);
  let firstTime = rowTime(row); let secondTime = rowTime(next); let duration = rowElapsed(next, row);
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
