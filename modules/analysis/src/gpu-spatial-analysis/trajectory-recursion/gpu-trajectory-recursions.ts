// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCandidateStatusPort,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {
  createRowBoundsNodes,
  MOVEMENT_SENTINEL_WGSL
} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUTrajectoryRecursions';

/** Default cap on the segments one row examines, per radius. */
export const GPU_TRAJECTORY_RECURSIONS_DEFAULT_MAXIMUM_TRACK_SEGMENTS = 4096;

/**
 * Largest accepted cap on the segments one row examines. The contributor is registered with
 * `tier: 'reference'`: each (fix, radius) loops over the segments of its track, so work is
 * quadratic in the track length (`fixCount * maximumTrackSegments * radiusCount`).
 */
export const GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS = 65536;

/** Largest number of radii evaluated in one pass (one output column each). */
export const GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT = 32;

/** Caller-owned outputs of {@link GPUTrajectoryRecursions}. At least one is required. */
export type GPUTrajectoryRecursionsOutput = {
  /**
   * Revisits of the circle around each fix: the number of times the track enters the circle,
   * not counting the visit that contains the fix itself. `fixCount * radiusCount` rows, row
   * `fix * radiusCount + radius`.
   */
  revisits?: GraphDataView<'uint32'>;
  /**
   * Total time the track spends inside the circle around each fix, over all visits including the
   * one that contains the fix, in the unit of the timestamps. Same layout as `revisits`.
   */
  residenceTimes?: GraphDataView<'float32'>;
  /**
   * Forward first passage time: the time from the fix, at the circle's centre, until the track
   * first leaves the circle around it, in the unit of the timestamps. Same layout as `revisits`.
   * If the track ends inside the circle the value is the time to the end of the track (censored).
   * This is the forward leg only. The first passage time of Fauchald and Tveraa (2003) and of
   * `adehabitatLT::fpt` is the time to cross the whole circle, from the last entry before the fix
   * to the first exit after it, which on a straight track at constant speed is twice this value.
   */
  firstPassageTimes?: GraphDataView<'float32'>;
  /** 1 where `firstPassageTimes` is censored (the track ended before leaving the circle). Same layout. */
  firstPassageCensored?: GraphDataView<'uint32'>;
  /**
   * `candidateOverflow` is 1 when a track has more than `maximumTrackSegments` segments, so some
   * rows examined only a window of their track.
   */
  status?: GPUCandidateStatusPort;
};

/** Properties for {@link GPUTrajectoryRecursions}. */
export type GPUTrajectoryRecursionsProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-recursions'`. */
  id?: string;
  /** Planar only. */
  spatialContext?: GPUSpatialContext;
  /**
   * Circle radii in the length unit of the positions, `radiusCount` rows (derived from this view,
   * at most {@link GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT}). A radius that is not positive
   * gives zero revisits, zero residence and zero first passage time. Per-frame contents.
   */
  radii: GraphDataView<'float32'>;
  /**
   * Segments examined per row (default 4096, at most 65,536). A longer track is examined in a
   * window of this many segments around the row and sets `status.candidateOverflow`. Compile-time.
   */
  maximumTrackSegments?: number;
  output: GPUTrajectoryRecursionsOutput;
};

/**
 * Recursion analysis of trajectories after Bracis, Bildstein and Mueller (Ecography 2018,
 * `recurse::getRecursions`), and a forward first passage time: for the circle of radius `r` around
 * each fix, how often the track returns, how long it stays, and how long it takes to leave.
 *
 * The path between fixes is the straight segment traversed at constant speed, so crossings of the
 * circle are exact (a quadratic per segment) and times are linear interpolations; residence
 * and passage times are not limited to fix resolution. Definitions used:
 * - A visit is a maximal part of the path inside the closed disc. The visit containing the focal
 *   fix always exists, so `revisits = visits - 1`. A segment that starts outside and touches the
 *   disc starts a visit.
 * - `residenceTimes` sums the time inside the disc over all segments of the track.
 * - `firstPassageTimes` follows the track forward from the focal fix; the exit time is
 *   interpolated on the segment that leaves the disc. The backward leg is not added, so the value
 *   is not the circle-crossing time that Fauchald and Tveraa (Ecology 2003) and
 *   `adehabitatLT::fpt` call first passage time (about half of it on a steady track).
 *
 * Only the track of the fix is examined. `recurse` options that are not provided: polygon
 * locations, a minimum time outside to end a visit (`threshold`), and its interval residence time.
 * The behaviour of the R package was not checked (R is not installed here and its licence is not
 * verified); the tests use a binary64 CPU implementation of the definitions above and tracks with
 * hand-computable answers.
 *
 * Planar only; positions should be in a local frame (float32 cancellation in the crossing test grows
 * with the distance from the origin). Per-frame: `radii` and the track data. Compile-time: the
 * number of radii, `maximumTrackSegments` and which outputs exist. Reference tier, see
 * {@link GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS}.
 */
export class GPUTrajectoryRecursions implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of fixes. */
  readonly rowCount: number;
  /** Number of radii. */
  readonly radiusCount: number;
  /** Segments examined per row. */
  readonly maximumTrackSegments: number;
  /** Validated properties. */
  readonly props: GPUTrajectoryRecursionsProps;

  constructor(props: GPUTrajectoryRecursionsProps) {
    const id = props.id ?? 'trajectory-recursions';
    resolveGPUSpatialContext(id, props.spatialContext, GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT);
    const {positions, timestamps, trackOffsets, radii, output} = props;
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    this.rowCount = positions.length;
    validateCapacity(id, 'fix count', this.rowCount);
    validateViewLength(id, 'timestamps', timestamps, this.rowCount);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    validateSinglePackedView(radii, ['float32'], `${id} radii`);
    this.radiusCount = radii.length;
    validateCapacity(
      id,
      'radius count',
      this.radiusCount,
      1,
      GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT
    );
    this.maximumTrackSegments =
      props.maximumTrackSegments ?? GPU_TRAJECTORY_RECURSIONS_DEFAULT_MAXIMUM_TRACK_SEGMENTS;
    validateCapacity(id, 'maximumTrackSegments', this.maximumTrackSegments);
    validateMaximumCount(
      id,
      'maximumTrackSegments',
      this.maximumTrackSegments,
      GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS
    );
    validateRequirement(
      id,
      Boolean(output.revisits || output.residenceTimes || output.firstPassageTimes),
      'output must request revisits, residenceTimes or firstPassageTimes'
    );
    const outputRows = this.rowCount * this.radiusCount;
    for (const [name, view, format] of [
      ['revisits', output.revisits, 'uint32'],
      ['residenceTimes', output.residenceTimes, 'float32'],
      ['firstPassageTimes', output.firstPassageTimes, 'float32'],
      ['firstPassageCensored', output.firstPassageCensored, 'uint32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} output.${name}`);
        validateViewLength(id, `output.${name}`, view, outputRows);
      }
    }
    validateRequirement(
      id,
      !output.firstPassageCensored || Boolean(output.firstPassageTimes),
      'output.firstPassageCensored requires output.firstPassageTimes'
    );
    this.id = id;
    this.props = props;
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), [
      positions,
      timestamps,
      trackOffsets,
      radii
    ]);
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.revisits,
      output.residenceTimes,
      output.firstPassageTimes,
      output.firstPassageCensored,
      output.status?.candidateOverflow
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, radiusCount, maximumTrackSegments} = this;
    const {positions, timestamps, trackOffsets, radii, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      timestamps,
      trackOffsets,
      radii,
      ...this.getOutputViews()
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'output.revisits',
      rowCount * radiusCount * 4,
      'use fewer fixes or radii per graph'
    );
    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const timeMode = getTrajectoryTimeMode(props);
    const declarations = `${getTrajectoryTimeSource(timeMode)}
const RADIUS_COUNT: u32 = ${radiusCount}u;
const CAP: u32 = ${maximumTrackSegments}u;
fn fixPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

// Segments [low, high) of the row's track that the row examines.
fn getWindow(row: u32) -> vec2<u32> {
  let start = rowStarts[rowStartsOffset + row];
  let end = rowEnds[rowEndsOffset + row];
  let segmentCount = end - start - 1u;
  if (segmentCount <= CAP) {
    return vec2<u32>(start, start + segmentCount);
  }
  let wanted = select(start, row - CAP / 2u, row >= start + CAP / 2u);
  let low = min(wanted, end - 1u - CAP);
  return vec2<u32>(low, low + CAP);
}

// Parameters of the closed disc inside a segment: (first, last, 1) or (0, 0, 0) when it misses.
fn getInsideInterval(pathStart: vec2<f32>, pathEnd: vec2<f32>, center: vec2<f32>, radius: f32) -> vec3<f32> {
  let step = pathEnd - pathStart;
  let offset = pathStart - center;
  let a = dot(step, step);
  let b = 2.0 * dot(offset, step);
  let c = dot(offset, offset) - radius * radius;
  if (a == 0.0) {
    return select(vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 1.0), c <= 0.0);
  }
  let discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    return vec3<f32>(0.0);
  }
  let root = sqrt(discriminant);
  let first = max((-b - root) / (2.0 * a), 0.0);
  let last = min((-b + root) / (2.0 * a), 1.0);
  return select(vec3<f32>(0.0), vec3<f32>(first, last, 1.0), first <= last);
}`;
    const rowBounds = {name: 'rowStarts', view: rowStarts, type: 'u32', access: 'read'} as const;
    const rowEndsBinding = {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'} as const;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...createRowBoundsNodes<Parameters>(graph, {
        id: `${id}-row-bounds`,
        operation: OPERATION,
        trackOffsets,
        rowCount,
        rowStarts,
        rowEnds,
        rowTracks
      })
    ];
    if (output.revisits || output.residenceTimes) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-visits`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: rowCount * radiusCount,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...getTrajectoryTimeBindings(props),
            rowBounds,
            rowEndsBinding,
            {name: 'radii', view: radii, type: 'f32', access: 'read'},
            ...(output.revisits
              ? [
                  {
                    name: 'revisits',
                    view: output.revisits,
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : []),
            ...(output.residenceTimes
              ? [
                  {
                    name: 'residenceTimes',
                    view: output.residenceTimes,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          declarations,
          body: /* wgsl */ `
  let row = index / RADIUS_COUNT;
  let radius = radii[radiiOffset + index % RADIUS_COUNT];
  var visits = 0u;
  var residence = 0.0;
  if (rowEnds[rowEndsOffset + row] != ${MOVEMENT_SENTINEL_WGSL} && radius > 0.0) {
    let center = fixPosition(row);
    let window = getWindow(row);
    for (var segment = window.x; segment < window.y; segment = segment + 1u) {
      let pathStart = fixPosition(segment);
      let outside = distance(pathStart, center) > radius;
      if (segment == window.x && !outside) {
        visits = visits + 1u;
      }
      let interval = getInsideInterval(pathStart, fixPosition(segment + 1u), center, radius);
      if (interval.z != 0.0) {
        if (outside) {
          visits = visits + 1u;
        }
        let duration = timeDifferenceRows(segment + 1u, segment);
        if (duration > 0.0) {
          residence = residence + (interval.y - interval.x) * duration;
        }
      }
    }
  }
${output.revisits ? '  revisits[revisitsOffset + index] = select(0u, visits - 1u, visits > 0u);' : ''}
${output.residenceTimes ? '  residenceTimes[residenceTimesOffset + index] = residence;' : ''}`
        })
      );
    }
    if (output.firstPassageTimes) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-passage`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: rowCount * radiusCount,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...getTrajectoryTimeBindings(props),
            rowBounds,
            rowEndsBinding,
            {name: 'radii', view: radii, type: 'f32', access: 'read'},
            {
              name: 'passageTimes',
              view: output.firstPassageTimes,
              type: 'f32',
              access: 'read_write'
            },
            ...(output.firstPassageCensored
              ? [
                  {
                    name: 'censored',
                    view: output.firstPassageCensored,
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          declarations,
          body: /* wgsl */ `
  let row = index / RADIUS_COUNT;
  let radius = radii[radiiOffset + index % RADIUS_COUNT];
  var passage = 0.0;
  var isCensored = 1u;
  let trackEnd = rowEnds[rowEndsOffset + row];
  if (trackEnd != ${MOVEMENT_SENTINEL_WGSL} && radius > 0.0) {
    let center = fixPosition(row);
    let window = getWindow(row);
    let last = min(window.y, trackEnd - 1u);
    var segment = max(row, window.x);
    var found = false;
    for (; segment < last; segment = segment + 1u) {
      let pathEnd = fixPosition(segment + 1u);
      if (distance(pathEnd, center) > radius) {
        // The segment starts inside the disc and ends outside: it leaves at the larger root.
        let interval = getInsideInterval(fixPosition(segment), pathEnd, center, radius);
        passage = timeDifferenceRows(segment, row) + interval.y * timeDifferenceRows(segment + 1u, segment);
        isCensored = 0u;
        found = true;
        break;
      }
    }
    if (!found) {
      passage = timeDifferenceRows(last, row);
    }
  }
  passageTimes[passageTimesOffset + index] = passage;
${output.firstPassageCensored ? '  censored[censoredOffset + index] = isCensored;' : '  _ = isCensored;'}`
        })
      );
    }
    if (output.status?.candidateOverflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-candidate-reset`,
          operation: OPERATION,
          view: output.status.candidateOverflow,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-candidate-overflow`,
          operation: OPERATION,
          variant: 'overflow',
          invocationCount: rowCount,
          bindings: [
            rowBounds,
            rowEndsBinding,
            {
              name: 'candidateOverflow',
              view: output.status.candidateOverflow,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          declarations: `const CAP: u32 = ${maximumTrackSegments}u;`,
          body: /* wgsl */ `
  let end = rowEnds[rowEndsOffset + index];
  if (end != ${MOVEMENT_SENTINEL_WGSL} && end - rowStarts[rowStartsOffset + index] - 1u > CAP) {
    atomicStore(&candidateOverflow[candidateOverflowOffset], 1u);
  }`
        })
      );
    }
    return nodes;
  }
}
