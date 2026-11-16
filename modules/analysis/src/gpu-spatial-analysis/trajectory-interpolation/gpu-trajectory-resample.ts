// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH} from './trajectory-clock-parameters';
import {
  createTrajectoryArcLengthNode,
  createTrajectoryClockTimesNode,
  createTrajectoryResampleNode
} from './trajectory-interpolation-kernels';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';
import {resolveGPUTrajectoryTemporalContext} from '../trajectory-analysis/trajectory-common';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** How {@link GPUTrajectoryResample} spaces samples along each track. */
export type GPUTrajectoryResampleSpacing = 'time' | 'arc-length' | 'clock';

/**
 * Caller-owned dense outputs of {@link GPUTrajectoryResample}.
 */
export type GPUTrajectoryResampleOutput = {
  /** Dense `[trackCount × sampleCount]` positions, row `track * sampleCount + k`. */
  samples: GraphDataView<'float32x2'>;
  /** Optional dense elevations aligned with `samples`. Requires `elevations`. */
  sampleElevations?: GraphDataView<'float32'>;
  /**
   * Optional dense sample times, in the timestamps' unit: relative to each track's first timestamp,
   * or with `spacing: 'clock'` relative to the clock start (`k * step`). Requires `timestamps`.
   */
  sampleTimes?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUTrajectoryResample}: the trajectory layout
 * ({@link GPUTrajectoryGeometryFields}: `positions`, `trackOffsets`, and `timestamps`, which is
 * optional here) plus the options below.
 *
 * `positions` are packed rows (planar, or longitude/latitude degrees), sorted by track and then
 * time. `timestamps` are `float32` relative or `uint32x2` Int64 words, non-decreasing inside each
 * track, and are required for `spacing: 'time'` and for `output.sampleTimes`. `trackOffsets` are
 * `trackCount + 1` monotonic row offsets, the same layout as `GPUTrajectoryMetrics`.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * track count, `sampleCount`, `spacing`, the time format, and which optional views are present.
 */
export type GPUTrajectoryResampleProps = Omit<GPUTrajectoryGeometryFields, 'timestamps'> & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-resample'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates, linear
   * interpolation and planar arc length. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` means `(longitude, latitude)` degrees interpolated along great circles
   * (spherical linear interpolation) on a sphere of `sphereRadius`, defaulting to the mean Earth
   * radius in the context's length unit. Arc lengths are measured in the context's length unit.
   * Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /** Optional packed elevation (z) per sample. Required for `output.sampleElevations`. */
  elevations?: GraphDataView<'float32'>;
  /** Packed sample times, see above. Required for `spacing: 'time'` and `output.sampleTimes`. */
  timestamps?: GPUTrajectoryGeometryFields['timestamps'];
  /** Samples per track, at least 1. Sample `k` targets `k / (sampleCount - 1)` of the track. */
  sampleCount: number;
  /**
   * `'time'` (default) spaces samples uniformly between the first and last timestamp;
   * `'arc-length'` spaces them uniformly along the path length in the context's length unit;
   * `'clock'` samples every
   * track at the shared instants `start + k * step` of `clock` (see below).
   */
  spacing?: GPUTrajectoryResampleSpacing;
  /**
   * Per-frame shared clock, required for `spacing: 'clock'` and invalid otherwise. Float32 from
   * `getGPUTrajectoryClockParameterValues` for float32 timestamps, or uint32 from
   * `getGPUTrajectoryClockWordParameterValues` for word timestamps. Changing the start or step
   * only rewrites this buffer; the compiled graph is reused.
   */
  clock?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Optional per-frame gap limit, one float32 row in the timestamps' unit, for `spacing: 'clock'`
   * and `'time'`. A sample strictly inside a bracketing interval longer than this limit is absent (NaN),
   * exactly as for `GPUTrajectoryPlayhead`'s `gap` status: an interval equal to the limit is still
   * interpolated, and samples at the fixes themselves stay present. Omitted or 0 means no limit.
   */
  maximumGap?: GraphDataView<'float32'>;
  /** Caller-owned output views. */
  output: GPUTrajectoryResampleOutput;
};

/**
 * Resamples every track to a fixed number of samples, the dense input for trajectory similarity,
 * k-means, and smooth fixed-vertex trails.
 *
 * Sample `k` of a track with `n > 0` rows targets progress `total * k / (sampleCount - 1)`, where
 * progress is time since the first sample (`spacing: 'time'`) or cumulative planar length
 * (`'arc-length'`), and `total` is the progress of the last row. The last sample is the last row
 * exactly; with `sampleCount == 1` the only sample targets progress 0. The bracketing segment is
 * found by the same upper-bound search as `GPUTrajectoryPlayhead`, so duplicate timestamps and
 * zero-length steps resolve to their last row and never divide by zero. A single-row track
 * repeats that row; an empty track writes zeros (derive validity from `trackOffsets`). With
 * `maximumGap` and `spacing: 'time'`, samples strictly inside a longer interval are NaN.
 *
 * Common clock: with `spacing: 'clock'` column `k` means the same instant `start + k * step` for
 * every track, the dense input `GPUTrajectoryEncounters` needs. A track is interpolated only
 * while that instant lies inside its own time range; before its first and after its last sample,
 * and for empty tracks, the position (and elevation) is NaN, which downstream contributors read as
 * "absent". With `maximumGap` set, a sample strictly inside a longer bracketing interval is absent
 * too. `sampleCount` is then the number of clock steps.
 *
 * Spatial context: with a great-circle `spatialContext`, samples lie on the great-circle arc of
 * the bracketing segment and are interpolated with spherical linear interpolation instead of a
 * linear blend of degrees. Sample longitudes are continuous with the start row of the bracketing
 * segment, so a segment crossing the antimeridian yields longitudes beyond +-180 (the segment end
 * is `b` shifted by a multiple of 360); wrap downstream if needed. Absent clock samples stay NaN.
 *
 * Composition: in arc-length mode one per-track kernel sums step lengths sequentially in row order
 * with a Neumaier-compensated float32 sum, the same as the line family's measure prefix
 * (deterministic, but O(track length) per invocation); then one kernel with one invocation per
 * output sample.
 *
 * Precision: progress in `'time'` and `'clock'` spacing, and `output.sampleTimes`, are float32
 * differences from the track's first timestamp (word timestamps are subtracted exactly, then
 * rounded once). They are exact while a track spans fewer than 2^24 timestamp units (4.6 hours of
 * milliseconds); beyond that a sample time is rounded to the float32 spacing at that span, 1 unit
 * in 2^24. Arc lengths are float32 with a relative error of about 1e-6.
 */
export class GPUTrajectoryResample implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryResampleProps;
  /** Resolved spacing. */
  readonly spacing: GPUTrajectoryResampleSpacing;
  /** Resolved coordinate and metric context; native planar when `props.spatialContext` is omitted. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUTrajectoryResampleProps) {
    this.id = props.id ?? 'trajectory-resample';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.spacing = props.spacing ?? 'time';
    const id = this.id;
    const {output} = props;
    const rowCount = props.positions.length;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    if (this.spacing !== 'time' && this.spacing !== 'arc-length' && this.spacing !== 'clock') {
      throw new Error(`${id} spacing must be 'time', 'arc-length' or 'clock'`);
    }
    resolveGPUTrajectoryTemporalContext(id, props.timestamps, props.temporalContext);
    if (this.spacing === 'clock') {
      if (!props.clock) {
        throw new Error(`${id} clock is required for clock spacing`);
      }
      if (!props.timestamps) {
        throw new Error(`${id} timestamps are required for clock spacing`);
      }
      const clockFormat = props.timestamps.format === 'uint32x2' ? 'uint32' : 'float32';
      validateSinglePackedView(props.clock, [clockFormat], `${id} clock`);
      if (props.clock.length < GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH) {
        throw new Error(
          `${id} clock must hold ${GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH} ${clockFormat} values`
        );
      }
    } else if (props.clock) {
      throw new Error(`${id} clock requires spacing 'clock'`);
    }
    if (props.maximumGap) {
      if (this.spacing === 'arc-length') {
        throw new Error(`${id} maximumGap requires spacing 'clock' or 'time'`);
      }
      validateSinglePackedView(props.maximumGap, ['float32'], `${id} maximumGap`);
    }
    validateCapacity(id, 'sampleCount', props.sampleCount);
    if (props.timestamps) {
      validateSinglePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
      if (props.timestamps.length !== rowCount) {
        throw new Error(`${id} timestamps length must equal positions length`);
      }
    } else if (this.spacing === 'time' || output.sampleTimes) {
      throw new Error(`${id} timestamps are required for time spacing and sampleTimes`);
    }
    if (props.elevations) {
      validateSinglePackedView(props.elevations, ['float32'], `${id} elevations`);
      if (props.elevations.length !== rowCount) {
        throw new Error(`${id} elevations length must equal positions length`);
      }
    }
    if (output.sampleElevations && !props.elevations) {
      throw new Error(`${id} sampleElevations requires elevations`);
    }
    const trackCount = props.trackOffsets.length - 1;
    const outputLength = trackCount * props.sampleCount;
    if (outputLength > 0xffffffff) {
      throw new Error(`${id} trackCount * sampleCount must fit in 32 bits`);
    }
    for (const [name, view, format] of [
      ['samples', output.samples, 'float32x2'],
      ['sampleElevations', output.sampleElevations, 'float32'],
      ['sampleTimes', output.sampleTimes, 'float32']
    ] as const) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, [format], `${id} ${name}`);
      if (view.length !== outputLength) {
        throw new Error(`${id} ${name} length must equal trackCount * sampleCount`);
      }
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {
        samples: output.samples,
        sampleElevations: output.sampleElevations,
        sampleTimes: this.spacing === 'clock' && props.maximumGap ? undefined : output.sampleTimes
      }
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.samples, output.sampleElevations, output.sampleTimes],
      [
        props.positions,
        output.sampleElevations ? props.elevations : undefined,
        this.spacing !== 'arc-length' || output.sampleTimes ? props.timestamps : undefined,
        props.trackOffsets,
        props.clock,
        props.maximumGap
      ]
    );
  }

  /** Returns the optional arc-length node and the per-sample resample node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, {
      ...this.props,
      elevations: this.props.output.sampleElevations ? this.props.elevations : undefined,
      timestamps:
        this.spacing !== 'arc-length' || this.props.output.sampleTimes
          ? this.props.timestamps
          : undefined
    });
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.elevations,
      props.timestamps,
      props.trackOffsets,
      props.clock,
      props.maximumGap,
      output.samples,
      output.sampleElevations,
      output.sampleTimes
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const geodesicRadius = getGPUGreatCircleRadius(this.spatialContext);
    let cumulativeLengths: GraphDataView<'float32'> | undefined;
    if (this.spacing === 'arc-length') {
      cumulativeLengths = createTransientView(
        graph,
        `${id}-cumulative-lengths`,
        'float32',
        Math.max(rowCount, 1)
      );
      nodes.push(
        createTrajectoryArcLengthNode<Parameters>(graph, {
          id: `${id}-arc-length`,
          positions: props.positions,
          trackOffsets: props.trackOffsets,
          cumulativeLengths,
          geodesicRadius,
          rowCount,
          trackCount
        })
      );
    }
    const isGapClock = Boolean(this.spacing === 'clock' && props.maximumGap && props.clock);
    if (isGapClock && output.sampleTimes && props.clock) {
      nodes.push(
        createTrajectoryClockTimesNode<Parameters>(graph, {
          id: `${id}-clock-times`,
          trackOffsets: props.trackOffsets,
          clock: props.clock,
          sampleTimes: output.sampleTimes,
          isWordMode: props.timestamps?.format === 'uint32x2',
          rowCount,
          trackCount,
          sampleCount: props.sampleCount
        })
      );
    }
    nodes.push(
      createTrajectoryResampleNode<Parameters>(graph, {
        id: `${id}-samples`,
        positions: props.positions,
        elevations: props.elevations,
        timestamps:
          this.spacing !== 'arc-length' || output.sampleTimes ? props.timestamps : undefined,
        clock: props.clock,
        maximumGap: props.maximumGap,
        trackOffsets: props.trackOffsets,
        cumulativeLengths,
        samples: output.samples,
        sampleElevations: output.sampleElevations,
        sampleTimes: isGapClock ? undefined : output.sampleTimes,
        geodesicRadius,
        rowCount,
        trackCount,
        sampleCount: props.sampleCount
      })
    );
    return nodes;
  }
}
