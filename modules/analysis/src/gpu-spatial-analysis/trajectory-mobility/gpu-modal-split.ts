// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTrajectoryBindings,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUStatusScalar,
  GPUTrajectoryGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../trajectory-analysis/trajectory-metrics-kernels';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {createTrackTableOffsetsNode} from '../trajectory-staypoints/movement-common';

const OPERATION = 'GPUModalSplit';
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** What {@link GPUModalSplit} accumulates per (track, mode). */
export type GPUModalSplitMetric = 'count' | 'distance' | 'duration';

/** Tripleg table consumed by {@link GPUModalSplit}. */
export type GPUModalSplitTriplegs = {
  /** Track of each tripleg, sorted. */
  tracks: GraphDataView<'uint32'>;
  /** First row of each tripleg. */
  startRows: GraphDataView<'uint32'>;
  /** Exclusive end row of each tripleg. */
  endRows: GraphDataView<'uint32'>;
  /** One row in `count`: active tripleg count, as `GPUTrajectorySegments` writes it. */
  status: {count: GPUStatusScalar};
};

/**
 * Properties for {@link GPUModalSplit}: the trajectory layout plus the tripleg table, the tripleg
 * modes, and the metric. `modes` holds one row per tripleg capacity row, values in
 * `[0, modeCount)`; when `modeSpeedBreaks` is given, modes are predicted from average tripleg
 * speed instead (trackintel `simple-coarse`) and written to `output.predictedModes`.
 */
export type GPUModalSplitProps = GPUTrajectoryGeometryFields & {
  /** Prefix for node and transient IDs. Defaults to `'modal-split'`. */
  id?: string;
  /** Planar or longitude/latitude great-circle; distances in the context's length unit. */
  spatialContext?: GPUSpatialContext;
  triplegs: GPUModalSplitTriplegs;
  /**
   * Mode of each tripleg, unless `modeSpeedBreaks` is given. A tripleg whose mode is the missing
   * key `0xffffffff` is left out of every class; any other value at or above `modeCount` counts
   * as the last class.
   */
  modes?: GraphDataView<'uint32'>;
  /**
   * Ascending upper bounds (exclusive) of each speed class, length unit per timestamp unit:
   * class `i` holds speeds in `[modeSpeedBreaks[i - 1], modeSpeedBreaks[i])`, `-inf` below the
   * first (compile-time, `modeCount` is its length). The last class has no upper bound: a speed at
   * or above the last break stays in the last class, so a finite last break only names the class
   * (trackintel leaves such a tripleg without a category). A tripleg with no speed, because no
   * time passes between its first and last row or its length is not finite, gets the missing key
   * `0xffffffff` and is left out of `output.modalSplit`. Trackintel `simple-coarse` is
   * `[15/3.6, 100/3.6, Infinity]`, in metres per second.
   */
  modeSpeedBreaks?: readonly number[];
  /** Number of modes; with `modeSpeedBreaks`, its length. */
  modeCount: number;
  /** Aggregation per (track, mode). Compile-time. */
  metric: GPUModalSplitMetric;
  output: {
    /** Per-(track, mode) accumulated metric, `trackCount * modeCount` rows, major axis track. */
    modalSplit: GraphDataView<'float32'>;
    /**
     * Optional predicted mode per tripleg (requires `modeSpeedBreaks`); `0xffffffff` for a tripleg
     * with no speed. Rows at and after the tripleg count are 0.
     */
    predictedModes?: GraphDataView<'uint32'>;
  };
};

/**
 * Modal split of triplegs (trackintel `calculate_modal_split`): one thread per track accumulates
 * count, summed haversine or Euclidean length, or summed duration per (track, mode). With
 * `modeSpeedBreaks`, the mode comes from trackintel's `simple-coarse` speed classes.
 */
export class GPUModalSplit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUModalSplitProps;
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUModalSplitProps) {
    this.id = props.id ?? 'modal-split';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {
      positions,
      timestamps,
      trackOffsets,
      triplegs,
      output,
      modes,
      modeSpeedBreaks,
      modeCount
    } = props;
    validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validateSinglePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    validateViewLength(id, 'timestamps', timestamps, positions.length);
    validateSinglePackedView(triplegs.tracks, ['uint32'], `${id} triplegs.tracks`);
    validateSinglePackedView(triplegs.startRows, ['uint32'], `${id} triplegs.startRows`);
    validateSinglePackedView(triplegs.endRows, ['uint32'], `${id} triplegs.endRows`);
    const capacity = triplegs.tracks.length;
    validateViewLength(id, 'triplegs.startRows', triplegs.startRows, capacity);
    validateViewLength(id, 'triplegs.endRows', triplegs.endRows, capacity);
    validateCapacity(id, 'modeCount', modeCount);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    const trackCount = trackOffsets.length - 1;
    validateSinglePackedView(output.modalSplit, ['float32'], `${id} output.modalSplit`);
    validateViewLength(id, 'output.modalSplit', output.modalSplit, trackCount * modeCount);
    validateRequirement(
      id,
      ['count', 'distance', 'duration'].includes(props.metric),
      "metric must be 'count', 'distance' or 'duration'"
    );
    if (modeSpeedBreaks) {
      validateRequirement(
        id,
        modeSpeedBreaks.every(value => Number.isFinite(Math.fround(value)) || value === Infinity),
        'modeSpeedBreaks must contain finite float32 values or positive Infinity'
      );
      validateRequirement(
        id,
        modeSpeedBreaks.length === modeCount,
        'modeSpeedBreaks must hold modeCount values'
      );
      validateRequirement(
        id,
        modeSpeedBreaks.every((value, index) => index === 0 || value > modeSpeedBreaks[index - 1]),
        'modeSpeedBreaks must be strictly ascending'
      );
      if (output.predictedModes) {
        validateSinglePackedView(output.predictedModes, ['uint32'], `${id} output.predictedModes`);
        validateViewLength(id, 'output.predictedModes', output.predictedModes, capacity);
      }
    } else {
      validateRequirement(id, Boolean(modes), 'modes or modeSpeedBreaks is required');
      validateSinglePackedView(modes!, ['uint32'], `${id} modes`);
      validateViewLength(id, 'modes', modes!, capacity);
    }
    validateTrajectoryOutputRanges(this.id, this.getUsedProps().output);
    if (modeSpeedBreaks) {
      validateGraphOutputsDisjointFromInputs(id, [output.modalSplit], [output.predictedModes]);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.modalSplit, modeSpeedBreaks ? output.predictedModes : undefined],
      [
        positions,
        timestamps,
        trackOffsets,
        triplegs.tracks,
        triplegs.startRows,
        triplegs.endRows,
        triplegs.status.count,
        modeSpeedBreaks ? undefined : modes
      ]
    );
  }

  private getUsedProps(): GPUModalSplitProps {
    const {props} = this;
    return {
      ...props,
      modes: props.modeSpeedBreaks ? undefined : props.modes,
      output: {
        ...props.output,
        predictedModes: props.modeSpeedBreaks ? props.output.predictedModes : undefined
      }
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, trackOffsets, triplegs, output, modeCount, modeSpeedBreaks} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.timestamps,
      trackOffsets,
      triplegs.tracks,
      triplegs.startRows,
      triplegs.endRows,
      triplegs.status.count,
      modeSpeedBreaks ? undefined : props.modes,
      output.modalSplit,
      modeSpeedBreaks ? output.predictedModes : undefined
    ]);
    const capacity = triplegs.tracks.length;
    const trackCount = trackOffsets.length - 1;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'output.modalSplit',
      trackCount * modeCount * 4,
      'reduce modeCount or the track count'
    );
    validateTrajectoryBindings(graph, this.id, this.getUsedProps());
    validateTrajectoryScratch(graph, id, {'tripleg track offsets': (trackCount + 1) * 4});
    const radius = getGPUGreatCircleRadius(this.spatialContext);
    const timeMode = getTrajectoryTimeMode(props);
    const tripTrackOffsets = createTransientView(
      graph,
      `${id}-tripleg-offsets`,
      'uint32',
      trackCount + 1
    );
    const modes = modeSpeedBreaks
      ? (output.predictedModes ?? createTransientView(graph, `${id}-modes`, 'uint32', capacity))
      : props.modes!;
    const nodes: GPUCommandNode<Parameters>[] = [
      createTrackTableOffsetsNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        tracks: triplegs.tracks,
        count: triplegs.status.count,
        offsets: tripTrackOffsets,
        capacity,
        trackCount
      })
    ];
    const distanceSource = `${radius === undefined ? '' : GEODESIC_WGSL}
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }
fn stepDistance(row: u32) -> f32 {
  return ${radius === undefined ? 'length(position(row) - position(row - 1u))' : `geodesicHaversineDistance(position(row - 1u), position(row), ${getWGSLFloatLiteral(radius)})`};
}
fn triplegLength(start: u32, end: u32) -> f32 {
  var summed = 0.0;
  for (var row = start + 1u; row < end; row++) { summed += stepDistance(row); }
  return summed;
}`;
    if (modeSpeedBreaks) {
      const breaks = modeSpeedBreaks
        .map(value => (Number.isFinite(value) ? getWGSLFloatLiteral(value) : '3.402823466e+38'))
        .join(', ');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-predict`,
          operation: OPERATION,
          variant: timeMode,
          invocationCount: capacity,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...getTrajectoryTimeBindings(props),
            {name: 'starts', view: triplegs.startRows, type: 'u32', access: 'read'},
            {name: 'ends', view: triplegs.endRows, type: 'u32', access: 'read'},
            {name: 'total', view: triplegs.status.count, type: 'u32', access: 'read'},
            {name: 'modes', view: modes, type: 'u32', access: 'read_write'}
          ],
          declarations: `${distanceSource}\n${getTrajectoryTimeSource(timeMode)}\n${IS_FINITE_VALUE_WGSL}\nconst SPEED_BREAKS = array<f32, ${modeCount}>(${breaks});\nconst MISSING_MODE: u32 = 0xffffffffu;`,
          body: `var mode = 0u;
  if (index < min(total[totalOffset], ${capacity}u)) {
    let start = starts[startsOffset + index];
    let end = ends[endsOffset + index];
    let duration = timeDifferenceRows(end - 1u, start);
    // No speed (no time passed, or a non-finite length): no class.
    mode = MISSING_MODE;
    if (duration > 0.0) {
      let speed = triplegLength(start, end) / duration;
      if (isFiniteValue(speed)) {
        // The last class has no upper bound: a speed at or above every break stays in it.
        mode = ${modeCount - 1}u;
        for (var candidate = 0u; candidate < ${modeCount}u; candidate++) {
          if (speed < SPEED_BREAKS[candidate]) { mode = candidate; break; }
        }
      }
    }
  }
  modes[modesOffset + index] = mode;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-aggregate`,
        operation: OPERATION,
        variant: `${props.metric}-${timeMode}`,
        invocationCount: trackCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...getTrajectoryTimeBindings(props),
          {name: 'starts', view: triplegs.startRows, type: 'u32', access: 'read'},
          {name: 'ends', view: triplegs.endRows, type: 'u32', access: 'read'},
          {name: 'modes', view: modes, type: 'u32', access: 'read'},
          {name: 'offsets', view: tripTrackOffsets, type: 'u32', access: 'read'},
          {name: 'modalSplit', view: output.modalSplit, type: 'f32', access: 'read_write'}
        ],
        declarations: `${distanceSource}\n${getTrajectoryTimeSource(timeMode)}`,
        body: `for (var mode = 0u; mode < ${modeCount}u; mode++) {
    modalSplit[modalSplitOffset + index * ${modeCount}u + mode] = 0.0;
  }
  for (var row = offsets[offsetsOffset + index]; row < offsets[offsetsOffset + index + 1u]; row++) {
    // The missing key 0xffffffff is in no class; any other out-of-range mode counts as the last.
    if (modes[modesOffset + row] == 0xffffffffu) { continue; }
    let mode = min(modes[modesOffset + row], ${modeCount - 1}u);
    let start = starts[startsOffset + row];
    let end = ends[endsOffset + row];
    let value = ${
      props.metric === 'count'
        ? '1.0'
        : props.metric === 'distance'
          ? 'triplegLength(start, end)'
          : 'timeDifferenceRows(end - 1u, start)'
    };
    modalSplit[modalSplitOffset + index * ${modeCount}u + mode] += value;
  }`
      })
    );
    return nodes;
  }
}
