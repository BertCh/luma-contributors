// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MODAL_SPEED_COMPARISON_WGSL} from './modal-speed-comparison';
import {
  resolveModalSpeedOptions,
  validateModalSpeedOptionName,
  type ModalSpeedOptions
} from './modal-split-options';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
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
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
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
  status: {count: GPUStatusScalar; overflow?: GPUStatusScalar; candidateOverflow?: GPUStatusScalar};
};

/**
 * Properties for {@link GPUModalSplit}: the trajectory layout plus the tripleg table, the tripleg
 * modes, and the metric. `modes` holds one row per tripleg capacity row, values in
 * `[0, modeCount)`; when `modeSpeedUpperBoundsFloat32` is given, modes are predicted from average tripleg
 * speed instead (trackintel `simple-coarse`) and written to `output.predictedModes`.
 */
export type GPUModalSplitProps = GPUTrajectoryGeometryFields &
  ModalSpeedOptions & {
    /** Prefix for node and transient IDs. Defaults to `'modal-split'`. */
    id?: string;
    /** Planar or longitude/latitude great-circle; distances in the context's length unit. */
    spatialContext?: GPUSpatialContext;
    triplegs: GPUModalSplitTriplegs;
    /**
     * Mode of each tripleg, unless `modeSpeedUpperBoundsFloat32` is given. A tripleg whose mode is the missing
     * key `0xffffffff` is left out of every class; any other value at or above `modeCount` counts
     * as the last class.
     */
    modes?: GraphDataView<'uint32'>;
    /** Number of modes; with `modeSpeedUpperBoundsFloat32`, its length. */
    modeCount: number;
    /** Aggregation per (track, mode). Compile-time. */
    metric: GPUModalSplitMetric;
    output: {
      /** Per-(track, mode) metric; all NaN when supplied tripleg loss is nonzero. */
      modalSplit: GraphDataView<'float32'>;
      /**
       * Optional predicted mode per tripleg; supplied-mode runs write missing keys. Prediction writes `0xffffffff` for a tripleg
       * with no speed. With complete input, rows at and after the tripleg count are 0.
       * If either supplied tripleg loss flag is nonzero, every capacity row is `0xffffffff`.
       */
      predictedModes?: GraphDataView<'uint32'>;
    };
  };

/**
 * Modal split of triplegs (trackintel `calculate_modal_split`): one thread per track accumulates
 * count, summed haversine or Euclidean length, or summed duration per (track, mode). With
 * `modeSpeedUpperBoundsFloat32`, the mode comes from trackintel's `simple-coarse` speed classes. Each step
 * length is encoded as binary32 and path lengths are summed in row order using nearest-even
 * binary32 addition. Exclusive classification compares length < duration * bound exactly,
 * without dividing and rounding a quotient.
 */
export class GPUModalSplit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUModalSplitProps;
  readonly spatialContext: GPUSpatialContext;

  constructor(input: GPUModalSplitProps) {
    let props = snapshotConstructionFields(input, [
      'id',
      'positions',
      'timestamps',
      'temporalContext',
      'trackOffsets',
      'spatialContext',
      'triplegs',
      'modes',
      'modeSpeedUpperBoundsFloat32',
      'unknownModePolicy',
      'modeCount',
      'metric',
      'output'
    ]);
    validateModalSpeedOptionName(props.id ?? 'modal-split', input);
    const triplegSnapshot = snapshotConstructionFields(props.triplegs, [
      'tracks',
      'startRows',
      'endRows',
      'status'
    ]);
    const speedBounds = resolveModalSpeedOptions(props.id ?? 'modal-split', props);
    props = {
      ...props,
      triplegs: {
        ...triplegSnapshot,
        status: snapshotConstructionFields(triplegSnapshot.status, [
          'count',
          'overflow',
          'candidateOverflow'
        ])
      },
      output: snapshotConstructionFields(props.output, ['modalSplit', 'predictedModes']),
      modeSpeedUpperBoundsFloat32: speedBounds
    };
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
      modeSpeedUpperBoundsFloat32,
      modeCount
    } = props;
    for (const [name, view] of Object.entries(triplegs.status)) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} triplegs.status.${name}`);
      validateRequirement(
        id,
        view.length >= 1,
        `triplegs.status.${name} must contain at least one row`
      );
    }
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
      trackOffsets.length >= 1,
      'trackOffsets must contain at least one terminal row'
    );
    const trackCount = trackOffsets.length - 1;
    validateSinglePackedView(output.modalSplit, ['float32'], `${id} output.modalSplit`);
    validateViewLength(id, 'output.modalSplit', output.modalSplit, trackCount * modeCount);
    validateRequirement(
      id,
      ['count', 'distance', 'duration'].includes(props.metric),
      "metric must be 'count', 'distance' or 'duration'"
    );
    if (!modeSpeedUpperBoundsFloat32) {
      validateRequirement(id, Boolean(modes), 'modes or modeSpeedUpperBoundsFloat32 is required');
      validateSinglePackedView(modes!, ['uint32'], `${id} modes`);
      validateViewLength(id, 'modes', modes!, capacity);
    }
    if (output.predictedModes) {
      validateSinglePackedView(output.predictedModes, ['uint32'], `${id} output.predictedModes`);
      validateViewLength(id, 'output.predictedModes', output.predictedModes, capacity);
    }
    validateTrajectoryOutputRanges(this.id, this.getUsedProps().output);
    if (modeSpeedUpperBoundsFloat32) {
      validateGraphOutputsDisjointFromInputs(id, [output.modalSplit], [output.predictedModes]);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.modalSplit, output.predictedModes],
      [
        positions,
        timestamps,
        trackOffsets,
        triplegs.tracks,
        triplegs.startRows,
        triplegs.endRows,
        triplegs.status.count,
        triplegs.status.overflow,
        triplegs.status.candidateOverflow,
        modeSpeedUpperBoundsFloat32 ? undefined : modes
      ]
    );
  }

  private getUsedProps(): GPUModalSplitProps {
    const {props} = this;
    return {
      ...props,
      modes: props.modeSpeedUpperBoundsFloat32 ? undefined : props.modes,
      output: {
        ...props.output,
        predictedModes: props.output.predictedModes
      }
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, trackOffsets, triplegs, output, modeCount, modeSpeedUpperBoundsFloat32} =
      props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.timestamps,
      trackOffsets,
      triplegs.tracks,
      triplegs.startRows,
      triplegs.endRows,
      triplegs.status.count,
      triplegs.status.overflow,
      triplegs.status.candidateOverflow,
      modeSpeedUpperBoundsFloat32 ? undefined : props.modes,
      output.modalSplit,
      output.predictedModes
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
    const lossBindings = [triplegs.status.overflow, triplegs.status.candidateOverflow].flatMap(
      (view, index) =>
        view
          ? [{name: `triplegLoss${index}`, view, type: 'u32' as const, access: 'read' as const}]
          : []
    );
    const hasLoss =
      lossBindings.map(binding => `${binding.name}[${binding.name}Offset] != 0u`).join(' || ') ||
      'false';
    const timeMode = getTrajectoryTimeMode(props);
    const tripTrackOffsets = createTransientView(
      graph,
      `${id}-tripleg-offsets`,
      'uint32',
      trackCount + 1
    );
    const modes = modeSpeedUpperBoundsFloat32
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
    const distanceSource = `${MODAL_SPEED_COMPARISON_WGSL}\n${radius === undefined ? '' : GEODESIC_WGSL}
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }
fn stepDistance(row: u32) -> f32 {
  return ${radius === undefined ? 'length(position(row) - position(row - 1u))' : `geodesicHaversineDistance(position(row - 1u), position(row), ${getWGSLFloatLiteral(radius)})`};
}
fn triplegLength(start: u32, end: u32) -> f32 {
  var summed = 0.0;
  for (var row = start + 1u; row < end; row++) { summed = modalAddLength(summed, stepDistance(row)); }
  return summed;
}`;
    if (modeSpeedUpperBoundsFloat32) {
      const breaks = modeSpeedUpperBoundsFloat32
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
    if (duration > 0.0 && isFiniteValue(duration)) {
      let pathLength = triplegLength(start, end);
      if (isFiniteValue(pathLength)) {
        // The last class has no upper bound: a speed at or above every break stays in it.
        mode = ${modeCount - 1}u;
        for (var candidate = 0u; candidate < ${modeCount}u; candidate++) {
          if (modalSpeedBelowBound(pathLength, duration, SPEED_BREAKS[candidate])) { mode = candidate; break; }
        }
      }
    }
  }
  modes[modesOffset + index] = mode;`
        })
      );
    }
    if (!modeSpeedUpperBoundsFloat32 && output.predictedModes) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-no-predictions`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'predicted', view: output.predictedModes, type: 'u32', access: 'read_write'}
          ],
          body: 'predicted[predictedOffset + index] = 0xffffffffu;'
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
    if (modes[modesOffset + row] == 0xffffffffu${props.unknownModePolicy === 'exclude' ? ` || modes[modesOffset + row] >= ${modeCount}u` : ''}) { continue; }
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
    if (lossBindings.length)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-suppress-incomplete`,
          operation: OPERATION,
          invocationCount: Math.max(
            output.modalSplit.length,
            modeSpeedUpperBoundsFloat32 ? modes.length : 0
          ),
          bindings: [
            ...lossBindings,
            {name: 'modalSplit', view: output.modalSplit, type: 'f32', access: 'read_write'},
            ...(modeSpeedUpperBoundsFloat32
              ? [{name: 'modes', view: modes, type: 'u32' as const, access: 'read_write' as const}]
              : [])
          ],
          declarations: GET_NAN_WGSL,
          body: `if (${hasLoss}) {
    if (index < ${output.modalSplit.length}u) { modalSplit[modalSplitOffset + index] = getNaN(); }
    ${modeSpeedUpperBoundsFloat32 ? `if (index < ${modes.length}u) { modes[modesOffset + index] = 0xffffffffu; }` : ''}
  }`
        })
      );
    return nodes;
  }
}
