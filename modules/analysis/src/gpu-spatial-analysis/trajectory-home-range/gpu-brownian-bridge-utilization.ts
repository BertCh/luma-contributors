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
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
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

const OPERATION = 'GPUBrownianBridgeUtilization';

/** Number of float32 elements in a {@link GPUBrownianBridgeUtilization} parameter view. */
export const GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH = 4;

/** Default number of midpoint-rule time steps per segment (the `time.step` of `move` is the shortest interval over 15). */
export const GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT = 15;

/** Largest accepted number of time steps per segment (a bounded per-thread loop). */
export const GPU_BROWNIAN_BRIDGE_MAXIMUM_SUB_STEP_COUNT = 256;

/**
 * Largest accepted fix count. The contributor is registered with `tier: 'reference'`: every grid
 * cell loops over all fixes (a segment test each, a Gaussian sum for nearby segments), so work is
 * `cellCount * fixCount`. A grid index over the segments would remove the loop; it is not built.
 */
export const GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT = 16384;

/**
 * Standard deviations beyond which a segment's bridge no longer adds to a cell. The Gaussian at
 * this distance is `exp(-24.5)`, about 2e-11 of its peak, so the omitted density is below float32
 * resolution of the cell total.
 */
export const GPU_BROWNIAN_BRIDGE_CUTOFF_SIGMA_COUNT = 7;

/** Per-frame settings of {@link GPUBrownianBridgeUtilization}. */
export type GPUBrownianBridgeUtilizationSettings = {
  /** Grid extent `[minX, minY, maxX, maxY]`; the cells are `gridSize` equal rectangles, row 0 at the smallest y. */
  extent: readonly [number, number, number, number];
};

/**
 * Packs the float parameters of {@link GPUBrownianBridgeUtilization}.
 *
 * Layout: `[minX, minY, maxX, maxY]` as float32.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUBrownianBridgeUtilizationParameterValues(
  settings: GPUBrownianBridgeUtilizationSettings,
  target: Float32Array = new Float32Array(GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} parameter target must hold ${GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH} elements`
    );
  }
  const [minX, minY, maxX, maxY] = settings.extent;
  for (const value of settings.extent) {
    validateFiniteNumber(OPERATION, 'extent', value);
  }
  validateRequirement(OPERATION, maxX > minX && maxY > minY, 'extent must have max > min');
  target.set([minX, minY, maxX, maxY]);
  return target;
}

/** Properties for {@link GPUBrownianBridgeUtilization}. */
export type GPUBrownianBridgeUtilizationProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'brownian-bridge-utilization'`. */
  id?: string;
  /** Planar only. */
  spatialContext?: GPUSpatialContext;
  /**
   * Brownian motion variance of the segment that starts at each fix (length squared per time unit
   * of the timestamps), one row per fix. The row of a track's last fix is not read. A constant
   * for the classic Brownian bridge (`estimateBrownianMotionVariance`), or per-segment values for
   * a dynamic one (`estimateWindowedBrownianMotionVariance`). Per-frame contents.
   */
  motionVariances: GraphDataView<'float32'>;
  /** Standard deviation of the location error of each fix (length unit), one row per fix. Per-frame contents. */
  locationErrors: GraphDataView<'float32'>;
  /** Grid size `[columns, rows]`. Compile-time. */
  gridSize: readonly [number, number];
  /** Per-frame parameters, at least 4 float32 rows, from {@link getGPUBrownianBridgeUtilizationParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Midpoint-rule steps per segment (1 to 256, default 15). Compile-time. */
  subStepCount?: number;
  output: {
    /**
     * Probability density at each cell centre, `columns * rows` rows (row-major, row 0 at the
     * smallest y), per unit of area. It integrates to 1 over the plane; the sum of
     * `density * cellArea` over the grid is below 1 by the mass outside the grid.
     */
    density: GraphDataView<'float32'>;
  };
};

/**
 * Brownian bridge utilization distribution of a track (Horne et al., Ecology 88, 2007; the
 * dynamic form of Kranstauber et al., 2012, differs only in the variance given per segment).
 *
 * Between fixes `a` and `b` at times `t_a` and `t_b` (`T = t_b - t_a`) the position at time
 * `t_a + alpha T` is bivariate normal with mean `a + alpha (b - a)` and variance
 * `T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta_a^2 + alpha^2 delta_b^2`, with `sigma^2` the
 * Brownian motion variance and `delta` the location errors. The density of a cell is the
 * time-weighted average over all segments of the time average of that normal density, evaluated at
 * the cell centre; the average over each segment is the midpoint rule with `subStepCount`
 * steps, where `move` uses a fixed time step. The mean over the
 * whole track is divided by the total duration of the segments.
 *
 * A segment is valid when its next fix is in the same track and the time increases. Segments
 * further than 7 standard deviations from a cell are skipped for that cell. A step with zero
 * variance (no motion variance and no location error) contributes nothing. The first and last
 * segments are used; `move` omits them by default.
 *
 * Planar only. Per-frame: `parameters`, `motionVariances`, `locationErrors`, positions and times.
 * Compile-time: the grid size, `subStepCount` and the view lengths. Reference tier, see
 * {@link GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT}.
 *
 * Reference: tests compare with a binary64 CPU implementation of the formulas above and check the
 * limit cases of a stationary animal and the mass inside the grid. The R package `move` was not
 * run.
 */
export class GPUBrownianBridgeUtilization implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of fixes. */
  readonly rowCount: number;
  /** Midpoint-rule steps per segment. */
  readonly subStepCount: number;
  /** Validated properties. */
  readonly props: GPUBrownianBridgeUtilizationProps;

  constructor(props: GPUBrownianBridgeUtilizationProps) {
    const id = props.id ?? 'brownian-bridge-utilization';
    resolveGPUSpatialContext(id, props.spatialContext, GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT);
    const {positions, timestamps, trackOffsets, gridSize, output} = props;
    validatePackedView(positions, ['float32x2'], `${id} positions`);
    validatePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedView(trackOffsets, ['uint32'], `${id} trackOffsets`);
    this.rowCount = positions.length;
    validateCapacity(id, 'fix count', this.rowCount, 2);
    validateMaximumCount(id, 'fix count', this.rowCount, GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT);
    validateViewLength(id, 'timestamps', timestamps, this.rowCount);
    resolveGPUTrajectoryTemporalContext(id, timestamps, props.temporalContext);
    validateRequirement(
      id,
      trackOffsets.length >= 2,
      'trackOffsets must contain at least two rows'
    );
    validatePackedView(props.motionVariances, ['float32'], `${id} motionVariances`);
    validateViewLength(id, 'motionVariances', props.motionVariances, this.rowCount);
    validatePackedView(props.locationErrors, ['float32'], `${id} locationErrors`);
    validateViewLength(id, 'locationErrors', props.locationErrors, this.rowCount);
    validateCapacity(id, 'gridSize columns', gridSize[0]);
    validateCapacity(id, 'gridSize rows', gridSize[1]);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH,
      `parameters must contain ${GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH} float32 rows`
    );
    this.subStepCount = props.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT;
    validateCapacity(
      id,
      'subStepCount',
      this.subStepCount,
      1,
      GPU_BROWNIAN_BRIDGE_MAXIMUM_SUB_STEP_COUNT
    );
    validatePackedView(output.density, ['float32'], `${id} output.density`);
    validateViewLength(id, 'output.density', output.density, gridSize[0] * gridSize[1]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.density],
      [
        positions,
        timestamps,
        trackOffsets,
        props.motionVariances,
        props.locationErrors,
        props.parameters
      ]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, subStepCount} = this;
    const {positions, timestamps, trackOffsets, gridSize, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      timestamps,
      trackOffsets,
      props.motionVariances,
      props.locationErrors,
      props.parameters,
      output.density
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'output.density',
      gridSize[0] * gridSize[1] * 4,
      'use a smaller grid'
    );
    const rowStarts = createTransientView(graph, `${id}-row-starts`, 'uint32', rowCount);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', rowCount);
    const rowTracks = createTransientView(graph, `${id}-row-tracks`, 'uint32', rowCount);
    const timeMode = getTrajectoryTimeMode(props);
    return [
      ...createRowBoundsNodes<Parameters>(graph, {
        id: `${id}-row-bounds`,
        operation: OPERATION,
        trackOffsets,
        rowCount,
        rowStarts,
        rowEnds,
        rowTracks
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-density`,
        operation: OPERATION,
        variant: timeMode,
        invocationCount: gridSize[0] * gridSize[1],
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...getTrajectoryTimeBindings(props),
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'motionVariances', view: props.motionVariances, type: 'f32', access: 'read'},
          {name: 'locationErrors', view: props.locationErrors, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'density', view: output.density, type: 'f32', access: 'read_write'}
        ],
        declarations: `${getTrajectoryTimeSource(timeMode)}
const COLUMNS: u32 = ${gridSize[0]}u;
const ROWS: u32 = ${gridSize[1]}u;
const FIX_COUNT: u32 = ${rowCount}u;
const SUB_STEPS: u32 = ${subStepCount}u;
const CUTOFF: f32 = ${GPU_BROWNIAN_BRIDGE_CUTOFF_SIGMA_COUNT}.0;
const TWO_PI: f32 = 6.283185307179586;
fn fixPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,
        body: /* wgsl */ `
  let minimum = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let maximum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let cellSize = (maximum - minimum) / vec2<f32>(f32(COLUMNS), f32(ROWS));
  let center = minimum + (vec2<f32>(f32(index % COLUMNS), f32(index / COLUMNS)) + vec2<f32>(0.5)) * cellSize;
  var weightedDensity = 0.0;
  var totalDuration = 0.0;
  for (var row = 0u; row + 1u < FIX_COUNT; row = row + 1u) {
    let trackEnd = rowEnds[rowEndsOffset + row];
    if (trackEnd == ${MOVEMENT_SENTINEL_WGSL} || row + 1u >= trackEnd) {
      continue;
    }
    let duration = timeDifferenceRows(row + 1u, row);
    if (!(duration > 0.0)) {
      continue;
    }
    totalDuration = totalDuration + duration;
    let pathStart = fixPosition(row);
    let pathEnd = fixPosition(row + 1u);
    let motion = max(motionVariances[motionVariancesOffset + row], 0.0);
    let errorFrom = locationErrors[locationErrorsOffset + row];
    let errorTo = locationErrors[locationErrorsOffset + row + 1u];
    let largestError = max(abs(errorFrom), abs(errorTo));
    let reach = CUTOFF * sqrt(0.25 * duration * motion + largestError * largestError);
    let lower = min(pathStart, pathEnd) - vec2<f32>(reach);
    let upper = max(pathStart, pathEnd) + vec2<f32>(reach);
    if (any(center < lower) || any(center > upper)) {
      continue;
    }
    var segmentDensity = 0.0;
    for (var step = 0u; step < SUB_STEPS; step = step + 1u) {
      let fraction = (f32(step) + 0.5) / f32(SUB_STEPS);
      let mean = pathStart + fraction * (pathEnd - pathStart);
      let variance = duration * fraction * (1.0 - fraction) * motion +
        (1.0 - fraction) * (1.0 - fraction) * errorFrom * errorFrom +
        fraction * fraction * errorTo * errorTo;
      if (variance > 0.0) {
        let offset = center - mean;
        segmentDensity = segmentDensity + exp(-dot(offset, offset) / (2.0 * variance)) / (TWO_PI * variance);
      }
    }
    weightedDensity = weightedDensity + duration * segmentDensity / f32(SUB_STEPS);
  }
  density[densityOffset + index] = select(0.0, weightedDensity / totalDuration, totalDuration > 0.0);`
      })
    ];
  }
}
