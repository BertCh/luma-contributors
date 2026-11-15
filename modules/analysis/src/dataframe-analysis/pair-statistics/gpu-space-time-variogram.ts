// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validatePairHistogramShape} from './pair-histogram';
import {validatePairStatisticsInputs} from './pair-statistics-grid';
import {getEmpiricalVariogramNodes} from './empirical-variogram-nodes';
import {
  validatePairStatisticsBounds,
  type GPUPairStatisticsBounds
} from './pair-statistics-parameters';

const OPERATION = 'GPUSpaceTimeVariogram';

/** Number of float32 elements in a `GPUSpaceTimeVariogram` parameter buffer. */
export const GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH = 8;

/** Per-frame parameters of {@link GPUSpaceTimeVariogram}. */
export type GPUSpaceTimeVariogramSettings = {
  /** Inclusive `[minX, minY, maxX, maxY]` extent; rows outside it are excluded. */
  bounds: Exclude<GPUPairStatisticsBounds, 'study-window'>;
  /** Largest spatial lag, in the units of the positions. */
  maximumDistance: number;
  /** Largest time lag, in the unit of the relative `times`. */
  maximumTimeLag: number;
};

/**
 * Packs the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, maximumTimeLag, 0, 0]`.
 *
 * @throws If a value is not finite, the bounds are inverted, or a maximum lag is not positive.
 */
export function getGPUSpaceTimeVariogramParameterValues(
  settings: GPUSpaceTimeVariogramSettings,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} target must hold ${GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds(OPERATION, settings.bounds, settings.maximumDistance);
  if (!Number.isFinite(settings.maximumTimeLag) || settings.maximumTimeLag <= 0) {
    throw new Error(`${OPERATION} maximumTimeLag must be positive and finite`);
  }
  target.set([...settings.bounds, settings.maximumDistance, settings.maximumTimeLag, 0, 0]);
  return target;
}

/** Properties of {@link GPUSpaceTimeVariogram}. */
export type GPUSpaceTimeVariogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  /** Defaults to values, giving the ordinary Matheron space-time semivariogram. */
  secondValues?: GraphDataView<'float32'>;
  /**
   * Event times relative to a float64 origin chosen on the CPU, in one unit. Never upload raw
   * epoch seconds or milliseconds: float32 spaces them 128 s (or about 2 minutes) apart. Build
   * them with `getGPUSpaceTimeRelativeTimes` from `gpu-spatial-analysis`, which also reports the
   * remaining float32 resolution.
   */
  times: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** Written with {@link getGPUSpaceTimeVariogramParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the neighbor-search lattice; affects speed, not results. */
  gridSize: readonly [number, number];
  /** Positive spatial classes over `(0, maximumDistance]`, in addition to the zero-lag class. */
  lagCount: number;
  /** Positive temporal classes over `(0, maximumTimeLag]`, in addition to the zero-lag class. */
  temporalLagCount: number;
  output: {
    /**
     * `(lagCount + 1) * (temporalLagCount + 1)` bins, spatial-major:
     * `spatialClass * (temporalLagCount + 1) + temporalClass`. Class 0 of each axis is the exact
     * zero lag, so bin 0 holds coincident points at the same time, the first row the purely
     * temporal variogram and the first column the purely spatial one.
     */
    semivariances: GraphDataView<'float32'>;
    pairCounts: GraphDataView<'uint32'>;
    /**
     * Per bin: 0 valid, 1 empty or invalid parameters, 2 nonfinite arithmetic, 3 the exact
     * integer accumulation overflowed.
     */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Empirical space-time semivariogram on the indexed pair histogram: signed
 * `sum (x_i-x_j)(y_i-y_j) / (2 N)` per bin, equal to the Matheron semivariogram when
 * `secondValues` is omitted. Each unordered pair of included rows with `d <= maximumDistance` and
 * `|t_i - t_j| <= maximumTimeLag` is counted once.
 *
 * Classes: class 0 of each axis holds exactly zero lag (identical float32 coordinates, identical
 * float32 relative times); positive lags fall in `1 + min(floor(lag / maximum * count),
 * count - 1)`, left-closed and right-open with the final edge included. A row is included when its
 * mask is nonzero and its coordinates, values and time are finite, with the position inside the
 * parameter bounds. Terms are accumulated as exact integers (positive and negative parts apart),
 * so results are reproducible; the fixed-point unit is the typical cross term, as in
 * `GPUVariogram`. Cost is the number of pairs in neighboring lattice cells, not `rows^2`.
 */
export class GPUSpaceTimeVariogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSpaceTimeVariogramProps) {
    this.id = props.id ?? 'space-time-variogram';
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH
    });
    validateCapacity(id, 'lagCount', props.lagCount);
    validateCapacity(id, 'temporalLagCount', props.temporalLagCount);
    // Count, positive and negative sums, and the large-term channel of each.
    validatePairHistogramShape(id, getBinCount(props), 5);
    for (const view of [props.secondValues, props.times])
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} input`);
        if (view.length !== rows) throw new Error(`${id} values must match positions`);
      }
    const bins = getBinCount(props);
    for (const [name, view] of Object.entries(props.output)) {
      validateSinglePackedView(
        view,
        [name === 'semivariances' ? 'float32' : 'uint32'],
        `${id} ${name}`
      );
      if (view.length < bins) throw new Error(`${id} output.${name} must hold ${bins} bins`);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.positions,
      props.values,
      props.secondValues,
      props.times,
      props.mask,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getEmpiricalVariogramNodes(graph, this.id, this.props);
  }
}

function getBinCount(props: {lagCount: number; temporalLagCount: number}): number {
  return (props.lagCount + 1) * (props.temporalLagCount + 1);
}
