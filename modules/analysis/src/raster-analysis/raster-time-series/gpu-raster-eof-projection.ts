// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GraphVectorView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesKernelNode,
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesBindings,
  validateRasterTimeSeriesViews,
  type RasterTimeSeriesField,
  type RasterTimeSeriesScalarView
} from './raster-time-series-chunks';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH,
  MAXIMUM_SNAPSHOT_COVARIANCE_SLICE_COUNT
} from './gpu-raster-snapshot-covariance';

/** Modes per cell of the output EOF maps. */
export const MAXIMUM_EOF_PROJECTION_MODE_COUNT = 32;

/** Properties of {@link GPURasterEOFProjection}. */
export type GPURasterEOFProjectionProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  modeCount: number;
  /** The same field given to `GPURasterSnapshotCovariance`, cell-major. */
  values: Rows;
  mask?: RasterTimeSeriesField<'uint32', Rows>;
  /** The same area weights given to `GPURasterSnapshotCovariance`. */
  weights?: RasterTimeSeriesField<'float32', Rows>;
  /** The parameters view of `GPURasterSnapshotCovariance`. */
  parameters: GraphDataView<'float32'>;
  /** Mode-major `mode * sliceCount + slice` temporal modes from the CPU eigendecomposition. */
  temporalModes: GraphDataView<'float32'>;
  /** `modeCount` singular values from the CPU eigendecomposition. */
  singularValues: GraphDataView<'float32'>;
  /**
   * `'weighted'` (default): the left singular vectors of the weighted matrix, orthonormal under the
   * plain dot product. `'value'`: divided by `sqrt(weight)`, orthonormal under the area-weighted
   * dot product, in the field's own space.
   */
  componentSpace?: 'weighted' | 'value';
  output: {
    /** Cell-major `cell * modeCount + mode`; NaN for an excluded cell or a mode with a zero singular value. */
    eofs: RasterTimeSeriesField<'float32', Rows>;
    /** Per cell: 0 projected, 1 excluded (missing sample, out of range or invalid weight). */
    status: RasterTimeSeriesField<'uint32', Rows>;
  };
};

/**
 * Projects a field onto the temporal modes of the method of snapshots, giving spatial EOF maps:
 * `u[cell, mode] = sqrt(weight) * sum_t((x[cell, t] - mean[cell]) * v[t, mode]) / s[mode]`. Its
 * inputs come from `getGPURasterSnapshotEOFDecomposition`, the CPU eigen step after readback.
 * A spatial EOF has the sign of its temporal mode, so the sign convention is fixed on the CPU.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: `parameters`, `temporalModes`, `singularValues` (buffers the caller updates).
 * Compile-time: `cellCount`, `sliceCount`, `modeCount`, `componentSpace`, optional views.
 */
export class GPURasterEOFProjection<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkProps: readonly GPURasterEOFProjectionProps<Rows>[];
  readonly componentSpace: 'weighted' | 'value';

  constructor(readonly props: GPURasterEOFProjectionProps<Rows>) {
    const id = props.id ?? 'raster-eof-projection';
    this.id = id;
    this.componentSpace = props.componentSpace ?? 'weighted';
    const {cellCount, sliceCount, modeCount} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'sliceCount', sliceCount, 2, MAXIMUM_SNAPSHOT_COVARIANCE_SLICE_COUNT);
    validateCapacity(id, 'modeCount', modeCount, 1, MAXIMUM_EOF_PROJECTION_MODE_COUNT);
    validateRequirement(id, modeCount <= sliceCount, 'modeCount must be at most sliceCount');
    validateRequirement(
      id,
      this.componentSpace === 'weighted' || this.componentSpace === 'value',
      "componentSpace must be 'weighted' or 'value'"
    );
    const observations = cellCount * sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observations);
    validateRequirement(
      id,
      [props.values, props.mask].every(view => !view || view.length >= observations),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      !props.weights || props.weights.length >= cellCount,
      'weights must hold one entry per cell'
    );
    validateRequirement(
      id,
      props.temporalModes.length >= modeCount * sliceCount,
      'temporalModes must hold modeCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.singularValues.length >= modeCount,
      'singularValues must hold modeCount entries'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      props.output.eofs.length >= cellCount * modeCount,
      'output.eofs must hold modeCount entries per cell'
    );
    validateRequirement(
      id,
      props.output.status.length >= cellCount,
      'output.status must hold one entry per cell'
    );
    this.chunkProps = planRasterTimeSeriesChunks(id, props, {
      values: props.sliceCount,
      mask: props.sliceCount,
      weights: 1,
      'output.eofs': props.modeCount,
      'output.status': 1
    });
    this.estimatedCommandNodeCount = this.chunkProps.length * 1;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
    const {values, mask, weights, parameters, temporalModes, singularValues} = this.props;
    return [
      values,
      parameters,
      temporalModes,
      singularValues,
      ...(mask ? [mask] : []),
      ...(weights ? [weights] : [])
    ];
  }

  private getOutputs(): RasterTimeSeriesScalarView[] {
    return [this.props.output.eofs, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterEOFProjectionProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterEOFProjectionProps
  ): readonly GPUCommandNode<Parameters>[] {
    const {componentSpace} = this;
    const id = props.id!;
    const {sliceCount: slices, modeCount: modes, values, mask, weights, parameters, output} = props;
    const scaleExpression =
      componentSpace === 'weighted' && weights ? 'sqrt(weights[weightsOffset + index])' : '1.0';
    return [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEOFProjection',
        invocationCount: props.cellCount,
        workgroupSize: 16,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          rasterBinding('temporalModes', props.temporalModes),
          rasterBinding('singularValues', props.singularValues),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          ...(weights ? [rasterBinding('weights', weights)] : []),
          rasterBinding('eofs', output.eofs, true),
          rasterBinding('status', output.status, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `let limit = parameters[parametersOffset + 1u];
let weight = ${weights ? 'weights[weightsOffset + index]' : '1.0'};
var kept = weight > 0.0 && weight <= 1.0;
var total = 0.0;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) || abs(value) > limit ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { kept = false; break; }
  total += value;
}
let mean = total / ${slices}.0;
for (var mode = 0u; mode < ${modes}u; mode++) {
  let singular = singularValues[singularValuesOffset + mode];
  var projected = missing();
  if (kept && singular > 0.0) {
    var sum = 0.0;
    for (var slice = 0u; slice < ${slices}u; slice++) { sum += (values[valuesOffset + index * ${slices}u + slice] - mean) * temporalModes[temporalModesOffset + mode * ${slices}u + slice]; }
    projected = (${scaleExpression}) * sum / singular;
  }
  eofs[eofsOffset + index * ${modes}u + mode] = projected;
}
status[statusOffset + index] = select(1u, 0u, kept);`
      })
    ];
  }
}
