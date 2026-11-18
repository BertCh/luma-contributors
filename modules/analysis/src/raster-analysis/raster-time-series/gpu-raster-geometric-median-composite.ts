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
import {validatePackedView} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesKernelNode,
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesBindings,
  validateRasterTimeSeriesViews,
  type RasterTimeSeriesField,
  type RasterTimeSeriesScalarView
} from './raster-time-series-chunks';
import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {getGPURasterGeometricMedianCompositeOutputLayout} from './raster-geometric-median-output-layout';

/** Slices per cell: each Weiszfeld iteration is one pass over the slices on one invocation. */
export const MAXIMUM_GEOMETRIC_MEDIAN_SLICE_COUNT = 256;
/** Bands per observation: the running estimate lives in a private array of this size. */
export const MAXIMUM_GEOMETRIC_MEDIAN_BAND_COUNT = 16;
/** Weiszfeld iterations are unrolled into the per-cell loop, so the count is capped. */
export const MAXIMUM_GEOMETRIC_MEDIAN_ITERATION_COUNT = 64;
/** Float32 elements of the per-frame parameters: distance epsilon. */
export const GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPURasterGeometricMedianComposite}. */
export type GPURasterGeometricMedianCompositeSettings = {
  /**
   * Smallest distance used as a Weiszfeld weight denominator, in value units: an estimate that lands
   * on an observation stays there instead of dividing by zero. Defaults to `1e-5`.
   */
  distanceEpsilon?: number;
};

/** Packs {@link GPURasterGeometricMedianCompositeSettings}: `[distanceEpsilon]`. */
export function getGPURasterGeometricMedianCompositeParameterValues(
  settings: GPURasterGeometricMedianCompositeSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH)
): Float32Array {
  const distanceEpsilon = settings.distanceEpsilon ?? 1e-5;
  validateFiniteNumber(
    'getGPURasterGeometricMedianCompositeParameterValues',
    'distanceEpsilon',
    distanceEpsilon,
    Number.MIN_VALUE,
    Infinity
  );
  target[0] = distanceEpsilon;
  return target;
}

/** Properties of {@link GPURasterGeometricMedianComposite}. */
export type GPURasterGeometricMedianCompositeProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  id?: string;
  cellCount: number;
  /** Slices in `[1, 256]`. */
  sliceCount: number;
  /** Bands per observation in `[1, 16]`. */
  bandCount: number;
  /** Weiszfeld iterations in `[1, 64]`; all of them run. */
  iterationCount: number;
  /** Observations `(cell * sliceCount + slice) * bandCount + band`; an observation with any non-finite band is missing. */
  values: Rows;
  /** Per-observation QA `cell * sliceCount + slice`; zero rejects the observation. */
  mask?: RasterTimeSeriesField<'uint32', Rows>;
  /** Float32, see {@link getGPURasterGeometricMedianCompositeParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** Geometric median `cell * bandCount + band`; NaN when status is nonzero. */
    composite: RasterTimeSeriesField<'float32', Rows>;
    /** Per cell: 0 composite, 1 no valid observation. */
    status: RasterTimeSeriesField<'uint32', Rows>;
    /** Optional per cell: Euclidean length of the last Weiszfeld step, a convergence indicator. */
    lastStep?: RasterTimeSeriesField<'float32', Rows>;
  };
};

/**
 * Geometric median composite per cell (Roberts, Mueller and McIntyre 2017): the point minimising the
 * sum of Euclidean distances to the valid observations over all bands together, so band
 * relationships are preserved and indices computed after compositing equal those computed before.
 * The output is synthetic; use {@link GPURasterMedoidComposite} when provenance matters.
 *
 * Weiszfeld's recursion `y <- sum(x_i / d_i) / sum(1 / d_i)` with `d_i = max(|x_i - y|, epsilon)`,
 * started at the mean of the valid observations, for exactly `iterationCount` iterations (no data
 * dependent exit). With few observations or a skewed cloud convergence can be slow: read `lastStep`.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `cellCount`, `sliceCount`,
 * `bandCount`, `iterationCount` and which optional views exist.
 */
export class GPURasterGeometricMedianComposite<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkProps: readonly GPURasterGeometricMedianCompositeProps<Rows>[];
  constructor(readonly props: GPURasterGeometricMedianCompositeProps<Rows>) {
    const id = props.id ?? 'raster-geometric-median-composite';
    this.id = id;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 1, MAXIMUM_GEOMETRIC_MEDIAN_SLICE_COUNT);
    validateCapacity(id, 'bandCount', props.bandCount, 1, MAXIMUM_GEOMETRIC_MEDIAN_BAND_COUNT);
    validateCapacity(
      id,
      'iterationCount',
      props.iterationCount,
      1,
      MAXIMUM_GEOMETRIC_MEDIAN_ITERATION_COUNT
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount * props.bandCount &&
        (!props.mask || props.mask.length >= observationCount),
      'values must hold cellCount times sliceCount times bandCount entries and mask cellCount times sliceCount'
    );
    const outputLayout = getGPURasterGeometricMedianCompositeOutputLayout({
      id,
      cellCount: props.cellCount,
      bandCount: props.bandCount,
      lastStep: props.output.lastStep !== undefined
    });
    for (const [name, descriptor] of Object.entries(outputLayout)) {
      const view = props.output[name as keyof typeof props.output]!;
      validateRequirement(
        id,
        view.length >= descriptor.length,
        `output.${name} must hold at least ${descriptor.length} entries`
      );
      for (const chunk of getGraphViewChunks(view)) {
        validatePackedView(chunk, [descriptor.format], `${id} output.${name}`);
      }
    }
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH} entries`
    );
    this.chunkProps = planRasterTimeSeriesChunks(id, props, {
      values: props.sliceCount * props.bandCount,
      mask: props.sliceCount,
      'output.composite': props.bandCount,
      'output.status': 1,
      'output.lastStep': 1
    });
    this.estimatedCommandNodeCount = this.chunkProps.length * 1;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
    const {values, mask, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterTimeSeriesScalarView[] {
    const {composite, status, lastStep} = this.props.output;
    return [composite, status, ...(lastStep ? [lastStep] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterGeometricMedianCompositeProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterGeometricMedianCompositeProps
  ): readonly GPUCommandNode<Parameters>[] {
    const id = props.id!;
    const {
      sliceCount: slices,
      bandCount: bands,
      iterationCount,
      values,
      mask,
      parameters,
      output
    } = props;
    return [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterGeometricMedianComposite',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          rasterBinding('composite', output.composite, true),
          rasterBinding('status', output.status, true),
          ...(output.lastStep ? [rasterBinding('lastStep', output.lastStep, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
fn isObservationValid(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  ${mask ? `if (mask[maskOffset + source] == 0u) { return false; }` : ''}
  for (var band = 0u; band < ${bands}u; band++) {
    if (!finite(values[valuesOffset + source * ${bands}u + band])) { return false; }
  }
  return true;
}`,
        body: `let distanceEpsilon = parameters[parametersOffset];
var estimate: array<f32, ${bands}>;
var numerator: array<f32, ${bands}>;
var validCount = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  if (!isObservationValid(index, slice)) { continue; }
  validCount++;
  for (var band = 0u; band < ${bands}u; band++) {
    estimate[band] += values[valuesOffset + (index * ${slices}u + slice) * ${bands}u + band];
  }
}
var step = 0.0;
if (validCount > 0u) {
  for (var band = 0u; band < ${bands}u; band++) { estimate[band] /= f32(validCount); }
  for (var iteration = 0u; iteration < ${iterationCount}u; iteration++) {
    var weightSum = 0.0;
    for (var band = 0u; band < ${bands}u; band++) { numerator[band] = 0.0; }
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (!isObservationValid(index, slice)) { continue; }
      let base = (index * ${slices}u + slice) * ${bands}u;
      var squared = 0.0;
      for (var band = 0u; band < ${bands}u; band++) {
        let difference = values[valuesOffset + base + band] - estimate[band];
        squared += difference * difference;
      }
      let weight = 1.0 / max(sqrt(squared), distanceEpsilon);
      weightSum += weight;
      for (var band = 0u; band < ${bands}u; band++) { numerator[band] += weight * values[valuesOffset + base + band]; }
    }
    var squaredStep = 0.0;
    for (var band = 0u; band < ${bands}u; band++) {
      let next = numerator[band] / weightSum;
      squaredStep += (next - estimate[band]) * (next - estimate[band]);
      estimate[band] = next;
    }
    step = sqrt(squaredStep);
  }
}
status[statusOffset + index] = select(0u, 1u, validCount == 0u);
${output.lastStep ? 'lastStep[lastStepOffset + index] = select(step, missing(), validCount == 0u);' : ''}
for (var band = 0u; band < ${bands}u; band++) {
  composite[compositeOffset + index * ${bands}u + band] = select(estimate[band], missing(), validCount == 0u);
}`
      })
    ];
  }
}
