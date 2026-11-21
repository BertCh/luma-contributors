// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateWeightsLossSources,
  createWeightsLossNodes
} from '../spatial-weights/spatial-weights-loss';
import type {InterpolationNeighborhoodWeights} from './interpolation-neighborhood-loss';

import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';

import {
  GPUFFT2D,
  GPUReduction,
  makeGPUFFT2DStats,
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import type {GPUApproximationStatusPort} from '../../analysis-core/contracts/status';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getGraphOutputViews,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH} from './gaussian-random-field-parameters';
import {
  getPerFramePredictionCovarianceWGSL,
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR
} from './prediction-covariance';
import {getRasterGridCorners} from './spatial-interpolation-parameters';
import {GPULocalKriging} from './gpu-local-kriging';

/** Twice this field side fits the published 2048-side FFT embedding. */
export const GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE = 1024;
/** One packed float32x2 coordinate binding fits 128 MiB. */
export const GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_ROW_COUNT = 2 ** 24;
/** Bounded serial mode sum per point, with a separate total-work limit. */
export const GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_MODE_COUNT = 4096;
/** Bounds field evaluation to about one billion mode-point products per run. */
export const GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_SPECTRAL_WORK = 2 ** 30;
/** Supplied candidate list bounds each residual solve to the local-kriging maximum. */
export const GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_NEIGHBOR_COUNT = 32;

export type GPUGaussianRandomFieldOutput = {
  /** One realization, row-major for a grid. Extra output rows are cleared to NaN. */
  values: GraphDataView<'float32'>;
  /** Nonzero for finite usable rows; padding and failed residual solves are zero. */
  validity: GraphDataView<'uint32'>;
  /** Scalar: finite modes, local conditioning, or a clipped FFT spectrum was used. */
  status: GPUApproximationStatusPort & {
    approximation: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
  /** Scalar sum of max(-eigenvalue,0) / padded cell count, in variance units; zero for modes. */
  clippedEigenvalueMass: GraphDataView<'float32'>;
};

export type GPUGaussianRandomFieldProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  /** Axis-aligned planar output lattice; supply exactly one of grid or positions. */
  grid?: GPURasterMetadata;
  /** Arbitrary planar points. Coordinates and parameters change per frame. */
  positions?: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  /** Auto uses FFT for grids up to 1024 per side and spectral modes otherwise. */
  method?: 'auto' | 'fft' | 'spectral';
  /** Compile-time finite-mode count, default 256. Not used by FFT. */
  modeCount?: number;
  /**
   * Hard observations on rows of the output domain. Supply candidate lists from a resident
   * neighbor index, in observation-row numbering, with distinct candidates and 0xffffffff padding.
   * Every observed query must include itself. Observed indices must be distinct.
   * This avoids an exhaustive scan or dense joint covariance and has no 4096-row cap.
   */
  conditioning?: {
    indices: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
    neighborhoods: GraphDataView<'uint32'>;
    /** Optional CSR offsets; omit for fixed-width candidate rows. */
    neighborhoodOffsets?: GraphDataView<'uint32'>;
    /** Live search loss; incomplete support suppresses all conditioned output rows. */
    status?: InterpolationNeighborhoodWeights['status'];
    neighborCount: number;
  };
  output: GPUGaussianRandomFieldOutput;
} & (
  | {grid: GPURasterMetadata; positions?: never}
  | {positions: GraphDataView<'float32x2'>; grid?: never}
);

function binding(name: string, view: GraphDataView, write = false): WGSLKernelBinding {
  return {
    name,
    view,
    type: view.format === 'uint32' ? 'u32' : 'f32',
    access: write ? 'read_write' : 'read'
  };
}

const RANDOM_WGSL = `${PHILOX_WGSL}
fn randomBlock(row: u32, tag: u32) -> vec4<u32> {
  return philox4x32(vec4<u32>(row, bitcast<u32>(params[paramsOffset + 10u]), tag, 0u),
    vec2<u32>(bitcast<u32>(params[paramsOffset + 9u]), ${PHILOX_PURPOSES.gaussianRandomField.id}u));
}
fn normalPair(words: vec4<u32>) -> vec2<f32> {
  let radius = sqrt(-2.0 * log(philoxOpenUnitFloat(words.x)));
  let angle = 6.283185307179586 * philoxUnitFloat(words.y);
  return radius * vec2<f32>(cos(angle), sin(angle));
}`;

/**
 * Stationary isotropic Gaussian fields in a native planar frame, without CPU readback.
 * FFT: covariance on a doubled radix-two torus, complex normal spectrum scaled by sqrt(N lambda),
 * then the real part of the published normalized inverse FFT. Negative eigenvalues are clipped
 * and their variance mass is published, never silently called exact.
 * Spectral: Heße et al. randomization with independent Gaussian sine/cosine amplitudes and
 * analytic two-dimensional radial inverse CDFs (Gaussian, exponential, Matern 3/2 and 5/2).
 * Spherical covariance is supported by FFT only; spectral rows report invalid for that model.
 * Zero partial sill skips spectral frequency generation and summation. A positive-sill draw with
 * an unrepresentable frequency or phase reports NaN and zero validity instead of clipping modes.
 * Nugget is independent per domain row (duplicate point rows are separate microscale samples).
 * Conditioning adds GPULocalKriging of observed-minus-simulated residuals, using caller-indexed
 * candidates. Include every exact observation in its query neighborhood and use distinct sites.
 * Generated grid positions stay relative to the first cell center, including during conditioning,
 * so large world origins cannot round away cell spacing before spectral phase evaluation.
 * Compile-time: grid dimensions, counts, method, modeCount, conditioning topology, spatialContext.
 * Per-frame: all view contents, mean, covariance, seed, realization, grid affine / phase origin.
 * No iterative fit or CPU twin: these are large data-parallel field evaluations.
 */
export class GPUGaussianRandomField implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rowCount: number;
  readonly method: 'fft' | 'spectral';
  readonly modeCount: number;
  readonly estimatedCommandNodeCount: number;
  private readonly paddedWidth: number;
  private readonly paddedHeight: number;

  readonly props: GPUGaussianRandomFieldProps;
  constructor(props: GPUGaussianRandomFieldProps) {
    this.props = props = snapshotInterpolationConstructionProps(
      props,
      [
        'conditioning',
        'grid',
        'id',
        'method',
        'modeCount',
        'output',
        'parameters',
        'positions',
        'spatialContext'
      ],
      ['values', 'validity', 'status', 'clippedEigenvalueMass'],
      ['approximation', 'candidateOverflow']
    );
    this.id = props.id ?? 'GPUGaussianRandomField';
    resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    validateRequirement(
      this.id,
      Boolean(props.grid) !== Boolean(props.positions),
      'grid or positions must be supplied exclusively'
    );
    validateRequirement(
      this.id,
      ['auto', 'fft', 'spectral'].includes(props.method ?? 'auto'),
      'method must be auto, fft or spectral'
    );
    if (props.grid) getRasterGridCorners(this.id, props.grid, false);
    this.rowCount = props.grid ? props.grid.width * props.grid.height : props.positions!.length;
    validateCapacity(
      this.id,
      'rowCount',
      this.rowCount,
      0,
      GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_ROW_COUNT
    );
    this.method =
      props.method === 'auto' || !props.method
        ? props.grid &&
          props.grid.width <= GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE &&
          props.grid.height <= GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE
          ? 'fft'
          : 'spectral'
        : props.method;
    validateRequirement(
      this.id,
      this.method !== 'fft' || Boolean(props.grid),
      'method fft requires grid'
    );
    if (this.method === 'fft') {
      validateCapacity(
        this.id,
        'grid.width',
        props.grid!.width,
        1,
        GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE
      );
      validateCapacity(
        this.id,
        'grid.height',
        props.grid!.height,
        1,
        GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE
      );
    }
    this.paddedWidth = 2 ** Math.ceil(Math.log2(Math.max(2, (props.grid?.width ?? 1) * 2)));
    this.paddedHeight = 2 ** Math.ceil(Math.log2(Math.max(2, (props.grid?.height ?? 1) * 2)));
    this.modeCount = props.modeCount ?? 256;
    validateCapacity(
      this.id,
      'modeCount',
      this.modeCount,
      1,
      GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_MODE_COUNT
    );
    if (this.method === 'spectral')
      validateCapacity(
        this.id,
        'rowCount * modeCount',
        this.rowCount * this.modeCount,
        0,
        GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_SPECTRAL_WORK
      );
    const inputs = [
      props.parameters,
      props.positions,
      ...(props.conditioning
        ? [
            props.conditioning.indices,
            props.conditioning.values,
            props.conditioning.mask,
            props.conditioning.neighborhoods,
            props.conditioning.neighborhoodOffsets,
            ...validateWeightsLossSources(this.id, [props.conditioning])
          ]
        : [])
    ].filter(view => view !== undefined) as GraphDataView[];
    const outputs = getGraphOutputViews(props.output);
    for (const [name, view, format] of [
      ['parameters', props.parameters, 'float32'],
      ['positions', props.positions, 'float32x2'],
      ['conditioning.indices', props.conditioning?.indices, 'uint32'],
      ['conditioning.values', props.conditioning?.values, 'float32'],
      ['conditioning.mask', props.conditioning?.mask, 'uint32'],
      ['conditioning.neighborhoods', props.conditioning?.neighborhoods, 'uint32'],
      ['conditioning.neighborhoodOffsets', props.conditioning?.neighborhoodOffsets, 'uint32'],
      ['output.status.candidateOverflow', props.output.status.candidateOverflow, 'uint32'],
      ['output.values', props.output.values, 'float32'],
      ['output.validity', props.output.validity, 'uint32'],
      ['output.status.approximation', props.output.status.approximation, 'uint32'],
      ['output.clippedEigenvalueMass', props.output.clippedEigenvalueMass, 'float32']
    ] as const) {
      if (view) validateSinglePackedView(view, [format], `${this.id} ${name}`);
    }
    validateViewLength(
      this.id,
      'parameters',
      props.parameters,
      GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH
    );
    validateRequirement(
      this.id,
      props.output.values.length >= this.rowCount,
      'output.values length must be at least rowCount'
    );
    validateRequirement(
      this.id,
      props.output.validity.length === props.output.values.length,
      'output.validity length must equal output.values length'
    );
    validateRequirement(
      this.id,
      props.output.status.approximation.length === 1 &&
        props.output.clippedEigenvalueMass.length === 1,
      'output scalar lengths must equal 1'
    );
    validateGraphOutputsDisjointFromInputs(this.id, outputs, inputs);
    outputs.forEach((view, index) =>
      validateGraphOutputsDisjointFromInputs(this.id, [view], outputs.slice(index + 1))
    );
    if (props.conditioning) {
      const conditioning = props.conditioning;
      validateCapacity(
        this.id,
        'conditioning.neighborCount',
        conditioning.neighborCount,
        1,
        GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_NEIGHBOR_COUNT
      );
      validateCapacity(
        this.id,
        'conditioning.indices.length',
        conditioning.indices.length,
        1,
        GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_ROW_COUNT
      );
      validateRequirement(
        this.id,
        conditioning.values.length === conditioning.indices.length &&
          (!conditioning.mask || conditioning.mask.length === conditioning.indices.length),
        'conditioning.values and mask lengths must equal indices length'
      );
      validateRequirement(
        this.id,
        conditioning.neighborhoodOffsets
          ? conditioning.neighborhoodOffsets.length === this.rowCount + 1
          : conditioning.neighborhoods.length === this.rowCount * conditioning.neighborCount,
        'conditioning.neighborhoods length must equal rowCount * neighborCount, or supply neighborhoodOffsets of rowCount + 1 for CSR'
      );
    }
    // Published FFT stats are exact for packed views; each scalar reduction needs at most four
    // levels at 2048 squared. Conditioning adds one gather, one bounded solve and exact observation restoration.
    this.estimatedCommandNodeCount =
      this.method === 'fft'
        ? 12 + 2 * makeGPUFFT2DStats(this.paddedWidth, this.paddedHeight).dispatchCountPerEncode
        : 6;
    if (props.conditioning) {
      const losses = validateWeightsLossSources(this.id, [props.conditioning]);
      this.estimatedCommandNodeCount += 3 + losses.length * 2;
    }
    if (props.output.status.candidateOverflow) {
      validateViewLength(
        this.id,
        'output.status.candidateOverflow',
        props.output.status.candidateOverflow,
        1
      );
      this.estimatedCommandNodeCount += Math.max(
        1,
        Math.ceil(
          validateWeightsLossSources(this.id, props.conditioning ? [props.conditioning] : [])
            .length / 7
        )
      );
    }
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount} = this;
    const views = [
      props.parameters,
      props.positions,
      ...getGraphOutputViews(props.output),
      ...(props.conditioning
        ? [
            props.conditioning.indices,
            props.conditioning.values,
            props.conditioning.mask,
            props.conditioning.neighborhoods,
            props.conditioning.neighborhoodOffsets,
            ...validateWeightsLossSources(id, [props.conditioning])
          ]
        : [])
    ];
    validateGraphViewsBelongToGraph(id, graph, views);
    for (const view of views)
      if (view)
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'view',
          view.length * view.rowByteLength,
          'reduce view length'
        );
    const paddedCount = this.paddedWidth * this.paddedHeight;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'field scratch',
      this.method === 'fft' ? paddedCount * 8 : rowCount * 8,
      'reduce grid dimensions or rowCount'
    );
    if (props.conditioning)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'conditioning scratch',
        Math.max(rowCount, props.conditioning.indices.length) * 16,
        'reduce conditioning indices or rowCount'
      );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const kernel = (
      variant: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode(graph, {
        id: `${id}-${variant}`,
        operation: 'GPUGaussianRandomField',
        variant,
        invocationCount,
        bindings: [binding('params', props.parameters), ...bindings],
        body,
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${declarations}`
      });
    const field = createTransientView(graph, `${id}-field`, 'float32', rowCount);
    const positions =
      props.positions ?? createTransientView(graph, `${id}-positions`, 'float32x2', rowCount);
    if (!props.positions)
      nodes.push(
        kernel(
          'grid-positions',
          rowCount,
          [binding('positions', positions, true)],
          `
positions[positionsOffset + index * 2u] = f32(index % ${props.grid!.width}u) * params[paramsOffset + 13u];
positions[positionsOffset + index * 2u + 1u] = f32(index / ${props.grid!.width}u) * params[paramsOffset + 14u];`
        )
      );
    if (this.method === 'fft') {
      const covariance = createTransientView(graph, `${id}-covariance`, 'float32x2', paddedCount);
      const spectrum = createTransientView(graph, `${id}-spectrum`, 'float32x2', paddedCount);
      const modes = createTransientView(graph, `${id}-modes`, 'float32x2', paddedCount);
      const realization = createTransientView(graph, `${id}-realization`, 'float32x2', paddedCount);
      const negativeMass = createTransientView(
        graph,
        `${id}-negative-mass`,
        'float32',
        paddedCount
      );
      nodes.push(
        kernel(
          'torus-covariance',
          paddedCount,
          [binding('covariance', covariance, true)],
          `
let column = index % ${this.paddedWidth}u;
let row = index / ${this.paddedWidth}u;
let displacement = vec2<f32>(f32(min(column, ${this.paddedWidth}u - column)) * params[paramsOffset + 13u], f32(min(row, ${this.paddedHeight}u - row)) * params[paramsOffset + 14u]);
let value = params[paramsOffset + 6u] * covarianceCorrelation(u32(params[paramsOffset + 4u]), length(displacement) / params[paramsOffset + 7u]) + select(0.0, params[paramsOffset + 5u], index == 0u);
covariance[covarianceOffset + index * 2u] = value;
covariance[covarianceOffset + index * 2u + 1u] = 0.0;`,
          getPerFramePredictionCovarianceWGSL(1, 1)
        )
      );
      nodes.push(
        ...new GPUFFT2D({
          id: `${id}-forward`,
          width: this.paddedWidth,
          height: this.paddedHeight,
          input: covariance,
          output: spectrum
        }).getCommandNodes(graph)
      );
      nodes.push(
        kernel(
          'random-spectrum',
          paddedCount,
          [
            binding('spectrum', spectrum),
            binding('modes', modes, true),
            binding('negativeMass', negativeMass, true)
          ],
          `
let eigenvalue = spectrum[spectrumOffset + index * 2u];
let normal = normalPair(randomBlock(index, 0u));
let amplitude = sqrt(max(0.0, eigenvalue) * ${paddedCount}.0);
modes[modesOffset + index * 2u] = normal.x * amplitude;
modes[modesOffset + index * 2u + 1u] = normal.y * amplitude;
negativeMass[negativeMassOffset + index] = max(0.0, -eigenvalue) / ${paddedCount}.0;`,
          RANDOM_WGSL
        )
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-clipped-mass`,
          input: negativeMass,
          output: props.output.clippedEigenvalueMass,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...new GPUFFT2D({
          id: `${id}-inverse`,
          width: this.paddedWidth,
          height: this.paddedHeight,
          direction: 'inverse',
          input: modes,
          output: realization
        }).getCommandNodes(graph)
      );
      nodes.push(
        kernel(
          'crop',
          rowCount,
          [binding('realization', realization), binding('field', field, true)],
          `
let source = (index / ${props.grid!.width}u) * ${this.paddedWidth}u + index % ${props.grid!.width}u;
field[fieldOffset + index] = realization[realizationOffset + source * 2u] + params[paramsOffset + 8u];`
        )
      );
    } else {
      const modes = createTransientView(graph, `${id}-modes`, 'float32x4', this.modeCount);
      nodes.push(
        kernel(
          'spectral-modes',
          this.modeCount,
          [binding('modes', modes, true)],
          `
// Clear every frame, including when an unused or unrepresentable mode exits early.
let modeOffset = modesOffset + index * 4u;
modes[modeOffset] = 0.0;
modes[modeOffset + 1u] = 0.0;
modes[modeOffset + 2u] = 0.0;
modes[modeOffset + 3u] = 0.0;
if (params[paramsOffset + 6u] == 0.0) { return; }
let words = randomBlock(index, 1u);
let radialUniform = philoxOpenUnitFloat(words.x);
let angle = 6.283185307179586 * philoxUnitFloat(words.y);
let model = u32(params[paramsOffset + 4u]);
var radius = sqrt(-12.0 * log(radialUniform)) / params[paramsOffset + 7u];
if (model != 2u) {
  var smoothness = 0.5;
  var factor = 3.0;
  if (model == 3u) { smoothness = 1.5; factor = ${MATERN_3_2_RANGE_FACTOR}; }
  if (model == 4u) { smoothness = 2.5; factor = ${MATERN_5_2_RANGE_FACTOR}; }
  radius = factor * sqrt(pow(radialUniform, -1.0 / smoothness) - 1.0) / params[paramsOffset + 7u];
}
if (!isFiniteValue(radius)) {
  modes[modeOffset] = getNaN();
  modes[modeOffset + 1u] = getNaN();
  return;
}
let amplitudes = normalPair(vec4<u32>(words.z, words.w, 0u, 0u));
modes[modesOffset + index * 4u] = radius * cos(angle);
modes[modesOffset + index * 4u + 1u] = radius * sin(angle);
modes[modesOffset + index * 4u + 2u] = amplitudes.x;
modes[modesOffset + index * 4u + 3u] = amplitudes.y;`,
          RANDOM_WGSL
        )
      );
      nodes.push(
        kernel(
          'spectral-field',
          rowCount,
          [binding('modes', modes), binding('positions', positions), binding('field', field, true)],
          `
let position = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u])${props.positions ? ' - vec2<f32>(params[paramsOffset + 11u], params[paramsOffset + 12u])' : ''};
if (!isFiniteValue(position.x) || !isFiniteValue(position.y)) {
  field[fieldOffset + index] = getNaN();
  return;
}
var total = 0.0;
if (params[paramsOffset + 6u] > 0.0) {
  for (var mode = 0u; mode < ${this.modeCount}u; mode++) {
    let base = modesOffset + mode * 4u;
    let frequency = vec2<f32>(modes[base], modes[base + 1u]);
    if (!isFiniteValue(frequency.x) || !isFiniteValue(frequency.y)) {
      field[fieldOffset + index] = getNaN();
      return;
    }
    let phase = dot(position, frequency);
    if (!isFiniteValue(phase)) {
      field[fieldOffset + index] = getNaN();
      return;
    }
    total += modes[base + 2u] * cos(phase) + modes[base + 3u] * sin(phase);
  }
}
let nugget = normalPair(randomBlock(index, 2u)).x * sqrt(params[paramsOffset + 5u]);
field[fieldOffset + index] = params[paramsOffset + 8u] + sqrt(params[paramsOffset + 6u] / ${this.modeCount}.0) * total + nugget;
if (params[paramsOffset + 4u] == 0.0) { field[fieldOffset + index] = getNaN(); }`,
          RANDOM_WGSL
        )
      );
      nodes.push(
        kernel(
          'spectral-status',
          1,
          [binding('mass', props.output.clippedEigenvalueMass, true)],
          'mass[massOffset] = 0.0;'
        )
      );
    }
    let predictions: GraphDataView<'float32x4'> | undefined;
    let statuses: GraphDataView<'uint32'> | undefined;
    if (props.conditioning) {
      const conditioning = props.conditioning;
      const observationPositions = createTransientView(
        graph,
        `${id}-observation-positions`,
        'float32x2',
        conditioning.indices.length
      );
      const observations = createTransientView(
        graph,
        `${id}-residuals`,
        'float32x4',
        conditioning.indices.length
      );
      predictions = createTransientView(graph, `${id}-predictions`, 'float32x4', rowCount);
      statuses = createTransientView(graph, `${id}-kriging-status`, 'uint32', rowCount);
      nodes.push(
        kernel(
          'residuals',
          conditioning.indices.length,
          [
            binding('indices', conditioning.indices),
            binding('data', conditioning.values),
            binding('positions', positions),
            binding('field', field),
            binding('observationPositions', observationPositions, true),
            binding('observations', observations, true),
            ...(conditioning.mask ? [binding('mask', conditioning.mask)] : [])
          ],
          `
let source = indices[indicesOffset + index];
let usable = source < ${rowCount}u ${conditioning.mask ? '&& mask[maskOffset + index] != 0u' : ''} && isFiniteValue(data[dataOffset + index]);
var position = vec2<f32>(getNaN());
var residual = getNaN();
if (usable) {
  position = vec2<f32>(positions[positionsOffset + source * 2u], positions[positionsOffset + source * 2u + 1u]);
  residual = data[dataOffset + index] - field[fieldOffset + source];
}
observationPositions[observationPositionsOffset + index * 2u] = position.x;
observationPositions[observationPositionsOffset + index * 2u + 1u] = position.y;
observations[observationsOffset + index * 4u] = residual;
observations[observationsOffset + index * 4u + 1u] = 0.0;
observations[observationsOffset + index * 4u + 2u] = 0.0;
observations[observationsOffset + index * 4u + 3u] = select(0.0, 1.0, usable && isFiniteValue(residual));`
        )
      );
      nodes.push(
        ...new GPULocalKriging({
          id: `${id}-residual-kriging`,
          spatialContext: props.spatialContext,
          positions: observationPositions,
          observations,
          supports: positions,
          parameters: props.parameters,
          mode: 'simple',
          emptyNeighborhood: 'prior',
          neighborhoods: conditioning.neighborhoods,
          neighborhoodOffsets: conditioning.neighborhoodOffsets,
          neighborhoodStatus: conditioning.status,
          neighborhoodCapacity: conditioning.neighborCount,
          neighborCount: conditioning.neighborCount,
          output: {values: predictions, status: statuses}
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      kernel(
        'publish',
        props.output.values.length,
        [
          binding('field', field),
          binding('values', props.output.values, true),
          binding('validity', props.output.validity, true),
          ...(predictions && statuses
            ? [binding('predictions', predictions), binding('statuses', statuses)]
            : [])
        ],
        `
var value = getNaN();
if (index < ${rowCount}u && params[paramsOffset] > 0.0) {
  value = field[fieldOffset + index];
  ${predictions ? 'if (statuses[statusesOffset + index] == 0u) { value += predictions[predictionsOffset + index * 4u]; } else { value = getNaN(); }' : ''}
}
values[valuesOffset + index] = value;
validity[validityOffset + index] = select(0u, 1u, isFiniteValue(value));`
      )
    );
    if (props.conditioning) {
      const conditioning = props.conditioning;
      nodes.push(
        kernel(
          'hard-observations',
          conditioning.indices.length,
          [
            binding('indices', conditioning.indices),
            binding('data', conditioning.values),
            binding('values', props.output.values, true),
            binding('validity', props.output.validity),
            ...(conditioning.mask ? [binding('mask', conditioning.mask)] : [])
          ],
          `
let source = indices[indicesOffset + index];
if (source >= ${rowCount}u) { return; }
${conditioning.mask ? 'if (mask[maskOffset + index] == 0u) { return; }' : ''}
let value = data[dataOffset + index];
if (validity[validityOffset + source] != 0u && isFiniteValue(value)) {
  values[valuesOffset + source] = value;
}`
        )
      );
    }
    nodes.push(
      kernel(
        'approximation',
        1,
        [
          binding('mass', props.output.clippedEigenvalueMass),
          binding('approximation', props.output.status.approximation, true)
        ],
        `approximation[approximationOffset] = select(0u, 1u, ${this.method === 'spectral' || Boolean(props.conditioning) ? 'true' : 'mass[massOffset] > 0.0'});`
      )
    );
    const lossViews = validateWeightsLossSources(
      id,
      props.conditioning ? [props.conditioning] : []
    );
    const candidateOverflow = props.output.status.candidateOverflow;
    if (candidateOverflow)
      nodes.push(
        ...createWeightsLossNodes(graph, `${id}-conditioning-loss`, lossViews, candidateOverflow)
      );
    for (const [lossIndex, loss] of lossViews.entries())
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-conditioning-incomplete-${lossIndex}`,
          operation: 'GaussianRandomFieldInputLoss',
          invocationCount: props.output.values.length,
          bindings: [
            binding('loss', loss),
            binding('values', props.output.values, true),
            binding('validity', props.output.validity, true)
          ],
          declarations: GET_NAN_WGSL,
          body: 'if (loss[lossOffset] != 0u) { values[valuesOffset + index] = getNaN(); validity[validityOffset + index] = 0u; }'
        })
      );
    return nodes;
  }
}
