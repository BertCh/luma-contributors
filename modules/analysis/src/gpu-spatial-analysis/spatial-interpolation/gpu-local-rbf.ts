// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateInterpolationPorts} from './interpolation-view-validation';
import {
  createInterpolationNeighborhoodLossNodes,
  getInterpolationNeighborhoodLossViews,
  decodeInterpolationPredictionStatus,
  type InterpolationNeighborhoodWeights
} from './interpolation-neighborhood-loss';

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

import {isGraphVectorView} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';

import {
  validateInterpolationPredictionViews,
  validateInterpolationOutputsDisjointFromInputs
} from './interpolation-view-validation';

import {
  GraphVectorView,
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
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateCapacity,
  validateFiniteNumber,
  validateNonZeroFiniteNumber,
  validateCommandNodeBudget,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';

/** Bounds the private dense system (up to six polynomial columns) and cubic solve work per query. */
export const GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT = 32;

/** Distances are planar Euclidean in the native units of the positions. */
const PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance' as const
});

/** Bounded local radial-basis interpolation in native planar units. */
export type GPULocalRBFProps<
  QueryRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  ValueRows extends
    | GraphDataView<'float32x4'>
    | GraphVectorView<'float32x4'> = GraphDataView<'float32x4'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  /**
   * Coordinate space of the positions. Compile-time. Omitted means native planar coordinates with
   * Euclidean distance (unchanged behavior); only that is accepted, so longitude/latitude throws:
   * project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Nonfinite values/positions are omitted. */
  values: GraphDataView<'float32'>;
  /** Packed or chunked query rows; output chunks must mirror their ordered lengths. */
  queries: QueryRows;
  kernel?:
    | 'thin-plate-spline'
    | 'cubic'
    | 'gaussian'
    | 'linear'
    | 'quintic'
    | 'multiquadric'
    | 'inverse-multiquadric'
    | 'inverse-quadratic';
  /** Polynomial degree up to 2. Thin plate/cubic require >=1, quintic requires2, linear/multiquadric >=0. */
  degree?: -1 | 0 | 1 | 2;
  /** Multiplier of distance before applying the radial kernel; positive, defaults 1. */
  epsilon?: number;
  smoothing?: number;
  /** Optional nonnegative per-sample smoothing, replacing the scalar diagonal. */
  sampleSmoothing?: GraphDataView<'float32'>;
  /** Optional query-major candidate rows supplied by a resident neighbor index; 0xffffffff pads. */
  neighborhoods?: GraphDataView<'uint32'>;
  /** Resident CSR including optional live producer overflow/candidateOverflow evidence. */
  neighborhoodWeights?: InterpolationNeighborhoodWeights;
  /** Producer loss for legacy fixed-width or CSR candidate ports. */
  neighborhoodStatus?: InterpolationNeighborhoodWeights['status'];
  /** Optional CSR offsets from GPUNeighborSearch; replaces fixed candidate row widths. */
  neighborhoodOffsets?: GraphDataView<'uint32'>;
  /** Fixed candidate slots per query. Each row must contain distinct sample indices. */
  neighborhoodCapacity?: number;
  neighborCount?: number;
  searchRadius?: number;
  output: {
    /** [prediction, pivot ratio, neighbor count, 0]. */
    values: ValueRows;
    /** 0 success; 1 no usable neighborhood; 2 deficient polynomial/kernel rank; 3 incomplete candidate input (overflow or candidateOverflow). */
    status: StatusRows;
  };
};

type PackedLocalRBFProps = Omit<GPULocalRBFProps, 'queries' | 'output'> & {
  queries: GraphDataView<'float32x2'>;
  output: {values: GraphDataView<'float32x4'>; status: GraphDataView<'uint32'>};
};

/** Five data bindings precede optional candidates and smoothing; dispatch parameters use a uniform. */
const LOCAL_RBF_CHUNK_BASE_STORAGE_BINDING_COUNT = 5;
/** Dense per-query solves retain the packed route's 32-thread workgroup size. */
const LOCAL_RBF_CHUNK_WORKGROUP_SIZE = 32;

/**
 * One GPU augmented kernel/polynomial solve per query. Selection retains the nearest 1..32 rows,
 * ties use original row index. Optional fixed-width or CSR candidates compose resident GPU indexes.
 * Polynomial coordinates are centered/scaled locally. Extrapolation outside
 * the sample hull is allowed; the radius/neighbor count explicitly delimit the model. Smoothing
 * adds a nonnegative kernel diagonal. Duplicate zero-smoothing observations and collinear affine
 * designs report rank failure. No global quadratic matrix or CPU solve is constructed.
 * Chunked queries borrow packed resident samples and candidate lists, use one node-owned
 * uniform for dispatch parameters, and preserve the packed result exactly. Candidate rows and
 * CSR offsets remain globally query-major. Chunked outputs have exactly one row per query.
 */
export class GPULocalRBF<
  QueryRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  ValueRows extends
    | GraphDataView<'float32x4'>
    | GraphVectorView<'float32x4'> = GraphDataView<'float32x4'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  /** Resolved coordinate space (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  readonly props: GPULocalRBFProps<QueryRows, ValueRows, StatusRows>;
  readonly neighborCount: number;
  readonly degree: -1 | 0 | 1 | 2;
  readonly kernel: NonNullable<GPULocalRBFProps['kernel']>;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan?: GraphChunkDispatchPlan;

  constructor(props: GPULocalRBFProps<QueryRows, ValueRows, StatusRows>) {
    this.props = props = snapshotInterpolationConstructionProps(
      props,
      [
        'degree',
        'epsilon',
        'id',
        'kernel',
        'neighborCount',
        'neighborhoodCapacity',
        'neighborhoodOffsets',
        'neighborhoods',
        'neighborhoodWeights',
        'neighborhoodStatus',
        'output',
        'positions',
        'queries',
        'sampleSmoothing',
        'searchRadius',
        'smoothing',
        'spatialContext',
        'values'
      ],
      ['values', 'status']
    );
    this.id = props.id ?? 'local-rbf';
    validateInterpolationPorts({id: this.id, contributor: 'GPULocalRBF'}, props, {
      positions: {required: true, viewType: 'data', formats: ['float32x2']},
      values: {required: true, viewType: 'data', formats: ['float32']},
      queries: {required: true, viewType: 'either', formats: ['float32x2']},
      neighborhoods: {required: false, viewType: 'data', formats: ['uint32']},
      neighborhoodOffsets: {required: false, viewType: 'data', formats: ['uint32']},
      sampleSmoothing: {required: false, viewType: 'data', formats: ['float32']},
      'output.values': {required: true, viewType: 'either', formats: ['float32x4']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT
    );
    this.kernel = props.kernel ?? 'thin-plate-spline';
    this.degree =
      props.degree ??
      (this.kernel === 'quintic'
        ? 2
        : ['thin-plate-spline', 'cubic'].includes(this.kernel)
          ? 1
          : this.kernel === 'linear' || this.kernel === 'multiquadric'
            ? 0
            : -1);
    this.neighborCount = props.neighborCount ?? 16;
    validateCapacity(
      this.id,
      'neighborCount',
      this.neighborCount,
      1,
      GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT
    );
    validateNonZeroFiniteNumber(this.id, 'epsilon', props.epsilon ?? 1);
    validateFiniteNumber(this.id, 'smoothing', props.smoothing ?? 0, 0);
    if (props.searchRadius !== undefined && props.searchRadius !== Infinity)
      validateNonZeroFiniteNumber(this.id, 'searchRadius', props.searchRadius);
    if (props.neighborhoods && !props.neighborhoodOffsets)
      validateCapacity(
        this.id,
        'neighborhoodCapacity',
        props.neighborhoodCapacity ?? this.neighborCount
      );
    validateRequirement(
      this.id,
      !(
        ![-1, 0, 1, 2].includes(this.degree) ||
        ![
          'thin-plate-spline',
          'cubic',
          'gaussian',
          'linear',
          'quintic',
          'multiquadric',
          'inverse-multiquadric',
          'inverse-quadratic'
        ].includes(this.kernel) ||
        (['thin-plate-spline', 'cubic'].includes(this.kernel) && this.degree < 1) ||
        (['linear', 'multiquadric'].includes(this.kernel) && this.degree < 0) ||
        (this.kernel === 'quintic' && this.degree < 2) ||
        !Number.isFinite(props.epsilon ?? 1) ||
        !((props.epsilon ?? 1) > 0) ||
        !Number.isFinite(props.smoothing ?? 0) ||
        !((props.smoothing ?? 0) >= 0) ||
        !(props.searchRadius === undefined || props.searchRadius > 0)
      ),
      `kernel and degree must form a supported pair (degree -1..2; thin-plate-spline/cubic >= 1, linear/multiquadric >= 0, quintic = 2); epsilon must be finite and positive, smoothing finite and nonnegative, searchRadius positive when supplied`
    );
    validateRequirement(
      this.id,
      !(
        props.positions.length !== props.values.length ||
        (props.sampleSmoothing && props.sampleSmoothing.length !== props.positions.length) ||
        (props.neighborhoodOffsets &&
          (!props.neighborhoods ||
            props.neighborhoodOffsets.length !== props.queries.length + 1)) ||
        (props.neighborhoods &&
          !props.neighborhoodOffsets &&
          (!Number.isSafeInteger(props.neighborhoodCapacity ?? this.neighborCount) ||
            (props.neighborhoodCapacity ?? this.neighborCount) < 1 ||
            props.neighborhoods.length !==
              props.queries.length * (props.neighborhoodCapacity ?? this.neighborCount))) ||
        props.output.values.length < props.queries.length ||
        props.output.status.length < props.queries.length
      ),
      `values and sampleSmoothing must match positions.length; neighborhoodOffsets requires neighborhoods and queries.length + 1 rows; fixed neighborhoods must contain queries.length * neighborhoodCapacity entries; output.values/status must cover queries.length`
    );

    const lossViews = getInterpolationNeighborhoodLossViews(this.id, props);
    const inputs: (GraphDataView | GraphVectorView | undefined)[] = [
      ...lossViews,
      props.positions,
      props.values,
      props.queries,
      props.neighborhoods,
      props.neighborhoodOffsets,
      props.sampleSmoothing
    ];
    if (
      isGraphVectorView(props.queries) ||
      isGraphVectorView(props.output.values) ||
      isGraphVectorView(props.output.status)
    ) {
      this.chunkPlan = planGraphChunkDispatch(this.id, {
        queries: props.queries,
        results: props.output.values,
        statuses: props.output.status
      });
      for (const chunk of this.chunkPlan.chunks) {
        validateSinglePackedView(chunk.views['queries'], ['float32x2'], `${this.id} queries`);
        validateSinglePackedView(chunk.views['results'], ['float32x4'], `${this.id} output.values`);
        validateSinglePackedView(chunk.views['statuses'], ['uint32'], `${this.id} output.status`);
      }
    } else {
      validateInterpolationPredictionViews(
        this.id,
        inputs.filter((view): view is GraphDataView => view !== undefined),
        [props.output.values, props.output.status]
      );
    }
    validateDisjointWritableViews(this.id, props.output);
    validateInterpolationOutputsDisjointFromInputs(this.id, Object.values(props.output), inputs);
    this.estimatedCommandNodeCount =
      (this.chunkPlan ? this.chunkPlan.nonemptyChunkCount : 1) * (1 + lossViews.length);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, chunkPlan} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      values: props.values,
      queries: props.queries,
      results: props.output.values,
      statuses: props.output.status,
      neighborhoods: props.neighborhoods,
      neighborhoodOffsets: props.neighborhoodOffsets,
      sampleSmoothing: props.sampleSmoothing
    });
    const losses = getInterpolationNeighborhoodLossViews(id, props);
    if (!chunkPlan) {
      const packed = props as PackedLocalRBFProps;
      return [
        createWGSLKernelNode(graph, this.getKernelProps(packed)),
        ...createInterpolationNeighborhoodLossNodes(
          graph,
          id,
          losses,
          packed.output,
          props.queries.length
        )
      ];
    }
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      LOCAL_RBF_CHUNK_BASE_STORAGE_BINDING_COUNT +
        Number(Boolean(props.neighborhoods)) +
        Number(Boolean(props.neighborhoodOffsets)) +
        Number(Boolean(props.sampleSmoothing)),
      LOCAL_RBF_CHUNK_WORKGROUP_SIZE
    );
    const nodes = chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const kernelProps = this.getKernelProps(
          {
            ...props,
            queries: chunk.views['queries'] as GraphDataView<'float32x2'>,
            output: {
              values: chunk.views['results'] as GraphDataView<'float32x4'>,
              status: chunk.views['statuses'] as GraphDataView<'uint32'>
            }
          },
          'queryIndex'
        );
        return createGraphChunkGroupKernelNode(graph, {
          ...kernelProps,
          id: `${id}-chunk-${chunk.chunkIndex}`,
          pipelineId: `${id}-chunk`,
          rowBase: chunk.rowBase,
          totalLength: chunkPlan.totalLength,
          body: `let queryIndex = getChunkParameter(0u) + index;\n${kernelProps.body}`
        });
      });
    for (const chunk of chunkPlan.chunks)
      if (chunk.length > 0)
        nodes.push(
          ...createInterpolationNeighborhoodLossNodes(
            graph,
            `${id}-chunk-${chunk.chunkIndex}`,
            losses,
            {
              values: chunk.views['results'] as GraphDataView<'float32x4'>,
              status: chunk.views['statuses'] as GraphDataView<'uint32'>
            },
            chunk.length
          )
        );
    return nodes;
  }

  private getKernelProps<Parameters>(
    props: PackedLocalRBFProps,
    queryRow: string = 'index'
  ): WGSLKernelProps<Parameters> {
    const {id, neighborCount, degree, kernel} = this;

    const polynomialCount = degree < 0 ? 0 : degree === 0 ? 1 : degree === 1 ? 3 : 6;
    const radial =
      kernel === 'thin-plate-spline'
        ? 'select(0.0, separation * separation * log(max(separation, 1e-30)), separation > 0.0)'
        : kernel === 'cubic'
          ? 'separation * separation * separation'
          : kernel === 'gaussian'
            ? 'exp(-separation * separation)'
            : kernel === 'quintic'
              ? '-pow(separation, 5.0)'
              : kernel === 'multiquadric'
                ? '-sqrt(1.0 + separation * separation)'
                : kernel === 'inverse-multiquadric'
                  ? 'inverseSqrt(1.0 + separation * separation)'
                  : kernel === 'inverse-quadratic'
                    ? '1.0 / (1.0 + separation * separation)'
                    : '-separation';
    // Polyharmonic kernels are homogeneous of this degree (thin-plate-spline up to a polynomial
    // term the constrained system absorbs), so the radial block is evaluated on distances divided
    // by the neighborhood scale and the smoothing is divided by scale^degree. The interpolant is
    // unchanged but the matrix is dimensionless for projected coordinates of any magnitude.
    const homogeneousDegree =
      kernel === 'thin-plate-spline'
        ? 2
        : kernel === 'cubic'
          ? 3
          : kernel === 'quintic'
            ? 5
            : kernel === 'linear'
              ? 1
              : 0;
    return {
      id,
      operation: 'GPULocalRBF',
      variant: `${kernel}-${degree}`,
      invocationCount: props.queries.length,
      workgroupSize: 32,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'values', view: props.values, type: 'f32', access: 'read'},
        {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
        {name: 'results', view: props.output.values, type: 'f32', access: 'read_write'},
        {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'},
        ...(props.neighborhoodOffsets
          ? [
              {
                name: 'neighborhoodOffsets',
                view: props.neighborhoodOffsets,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(props.neighborhoods
          ? [
              {
                name: 'neighborhoods',
                view: props.neighborhoods,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(props.sampleSmoothing
          ? [
              {
                name: 'sampleSmoothing',
                view: props.sampleSmoothing,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      declarations: `${getPackedPredictionAccessors({positions: 2, queries: 2}, ['results'])}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${SCALED_QUOTIENT_WGSL}\n${getEquilibratedSolveWGSL(neighborCount + polynomialCount)}
var<private> radialScale: f32 = 1.0;
fn radial(first: vec2<f32>, second: vec2<f32>) -> f32 {
  let separation = distance(first, second) * ${getWGSLFloatLiteral(props.epsilon ?? 1)} * radialScale;
  return ${radial};
}
fn scaleSmoothing(smoothing: f32, coordinateScale: f32) -> f32 {
  // Repeated divisions can reassociate into smoothing * pow(1 / scale, degree) on Metal.
  // Keep the power in integer exponents and a bounded [1/32, 1] mantissa instead.
  ${
    homogeneousDegree > 0
      ? `let divisor = frexp(coordinateScale);
  return getScaledQuotient(smoothing, pow(divisor.fract, ${homogeneousDegree}.0), 1.0, -${homogeneousDegree} * divisor.exp);`
      : 'return smoothing;'
  }
}
fn polynomial(position: vec2<f32>, column: u32) -> f32 {
  switch column { case 0u: { return 1.0; } case 1u: { return position.x; } case 2u: { return position.y; } case 3u: { return position.x * position.x; } case 4u: { return position.x * position.y; } default: { return position.y * position.y; } }
}`,
      body: `
write_results(index, vec4<f32>(getNaN(), 0.0, 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
let query = read_queries(index);
if (!isFiniteValue(query.x) || !isFiniteValue(query.y)) { return; }
var neighbors: array<u32, ${neighborCount}>;
var distances: array<f32, ${neighborCount}>;
var retained = 0u;
${props.neighborhoodOffsets ? `let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + ${queryRow}]; let candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + ${queryRow} + 1u]; if (candidateBegin > candidateEnd || candidateEnd > arrayLength(&neighborhoods) - neighborhoodsOffset) { return; }` : ''}
for (var candidate = ${props.neighborhoodOffsets ? 'candidateBegin' : '0u'}; candidate < ${props.neighborhoodOffsets ? 'candidateEnd' : props.neighborhoods ? `${props.neighborhoodCapacity ?? neighborCount}u` : `${props.positions.length}u`}; candidate++) {
  let sample = ${props.neighborhoods ? `neighborhoods[neighborhoodsOffset + ${props.neighborhoodOffsets ? '' : `${queryRow} * ${props.neighborhoodCapacity ?? neighborCount}u + `}candidate]` : 'candidate'};
  if (sample >= ${props.positions.length}u) { continue; }
  ${props.sampleSmoothing ? 'if (!isFiniteValue(sampleSmoothing[sampleSmoothingOffset + sample]) || sampleSmoothing[sampleSmoothingOffset + sample] < 0.0) { continue; }' : ''}
  let position = read_positions(sample);
  if (!isFiniteValue(values[valuesOffset + sample]) || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { continue; }
  let separation = distance(position, query);
  if (!(separation <= ${getWGSLFloatLiteral(Number.isFinite(props.searchRadius) ? props.searchRadius! : 3.402823e38)})) { continue; }
  var insertion = retained;
  for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor] || (separation == distances[neighbor] && sample < neighbors[neighbor])) { insertion = neighbor; break; } }
  if (insertion >= ${neighborCount}u) { continue; }
  var shift = min(retained, ${neighborCount - 1}u);
  loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
  neighbors[insertion] = sample; distances[insertion] = separation; retained = min(retained + 1u, ${neighborCount}u);
}
if (retained == 0u || retained < ${polynomialCount}u) { return; }
let coordinateScale = max(distances[retained - 1u], 1e-20);
${homogeneousDegree > 0 ? `radialScale = 1.0 / coordinateScale;` : ''}
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  for (var column = 0u; column < retained; column++) { matrix[row * SYSTEM_CAPACITY + column] = radial(position, read_positions(neighbors[column])); }
  matrix[row * SYSTEM_CAPACITY + row] += scaleSmoothing(${props.sampleSmoothing ? 'sampleSmoothing[sampleSmoothingOffset + neighbors[row]]' : getWGSLFloatLiteral(props.smoothing ?? 0)}, coordinateScale);
  right[row] = values[valuesOffset + neighbors[row]];
  for (var column = 0u; column < ${polynomialCount}u; column++) {
    let value = polynomial((position - query) / coordinateScale, column);
    matrix[row * SYSTEM_CAPACITY + retained + column] = value;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = value;
  }
}
for (var row = 0u; row < ${polynomialCount}u; row++) {
  right[retained + row] = 0.0;
  for (var column = 0u; column < ${polynomialCount}u; column++) { matrix[(retained + row) * SYSTEM_CAPACITY + retained + column] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + ${polynomialCount}u)) { results[resultsOffset + index * 4u + 1u] = pivotRatio; return; }
var prediction = ${polynomialCount ? 'solution[retained]' : '0.0'};
for (var row = 0u; row < retained; row++) { prediction += solution[row] * radial(query, read_positions(neighbors[row])); }
if (!isFiniteValue(prediction)) { return; }
write_results(index, vec4<f32>(prediction, pivotRatio, f32(retained), 0.0));
statuses[statusesOffset + index] = 0u;`
    };
  }
}

/** Decode one query status: 0 complete, 1 invalid/insufficient input, 2 singular solve, 3 incomplete candidates. */
export function decodeGPULocalRBFStatus(
  input: Uint32Array | Uint8Array | ArrayBuffer,
  row: number = 0
) {
  return decodeInterpolationPredictionStatus('decodeGPULocalRBFStatus', input, row);
}
