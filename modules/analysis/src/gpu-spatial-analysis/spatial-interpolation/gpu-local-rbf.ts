// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedPredictionAccessors,
  validatePredictionViews
} from '../../analysis-core/kernels/gpu-prediction-utils';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';

/** Bounded local radial-basis interpolation in native planar units. */
export type GPULocalRBFProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Nonfinite values/positions are omitted. */
  values: GraphDataView<'float32'>;
  queries: GraphDataView<'float32x2'>;
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
  /** Optional CSR offsets from GPUNeighborSearch; replaces fixed candidate row widths. */
  neighborhoodOffsets?: GraphDataView<'uint32'>;
  /** Fixed candidate slots per query. Each row must contain distinct sample indices. */
  neighborhoodCapacity?: number;
  neighborCount?: number;
  searchRadius?: number;
  output: {
    /** [prediction, pivot ratio, neighbor count, 0]. */
    values: GraphDataView<'float32x4'>;
    /** 0 success; 1 no usable neighborhood; 2 deficient polynomial/kernel rank. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * One GPU augmented kernel/polynomial solve per query. Selection retains the nearest 1..32 rows,
 * ties use original row index. Optional fixed-width or CSR candidates compose resident GPU indexes. Polynomial coordinates are centered/scaled locally. Extrapolation outside
 * the sample hull is allowed; the radius/neighbor count explicitly delimit the model. Smoothing
 * adds a nonnegative kernel diagonal. Duplicate zero-smoothing observations and collinear affine
 * designs report rank failure. No global quadratic matrix or CPU solve is constructed.
 */
export class GPULocalRBF implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalRBFProps;
  readonly neighborCount: number;
  readonly degree: -1 | 0 | 1 | 2;
  readonly kernel: NonNullable<GPULocalRBFProps['kernel']>;

  constructor(props: GPULocalRBFProps) {
    this.props = props;
    this.id = props.id ?? 'local-rbf';
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
    validateCapacity(this.id, 'neighborCount', this.neighborCount, 1);
    if (
      this.neighborCount > 32 ||
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
    ) {
      throw new Error(`${this.id} invalid interpolation model`);
    }
    if (
      props.positions.length !== props.values.length ||
      (props.sampleSmoothing && props.sampleSmoothing.length !== props.positions.length) ||
      (props.neighborhoodOffsets &&
        (!props.neighborhoods || props.neighborhoodOffsets.length !== props.queries.length + 1)) ||
      (props.neighborhoods &&
        !props.neighborhoodOffsets &&
        (!Number.isSafeInteger(props.neighborhoodCapacity ?? this.neighborCount) ||
          (props.neighborhoodCapacity ?? this.neighborCount) < 1 ||
          props.neighborhoods.length !==
            props.queries.length * (props.neighborhoodCapacity ?? this.neighborCount))) ||
      props.output.values.length < props.queries.length ||
      props.output.status.length < props.queries.length
    ) {
      throw new Error(`${this.id} invalid view lengths`);
    }
    validatePredictionViews(
      this.id,
      [
        props.positions,
        props.values,
        props.queries,
        ...(props.neighborhoods ? [props.neighborhoods] : []),
        ...(props.neighborhoodOffsets ? [props.neighborhoodOffsets] : []),
        ...(props.sampleSmoothing ? [props.sampleSmoothing] : [])
      ],
      [props.output.values, props.output.status]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, neighborCount, degree, kernel} = this;
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
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.values,
      props.queries,
      props.output.values,
      props.output.status,
      ...(props.neighborhoods ? [props.neighborhoods] : []),
      ...(props.neighborhoodOffsets ? [props.neighborhoodOffsets] : []),
      ...(props.sampleSmoothing ? [props.sampleSmoothing] : [])
    ]);
    return [
      createWGSLKernelNode(graph, {
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
        declarations: `${getPackedPredictionAccessors({positions: 2, queries: 2}, ['results'])}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getEquilibratedSolveWGSL(neighborCount + polynomialCount)}
var<private> radialScale: f32 = 1.0;
fn radial(first: vec2<f32>, second: vec2<f32>) -> f32 {
  let separation = distance(first, second) * ${getWGSLFloatLiteral(props.epsilon ?? 1)} * radialScale;
  return ${radial};
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
${props.neighborhoodOffsets ? `let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + index]; let candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + index + 1u]; if (candidateBegin > candidateEnd || candidateEnd > ${props.neighborhoods!.length}u) { return; }` : ''}
for (var candidate = ${props.neighborhoodOffsets ? 'candidateBegin' : '0u'}; candidate < ${props.neighborhoodOffsets ? 'candidateEnd' : props.neighborhoods ? `${props.neighborhoodCapacity ?? neighborCount}u` : `${props.positions.length}u`}; candidate++) {
  let sample = ${props.neighborhoods ? `neighborhoods[neighborhoodsOffset + ${props.neighborhoodOffsets ? '' : `index * ${props.neighborhoodCapacity ?? neighborCount}u + `}candidate]` : 'candidate'};
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
let smoothingScale = ${homogeneousDegree > 0 ? `1.0 / pow(coordinateScale, ${homogeneousDegree}.0)` : '1.0'};
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  for (var column = 0u; column < retained; column++) { matrix[row * SYSTEM_CAPACITY + column] = radial(position, read_positions(neighbors[column])); }
  matrix[row * SYSTEM_CAPACITY + row] += smoothingScale * ${props.sampleSmoothing ? 'sampleSmoothing[sampleSmoothingOffset + neighbors[row]]' : getWGSLFloatLiteral(props.smoothing ?? 0)};
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
      })
    ];
  }
}
