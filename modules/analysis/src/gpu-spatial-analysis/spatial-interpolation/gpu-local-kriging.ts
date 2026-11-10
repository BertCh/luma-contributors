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
  getWGSLFloatLiteral,
  type WGSLKernelBinding
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
import {
  getCoregionalizationCoefficient,
  getGPULocalKrigingParameterLength,
  getPerFramePredictionCovarianceWGSL,
  getPredictionCovarianceWGSL,
  validatePredictionCovariance,
  type GPUPredictionCovariance
} from './prediction-covariance';

export {getPredictionCovarianceWGSL, validatePredictionCovariance};
export type {GPUPredictionCovariance};

/**
 * Explicit local, planar prediction contract. Counts specialize the graph. Covariance parameters
 * specialize it unless the `parameters` buffer is supplied, in which case they change per frame
 * without a recompile.
 */
export type GPULocalKrigingProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Rows [value, observationErrorVariance, variableIndex (0..variableCount-1), validity (0 skips)]. */
  observations: GraphDataView<'float32x4'>;
  /** Query-major quadrature: float32x2 has equal weights; float32x4 stores [x,y,weight,reserved]. */
  supports: GraphDataView<'float32x2' | 'float32x4'>;
  supportCount?: number;
  mode: 'simple' | 'ordinary' | 'universal' | 'external-drift' | 'co-kriging';
  /**
   * Compile-time covariance of the univariate modes and of the two-by-two `coregionalization`
   * shortcut. Required unless `parameters` is supplied, in which case it is ignored.
   */
  covariance?: GPUPredictionCovariance;
  /**
   * Per-frame covariance buffer written by `getGPULocalKrigingParameterValues`: model, nugget, sill
   * and range change between frames without a recompile. Counts one more storage binding. Replaces
   * `covariance`, `coregionalization` and the values of `coregionalizationModels`; a buffer that
   * was never written reports status 1 for every query.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Covariance structures in `parameters`, 1..4, for co-kriging. Compile time. Defaults to the
   * length of `coregionalizationModels`, else 1.
   */
  covarianceStructureCount?: number;
  /** Known means for simple or known-mean co-kriging. Defaults to zero. */
  means?: readonly number[];
  /** Number of response variables for co-kriging, 2..4. */
  variableCount?: number;
  /** Separate unknown constant for every variable, or supplied known means (default). */
  coKrigingMean?: 'known' | 'ordinary';
  /** Sum of up to four PSD covariance structures, with row-major square loading factors. */
  coregionalizationModels?: readonly {
    covariance: GPUPredictionCovariance;
    loadings: readonly number[];
  }[];
  /** Two-by-two row-major loading matrix. B = L L^T guarantees a PSD coregionalization model. */
  coregionalization?: readonly [number, number, number, number];
  targetVariable?: number;
  /**
   * Optional candidate rows supplied by a resident neighbor index, replacing the scan of every
   * sample (query-major with `neighborhoodCapacity` slots, or CSR with `neighborhoodOffsets`).
   * 0xffffffff pads. The nearest `neighborCount` candidates to the support center are retained.
   * Each extra binding counts against the device storage-buffer limit of the kernel.
   */
  neighborhoods?: GraphDataView<'uint32'>;
  /** Optional CSR offsets (query count + 1) replacing fixed candidate row widths. */
  neighborhoodOffsets?: GraphDataView<'uint32'>;
  /** Fixed candidate slots per query. Each row must contain distinct sample indices. */
  neighborhoodCapacity?: number;
  neighborCount?: number;
  searchRadius?: number;
  /** External drift includes all supplied columns, including an intercept if wanted; 1..8; flat float32 stores row-major driftCount columns. */
  driftCount?: number;
  sampleDrift?: GraphDataView<'float32' | 'float32x4'>;
  /** Query-major support-averaged external drift, one row per query. */
  queryDrift?: GraphDataView<'float32' | 'float32x4'>;
  output: {
    /** [prediction, latent point/block variance, pivot ratio, retained neighbor count]. */
    values: GraphDataView<'float32x4'>;
    /** 0 success; 1 invalid/missing/insufficient data; 2 singular or unreliable variance. */
    status: GraphDataView<'uint32'>;
    /** Optional query-major weights in neighbor order (distance, original row); padded with 0. */
    weights?: GraphDataView<'float32'>;
  };
};

/**
 * Local covariance kriging with equilibrated partial-pivot bounded solves. Neighborhood selection
 * scans every sample on GPU, or only the candidates of a resident neighbor index when
 * `neighborhoods` is supplied, and retains at most 32 rows. No readback, implicit repacking, or CPU
 * solve occurs.
 * Universal drift defaults to affine [1, x, y], centered/scaled locally, or uses up to eight
 * caller-supplied columns. Co-kriging sums up to four PSD structures over two to four variables
 * with supplied known means or one unknown constant for each variable. Duplicates are separate
 * observations: a zero-error singular system is reported, never silently jittered. Positive
 * measurement errors smooth exact hits. All block quantities use the supplied normalized nonnegative
 * support quadrature, including the within-block covariance and process nugget.
 */
export class GPULocalKriging implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalKrigingProps;
  readonly neighborCount: number;
  readonly supportCount: number;
  readonly driftCount: number;
  readonly queryCount: number;
  readonly variableCount: number;
  readonly suppliedDrift: boolean;
  readonly covarianceStructureCount: number;

  constructor(props: GPULocalKrigingProps) {
    this.props = props;
    this.id = props.id ?? 'local-kriging';
    this.neighborCount = props.neighborCount ?? 16;
    this.supportCount = props.supportCount ?? 1;
    this.variableCount = props.mode === 'co-kriging' ? (props.variableCount ?? 2) : 1;
    this.suppliedDrift =
      props.mode === 'external-drift' ||
      (props.mode === 'universal' && Boolean(props.sampleDrift || props.queryDrift));
    this.driftCount = this.suppliedDrift
      ? (props.driftCount ?? 1)
      : props.mode === 'co-kriging' && props.coKrigingMean === 'ordinary'
        ? this.variableCount
        : props.mode === 'universal'
          ? 3
          : props.mode === 'ordinary'
            ? 1
            : props.mode === 'external-drift'
              ? (props.driftCount ?? 1)
              : 0;
    this.queryCount = props.supports.length / this.supportCount;
    validateCapacity(this.id, 'neighborCount', this.neighborCount, 1);
    validateCapacity(this.id, 'supportCount', this.supportCount, 1);
    validateCapacity(this.id, 'queryCount', this.queryCount, 0);
    this.covarianceStructureCount =
      props.covarianceStructureCount ?? props.coregionalizationModels?.length ?? 1;
    if (props.covariance) {
      validatePredictionCovariance(props.covariance);
    } else if (!props.parameters) {
      throw new Error(`${this.id} needs covariance or parameters`);
    }
    if (
      props.parameters &&
      (props.parameters.format !== 'float32' ||
        !Number.isInteger(this.covarianceStructureCount) ||
        this.covarianceStructureCount < 1 ||
        this.covarianceStructureCount > 4 ||
        (props.mode !== 'co-kriging' && this.covarianceStructureCount !== 1) ||
        props.parameters.length <
          getGPULocalKrigingParameterLength(this.covarianceStructureCount, this.variableCount))
    ) {
      throw new Error(`${this.id} invalid parameters`);
    }
    if (
      this.neighborCount > 32 ||
      this.supportCount > 64 ||
      this.driftCount < 0 ||
      this.driftCount > 8 ||
      !Number.isInteger(this.driftCount) ||
      props.positions.length !== props.observations.length ||
      props.output.values.length < this.queryCount ||
      props.output.status.length < this.queryCount ||
      (props.output.weights && props.output.weights.length < this.queryCount * this.neighborCount)
    ) {
      throw new Error(`${this.id} invalid capacities`);
    }
    if (
      (props.neighborhoodOffsets &&
        (!props.neighborhoods || props.neighborhoodOffsets.length !== this.queryCount + 1)) ||
      (props.neighborhoods &&
        !props.neighborhoodOffsets &&
        props.neighborhoods.length !== this.queryCount * (props.neighborhoodCapacity ?? 0))
    ) {
      throw new Error(`${this.id} invalid neighborhoods`);
    }
    if (
      !['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'].includes(props.mode) ||
      !(props.searchRadius === undefined || props.searchRadius > 0)
    ) {
      throw new Error(`${this.id} invalid mode or radius`);
    }
    if (
      this.suppliedDrift &&
      (!props.sampleDrift ||
        !props.queryDrift ||
        props.sampleDrift.length !==
          props.positions.length * (props.sampleDrift.format === 'float32' ? this.driftCount : 1) ||
        props.queryDrift.length !==
          this.queryCount * (props.queryDrift.format === 'float32' ? this.driftCount : 1) ||
        (this.driftCount > 4 &&
          (props.sampleDrift.format !== 'float32' || props.queryDrift.format !== 'float32')) ||
        this.driftCount < 1)
    ) {
      throw new Error(`${this.id} missing external drift`);
    }
    if (
      props.mode === 'co-kriging' &&
      !props.parameters &&
      !props.coregionalization &&
      !props.coregionalizationModels
    ) {
      throw new Error(`${this.id} co-kriging requires PSD loading factors`);
    }
    if (
      ![...(props.means ?? [0, 0]), ...(props.coregionalization ?? [1, 0, 0, 1])].every(
        Number.isFinite
      )
    ) {
      throw new Error(`${this.id} nonfinite model`);
    }
    const structures = props.coregionalizationModels;
    if (
      this.variableCount < (props.mode === 'co-kriging' ? 2 : 1) ||
      this.variableCount > 4 ||
      !Number.isInteger(this.variableCount) ||
      !Number.isInteger(props.targetVariable ?? 0) ||
      (props.targetVariable ?? 0) < 0 ||
      (props.targetVariable ?? 0) >= this.variableCount ||
      (props.means && props.means.length < this.variableCount) ||
      (props.coKrigingMean && !['known', 'ordinary'].includes(props.coKrigingMean)) ||
      (props.mode === 'co-kriging' &&
        this.variableCount !== 2 &&
        !structures &&
        !props.parameters) ||
      (structures && (structures.length < 1 || structures.length > 4))
    ) {
      throw new Error(`${this.id} invalid multivariate model`);
    }
    for (const structure of structures ?? []) {
      validatePredictionCovariance(structure.covariance);
      if (
        structure.loadings.length !== this.variableCount ** 2 ||
        !structure.loadings.every(Number.isFinite)
      ) {
        throw new Error(`${this.id} invalid PSD loading dimensions`);
      }
    }
    validatePredictionViews(
      this.id,
      [
        props.positions,
        props.observations,
        props.supports,
        ...[props.sampleDrift, props.queryDrift].filter(
          (view): view is GraphDataView<'float32' | 'float32x4'> => Boolean(view)
        ),
        ...(props.neighborhoods ? [props.neighborhoods] : []),
        ...(props.neighborhoodOffsets ? [props.neighborhoodOffsets] : []),
        ...(props.parameters ? [props.parameters] : [])
      ],
      [
        props.output.values,
        props.output.status,
        ...(props.output.weights ? [props.output.weights] : [])
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, neighborCount, supportCount, driftCount, queryCount} = this;
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
      {name: 'supports', view: props.supports, type: 'f32', access: 'read'},
      {name: 'results', view: props.output.values, type: 'f32', access: 'read_write'},
      {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
    ];
    if (this.suppliedDrift) {
      bindings.push(
        {name: 'sampleDrift', view: props.sampleDrift!, type: 'f32', access: 'read'},
        {name: 'queryDrift', view: props.queryDrift!, type: 'f32', access: 'read'}
      );
    }
    if (props.neighborhoodOffsets) {
      bindings.push({
        name: 'neighborhoodOffsets',
        view: props.neighborhoodOffsets,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.neighborhoods) {
      bindings.push({
        name: 'neighborhoods',
        view: props.neighborhoods,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.parameters) {
      bindings.push({name: 'params', view: props.parameters, type: 'f32', access: 'read'});
    }
    if (props.output.weights) {
      bindings.push({
        name: 'weights',
        view: props.output.weights,
        type: 'f32',
        access: 'read_write'
      });
    }
    validateGraphViewsBelongToGraph(
      id,
      graph,
      bindings.map(binding => binding.view)
    );
    const targetVariable = props.mode === 'co-kriging' ? (props.targetVariable ?? 0) : 0;
    const factors = props.coregionalization ?? [1, 0, 0, 1];
    const means = props.means ?? Array.from({length: this.variableCount}, () => 0);
    const sampleDriftStride = props.sampleDrift?.format === 'float32' ? driftCount : 4;
    const queryDriftStride = props.queryDrift?.format === 'float32' ? driftCount : 4;
    const drift = this.suppliedDrift
      ? `sampleDrift[sampleDriftOffset + neighbors[row] * ${sampleDriftStride}u + column]`
      : props.mode === 'co-kriging'
        ? 'select(0.0, 1.0, u32(observation.z) == column)'
        : props.mode === 'universal'
          ? 'select((read_positions(neighbors[row]) - center)[min(column - 1u, 1u)] / coordinateScale, 1.0, column == 0u)'
          : '1.0';
    const queryDrift = this.suppliedDrift
      ? `queryDrift[queryDriftOffset + index * ${queryDriftStride}u + column]`
      : props.mode === 'co-kriging'
        ? 'select(0.0, 1.0, TARGET_VARIABLE == column)'
        : 'select(0.0, 1.0, column == 0u)';
    const structures =
      props.coregionalizationModels ??
      (props.covariance ? [{covariance: props.covariance, loadings: factors}] : []);
    // The system is divided by the largest structure variance (times its cross loading), so the
    // covariance block is O(1) next to the unit-scale drift rows for any sill from 1e-4 to 1e6.
    // Weights are unchanged, the Lagrange multipliers scale by the same constant.
    const covarianceScale = structures.reduce((largest, structure) => {
      let loadingScale = 1;
      if (props.mode === 'co-kriging') {
        loadingScale = 0;
        for (let pair = 0; pair < this.variableCount ** 2; pair++) {
          loadingScale = Math.max(
            loadingScale,
            Math.abs(getCoregionalizationCoefficient(structure.loadings, this.variableCount, pair))
          );
        }
      }
      const variance = structure.covariance.sill + (structure.covariance.nugget ?? 0);
      return Math.max(largest, variance * loadingScale);
    }, 0);
    const crossModel =
      props.mode === 'co-kriging'
        ? structures
            .map((structure, structureIndex) => {
              const coefficients = Array.from({length: this.variableCount ** 2}, (_, pair) =>
                getWGSLFloatLiteral(
                  getCoregionalizationCoefficient(structure.loadings, this.variableCount, pair)
                )
              );
              return `${getPredictionCovarianceWGSL(structure.covariance).replace('fn covariance(', `fn covariance${structureIndex}(`)}\nconst CROSS${structureIndex}: array<f32, ${coefficients.length}> = array<f32, ${coefficients.length}>(${coefficients.join(', ')});`;
            })
            .join('\n')
        : '';
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPULocalKriging',
        variant: props.mode,
        bindings,
        invocationCount: queryCount,
        workgroupSize: 32,
        declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4, supports: props.supports.format === 'float32x4' ? 4 : 2}, ['results'])}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getEquilibratedSolveWGSL(neighborCount + driftCount)}\n${props.parameters ? getPerFramePredictionCovarianceWGSL(this.covarianceStructureCount, this.variableCount) : getPredictionCovarianceWGSL(props.covariance!)}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const DRIFT_COUNT: u32 = ${driftCount}u;
const SUPPORT_COUNT: u32 = ${supportCount}u;
const SAMPLE_COUNT: u32 = ${props.positions.length}u;
const TARGET_VARIABLE: u32 = ${targetVariable}u;
const MEANS: array<f32, ${means.length}> = array<f32, ${means.length}>(${means.map(getWGSLFloatLiteral).join(', ')});
${
  props.parameters
    ? ''
    : `const COVARIANCE_SCALE: f32 = ${getWGSLFloatLiteral(covarianceScale > 0 && Number.isFinite(covarianceScale) ? covarianceScale : 1)};
${crossModel}
fn crossCovariance(first: vec2<f32>, second: vec2<f32>, firstVariable: u32, secondVariable: u32) -> f32 {
  return ${props.mode === 'co-kriging' ? structures.map((_, structureIndex) => `covariance${structureIndex}(first, second) * CROSS${structureIndex}[firstVariable * ${this.variableCount}u + secondVariable]`).join(' + ') : 'covariance(first, second)'};
}`
}`,
        body: `
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
${props.output.weights ? 'for (var column = 0u; column < NEIGHBOR_COUNT; column++) { weights[weightsOffset + index * NEIGHBOR_COUNT + column] = 0.0; }' : ''}
${props.parameters ? 'if (!loadCovarianceParameters()) { return; }\nlet covarianceScale = params[paramsOffset];' : 'let covarianceScale = COVARIANCE_SCALE;'}
var center = vec2<f32>(0.0);
var supportWeightSum = 0.0;
var supportWeights: array<f32, ${supportCount}>;
for (var support = 0u; support < SUPPORT_COUNT; support++) {
  let supportValue = read_supports(index * SUPPORT_COUNT + support);
  let point = supportValue.xy;
  let weight = ${props.supports.format === 'float32x4' ? 'supportValue.z' : '1.0'};
  if (!isFiniteValue(weight) || weight < 0.0) { return; }
  supportWeights[support] = weight; supportWeightSum += weight;
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  center += point * weight;
}
if (!(supportWeightSum > 0.0) || !isFiniteValue(supportWeightSum)) { return; }
center /= supportWeightSum;
for (var support = 0u; support < SUPPORT_COUNT; support++) { supportWeights[support] /= supportWeightSum; }
var neighbors: array<u32, ${neighborCount}>;
var distances: array<f32, ${neighborCount}>;
var retained = 0u;
${props.neighborhoodOffsets ? `let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + index];\nlet candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + index + 1u];\nif (candidateBegin > candidateEnd || candidateEnd > ${props.neighborhoods!.length}u) { return; }` : ''}
for (var candidate = ${props.neighborhoodOffsets ? 'candidateBegin' : '0u'}; candidate < ${props.neighborhoodOffsets ? 'candidateEnd' : props.neighborhoods ? `${props.neighborhoodCapacity}u` : 'SAMPLE_COUNT'}; candidate++) {
  let sample = ${props.neighborhoods ? `neighborhoods[neighborhoodsOffset + ${props.neighborhoodOffsets ? '' : `index * ${props.neighborhoodCapacity}u + `}candidate]` : 'candidate'};
  if (sample >= SAMPLE_COUNT) { continue; }
  let observation = read_observations(sample);
  let position = read_positions(sample);
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { continue; }
  ${props.mode === 'co-kriging' ? `if (!isFiniteValue(observation.z) || observation.z < 0.0 || observation.z >= ${this.variableCount}.0 || floor(observation.z) != observation.z) { continue; }` : ''}
  ${this.suppliedDrift ? `var usable = true; for (var column = 0u; column < DRIFT_COUNT; column++) { usable = usable && isFiniteValue(sampleDrift[sampleDriftOffset + sample * ${sampleDriftStride}u + column]); } if (!usable) { continue; }` : ''}
  let separation = distance(position, center);
  if (!(separation <= ${getWGSLFloatLiteral(Number.isFinite(props.searchRadius) ? props.searchRadius! : 3.402823e38)})) { continue; }
  var insertion = retained;
  for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor]) { insertion = neighbor; break; } }
  if (insertion >= NEIGHBOR_COUNT) { continue; }
  var shift = min(retained, NEIGHBOR_COUNT - 1u);
  loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
  neighbors[insertion] = sample; distances[insertion] = separation; retained = min(retained + 1u, NEIGHBOR_COUNT);
}
if (retained == 0u || retained < DRIFT_COUNT) { return; }
let coordinateScale = max(distances[retained - 1u], 1e-20);
var crossValues: array<f32, ${neighborCount}>;
var queryDesign: array<f32, ${Math.max(driftCount, 1)}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let variable = ${props.mode === 'co-kriging' ? 'u32(observation.z)' : '0u'};
  for (var column = 0u; column < retained; column++) {
    let otherObservation = read_observations(neighbors[column]);
    matrix[row * SYSTEM_CAPACITY + column] = crossCovariance(position, read_positions(neighbors[column]), variable, ${props.mode === 'co-kriging' ? 'u32(otherObservation.z)' : '0u'}) / covarianceScale;
  }
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  var crossValue = 0.0;
  for (var support = 0u; support < SUPPORT_COUNT; support++) { crossValue += crossCovariance(position, read_supports(index * SUPPORT_COUNT + support).xy, variable, TARGET_VARIABLE) * supportWeights[support]; }
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
  for (var column = 0u; column < DRIFT_COUNT; column++) {
    let value = ${drift};
    matrix[row * SYSTEM_CAPACITY + retained + column] = value;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = value;
  }
}
for (var column = 0u; column < DRIFT_COUNT; column++) {
  let value = ${queryDrift};
  if (!isFiniteValue(value)) { return; }
  queryDesign[column] = value; right[retained + column] = value;
  for (var other = 0u; other < DRIFT_COUNT; other++) { matrix[(retained + column) * SYSTEM_CAPACITY + retained + other] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + DRIFT_COUNT)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
var prediction = ${driftCount ? '0.0' : 'MEANS[TARGET_VARIABLE]'};
var variance = 0.0;
for (var first = 0u; first < SUPPORT_COUNT; first++) {
  for (var second = 0u; second < SUPPORT_COUNT; second++) {
    variance += crossCovariance(read_supports(index * SUPPORT_COUNT + first).xy, read_supports(index * SUPPORT_COUNT + second).xy, TARGET_VARIABLE, TARGET_VARIABLE) * supportWeights[first] * supportWeights[second];
  }
}
let varianceScale = max(variance, 1e-20);
for (var row = 0u; row < retained; row++) {
  let observation = read_observations(neighbors[row]);
  prediction += solution[row] * (observation.x ${driftCount ? '' : `- MEANS[${props.mode === 'co-kriging' ? 'u32(observation.z)' : '0u'}]`});
  variance -= solution[row] * crossValues[row];
}
for (var column = 0u; column < DRIFT_COUNT; column++) { variance -= solution[retained + column] * queryDesign[column] * covarianceScale; }
if (!isFiniteValue(prediction) || !isFiniteValue(variance) || variance < -1e-4 * varianceScale) { return; }
write_results(index, vec4<f32>(prediction, max(variance, 0.0), pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;
${props.output.weights ? 'for (var row = 0u; row < retained; row++) { weights[weightsOffset + index * NEIGHBOR_COUNT + row] = solution[row]; }' : ''}`
      })
    ];
  }
}
