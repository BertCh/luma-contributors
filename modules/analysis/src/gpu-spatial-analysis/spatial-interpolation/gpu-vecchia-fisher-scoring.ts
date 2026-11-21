// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPackedReadbackValues} from '../../analysis-core/kernels/packed-readback';
import {getVecchiaReadbackValues, decodeVecchiaStatus} from './vecchia-readback';
import {
  snapshotVecchiaLossProps,
  validateVecchiaPorts,
  createVecchiaNeighborhoodStatusNodes,
  type VecchiaNeighborhoodStatus
} from './vecchia-neighborhood-status';
import {MAXIMUM_VECCHIA_NEIGHBOR_COUNT} from './vecchia-conditioning-sets';
import {getInterpolationNeighborhoodLossViews} from './interpolation-neighborhood-loss';

import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';

import {
  validateInterpolationPredictionViews,
  validateInterpolationOutputPasses,
  validateInterpolationOutputsDisjointFromInputs
} from './interpolation-view-validation';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {
  getPerFramePredictionCovarianceWGSL,
  MATERN_3_2_RANGE_FACTOR
} from './prediction-covariance';
import {createVecchiaFixedPointSumNodes} from './vecchia-likelihood-nodes';
import {GPU_VECCHIA_PARAMETER_LENGTH} from './vecchia-parameters';

const OPERATION = 'GPUVecchiaFisherScoring';
const MAXIMUM_NEIGHBOR_COUNT = MAXIMUM_VECCHIA_NEIGHBOR_COUNT;
/** Number of parameters: log sill, log effective range, log nugget. */
const PARAMETER_COUNT = 3;
/** Summed cells: the log-likelihood, then the score, then the upper triangle of the information. */
export const GPU_VECCHIA_FISHER_CELL_COUNT = 10;

/** Outputs of {@link GPUVecchiaFisherScoring}, all caller-owned. */
export type GPUVecchiaFisherScoringOutput = {
  /** `rowCount * 10` elements: per site `[logLik, score(3), information(6)]`. NaN for excluded or failed. */
  siteTerms: GraphDataView<'float32'>;
  /** One word per site: 0 used, 1 excluded, 2 solve failed, 3 variance nonpositive or non-finite. */
  siteStatus: GraphDataView<'uint32'>;
  /**
   * 20 elements, `[high, low]` per cell of `[logLik, score(3), information(6)]`; each value is
   * `high + low`. Information is `I_00, I_01, I_02, I_11, I_12, I_22`. NaN if any site failed.
   */
  totals: GraphDataView<'float32'>;
  /** Three words `[failedSiteCount, excludedSiteCount, usedSiteCount]`. */
  status: GraphDataView<'uint32'>;
  /** Optional three words [complete, overflow, candidateOverflow] for the input CSR. */
  neighborhoodStatus?: GraphDataView<'uint32'>;
};

/** Distances are planar Euclidean in the native units of the positions. */
const PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance' as const
});

/** Props of {@link GPUVecchiaFisherScoring}. */
export type GPUVecchiaFisherScoringProps = {
  id?: string;
  /**
   * Coordinate space of the positions. Compile-time. Omitted means native planar coordinates with
   * Euclidean distance (unchanged behavior); only that is accepted, so longitude/latitude throws:
   * project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Rows `[value, observationErrorVariance, reserved, validity (> 0 includes; zero, negative and NaN exclude)]`, mean zero. */
  observations: GraphDataView<'float32x4'>;
  neighborhoodOffsets: GraphDataView<'uint32'>;
  neighborhoods: GraphDataView<'uint32'>;
  /** Largest conditioning set, 1..32. Compile time. */
  neighborCount: number;
  /** Optional producer loss flags; any supplied loss suppresses dependent totals. */
  neighborhoodStatus?: VecchiaNeighborhoodStatus;
  /** Written by `getGPUVecchiaLogLikelihoodParameterValues`. Per-frame. */
  parameters: GraphDataView<'float32'>;
  output: GPUVecchiaFisherScoringOutput;
};

/**
 * One Fisher scoring evaluation of a Vecchia likelihood (Guinness 2019): the log-likelihood, its
 * score with respect to `theta = (log sill, log effective range, log nugget)` and the expected
 * information. Per site, with `b = K_cc^-1 k_ci`, `d = k_ii - k_ci' b`, `r = y_i - b' y_c`,
 * `v_j = dk_ci/dtheta_j - dK_cc/dtheta_j b`, `w_j = K_cc^-1 v_j` and
 * `dd_j = dk_ii/dtheta_j - dk_ci'/dtheta_j b - b' v_j`:
 *
 *   score_j = -1/2 (dd_j / d) (1 - r^2 / d) + (r / d) w_j' y_c
 *   info_jk = 1/2 (dd_j dd_k) / d^2 + (v_j' w_k) / d
 *
 * The small system is rebuilt and solved four times per site (`b` and each `w_j`) with the shared
 * equilibrated solve, since the solver overwrites the matrix. Totals are fixed-point sums, so they
 * are exact for the float32 per-site terms. Exponential and Matern 3/2 only: other model codes
 * have no range derivative here and report status 3.
 *
 * The GPU route accepts at most 32 usable neighbors per site (MAXIMUM_VECCHIA_NEIGHBOR_COUNT),
 * taking the first usable CSR entries. CPU evaluation uses all usable slots.
 * Compile-time: row count, conditioning-set width. Per-frame: the covariance `parameters`.
 */
export class GPUVecchiaFisherScoring implements GPUCommandNodeProducer {
  readonly id: string;
  /** Resolved coordinate space (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  readonly props: GPUVecchiaFisherScoringProps;
  readonly rowCount: number;

  constructor(props: GPUVecchiaFisherScoringProps) {
    validateVecchiaPorts(props.id ?? 'vecchia-fisher-scoring', props, true, false);
    const originalProps = props;
    this.props = props = snapshotInterpolationConstructionProps(
      props,
      [
        'id',
        'neighborCount',
        'neighborhoodOffsets',
        'neighborhoods',
        'observations',
        'output',
        'parameters',
        'positions',
        'spatialContext'
      ],
      ['siteTerms', 'siteStatus', 'totals', 'status']
    );
    this.props = props = snapshotVecchiaLossProps(originalProps, () => props);
    const lossViews = getInterpolationNeighborhoodLossViews(
      props.id ?? 'vecchia-fisher-scoring',
      props
    );
    this.id = props.id ?? 'vecchia-fisher-scoring';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT
    );
    const {id} = this;
    const {positions, observations, neighborhoodOffsets, neighborhoods, parameters, output} = props;
    this.rowCount = positions.length;
    if (output.neighborhoodStatus)
      validateViewLength(id, 'output.neighborhoodStatus', output.neighborhoodStatus, 3);
    validateCapacity(id, 'neighborCount', props.neighborCount, 1, MAXIMUM_NEIGHBOR_COUNT);
    validateViewLength(id, 'observations', observations, this.rowCount);
    validateViewLength(id, 'neighborhoodOffsets', neighborhoodOffsets, this.rowCount + 1);
    validateViewLength(id, 'parameters', parameters, GPU_VECCHIA_PARAMETER_LENGTH);
    validateViewLength(
      id,
      'output.siteTerms',
      output.siteTerms,
      this.rowCount * GPU_VECCHIA_FISHER_CELL_COUNT
    );
    validateViewLength(id, 'output.siteStatus', output.siteStatus, this.rowCount);
    validateViewLength(id, 'output.totals', output.totals, GPU_VECCHIA_FISHER_CELL_COUNT * 2);
    validateViewLength(id, 'output.status', output.status, 3);
    validateInterpolationOutputPasses(id, output, [
      {write: {siteTerms: output.siteTerms, siteStatus: output.siteStatus, status: output.status}},
      {read: {status: output.status}, write: {totals: output.totals}},
      {write: {totals: output.totals, neighborhoodStatus: output.neighborhoodStatus}}
    ]);
    validateInterpolationOutputsDisjointFromInputs(id, Object.values(output), [
      positions,
      observations,
      neighborhoodOffsets,
      neighborhoods,
      parameters,
      ...lossViews
    ]);
    validateInterpolationPredictionViews(
      id,
      [positions, observations, neighborhoodOffsets, neighborhoods, parameters, ...lossViews],
      [
        output.siteTerms,
        output.siteStatus,
        output.totals,
        output.status,
        ...(output.neighborhoodStatus ? [output.neighborhoodStatus] : [])
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      observations: props.observations,
      neighborhoodOffsets: props.neighborhoodOffsets,
      neighborhoods: props.neighborhoods,
      parameters: props.parameters,
      ...Object.fromEntries(
        getInterpolationNeighborhoodLossViews(id, props).map((view, index) => [
          `neighborhoodLoss${index}`,
          view
        ])
      ),
      ...props.output
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'fixed-point words',
      GPU_VECCHIA_FISHER_CELL_COUNT * 3 * 4,
      'reduce the likelihood cell count'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'fixed-point scale',
      4,
      'use a device with storage bindings'
    );
    const {
      positions,
      observations,
      neighborhoodOffsets,
      neighborhoods,
      parameters,
      output,
      neighborCount
    } = props;
    const bindings = [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'observations', view: observations, type: 'f32', access: 'read'},
      {name: 'offsets', view: neighborhoodOffsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: neighborhoods, type: 'u32', access: 'read'},
      {name: 'params', view: parameters, type: 'f32', access: 'read'},
      {name: 'siteTerms', view: output.siteTerms, type: 'f32', access: 'read_write'},
      {name: 'siteStatus', view: output.siteStatus, type: 'u32', access: 'read_write'},
      {name: 'counters', view: output.status, type: 'atomic<u32>', access: 'read_write'}
    ] as const;
    const cellCount = GPU_VECCHIA_FISHER_CELL_COUNT;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-clear-counts`,
        operation: OPERATION,
        variant: 'clear-counts',
        bindings: [{name: 'counters', view: output.status, type: 'u32', access: 'read_write'}],
        invocationCount: 3,
        workgroupSize: 4,
        body: 'counters[countersOffset + index] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-site-terms`,
        operation: OPERATION,
        variant: 'fisher-site-terms',
        bindings,
        invocationCount: rowCount,
        workgroupSize: 32,
        declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4})}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
${getEquilibratedSolveWGSL(neighborCount)}
${getPerFramePredictionCovarianceWGSL(1, 1)}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const SAMPLE_COUNT: u32 = ${positions.length}u;
const PARAMETER_COUNT: u32 = ${PARAMETER_COUNT}u;
const CELL_COUNT: u32 = ${cellCount}u;
const LOG_TWO_PI: f32 = ${getWGSLFloatLiteral(Math.log(2 * Math.PI))};
const MATERN_3_2_RANGE_FACTOR: f32 = ${getWGSLFloatLiteral(MATERN_3_2_RANGE_FACTOR)};
var<private> sitePositions: array<vec2<f32>, ${neighborCount}>;
var<private> siteValues: array<f32, ${neighborCount}>;
var<private> siteErrors: array<f32, ${neighborCount}>;
var<private> weightsB: array<f32, ${neighborCount}>;
var<private> crossK: array<f32, ${neighborCount}>;
var<private> solvedW: array<f32, ${PARAMETER_COUNT * neighborCount}>;
var<private> rawV: array<f32, ${PARAMETER_COUNT * neighborCount}>;
var<private> dd: array<f32, ${PARAMETER_COUNT}>;
var<private> rd: array<f32, ${PARAMETER_COUNT}>;
fn correlationRangeDerivative(modelCode: u32, scaled: f32) -> f32 {
  // -t dR/dt of the effective-range correlation; other codes produce NaN and a status 3.
  if (modelCode == 1u) { let a = 3.0 * scaled; return a * exp(-a); }
  if (modelCode == 3u) { let a = MATERN_3_2_RANGE_FACTOR * scaled; return a * a * exp(-a); }
  return getNaN();
}
fn parameterDerivative(parameterIndex: u32, first: vec2<f32>, second: vec2<f32>) -> f32 {
  let values = covarianceStructures[0];
  let separation = distance(first, second) / values.w;
  let coincident = all(first == second);
  if (parameterIndex == 0u) { return values.z * covarianceCorrelation(u32(values.x), separation); }
  if (parameterIndex == 1u) { return values.z * correlationRangeDerivative(u32(values.x), separation); }
  return select(0.0, values.y, coincident);
}
fn fillSystem(count: u32, covarianceScale: f32) {
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column < count; column++) {
      matrix[row * SYSTEM_CAPACITY + column] = crossCovariance(sitePositions[row], sitePositions[column], 0u, 0u) / covarianceScale;
    }
    matrix[row * SYSTEM_CAPACITY + row] += siteErrors[row] / covarianceScale;
  }
}`,
        body: `for (var cell = 0u; cell < CELL_COUNT; cell++) { siteTerms[siteTermsOffset + index * CELL_COUNT + cell] = getNaN(); }
let observation = read_observations(index);
if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0) {
  siteStatus[siteStatusOffset + index] = 1u;
  atomicAdd(&counters[countersOffset + 1u], 1u);
  return;
}
siteStatus[siteStatusOffset + index] = 2u;
if (!loadCovarianceParameters()) { atomicAdd(&counters[countersOffset], 1u); return; }
let covarianceScale = params[paramsOffset];
let sitePosition = read_positions(index);
if (!isFiniteValue(sitePosition.x) || !isFiniteValue(sitePosition.y)) { atomicAdd(&counters[countersOffset], 1u); return; }
var count = 0u;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u] && count < NEIGHBOR_COUNT; slot++) {
  let sample = neighbors[neighborsOffset + slot];
  if (sample >= SAMPLE_COUNT) { continue; }
  let neighborObservation = read_observations(sample);
  let neighborPosition = read_positions(sample);
  if (!(neighborObservation.w > 0.0) || !isFiniteValue(neighborObservation.x) || !isFiniteValue(neighborObservation.y) || neighborObservation.y < 0.0 || !isFiniteValue(neighborPosition.x) || !isFiniteValue(neighborPosition.y)) { continue; }
  sitePositions[count] = neighborPosition;
  siteValues[count] = neighborObservation.x;
  siteErrors[count] = neighborObservation.y;
  count++;
}
if (count > 0u) {
  fillSystem(count, covarianceScale);
  for (var row = 0u; row < count; row++) {
    crossK[row] = crossCovariance(sitePositions[row], sitePosition, 0u, 0u);
    right[row] = crossK[row] / covarianceScale;
  }
  if (!solveSystem(count)) { atomicAdd(&counters[countersOffset], 1u); return; }
  for (var row = 0u; row < count; row++) { weightsB[row] = solution[row]; }
}
var conditionalVariance = crossCovariance(sitePosition, sitePosition, 0u, 0u) + observation.y;
var prediction = 0.0;
for (var row = 0u; row < count; row++) {
  conditionalVariance -= weightsB[row] * crossK[row];
  prediction += weightsB[row] * siteValues[row];
}
let residual = observation.x - prediction;
let variance = conditionalVariance;
if (!(variance > 0.0) || !isFiniteValue(variance)) { siteStatus[siteStatusOffset + index] = 3u; atomicAdd(&counters[countersOffset], 1u); return; }
for (var parameter = 0u; parameter < PARAMETER_COUNT; parameter++) {
  var dvariance = parameterDerivative(parameter, sitePosition, sitePosition);
  for (var row = 0u; row < count; row++) {
    let dk = parameterDerivative(parameter, sitePositions[row], sitePosition);
    var vectorV = dk;
    for (var column = 0u; column < count; column++) {
      vectorV -= parameterDerivative(parameter, sitePositions[row], sitePositions[column]) * weightsB[column];
    }
    rawV[parameter * NEIGHBOR_COUNT + row] = vectorV;
    dvariance -= (dk + vectorV) * weightsB[row];
  }
  dd[parameter] = dvariance;
  if (count > 0u) {
    fillSystem(count, covarianceScale);
    for (var row = 0u; row < count; row++) { right[row] = rawV[parameter * NEIGHBOR_COUNT + row] / covarianceScale; }
    if (!solveSystem(count)) { atomicAdd(&counters[countersOffset], 1u); return; }
    var quadraticW = 0.0;
    for (var row = 0u; row < count; row++) {
      solvedW[parameter * NEIGHBOR_COUNT + row] = solution[row];
      quadraticW += solution[row] * siteValues[row];
    }
    rd[parameter] = -quadraticW;
  } else {
    rd[parameter] = 0.0;
  }
}
let standardized = residual * residual / variance;
let logLikelihood = -0.5 * (LOG_TWO_PI + log(variance) + standardized);
var output: array<f32, ${cellCount}>;
output[0] = logLikelihood;
for (var first = 0u; first < PARAMETER_COUNT; first++) {
  output[1u + first] = -0.5 * (dd[first] / variance) * (1.0 - standardized) - residual * rd[first] / variance;
}
var cell = 1u + PARAMETER_COUNT;
for (var first = 0u; first < PARAMETER_COUNT; first++) {
  for (var second = first; second < PARAMETER_COUNT; second++) {
    var quadratic = 0.0;
    for (var row = 0u; row < count; row++) {
      quadratic += rawV[first * NEIGHBOR_COUNT + row] * solvedW[second * NEIGHBOR_COUNT + row];
    }
    output[cell] = 0.5 * dd[first] * dd[second] / (variance * variance) + quadratic / variance;
    cell += 1u;
  }
}
for (var entry = 0u; entry < CELL_COUNT; entry++) {
  if (!isFiniteValue(output[entry])) { siteStatus[siteStatusOffset + index] = 3u; atomicAdd(&counters[countersOffset], 1u); return; }
}
for (var entry = 0u; entry < CELL_COUNT; entry++) { siteTerms[siteTermsOffset + index * CELL_COUNT + entry] = output[entry]; }
siteStatus[siteStatusOffset + index] = 0u;
atomicAdd(&counters[countersOffset + 2u], 1u);`
      }),
      ...createVecchiaFixedPointSumNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        rowCount,
        cellCount,
        terms: output.siteTerms,
        counters: output.status,
        totals: output.totals
      }),
      ...createVecchiaNeighborhoodStatusNodes(
        graph,
        id,
        props.neighborhoodStatus,
        output.totals,
        output.neighborhoodStatus
      )
    ];
  }
}

/** Reads a {@link GPUVecchiaFisherScoring} result back into one evaluation. */
export function getVecchiaFisherEvaluation(
  totals: GPUPackedReadbackValues,
  status: GPUPackedReadbackValues
): {
  logLikelihood: number;
  score: number[];
  information: number[];
  failedSiteCount: number;
  excludedSiteCount: number;
  usedSiteCount: number;
} {
  const components = getVecchiaReadbackValues(
    'getVecchiaFisherEvaluation',
    'totals',
    totals,
    GPU_VECCHIA_FISHER_CELL_COUNT * 2,
    'float32'
  );
  status = getVecchiaReadbackValues('getVecchiaFisherEvaluation', 'status', status, 3, 'uint32');
  const read = (cell: number): number => components[cell * 2] + components[cell * 2 + 1];
  return {
    logLikelihood: status[0] > 0 ? Number.NaN : read(0),
    score: [read(1), read(2), read(3)],
    information: [read(4), read(5), read(6), read(7), read(8), read(9)],
    failedSiteCount: status[0],
    excludedSiteCount: status[1],
    usedSiteCount: status[2]
  };
}

/** Decodes solve counts separately from loss of the input CSR. */
export function decodeGPUVecchiaFisherScoringStatus(
  status: GPUPackedReadbackValues,
  neighborhoodStatus?: GPUPackedReadbackValues
) {
  return decodeVecchiaStatus('decodeGPUVecchiaFisherScoringStatus', status, neighborhoodStatus);
}

/** Reads float32 totals and uint32 status from their respective typed or raw-byte ports. */
export function decodeGPUVecchiaFisherScoringResult(ports: {
  totals: Float32Array | Uint8Array | ArrayBuffer;
  status: Uint32Array | Uint8Array | ArrayBuffer;
  neighborhoodStatus?: Uint32Array | Uint8Array | ArrayBuffer;
}) {
  const totals = getVecchiaReadbackValues(
    'decodeGPUVecchiaFisherScoringResult',
    'totals',
    ports.totals,
    20,
    'float32',
    true
  );
  const result = getVecchiaFisherEvaluation(totals, ports.status);
  const status = decodeGPUVecchiaFisherScoringStatus(ports.status, ports.neighborhoodStatus);
  status.complete =
    status.complete &&
    [result.logLikelihood, ...result.score, ...result.information].every(Number.isFinite);
  return {
    ...result,
    logLikelihood: status.complete ? result.logLikelihood : NaN,
    score: status.complete ? result.score : result.score.map(() => NaN),
    information: status.complete ? result.information : result.information.map(() => NaN),
    status
  };
}
