// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {SpaceTimeVariogramModel} from '../../dataframe-analysis/pair-statistics/space-time-variogram-model';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {
  getGPUSpaceTimeKrigingParameterValues,
  getPerFrameSpaceTimeCovarianceWGSL,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
} from './prediction-covariance';
import {
  createVecchiaLogLikelihoodSumNodes,
  type GPUVecchiaLikelihoodOutput
} from './vecchia-likelihood-nodes';
import {VECCHIA_NUGGET_FLOOR_FRACTION} from './vecchia-parameters';
import type {GPUPackedReadbackValues} from '../../analysis-core/kernels/packed-readback';
import {decodeVecchiaStatus} from './vecchia-readback';
import {decodeGPUVecchiaLogLikelihoodResult} from './gpu-vecchia-log-likelihood';

const OPERATION = 'GPUVecchiaSpaceTimeLogLikelihood';
const MAXIMUM_NEIGHBOR_COUNT = MAXIMUM_VECCHIA_NEIGHBOR_COUNT;
/** Number of float32 elements in a space-time Vecchia parameter view. */
export const GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH = GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH;

/** Output layout shared with the spatial likelihood. */
export type GPUVecchiaSpaceTimeLogLikelihoodOutput = GPUVecchiaLikelihoodOutput;

/** Decodes space-time solve counts and optional input-neighborhood loss evidence. */
export function decodeGPUVecchiaSpaceTimeLogLikelihoodStatus(
  status: GPUPackedReadbackValues,
  neighborhoodStatus?: GPUPackedReadbackValues
) {
  return decodeVecchiaStatus(
    'decodeGPUVecchiaSpaceTimeLogLikelihoodStatus',
    status,
    neighborhoodStatus
  );
}

/** Space-time totals have the same port element types as the spatial likelihood. */
export function decodeGPUVecchiaSpaceTimeLogLikelihoodResult(
  ports: Parameters<typeof decodeGPUVecchiaLogLikelihoodResult>[0]
) {
  return decodeGPUVecchiaLogLikelihoodResult(ports);
}

/** Rows `[value, observationErrorVariance, reserved, validity]`; validity must be strictly `> 0` (zero, negative and NaN exclude), as for the spatial likelihood. */
export type GPUVecchiaSpaceTimeLogLikelihoodProps = {
  id?: string;
  /** Planar coordinates only; project longitude and latitude first. */
  spatialContext?: GPUSpatialContext;
  /** Context of `times`, encoding `float32-relative`. */
  temporalContext: GPUTemporalContext;
  positions: GraphDataView<'float32x2'>;
  /** Relative float32 time per row. */
  times: GraphDataView<'float32'>;
  /** Rows [value, observationErrorVariance, reserved, validity]; strictly > 0 includes, zero/negative/NaN exclude. */
  observations: GraphDataView<'float32x4'>;
  neighborhoodOffsets: GraphDataView<'uint32'>;
  neighborhoods: GraphDataView<'uint32'>;
  /** Largest conditioning set, 1..32. Compile time. */
  neighborCount: number;
  /** Optional producer loss flags; any supplied loss suppresses dependent totals. */
  neighborhoodStatus?: VecchiaNeighborhoodStatus;
  /** Written by `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues`. Per-frame. */
  parameters: GraphDataView<'float32'>;
  output: GPUVecchiaLikelihoodOutput;
};

/**
 * Space-time Vecchia log-likelihood over caller-ordered conditioning sets, with the per-frame
 * `SpaceTimeVariogramModel` covariance of `GPUSpaceTimeKriging`. A site's conditioning set comes
 * straight from `neighborhoods` (no search), and it solves the same equilibrated small system as
 * kriging; the likelihood terms and sum are the shared spatial ones, with the same failure rule.
 * A site with no usable conditioning row (an empty set, or every listed row excluded or
 * non-finite) contributes its marginal term.
 *
 * Bindings are limited to the core eight storage buffers per stage by reading site positions and
 * times directly; there is no separate query view.
 *
 * The GPU route accepts at most 32 usable neighbors per site (MAXIMUM_VECCHIA_NEIGHBOR_COUNT),
 * taking the first usable CSR entries. CPU evaluation uses all usable slots.
 * Compile-time: row count, conditioning-set width. Per-frame: every covariance parameter.
 */
export class GPUVecchiaSpaceTimeLogLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVecchiaSpaceTimeLogLikelihoodProps;
  readonly rowCount: number;
  readonly temporalContext: GPUTemporalContext;

  constructor(props: GPUVecchiaSpaceTimeLogLikelihoodProps) {
    validateVecchiaPorts(props.id ?? 'vecchia-space-time-log-likelihood', props, false, true);
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
        'spatialContext',
        'temporalContext',
        'times'
      ],
      ['conditionals', 'conditionalStatus', 'siteTerms', 'siteStatus', 'logLikelihood', 'status']
    );
    this.props = props = snapshotVecchiaLossProps(originalProps, () => props);
    const lossViews = getInterpolationNeighborhoodLossViews(
      props.id ?? 'vecchia-space-time-log-likelihood',
      props
    );
    this.id = props.id ?? 'vecchia-space-time-log-likelihood';
    const {id} = this;
    const {positions, times, observations, neighborhoodOffsets, neighborhoods, parameters, output} =
      props;
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    this.temporalContext = resolveGPUTemporalContext(id, props.temporalContext, {
      encodings: ['float32-relative'] as const
    });
    this.rowCount = positions.length;
    if (output.neighborhoodStatus)
      validateViewLength(id, 'output.neighborhoodStatus', output.neighborhoodStatus, 3);
    validateCapacity(id, 'neighborCount', props.neighborCount, 1, MAXIMUM_NEIGHBOR_COUNT);
    validateViewLength(id, 'times', times, this.rowCount);
    validateViewLength(id, 'observations', observations, this.rowCount);
    validateViewLength(id, 'neighborhoodOffsets', neighborhoodOffsets, this.rowCount + 1);
    validateViewLength(id, 'parameters', parameters, GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH);
    validateViewLength(id, 'output.conditionals', output.conditionals, this.rowCount);
    validateViewLength(id, 'output.conditionalStatus', output.conditionalStatus, this.rowCount);
    validateViewLength(id, 'output.siteTerms', output.siteTerms, this.rowCount);
    validateViewLength(id, 'output.siteStatus', output.siteStatus, this.rowCount);
    validateViewLength(id, 'output.logLikelihood', output.logLikelihood, 2);
    validateViewLength(id, 'output.status', output.status, 3);
    validateInterpolationOutputPasses(id, output, [
      {write: {conditionals: output.conditionals, conditionalStatus: output.conditionalStatus}},
      {
        read: {conditionals: output.conditionals, conditionalStatus: output.conditionalStatus},
        write: {siteTerms: output.siteTerms, siteStatus: output.siteStatus, status: output.status}
      },
      {read: {status: output.status}, write: {logLikelihood: output.logLikelihood}},
      {write: {logLikelihood: output.logLikelihood, neighborhoodStatus: output.neighborhoodStatus}}
    ]);
    validateInterpolationOutputsDisjointFromInputs(id, Object.values(output), [
      positions,
      observations,
      neighborhoodOffsets,
      neighborhoods,
      parameters,
      times,
      ...lossViews
    ]);
    validateInterpolationPredictionViews(
      id,
      [
        positions,
        times,
        observations,
        neighborhoodOffsets,
        neighborhoods,
        parameters,
        ...lossViews
      ],
      [
        output.conditionals,
        output.conditionalStatus,
        output.siteTerms,
        output.siteStatus,
        output.logLikelihood,
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
      times: props.times,
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
      3 * 4,
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
      times,
      observations,
      neighborhoodOffsets,
      neighborhoods,
      parameters,
      neighborCount,
      output
    } = props;
    const bindings = [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'times', view: times, type: 'f32', access: 'read'},
      {name: 'observations', view: observations, type: 'f32', access: 'read'},
      {name: 'offsets', view: neighborhoodOffsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: neighborhoods, type: 'u32', access: 'read'},
      {name: 'params', view: parameters, type: 'f32', access: 'read'},
      {name: 'results', view: output.conditionals, type: 'f32', access: 'read_write'},
      {name: 'statuses', view: output.conditionalStatus, type: 'u32', access: 'read_write'}
    ] as const;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-solve`,
        operation: OPERATION,
        variant: 'space-time-solve',
        bindings,
        invocationCount: rowCount,
        workgroupSize: 32,
        declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4}, ['results'])}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
${getEquilibratedSolveWGSL(neighborCount)}
${getPerFrameSpaceTimeCovarianceWGSL()}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const SAMPLE_COUNT: u32 = ${positions.length}u;`,
        body: `
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
if (!loadSpaceTimeCovariance()) { return; }
let covarianceScale = params[paramsOffset];
let sitePosition = read_positions(index);
let siteTime = times[timesOffset + index];
if (!isFiniteValue(sitePosition.x) || !isFiniteValue(sitePosition.y) || !isFiniteValue(siteTime)) { return; }
var rows: array<u32, ${neighborCount}>;
var retained = 0u;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u] && retained < NEIGHBOR_COUNT; slot++) {
  let sample = neighbors[neighborsOffset + slot];
  if (sample >= SAMPLE_COUNT) { continue; }
  let observation = read_observations(sample);
  let position = read_positions(sample);
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y) || !isFiniteValue(times[timesOffset + sample])) { continue; }
  rows[retained] = sample;
  retained++;
}
if (retained == 0u) {
  // No usable conditioning row (an empty set, or every listed row excluded): the site is marginal,
  // as in GPUVecchiaFisherScoring. Mean 0, variance C(0, 0), no system solved.
  let marginalVariance = spaceTimeCovariance(0.0, 0.0, true, true);
  if (isFiniteValue(marginalVariance)) {
    write_results(index, vec4<f32>(0.0, marginalVariance, 1.0, 0.0));
    statuses[statusesOffset + index] = 0u;
  }
  return;
}
var crossValues: array<f32, ${neighborCount}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(rows[row]);
  let rowTime = times[timesOffset + rows[row]];
  for (var column = 0u; column < retained; column++) {
    let otherPosition = read_positions(rows[column]);
    let otherTime = times[timesOffset + rows[column]];
    matrix[row * SYSTEM_CAPACITY + column] = spaceTimeCovariance(distance(position, otherPosition), abs(rowTime - otherTime), all(position == otherPosition), rowTime == otherTime) / covarianceScale;
  }
  let observation = read_observations(rows[row]);
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  let crossValue = spaceTimeCovariance(distance(position, sitePosition), abs(rowTime - siteTime), all(position == sitePosition), rowTime == siteTime);
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
var mean = 0.0;
var variance = spaceTimeCovariance(0.0, 0.0, true, true);
for (var row = 0u; row < retained; row++) {
  mean += solution[row] * read_observations(rows[row]).x;
  variance -= solution[row] * crossValues[row];
}
if (!isFiniteValue(mean) || !isFiniteValue(variance)) { return; }
write_results(index, vec4<f32>(mean, variance, pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;`
      }),
      ...createVecchiaLogLikelihoodSumNodes(graph, {
        id,
        operation: OPERATION,
        rowCount,
        observations,
        neighborhoodOffsets,
        parameters,
        output
      }),
      ...createVecchiaNeighborhoodStatusNodes(
        graph,
        id,
        props.neighborhoodStatus,
        output.logLikelihood,
        output.neighborhoodStatus
      )
    ];
  }
}

/** Whether a space-time model has a smooth (Matern 3/2 or smoother) component needing the floor. */
function hasSmoothComponent(model: SpaceTimeVariogramModel): boolean {
  const isSmooth = (name: string): boolean => name !== 'exponential' && name !== 'spherical';
  switch (model.type) {
    case 'separable':
    case 'product-sum':
      return isSmooth(model.spatial.model) || isSmooth(model.temporal.model);
    case 'metric':
      return isSmooth(model.joint.model);
    case 'sum-metric':
      return (
        isSmooth(model.spatial.model) ||
        isSmooth(model.temporal.model) ||
        isSmooth(model.joint.model)
      );
    case 'gneiting':
      return model.alpha > 0.5 || model.gamma > 0.5;
    default:
      return true;
  }
}

/** Exact coincident discontinuity as both spatial and temporal lags approach zero from above. */
function getSpaceTimeCoincidentNugget(model: SpaceTimeVariogramModel): number {
  switch (model.type) {
    case 'separable':
      return (
        model.sill * (model.spatial.nugget + (1 - model.spatial.nugget) * model.temporal.nugget)
      );
    case 'product-sum': {
      const spatialNugget = model.spatial.nugget;
      const temporalNugget = model.temporal.nugget;
      // Expand C(0,0) - C(0+,0+) into nonnegative terms, avoiding cancellation.
      return (
        spatialNugget +
        temporalNugget +
        model.k *
          (spatialNugget * temporalNugget +
            spatialNugget * model.temporal.sill +
            model.spatial.sill * temporalNugget)
      );
    }
    case 'metric':
      return model.joint.nugget;
    case 'sum-metric':
      return model.spatial.nugget + model.temporal.nugget + model.joint.nugget;
    case 'gneiting':
      return model.nugget;
  }
}

/**
 * Packs per-frame parameters of `GPUVecchiaSpaceTimeLogLikelihood`. Enforces the float32 nugget
 * floor: a model with a Matern 3/2 or smoother component needs a nugget at a coincident space-time
 * point of at least `VECCHIA_NUGGET_FLOOR_FRACTION` of the total variance.
 *
 * @param model Space-time covariance.
 * @param target Optional destination of at least 28 elements.
 * @param searchAnisotropy Space units per unit of time, needed when the model has none.
 * @throws If the model has a smooth component and its nugget is below the floor.
 */
export function getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
  model: SpaceTimeVariogramModel,
  target?: Float32Array,
  searchAnisotropy: number = 1
): Float32Array {
  const values = getGPUSpaceTimeKrigingParameterValues({covariance: model, searchAnisotropy});
  const total = values[0];
  const nugget = getSpaceTimeCoincidentNugget(model);
  validateRequirement(
    OPERATION,
    !hasSmoothComponent(model) || nugget >= VECCHIA_NUGGET_FLOOR_FRACTION * total * (1 - 1e-6),
    `nugget must be at least ${VECCHIA_NUGGET_FLOOR_FRACTION} of the total variance for a smooth component`
  );
  if (target) {
    validateRequirement(
      OPERATION,
      target.length >= values.length,
      `target must hold at least ${values.length} elements; received ${target.length}`
    );
    target.set(values);
    return target;
  }
  return values;
}
