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
  assertCompatibleGPUTemporalContexts,
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
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedPredictionAccessors,
  validatePredictionViews
} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {
  getPerFrameSpaceTimeCovarianceWGSL,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
} from './prediction-covariance';

const MAXIMUM_NEIGHBOR_COUNT = 32;
/** Most samples: every query scans every sample in one invocation, `O(queries x samples)`. */
const GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT = 1048576;
const MAXIMUM_STAGED_CANDIDATE_COUNT = 64;
const MAXIMUM_NEIGHBORS_PER_OCTANT = 8;

/** Neighbor selection of {@link GPUSpaceTimeKriging}; see the class documentation. */
export type GPUSpaceTimeKrigingNeighborSearch = 'metric' | 'octant' | 'staged';

/** Props of {@link GPUSpaceTimeKriging}. */
export type GPUSpaceTimeKrigingProps = {
  id?: string;
  /** Planar coordinates only; project longitude and latitude first. */
  spatialContext?: GPUSpatialContext;
  /** Context of `times`, encoding `float32-relative` (`GPUTimeWordsRelative` provides one). */
  temporalContext: GPUTemporalContext;
  /** Context of `queryTimes`, default `temporalContext`; must have the same unit and epoch. */
  queryTemporalContext?: GPUTemporalContext;
  positions: GraphDataView<'float32x2'>;
  /** Relative float32 time per sample. */
  times: GraphDataView<'float32'>;
  /** Rows `[value, observationErrorVariance, reserved, validity (0 skips)]`. */
  observations: GraphDataView<'float32x4'>;
  queries: GraphDataView<'float32x2'>;
  /** Relative float32 time per query, same epoch and unit as `times`. */
  queryTimes: GraphDataView<'float32'>;
  /** Known mean or unknown constant mean. */
  mode: 'simple' | 'ordinary';
  /** Written by `getGPUSpaceTimeKrigingParameterValues`; every value changes without a recompile. */
  parameters: GraphDataView<'float32'>;
  /** Retained neighbors, 1..32 (compile time). Defaults to 16. */
  neighborCount?: number;
  /** Neighbor selection (compile time). Defaults to `'metric'`. */
  neighborSearch?: GPUSpaceTimeKrigingNeighborSearch;
  /** Candidates per octant for `'octant'`, 1..8. Defaults to `ceil(neighborCount / 8)`. */
  neighborsPerOctant?: number;
  /** Metric candidates for `'staged'`, `neighborCount..64`. Defaults to `2 * neighborCount`. */
  stagedCandidateCount?: number;
  output: {
    /** `[prediction, latent variance, pivot ratio, retained neighbor count]`. */
    values: GraphDataView<'float32x4'>;
    /** 0 success; 1 invalid or insufficient data; 2 singular or unreliable. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Local space-time kriging with one neighbor search per query, over a per-frame covariance from
 * `SpaceTimeVariogramModel` (the gstat definitions, plus Gneiting). Uses the same equilibrated
 * partial-pivot solve and the same pivot-ratio limit as `GPULocalKriging`, and keeps that
 * contributor's WGSL unchanged. Samples are scanned for every query.
 *
 * Neighbor search, compile-time:
 * - `'metric'`: the `neighborCount` samples nearest in `sqrt(h^2 + (searchAnisotropy u)^2)`.
 * - `'octant'`: per quadrant of the sample offset and per side in time (`time < query` is the past,
 *   otherwise the future), at most `neighborsPerOctant` nearest in the metric; then the
 *   `neighborCount` nearest of those. A sample exactly at the query location goes to the quadrant
 *   with positive offsets.
 * - `'staged'`: the `stagedCandidateCount` samples nearest in the metric, then the `neighborCount`
 *   of those with the largest model covariance to the query, ties going to the nearer candidate.
 *
 * `gstat` documents staged and octant search as not yet available, so these are this library's
 * definitions. `searchRadius` and `maximumTimeLag` exclude samples before the search. Equal metric
 * values resolve to the lower row.
 *
 * Counts, the neighbor search and the mode specialize the graph. The parameter buffer, including
 * the covariance model, changes per frame without a recompile.
 */
export class GPUSpaceTimeKriging implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSpaceTimeKrigingProps;
  readonly neighborCount: number;
  readonly neighborSearch: GPUSpaceTimeKrigingNeighborSearch;
  readonly neighborsPerOctant: number;
  readonly stagedCandidateCount: number;
  readonly queryCount: number;
  readonly temporalContext: GPUTemporalContext;

  constructor(props: GPUSpaceTimeKrigingProps) {
    this.props = props;
    this.id = props.id ?? 'space-time-kriging';
    const {id} = this;
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const support = {encodings: ['float32-relative'] as const};
    this.temporalContext = resolveGPUTemporalContext(id, props.temporalContext, support);
    if (props.queryTemporalContext) {
      resolveGPUTemporalContext(id, props.queryTemporalContext, support);
      assertCompatibleGPUTemporalContexts(id, props.temporalContext, props.queryTemporalContext);
    }
    this.neighborCount = props.neighborCount ?? 16;
    this.neighborSearch = props.neighborSearch ?? 'metric';
    this.neighborsPerOctant = props.neighborsPerOctant ?? Math.ceil(this.neighborCount / 8);
    this.stagedCandidateCount = props.stagedCandidateCount ?? 2 * this.neighborCount;
    this.queryCount = props.queries.length;
    validateCapacity(
      id,
      'positions.length',
      props.positions.length,
      1,
      GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT
    );
    validateCapacity(id, 'neighborCount', this.neighborCount, 1);
    validateCapacity(id, 'queryCount', this.queryCount, 0);
    validateCapacity(id, 'neighborsPerOctant', this.neighborsPerOctant, 1);
    validateCapacity(id, 'stagedCandidateCount', this.stagedCandidateCount, 1);
    if (
      this.neighborCount > MAXIMUM_NEIGHBOR_COUNT ||
      this.neighborsPerOctant > MAXIMUM_NEIGHBORS_PER_OCTANT ||
      this.stagedCandidateCount > MAXIMUM_STAGED_CANDIDATE_COUNT ||
      this.stagedCandidateCount < this.neighborCount ||
      !['metric', 'octant', 'staged'].includes(this.neighborSearch) ||
      !['simple', 'ordinary'].includes(props.mode)
    ) {
      throw new Error(`${id} invalid neighbor search or mode`);
    }
    if (
      props.times.length !== props.positions.length ||
      props.observations.length !== props.positions.length ||
      props.queryTimes.length !== this.queryCount ||
      props.parameters.length < GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH ||
      props.parameters.format !== 'float32' ||
      props.output.values.length < this.queryCount ||
      props.output.status.length < this.queryCount
    ) {
      throw new Error(`${id} invalid capacities`);
    }
    validatePredictionViews(
      id,
      [
        props.positions,
        props.times,
        props.observations,
        props.queries,
        props.queryTimes,
        props.parameters
      ],
      [props.output.values, props.output.status]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, neighborCount, neighborSearch, neighborsPerOctant, stagedCandidateCount} =
      this;
    const driftCount = props.mode === 'ordinary' ? 1 : 0;
    const candidateCapacity =
      neighborSearch === 'octant'
        ? 8 * neighborsPerOctant
        : neighborSearch === 'staged'
          ? stagedCandidateCount
          : neighborCount;
    const bindings = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'times', view: props.times, type: 'f32', access: 'read'},
      {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
      {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
      {name: 'queryTimes', view: props.queryTimes, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'results', view: props.output.values, type: 'f32', access: 'read_write'},
      {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
    ] as const;
    validateGraphViewsBelongToGraph(
      id,
      graph,
      bindings.map(binding => binding.view)
    );
    const insert =
      neighborSearch === 'octant'
        ? `let offset = position - queryPosition;
  let octant = select(0u, 1u, offset.x < 0.0) + select(0u, 2u, offset.y < 0.0) + select(0u, 4u, sampleTime < queryTime);
  octantCounts[octant] = insertCandidate(octant * NEIGHBORS_PER_OCTANT, NEIGHBORS_PER_OCTANT, octantCounts[octant], sample, metric);`
        : 'candidateCount = insertCandidate(0u, CANDIDATE_CAPACITY, candidateCount, sample, metric);';
    const gather =
      neighborSearch === 'octant'
        ? `var cursors: array<u32, 8>;
for (var slot = 0u; slot < NEIGHBOR_COUNT; slot++) {
  var chosen = 8u;
  for (var octant = 0u; octant < 8u; octant++) {
    if (cursors[octant] >= octantCounts[octant]) { continue; }
    if (chosen == 8u) { chosen = octant; continue; }
    let current = octant * NEIGHBORS_PER_OCTANT + cursors[octant];
    let best = chosen * NEIGHBORS_PER_OCTANT + cursors[chosen];
    if (candidateDistance[current] < candidateDistance[best] || (candidateDistance[current] == candidateDistance[best] && candidateRow[current] < candidateRow[best])) { chosen = octant; }
  }
  if (chosen == 8u) { break; }
  neighbors[retained] = candidateRow[chosen * NEIGHBORS_PER_OCTANT + cursors[chosen]];
  cursors[chosen] += 1u;
  retained += 1u;
}`
        : neighborSearch === 'staged'
          ? `var selectedCovariance: array<f32, ${neighborCount}>;
for (var slot = 0u; slot < candidateCount; slot++) {
  let sample = candidateRow[slot];
  let position = read_positions(sample);
  let sampleTime = times[timesOffset + sample];
  let covarianceValue = spaceTimeCovariance(distance(position, queryPosition), abs(sampleTime - queryTime), all(position == queryPosition), sampleTime == queryTime);
  if (!isFiniteValue(covarianceValue)) { return; }
  var insertion = retained;
  for (var other = 0u; other < retained; other++) { if (covarianceValue > selectedCovariance[other]) { insertion = other; break; } }
  if (insertion >= NEIGHBOR_COUNT) { continue; }
  var shift = min(retained, NEIGHBOR_COUNT - 1u);
  loop { if (shift <= insertion) { break; } selectedCovariance[shift] = selectedCovariance[shift - 1u]; neighbors[shift] = neighbors[shift - 1u]; shift--; }
  selectedCovariance[insertion] = covarianceValue; neighbors[insertion] = sample; retained = min(retained + 1u, NEIGHBOR_COUNT);
}`
          : `for (var slot = 0u; slot < candidateCount; slot++) { neighbors[slot] = candidateRow[slot]; retained += 1u; }`;
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUSpaceTimeKriging',
        variant: `${props.mode}-${neighborSearch}`,
        bindings,
        invocationCount: this.queryCount,
        workgroupSize: 32,
        declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4, queries: 2}, ['results'])}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
${getEquilibratedSolveWGSL(neighborCount + driftCount)}
${getPerFrameSpaceTimeCovarianceWGSL()}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const DRIFT_COUNT: u32 = ${driftCount}u;
const SAMPLE_COUNT: u32 = ${props.positions.length}u;
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
${neighborSearch === 'octant' ? `const NEIGHBORS_PER_OCTANT: u32 = ${neighborsPerOctant}u;` : ''}
var<private> candidateDistance: array<f32, ${candidateCapacity}>;
var<private> candidateRow: array<u32, ${candidateCapacity}>;
fn insertCandidate(base: u32, capacity: u32, count: u32, row: u32, metric: f32) -> u32 {
  var insertion = count;
  for (var slot = 0u; slot < count; slot++) { if (metric < candidateDistance[base + slot]) { insertion = slot; break; } }
  if (insertion >= capacity) { return count; }
  var shift = min(count, capacity - 1u);
  loop { if (shift <= insertion) { break; } candidateDistance[base + shift] = candidateDistance[base + shift - 1u]; candidateRow[base + shift] = candidateRow[base + shift - 1u]; shift--; }
  candidateDistance[base + insertion] = metric; candidateRow[base + insertion] = row;
  return min(count + 1u, capacity);
}`,
        body: `
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
if (!loadSpaceTimeCovariance()) { return; }
let covarianceScale = params[paramsOffset];
let searchRadius = params[paramsOffset + 24u];
let maximumTimeLag = params[paramsOffset + 25u];
let searchAnisotropy = params[paramsOffset + 26u];
let queryPosition = read_queries(index);
let queryTime = queryTimes[queryTimesOffset + index];
if (!isFiniteValue(queryPosition.x) || !isFiniteValue(queryPosition.y) || !isFiniteValue(queryTime)) { return; }
var candidateCount = 0u;
${neighborSearch === 'octant' ? 'var octantCounts: array<u32, 8>;' : ''}
for (var sample = 0u; sample < SAMPLE_COUNT; sample++) {
  let observation = read_observations(sample);
  let position = read_positions(sample);
  let sampleTime = times[timesOffset + sample];
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y) || !isFiniteValue(sampleTime)) { continue; }
  let spatialLag = distance(position, queryPosition);
  let timeLag = abs(sampleTime - queryTime);
  if (!(spatialLag <= searchRadius) || !(timeLag <= maximumTimeLag)) { continue; }
  let scaledTime = searchAnisotropy * timeLag;
  let metric = sqrt(spatialLag * spatialLag + scaledTime * scaledTime);
  ${insert}
}
var neighbors: array<u32, ${neighborCount}>;
var retained = 0u;
${gather}
if (retained == 0u || retained < DRIFT_COUNT) { return; }
var crossValues: array<f32, ${neighborCount}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let sampleTime = times[timesOffset + neighbors[row]];
  for (var column = 0u; column < retained; column++) {
    let otherPosition = read_positions(neighbors[column]);
    let otherTime = times[timesOffset + neighbors[column]];
    matrix[row * SYSTEM_CAPACITY + column] = spaceTimeCovariance(distance(position, otherPosition), abs(sampleTime - otherTime), all(position == otherPosition), sampleTime == otherTime) / covarianceScale;
  }
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  let crossValue = spaceTimeCovariance(distance(position, queryPosition), abs(sampleTime - queryTime), all(position == queryPosition), sampleTime == queryTime);
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
  for (var column = 0u; column < DRIFT_COUNT; column++) {
    matrix[row * SYSTEM_CAPACITY + retained + column] = 1.0;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = 1.0;
  }
}
for (var column = 0u; column < DRIFT_COUNT; column++) {
  right[retained + column] = 1.0;
  for (var other = 0u; other < DRIFT_COUNT; other++) { matrix[(retained + column) * SYSTEM_CAPACITY + retained + other] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + DRIFT_COUNT)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
let mean = params[paramsOffset + 27u];
var prediction = ${driftCount ? '0.0' : 'mean'};
var variance = spaceTimeCovariance(0.0, 0.0, true, true);
let varianceScale = max(variance, 1e-20);
for (var row = 0u; row < retained; row++) {
  let observation = read_observations(neighbors[row]);
  prediction += solution[row] * (observation.x ${driftCount ? '' : '- mean'});
  variance -= solution[row] * crossValues[row];
}
for (var column = 0u; column < DRIFT_COUNT; column++) { variance -= solution[retained + column] * covarianceScale; }
if (!isFiniteValue(prediction) || !isFiniteValue(variance) || variance < -1e-4 * varianceScale) { return; }
write_results(index, vec4<f32>(prediction, max(variance, 0.0), pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;`
      })
    ];
  }
}
