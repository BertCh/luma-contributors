// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createGraphChunkGroupKernelNode} from '../../analysis-core/kernels/graph-chunk-groups';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {getPerFrameSpaceTimeCovarianceWGSL} from './prediction-covariance';
import type {GPUSpaceTimeKriging, SpaceTimeKrigingRowViews} from './gpu-space-time-kriging';
import {
  getSpaceTimeKrigingGridWGSL,
  getSpaceTimeKrigingGridScanWGSL
} from './space-time-kriging-index';

/** Same bounded selection and solve for the indexed and reference routes. @internal */
export function createSpaceTimeKrigingPredictionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUSpaceTimeKriging<SpaceTimeKrigingRowViews>,
  nodeId: string,
  chunkQueryCount: number,
  geometry: GraphDataView<'float32x3'>,
  queryGeometry: GraphDataView<'float32x3'>,
  parameters: GraphDataView<'float32'>,
  values: GraphDataView<'float32x4'>,
  status: GraphDataView<'uint32'>,
  grid?: {cellOffsets: GraphDataView<'uint32'>; sortedRows: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  const {props, neighborCount, neighborSearch, neighborsPerOctant, stagedCandidateCount, gridSide} =
    contributor;
  const indexed = Boolean(grid);
  const driftCount = props.mode === 'ordinary' ? 1 : 0;
  const candidateCapacity =
    neighborSearch === 'octant'
      ? 8 * neighborsPerOctant
      : neighborSearch === 'staged'
        ? stagedCandidateCount
        : neighborCount;
  const bindings = [
    {name: 'geometry', view: geometry, type: 'f32', access: 'read'},
    {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
    {name: 'queryGeometry', view: queryGeometry, type: 'f32', access: 'read'},
    {name: 'params', view: parameters, type: 'f32', access: 'read'},
    ...(grid
      ? [
          {name: 'cellOffsets', view: grid.cellOffsets, type: 'u32', access: 'read'} as const,
          {name: 'sortedRows', view: grid.sortedRows, type: 'u32', access: 'read'} as const
        ]
      : []),
    {name: 'results', view: values, type: 'f32', access: 'read_write'},
    {name: 'statuses', view: status, type: 'u32', access: 'read_write'}
  ] as const;
  const insert =
    neighborSearch === 'octant'
      ? `let offset = position - queryPosition;
  let octant = select(0u, 1u, offset.x < 0.0) + select(0u, 2u, offset.y < 0.0) + select(0u, 4u, sampleTime < queryTime);
  octantCounts[octant] = insertCandidate(octant * NEIGHBORS_PER_OCTANT, NEIGHBORS_PER_OCTANT, octantCounts[octant], sample, metric);`
      : 'candidateCount = insertCandidate(0u, CANDIDATE_CAPACITY, candidateCount, sample, metric);';
  const visitSample = `  let observation = read_observations(sample);
  let position = read_positions(sample);
  let sampleTime = geometry[sample * 3u + 2u];
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y) || !isFiniteValue(sampleTime)) { continue; }
  let spatialLag = distance(position, queryPosition);
  let timeLag = abs(sampleTime - queryTime);
  if (!(spatialLag <= searchRadius) || !(timeLag <= maximumTimeLag)) { continue; }
  let scaledTime = searchAnisotropy * timeLag;
  let metric = sqrt(spatialLag * spatialLag + scaledTime * scaledTime);
  ${insert}`;
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
  let sampleTime = geometry[sample * 3u + 2u];
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
  return createGraphChunkGroupKernelNode(graph, {
    id: nodeId,
    operation: 'GPUSpaceTimeKriging',
    variant: `${props.mode}-${neighborSearch}`,
    bindings,
    invocationCount: chunkQueryCount,
    workgroupSize: 32,
    declarations: `${getPackedPredictionAccessors({observations: 4}, ['results'])}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
${getEquilibratedSolveWGSL(neighborCount + driftCount)}
${getPerFrameSpaceTimeCovarianceWGSL()}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const DRIFT_COUNT: u32 = ${driftCount}u;
fn read_positions(row: u32) -> vec2<f32> { return vec2<f32>(geometry[row * 3u], geometry[row * 3u + 1u]); }
fn read_queries(row: u32) -> vec2<f32> { return vec2<f32>(queryGeometry[row * 3u], queryGeometry[row * 3u + 1u]); }
${indexed ? getSpaceTimeKrigingGridWGSL(gridSide) : ''}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
${neighborSearch === 'octant' ? `const NEIGHBORS_PER_OCTANT: u32 = ${neighborsPerOctant}u;` : ''}
var<private> candidateDistance: array<f32, ${candidateCapacity}>;
var<private> candidateRow: array<u32, ${candidateCapacity}>;
fn insertCandidate(base: u32, capacity: u32, count: u32, row: u32, metric: f32) -> u32 {
  var insertion = count;
  for (var slot = 0u; slot < count; slot++) { if (metric < candidateDistance[base + slot] || (metric == candidateDistance[base + slot] && row < candidateRow[base + slot])) { insertion = slot; break; } }
  if (insertion >= capacity) { return count; }
  var shift = min(count, capacity - 1u);
  loop { if (shift <= insertion) { break; } candidateDistance[base + shift] = candidateDistance[base + shift - 1u]; candidateRow[base + shift] = candidateRow[base + shift - 1u]; shift--; }
  candidateDistance[base + insertion] = metric; candidateRow[base + insertion] = row;
  return min(count + 1u, capacity);
}`,
    body: `
if (index >= arrayLength(&queryGeometry) / 3u) { return; }
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
if (!loadSpaceTimeCovariance()) { return; }
let covarianceScale = params[paramsOffset];
let searchRadius = params[paramsOffset + 24u];
let maximumTimeLag = params[paramsOffset + 25u];
let searchAnisotropy = params[paramsOffset + 26u];
let queryPosition = read_queries(index);
let queryTime = queryGeometry[index * 3u + 2u];
if (!isFiniteValue(queryPosition.x) || !isFiniteValue(queryPosition.y) || !isFiniteValue(queryTime)) { return; }
var candidateCount = 0u;
${neighborSearch === 'octant' ? 'var octantCounts: array<u32, 8>;' : ''}
${indexed ? getSpaceTimeKrigingGridScanWGSL(visitSample, neighborSearch) : `for (var sample = 0u; sample < arrayLength(&geometry) / 3u; sample++) { ${visitSample} }`}
var neighbors: array<u32, ${neighborCount}>;
var retained = 0u;
${gather}
if (retained == 0u || retained < DRIFT_COUNT) { return; }
var crossValues: array<f32, ${neighborCount}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let sampleTime = geometry[neighbors[row] * 3u + 2u];
  for (var column = 0u; column < retained; column++) {
    let otherPosition = read_positions(neighbors[column]);
    let otherTime = geometry[neighbors[column] * 3u + 2u];
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
  });
}
