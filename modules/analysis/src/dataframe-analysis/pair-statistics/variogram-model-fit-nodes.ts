// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import type {GPUVariogramModelFit} from './gpu-variogram-model-fit';
import {COOPERATIVE_VARIOGRAM_FIT_WGSL} from './variogram-model-fit-cooperative';

/** Parallel range ladder and cooperative bin reductions; refinement steps and rounds remain ordered. */
export function getParallelVariogramModelFitNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUVariogramModelFit
): GPUCommandNode<Parameters>[] {
  const {id, props, models, rangeSteps, maximumIterations} = contributor;
  const modelCodes = models.map(model => ['spherical', 'exponential', 'gaussian'].indexOf(model));
  const previousStart = 4;
  const candidateStart = previousStart + models.length * 4;
  const scratch = createTransientView(
    graph,
    `${id}-search`,
    'float32',
    candidateStart + models.length * rangeSteps * 4
  );
  const bindings = [
    {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
    {name: 'semivariances', view: props.semivariances, type: 'f32', access: 'read'},
    {name: 'pairCounts', view: props.pairCounts, type: 'u32', access: 'read'},
    {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'},
    {name: 'parameters', view: props.output.parameters, type: 'f32', access: 'read_write'},
    {name: 'diagnostics', view: props.output.diagnostics, type: 'f32', access: 'read_write'},
    {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
  ] as const;
  const declarations = `${getPackedPredictionAccessors({}, ['parameters', 'diagnostics'])}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
const BIN_COUNT: u32 = ${props.distances.length}u;
const MODEL_CODES: array<u32, ${models.length}> = array<u32, ${models.length}>(${modelCodes.map(code => `${code}u`).join(',')});
fn readFit(offset: u32) -> vec4f {
  return vec4f(scratch[scratchOffset + offset], scratch[scratchOffset + offset + 1u], scratch[scratchOffset + offset + 2u], scratch[scratchOffset + offset + 3u]);
}
fn writeFit(offset: u32, value: vec4f) {
  scratch[scratchOffset + offset] = value.x;
  scratch[scratchOffset + offset + 1u] = value.y;
  scratch[scratchOffset + offset + 2u] = value.z;
  scratch[scratchOffset + offset + 3u] = value.w;
}
var<private> model: u32;
var<private> previous: vec4<f32>;
var<private> reweight: bool;
fn shape(scaled: f32) -> f32 {
  if (model == 0u) { return select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0); }
  if (model == 1u) { return 1.0 - exp(-3.0 * scaled); }
  return 1.0 - exp(-3.0 * scaled * scaled);
}
fn usable(bin: u32) -> bool { return pairCounts[pairCountsOffset + bin] > 0u && isFiniteValue(distances[distancesOffset + bin]) && distances[distancesOffset + bin] > 0.0 && isFiniteValue(semivariances[semivariancesOffset + bin]); }
fn weight(bin: u32) -> f32 {
  var value = ${props.weighting === 'none' ? '1.0' : 'f32(pairCounts[pairCountsOffset + bin])'};
  if (reweight) { let fitted = previous.x + previous.y * shape(distances[distancesOffset + bin] / previous.z); value /= max(fitted * fitted, 1e-20); }
  return value;
}
fn residual(nugget: f32, sill: f32, range: f32) -> f32 {
  var value = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) { if (usable(bin)) { let difference = semivariances[semivariancesOffset + bin] - nugget - sill * shape(distances[distancesOffset + bin] / range); value += weight(bin) * difference * difference; } }
  return value;
}
fn fitLinear(range: f32) -> vec4<f32> {
  var weightSum = 0.0; var shapeSum = 0.0; var squareSum = 0.0; var valueSum = 0.0; var crossSum = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) {
    if (!usable(bin)) { continue; }
    let binWeight = weight(bin); let binShape = shape(distances[distancesOffset + bin] / range); let value = semivariances[semivariancesOffset + bin];
    weightSum += binWeight; shapeSum += binWeight * binShape; squareSum += binWeight * binShape * binShape; valueSum += binWeight * value; crossSum += binWeight * binShape * value;
  }
  let nuggetOnly = max(valueSum / weightSum, 0.0);
  let sillOnly = max(crossSum / max(squareSum, 1e-30), 0.0);
  var best = vec4<f32>(nuggetOnly, 0.0, range, residual(nuggetOnly, 0.0, range));
  let sillResidual = residual(0.0, sillOnly, range);
  if (sillResidual < best.w) { best = vec4<f32>(0.0, sillOnly, range, sillResidual); }
  let determinant = weightSum * squareSum - shapeSum * shapeSum;
  if (determinant > 1e-6 * weightSum * squareSum) {
    let nugget = (squareSum * valueSum - shapeSum * crossSum) / determinant;
    let sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
    if (nugget >= 0.0 && sill >= 0.0) { let error = residual(nugget, sill, range); if (error < best.w) { best = vec4<f32>(nugget, sill, range, error); } }
  }
  return best;
}
`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-prepare`,
      operation: 'GPUVariogramModelFit',
      variant: 'prepare-search',
      invocationCount: 1,
      workgroupSize: 2,
      bindings,
      declarations,
      body: `
var minimumLag = 3.402823e38;
var maximumLag = 0.0;
var binCount = 0u;
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  if (usable(bin)) {
    minimumLag = min(minimumLag, distances[distancesOffset + bin]);
    maximumLag = max(maximumLag, distances[distancesOffset + bin]);
    binCount++;
  }
}
let valid = binCount >= 2u && maximumLag > minimumLag;
writeFit(0u, vec4f(0.0, 0.0, f32(binCount), 0.0));
if (valid) {
  let lowest = log(minimumLag / 10.0);
  let highest = log(3.0 * maximumLag);
  scratch[scratchOffset] = lowest;
  scratch[scratchOffset + 1u] = (highest - lowest) / ${rangeSteps - 1}.0;
}
for (var modelIndex = 0u; modelIndex < ${models.length}u; modelIndex++) {
  write_parameters(modelIndex, vec4f(getNaN()));
  write_diagnostics(modelIndex, vec4f(getNaN()));
  statuses[statusesOffset + modelIndex] = select(1u, 0u, valid);
  writeFit(${previousStart}u + modelIndex * 4u, vec4f(0.0, 0.0, 1.0, 3.402823e38));
}`
    })
  ];
  const roundCount = props.weighting === undefined || props.weighting === 'cressie' ? 4 : 1;
  for (let round = 0; round < roundCount; round++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidates-${round}`,
        operation: 'GPUVariogramModelFit',
        variant: 'range-candidates',
        invocationCount: models.length * rangeSteps,
        workgroupSize: 64,
        bindings,
        declarations,
        body: `
let modelIndex = index / ${rangeSteps}u;
if (statuses[statusesOffset + modelIndex] != 0u) { return; }
model = MODEL_CODES[modelIndex];
previous = readFit(${previousStart}u + modelIndex * 4u);
reweight = ${round > 0};
let step = index % ${rangeSteps}u;
let range = exp(scratch[scratchOffset] + scratch[scratchOffset + 1u] * f32(step));
writeFit(${candidateStart}u + index * 4u, fitLinear(range));`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-refine-${round}`,
        operation: 'GPUVariogramModelFit',
        variant: 'refine-range',
        invocationCount: models.length * 64,
        workgroupSize: 64,
        guardIndex: false,
        bindings,
        declarations: declarations + COOPERATIVE_VARIOGRAM_FIT_WGSL,
        body: `
let modelIndex = workgroupIndex;
model = MODEL_CODES[modelIndex];
previous = readFit(${previousStart}u + modelIndex * 4u);
reweight = ${round > 0};
if (localInvocationIndex == 0u) {
  sharedValid = 0u;
  if (statuses[statusesOffset + modelIndex] == 0u) {
    var selected = previous;
    selected.w = 3.402823e38;
    var selectedStep = 0u;
    for (var step = 0u; step < ${rangeSteps}u; step++) {
      let candidate = readFit(${candidateStart}u + (modelIndex * ${rangeSteps}u + step) * 4u);
      if (candidate.w < selected.w) { selected = candidate; selectedStep = step; }
    }
    if (selected.w < 3.402823e38) {
      sharedBest = selected;
      sharedBestStep = selectedStep;
      sharedLimits = vec2f(scratch[scratchOffset], scratch[scratchOffset + 1u]);
      sharedValid = 1u;
    } else { statuses[statusesOffset + modelIndex] = 2u; }
  }
}
if (workgroupUniformLoad(&sharedValid) == 0u) { return; }
var best = workgroupUniformLoad(&sharedBest);
let bestStep = workgroupUniformLoad(&sharedBestStep);
let limits = workgroupUniformLoad(&sharedLimits);
let boundary = bestStep == 0u || bestStep == ${rangeSteps - 1}u;
let lowest = limits.x;
let stepSize = limits.y;
var left = lowest + stepSize * f32(select(bestStep - 1u, 0u, bestStep == 0u));
var rightBound = lowest + stepSize * f32(min(bestStep + 1u, ${rangeSteps - 1}u));
var inner = rightBound - 0.61803398875 * (rightBound - left);
var outer = left + 0.61803398875 * (rightBound - left);
var innerFit = fitLinearCooperative(exp(inner), localInvocationIndex);
var outerFit = fitLinearCooperative(exp(outer), localInvocationIndex);
for (var iteration = 0u; iteration < ${maximumIterations}u; iteration++) {
  if (rightBound - left <= 1e-5) { break; }
  if (innerFit.w <= outerFit.w) {
    rightBound = outer; outer = inner; outerFit = innerFit;
    inner = rightBound - 0.61803398875 * (rightBound - left); innerFit = fitLinearCooperative(exp(inner), localInvocationIndex);
  } else {
    left = inner; inner = outer; innerFit = outerFit;
    outer = left + 0.61803398875 * (rightBound - left); outerFit = fitLinearCooperative(exp(outer), localInvocationIndex);
  }
}
if (innerFit.w < best.w) { best = innerFit; }
if (outerFit.w < best.w) { best = outerFit; }
if (localInvocationIndex == 0u) { writeFit(${previousStart}u + modelIndex * 4u, best); }
${
  round === roundCount - 1
    ? `
let bracket = rightBound - left;
if (!isFiniteValue(best.w) || !isFiniteValue(best.x) || !isFiniteValue(best.y) || !isFiniteValue(best.z)) { if (localInvocationIndex == 0u) { statuses[statusesOffset + modelIndex] = 2u; } return; }
var partialUnweighted = 0.0;
for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += 64u) {
  if (usable(bin)) {
    let difference = semivariances[semivariancesOffset + bin] - best.x - best.y * shape(distances[distancesOffset + bin] / best.z);
    partialUnweighted += difference * difference;
  }
}
fitSums[localInvocationIndex] = FitSums(vec4f(partialUnweighted, 0.0, 0.0, 0.0), 0.0);
let unweighted = reduceFitSums(localInvocationIndex).moments.x;
if (!isFiniteValue(unweighted) || !isFiniteValue(bracket)) { if (localInvocationIndex == 0u) { statuses[statusesOffset + modelIndex] = 2u; } return; }
if (localInvocationIndex == 0u) {
let binCount = scratch[scratchOffset + 2u];
write_parameters(modelIndex, best);
write_diagnostics(modelIndex, vec4f(binCount, unweighted, binCount * log(max(unweighted / binCount, 1e-30)) + 6.0, bracket));
statuses[statusesOffset + modelIndex] = select(0u, 3u, boundary || bracket > 1e-5);
}`
    : ''
}`
      })
    );
  }
  return nodes;
}
