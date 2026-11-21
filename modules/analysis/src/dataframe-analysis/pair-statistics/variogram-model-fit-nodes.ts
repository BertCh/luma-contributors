// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import type {GPUVariogramModelFit} from './gpu-variogram-model-fit';
import {COOPERATIVE_VARIOGRAM_FIT_WGSL} from './variogram-model-fit-cooperative';
import {VARIOGRAM_MODEL_SHAPE_WGSL, type VariogramModelType} from './variogram-model';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** One aligned slot per Cressie round keeps offsets and pipeline sources identical. */
const ROUND_PARAMETER_SLOT_WORD_COUNT = 64;
/** Reweight and publish flags are the only shader-visible words in each slot. */
const ROUND_PARAMETER_WORD_COUNT = 2;
/** The initial fit and three reweighting rounds retain the existing contract. @internal */
export const CRESSIE_ROUND_COUNT = 4;

const MODEL_CODES: Record<VariogramModelType, number> = {
  spherical: 0,
  exponential: 1,
  gaussian: 2,
  'matern-3/2': 3,
  'matern-5/2': 4
};

/** Spacing of float32 values relative to their magnitude, `2^-23`. */
const FLOAT32_RELATIVE_RESOLUTION = 2 ** -23;
/** WGSL literal of the square root of the largest float32, `sqrt(3.4028235e38)`. */
const FLOAT32_MAXIMUM_SQUARE_ROOT_WGSL = '1.8446743e19';

/** Keep ordinary fits on the original float32 summation order. */
const MAXIMUM_SERIAL_PREPARE_BIN_COUNT = 1024;
const PREPARE_WORKGROUP_SIZE = 64;
const PREPARE_BIN_TILE_SIZE = 1024;
const PREPARE_SUMMARY_WORD_COUNT = 5;
const PREPARE_SUMMARY_WGSL = /* wgsl */ `
struct BinSummary {
  minimumLag: f32,
  maximumLag: f32,
  largestValue: f32,
  baseWeight: f32,
  count: u32,
}
var<workgroup> binSummaries: array<BinSummary, ${PREPARE_WORKGROUP_SIZE}>;
fn emptyBinSummary() -> BinSummary {
  return BinSummary(bitcast<f32>(0x7f7fffffu), 0.0, 0.0, 0.0, 0u);
}
fn mergeBinSummaries(left: BinSummary, right: BinSummary) -> BinSummary {
  return BinSummary(
    min(left.minimumLag, right.minimumLag),
    max(left.maximumLag, right.maximumLag),
    max(left.largestValue, right.largestValue),
    left.baseWeight + right.baseWeight,
    left.count + right.count
  );
}
fn reduceBinSummaries(lane: u32, partial: BinSummary) -> BinSummary {
  binSummaries[lane] = partial;
  workgroupBarrier();
  for (var stride = ${PREPARE_WORKGROUP_SIZE / 2}u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      binSummaries[lane] = mergeBinSummaries(binSummaries[lane], binSummaries[lane + stride]);
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&binSummaries[0]);
}
`;

/** Parallel range ladder and cooperative bin reductions; refinement steps and rounds remain ordered. */
export function getParallelVariogramModelFitNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUVariogramModelFit
): GPUCommandNode<Parameters>[] {
  const {id, props, models, rangeSteps, maximumIterations} = contributor;
  validateGraphChunkBindingRanges(graph, id, {
    distances: props.distances,
    semivariances: props.semivariances,
    pairCounts: props.pairCounts,
    ...props.output
  });
  validateMaximumCount(
    id,
    'storage bindings',
    8,
    graph.device.limits.maxStorageBuffersPerShaderStage
  );
  const roundCount =
    props.weighting === undefined || props.weighting === 'cressie' ? CRESSIE_ROUND_COUNT : 1;
  const roundParameterWordCount = roundCount * ROUND_PARAMETER_SLOT_WORD_COUNT;
  const useTiledPrepare = props.distances.length > MAXIMUM_SERIAL_PREPARE_BIN_COUNT;
  const tileCount = Math.ceil(props.distances.length / PREPARE_BIN_TILE_SIZE);
  const partialWordCount = tileCount * PREPARE_SUMMARY_WORD_COUNT;
  const scratchWordCount = 4 + models.length * (rangeSteps + 1) * 4;
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'search scratch',
    scratchWordCount * 4,
    'reduce rangeSteps or models length'
  );
  validateMaximumCount(
    id,
    'search scratch byteLength',
    scratchWordCount * 4,
    graph.device.limits.maxBufferSize
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    useTiledPrepare ? 'round parameter buffer' : 'round parameter slot',
    (useTiledPrepare ? roundParameterWordCount : ROUND_PARAMETER_WORD_COUNT) * 4,
    useTiledPrepare
      ? 'use a device with a larger storage binding'
      : 'use a device with an eight-byte storage binding'
  );
  validateMaximumCount(
    id,
    'round parameter buffer byteLength',
    roundParameterWordCount * 4,
    graph.device.limits.maxBufferSize
  );
  if (useTiledPrepare) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'bin summary scratch',
      partialWordCount * 4,
      'reduce the empirical bin count'
    );
    validateMaximumCount(
      id,
      'bin summary scratch byteLength',
      partialWordCount * 4,
      graph.device.limits.maxBufferSize
    );
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  let roundParameters: GraphDataView<'uint32'>;
  if (useTiledPrepare) {
    roundParameters = createTransientView(
      graph,
      `${id}-round-parameters`,
      'uint32',
      roundParameterWordCount
    );
  } else {
    const roundValues = new Uint32Array(roundParameterWordCount);
    for (let round = 0; round < roundCount; round++) {
      roundValues.set(
        [Number(round > 0), Number(round === roundCount - 1)],
        round * ROUND_PARAMETER_SLOT_WORD_COUNT
      );
    }
    const upload = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-round-parameters`,
      roundValues
    );
    roundParameters = upload.parameters;
    nodes.push(...upload.nodes);
  }
  const modelCodes = models.map(model => MODEL_CODES[model]);
  const previousStart = 4;
  const candidateStart = previousStart + models.length * 4;
  const scratch = createTransientView(
    graph,
    `${id}-search`,
    'float32',
    candidateStart + models.length * rangeSteps * 4
  );
  const inputs = [
    {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
    {name: 'semivariances', view: props.semivariances, type: 'f32', access: 'read'},
    {name: 'pairCounts', view: props.pairCounts, type: 'u32', access: 'read'}
  ] as const;
  const outputs = [
    {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'},
    {name: 'parameters', view: props.output.parameters, type: 'f32', access: 'read_write'},
    {name: 'diagnostics', view: props.output.diagnostics, type: 'f32', access: 'read_write'},
    {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
  ] as const;
  const usableBinWGSL = `fn usable(bin: u32) -> bool { return pairCounts[pairCountsOffset + bin] > 0u && isFiniteValue(distances[distancesOffset + bin]) && distances[distancesOffset + bin] > 0.0 && isFiniteValue(semivariances[semivariancesOffset + bin]); }`;
  const writeFitWGSL = `fn writeFit(offset: u32, value: vec4f) {
  scratch[scratchOffset + offset] = value.x;
  scratch[scratchOffset + offset + 1u] = value.y;
  scratch[scratchOffset + offset + 2u] = value.z;
  scratch[scratchOffset + offset + 3u] = value.w;
}`;
  const declarations = `${getPackedPredictionAccessors({}, ['parameters', 'diagnostics'])}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
const BIN_COUNT: u32 = ${props.distances.length}u;
const MODEL_CODES: array<u32, ${models.length}> = array<u32, ${models.length}>(${modelCodes.map(code => `${code}u`).join(',')});
fn readFit(offset: u32) -> vec4f {
  return vec4f(scratch[scratchOffset + offset], scratch[scratchOffset + offset + 1u], scratch[scratchOffset + offset + 2u], scratch[scratchOffset + offset + 3u]);
}
${writeFitWGSL}
var<private> model: u32;
var<private> previous: vec4<f32>;
var<private> reweight: bool;
${VARIOGRAM_MODEL_SHAPE_WGSL}
// The shared float32 shapes: they keep their digits for bins at a small fraction of a candidate
// range, where the direct 1 - exp(-x) forms cancel (interpolation-density-08).
fn shape(scaled: f32) -> f32 { return getVariogramModelShape(model, scaled); }
${usableBinWGSL}
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
  // Cressie weights are N / max(fitted^2, 1e-20) and reach 1e20 for semivariances near 1e-8, so
  // the normal-equation products weightSum * squareSum would leave f32 and the joint nugget and
  // sill fit would be skipped. Every weight is divided by the total first, as the cooperative
  // refinement does; the residual below keeps the caller's unnormalised weights.
  var totalWeight = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) { if (usable(bin)) { totalWeight += weight(bin); } }
  let weightScale = select(1.0, 1.0 / totalWeight, totalWeight > 0.0 && totalWeight < 3.0e38);
  var weightSum = 0.0; var shapeSum = 0.0; var squareSum = 0.0; var valueSum = 0.0; var crossSum = 0.0;
  for (var bin = 0u; bin < BIN_COUNT; bin++) {
    if (!usable(bin)) { continue; }
    let binWeight = weight(bin) * weightScale; let binShape = shape(distances[distancesOffset + bin] / range); let value = semivariances[semivariancesOffset + bin];
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
  const publishSearchWGSL = `let valid = binCount >= 2u && maximumLag > minimumLag;
// The objective is a weighted sum of squared residuals in the caller's weights. A residual
// cannot be smaller than the float32 resolution of the largest semivariance (2^-23 of it), so
// when that resolution, squared and summed with the weights, is not finite, the objective of
// even an exact fit is a matter of rounding (0 or infinity) and candidates cannot be ranked.
// Such a frame is a numerical failure for every model. The test is written without the
// product so that it does not rely on overflow: resolution >= sqrt(FLT_MAX / weight).
let resolution = largestValue * ${getWGSLFloatLiteral(FLOAT32_RELATIVE_RESOLUTION)};
let isRepresentable = !(resolution >= ${FLOAT32_MAXIMUM_SQUARE_ROOT_WGSL} / sqrt(max(baseWeight, 1.0)));
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
  statuses[statusesOffset + modelIndex] = select(1u, select(2u, 0u, isRepresentable), valid);
  writeFit(${previousStart}u + modelIndex * 4u, vec4f(0.0, 0.0, 1.0, 3.402823e38));
}`;

  if (useTiledPrepare) {
    const partials = createTransientView(graph, `${id}-bin-summaries`, 'uint32', partialWordCount);
    const partialsBinding = {name: 'partials', view: partials, type: 'u32'} as const;
    const prepareTileCountWGSL = `const TILE_COUNT: u32 = ${tileCount}u;`;
    const usableBinsWGSL = `${IS_FINITE_VALUE_WGSL}
const BIN_COUNT: u32 = ${props.distances.length}u;
${usableBinWGSL}
fn getBaseWeight(bin: u32) -> f32 {
  return ${props.weighting === 'none' ? '1.0' : 'f32(pairCounts[pairCountsOffset + bin])'};
}`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-prepare-tiles`,
        operation: 'GPUVariogramModelFit',
        variant: 'prepare-bin-tiles',
        invocationCount: tileCount * PREPARE_WORKGROUP_SIZE,
        workgroupSize: PREPARE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [...inputs, {...partialsBinding, access: 'read_write'}],
        declarations: `${usableBinsWGSL}
${PREPARE_SUMMARY_WGSL}
${prepareTileCountWGSL}`,
        body: `
if (workgroupIndex >= TILE_COUNT) { return; }
let firstBin = workgroupIndex * ${PREPARE_BIN_TILE_SIZE}u;
let tileBinCount = min(BIN_COUNT - firstBin, ${PREPARE_BIN_TILE_SIZE}u);
var partial = emptyBinSummary();
for (var tileBin = localInvocationIndex; tileBin < tileBinCount; tileBin += ${PREPARE_WORKGROUP_SIZE}u) {
  let bin = firstBin + tileBin;
  if (usable(bin)) {
    let lag = distances[distancesOffset + bin];
    partial.minimumLag = min(partial.minimumLag, lag);
    partial.maximumLag = max(partial.maximumLag, lag);
    partial.largestValue = max(partial.largestValue, abs(semivariances[semivariancesOffset + bin]));
    partial.baseWeight += getBaseWeight(bin);
    partial.count++;
  }
}
let summary = reduceBinSummaries(localInvocationIndex, partial);
if (localInvocationIndex == 0u) {
  let slot = partialsOffset + workgroupIndex * ${PREPARE_SUMMARY_WORD_COUNT}u;
  partials[slot] = bitcast<u32>(summary.minimumLag);
  partials[slot + 1u] = bitcast<u32>(summary.maximumLag);
  partials[slot + 2u] = bitcast<u32>(summary.largestValue);
  partials[slot + 3u] = bitcast<u32>(summary.baseWeight);
  partials[slot + 4u] = summary.count;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPUVariogramModelFit',
        variant: 'prepare-search',
        invocationCount: PREPARE_WORKGROUP_SIZE,
        workgroupSize: PREPARE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          ...outputs,
          {...partialsBinding, access: 'read'},
          {name: 'roundParameters', view: roundParameters, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getPackedPredictionAccessors({}, ['parameters', 'diagnostics'])}
${GET_NAN_WGSL}
${writeFitWGSL}
${PREPARE_SUMMARY_WGSL}
${prepareTileCountWGSL}`,
        body: `
// Initialize one aligned parameter slot per round on each encoding, including replays.
let roundCount = (arrayLength(&roundParameters) - roundParametersOffset) / ${ROUND_PARAMETER_SLOT_WORD_COUNT}u;
if (localInvocationIndex < roundCount) {
  let slot = roundParametersOffset + localInvocationIndex * ${ROUND_PARAMETER_SLOT_WORD_COUNT}u;
  roundParameters[slot] = u32(localInvocationIndex > 0u);
  roundParameters[slot + 1u] = u32(localInvocationIndex + 1u == roundCount);
}
var partial = emptyBinSummary();
for (var tile = localInvocationIndex; tile < TILE_COUNT; tile += ${PREPARE_WORKGROUP_SIZE}u) {
  let slot = partialsOffset + tile * ${PREPARE_SUMMARY_WORD_COUNT}u;
  partial = mergeBinSummaries(partial, BinSummary(
    bitcast<f32>(partials[slot]),
    bitcast<f32>(partials[slot + 1u]),
    bitcast<f32>(partials[slot + 2u]),
    bitcast<f32>(partials[slot + 3u]),
    partials[slot + 4u]
  ));
}
let summary = reduceBinSummaries(localInvocationIndex, partial);
if (localInvocationIndex != 0u) { return; }
let minimumLag = summary.minimumLag;
let maximumLag = summary.maximumLag;
let largestValue = summary.largestValue;
let baseWeight = summary.baseWeight;
let binCount = summary.count;
${publishSearchWGSL}`
      })
    );
  } else {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPUVariogramModelFit',
        variant: 'prepare-search',
        // contract-allow: single-invocation -- MAXIMUM_SERIAL_PREPARE_BIN_COUNT = 1024 bounds the original ordered scan. Larger inputs use the tiled route.
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [...inputs, ...outputs],
        declarations,
        body: `
var minimumLag = 3.402823e38;
var maximumLag = 0.0;
var binCount = 0u;
var largestValue = 0.0;
var baseWeight = 0.0;
reweight = false;
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  if (usable(bin)) {
    minimumLag = min(minimumLag, distances[distancesOffset + bin]);
    maximumLag = max(maximumLag, distances[distancesOffset + bin]);
    largestValue = max(largestValue, abs(semivariances[semivariancesOffset + bin]));
    baseWeight += weight(bin);
    binCount++;
  }
}
${publishSearchWGSL}`
      })
    );
  }
  for (let round = 0; round < roundCount; round++) {
    const roundParameterView = graph.createDataView(roundParameters.buffer, {
      format: 'uint32',
      length: ROUND_PARAMETER_WORD_COUNT,
      byteOffset: round * ROUND_PARAMETER_SLOT_WORD_COUNT * 4
    });
    const roundBindings = [
      ...inputs,
      ...outputs,
      {name: 'roundParameters', view: roundParameterView, type: 'u32', access: 'read'} as const
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidates-${round}`,
        operation: 'GPUVariogramModelFit',
        variant: 'range-candidates',
        invocationCount: models.length * rangeSteps,
        workgroupSize: 64,
        bindings: roundBindings,
        declarations,
        body: `
let modelIndex = index / ${rangeSteps}u;
if (statuses[statusesOffset + modelIndex] != 0u) { return; }
model = MODEL_CODES[modelIndex];
previous = readFit(${previousStart}u + modelIndex * 4u);
reweight = roundParameters[roundParametersOffset] != 0u;
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
        bindings: roundBindings,
        declarations:
          declarations +
          COOPERATIVE_VARIOGRAM_FIT_WGSL +
          '\nvar<workgroup> sharedPublishRound: u32;',
        body: `
let modelIndex = workgroupIndex;
model = MODEL_CODES[modelIndex];
previous = readFit(${previousStart}u + modelIndex * 4u);
reweight = roundParameters[roundParametersOffset] != 0u;
if (localInvocationIndex == 0u) {
  sharedPublishRound = roundParameters[roundParametersOffset + 1u];
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
if (workgroupUniformLoad(&sharedPublishRound) != 0u) {
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
}
}`
      })
    );
  }
  return nodes;
}
