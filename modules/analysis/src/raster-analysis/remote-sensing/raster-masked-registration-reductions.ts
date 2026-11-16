// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPURasterRegistrationProps} from './gpu-raster-registration';
import {RASTER_FINITE_WGSL, rasterBinding} from './raster-kernel-utils';

/** Power-of-two tree width fits the baseline 256-invocation and 16 KiB workgroup limits. */
export const MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE = 256;
/** Two exact integer counts and two bit-cast float means per raster tile. */
export const MASKED_REGISTRATION_MEAN_WORD_COUNT = 4;
/** Exact paired count, two means, covariance, and two centered sums of squares per tile/candidate. */
export const MASKED_REGISTRATION_MOMENT_WORD_COUNT = 6;

const MEAN_DECLARATIONS = `
struct RasterMeans { referenceCount: u32, movingCount: u32, referenceMean: f32, movingMean: f32 }
fn combineMeans(first: RasterMeans, second: RasterMeans) -> RasterMeans {
  var result: RasterMeans;
  result.referenceCount = first.referenceCount + second.referenceCount;
  result.movingCount = first.movingCount + second.movingCount;
  if (result.referenceCount > 0u) {
    result.referenceMean = first.referenceMean * (f32(first.referenceCount) / f32(result.referenceCount)) + second.referenceMean * (f32(second.referenceCount) / f32(result.referenceCount));
  }
  if (result.movingCount > 0u) {
    result.movingMean = first.movingMean * (f32(first.movingCount) / f32(result.movingCount)) + second.movingMean * (f32(second.movingCount) / f32(result.movingCount));
  }
  return result;
}
var<workgroup> sharedMeans: array<RasterMeans, ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE}>;`;

const MOMENT_DECLARATIONS = `
struct RasterMoments { count: u32, referenceMean: f32, movingMean: f32, covariance: f32, referenceVariance: f32, movingVariance: f32 }
// Chan's pairwise form of Welford: fixed tree order, with exact uint32 support throughout.
fn combineMoments(first: RasterMoments, second: RasterMoments) -> RasterMoments {
  if (first.count == 0u) { return second; }
  if (second.count == 0u) { return first; }
  let count = first.count + second.count;
  let fraction = f32(second.count) / f32(count);
  let weight = f32(first.count) * fraction;
  let referenceDifference = second.referenceMean - first.referenceMean;
  let movingDifference = second.movingMean - first.movingMean;
  return RasterMoments(count,
    first.referenceMean + referenceDifference * fraction,
    first.movingMean + movingDifference * fraction,
    first.covariance + second.covariance + referenceDifference * movingDifference * weight,
    first.referenceVariance + second.referenceVariance + referenceDifference * referenceDifference * weight,
    first.movingVariance + second.movingVariance + movingDifference * movingDifference * weight);
}
var<workgroup> sharedMoments: array<RasterMoments, ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE}>;`;

function getReductionTree(shared: string, combine: string): string {
  return `workgroupBarrier();
for (var stride = ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
  if (localInvocationIndex < stride) { ${shared}[localInvocationIndex] = ${combine}(${shared}[localInvocationIndex], ${shared}[localInvocationIndex + stride]); }
  workgroupBarrier();
}`;
}

/** Each cell contributes once; the final workgroup folds at most four tiles per lane at 512². */
export function createMaskedRegistrationMeanNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegistrationProps,
  id: string,
  statistics: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  const count = props.grid.width * props.grid.height;
  const groupCount = Math.ceil(count / MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE);
  const partials = createTransientView(
    graph,
    `${id}-masked-mean-partials`,
    'uint32',
    groupCount * MASKED_REGISTRATION_MEAN_WORD_COUNT
  );
  const tree = getReductionTree('sharedMeans', 'combineMeans');
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-masked-mean-groups`,
      operation: 'GPURasterRegistration',
      invocationCount: count,
      workgroupSize: MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        rasterBinding('reference', props.reference),
        rasterBinding('moving', props.moving),
        ...(props.referenceMask ? [rasterBinding('referenceValidity', props.referenceMask)] : []),
        ...(props.movingMask ? [rasterBinding('movingValidity', props.movingMask)] : []),
        rasterBinding('partials', partials, true)
      ],
      declarations: RASTER_FINITE_WGSL + MEAN_DECLARATIONS,
      body: `var state: RasterMeans;
if (index < ${count}u) {
  let referenceValue = reference[referenceOffset + index]; let movingValue = moving[movingOffset + index];
  if (finite(referenceValue) ${props.referenceMask ? '&& referenceValidity[referenceValidityOffset + index] != 0u' : ''}) { state.referenceCount = 1u; state.referenceMean = referenceValue; }
  if (finite(movingValue) ${props.movingMask ? '&& movingValidity[movingValidityOffset + index] != 0u' : ''}) { state.movingCount = 1u; state.movingMean = movingValue; }
}
sharedMeans[localInvocationIndex] = state;
${tree}
if (localInvocationIndex == 0u) { let result = sharedMeans[0]; let slot = partialsOffset + workgroupIndex * 4u;
  partials[slot] = result.referenceCount; partials[slot + 1u] = result.movingCount;
  partials[slot + 2u] = bitcast<u32>(result.referenceMean); partials[slot + 3u] = bitcast<u32>(result.movingMean);
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-masked-statistics`,
      operation: 'GPURasterRegistration',
      invocationCount: MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      workgroupSize: MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        rasterBinding('partials', partials),
        rasterBinding('statistics', statistics, true)
      ],
      declarations: MEAN_DECLARATIONS,
      body: `var state: RasterMeans;
for (var group = localInvocationIndex; group < ${groupCount}u; group += ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE}u) {
  let slot = partialsOffset + group * 4u;
  state = combineMeans(state, RasterMeans(partials[slot], partials[slot + 1u], bitcast<f32>(partials[slot + 2u]), bitcast<f32>(partials[slot + 3u])));
}
sharedMeans[localInvocationIndex] = state;
${tree}
if (localInvocationIndex == 0u) { let result = sharedMeans[0];
  statistics[statisticsOffset] = result.referenceMean; statistics[statisticsOffset + 1u] = result.movingMean;
  // Counts are at most 512 squared, exactly representable in this existing float32 layout.
  statistics[statisticsOffset + 2u] = f32(result.referenceCount); statistics[statisticsOffset + 3u] = f32(result.movingCount);
}`
    })
  ];
}

/** Parallel paired-valid bilinear NCC; no invocation scans the raster or one full candidate. */
export function createMaskedRegistrationRefinementNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegistrationProps,
  id: string,
  coarse: GraphDataView<'float32'>,
  statistics: GraphDataView<'float32'>,
  refined: GraphDataView<'float32'>,
  refinedSupport: GraphDataView<'uint32'>,
  candidateWidth: number,
  radius: number,
  upsample: number
): GPUCommandNode<Parameters>[] {
  const {width, height} = props.grid;
  const count = width * height;
  const groupCount = Math.ceil(count / MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE);
  const candidateCount = candidateWidth ** 2;
  const maximumShift = props.maximumShift ?? [width - 1, height - 1];
  const partials = createTransientView(
    graph,
    `${id}-masked-refinement-moments`,
    'uint32',
    groupCount * candidateCount * MASKED_REGISTRATION_MOMENT_WORD_COUNT
  );
  const tree = getReductionTree('sharedMoments', 'combineMoments');
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-masked-refine-groups`,
      operation: 'GPURasterRegistration',
      invocationCount: candidateCount * groupCount * MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      workgroupSize: MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        rasterBinding('reference', props.reference),
        rasterBinding('moving', props.moving),
        ...(props.referenceMask ? [rasterBinding('referenceValidity', props.referenceMask)] : []),
        ...(props.movingMask ? [rasterBinding('movingValidity', props.movingMask)] : []),
        rasterBinding('coarse', coarse),
        rasterBinding('statistics', statistics),
        rasterBinding('partials', partials, true)
      ],
      declarations:
        RASTER_FINITE_WGSL +
        MOMENT_DECLARATIONS +
        `
fn sampleMoments(cell: u32, candidate: u32) -> RasterMoments {
  var result: RasterMoments;
  if (cell >= ${count}u) { return result; }
  let shift = vec2<f32>(coarse[coarseOffset], coarse[coarseOffset + 1u]) + (vec2<f32>(f32(candidate % ${candidateWidth}u), f32(candidate / ${candidateWidth}u)) - vec2<f32>(${radius}.0)) / ${upsample}.0;
  if (any(abs(shift) > vec2<f32>(${maximumShift[0]}.0, ${maximumShift[1]}.0))) { return result; }
  let referenceValue = reference[referenceOffset + cell] - statistics[statisticsOffset];
  if (!finite(referenceValue) ${props.referenceMask ? '|| referenceValidity[referenceValidityOffset + cell] == 0u' : ''}) { return result; }
  let coordinate = vec2<f32>(f32(cell % ${width}u), f32(cell / ${width}u)) - shift;
  if (any(coordinate < vec2<f32>(0.0)) || any(coordinate > vec2<f32>(${width - 1}.0, ${height - 1}.0))) { return result; }
  let lower = vec2<i32>(floor(coordinate)); let fraction = fract(coordinate); var movingValue = 0.0;
  for (var corner = 0u; corner < 4u; corner++) {
    let offset = vec2<i32>(i32(corner % 2u), i32(corner / 2u));
    let weight = select(1.0 - fraction.x, fraction.x, offset.x != 0) * select(1.0 - fraction.y, fraction.y, offset.y != 0);
    if (weight <= 0.0) { continue; }
    let position = lower + offset;
    if (any(position < vec2<i32>(0)) || any(position >= vec2<i32>(${width}, ${height}))) { return result; }
    let source = u32(position.y) * ${width}u + u32(position.x); let value = moving[movingOffset + source];
    if (!finite(value) ${props.movingMask ? '|| movingValidity[movingValidityOffset + source] == 0u' : ''}) { return result; }
    movingValue += weight * (value - statistics[statisticsOffset + 1u]);
  }
  result.count = 1u; result.referenceMean = referenceValue; result.movingMean = movingValue;
  return result;
}`,
      body: `if (workgroupIndex >= ${candidateCount * groupCount}u) { return; }
let candidate = workgroupIndex / ${groupCount}u;
let cell = (workgroupIndex % ${groupCount}u) * ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE}u + localInvocationIndex;
sharedMoments[localInvocationIndex] = sampleMoments(cell, candidate);
${tree}
if (localInvocationIndex == 0u) { let result = sharedMoments[0]; let slot = partialsOffset + workgroupIndex * 6u;
  partials[slot] = result.count; partials[slot + 1u] = bitcast<u32>(result.referenceMean); partials[slot + 2u] = bitcast<u32>(result.movingMean);
  partials[slot + 3u] = bitcast<u32>(result.covariance); partials[slot + 4u] = bitcast<u32>(result.referenceVariance); partials[slot + 5u] = bitcast<u32>(result.movingVariance);
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-masked-refine`,
      operation: 'GPURasterRegistration',
      invocationCount: candidateCount * MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      workgroupSize: MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      bindings: [
        rasterBinding('partials', partials),
        rasterBinding('statistics', statistics),
        rasterBinding('refined', refined, true),
        rasterBinding('refinedSupport', refinedSupport, true)
      ],
      declarations: RASTER_FINITE_WGSL + MOMENT_DECLARATIONS,
      body: `var state: RasterMoments;
for (var group = localInvocationIndex; group < ${groupCount}u; group += ${MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE}u) {
  let slot = partialsOffset + (workgroupIndex * ${groupCount}u + group) * 6u;
  state = combineMoments(state, RasterMoments(partials[slot], bitcast<f32>(partials[slot + 1u]), bitcast<f32>(partials[slot + 2u]), bitcast<f32>(partials[slot + 3u]), bitcast<f32>(partials[slot + 4u]), bitcast<f32>(partials[slot + 5u])));
}
sharedMoments[localInvocationIndex] = state;
${tree}
if (localInvocationIndex == 0u) { let result = sharedMoments[0]; var score = missing();
  let required = max(${getWGSLFloatLiteral(props.minimumOverlapCount ?? 3)}, ceil(min(statistics[statisticsOffset + 2u], statistics[statisticsOffset + 3u]) * ${getWGSLFloatLiteral(props.minimumOverlapFraction ?? 0.3)}));
  if (f32(result.count) >= required && result.referenceVariance > 1e-12 && result.movingVariance > 1e-12) { score = clamp(result.covariance / sqrt(result.referenceVariance * result.movingVariance), -1.0, 1.0); }
  refined[refinedOffset + workgroupIndex] = score; refinedSupport[refinedSupportOffset + workgroupIndex] = result.count;
}`
    })
  ];
}
