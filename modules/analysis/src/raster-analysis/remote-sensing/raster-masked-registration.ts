// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  GPUFFT2D,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding, validateRasterCount} from './raster-kernel-utils';
import type {GPURasterRegistrationProps} from './gpu-raster-registration';

export function validateMaskedRegistrationProps(
  props: GPURasterRegistrationProps,
  id: string
): void {
  // Six padded correlation fields plus FFT scratch bound this initial masked implementation.
  validateRasterCount(props.grid.width, 512);
  validateRasterCount(props.grid.height, 512);
  assert(props.grid.width >= 2 && props.grid.height >= 2);
  validateRasterCount(props.upsampleFactor ?? 20, 32);
  const maximumShift = props.maximumShift ?? [props.grid.width - 1, props.grid.height - 1];
  assert(
    maximumShift.every(
      (value, dimension) =>
        Number.isInteger(value) &&
        value >= 0 &&
        value < (dimension === 0 ? props.grid.width : props.grid.height)
    )
  );
  const fraction = props.minimumOverlapFraction ?? 0.3;
  assert(Number.isFinite(fraction) && fraction >= 0 && fraction <= 1);
  validateRasterCount(props.minimumOverlapCount ?? 3, props.grid.width * props.grid.height);
  assert((props.minimumOverlapCount ?? 3) >= 3);
  const minimumCorrelation = props.minimumCorrelation ?? 0;
  assert(
    Number.isFinite(minimumCorrelation) && minimumCorrelation >= -1 && minimumCorrelation <= 1
  );
  const ambiguity = props.ambiguityTolerance ?? 1e-4;
  assert(Number.isFinite(ambiguity) && ambiguity >= 0 && ambiguity < 1);
  const inputs = [
    props.reference,
    props.moving,
    props.referenceValidity,
    props.movingValidity
  ].filter(view => view !== undefined);
  const outputs = Object.values(props.output).filter(view => view !== undefined);
  for (const view of inputs) {
    validatePackedView(view, [view.format], id);
    assert(view.length >= props.grid.width * props.grid.height);
  }
  for (const view of outputs) {
    validatePackedView(view, [view.format], id);
    assert(view.length >= 1);
  }
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  outputs.forEach((view, index) =>
    validateGraphOutputsDisjointFromInputs(id, [view], outputs.slice(index + 1))
  );
}

function getPaddedDimension(length: number): number {
  return 2 ** Math.ceil(Math.log2(2 * length - 1));
}

export function getMaskedRegistrationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegistrationProps,
  id: string
): GPUCommandNode<Parameters>[] {
  validateGraphViewsBelongToGraph(id, graph, [
    props.reference,
    props.moving,
    props.referenceValidity,
    props.movingValidity,
    ...Object.values(props.output)
  ]);
  const width = props.grid.width;
  const height = props.grid.height;
  const count = width * height;
  const paddedWidth = getPaddedDimension(width);
  const paddedHeight = getPaddedDimension(height);
  const paddedCount = paddedWidth * paddedHeight;
  const maximumShift = props.maximumShift ?? [width - 1, height - 1];
  const searchWidth = maximumShift[0] * 2 + 1;
  const searchHeight = maximumShift[1] * 2 + 1;
  const searchCount = searchWidth * searchHeight;
  const upsample = props.upsampleFactor ?? 20;
  const radius = upsample === 1 ? 0 : Math.ceil(upsample / 2);
  const candidateWidth = radius * 2 + 1;
  const candidateCount = candidateWidth ** 2;
  const scalar = (view: GraphDataView<'float32x2'>) =>
    graph.createDataView(view.buffer, {
      format: 'float32',
      byteOffset: view.byteOffset,
      length: view.length * 2
    });
  const reference = createTransientView(
    graph,
    `${id}-masked-reference`,
    'float32x2',
    paddedCount * 3
  );
  const moving = createTransientView(graph, `${id}-masked-moving`, 'float32x2', paddedCount * 3);
  const referenceSpectrum = createTransientView(
    graph,
    `${id}-masked-reference-spectrum`,
    'float32x2',
    paddedCount * 3
  );
  const movingSpectrum = createTransientView(
    graph,
    `${id}-masked-moving-spectrum`,
    'float32x2',
    paddedCount * 3
  );
  const products = createTransientView(
    graph,
    `${id}-masked-products`,
    'float32x2',
    paddedCount * 6
  );
  const correlations = createTransientView(
    graph,
    `${id}-masked-correlations`,
    'float32x2',
    paddedCount * 6
  );
  const statistics = createTransientView(graph, `${id}-masked-statistics`, 'float32', 4);
  const scores = createTransientView(graph, `${id}-masked-scores`, 'float32', searchCount);
  const supports = createTransientView(graph, `${id}-masked-supports`, 'uint32', searchCount);
  const coarse = createTransientView(graph, `${id}-masked-coarse`, 'float32', 2);
  const refined = createTransientView(graph, `${id}-masked-refined`, 'float32', candidateCount);
  const refinedSupport = createTransientView(
    graph,
    `${id}-masked-refined-support`,
    'uint32',
    candidateCount
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  const maskBindings = [
    ...(props.referenceValidity
      ? [rasterBinding('referenceValidity', props.referenceValidity)]
      : []),
    ...(props.movingValidity ? [rasterBinding('movingValidity', props.movingValidity)] : [])
  ];
  const referenceValid = (source: string) =>
    `finite(referenceValues[referenceValuesOffset + ${source}])${props.referenceValidity ? ` && referenceValidity[referenceValidityOffset + ${source}] != 0u` : ''}`;
  const movingValid = (source: string) =>
    `finite(movingValues[movingValuesOffset + ${source}])${props.movingValidity ? ` && movingValidity[movingValidityOffset + ${source}] != 0u` : ''}`;
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-masked-statistics`,
      operation: 'GPURasterRegistration',
      invocationCount: 1,
      bindings: [
        rasterBinding('referenceValues', props.reference),
        rasterBinding('movingValues', props.moving),
        ...maskBindings,
        rasterBinding('statistics', statistics, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `var referenceMean = 0.0; var movingMean = 0.0; var referenceCount = 0u; var movingCount = 0u;
for (var cell = 0u; cell < ${count}u; cell++) {
  if (${referenceValid('cell')}) { referenceCount++; referenceMean = referenceMean * (1.0 - 1.0 / f32(referenceCount)) + referenceValues[referenceValuesOffset + cell] / f32(referenceCount); }
  if (${movingValid('cell')}) { movingCount++; movingMean = movingMean * (1.0 - 1.0 / f32(movingCount)) + movingValues[movingValuesOffset + cell] / f32(movingCount); }
}
statistics[statisticsOffset] = referenceMean; statistics[statisticsOffset + 1u] = movingMean; statistics[statisticsOffset + 2u] = f32(referenceCount); statistics[statisticsOffset + 3u] = f32(movingCount);`
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-masked-pack`,
      operation: 'GPURasterRegistration',
      invocationCount: paddedCount,
      bindings: [
        rasterBinding('referenceValues', props.reference),
        rasterBinding('movingValues', props.moving),
        ...maskBindings,
        rasterBinding('statistics', statistics),
        rasterBinding('reference', scalar(reference), true),
        rasterBinding('moving', scalar(moving), true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `let column = index % ${paddedWidth}u; let row = index / ${paddedWidth}u;
var referenceMask = 0.0; var movingMask = 0.0; var referenceValue = 0.0; var movingValue = 0.0;
if (column < ${width}u && row < ${height}u) { let cell = row * ${width}u + column;
  if (${referenceValid('cell')}) { referenceMask = 1.0; referenceValue = referenceValues[referenceValuesOffset + cell] - statistics[statisticsOffset]; }
  if (${movingValid('cell')}) { movingMask = 1.0; movingValue = movingValues[movingValuesOffset + cell] - statistics[statisticsOffset + 1u]; }
}
let referenceChannels = array<f32, 3>(referenceMask, referenceValue, referenceValue * referenceValue); let movingChannels = array<f32, 3>(movingMask, movingValue, movingValue * movingValue);
for (var channel = 0u; channel < 3u; channel++) { let offset = (channel * ${paddedCount}u + index) * 2u; reference[referenceOffset + offset] = referenceChannels[channel]; reference[referenceOffset + offset + 1u] = 0.0; moving[movingOffset + offset] = movingChannels[channel]; moving[movingOffset + offset + 1u] = 0.0; }`
    })
  );
  nodes.push(
    ...new GPUFFT2D({
      id: `${id}-masked-reference-fft`,
      input: reference,
      output: referenceSpectrum,
      width: paddedWidth,
      height: paddedHeight,
      batchCount: 3
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUFFT2D({
      id: `${id}-masked-moving-fft`,
      input: moving,
      output: movingSpectrum,
      width: paddedWidth,
      height: paddedHeight,
      batchCount: 3
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-masked-products`,
      operation: 'GPURasterRegistration',
      invocationCount: paddedCount * 6,
      bindings: [
        rasterBinding('reference', scalar(referenceSpectrum)),
        rasterBinding('moving', scalar(movingSpectrum)),
        rasterBinding('products', scalar(products), true)
      ],
      body: `let channel = index / ${paddedCount}u; let cell = index % ${paddedCount}u;
let referenceChannels = array<u32, 6>(0u, 1u, 0u, 2u, 0u, 1u); let movingChannels = array<u32, 6>(0u, 0u, 1u, 0u, 2u, 1u);
let firstOffset = (referenceChannels[channel] * ${paddedCount}u + cell) * 2u; let secondOffset = (movingChannels[channel] * ${paddedCount}u + cell) * 2u;
let first = vec2<f32>(reference[referenceOffset + firstOffset], reference[referenceOffset + firstOffset + 1u]); let second = vec2<f32>(moving[movingOffset + secondOffset], moving[movingOffset + secondOffset + 1u]);
products[productsOffset + index * 2u] = dot(first, second); products[productsOffset + index * 2u + 1u] = first.y * second.x - first.x * second.y;`
    })
  );
  nodes.push(
    ...new GPUFFT2D({
      id: `${id}-masked-inverse-fft`,
      input: products,
      output: correlations,
      width: paddedWidth,
      height: paddedHeight,
      batchCount: 6,
      direction: 'inverse'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-masked-normalize`,
      operation: 'GPURasterRegistration',
      invocationCount: searchCount,
      bindings: [
        rasterBinding('correlations', scalar(correlations)),
        rasterBinding('statistics', statistics),
        rasterBinding('scores', scores, true),
        rasterBinding('supports', supports, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `let shift = vec2<i32>(i32(index % ${searchWidth}u) - ${maximumShift[0]}, i32(index / ${searchWidth}u) - ${maximumShift[1]});
let coordinate = vec2<u32>((shift + vec2<i32>(${paddedWidth}, ${paddedHeight})) % vec2<i32>(${paddedWidth}, ${paddedHeight})); let cell = coordinate.y * ${paddedWidth}u + coordinate.x;
var moments: array<f32, 6>; for (var channel = 0u; channel < 6u; channel++) { moments[channel] = correlations[correlationsOffset + (channel * ${paddedCount}u + cell) * 2u]; }
let support = max(0.0, floor(moments[0] + 0.5)); supports[supportsOffset + index] = u32(support); var score = missing();
let required = max(${props.minimumOverlapCount ?? 3}.0, ceil(min(statistics[statisticsOffset + 2u], statistics[statisticsOffset + 3u]) * ${props.minimumOverlapFraction ?? 0.3}));
if (support >= required) { let referenceVariance = moments[3] - moments[1] * moments[1] / support; let movingVariance = moments[4] - moments[2] * moments[2] / support;
  if (referenceVariance > max(1e-12, abs(moments[3]) * 1e-6) && movingVariance > max(1e-12, abs(moments[4]) * 1e-6)) { score = clamp((moments[5] - moments[1] * moments[2] / support) / sqrt(referenceVariance * movingVariance), -1.0, 1.0); }
}
scores[scoresOffset + index] = score;`
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-masked-coarse`,
      operation: 'GPURasterRegistration',
      invocationCount: 1,
      bindings: [
        rasterBinding('scores', scores),
        rasterBinding('coarse', coarse, true),
        rasterBinding('status', props.output.status, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `var best = -2.0; var selected = 0u;
for (var candidate = 0u; candidate < ${searchCount}u; candidate++) { let score = scores[scoresOffset + candidate]; if (finite(score) && score > best) { best = score; selected = candidate; } }
let coordinate = vec2<i32>(i32(selected % ${searchWidth}u), i32(selected / ${searchWidth}u)); var secondary = -2.0;
for (var candidate = 0u; candidate < ${searchCount}u; candidate++) { let position = vec2<i32>(i32(candidate % ${searchWidth}u), i32(candidate / ${searchWidth}u)); if (any(abs(position - coordinate) > vec2<i32>(1))) { let score = scores[scoresOffset + candidate]; if (finite(score)) { secondary = max(secondary, score); } } }
var code = 0u; if (best < ${props.minimumCorrelation ?? 0}) { code = 3u; } else if (best - secondary <= max(abs(best), 1e-12) * ${props.ambiguityTolerance ?? 1e-4}) { code = 2u; }
coarse[coarseOffset] = f32(coordinate.x - ${maximumShift[0]}); coarse[coarseOffset + 1u] = f32(coordinate.y - ${maximumShift[1]}); status[statusOffset] = code;`
    })
  );
  nodes.push(
    ...getMaskedRefinementNodes(
      graph,
      props,
      id,
      coarse,
      statistics,
      refined,
      refinedSupport,
      candidateWidth,
      radius,
      upsample
    )
  );
  return nodes;
}

function getMaskedRefinementNodes<Parameters>(
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
  const width = props.grid.width;
  const height = props.grid.height;
  const count = width * height;
  const candidateCount = candidateWidth ** 2;
  const maximumShift = props.maximumShift ?? [width - 1, height - 1];
  const shiftOutput = graph.createDataView(props.output.shift.buffer, {
    format: 'float32',
    byteOffset: props.output.shift.byteOffset,
    length: props.output.shift.length * 2
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-masked-refine`,
      operation: 'GPURasterRegistration',
      invocationCount: candidateCount,
      workgroupSize: 32,
      bindings: [
        rasterBinding('reference', props.reference),
        rasterBinding('moving', props.moving),
        ...(props.referenceValidity
          ? [rasterBinding('referenceValidity', props.referenceValidity)]
          : []),
        ...(props.movingValidity ? [rasterBinding('movingValidity', props.movingValidity)] : []),
        rasterBinding('coarse', coarse),
        rasterBinding('statistics', statistics),
        rasterBinding('refined', refined, true),
        rasterBinding('refinedSupport', refinedSupport, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `let shift = vec2<f32>(coarse[coarseOffset], coarse[coarseOffset + 1u]) + (vec2<f32>(f32(index % ${candidateWidth}u), f32(index / ${candidateWidth}u)) - vec2<f32>(${radius}.0)) / ${upsample}.0;
refined[refinedOffset + index] = missing(); refinedSupport[refinedSupportOffset + index] = 0u;
if (any(abs(shift) > vec2<f32>(${maximumShift[0]}.0, ${maximumShift[1]}.0))) { return; }
var support = 0u; var referenceMean = 0.0; var movingMean = 0.0; var covariance = 0.0; var referenceVariance = 0.0; var movingVariance = 0.0;
for (var cell = 0u; cell < ${count}u; cell++) {
  let referenceValue = reference[referenceOffset + cell] - statistics[statisticsOffset];
  if (!finite(referenceValue) ${props.referenceValidity ? '|| referenceValidity[referenceValidityOffset + cell] == 0u' : ''}) { continue; }
  let coordinate = vec2<f32>(f32(cell % ${width}u), f32(cell / ${width}u)) - shift;
  if (any(coordinate < vec2<f32>(0.0)) || any(coordinate > vec2<f32>(${width - 1}.0, ${height - 1}.0))) { continue; }
  let lower = vec2<i32>(floor(coordinate)); let fraction = fract(coordinate); var movingValue = 0.0; var valid = true;
  for (var corner = 0u; corner < 4u; corner++) {
    let offset = vec2<i32>(i32(corner % 2u), i32(corner / 2u));
    let weight = select(1.0 - fraction.x, fraction.x, offset.x != 0) * select(1.0 - fraction.y, fraction.y, offset.y != 0);
    if (weight <= 0.0) { continue; }
    let position = lower + offset;
    if (any(position < vec2<i32>(0)) || any(position >= vec2<i32>(${width}, ${height}))) { valid = false; break; }
    let source = u32(position.y) * ${width}u + u32(position.x); let value = moving[movingOffset + source];
    if (!finite(value) ${props.movingValidity ? '|| movingValidity[movingValidityOffset + source] == 0u' : ''}) { valid = false; break; }
    movingValue += weight * (value - statistics[statisticsOffset + 1u]);
  }
  if (!valid) { continue; }
  support++; let referenceDifference = referenceValue - referenceMean; let movingDifference = movingValue - movingMean;
  referenceMean += referenceDifference / f32(support); movingMean += movingDifference / f32(support);
  covariance += referenceDifference * (movingValue - movingMean); referenceVariance += referenceDifference * (referenceValue - referenceMean); movingVariance += movingDifference * (movingValue - movingMean);
}
refinedSupport[refinedSupportOffset + index] = support;
let required = max(${props.minimumOverlapCount ?? 3}.0, ceil(min(statistics[statisticsOffset + 2u], statistics[statisticsOffset + 3u]) * ${props.minimumOverlapFraction ?? 0.3}));
if (f32(support) >= required && referenceVariance > 1e-12 && movingVariance > 1e-12) { refined[refinedOffset + index] = clamp(covariance / sqrt(referenceVariance * movingVariance), -1.0, 1.0); }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-masked-finish`,
      operation: 'GPURasterRegistration',
      invocationCount: 1,
      bindings: [
        rasterBinding('refined', refined),
        rasterBinding('refinedSupport', refinedSupport),
        rasterBinding('coarse', coarse),
        rasterBinding('status', props.output.status, true),
        rasterBinding('shift', shiftOutput, true),
        rasterBinding('peak', props.output.peak, true),
        ...(props.output.overlap ? [rasterBinding('overlap', props.output.overlap, true)] : [])
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `var best = -2.0; var selected = 0u; var bestDistance = 0xffffffffu;
for (var candidate = 0u; candidate < ${candidateCount}u; candidate++) {
  let score = refined[refinedOffset + candidate]; let offset = vec2<i32>(i32(candidate % ${candidateWidth}u) - ${radius}, i32(candidate / ${candidateWidth}u) - ${radius}); let distance = u32(offset.x * offset.x + offset.y * offset.y);
  if (finite(score) && (score > best || (score == best && distance < bestDistance))) { best = score; selected = candidate; bestDistance = distance; }
}
var code = status[statusOffset]; if (code == 0u && best < ${props.minimumCorrelation ?? 0}) { code = 3u; }
let valid = code == 0u; let result = vec2<f32>(coarse[coarseOffset], coarse[coarseOffset + 1u]) + (vec2<f32>(f32(selected % ${candidateWidth}u), f32(selected / ${candidateWidth}u)) - vec2<f32>(${radius}.0)) / ${upsample}.0;
shift[shiftOffset] = select(missing(), result.x, valid); shift[shiftOffset + 1u] = select(missing(), result.y, valid); peak[peakOffset] = select(missing(), best, valid); status[statusOffset] = code;
${props.output.overlap ? 'overlap[overlapOffset] = select(0u, refinedSupport[refinedSupportOffset + selected], valid);' : ''}`
    })
  ];
}
