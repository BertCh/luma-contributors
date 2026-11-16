// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFFT2D,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, createRasterArgmaxNodes, rasterBinding} from './raster-kernel-utils';
import {validateRasterWritableDispatches} from '../raster-time-series/raster-time-series-chunks-b';
import {
  validateRasterStorageBindings,
  validateRasterRegistrationFFT
} from './raster-binding-validation';
import {
  createMaskedRegistrationMeanNodes,
  createMaskedRegistrationRefinementNodes,
  MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE,
  MASKED_REGISTRATION_MEAN_WORD_COUNT,
  MASKED_REGISTRATION_MOMENT_WORD_COUNT
} from './raster-masked-registration-reductions';
import type {GPURasterRegistrationProps} from './gpu-raster-registration';

/** Masked mode side length: six padded (about twice the side, rounded to a power of two) correlation fields plus FFT scratch must fit device storage (512 gives 1024 squared padded cells). */
export const MAXIMUM_MASKED_REGISTRATION_DIMENSION = 512;
/** Same refinement-window limit as periodic mode: (2 * ceil(factor / 2) + 1) squared candidates. */
const MAXIMUM_MASKED_UPSAMPLE_FACTOR = 32;
/** Fewer paired cells make a Pearson correlation undefined or meaningless (three points fit any line). */
const MINIMUM_MASKED_OVERLAP_COUNT = 3;

export function validateMaskedRegistrationProps(
  props: GPURasterRegistrationProps,
  id: string
): void {
  validateCapacity(id, 'grid.width', props.grid.width, 2, MAXIMUM_MASKED_REGISTRATION_DIMENSION);
  validateCapacity(id, 'grid.height', props.grid.height, 2, MAXIMUM_MASKED_REGISTRATION_DIMENSION);
  validateCapacity(
    id,
    'upsampleFactor',
    props.upsampleFactor ?? 20,
    1,
    MAXIMUM_MASKED_UPSAMPLE_FACTOR
  );
  const maximumShift = props.maximumShift ?? [props.grid.width - 1, props.grid.height - 1];
  validateRequirement(
    id,
    maximumShift.every(
      (value, dimension) =>
        Number.isInteger(value) &&
        value >= 0 &&
        value < (dimension === 0 ? props.grid.width : props.grid.height)
    ),
    'maximumShift must be nonnegative integers below grid.width and grid.height'
  );
  const fraction = props.minimumOverlapFraction ?? 0.3;
  validateRequirement(
    id,
    Number.isFinite(fraction) && fraction >= 0 && fraction <= 1,
    'minimumOverlapFraction must be a finite number in [0, 1]'
  );
  validateCapacity(
    id,
    'minimumOverlapCount',
    props.minimumOverlapCount ?? MINIMUM_MASKED_OVERLAP_COUNT,
    MINIMUM_MASKED_OVERLAP_COUNT,
    props.grid.width * props.grid.height
  );
  const minimumCorrelation = props.minimumCorrelation ?? 0;
  validateRequirement(
    id,
    Number.isFinite(minimumCorrelation) && minimumCorrelation >= -1 && minimumCorrelation <= 1,
    'minimumCorrelation must be a finite number in [-1, 1]'
  );
  const ambiguity = props.ambiguityTolerance ?? 1e-4;
  validateRequirement(
    id,
    Number.isFinite(ambiguity) && ambiguity >= 0 && ambiguity < 1,
    'ambiguityTolerance must be a finite number in [0, 1)'
  );
  const inputs = [props.reference, props.moving, props.referenceMask, props.movingMask].filter(
    view => view !== undefined
  );
  const outputs = Object.values(props.output).filter(view => view !== undefined);
  for (const view of inputs) {
    validateSinglePackedView(view, [view.format], id);
    validateRequirement(
      id,
      view.length >= props.grid.width * props.grid.height,
      'reference, moving, referenceMask and movingMask views must hold at least grid.width times grid.height values'
    );
  }
  for (const view of outputs) {
    validateSinglePackedView(view, [view.format], id);
    validateRequirement(id, view.length >= 1, 'output views must not be empty');
  }
  validateRegistrationWritableViews(props, id);
}

/** Caller inputs are consumed before output stages; status is read-only only in periodic finish. */
export function validateRegistrationWritableViews(
  props: GPURasterRegistrationProps,
  id: string
): void {
  const inputs = [props.reference, props.moving, props.referenceMask, props.movingMask].filter(
    view => view !== undefined
  );
  const outputs = Object.values(props.output).filter(view => view !== undefined);
  const masked = props.mode === 'masked-linear';
  validateRasterWritableDispatches(id, inputs, outputs, [
    {inputs, outputs: []},
    {
      inputs: [],
      outputs: [
        props.output.status,
        ...(props.output.secondPeakRatio ? [props.output.secondPeakRatio] : [])
      ]
    },
    {
      inputs: masked ? [] : [props.output.status],
      outputs: [
        props.output.shift,
        props.output.peak,
        ...(masked
          ? [props.output.status, ...(props.output.overlap ? [props.output.overlap] : [])]
          : [])
      ]
    }
  ]);
}

function getPaddedDimension(length: number): number {
  return 2 ** Math.ceil(Math.log2(2 * length - 1));
}

export function getMaskedRegistrationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegistrationProps,
  id: string
): GPUCommandNode<Parameters>[] {
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
  validateRasterStorageBindings(
    graph,
    id,
    [
      props.reference,
      props.moving,
      ...[props.referenceMask, props.movingMask].filter(view => view !== undefined),
      ...Object.values(props.output).filter(view => view !== undefined)
    ],
    {
      'masked FFT input and spectrum fields': paddedCount * 3 * 8,
      'masked correlation fields': paddedCount * 6 * 8,
      'masked statistics': 16,
      'masked scores and supports': searchCount * 4,
      'masked coarse shift': 8,
      'masked refinement scores and supports': candidateCount * 4,
      'masked argmax scalars': 4,
      'masked argmax partials': Math.ceil(searchCount / 256) * 4,
      'masked mean partials':
        Math.ceil(count / MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE) *
        MASKED_REGISTRATION_MEAN_WORD_COUNT *
        4,
      'masked refinement moments':
        Math.ceil(count / MASKED_REGISTRATION_REDUCTION_WORKGROUP_SIZE) *
        candidateCount *
        MASKED_REGISTRATION_MOMENT_WORD_COUNT *
        4
    }
  );
  validateRasterRegistrationFFT(graph, id, paddedWidth, paddedHeight, 6);
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
    ...(props.referenceMask ? [rasterBinding('referenceValidity', props.referenceMask)] : []),
    ...(props.movingMask ? [rasterBinding('movingValidity', props.movingMask)] : [])
  ];
  const referenceValid = (source: string) =>
    `finite(referenceValues[referenceValuesOffset + ${source}])${props.referenceMask ? ` && referenceValidity[referenceValidityOffset + ${source}] != 0u` : ''}`;
  const movingValid = (source: string) =>
    `finite(movingValues[movingValuesOffset + ${source}])${props.movingMask ? ` && movingValidity[movingValidityOffset + ${source}] != 0u` : ''}`;
  nodes.push(...createMaskedRegistrationMeanNodes(graph, props, id, statistics));
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
let required = max(${props.minimumOverlapCount ?? 3}.0, ceil(min(statistics[statisticsOffset + 2u], statistics[statisticsOffset + 3u]) * ${getWGSLFloatLiteral(props.minimumOverlapFraction ?? 0.3)}));
if (support >= required) { let referenceVariance = moments[3] - moments[1] * moments[1] / support; let movingVariance = moments[4] - moments[2] * moments[2] / support;
  if (referenceVariance > max(1e-12, abs(moments[3]) * 1e-6) && movingVariance > max(1e-12, abs(moments[4]) * 1e-6)) { score = clamp((moments[5] - moments[1] * moments[2] / support) / sqrt(referenceVariance * movingVariance), -1.0, 1.0); }
}
scores[scoresOffset + index] = score;`
    })
  );
  const bestScore = createTransientView(graph, `${id}-masked-best-score`, 'float32', 1);
  const bestIndex = createTransientView(graph, `${id}-masked-best-index`, 'uint32', 1);
  const secondaryScore = createTransientView(graph, `${id}-masked-secondary-score`, 'float32', 1);
  const secondaryIndex = createTransientView(graph, `${id}-masked-secondary-index`, 'uint32', 1);
  // Best finite score (lowest search cell wins ties), then the best finite score at least two
  // search cells away from it in either axis; both are parallel argmax reductions.
  nodes.push(
    ...createRasterArgmaxNodes(graph, {
      id: `${id}-masked-best`,
      operation: 'GPURasterRegistration',
      count: searchCount,
      bindings: [rasterBinding('scores', scores)],
      declarations: RASTER_FINITE_WGSL,
      candidate:
        'candidateValue = scores[scoresOffset + index]; candidateEligible = finite(candidateValue);',
      initialValue: '-2.0',
      outputValue: bestScore,
      outputIndex: bestIndex
    }),
    ...createRasterArgmaxNodes(graph, {
      id: `${id}-masked-secondary`,
      operation: 'GPURasterRegistration',
      count: searchCount,
      bindings: [rasterBinding('scores', scores), rasterBinding('bestIndex', bestIndex)],
      declarations: RASTER_FINITE_WGSL,
      candidate: `let selected = bestIndex[bestIndexOffset]; let coordinate = vec2<i32>(i32(selected % ${searchWidth}u), i32(selected / ${searchWidth}u));
    let position = vec2<i32>(i32(index % ${searchWidth}u), i32(index / ${searchWidth}u));
    candidateValue = scores[scoresOffset + index]; candidateEligible = finite(candidateValue) && any(abs(position - coordinate) > vec2<i32>(1));`,
      initialValue: '-2.0',
      outputValue: secondaryScore,
      outputIndex: secondaryIndex
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-masked-coarse`,
      operation: 'GPURasterRegistration',
      invocationCount: 1,
      bindings: [
        rasterBinding('bestScore', bestScore),
        rasterBinding('bestIndex', bestIndex),
        rasterBinding('secondaryScore', secondaryScore),
        rasterBinding('coarse', coarse, true),
        rasterBinding('status', props.output.status, true),
        ...(props.output.secondPeakRatio
          ? [rasterBinding('secondPeakRatio', props.output.secondPeakRatio, true)]
          : [])
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `let best = bestScore[bestScoreOffset]; let secondary = secondaryScore[secondaryScoreOffset]; let selected = bestIndex[bestIndexOffset];
let coordinate = vec2<i32>(i32(selected % ${searchWidth}u), i32(selected / ${searchWidth}u));
var code = 0u; if (best < ${getWGSLFloatLiteral(props.minimumCorrelation ?? 0)}) { code = 3u; } else if (best - secondary <= max(abs(best), 1e-12) * ${getWGSLFloatLiteral(props.ambiguityTolerance ?? 1e-4)}) { code = 2u; }
${props.output.secondPeakRatio ? 'var ratio = missing(); if (code != 3u && best > 0.0) { ratio = clamp(max(secondary, 0.0) / best, 0.0, 1.0); } secondPeakRatio[secondPeakRatioOffset] = ratio;' : ''}
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
  const candidateCount = candidateWidth ** 2;
  const shiftOutput = graph.createDataView(props.output.shift.buffer, {
    format: 'float32',
    byteOffset: props.output.shift.byteOffset,
    length: props.output.shift.length * 2
  });
  return [
    ...createMaskedRegistrationRefinementNodes(
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
    ),
    createWGSLKernelNode(graph, {
      id: `${id}-masked-finish`,
      operation: 'GPURasterRegistration',
      // Serial over the refinement candidates (at most 33 x 33 = 1089), the tie rule prefers the
      // candidate nearest the coarse peak; the loop is bounded by the upsample factor, not by raster size.
      // contract-allow: single-invocation tie selection over at most (MAXIMUM_MASKED_UPSAMPLE_FACTOR + 1) squared candidates.
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
var code = status[statusOffset]; if (code == 0u && best < ${getWGSLFloatLiteral(props.minimumCorrelation ?? 0)}) { code = 3u; }
let valid = code == 0u; let result = vec2<f32>(coarse[coarseOffset], coarse[coarseOffset + 1u]) + (vec2<f32>(f32(selected % ${candidateWidth}u), f32(selected / ${candidateWidth}u)) - vec2<f32>(${radius}.0)) / ${upsample}.0;
shift[shiftOffset] = select(missing(), result.x, valid); shift[shiftOffset + 1u] = select(missing(), result.y, valid); peak[peakOffset] = select(missing(), best, valid); status[statusOffset] = code;
${props.output.overlap ? 'overlap[overlapOffset] = select(0u, refinedSupport[refinedSupportOffset + selected], valid);' : ''}`
    })
  ];
}
