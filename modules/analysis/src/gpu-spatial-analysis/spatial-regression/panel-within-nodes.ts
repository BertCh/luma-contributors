// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGPUStorageBindingByteLength,
  validateCapacity
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';
import {getRegressionTileLayout} from './regression-row-tiles';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

const PANEL_CENTER_ARITHMETIC_WGSL = `${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
fn addPanelCenters(first: vec2<f32>, second: vec2<f32>) -> vec2<f32> {
  let high = first.x + second.x;
  let low = first.y + second.y + getAdditionRoundingError(first.x, second.x, high);
  let combined = high + low;
  return vec2<f32>(combined, getAdditionRoundingError(high, low, combined));
}`;

type PanelWithinProps = {
  id: string;
  entityCount: number;
  periodCount: number;
  predictorCount: number;
  effects?: 'unit' | 'time';
  predictors: GraphDataView<'float32'>;
  response: GraphDataView<'float32'>;
  overflow?: GraphDataView<'uint32'>;
  output: {
    predictors: GraphDataView<'float32'>;
    response: GraphDataView<'float32'>;
    /** Optional split means, including the final runtime-zero word; exact scratch shape. */
    means?: GraphDataView<'float32'>;
  };
};

/** Actual mean-scratch shapes, also used before the parent declares any resources. */
export function getPanelWithinScratchLengths(
  entityCount: number,
  periodCount: number,
  predictorCount: number,
  effects: 'unit' | 'time' = 'unit'
): {partialLength: number; meanLength: number; tileCount: number} {
  const {tileCount} = getRegressionTileLayout(effects === 'time' ? entityCount : periodCount);
  const meanCount = (effects === 'time' ? periodCount : entityCount) * (predictorCount + 1);
  return {partialLength: meanCount * tileCount * 2, meanLength: meanCount * 2 + 1, tileCount};
}

/**
 * Two-part tiled means without a reference observation. Scaling by a power of two keeps sums
 * finite; integer rounding errors preserve both large common offsets and isolated outliers.
 * The final residual subtracts the high and low center separately, rounded as written.
 */
export function getPanelWithinNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PanelWithinProps
): GPUCommandNode<Parameters>[] {
  const {entityCount, periodCount, predictorCount, id} = props;
  const timeEffects = props.effects === 'time';
  const groupCount = timeEffects ? periodCount : entityCount;
  const observationCount = timeEffects ? entityCount : periodCount;
  const {partialLength, meanLength, tileCount} = getPanelWithinScratchLengths(
    entityCount,
    periodCount,
    predictorCount,
    props.effects
  );
  validateGraphChunkBindingRanges(graph, id, {
    predictors: props.predictors,
    response: props.response,
    overflow: props.overflow,
    outputPredictors: props.output.predictors,
    outputResponse: props.output.response,
    outputMeans: props.output.means
  });
  const {tileRowCount: tilePeriodCount} = getRegressionTileLayout(observationCount);
  const scaleExponent = Math.ceil(Math.log2(observationCount));
  const scale = 2 ** -scaleExponent;
  const divisor = observationCount * scale;
  for (const [name, length] of [
    ['panel mean partials', partialLength],
    ['panel means', meanLength]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      length * 4,
      'reduce panel dimensions'
    );
  }
  const partial = createTransientView(graph, `${id}-mean-partials`, 'float32', partialLength);
  if (props.output.means)
    validateCapacity(id, 'output.means.length', props.output.means.length, meanLength, meanLength);
  const means =
    props.output.means ?? createTransientView(graph, `${id}-means`, 'float32', meanLength);
  const sourceBindings = [
    {name: 'sourcePredictors', view: props.predictors, type: 'f32', access: 'read'},
    {name: 'sourceResponse', view: props.response, type: 'f32', access: 'read'}
  ] as const;
  // A storage-written zero keeps rounded differences intact under fast math.
  const declarations = `${PANEL_CENTER_ARITHMETIC_WGSL}
fn readPanelValue(entity: u32, period: u32, component: u32) -> f32 {
  let row = ${timeEffects ? `entity * ${entityCount}u + period` : `period * ${entityCount}u + entity`};
  if (component == 0u) { return sourceResponse[sourceResponseOffset + row]; }
  return sourcePredictors[sourcePredictorsOffset + row * ${predictorCount}u + component - 1u];
}`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-mean-initialize`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'within-mean-initialize',
      invocationCount: 1,
      bindings: [{name: 'means', view: means, type: 'f32', access: 'read_write'}],
      body: `means[meansOffset + ${meanLength - 1}u] = 0.0;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-mean-tiles`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'within-mean-tiles',
      invocationCount: partialLength / 2,
      workgroupSize: 64,
      bindings: [
        ...sourceBindings,
        {name: 'means', view: means, type: 'f32', access: 'read'},
        {name: 'partial', view: partial, type: 'f32', access: 'read_write'}
      ],
      declarations,
      body: `let tile = index % ${tileCount}u;
let column = index / ${tileCount}u;
let entity = column / ${predictorCount + 1}u;
let component = column % ${predictorCount + 1}u;
let runtimeZero = bitcast<u32>(means[meansOffset + ${meanLength - 1}u]);
var total = vec2<f32>(0.0);
for (var period = tile * ${tilePeriodCount}u; period < min((tile + 1u) * ${tilePeriodCount}u, ${observationCount}u); period++) {
  let value = getRoundedDifference(readPanelValue(entity, period, component) * ${scale}, 0.0, runtimeZero);
  total = addPanelCenters(total, vec2<f32>(value, 0.0));
}
partial[partialOffset + index * 2u] = total.x;
partial[partialOffset + index * 2u + 1u] = total.y;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-mean-merge`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'within-mean-merge',
      invocationCount: (meanLength - 1) / 2,
      workgroupSize: 64,
      bindings: [
        {name: 'partial', view: partial, type: 'f32', access: 'read'},
        {name: 'means', view: means, type: 'f32', access: 'read_write'}
      ],
      declarations: PANEL_CENTER_ARITHMETIC_WGSL,
      body: `var levels: array<vec2<f32>, 32>;
var depth = 0u;
for (var tile = 0u; tile < ${tileCount}u; tile++) {
  let start = partialOffset + (index * ${tileCount}u + tile) * 2u;
  var centerBlock = vec2<f32>(partial[start], partial[start + 1u]);
  var carry = tile;
  for (var level = 0u; level < 32u; level++) {
    if ((carry & 1u) == 0u) { break; }
    depth--;
    centerBlock = addPanelCenters(levels[depth], centerBlock);
    carry >>= 1u;
  }
  levels[depth] = centerBlock;
  depth++;
}
var total = vec2<f32>(0.0);
for (var level = 0u; level < 32u; level++) {
  if (depth == 0u) { break; }
  depth--;
  total = addPanelCenters(levels[depth], total);
}
let high = total.x / ${divisor};
let product = high * ${divisor};
let runtimeZero = bitcast<u32>(means[meansOffset + ${meanLength - 1}u]);
let remainder = getRoundedDifference(total.x, product, runtimeZero)
  - getMultiplicationRoundingError(high, ${divisor}, product) + total.y;
means[meansOffset + index * 2u] = high;
means[meansOffset + index * 2u + 1u] = remainder / ${divisor};`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-within`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: timeEffects ? 'time-demean' : 'unit-demean',
      invocationCount: groupCount * observationCount,
      workgroupSize: 64,
      bindings: [
        ...sourceBindings,
        {name: 'means', view: means, type: 'f32', access: 'read'},
        {name: 'predictors', view: props.output.predictors, type: 'f32', access: 'read_write'},
        {name: 'response', view: props.output.response, type: 'f32', access: 'read_write'},
        ...(props.overflow
          ? [{name: 'overflow', view: props.overflow, type: 'u32', access: 'read'} as const]
          : [])
      ],
      declarations: `${declarations}\n${GET_NAN_WGSL}`,
      body: `let entity = ${timeEffects ? `index / ${entityCount}u` : `index % ${entityCount}u`};
let period = ${timeEffects ? `index % ${entityCount}u` : `index / ${entityCount}u`};
let runtimeZero = bitcast<u32>(means[meansOffset + ${meanLength - 1}u]);
for (var component = 0u; component < ${predictorCount + 1}u; component++) {
  let center = meansOffset + (entity * ${predictorCount + 1}u + component) * 2u;
  let spread = getRoundedDifference(readPanelValue(entity, period, component), means[center], runtimeZero);
  let centered = getRoundedDifference(spread, means[center + 1u], runtimeZero);
  if (component == 0u) { response[responseOffset + index] = centered; }
  else { predictors[predictorsOffset + index * ${predictorCount}u + component - 1u] = centered; }
}
${props.overflow ? 'if (overflow[overflowOffset] != 0u) { response[responseOffset + index] = getNaN(); }' : ''}`
    })
  ];
}
