// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../raster-band-utils';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterCount,
  validateRasterViews,
  validateRasterGraph
} from './raster-kernel-utils';

export type GPURasterTextureProps = {
  id?: string;
  grid: GPURasterMetadata;
  values: GraphDataView<'float32'>;
  validity?: GraphDataView<'uint32'>;
  /** 2–32 quantization bins, clamped to the supplied finite increasing range. */
  levels: number;
  range: readonly [number, number];
  /** Square radius 1–8. Both pair endpoints must lie inside the clipped window. */
  radius: number;
  /** One nonzero integer direction with both components in [-8, 8]. */
  offset: readonly [number, number];
  symmetric?: boolean;
  /** Seven floats per cell: contrast, dissimilarity, homogeneity, ASM, energy, correlation, entropy. */
  features: GraphDataView<'float32'>;
  pairCount: GraphDataView<'uint32'>;
};
/** Local normalized GLCM; natural-log entropy, constant-window correlation 1, empty features NaN.
 * Run once per direction; symmetry counts each directed pair twice. requiredHalo is radius,
 * since endpoints outside each window are excluded. Interior seams require a source halo.
 */
export class GPURasterTexture implements GPUCommandNodeProducer {
  readonly id: string;
  readonly requiredHalo: number;
  constructor(readonly props: GPURasterTextureProps) {
    this.id = props.id ?? 'raster-texture';
    validateRasterGrid(this.id, props.grid);
    this.requiredHalo = props.radius;
    validateRasterCount(props.grid.width);
    validateRasterCount(props.grid.height);
    const count = props.grid.width * props.grid.height;
    validateRasterCount(count * 7);
    validateRasterCount(props.levels, 32);
    assert(props.levels >= 2);
    validateRasterCount(props.radius, 8);
    assert(
      props.range.every(value => Number.isFinite(Math.fround(value))) &&
        props.range[1] > props.range[0] &&
        Number.isFinite(Math.fround(props.range[1] - props.range[0]))
    );
    assert(
      props.offset.every(value => Number.isInteger(value) && Math.abs(value) <= 8) &&
        props.offset.some(value => value !== 0)
    );
    assert(props.values.length >= count && (!props.validity || props.validity.length >= count));
    assert(props.features.length >= count * 7 && props.pairCount.length >= count);
    validateRasterViews(
      this.id,
      [props.values, ...(props.validity ? [props.validity] : [])],
      [props.features, props.pairCount]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateRasterGraph(
      id,
      graph,
      [props.values, ...(props.validity ? [props.validity] : [])],
      [props.features, props.pairCount]
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterTexture',
        invocationCount: props.grid.width * props.grid.height,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', props.values),
          ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
          rasterBinding('features', props.features, true),
          rasterBinding('pairCount', props.pairCount, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}
fn quantize(value: f32) -> u32 { return u32(clamp(floor((value - ${props.range[0]}) / ${props.range[1] - props.range[0]} * ${props.levels}.0), 0.0, ${props.levels - 1}.0)); }`,
        body: `var histogram: array<u32, ${props.levels * props.levels}>;
let center = vec2<i32>(i32(index % ${props.grid.width}u), i32(index / ${props.grid.width}u));
let lower = max(center - vec2<i32>(${props.radius}), vec2<i32>(0));
let upper = min(center + vec2<i32>(${props.radius}), vec2<i32>(${props.grid.width - 1}, ${props.grid.height - 1}));
var count = 0u;
for (var row = lower.y; row <= upper.y; row++) {
  for (var column = lower.x; column <= upper.x; column++) {
    let following = vec2<i32>(column, row) + vec2<i32>(${props.offset[0]}, ${props.offset[1]});
    if (any(following < lower) || any(following > upper)) { continue; }
    let firstIndex = u32(row) * ${props.grid.width}u + u32(column);
    let secondIndex = u32(following.y) * ${props.grid.width}u + u32(following.x);
    let firstValue = values[valuesOffset + firstIndex]; let secondValue = values[valuesOffset + secondIndex];
    if (!finite(firstValue) || !finite(secondValue) ${props.validity ? '|| validity[validityOffset + firstIndex] == 0u || validity[validityOffset + secondIndex] == 0u' : ''}) { continue; }
    let first = quantize(firstValue); let second = quantize(secondValue);
    histogram[first * ${props.levels}u + second]++; count++;
    ${props.symmetric ? `histogram[second * ${props.levels}u + first]++; count++;` : ''}
  }
}
pairCount[pairCountOffset + index] = count;
for (var metric = 0u; metric < 7u; metric++) { features[featuresOffset + index * 7u + metric] = missing(); }
if (count == 0u) { return; }
var contrast = 0.0; var dissimilarity = 0.0; var homogeneity = 0.0; var angular = 0.0; var entropy = 0.0; var firstMean = 0.0; var secondMean = 0.0;
for (var cell = 0u; cell < ${props.levels * props.levels}u; cell++) {
  let probability = f32(histogram[cell]) / f32(count); let first = f32(cell / ${props.levels}u); let second = f32(cell % ${props.levels}u); let difference = first - second;
  contrast += probability * difference * difference; dissimilarity += probability * abs(difference); homogeneity += probability / (1.0 + difference * difference); angular += probability * probability;
  if (probability > 0.0) { entropy -= probability * log(probability); }
  firstMean += probability * first; secondMean += probability * second;
}
var covariance = 0.0; var firstVariance = 0.0; var secondVariance = 0.0;
for (var cell = 0u; cell < ${props.levels * props.levels}u; cell++) {
  let probability = f32(histogram[cell]) / f32(count); let first = f32(cell / ${props.levels}u) - firstMean; let second = f32(cell % ${props.levels}u) - secondMean;
  covariance += probability * first * second; firstVariance += probability * first * first; secondVariance += probability * second * second;
}
var correlation = 1.0;
if (firstVariance > 1e-15 && secondVariance > 1e-15) { correlation = covariance / sqrt(firstVariance * secondVariance); }
let result = array<f32, 7>(contrast, dissimilarity, homogeneity, angular, sqrt(angular), correlation, entropy);
for (var metric = 0u; metric < 7u; metric++) { features[featuresOffset + index * 7u + metric] = result[metric]; }`
      })
    ];
  }
}
