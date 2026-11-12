// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterHaloStage, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../raster-band-utils';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterViews,
  validateRasterGraph
} from './raster-kernel-utils';

export type GPURasterTextureProps = {
  id?: string;
  grid: GPURasterMetadata;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
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
/** Quantization levels: each invocation keeps a levels squared u32 histogram in private memory (32 squared is 4 KiB). */
export const MAXIMUM_TEXTURE_LEVELS = 32;
/** Window radius and offset magnitude: each cell scans (2 * radius + 1) squared neighbours in one invocation. */
export const MAXIMUM_TEXTURE_RADIUS = 8;

/** Local normalized GLCM; natural-log entropy, constant-window correlation 1, empty features NaN.
 * Run once per direction; symmetry counts each directed pair twice. requiredHalo is radius,
 * since endpoints outside each window are excluded. Interior seams require a source halo.
 */
export class GPURasterTexture implements GPUCommandNodeProducer, GPURasterHaloStage {
  readonly id: string;
  readonly requiredHalo: number;
  constructor(readonly props: GPURasterTextureProps) {
    this.id = props.id ?? 'raster-texture';
    validateRasterGrid(this.id, props.grid);
    this.requiredHalo = props.radius;
    validateCapacity(this.id, 'grid.width', props.grid.width);
    validateCapacity(this.id, 'grid.height', props.grid.height);
    const count = props.grid.width * props.grid.height;
    validateCapacity(this.id, 'feature rows (seven per cell)', count * 7);
    validateCapacity(this.id, 'levels', props.levels, 2, MAXIMUM_TEXTURE_LEVELS);
    validateCapacity(this.id, 'radius', props.radius, 1, MAXIMUM_TEXTURE_RADIUS);
    validateRequirement(
      this.id,
      props.range.every(value => Number.isFinite(Math.fround(value))) &&
        props.range[1] > props.range[0] &&
        Number.isFinite(Math.fround(props.range[1] - props.range[0])),
      'range must be finite float32 values with range[1] > range[0]'
    );
    validateRequirement(
      this.id,
      props.offset.every(
        value => Number.isInteger(value) && Math.abs(value) <= MAXIMUM_TEXTURE_RADIUS
      ) && props.offset.some(value => value !== 0),
      `offset must be nonzero integers with magnitude at most ${MAXIMUM_TEXTURE_RADIUS}`
    );
    validateRequirement(
      this.id,
      props.values.length >= count && (!props.mask || props.mask.length >= count),
      'values and mask must hold at least grid.width times grid.height entries'
    );
    validateRequirement(
      this.id,
      props.features.length >= count * 7 && props.pairCount.length >= count,
      'features must hold seven entries per cell and pairCount one per cell'
    );
    validateRasterViews(
      this.id,
      [props.values, ...(props.mask ? [props.mask] : [])],
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
      [props.values, ...(props.mask ? [props.mask] : [])],
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
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
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
    if (!finite(firstValue) || !finite(secondValue) ${props.mask ? '|| mask[maskOffset + firstIndex] == 0u || mask[maskOffset + secondIndex] == 0u' : ''}) { continue; }
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
