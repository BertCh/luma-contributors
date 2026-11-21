// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addRasterZonalDistributionRecipe,
  addRegionalRasterChangeRecipe,
  type GPURasterZonalDistributionRecipeProps,
  type GPURegionalRasterChangeRecipeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUZonalDistributionOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/** Four fixed, whole-cell quadrants over the 64 by 64 raster crop. */
export function makeDixieQuadrantIds(): Uint32Array {
  const values = new Uint32Array(64 * 64);
  for (let row = 0; row < 64; row++) {
    for (let column = 0; column < 64; column++) {
      values[row * 64 + column] = (row >= 32 ? 2 : 0) + (column >= 32 ? 1 : 0);
    }
  }
  return values;
}

/** Uses the raster adapter, which preserves strict mask support before zonal distribution. */
export function addDixieRasterZonalDistribution(
  graph: GPUCommandGraph<void>,
  props: GPURasterZonalDistributionRecipeProps
) {
  return addRasterZonalDistributionRecipe(graph, props);
}

/** Two-date common-support quadrant summary with exact projected cell areas. */
export function addDixieRegionalChange(
  graph: GPUCommandGraph<void>,
  props: GPURegionalRasterChangeRecipeProps
) {
  return addRegionalRasterChangeRecipe(graph, props);
}

/** One graph with calibrated whole-cell distributions for both dates and paired change support. */
export function addDixieRegionalSupport(
  graph: GPUCommandGraph<void>,
  props: {
    grid: GPURasterMetadata;
    before: GPURasterBufferBand;
    after: GPURasterBufferBand;
    zones: Parameters<typeof addDixieRasterZonalDistribution>[1]['zones'];
    probabilities: Parameters<typeof addDixieRasterZonalDistribution>[1]['probabilities'];
    beforeOutputs?: Partial<GPUZonalDistributionOutput>;
    afterOutputs?: Partial<GPUZonalDistributionOutput>;
    change: GPURegionalRasterChangeRecipeProps['change'];
    cellAreas: GPURegionalRasterChangeRecipeProps['cellAreas'];
    threshold: GPURegionalRasterChangeRecipeProps['threshold'];
  }
) {
  const beforeDistribution = addDixieRasterZonalDistribution(graph, {
    id: 'dixie-regions-before',
    grid: props.grid,
    input: props.before,
    zones: props.zones,
    zoneCount: 4,
    probabilities: props.probabilities,
    output: props.beforeOutputs
  });
  const afterDistribution = addDixieRasterZonalDistribution(graph, {
    id: 'dixie-regions-after',
    grid: props.grid,
    input: props.after,
    zones: props.zones,
    zoneCount: 4,
    probabilities: props.probabilities,
    output: props.afterOutputs
  });
  const change = addDixieRegionalChange(graph, {
    id: 'dixie-regional-change',
    grid: props.grid,
    change: props.change,
    zones: props.zones,
    zoneCount: 4,
    cellAreas: props.cellAreas,
    threshold: props.threshold
  });
  return {beforeDistribution, afterDistribution, change};
}
