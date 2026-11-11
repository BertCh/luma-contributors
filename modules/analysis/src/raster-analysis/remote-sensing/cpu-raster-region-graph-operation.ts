// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {MAXIMUM_REGION_GRAPH_CELL_COUNT} from './gpu-raster-region-graph';
import type {RasterRegionGraphOperationProps} from './gpu-raster-region-graph-operation';

export type CPURasterRegionGraphOptions = Pick<
  RasterRegionGraphOperationProps,
  'grid' | 'regionCount' | 'mergeMode' | 'mergeThreshold' | 'maximumIterations'
>;
export type CPURasterRegionGraphResult = {
  adjacency: Uint32Array;
  weights: Float32Array;
  mapping: Uint32Array;
  /** One row, zero after the complete CPU merge. */
  iterationLimitReached: Uint32Array;
};

/**
 * Builds four-neighbour adjacency and merges regions to completion. Fixed mode uses original
 * mean differences; adaptive mode repeatedly merges the closest adjacent component pair, with
 * lexicographic source-root ties and observation-count-weighted means.
 *
 * What differs from the GPU contributor:
 * - Accumulation, means and edge comparisons use float64, rounded only in output weights;
 *   threshold decisions near float32 rounding boundaries may differ.
 * - Both modes finish all possible merges, ignoring the GPU's `maximumIterations` budget.
 *   `iterationLimitReached` is always zero. Fixed mode returns complete connected components.
 * - The 256-region and 256-round GPU limits are dropped; the uint32 adjacency cell bound remains.
 *
 * @throws If labels/values do not cover the grid, or options are invalid.
 */
export function computeRasterRegionGraphOnCPU(
  labels: Uint32Array,
  values: Float32Array,
  options: CPURasterRegionGraphOptions
): CPURasterRegionGraphResult {
  validateRegionOptions('computeRasterRegionGraphOnCPU', options);
  const {grid, regionCount, mergeThreshold} = options;
  const cellCount = grid.width * grid.height;
  if (labels.length < cellCount)
    throw new Error('computeRasterRegionGraphOnCPU labels must cover grid cells');
  if (values.length < cellCount)
    throw new Error('computeRasterRegionGraphOnCPU values must cover grid cells');
  const adjacency = new Uint32Array(regionCount ** 2);
  const sums = new Float64Array(regionCount);
  const counts = new Float64Array(regionCount);
  const mapping = Uint32Array.from({length: regionCount}, (_, region) => region);
  for (let cell = 0; cell < cellCount; cell++) {
    const label = labels[cell];
    if (label >= regionCount) continue;
    if (Number.isFinite(values[cell])) {
      sums[label] += values[cell];
      counts[label]++;
    }
    for (const neighbor of [
      (cell % grid.width) + 1 < grid.width ? cell + 1 : cellCount,
      cell + grid.width
    ]) {
      if (neighbor >= cellCount || labels[neighbor] >= regionCount || labels[neighbor] === label)
        continue;
      const other = labels[neighbor];
      adjacency[label * regionCount + other]++;
      adjacency[other * regionCount + label]++;
    }
  }
  const originalMeans = Float64Array.from(sums, (sum, region) =>
    counts[region] > 0 ? sum / counts[region] : NaN
  );
  const weights = new Float32Array(regionCount ** 2).fill(NaN);
  if ((options.mergeMode ?? 'fixed') === 'fixed') {
    // Union original eligible edges; unlike bounded label propagation this reaches closure.
    for (let first = 0; first < regionCount; first++) {
      for (let second = first + 1; second < regionCount; second++) {
        if (adjacency[first * regionCount + second] === 0) continue;
        const weight = Math.abs(originalMeans[first] - originalMeans[second]);
        weights[first * regionCount + second] = weight;
        weights[second * regionCount + first] = weight;
        if (Number.isFinite(weight) && weight <= mergeThreshold)
          mergeRoots(mapping, mapping[first], mapping[second]);
      }
    }
  } else {
    while (true) {
      const componentSums = new Float64Array(regionCount);
      const componentCounts = new Float64Array(regionCount);
      for (let region = 0; region < regionCount; region++) {
        componentSums[mapping[region]] += sums[region];
        componentCounts[mapping[region]] += counts[region];
      }
      const means = Float64Array.from(componentSums, (sum, region) =>
        componentCounts[region] > 0 ? sum / componentCounts[region] : NaN
      );
      let selectedFirst = regionCount;
      let selectedSecond = regionCount;
      let selectedWeight = mergeThreshold;
      weights.fill(NaN);
      for (let first = 0; first < regionCount; first++) {
        for (let second = first + 1; second < regionCount; second++) {
          if (adjacency[first * regionCount + second] === 0 || mapping[first] === mapping[second])
            continue;
          const weight = Math.abs(means[mapping[first]] - means[mapping[second]]);
          weights[first * regionCount + second] = weight;
          weights[second * regionCount + first] = weight;
          const lower = Math.min(mapping[first], mapping[second]);
          const upper = Math.max(mapping[first], mapping[second]);
          if (
            Number.isFinite(weight) &&
            weight <= mergeThreshold &&
            (weight < selectedWeight ||
              (weight === selectedWeight &&
                (lower < selectedFirst || (lower === selectedFirst && upper < selectedSecond))))
          ) {
            selectedFirst = lower;
            selectedSecond = upper;
            selectedWeight = weight;
          }
        }
      }
      if (selectedFirst === regionCount) break;
      mergeRoots(mapping, selectedFirst, selectedSecond);
    }
  }
  return {adjacency, weights, mapping, iterationLimitReached: new Uint32Array([0])};
}

function mergeRoots(mapping: Uint32Array, first: number, second: number): void {
  const lower = Math.min(first, second);
  const upper = Math.max(first, second);
  for (let region = 0; region < mapping.length; region++)
    if (mapping[region] === upper) mapping[region] = lower;
}

function validateRegionOptions(id: string, options: CPURasterRegionGraphOptions): void {
  validateCapacity(id, 'grid.width', options.grid.width);
  validateCapacity(id, 'grid.height', options.grid.height);
  validateCapacity(
    id,
    'grid cell count',
    options.grid.width * options.grid.height,
    1,
    MAXIMUM_REGION_GRAPH_CELL_COUNT
  );
  validateRequirement(
    id,
    options.grid.affine.length === 6 && options.grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite values'
  );
  validateCapacity(id, 'regionCount', options.regionCount);
  validateCapacity(id, 'regionCount squared', options.regionCount ** 2);
  validateCapacity(id, 'maximumIterations', options.maximumIterations);
  validateFiniteNumber(id, 'mergeThreshold', options.mergeThreshold, 0);
  validateRequirement(id, options.mergeThreshold >= 0, 'mergeThreshold must be nonnegative');
  validateRequirement(
    id,
    ['fixed', 'adaptive'].includes(options.mergeMode ?? 'fixed'),
    "mergeMode must be 'fixed' or 'adaptive'"
  );
}

/** CPU twin; drops GPU region/round caps and computes merges to completion. */
export class CPURasterRegionGraphOperation extends GPUCPUOperation {
  constructor(props: RasterRegionGraphOperationProps) {
    const id = props.id ?? 'raster-region-graph';
    validateRegionOptions(id, props);
    super({
      id,
      inputs: {labels: props.labels, values: props.values},
      outputs: {
        [props.output.adjacency]: gpuGraph.vector('uint32', {length: props.regionCount ** 2}),
        [props.output.weights]: gpuGraph.vector('float32', {length: props.regionCount ** 2}),
        [props.output.mapping]: gpuGraph.vector('uint32', {length: props.regionCount}),
        [props.output.status.iterationLimitReached]: gpuGraph.vector('uint32', {length: 1})
      },
      execute({inputs}) {
        const result = computeRasterRegionGraphOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'labels'),
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          props
        );
        return {
          [props.output.adjacency]: result.adjacency,
          [props.output.weights]: result.weights,
          [props.output.mapping]: result.mapping,
          [props.output.status.iterationLimitReached]: result.iterationLimitReached
        };
      }
    });
  }
}
