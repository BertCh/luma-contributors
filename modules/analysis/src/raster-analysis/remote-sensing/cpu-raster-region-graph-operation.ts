// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  MAXIMUM_REGION_GRAPH_CELL_COUNT,
  MAXIMUM_REGION_GRAPH_REGION_COUNT
} from './gpu-raster-region-graph';
import {
  snapshotRasterRegionGraphOperationProps,
  type RasterRegionGraphOperationProps
} from './gpu-raster-region-graph-operation';

export type CPURasterRegionGraphOptions = Pick<
  RasterRegionGraphOperationProps,
  'grid' | 'regionCount' | 'mergeMode' | 'searchStructure' | 'mergeThreshold' | 'maximumIterations'
> & {labelsComplete?: Uint32Array};
export type CPURasterRegionGraphResult = {
  adjacency: Uint32Array;
  weights: Float32Array;
  mapping: Uint32Array;
  /** One row, 1 when another propagation, Boruvka phase or adaptive merge remains possible. */
  iterationLimitReached: Uint32Array;
  /** Source-label completeness; independent of the local merge iteration budget. */
  complete: Uint32Array;
};

/**
 * Builds four-neighbour adjacency and merges regions within the iteration budget. Fixed mode uses original
 * mean differences; adaptive mode repeatedly merges the closest adjacent component pair, with
 * lexicographic source-root ties and observation-count-weighted means.
 *
 * What differs from the GPU contributor:
 * - Accumulation, means and edge comparisons use float64, rounded only in output weights;
 *   threshold decisions near float32 rounding boundaries may differ.
 * - Both modes honor `maximumIterations`, including dense propagation and Boruvka phases.
 *   Auto uses dense propagation for small budgets and Boruvka when the budget guarantees closure.
 *   The CPU has no device-dependent sparse-scratch fallback; select dense explicitly to match it.
 * - The 256-region and 256-round GPU limits are dropped; the uint32 adjacency cell bound remains.
 *
 * @throws If labels/values do not cover the grid, or options are invalid.
 */
export function computeRasterRegionGraphOnCPU(
  labels: Uint32Array,
  values: Float32Array,
  // Preserve legacy annotated operation props when no certificate is supplied. A graph value
  // name is not CPU evidence: callers supplying labelsComplete must pass a Uint32Array instead.
  options: CPURasterRegionGraphOptions | RasterRegionGraphOperationProps
): CPURasterRegionGraphResult {
  validateRegionOptions('computeRasterRegionGraphOnCPU', options);
  validateRequirement(
    'computeRasterRegionGraphOnCPU',
    options.labelsComplete === undefined ||
      (options.labelsComplete instanceof Uint32Array && options.labelsComplete.length === 1),
    'labelsComplete must be a Uint32Array holding exactly one word; pass CPU evidence instead of a graph value name'
  );
  const {grid, regionCount, mergeThreshold} = options;
  const fixed = (options.mergeMode ?? 'fixed') === 'fixed';
  const dense =
    options.searchStructure === 'dense' ||
    ((options.searchStructure ?? 'auto') === 'auto' &&
      regionCount <= MAXIMUM_REGION_GRAPH_REGION_COUNT &&
      options.maximumIterations < regionCount - 1);
  const cellCount = grid.width * grid.height;
  if (labels.length < cellCount)
    throw new Error('computeRasterRegionGraphOnCPU labels must cover grid cells');
  if (values.length < cellCount)
    throw new Error('computeRasterRegionGraphOnCPU values must cover grid cells');
  const adjacency = new Uint32Array(regionCount ** 2);
  const sums = new Float64Array(regionCount);
  const counts = new Float64Array(regionCount);
  const mapping = Uint32Array.from({length: regionCount}, (_, region) => region);
  // Only Boruvka needs ordered edges. The first occurrence of each undirected region pair
  // preserves the raster edge-order tie break without retaining every cell boundary.
  const boundaryEdges: {first: number; second: number}[] | undefined =
    fixed && !dense ? [] : undefined;
  let iterationLimitReached = 0;
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
      if (boundaryEdges && adjacency[label * regionCount + other] === 0)
        boundaryEdges.push({first: label, second: other});
      adjacency[label * regionCount + other]++;
      adjacency[other * regionCount + label]++;
    }
  }
  const originalMeans = Float64Array.from(sums, (sum, region) =>
    counts[region] > 0 ? sum / counts[region] : NaN
  );
  const weights = new Float32Array(regionCount ** 2).fill(NaN);
  if (fixed) {
    const preciseWeights = new Float64Array(regionCount ** 2).fill(NaN);
    // Original-region means define static edge weights for both fixed search routes.
    for (let first = 0; first < regionCount; first++) {
      for (let second = first + 1; second < regionCount; second++) {
        if (adjacency[first * regionCount + second] === 0) continue;
        const weight = Math.abs(originalMeans[first] - originalMeans[second]);
        weights[first * regionCount + second] = weight;
        weights[second * regionCount + first] = weight;
        preciseWeights[first * regionCount + second] = weight;
        preciseWeights[second * regionCount + first] = weight;
      }
    }
    if (dense) {
      for (let round = 0; round < options.maximumIterations; round++) {
        const previous = mapping.slice();
        let changed = false;
        for (let region = 0; region < regionCount; region++) {
          for (let neighbor = 0; neighbor < regionCount; neighbor++) {
            const weight = preciseWeights[region * regionCount + neighbor];
            if (Number.isFinite(weight) && weight <= mergeThreshold)
              mapping[region] = Math.min(mapping[region], previous[neighbor]);
          }
          changed ||= mapping[region] !== previous[region];
        }
        if (!changed) break;
      }
    } else {
      const rounds = Math.min(options.maximumIterations, Math.ceil(Math.log2(regionCount)));
      for (let round = 0; round < rounds; round++) {
        const selectedEdges = new Int32Array(regionCount).fill(-1);
        const selectedWeights = new Float64Array(regionCount).fill(Infinity);
        // Boundaries retain raster traversal order, matching the GPU edge-index tie break.
        boundaryEdges!.forEach(({first, second}, edge) => {
          if (mapping[first] === mapping[second]) return;
          const weight = preciseWeights[first * regionCount + second];
          if (!Number.isFinite(weight) || weight > mergeThreshold) return;
          for (const root of [mapping[first], mapping[second]]) {
            if (weight < selectedWeights[root]) {
              selectedWeights[root] = weight;
              selectedEdges[root] = edge;
            }
          }
        });
        for (const edge of selectedEdges) {
          if (edge < 0) continue;
          const {first, second} = boundaryEdges![edge];
          mergeRoots(mapping, mapping[first], mapping[second]);
        }
      }
    }
    for (let region = 0; region < regionCount; region++) {
      for (let neighbor = 0; neighbor < regionCount; neighbor++) {
        const weight = preciseWeights[region * regionCount + neighbor];
        if (
          Number.isFinite(weight) &&
          weight <= mergeThreshold &&
          mapping[neighbor] < mapping[region]
        )
          iterationLimitReached = 1;
      }
    }
  } else {
    for (let round = 0; round <= options.maximumIterations; round++) {
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
      if (round === options.maximumIterations) {
        iterationLimitReached = 1;
        break;
      }
      mergeRoots(mapping, selectedFirst, selectedSecond);
    }
  }
  return {
    adjacency,
    weights,
    mapping,
    iterationLimitReached: new Uint32Array([iterationLimitReached]),
    complete: new Uint32Array([
      Number(options.labelsComplete === undefined || options.labelsComplete[0] !== 0)
    ])
  };
}

function mergeRoots(mapping: Uint32Array, first: number, second: number): void {
  const lower = Math.min(first, second);
  const upper = Math.max(first, second);
  for (let region = 0; region < mapping.length; region++)
    if (mapping[region] === upper) mapping[region] = lower;
}

function validateRegionOptions(
  id: string,
  options: Omit<CPURasterRegionGraphOptions, 'labelsComplete'>
): void {
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
  validateRequirement(
    id,
    ['auto', 'dense', 'boruvka'].includes(options.searchStructure ?? 'auto'),
    "searchStructure must be 'auto', 'dense' or 'boruvka'"
  );
  validateRequirement(
    id,
    options.mergeMode !== 'adaptive' || options.searchStructure !== 'boruvka',
    "searchStructure 'boruvka' requires mergeMode 'fixed'"
  );
}

/** CPU twin; drops GPU region/round caps and preserves the explicit iteration budget. */
export class CPURasterRegionGraphOperation extends GPUCPUOperation {
  constructor(inputProps: RasterRegionGraphOperationProps) {
    const props = snapshotRasterRegionGraphOperationProps(inputProps);
    const id = props.id ?? 'raster-region-graph';
    validateRegionOptions(id, props);
    const inputNames = {
      labels: props.labels,
      values: props.values,
      ...(props.labelsComplete !== undefined ? {labelsComplete: props.labelsComplete} : {})
    };
    validateOperationGraphNames(id, inputNames, {
      adjacency: props.output.adjacency,
      weights: props.output.weights,
      mapping: props.output.mapping,
      iterationLimitReached: props.output.status.iterationLimitReached,
      ...(props.output.status.complete !== undefined
        ? {complete: props.output.status.complete}
        : {})
    });
    validateRequirement(
      id,
      props.labelsComplete === undefined || props.output.status.complete !== undefined,
      'labelsComplete requires output.status.complete to retain source-label completeness'
    );
    super({
      id,
      inputs: inputNames,
      outputs: {
        [props.output.adjacency]: gpuGraph.vector('uint32', {length: props.regionCount ** 2}),
        [props.output.weights]: gpuGraph.vector('float32', {length: props.regionCount ** 2}),
        [props.output.mapping]: gpuGraph.vector('uint32', {length: props.regionCount}),
        [props.output.status.iterationLimitReached]: gpuGraph.vector('uint32', {length: 1}),
        ...(props.output.status.complete !== undefined
          ? {[props.output.status.complete]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const result = computeRasterRegionGraphOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'labels'),
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          {
            ...props,
            labelsComplete:
              props.labelsComplete !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'labelsComplete')
                : undefined
          }
        );
        return {
          [props.output.adjacency]: result.adjacency,
          [props.output.weights]: result.weights,
          [props.output.mapping]: result.mapping,
          [props.output.status.iterationLimitReached]: result.iterationLimitReached,
          ...(props.output.status.complete !== undefined
            ? {[props.output.status.complete]: result.complete}
            : {})
        };
      }
    });
  }
}
