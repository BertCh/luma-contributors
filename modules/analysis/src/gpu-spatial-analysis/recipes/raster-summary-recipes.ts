// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRecipeView as getOrCreateView} from './expansion-recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPURasterNeighborhood} from '@luma.gl/graph/gpu-raster';
import {GPURasterZonalStatistics} from '../../raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {
  GPUChangeDetection,
  type GPUChangeDetectionProps
} from '../../raster-analysis/change-detection/gpu-change-detection';
import {createRecipeKernelNode as createWGSLKernelNode} from './expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {RecipeBuilder} from './recipe-utils';
import {
  addZonalDistributionRecipe,
  type GPUZonalDistributionRecipeProps
} from './zonal-distribution-recipe';

/** @internal Resolve raw nodata before calibration; every missing output is NaN. */
function prepareBand<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  grid: GPURasterMetadata,
  input: GPURasterBufferBand
): GPURasterBufferBand<'float32'> {
  const count = grid.width * grid.height;
  const values = getOrCreateView(graph, `${id}-values`, 'float32', count);
  const validity = getOrCreateView(graph, `${id}-validity`, 'uint32', count);
  new GPURasterNeighborhood({
    id,
    width: grid.width,
    height: grid.height,
    input,
    output: values,
    outputValidity: validity,
    radius: 0,
    kernel: [1]
  }).addToGraph(graph);
  return {id, format: 'float32', storage: {kind: 'buffer', values}, validity};
}
export type GPURasterZonalDistributionRecipeProps = Omit<
  GPUZonalDistributionRecipeProps,
  'source' | 'values'
> & {
  grid: GPURasterMetadata;
  input: GPURasterBufferBand;
  zones: GraphDataView<'uint32'>;
  zoneCount: number;
};
/** Whole-cell, unweighted distributions, with native-band validity and calibration applied first. */
export function addRasterZonalDistributionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterZonalDistributionRecipeProps
) {
  const id = props.id ?? 'raster-zonal-distribution-recipe';
  const band = prepareBand(graph, `${id}-calibrate`, props.grid, props.input);
  const result = addZonalDistributionRecipe(graph, {
    ...props,
    id,
    values: band.storage.values,
    source: {kind: 'zones', zones: props.zones, zoneCount: props.zoneCount, mask: band.validity}
  });
  return {...result, intermediates: {...result.intermediates, band, grid: props.grid}};
}

export type GPUExposureByZoneRecipeProps = {
  id?: string;
  grid: GPURasterMetadata;
  hazard: GPURasterBufferBand;
  /** Nonnegative count per cell, or density per unit area according to weightMeaning. */
  population: GPURasterBufferBand;
  weightMeaning: 'count' | 'density';
  /** Required for densities. Explicit per-cell areas in the density denominator's units. */
  cellAreas?: GraphDataView<'float32'>;
  /** One whole-cell zone ID per cell. No implicit polygon boundary allocation. */
  zones: GraphDataView<'uint32'>;
  zoneCount: number;
  /** One-row per-frame hazard cutoff; values >= cutoff count as exposed. */
  threshold: GraphDataView<'float32'>;
};
/** Denominators retain population with missing hazards; observed coverage is returned separately. */
export function addExposureByZoneRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUExposureByZoneRecipeProps
) {
  const id = props.id ?? 'exposure-by-zone-recipe';
  const builder = new RecipeBuilder(graph);
  const count = props.grid.width * props.grid.height;
  if (
    props.zones.length !== count ||
    props.threshold.length !== 1 ||
    (props.weightMeaning === 'density' && !props.cellAreas) ||
    (props.cellAreas && props.cellAreas.length !== count)
  )
    throw new Error(`${id}: incompatible grid or missing density areas`);
  const hazard = prepareBand(graph, `${id}-hazard`, props.grid, props.hazard);
  const population = prepareBand(graph, `${id}-population`, props.grid, props.population);
  const totals = getOrCreateView(graph, `${id}-totals`, 'float32', count);
  const observed = getOrCreateView(graph, `${id}-observed`, 'float32', count);
  const exposed = getOrCreateView(graph, `${id}-exposed`, 'float32', count);
  builder.addGlue(
    'weights',
    createWGSLKernelNode(graph, {
      id: `${id}-weights`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'hazard', view: hazard.storage.values, type: 'f32', access: 'read'},
        {name: 'population', view: population.storage.values, type: 'f32', access: 'read'},
        {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
        {name: 'observed', view: observed, type: 'f32', access: 'read_write'},
        {name: 'exposed', view: exposed, type: 'f32', access: 'read_write'},
        ...(props.weightMeaning === 'density' && props.cellAreas
          ? [{name: 'areas', view: props.cellAreas, type: 'f32' as const, access: 'read' as const}]
          : [])
      ],
      body: `let value = hazard[hazardOffset + index]; let weight = population[populationOffset + index]${props.weightMeaning === 'density' ? ' * areas[areasOffset + index]' : ''};
let validWeight = isFiniteValue(weight) && weight >= 0.0 ${props.weightMeaning === 'density' ? '&& areas[areasOffset + index] > 0.0' : ''};
let known = validWeight && isFiniteValue(value);
let classified = known && isFiniteValue(threshold[thresholdOffset]);
let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
totals[totalsOffset + index] = select(missing, weight, validWeight);
observed[observedOffset + index] = select(missing, weight, known);
exposed[exposedOffset + index] = select(missing, select(0.0, weight, value >= threshold[thresholdOffset]), classified);`
    })
  );
  const totalPopulation = getOrCreateView(
    graph,
    `${id}-total-population`,
    'float32',
    props.zoneCount
  );
  const observedPopulation = getOrCreateView(
    graph,
    `${id}-observed-population`,
    'float32',
    props.zoneCount
  );
  const exposedPopulation = getOrCreateView(
    graph,
    `${id}-exposed-population`,
    'float32',
    props.zoneCount
  );
  const coverage = getOrCreateView(graph, `${id}-coverage`, 'float32', props.zoneCount);
  const exposedShare = getOrCreateView(graph, `${id}-share`, 'float32', props.zoneCount);
  const thresholdInvalidCount = getOrCreateView(graph, `${id}-threshold-invalid`, 'uint32', 1);
  const stages = [];
  for (const [name, values, sums] of [
    ['total', totals, totalPopulation],
    ['observed', observed, observedPopulation],
    ['exposed', exposed, exposedPopulation]
  ] as const) {
    const overflow = getOrCreateView(graph, `${id}-${name}-overflow`, 'uint32', 1);
    builder.add(
      new GPURasterZonalStatistics({
        id: `${id}-${name}`,
        grid: props.grid,
        zones: props.zones,
        values: {id: name, format: 'float32', storage: {kind: 'buffer', values}},
        zoneCapacity: props.zoneCount,
        output: {sums, status: {overflow}}
      })
    );
    stages.push({stage: `${name}-population`, status: {overflow}});
  }
  builder.addGlue(
    'shares',
    createWGSLKernelNode(graph, {
      id: `${id}-shares`,
      operation: id,
      invocationCount: props.zoneCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'total', view: totalPopulation, type: 'f32', access: 'read'},
        {name: 'observed', view: observedPopulation, type: 'f32', access: 'read'},
        {name: 'exposed', view: exposedPopulation, type: 'f32', access: 'read_write'},
        {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
        {name: 'thresholdInvalid', view: thresholdInvalidCount, type: 'u32', access: 'read_write'},
        {name: 'coverage', view: coverage, type: 'f32', access: 'read_write'},
        {name: 'share', view: exposedShare, type: 'f32', access: 'read_write'}
      ],
      body: `let totalWeight = total[totalOffset + index]; let observedWeight = observed[observedOffset + index]; let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
coverage[coverageOffset + index] = select(missing, observedWeight / totalWeight, totalWeight > 0.0);
let thresholdValid = isFiniteValue(threshold[thresholdOffset]);
share[shareOffset + index] = select(missing, exposed[exposedOffset + index] / observedWeight, observedWeight > 0.0 && thresholdValid);
if (!thresholdValid) { exposed[exposedOffset + index] = missing; }
if (index == 0u) { thresholdInvalid[thresholdInvalidOffset] = select(1u, 0u, thresholdValid); }`
    })
  );
  return {
    contributors: builder.contributors,
    outputs: {
      totalPopulation,
      observedPopulation,
      exposedPopulation,
      coverage,
      exposedShare,
      thresholdInvalidCount
    },
    intermediates: {hazard, population, zones: props.zones, grid: props.grid},
    status: {
      stages: [
        ...stages,
        {stage: 'exposure-threshold', status: {invalidCount: thresholdInvalidCount}}
      ]
    }
  };
}

export type GPURegionalRasterChangeRecipeProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Calibrated aligned slices, with missing samples encoded as NaN. Single-band only. */
  change: Omit<GPUChangeDetectionProps, 'id' | 'output' | 'cellCount' | 'bandCount'>;
  zones: GraphDataView<'uint32'>;
  zoneCount: number;
  cellAreas: GraphDataView<'float32'>;
  /** Absolute difference threshold, one-row live view. No inferential regional claim. */
  threshold: GraphDataView<'float32'>;
};
/** Two-period raster differences over common support, with independently reported coverage. */
export function addRegionalRasterChangeRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURegionalRasterChangeRecipeProps
) {
  const id = props.id ?? 'regional-raster-change-recipe';
  const builder = new RecipeBuilder(graph);
  const count = props.grid.width * props.grid.height;
  if (
    props.zones.length !== count ||
    props.cellAreas.length !== count ||
    props.threshold.length !== 1
  )
    throw new Error(`${id}: incompatible grid`);
  const difference = getOrCreateView(graph, `${id}-difference`, 'float32', count);
  builder.add(
    new GPUChangeDetection({
      ...props.change,
      id: `${id}-change`,
      cellCount: count,
      bandCount: 1,
      output: {difference}
    })
  );
  const validAreas = getOrCreateView(graph, `${id}-valid-areas`, 'float32', count);
  const changedAreas = getOrCreateView(graph, `${id}-changed-areas`, 'float32', count);
  const totalAreas = getOrCreateView(graph, `${id}-total-areas`, 'float32', count);
  builder.addGlue(
    'area',
    createWGSLKernelNode(graph, {
      id: `${id}-area`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'difference', view: difference, type: 'f32', access: 'read'},
        {name: 'areas', view: props.cellAreas, type: 'f32', access: 'read'},
        {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
        {name: 'valid', view: validAreas, type: 'f32', access: 'read_write'},
        {name: 'changed', view: changedAreas, type: 'f32', access: 'read_write'},
        {name: 'total', view: totalAreas, type: 'f32', access: 'read_write'}
      ],
      body: `let area = areas[areasOffset + index]; let delta = difference[differenceOffset + index];
let validArea = isFiniteValue(area) && area > 0.0; let comparisonValid = validArea && isFiniteValue(delta);
let missing = bitcast<f32>(0x7fc00000u | (index & 0u)); let cutoff = threshold[thresholdOffset];
total[totalOffset + index] = select(missing, area, validArea);
valid[validOffset + index] = select(missing, area, comparisonValid);
changed[changedOffset + index] = select(missing, select(0.0, area, abs(delta) >= cutoff), comparisonValid && isFiniteValue(cutoff) && cutoff >= 0.0);`
    })
  );
  const validCounts = getOrCreateView(graph, `${id}-valid-counts`, 'uint32', props.zoneCount);
  const meanChanges = getOrCreateView(graph, `${id}-means`, 'float32', props.zoneCount);
  const areaOutputs = {
    totalArea: getOrCreateView(graph, `${id}-total-area`, 'float32', props.zoneCount),
    validArea: getOrCreateView(graph, `${id}-valid-area`, 'float32', props.zoneCount),
    changedArea: getOrCreateView(graph, `${id}-changed-area`, 'float32', props.zoneCount)
  };
  const stages = [];
  for (const [name, values, sums] of [
    ['total', totalAreas, areaOutputs.totalArea],
    ['valid', validAreas, areaOutputs.validArea],
    ['changed', changedAreas, areaOutputs.changedArea]
  ] as const) {
    const overflow = getOrCreateView(graph, `${id}-${name}-overflow`, 'uint32', 1);
    builder.add(
      new GPURasterZonalStatistics({
        id: `${id}-${name}`,
        grid: props.grid,
        zones: props.zones,
        values: {id: name, format: 'float32', storage: {kind: 'buffer', values}},
        zoneCapacity: props.zoneCount,
        output: {sums, status: {overflow}}
      })
    );
    stages.push({stage: `${name}-area`, status: {overflow}});
  }
  const summaryOverflow = getOrCreateView(graph, `${id}-summary-overflow`, 'uint32', 1);
  builder.add(
    new GPURasterZonalStatistics({
      id: `${id}-summary`,
      grid: props.grid,
      zones: props.zones,
      values: {id: 'difference', format: 'float32', storage: {kind: 'buffer', values: difference}},
      zoneCapacity: props.zoneCount,
      output: {valueCounts: validCounts, means: meanChanges, status: {overflow: summaryOverflow}}
    })
  );
  stages.push({stage: 'change-summary', status: {overflow: summaryOverflow}});
  const coverage = getOrCreateView(graph, `${id}-coverage`, 'float32', props.zoneCount);
  const thresholdInvalidCount = getOrCreateView(graph, `${id}-threshold-invalid`, 'uint32', 1);
  builder.addGlue(
    'coverage',
    createWGSLKernelNode(graph, {
      id: `${id}-coverage`,
      operation: id,
      invocationCount: props.zoneCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'total', view: areaOutputs.totalArea, type: 'f32', access: 'read'},
        {name: 'valid', view: areaOutputs.validArea, type: 'f32', access: 'read'},
        {name: 'changed', view: areaOutputs.changedArea, type: 'f32', access: 'read_write'},
        {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
        {name: 'thresholdInvalid', view: thresholdInvalidCount, type: 'u32', access: 'read_write'},
        {name: 'coverage', view: coverage, type: 'f32', access: 'read_write'}
      ],
      body: `let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
coverage[coverageOffset + index] = select(missing, valid[validOffset + index] / total[totalOffset + index], total[totalOffset + index] > 0.0);
let thresholdValid = isFiniteValue(threshold[thresholdOffset]) && threshold[thresholdOffset] >= 0.0;
if (!thresholdValid) { changed[changedOffset + index] = missing; }
if (index == 0u) { thresholdInvalid[thresholdInvalidOffset] = select(1u, 0u, thresholdValid); }`
    })
  );
  return {
    contributors: builder.contributors,
    outputs: {
      difference,
      validCounts,
      meanChanges,
      ...areaOutputs,
      coverage,
      thresholdInvalidCount
    },
    intermediates: {grid: props.grid, zones: props.zones},
    status: {
      stages: [
        ...stages,
        {stage: 'change-threshold', status: {invalidCount: thresholdInvalidCount}}
      ]
    }
  };
}
