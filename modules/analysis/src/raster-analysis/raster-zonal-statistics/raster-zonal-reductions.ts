// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getCellWeightWGSL} from './raster-zonal-cell-weight';
import type {
  GPURasterZonalStatisticsProps,
  GPURasterZonalStatisticsSumOrder
} from './gpu-raster-zonal-statistics';

const GPU_RASTER_ZONAL_STATISTICS_NO_ZONE = 0xffffffff;

/**
 * Whether the encoding runs the sorted per-zone sums: `sumOrder` is `'sorted'` and a column needs a
 * sum (moments, geographic weights, sums or means).
 */
export function hasSortedSums(
  sumOrder: GPURasterZonalStatisticsSumOrder,
  props: GPURasterZonalStatisticsProps
): boolean {
  const {output} = props;
  const momentPath =
    props.cellSizeMode === 'geographic' ||
    Boolean(
      output?.variances ||
        output?.populationVariances ||
        output?.standardDeviations ||
        output?.weightSums
    );
  return sumOrder === 'sorted' && (momentPath || Boolean(output?.sums || output?.means));
}

/** Common route selection for constructor alias validation and graph declaration. @internal */
export function getRasterZonalReductionPlan(
  props: GPURasterZonalStatisticsProps,
  sumOrder: GPURasterZonalStatisticsSumOrder,
  maximumWorkgroupStorageSize: number
) {
  const {output} = props;
  const weighted = props.cellSizeMode === 'geographic';
  const needsVariance = Boolean(
    output.variances || output.populationVariances || output.standardDeviations
  );
  const momentPath = weighted || needsVariance || Boolean(output.weightSums);
  const needsMeans = Boolean(output.means) || needsVariance;
  const needsWeightSums = Boolean(output.weightSums) || (weighted && needsMeans);
  const sortedSums = hasSortedSums(sumOrder, props);
  const needsValueCounts =
    Boolean(output.valueCounts) ||
    (sortedSums && Boolean(output.means)) ||
    (momentPath && needsMeans);
  const columnCount = [
    output.cellCounts,
    needsValueCounts,
    output.minimums,
    output.maximums
  ].filter(Boolean).length;
  // WebGPU accounts for each statically allocated workgroup variable separately, rounded to
  // 16 bytes. Three arrays at capacity 1365 take 3*5472=16416, not 3*5460=16380 bytes.
  const fusedWorkgroupBytes = columnCount * Math.ceil((props.zoneCapacity * 4) / 16) * 16;
  return {
    weighted,
    needsVariance,
    momentPath,
    needsMeans,
    needsWeightSums,
    sortedSums,
    needsValueCounts,
    fused: columnCount > 0 && fusedWorkgroupBytes <= maximumWorkgroupStorageSize
  };
}

/**
 * Sums, weight sums, means and variances from the per-cell contributions of the mask kernel:
 * one sorted segmented reduction (or `GPUGroupAggregation` sums with `sumOrder: 'atomic'`), a
 * mean kernel, then, for variances, a deviation pass, a second reduction and a finalize kernel.
 */
export function getRasterZonalMomentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURasterZonalStatisticsProps,
  plan: {
    weighted: boolean;
    needsMeans: boolean;
    needsVariance: boolean;
    needsWeightSums: boolean;
    sortedSums: boolean;
    canonicalValues?: GraphDataView<'float32'>;
    canonicalValidity?: GraphDataView<'uint32'>;
    valueMask?: GraphDataView<'uint32'>;
    sortRows?: GraphDataView<'uint32'>;
    contributions: GraphDataView<'float32'>;
    weightContributions?: GraphDataView<'float32'>;
    valueCounts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {zones, zoneCapacity, output} = props;
  const {width, height} = props.grid;
  const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
  const {weighted, sortedSums} = plan;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sumsView =
    output.sums ??
    (plan.needsMeans
      ? createTransientView(graph, `${id}-sorted-sums`, 'float32', zoneCapacity)
      : undefined);
  const weightSumsView =
    output.weightSums ??
    (plan.needsWeightSums
      ? createTransientView(graph, `${id}-weight-sums`, 'float32', zoneCapacity)
      : undefined);
  const reductions = [
    ...(sumsView ? [{name: 'sums', contributions: plan.contributions, output: sumsView}] : []),
    ...(weightSumsView
      ? [{name: 'weights', contributions: plan.weightContributions!, output: weightSumsView}]
      : [])
  ];
  nodes.push(...getSumNodes(graph, props, `${id}-sorted`, `${id}-`, sortedSums, plan, reductions));

  const meansView =
    output.means ??
    (plan.needsVariance
      ? createTransientView(graph, `${id}-mean-values`, 'float32', zoneCapacity)
      : undefined);
  if (meansView) {
    const bindings: WGSLKernelBinding[] = [
      {name: 'sums', view: sumsView!, type: 'f32', access: 'read'},
      {name: 'valueCounts', view: plan.valueCounts!, type: 'u32', access: 'read'}
    ];
    if (weighted) {
      bindings.push({name: 'weightSums', view: weightSumsView!, type: 'f32', access: 'read'});
    }
    bindings.push({name: 'means', view: meansView, type: 'u32', access: 'read_write'});
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation: 'GPURasterZonalStatistics',
        variant: weighted ? 'weighted-mean' : 'sorted-mean',
        bindings,
        invocationCount: zoneCapacity,
        body: weighted
          ? `let count = valueCounts[valueCountsOffset + index];
  let weightSum = weightSums[weightSumsOffset + index];
  var mean = 0x7fc00000u;
  // A zone whose valid cells have no area has no weighted mean.
  if (count > 0u && weightSum > 0.0) {
    mean = bitcast<u32>(sums[sumsOffset + index] / weightSum);
  }
  means[meansOffset + index] = mean;`
          : `let count = valueCounts[valueCountsOffset + index];
  var mean = 0x7fc00000u;
  if (count > 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(count));
  }
  means[meansOffset + index] = mean;`
      })
    );
  }
  if (!plan.needsVariance) {
    return nodes;
  }

  // Deviation passes around per-zone means: optional `w (v - m)`, `w (v - m)^2` and `w^2` per
  // valid cell, reduced per zone. A float32 sum of values near 1e7 over many cells leaves the
  // first mean units off, so the first pass refines the mean with the sum of the (small)
  // deviations around it, and the second pass sums the corrected two-pass terms around the
  // refined mean (GEO-105 in GPUZonalStatistics).
  const cellCount = width * height;
  const createDeviationNode = (
    nodeId: string,
    means: GraphDataView<'float32'>,
    terms: {
      squared?: GraphDataView<'float32'>;
      linear?: GraphDataView<'float32'>;
      weightSquares?: GraphDataView<'float32'>;
    }
  ) => {
    const bindings: WGSLKernelBinding[] = [
      {name: 'zones', view: zones, type: 'u32', access: 'read'},
      {name: 'values', view: plan.canonicalValues!, type: 'f32', access: 'read'},
      {name: 'validity', view: plan.canonicalValidity!, type: 'u32', access: 'read'},
      {name: 'means', view: means, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['squaredDeviations', terms.squared],
      ['linearDeviations', terms.linear],
      ['weightSquares', terms.weightSquares]
    ] as const) {
      if (view) bindings.push({name, view, type: 'f32', access: 'read_write'});
    }
    if (weighted) {
      bindings.push({name: 'settings', view: props.parameters!, type: 'f32', access: 'read'});
    }
    return createWGSLKernelNode<Parameters>(graph, {
      id: nodeId,
      operation: 'GPURasterZonalStatistics',
      variant: [
        weighted ? 'weighted-deviations' : 'deviations',
        terms.squared ? 'squared' : '',
        terms.linear ? 'linear' : '',
        terms.weightSquares ? 'weight-squares' : ''
      ]
        .filter(Boolean)
        .join('-'),
      bindings,
      invocationCount: cellCount,
      declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${ignoredZone}u;
${IS_FINITE_VALUE_WGSL}
${weighted ? getCellWeightWGSL(width, height, props.parameterLayoutVersion ?? 2) : ''}`,
      body: `let zone = zones[zonesOffset + index];
  let value = values[valuesOffset + index];
  let isValid = zone < CAPACITY && zone != IGNORED && validity[validityOffset + index] != 0u &&
    isFiniteValue(value);
  let weight = ${weighted ? 'getCellWeight(index)' : '1.0'};
  var squaredDeviation = 0.0;
  var linearDeviation = 0.0;
  var weightSquare = 0.0;
  if (isValid && weight > 0.0) {
    let difference = value - means[meansOffset + zone];
    squaredDeviation = weight * difference * difference;
    linearDeviation = weight * difference;
    weightSquare = weight * weight;
  }
  ${terms.squared ? 'squaredDeviations[squaredDeviationsOffset + index] = squaredDeviation;' : ''}
  ${terms.linear ? 'linearDeviations[linearDeviationsOffset + index] = linearDeviation;' : ''}
  ${terms.weightSquares ? 'weightSquares[weightSquaresOffset + index] = weightSquare;' : ''}`
    });
  };

  // First pass: refine the float32 mean, mean + sum w (v - mean) / V1.
  const firstDeviations = createTransientView(
    graph,
    `${id}-first-deviations`,
    'float32',
    cellCount
  );
  const firstDeviationSums = createTransientView(
    graph,
    `${id}-first-deviation-sums`,
    'float32',
    zoneCapacity
  );
  const refinedMeans = createTransientView(graph, `${id}-refined-means`, 'float32', zoneCapacity);
  nodes.push(
    createDeviationNode(`${id}-first-deviations`, meansView!, {linear: firstDeviations}),
    ...getSumNodes(
      graph,
      props,
      `${id}-first-deviation-sorted`,
      `${id}-first-deviation-`,
      sortedSums,
      plan,
      [{name: 'linear-deviations', contributions: firstDeviations, output: firstDeviationSums}]
    ),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-refined-means`,
      operation: 'GPURasterZonalStatistics',
      variant: weighted ? 'weighted-refined-means' : 'refined-means',
      bindings: [
        {name: 'means', view: meansView!, type: 'f32', access: 'read'},
        {name: 'linearDeviationSums', view: firstDeviationSums, type: 'f32', access: 'read'},
        weighted
          ? {name: 'weightSums', view: weightSumsView!, type: 'f32', access: 'read'}
          : {name: 'valueCounts', view: plan.valueCounts!, type: 'u32', access: 'read'},
        {name: 'refinedMeans', view: refinedMeans, type: 'f32', access: 'read_write'}
      ],
      invocationCount: zoneCapacity,
      declarations: IS_FINITE_VALUE_WGSL,
      // Means that are not finite (empty zones), or whose correction is not, are kept.
      body: `let mean = means[meansOffset + index];
  let total = ${weighted ? 'weightSums[weightSumsOffset + index]' : 'f32(valueCounts[valueCountsOffset + index])'};
  var refined = mean;
  if (isFiniteValue(mean) && total > 0.0) {
    let candidate = mean + linearDeviationSums[linearDeviationSumsOffset + index] / total;
    refined = select(mean, candidate, isFiniteValue(candidate));
  }
  refinedMeans[refinedMeansOffset + index] = refined;`
    })
  );

  // Second pass around the refined mean: squared and linear deviations, and w^2 when weighted.
  const deviations = createTransientView(graph, `${id}-deviations`, 'float32', cellCount);
  const linearDeviations = createTransientView(
    graph,
    `${id}-linear-deviations`,
    'float32',
    cellCount
  );
  const weightSquares = weighted
    ? createTransientView(graph, `${id}-weight-squares`, 'float32', cellCount)
    : undefined;
  nodes.push(
    createDeviationNode(`${id}-deviations`, refinedMeans, {
      squared: deviations,
      linear: linearDeviations,
      weightSquares
    })
  );
  const squaredDeviationSums = createTransientView(
    graph,
    `${id}-squared-deviation-sums`,
    'float32',
    zoneCapacity
  );
  const linearDeviationSums = createTransientView(
    graph,
    `${id}-linear-deviation-sums`,
    'float32',
    zoneCapacity
  );
  const weightSquareSums = weightSquares
    ? createTransientView(graph, `${id}-weight-square-sums`, 'float32', zoneCapacity)
    : undefined;
  nodes.push(
    ...getSumNodes(graph, props, `${id}-variance-sorted`, `${id}-variance-`, sortedSums, plan, [
      {name: 'deviations', contributions: deviations, output: squaredDeviationSums},
      {name: 'linear-deviations', contributions: linearDeviations, output: linearDeviationSums},
      ...(weightSquares && weightSquareSums
        ? [{name: 'weight-squares', contributions: weightSquares, output: weightSquareSums}]
        : [])
    ])
  );

  const finalizeBindings: WGSLKernelBinding[] = [
    {name: 'valueCounts', view: plan.valueCounts!, type: 'u32', access: 'read'},
    {name: 'squaredDeviationSums', view: squaredDeviationSums, type: 'f32', access: 'read'},
    {name: 'linearDeviationSums', view: linearDeviationSums, type: 'f32', access: 'read'}
  ];
  if (weightSquareSums) {
    finalizeBindings.push(
      {name: 'weightSums', view: weightSumsView!, type: 'f32', access: 'read'},
      {name: 'weightSquareSums', view: weightSquareSums, type: 'f32', access: 'read'}
    );
  }
  for (const name of ['variances', 'populationVariances', 'standardDeviations'] as const) {
    if (output[name]) {
      finalizeBindings.push({name, view: output[name], type: 'u32', access: 'read_write'});
    }
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-variances`,
      operation: 'GPURasterZonalStatistics',
      variant: weighted ? 'weighted-corrected-variances' : 'corrected-variances',
      bindings: finalizeBindings,
      invocationCount: zoneCapacity,
      // Reliability weights, as GPUZonalStatistics: sum w (v - m)^2 / (V1 - V2 / V1), which is
      // sum (v - m)^2 / (n - 1) for unit weights. The corrected two-pass numerator
      // max(0, ssd - (lds / V1) * lds) removes the remaining error of the refined mean.
      body: `let valueCount = valueCounts[valueCountsOffset + index];
  var variance = 0x7fc00000u;
  var populationVariance = 0x7fc00000u;
  var deviation = 0x7fc00000u;
  var denominator = 0.0;
  ${
    weightSquareSums
      ? `let weightSum = weightSums[weightSumsOffset + index];
  if (weightSum != 0.0) {
    denominator = weightSum - weightSquareSums[weightSquareSumsOffset + index] / weightSum;
  }`
      : 'denominator = f32(valueCount) - 1.0;'
  }
  let total = ${weightSquareSums ? 'weightSum' : 'f32(valueCount)'};
  if (valueCount >= 1u && total > 0.0) {
    let linearDeviationSum = linearDeviationSums[linearDeviationSumsOffset + index];
    let squaredDeviationSum = squaredDeviationSums[squaredDeviationSumsOffset + index];
    let numerator = max(0.0, squaredDeviationSum - (linearDeviationSum / total) * linearDeviationSum);
    populationVariance = bitcast<u32>(numerator / total);
    if (valueCount >= 2u && denominator > 0.0) {
      let result = numerator / denominator;
      variance = bitcast<u32>(result);
      deviation = bitcast<u32>(sqrt(result));
    }
  }
  ${output.populationVariances ? 'populationVariances[populationVariancesOffset + index] = populationVariance;' : ''}
  ${output.variances ? 'variances[variancesOffset + index] = variance;' : ''}
  ${output.standardDeviations ? 'standardDeviations[standardDeviationsOffset + index] = deviation;' : ''}`
    })
  );
  return nodes;
}

/** Per-zone sums of contributions: sorted segmented sums, or one atomic aggregation each. */
function getSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterZonalStatisticsProps,
  sortedId: string,
  atomicPrefix: string,
  sortedSums: boolean,
  plan: {sortRows?: GraphDataView<'uint32'>; valueMask?: GraphDataView<'uint32'>},
  reductions: {
    name: string;
    contributions: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }[]
): GPUCommandNode<Parameters>[] {
  if (reductions.length === 0) {
    return [];
  }
  if (sortedSums) {
    return getSortedSegmentSumNodes<Parameters>(graph, {
      id: sortedId,
      operation: 'GPURasterZonalStatistics',
      segmentCount: props.zoneCapacity,
      segmentKeys: plan.sortRows!,
      reductions
    });
  }
  return reductions.flatMap(reduction =>
    new GPUGroupAggregation({
      id: `${atomicPrefix}${reduction.name}`,
      keys: props.zones,
      mask: plan.valueMask!,
      values: reduction.contributions,
      output: reduction.output,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
}
