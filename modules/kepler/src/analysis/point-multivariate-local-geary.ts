// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUMultivariateLocalGearyPermutationTest} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {addKeplerLinearKernel, assertKeplerStorageBindingFits} from './analysis-scaling';
import type {KeplerPointWeightsBinding} from './point-weights';
import {getKeplerPointWeightsViews} from './point-weights-chunked';

/**
 * Bounds of the tiled GPU contributor. Up to 256 rows and 4096 permutations the exact baseline
 * runs (the oracle); beyond that the tiled strategy gives identical results per tile-size-
 * independent streams. Device storage binding limits can still lower the row capacity.
 */
export const MAXIMUM_KEPLER_MULTIVARIATE_GEARY_ROWS = 2 ** 24 - 1;
export const MAXIMUM_KEPLER_MULTIVARIATE_GEARY_PERMUTATIONS = 9999;
/** Row count up to which the contributor still uses its exact single-dispatch baseline. */
export const EXACT_KEPLER_MULTIVARIATE_GEARY_ROWS = 256;
export const MINIMUM_KEPLER_MULTIVARIATE_GEARY_VARIABLES = 2;
export const MAXIMUM_KEPLER_MULTIVARIATE_GEARY_VARIABLES = 8;

export type KeplerPointMultivariateLocalGearySettings = Readonly<{
  /** Conditional permutations per row, 1 to 9999. */
  permutationCount: number;
  seed: number;
  significanceLevel: number;
  /** Bonferroni family is every included row. */
  correction: 'none' | 'bonferroni';
  tail: 'equal-tail' | 'esda-min-tail';
}>;

export type KeplerPointMultivariateLocalGearyOutputs = {
  /** Mean multivariate local Geary per row, NaN where not tested. */
  statistic?: GraphDataView<'float32'>;
  pValues?: GraphDataView<'float32'>;
  /** 0 not significant or untested, 1 significant similarity, 2 significant dissimilarity. */
  classes?: GraphDataView<'uint32'>;
};

export type KeplerPointMultivariateLocalGearyProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerPointMultivariateLocalGearySettings;
  /** Two to eight packed float32 variables over the original source rows. */
  variables: readonly GraphDataView<'float32'>[];
  /** Declared population intersected with source numeric validity. */
  mask: GraphDataView<'uint32'>;
  pointWeights: KeplerPointWeightsBinding;
  outputs?: KeplerPointMultivariateLocalGearyOutputs;
};

/**
 * GeoDa-style multivariate local Geary on original point rows: population-standardized variables,
 * binary active weights normalized per row, and conditional randomization that samples whole
 * neighbor vectors. Rows are tested only when all variables, the population mask and the
 * coordinates are valid. The contributor runs its exact baseline up to 256 rows and 4096
 * permutations and a row- and permutation-tiled strategy beyond; rows with more than 64 active
 * neighbors are reported untested (class 0), and storage binding limits are checked up front.
 */
export function addKeplerPointMultivariateLocalGeary<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointMultivariateLocalGearyProps
) {
  const id = props.id ?? 'kepler-point-multivariate-local-geary';
  const settings = Object.freeze({...props.settings});
  const {variables, mask, pointWeights} = props;
  const rowCount = variables[0]?.length ?? 0;
  const variableCount = variables.length;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    pointWeights.source.datasetId !== props.source.datasetId ||
    pointWeights.source.revision !== props.source.revision ||
    (props.population === 'selected'
      ? !Number.isSafeInteger(props.selectionRevision) || props.selectionRevision! < 1
      : props.population !== 'all' || props.selectionRevision !== undefined) ||
    variableCount < MINIMUM_KEPLER_MULTIVARIATE_GEARY_VARIABLES ||
    variableCount > MAXIMUM_KEPLER_MULTIVARIATE_GEARY_VARIABLES ||
    rowCount < 2 ||
    rowCount > MAXIMUM_KEPLER_MULTIVARIATE_GEARY_ROWS ||
    variables.some(view => view.length !== rowCount) ||
    mask.length !== rowCount ||
    pointWeights.coordinateMask.length !== rowCount ||
    !Number.isSafeInteger(settings.permutationCount) ||
    settings.permutationCount < 1 ||
    settings.permutationCount > MAXIMUM_KEPLER_MULTIVARIATE_GEARY_PERMUTATIONS ||
    !Number.isSafeInteger(settings.seed) ||
    settings.seed < 0 ||
    settings.seed > 2 ** 32 - 1 ||
    !(settings.significanceLevel > 0 && settings.significanceLevel < 1)
  )
    throw new Error('Multivariate Local Geary requires current source, bounded rows and variables');
  // Interleaved values and standardized copies are rows x variables; accumulators are 4 x rows.
  assertKeplerStorageBindingFits(
    graph.device.limits,
    rowCount * variableCount,
    4,
    'Multivariate Local Geary variables'
  );
  assertKeplerStorageBindingFits(graph.device.limits, 4 * rowCount, 4, 'Multivariate Local Geary');
  for (const view of variables)
    validatePackedView(view, ['float32'], 'Multivariate Local Geary variables');
  for (const view of [mask, pointWeights.coordinateMask])
    validatePackedView(view, ['uint32'], 'Multivariate Local Geary mask');
  const provided = props.outputs ?? {};
  const suppliedViews = [provided.statistic, provided.pValues, provided.classes].filter(
    view => view !== undefined
  );
  for (let first = 0; first < suppliedViews.length; first++)
    for (let second = first + 1; second < suppliedViews.length; second++)
      if (doGraphDataViewsOverlap(suppliedViews[first], suppliedViews[second]))
        throw new Error('Multivariate Local Geary output ports must not overlap');
  for (const view of [
    ...variables,
    mask,
    pointWeights.coordinateMask,
    ...getKeplerPointWeightsViews(pointWeights.weights),
    ...suppliedViews
  ])
    if (view.buffer.graph !== graph)
      throw new Error('Multivariate Local Geary views must belong to the consumer graph');
  const statistic =
    provided.statistic ?? createTransientView(graph, `${id}-statistic`, 'float32', rowCount);
  const pValues =
    provided.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rowCount);
  const classes =
    provided.classes ?? createTransientView(graph, `${id}-classes`, 'uint32', rowCount);
  const ungatedClasses = createTransientView(graph, `${id}-ungated-classes`, 'uint32', rowCount);
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const interleaved = createTransientView(
    graph,
    `${id}-values`,
    'float32',
    rowCount * variableCount
  );
  addKeplerLinearKernel(
    graph,
    `${id}-effective-mask`,
    rowCount,
    [
      {name: 'populationMask', view: mask},
      {name: 'coordinateMask', view: pointWeights.coordinateMask},
      {name: 'effectiveMask', view: effectiveMask, write: true}
    ],
    `effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, populationMask[populationMaskOffset + index] != 0u && coordinateMask[coordinateMaskOffset + index] != 0u);`
  );
  // Row-major interleave; the contributor treats a nonfinite value in any column as incomplete.
  addKeplerLinearKernel(
    graph,
    `${id}-interleave`,
    rowCount,
    [
      ...variables.map((view, column) => ({name: `column${column}`, view})),
      {name: 'interleaved', view: interleaved, write: true}
    ],
    variables
      .map(
        (_, column) =>
          `interleaved[interleavedOffset + index * ${variableCount}u + ${column}u] = column${column}[column${column}Offset + index];`
      )
      .join('\n')
  );
  const contributor = new GPUMultivariateLocalGearyPermutationTest({
    id: `${id}-geary`,
    weights: {...pointWeights.weights, status: pointWeights.status},
    values: interleaved,
    variableCount,
    mask: effectiveMask,
    permutationCount: settings.permutationCount,
    seed: [settings.seed, 0],
    significanceLevel: settings.significanceLevel,
    correction: settings.correction,
    tail: settings.tail,
    output: {statistic, adjustedPValues: pValues, classes: ungatedClasses}
  });
  graph.add(contributor);
  // 0xffffffff marks untested rows; report them as class 0 so palettes stay in range.
  addKeplerLinearKernel(
    graph,
    `${id}-class-gate`,
    rowCount,
    [
      {name: 'ungated', view: ungatedClasses},
      {name: 'gated', view: classes, write: true}
    ],
    `let value = ungated[ungatedOffset + index];
gated[gatedOffset + index] = select(0u, value, value < 3u);`
  );
  return {
    operation: 'multivariate-local-geary' as const,
    source: Object.freeze({...props.source}),
    population: props.population,
    ...(props.selectionRevision === undefined ? {} : {selectionRevision: props.selectionRevision}),
    settings,
    variableCount,
    rowCount,
    effectiveMask,
    outputs: {statistic, pValues, classes},
    pointWeights,
    destroyParameters() {}
  };
}
