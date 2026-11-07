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
import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPULocalPermutationTest,
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS,
  type KeplerMultipleTestingMethod
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {addKeplerLinearKernel, KEPLER_ANALYSIS_ROW_TILE_SIZE} from './analysis-scaling';
import {
  getKeplerPointWeightsMaximumNeighbors,
  type KeplerPointWeightsBinding
} from './point-weights';
import {copyPointStatus} from './point-hotspots';
import {getKeplerPointWeightsViews} from './point-weights-chunked';

export type KeplerPointBivariateLocalMoranSettings = Readonly<{
  islands: 'exclude' | 'keep';
  multipleTesting: KeplerMultipleTestingMethod;
  significanceLevel: number;
  /** Conditional permutations per row, 1 to 9999. */
  permutationCount: number;
  seed: number;
}>;

export type KeplerPointBivariateLocalMoranOutputs = {
  /** Bivariate Local Moran I per row (esda `Moran_Local_BV` scaling), NaN where not tested. */
  localI?: GraphDataView<'float32'>;
  pValues?: GraphDataView<'float32'>;
  adjustedPValues?: GraphDataView<'float32'>;
  /** 1 HH, 2 LH, 3 LL, 4 HL in (x_i, lag of y), 0 where not significant. */
  quadrants?: GraphDataView<'uint32'>;
  significant?: GraphDataView<'uint32'>;
  status?: {
    /** Incomplete weights or an active degree above the permutation scratch limit. */
    inferenceOverflow?: GraphDataView<'uint32'>;
    weightsOverflow?: GraphDataView<'uint32'>;
  };
};

export type KeplerPointBivariateLocalMoranProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerPointBivariateLocalMoranSettings;
  /** Focus variable x: the value at location i. */
  values: GraphDataView<'float32'>;
  /** Neighbor variable y: the weighted lag is taken over y (GeoDa bivariate Local Moran). */
  secondValues: GraphDataView<'float32'>;
  /** Declared population intersected with source numeric validity. */
  mask: GraphDataView<'uint32'>;
  pointWeights: KeplerPointWeightsBinding;
  outputs?: KeplerPointBivariateLocalMoranOutputs;
};

/**
 * Bivariate Local Moran's I (Anselin 2019; GeoDa and esda `Moran_Local_BV`) on original point
 * rows: `I_i = (n - 1) z^x_i sum_j w_ij z^y_j / sqrt(SS_x SS_y)` with conditional permutation
 * inference that draws the y values of the neighbor slots from the other rows. Rows are included
 * only when x, y, the population mask and the coordinates are all valid. Binary weights are used
 * as built by the shared point topology; there is no analytic variance, so permutation inference
 * is always used, with any multiple-testing correction applied over tested rows. Permutations are
 * keyed by the global row index and tiled across rows, so results do not depend on row tiling or
 * on the number of rows, up to 2^24 - 1.
 */
export function addKeplerPointBivariateLocalMoran<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointBivariateLocalMoranProps
) {
  const id = props.id ?? 'kepler-point-bivariate-local-moran';
  const settings = Object.freeze({...props.settings});
  const {values, secondValues, mask, pointWeights} = props;
  const rowCount = values.length;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    pointWeights.source.datasetId !== props.source.datasetId ||
    pointWeights.source.revision !== props.source.revision ||
    (props.population === 'selected'
      ? !Number.isSafeInteger(props.selectionRevision) || props.selectionRevision! < 1
      : props.population !== 'all' || props.selectionRevision !== undefined) ||
    rowCount < 3 ||
    rowCount > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS ||
    secondValues.length !== rowCount ||
    mask.length !== rowCount ||
    pointWeights.coordinateMask.length !== rowCount ||
    !Number.isSafeInteger(settings.permutationCount) ||
    settings.permutationCount < 1 ||
    settings.permutationCount > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS ||
    !Number.isSafeInteger(settings.seed) ||
    settings.seed < 0 ||
    settings.seed > 2 ** 32 - 1 ||
    !(settings.significanceLevel > 0 && settings.significanceLevel < 1)
  )
    throw new Error('Bivariate Local Moran requires current source, rows and permutation bounds');
  for (const view of [values, secondValues])
    validatePackedView(view, ['float32'], 'Bivariate Local Moran values');
  for (const view of [mask, pointWeights.coordinateMask])
    validatePackedView(view, ['uint32'], 'Bivariate Local Moran mask');
  const provided = props.outputs ?? {};
  const suppliedViews = [
    provided.localI,
    provided.pValues,
    provided.adjustedPValues,
    provided.quadrants,
    provided.significant,
    provided.status?.inferenceOverflow,
    provided.status?.weightsOverflow
  ].filter(view => view !== undefined);
  for (let first = 0; first < suppliedViews.length; first++)
    for (let second = first + 1; second < suppliedViews.length; second++)
      if (doGraphDataViewsOverlap(suppliedViews[first], suppliedViews[second]))
        throw new Error('Bivariate Local Moran output ports must not overlap');
  for (const view of [
    values,
    secondValues,
    mask,
    pointWeights.coordinateMask,
    ...getKeplerPointWeightsViews(pointWeights.weights),
    ...suppliedViews
  ])
    if (view.buffer.graph !== graph)
      throw new Error('Bivariate Local Moran views must belong to the consumer graph');
  const localI =
    provided.localI ?? createTransientView(graph, `${id}-local-i`, 'float32', rowCount);
  const pValues =
    provided.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rowCount);
  const adjustedPValues =
    provided.adjustedPValues ??
    createTransientView(graph, `${id}-adjusted-p-values`, 'float32', rowCount);
  const quadrants =
    provided.quadrants ?? createTransientView(graph, `${id}-quadrants`, 'uint32', rowCount);
  const significant =
    provided.significant ?? createTransientView(graph, `${id}-significant`, 'uint32', rowCount);
  const inferenceOverflow =
    provided.status?.inferenceOverflow ??
    createTransientView(graph, `${id}-inference-overflow`, 'uint32', 1);
  const weightsOverflow =
    provided.status?.weightsOverflow ??
    createTransientView(graph, `${id}-weights-overflow`, 'uint32', 1);
  const ungatedQuadrants = createTransientView(
    graph,
    `${id}-ungated-quadrants`,
    'uint32',
    rowCount
  );
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-permutation-parameters`,
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues(
      {
        seed: settings.seed,
        permutationCount: settings.permutationCount,
        significanceLevel: settings.significanceLevel
      },
      {maximumPermutations: settings.permutationCount}
    )
  });
  try {
    addKeplerLinearKernel(
      graph,
      `${id}-effective-mask`,
      rowCount,
      [
        {name: 'firstValues', view: values},
        {name: 'secondValues', view: secondValues},
        {name: 'populationMask', view: mask},
        {name: 'coordinateMask', view: pointWeights.coordinateMask},
        {name: 'effectiveMask', view: effectiveMask, write: true}
      ],
      `let included = isFiniteValue(firstValues[firstValuesOffset + index]) && isFiniteValue(secondValues[secondValuesOffset + index]) && populationMask[populationMaskOffset + index] != 0u && coordinateMask[coordinateMaskOffset + index] != 0u;
effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, included);`
    );
    const contributor = new GPULocalPermutationTest({
      id: `${id}-permutation`,
      values,
      secondValues,
      weights: {...pointWeights.weights, status: pointWeights.status},
      mask: effectiveMask,
      statistic: 'localBivariateMoran',
      alternative: 'two-sided',
      islands: settings.islands,
      multipleTesting: settings.multipleTesting,
      maximumNeighbors: getKeplerPointWeightsMaximumNeighbors(pointWeights.settings),
      extendedNeighborScratch: getKeplerPointWeightsMaximumNeighbors(pointWeights.settings) > 64,
      rowTileSize: KEPLER_ANALYSIS_ROW_TILE_SIZE,
      maximumPermutations: settings.permutationCount,
      parameters: parameters.importToGraph(graph),
      output: {
        exceedances: createTransientView(graph, `${id}-exceedances`, 'uint32', rowCount),
        pseudoPValues: pValues,
        observed: localI,
        adjustedPValues,
        significant,
        quadrants: ungatedQuadrants,
        status: {overflow: inferenceOverflow}
      }
    });
    graph.add(contributor);
    addKeplerLinearKernel(
      graph,
      `${id}-quadrant-gate`,
      rowCount,
      [
        {name: 'ungated', view: ungatedQuadrants},
        {name: 'significantRows', view: significant},
        {name: 'gated', view: quadrants, write: true}
      ],
      `gated[gatedOffset + index] = select(0u, ungated[ungatedOffset + index], significantRows[significantRowsOffset + index] != 0u);`
    );
    copyPointStatus(
      graph,
      `${id}-copy-weights-overflow`,
      pointWeights.status.overflow,
      weightsOverflow
    );
    return {
      operation: 'bivariate-local-moran' as const,
      source: Object.freeze({...props.source}),
      population: props.population,
      ...(props.selectionRevision === undefined
        ? {}
        : {selectionRevision: props.selectionRevision}),
      settings,
      numericalPolicy: 'float32' as const,
      weightTransform: 'binary' as const,
      inferenceMethod: 'permutation-conditional-two-sided' as const,
      inference: contributor.getInferenceMetadata({
        seed: settings.seed,
        permutationCount: settings.permutationCount,
        significanceLevel: settings.significanceLevel
      }),
      rowCount,
      effectiveMask,
      outputs: {localI, pValues, adjustedPValues, quadrants, significant},
      status: {inferenceOverflow, weightsOverflow},
      parameterByteLength: parameters.buffer.byteLength,
      pointWeights,
      destroyParameters() {
        parameters.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    throw error;
  }
}
