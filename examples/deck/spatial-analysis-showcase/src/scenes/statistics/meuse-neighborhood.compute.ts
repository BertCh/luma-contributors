// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addLocalVariabilityRecipe,
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUGammaStatistic,
  GPULocalGeary,
  GPULocalPermutationTest,
  GPUQuantileLISA,
  GPULocalSpatialHeteroscedasticity,
  GPULOSHResidualPermutationTest,
  GPUMultivariateLocalGearyPermutationTest,
  GPUNeighborSearch,
  GPUSpatialPearson
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';

const NEIGHBOR_COUNT = 8;
const MAXIMUM_PERMUTATIONS = 999;
const QUANTILE_PERMUTATIONS = 1;
const DISPLAY_FIELDS = [
  'local-geary',
  'local-geary-p',
  'multivariate-geary-p',
  'losh',
  'losh-p',
  'lee-l',
  'lee-l-p',
  'quantile-membership',
  'quantile-join-count'
] as const;

type QuantileCountOption = '4' | '5' | '10';
type QuantileBandOption = 'lowest' | 'middle' | 'highest';

export type MeuseNeighborhoodOptions = {
  response: 'zinc' | 'lead';
  display: (typeof DISPLAY_FIELDS)[number];
  gammaOperation: 'cross-product' | 'squared-difference' | 'absolute-difference';
  loshPower: 'variance' | 'absolute-deviation';
  quantileCount: QuantileCountOption;
  quantileBand: QuantileBandOption;
};

const DISPLAY_LABELS: Record<MeuseNeighborhoodOptions['display'], string> = {
  'local-geary': 'Local Geary statistic',
  'local-geary-p': 'Local Geary BY-adjusted pseudo-p',
  'multivariate-geary-p': 'Multivariate Local Geary equal-tail pseudo-p',
  losh: 'LOSH local dispersion',
  'losh-p': 'LOSH fixed-residual BY-adjusted pseudo-p',
  'lee-l': 'Local Lee L statistic',
  'lee-l-p': 'Local Lee L BY-adjusted pseudo-p',
  'quantile-membership': 'Selected type-5 quantile membership',
  'quantile-join-count': 'Selected-quantile local join count'
};

function getDisplayIndex(display: MeuseNeighborhoodOptions['display']): number {
  return DISPLAY_FIELDS.indexOf(display);
}

function getSelectedQuantile(count: QuantileCountOption, band: QuantileBandOption): number {
  const quantileCount = Number(count);
  switch (band) {
    case 'lowest':
      return 1;
    case 'middle':
      return Math.ceil(quantileCount / 2);
    case 'highest':
      return quantileCount;
  }
}

/** GPU local association and variability tests for the measured Meuse metal field. */
export async function createMeuseNeighborhood(
  ctx: SceneContext<MeuseNeighborhoodOptions>
): Promise<SceneInstance<MeuseNeighborhoodOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const rowCount = dataset.column('sampleDist').length;
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const mapPositions = dataset.projectColumn('samplePosition', origin);
  const sampleSource = dataset.column('samplePositionRD');
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < rowCount; row++) {
    minimumX = Math.min(minimumX, sampleSource[row * 2]);
    maximumX = Math.max(maximumX, sampleSource[row * 2]);
    minimumY = Math.min(minimumY, sampleSource[row * 2 + 1]);
    maximumY = Math.max(maximumY, sampleSource[row * 2 + 1]);
  }
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const positions = Float32Array.from(
    sampleSource,
    (value, index) => value - (index % 2 ? centerY : centerX)
  );
  const responseColumns = {
    zinc: dataset.column('logZinc'),
    copper: dataset.column('logCopper'),
    lead: dataset.column('logLead')
  };
  const copperValues = Float32Array.from(responseColumns.copper);
  const multivariateValues = new Float32Array(rowCount * 3);
  for (let row = 0; row < rowCount; row++) {
    multivariateValues.set(
      [responseColumns.zinc[row], responseColumns.copper[row], responseColumns.lead[row]],
      row * 3
    );
  }
  const riverCoordinates =
    (dataset.geojson?.features[0]?.geometry as {coordinates: [number, number][]} | undefined)
      ?.coordinates ?? [];
  const riverSegments = new Float32Array(Math.max(0, riverCoordinates.length - 1) * 4);
  for (let index = 0; index + 1 < riverCoordinates.length; index++) {
    const first = projection.project(...riverCoordinates[index]);
    const second = projection.project(...riverCoordinates[index + 1]);
    riverSegments.set([first[0], first[1], second[0], second[1]], index * 4);
  }

  const resources = new SpatialAnalysisResources(device, 'meuse-neighborhood');
  const positionsBuffer = resources.createBuffer('sample-positions', positions);
  const valuesBuffer = resources.createBuffer(
    'selected-log-metal',
    Float32Array.from(responseColumns[ctx.options.response])
  );
  const copperBuffer = resources.createBuffer('log-copper', copperValues);
  const multivariateBuffer = resources.createBuffer('log-zinc-copper-lead', multivariateValues);
  const searchParametersBuffer = resources.createBuffer(
    'neighbor-search-parameters',
    getGPUNeighborSearchParameterValues({
      bounds: [
        Math.min(...positions.filter((_, index) => index % 2 === 0)),
        Math.min(...positions.filter((_, index) => index % 2 === 1)),
        Math.max(...positions.filter((_, index) => index % 2 === 0)),
        Math.max(...positions.filter((_, index) => index % 2 === 1))
      ],
      weightKind: 'binary',
      rowStandardize: true
    })
  );
  const autocorrelationParametersBuffer = resources.createBuffer(
    'spatial-autocorrelation-parameters',
    getGPUSpatialAutocorrelationParameterValues()
  );
  const permutationParametersBuffer = resources.createBuffer(
    'permutation-parameters',
    getGPUPermutationParameterValues(
      {seed: 31, permutationCount: MAXIMUM_PERMUTATIONS},
      {maximumPermutations: MAXIMUM_PERMUTATIONS}
    )
  );
  const quantileParameterBuffer = resources.createBuffer(
    'quantile-lisa-settings',
    Uint32Array.of(
      Number(ctx.options.quantileCount),
      getSelectedQuantile(ctx.options.quantileCount, ctx.options.quantileBand)
    )
  );
  const quantilePermutationParametersBuffer = resources.createBuffer(
    'quantile-lisa-permutation-settings',
    getGPUPermutationParameterValues(
      {seed: 31, permutationCount: QUANTILE_PERMUTATIONS},
      {maximumPermutations: QUANTILE_PERMUTATIONS}
    )
  );
  const mapControlBuffer = resources.createBuffer(
    'map-control',
    Uint32Array.of(
      getDisplayIndex(ctx.options.display),
      ctx.options.loshPower === 'variance' ? 2 : 1
    )
  );
  const mapValuesBuffer = resources.createBuffer('mapped-statistic', rowCount * 4);
  const neighborOffsetsBuffer = resources.createBuffer('knn-offsets', (rowCount + 1) * 4);
  const neighborIdsBuffer = resources.createBuffer('knn-neighbors', rowCount * NEIGHBOR_COUNT * 4);
  const neighborWeightsBuffer = resources.createBuffer(
    'knn-weights',
    rowCount * NEIGHBOR_COUNT * 4
  );
  const neighborDistancesBuffer = resources.createBuffer(
    'knn-distances',
    rowCount * NEIGHBOR_COUNT * 4
  );
  const neighborOverflowBuffer = resources.createBuffer('knn-overflow', 4);
  const neighborRequiredBuffer = resources.createBuffer('knn-required-count', 4);
  const neighborInvalidBuffer = resources.createBuffer('knn-invalid-count', 4);
  const localGearyBuffer = resources.createBuffer('local-geary', rowCount * 4);
  const localGearyGlobalBuffer = resources.createBuffer('local-geary-global', 5 * 4);
  const localGearyPBuffer = resources.createBuffer('local-geary-pseudo-p', rowCount * 4);
  const localGearyAdjustedBuffer = resources.createBuffer('local-geary-adjusted-p', rowCount * 4);
  const localGearyClassesBuffer = resources.createBuffer('local-geary-classes', rowCount * 4);
  const localGearyExceedancesBuffer = resources.createBuffer(
    'local-geary-exceedances',
    rowCount * 4
  );
  const localGearyOverflowBuffer = resources.createBuffer('local-geary-overflow', 4);
  const quantileMembershipBuffer = resources.createBuffer('quantile-membership', rowCount * 4);
  const quantileBoundariesBuffer = resources.createBuffer('quantile-boundaries', 2 * 4);
  const quantileValidCountBuffer = resources.createBuffer('quantile-valid-count', 4);
  const quantileClassesBuffer = resources.createBuffer('quantile-classes', rowCount * 4);
  const quantileInvalidBuffer = resources.createBuffer('quantile-invalid', 4);
  const quantileExceedancesBuffer = resources.createBuffer('quantile-exceedances', rowCount * 4);
  const quantilePseudoPBuffer = resources.createBuffer('quantile-pseudo-p', rowCount * 4);
  const quantileObservedBuffer = resources.createBuffer(
    'quantile-observed-join-count',
    rowCount * 4
  );
  const quantileSignificantBuffer = resources.createBuffer('quantile-significant', rowCount * 4);
  const quantileIslandCountBuffer = resources.createBuffer('quantile-island-count', 4);
  const quantileOverflowBuffer = resources.createBuffer('quantile-overflow', 4);
  const multivariateGearyBuffer = resources.createBuffer('multivariate-local-geary', rowCount * 4);
  const multivariateGearyPBuffer = resources.createBuffer(
    'multivariate-local-geary-pseudo-p',
    rowCount * 4
  );
  const multivariateGearyClassesBuffer = resources.createBuffer(
    'multivariate-local-geary-classes',
    rowCount * 4
  );
  const loshBuffer = resources.createBuffer('losh', rowCount * 4);
  const loshStatusBuffer = resources.createBuffer('losh-status', rowCount * 4);
  const loshInferenceBuffer = resources.createBuffer('losh-inference-statistic', rowCount * 4);
  const loshInferenceStatusBuffer = resources.createBuffer('losh-inference-status', rowCount * 4);
  const loshAbsoluteBuffer = resources.createBuffer('losh-absolute-deviation', rowCount * 4);
  const loshAbsoluteStatusBuffer = resources.createBuffer(
    'losh-absolute-deviation-status',
    rowCount * 4
  );
  const loshAbsoluteInferenceBuffer = resources.createBuffer(
    'losh-absolute-deviation-inference-statistic',
    rowCount * 4
  );
  const loshAbsoluteInferenceStatusBuffer = resources.createBuffer(
    'losh-absolute-deviation-inference-status',
    rowCount * 4
  );
  const analyticMoranBuffer = resources.createBuffer('analytic-local-moran', rowCount * 4);
  const analyticMoranZBuffer = resources.createBuffer('analytic-local-moran-z', rowCount * 4);
  const analyticMoranPBuffer = resources.createBuffer('analytic-local-moran-p', rowCount * 4);
  const analyticMoranAdjustedBuffer = resources.createBuffer(
    'analytic-local-moran-adjusted-p',
    rowCount * 4
  );
  const analyticMoranQuadrantsBuffer = resources.createBuffer(
    'analytic-local-moran-quadrants',
    rowCount * 4
  );
  const analyticMoranStatusBuffer = resources.createBuffer(
    'analytic-local-moran-status',
    rowCount * 4
  );
  const loshPermutationBuffer = resources.createBuffer(
    'losh-fixed-residual-pseudo-p',
    rowCount * 4
  );
  const loshAdjustedBuffer = resources.createBuffer('losh-fixed-residual-adjusted-p', rowCount * 4);
  const loshExceedancesBuffer = resources.createBuffer(
    'losh-fixed-residual-exceedances',
    rowCount * 4
  );
  const loshPermutationOverflowBuffer = resources.createBuffer('losh-fixed-residual-overflow', 4);
  const loshInferenceSignificantBuffer = resources.createBuffer(
    'losh-inference-significant',
    rowCount * 4
  );
  const loshAbsolutePermutationBuffer = resources.createBuffer(
    'losh-absolute-deviation-pseudo-p',
    rowCount * 4
  );
  const loshAbsoluteAdjustedBuffer = resources.createBuffer(
    'losh-absolute-deviation-adjusted-p',
    rowCount * 4
  );
  const loshAbsoluteExceedancesBuffer = resources.createBuffer(
    'losh-absolute-deviation-exceedances',
    rowCount * 4
  );
  const loshAbsolutePermutationOverflowBuffer = resources.createBuffer(
    'losh-absolute-deviation-overflow',
    4
  );
  const loshAbsoluteInferenceSignificantBuffer = resources.createBuffer(
    'losh-absolute-deviation-significant',
    rowCount * 4
  );
  const leeLBuffer = resources.createBuffer('local-lee-l', rowCount * 4);
  const leeLPBuffer = resources.createBuffer('local-lee-l-pseudo-p', rowCount * 4);
  const leeLAdjustedBuffer = resources.createBuffer('local-lee-l-adjusted-p', rowCount * 4);
  const leeLExceedancesBuffer = resources.createBuffer('local-lee-l-exceedances', rowCount * 4);
  const leeLOverflowBuffer = resources.createBuffer('local-lee-l-overflow', 4);
  const spatialPearsonBuffer = resources.createBuffer('spatial-pearson', 4);
  const gammaCrossProductBuffer = resources.createBuffer('gamma-cross-product', 4);
  const gammaSquaredDifferenceBuffer = resources.createBuffer('gamma-squared-difference', 4);
  const gammaAbsoluteDifferenceBuffer = resources.createBuffer('gamma-absolute-difference', 4);
  const generationInput = resources.createBuffer('summary-generation-input', Uint32Array.of(0));
  const generationOutput = resources.createBuffer('summary-generation-output', 4);
  const mapPositionsBuffer = resources.createBuffer(
    'sample-map-positions',
    mapPositions,
    Buffer.VERTEX
  );
  const riverBuffer = resources.createBuffer('river-outline', riverSegments);

  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-neighborhood'});
  const positionsView = importGraphBuffer(
    graph,
    'sample-positions',
    positionsBuffer,
    'float32x2',
    rowCount
  );
  const valuesView = importGraphBuffer(
    graph,
    'selected-log-metal',
    valuesBuffer,
    'float32',
    rowCount
  );
  const copperView = importGraphBuffer(graph, 'log-copper', copperBuffer, 'float32', rowCount);
  const multivariateView = importGraphBuffer(
    graph,
    'log-zinc-copper-lead',
    multivariateBuffer,
    'float32',
    rowCount * 3
  );
  const searchParametersView = importGraphBuffer(
    graph,
    'neighbor-search-parameters',
    searchParametersBuffer,
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const autocorrelationParametersView = importGraphBuffer(
    graph,
    'spatial-autocorrelation-parameters',
    autocorrelationParametersBuffer,
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const permutationParametersView = importGraphBuffer(
    graph,
    'permutation-parameters',
    permutationParametersBuffer,
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  const quantileParameterView = importGraphBuffer(
    graph,
    'quantile-lisa-settings',
    quantileParameterBuffer,
    'uint32',
    2
  );
  const quantilePermutationParametersView = importGraphBuffer(
    graph,
    'quantile-lisa-permutation-settings',
    quantilePermutationParametersBuffer,
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  const offsetsView = importGraphBuffer(
    graph,
    'knn-offsets',
    neighborOffsetsBuffer,
    'uint32',
    rowCount + 1
  );
  const neighborsView = importGraphBuffer(
    graph,
    'knn-neighbors',
    neighborIdsBuffer,
    'uint32',
    rowCount * NEIGHBOR_COUNT
  );
  const weightsView = importGraphBuffer(
    graph,
    'knn-weights',
    neighborWeightsBuffer,
    'float32',
    rowCount * NEIGHBOR_COUNT
  );
  const distancesView = importGraphBuffer(
    graph,
    'knn-distances',
    neighborDistancesBuffer,
    'float32',
    rowCount * NEIGHBOR_COUNT
  );
  const neighborOverflowView = importGraphBuffer(
    graph,
    'knn-overflow',
    neighborOverflowBuffer,
    'uint32',
    1
  );
  const neighborRequiredView = importGraphBuffer(
    graph,
    'knn-required-count',
    neighborRequiredBuffer,
    'uint32',
    1
  );
  const neighborInvalidView = importGraphBuffer(
    graph,
    'knn-invalid-count',
    neighborInvalidBuffer,
    'uint32',
    1
  );
  const weightsSearch = new GPUNeighborSearch({
    id: 'meuse-eight-nearest-row-weights',
    mode: 'knn',
    k: NEIGHBOR_COUNT,
    positions: positionsView,
    parameters: searchParametersView,
    gridSize: [64, 64],
    weightTransform: 'row',
    rowStandardize: true,
    output: {
      offsets: offsetsView,
      neighbors: neighborsView,
      weights: weightsView,
      distances: distancesView,
      status: {
        overflow: neighborOverflowView,
        requiredCount: neighborRequiredView,
        invalidCount: neighborInvalidView
      }
    }
  });
  const weights = weightsSearch.outputWeights;
  graph.add(weightsSearch);
  const quantileMembershipView = importGraphBuffer(
    graph,
    'quantile-membership',
    quantileMembershipBuffer,
    'uint32',
    rowCount
  );
  const quantileBoundariesView = importGraphBuffer(
    graph,
    'quantile-boundaries',
    quantileBoundariesBuffer,
    'float32',
    2
  );
  const quantileValidCountView = importGraphBuffer(
    graph,
    'quantile-valid-count',
    quantileValidCountBuffer,
    'uint32',
    1
  );
  const quantileClassesView = importGraphBuffer(
    graph,
    'quantile-classes',
    quantileClassesBuffer,
    'uint32',
    rowCount
  );
  const quantileInvalidView = importGraphBuffer(
    graph,
    'quantile-invalid',
    quantileInvalidBuffer,
    'uint32',
    1
  );
  const quantileExceedancesView = importGraphBuffer(
    graph,
    'quantile-exceedances',
    quantileExceedancesBuffer,
    'uint32',
    rowCount
  );
  const quantilePseudoPView = importGraphBuffer(
    graph,
    'quantile-pseudo-p',
    quantilePseudoPBuffer,
    'float32',
    rowCount
  );
  const quantileObservedView = importGraphBuffer(
    graph,
    'quantile-observed-join-count',
    quantileObservedBuffer,
    'float32',
    rowCount
  );
  const quantileSignificantView = importGraphBuffer(
    graph,
    'quantile-significant',
    quantileSignificantBuffer,
    'uint32',
    rowCount
  );
  const quantileIslandCountView = importGraphBuffer(
    graph,
    'quantile-island-count',
    quantileIslandCountBuffer,
    'uint32',
    1
  );
  const quantileOverflowView = importGraphBuffer(
    graph,
    'quantile-overflow',
    quantileOverflowBuffer,
    'uint32',
    1
  );
  graph.add(
    new GPUQuantileLISA({
      id: 'meuse-measured-metal-quantile-lisa',
      values: valuesView,
      weights,
      quantileParameters: quantileParameterView,
      parameters: quantilePermutationParametersView,
      maximumPermutations: QUANTILE_PERMUTATIONS,
      maximumNeighbors: NEIGHBOR_COUNT,
      output: {
        membership: quantileMembershipView,
        boundaries: quantileBoundariesView,
        validCount: quantileValidCountView,
        classes: quantileClassesView,
        invalidQuantile: quantileInvalidView,
        exceedances: quantileExceedancesView,
        pseudoPValues: quantilePseudoPView,
        observed: quantileObservedView,
        significant: quantileSignificantView,
        islandCount: quantileIslandCountView,
        status: {overflow: quantileOverflowView}
      }
    })
  );
  const localGearyView = importGraphBuffer(
    graph,
    'local-geary',
    localGearyBuffer,
    'float32',
    rowCount
  );
  const localGearyGlobalView = importGraphBuffer(
    graph,
    'local-geary-global',
    localGearyGlobalBuffer,
    'float32',
    5
  );
  const localGearyPView = importGraphBuffer(
    graph,
    'local-geary-pseudo-p',
    localGearyPBuffer,
    'float32',
    rowCount
  );
  const localGearyAdjustedView = importGraphBuffer(
    graph,
    'local-geary-adjusted-p',
    localGearyAdjustedBuffer,
    'float32',
    rowCount
  );
  const localGearyClassesView = importGraphBuffer(
    graph,
    'local-geary-classes',
    localGearyClassesBuffer,
    'uint32',
    rowCount
  );
  const localGearyExceedancesView = importGraphBuffer(
    graph,
    'local-geary-exceedances',
    localGearyExceedancesBuffer,
    'uint32',
    rowCount
  );
  const localGearyOverflowView = importGraphBuffer(
    graph,
    'local-geary-overflow',
    localGearyOverflowBuffer,
    'uint32',
    1
  );
  graph.add(
    new GPULocalGeary({
      id: 'meuse-local-geary-observed',
      weights,
      values: valuesView,
      parameters: autocorrelationParametersView,
      output: {statistic: localGearyView, globalStatistics: localGearyGlobalView}
    })
  );
  graph.add(
    new GPULocalPermutationTest({
      id: 'meuse-local-geary-conditional-test',
      weights,
      values: valuesView,
      statistic: 'localGeary',
      alternative: 'lesser',
      multipleTesting: 'benjamini-yekutieli',
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      maximumNeighbors: NEIGHBOR_COUNT,
      parameters: permutationParametersView,
      output: {
        exceedances: localGearyExceedancesView,
        pseudoPValues: localGearyPView,
        adjustedPValues: localGearyAdjustedView,
        classes: localGearyClassesView,
        status: {overflow: localGearyOverflowView}
      }
    })
  );

  const multivariateGearyView = importGraphBuffer(
    graph,
    'multivariate-local-geary',
    multivariateGearyBuffer,
    'float32',
    rowCount
  );
  const multivariateGearyPView = importGraphBuffer(
    graph,
    'multivariate-local-geary-pseudo-p',
    multivariateGearyPBuffer,
    'float32',
    rowCount
  );
  const multivariateGearyClassesView = importGraphBuffer(
    graph,
    'multivariate-local-geary-classes',
    multivariateGearyClassesBuffer,
    'uint32',
    rowCount
  );
  graph.add(
    new GPUMultivariateLocalGearyPermutationTest({
      id: 'meuse-joint-metal-local-geary',
      weights,
      values: multivariateView,
      variableCount: 3,
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      seed: [31, 0],
      tail: 'equal-tail',
      correction: 'none',
      output: {
        statistic: multivariateGearyView,
        pValues: multivariateGearyPView,
        classes: multivariateGearyClassesView
      }
    })
  );

  const recipe = addLocalVariabilityRecipe(graph, {
    id: 'meuse-local-dispersion-and-moran',
    weights,
    values: valuesView,
    parameters: autocorrelationParametersView,
    multipleTesting: 'benjamini-yekutieli',
    outputs: {
      losh: importGraphBuffer(graph, 'losh', loshBuffer, 'float32', rowCount),
      loshStatus: importGraphBuffer(graph, 'losh-status', loshStatusBuffer, 'uint32', rowCount),
      localI: importGraphBuffer(
        graph,
        'analytic-local-moran',
        analyticMoranBuffer,
        'float32',
        rowCount
      ),
      zScores: importGraphBuffer(
        graph,
        'analytic-local-moran-z',
        analyticMoranZBuffer,
        'float32',
        rowCount
      ),
      pValues: importGraphBuffer(
        graph,
        'analytic-local-moran-p',
        analyticMoranPBuffer,
        'float32',
        rowCount
      ),
      adjustedPValues: importGraphBuffer(
        graph,
        'analytic-local-moran-adjusted-p',
        analyticMoranAdjustedBuffer,
        'float32',
        rowCount
      ),
      quadrants: importGraphBuffer(
        graph,
        'analytic-local-moran-quadrants',
        analyticMoranQuadrantsBuffer,
        'uint32',
        rowCount
      ),
      moranStatus: importGraphBuffer(
        graph,
        'analytic-local-moran-status',
        analyticMoranStatusBuffer,
        'uint32',
        rowCount
      )
    }
  });
  const loshPView = importGraphBuffer(
    graph,
    'losh-fixed-residual-pseudo-p',
    loshPermutationBuffer,
    'float32',
    rowCount
  );
  const loshAdjustedView = importGraphBuffer(
    graph,
    'losh-fixed-residual-adjusted-p',
    loshAdjustedBuffer,
    'float32',
    rowCount
  );
  const loshExceedancesView = importGraphBuffer(
    graph,
    'losh-fixed-residual-exceedances',
    loshExceedancesBuffer,
    'uint32',
    rowCount
  );
  const loshPermutationOverflowView = importGraphBuffer(
    graph,
    'losh-fixed-residual-overflow',
    loshPermutationOverflowBuffer,
    'uint32',
    1
  );
  const loshAbsoluteView = importGraphBuffer(
    graph,
    'losh-absolute-deviation',
    loshAbsoluteBuffer,
    'float32',
    rowCount
  );
  const loshAbsoluteStatusView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-status',
    loshAbsoluteStatusBuffer,
    'uint32',
    rowCount
  );
  const loshAbsoluteInferenceView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-inference-statistic',
    loshAbsoluteInferenceBuffer,
    'float32',
    rowCount
  );
  const loshAbsoluteInferenceStatusView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-inference-status',
    loshAbsoluteInferenceStatusBuffer,
    'uint32',
    rowCount
  );
  const loshAbsolutePermutationView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-pseudo-p',
    loshAbsolutePermutationBuffer,
    'float32',
    rowCount
  );
  const loshAbsoluteAdjustedView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-adjusted-p',
    loshAbsoluteAdjustedBuffer,
    'float32',
    rowCount
  );
  const loshAbsoluteExceedancesView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-exceedances',
    loshAbsoluteExceedancesBuffer,
    'uint32',
    rowCount
  );
  const loshAbsoluteOverflowView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-overflow',
    loshAbsolutePermutationOverflowBuffer,
    'uint32',
    1
  );
  const loshAbsoluteSignificantView = importGraphBuffer(
    graph,
    'losh-absolute-deviation-significant',
    loshAbsoluteInferenceSignificantBuffer,
    'uint32',
    rowCount
  );
  graph.add(
    new GPULocalSpatialHeteroscedasticity({
      id: 'meuse-local-absolute-deviation',
      weights,
      values: valuesView,
      residualPower: 1,
      output: {
        statistic: loshAbsoluteView,
        status: loshAbsoluteStatusView
      }
    })
  );
  graph.add(
    new GPULOSHResidualPermutationTest({
      id: 'meuse-fixed-loSh-residual-label-test',
      weights,
      values: valuesView,
      parameters: permutationParametersView,
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      maximumNeighbors: NEIGHBOR_COUNT,
      alternative: 'greater',
      multipleTesting: 'benjamini-yekutieli',
      output: {
        statistic: importGraphBuffer(
          graph,
          'losh-inference-statistic',
          loshInferenceBuffer,
          'float32',
          rowCount
        ),
        status: importGraphBuffer(
          graph,
          'losh-inference-status',
          loshInferenceStatusBuffer,
          'uint32',
          rowCount
        ),
        inference: {
          pseudoPValues: loshPView,
          adjustedPValues: loshAdjustedView,
          exceedances: loshExceedancesView,
          significant: importGraphBuffer(
            graph,
            'losh-inference-significant',
            loshInferenceSignificantBuffer,
            'uint32',
            rowCount
          ),
          status: {overflow: loshPermutationOverflowView}
        }
      }
    })
  );
  graph.add(
    new GPULOSHResidualPermutationTest({
      id: 'meuse-absolute-deviation-residual-label-test',
      weights,
      values: valuesView,
      parameters: permutationParametersView,
      residualPower: 1,
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      maximumNeighbors: NEIGHBOR_COUNT,
      alternative: 'greater',
      multipleTesting: 'benjamini-yekutieli',
      output: {
        statistic: loshAbsoluteInferenceView,
        status: loshAbsoluteInferenceStatusView,
        inference: {
          pseudoPValues: loshAbsolutePermutationView,
          adjustedPValues: loshAbsoluteAdjustedView,
          exceedances: loshAbsoluteExceedancesView,
          significant: loshAbsoluteSignificantView,
          status: {overflow: loshAbsoluteOverflowView}
        }
      }
    })
  );

  const leeLView = importGraphBuffer(graph, 'local-lee-l', leeLBuffer, 'float32', rowCount);
  const leeLPView = importGraphBuffer(
    graph,
    'local-lee-l-pseudo-p',
    leeLPBuffer,
    'float32',
    rowCount
  );
  const leeLAdjustedView = importGraphBuffer(
    graph,
    'local-lee-l-adjusted-p',
    leeLAdjustedBuffer,
    'float32',
    rowCount
  );
  graph.add(
    new GPULocalPermutationTest({
      id: 'meuse-local-lee-l-zinc-copper',
      weights,
      values: valuesView,
      secondValues: copperView,
      statistic: 'localLeeL',
      alternative: 'two-sided',
      multipleTesting: 'benjamini-yekutieli',
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      maximumNeighbors: NEIGHBOR_COUNT,
      parameters: permutationParametersView,
      output: {
        exceedances: importGraphBuffer(
          graph,
          'local-lee-l-exceedances',
          leeLExceedancesBuffer,
          'uint32',
          rowCount
        ),
        pseudoPValues: leeLPView,
        adjustedPValues: leeLAdjustedView,
        observed: leeLView,
        status: {
          overflow: importGraphBuffer(
            graph,
            'local-lee-l-overflow',
            leeLOverflowBuffer,
            'uint32',
            1
          )
        }
      }
    })
  );
  const spatialPearsonView = importGraphBuffer(
    graph,
    'spatial-pearson',
    spatialPearsonBuffer,
    'float32',
    1
  );
  graph.add(
    new GPUSpatialPearson({
      id: 'meuse-zinc-copper-edge-pearson',
      weights,
      values: valuesView,
      secondValues: copperView,
      output: {statistic: spatialPearsonView}
    })
  );
  const gammaViews = [
    {
      operation: 'cross-product' as const,
      view: importGraphBuffer(graph, 'gamma-cross-product', gammaCrossProductBuffer, 'float32', 1)
    },
    {
      operation: 'squared-difference' as const,
      view: importGraphBuffer(
        graph,
        'gamma-squared-difference',
        gammaSquaredDifferenceBuffer,
        'float32',
        1
      )
    },
    {
      operation: 'absolute-difference' as const,
      view: importGraphBuffer(
        graph,
        'gamma-absolute-difference',
        gammaAbsoluteDifferenceBuffer,
        'float32',
        1
      )
    }
  ];
  for (const {operation, view} of gammaViews) {
    graph.add(
      new GPUGammaStatistic({
        id: `meuse-${operation}-gamma`,
        weights,
        values: valuesView,
        operation,
        output: {statistic: view}
      })
    );
  }

  const mapControlView = importGraphBuffer(graph, 'map-control', mapControlBuffer, 'uint32', 2);
  const mapValuesView = importGraphBuffer(
    graph,
    'mapped-statistic',
    mapValuesBuffer,
    'float32',
    rowCount
  );
  const generationInputView = importGraphBuffer(
    graph,
    'summary-generation-input',
    generationInput,
    'uint32',
    1
  );
  const generationOutputView = importGraphBuffer(
    graph,
    'summary-generation-output',
    generationOutput,
    'uint32',
    1
  );
  // Keep each selector at or below eight storage bindings. Each mode-specific pass writes only
  // its own modes; the Geary pass initializes all other rows to NaN.
  addKernelPass(graph, {
    id: 'meuse-neighborhood-map-selection',
    invocationCount: rowCount,
    bindings: [
      {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
      {name: 'localGeary', view: localGearyView, type: 'f32', access: 'read'},
      {name: 'localGearyP', view: localGearyAdjustedView, type: 'f32', access: 'read'},
      {name: 'multivariateP', view: multivariateGearyPView, type: 'f32', access: 'read'},
      {name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}
    ],
    body: `let mode = control[controlOffset]; var value = bitcast<f32>(0x7fc00000u | (index & 0u)); if (mode == 0u) { value = localGeary[localGearyOffset + index]; } else if (mode == 1u) { value = localGearyP[localGearyPOffset + index]; } else if (mode == 2u) { value = multivariateP[multivariatePOffset + index]; } mapped[mappedOffset + index] = value;`
  });
  addKernelPass(graph, {
    id: 'meuse-neighborhood-losh-map-selection',
    invocationCount: rowCount,
    bindings: [
      {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
      {name: 'losh', view: recipe.outputs.losh, type: 'f32', access: 'read'},
      {name: 'loshP', view: loshAdjustedView, type: 'f32', access: 'read'},
      {name: 'loshAbsolute', view: loshAbsoluteView, type: 'f32', access: 'read'},
      {name: 'loshAbsoluteP', view: loshAbsoluteAdjustedView, type: 'f32', access: 'read'},
      {name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}
    ],
    body: 'let mode = control[controlOffset]; let loshPower = control[controlOffset + 1u]; if (mode == 3u) { mapped[mappedOffset + index] = select(losh[loshOffset + index], loshAbsolute[loshAbsoluteOffset + index], loshPower == 1u); } else if (mode == 4u) { mapped[mappedOffset + index] = select(loshP[loshPOffset + index], loshAbsoluteP[loshAbsolutePOffset + index], loshPower == 1u); }'
  });
  addKernelPass(graph, {
    id: 'meuse-neighborhood-lee-map-selection',
    invocationCount: rowCount,
    bindings: [
      {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
      {name: 'leeL', view: leeLView, type: 'f32', access: 'read'},
      {name: 'leeLP', view: leeLAdjustedView, type: 'f32', access: 'read'},
      {name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}
    ],
    body: 'let mode = control[controlOffset]; if (mode == 5u) { mapped[mappedOffset + index] = leeL[leeLOffset + index]; } else if (mode == 6u) { mapped[mappedOffset + index] = leeLP[leeLPOffset + index]; }'
  });
  addKernelPass(graph, {
    id: 'meuse-neighborhood-quantile-map-selection',
    invocationCount: rowCount,
    bindings: [
      {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
      {name: 'membership', view: quantileMembershipView, type: 'u32', access: 'read'},
      {name: 'joinCounts', view: quantileObservedView, type: 'f32', access: 'read'},
      {name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}
    ],
    body: 'let mode = control[controlOffset]; if (mode == 7u) { mapped[mappedOffset + index] = f32(membership[membershipOffset + index]); } else if (mode == 8u) { mapped[mappedOffset + index] = joinCounts[joinCountsOffset + index]; }'
  });
  addKernelPass(graph, {
    id: 'meuse-neighborhood-generation-snapshot',
    invocationCount: 1,
    bindings: [
      {name: 'generation', view: generationInputView, type: 'u32', access: 'read'},
      {name: 'snapshot', view: generationOutputView, type: 'u32', access: 'read_write'}
    ],
    body: 'snapshot[snapshotOffset] = generation[generationOffset];'
  });

  const compiled = resources.track(await graph.compileAsync());
  let generation = 0;
  let graphDirty = true;
  let latestMapValues: Float32Array | null = null;
  let latestQuantileMembership: Uint32Array | null = null;
  let latestQuantileJoinCounts: Float32Array | null = null;
  let latestQuantileBoundaries: [number, number] = [Number.NaN, Number.NaN];
  const summaryBuffers = [
    mapValuesBuffer,
    quantileMembershipBuffer,
    quantileBoundariesBuffer,
    quantileValidCountBuffer,
    quantileInvalidBuffer,
    quantileObservedBuffer,
    localGearyPBuffer,
    localGearyAdjustedBuffer,
    localGearyBuffer,
    localGearyExceedancesBuffer,
    localGearyClassesBuffer,
    multivariateGearyPBuffer,
    multivariateGearyClassesBuffer,
    loshBuffer,
    loshStatusBuffer,
    loshAbsoluteBuffer,
    loshAbsoluteStatusBuffer,
    loshInferenceBuffer,
    loshInferenceStatusBuffer,
    loshPermutationBuffer,
    loshAdjustedBuffer,
    loshExceedancesBuffer,
    loshAbsoluteInferenceBuffer,
    loshAbsoluteInferenceStatusBuffer,
    loshAbsolutePermutationBuffer,
    loshAbsoluteAdjustedBuffer,
    loshAbsoluteExceedancesBuffer,
    leeLBuffer,
    leeLPBuffer,
    leeLAdjustedBuffer,
    analyticMoranPBuffer,
    analyticMoranAdjustedBuffer,
    neighborOverflowBuffer,
    neighborRequiredBuffer,
    neighborInvalidBuffer,
    localGearyOverflowBuffer,
    loshPermutationOverflowBuffer,
    loshAbsolutePermutationOverflowBuffer,
    leeLOverflowBuffer,
    spatialPearsonBuffer,
    gammaCrossProductBuffer,
    gammaSquaredDifferenceBuffer,
    gammaAbsoluteDifferenceBuffer,
    generationOutput
  ];
  const reader = new SummaryReader(
    resources,
    'meuse-neighborhood-summary',
    summaryBuffers.map((buffer, index) => ({
      buffer,
      size: index === summaryBuffers.length - 1 ? 4 : buffer.byteLength
    })),
    bytes => {
      let offset = 0;
      const readFloats = (length: number): Float32Array => {
        const values = new Float32Array(bytes, offset, length);
        offset += length * 4;
        return values;
      };
      const readIntegers = (length: number): Uint32Array => {
        const values = new Uint32Array(bytes, offset, length);
        offset += length * 4;
        return values;
      };
      const mapValues = new Float32Array(readFloats(rowCount));
      const quantileMembership = readIntegers(rowCount);
      const quantileBoundaries = readFloats(2);
      const quantileValidCount = readIntegers(1)[0];
      const quantileInvalid = readIntegers(1)[0];
      const quantileJoinCounts = readFloats(rowCount);
      const gearyPValues = readFloats(rowCount);
      const gearyAdjusted = readFloats(rowCount);
      const localGeary = readFloats(rowCount);
      readIntegers(rowCount);
      readIntegers(rowCount);
      const multivariateP = readFloats(rowCount);
      readIntegers(rowCount);
      const losh = readFloats(rowCount);
      const loshStatus = readIntegers(rowCount);
      const loshAbsolute = readFloats(rowCount);
      const loshAbsoluteStatus = readIntegers(rowCount);
      const loshInference = readFloats(rowCount);
      const loshInferenceStatus = readIntegers(rowCount);
      const loshP = readFloats(rowCount);
      const loshAdjusted = readFloats(rowCount);
      readIntegers(rowCount);
      const loshAbsoluteInference = readFloats(rowCount);
      const loshAbsoluteInferenceStatus = readIntegers(rowCount);
      const loshAbsoluteP = readFloats(rowCount);
      const loshAbsoluteAdjusted = readFloats(rowCount);
      readIntegers(rowCount);
      const leeL = readFloats(rowCount);
      const leeLP = readFloats(rowCount);
      const leeLAdjusted = readFloats(rowCount);
      const analyticMoranP = readFloats(rowCount);
      const analyticMoranAdjusted = readFloats(rowCount);
      const weightsStatus = readIntegers(3);
      const localGearyStatus = readIntegers(1);
      const loshPermutationStatus = readIntegers(1);
      const loshAbsolutePermutationStatus = readIntegers(1);
      const leeLStatus = readIntegers(1);
      const edgeStatistics = readFloats(4);
      const generationResult = readIntegers(1)[0];
      if (generationResult !== generation) return;
      latestMapValues = mapValues;
      latestQuantileMembership = quantileMembership;
      latestQuantileJoinCounts = quantileJoinCounts;
      latestQuantileBoundaries = [quantileBoundaries[0], quantileBoundaries[1]];
      const summarizeP = (values: Float32Array) => {
        let tested = 0;
        let rawSignificant = 0;
        for (const value of values) {
          if (!Number.isFinite(value)) continue;
          tested++;
          if (value <= 0.05) rawSignificant++;
        }
        return `${rawSignificant}/${tested} rows p ≤ .05`;
      };
      const summarizeAdjusted = (values: Float32Array) => {
        const count = Array.from(values).filter(
          value => Number.isFinite(value) && value <= 0.05
        ).length;
        return `${count} BY-adjusted p ≤ .05`;
      };
      const finiteMean = (values: Float32Array) => {
        const finiteValues = Array.from(values).filter(Number.isFinite);
        return finiteValues.length
          ? finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
          : NaN;
      };
      const summarizeStatuses = (statuses: Uint32Array) => {
        const counts = new Map<number, number>();
        for (const status of statuses) counts.set(status, (counts.get(status) ?? 0) + 1);
        return Array.from(counts, ([status, count]) => `${status}:${count}`).join(', ');
      };
      let loshStatisticDifference = 0;
      let comparableLoshRows = 0;
      for (let row = 0; row < rowCount; row++) {
        if (Number.isFinite(losh[row]) && Number.isFinite(loshInference[row])) {
          loshStatisticDifference = Math.max(
            loshStatisticDifference,
            Math.abs(losh[row] - loshInference[row])
          );
          comparableLoshRows++;
        }
      }
      let absoluteLoshStatisticDifference = 0;
      let comparableAbsoluteLoshRows = 0;
      for (let row = 0; row < rowCount; row++) {
        if (Number.isFinite(loshAbsolute[row]) && Number.isFinite(loshAbsoluteInference[row])) {
          absoluteLoshStatisticDifference = Math.max(
            absoluteLoshStatisticDifference,
            Math.abs(loshAbsolute[row] - loshAbsoluteInference[row])
          );
          comparableAbsoluteLoshRows++;
        }
      }
      ctx.setReadout(
        'weightsStatus',
        `${weightsStatus[0] ? 'overflow' : 'complete'} · required slots ${weightsStatus[1]} · invalid rows ${weightsStatus[2]} · 8-neighbor row-standardized kNN`
      );
      ctx.setReadout(
        'localGearyStatus',
        `mean local Geary ${finiteMean(localGeary).toFixed(3)} · ${summarizeP(gearyPValues)} · ${summarizeAdjusted(gearyAdjusted)} · overflow ${localGearyStatus[0]}`
      );
      ctx.setReadout(
        'multivariateStatus',
        `seed 31 · ${summarizeP(multivariateP)} · equal-tail minimum p ${2 / (MAXIMUM_PERMUTATIONS + 1)} · no multiplicity adjustment`
      );
      ctx.setReadout(
        'variabilityStatus',
        `${ctx.options.loshPower === 'variance' ? 'Variance power 2' : 'Absolute-deviation power 1'}: mean ${finiteMean(ctx.options.loshPower === 'variance' ? losh : loshAbsolute).toFixed(3)} · recipe status ${summarizeStatuses(ctx.options.loshPower === 'variance' ? loshStatus : loshAbsoluteStatus)} · residual-test status ${summarizeStatuses(ctx.options.loshPower === 'variance' ? loshInferenceStatus : loshAbsoluteInferenceStatus)} · LOSH/test max |Δ| ${ctx.options.loshPower === 'variance' ? (comparableLoshRows ? loshStatisticDifference.toExponential(1) : '—') : comparableAbsoluteLoshRows ? absoluteLoshStatisticDifference.toExponential(1) : '—'} · fixed-residual raw pseudo-p ${summarizeP(ctx.options.loshPower === 'variance' ? loshP : loshAbsoluteP)} · ${summarizeAdjusted(ctx.options.loshPower === 'variance' ? loshAdjusted : loshAbsoluteAdjusted)} · overflow ${ctx.options.loshPower === 'variance' ? loshPermutationStatus[0] : loshAbsolutePermutationStatus[0]} · analytic Moran ${summarizeP(analyticMoranP)} · ${summarizeAdjusted(analyticMoranAdjusted)}`
      );
      ctx.setReadout(
        'leeLStatus',
        `mean local Lee L focal metal/copper ${finiteMean(leeL).toFixed(3)} · ${summarizeP(leeLP)} · ${summarizeAdjusted(leeLAdjusted)} · overflow ${leeLStatus[0]}`
      );
      ctx.setReadout(
        'globalAssociation',
        `Spatial Pearson ${ctx.options.response}/copper edge correlation ${edgeStatistics[0].toFixed(3)} · Gamma ${ctx.options.gammaOperation} ${edgeStatistics[['cross-product', 'squared-difference', 'absolute-difference'].indexOf(ctx.options.gammaOperation) + 1].toFixed(3)} · descriptive statistics only; no analytic p-values`
      );
      if (quantileInvalid !== 0) {
        ctx.setReadout(
          'quantileSummary',
          `Selected quantile unavailable (invalid setting ${quantileInvalid}).`
        );
        ctx.setReadout('quantileJoinSummary', 'Join counts unavailable for the selected quantile.');
      } else {
        const selectedQuantile = getSelectedQuantile(
          ctx.options.quantileCount,
          ctx.options.quantileBand
        );
        const memberCount = Array.from(quantileMembership).reduce(
          (total, member) => total + (member === 1 ? 1 : 0),
          0
        );
        const formatBoundary = (value: number) =>
          Number.isFinite(value) ? value.toFixed(3) : value < 0 ? '−∞' : '+∞';
        ctx.setReadout(
          'quantileSummary',
          `${quantileValidCount} finite rows · ${memberCount} in q${selectedQuantile}/${ctx.options.quantileCount} · type-5 interval [${formatBoundary(quantileBoundaries[0])}, ${formatBoundary(quantileBoundaries[1])}) log units`
        );
        let summedJoinCount = 0;
        let maximumJoinCount = 0;
        let finiteJoinCountRows = 0;
        for (const joinCount of quantileJoinCounts) {
          if (!Number.isFinite(joinCount)) continue;
          summedJoinCount += joinCount;
          maximumJoinCount = Math.max(maximumJoinCount, joinCount);
          finiteJoinCountRows++;
        }
        ctx.setReadout(
          'quantileJoinSummary',
          `${summedJoinCount.toFixed(0)} summed directed local joins · max ${maximumJoinCount.toFixed(0)} / ${NEIGHBOR_COUNT} · ${finiteJoinCountRows} finite rows; binary neighbor edges, not row-standardized weight sums`
        );
      }
      ctx.setChart('inferenceResolution', {
        kind: 'bars',
        title: 'Permutation resolution vs Bonferroni threshold',
        labels: ['One-sided local min p', 'Two-sided equal-tail min p', 'Bonferroni α/155'],
        values: [1 / (MAXIMUM_PERMUTATIONS + 1), 2 / (MAXIMUM_PERMUTATIONS + 1), 0.05 / rowCount],
        description:
          'The multivariate test uses P=999, so its equal-tail minimum is 0.002, while Bonferroni α/155 is about 0.000323. That adjustment cannot reject any result at this permutation count; the scene therefore reports raw multivariate p-values and does not imply familywise significance.'
      });
      ctx.requestLayers();
      ctx.refreshTooltip();
    }
  );
  ctx.setReadout('sampleCount', rowCount);
  ctx.setReadout('weightsStatus', 'Eight-neighbor weights pending GPU execution.');
  ctx.setReadout('localGearyStatus', 'Local Geary inference pending.');
  ctx.setReadout('multivariateStatus', 'Joint-vector local Geary inference pending.');
  ctx.setReadout('variabilityStatus', 'LOSH diagnostics pending.');
  ctx.setReadout('leeLStatus', 'Local Lee L inference pending.');
  ctx.setReadout('globalAssociation', 'Spatial Pearson and Gamma pending.');
  ctx.setReadout('quantileSummary', 'Selected type-5 quantile membership pending GPU execution.');
  ctx.setReadout('quantileJoinSummary', 'Raw directed local join counts pending GPU execution.');
  ctx.setChart('inferenceResolution', null);
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        reader.request(commandEncoder);
        graphDirty = false;
      } else {
        reader.flush(commandEncoder);
      }
    },
    setOption(id, _value, state) {
      if (id === 'response') {
        const responseValues = Float32Array.from(responseColumns[state.response]);
        valuesBuffer.write(responseValues);
      }
      generation++;
      generationInput.write(Uint32Array.of(generation));
      mapControlBuffer.write(
        Uint32Array.of(getDisplayIndex(state.display), state.loshPower === 'variance' ? 2 : 1)
      );
      quantileParameterBuffer.write(
        Uint32Array.of(
          Number(state.quantileCount),
          getSelectedQuantile(state.quantileCount, state.quantileBand)
        )
      );
      latestMapValues = null;
      latestQuantileMembership = null;
      latestQuantileJoinCounts = null;
      ctx.setReadout('quantileSummary', 'Selected quantile values pending GPU execution.');
      ctx.setReadout(
        'quantileJoinSummary',
        'Raw directed local join counts pending GPU execution.'
      );
      ctx.setReadout('weightsStatus', 'Recomputing eight-neighbor weights and local statistics.');
      ctx.setReadout(
        'localGearyStatus',
        'Local Geary values and permutation summaries are pending.'
      );
      ctx.setReadout('multivariateStatus', 'Joint-vector permutation summaries are pending.');
      ctx.setReadout('variabilityStatus', 'LOSH statuses and fixed-residual test are pending.');
      ctx.setReadout('leeLStatus', 'Local Lee L values and BY-adjusted summaries are pending.');
      ctx.setReadout('globalAssociation', 'Spatial Pearson and Gamma summaries are pending.');
      ctx.setChart('inferenceResolution', null);
      reader.markStale();
      graphDirty = true;
      if (id === 'response' || id === 'display' || id === 'quantileCount' || id === 'quantileBand')
        ctx.refreshLegends();
      ctx.requestLayers();
      ctx.refreshTooltip();
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [];
      if (latestMapValues) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-neighborhood-statistics',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: mapPositionsBuffer,
            values: mapValuesBuffer,
            valueFormat: 'float32',
            instanceCount: rowCount,
            radiusPixels: 5,
            colormap:
              ctx.options.display === 'quantile-membership'
                ? 'category'
                : ctx.options.display.endsWith('-p')
                  ? 'magma'
                  : 'cividis',
            valueRange:
              ctx.options.display === 'quantile-membership'
                ? [0, 1]
                : ctx.options.display === 'quantile-join-count'
                  ? [0, NEIGHBOR_COUNT]
                  : ctx.options.display.endsWith('-p')
                    ? [0, 1]
                    : ctx.options.display === 'local-geary'
                      ? [0, 3]
                      : ctx.options.display === 'losh'
                        ? [0, 2]
                        : [-1, 1],
            ...(ctx.options.display === 'quantile-membership'
              ? {
                  palette: [
                    [202, 211, 219, 255],
                    [226, 126, 44, 255]
                  ] as const
                }
              : {}),
            noDataColor: [0, 0, 0, 0],
            outlineColor: [19, 24, 28, 210],
            outlineWidthPixels: 1
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'meuse-neighborhood-river',
          coordinateOrigin: [origin[0], origin[1], 0],
          segments: riverBuffer,
          instanceCount: Math.max(0, riverSegments.length / 4),
          color: [49, 113, 151, 230],
          widthPixels: 2
        })
      );
      return layers;
    },
    getTooltip(event: ScenePointerEvent): string | null {
      if (!event.coordinate || !latestMapValues) return null;
      const projected = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < rowCount; row++) {
        const distance = Math.hypot(
          mapPositions[row * 2] - projected[0],
          mapPositions[row * 2 + 1] - projected[1]
        );
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      if (nearest < 0 || nearestDistance > 60) return null;
      const value = latestMapValues[nearest];
      if (
        ctx.options.display === 'quantile-membership' ||
        ctx.options.display === 'quantile-join-count'
      ) {
        if (!latestQuantileMembership || !latestQuantileJoinCounts) return null;
        const selectedQuantile = getSelectedQuantile(
          ctx.options.quantileCount,
          ctx.options.quantileBand
        );
        const member = latestQuantileMembership[nearest] === 1;
        const boundaryText = `[${Number.isFinite(latestQuantileBoundaries[0]) ? latestQuantileBoundaries[0].toFixed(3) : '−∞'}, ${Number.isFinite(latestQuantileBoundaries[1]) ? latestQuantileBoundaries[1].toFixed(3) : '+∞'})`;
        return `Nearest ${ctx.options.response} sample within ${Math.round(nearestDistance)} m · selected q${selectedQuantile}/${ctx.options.quantileCount} ${member ? 'member' : 'not a member'} · type-5 interval ${boundaryText} log units · raw directed local join count ${latestQuantileJoinCounts[nearest].toFixed(0)}/${NEIGHBOR_COUNT}`;
      }
      return Number.isFinite(value)
        ? `Meuse ${ctx.options.response} · ${DISPLAY_LABELS[ctx.options.display]} ${value.toFixed(4)} · ${Math.round(nearestDistance)} m from site`
        : `Meuse ${ctx.options.response} · ${DISPLAY_LABELS[ctx.options.display]} unavailable for this row`;
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
