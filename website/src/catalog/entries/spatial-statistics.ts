// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: spatial statistics. Autocorrelation, hot spots, clustering, outliers, permutation
// inference, scan and space-time tests, rate smoothing and spatial validation.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';
const REFERENCE = 'api/gpu-spatial-analysis/operators-statistics';

const weightsParameter: ParamDoc = {
  name: 'weights',
  type: 'GPUSpatialWeights | GPUChunkedSpatialWeights',
  required: true,
  description: 'Neighbor structure (CSR) from queen, KNN or distance weights.'
};
const valuesParameter: ParamDoc = {
  name: 'values',
  type: 'GPUFloat32OrUint32Values',
  required: true,
  description: 'One value per row; non-finite rows are excluded.'
};
const maskParameter: ParamDoc = {
  name: 'mask',
  type: 'GraphDataView<"uint32"> | undefined',
  description: 'Nonzero rows take part in the analysis.'
};
const islandsParameter: ParamDoc = {
  name: 'islands',
  type: 'GPUIslandPolicy | undefined',
  default: "'exclude'",
  description: "Rows without neighbors: 'exclude' them or 'keep' (esda convention)."
};
const multipleTestingParameter: ParamDoc = {
  name: 'multipleTesting',
  type: 'GPUMultipleTestingMethod | undefined',
  default: "'none'",
  description: 'Bonferroni, Holm, Benjamini-Hochberg or Benjamini-Yekutieli.'
};
const alternativeParameter: ParamDoc = {
  name: 'alternative',
  type: 'GPUInferenceAlternative | undefined',
  default: "'two-sided'",
  description: "Tail of the test: 'two-sided', 'greater' or 'lesser'."
};
const permutationParametersParameter: ParamDoc = {
  name: 'parameters',
  type: 'GraphDataView<"uint32">',
  required: true,
  description: 'Per-frame seed and permutation count from getGPUPermutationParameterValues.'
};
const maximumPermutationsParameter: ParamDoc = {
  name: 'maximumPermutations',
  type: 'number',
  required: true,
  description: 'Compile-time cap on permutations; the per-frame count may not exceed it.'
};
const autocorrelationParametersParameter: ParamDoc = {
  name: 'parameters',
  type: 'GraphDataView<"float32">',
  required: true,
  description: 'Per-frame settings from getGPUSpatialAutocorrelationParameterValues.'
};
const positionsParameter: ParamDoc = {
  name: 'positions',
  type: 'GraphDataView<"float32x2">',
  required: true,
  description: 'Planar point coordinates, one float32x2 per point.'
};
const zScoresOutput: OutputDoc = {
  name: 'zScores',
  type: 'GraphDataView<"float32">',
  description: 'Analytic z-score per row; NaN where undefined.'
};
const labelsOutput: OutputDoc = {
  name: 'labels',
  type: 'GraphDataView<"uint32">',
  description: 'Cluster id per point, or 0xffffffff for noise.'
};

const SETUP_NOTE = '// graph, weights and views are set up as in the Local Moran example';

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------- autocorrelation
  {
    id: 'local-morans-i',
    symbol: 'GPULocalMoran',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: "Local Moran's I",
    summary:
      'Finds clusters and outliers of similar values among neighbors (LISA), with z-scores and quadrants.',
    explanation:
      "Maps where high values sit beside high values (HH), low beside low (LL), or oppose their neighbors (HL, LH). Each row's value is compared with the weighted mean of its neighbors, and an analytic z-score approximates the esda permutation test.",
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table', 'polygons']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      weightsParameter,
      valuesParameter,
      autocorrelationParametersParameter,
      {
        name: 'quadrantGating',
        type: 'GPULocalMoranQuadrantGating | undefined',
        default: "'analytic'",
        description: "'analytic' keeps significant quadrants only; 'none' returns all."
      },
      multipleTestingParameter,
      islandsParameter,
      {
        name: 'minimumNeighbors',
        type: 'number | undefined',
        default: '1',
        description: 'Rows with fewer neighbors get NaN z and quadrant 0.'
      },
      {
        name: 'output',
        type: 'GPULocalMoranOutput',
        required: true,
        description: 'Caller-owned result views; only zScores is required.'
      }
    ],
    outputs: [
      zScoresOutput,
      {
        name: 'localI',
        type: 'GraphDataView<"float32">',
        description: "Local Moran's I per row (esda Is)."
      },
      {
        name: 'spatialLag',
        type: 'GraphDataView<"float32">',
        description: 'Weighted neighbor deviation, the Moran scatterplot y axis.'
      },
      {
        name: 'quadrants',
        type: 'GraphDataView<"uint32">',
        description: '1 HH, 2 LH, 3 LL, 4 HL; 0 when not significant.'
      },
      {
        name: 'adjustedPValues',
        type: 'GraphDataView<"float32">',
        description: 'p-values after multiple-testing correction.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULocalMoran, getGPUSpatialAutocorrelationParameterValues, GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const graph = new GPUCommandGraph(device, {id: 'lisa'});
const parameters = new GPUParameterBuffer(device, {
  id: 'moran-parameters', format: 'float32',
  length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
});
// weights (queen or KNN CSR), values, zScores, quadrants: graph views
graph.add(new GPULocalMoran({
  weights, values, parameters: parameters.importToGraph(graph),
  multipleTesting: 'benjamini-hochberg',
  output: {zScores, quadrants}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'hotspots',
      caption: 'Local Moran clusters of September 27 rainfall.',
      seed: 501
    },
    related: [
      'hot-spot-analysis-gi-star',
      'local-permutation-test',
      'local-gearys-c',
      'global-spatial-statistics'
    ],
    lensScenario: 'gpu-spatial-analysis.local-moran',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'local-gearys-c',
    symbol: 'GPULocalGeary',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: "Local Geary's C",
    summary:
      'Scores how different each row is from its neighbors, a dissimilarity complement to local Moran.',
    explanation:
      'Finds places that resemble their surroundings (low c) or contrast with them (high c). It sums squared differences to neighbors, standardized by the population variance.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      autocorrelationParametersParameter,
      maskParameter,
      islandsParameter,
      {
        name: 'output',
        type: 'GPULocalGearyOutput',
        required: true,
        description: 'Caller-owned result views.'
      }
    ],
    outputs: [
      {
        name: 'statistic',
        type: 'GraphDataView<"float32">',
        description: 'Local Geary value per row.'
      },
      {
        name: 'globalStatistics',
        type: 'GraphDataView<"float32">',
        description: '[n, mean, variance, sd, islandCount] used.'
      }
    ],
    example: `import {GPULocalGeary} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPULocalGeary({
  weights, values,
  parameters: parameters.importToGraph(graph),
  islands: 'keep',
  output: {statistic}
}));`,
    preview: {
      kind: 'hotspots',
      caption: 'Low c where neighbors agree, high c at contrasts.',
      seed: 502
    },
    related: ['local-morans-i', 'multivariate-local-gearys-c', 'local-permutation-test'],
    lensScenario: 'gpu-spatial-analysis.local-geary',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'hot-spot-analysis-gi-star',
    symbol: 'GPUHotSpotAnalysis',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Hot spots (Getis-Ord Gi*)',
    summary:
      'Marks statistically significant hot and cold spots with z-scores and 90, 95 and 99% confidence bins.',
    explanation:
      'Shows where high or low values cluster, as in the ArcGIS Hot Spot Analysis tool. Gi* compares the sum of values in each neighborhood (including the row itself) with what chance would give.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table', 'polygons']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      weightsParameter,
      valuesParameter,
      autocorrelationParametersParameter,
      {
        name: 'selfWeight',
        type: 'number | undefined',
        description: 'Weight of the focal row itself; 1 gives Gi*, 0 gives Gi.'
      },
      {
        name: 'minimumNeighbors',
        type: 'number | undefined',
        default: '1',
        description: 'Esri recommends 8 for Gi*.'
      },
      multipleTestingParameter,
      islandsParameter,
      {
        name: 'output',
        type: 'GPUHotSpotAnalysisOutput',
        required: true,
        description: 'Caller-owned result views.'
      }
    ],
    outputs: [
      zScoresOutput,
      {
        name: 'bins',
        type: 'GraphDataView<"sint32">',
        description: '+/-3, 2, 1 for 99, 95, 90% hot or cold; 0 otherwise.'
      },
      {
        name: 'adjustedPValues',
        type: 'GraphDataView<"float32">',
        description: 'p-values after multiple-testing correction.'
      }
    ],
    example: `import {GPUHotSpotAnalysis} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUHotSpotAnalysis({
  weights, values,
  parameters: parameters.importToGraph(graph),
  selfWeight: 1,
  minimumNeighbors: 8,
  multipleTesting: 'benjamini-hochberg',
  output: {zScores, bins}
}));`,
    preview: {
      kind: 'hotspots',
      caption: 'Unusually wet and dry gauge neighborhoods during Helene.',
      seed: 503
    },
    related: ['local-morans-i', 'emerging-hot-spots', 'multiple-testing-adjustment'],
    lensScenario: 'gpu-spatial-analysis.hot-spot-analysis',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'global-spatial-statistics',
    symbol: 'GPUGlobalSpatialStatistics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Global autocorrelation',
    summary:
      "Computes Moran's I, Geary's C, Getis-Ord G, bivariate Moran and join counts in one pass.",
    explanation:
      'Tests whether a whole map is clustered, dispersed or random before mapping local statistics. Each statistic has an analytic z and p.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      {
        name: 'statistics',
        type: 'readonly GPUGlobalSpatialStatistic[]',
        required: true,
        description: "Any of 'moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'."
      },
      {
        name: 'secondValues',
        type: 'GPUFloat32OrUint32Values | undefined',
        description: 'Second variable for bivariate Moran.'
      },
      alternativeParameter,
      islandsParameter,
      maskParameter,
      {
        name: 'output',
        type: 'GPUGlobalSpatialStatisticsOutput',
        required: true,
        description: 'Summary block laid out by GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Statistic, expectation, variance, z and p per requested block.'
      },
      {
        name: 'joinCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Ordered-pair [BB, BW, WW] join counts.'
      }
    ],
    example: `import {GPUGlobalSpatialStatistics} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUGlobalSpatialStatistics({
  weights, values,
  statistics: ['moran', 'geary', 'getisOrdG'],
  alternative: 'greater',
  output: {summary}
}));
// read summary with GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`,
    preview: {kind: 'histogram', caption: 'One number per statistic for the whole map.', seed: 504},
    related: ['global-permutation-test', 'local-morans-i'],
    lensScenario: 'gpu-spatial-analysis.global-spatial-statistics',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'spatial-pearson',
    symbol: 'GPUSpatialPearson',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial Pearson correlation',
    summary:
      'Correlates one variable at a location with another at its neighbors, across weighted edges.',
    explanation:
      'Best for bivariate spatial association: does high income here go with high rents next door? It is the weighted Pearson correlation of x_i and y_j over the directed edges.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      {
        name: 'secondValues',
        type: 'GraphDataView<"uint32"> | GraphDataView<"float32">',
        required: true,
        description: 'The variable observed at the neighbors.'
      },
      maskParameter,
      {
        name: 'output',
        type: 'GPUEdgeAssociationOutput',
        required: true,
        description: 'One-value result view.'
      }
    ],
    outputs: [
      {
        name: 'statistic',
        type: 'GraphDataView<"float32">',
        description: 'Correlation r; NaN when undefined. No analytic p-value.'
      }
    ],
    example: `import {GPUSpatialPearson} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSpatialPearson({
  weights,
  values: income,
  secondValues: rent,
  output: {statistic}
}));`,
    preview: {
      kind: 'regression',
      caption: 'Neighbor values plotted against each other.',
      seed: 505
    },
    related: ['gamma-statistic', 'global-spatial-statistics'],
    lensScenario: 'gpu-spatial-analysis.spatial-pearson',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'gamma-statistic',
    symbol: 'GPUGammaStatistic',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: "Hubert's Gamma",
    summary:
      'Sums a pairwise similarity over weighted neighbor edges, the general form of spatial association.',
    explanation:
      'Best for custom association measures: cross-product gives Moran-like, squared difference Geary-like behavior. Pair it with the global permutation machinery for a pseudo p-value.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      {
        name: 'secondValues',
        type: 'GPUFloat32OrUint32Values | undefined',
        description: 'Second variable; defaults to values.'
      },
      {
        name: 'operation',
        type: 'GPUGammaOperation | undefined',
        description: "Edge operation, for example 'cross-product' or 'squared-difference'."
      },
      maskParameter,
      {
        name: 'output',
        type: 'GPUEdgeAssociationOutput',
        required: true,
        description: 'One-value result view.'
      }
    ],
    outputs: [
      {
        name: 'statistic',
        type: 'GraphDataView<"float32">',
        description: 'Gamma over the directed edges.'
      }
    ],
    example: `import {GPUGammaStatistic} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUGammaStatistic({
  weights, values,
  operation: 'squared-difference',
  output: {statistic}
}));`,
    preview: {
      kind: 'spatial-weights',
      caption: 'Every weighted edge contributes one term.',
      seed: 506
    },
    related: ['spatial-pearson', 'global-permutation-test'],
    lensScenario: 'gpu-spatial-analysis.gamma-statistic',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'local-spatial-heteroscedasticity',
    symbol: 'GPULocalSpatialHeteroscedasticity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'LOSH (local variability)',
    summary:
      'Measures how variable values are within each neighborhood, to find unstable or heterogeneous areas.',
    explanation:
      'Complements local Moran: LOSH says where the spread of neighbors is unusually high or low, not where values are high. It is the neighbor-weighted mean of absolute deviations over the global mean residual.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'measure'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'One float32 value per row.'
      },
      {
        name: 'residualPower',
        type: '2 | 1 | undefined',
        description: 'Exponent a of the absolute deviations: 2 (variance-like) or 1.'
      },
      maskParameter,
      {
        name: 'output',
        type: '{ statistic: GraphDataView<"float32">; status: GraphDataView<"uint32"> }',
        required: true,
        description: 'Result views.'
      }
    ],
    outputs: [
      {name: 'statistic', type: 'GraphDataView<"float32">', description: 'LOSH H_i per row.'},
      {name: 'status', type: 'GraphDataView<"uint32">', description: 'Status code word.'},
      {
        name: 'normalizedResiduals',
        type: 'GraphDataView<"float32">',
        description: 'Optional normalized residual powers for permutation.'
      }
    ],
    example: `import {GPULocalSpatialHeteroscedasticity} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPULocalSpatialHeteroscedasticity({
  weights, values,
  residualPower: 2,
  output: {statistic, status}
}));`,
    preview: {kind: 'hotspots', caption: 'High LOSH marks uneven neighborhoods.', seed: 507},
    related: ['losh-residual-permutation-test', 'local-morans-i'],
    lensScenario: 'gpu-spatial-analysis.local-spatial-heteroscedasticity',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'multivariate-local-gearys-c',
    symbol: 'GPUMultivariateLocalGeary',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: "Multivariate local Geary's C",
    summary: 'Finds places that are similar to their neighbors across several variables at once.',
    explanation:
      'Best for multi-attribute clustering, such as a socioeconomic profile. It averages the row-standardized local Geary over the variables stored row-major in one column.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      weightsParameter,
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Row-major rows x variableCount values.'
      },
      {
        name: 'variableCount',
        type: 'number',
        required: true,
        description: 'Number of variables per row.'
      },
      maskParameter,
      {
        name: 'output',
        type: '{ statistic: GraphDataView<"float32">; status: GraphDataView<"uint32"> }',
        required: true,
        description: 'Result views.'
      }
    ],
    outputs: [
      {
        name: 'statistic',
        type: 'GraphDataView<"float32">',
        description: 'Mean local Geary per row; low means alike.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Per-row completion evidence bitmask.'
      }
    ],
    example: `import {GPUMultivariateLocalGeary} from '${IMPORT_PATH}';

${SETUP_NOTE}
// values: income, age, density interleaved per row
graph.add(new GPUMultivariateLocalGeary({
  weights, values,
  variableCount: 3,
  output: {statistic, status}
}));`,
    preview: {
      kind: 'hotspots',
      caption: 'Low values: neighbors agree on every variable.',
      seed: 508
    },
    related: ['multivariate-local-geary-permutation-test', 'local-gearys-c'],
    lensScenario: 'gpu-spatial-analysis.multivariate-local-geary',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- permutation inference
  {
    id: 'global-permutation-test',
    symbol: 'GPUGlobalPermutationTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Global permutation test',
    summary: 'Gets a pseudo p-value for a global statistic by shuffling values across the map.',
    explanation:
      "Best when normal assumptions are doubtful. It reshuffles values, recomputes Moran's I, Geary's C, G, bivariate Moran or Lee's L each time, and ranks the observed value in that reference distribution.",
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      {
        name: 'statistic',
        type: 'GPUGlobalPermutationStatistic',
        required: true,
        description: "'moran', 'geary', 'getisOrdG', 'bivariateMoran' or 'leesL'."
      },
      permutationParametersParameter,
      maximumPermutationsParameter,
      alternativeParameter,
      {
        name: 'stopAfterExceedances',
        type: 'number | undefined',
        description: 'Stop early once this many permutations exceed the observed value.'
      },
      {
        name: 'output',
        type: 'GPUGlobalPermutationTestOutput',
        required: true,
        description: 'Summary, optional distribution and histogram.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Observed, pseudo p, simulated z, laid out by GPU_GLOBAL_PERMUTATION_SUMMARY.'
      },
      {
        name: 'referenceDistribution',
        type: 'GraphDataView<"float32">',
        description: 'Statistic of each permutation.'
      },
      {
        name: 'histogram',
        type: 'GraphDataView<"uint32">',
        description: 'Binned reference distribution.'
      }
    ],
    example: `import {GPUGlobalPermutationTest, GPU_PERMUTATION_PARAMETER_LENGTH, getGPUPermutationParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const permutation = new GPUParameterBuffer(device, {
  id: 'permutation', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues({seed: 3, permutationCount: 999})
});
graph.add(new GPUGlobalPermutationTest({
  weights, values, statistic: 'moran',
  parameters: permutation.importToGraph(graph),
  maximumPermutations: 999,
  output: {summary, histogram}
}));`,
    preview: {kind: 'histogram', caption: 'Observed value against 999 shuffled maps.', seed: 509},
    related: ['global-spatial-statistics', 'local-permutation-test'],
    lensScenario: 'gpu-spatial-analysis.global-permutation-test',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'local-permutation-test',
    symbol: 'GPULocalPermutationTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local permutation test',
    summary:
      'Gives each row a conditional-permutation pseudo p-value for local Moran, Geary, G and join counts.',
    explanation:
      "Best for esda-style significance of LISA clusters without normality assumptions. For each row it holds the value fixed and resamples its neighbors' values.",
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table', 'polygons']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      valuesParameter,
      {
        name: 'statistic',
        type: 'GPULocalPermutationStatistic',
        required: true,
        description: "For example 'localMoran', 'localGeary', 'localG', 'localGStar'."
      },
      permutationParametersParameter,
      maximumPermutationsParameter,
      alternativeParameter,
      multipleTestingParameter,
      {
        name: 'output',
        type: 'GPULocalPermutationTestOutput',
        required: true,
        description: 'Exceedances and pseudo p-values, plus optional extras.'
      }
    ],
    outputs: [
      {
        name: 'pseudoPValues',
        type: 'GraphDataView<"float32">',
        description: 'Pseudo p-value per row; NaN if not tested.'
      },
      {
        name: 'exceedances',
        type: 'GraphDataView<"uint32">',
        description: 'Permutations at least as extreme as observed.'
      },
      {
        name: 'significant',
        type: 'GraphDataView<"uint32">',
        description: '1 where the pseudo p-value is significant.'
      },
      {
        name: 'status',
        type: '{overflow: GraphDataView<"uint32">}',
        description: 'Completeness status; overflow is nonzero on truncation.'
      }
    ],
    example: `import {GPULocalPermutationTest} from '${IMPORT_PATH}';

${SETUP_NOTE}
// permutation: GPUParameterBuffer as in the global permutation test
graph.add(new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran',
  parameters: permutation.importToGraph(graph),
  maximumPermutations: 999,
  multipleTesting: 'benjamini-hochberg',
  output: {exceedances, pseudoPValues, significant, status: {overflow}}
}));`,
    preview: {kind: 'hotspots', caption: 'Rows that survive 999 reshuffles light up.', seed: 510},
    related: ['local-morans-i', 'global-permutation-test', 'multiple-testing-adjustment'],
    lensScenario: 'gpu-spatial-analysis.local-permutation-test',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'losh-residual-permutation-test',
    symbol: 'GPULOSHResidualPermutationTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'LOSH permutation test',
    summary: 'Tests local variability (LOSH) by conditional sampling of the normalized residuals.',
    explanation:
      'Finds which high-LOSH areas are more variable than chance. It computes LOSH and resamples the fixed residual field without replacement.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      weightsParameter,
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'One float32 value per row.'
      },
      {
        name: 'alternative',
        type: '"two-sided" | "greater" | "lesser"',
        required: true,
        description: 'Tail of the test.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Seed and permutation count.'
      },
      maximumPermutationsParameter,
      {
        name: 'residualPower',
        type: '2 | 1 | undefined',
        description: 'Exponent of the absolute residuals.'
      },
      multipleTestingParameter,
      {
        name: 'output',
        type: '{statistic, status, inference}',
        required: true,
        description: 'Result views plus the inference views that hold the p-values.'
      }
    ],
    outputs: [
      {name: 'statistic', type: 'GraphDataView<"float32">', description: 'LOSH per row.'},
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Per-row completion evidence bitmask.'
      },
      {
        name: 'inference',
        type: 'GPULocalPermutationTestOutput',
        description: 'Required: exceedances, pseudoPValues, adjustedPValues, significant, status.'
      }
    ],
    example: `import {GPULOSHResidualPermutationTest} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPULOSHResidualPermutationTest({
  weights, values,
  alternative: 'greater',
  parameters: permutation.importToGraph(graph),
  maximumPermutations: 499,
  output: {
    statistic, status,
    inference: {exceedances, pseudoPValues, adjustedPValues, significant, status: inferenceStatus}
  }
}));`,
    preview: {kind: 'hotspots', caption: 'Observed LOSH against its shuffled null.', seed: 511},
    related: ['local-spatial-heteroscedasticity', 'local-permutation-test'],
    lensScenario: 'gpu-spatial-analysis.losh-residual-permutation-test',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'multivariate-local-geary-permutation-test',
    symbol: 'GPUMultivariateLocalGearyPermutationTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multivariate Geary test',
    summary:
      'Tests multivariate local Geary with permutations and classifies clusters, outliers and negative association.',
    explanation:
      'Adds significance and classes for a multi-variable profile. The seed, tail and tiling are plain props, so no parameter buffer is needed.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      weightsParameter,
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Row-major rows x variableCount values.'
      },
      {
        name: 'variableCount',
        type: 'number',
        required: true,
        description: 'Number of variables per row.'
      },
      {
        name: 'maximumPermutations',
        type: 'number | undefined',
        description: 'Number of conditional permutations.'
      },
      {
        name: 'seed',
        type: 'readonly [number, number] | undefined',
        description: 'Two-word random seed.'
      },
      {
        name: 'significance',
        type: 'number | undefined',
        description: 'Significance level for classes.'
      },
      {
        name: 'multipleTesting',
        type: 'GPUMultipleTestingMethod | undefined',
        default: "'none'",
        description: 'Correction applied to the pseudo p-values.'
      },
      {
        name: 'strategy',
        type: '"exact" | "tiled" | undefined',
        description: 'Exact or row-tiled permutation scheduling.'
      }
    ],
    outputs: [
      {
        name: 'statistic',
        type: 'GraphDataView<"float32">',
        description: 'Mean multivariate local Geary per row.'
      },
      {name: 'pValues', type: 'GraphDataView<"float32">', description: 'Pseudo p-value per row.'},
      {
        name: 'classes',
        type: 'GraphDataView<"uint32">',
        description: 'Cluster, outlier or not-significant class code.'
      }
    ],
    example: `import {GPUMultivariateLocalGearyPermutationTest} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUMultivariateLocalGearyPermutationTest({
  weights, values,
  variableCount: 3,
  maximumPermutations: 999,
  seed: [1, 2],
  multipleTesting: 'benjamini-hochberg',
  output: {statistic, pValues, classes}
}));`,
    preview: {
      kind: 'hotspots',
      caption: 'Significant multi-variable clusters, classified.',
      seed: 512
    },
    related: ['multivariate-local-gearys-c', 'local-permutation-test'],
    lensScenario: 'gpu-spatial-analysis.multivariate-local-geary-permutation-test',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'multiple-testing-adjustment',
    symbol: 'GPUMultipleTestingAdjustment',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multiple-testing adjustment',
    summary:
      'Corrects thousands of per-row p-values for false positives: Bonferroni, Holm, BH or BY.',
    explanation:
      'Use it whenever you map many local tests at once, or the map is full of false hot spots. Results match statsmodels multipletests.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'pValues',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-row p-values; NaN rows are not tested.'
      },
      {
        name: 'method',
        type: 'GPUMultipleTestingMethod',
        required: true,
        description: "'bonferroni', 'holm', 'benjamini-hochberg' or 'benjamini-yekutieli'."
      },
      {
        name: 'minimumFamilySize',
        type: 'number | undefined',
        description: 'Smallest number of tested rows.'
      },
      {
        name: 'output',
        type: 'GPUMultipleTestingAdjustmentOutput',
        required: true,
        description: 'Adjusted p-values and optional tested count.'
      }
    ],
    outputs: [
      {
        name: 'adjustedPValues',
        type: 'GraphDataView<"float32">',
        description: 'Adjusted p-value in [0, 1]; must not alias the input.'
      },
      {
        name: 'testedCount',
        type: 'GraphDataView<"uint32">',
        description: 'Family size m actually used.'
      }
    ],
    example: `import {GPUMultipleTestingAdjustment} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUMultipleTestingAdjustment({
  pValues: pseudoPValues,
  method: 'benjamini-hochberg',
  output: {adjustedPValues, testedCount}
}));`,
    preview: {
      kind: 'histogram',
      caption: 'p-values pushed up; fewer rows stay significant.',
      seed: 513
    },
    related: ['local-permutation-test', 'hot-spot-analysis-gi-star', 'local-morans-i'],
    lensScenario: 'gpu-spatial-analysis.multiple-testing-adjustment',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- rates
  {
    id: 'assuncao-reis-standardization',
    symbol: 'GPUAssuncaoReisStandardization',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Assuncao-Reis rates',
    summary:
      'Standardizes rates by population so Moran statistics are not driven by small denominators.',
    explanation:
      'Best before local or global Moran on rates such as deaths per capita. Each rate becomes an empirical-Bayes z that accounts for its population size, as in esda Moran_Rate.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'summarize'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {
        name: 'events',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Event counts per row (numerator).'
      },
      {
        name: 'populations',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Population at risk per row (denominator).'
      },
      maskParameter,
      {
        name: 'output',
        type: 'GPUAssuncaoReisStandardizationOutput',
        required: true,
        description: 'Standardized rates plus optional statistics.'
      }
    ],
    outputs: [
      {
        name: 'standardizedRates',
        type: 'GraphDataView<"float32">',
        description: 'EB-standardized rate z_i; NaN where invalid.'
      },
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: '[n, b, s2, a, totalPopulation].'
      },
      {
        name: 'status',
        type: '{invalidCount?: GraphDataView<"uint32">}',
        description: 'Optional count of invalid rows.'
      }
    ],
    example: `import {GPUAssuncaoReisStandardization} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUAssuncaoReisStandardization({
  events: deaths,
  populations,
  output: {standardizedRates}
}));
// feed standardizedRates to GPULocalMoran as values`,
    preview: {
      kind: 'hotspots',
      caption: 'Noisy small-area rates become comparable.',
      seed: 514
    },
    related: ['empirical-bayes-rates', 'local-morans-i'],
    lensScenario: 'gpu-spatial-analysis.assuncao-reis-standardization',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'empirical-bayes-rates',
    symbol: 'GPUEmpiricalBayesRates',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Empirical Bayes rates',
    summary:
      'Shrinks unstable small-population rates toward the global rate and reports excess risk.',
    explanation:
      'Maps rates without the noise of small denominators. The global prior pulls extreme rates in small places the most.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize', 'model'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {
        name: 'events',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Event counts per row.'
      },
      {
        name: 'populations',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Population at risk per row.'
      },
      {
        name: 'clampPriorVariance',
        type: 'boolean | undefined',
        default: 'true',
        description: 'Clamp a negative prior variance to zero.'
      },
      maskParameter,
      {
        name: 'output',
        type: 'GPUEmpiricalBayesRatesOutput',
        required: true,
        description: 'Any of the optional rate columns.'
      }
    ],
    outputs: [
      {
        name: 'smoothedRates',
        type: 'GraphDataView<"float32">',
        description: 'Empirical-Bayes smoothed rate per row.'
      },
      {name: 'rawRates', type: 'GraphDataView<"float32">', description: 'Raw rate e / b per row.'},
      {
        name: 'excessRisk',
        type: 'GraphDataView<"float32">',
        description: 'Observed over expected risk.'
      }
    ],
    example: `import {GPUEmpiricalBayesRates} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUEmpiricalBayesRates({
  events: cases,
  populations,
  output: {smoothedRates, rawRates, excessRisk}
}));`,
    preview: {
      kind: 'polygons',
      caption: 'Raw rates pulled toward the global mean.',
      seed: 515
    },
    related: ['spatial-empirical-bayes-rates', 'assuncao-reis-standardization'],
    lensScenario: 'gpu-spatial-analysis.empirical-bayes-rates',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'spatial-empirical-bayes-rates',
    symbol: 'GPUSpatialEmpiricalBayesRates',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial Empirical Bayes rates',
    summary:
      'Smooths each rate toward the pooled rate of its neighbors rather than the global rate.',
    explanation:
      'Best when risk varies regionally and a global prior would blur real gradients. It matches esda Spatial_Rate and Spatial_Empirical_Bayes.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['summarize', 'model'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {
        name: 'events',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Event counts per row.'
      },
      {
        name: 'populations',
        type: 'GPUFloat32OrUint32Values',
        required: true,
        description: 'Population at risk per row.'
      },
      weightsParameter,
      maskParameter,
      {
        name: 'output',
        type: 'GPUSpatialEmpiricalBayesRatesOutput',
        required: true,
        description: 'Any of the optional rate columns.'
      }
    ],
    outputs: [
      {
        name: 'spatialRates',
        type: 'GraphDataView<"float32">',
        description: 'Neighborhood-pooled rate per row.'
      },
      {
        name: 'smoothedRates',
        type: 'GraphDataView<"float32">',
        description: 'Spatial empirical-Bayes rate per row.'
      },
      {
        name: 'priorMeans',
        type: 'GraphDataView<"float32">',
        description: 'Local prior mean per row.'
      }
    ],
    example: `import {GPUSpatialEmpiricalBayesRates} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSpatialEmpiricalBayesRates({
  events: cases,
  populations,
  weights,
  output: {spatialRates, smoothedRates}
}));`,
    preview: {
      kind: 'polygons',
      caption: 'Smoothed toward neighbors, not the whole map.',
      seed: 516
    },
    related: ['empirical-bayes-rates', 'assuncao-reis-standardization'],
    lensScenario: 'gpu-spatial-analysis.spatial-empirical-bayes-rates',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- space-time
  {
    id: 'emerging-hot-spots',
    symbol: 'GPUEmergingHotSpots',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Emerging hot spots',
    summary:
      'Classifies each cell of a space-time cube as new, intensifying, persistent, diminishing or sporadic.',
    explanation:
      'Shows how hot spots change over time on a regular lattice. It runs Gi* on every time slice and a Mann-Kendall trend on each cell series, as in the ArcGIS tool.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['find-patterns', 'detect-change'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"uint32"> | GraphDataView<"float32">',
        required: true,
        description: 'Dense cube, slice after slice.'
      },
      {
        name: 'gridWidth',
        type: 'number | undefined',
        description: 'Lattice width in cells (lattice mode).'
      },
      {
        name: 'gridHeight',
        type: 'number | undefined',
        description: 'Lattice height in cells (lattice mode).'
      },
      {
        name: 'sliceCount',
        type: 'number',
        required: true,
        description: 'Number of time slices in the cube.'
      },
      {
        name: 'maximumRadius',
        type: 'number | undefined',
        description: 'Compile-time cap on the neighborhood radius in cells.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Radius, temporal window and confidence level.'
      },
      {
        ...multipleTestingParameter,
        default: "'benjamini-hochberg'",
        description: 'Benjamini-Hochberg by default; or Bonferroni, Holm, Yekutieli, none.'
      },
      {
        name: 'output',
        type: 'GPUEmergingHotSpotsOutput',
        required: true,
        description: 'Per-bin Gi* z and per-cell trend and category.'
      }
    ],
    outputs: [
      {
        name: 'giZScores',
        type: 'GraphDataView<"float32">',
        description: 'Gi* z per space-time bin.'
      },
      {name: 'trendZ', type: 'GraphDataView<"float32">', description: 'Mann-Kendall z per cell.'},
      {
        name: 'trendP',
        type: 'GraphDataView<"float32">',
        description: 'Two-sided Mann-Kendall p-value per cell.'
      },
      {
        name: 'trendS',
        type: 'GraphDataView<"sint32">',
        description: 'Mann-Kendall S statistic per cell.'
      },
      {
        name: 'category',
        type: 'GraphDataView<"uint32">',
        description: 'Category code 0 to 17 (GPU_EMERGING_HOT_SPOTS_CATEGORIES).'
      },
      {
        name: 'hotSliceCount',
        type: 'GraphDataView<"uint32">',
        description: 'Significant hot slices per cell.'
      },
      {
        name: 'coldSliceCount',
        type: 'GraphDataView<"uint32">',
        description: 'Significant cold slices per cell.'
      }
    ],
    example: `import {GPUEmergingHotSpots, getGPUEmergingHotSpotsParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// cube: width x height x 12 months of counts, as a graph view
const values = getGPUEmergingHotSpotsParameterValues({radius: 1, temporalWindow: 1, confidenceLevel: 0.95});
const parameters = new GPUParameterBuffer(device, {
  id: 'emerging', format: 'float32', length: values.length, values
});
graph.add(new GPUEmergingHotSpots({
  values: cube, gridWidth: 64, gridHeight: 64,
  sliceCount: 12, maximumRadius: 2,
  parameters: parameters.importToGraph(graph),
  multipleTesting: 'benjamini-hochberg',
  output: {giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount}
}));`,
    preview: {
      kind: 'time-series',
      caption: 'Hot spots appear, persist and fade across slices.',
      seed: 517
    },
    related: ['hot-spot-analysis-gi-star', 'spatial-scan-statistic'],
    lensScenario: 'gpu-spatial-analysis.emerging-hot-spots',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'knox-test',
    symbol: 'GPUKnoxTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Knox test',
    summary:
      'Tests whether events close in space are also close in time, such as contagious outbreaks.',
    explanation:
      'Best for space-time interaction in event data. It counts pairs close in both space and time and compares the count with permutations of the times.',
    dataTypes: {input: ['points', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'pairs',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Event pairs that are close in space.'
      },
      {
        name: 'times',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Relative event times (see getGPUSpaceTimeRelativeTimes).'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Seed, permutations and time threshold.'
      },
      maximumPermutationsParameter,
      {
        name: 'stopAfterExceedances',
        type: 'number | undefined',
        description: 'Stop early after this many exceedances.'
      },
      {
        name: 'output',
        type: 'GPUKnoxTestOutput',
        required: true,
        description: 'Statistics and summary views.'
      }
    ],
    outputs: [
      {
        name: 'statistics',
        type: 'GraphDataView<"uint32">',
        description: 'Observed Knox count, then one per permutation.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Counts and p-values (GPU_SPACE_TIME_SUMMARY).'
      },
      {
        name: 'permutationCount',
        type: 'GraphDataView<"uint32">',
        description: 'Optional permutations actually used.'
      }
    ],
    example: `import {GPUKnoxTest, GPU_SPACE_TIME_PARAMETER_LENGTH, getGPUSpaceTimeParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const parameters = new GPUParameterBuffer(device, {
  id: 'knox', format: 'uint32', length: GPU_SPACE_TIME_PARAMETER_LENGTH,
  values: getGPUSpaceTimeParameterValues({seed: 1, permutationCount: 999, timeThreshold: 7})
});
// pairs: events within 500 m (radius search); times: days since the first event
graph.add(new GPUKnoxTest({
  pairs, times, parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, output: {statistics, summary}
}));`,
    preview: {kind: 'time-series', caption: 'Pairs near in space and in time.', seed: 518},
    related: ['mantel-test', 'spatial-scan-statistic'],
    lensScenario: 'gpu-spatial-analysis.knox-test',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'mantel-test',
    symbol: 'GPUMantelTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Mantel test',
    summary:
      'Correlates spatial distance with time difference across event pairs, without a fixed threshold.',
    explanation:
      'Best for space-time clustering when you do not want to pick close-in-space and close-in-time cutoffs. It is the Pearson r between d_ij and |t_i - t_j|, with a permutation p-value.',
    dataTypes: {input: ['points', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'pairs',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Event pairs with their distances.'
      },
      {
        name: 'times',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Relative event times.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Seed and permutation count.'
      },
      maximumPermutationsParameter,
      {
        name: 'stopAfterExceedances',
        type: 'number | undefined',
        description: 'Stop early after this many exceedances.'
      },
      {
        name: 'output',
        type: 'GPUMantelTestOutput',
        required: true,
        description: 'Statistics and summary views.'
      }
    ],
    outputs: [
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: 'Observed r, then one per permutation.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'r and p-values (GPU_SPACE_TIME_SUMMARY).'
      },
      {
        name: 'permutationCount',
        type: 'GraphDataView<"uint32">',
        description: 'Optional permutations actually used.'
      }
    ],
    example: `import {GPUMantelTest} from '${IMPORT_PATH}';

${SETUP_NOTE}
// parameters: GPUParameterBuffer as in the Knox example
graph.add(new GPUMantelTest({
  pairs, times,
  parameters: parameters.importToGraph(graph),
  maximumPermutations: 999,
  output: {statistics, summary}
}));`,
    preview: {
      kind: 'regression',
      caption: 'Distance in space against distance in time.',
      seed: 519
    },
    related: ['knox-test', 'spatial-scan-statistic'],
    lensScenario: 'gpu-spatial-analysis.mantel-test',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'spatial-scan-statistic',
    symbol: 'GPUSpatialScanStatistic',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial scan statistic',
    summary:
      "Detects the most likely clusters of cases in space or space-time (Kulldorff's SaTScan).",
    explanation:
      'Best for disease or crime surveillance with a population baseline. It scans circular windows or cylinders for the highest likelihood ratio and tests each by Monte Carlo replicates.',
    dataTypes: {input: ['points', 'cells', 'time-series'], output: ['table']},
    useCases: ['find-patterns', 'detect-change'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      positionsParameter,
      {
        name: 'cases',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Case count per zone.'
      },
      {
        name: 'baseline',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Expected population or at-risk baseline per zone.'
      },
      {
        name: 'model',
        type: 'GPUSpatialScanModel | undefined',
        default: "'poisson'",
        description: "'poisson', 'bernoulli' or 'space-time-permutation'."
      },
      {
        name: 'timeBuckets',
        type: 'number | undefined',
        description: 'Time slices; more than 1 scans cylinders.'
      },
      {
        name: 'maximumPermutations',
        type: 'number',
        required: true,
        description: 'Compile-time cap on Monte Carlo replicates.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Seed, replicate count and window limits.'
      },
      {
        name: 'output',
        type: 'GPUSpatialScanStatisticOutput',
        required: true,
        description: 'Cluster indices and statistics.'
      }
    ],
    outputs: [
      {
        name: 'clusterStatistics',
        type: 'GraphDataView<"float32">',
        description: 'Log likelihood ratio, relative risk and p-value per cluster.'
      },
      {
        name: 'clusterIndices',
        type: 'GraphDataView<"uint32">',
        description: 'Zones and time range of each cluster.'
      },
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: 'Observed statistic then one per permutation.'
      },
      {name: 'summary', type: 'GraphDataView<"uint32">', description: 'Cluster count and status.'}
    ],
    example: `import {GPUSpatialScanStatistic, getGPUSpatialScanStatisticParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const values = getGPUSpatialScanStatisticParameterValues({seed: 7, permutationCount: 999});
const parameters = new GPUParameterBuffer(device, {
  id: 'scan', format: 'uint32', length: values.length, values
});
// positions: zone centroids; cases and baseline: per-zone columns
graph.add(new GPUSpatialScanStatistic({
  positions, cases, baseline, model: 'poisson',
  maximumPermutations: 999,
  parameters: parameters.importToGraph(graph),
  output: {clusterIndices, clusterStatistics, statistics, summary}
}));`,
    preview: {kind: 'point-clusters', caption: 'The most likely cluster, circled.', seed: 520},
    related: ['knox-test', 'emerging-hot-spots', 'dbscan-clustering'],
    lensScenario: 'gpu-spatial-analysis.spatial-scan-statistic',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- clustering
  {
    id: 'dbscan-clustering',
    symbol: 'GPUSpatialClustering',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'DBSCAN clustering',
    summary:
      'Groups nearby points into density-based clusters of any shape and labels the rest as noise.',
    explanation:
      'Best for clusters of unknown number, such as incident hot zones or stop locations. A point is core when it has enough neighbors within epsilon; clusters are connected core points.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      positionsParameter,
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Bounds, epsilon and minimumPoints from the packer.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Neighbor-search lattice size in cells.'
      },
      {
        name: 'sourceIds',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Ids reported as cluster roots.'
      },
      {
        name: 'sumOrder',
        type: '"sorted" | "atomic" | undefined',
        description: 'Centroid sum order: deterministic or fast.'
      },
      {
        name: 'output',
        type: 'GPUSpatialClusteringOutput',
        required: true,
        description: 'Labels plus optional core flags and clusters.'
      }
    ],
    outputs: [
      labelsOutput,
      {name: 'coreFlags', type: 'GraphDataView<"uint32">', description: '1 for core points.'},
      {name: 'clusterCount', type: 'GraphDataView<"uint32">', description: 'Number of clusters k.'}
    ],
    example: `import {GPUSpatialClustering, getGPUSpatialClusteringParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// The demo above: bounds spanning 1 unit, epsilon 2.6% of it, 6 points
const values = getGPUSpatialClusteringParameterValues({bounds, epsilon: 0.026, minimumPoints: 6});
const parameters = new GPUParameterBuffer(device, {
  id: 'dbscan', format: 'float32', length: values.length, values
});
graph.add(new GPUSpatialClustering({
  positions, gridSize: [256, 256],
  parameters: parameters.importToGraph(graph),
  output: {labels, clusterCount}
}));`,
    preview: {
      kind: 'point-clusters',
      caption: 'Density-connected Japanese earthquake epicenters.',
      seed: 521
    },
    related: ['hdbscan-clustering', 'optics-clustering', 'k-means-clustering'],
    lensScenario: 'gpu-spatial-analysis.spatial-clustering',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'hdbscan-clustering',
    symbol: 'GPUHierarchicalDensityClustering',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'HDBSCAN clustering',
    summary:
      'Finds clusters of varying density without choosing a distance, matching scikit-learn HDBSCAN.',
    explanation:
      'Best when clusters have different densities and one epsilon would fail. It keeps the most stable clusters of a mutual-reachability tree.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      positionsParameter,
      {
        name: 'minimumClusterSize',
        type: 'number',
        required: true,
        description: 'Smallest group that counts as a cluster.'
      },
      {
        name: 'minimumSamples',
        type: 'number | undefined',
        description: 'Neighbors for core distance; defaults to minimumClusterSize.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext | undefined',
        description: 'Units and frame of the positions.'
      },
      {
        name: 'output',
        type: 'GPUHierarchicalDensityClusteringOutput',
        required: true,
        description: 'Labels plus optional core distances.'
      }
    ],
    outputs: [
      labelsOutput,
      {
        name: 'coreDistances',
        type: 'GraphDataView<"float32">',
        description: 'Distance to the minimumSamples-th neighbor.'
      },
      {
        name: 'status',
        type: 'Partial<GPUCardinalityPort>',
        description: 'Optional; count is the number of selected clusters.'
      }
    ],
    example: `import {GPUHierarchicalDensityClustering} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUHierarchicalDensityClustering({
  positions,
  minimumClusterSize: 15,
  minimumSamples: 5,
  output: {labels, coreDistances}
}));`,
    preview: {
      kind: 'point-clusters',
      caption: 'Tight and loose clusters found together.',
      seed: 522
    },
    related: ['dbscan-clustering', 'optics-clustering'],
    lensScenario: 'gpu-spatial-analysis.hierarchical-density-clustering',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'optics-clustering',
    symbol: 'GPUOPTICS',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'OPTICS ordering',
    summary:
      'Orders points by density reachability and extracts clusters at one or many distance scales.',
    explanation:
      'Explores density structure before committing to an epsilon. The reachability plot shows valleys as clusters, extracted by DBSCAN cut or xi steepness.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      positionsParameter,
      {
        name: 'minimumSamples',
        type: 'number',
        required: true,
        description: 'Neighbors, self included, for core distance.'
      },
      {
        name: 'maximumDistance',
        type: 'number',
        required: true,
        description: 'Search radius bounding the neighborhood.'
      },
      {
        name: 'extractionMethod',
        type: '"dbscan" | "xi" | undefined',
        default: "'dbscan'",
        description: 'How labels are cut from the reachability plot.'
      },
      {
        name: 'extractionDistance',
        type: 'number | undefined',
        description: 'Cut distance for dbscan extraction.'
      },
      {
        name: 'xi',
        type: 'number | undefined',
        default: '0.05',
        description: 'Minimum steepness for xi extraction.'
      },
      {name: 'minimumClusterSize', type: 'number | undefined', description: 'Smallest xi cluster.'},
      {
        name: 'output',
        type: 'GPUOPTICSOutput',
        required: true,
        description: 'Ordering, reachability and labels.'
      }
    ],
    outputs: [
      {name: 'ordering', type: 'GraphDataView<"uint32">', description: 'Row ids in visit order.'},
      {
        name: 'coreDistances',
        type: 'GraphDataView<"float32">',
        description: 'Core distance per row; infinity for non-core rows.'
      },
      {
        name: 'reachability',
        type: 'GraphDataView<"float32">',
        description: 'Reachability distance per row.'
      },
      {
        name: 'predecessors',
        type: 'GraphDataView<"uint32">',
        description: 'Predecessor row, or 0xffffffff.'
      },
      {
        name: 'labels',
        type: 'GraphDataView<"uint32">',
        description: 'Cluster id per row, or 0xffffffff for noise.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: '[visitedCount, complete, invalidRowCount].'
      }
    ],
    example: `import {GPUOPTICS} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUOPTICS({
  positions,
  minimumSamples: 8,
  maximumDistance: 400,
  extractionMethod: 'xi',
  output: {ordering, coreDistances, reachability, predecessors, labels, status}
}));`,
    preview: {kind: 'point-clusters', caption: 'Reachability valleys are clusters.', seed: 523},
    related: ['dbscan-clustering', 'hdbscan-clustering'],
    lensScenario: 'gpu-spatial-analysis.optics',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'k-means-clustering',
    symbol: 'GPUSpatialKMeans',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'K-means',
    summary:
      'Partitions points into k compact groups with deterministic k-means++ and Lloyd iterations.',
    explanation:
      'Carves points into k service areas or zones when you know k. Seeded k-means++ with several restarts; the best run is kept.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['find-patterns', 'summarize'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      positionsParameter,
      {name: 'k', type: 'number', required: true, description: 'Number of clusters.'},
      {
        name: 'maximumIterations',
        type: 'number',
        required: true,
        description: 'Compile-time cap on Lloyd iterations.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Seed and tolerance from getGPUSpatialKMeansParameterValues.'
      },
      {
        name: 'initialization',
        type: 'GPUSpatialKMeansInitialization | undefined',
        default: "'kmeans++'",
        description: "'kmeans++' or 'first-valid'."
      },
      {
        name: 'restartCount',
        type: 'number | undefined',
        default: '1',
        description: 'Restarts; the lowest inertia wins.'
      },
      {
        name: 'centerCoordinates',
        type: 'boolean | undefined',
        default: 'false',
        description: 'Center coordinates first for precision.'
      },
      {
        name: 'output',
        type: 'GPUSpatialKMeansOutput',
        required: true,
        description: 'Labels, centers and optional diagnostics.'
      }
    ],
    outputs: [
      labelsOutput,
      {
        name: 'centers',
        type: 'GraphDataView<"float32x2">',
        description: 'Cluster centers, k rows.'
      },
      {
        name: 'inertia',
        type: 'GraphDataView<"float32">',
        description: 'Total within-cluster squared distance.'
      }
    ],
    example: `import {GPUSpatialKMeans, GPU_SPATIAL_KMEANS_PARAMETER_LENGTH, getGPUSpatialKMeansParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const parameters = new GPUParameterBuffer(device, {
  id: 'kmeans', format: 'uint32', length: GPU_SPATIAL_KMEANS_PARAMETER_LENGTH,
  values: getGPUSpatialKMeansParameterValues({seed: 5})
});
graph.add(new GPUSpatialKMeans({
  positions, k: 8, maximumIterations: 50, restartCount: 4,
  parameters: parameters.importToGraph(graph),
  output: {labels, centers, inertia}
}));`,
    preview: {
      kind: 'point-clusters',
      caption: 'Five centers among Japanese earthquake epicenters.',
      seed: 524
    },
    related: ['dbscan-clustering', 'hdbscan-clustering'],
    lensScenario: 'gpu-spatial-analysis.spatial-k-means',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- outliers and similarity
  {
    id: 'local-outlier-factor',
    symbol: 'GPULocalOutlierFactor',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local outlier factor',
    summary: 'Scores how much sparser each point is than its neighbors, to flag local outliers.',
    explanation:
      'Finds anomalies relative to their local density, such as a lone sensor in a dense area. A score near 1 is as dense as the neighbors and well above 1 is an outlier.',
    dataTypes: {input: ['points', 'spatial-weights'], output: ['points']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      {
        name: 'neighbors',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'k nearest neighbors with distances.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Threshold and densityFloor from the packer.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext | undefined',
        description: 'Units of densityFloor.'
      },
      {
        name: 'output',
        type: 'GPULocalOutlierFactorOutput',
        required: true,
        description: 'LOF scores plus optional outlier mask.'
      }
    ],
    outputs: [
      {name: 'kDistance', type: 'GraphDataView<"float32">', description: 'k-distance per row.'},
      {
        name: 'localReachabilityDensity',
        type: 'GraphDataView<"float32">',
        description: 'Local reachability density per row.'
      },
      {name: 'lof', type: 'GraphDataView<"float32">', description: 'Local outlier factor per row.'},
      {
        name: 'outlier',
        type: 'GraphDataView<"uint32">',
        description: '1 where lof exceeds the threshold.'
      },
      {
        name: 'outlierCount',
        type: 'GraphDataView<"uint32">',
        description: 'Number of flagged rows.'
      }
    ],
    example: `import {GPULocalOutlierFactor, getGPULocalOutlierFactorParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const values = getGPULocalOutlierFactorParameterValues({threshold: 1.5});
const parameters = new GPUParameterBuffer(device, {
  id: 'lof', format: 'float32', length: values.length, values
});
// neighbors: 20-nearest-neighbor search with distances
graph.add(new GPULocalOutlierFactor({
  neighbors, parameters: parameters.importToGraph(graph),
  output: {kDistance, localReachabilityDensity, lof, outlier, outlierCount}
}));`,
    preview: {kind: 'points', caption: 'Isolated points score high.', seed: 525},
    related: ['dbscan-clustering', 'similar-locations'],
    lensScenario: 'gpu-spatial-analysis.local-outlier-factor',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'similar-locations',
    symbol: 'GPUSimilarLocations',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Similar locations',
    summary: 'Ranks every place by how closely its attributes match a chosen reference place.',
    explanation:
      'Best for site selection and "find places like this one". Attributes are standardized, then ranked by weighted Euclidean distance to the reference rows.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['relate', 'find-patterns'],
    domains: ['urban', 'data-engineering'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Row-major rows x columnCount attributes.'
      },
      {
        name: 'columnCount',
        type: 'number',
        required: true,
        description: 'Number of attributes per row.'
      },
      {
        name: 'referenceMask',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Nonzero marks the reference rows.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Result count, direction and attribute weights.'
      },
      {
        name: 'standardization',
        type: 'GPUSimilarLocationsStandardization | undefined',
        description: "'zscore' or 'rank'."
      },
      {
        name: 'resultCapacity',
        type: 'number | undefined',
        description: 'Compile-time size of topIds.'
      },
      maskParameter,
      {
        name: 'output',
        type: 'GPUSimilarLocationsOutput',
        required: true,
        description: 'Ranks, distances and top ids.'
      }
    ],
    outputs: [
      {
        name: 'ranks',
        type: 'GraphDataView<"uint32">',
        description: 'Similarity rank per row; 0 is most similar.'
      },
      {
        name: 'distances',
        type: 'GraphDataView<"float32">',
        description: 'Weighted distance to the reference.'
      },
      {
        name: 'topIds',
        type: 'GraphDataView<"uint32">',
        description: 'Row ids of the best matches in rank order.'
      },
      {
        name: 'status',
        type: 'Pick<GPUCardinalityPort, "count">',
        description: 'count is the number of valid topIds.'
      }
    ],
    example: `import {GPUSimilarLocations, getGPUSimilarLocationsParameterValues} from '${IMPORT_PATH}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const values = getGPUSimilarLocationsParameterValues({resultCount: 10}, 4);
const parameters = new GPUParameterBuffer(device, {
  id: 'similar', format: 'float32', length: values.length, values
});
graph.add(new GPUSimilarLocations({
  values: attributes, columnCount: 4, referenceMask,
  parameters: parameters.importToGraph(graph), resultCapacity: 10,
  output: {ranks, distances, topIds, status: {count}}
}));`,
    preview: {
      kind: 'points',
      caption: 'Darker means a closer match to the reference.',
      seed: 526
    },
    related: ['local-outlier-factor', 'k-means-clustering'],
    lensScenario: 'gpu-spatial-analysis.similar-locations',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- segregation and interaction
  {
    id: 'segregation-indices',
    symbol: 'GPUSegregation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Segregation indices',
    summary:
      'Computes dissimilarity, entropy and related segregation indices, aspatial or spatial, global or local.',
    explanation:
      'Quantifies how unevenly population groups are spread across tracts, as in the PySAL segregation package. Scales add spatial smoothing so neighbors share exposure.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table', 'polygons']},
    useCases: ['measure', 'summarize'],
    domains: ['urban', 'demographics'],
    params: [
      {
        name: 'groupCount',
        type: 'number',
        required: true,
        description: 'Number of population groups K.'
      },
      {
        name: 'groupCounts',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Row-major units x K group populations.'
      },
      {
        name: 'scales',
        type: 'readonly (GPUSpatialWeights | null)[] | undefined',
        description: 'Spatial scales; null is the aspatial index.'
      },
      {
        name: 'selfWeight',
        type: 'number | undefined',
        description: 'Weight of the unit itself in its environment.'
      },
      {
        name: 'spatialForm',
        type: '"environment" | "smoothed-population" | undefined',
        description: 'How neighbors enter the local environment.'
      },
      {
        name: 'atkinsonInequalityAversion',
        type: 'number | undefined',
        description: 'Atkinson index aversion parameter b.'
      },
      {
        name: 'output',
        type: 'GPUSegregationOutput',
        required: true,
        description: 'Global indices plus optional local columns.'
      }
    ],
    outputs: [
      {
        name: 'indices',
        type: 'GraphDataView<"float32">',
        description: 'Global indices per scale (see getGPUSegregationLayout).'
      },
      {
        name: 'localEntropy',
        type: 'GraphDataView<"float32">',
        description: 'Local entropy of each unit environment, in nats.'
      },
      {
        name: 'localDissimilarity',
        type: 'GraphDataView<"float32">',
        description: 'Per-unit contribution to the dissimilarity index.'
      }
    ],
    example: `import {GPUSegregation} from '${IMPORT_PATH}';

${SETUP_NOTE}
// groupCounts: tracts x 3 groups; scales: aspatial plus a queen-contiguity scale
graph.add(new GPUSegregation({
  groupCount: 3,
  groupCounts,
  scales: [null, queenWeights],
  output: {indices, localEntropy, localDissimilarity}
}));`,
    preview: {
      kind: 'classification',
      caption: 'Local entropy: mixed versus segregated tracts.',
      seed: 527
    },
    related: ['spatial-interaction', 'global-spatial-statistics'],
    lensScenario: 'gpu-spatial-analysis.segregation',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'spatial-interaction',
    symbol: 'GPUSpatialInteraction',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Gravity model',
    summary:
      'Fits a calibrated Poisson gravity model to origin-destination flows with one distance covariate.',
    explanation:
      'Use it to predict flows between places and see which origins or destinations are over or under-served. Production, attraction or both constraints are balanced on the GPU while the distance decay is calibrated.',
    dataTypes: {input: ['table', 'network'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'origins',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Origin index per flow pair.'
      },
      {
        name: 'destinations',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Destination index per flow pair.'
      },
      {
        name: 'flows',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Observed flow per pair.'
      },
      {
        name: 'covariate',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Distance or cost per pair.'
      },
      {
        name: 'constraint',
        type: '"unconstrained" | "origin" | "destination" | "doubly"',
        required: true,
        description: 'Which margins the model preserves.'
      },
      {
        name: 'coefficients',
        type: 'readonly number[]',
        required: true,
        description: 'Candidate distance-decay coefficients.'
      },
      {
        name: 'calibration',
        type: '"grid" | "bounded" | undefined',
        default: "'grid'",
        description: 'Search the candidates or a bounded interval.'
      },
      {
        name: 'output',
        type: '{predictions, summary}',
        required: true,
        description: 'Predictions, summary, status and candidate scores.'
      }
    ],
    moreRequired: ['originCount', 'destinationCount'],
    outputs: [
      {
        name: 'predictions',
        type: 'GraphDataView<"float32">',
        description: 'Modeled flow per pair.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Selected coefficient, deviance and balance error.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: '0 converged; 1 no mass; 2 invalid input; 3 no candidate.'
      },
      {
        name: 'candidates',
        type: 'GraphDataView<"float32">',
        description: 'Per candidate: deviance, marginal error, iterations, code.'
      },
      {
        name: 'originResiduals',
        type: 'GraphDataView<"float32">',
        description: 'Residual per origin.'
      }
    ],
    example: `import {GPUSpatialInteraction} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSpatialInteraction({
  origins, destinations, originCount: 120, destinationCount: 120,
  flows, covariate: distances,
  constraint: 'doubly',
  coefficients: [0.5, 1, 1.5, 2],
  output: {predictions, summary, status, candidates}
}));`,
    preview: {kind: 'regression', caption: 'Observed flows against the gravity fit.', seed: 528},
    related: ['spatial-cross-validated-ols', 'segregation-indices'],
    lensScenario: 'gpu-spatial-analysis.spatial-interaction',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- spatial validation
  {
    id: 'spatial-cross-validation',
    symbol: 'GPUSpatialCrossValidation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial cross-validation',
    summary:
      'Builds block holdout folds with a buffer, so test points are not leaked through spatial autocorrelation.',
    explanation:
      'Best before scoring any spatial model: random folds overstate accuracy when neighbors are alike. Seeded blocks are held out and training points within the buffer distance are dropped.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      positionsParameter,
      {
        name: 'blocks',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Block id per point.'
      },
      {name: 'foldCount', type: 'number', required: true, description: 'Number of folds.'},
      {
        name: 'bufferDistance',
        type: 'number | undefined',
        description: 'Planar distance excluded around each test point.'
      },
      {
        name: 'seed',
        type: 'number | undefined',
        description: 'Seed of the block-to-fold assignment.'
      },
      {
        name: 'design',
        type: 'GPUSpatialValidationDesign | undefined',
        description: "Fold design, for example 'groups' or 'environmental'."
      },
      maskParameter,
      {
        name: 'output',
        type: '{folds, training}',
        required: true,
        description: 'Fold, training and testing masks.'
      }
    ],
    outputs: [
      {name: 'folds', type: 'GraphDataView<"uint32">', description: 'Fold id per point.'},
      {name: 'training', type: 'GraphDataView<"uint32">', description: 'Training mask per fold.'},
      {name: 'testing', type: 'GraphDataView<"uint32">', description: 'Testing mask per fold.'},
      {
        name: 'counts',
        type: 'GraphDataView<"uint32">',
        description: 'Per fold: training, test and buffered-out counts.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Per fold bitmask: 1 empty training, 2 empty test.'
      }
    ],
    example: `import {GPUSpatialCrossValidation} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSpatialCrossValidation({
  positions, blocks,
  foldCount: 5,
  seed: 42,
  bufferDistance: 250,
  output: {folds, training, testing, counts, status}
}));`,
    preview: {kind: 'cells', caption: 'Blocks held out, buffer ring dropped.', seed: 529},
    related: ['spatial-cross-validated-ols', 'nested-interpolation-validation'],
    lensScenario: 'gpu-spatial-analysis.spatial-cross-validation',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'spatial-cross-validated-ols',
    symbol: 'GPUSpatialCrossValidatedOLS',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cross-validated regression',
    summary:
      'Fits OLS on every spatial fold and reports honest out-of-fold predictions and metrics.',
    explanation:
      'Checks how well a regression transfers to new places. Each fold trains on buffered blocks, predicts the held-out block, and everything stays resident on the GPU.',
    dataTypes: {input: ['table', 'points'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'folds',
        type: 'Omit<GPUSpatialCrossValidationProps, "id">',
        required: true,
        description: 'Spatial fold design, as for GPUSpatialCrossValidation.'
      },
      {
        name: 'predictors',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Row-major rows x predictorCount design matrix.'
      },
      {
        name: 'predictorCount',
        type: 'number',
        required: true,
        description: 'Number of predictor columns.'
      },
      {
        name: 'response',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Response value per row.'
      },
      {
        name: 'output',
        type: '{coefficients, predictions}',
        required: true,
        description: 'Coefficients, predictions and metrics.'
      }
    ],
    outputs: [
      {
        name: 'coefficients',
        type: 'GraphDataView<"float32">',
        description: 'Fitted coefficients per fold.'
      },
      {
        name: 'predictions',
        type: 'GraphDataView<"float32">',
        description: 'Out-of-fold prediction per row.'
      },
      {name: 'fitStatus', type: 'GraphDataView<"uint32">', description: 'Per fold fit status.'},
      {name: 'metrics', type: 'GraphDataView<"float32">', description: 'Error metrics per fold.'},
      {
        name: 'metricStatus',
        type: 'GraphDataView<"uint32">',
        description: 'Per fold bitmask: 1 no scored rows, 2 missing outcomes.'
      }
    ],
    example: `import {GPUSpatialCrossValidatedOLS} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSpatialCrossValidatedOLS({
  folds: {positions, blocks, foldCount: 5, bufferDistance: 250, output: foldOutput},
  predictors, predictorCount: 3, response,
  output: {coefficients, predictions, fitStatus, metrics, metricStatus}
}));`,
    preview: {
      kind: 'regression',
      caption: 'Out-of-fold predictions against observations.',
      seed: 530
    },
    related: ['spatial-cross-validation', 'nested-interpolation-validation'],
    lensScenario: 'gpu-spatial-analysis.spatial-cross-validated-ols',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'nested-interpolation-validation',
    symbol: 'GPUNestedInterpolationValidation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Nested interpolation validation',
    summary:
      'Tunes interpolation hyperparameters on inner folds and scores the winner on outer spatial folds.',
    explanation:
      'Chooses a kernel, degree or smoothing without optimistic bias. An inner buffered search picks the best candidate, then the outer test blocks score it; all on the GPU.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['interpolate', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      positionsParameter,
      {
        name: 'response',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Observed value per point.'
      },
      {
        name: 'training',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Outer training masks from spatial cross-validation.'
      },
      {
        name: 'testing',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Outer testing masks.'
      },
      {name: 'foldCount', type: 'number', required: true, description: 'Number of outer folds.'},
      {
        name: 'innerBlocks',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Block ids for the inner folds.'
      },
      {
        name: 'candidates',
        type: 'readonly GPUNestedInterpolationCandidate[]',
        required: true,
        description: 'Kriging covariance or RBF kernel, epsilon and smoothing options.'
      },
      {
        name: 'output',
        type: '{selected, scores}',
        required: true,
        description: 'Selected candidate, scores and metrics.'
      }
    ],
    outputs: [
      {
        name: 'selected',
        type: 'GraphDataView<"uint32">',
        description: 'Winning candidate per outer fold.'
      },
      {name: 'scores', type: 'GraphDataView<"float32">', description: 'Inner score per candidate.'},
      {
        name: 'predictions',
        type: 'GraphDataView<"float32">',
        description: 'Outer predictions per point.'
      },
      {
        name: 'metrics',
        type: 'GraphDataView<"float32">',
        description: 'Per fold: scored, missing, MSE, MAE, bias, R2.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Bitmask: 1 overlapping outer masks, 2 no selected candidate.'
      }
    ],
    example: `import {GPUNestedInterpolationValidation} from '${IMPORT_PATH}';

${SETUP_NOTE}
// training and testing come from GPUSpatialCrossValidation
graph.add(new GPUNestedInterpolationValidation({
  positions, response, training, testing, foldCount: 5,
  innerBlocks, innerFoldCount: 3,
  candidates: [
    {mode: 'rbf', kernel: 'gaussian', epsilon: 0.5},
    {mode: 'rbf', kernel: 'gaussian', epsilon: 1, smoothing: 0.1}
  ],
  output: {selected, scores, predictions, metrics, status}
}));`,
    preview: {kind: 'interpolation', caption: 'Candidates compared on held-out blocks.', seed: 531},
    related: ['spatial-cross-validation', 'spatial-cross-validated-ols'],
    lensScenario: 'gpu-spatial-analysis.nested-interpolation-validation',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  }
];
