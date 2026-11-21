// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "categories-cells-tests": categorical association and entropy, the geographical
// detector, change points, REDCAP regions, false discovery rate masks, Sinkhorn divergence, the
// contiguity-order correlogram and HEALPix cells.

import type {CatalogEntry} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

const GRAPH_IMPORT = `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';`;

export const entries: CatalogEntry[] = [
  {
    id: 'colocation-quotient',
    symbol: 'GPUColocationQuotient',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Colocation quotient',
    summary:
      'Measures whether rows of one category have neighbors of another more often than chance.',
    explanation:
      'Use it for categorical points or areas: above 1 is attraction, 1 is independence, below 1 is avoidance. Each row gets a local quotient and each category pair a global one, from neighbor shares over a weights graph.',
    dataTypes: {input: ['points', 'spatial-weights'], output: ['table']},
    useCases: ['find-patterns', 'relate'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Square weights; weight values are shares of the neighborhood.'
      },
      {
        name: 'categories',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Category per row; 0xffffffff leaves the row out.'
      },
      {
        name: 'categoryCount',
        type: 'number',
        required: true,
        description: 'Number of categories, at most 32.'
      },
      {
        name: 'output',
        type: 'GPUColocationQuotientOutput',
        required: true,
        description: 'Global quotients, optional local quotients and row counts.'
      }
    ],
    outputs: [
      {
        name: 'globalQuotients',
        type: 'GraphDataView<"float32">',
        description: 'categoryCount squared entries: CLQ(a -> b).'
      },
      {
        name: 'localQuotients',
        type: 'GraphDataView<"float32">',
        description: 'Optional rows x categoryCount quotients.'
      },
      {
        name: 'categoryRowCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Optional rows per category.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUColocationQuotient} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'clq'});
// weights: k nearest neighbors as GPUSpatialWeights; categories: era of each earthquake (0-2)
graph.add(
  new GPUColocationQuotient({
    weights,
    categories,
    categoryCount: 3,
    output: {globalQuotients, localQuotients}
  })
);
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {
      kind: 'point-clusters',
      seed: 3101,
      caption: 'Events of one era sit next to events of the same era.'
    },
    related: ['local-category-dependence', 'category-pair-counts', 'neighborhood-summary'],
    reference: 'api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density',
    maturity: 'stable'
  },
  {
    id: 'local-category-dependence',
    symbol: 'GPULocalCategoryDependence',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local category dependence',
    summary:
      'Counts how many neighbors share each row category and tests it against the category share.',
    explanation:
      'Use it to find where a class is locally clustered, as in Boots local indicators for categorical data. Each row reports like neighbors, their proportion, the expected proportion and a binomial upper tail.',
    dataTypes: {input: ['polygons', 'spatial-weights'], output: ['table']},
    useCases: ['find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Square weights; every listed slot is one neighbor.'
      },
      {
        name: 'categories',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Category per row; 0xffffffff leaves the row out.'
      },
      {
        name: 'categoryCount',
        type: 'number',
        required: true,
        description: 'Number of categories, at most 32.'
      },
      {
        name: 'output',
        type: 'GPULocalCategoryDependenceOutput',
        required: true,
        description: 'Statistics per row and optional category row counts.'
      }
    ],
    outputs: [
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: 'Six columns per row, see GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.'
      },
      {
        name: 'categoryRowCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Optional rows per category.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {
  GPULocalCategoryDependence,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'licd'});
// weights: rook contiguity of tracts; categories: poverty tercile of each tract (0-2)
graph.add(
  new GPULocalCategoryDependence({
    weights,
    categories,
    categoryCount: 3,
    output: {statistics} // rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length floats
  })
);
// column upperTailProbability: small where a tract has more like neighbors than chance`,
    preview: {
      kind: 'classification',
      seed: 3102,
      caption: 'Tracts whose neighbors share their class stand out.'
    },
    related: ['colocation-quotient', 'category-pair-counts', 'local-morans-i'],
    reference: 'api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density',
    maturity: 'stable'
  },
  {
    id: 'spatial-entropy',
    symbol: 'GPUSpatialEntropy',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial entropy',
    summary:
      'Summarizes how much of a categorical map entropy is spatial, as mutual information at links.',
    explanation:
      'Use it to compare maps: Shannon entropy ignores arrangement, while the mutual information between the two ends of a link is large when like classes sit together. It follows Leibovici and Altieri, Cocchi and Roli.',
    dataTypes: {input: ['polygons', 'spatial-weights'], output: ['table']},
    useCases: ['measure', 'find-patterns'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Square weights; each listed slot is one ordered link.'
      },
      {
        name: 'categories',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Category per row; 0xffffffff for none.'
      },
      {
        name: 'categoryCount',
        type: 'number',
        required: true,
        description: 'Number of categories, at most 32.'
      },
      {
        name: 'output',
        type: 'GPUSpatialEntropyOutput',
        required: true,
        description: 'Summary words and optional ordered pair counts.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Entropies in nats, see GPU_SPATIAL_ENTROPY_LAYOUT.'
      },
      {
        name: 'pairCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Optional ordered link counts, categoryCount squared.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUSpatialEntropy, GPU_SPATIAL_ENTROPY_LAYOUT} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'entropy'});
// summary: GraphDataView<'float32'> with GPU_SPATIAL_ENTROPY_LAYOUT.length (8) words
graph.add(
  new GPUSpatialEntropy({
    weights, // rook contiguity of tracts
    categories, // poverty class of each tract
    categoryCount: 3,
    output: {summary}
  })
);
// After readback: summary[GPU_SPATIAL_ENTROPY_LAYOUT.mutualInformation]
// is near 0 for a spatially random map`,
    preview: {
      kind: 'classification',
      seed: 3103,
      caption: 'A sorted map keeps its mutual information.'
    },
    related: ['category-pair-counts', 'colocation-quotient', 'global-spatial-statistics'],
    reference: 'api/gpu-spatial-analysis/geographical-detector-redcap-entropy',
    maturity: 'stable'
  },
  {
    id: 'category-pair-counts',
    symbol: 'GPUCategoryPairCounts',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Category pair counts',
    summary: 'Counts links between every ordered pair of categories and the rows in each category.',
    explanation:
      'The join-count table of a categorical map, and the shared input of spatial entropy and the colocation quotient. Counts are exact integers from atomics, so they do not depend on thread order.',
    dataTypes: {input: ['polygons', 'spatial-weights'], output: ['table']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Square weights; every listed slot is one ordered link.'
      },
      {
        name: 'categories',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Category per row; 0xffffffff leaves the row out.'
      },
      {
        name: 'categoryCount',
        type: 'number',
        required: true,
        description: 'Number of categories, at most 32.'
      },
      {
        name: 'output',
        type: 'GPUCategoryPairCountsOutput',
        required: true,
        description: 'Pair counts and rows per category.'
      }
    ],
    outputs: [
      {
        name: 'pairCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Entry a * categoryCount + b counts links from a to b.'
      },
      {
        name: 'categoryRowCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Rows per category.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUCategoryPairCounts} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'pairs'});
// pairCounts: 9 uint32 for 3 categories; rowCounts: 3 uint32
graph.add(
  new GPUCategoryPairCounts({
    weights, // rook contiguity of tracts
    categories, // poverty class of each tract (0-2)
    categoryCount: 3,
    output: {pairCounts, categoryRowCounts: rowCounts}
  })
);
// pairCounts[a * 3 + b]: links from a tract of class a to a tract of class b`,
    preview: {
      kind: 'classification',
      seed: 3104,
      caption: 'Links counted by the classes of their ends.'
    },
    related: ['spatial-entropy', 'colocation-quotient', 'local-category-dependence'],
    reference: 'api/gpu-spatial-analysis/geographical-detector-redcap-entropy',
    maturity: 'stable'
  },
  {
    id: 'geographical-detector',
    symbol: 'GPUGeographicalDetector',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geographical detector',
    summary:
      'Computes the q statistic: the share of a variable variance explained by a stratification.',
    explanation:
      'Use it to ask whether zones, classes or districts explain a variable, after Wang et al. (2010). The summary has q, an F statistic and its non-centrality; pass it to getGeographicalDetectorPValue for a p-value.',
    dataTypes: {input: ['table', 'polygons'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Response variable Y, one value per row.'
      },
      {
        name: 'labels',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Stratum per row; 0xffffffff leaves the row out.'
      },
      {
        name: 'labelCapacity',
        type: 'number',
        description: 'Distinct label values [0, labelCapacity); defaults to the row count.'
      },
      {
        name: 'output',
        type: 'GPUGeographicalDetectorOutput',
        required: true,
        description: 'The summary words.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'q, F, non-centrality and counts, see GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUGeographicalDetector, GPU_GEOGRAPHICAL_DETECTOR_LAYOUT} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'detector'});
// values: poverty share per tract; labels: zone id of each tract (0-3)
graph.add(
  new GPUGeographicalDetector({
    values,
    labels,
    labelCapacity: 4,
    output: {summary} // GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length words
  })
);
// After readback: summary[GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.q], then
// getGeographicalDetectorPValue(summary)`,
    preview: {
      kind: 'classification',
      seed: 3105,
      caption: 'Zones that explain the variable have a high q.'
    },
    related: ['region-partition-evaluation', 'spatial-pearson', 'segregation-indices'],
    reference: 'api/gpu-spatial-analysis/geographical-detector-redcap-entropy',
    maturity: 'stable'
  },
  {
    id: 'pettitt-test',
    symbol: 'GPUPettittTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Pettitt test',
    summary:
      'Finds the most likely change point of every cell series with a rank-based test and p-value.',
    explanation:
      'Use it to date a single shift in level in each location of a space-time cube without assuming a distribution. It returns the statistic K, the first slice after the change and an asymptotic p-value.',
    dataTypes: {input: ['time-series', 'raster'], output: ['table', 'raster']},
    useCases: ['detect-change', 'model'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'cellCount * sliceCount values; NaN is skipped.'
      },
      {
        name: 'layout',
        type: 'GPUSpaceTimeWeightsLayout',
        required: true,
        description: '"cell-major" or "period-major".'
      },
      {name: 'cellCount', type: 'number', required: true, description: 'Compile-time cells.'},
      {
        name: 'sliceCount',
        type: 'number',
        required: true,
        description: 'Compile-time slices, at most 512.'
      },
      {
        name: 'output',
        type: 'GPUPettittTestOutput',
        required: true,
        description: 'Statistic, change slice and p-value per cell.'
      }
    ],
    outputs: [
      {name: 'statistic', type: 'GraphDataView<"uint32">', description: 'Pettitt K per cell.'},
      {
        name: 'changeSlice',
        type: 'GraphDataView<"uint32">',
        description: 'First slice after the most likely change.'
      },
      {
        name: 'pValues',
        type: 'GraphDataView<"float32">',
        description: 'Asymptotic two-sided p-value per cell.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUPettittTest} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'pettitt'});
// values: 1440 cells x 16 scenes of NDVI, cell-major
graph.add(
  new GPUPettittTest({
    values,
    layout: 'cell-major',
    cellCount: 1440,
    sliceCount: 16,
    output: {statistic, changeSlice, pValues}
  })
);
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {kind: 'time-series', seed: 3106, caption: 'Each cell gets the date of its change.'},
    related: ['location-change-points', 'seasonal-break-detection', 'change-detection'],
    reference: 'api/gpu-spatial-analysis/change-points-and-expectation-based-scan',
    maturity: 'stable'
  },
  {
    id: 'location-change-points',
    symbol: 'GPULocationChangePoints',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Location change points',
    summary:
      'Finds several change points in every cell series by optimal partitioning with a penalty.',
    explanation:
      'Use it when a series changes more than once. A dynamic program minimizes segment cost plus a penalty per change point, as PELT does. Mean, variance, mean-variance, Poisson and slope costs are available.',
    dataTypes: {input: ['time-series', 'raster'], output: ['table', 'raster']},
    useCases: ['detect-change'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'cellCount * sliceCount values; NaN marks a missing slice.'
      },
      {
        name: 'layout',
        type: 'GPUSpaceTimeWeightsLayout',
        required: true,
        description: '"cell-major" or "period-major".'
      },
      {name: 'cellCount', type: 'number', required: true, description: 'Compile-time cells.'},
      {
        name: 'sliceCount',
        type: 'number',
        required: true,
        description: 'Compile-time slices, at most 512.'
      },
      {
        name: 'cost',
        type: 'GPULocationChangePointCost',
        required: true,
        description: 'Compile-time segment cost.'
      },
      {
        name: 'maximumChangePoints',
        type: 'number',
        required: true,
        description: 'Compile-time capacity per cell, at most 128.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame penalty and minimum segment length.'
      },
      {
        name: 'output',
        type: 'GPULocationChangePointsOutput',
        required: true,
        description: 'Change point counts and slices per cell.'
      }
    ],
    outputs: [
      {
        name: 'changePointCount',
        type: 'GraphDataView<"uint32">',
        description: 'Change points per cell, clamped to the capacity.'
      },
      {
        name: 'changePoints',
        type: 'GraphDataView<"uint32">',
        description: 'First slice of each later segment; unused rows are 0xffffffff.'
      },
      {
        name: 'optimalCost',
        type: 'GraphDataView<"float32">',
        description: 'Optional penalized optimum per cell.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPULocationChangePoints} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'change-points'});
const detector = new GPULocationChangePoints({
  values, layout: 'cell-major', cellCount: 1440, sliceCount: 16,
  cost: 'mean', maximumChangePoints: 8,
  parameters, // float32 view, 2 values
  output: {changePointCount, changePoints}
});
graph.add(detector);
// Write [penalty, minimumSegmentLength] into the parameters buffer
parametersBuffer.write(detector.getParameterValues({penalty: 0.2, minimumSegmentLength: 2}));`,
    preview: {
      kind: 'time-series',
      seed: 3107,
      caption: 'Series cut into segments of steady level.'
    },
    related: ['pettitt-test', 'seasonal-break-detection', 'change-detection'],
    reference: 'api/gpu-spatial-analysis/change-points-and-expectation-based-scan',
    maturity: 'stable'
  },
  {
    id: 'redcap-single-linkage',
    symbol: 'GPURedcapSingleLinkage',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'REDCAP single linkage',
    summary: 'Builds contiguous regions by cutting the longest edges of a minimum spanning forest.',
    explanation:
      'Use it for regionalization: merge neighboring areas with similar attributes into the number of regions you ask for. The forest links contiguous rows by attribute distance and the cut sets regionCount.',
    dataTypes: {input: ['polygons', 'spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'classify'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Symmetric contiguity weights.'
      },
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Attribute table, row-major.'
      },
      {
        name: 'columnCount',
        type: 'number',
        default: '1',
        description: 'Attribute columns, at most 32.'
      },
      {
        name: 'standardize',
        type: 'boolean',
        default: 'true',
        description: 'Scale each column to mean 0 and unit deviation.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Per-frame regionCount.'
      },
      {
        name: 'componentIterations',
        type: 'number',
        default: '32',
        description: 'Label propagation iterations.'
      },
      {
        name: 'output',
        type: 'GPURedcapSingleLinkageOutput',
        required: true,
        description: 'Region labels, count and status.'
      }
    ],
    outputs: [
      {
        name: 'labels',
        type: 'GraphDataView<"uint32">',
        description: 'Region label per row: the smallest row index of the region.'
      },
      {
        name: 'regionCount',
        type: 'GraphDataView<"uint32">',
        description: 'Optional region count after the cuts.'
      },
      {
        name: 'status',
        type: 'GPUIterationStatusPort',
        description: 'Optional converged and invalid counts.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {
  getGPURedcapSingleLinkageParameterValues,
  GPURedcapSingleLinkage
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'redcap'});
// attributes: poverty, unemployment and 65+ share per tract, 3 columns
graph.add(
  new GPURedcapSingleLinkage({
    weights, // symmetric rook contiguity
    values: attributes,
    columnCount: 3,
    parameters,
    output: {labels, regionCount}
  })
);
parametersBuffer.write(getGPURedcapSingleLinkageParameterValues({regionCount: 8}));`,
    preview: {kind: 'classification', seed: 3108, caption: 'Contiguous regions of similar tracts.'},
    related: ['skater-regions', 'ward-regions', 'spatial-weights-minimum-spanning-tree'],
    reference: 'api/gpu-spatial-analysis/geographical-detector-redcap-entropy',
    maturity: 'stable'
  },
  {
    id: 'raster-false-discovery-rate-mask',
    symbol: 'GPURasterFalseDiscoveryRateMask',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster false discovery rate mask',
    summary:
      'Turns a raster of p-values into a significance mask that controls the false discovery rate.',
    explanation:
      'Use it after a per-cell test, so many tests do not make noise look like signal. Benjamini-Hochberg is the default and Benjamini-Yekutieli suits dependent cells. It reports the critical p-value and a p-value histogram.',
    dataTypes: {input: ['raster'], output: ['raster', 'table']},
    useCases: ['model', 'find-patterns'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      {
        name: 'cellCount',
        type: 'number',
        required: true,
        description: 'Number of cells of the p-value raster.'
      },
      {
        name: 'pValues',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'p-values; values outside [0, 1] are untested.'
      },
      {
        name: 'validity',
        type: 'GraphDataView<"uint32">',
        description: 'Per-cell QA; zero makes the cell untested.'
      },
      {
        name: 'method',
        type: 'GPURasterFalseDiscoveryRateMethod',
        default: "'benjamini-hochberg'",
        description: 'Correction method.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame alpha.'
      },
      {
        name: 'output',
        type: 'GPURasterFalseDiscoveryRateMaskOutput',
        required: true,
        description: 'Mask and optional adjusted p-values and summary.'
      }
    ],
    outputs: [
      {
        name: 'significant',
        type: 'GraphDataView<"uint32">',
        description: '1 where the adjusted p-value is at most alpha.'
      },
      {
        name: 'adjustedPValues',
        type: 'GraphDataView<"float32">',
        description: 'Optional adjusted p-value per cell.'
      },
      {
        name: 'criticalPValue',
        type: 'GraphDataView<"float32">',
        description: 'Optional largest significant raw p-value.'
      },
      {
        name: 'histogram',
        type: 'GraphDataView<"uint32">',
        description: 'Optional counts of tested p-values per bin.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {
  getGPURasterFalseDiscoveryRateMaskParameterValues,
  GPURasterFalseDiscoveryRateMask
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'fdr'});
graph.add(
  new GPURasterFalseDiscoveryRateMask({
    cellCount: 1440,
    pValues, // e.g. the pValues output of GPUPettittTest
    method: 'benjamini-hochberg',
    parameters,
    output: {significant, criticalPValue, histogram}
  })
);
parametersBuffer.write(getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05}));`,
    preview: {
      kind: 'raster',
      seed: 3109,
      caption: 'Only cells that survive the correction stay lit.'
    },
    related: ['multiple-testing-adjustment', 'local-permutation-test', 'change-detection'],
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'sinkhorn-divergence',
    symbol: 'GPUSinkhornDivergence',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Sinkhorn divergence',
    summary:
      'Measures how far mass must move between two grid fields, as a debiased transport cost.',
    explanation:
      'Use it to compare density maps or a forecast and an observation without the double penalty of cell by cell error. It is zero for equal fields and grows with distance, in squared position units.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['measure', 'detect-change'],
    domains: ['spatial-statistics', 'remote-sensing'],
    params: [
      {
        name: 'grid',
        type: 'GPURasterMetadata',
        required: true,
        description: 'Only its width and height are used.'
      },
      {
        name: 'source',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Source masses in raster row order.'
      },
      {
        name: 'target',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Target masses in raster row order.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame cell size, epsilon and tolerance.'
      },
      {
        name: 'maximumIterations',
        type: 'number',
        default: '100',
        description: 'Unrolled Sinkhorn iterations, at most 1000.'
      },
      {
        name: 'transportRadius',
        type: 'number',
        description: 'Radius in cells of each 1D blur window.'
      },
      {
        name: 'output',
        type: 'GPUSinkhornDivergenceOutput',
        required: true,
        description: 'Divergence and optional components and status.'
      }
    ],
    outputs: [
      {
        name: 'divergence',
        type: 'GraphDataView<"float32">',
        description: 'The debiased divergence S.'
      },
      {
        name: 'components',
        type: 'GraphDataView<"float32">',
        description: 'Optional OT(a, b), OT(a, a) and OT(b, b).'
      },
      {
        name: 'marginalError',
        type: 'GraphDataView<"float32">',
        description: 'Optional L1 marginal error of the last iteration.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {
  getGPUSinkhornDivergenceParameterValues,
  GPUSinkhornDivergence
} from '${IMPORT_PATH}';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'sinkhorn'});
graph.add(
  new GPUSinkhornDivergence({
    grid: makeGPURasterMetadata({width: 32, height: 18}),
    source, target, parameters, // masses per cell; parameters has 6 values
    maximumIterations: 40,
    output: {divergence, components}
  })
);
parametersBuffer.write(
  getGPUSinkhornDivergenceParameterValues({cellWidth: 1, cellHeight: 1, epsilon: 12})
);`,
    preview: {
      kind: 'raster',
      seed: 3110,
      caption: 'The farther the mass moves, the larger the divergence.'
    },
    related: ['distance-field', 'change-detection', 'spatial-pearson'],
    reference: 'api/gpu-spatial-analysis/sinkhorn-divergence',
    maturity: 'stable'
  },
  {
    id: 'contiguity-order-correlogram-recipe',
    symbol: 'addContiguityOrderCorrelogramRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Contiguity order correlogram',
    summary: 'Computes global Moran I for neighbors at exactly 1, 2, ... k contiguity steps.',
    explanation:
      'Use it to see how far autocorrelation reaches. Each order uses the ring of neighbors at exactly that many steps, row-standardized, with analytic inference. Islands at an order are left out of that order.',
    dataTypes: {input: ['polygons', 'spatial-weights'], output: ['table']},
    useCases: ['find-patterns', 'measure'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Contiguity or k nearest neighbor weights; only the pattern is used.'
      },
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Values, one per row.'
      },
      {
        name: 'mask',
        type: 'GraphDataView<"uint32">',
        description: 'Optional row selection.'
      },
      {
        name: 'maximumOrder',
        type: 'number',
        required: true,
        description: 'Largest contiguity order to evaluate.'
      },
      {name: 'slotCapacity', type: 'number', description: 'Slot capacity of each order ring.'},
      {
        name: 'output',
        type: 'Partial<GPUContiguityOrderCorrelogramRecipeOutputs>',
        description: 'Caller-owned outputs.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: 'GPUContiguityOrderCorrelogramRecipeOutputs',
        description: 'correlogram (maximumOrder rows of statistics) and incompleteOrders.'
      },
      {
        name: 'contributors',
        type: 'readonly GPUCommandNodeProducer[]',
        description: 'The operations the recipe added.'
      },
      {
        name: 'intermediates',
        type: 'GPURecipeIntermediates',
        description: 'The ring weights of each order.'
      },
      {name: 'status', type: 'GPURecipeStatus', description: 'Per-stage overflow status.'}
    ],
    example: `${GRAPH_IMPORT}
import {
  addContiguityOrderCorrelogramRecipe,
  GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'correlogram'});
// correlogram: 5 * GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS.length floats
addContiguityOrderCorrelogramRecipe(graph, {
  weights, // rook contiguity of tracts
  values, // poverty share per tract
  maximumOrder: 5,
  output: {correlogram, incompleteOrders}
});
// Moran's I of order k: correlogram[(k - 1) * columns.length + columns.statistic]`,
    preview: {
      kind: 'spatial-weights',
      seed: 3111,
      caption: 'Autocorrelation fades with neighbor order.'
    },
    related: ['spatial-correlogram', 'global-spatial-statistics', 'spatial-weights-algebra'],
    reference: 'api/gpu-spatial-analysis/geographical-detector-redcap-entropy',
    maturity: 'stable'
  },
  {
    id: 'healpix-point-to-cell',
    symbol: 'GPUHealpixPointToCell',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'HEALPix point to cell',
    summary:
      'Converts longitude/latitude points to HEALPix nested or UNIQ cell keys, one key per row.',
    explanation:
      'HEALPix cells have equal area, which suits global statistics. Keys are uint32x2 words like GPUPointToCell. The projection is spherical and float32, so a point near a cell edge may land in the neighboring cell.',
    dataTypes: {input: ['points'], output: ['cells']},
    useCases: ['index'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {
        name: 'order',
        type: 'number',
        required: true,
        description: 'HEALPix order 0 to 29; 12 * 4^order cells.'
      },
      {
        name: 'scheme',
        type: 'GPUHealpixIndexScheme',
        default: "'nested'",
        description: "'nested' or 'uniq' keys."
      },
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Longitude/latitude in degrees.'
      },
      {
        name: 'mask',
        type: 'GraphDataView<"uint32">',
        description: 'Per-row mask; zero produces a zero key.'
      },
      {
        name: 'output',
        type: 'GPUHealpixPointToCellOutput',
        required: true,
        description: 'Destination views for cells and optional validity.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'One 64-bit key per row; zero when masked (a real cell in nested).'
      },
      {
        name: 'validity',
        type: 'GraphDataView<"uint32">',
        description: 'Optional 1 for rows that received a key.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUHealpixPointToCell} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'healpix-keys'});
const positions = graph.importGPUData('positions', lonLatData); // GPUData<'float32x2'>
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>, caller-owned

graph.add(new GPUHealpixPointToCell({order: 3, positions, output: {cells}}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {kind: 'cells', seed: 3112, caption: 'Each point falls into one equal-area cell.'},
    related: ['point-to-cell', 'healpix-cell-aggregation', 'cell-geometry'],
    reference: 'api/gpu-spatial-analysis/healpix-cells-dggs-json',
    maturity: 'stable'
  },
  {
    id: 'healpix-cell-aggregation',
    symbol: 'GPUHealpixCellAggregation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'HEALPix cell aggregation',
    summary: 'Groups keyed points into HEALPix cells with row counts, sums, minima and maxima.',
    explanation:
      'The step after GPUHealpixPointToCell. Distinct cells come out in key order, which is also the order along the HEALPix curve. Sums add in row order inside each cell, so results are bitwise reproducible.',
    dataTypes: {input: ['cells', 'points'], output: ['table', 'cells']},
    useCases: ['summarize', 'index'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {name: 'order', type: 'number', required: true, description: 'HEALPix order of the keys.'},
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'One 64-bit key per row.'
      },
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        description: 'Optional values to sum and take extremes of.'
      },
      {
        name: 'mask',
        type: 'GraphDataView<"uint32">',
        description: 'Optional per-row selection; nonzero keeps the row.'
      },
      {
        name: 'scheme',
        type: 'GPUHealpixIndexScheme',
        default: "'nested'",
        description: 'Key scheme, as written by GPUHealpixPointToCell.'
      },
      {
        name: 'cellCapacity',
        type: 'number',
        required: true,
        description: 'Number of distinct cells the outputs can hold.'
      },
      {
        name: 'output',
        type: 'GPUHealpixCellAggregationOutput',
        required: true,
        description: 'Cell table columns and status scalars.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Distinct cell keys in ascending order.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Rows per cell.'},
      {
        name: 'sums',
        type: 'GraphDataView<"float32">',
        description: 'Optional per-cell sum of values.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Cell count, required count and overflow flag.'
      }
    ],
    example: `${GRAPH_IMPORT}
import {GPUHealpixCellAggregation, GPUHealpixPointToCell} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'healpix-table'});
graph.add(new GPUHealpixPointToCell({order: 3, positions, output: {cells: keys}}));
graph.add(
  new GPUHealpixCellAggregation({
    order: 3,
    cells: keys,
    values, // optional float32 column
    cellCapacity: 768,
    output: {cells, counts, sums, status: {count, requiredCount, overflow}}
  })
);
// cells[0 .. count) are the occupied cells; counts[i] their rows`,
    preview: {kind: 'cells', seed: 3113, caption: 'Points counted per equal-area cell.'},
    related: ['healpix-point-to-cell', 'cell-aggregation', 'healpix-dggs-json'],
    reference: 'api/gpu-spatial-analysis/healpix-cells-dggs-json',
    maturity: 'stable'
  },
  {
    id: 'healpix-dggs-json',
    symbol: 'createHealpixDGGSJSON',
    kind: 'function',
    importPath: IMPORT_PATH,
    title: 'HEALPix DGGS-JSON',
    summary: 'Packs read-back HEALPix cell values into an OGC DGGS-JSON zone data document.',
    explanation:
      'Use it to hand aggregated cells to a DGGS client. The document holds the cells below one ancestral zone as a dense array of sub-zones, with null for missing cells. Zone ids are nested indices, not OGC zone identifiers.',
    dataTypes: {input: ['cells', 'table'], output: ['table']},
    useCases: ['index', 'summarize'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {name: 'order', type: 'number', required: true, description: 'HEALPix order of the cells.'},
      {
        name: 'scheme',
        type: 'GPUHealpixIndexScheme',
        default: "'nested'",
        description: 'Key scheme of the cells.'
      },
      {
        name: 'ancestorOrder',
        type: 'number',
        default: '0',
        description: 'Order of the ancestral zone, at most order.'
      },
      {
        name: 'ancestor',
        type: 'bigint',
        default: '0n',
        description: 'Nested index of the ancestral zone.'
      },
      {
        name: 'cells',
        type: 'readonly bigint[]',
        required: true,
        description: 'Read-back cell keys.'
      },
      {
        name: 'properties',
        type: 'Readonly<Record<string, HealpixDGGSJSONProperty>>',
        required: true,
        description: 'Named properties aligned with cells.'
      }
    ],
    outputs: [
      {name: '$schema', type: 'string', description: 'DGGS-JSON schema URI.'},
      {name: 'dggrs', type: 'string', description: 'DGGRS definition URI.'},
      {name: 'zoneId', type: 'string', description: 'healpix-nested:<ancestorOrder>:<ancestor>.'},
      {name: 'depths', type: 'number[]', description: 'Relative depth of the data.'},
      {
        name: 'schema',
        type: 'HealpixDGGSJSON["schema"]',
        description: 'JSON schema of the properties.'
      },
      {
        name: 'values',
        type: 'HealpixDGGSJSON["values"]',
        description: 'Dense data array per property.'
      }
    ],
    example: `import {createHealpixDGGSJSON} from '${IMPORT_PATH}';

// keys and counts: the first \`count\` rows of GPUHealpixCellAggregation output.cells / counts,
// read back; keys as bigint = (high << 32n) | low
const ancestor = 4n; // base cell 4: cells outside the ancestral zone make the call throw
const inside = keys.flatMap((key, row) => (key >> 6n === ancestor ? [row] : [])); // 2 * (3 - 0) bits
const document = createHealpixDGGSJSON({
  order: 3,
  ancestorOrder: 0,
  ancestor,
  cells: inside.map(row => keys[row]),
  properties: {count: {title: 'coastline vertices', values: inside.map(row => counts[row])}}
});
// document.values.count[0].data has 4^3 entries; null where a cell is empty`,
    preview: {kind: 'cells', seed: 3114, caption: 'Cells below one ancestor, as a dense array.'},
    related: ['healpix-cell-aggregation', 'healpix-point-to-cell', 'cell-rollup'],
    reference: 'api/gpu-spatial-analysis/healpix-cells-dggs-json',
    maturity: 'stable'
  }
];
