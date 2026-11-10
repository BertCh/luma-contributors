// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: tables, point clouds, geospatial buffers and computation records. Table statistics,
// classification, joins and variograms; point clouds; datum and buffer helpers; computation records
// and snapshots. Every example assumes `graph` is a GPUCommandGraph, `device` a luma Device and the
// named views were created on the graph.

import type {CatalogEntry} from '../schema';

export const entries: CatalogEntry[] = [
  {
    id: 'column-profile',
    symbol: 'GPUColumnProfile',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Column Profile',
    summary:
      'One fused pass gives count, nulls, moments, histogram, distinct estimate and top categories per column.',
    explanation:
      'Best for a dataset statistics panel that recomputes every frame as filters change. Up to 16 columns share one mask; distinct counts use HyperLogLog and categories come from dictionary codes.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      {
        name: 'columns',
        type: 'readonly GPUColumnProfileColumn[]',
        required: true,
        description: 'One to 16 numeric or category columns of one table.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows are profiled.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        description: 'Optional per-frame histogram domains; NaN means automatic.'
      },
      {
        name: 'histogramBinCount',
        type: 'number',
        required: true,
        description: 'Equal-width histogram bins per column.'
      },
      {
        name: 'hyperLogLogPrecision',
        type: 'number',
        description: 'Registers per column are 2^p; higher is more exact.'
      },
      {name: 'topCategoryCount', type: 'number', description: 'Top-category slots per column.'}
    ],
    outputs: [
      {
        name: 'statistics',
        type: "GraphDataView<'float32'>",
        description: 'Minimum, maximum, mean, variance and more, per column.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Valid and null counts per column.'
      },
      {
        name: 'histograms',
        type: "GraphDataView<'uint32'>",
        description: 'Histogram bins per column.'
      },
      {
        name: 'topCategories',
        type: "GraphDataView<'uint32'>",
        description: 'Most frequent category codes and their counts.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUColumnProfile} from '@luma-contributors/analysis/dataframe-analysis';

// fare, vendorCodes, mask, statistics, counts, histograms: views on graph
graph.add(new GPUColumnProfile({
  columns: [
    {values: fare},
    {values: vendorCodes, kind: 'category', categoryCount: 500}
  ],
  mask,
  histogramBinCount: 64,
  topCategoryCount: 10,
  output: {statistics, counts, histograms}
}));`,
    preview: {
      kind: 'histogram',
      seed: 9101,
      caption: 'Per-column histogram and summary, recomputed on the GPU.'
    },
    related: ['column-quantiles', 'group-statistics', 'variable-standardization'],
    lensScenario: 'gpu-dataframe.column-profile',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'column-quantiles',
    symbol: 'GPUColumnQuantiles',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Column Quantiles',
    summary:
      'Exact quantiles of a float32 column by radix selection, plus a percentile-range filter mask.',
    explanation:
      'Best for exact medians, quartiles and trimmed ranges that stay on the GPU. There is no sort and no readback, so brushing a percentile range is a parameter write.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize', 'classify'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed float32 column.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Probabilities and filter range from getGPUColumnQuantilesParameterValues.'
      },
      {
        name: 'quantileCount',
        type: 'number',
        required: true,
        description: 'Number of probability slots (1 to 64), compile-time.'
      }
    ],
    outputs: [
      {
        name: 'quantiles',
        type: "GraphDataView<'float32'>",
        description: 'One value per requested probability.'
      },
      {name: 'validCount', type: "GraphDataView<'uint32'>", description: 'Rows that took part.'},
      {
        name: 'outputMask',
        type: "GraphDataView<'uint32'>",
        description: 'Rows inside the percentile filter range.'
      },
      {
        name: 'filterBounds',
        type: "GraphDataView<'float32'>",
        description: 'Lower and upper value of the filter range.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUColumnQuantiles, getGPUColumnQuantilesParameterLength, getGPUColumnQuantilesParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// values, quantiles, validCount, outputMask: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'q', format: 'float32',
  length: getGPUColumnQuantilesParameterLength(3)});
graph.add(new GPUColumnQuantiles({
  values,
  quantileCount: 3,
  parameters: parameters.importToGraph(graph),
  output: {quantiles, validCount, outputMask}
}));
parameters.write(getGPUColumnQuantilesParameterValues({
  quantiles: [0.25, 0.5, 0.75],
  filterRange: [0.05, 0.95]
}));`,
    preview: {
      kind: 'histogram',
      seed: 9102,
      caption: 'Quartiles and a 5 to 95 percent filter band over one column.'
    },
    related: ['class-breaks', 'column-profile'],
    lensScenario: 'gpu-dataframe.column-quantiles',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'class-breaks',
    symbol: 'GPUClassBreaks',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Class Breaks',
    summary:
      'Choropleth class breaks on the GPU every frame: equal interval, quantile, natural breaks and more.',
    explanation:
      'Classifies a column for a map without reading data back. The method, class count and custom edges are per-frame parameters, so switching never recompiles.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify'],
    domains: ['cartography', 'spatial-statistics'],
    params: [
      {
        name: 'values',
        type: "GraphDataView<'float32'> | GraphDataView<'uint32'> | GraphDataView<'sint32'>",
        required: true,
        description: 'Packed column to classify.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Method and class count from getGPUClassBreaksParameterValues.'
      },
      {
        name: 'maximumClassCount',
        type: 'number',
        required: true,
        description: 'Compile-time class capacity (1 to 64).'
      },
      {
        name: 'methods',
        type: 'readonly GPUClassBreaksMethod[]',
        default: 'every method',
        description: 'Methods compiled into the graph.'
      },
      {
        name: 'naturalBreaksMode',
        type: "'exact' | 'monotone' | 'approximate'",
        default: "'approximate'",
        description: 'Natural breaks: approximate, monotone or exact.'
      }
    ],
    outputs: [
      {
        name: 'breaks',
        type: "GraphDataView<'float32'>",
        description: 'Class edges, classCount + 1 of them.'
      },
      {
        name: 'classCount',
        type: "GraphDataView<'uint32'>",
        description: 'Number of classes produced.'
      },
      {name: 'classCounts', type: "GraphDataView<'uint32'>", description: 'Rows per class.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUClassBreaks, getGPUClassBreaksParameterLength, getGPUClassBreaksParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// values, breaks, classCount, classCounts: views on graph
const parameters = new GPUParameterBuffer(device, {
  id: 'breaks', format: 'float32', length: getGPUClassBreaksParameterLength(7)
});
graph.add(new GPUClassBreaks({
  values, maximumClassCount: 7,
  parameters: parameters.importToGraph(graph),
  output: {breaks, classCount, classCounts}
}));
parameters.write(getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 5}, 7));`,
    preview: {
      kind: 'classification',
      seed: 9103,
      caption: 'Five natural-breaks classes over a skewed column.'
    },
    related: ['column-quantiles', 'color-scale', 'bivariate-classification', 'class-assignment'],
    lensScenario: 'gpu-dataframe.class-breaks',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'color-scale',
    symbol: 'GPUColorScale',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Color Scale',
    summary:
      'Maps a column to RGBA with linear, log, threshold or ordinal scales, all chosen per frame.',
    explanation:
      'Colors features from data on the GPU. Scale type, domain and palette are data, so animating a legend never rebuilds the graph.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify'],
    domains: ['cartography'],
    params: [
      {
        name: 'values',
        type: "GraphDataView<'float32'> | GraphDataView<'uint32'> | GraphDataView<'sint32'>",
        required: true,
        description: 'Packed values to color.'
      },
      {
        name: 'domain',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Monotonic scale domain, for example class breaks.'
      },
      {
        name: 'palette',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Packed rgba8 palette.'
      },
      {
        name: 'domainCount',
        type: "GraphDataView<'uint32'>",
        description: 'Optional one-row view with the active class count.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Scale settings from getGPUColorScaleParameterValues.'
      },
      {
        name: 'maximumDomainCount',
        type: 'number',
        required: true,
        description: 'Largest active domain length, compile-time.'
      },
      {
        name: 'maximumPaletteCount',
        type: 'number',
        required: true,
        description: 'Largest active palette length, compile-time.'
      }
    ],
    outputs: [
      {name: 'colors', type: "GraphDataView<'uint32'>", description: 'Packed rgba8 color per row.'},
      {name: 'classIndices', type: "GraphDataView<'uint32'>", description: 'Class index per row.'},
      {name: 'classCounts', type: "GraphDataView<'uint32'>", description: 'Rows per class.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUColorScale, getGPUColorScaleParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// values, breaks, classCount, palette, colors: views on graph
const scale = new GPUParameterBuffer(device, {id: 'scale', format: 'float32', length: 9});
graph.add(new GPUColorScale({
  values, domain: breaks, domainCount: classCount, palette,
  parameters: scale.importToGraph(graph),
  maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors}
}));
scale.write(getGPUColorScaleParameterValues({scale: 'threshold', domainCount: 8, paletteCount: 9}));`,
    preview: {
      kind: 'classification',
      seed: 9104,
      caption: 'A threshold palette applied to a column.'
    },
    related: ['class-breaks', 'scalar-scale', 'bivariate-classification'],
    lensScenario: 'gpu-dataframe.color-scale',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'scalar-scale',
    symbol: 'GPUScalarScale',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Scalar Scale',
    summary:
      'Maps a column to a numeric range, such as radius or width, with source-aligned output and no readback.',
    explanation:
      'Drives size, width or elevation from data. It mirrors native scale semantics (linear, log, point) and keeps invalid rows flagged instead of compacting them.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify'],
    domains: ['cartography'],
    params: [
      {
        name: 'values',
        type: "GraphDataView<'uint32'> | GraphDataView<'float32'>",
        required: true,
        description: 'Scalar float32 values, or uint32 category codes for point scales.'
      },
      {
        name: 'validity',
        type: "GraphDataView<'uint32'>",
        description: 'Source validity; display filtering is separate.'
      },
      {
        name: 'domain',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Endpoint pair, or ordered domain entries for point scales.'
      },
      {
        name: 'range',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Output endpoints.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Scale type and settings from getGPUScalarScaleParameterValues.'
      },
      {
        name: 'maximumDomainCount',
        type: 'number',
        required: true,
        description: 'Domain capacity, 2 to 256.'
      }
    ],
    outputs: [
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        description: 'Scaled value per source row.'
      },
      {
        name: 'validity',
        type: "GraphDataView<'uint32'>",
        description: 'Optional validity per source row.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUScalarScale, getGPUScalarScaleParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// values, domain, range, widths: views on graph
const scale = new GPUParameterBuffer(device, {id: 'scale', format: 'float32', length: 4});
graph.add(new GPUScalarScale({
  values, domain, range,
  parameters: scale.importToGraph(graph),
  maximumDomainCount: 2,
  output: {values: widths}
}));
scale.write(getGPUScalarScaleParameterValues({scale: 'linear', domainCount: 2}));`,
    preview: {kind: 'classification', seed: 9105, caption: 'Values mapped to a radius range.'},
    related: ['color-scale', 'class-breaks'],
    lensScenario: 'gpu-dataframe.scalar-scale',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'bivariate-classification',
    symbol: 'GPUBivariateClassification',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Bivariate Classification',
    summary: 'Classifies two columns into an n by n grid and colors rows from a bivariate palette.',
    explanation:
      'Best for bivariate choropleths, such as income against transit access. Each axis has its own breaks; an optional third column fades the alpha.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify'],
    domains: ['cartography', 'demographics'],
    params: [
      {
        name: 'valuesX',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed float32 X value per row.'
      },
      {
        name: 'valuesY',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed float32 Y value per row.'
      },
      {
        name: 'breaksX',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'X class edges.'
      },
      {
        name: 'breaksY',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Y class edges.'
      },
      {
        name: 'palette',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Row-major rgba8 palette, y class by x class.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Class counts from getGPUBivariateClassificationParameterValues.'
      },
      {
        name: 'maximumClassCount',
        type: 'number',
        required: true,
        description: 'Largest class count per axis, 1 to 16.'
      },
      {
        name: 'alphaValues',
        type: "GraphDataView<'float32'>",
        description: 'Optional value-by-alpha variable.'
      }
    ],
    outputs: [
      {
        name: 'classIds',
        type: "GraphDataView<'uint32'>",
        description: 'Combined class id per row.'
      },
      {name: 'colors', type: "GraphDataView<'uint32'>", description: 'Packed rgba8 color per row.'},
      {name: 'classCounts', type: "GraphDataView<'uint32'>", description: 'Rows per class cell.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUBivariateClassification, getGPUBivariateClassificationParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// valuesX, valuesY, breaksX, breaksY, palette, colors: views on graph
const classes = new GPUParameterBuffer(device, {id: 'bivariate', format: 'float32', length: 8});
graph.add(new GPUBivariateClassification({
  valuesX, valuesY, breaksX, breaksY, palette,
  parameters: classes.importToGraph(graph),
  maximumClassCount: 3,
  output: {colors}
}));
classes.write(getGPUBivariateClassificationParameterValues({classCountX: 3, classCountY: 3}));`,
    preview: {kind: 'classification', seed: 9106, caption: 'A 3 by 3 bivariate class grid.'},
    related: ['class-breaks', 'color-scale'],
    lensScenario: 'gpu-dataframe.bivariate-classification',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'variable-standardization',
    symbol: 'GPUVariableStandardization',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Variable Standardization',
    summary:
      'Z-score, deviation, MAD, range-adjust and range-standardize recipes for a float32 column.',
    explanation:
      'Puts variables on a common scale before clustering, scoring or regression. Reductions span the whole column; masked and non-finite rows become NaN.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      {
        name: 'values',
        type: "Column<'float32'>",
        required: true,
        description: 'Packed scalar float32 column, a single chunk.'
      },
      {name: 'mask', type: "Column<'uint32'>", description: 'Nonzero includes the row.'},
      {
        name: 'method',
        type: 'GPUVariableStandardizationMethod',
        required: true,
        description: 'z-score, deviation-from-mean, mad, range-adjust or range-standardize.'
      },
      {
        name: 'variance',
        type: "'sample' | 'population'",
        default: "'sample'",
        description: 'Z-score convention.'
      },
      {
        name: 'output',
        type: "Column<'float32'>",
        required: true,
        description: 'Caller-owned float32 output of the same length.'
      }
    ],
    outputs: [
      {
        name: 'output',
        type: "Column<'float32'>",
        description: 'Standardized value per row; excluded rows are NaN.'
      }
    ],
    example: `import {GPUVariableStandardization} from '@luma-contributors/analysis/dataframe-analysis';

// values, output: float32 columns of the same length
graph.add(new GPUVariableStandardization({
  values, method: 'z-score', variance: 'sample', output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {kind: 'histogram', seed: 9107, caption: 'A skewed column recentered and rescaled.'},
    related: ['composite-score', 'column-profile'],
    lensScenario: 'gpu-dataframe.variable-standardization',
    reference: 'api/variable-standardization',
    maturity: 'stable'
  },
  {
    id: 'composite-score',
    symbol: 'GPUCompositeScore',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Composite Score',
    summary:
      'Builds one index per row from several indicators with scaling, weights, directions and PCA.',
    explanation:
      'Best for vulnerability, deprivation or suitability indices. Scale each indicator (min-max, z-score, rank), flip those where higher is worse, then combine them by weighted sum, geometric mean or principal component.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['demographics', 'urban'],
    params: [
      {
        name: 'indicators',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Row-major float32, row * indicatorCount + column.'
      },
      {
        name: 'indicatorCount',
        type: 'number',
        required: true,
        description: 'Number of indicators, 1 to 16.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Scaler, aggregation, weights and directions.'
      },
      {
        name: 'enableRank',
        type: 'boolean',
        description: 'Compiles the sorts behind the rank scaler.'
      },
      {
        name: 'enablePrincipalComponent',
        type: 'boolean',
        description: 'Compiles the PCA behind the principal-component score.'
      }
    ],
    outputs: [
      {name: 'score', type: "GraphDataView<'float32'>", description: 'Composite score per row.'},
      {
        name: 'columnStatistics',
        type: "GraphDataView<'float32'>",
        description: 'Per-indicator statistics.'
      },
      {
        name: 'loadings',
        type: "GraphDataView<'float32'>",
        description: 'Principal component loadings.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCompositeScore, GPU_COMPOSITE_SCORE_PARAMETER_LENGTH, getGPUCompositeScoreParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// indicators (rows * 4 float32), score: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'composite', format: 'float32',
  length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH});
graph.add(new GPUCompositeScore({
  indicators, indicatorCount: 4, enableRank: true,
  parameters: parameters.importToGraph(graph),
  output: {score}
}));
parameters.write(getGPUCompositeScoreParameterValues({
  scaler: 'rank', aggregation: 'weighted-sum',
  weights: [1, 2, 0.5, 1], directions: [1, -1, 1, 1]
}));`,
    preview: {
      kind: 'classification',
      seed: 9108,
      caption: 'Four indicators folded into one index per cell.'
    },
    related: ['variable-standardization', 'inequality', 'class-breaks'],
    lensScenario: 'gpu-dataframe.composite-score',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'inequality',
    symbol: 'GPUInequality',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Inequality Indices',
    summary:
      'Per-zone Gini, Theil, Atkinson, Hoover and Palma indices and the Lorenz curve from a value column.',
    explanation:
      'Compares how unevenly income or access is distributed inside each zone. Optional population weights turn rows into shares.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['measure', 'summarize'],
    domains: ['demographics', 'urban'],
    params: [
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Non-negative float32 values, such as incomes.'
      },
      {
        name: 'zoneIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'uint32 zone per row.'
      },
      {name: 'zoneCount', type: 'number', required: true, description: 'Number of zones.'},
      {
        name: 'weights',
        type: "GraphDataView<'float32'>",
        description: 'Optional float32 population weights.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Atkinson epsilon and Palma shares.'
      },
      {name: 'lorenzKnotCount', type: 'number', description: 'Lorenz curve knots per zone.'}
    ],
    outputs: [
      {name: 'gini', type: "GraphDataView<'float32'>", description: 'Gini index per zone.'},
      {name: 'theilT', type: "GraphDataView<'float32'>", description: 'Theil T index per zone.'},
      {name: 'palma', type: "GraphDataView<'float32'>", description: 'Palma ratio per zone.'},
      {
        name: 'lorenzKnots',
        type: "GraphDataView<'float32'>",
        description: 'Lorenz curve knots per zone.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUInequality, GPU_INEQUALITY_PARAMETER_LENGTH, getGPUInequalityParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// incomes, zoneIds, gini, lorenzKnots: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'inequality', format: 'float32',
  length: GPU_INEQUALITY_PARAMETER_LENGTH});
graph.add(new GPUInequality({
  values: incomes, zoneIds, zoneCount: 120, lorenzKnotCount: 32,
  parameters: parameters.importToGraph(graph),
  output: {gini, lorenzKnots}
}));
parameters.write(getGPUInequalityParameterValues({epsilon: 1}));`,
    preview: {kind: 'histogram', seed: 9109, caption: 'Lorenz curves and Gini per zone.'},
    related: ['composite-score', 'group-statistics'],
    lensScenario: 'gpu-dataframe.inequality',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'group-statistics',
    symbol: 'GPUGroupStatistics',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Group Statistics',
    summary:
      'Groups rows by a 32- or 64-bit key and computes moments, median, percentiles and mode per group.',
    explanation:
      'Best for a group-by that never leaves the GPU, such as stats per tract or per category. Results land in a capacity-bounded table with an overflow flag.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      {
        name: 'keys',
        type: "GraphDataView<'uint32'> | GraphDataView<'uint32x2'>",
        required: true,
        description: 'uint32 or uint32x2 (64-bit) group key per row.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {name: 'keyCount', type: 'number', description: 'Dense mode: number of keys.'},
      {
        name: 'columns',
        type: 'readonly GPUGroupStatisticsColumn[]',
        required: true,
        description: 'Zero to four value columns with their statistics.'
      },
      {
        name: 'variance',
        type: "'sample' | 'population'",
        default: "'sample'",
        description: 'Variance convention.'
      },
      {
        name: 'percentiles',
        type: "GraphDataView<'float32'>",
        description: 'Per-frame quantile fractions in [0, 1].'
      },
      {name: 'sumScale', type: 'number', description: 'Fixed-point scale of sums and means.'}
    ],
    outputs: [
      {name: 'keys', type: "GraphDataView<'uint32'>", description: 'Distinct group keys.'},
      {name: 'counts', type: "GraphDataView<'uint32'>", description: 'Rows per group.'},
      {
        name: 'status',
        type: 'GPUCompactOutput status',
        description: 'Group count and overflow flag.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';

// keys, fare, groupKeys, groupCounts, counts, means, medians, status: views on graph
graph.add(new GPUGroupStatistics({
  keys,
  columns: [{
    values: fare,
    statistics: ['count', 'mean', 'median'],
    output: {counts, means, medians}
  }],
  variance: 'sample',
  output: {keys: groupKeys, counts: groupCounts, status}
}));`,
    preview: {kind: 'histogram', seed: 9110, caption: 'Mean and spread per group.'},
    related: ['column-profile', 'key-join', 'temporal-reduction'],
    lensScenario: 'gpu-dataframe.group-statistics',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'key-join',
    symbol: 'GPUKeyJoin',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Key Join',
    summary:
      'Attribute join of a left table onto a right table by 32- or 64-bit key, entirely on the GPU.',
    explanation:
      'Attaches attributes by id, such as census fields onto features. Gather copies the first match; aggregates reduce all matches (count, sum, mean, minimum, maximum).',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['relate'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'leftKeys',
        type: "GraphDataView<'uint32'> | GraphDataView<'uint32x2'>",
        required: true,
        description: 'uint32 or uint32x2 keys of the left table.'
      },
      {
        name: 'rightKeys',
        type: "GraphDataView<'uint32'> | GraphDataView<'uint32x2'>",
        required: true,
        description: 'Right keys in the same format.'
      },
      {name: 'leftMask', type: "GraphDataView<'uint32'>", description: 'Zero rows never match.'},
      {name: 'rightMask', type: "GraphDataView<'uint32'>", description: 'Zero rows are ignored.'},
      {name: 'kind', type: 'GPUKeyJoinKind', description: "Join kind: 'left' or 'inner'."},
      {
        name: 'gather',
        type: 'readonly GPUKeyJoinGather[]',
        description: '1:1 right columns copied to the left (first match).'
      },
      {
        name: 'aggregates',
        type: 'readonly GPUKeyJoinAggregate[]',
        description: '1:n aggregates over all matching right rows.'
      }
    ],
    outputs: [
      {
        name: 'matched',
        type: "GraphDataView<'uint32'>",
        description: 'Left rows that found a match.'
      },
      {
        name: 'rightRows',
        type: "GraphDataView<'uint32'>",
        description: 'Matching right row per left row.'
      },
      {
        name: 'matchCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Number of right matches per left row.'
      }
    ],
    example: `import {GPUKeyJoin} from '@luma-contributors/analysis/dataframe-analysis';

// leftKeys, rightKeys, rightPopulation, rightIncome, leftPopulation, leftIncomeSum,
// matched, rightRows, matchCounts: views on graph
graph.add(new GPUKeyJoin({
  leftKeys, rightKeys, kind: 'left',
  gather: [{column: rightPopulation, output: leftPopulation}],
  aggregates: [{operation: 'sum', column: rightIncome, output: leftIncomeSum}],
  output: {matched, rightRows, matchCounts}
}));`,
    preview: {
      kind: 'pipeline',
      seed: 9111,
      caption: 'Right-table columns gathered onto left rows by key.'
    },
    related: ['group-statistics', 'transfer-mask', 'point-in-polygon-join'],
    lensScenario: 'gpu-dataframe.key-join',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'calendar-buckets',
    symbol: 'GPUCalendarBuckets',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Calendar Buckets',
    summary:
      'Decodes epoch-millisecond timestamps into year, month, hour, weekday, ISO week and more.',
    explanation:
      'Buckets events by hour of day or weekday with no CPU date math. Exact Int64 words, per-row UTC offsets or a time-zone table keep local time right.',
    dataTypes: {input: ['time-series'], output: ['table']},
    useCases: ['summarize'],
    domains: ['urban', 'data-engineering'],
    params: [
      {
        name: 'timestamps',
        type: "GraphDataView<'uint32x2'>",
        required: true,
        description: 'Epoch milliseconds as Int64 word pairs.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Zero makes the row invalid.'},
      {
        name: 'utcOffsets',
        type: "GraphDataView<'sint32'>",
        description: 'Optional UTC offset in minutes per row.'
      },
      {
        name: 'timeZone',
        type: 'GPUCalendarBucketsTimeZone',
        description: 'Optional time zone transition table.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'sint32'>",
        required: true,
        description: 'UTC offset and week start from getGPUCalendarBucketsParameterValues.'
      }
    ],
    outputs: [
      {name: 'hour', type: "GraphDataView<'uint32'>", description: 'Hour of day per row.'},
      {name: 'weekday', type: "GraphDataView<'uint32'>", description: 'Weekday per row.'},
      {name: 'month', type: "GraphDataView<'uint32'>", description: 'Month per row.'},
      {
        name: 'hourWeekdayCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Rows per hour and weekday cell.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCalendarBuckets, getGPUCalendarBucketsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// timestamps (uint32x2 Int64 words from getInt64TimeWords), hour, weekday: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'calendar', format: 'sint32', length: 2});
graph.add(new GPUCalendarBuckets({
  timestamps,
  parameters: parameters.importToGraph(graph),
  output: {hour, weekday}
}));
// UTC-5, weeks start on Monday
parameters.write(getGPUCalendarBucketsParameterValues(-300, 0));`,
    preview: {
      kind: 'time-series',
      seed: 9112,
      caption: 'Events folded into hour and weekday cells.'
    },
    related: ['temporal-reduction', 'time-window-filter'],
    lensScenario: 'gpu-dataframe.calendar-buckets',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'time-window-filter',
    symbol: 'GPUTimeWindowFilter',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Time Window Filter',
    summary:
      'Filters timestamped rows or intervals against a per-frame window and publishes compact ids and fade weights.',
    explanation:
      'Scrubs a timeline over events or trails. The window is a parameter write; exact Int64 time words and sorted-range mode give O(log n) frames.',
    dataTypes: {input: ['time-series', 'trajectories'], output: ['table']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['urban', 'data-engineering'],
    params: [
      {
        name: 'timestamps',
        type: 'Float32Rows | GPUInt64TimeWordRows',
        required: true,
        description: 'Row times, or interval start times.'
      },
      {
        name: 'endTimestamps',
        type: 'Float32Rows | GPUInt64TimeWordRows',
        description: 'Optional interval end times.'
      },
      {
        name: 'window',
        type: "GraphDataView<'uint32'> | GraphDataView<'float32'>",
        required: true,
        description: 'Per-frame window from getGPUTimeWindowFilterParameterValues.'
      },
      {
        name: 'sourceIds',
        type: 'GPUUint32Rows',
        description: 'Optional stable ids aligned with rows.'
      },
      {name: 'trackIds', type: 'GPUUint32Rows', description: 'Optional dense track id per row.'},
      {
        name: 'sorted',
        type: 'boolean',
        default: 'false',
        description: 'Rows are sorted by time: one contiguous range.'
      },
      {
        name: 'validateOrder',
        type: 'boolean',
        default: 'false',
        description: 'Sorted mode only: verify the order.'
      }
    ],
    outputs: [
      {name: 'ids', type: "GraphDataView<'uint32'>", description: 'Accepted row ids, compacted.'},
      {
        name: 'status',
        type: 'GPUCompactOutput status',
        description: 'Count and overflow flag; can drive an indirect draw.'
      },
      {name: 'outputMask', type: 'GPUUint32Rows', description: 'Optional per-row accept mask.'},
      {name: 'fadeWeights', type: 'Float32Rows', description: 'Optional fade weight per row.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues, GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} from '@luma-contributors/analysis/dataframe-analysis';

// timestamps, ids, status: views on graph
const window = new GPUParameterBuffer(device, {id: 'window', format: 'float32',
  length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH});
graph.add(new GPUTimeWindowFilter({
  timestamps, window: window.importToGraph(graph),
  output: {ids, status}
}));
const compiled = await graph.compileAsync();
// every frame
window.write(getGPUTimeWindowFilterParameterValues({start: now - 60, end: now}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'time-series',
      seed: 9113,
      caption: 'A moving window selecting events on a timeline.'
    },
    related: ['temporal-reduction', 'calendar-buckets'],
    lensScenario: 'gpu-dataframe.time-window-filter',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'temporal-reduction',
    symbol: 'GPUTemporalReduction',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Temporal Reduction',
    summary:
      'Reduces timestamped rows to per-cell, per-time-bucket minimum, maximum, first, last and count.',
    explanation:
      'Use it so a scrubbing view draws one feature per cell and bucket rather than every event. Bucket origin and width are per-frame parameters.',
    dataTypes: {input: ['time-series', 'cells'], output: ['table']},
    useCases: ['summarize'],
    domains: ['urban', 'data-engineering'],
    params: [
      {
        name: 'cellIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'uint32 cell index per row.'
      },
      {
        name: 'timestamps',
        type: "GraphDataView<'float32'> | GraphDataView<'uint32x2'>",
        required: true,
        description: 'Relative float32 times, or exact Int64 words.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Float32 value per row.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'uint32'> | GraphDataView<'float32'>",
        required: true,
        description: 'Bucket origin and width.'
      },
      {
        name: 'cellCount',
        type: 'number',
        required: true,
        description: 'Number of cells, compile-time.'
      },
      {
        name: 'bucketCount',
        type: 'number',
        required: true,
        description: 'Coarse time buckets per cell, compile-time.'
      },
      {name: 'sorted', type: 'boolean', default: 'false', description: 'Rows are sorted by time.'}
    ],
    outputs: [
      {name: 'counts', type: "GraphDataView<'uint32'>", description: 'Rows per cell and bucket.'},
      {
        name: 'minimums',
        type: "GraphDataView<'float32'>",
        description: 'Minimum value per cell and bucket.'
      },
      {
        name: 'maximums',
        type: "GraphDataView<'float32'>",
        description: 'Maximum value per cell and bucket.'
      },
      {
        name: 'first',
        type: "GraphDataView<'float32'>",
        description: 'First value per cell and bucket.'
      },
      {
        name: 'last',
        type: "GraphDataView<'float32'>",
        description: 'Last value per cell and bucket.'
      },
      {
        name: 'occupiedSlots',
        type: 'GPUCompactOutput',
        description: 'Occupied slot ids (cell * bucketCount + bucket) and count.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues, GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH} from '@luma-contributors/analysis/dataframe-analysis';

// cellIds, timestamps, values, counts, minimums, maximums, first, last, occupiedSlots: views on graph
const buckets = new GPUParameterBuffer(device, {id: 'buckets', format: 'float32',
  length: GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH});
graph.add(new GPUTemporalReduction({
  cellIds, timestamps, values, cellCount: 500, bucketCount: 24,
  parameters: buckets.importToGraph(graph),
  output: {counts, minimums, maximums, first, last, occupiedSlots}
}));
buckets.write(getGPUTemporalReductionParameterValues(originTime, 3600));`,
    preview: {kind: 'time-series', seed: 9114, caption: 'Hourly min and max per cell.'},
    related: ['time-window-filter', 'calendar-buckets', 'group-statistics', 'cell-aggregation'],
    lensScenario: 'gpu-dataframe.temporal-reduction',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'ripley-k',
    symbol: 'GPURipley',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: "Ripley's K and L",
    summary:
      "Ripley's K, Besag's L and the pair-correlation function of a point pattern, with edge correction.",
    explanation:
      'Tests whether points cluster or repel across distance scales. One deterministic pass counts pairs per radius inside a rectangle or polygon window.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Coordinate and metric descriptor of positions.'
      },
      {
        name: 'studyWindow',
        type: 'GPUStudyWindow',
        default: "'rectangle'",
        description: "Study window: 'rectangle' or a prepared polygon."
      },
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumDistance, edgeCorrection and seed.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {name: 'radiusCount', type: 'number', required: true, description: 'Number of radii.'},
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'}
    ],
    outputs: [
      {name: 'k', type: "GraphDataView<'float32'>", description: 'Ripley K per radius.'},
      {name: 'l', type: "GraphDataView<'float32'>", description: "Besag's L per radius."},
      {name: 'lMinusR', type: "GraphDataView<'float32'>", description: 'L(r) minus r per radius.'},
      {
        name: 'pairCorrelation',
        type: "GraphDataView<'float32'>",
        description: 'Annulus pair-correlation per radius.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPURipley, GPU_RIPLEY_PARAMETER_LENGTH, getGPURipleyParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// positions (float32x2), k, lMinusR: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'ripley', format: 'float32',
  length: GPU_RIPLEY_PARAMETER_LENGTH});
graph.add(new GPURipley({
  positions, gridSize: [64, 64], radiusCount: 40,
  parameters: parameters.importToGraph(graph),
  output: {k, lMinusR}
}));
parameters.write(getGPURipleyParameterValues({
  bounds: [0, 0, 1000, 1000], maximumDistance: 250, edgeCorrection: 'isotropic'
}));`,
    preview: {
      kind: 'point-clusters',
      seed: 9115,
      caption: 'Clustered points and the K curve above random.'
    },
    related: [
      'ripley-envelope',
      'ripley-distance-functions',
      'point-pattern-indices',
      'point-pattern-envelope'
    ],
    lensScenario: 'gpu-dataframe.ripley',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'ripley-envelope',
    symbol: 'GPURipleyEnvelope',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: "Ripley's Envelope",
    summary: 'Monte Carlo complete-spatial-randomness envelopes for K, L or L(r) minus r.',
    explanation:
      'Judges whether an observed Ripley curve leaves what random patterns produce. Simulations run on the GPU in the same window; pointwise and global bands come back with the theoretical curve.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'studyWindow',
        type: 'GPUStudyWindow',
        default: "'rectangle'",
        description: "Study window: 'rectangle' or a prepared polygon."
      },
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Same 8-element layout as the observed GPURipley.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {
        name: 'radiusCount',
        type: 'number',
        required: true,
        description: 'Must equal the observed radiusCount.'
      },
      {
        name: 'simulationCount',
        type: 'number',
        required: true,
        description: 'Number of CSR simulations, 2 * rank to 199.'
      },
      {
        name: 'rank',
        type: 'number',
        description: 'Which sorted simulated value bounds the envelope.'
      },
      {
        name: 'statistic',
        type: 'GPURipleyEnvelopeStatistic',
        description: 'Scale of the reported envelopes.'
      }
    ],
    outputs: [
      {
        name: 'lower',
        type: "GraphDataView<'float32'>",
        description: 'Pointwise lower envelope per radius.'
      },
      {
        name: 'upper',
        type: "GraphDataView<'float32'>",
        description: 'Pointwise upper envelope per radius.'
      },
      {
        name: 'globalLower',
        type: "GraphDataView<'float32'>",
        description: 'Global lower envelope.'
      },
      {
        name: 'globalUpper',
        type: "GraphDataView<'float32'>",
        description: 'Global upper envelope.'
      },
      {name: 'theoretical', type: "GraphDataView<'float32'>", description: 'Theoretical CSR curve.'}
    ],
    example: `import {GPURipleyEnvelope} from '@luma-contributors/analysis/dataframe-analysis';

// positions, parameters (as for GPURipley), lower, upper, theoretical: views on graph
graph.add(new GPURipleyEnvelope({
  positions, parameters, gridSize: [64, 64], radiusCount: 40,
  simulationCount: 99, rank: 5, statistic: 'l-minus-r',
  output: {lower, upper, theoretical}
}));`,
    preview: {
      kind: 'regression',
      seed: 9116,
      caption: 'An observed curve against a simulated random band.'
    },
    related: ['ripley-k', 'point-pattern-envelope'],
    lensScenario: 'gpu-dataframe.ripley-envelope',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'ripley-distance-functions',
    symbol: 'GPURipleyDistanceFunctions',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'G, F and J Functions',
    summary:
      'Nearest-neighbor G, empty-space F and J = (1 - G) / (1 - F) of a point pattern in a window.',
    explanation:
      'Complements K when you care about gaps and nearest neighbors. J below 1 suggests clustering and above 1 regularity.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'studyWindow',
        type: 'GPUStudyWindow',
        default: "'rectangle'",
        description: "Study window: 'rectangle' or a prepared polygon."
      },
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumDistance and edgeCorrection.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {
        name: 'referenceGrid',
        type: 'readonly [number, number]',
        required: true,
        description: '[columns, rows] of the reference location lattice for F.'
      },
      {name: 'radiusCount', type: 'number', required: true, description: 'Number of radii.'},
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'}
    ],
    outputs: [
      {
        name: 'g',
        type: "GraphDataView<'float32'>",
        description: 'Nearest-neighbor function per radius.'
      },
      {
        name: 'f',
        type: "GraphDataView<'float32'>",
        description: 'Empty-space function per radius.'
      },
      {name: 'j', type: "GraphDataView<'float32'>", description: 'J function per radius.'},
      {name: 'radii', type: "GraphDataView<'float32'>", description: 'Radius of each row.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPURipleyDistanceFunctions, GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH, getGPURipleyDistanceFunctionsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// positions, g, f, j: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'gfj', format: 'float32',
  length: GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH});
graph.add(new GPURipleyDistanceFunctions({
  positions, gridSize: [64, 64], referenceGrid: [32, 32], radiusCount: 40,
  parameters: parameters.importToGraph(graph),
  output: {g, f, j}
}));
parameters.write(getGPURipleyDistanceFunctionsParameterValues({
  bounds: [0, 0, 1000, 1000], maximumDistance: 100
}));`,
    preview: {kind: 'points', seed: 9117, caption: 'Nearest-neighbor and empty-space distances.'},
    related: ['ripley-k', 'point-pattern-indices'],
    lensScenario: 'gpu-dataframe.ripley-distance-functions',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'point-pattern-indices',
    symbol: 'GPUPointPatternIndices',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Point Pattern Indices',
    summary:
      'Clark-Evans nearest-neighbor index and quadrat-count dispersion tests for a point pattern.',
    explanation:
      'Best for a quick yes-or-no on clustering versus randomness. Clark-Evans compares mean nearest-neighbor distance with random; the quadrat test compares cell counts with Poisson.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'studyWindow',
        type: 'GPUStudyWindow',
        default: "'rectangle'",
        description: "Study window: 'rectangle' or a prepared polygon."
      },
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumDistance, correction and alternative.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {
        name: 'quadratGrid',
        type: 'readonly [number, number]',
        description: '[columns, rows] of the quadrat tiling.'
      },
      {
        name: 'quadratSamples',
        type: 'number',
        default: '16',
        description: 'Midpoint samples per quadrat axis.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'}
    ],
    outputs: [
      {
        name: 'clarkEvans',
        type: "GraphDataView<'float32'>",
        description: 'Clark-Evans statistics.'
      },
      {
        name: 'nearestNeighborDistances',
        type: "GraphDataView<'float32'>",
        description: 'Nearest-neighbor distance per point.'
      },
      {name: 'quadratCounts', type: "GraphDataView<'uint32'>", description: 'Points per quadrat.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUPointPatternIndices, GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH, getGPUPointPatternIndicesParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// positions, clarkEvans, quadratCounts: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'indices', format: 'float32',
  length: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH});
graph.add(new GPUPointPatternIndices({
  positions, gridSize: [64, 64], quadratGrid: [8, 8],
  parameters: parameters.importToGraph(graph),
  output: {clarkEvans, quadratCounts}
}));
parameters.write(getGPUPointPatternIndicesParameterValues({
  bounds: [0, 0, 1000, 1000], maximumDistance: 200
}));`,
    preview: {
      kind: 'points',
      seed: 9118,
      caption: 'Nearest-neighbor links and an 8 by 8 quadrat grid.'
    },
    related: ['ripley-k', 'ripley-distance-functions'],
    lensScenario: 'gpu-dataframe.point-pattern-indices',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'marked-point-statistics',
    symbol: 'GPUMarkedPointStatistics',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Marked Point Statistics',
    summary:
      'Cross-type K and L, inhomogeneous K and annular pair and mark correlation on a rectangle.',
    explanation:
      'Best when points carry a type or a mark, such as two species or trees with sizes. It tests whether types attract each other or whether marks are spatially correlated.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {name: 'types', type: "GraphDataView<'uint32'>", description: 'Optional uint32 categories.'},
      {
        name: 'typePair',
        type: 'readonly [number, number]',
        description: 'The two types compared in the cross-type statistic.'
      },
      {
        name: 'marks',
        type: "GraphDataView<'float32'>",
        description: 'Optional finite scalar mark per point.'
      },
      {
        name: 'intensities',
        type: "GraphDataView<'float32'>",
        description: 'Optional positive point intensity.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumRadius and minimum intensity.'
      },
      {name: 'radiusCount', type: 'number', required: true, description: 'Number of radii.'},
      {
        name: 'edgeCorrection',
        type: "'none' | 'translation'",
        default: "'translation'",
        description: 'Rectangle translation correction or none.'
      }
    ],
    outputs: [
      {
        name: 'statistics',
        type: "GraphDataView<'float32'>",
        description: 'Four statistics per radius.'
      },
      {
        name: 'pairCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Pairs counted per radius.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Per radius: 0 valid, 1 invalid, 2 nonfinite.'
      }
    ],
    example: `import {GPUMarkedPointStatistics} from '@luma-contributors/analysis/dataframe-analysis';

// positions, types, parameters, statistics, pairCounts, status: views on graph
graph.add(new GPUMarkedPointStatistics({
  positions, types, typePair: [0, 1],
  parameters, radiusCount: 32,
  output: {statistics, pairCounts, status}
}));`,
    preview: {
      kind: 'point-clusters',
      seed: 9119,
      caption: 'Two point types and their cross-K curve.'
    },
    related: ['ripley-k', 'point-pattern-indices', 'inhomogeneous-point-pattern'],
    lensScenario: 'gpu-dataframe.marked-point-statistics',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'variogram',
    symbol: 'GPUVariogram',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Empirical Variogram',
    summary:
      'Empirical semivariogram of point values, optionally directional, from one deterministic pair pass.',
    explanation:
      'Shows how similarity decays with distance before kriging. Pairs fall into equal-width lag bins, optionally split by direction sector, with robust and fixed-point accumulation.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'hydrology', 'ecology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed observed value per row.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumDistance and azimuthOffset.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {
        name: 'lagCount',
        type: 'number',
        required: true,
        description: 'Equal-width lag bins over [0, maximumDistance].'
      },
      {name: 'directionCount', type: 'number', description: 'Direction sectors over [0, pi).'},
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'}
    ],
    outputs: [
      {
        name: 'semivariances',
        type: "GraphDataView<'float32'>",
        description: 'Semivariance per lag bin.'
      },
      {name: 'pairCounts', type: "GraphDataView<'uint32'>", description: 'Pairs per lag bin.'},
      {
        name: 'meanDistances',
        type: "GraphDataView<'float32'>",
        description: 'Mean pair distance per lag bin.'
      }
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUVariogram, GPU_VARIOGRAM_PARAMETER_LENGTH, getGPUVariogramParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// positions (float32x2), values, semivariances, pairCounts, meanDistances: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'variogram', format: 'float32',
  length: GPU_VARIOGRAM_PARAMETER_LENGTH});
graph.add(new GPUVariogram({
  positions, values, gridSize: [64, 64], lagCount: 24, directionCount: 4,
  parameters: parameters.importToGraph(graph),
  output: {semivariances, pairCounts, meanDistances}
}));
parameters.write(getGPUVariogramParameterValues({bounds: [0, 0, 1000, 1000], maximumDistance: 500}));`,
    preview: {
      kind: 'regression',
      seed: 9120,
      caption: 'Semivariance rising with lag distance to a sill.'
    },
    related: ['variogram-model-fit', 'cross-variogram', 'space-time-variogram', 'kriging'],
    lensScenario: 'gpu-dataframe.variogram',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'variogram-model-fit',
    symbol: 'GPUVariogramModelFit',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Variogram Model Fit',
    summary:
      'Fits spherical, exponential or Gaussian models to empirical semivariances entirely on the GPU.',
    explanation:
      'Turns an empirical variogram into nugget, sill and range for kriging without a CPU fit. It searches a log-range ladder with golden-section refinement and weights bins by Cressie, pairs or none.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      {
        name: 'distances',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Mean distance per lag bin.'
      },
      {
        name: 'semivariances',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Empirical semivariance per lag bin.'
      },
      {
        name: 'pairCounts',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Pairs per lag bin.'
      },
      {
        name: 'models',
        type: 'readonly VariogramModelType[]',
        description: 'Candidate models; output row order follows this.'
      },
      {
        name: 'weighting',
        type: 'VariogramModelWeighting',
        description: 'Least-squares weighting: cressie, pairs or none.'
      },
      {name: 'rangeSteps', type: 'number', description: 'Steps of the log-range ladder.'},
      {
        name: 'maximumIterations',
        type: 'number',
        description: 'Golden-section refinement iterations.'
      }
    ],
    outputs: [
      {
        name: 'parameters',
        type: "GraphDataView<'float32x4'>",
        description: 'Nugget, partial sill, effective range, residual per model.'
      },
      {
        name: 'diagnostics',
        type: "GraphDataView<'float32x4'>",
        description: 'Usable bins, squared residual, AIC, bracket per model.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: '0 converged, 1 few bins, 2 numeric failure, 3 boundary.'
      }
    ],
    example: `import {GPUVariogramModelFit} from '@luma-contributors/analysis/dataframe-analysis';

// distances, semivariances, pairCounts: GPUVariogram output views; fit: output views
graph.add(new GPUVariogramModelFit({
  distances: meanDistances, semivariances, pairCounts,
  models: ['spherical', 'exponential', 'gaussian'],
  weighting: 'cressie',
  output: {parameters, diagnostics, status}
}));`,
    preview: {
      kind: 'regression',
      seed: 9121,
      caption: 'Three candidate models fitted to the same bins.'
    },
    related: ['variogram', 'cross-variogram', 'kriging'],
    lensScenario: 'gpu-dataframe.variogram-model-fit',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'cross-variogram',
    symbol: 'GPUCrossVariogram',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Cross Variogram',
    summary: 'Pair accumulation of (x_i - x_j)(y_i - y_j) / 2 between two variables at each lag.',
    explanation:
      'Shows how two variables co-vary with distance, for cokriging. Without secondValues it reduces to the ordinary Matheron semivariogram.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed float32x2 point positions.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'First variable.'
      },
      {
        name: 'secondValues',
        type: "GraphDataView<'float32'>",
        default: 'values',
        description: 'Second variable for the cross semivariogram.'
      },
      {
        name: 'times',
        type: "GraphDataView<'float32'>",
        description: 'Optional times for a space-time lag.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: '[maximumDistance, maximumTimeLag], positive and finite.'
      },
      {name: 'lagCount', type: 'number', required: true, description: 'Spatial lag bins.'},
      {name: 'temporalLagCount', type: 'number', description: 'Temporal lag bins.'}
    ],
    outputs: [
      {
        name: 'semivariances',
        type: "GraphDataView<'float32'>",
        description: 'Cross semivariance per lag.'
      },
      {name: 'pairCounts', type: "GraphDataView<'uint32'>", description: 'Pairs per lag.'},
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Per lag: 0 valid, 1 empty, 2 nonfinite.'
      }
    ],
    example: `import {GPUCrossVariogram} from '@luma-contributors/analysis/dataframe-analysis';

// positions, ozone, pm25, parameters, semivariances, pairCounts, status: views on graph
graph.add(new GPUCrossVariogram({
  positions, values: ozone, secondValues: pm25,
  parameters, lagCount: 20,
  output: {semivariances, pairCounts, status}
}));`,
    preview: {kind: 'regression', seed: 9122, caption: 'Two variables co-varying with distance.'},
    related: ['variogram', 'space-time-variogram', 'variogram-model-fit'],
    lensScenario: 'gpu-dataframe.cross-variogram',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'space-time-variogram',
    symbol: 'GPUSpaceTimeVariogram',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Space-Time Variogram',
    summary: 'Semivariance across both spatial and temporal lags of timestamped point values.',
    explanation:
      'Checks whether similarity decays with distance and with time before space-time kriging. Output is a spatial by temporal lag grid.',
    dataTypes: {input: ['points', 'time-series'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed float32x2 point positions.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Observed value per row.'
      },
      {
        name: 'times',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Observation time per row.'
      },
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: '[maximumDistance, maximumTimeLag], positive and finite.'
      },
      {name: 'lagCount', type: 'number', required: true, description: 'Spatial lag bins.'},
      {name: 'temporalLagCount', type: 'number', required: true, description: 'Temporal lag bins.'}
    ],
    outputs: [
      {
        name: 'semivariances',
        type: "GraphDataView<'float32'>",
        description: 'Semivariance per spatial and temporal lag.'
      },
      {
        name: 'pairCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Pairs per spatial and temporal lag.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Per lag: 0 valid, 1 empty, 2 nonfinite.'
      }
    ],
    example: `import {GPUSpaceTimeVariogram} from '@luma-contributors/analysis/dataframe-analysis';

// positions, values, times, parameters, semivariances, pairCounts, status: views on graph
graph.add(new GPUSpaceTimeVariogram({
  positions, values, times, parameters,
  lagCount: 16, temporalLagCount: 8,
  output: {semivariances, pairCounts, status}
}));`,
    preview: {
      kind: 'density-surface',
      seed: 9123,
      caption: 'Semivariance over space lag by time lag.'
    },
    related: ['variogram', 'cross-variogram', 'variogram-model-fit'],
    lensScenario: 'gpu-dataframe.space-time-variogram',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'spatial-correlogram',
    symbol: 'GPUSpatialCorrelogram',
    kind: 'class',
    importPath: '@luma-contributors/analysis/dataframe-analysis',
    title: 'Spatial Correlogram',
    summary:
      "Global Moran's I at a ladder of distance bands, with analytic z-scores, in one pair pass.",
    explanation:
      'Finds the distance at which spatial autocorrelation fades, and so pick a neighborhood scale. Bands are cumulative or annular; peakBands flags the strongest.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed planar points, one row per event.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed analysis value per row.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Bounds, maximumDistance and varianceAssumption.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Maximum [columns, rows] of the neighbor-search lattice.'
      },
      {
        name: 'bandCount',
        type: 'number',
        required: true,
        description: 'Number of distance bands, 1 to 64.'
      },
      {name: 'bandMode', type: 'GPUSpatialCorrelogramBandMode', description: 'Band layout.'},
      {name: 'mask', type: "GraphDataView<'uint32'>", description: 'Nonzero rows take part.'}
    ],
    outputs: [
      {name: 'moransI', type: "GraphDataView<'float32'>", description: "Moran's I per band."},
      {
        name: 'zScores',
        type: "GraphDataView<'float32'>",
        description: 'Analytic z-score per band.'
      },
      {name: 'pValues', type: "GraphDataView<'float32'>", description: 'p-value per band.'},
      {name: 'pairCounts', type: "GraphDataView<'uint32'>", description: 'Pairs per band.'}
    ],
    example: `import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUSpatialCorrelogram, GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH, getGPUSpatialCorrelogramParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

// positions, values, moransI, zScores: views on graph
const parameters = new GPUParameterBuffer(device, {id: 'correlogram', format: 'float32',
  length: GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH});
graph.add(new GPUSpatialCorrelogram({
  positions, values, gridSize: [64, 64], bandCount: 12,
  parameters: parameters.importToGraph(graph),
  output: {moransI, zScores}
}));
parameters.write(getGPUSpatialCorrelogramParameterValues({
  bounds: [0, 0, 1000, 1000], maximumDistance: 600
}));`,
    preview: {kind: 'histogram', seed: 9124, caption: "Moran's I falling across distance bands."},
    related: ['variogram', 'global-spatial-statistics'],
    lensScenario: 'gpu-dataframe.spatial-correlogram',
    reference: 'api/gpu-dataframe-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-neighborhoods',
    symbol: 'GPUPointCloudNeighborhoods',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud Neighborhoods',
    summary:
      'Exact 3D radius neighborhoods with nearest-k capacity, optional grid acceleration and observable truncation.',
    explanation:
      'The first step of normals, shape features and outlier filtering. Each row keeps its nearest 1 to 64 neighbors within the radius, and counts reports the full support so truncation is visible.',
    dataTypes: {input: ['point-cloud'], output: ['table']},
    useCases: ['index', 'relate'],
    domains: ['three-d'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 Cartesian positions.'
      },
      {
        name: 'queryRows',
        type: "GraphDataView<'uint32'>",
        description: 'Optional owned query rows into positions.'
      },
      {
        name: 'radius',
        type: 'number',
        required: true,
        description: 'Inclusive Euclidean 3D search radius.'
      },
      {
        name: 'capacity',
        type: 'number',
        required: true,
        description: 'Nearest 1 to 64 other points retained per row.'
      },
      {
        name: 'strategy',
        type: "'serial' | 'cooperative'",
        default: "'serial'",
        description: "'serial' or 'cooperative' (dense candidate sets)."
      },
      {
        name: 'indices',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Row-major original ids; missing slots are 0xffffffff.'
      },
      {
        name: 'squaredDistances',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Squared distance of each retained neighbor.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Full radius support, including candidates over capacity.'
      }
    ],
    outputs: [
      {
        name: 'indices',
        type: "GraphDataView<'uint32'>",
        description: 'N * capacity neighbor ids, nearest first.'
      },
      {
        name: 'squaredDistances',
        type: "GraphDataView<'float32'>",
        description: 'N * capacity squared distances.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Neighbors within radius per point; above capacity means truncated.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Optional: 1 invalid query, 2 truncated.'
      }
    ],
    example: `import {GPUPointCloudNeighborhoods} from '@luma-contributors/analysis/pointcloud-analysis';

// positions (float32x3 in a local frame), indices, squaredDistances, counts: views on graph
graph.add(new GPUPointCloudNeighborhoods({
  positions, radius: 5, capacity: 8,
  indices, squaredDistances, counts
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {
      kind: 'point-cloud',
      seed: 9125,
      caption: 'Radius neighbors found around each lidar point.'
    },
    related: [
      'pointcloud-features',
      'pointcloud-statistical-filter',
      'pointcloud-local-positions',
      'neighbor-search'
    ],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-features',
    symbol: 'GPUPointCloudFeatures',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud Features',
    summary:
      'Normals, eigenvalues and linearity, planarity and scattering from 3D neighborhoods via covariance and Jacobi.',
    explanation:
      'Separates ground, walls, poles and vegetation. It builds a population covariance per neighborhood and solves its eigenvectors with a bounded Jacobi sweep.',
    dataTypes: {input: ['point-cloud'], output: ['table']},
    useCases: ['measure', 'find-patterns'],
    domains: ['three-d', 'remote-sensing'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions.'
      },
      {
        name: 'indices',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Neighbor ids from GPUPointCloudNeighborhoods.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Neighbor counts from GPUPointCloudNeighborhoods.'
      },
      {
        name: 'capacity',
        type: 'number',
        required: true,
        description: 'Neighbor capacity of the input.'
      },
      {
        name: 'viewpoint',
        type: 'readonly [number, number, number]',
        default: 'nonnegative z',
        description: 'Orient normals toward this point.'
      },
      {
        name: 'normals',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Unit normal per point.'
      },
      {
        name: 'eigenvalues',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Descending eigenvalues per point.'
      },
      {
        name: 'shapes',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Linearity, planarity, scattering per point.'
      }
    ],
    moreRequired: ['status'],
    outputs: [
      {name: 'normals', type: "GraphDataView<'float32x3'>", description: 'Unit normals.'},
      {
        name: 'eigenvalues',
        type: "GraphDataView<'float32x3'>",
        description: 'Descending eigenvalues of the covariance.'
      },
      {
        name: 'shapes',
        type: "GraphDataView<'float32x3'>",
        description: 'Linearity, planarity, scattering.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Bit 1 degenerate, bit 2 truncated, bit 4 eigensolve bound.'
      }
    ],
    example: `import {GPUPointCloudNeighborhoods, GPUPointCloudFeatures} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, indices, squaredDistances, counts, normals, eigenvalues, shapes, status: views on graph
graph.add(new GPUPointCloudNeighborhoods({
  positions, radius: 5, capacity: 8, indices, squaredDistances, counts
}));
graph.add(new GPUPointCloudFeatures({
  positions, indices, counts, capacity: 8,
  normals, eigenvalues, shapes, status
}));`,
    preview: {
      kind: 'point-cloud',
      seed: 9126,
      caption: 'Points colored by planarity and normal direction.'
    },
    related: ['pointcloud-neighborhoods', 'pointcloud-statistical-filter'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-statistical-filter',
    symbol: 'GPUPointCloudStatisticalFilter',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Statistical Outlier Filter',
    summary:
      'Flags points whose mean neighbor distance exceeds the cloud mean by k standard deviations.',
    explanation:
      'Drops noise points before surfacing or registration. It consumes neighborhood distances and writes a row-preserving inlier mask, with no compaction.',
    dataTypes: {input: ['point-cloud'], output: ['point-cloud']},
    useCases: ['transform', 'find-patterns'],
    domains: ['three-d', 'remote-sensing'],
    params: [
      {
        name: 'squaredDistances',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Squared distances from GPUPointCloudNeighborhoods.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Neighbor counts from GPUPointCloudNeighborhoods.'
      },
      {
        name: 'capacity',
        type: 'number',
        required: true,
        description: 'Neighbor capacity of the input.'
      },
      {
        name: 'standardDeviations',
        type: 'number',
        default: '2',
        description: 'Threshold in standard deviations above the mean.'
      },
      {
        name: 'meanDistances',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Mean retained neighbor distance; -1 means none.'
      },
      {
        name: 'mask',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Original-row inlier flags.'
      }
    ],
    outputs: [
      {
        name: 'mask',
        type: "GraphDataView<'uint32'>",
        description: 'One inlier flag per original row.'
      },
      {
        name: 'meanDistances',
        type: "GraphDataView<'float32'>",
        description: 'Mean neighbor distance per point.'
      }
    ],
    example: `import {GPUPointCloudNeighborhoods, GPUPointCloudStatisticalFilter} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, indices, squaredDistances, counts, meanDistances, mask: views on graph
graph.add(new GPUPointCloudNeighborhoods({
  positions, radius: 2, capacity: 16, indices, squaredDistances, counts
}));
graph.add(new GPUPointCloudStatisticalFilter({
  squaredDistances, counts, capacity: 16,
  standardDeviations: 2, meanDistances, mask
}));`,
    preview: {
      kind: 'point-cloud',
      seed: 9127,
      caption: 'Sparse noise points rejected from a dense cloud.'
    },
    related: ['pointcloud-neighborhoods', 'pointcloud-voxel-merge'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-voxel-merge',
    symbol: 'GPUPointCloudVoxelMerge',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud Voxel Merge',
    summary:
      'Merges per-tile voxel counts, centroids and representatives with associative sufficient statistics.',
    explanation:
      'Combines tiled voxel downsampling into one global result. The caller certifies disjoint owned rows; halo copies are never inferred or deduplicated.',
    dataTypes: {input: ['point-cloud'], output: ['point-cloud']},
    useCases: ['summarize', 'transform'],
    domains: ['three-d'],
    params: [
      {
        name: 'partitionPolicy',
        type: "'disjoint-owned-rows'",
        required: true,
        description: "Must be 'disjoint-owned-rows'; the caller certifies it."
      },
      {
        name: 'parts',
        type: 'readonly {counts, centroids, representatives, sourceRowOffset, attribute?}}[]',
        required: true,
        description: 'Per-partition counts, centroids and representatives.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Merged point count per voxel.'
      },
      {
        name: 'centroids',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Merged centroid per voxel.'
      },
      {
        name: 'representatives',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Global representative point id per voxel.'
      },
      {
        name: 'attribute',
        type: "{means: GraphDataView<AttributeFormat>; counts: GraphDataView<'uint32'>}",
        description: 'Optional merged attribute means and counts.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Bit 1 count overflow, bit 2 representative id overflow.'
      }
    ],
    outputs: [
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Points per voxel across all parts.'
      },
      {
        name: 'centroids',
        type: "GraphDataView<'float32x3'>",
        description: 'Arithmetic mean position per voxel.'
      },
      {
        name: 'representatives',
        type: "GraphDataView<'uint32'>",
        description: 'Representative original id per voxel.'
      }
    ],
    example: `import {GPUPointCloudVoxelMerge} from '@luma-contributors/analysis/pointcloud-analysis';

// each part: {counts, centroids, representatives, sourceRowOffset} from one tile
graph.add(new GPUPointCloudVoxelMerge({
  partitionPolicy: 'disjoint-owned-rows',
  parts: [{...tileA, sourceRowOffset: 0}, {...tileB, sourceRowOffset: 500000}],
  counts, centroids, representatives, status
}));`,
    preview: {
      kind: 'point-cloud',
      seed: 9128,
      caption: 'Tile-wise voxel centroids merged into one grid.'
    },
    related: ['pointcloud-statistical-filter', 'pointcloud-neighborhoods'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-surfaces',
    symbol: 'GPUPointCloudSurfaces',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud Surfaces',
    summary:
      'Cell-minimum terrain (DTM), cell-maximum surface (DSM) and canopy height from classified points.',
    explanation:
      'Rasterizes a classified lidar cloud on a planar grid. Terrain is the minimum ground-class elevation per cell, surface the maximum of all points, and coverage tells you which cells are supported.',
    dataTypes: {input: ['point-cloud'], output: ['terrain', 'raster']},
    useCases: ['transform', 'measure'],
    domains: ['three-d', 'remote-sensing', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions.'
      },
      {
        name: 'classifications',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'uint32 class per point.'
      },
      {name: 'groundClass', type: 'number', default: '2', description: 'Class counted as ground.'},
      {
        name: 'grid',
        type: 'PointCloudRasterGrid',
        required: true,
        description: '{origin, cellSize, width, height} planar grid.'
      },
      {
        name: 'terrain',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Minimum ground elevation per cell.'
      },
      {
        name: 'surface',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Maximum elevation per cell.'
      },
      {
        name: 'canopy',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Surface minus terrain; negatives are preserved.'
      },
      {
        name: 'coverage',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Bit 1 has any point, bit 2 has ground.'
      }
    ],
    outputs: [
      {
        name: 'terrain',
        type: "GraphDataView<'float32'>",
        description: 'DTM per cell; zero if unsupported.'
      },
      {name: 'surface', type: "GraphDataView<'float32'>", description: 'DSM per cell.'},
      {name: 'canopy', type: "GraphDataView<'float32'>", description: 'Canopy height per cell.'},
      {name: 'coverage', type: "GraphDataView<'uint32'>", description: 'Support bits per cell.'}
    ],
    example: `import {GPUPointCloudSurfaces} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, classes, terrain, surface, canopy, coverage: views on graph
graph.add(new GPUPointCloudSurfaces({
  positions, classifications: classes,
  grid: {origin: [0, 0], cellSize: 1, width: 256, height: 256},
  terrain, surface, canopy, coverage
}));`,
    preview: {
      kind: 'terrain',
      seed: 9129,
      caption: 'Ground and canopy surfaces rasterized from points.'
    },
    related: ['pointcloud-height', 'pointcloud-ground', 'pointcloud-smrf'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-height',
    symbol: 'GPUPointCloudHeight',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Height Above Ground',
    summary: 'Height of each point above the containing terrain cell, preserving negative heights.',
    explanation:
      'Normalizes a cloud to terrain so trees, buildings and cars share one baseline. Sampling is cell-constant by default, with optional bilinear or inverse-distance interpolation.',
    dataTypes: {input: ['point-cloud', 'terrain'], output: ['point-cloud']},
    useCases: ['measure', 'transform'],
    domains: ['three-d', 'remote-sensing'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions.'
      },
      {
        name: 'grid',
        type: 'PointCloudRasterGrid',
        required: true,
        description: '{origin, cellSize, width, height} planar grid.'
      },
      {
        name: 'terrain',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Terrain elevation per cell.'
      },
      {
        name: 'coverage',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Nonzero cells are supported.'
      },
      {
        name: 'coverageBit',
        type: 'number',
        default: 'any nonzero',
        description: 'Bit to test; use 2 with GPUPointCloudSurfaces.'
      },
      {
        name: 'interpolation',
        type: "{method: 'bilinear'} | {method: 'idw'; radius: number; power?: number; minimumSupport?: number}",
        description: 'Bilinear or IDW sampling instead of cell-constant.'
      },
      {
        name: 'heights',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Signed height per point.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: '0 valid, 1 outside, 2 unsupported, 3 nonfinite.'
      }
    ],
    outputs: [
      {
        name: 'heights',
        type: "GraphDataView<'float32'>",
        description: 'Height above terrain per point.'
      },
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Validity code per point.'}
    ],
    example: `import {GPUPointCloudHeight} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, terrain, coverage (from GPUPointCloudSurfaces), heights, status: views on graph
graph.add(new GPUPointCloudHeight({
  positions,
  grid: {origin: [0, 0], cellSize: 1, width: 256, height: 256},
  terrain, coverage, coverageBit: 2,
  heights, status
}));`,
    preview: {
      kind: 'terrain',
      seed: 9130,
      caption: 'Point heights measured up from the terrain grid.'
    },
    related: ['pointcloud-surfaces', 'pointcloud-ground'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-ground',
    symbol: 'GPUPointCloudGround',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Progressive Ground Filter',
    summary:
      'Bounded progressive morphological filter that classifies ground points over growing square windows.',
    explanation:
      'Best for a quick ground classification from a cell-minimum surface. Each stage opens the surface with a larger window and rejects points that rise above a slope-based threshold; it is a PMF, not SMRF.',
    dataTypes: {input: ['point-cloud'], output: ['point-cloud']},
    useCases: ['classify', 'transform'],
    domains: ['three-d', 'remote-sensing', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions.'
      },
      {
        name: 'classifications',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Original uint32 classes.'
      },
      {
        name: 'grid',
        type: 'PointCloudRasterGrid',
        required: true,
        description: '{origin, cellSize, width, height} planar grid.'
      },
      {
        name: 'minimumSurface',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Minimum elevation per occupied cell.'
      },
      {
        name: 'radii',
        type: 'readonly number[]',
        required: true,
        description: 'Strictly increasing window radii in cells, 1 to 32.'
      },
      {
        name: 'initialThreshold',
        type: 'number',
        required: true,
        description: 'Starting elevation threshold.'
      },
      {
        name: 'slope',
        type: 'number',
        required: true,
        description: 'Terrain slope used to grow the threshold.'
      },
      {
        name: 'maximumThreshold',
        type: 'number',
        required: true,
        description: 'Upper limit of the threshold.'
      }
    ],
    moreRequired: ['coverage', 'groundMask', 'outputClassifications', 'status'],
    outputs: [
      {name: 'groundMask', type: "GraphDataView<'uint32'>", description: 'Ground flag per point.'},
      {
        name: 'outputClassifications',
        type: "GraphDataView<'uint32'>",
        description: 'Ground becomes LAS class 2; others keep or become 1.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Bit 1 unsupported, bit 2 incomplete window.'
      }
    ],
    example: `import {GPUPointCloudGround} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, classes, minimum, coverage, groundMask, outputClasses, status: views on graph
graph.add(new GPUPointCloudGround({
  positions, classifications: classes,
  grid: {origin: [0, 0], cellSize: 1, width: 256, height: 256},
  minimumSurface: minimum, coverage,
  radii: [1, 2, 4], initialThreshold: 0.2, slope: 0.1, maximumThreshold: 1,
  groundMask, outputClassifications: outputClasses, status
}));`,
    preview: {
      kind: 'terrain',
      seed: 9131,
      caption: 'Ground points separated from canopy and buildings.'
    },
    related: ['pointcloud-smrf', 'pointcloud-surfaces', 'pointcloud-height'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-smrf',
    symbol: 'GPUPointCloudSMRF',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'SMRF Ground Filter',
    summary:
      'Simple morphological filter following the PDAL SMRF stages to classify ground points.',
    explanation:
      'Best for production-style ground classification of airborne lidar. It runs diamond openings, low-outlier and net-cut steps and inpaints a provisional terrain before thresholding points.',
    dataTypes: {input: ['point-cloud'], output: ['point-cloud', 'terrain']},
    useCases: ['classify', 'transform'],
    domains: ['three-d', 'remote-sensing', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions.'
      },
      {
        name: 'classifications',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Original uint32 classes.'
      },
      {
        name: 'grid',
        type: 'PointCloudRasterGrid',
        required: true,
        description: '{origin, cellSize, width, height} planar grid.'
      },
      {name: 'slope', type: 'number', description: 'Terrain slope tolerance.'},
      {
        name: 'window',
        type: 'number',
        default: '18 * cellSize',
        description: 'Maximum physical window radius.'
      },
      {name: 'threshold', type: 'number', description: 'Elevation threshold above the surface.'},
      {name: 'cut', type: 'number', description: 'Net-cut spacing; zero disables.'},
      {name: 'groundClass', type: 'number', description: 'Class written for ground points.'}
    ],
    moreRequired: ['groundSurface', 'groundMask', 'outputClassifications', 'status'],
    outputs: [
      {
        name: 'groundSurface',
        type: "GraphDataView<'float32'>",
        description: 'Provisional inpainted terrain per cell.'
      },
      {name: 'groundMask', type: "GraphDataView<'uint32'>", description: 'Ground flag per point.'},
      {
        name: 'outputClassifications',
        type: "GraphDataView<'uint32'>",
        description: 'Updated classes per point.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: '0 valid, 1 outside, 2 no anchors, 4 excluded, 8 inpainted.'
      }
    ],
    example: `import {GPUPointCloudSMRF} from '@luma-contributors/analysis/pointcloud-analysis';

// positions, classes, groundSurface, groundMask, outputClasses, status: views on graph
graph.add(new GPUPointCloudSMRF({
  positions, classifications: classes,
  grid: {origin: [0, 0], cellSize: 1, width: 256, height: 256},
  slope: 0.15, threshold: 0.5, window: 16,
  groundSurface, groundMask, outputClassifications: outputClasses, status
}));`,
    preview: {kind: 'terrain', seed: 9132, caption: 'SMRF ground surface under a vegetated cloud.'},
    related: ['pointcloud-ground', 'pointcloud-surfaces', 'pointcloud-height'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-icp',
    symbol: 'GPUPointCloudICP',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud ICP',
    summary:
      'Bounded GPU iterative closest point with distance rejection, point-to-plane and robust weighting.',
    explanation:
      'Registers one scan onto another. Each iteration matches nearest points within maximumDistance and solves a rigid transform by Horn least squares; history reports inliers and RMSE per step.',
    dataTypes: {input: ['point-cloud'], output: ['table']},
    useCases: ['detect-change', 'transform'],
    domains: ['three-d', 'remote-sensing'],
    params: [
      {
        name: 'source',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions to move.'
      },
      {
        name: 'target',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Packed float32x3 positions to match.'
      },
      {
        name: 'maximumDistance',
        type: 'number',
        required: true,
        description: 'Correspondences farther than this are rejected.'
      },
      {
        name: 'metric',
        type: "'point-to-point' | 'point-to-plane'",
        default: "'point-to-point'",
        description: "'point-to-point' or 'point-to-plane'."
      },
      {
        name: 'targetNormals',
        type: "GraphDataView<'float32x3'>",
        description: 'Normals of target, for point-to-plane.'
      },
      {
        name: 'robust',
        type: "{ method: 'huber' | 'tukey'; scale: number; }",
        description: 'Huber or Tukey reweighting and its scale.'
      },
      {name: 'iterations', type: 'number', description: 'Iteration budget.'},
      {name: 'targetGrid', type: 'PointCloudGrid', description: 'Exact static target index.'}
    ],
    moreRequired: ['transform', 'history', 'inlierMask', 'status'],
    outputs: [
      {
        name: 'transform',
        type: "GraphDataView<'float32'>",
        description: 'Column-major 4x4 rigid transform, source to target.'
      },
      {
        name: 'history',
        type: "GraphDataView<'float32x3'>",
        description: 'Per iteration: inliers, RMSE, translation step.'
      },
      {
        name: 'inlierMask',
        type: "GraphDataView<'uint32'>",
        description: 'Final correspondence flags per source point.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: '0 converged, 1 budget exhausted, 2 few matches.'
      }
    ],
    example: `import {GPUPointCloudICP} from '@luma-contributors/analysis/pointcloud-analysis';

// source, target (shared local frame), transform, history, inlierMask, status: views on graph
graph.add(new GPUPointCloudICP({
  source, target, maximumDistance: 1, iterations: 20,
  transform, history, inlierMask, status
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {kind: 'point-cloud', seed: 9133, caption: 'Two scans converging onto each other.'},
    related: ['pointcloud-change', 'pointcloud-local-positions'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-change',
    symbol: 'GPUPointCloudChange',
    kind: 'class',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Point Cloud Change (M3C2)',
    summary:
      'M3C2-style signed change along normals with confidence bounds, roughness and registration error.',
    explanation:
      'Measures erosion, deformation or growth between two aligned scans. For each core point it compares cylindrical neighborhoods along the normal and reports the displacement with a 95 percent detection bound.',
    dataTypes: {input: ['point-cloud'], output: ['table']},
    useCases: ['detect-change', 'measure'],
    domains: ['three-d', 'remote-sensing', 'hydrology'],
    params: [
      {
        name: 'reference',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Earlier cloud, float32x3.'
      },
      {
        name: 'comparison',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Later cloud, float32x3.'
      },
      {
        name: 'cores',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Core points where change is measured.'
      },
      {
        name: 'normals',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'Unit normal per core point.'
      },
      {
        name: 'alignment',
        type: "'already-aligned' | { transform: GraphDataView<'float32'>; }",
        required: true,
        description: "'already-aligned' or a comparison-to-reference transform."
      },
      {name: 'radius', type: 'number', required: true, description: 'Cylinder radius.'},
      {
        name: 'halfDepth',
        type: 'number',
        required: true,
        description: 'Cylinder half-length along the normal.'
      },
      {
        name: 'registrationError',
        type: 'number',
        description: 'Registration uncertainty added to the bound.'
      }
    ],
    moreRequired: ['changes', 'roughness', 'counts', 'status'],
    outputs: [
      {
        name: 'changes',
        type: "GraphDataView<'float32x3'>",
        description: 'Signed change, lower bound, upper bound.'
      },
      {
        name: 'roughness',
        type: "GraphDataView<'float32x3'>",
        description: 'Reference roughness, comparison roughness, pooled.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Reference and comparison point counts per core.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: '0 valid, 1 invalid core, 2 low support.'
      }
    ],
    example: `import {GPUPointCloudChange} from '@luma-contributors/analysis/pointcloud-analysis';

// reference, comparison, cores, normals, changes, roughness, counts, status: views on graph
graph.add(new GPUPointCloudChange({
  reference, comparison, cores, normals, alignment: 'already-aligned',
  radius: 1, halfDepth: 2, registrationError: 0.02,
  changes, roughness, counts, status
}));`,
    preview: {
      kind: 'point-cloud',
      seed: 9134,
      caption: 'Signed change colored along surface normals.'
    },
    related: ['pointcloud-icp', 'pointcloud-features', 'pointcloud-local-positions'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'pointcloud-local-positions',
    symbol: 'makePointCloudLocalPositions',
    kind: 'function',
    importPath: '@luma-contributors/analysis/pointcloud-analysis',
    title: 'Local Origin Positions',
    summary:
      'Float32 positions relative to a float64 origin, so projected coordinates keep millimetre precision.',
    explanation:
      'Best before uploading any cloud with large coordinates. Absolute float32 eastings lose about 6 cm at 1e6 m; subtracting a shared origin in float64 first avoids it, and every cloud compared together needs the same origin.',
    dataTypes: {input: ['point-cloud'], output: ['point-cloud']},
    useCases: ['transform'],
    domains: ['three-d', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'ArrayLike<number>',
        required: true,
        description: 'Interleaved xyz coordinates in any ArrayLike<number>.'
      },
      {
        name: 'origin',
        type: 'readonly [number, number, number]',
        required: true,
        description: 'Shared [x, y, z] origin, subtracted in float64.'
      }
    ],
    outputs: [
      {
        name: 'local',
        type: 'Float32Array',
        description: 'Interleaved xyz positions relative to origin.'
      }
    ],
    example: `import {getPointCloudLocalOrigin, makePointCloudLocalPositions} from '@luma-contributors/analysis/pointcloud-analysis';

// absolute: Float64Array of interleaved projected x, y, z
const origin = getPointCloudLocalOrigin(absolute);
const local = makePointCloudLocalPositions(absolute, origin);
// upload local as float32x3; add origin back when reading results`,
    preview: {
      kind: 'point-cloud',
      seed: 9135,
      caption: 'A far-from-origin cloud recentered for float32.'
    },
    related: ['pointcloud-icp', 'pointcloud-change', 'pointcloud-neighborhoods'],
    reference: 'api/pointcloud-analysis',
    maturity: 'stable'
  },
  {
    id: 'datum-grid-transform',
    symbol: 'GPUDatumGridTransform',
    kind: 'class',
    importPath: '@luma-contributors/analysis/geospatial-analysis',
    title: 'Datum Grid Transform',
    summary:
      'Bilinear horizontal and vertical datum shifts from caller-decoded grids, forward or inverse.',
    explanation:
      'Moves positions between datums or vertical references, as NADCON or geoid grids define. The first grid covering a point wins, nodata fails the row, and inverse runs a bounded iteration.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['transform'],
    domains: ['cartography', 'three-d'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'float32x3: longitude, latitude (degrees), height.'
      },
      {
        name: 'samples',
        type: "GraphDataView<'float32x3'>",
        required: true,
        description: 'float32x3 shifts: longitude, latitude, height.'
      },
      {
        name: 'validity',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Zero marks nodata.'
      },
      {
        name: 'grids',
        type: 'readonly GPUDatumGrid[]',
        required: true,
        description: 'Ordered regular grids; the first covering grid wins.'
      },
      {
        name: 'operation',
        type: "'horizontal' | 'vertical' | 'combined'",
        required: true,
        description: "'horizontal', 'vertical' or 'combined'."
      },
      {name: 'direction', type: "'forward' | 'inverse'", description: "'forward' or 'inverse'."},
      {name: 'maximumIterations', type: 'number', description: 'Inverse iteration bound.'},
      {
        name: 'tolerance',
        type: 'number',
        default: '1e-5',
        description: 'Inverse residual tolerance in degrees.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x3'>", description: 'Shifted positions.'},
      {
        name: 'positionsLow',
        type: "GraphDataView<'float32x3'>",
        description: 'Optional residual words keeping sub-float32 detail.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'success, invalid, noCoverage, nodata or inverseExhausted.'
      }
    ],
    example: `import {GPUDatumGridTransform} from '@luma-contributors/analysis/geospatial-analysis';

// positions, samples, validity, shiftedPositions, statusWords: views on graph
graph.add(new GPUDatumGridTransform({
  positions, samples, validity,
  grids: [{width: 65, height: 31, sampleOffset: 0, origin: [-96, 35.5], spacing: [0.25, 0.25]}],
  operation: 'horizontal', direction: 'forward',
  output: {positions: shiftedPositions, status: statusWords}
}));`,
    preview: {
      kind: 'raster',
      seed: 9136,
      caption: 'Magnified NAD27–NAD83 shifts across the Midwest, colored by distance.'
    },
    related: ['local-frame-projection'],
    reference: 'api/geospatial',
    maturity: 'stable'
  },
  {
    id: 'point-buffer',
    symbol: 'GPUPointBuffer',
    kind: 'class',
    importPath: '@luma-contributors/analysis/geospatial-analysis',
    title: 'Point Buffer',
    summary:
      'Generates inscribed circle vertices around points, in geographic or planar coordinates.',
    explanation:
      'Draws radius circles around many points with a per-frame radius. Geographic offsets are degrees you add to float64 centers, so small radii keep precision.',
    dataTypes: {input: ['points'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'urban'],
    params: [
      {
        name: 'points',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'float32x2 centers: degrees or planar units.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Radius from getGPUPointBufferParameterValues.'
      },
      {
        name: 'offsets',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'float32x2 ring vertex offsets, verticesPerRing per point.'
      },
      {
        name: 'verticesPerRing',
        type: 'number',
        required: true,
        description: 'Vertices per ring, at least 3.'
      },
      {
        name: 'coordinateSystem',
        type: "'geographic' | 'planar'",
        required: true,
        description: "'geographic' or 'planar'."
      }
    ],
    outputs: [
      {
        name: 'offsets',
        type: "GraphDataView<'float32x2'>",
        description: 'Vertex offsets from each center.'
      }
    ],
    example: `import {GPUPointBuffer, getGPUPointBufferParameterValues} from '@luma-contributors/analysis/geospatial-analysis';

// points, parameters, offsets: views on graph
graph.add(new GPUPointBuffer({
  points, parameters, offsets,
  verticesPerRing: 32, coordinateSystem: 'geographic'
}));
// 250 m radius
const radiusValues = getGPUPointBufferParameterValues(250);`,
    preview: {kind: 'buffers', seed: 9137, caption: 'Circles of one radius around every point.'},
    related: ['buffer-selection', 'buffer-surface'],
    reference: 'api/geospatial',
    maturity: 'stable'
  },
  {
    id: 'computation-recorder',
    symbol: 'GPUComputationRecorder',
    kind: 'class',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Computation Recorder',
    summary:
      'Records a command graph under construction, with scopes, nodes and sources, for a computation record.',
    explanation:
      'Captures what a graph does so it can be drawn and replayed on the lens pages. Wrap node creation in withScope and describe the compiled graph afterwards.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'sourceText',
        type: 'boolean',
        description: 'Keep full source text on recorded sources.'
      }
    ],
    outputs: [
      {
        name: 'recorder',
        type: 'GPUComputationRecorder',
        description: 'Pass to describeGPUCommandGraph to get graph input.'
      }
    ],
    example: `import {GPUComputationRecorder, createGPUComputationRecord, describeGPUCommandGraph, getGPUComputationAdapter} from '@luma-contributors/analysis/computation-record';

const recorder = new GPUComputationRecorder({sourceText: true});
recorder.withScope({namespace: 'solve', op: 'relax'}, () => {
  // add the real nodes here: recorder.addNode(graph, node)
});
const compiled = await graph.compileAsync();
const record = createGPUComputationRecord({
  id: 'relaxation', adapter: getGPUComputationAdapter(device, 'portable'),
  domains: ['gpu-core'], graphs: [describeGPUCommandGraph(compiled, {recorder})]
});`,
    preview: {
      kind: 'pipeline',
      seed: 9138,
      caption: 'Nodes of a command graph captured as they are added.'
    },
    related: ['computation-record', 'snapshot-tap', 'snapshot-arena'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'computation-record',
    symbol: 'createGPUComputationRecord',
    kind: 'function',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Computation Record',
    summary:
      'Assembles a portable record of graphs, sources, formulas and decisions, deduplicating sources by id.',
    explanation:
      'Publishes a reproducible description of a run: which graphs ran on which adapter, with which data version and seed. Validate it with validateGPUComputationRecord before sharing.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {name: 'id', type: 'string', required: true, description: 'Record id.'},
      {
        name: 'adapter',
        type: 'GPUComputationAdapter',
        required: true,
        description: 'Adapter description from getGPUComputationAdapter.'
      },
      {
        name: 'domains',
        type: 'readonly string[]',
        required: true,
        description: 'Domains the computation belongs to.'
      },
      {
        name: 'graphs',
        type: 'readonly GPUComputationGraphInput[]',
        required: true,
        description: 'Graph descriptions from describeGPUCommandGraph.'
      },
      {
        name: 'formulas',
        type: 'readonly GPUComputationFormulaRef[]',
        description: 'Formula references.'
      },
      {
        name: 'sources',
        type: 'readonly GPUComputationSourceRef[]',
        description: 'Source references, deduplicated by id.'
      },
      {name: 'datasetVersion', type: 'string', description: 'Version of the input data.'},
      {name: 'seed', type: 'string | number', description: 'Random seed of the run.'}
    ],
    outputs: [
      {name: 'schema', type: 'string', description: 'Schema identifier.'},
      {name: 'id', type: 'string', description: 'Record id.'},
      {name: 'runId', type: 'string', description: 'Run identity hash; sidecars join on it.'},
      {name: 'createdAt', type: 'string', description: 'ISO 8601 creation time.'},
      {
        name: 'adapter',
        type: 'GPUComputationAdapter',
        description: 'Adapter that ran the computation.'
      },
      {name: 'domains', type: 'readonly string[]', description: 'Domain packs of the record.'},
      {
        name: 'graphs',
        type: 'readonly GPUComputationGraph[]',
        description: 'Graph forms, intent to encoded.'
      },
      {
        name: 'axes, refs, decisions, choices, workflow',
        type: 'mixed views',
        description: 'Further result fields; see the reference.'
      }
    ],
    example: `import {createGPUComputationRecord, describeGPUCommandGraph, getGPUComputationAdapter, validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';

const record = createGPUComputationRecord({
  id: 'relaxation',
  adapter: getGPUComputationAdapter(device, 'portable'),
  domains: ['gpu-core'],
  graphs: [describeGPUCommandGraph(compiled, {recorder})],
  datasetVersion: 'example-v1',
  seed: 42
});
const report = validateGPUComputationRecord(record);`,
    preview: {
      kind: 'pipeline',
      seed: 9139,
      caption: 'A graph, its adapter and its data version in one record.'
    },
    related: ['computation-recorder', 'computation-metrics', 'computation-captures'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'computation-metrics',
    symbol: 'createGPUComputationMetrics',
    kind: 'function',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Computation Metrics',
    summary:
      'Builds per-node nanosecond timing relative to the earliest GPU begin, plus counters and series.',
    explanation:
      'Attaches GPU and CPU timings to a record so a lens page can animate the run. Timestamps disable pass coalescing, which the encoding flags record.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['measure'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'runId',
        type: 'string',
        required: true,
        description: 'Run id shared with the record.'
      },
      {name: 'graph', type: 'string', required: true, description: 'Id of the timed graph.'},
      {
        name: 'timing',
        type: 'GPUComputationTimingInput',
        required: true,
        description: 'Per-node CPU encode and GPU begin and end times.'
      },
      {
        name: 'encoding',
        type: 'GPUComputationEncodingFlags',
        required: true,
        description: 'Flags: coalesced, timestamped, taps, lensTwin.'
      },
      {name: 'adapter', type: 'GPUComputationAdapter', description: 'Adapter description.'},
      {
        name: 'series',
        type: 'readonly GPUComputationSeries[]',
        description: 'Optional per-iteration series.'
      },
      {
        name: 'counters',
        type: 'Readonly<Record<string, Readonly<Record<string, number>>>>',
        description: 'Optional counters per node.'
      },
      {
        name: 'passIndices',
        type: 'Readonly<Record<string, number>>',
        description: 'Node id to physical pass index.'
      }
    ],
    outputs: [
      {name: 'schema', type: 'string', description: 'Schema identifier.'},
      {name: 'runId', type: 'string', description: 'Run the metrics measure.'},
      {name: 'graph', type: 'string', description: 'Id of the measured graph.'},
      {name: 'timebase', type: '"ns" | "ms"', description: 'Unit of GPU begin and end times.'},
      {
        name: 'adapter',
        type: 'GPUComputationAdapter',
        description: 'Optional adapter description.'
      },
      {
        name: 'encoding',
        type: 'GPUComputationEncodingFlags',
        description: 'Observer effects in force.'
      },
      {
        name: 'nodes',
        type: 'Record<string, GPUComputationNodeMetrics>',
        description: 'Timings and counters keyed by node id.'
      },
      {
        name: 'series',
        type: 'readonly GPUComputationSeries[]',
        description: 'Optional per-iteration counter series.'
      }
    ],
    example: `import {createGPUComputationMetrics} from '@luma-contributors/analysis/computation-record';

const metrics = createGPUComputationMetrics({
  runId: record.runId,
  graph: 'compiled',
  encoding: {coalesced: false, timestamped: true, taps: 0, lensTwin: true},
  timing: {nodes: [{
    id: 'init', cpuEncodeTimeMilliseconds: 0.1,
    gpuBeginNanoseconds: 2000, gpuEndNanoseconds: 9000
  }]}
});`,
    preview: {
      kind: 'time-series',
      seed: 9140,
      caption: 'Node durations laid out along a GPU timeline.'
    },
    related: ['computation-record', 'computation-captures'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'computation-captures',
    symbol: 'createGPUComputationCaptures',
    kind: 'function',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Computation Captures',
    summary:
      'Builds the captures sidecar of a record: snapshots, stamps and replays, with empty arrays filled in.',
    explanation:
      'Bundles the data frames you read back from snapshot taps with the run id. Validate record and captures together with validateGPUComputationSidecars.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'runId',
        type: 'string',
        required: true,
        description: 'Run id shared with the record.'
      },
      {
        name: 'snapshots',
        type: 'readonly GPUComputationCapture[]',
        description: 'Captured resource frames.'
      },
      {
        name: 'stamps',
        type: 'readonly GPUComputationCapture[]',
        description: 'Captured timestamps.'
      },
      {
        name: 'replays',
        type: 'readonly GPUComputationCapture[]',
        description: 'Captured replay frames.'
      }
    ],
    outputs: [
      {name: 'schema', type: 'string', description: 'Schema identifier.'},
      {name: 'runId', type: 'string', description: 'Run the captures belong to.'},
      {
        name: 'snapshots',
        type: 'readonly GPUComputationCapture[]',
        description: 'Resource copies at taps and steps.'
      },
      {
        name: 'stamps',
        type: 'readonly GPUComputationCapture[]',
        description: 'Per-datum stamps such as settle round.'
      },
      {
        name: 'replays',
        type: 'readonly GPUComputationCapture[]',
        description: 'GPU replays of folded state.'
      }
    ],
    example: `import {createGPUComputationCaptures, validateGPUComputationSidecars} from '@luma-contributors/analysis/computation-record';

// snapshots: captures read back from a GPUSnapshotArena
const captures = createGPUComputationCaptures({runId: record.runId, snapshots});
const report = validateGPUComputationSidecars(record, {captures});`,
    preview: {
      kind: 'pipeline',
      seed: 9141,
      caption: 'Snapshot frames attached to the run they came from.'
    },
    related: ['computation-record', 'snapshot-arena', 'snapshot-tap'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'snapshot-arena',
    symbol: 'GPUSnapshotArena',
    kind: 'class',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Snapshot Arena',
    summary: 'A bounded readback buffer that collects snapshot frames and maps once.',
    explanation:
      'Observes intermediate state of a graph without stalling it per frame. Slots name what is copied; capacity bounds how many frames fit before one mapped read.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {name: 'id', type: 'string', description: 'Arena id.'},
      {
        name: 'slots',
        type: 'readonly GPUSnapshotSlot[]',
        required: true,
        description: 'Named slots: name, byteLength and elementType.'
      },
      {name: 'capacity', type: 'number', required: true, description: 'Maximum number of frames.'},
      {name: 'maximumByteLength', type: 'number', description: 'Optional cap on the buffer size.'}
    ],
    outputs: [
      {name: 'frames', type: 'GPUSnapshotFrame[]', description: 'Frames read back after the run.'}
    ],
    example: `import {GPUSnapshotArena, GPUSnapshotTap} from '@luma-contributors/analysis/computation-record';

const arena = new GPUSnapshotArena(device, {
  id: 'residual',
  slots: [{name: 'residual', byteLength: 4096, elementType: 'f32'}],
  capacity: 8
});
graph.add(new GPUSnapshotTap({
  id: 'tap-residual', arena, views: {residual: residualView}, afterNode: 'relax'
}));`,
    preview: {
      kind: 'pipeline',
      seed: 9142,
      caption: 'Frames collected in one buffer along a graph.'
    },
    related: ['snapshot-tap', 'computation-captures'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'snapshot-tap',
    symbol: 'GPUSnapshotTap',
    kind: 'class',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Snapshot Tap',
    summary:
      'A copy node that snapshots views of a graph into an arena at a chosen point in the sequence.',
    explanation:
      'Watches a value evolve across iterations, such as a residual. Place it after a node, give it views, and each encoding adds one frame to the arena.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'id',
        type: 'string',
        required: true,
        description: 'Node id; also the default frame label.'
      },
      {
        name: 'arena',
        type: 'GPUSnapshotArena',
        required: true,
        description: 'Arena receiving one frame per encoding.'
      },
      {
        name: 'views',
        type: 'Readonly<Record<string, GraphDataView | GraphBufferHandle>>',
        required: true,
        description: 'Views to copy, keyed by slot name.'
      },
      {name: 'afterNode', type: 'string', description: 'Node the observation follows.'},
      {name: 'label', type: 'string', description: 'Frame description.'},
      {
        name: 'iteration',
        type: 'number',
        default: 'frame index',
        description: 'Frame iteration coordinate.'
      },
      {
        name: 'recorder',
        type: 'GPUComputationRecorder',
        description: 'Records the tap in a computation record.'
      }
    ],
    outputs: [
      {
        name: 'frame',
        type: 'GPUSnapshotFrame',
        description: 'One frame per encoding, in the arena.'
      }
    ],
    example: `import {GPUSnapshotArena, GPUSnapshotTap} from '@luma-contributors/analysis/computation-record';

// arena: GPUSnapshotArena; residualView: view on graph
graph.add(new GPUSnapshotTap({
  id: 'tap-residual',
  arena,
  views: {residual: residualView},
  afterNode: 'relax',
  label: 'residual after relax',
  recorder
}));`,
    preview: {
      kind: 'pipeline',
      seed: 9143,
      caption: 'A tap copying a view between two graph nodes.'
    },
    related: ['snapshot-arena', 'computation-recorder', 'computation-captures'],
    reference: 'api/computation-record',
    maturity: 'stable'
  },
  {
    id: 'transfer-mask',
    symbol: 'GPUTransferMask',
    kind: 'class',
    importPath: '@luma-contributors/analysis/computation-record',
    title: 'Transfer Mask',
    summary: 'Clears and scatters a selection mask across a GPU-resident index or CSR relation.',
    explanation:
      'Carries a selection from one table to another, such as picked tracts to their parcels. Mode any marks reached targets; count tallies selected incoming entries.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['relate'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'id',
        type: 'string',
        required: true,
        description: 'Prefix for the clear and scatter node ids.'
      },
      {
        name: 'source',
        type: 'GraphDataView<GPUVectorFormat>',
        required: true,
        description: 'uint32 flags over the source key space.'
      },
      {
        name: 'relation',
        type: 'GPUTransferMaskRelation',
        required: true,
        description: "Index {layout: 'index', targets} or CSR {layout: 'csr', offsets, neighbors}."
      },
      {
        name: 'target',
        type: 'GraphDataView<GPUVectorFormat>',
        required: true,
        description: 'uint32 output over the target key space; cleared first.'
      },
      {
        name: 'mode',
        type: "'any' | 'count'",
        description: "'any' writes one; 'count' counts incoming selected entries."
      },
      {
        name: 'recorder',
        type: 'GPUComputationRecorder',
        description: 'Annotates the nodes for a computation record.'
      }
    ],
    outputs: [
      {
        name: 'target',
        type: 'GraphDataView<GPUVectorFormat>',
        description: 'Selection flags or counts per target key.'
      }
    ],
    example: `import {GPUTransferMask} from '@luma-contributors/analysis/computation-record';

// parcelFlags, tractsOfParcel, tractFlags: uint32 views on graph
graph.add(new GPUTransferMask({
  id: 'carry-selection',
  source: parcelFlags,
  relation: {layout: 'index', targets: tractsOfParcel},
  target: tractFlags,
  mode: 'any'
}));`,
    preview: {
      kind: 'pipeline',
      seed: 9144,
      caption: 'A selection carried across a relation to another table.'
    },
    related: ['key-join', 'snapshot-tap'],
    reference: 'api/computation-record',
    maturity: 'stable'
  }
];
