// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "space-time-models": Gaussian-process likelihoods (Vecchia), space-time kriging,
// space-time local statistics, geographically and temporally weighted regression, and the
// validation and scoring of predictions (area of applicability, conformal intervals, scores).

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

const param = (
  name: string,
  type: string,
  description: string,
  extra: {required?: boolean; default?: string} = {}
): ParamDoc => ({name, type, description, ...extra});

const output = (name: string, type: string, description: string): OutputDoc => ({
  name,
  type,
  description
});

/** A runnable-shaped example. `output(id, format, length)` stands for a caller-owned storage view. */
function makeExample(symbols: string[], body: string[]): string {
  return [
    `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    `import {${symbols.join(', ')}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    '// output(id, format, length): caller-owned storage views',
    ...body,
    'const encoder = device.createCommandEncoder();',
    '(await graph.compileAsync()).encode(encoder, {parameters: undefined});',
    'device.submit(encoder.finish());'
  ].join('\n');
}

const F32 = "GraphDataView<'float32'>";
const U32 = "GraphDataView<'uint32'>";
const POSITIONS = param('positions', "GraphDataView<'float32x2'>", 'Planar position per row.', {
  required: true
});
const OBSERVATIONS = param(
  'observations',
  "GraphDataView<'float32x4'>",
  'Rows [value, error variance, reserved, validity]; values mean zero.',
  {required: true}
);
const NEIGHBORHOODS = param(
  'neighborhoods',
  U32,
  'Earlier rows each row conditions on, from buildVecchiaConditioningSets.',
  {required: true}
);
const NEIGHBOR_COUNT = param('neighborCount', 'number', 'Largest conditioning set, 1 to 32.', {
  required: true
});
const TEMPORAL_CONTEXT = param(
  'temporalContext',
  'GPUTemporalContext',
  'Meaning of times: float32-relative ticks after an epoch.',
  {required: true}
);
const TIMES = param('times', F32, 'Relative float32 time per row.', {
  required: true
});
const SITE_TERMS = output('siteTerms', F32, "Each site's term of the log-likelihood sum.");
const LIKELIHOOD_OUTPUTS = [
  output('conditionals', "GraphDataView<'float32x4'>", 'Conditional mean and variance per site.'),
  output('conditionalStatus', U32, 'Per site: 0 when the conditional solve succeeded.'),
  SITE_TERMS,
  output('siteStatus', U32, 'Per site: 0 valid, nonzero excluded or failed.'),
  output('logLikelihood', F32, 'Total as a [high, low] float32 pair.'),
  output('status', U32, 'Included, excluded and failed site counts.')
];
const CALIBRATION = param(
  'calibration',
  'GPUConformalCalibration',
  'Held-out observations and predictions of the calibration rows.',
  {required: true}
);
const QUERY = param('query', 'GPUConformalQuery', 'Predictions (and scales) of the query rows.', {
  required: true
});
const CONFORMAL_PARAMETERS = param(
  'parameters',
  F32,
  'Per-frame miscoverage, from getGPUConformalPredictionParameterValues.',
  {required: true}
);

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------- Gaussian-process likelihood
  {
    id: 'vecchia-log-likelihood',
    symbol: 'GPUVecchiaLogLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Vecchia log-likelihood',
    summary:
      'Gaussian-process log-likelihood of point data, each site conditioned on a few earlier neighbors.',
    explanation:
      'Fits covariance models to thousands of sites without an n x n solve, as GpGp does. Order the sites max-min, build conditioning sets once, then evaluate an exponential or Matern 3/2 covariance (nugget, sill, range) per frame. Subtract a trend first: the values are taken as mean zero.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      POSITIONS,
      OBSERVATIONS,
      param('neighborhoodOffsets', U32, 'CSR offsets of the conditioning sets, rows + 1.', {
        required: true
      }),
      NEIGHBORHOODS,
      NEIGHBOR_COUNT,
      param(
        'parameters',
        F32,
        'Per-frame covariance from getGPUVecchiaLogLikelihoodParameterValues.',
        {required: true}
      )
    ],
    outputs: LIKELIHOOD_OUTPUTS,
    example: makeExample(
      [
        'GPUVecchiaLogLikelihood',
        'buildVecchiaConditioningSets',
        'getVecchiaMaxMinOrdering',
        'getGPUVecchiaLogLikelihoodParameterValues'
      ],
      [
        'const order = getVecchiaMaxMinOrdering(positionData);',
        'const sets = buildVecchiaConditioningSets({positions: positionData, neighborCount: 10, order});',
        "parameterBuffer.write(getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', range: 1.5, sill: 1.7, nugget: 0.05}));",
        'graph.add(',
        '  new GPUVecchiaLogLikelihood({',
        '    positions, observations, // [residual, 0, 0, 1] per site',
        "    neighborhoodOffsets: graph.importGPUData('offsets', sets.offsets),",
        "    neighborhoods: graph.importGPUData('neighbors', sets.neighbors),",
        '    neighborCount: 10,',
        '    parameters: parameterBuffer.importToGraph(graph),',
        '    output: likelihoodOutput // conditionals, siteTerms, logLikelihood, status ...',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1301,
      caption: 'Helene log rain minus a planar trend; right, the (range, sill) surface.'
    },
    related: ['vecchia-fisher-scoring', 'vecchia-space-time-log-likelihood', 'space-time-kriging'],
    reference: 'api/gpu-spatial-analysis/vecchia-likelihood',
    maturity: 'stable'
  },
  {
    id: 'vecchia-fisher-scoring',
    symbol: 'GPUVecchiaFisherScoring',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Vecchia Fisher scoring',
    summary:
      'Vecchia log-likelihood with its score and expected information, for fitting covariance parameters.',
    explanation:
      'For an exponential or Matern 3/2 covariance, one evaluation gives the gradient in log sill, log range and log nugget and the Fisher information. A scoring step adds the information inverse times the score, as GpGp `fit_model` does. Iterate until the step is small.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      POSITIONS,
      OBSERVATIONS,
      param('neighborhoodOffsets', U32, 'CSR offsets of the conditioning sets.', {required: true}),
      NEIGHBORHOODS,
      NEIGHBOR_COUNT,
      param(
        'parameters',
        F32,
        'Per-frame covariance from getGPUVecchiaLogLikelihoodParameterValues.',
        {
          required: true
        }
      )
    ],
    outputs: [
      output('siteTerms', F32, 'Per site: likelihood term, score and information cells.'),
      output('siteStatus', U32, 'Per site: 0 valid, nonzero excluded or failed.'),
      output(
        'totals',
        F32,
        'Summed cells as [high, low] pairs; read with getVecchiaFisherEvaluation.'
      ),
      output('status', U32, 'Included, excluded and failed site counts.')
    ],
    example: makeExample(
      ['GPUVecchiaFisherScoring', 'getVecchiaFisherEvaluation'],
      [
        'graph.add(',
        '  new GPUVecchiaFisherScoring({',
        '    positions, observations, neighborhoodOffsets, neighborhoods, // as for the log-likelihood',
        '    neighborCount: 10,',
        '    parameters, // getGPUVecchiaLogLikelihoodParameterValues(covariance)',
        '    output: {',
        "      siteTerms: output('terms', 'float32', rowCount * GPU_VECCHIA_FISHER_CELL_COUNT),",
        "      siteStatus: output('site-status', 'uint32', rowCount),",
        "      totals: output('totals', 'float32', 2 * GPU_VECCHIA_FISHER_CELL_COUNT),",
        "      status: output('status', 'uint32', 3)",
        '    }',
        '  })',
        ');',
        '// after readback: const {logLikelihood, score, information} = getVecchiaFisherEvaluation(totals, status);'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1302,
      caption: 'The arrow is one Fisher scoring step.'
    },
    related: ['vecchia-log-likelihood', 'vecchia-space-time-log-likelihood'],
    reference: 'api/gpu-spatial-analysis/vecchia-likelihood',
    maturity: 'stable'
  },
  {
    id: 'vecchia-space-time-log-likelihood',
    symbol: 'GPUVecchiaSpaceTimeLogLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time Vecchia log-likelihood',
    summary:
      'Vecchia log-likelihood of space-time observations under a separable or metric space-time model.',
    explanation:
      'The space-time version of the Vecchia likelihood: rows carry a time, and conditioning sets are built in scaled space-time. Evaluate spatial and temporal ranges per frame to fit a space-time covariance.',
    dataTypes: {input: ['points', 'time-series'], output: ['table']},
    useCases: ['model', 'interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      TEMPORAL_CONTEXT,
      POSITIONS,
      TIMES,
      OBSERVATIONS,
      NEIGHBORHOODS,
      NEIGHBOR_COUNT,
      param(
        'parameters',
        F32,
        'Per-frame model from getGPUVecchiaSpaceTimeLogLikelihoodParameterValues.',
        {required: true}
      ),
      param('spatialContext', 'GPUSpatialContext | undefined', 'Planar coordinates only.')
    ],
    moreRequired: ['neighborhoodOffsets'],
    outputs: LIKELIHOOD_OUTPUTS,
    example: makeExample(
      ['GPUVecchiaSpaceTimeLogLikelihood', 'getGPUVecchiaSpaceTimeLogLikelihoodParameterValues'],
      [
        'parameterBuffer.write(getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(spaceTimeModel));',
        'graph.add(',
        '  new GPUVecchiaSpaceTimeLogLikelihood({',
        '    temporalContext, // float32-relative days',
        '    positions, times, observations,',
        '    neighborhoodOffsets, neighborhoods, // conditioning sets in scaled space-time',
        '    neighborCount: 10,',
        '    parameters: parameterBuffer.importToGraph(graph),',
        '    output: likelihoodOutput',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1303,
      caption: 'Two Helene days, each detrended: nearly uncorrelated, best at shortest range.'
    },
    related: ['vecchia-log-likelihood', 'space-time-kriging'],
    reference: 'api/gpu-spatial-analysis/vecchia-likelihood',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- space-time kriging
  {
    id: 'space-time-kriging',
    symbol: 'GPUSpaceTimeKriging',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time kriging',
    summary: 'Predicts a value and its variance at any place and time from space-time samples.',
    explanation:
      'Kriging with a space-time covariance (the gstat `krigeST` idea): nearby samples from other times count by the temporal range. Each query keeps up to 32 neighbors, chosen by metric, octant or staged search. Every covariance value changes per frame.',
    dataTypes: {
      input: ['points', 'time-series'],
      output: ['points', 'raster']
    },
    useCases: ['interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      TEMPORAL_CONTEXT,
      param('positions', "GraphDataView<'float32x2'>", 'Sample positions.', {
        required: true
      }),
      param('times', F32, 'Relative float32 time per sample.', {
        required: true
      }),
      param(
        'observations',
        "GraphDataView<'float32x4'>",
        'Rows [value, error variance, reserved, validity].',
        {
          required: true
        }
      ),
      param('queries', "GraphDataView<'float32x2'>", 'Prediction positions.', {
        required: true
      }),
      param('queryTimes', F32, 'Relative time per query.', {required: true}),
      param('mode', "'simple' | 'ordinary'", 'Known mean, or unknown constant mean.', {
        required: true
      }),
      param('parameters', F32, 'Per-frame covariance from getGPUSpaceTimeKrigingParameterValues.', {
        required: true
      })
    ],
    outputs: [
      output(
        'values',
        "GraphDataView<'float32x4'>",
        'Prediction, variance, pivot ratio, neighbor count.'
      ),
      output('status', U32, '0 success, 1 insufficient data, 2 singular.')
    ],
    example: makeExample(
      ['GPUSpaceTimeKriging', 'getGPUSpaceTimeKrigingParameterValues'],
      [
        'parameterBuffer.write(getGPUSpaceTimeKrigingParameterValues({covariance: spaceTimeModel}));',
        'graph.add(',
        '  new GPUSpaceTimeKriging({',
        '    temporalContext,',
        '    positions, times, observations, // samples',
        '    queries, queryTimes, // where and when to predict',
        "    mode: 'ordinary',",
        '    neighborCount: 16,',
        '    parameters: parameterBuffer.importToGraph(graph),',
        "    output: {values: output('values', 'float32x4', queryCount), status: output('status', 'uint32', queryCount)}",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'interpolation',
      seed: 1304,
      caption: 'Helene residuals between two days; variance grows away from both.'
    },
    related: ['vecchia-space-time-log-likelihood', 'space-time-variogram'],
    reference: 'api/gpu-spatial-analysis/interpolation-expansion',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- space-time statistics
  {
    id: 'space-time-local-moran',
    symbol: 'GPUSpaceTimeLocalMoran',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time local Moran',
    summary:
      "Local Moran's I of a space-time cube: each cell against its neighbors one lag earlier.",
    explanation:
      "Finds hot and cold spots that persist or spread over time. `'lagged'` relates a cell now to its neighbors `lag` periods earlier; `'differential'` tests the change between periods. Significance comes from conditional permutations.",
    dataTypes: {
      input: ['cells', 'time-series', 'spatial-weights'],
      output: ['cells']
    },
    useCases: ['find-patterns', 'detect-change'],
    domains: ['spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Spatial weights over the cells.', {
        required: true
      }),
      param('cube', F32, 'Dense cube, cellCount * periodCount; NaN is missing.', {required: true}),
      param('layout', 'GPUSpaceTimeWeightsLayout', 'Cube layout.', {
        required: true
      }),
      param('periodCount', 'number', 'Number of periods (compile-time).', {
        required: true
      }),
      param('period', 'number', 'Period analysed, in [lag, periodCount) (compile-time).', {
        required: true
      }),
      param('variant', 'GPUSpaceTimeLocalMoranVariant', 'Recipe: lagged or differential.', {
        required: true
      }),
      param('lag', 'number | undefined', 'Lag in periods, at least 1.', {
        default: '1'
      }),
      param('maximumPermutations', 'number', 'Permutation capacity.', {
        required: true
      })
    ],
    moreRequired: ['parameters'],
    outputs: [
      output('pseudoPValues', F32, 'Conditional permutation pseudo p-value per cell.'),
      output('exceedances', U32, 'Permutations at least as extreme, per cell.'),
      output('observed', "GraphDataView<'float32'> | undefined", 'Local statistic per cell.'),
      output(
        'quadrants',
        "GraphDataView<'uint32'> | undefined",
        'High-high, low-low, ... per cell; lagged variant only.'
      ),
      output('status.overflow', U32, '1 if a capacity overflowed.')
    ],
    example: makeExample(
      ['GPUSpaceTimeLocalMoran'],
      [
        'graph.add(',
        '  new GPUSpaceTimeLocalMoran({',
        '    weights, // queen weights over the cells',
        "    cube: graph.importGPUData('flashes', cubeData), // period-major",
        '    layout, periodCount: 12, period: 6, lag: 1,',
        "    variant: 'lagged',",
        '    maximumPermutations: 999,',
        '    parameters, // per-frame permutation count and seed',
        '    output: {',
        "      exceedances: output('exceedances', 'uint32', cellCount),",
        "      pseudoPValues: output('p', 'float32', cellCount),",
        "      status: {overflow: output('overflow', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'hotspots',
      seed: 1305,
      caption: 'High-High: a storm cell that persists. Uncorrected pseudo p.'
    },
    related: ['space-time-cross-correlation'],
    reference: 'api/gpu-spatial-analysis/space-time-statistics',
    maturity: 'stable'
  },
  {
    id: 'space-time-cross-correlation',
    symbol: 'GPUSpaceTimeCrossCorrelation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time cross-correlation',
    summary:
      'Per-cell correlation of two space-time series at several lags, with corrected p-values.',
    explanation:
      'Asks whether one variable follows another, cell by cell: lag `l` correlates `first[t]` with `second[t + l]`. P-values are corrected across every tested cell and lag, by default Benjamini-Hochberg.',
    dataTypes: {input: ['cells', 'time-series'], output: ['cells', 'table']},
    useCases: ['find-patterns', 'relate'],
    domains: ['spatial-statistics'],
    params: [
      param('first', F32, 'First variable, cellCount * periodCount.', {
        required: true
      }),
      param('second', F32, 'Second variable, same shape.', {required: true}),
      param('layout', 'GPUSpaceTimeWeightsLayout', 'Cube layout of both.', {
        required: true
      }),
      param('cellCount', 'number', 'Number of cells (compile-time).', {
        required: true
      }),
      param('periodCount', 'number', 'Number of periods (compile-time).', {
        required: true
      }),
      param('lags', 'readonly number[]', 'Distinct integer lags, at most 64 (compile-time).', {
        required: true
      }),
      param(
        'multipleTesting',
        'GPUMultipleTestingMethod | undefined',
        'Correction across cells and lags.',
        {
          default: "'benjamini-hochberg'"
        }
      )
    ],
    outputs: [
      output('correlations', F32, 'Correlation per cell and lag.'),
      output('pValues', F32, 'Two-sided p-value per cell and lag.'),
      output('adjustedPValues', "GraphDataView<'float32'> | undefined", 'Corrected p-values.'),
      output(
        'pairCounts',
        "GraphDataView<'uint32'> | undefined",
        'Overlapping periods per cell and lag.'
      ),
      output('testedCount', "GraphDataView<'uint32'> | undefined", 'Pairs entering the correction.')
    ],
    example: makeExample(
      ['GPUSpaceTimeCrossCorrelation'],
      [
        'const lags = [-2, -1, 0, 1, 2];',
        'graph.add(',
        '  new GPUSpaceTimeCrossCorrelation({',
        '    first, second, // cellCount * periodCount each',
        '    layout, cellCount, periodCount: 24, lags,',
        "    multipleTesting: 'benjamini-hochberg',",
        '    output: {',
        "      correlations: output('r', 'float32', cellCount * lags.length),",
        "      pValues: output('p', 'float32', cellCount * lags.length)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'cells',
      seed: 1306,
      caption: 'Flash count against mean flash area per cell; p-values optimistic.'
    },
    related: ['space-time-local-moran'],
    reference: 'api/gpu-spatial-analysis/space-time-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- GTWR
  {
    id: 'geographically-temporally-weighted-regression',
    symbol: 'GPUGeographicallyTemporallyWeightedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geographically and temporally weighted regression',
    summary: 'Local regression whose coefficients vary over space and time (GTWR).',
    explanation:
      'GWR with time: each row is fit from rows near it in space-time distance, where `timeScale` converts time to space. Bandwidth and time scale change per frame, so a search over both needs no recompile.',
    dataTypes: {
      input: ['polygons', 'time-series', 'table'],
      output: ['table']
    },
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('positions', "GraphDataView<'float32x2'>", 'Planar (x, y) of every row.', {
        required: true
      }),
      param('times', F32, 'Time of every row, float32 ticks.', {
        required: true
      }),
      TEMPORAL_CONTEXT,
      param('predictors', F32, 'rows x predictorCount, row-major; intercept added.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {
        required: true
      }),
      param('response', F32, 'Response per row.', {required: true}),
      param('kernel', "'gaussian' | 'bisquare' | undefined", 'Space-time kernel (compile-time).', {
        default: "'gaussian'"
      }),
      param(
        'parameters',
        F32,
        'Per-frame bandwidth and timeScale from getGPUGTWRParameterValues.',
        {
          required: true
        }
      )
    ],
    outputs: [
      output('coefficients', F32, 'Local coefficients per row, intercept first.'),
      output('localStatus', U32, 'Per row: 0 when the local solve succeeded.'),
      output('fitted', F32, 'Fitted value per row.'),
      output('hatDiagonal', F32, 'Leverage per row.'),
      output('summary', F32, 'Residual sum of squares, effective parameters, AICc.')
    ],
    example: makeExample(
      ['GPUGeographicallyTemporallyWeightedRegression', 'getGPUGTWRParameterValues'],
      [
        'parameterBuffer.write(getGPUGTWRParameterValues({bandwidth: 0.25, timeScale: 0.1}));',
        'graph.add(',
        '  new GPUGeographicallyTemporallyWeightedRegression({',
        '    positions, times, temporalContext,',
        '    predictors, predictorCount: 1, response,',
        '    parameters: parameterBuffer.importToGraph(graph),',
        '    output: {',
        "      coefficients: output('beta', 'float32', 2 * rowCount),",
        "      localStatus: output('local', 'uint32', rowCount),",
        "      fitted: output('fitted', 'float32', rowCount),",
        "      hatDiagonal: output('hat', 'float32', rowCount),",
        "      summary: output('summary', 'float32', 12)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'polygons',
      seed: 1307,
      caption: 'Simulated response on real tract poverty shares; AICc surface.'
    },
    related: [
      'multiscale-geographically-temporally-weighted-regression',
      'geographically-weighted-regression'
    ],
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'multiscale-geographically-temporally-weighted-regression',
    symbol: 'GPUMultiscaleGeographicallyTemporallyWeightedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multiscale GTWR',
    summary: 'GTWR with its own space-time bandwidth per coefficient, fit by backfitting.',
    explanation:
      'Some effects vary sharply and others barely; one bandwidth cannot fit both. MGTWR backfits each coefficient with its own bandwidth and time scale, as MGWR does in space. Settings change per frame; iterations are compile-time.',
    dataTypes: {
      input: ['polygons', 'time-series', 'table'],
      output: ['table']
    },
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('positions', "GraphDataView<'float32x2'>", 'Planar (x, y) of every row.', {
        required: true
      }),
      param('times', F32, 'Time of every row, float32 ticks.', {
        required: true
      }),
      TEMPORAL_CONTEXT,
      param('predictors', F32, 'rows x predictorCount, row-major; intercept added.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {
        required: true
      }),
      param('response', F32, 'Response per row.', {required: true}),
      param('iterationCount', 'number', 'Backfitting iterations (compile-time).', {required: true}),
      param(
        'parameters',
        F32,
        'Per-frame settings per component from getGPUMGTWRParameterValues.',
        {
          required: true
        }
      )
    ],
    outputs: [
      output('coefficients', F32, 'Local coefficients per row, intercept first.'),
      output('fitted', F32, 'Fitted value per row.'),
      output('residuals', F32, 'Response minus fitted per row.'),
      output('localStatus', U32, 'Per row: 0 when every local solve succeeded.'),
      output('scores', F32, 'Fit scores of the backfitting.')
    ],
    example: makeExample(
      ['GPUMultiscaleGeographicallyTemporallyWeightedRegression', 'getGPUMGTWRParameterValues'],
      [
        'const settings = {initial: {bandwidth: 0.3, timeScale: 0.1}, components: [',
        '  {bandwidth: 0.6, timeScale: 0.1}, // intercept: smooth',
        '  {bandwidth: 0.15, timeScale: 0.1} // slope: local',
        ']};',
        'parameterBuffer.write(getGPUMGTWRParameterValues(settings));',
        'graph.add(',
        '  new GPUMultiscaleGeographicallyTemporallyWeightedRegression({',
        '    positions, times, temporalContext, predictors, predictorCount: 1, response,',
        '    iterationCount: 8,',
        '    parameters: parameterBuffer.importToGraph(graph),',
        '    output // coefficients, fitted, residuals, localStatus, scores',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'polygons',
      seed: 1308,
      caption: 'Simulated response; each coefficient at its own bandwidth.'
    },
    related: ['geographically-temporally-weighted-regression', 'multiscale-gwr'],
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- validation
  {
    id: 'area-of-applicability',
    symbol: 'GPUAreaOfApplicability',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Area of applicability',
    summary:
      'Marks where a model sees predictors unlike its training data (Meyer and Pebesma AOA).',
    explanation:
      'Predictions far from the training data in predictor space cannot be trusted. The dissimilarity index is the distance to the nearest training row, scaled by the mean training distance; rows above the cross-validated threshold fall outside the area.',
    dataTypes: {input: ['raster', 'table'], output: ['raster', 'table']},
    useCases: ['model', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      param('training', F32, 'Training predictors, rows x predictorCount.', {
        required: true
      }),
      param('query', F32, 'Query predictors, queryRows x predictorCount.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {
        required: true
      }),
      param(
        'weights',
        "GraphDataView<'float32'> | undefined",
        'Variable importance per predictor.'
      ),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Training row filter.'),
      param(
        'crossValidation',
        'GPUAreaOfApplicabilityFolds | undefined',
        'Folds: neighbors are searched outside the held-out fold.'
      )
    ],
    outputs: [
      output('trainDissimilarity', F32, 'Dissimilarity index of each training row.'),
      output('queryDissimilarity', F32, 'Dissimilarity index of each query row.'),
      output('applicable', U32, '1 inside the area of applicability.'),
      output('summary', F32, '[meanDistance, threshold, usable, scored, Q1, Q3].'),
      output('status', U32, 'One GPU status word; zero means complete.')
    ],
    example: makeExample(
      ['GPUAreaOfApplicability'],
      [
        'graph.add(',
        '  new GPUAreaOfApplicability({',
        '    training, query, predictorCount: 4, // band values',
        '    output: {',
        "      trainDissimilarity: output('train-di', 'float32', trainingCount),",
        "      queryDissimilarity: output('di', 'float32', queryCount),",
        "      applicable: output('aoa', 'uint32', queryCount),",
        "      summary: output('summary', 'float32', 6), // meanDistance, threshold, usable, scored, Q1, Q3",
        "      status: output('status', 'uint32', 1)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'raster',
      seed: 1309,
      caption: 'Training disc of Sentinel-2 pixels, no folds; unlike pixels fall outside.'
    },
    related: ['conformal-prediction-intervals'],
    reference: 'api/gpu-spatial-analysis/spatial-validation-advanced',
    maturity: 'stable'
  },
  {
    id: 'conformal-prediction-intervals',
    symbol: 'GPUConformalPredictionIntervals',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Conformal prediction intervals',
    summary:
      'Split and CV+ interval estimates; coverage depends on assumptions about the data, folds and fitting.',
    explanation:
      'Computes split or CV+ bounds from held-out residuals; supplying folds selects CV+. Split coverage requires exchangeable scores from an independently fitted rule; CV+ requires exchangeable observations, balanced, data-independent folds and permutation-invariant fitting, with a finite-sample bound depending on fold count. Arbitrary spatial or group folds do not inherit a universal coverage bound; miscoverage changes per frame.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      CALIBRATION,
      QUERY,
      param('foldCount', 'number | undefined', '1 is split conformal; more is CV+.', {
        default: '1'
      }),
      CONFORMAL_PARAMETERS
    ],
    outputs: [
      output('intervals', F32, 'Lower and upper bound per query row.'),
      output('status', U32, 'Status bits per query row.'),
      output('calibrationCount', U32, 'Calibration rows used.')
    ],
    example: makeExample(
      ['GPUConformalPredictionIntervals', 'getGPUConformalPredictionParameterValues'],
      [
        'parameterBuffer.write(getGPUConformalPredictionParameterValues({miscoverage: 0.1}));',
        'graph.add(',
        '  new GPUConformalPredictionIntervals({',
        '    calibration: {observations: heldOut, predictions: heldOutPredictions},',
        '    query: {predictions},',
        '    parameters: parameterBuffer.importToGraph(graph),',
        '    output: {',
        "      intervals: output('intervals', 'float32', 2 * queryCount),",
        "      status: output('status', 'uint32', queryCount),",
        "      calibrationCount: output('count', 'uint32', 1)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1310,
      caption: 'Sorted calibration errors; the rank sets every interval.'
    },
    related: ['geo-conformal-prediction-intervals', 'area-of-applicability'],
    reference: 'api/gpu-spatial-analysis/spatial-validation-advanced',
    maturity: 'stable'
  },
  {
    id: 'geo-conformal-prediction-intervals',
    symbol: 'GPUGeoConformalPredictionIntervals',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geographic conformal intervals',
    summary: 'Conformal intervals that weight calibration residuals by distance to each query.',
    explanation:
      'Errors differ by place, so one global quantile is too wide in some areas and too narrow in others. Each query takes the weighted residual quantile of its neighboring calibration rows, as in GeoConformal prediction.',
    dataTypes: {
      input: ['points', 'table', 'spatial-weights'],
      output: ['table']
    },
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics'],
    params: [
      CALIBRATION,
      QUERY,
      param(
        'neighbors',
        'GPUGeoConformalNeighbors',
        'Calibration neighbors of each query, weighted.',
        {
          required: true
        }
      ),
      CONFORMAL_PARAMETERS
    ],
    outputs: [
      output('intervals', F32, 'Lower and upper bound per query row.'),
      output('status', U32, 'Status bits per query row.')
    ],
    example: makeExample(
      ['GPUGeoConformalPredictionIntervals', 'getGPUConformalPredictionParameterValues'],
      [
        'parameterBuffer.write(getGPUConformalPredictionParameterValues({miscoverage: 0.1, selfWeight: 1}));',
        'graph.add(',
        '  new GPUGeoConformalPredictionIntervals({',
        '    calibration: {observations: heldOut, predictions: heldOutPredictions},',
        '    query: {predictions},',
        '    neighbors: {offsets, indices, weights}, // kernel weights to calibration rows',
        '    parameters: parameterBuffer.importToGraph(graph),',
        "    output: {intervals: output('intervals', 'float32', 2 * queryCount), status: output('status', 'uint32', queryCount)}",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1311,
      caption: 'Each station calibrated by its neighbors.'
    },
    related: ['conformal-prediction-intervals'],
    reference: 'api/gpu-spatial-analysis/spatial-validation-advanced',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- scoring rules
  {
    id: 'ensemble-scores',
    symbol: 'GPUEnsembleScores',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Ensemble scores',
    summary: 'Sample CRPS per site, energy score and variogram score of an ensemble forecast.',
    explanation:
      'Scores simulated fields against what was observed, matching scoringRules `crps_sample`, `es_sample` and `vs_sample`. `fair` gives the unbiased member-pair term. Lower is better.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['measure', 'model'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      param('ensemble', F32, 'Members, member-major: member * dimension + site.', {required: true}),
      param('observations', F32, 'Observed value per site.', {
        required: true
      }),
      param('memberCount', 'number', 'Number of members.', {required: true}),
      param('dimension', 'number', 'Number of sites of one field.', {
        required: true
      }),
      param(
        'estimator',
        "'fair' | 'empirical' | undefined",
        'Estimator for CRPS and energy score.',
        {default: "'empirical'"}
      ),
      param('variogramOrder', 'number | undefined', 'Order p of the variogram score.', {
        default: '0.5'
      }),
      param(
        'variogramPairs',
        "GraphDataView<'uint32x2'> | undefined",
        'Site pairs of the variogram score.'
      )
    ],
    outputs: [
      output('crps', "GraphDataView<'float32'> | undefined", 'Sample CRPS per site.'),
      output('energyScore', "GraphDataView<'float32'> | undefined", 'Energy score of the field.'),
      output(
        'variogramScore',
        "GraphDataView<'float32'> | undefined",
        'Variogram score over the listed pairs.'
      )
    ],
    example: makeExample(
      ['GPUEnsembleScores'],
      [
        'graph.add(',
        '  new GPUEnsembleScores({',
        '    ensemble, // e.g. conditional simulation output.values',
        '    observations,',
        '    memberCount: 20, dimension: siteCount,',
        "    estimator: 'fair',",
        "    output: {crps: output('crps', 'float32', siteCount), energyScore: output('es', 'float32', 1)}",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1312,
      caption: 'Members modelled from day one plus correlated noise.'
    },
    related: ['gaussian-predictive-scores'],
    reference: 'api/gpu-spatial-analysis/scoring-rules',
    maturity: 'stable'
  },
  {
    id: 'gaussian-predictive-scores',
    symbol: 'GPUGaussianPredictiveScores',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Gaussian predictive scores',
    summary: 'Closed-form CRPS of normal predictive distributions, such as kriging predictions.',
    explanation:
      'Scores a probabilistic prediction, not just its mean: CRPS rewards being close and penalizes both overconfidence and vagueness. Matches scoringRules `crps_norm`; zero variance scores the absolute error.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['measure', 'model'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      param('means', F32, 'Predictive mean per site.', {required: true}),
      param('variances', F32, 'Of an observation: latent variance + nugget + observation error.', {
        required: true
      }),
      param('observations', F32, 'Observed value per site.', {
        required: true
      })
    ],
    outputs: [output('crps', F32, 'CRPS per site.')],
    example: makeExample(
      ['GPUGaussianPredictiveScores'],
      [
        'graph.add(',
        '  new GPUGaussianPredictiveScores({',
        '    means, variances, // kriging prediction; latent variance + nugget + error',
        '    observations,',
        "    output: {crps: output('crps', 'float32', siteCount)}",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'points',
      seed: 1313,
      caption: 'CRPS over spread and bias of a day-one forecast.'
    },
    related: ['ensemble-scores', 'space-time-kriging'],
    reference: 'api/gpu-spatial-analysis/scoring-rules',
    maturity: 'stable'
  }
];
