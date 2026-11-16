// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// These bounded GPU adapters preserve input order and publish numeric metric/value diagnostic
// tables. Numeric seconds are read before subtracting the shared integer origin and packing f32.
// Column roles retain the binding ABI, including comma-separated lists shared across inputs.
// Apply policies follow the adapter preparation: GTWR and covariance settings, conformal alpha
// and FDR alpha use parameter buffers. Coordinates/origins, Vecchia mean/anisotropy and geographic
// conformal bandwidth change prepared inputs. Kernel/mode, counts and iteration budgets rebuild.

function describeNumericField(
  role: string,
  parameter: string,
  label: string,
  required = true
): KeplerOperationFieldDescriptor {
  return {role, parameter, label, required, accepts: ['number', 'integer']};
}

function describePredictors(maximum: number): KeplerOperationFieldDescriptor {
  return {
    ...describeNumericField(
      'predictors',
      'predictorProperties',
      'Predictors (same order on each input)'
    ),
    multiple: {minimum: 1, maximum}
  };
}

function describeTime(): KeplerOperationFieldDescriptor {
  // A date/string role would promise conversion that readValidationTimes does not perform.
  return describeNumericField('time', 'timeProperty', 'Time in numeric seconds (shared origin)');
}

function describeErrorVariance(): KeplerOperationFieldDescriptor {
  return describeNumericField(
    'errorVariance',
    'errorVarianceProperty',
    'Observation error variance in squared response units (empty: zero)',
    false
  );
}

function describeConstant(
  name: string,
  level: 'basic' | 'advanced' = 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen'],
  apply: 'live' | 'rebuild' = 'rebuild'
): KeplerOperationParameterDescriptor {
  return {
    name,
    level,
    apply,
    defaultPolicy: {from: 'constant'},
    ...(activeWhen ? {activeWhen} : {})
  };
}

function describeDistance(
  name: string,
  input: string,
  factor: number,
  apply: 'live' | 'rebuild' = 'rebuild'
): KeplerOperationParameterDescriptor {
  return {
    name,
    level: 'basic',
    apply,
    defaultPolicy: {from: 'data', input, rule: 'extent-diagonal', factor}
  };
}

function describeCapacity(...inputs: string[]): KeplerOperationRequirement[] {
  return inputs.flatMap(input => [
    {
      input,
      id: 'maximum-rows',
      severity: 'block',
      limit: 2048,
      remedy: 'Filter or partition this input to at most 2,048 points'
    },
    {
      input,
      id: 'maximum-vertices',
      severity: 'block',
      limit: 2048,
      remedy: 'Use single Point features, with at most 2,048 vertices per input'
    }
  ]);
}

function describeCoordinates(...inputs: string[]): KeplerOperationRequirement[] {
  return inputs.map(input => ({
    input,
    id: 'declared-coordinates',
    severity: 'warn',
    remedy: 'Declare longitude/latitude or planar coordinates, using the same model for both inputs'
  }));
}

function describeNonnegative(input: string, role: string): KeplerOperationRequirement {
  return {
    input,
    role,
    id: 'non-negative-values',
    severity: 'warn',
    remedy: 'Choose a finite non-negative variance; negative or missing values are excluded'
  };
}

function describeOutput(
  tableId: string,
  input: string,
  column: string,
  diverging = false
): KeplerOperationOutputDescriptor {
  return {
    tableId,
    kind: 'points',
    kindFrom: input,
    presentation: diverging
      ? {column, scale: 'diverging', center: 0}
      : {column, scale: 'sequential'},
    interpretation: {summary: 'value-range', cautions: []},
    next: ['how-related', 'value-where-unmeasured']
  };
}

function describeDiagnostics(bindingId: string): KeplerOperationOutputDescriptor {
  return {
    tableId: `${bindingId}-diagnostics`,
    kind: 'summary',
    interpretation: {summary: 'row-count', cautions: []}
  };
}

function describeCovariance(input: string): KeplerOperationParameterDescriptor[] {
  return [
    describeConstant('model', 'basic', undefined, 'live'),
    {
      name: 'sill',
      level: 'basic',
      apply: 'live',
      defaultPolicy: {from: 'data', input, rule: 'column-variance', role: 'value'}
    },
    describeConstant('nugget', 'basic', undefined, 'live'),
    describeDistance('range', input, 1 / 3, 'live'),
    describeConstant('neighborCount')
  ];
}

const NUMERICAL_NOTES = [
  'GPU inputs are Float32. Missing numeric values are passed to contributor exclusion/status handling; nonfinite outputs become null. Input order and feature_index are preserved.',
  'Read the metric/value diagnostics table alongside the map. A null value or nonzero status must not be read as a successful prediction.'
];
const TIME_NOTES = [
  'Times must be numeric seconds on every input, not milliseconds or date strings. Set timeOrigin to an integer Unix second near the observations: it is subtracted before Float32 rounding to preserve short time differences.',
  'For EPSG:4326 the spatial frame uses metres; planar inputs retain coordinate units. Time scales and anisotropy must match those spatial units and seconds.'
];
const COVARIANCE_NOTES = [
  'The initial range is one third of the observation extent diagonal and the initial partial sill is the observed column variance; these are starting policies, not fitted estimates. The nugget is in squared response units. Vecchia defaults raise it to the published model-specific numerical floor when needed; explicit incompatible nuggets must be corrected before running.',
  'Only exponential and Matern 3/2 covariance models are supported. Inspect solve statuses and pivot diagnostics; model-based variance does not include uncertainty in fitted covariance parameters.'
];

/** Hand-written descriptions of the twelve validation and space-time regression bindings. */
export const VALIDATION_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  ...[false, true].map((multiscale): KeplerOperationDescriptor => {
    const bindingId = multiscale ? 'regression-mgtwr' : 'regression-gtwr';
    return {
      bindingId,
      questions: ['how-related', 'what-changed'],
      action: multiscale
        ? 'Fit multiscale space-time relationships'
        : 'Fit local space-time relationships',
      prominence: 3,
      inputs: [
        {
          name: 'points',
          kind: 'points',
          fields: [
            describeNumericField('response', 'responseProperty', 'Response to explain'),
            describePredictors(11),
            describeTime()
          ]
        }
      ],
      parameters: [
        describeConstant('timeOrigin'),
        describeConstant('kernel', 'basic'),
        describeDistance('bandwidth', 'points', 1 / 3, 'live'),
        describeConstant('timeScale', 'basic', undefined, 'live'),
        ...(multiscale
          ? [
              describeConstant('componentBandwidths', 'basic', undefined, 'live'),
              describeConstant('componentTimeScales', 'basic', undefined, 'live'),
              describeConstant('iterationCount')
            ]
          : []),
        describeConstant('coordinateSystem')
      ],
      requirements: [...describeCapacity('points'), ...describeCoordinates('points')],
      outputs: [
        describeOutput(bindingId, 'points', 'residual', true),
        describeDiagnostics(bindingId)
      ],
      notes: [
        ...NUMERICAL_NOTES,
        ...TIME_NOTES,
        'Fixed bandwidths and Gaussian or bisquare kernels are used; no automatic bandwidth search or adaptive kernel is enabled. The extent-based bandwidth is a starting scale, not a fitted optimum.',
        'Choose 1 to 11 distinct predictors. The name intercept is reserved. Coefficients are coef_intercept and coef_<predictor>; fitted, residual and status retain one row per observation. Status 1 is singular and status 2 is excluded.',
        'The residual map describes an in-sample fit, not out-of-sample accuracy or causal effects. Inspect singular_rows and excluded_rows in diagnostics.',
        ...(multiscale
          ? [
              'Component bandwidths and component time scales need one entry per coefficient, intercept first. Defaults repeat the resolved bandwidth and time scale for the selected predictors; explicit lists must match that coefficient count.',
              'Backfitting runs every requested iteration; backfit_score diagnostics report progress and do not imply convergence or trigger early stopping.'
            ]
          : [
              'Diagnostics include residual_sum_squares, r_squared, aicc and hat_trace; these describe the fitted sample.'
            ])
      ]
    };
  }),
  ...['likelihood', 'fisher', 'space-time'].map((method): KeplerOperationDescriptor => {
    const bindingId = `validation-vecchia-${method}`;
    const temporal = method === 'space-time';
    return {
      bindingId,
      questions: ['value-where-unmeasured', 'how-related'],
      action: temporal
        ? 'Evaluate space-time covariance'
        : method === 'fisher'
          ? 'Fit covariance with Fisher scoring'
          : 'Evaluate or fit spatial covariance',
      prominence: 3,
      inputs: [
        {
          name: 'points',
          kind: 'points',
          rowSemantics: ['surface-samples'],
          fields: [
            describeNumericField('value', 'valueProperty', 'Observed response'),
            describeErrorVariance(),
            ...(temporal ? [describeTime()] : [])
          ]
        }
      ],
      parameters: [
        describeConstant('mean', 'basic'),
        ...describeCovariance('points'),
        describeConstant('fit', 'basic'),
        describeConstant('maximumIterations', 'advanced', {parameter: 'fit', equals: [true]}),
        ...(temporal
          ? [describeConstant('timeOrigin'), describeConstant('anisotropy', 'basic')]
          : []),
        describeConstant('coordinateSystem')
      ],
      requirements: [
        ...describeCapacity('points'),
        ...describeCoordinates('points'),
        describeNonnegative('points', 'errorVariance')
      ],
      outputs: [
        describeOutput(bindingId, 'points', 'site_log_likelihood'),
        describeDiagnostics(bindingId)
      ],
      notes: [
        ...NUMERICAL_NOTES,
        ...COVARIANCE_NOTES,
        ...(temporal ? TIME_NOTES : []),
        'Input order defines the Vecchia ordering: conditioning uses earlier usable rows, with at most 32 neighbors. Changing row order can change this approximate likelihood.',
        'The supplied known mean is subtracted before Float32 rounding; this does not estimate a trend. A failed site invalidates the total likelihood. Site status: 0 used, 1 excluded, 2 solve failed, 3 invalid conditional variance.',
        'Fit is off by default. When enabled, bounded Fisher scoring or Nelder-Mead work reports fit_requested, converged, budget_used and fit_failed; a completed budget does not imply convergence.',
        ...(method === 'fisher'
          ? [
              'Site score_log_* and information_* columns are derivatives and information for covariance fitting, not prediction errors.'
            ]
          : [
              'conditional_mean and conditional_variance are ordered conditional quantities, not held-out forecast validation.'
            ]),
        ...(temporal
          ? [
              'Space-time fitting keeps the metric anisotropy fixed; it is not estimated by the optimizer.'
            ]
          : [])
      ]
    };
  }),
  {
    bindingId: 'interpolation-space-time-kriging',
    questions: ['value-where-unmeasured'],
    action: 'Predict at supplied places and times',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['surface-samples'],
        fields: [
          describeNumericField('value', 'valueProperty', 'Observed sample value'),
          describeErrorVariance(),
          describeTime()
        ]
      },
      {name: 'queries', kind: 'points', fields: [describeTime()]}
    ],
    parameters: [
      describeConstant('timeOrigin'),
      ...describeCovariance('points'),
      describeConstant('anisotropy', 'basic', undefined, 'live'),
      describeConstant('mode', 'basic'),
      describeConstant('mean', 'basic', {parameter: 'mode', equals: ['simple']}, 'live'),
      describeConstant('coordinateSystem')
    ],
    requirements: [
      ...describeCapacity('points', 'queries'),
      ...describeCoordinates('points', 'queries'),
      describeNonnegative('points', 'errorVariance')
    ],
    outputs: [
      describeOutput('interpolation-space-time-kriging', 'queries', 'prediction'),
      describeDiagnostics('interpolation-space-time-kriging')
    ],
    notes: [
      ...NUMERICAL_NOTES,
      ...TIME_NOTES,
      ...COVARIANCE_NOTES,
      'Supply query points and their times explicitly; no regular grid is generated and no variogram is fitted. Both inputs use the same time property name and coordinate frame.',
      'Variance is latent conditional model variance, not a calibrated prediction interval. Inspect variance, pivot_ratio and retained_neighbors alongside prediction.',
      'Status 0 is success, 1 invalid or insufficient samples, 2 singular or unreliable solve. Neighbor count is at most 32; the known mean is used only in simple mode.'
    ]
  },
  {
    bindingId: 'validation-area-of-applicability',
    questions: ['value-where-unmeasured'],
    action: 'Check predictor support for predictions',
    prominence: 3,
    inputs: ['training', 'queries'].map(name => ({
      name,
      kind: 'points',
      fields: [describePredictors(16)]
    })),
    parameters: [],
    requirements: describeCapacity('training', 'queries'),
    outputs: [
      describeOutput('validation-area-of-applicability', 'queries', 'dissimilarity'),
      describeOutput('validation-area-of-applicability-training', 'training', 'dissimilarity'),
      describeDiagnostics('validation-area-of-applicability')
    ],
    notes: [
      ...NUMERICAL_NOTES,
      'Training and query inputs must share 1 to 16 distinct predictor names in the same order. Standardization uses training sample deviations and equal variable weights.',
      'Training distances use leave-one-out nearest neighbors. Threshold = min(Q3 + 1.5 IQR, maximum training dissimilarity). Diagnostics expose threshold, usable_training_rows and status; nonzero status indicates degenerate training.',
      'Applicability is predictor support, not calibrated predictive accuracy. Spatial cross-validation folds and variable importance are not supplied. No response or query observation is required.'
    ]
  },
  ...[false, true].map((geographic): KeplerOperationDescriptor => {
    const bindingId = geographic ? 'validation-geo-conformal' : 'validation-conformal';
    const prediction = describeNumericField(
      'prediction',
      'predictionProperty',
      'Prediction from the held-out model'
    );
    const scale = describeNumericField(
      'scale',
      'scaleProperty',
      'Positive residual scale (empty: unit scale)',
      false
    );
    return {
      bindingId,
      questions: ['value-where-unmeasured'],
      action: geographic
        ? 'Build locally weighted intervals'
        : 'Calibrate held-out prediction intervals',
      prominence: 3,
      inputs: [
        {
          name: 'calibration',
          kind: 'points',
          fields: [
            describeNumericField('observation', 'observationProperty', 'Held-out observation'),
            prediction,
            scale
          ]
        },
        {name: 'queries', kind: 'points', fields: [prediction, scale]}
      ],
      parameters: [
        describeConstant('miscoverage', 'basic', undefined, 'live'),
        ...(geographic
          ? [
              describeConstant('neighborCount'),
              describeDistance('bandwidth', 'calibration', 1 / 3),
              describeConstant('selfWeight', 'advanced', undefined, 'live'),
              describeConstant('coordinateSystem')
            ]
          : [])
      ],
      requirements: [
        ...describeCapacity('calibration', 'queries'),
        ...(geographic ? describeCoordinates('calibration', 'queries') : []),
        ...['calibration', 'queries'].map(
          (input): KeplerOperationRequirement => ({
            input,
            role: 'scale',
            id: 'positive-values',
            severity: 'warn',
            remedy:
              'If a scale column is selected, use finite positive values on both inputs; otherwise leave it empty for unit scale'
          })
        )
      ],
      outputs: [describeOutput(bindingId, 'queries', 'upper'), describeDiagnostics(bindingId)],
      notes: [
        ...NUMERICAL_NOTES,
        'Calibration observations must be held out from model training. Supply predictions from that same model on calibration and query rows; no model is fitted here and query observations are not read.',
        'Prediction and optional scale names are shared across inputs. Positive scales normalize calibration residuals and rescale query intervals; empty uses unit scale.',
        'Read lower and upper together. Null endpoints can be unbounded, not zero-width intervals. Status bits: 1 lower unbounded, 2 upper unbounded, 4 invalid query, 8 no calibration, 16 neighbors truncated. Diagnostics report calibration_count, miscoverage and invalid/unbounded rows.',
        geographic
          ? 'Gaussian weighting uses at most 256 nearest usable calibration points. Local weighting does not provide a distribution-free spatial coverage guarantee; bandwidth is an extent-based starting policy, not a calibrated optimum.'
          : 'Split holdout conformal requires exchangeability for marginal coverage. Marginal coverage does not guarantee coverage at every location or for each subgroup.'
      ]
    };
  }),
  ...[false, true].map((ensemble): KeplerOperationDescriptor => {
    const bindingId = ensemble ? 'validation-ensemble-scores' : 'validation-gaussian-scores';
    return {
      bindingId,
      questions: ['value-where-unmeasured'],
      action: ensemble
        ? 'Score joint ensemble forecasts'
        : 'Score Gaussian predictive distributions',
      prominence: 3,
      inputs: [
        {
          name: 'points',
          kind: 'points',
          fields: [
            describeNumericField('observation', 'observationProperty', 'Held-out observation'),
            ...(ensemble
              ? [
                  {
                    ...describeNumericField(
                      'members',
                      'memberProperties',
                      'Ensemble members of the same joint field'
                    ),
                    multiple: {minimum: 1, maximum: 64}
                  }
                ]
              : [
                  describeNumericField('mean', 'meanProperty', 'Predictive mean'),
                  describeNumericField(
                    'variance',
                    'varianceProperty',
                    'Predictive variance in squared response units'
                  )
                ])
          ]
        }
      ],
      parameters: ensemble
        ? [describeConstant('fair', 'basic'), describeConstant('variogramOrder')]
        : [],
      requirements: [
        ...describeCapacity('points'),
        ...(!ensemble ? [describeNonnegative('points', 'variance')] : [])
      ],
      outputs: [describeOutput(bindingId, 'points', 'crps'), describeDiagnostics(bindingId)],
      notes: [
        ...NUMERICAL_NOTES,
        'Lower CRPS means a better predictive distribution, in response units. Use observations independent of model training to measure predictive performance. Diagnostics include mean_crps, scored_rows and invalid_rows.',
        ...(ensemble
          ? [
              'Select 1 to 64 distinct member columns. All members must describe the same joint field. Fair CRPS and energy scores need at least two members; with one member leave fair off.',
              'Whole-field energy_score and variogram_score are diagnostics. Variogram pairs are both directions of consecutive input rows with equal weights: row order defines the pair list and can change the score.'
            ]
          : [
              'Only Gaussian predictive distributions are scored. Negative or missing variance produces a null score; zero variance is a point forecast. Variance is squared response units, not standard deviation.'
            ])
      ]
    };
  }),
  {
    bindingId: 'validation-fdr',
    questions: ['where-significant'],
    action: 'Adjust a declared family of tests',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [describeNumericField('p-value', 'pValueProperty', 'P-value (one point per test)')]
      }
    ],
    parameters: [
      describeConstant('alpha', 'basic', undefined, 'live'),
      describeConstant('method', 'basic')
    ],
    requirements: describeCapacity('points'),
    outputs: [
      {
        ...describeOutput('validation-fdr', 'points', 'significant'),
        presentation: {
          column: 'significant',
          scale: 'class-scheme',
          classScheme: 'significance-flag',
          classEncoding: 'contributor-integer'
        }
      },
      describeDiagnostics('validation-fdr')
    ],
    notes: [
      ...NUMERICAL_NOTES,
      'The selected input is the complete testing family. Filtering it changes the adjustment; recompute the full population before keeping an inferential result.',
      'Only finite p-values in [0, 1] enter the family. Invalid rows are untested, with null adjusted_pvalue and significant = 0; that zero does not establish a nonsignificant test.',
      'Benjamini-Yekutieli is the binding default for dependent tests. Benjamini-Hochberg requires its dependence assumptions; Holm and Bonferroni control family-wise error rather than false discovery rate.',
      'Read tested_count, significant_count, critical_pvalue and alpha in diagnostics. Raster cells must be supplied as Point features at cell centers; native raster tiles are not read.'
    ]
  }
];
