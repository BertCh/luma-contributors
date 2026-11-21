// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisCaution,
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisMeasurementLevel
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: significance, multipleTesting, rateScale only reclassify or rescale computed values in a
// post-pass. Other settings change prepared inputs, allocation, graph construction or compiled
// shader constants (including neighbourhood sizes and permutation loops), so require rebuilding.

// Descriptors for the six curated bindings in `modules/kepler/src/curated/bindings`. Distances
// (`epsilon`, `cellSize`, `range`, `searchRadius`, `maximumLag`) are in metres in the local frame
// for EPSG:4326 data and in coordinate units for planar data, as for the catalog bindings.
//
// Encodings the descriptors rely on (all read from the binding code, checked by the spec):
// - `curated-rate-cluster-map`, column `cluster`: the strings `HH`, `HL`, `LH`, `LL`,
//   `not-significant`, `not-tested`, the encoding of `local-moran-statistics`. Rows that were
//   not analysed carry `not-tested`; their `status` column names the reason.
// - `curated-clusters-and-footprints`, column `cluster` of `points`: an integer, -1 for noise and
//   0, 1, 2, ... for clusters.

function describeConstant(
  name: string,
  level: 'basic' | 'advanced'
): KeplerOperationParameterDescriptor {
  return {
    name,
    apply: ['significance', 'multipleTesting', 'rateScale'].includes(name) ? 'live' : 'rebuild',
    defaultPolicy: {from: 'constant'},
    level
  };
}

function describeCapacity(
  input: string,
  maximumRows: number,
  maximumVertices?: number
): KeplerOperationRequirement[] {
  const requirements: KeplerOperationRequirement[] = [
    {
      input,
      id: 'maximum-rows',
      severity: 'block',
      limit: maximumRows,
      remedy: 'Filter the layer to the area of interest or aggregate it first'
    }
  ];
  if (maximumVertices !== undefined) {
    requirements.push({
      input,
      id: 'maximum-vertices',
      severity: 'block',
      limit: maximumVertices,
      remedy: 'Simplify the geometry or split the layer into tiles'
    });
  }
  return requirements;
}

function describeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {
    input,
    id: 'declared-coordinates',
    severity: 'warn',
    remedy: 'State whether the coordinates are longitude and latitude or planar'
  };
}

/** A role that is a count: events and population are totals, and the rate is formed in the binding. */
function describeCountRole(
  role: string,
  label: string,
  parameter: string
): GPUAnalysisFieldRoleSpec & {parameter: string} {
  return {
    role,
    label,
    accepts: ['number', 'integer'],
    measurement: ['count'],
    required: true,
    parameter
  };
}

/** The areas of a rate analysis: cases and population per area. */
const RATE_AREAS_INPUT = {
  name: 'areas',
  kind: 'polygons',
  rowSemantics: ['objects'],
  fields: [
    describeCountRole('cases', 'Cases (numerator)', 'casesProperty'),
    describeCountRole('population', 'Population (denominator)', 'populationProperty')
  ]
} as const;

function describeRateRequirements(
  maximumRows: number,
  minimumRows: number
): KeplerOperationRequirement[] {
  return [
    ...describeCapacity('areas', maximumRows, 200000),
    {
      input: 'areas',
      id: 'minimum-rows',
      severity: 'block',
      limit: minimumRows,
      remedy: `Use a layer with at least ${minimumRows} areas that have cases and a positive population`
    },
    {
      input: 'areas',
      id: 'minimum-rows',
      severity: 'warn',
      limit: 30,
      remedy: 'Add more areas or treat the result as exploratory'
    },
    {
      input: 'areas',
      id: 'positive-values',
      severity: 'warn',
      role: 'population',
      remedy:
        'Areas with no population are listed with a status and not analysed; merge them into a neighbour'
    },
    {
      input: 'areas',
      id: 'non-negative-values',
      severity: 'warn',
      role: 'cases',
      remedy: 'Areas with negative cases are listed with a status and not analysed'
    },
    {
      input: 'areas',
      id: 'every-row-has-a-neighbor',
      severity: 'warn',
      remedy: 'Areas that touch no other area are not tested; check the boundaries share vertices'
    },
    describeDeclaredCoordinates('areas'),
    {
      input: 'areas',
      id: 'row-semantics',
      severity: 'warn',
      expects: ['objects'],
      remedy: 'Aggregate the events to areas first, then analyse the rate'
    }
  ];
}

/** Value of an area statistic: a rate or an amount, not a raw count of areas of unequal size. */
function describeValueRole(label: string): GPUAnalysisFieldRoleSpec & {parameter: string} {
  const measurement: readonly GPUAnalysisMeasurementLevel[] = ['rate', 'amount'];
  return {
    role: 'value',
    label,
    accepts: ['number', 'integer'],
    measurement,
    required: true,
    parameter: 'valueProperty'
  };
}

/** Response and predictors take any numeric level except identifiers and wrapped angles. */
const REGRESSION_MEASUREMENT: readonly GPUAnalysisMeasurementLevel[] = [
  'count',
  'amount',
  'rate',
  'ordinal'
];

/**
 * Run-time cautions of the curated bindings. `interpretation.cautions` lists only cautions that
 * are always true of an output; these depend on the run, so the report layer reads the named
 * summary key of the run and raises the caution when the condition holds.
 */
export const CURATED_RUNTIME_CAUTION_SUMMARY_KEYS: Readonly<
  Record<string, Readonly<Partial<Record<GPUAnalysisCaution, {key: string; raiseWhen: string}>>>>
> = {
  'curated-rate-cluster-map': {
    'significance-not-attainable': {
      key: 'significanceNotAttainable',
      raiseWhen: 'equals 1 (permutationsNeeded says how many permutations would allow a flag)'
    },
    'row-order-dependent-inference': {key: 'tiedAnchorCount', raiseWhen: 'is above 0'}
  },
  'curated-excess-risk': {
    'small-denominators': {
      key: 'smallDenominatorAreaCount',
      raiseWhen: 'is above 0 (areas with fewer than smallDenominatorThreshold expected cases)'
    }
  },
  'curated-regression-with-diagnostics': {
    'row-order-dependent-inference': {key: 'tiedAnchorCount', raiseWhen: 'is above 0'}
  },
  'curated-kriging-with-diagnostics': {
    'poor-model-fit': {key: 'poorModelFit', raiseWhen: 'equals 1 (poorModelFitReasons says why)'}
  },
  'curated-contiguity-correlogram': {
    'row-order-dependent-inference': {key: 'tiedAnchorCount', raiseWhen: 'is above 0'}
  }
};

export const CURATED_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'curated-rate-cluster-map',
    questions: ['where-significant'],
    action: 'Map rate clusters',
    prominence: 1,
    inputs: [{...RATE_AREAS_INPUT, fields: [...RATE_AREAS_INPUT.fields]}],
    parameters: [
      describeConstant('criterion', 'basic'),
      describeConstant('permutations', 'advanced'),
      describeConstant('seed', 'advanced'),
      describeConstant('significance', 'basic'),
      // Benjamini-Hochberg over the tested areas, the recipe's default; a choice about the process.
      describeConstant('multipleTesting', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describeRateRequirements(5000, 4),
    notes: [
      'p-values are conditional-permutation pseudo p-values counting permuted statistics with an absolute value at least the observed one; the absolute-value and doubled-smaller-tail definitions can differ materially on skewed data; either may give the smaller p-value.',
      'With 999 permutations the smallest p-value is 0.001, so a strict correction over many areas may flag nothing; the result says so when no area could be flagged.'
    ],
    outputs: [
      {
        tableId: 'rate-clusters',
        kind: 'polygons',
        kindFrom: 'areas',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'local-moran-clusters',
          classEncoding: 'catalog-text'
        },
        // The p-values are corrected by default, so `multiple-testing-uncorrected` is not always
        // true; a host adds it from the run when the correction was `none`.
        interpretation: {summary: 'local-cluster-count', cautions: ['pseudo-p-values']},
        next: ['how-related', 'prepare-data']
      }
    ]
  },
  {
    bindingId: 'curated-excess-risk',
    questions: ['where-significant'],
    action: 'Map excess risk and smoothed rates',
    prominence: 2,
    inputs: [{...RATE_AREAS_INPUT, fields: [...RATE_AREAS_INPUT.fields]}],
    parameters: [
      describeConstant('criterion', 'basic'),
      // 1000 per thousand is a presentation choice for the rate columns, not a data statistic.
      describeConstant('rateScale', 'basic'),
      describeConstant('islandPolicy', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describeRateRequirements(5000, 3),
    notes: [
      'The raw excess risk (observed over expected cases) is unstable for small populations: an area with fewer than 5 expected cases moves by more than 20 percent with each case, and the highest raw values of a map are often tiny areas driven by one case. The map shows the smoothed excess risk by default; the raw ratio stays in the table.'
    ],
    outputs: [
      {
        tableId: 'excess-risk',
        kind: 'polygons',
        kindFrom: 'areas',
        // The empirical Bayes smoothed rate over the overall rate, centred on 1: above is more
        // cases than the overall rate explains. The raw ratio (`excess_risk`) stays in the table.
        presentation: {column: 'smoothed_excess_risk', scale: 'diverging', center: 1},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant']
      }
    ]
  },
  {
    bindingId: 'curated-clusters-and-footprints',
    questions: ['which-groups-or-regions'],
    action: 'Find clusters and outlines',
    prominence: 1,
    notes: [
      'Footprints are convex hulls, so a crescent-shaped group is outlined with its gap filled; areas on longitude and latitude data are on a sphere and are smaller than the ellipsoid area by about 0.12 percent at 40 degrees latitude.'
    ],
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events', 'objects'], fields: []}],
    parameters: [
      // The knee of the sorted distance to the 3rd nearest neighbour (k = 2 * 2 - 1, Schubert et
      // al. 2017), as for `density-clustering`; the binding's minimum points counts the point itself.
      {
        name: 'epsilon',
        // When the knee rule declines (uniform points, few points) three times the median
        // nearest-neighbour distance is used instead of the binding's 200 m: the same rule, factor
        // and 1 percent margin as `density-clustering` (`descriptors-clustering.ts`), where the
        // factor was chosen by comparing 2, 2.5, 3 and 4 on eight kinds of data.
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'k-distance-knee',
          factor: 1.01,
          fallback: {rule: 'median-nearest-neighbor-distance', factor: 3}
        },
        level: 'basic'
      },
      describeConstant('minimumSamples', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 6000),
      {
        input: 'points',
        id: 'no-duplicate-locations',
        severity: 'warn',
        remedy: 'Coincident points count toward density; aggregate them or raise the minimum points'
      },
      describeDeclaredCoordinates('points'),
      {
        input: 'points',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['events', 'objects'],
        remedy:
          'Interpolate measured values with IDW or kriging instead of clustering their locations'
      }
    ],
    outputs: [
      {
        tableId: 'points',
        kind: 'points',
        kindFrom: 'points',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'cluster-membership',
          classEncoding: 'catalog-signed-label'
        },
        interpretation: {summary: 'cluster-count-and-noise', cautions: []},
        next: ['how-much-per-area', 'how-related']
      },
      {
        tableId: 'footprints',
        kind: 'polygons',
        presentation: {column: 'point_count', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'curated-regression-with-diagnostics',
    questions: ['how-related'],
    action: 'Fit a regression with diagnostics',
    prominence: 1,
    notes: [
      'AIC and AICc count the error variance as a parameter (as R AIC(lm) and MuMIn AICc); statsmodels and spreg report an AIC exactly 2 lower.',
      'Predictors are standardised internally, so their units do not matter; a constant or exactly collinear predictor is refused.',
      'The residual Moran statistic is tested by permutation of the residuals, an approximation; the Lagrange multiplier tests and spatial lag, error and geographically weighted models are not computed.'
    ],
    inputs: [
      {
        name: 'areas',
        // The binding also accepts points (nearest neighbours for the residual test); the data
        // kind of an input is a single value in the contract, so only polygons are declared.
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'response',
            label: 'Response',
            accepts: ['number', 'integer'],
            measurement: REGRESSION_MEASUREMENT,
            required: true,
            parameter: 'responseProperty'
          },
          {
            role: 'predictors',
            label: 'Predictors',
            accepts: ['number', 'integer'],
            measurement: REGRESSION_MEASUREMENT,
            required: true,
            multiple: {minimum: 1, maximum: 15},
            parameter: 'predictorProperties'
          }
        ]
      }
    ],
    parameters: [
      // `auto` is queen contiguity for polygons and nearest neighbours for points.
      describeConstant('neighborhood', 'basic'),
      {
        ...describeConstant('neighborCount', 'advanced'),
        activeWhen: {parameter: 'neighborhood', equals: ['auto', 'knn']}
      },
      describeConstant('permutations', 'advanced'),
      describeConstant('seed', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('areas', 5000, 200000),
      {
        input: 'areas',
        id: 'minimum-rows',
        severity: 'block',
        limit: 5,
        remedy:
          'Use more observations (at least the number of predictors plus four) or fewer predictors'
      },
      {
        input: 'areas',
        id: 'minimum-rows',
        severity: 'warn',
        limit: 30,
        remedy: 'Add observations; coefficients from so few rows are unstable'
      },
      {
        input: 'areas',
        id: 'varying-values',
        severity: 'block',
        role: 'response',
        remedy: 'Choose a response that differs between features'
      },
      {
        input: 'areas',
        id: 'varying-values',
        severity: 'block',
        role: 'predictors',
        remedy: 'Drop predictors that are constant; they duplicate the intercept'
      },
      {
        input: 'areas',
        id: 'no-missing-values',
        severity: 'warn',
        role: 'predictors',
        remedy: 'Rows with a missing value are listed with a status and left out of the fit'
      },
      {
        input: 'areas',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Rows with no neighbour are left out of the residual Moran statistic'
      },
      describeDeclaredCoordinates('areas'),
      {
        input: 'areas',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['objects'],
        remedy: 'Aggregate the events to areas first, then relate the rate'
      }
    ],
    outputs: [
      {
        tableId: 'coefficients',
        kind: 'summary',
        interpretation: {summary: 'regression-fit', cautions: []}
      },
      {
        tableId: 'diagnostics',
        kind: 'summary',
        interpretation: {summary: 'row-count', cautions: ['pseudo-p-values']}
      },
      {
        tableId: 'residuals',
        kind: 'polygons',
        kindFrom: 'areas',
        presentation: {column: 'residual', scale: 'diverging', center: 0},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant']
      }
    ]
  },
  {
    bindingId: 'curated-kriging-with-diagnostics',
    questions: ['value-where-unmeasured'],
    action: 'Predict values with kriging',
    prominence: 1,
    notes: [
      'The variogram is fitted once. The poor model fit warning (summary key poorModelFit) uses incomplete isotropic screening checks that may miss white noise, trends or periodic data; absence of a warning does not establish a good or identified fit. A high omnidirectional nugget can mask anisotropy.',
      'The variance uses the squared units specified by the variogram. Its calibration depends on how well that variogram describes the data.',
      'Leave-one-out predictions use the nearest samples within the search radius, as the grid does, and the variogram is not refitted for each, so the errors are slightly optimistic.'
    ],
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['surface-samples'],
        fields: [
          {
            role: 'value',
            label: 'Measured value',
            accepts: ['number', 'integer'],
            // A quantity that exists everywhere, measured at a point; a count of events at a
            // point says how many happened there, not what the surface is.
            measurement: ['amount', 'rate'],
            required: true,
            parameter: 'valueProperty'
          }
        ]
      }
    ],
    parameters: [
      // Shorter side over 250 times 2: 15,625 cells for a square extent, within the 40,000 cap.
      {
        name: 'cellSize',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'extent-shorter-side-over-250',
          factor: 2
        },
        level: 'basic'
      },
      describeConstant('variogram', 'basic'),
      describeConstant('model', 'basic'),
      // Used only with a stated variogram, which no rule here supplies.
      {
        ...describeConstant('nugget', 'advanced'),
        activeWhen: {parameter: 'variogram', equals: ['stated']}
      },
      {
        ...describeConstant('sill', 'advanced'),
        activeWhen: {parameter: 'variogram', equals: ['stated']}
      },
      {
        ...describeConstant('range', 'advanced'),
        activeWhen: {parameter: 'variogram', equals: ['stated']}
      },
      describeConstant('binCount', 'advanced'),
      // 0 means a third of the data diagonal, the maximum lag of the design.
      describeConstant('maximumLag', 'advanced'),
      describeConstant('neighborCount', 'advanced'),
      describeConstant('searchRadius', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 2000),
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'block',
        limit: 8,
        remedy: 'Use at least eight measurements at distinct locations'
      },
      {
        input: 'points',
        id: 'varying-values',
        severity: 'block',
        role: 'value',
        activeWhen: {parameter: 'variogram', equals: ['fit']},
        remedy:
          'A constant column cannot fit a variogram; choose a varying column or state a variogram'
      },
      {
        input: 'points',
        id: 'no-duplicate-locations',
        severity: 'warn',
        remedy: 'Measurements at the same place are averaged into one'
      },
      describeDeclaredCoordinates('points'),
      {
        input: 'points',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['surface-samples'],
        remedy: 'For the locations of events use a density analysis instead'
      }
    ],
    outputs: [
      {
        tableId: 'kriging-cells',
        kind: 'surface',
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['shape-of-the-surface']
      },
      {
        tableId: 'cross-validation',
        kind: 'points',
        kindFrom: 'points',
        // Standardised errors are about zero for a sound variogram, so the scale centres on 0.
        presentation: {column: 'standardized_error', scale: 'diverging', center: 0},
        interpretation: {summary: 'row-count', cautions: []}
      },
      {
        tableId: 'variogram',
        kind: 'summary',
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'curated-contiguity-correlogram',
    questions: ['how-related'],
    action: 'Plot association by neighbour order',
    prominence: 2,
    notes: ['Each order is tested separately; no correction is made for testing several orders.'],
    inputs: [
      {
        name: 'areas',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [describeValueRole('Value to test')]
      }
    ],
    parameters: [
      // Orders 1 to 6 cover the first rings of most area layers; a count, so constant.
      describeConstant('maximumOrder', 'basic'),
      describeConstant('criterion', 'basic'),
      describeConstant('permutations', 'advanced'),
      describeConstant('seed', 'advanced'),
      describeConstant('significance', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('areas', 5000, 200000),
      {
        input: 'areas',
        id: 'minimum-rows',
        severity: 'block',
        limit: 5,
        remedy: 'Use a layer with at least five areas that have a value'
      },
      {
        input: 'areas',
        id: 'minimum-rows',
        severity: 'warn',
        limit: 30,
        remedy: 'Add more areas or treat the result as exploratory'
      },
      {
        input: 'areas',
        id: 'varying-values',
        severity: 'block',
        role: 'value',
        remedy: 'Choose a column whose values differ between features'
      },
      {
        input: 'areas',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Areas that touch no other area are left out of every order'
      },
      describeDeclaredCoordinates('areas'),
      {
        input: 'areas',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['objects'],
        remedy: 'Aggregate the points to areas first, then analyse the rate'
      }
    ],
    outputs: [
      {
        tableId: 'correlogram',
        kind: 'summary',
        presentation: {column: 'morans_i', scale: 'diverging', center: 0},
        interpretation: {summary: 'global-statistic', cautions: ['pseudo-p-values']},
        next: ['where-significant']
      }
    ]
  }
];
