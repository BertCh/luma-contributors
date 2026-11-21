// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisMeasurementLevel
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: significance, multipleTesting, method, alpha and resultCount only reclassify computed values in a
// post-pass. Other settings change prepared inputs, allocation, graph construction or compiled
// shader constants (including neighbourhood sizes and permutation loops), so require rebuilding.

// Descriptors for the 20 statistical bindings of the `analysis` catalog domain: spatial weights,
// global and local autocorrelation, hot spots, rate smoothing, multiple testing, the Knox test,
// regressions and the two optimisation and similarity bindings. The measurement levels below are
// the reason these descriptors exist: a statistic on raw totals of areas of unequal size or
// population maps where the areas are large or the people live, not where the process is unusual.
//
// Encodings the descriptors rely on (all read from the binding code, checked by the spec):
// - `local-moran-statistics`, column `cluster`: the strings `HH`, `HL`, `LH`, `LL`,
//   `not-significant`, `not-tested`.
// - `hotspot-gi-star`, column `classification`: the strings `hot-90`, `hot-95`, `hot-99`,
//   `cold-90`, `cold-95`, `cold-99`, `not-significant`.
// - `recipe-hot-spot-lattice`, column `bin`: the integers -3 to 3 (sign is hot or cold, size is
//   90, 95 and 99 percent).
// - `multiple-testing-adjustment`, column `rejected`: a boolean.

/**
 * Margin on the outlier-resistant nearest-neighbor distance. The rule returns the distance at
 * which the farthest feature (outliers apart) just reaches its neighbor. The binding includes
 * equality; fixed planar DBSCAN also compensates frame translation errors. Neither change makes
 * the profile's geographic great-circle metric equal to the binding's local equidistant metric.
 * AX-87's four-point geographic probe still has two islands at the spherical band and none with
 * this margin. Polygon anchor arithmetic can also differ, so retain this safeguard and the
 * resolver's independent upward rounding.
 */
const DISTANCE_BAND_MARGIN = 1.01;

/**
 * Resolution/stability compromise measured by this layer on 40 paired seeds at each of 300,
 * 1,000 and 5,000 points (uniform nulls; a 50-point Gaussian cluster of standard deviation 10
 * in a 1,000 by 1,000 background). Factors 1.5 / 2 / 2.5 found the cluster in 100 / 90 / 65
 * percent of 300-point runs and all larger runs. Null runs with any discovery were respectively
 * 17.5 / 20 / 12.5 percent at 300, 27.5 / 10 / 22.5 at 1,000, and 12.5 / 22.5 / 25 at 5,000.
 * Factor 1.5 retains the smaller clusters with a similar pooled null frequency (19.2 versus
 * 17.5 percent for factor 2). This is a resolution choice, not a calibration fix; counts per
 * cell are still sparse and every tested factor exceeds nominal 10 percent in some fixtures.
 * Near the input limit the resolver must raise the cell size to respect the 40,000-cell cap.
 */
const HOT_SPOT_CELL_SPACING_FACTOR = 1.5;

/**
 * Permutations this layer presets. 999 is GeoDa's default; with it a pseudo p-value reaches 0.001,
 * which a Benjamini-Hochberg or Bonferroni threshold over hundreds of areas needs. The bindings
 * default to 999 (local Moran), 99 (local Geary) or 199 (local LISA variants, Knox).
 */
const PERMUTATION_PRESET: KeplerOperationParameterDescriptor['defaultPolicy'] = {
  from: 'preset',
  value: 999,
  reason:
    '999 permutations (GeoDa’s default) let a pseudo p-value reach 0.001; fewer permutations can be too coarse for a corrected threshold'
};

/**
 * A parameter whose binding default stands: counts such as the k of k-nearest neighbors (the
 * binding's 6; Esri recommends 8 and GeoDa uses 4, so no single value is the standard), seeds,
 * enumerations and thresholds that are choices about the process.
 */
function describeConstant(
  name: string,
  level: 'basic' | 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen']
): KeplerOperationParameterDescriptor {
  return {
    name,
    apply: ['significance', 'multipleTesting', 'method', 'alpha', 'resultCount'].includes(name)
      ? 'live'
      : 'rebuild',
    defaultPolicy: {from: 'constant'},
    level,
    ...(activeWhen ? {activeWhen} : {})
  };
}

function describePermutations(): KeplerOperationParameterDescriptor {
  return {
    apply: 'rebuild',
    name: 'permutations',
    defaultPolicy: PERMUTATION_PRESET,
    level: 'advanced'
  };
}

/** The neighborhood types and where each parameter of the weights rule is used. */
const KNN_ONLY = {parameter: 'neighborhood', equals: ['knn']} as const;
const DISTANCE_BAND_ONLY = {parameter: 'neighborhood', equals: ['distance-band']} as const;

/**
 * Neighborhood of a feature-based statistic. The binding defaults to k nearest neighbors on
 * centroids. For polygons that is a poor default: on 324 polygons of very unequal size a distance
 * band gave 2 to 213 neighbors per polygon and k nearest neighbors link a small polygon to
 * neighbors that do not touch it, while queen contiguity gave 3 to 8. Points keep the binding's
 * default.
 */
function describeNeighborhood(input: string, extraReason = ''): KeplerOperationParameterDescriptor {
  return {
    name: 'neighborhood',
    apply: 'rebuild',
    defaultPolicy: {
      from: 'preset',
      value: 'queen',
      input,
      whenGeometry: 'polygon',
      reason: `Polygons are neighbors when they touch (queen contiguity). A distance or k-nearest rule on centroids gave 2 to 213 neighbors on polygons of very unequal size, queen gave 3 to 8.${extraReason}`
    },
    level: 'basic'
  };
}

/**
 * The parameters every feature-based weights rule shares. `distanceBand` is in the unit of the
 * coordinate system parameter: metres in the local frame for EPSG:4326 data and coordinate units
 * for planar data (`makeLocalFrame` in `catalog-local-frame.ts`), which is what the rule returns.
 * The outlier-resistant rule is the upper fence of the nearest-neighbor distances: GeoDa's and
 * Esri's default (the largest nearest-neighbor distance, so every feature has a neighbor) with
 * isolated points set aside, because one far point took a band from 65 m to 56,000 m and gave
 * every other feature 299 neighbors. The explanation counts points beyond the unscaled fence;
 * actual isolation depends on the final band after the margin and rounding. For polygons the distance is between centroids, as the binding measures it. The
 * band is only read for the distance-band neighborhood and `neighborCount` (a count, so
 * constant) only for k-nearest neighbors, so both carry `activeWhen`.
 */
function describeWeightsParameters(
  input: string,
  options: {
    neighborCount?: 'basic' | 'advanced';
    distanceBand?: 'basic' | 'advanced';
    neighborhoodReason?: string;
  } = {}
): KeplerOperationParameterDescriptor[] {
  return [
    describeNeighborhood(input, options.neighborhoodReason),
    describeConstant('neighborCount', options.neighborCount ?? 'basic', KNN_ONLY),
    {
      name: 'distanceBand',
      apply: 'rebuild',
      defaultPolicy: {
        from: 'data',
        input,
        rule: 'outlier-resistant-nearest-neighbor-distance',
        factor: DISTANCE_BAND_MARGIN
      },
      level: options.distanceBand ?? 'basic',
      activeWhen: DISTANCE_BAND_ONLY
    },
    describeConstant('coordinateSystem', 'advanced')
  ];
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

/** Metric distances are computed, so the coordinate system parameter must be right. */
function describeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {
    input,
    id: 'declared-coordinates',
    severity: 'warn',
    remedy: 'State whether the coordinates are longitude and latitude or planar'
  };
}

/**
 * Requirements of the feature-based local statistics. The code throws below four usable features
 * and for a constant value; Esri documents results as unreliable below 30 features ("fewer than
 * 30", so exactly 30 passes).
 */
function describeLocalStatisticRequirements(
  input: string,
  role: string,
  expects: readonly ('events' | 'objects' | 'surface-samples')[],
  capacity: {rows: number; vertices: number} = {rows: 3000, vertices: 200000}
): KeplerOperationRequirement[] {
  return [
    ...describeCapacity(input, capacity.rows, capacity.vertices),
    {
      input,
      id: 'minimum-rows',
      severity: 'block',
      limit: 4,
      remedy: 'Use a layer with at least four features that have a value'
    },
    {
      input,
      id: 'minimum-rows',
      severity: 'warn',
      limit: 30,
      remedy: 'Add more features or treat the result as exploratory'
    },
    {
      input,
      id: 'varying-values',
      severity: 'block',
      role,
      remedy: 'Choose a column whose values differ between features'
    },
    {
      input,
      id: 'every-row-has-a-neighbor',
      severity: 'warn',
      remedy: 'Raise the distance band or neighbor count, or drop the isolated features'
    },
    describeDeclaredCoordinates(input),
    {
      input,
      id: 'row-semantics',
      severity: 'warn',
      expects,
      remedy: 'Aggregate the points to areas or cells first, then analyse the rate'
    }
  ];
}

/**
 * Value of an area statistic. The vocabulary defines `count` and `amount` as totals that add up
 * over areas, and a total per area mostly measures the size of the area (or its population): on
 * 324 unequal rectangles with a uniform rate, revenue (rate times area) was the preferred column
 * and Gi* flagged the 34 largest polygons as hot. So for polygons only a `rate` (a ratio, share or
 * density, which does not add up) is meant; for points, whose rows are the objects themselves, an
 * `amount` measured on each object is fine too. A column of another level draws the
 * `counts-where-rates-expected` caution, never a block.
 */
function describeAreaValueRole(
  label: string,
  required: GPUAnalysisFieldRoleSpec['required'] = true
): Omit<KeplerOperationFieldDescriptor, 'parameter'> {
  return {
    role: 'value',
    label,
    accepts: ['number', 'integer'],
    measurement: ['rate', 'amount'],
    measurementByGeometry: {polygon: ['rate'], point: ['rate', 'amount']},
    required
  };
}

/** Local autocorrelation result shown by its cluster class, with the usual next steps. */
const LOCAL_NEXT = ['how-related', 'prepare-data'] as const;

/**
 * Response of a regression: a quantity (amount), a rate or an ordinal score. A count is not
 * listed, so it draws the measurement warning: a count response violates the constant variance
 * and unbounded range of a linear model (use a rate, or a count model).
 */
const RESPONSE_MEASUREMENT: readonly GPUAnalysisMeasurementLevel[] = ['amount', 'rate', 'ordinal'];

/** Predictors may be any numeric level; identifiers and wrapped angles are not offered. */
const PREDICTOR_MEASUREMENT: readonly GPUAnalysisMeasurementLevel[] = [
  'count',
  'amount',
  'rate',
  'ordinal'
];

/**
 * Requirements shared by the regressions. The binding refuses fewer complete rows than
 * `minimumRows` for ONE predictor; the general rule is predictors plus two (plus three for GWR),
 * and a model with ten times as many rows as coefficients is a common minimum, so the block limit
 * is the smallest case and the warning limit is 30. A constant response gives a null R squared and
 * a constant predictor duplicates the intercept (a singular fit).
 */
function describeRegressionRequirements(
  input: string,
  maximumRows: number,
  minimumRows: number
): KeplerOperationRequirement[] {
  return [
    ...describeCapacity(input, maximumRows),
    {
      input,
      id: 'minimum-rows',
      severity: 'block',
      limit: minimumRows,
      remedy: `This is the smallest case, one predictor; the binding needs at least ${minimumRows - 1} complete rows more than there are predictors. Use more observations or fewer predictors`
    },
    {
      input,
      id: 'minimum-rows',
      severity: 'warn',
      limit: 30,
      remedy: 'Add observations; coefficients from so few rows are unstable'
    },
    {
      input,
      id: 'varying-values',
      severity: 'block',
      role: 'response',
      remedy: 'Choose a response that differs between rows; a constant one has no fit to measure'
    },
    {
      input,
      id: 'varying-values',
      severity: 'block',
      role: 'predictors',
      remedy: 'Drop predictors that are constant; they duplicate the intercept'
    },
    {
      input,
      id: 'no-missing-values',
      severity: 'warn',
      role: 'response',
      remedy: 'Rows with a missing response are dropped (complete cases only)'
    },
    {
      input,
      id: 'no-missing-values',
      severity: 'warn',
      role: 'predictors',
      remedy: 'Rows with a missing predictor are dropped (complete cases only)'
    },
    {
      input,
      id: 'row-semantics',
      severity: 'warn',
      expects: ['objects'],
      remedy: 'Aggregate the events to areas first, then relate the rate'
    }
  ];
}

/** Role of the predictors of a regression with `maximum` columns. */
function describePredictorsRole(maximum: number): KeplerOperationFieldDescriptor {
  return {
    role: 'predictors',
    label: 'Explanatory columns',
    accepts: ['number', 'integer'],
    measurement: PREDICTOR_MEASUREMENT,
    required: true,
    multiple: {minimum: 1, maximum},
    parameter: 'predictorProperties'
  };
}

/**
 * One descriptor per binding id, in the order of the task. Two bindings exist both here and as
 * hand-written host panels with a different statistic (see `local-moran-statistics` and, in the
 * interpolation file, `kriging-grid`).
 */
export const STATISTICS_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'spatial-weights',
    questions: ['prepare-data'],
    action: 'Build spatial weights',
    prominence: 2,
    // Points are accepted with knn and distance-band rules; polygons also allow queen and rook.
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    parameters: [
      ...describeWeightsParameters('features'),
      describeConstant('rowStandardize', 'advanced')
    ],
    requirements: [
      ...describeCapacity('features', 3000, 200000),
      {
        input: 'features',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Raise the distance band or neighbor count'
      },
      describeDeclaredCoordinates('features')
    ],
    outputs: [
      {
        tableId: 'weights',
        kind: 'spatial-weights',
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'Every link is directed from a feature to its neighbor; k-nearest-neighbor links are not symmetric.',
      'More than 3,000 features are refused: the distance search compares every pair.'
    ]
  },
  {
    bindingId: 'global-spatial-statistics',
    // It answers whether there is clustering at all, which is the significance question without a
    // map of where.
    questions: ['where-significant'],
    action: 'Test for clustering overall',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects', 'surface-samples'],
        fields: [{...describeAreaValueRole('Value to test'), parameter: 'valueProperty'}]
      }
    ],
    parameters: describeWeightsParameters('features'),
    requirements: [
      ...describeCapacity('features', 3000, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        limit: 4,
        remedy: 'Use a layer with at least four features that have a value'
      },
      {
        input: 'features',
        id: 'varying-values',
        severity: 'block',
        role: 'value',
        remedy: 'Choose a column whose values differ between features'
      },
      {
        input: 'features',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Raise the distance band or neighbor count'
      },
      describeDeclaredCoordinates('features'),
      {
        input: 'features',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['objects', 'surface-samples'],
        remedy: 'Aggregate the points to areas or cells first, then test the rate'
      }
    ],
    outputs: [
      {
        tableId: 'global',
        kind: 'summary',
        // Analytic normal approximations, not permutations, so no always-on caution applies.
        interpretation: {summary: 'global-statistic', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      "Moran's I and Geary's C both use randomization variance with a normal approximation and no permutation test."
    ]
  },
  {
    bindingId: 'local-moran-statistics',
    questions: ['where-significant'],
    action: 'Find clusters and outliers',
    prominence: 1,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        // Local Moran compares each area with its neighbors, so rows must be things with a value.
        rowSemantics: ['objects'],
        fields: [{...describeAreaValueRole('Value to map'), parameter: 'valueProperty'}]
      }
    ],
    parameters: [
      ...describeWeightsParameters('features').map(parameter =>
        parameter.name === 'neighborCount' ? {...parameter, level: 'advanced' as const} : parameter
      ),
      describePermutations(),
      describeConstant('seed', 'advanced'),
      describeConstant('significance', 'basic'),
      describeConstant('multipleTesting', 'basic')
    ],
    requirements: describeLocalStatisticRequirements('features', 'value', ['objects']),
    outputs: [
      {
        tableId: 'local-moran',
        kind: 'polygons',
        kindFrom: 'features',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'local-moran-clusters',
          classEncoding: 'catalog-text'
        },
        interpretation: {
          summary: 'local-cluster-count',
          cautions: ['pseudo-p-values', 'multiple-testing-uncorrected']
        },
        next: LOCAL_NEXT
      }
    ],
    notes: [
      'This form uses row-standardized weights and conditional-permutation pseudo p-values, with 999 permutations and Benjamini-Hochberg correction by default. Raw and adjusted p-values are published; none and Bonferroni corrections are also available.',
      'Its results will not match the host Local Moran panel or the rate cluster map, which use their own weights, permutation counts and correction.',
      'The cluster classes mark cores of clusters, not whole clusters.'
    ]
  },
  {
    bindingId: 'hotspot-gi-star',
    questions: ['where-significant'],
    action: 'Find hot and cold spots',
    prominence: 1,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        // Gi* sums values around each feature, so a point row must be a thing with a value.
        rowSemantics: ['objects'],
        fields: [{...describeAreaValueRole('Value to map'), parameter: 'valueProperty'}]
      }
    ],
    parameters: describeWeightsParameters('features'),
    requirements: describeLocalStatisticRequirements('features', 'value', ['objects']),
    outputs: [
      {
        tableId: 'hotspots',
        kind: 'polygons',
        kindFrom: 'features',
        presentation: {
          column: 'classification',
          scale: 'class-scheme',
          classScheme: 'hot-spot-confidence-bins',
          classEncoding: 'catalog-text'
        },
        // Analytic z-scores with fixed thresholds and no correction for the number of tests.
        interpretation: {
          summary: 'significant-hot-cold-count',
          cautions: ['multiple-testing-uncorrected']
        },
        next: ['which-groups-or-regions', 'how-related']
      }
    ],
    notes: [
      'Rows without neighbors have null z-scores and p-values and are classified not-tested; diagnostics count these islands.',
      'The p-values are normal approximations with fixed thresholds 1.645, 1.96 and 2.576 and no correction for the number of tests.'
    ]
  },
  {
    bindingId: 'local-geary-statistics',
    questions: ['where-significant'],
    action: 'Find similar and dissimilar places',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [{...describeAreaValueRole('Value to map'), parameter: 'valueProperty'}]
      }
    ],
    parameters: [
      ...describeWeightsParameters('features'),
      describePermutations(),
      describeConstant('seed', 'advanced')
    ],
    requirements: describeLocalStatisticRequirements('features', 'value', ['objects']),
    outputs: [
      {
        tableId: 'local-geary',
        kind: 'polygons',
        kindFrom: 'features',
        // The binding publishes no cluster class: small local C means a feature is similar to its
        // neighbors, so the statistic itself is mapped.
        presentation: {column: 'local_c', scale: 'sequential'},
        interpretation: {
          summary: 'value-range',
          cautions: ['pseudo-p-values', 'multiple-testing-uncorrected']
        },
        next: ['where-significant', 'prepare-data']
      }
    ],
    notes: [
      'The p-value is one-sided for small local C (similarity to the neighbors) and has no correction for the number of tests; the binding publishes no cluster classes.'
    ]
  },
  {
    bindingId: 'empirical-bayes-rates',
    // The stabilised rate per area is the answer to "how much per area" for districts with cases
    // and population. It tests nothing, so `where-significant` is only the second step toward it
    // (map the smoothed rates with a local statistic); it is not plumbing.
    questions: ['how-much-per-area', 'prepare-data'],
    action: 'Stabilize rates per area',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'cases',
            label: 'Events (numerator)',
            // Counts prefer an integer column.
            accepts: ['integer', 'number'],
            // Events and population are both totals; the rate is formed here.
            measurement: ['count'],
            required: true,
            parameter: 'numeratorProperty'
          },
          {
            role: 'population',
            label: 'Population (denominator)',
            accepts: ['integer', 'number'],
            measurement: ['count'],
            required: true,
            parameter: 'denominatorProperty'
          }
        ]
      }
    ],
    parameters: [],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        limit: 2,
        remedy: 'Use at least two areas with a positive population'
      },
      {
        input: 'features',
        id: 'positive-values',
        severity: 'warn',
        role: 'population',
        remedy: 'Areas with no population are skipped; remove them or merge them into a neighbor'
      },
      {
        input: 'features',
        id: 'non-negative-values',
        severity: 'warn',
        role: 'cases',
        remedy: 'Areas with negative events are skipped'
      }
    ],
    outputs: [
      {
        tableId: 'rates',
        kind: 'polygons',
        kindFrom: 'features',
        presentation: {column: 'eb_rate', scale: 'sequential'},
        // Global prior only: every area shrinks toward the overall rate, with no spatial term.
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'Global empirical Bayes: every rate shrinks toward the overall rate, small populations most; there is no spatial term and no test.',
      'Areas with more events than population are accepted and give a rate above 1; check the two columns are not swapped.'
    ]
  },
  {
    bindingId: 'multiple-testing-adjustment',
    questions: ['prepare-data'],
    action: 'Correct p-values for many tests',
    prominence: 3,
    inputs: [
      {
        name: 'features',
        // Any layer that carries a p-value column, for example a local statistic result.
        kind: 'attributes',
        fields: [
          {
            role: 'pvalue',
            label: 'p-value',
            accepts: ['number'],
            // A probability is a ratio between 0 and 1.
            measurement: ['rate'],
            required: true,
            parameter: 'pValueProperty'
          }
        ]
      }
    ],
    parameters: [describeConstant('method', 'basic'), describeConstant('alpha', 'basic')],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        limit: 1,
        remedy: 'Choose a column that holds p-values between 0 and 1'
      }
    ],
    outputs: [
      {
        tableId: 'adjusted',
        kind: 'attributes',
        kindFrom: 'features',
        presentation: {column: 'rejected', scale: 'class-scheme', classScheme: 'significance-flag'},
        // No summary template states "N of M tests are rejected"; `row-count` is the nearest and the
        // note below points to the run summary, which carries `rejectedCount`.
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'The number of rejected tests is `rejectedCount` in the run summary, out of `testCount`. Rows whose p-value is missing or outside 0 to 1 are skipped and do not count as tests.',
      'Benjamini-Hochberg assumes independent or positively dependent tests. A pseudo p-value from permutations cannot go below 1 / (permutations + 1), so a strict correction over many areas needs 999 permutations or more.'
    ]
  },
  {
    bindingId: 'knox-test',
    // The test asks whether events close in space are also close in time. It gives one answer for
    // the whole layer and no map, so it is a significance test first (the operator taxonomy files
    // it with the significance statistics) and a time question second.
    questions: ['where-significant', 'what-changed'],
    action: 'Test for space-time clustering',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['events'],
        fields: [
          {
            role: 'time',
            label: 'Time of each event',
            // The binding reads numbers only (numeric strings count; ISO text fails with "Too few
            // features"), so a time column must hold epoch values, and `temporalThreshold` is in the
            // same unit. Integer first: epoch milliseconds and seconds are integers.
            accepts: ['integer', 'number'],
            required: true,
            parameter: 'timeProperty'
          }
        ]
      }
    ],
    parameters: [
      // Close in space: twice the median nearest-neighbor distance. An event typically has a
      // neighbor at the median; twice that counts a few neighbors per event (about 100 of 1,770
      // pairs for three tight clusters of 20), a modest share of pairs, as a Knox test wants.
      {
        name: 'spatialThreshold',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'median-nearest-neighbor-distance',
          factor: 2
        },
        level: 'basic'
      },
      // Close in time: one twentieth of the time span, in the unit of the time column (epoch
      // milliseconds make the constant 1 a single millisecond, which found no joint pairs).
      // With the spatial factor above the null runs (random times) gave p between 0.08 and 0.98.
      {
        name: 'temporalThreshold',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'column-range',
          role: 'time',
          factor: 0.05
        },
        level: 'basic'
      },
      {
        apply: 'rebuild',
        name: 'permutations',
        defaultPolicy: PERMUTATION_PRESET,
        level: 'advanced'
      },
      describeConstant('seed', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 2000),
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'block',
        limit: 3,
        remedy: 'Use at least three events that have a time'
      },
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'warn',
        limit: 30,
        remedy: 'Add events; a joint count from so few pairs has little power'
      },
      {
        input: 'points',
        id: 'varying-values',
        severity: 'warn',
        role: 'time',
        remedy: 'Choose a time column whose values differ between events'
      },
      describeDeclaredCoordinates('points'),
      {
        input: 'points',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['events'],
        remedy: 'Use a layer where each row is one event with a time'
      }
    ],
    outputs: [
      {
        tableId: 'knox',
        kind: 'summary',
        interpretation: {summary: 'global-statistic', cautions: ['pseudo-p-values']},
        next: ['what-changed']
      }
    ],
    notes: [
      'One threshold pair is tested; the result depends on it, so read it with the thresholds that were used. No edge correction.',
      'The time column must be numeric, for example epoch seconds or milliseconds, and the temporal threshold is in the same unit. Text timestamps (ISO strings) must be converted to numbers first; otherwise every row is skipped and the run fails with "Too few features".'
    ]
  },
  {
    bindingId: 'similar-locations',
    questions: ['which-groups-or-regions'],
    action: 'Find places like this one',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'attributes',
        fields: [
          {
            role: 'attributes',
            label: 'Attributes to compare',
            accepts: ['number', 'integer'],
            // Attributes are standardised, so levels mix; identifiers and wrapped angles do not.
            measurement: ['count', 'amount', 'rate', 'ordinal'],
            // Totals of polygons of unequal size would group large areas together.
            measurementByGeometry: {polygon: ['rate']},
            required: true,
            multiple: {minimum: 1, maximum: 16},
            parameter: 'properties'
          }
        ]
      }
    ],
    parameters: [
      // The binding requires an explicit reference row.
      describeConstant('referenceIndex', 'basic'),
      describeConstant('resultCount', 'basic')
    ],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        limit: 2,
        remedy: 'Use a layer with at least two features that have every attribute'
      }
    ],
    outputs: [
      {
        tableId: 'similarity',
        kind: 'attributes',
        kindFrom: 'features',
        // Rank 0 is the reference, 1 the most similar. The flag column `is_similar` marks the first
        // `resultCount`; the rank gives the order.
        presentation: {column: 'rank', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['which-groups-or-regions']
      }
    ],
    notes: [
      'Explicitly choose referenceIndex (source_index); there is no selected reference row by default.',
      'Similarity is the distance between standardized attributes; geography is ignored.'
    ]
  },
  {
    bindingId: 'location-allocation',
    questions: ['what-is-near-or-reachable'],
    action: 'Choose best facility sites',
    prominence: 2,
    inputs: [
      {
        name: 'demand',
        kind: 'points',
        fields: [
          {
            role: 'weight',
            label: 'Demand weight',
            accepts: ['number', 'integer'],
            // The cost is weight times distance, so the weight must be a total, not a rate.
            measurement: ['count', 'amount'],
            required: false,
            parameter: 'weightProperty'
          }
        ]
      },
      {name: 'candidates', kind: 'points', fields: []}
    ],
    parameters: [
      describeConstant('facilityCount', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('demand', 5000),
      ...describeCapacity('candidates', 500),
      {
        input: 'candidates',
        id: 'minimum-rows',
        severity: 'block',
        // The binding needs at least as many candidates as facilities; 3 is the default facility
        // count, so this limit is only right for the default.
        limit: 3,
        remedy: 'Offer at least as many candidate sites as facilities (3 for the default count)'
      },
      {
        input: 'demand',
        id: 'non-negative-values',
        severity: 'warn',
        role: 'weight',
        remedy: 'Demand points with a negative weight are skipped'
      },
      describeDeclaredCoordinates('demand'),
      describeDeclaredCoordinates('candidates')
    ],
    outputs: [
      {
        tableId: 'facilities',
        kind: 'points',
        kindFrom: 'candidates',
        presentation: {column: 'demand_weight', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-much-per-area']
      },
      {
        tableId: 'assignments',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'The minimum of 3 candidates is the default facility count; with a different count the minimum is that count.',
      'A greedy add followed by single swaps, so the result can be a local optimum; distances are straight lines.'
    ]
  },
  {
    bindingId: 'weights-algebra',
    questions: ['prepare-data'],
    action: 'Combine two neighbor rules',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('neighborCount', 'basic'),
      // The right operand is a distance band with decay 1 / (1 + d / band).
      {
        name: 'distanceBand',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'outlier-resistant-nearest-neighbor-distance',
          factor: DISTANCE_BAND_MARGIN
        },
        level: 'basic'
      },
      describeConstant('operation', 'basic'),
      // Only neighbors present in both operands take the rule (union and intersection).
      describeConstant('weightRule', 'advanced', {
        parameter: 'operation',
        equals: ['union', 'intersection']
      }),
      describeConstant('coordinateSystem', 'advanced'),
      describeConstant('chunkRows', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 4000),
      {
        input: 'points',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Use the union, or raise the distance band'
      },
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      {
        tableId: 'weights-edges',
        kind: 'spatial-weights',
        presentation: {column: 'weight', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ]
  },
  {
    bindingId: 'weights-transpose',
    questions: ['prepare-data'],
    action: 'Reverse neighbor links',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('neighborCount', 'basic'),
      // Zero means binary weights; a decay distance is a modelling choice, not a data statistic.
      describeConstant('decayDistance', 'advanced'),
      describeConstant('coordinateSystem', 'advanced'),
      describeConstant('chunkRows', 'advanced')
    ],
    requirements: [...describeCapacity('points', 4000), describeDeclaredCoordinates('points')],
    outputs: [
      {
        tableId: 'weights-edges',
        kind: 'spatial-weights',
        presentation: {column: 'weight', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ]
  },
  {
    bindingId: 'weights-transform',
    questions: ['prepare-data'],
    action: 'Standardize spatial weights',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('neighborCount', 'basic'),
      describeConstant('decayDistance', 'advanced'),
      describeConstant('coordinateSystem', 'advanced'),
      describeConstant('chunkRows', 'advanced'),
      describeConstant('operation', 'basic')
    ],
    requirements: [...describeCapacity('points', 4000), describeDeclaredCoordinates('points')],
    outputs: [
      {
        tableId: 'weights-edges',
        kind: 'spatial-weights',
        presentation: {column: 'weight', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ]
  },
  {
    bindingId: 'regression-ols',
    questions: ['how-related'],
    action: 'Explain a value with other columns',
    prominence: 1,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'response',
            label: 'Value to explain',
            // Booleans are not read by the binding, so only numeric columns are offered.
            accepts: ['number', 'integer'],
            measurement: RESPONSE_MEASUREMENT,
            required: true,
            parameter: 'responseProperty'
          },
          describePredictorsRole(15)
        ]
      }
    ],
    parameters: [],
    // The code needs more complete rows than coefficients (rows greater than predictors plus
    // one), so one predictor needs three rows.
    requirements: describeRegressionRequirements('points', 100000, 3),
    outputs: [
      {
        tableId: 'ols-coefficients',
        kind: 'summary',
        interpretation: {summary: 'regression-fit', cautions: []},
        next: ['where-significant']
      },
      {
        tableId: 'ols-residuals',
        kind: 'points',
        kindFrom: 'points',
        presentation: {column: 'residual', scale: 'diverging', center: 0},
        interpretation: {summary: 'regression-fit', cautions: []},
        // Clustered residuals mean the model misses a spatial pattern.
        next: ['where-significant']
      }
    ],
    notes: [
      'This regression does not test the residuals for spatial autocorrelation, unlike the regression with diagnostics; map the residuals and test them before trusting the standard errors.',
      'A count as the response draws a warning: use a rate, or a count model. Coefficients come with t statistics only, no p-values.'
    ]
  },
  {
    bindingId: 'global-join-counts-getis-ord',
    // GPU only: verified by reading, not run by the spec. G needs non-negative values and the
    // join counts treat a non-zero value as black, so a continuous column is meaningful for G
    // only and a 0/1 column for the joins.
    questions: ['where-significant'],
    action: 'Test clustering (G, join counts)',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {
            ...describeAreaValueRole('Value to test (0 or 1 for joins)'),
            parameter: 'valueProperty'
          }
        ]
      }
    ],
    parameters: describeWeightsParameters('features', {
      neighborhoodReason:
        ' The binding itself advises queen or rook for join counts: k-nearest-neighbor links are not symmetric.'
    }),
    requirements: [
      ...describeCapacity('features', 3000, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        limit: 4,
        remedy: 'Use a layer with at least four features that have a value'
      },
      {
        input: 'features',
        id: 'non-negative-values',
        severity: 'warn',
        role: 'value',
        remedy: 'General G is defined for non-negative values'
      },
      {
        input: 'features',
        id: 'varying-values',
        severity: 'warn',
        role: 'value',
        remedy: 'Choose a column whose values differ between features'
      },
      {
        input: 'features',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Use queen or rook neighbors for join counts; knn links are not symmetric'
      },
      describeDeclaredCoordinates('features'),
      {
        input: 'features',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['objects'],
        remedy: 'Aggregate the points to areas or cells first'
      }
    ],
    outputs: [
      {
        tableId: 'global',
        kind: 'summary',
        interpretation: {summary: 'global-statistic', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'Join counts treat every non-zero value as black, so a continuous column makes every area black; use a 0 or 1 column for the joins and read General G on the value itself.',
      'Weights are binary. This operation needs a GPU.'
    ]
  },
  {
    bindingId: 'local-lisa-variants',
    // GPU only: verified by reading.
    questions: ['where-significant'],
    action: 'Find local associations',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {...describeAreaValueRole('Value to map'), parameter: 'valueProperty'},
          {
            role: 'secondValue',
            label: 'Second value (bivariate Moran)',
            accepts: ['number', 'integer'],
            measurement: ['rate', 'amount'],
            measurementByGeometry: {polygon: ['rate'], point: ['rate', 'amount']},
            required: {whenParameter: 'variant', equals: ['local-bivariate-moran']},
            parameter: 'secondValueProperty'
          }
        ]
      }
    ],
    parameters: [
      describeConstant('variant', 'basic'),
      ...describeWeightsParameters('features', {distanceBand: 'advanced'}),
      describePermutations(),
      describeConstant('seed', 'advanced'),
      describeConstant('significance', 'basic')
    ],
    requirements: [
      ...describeLocalStatisticRequirements('features', 'value', ['objects']),
      {
        input: 'features',
        id: 'non-negative-values',
        severity: 'block',
        role: 'value',
        // G and G* need non-negative values; signed bivariate Moran is valid.
        activeWhen: {parameter: 'variant', equals: ['local-getis-ord-g', 'local-getis-ord-g-star']},
        remedy:
          'Local G and G* fail on a negative value; remove or shift it, or use the bivariate Moran variant'
      },
      {
        input: 'features',
        id: 'minimum-distinct-values',
        severity: 'warn',
        limit: 2,
        role: 'value',
        activeWhen: {parameter: 'variant', equals: ['local-join-count']},
        // The contract can check distinctness, but cannot express an exact 0/1 range.
        remedy: 'The local join count variant needs a column that holds only 0 and 1'
      }
    ],
    outputs: [
      {
        tableId: 'local-lisa',
        kind: 'polygons',
        kindFrom: 'features',
        // `significant` alone cannot tell a high cluster from a low one. The statistic can: a high
        // G* or join count is a high-value neighborhood. A diverging scale centred on 0 would be
        // wrong for G and G* (never negative), so it is sequential.
        presentation: {column: 'statistic', scale: 'sequential'},
        interpretation: {
          summary: 'local-cluster-count',
          cautions: ['pseudo-p-values', 'multiple-testing-uncorrected']
        },
        next: ['prepare-data', 'how-related']
      }
    ],
    notes: [
      'Local G and G* fail the whole run on a negative value; local join counts need a column of 0 and 1; a feature with more than 32 neighbors also fails the run (the contract cannot cap neighbors, so lower the neighbor count or the distance band). This operation needs a GPU.',
      'The map shows the statistic; significance is in the `significant` and `p_value` columns, with no correction for the number of tests.'
    ]
  },
  {
    bindingId: 'recipe-hot-spot-lattice',
    questions: ['where-significant'],
    action: 'Find hot spots in points',
    prominence: 1,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        // The lattice counts occurrences, so the rows are incidents.
        rowSemantics: ['events'],
        fields: [
          {
            role: 'value',
            label: 'Value to sum per cell (empty counts points)',
            accepts: ['number', 'integer'],
            // Cell sums add up, so the value must be a count or an amount, never a rate.
            measurement: ['count', 'amount'],
            required: false,
            parameter: 'valueProperty'
          }
        ]
      }
    ],
    parameters: [
      // A smaller cell resolves small clusters but has sparser counts. The factor comparison
      // above weighs both null discoveries and planted-cluster detection, not flagged-cell counts.
      {
        name: 'cellSize',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'mean-point-spacing',
          factor: HOT_SPOT_CELL_SPACING_FACTOR
        },
        level: 'basic'
      },
      describeConstant('criterion', 'advanced'),
      describeConstant('radius', 'basic'),
      describeConstant('minimumNeighbors', 'advanced'),
      describeConstant('multipleTesting', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 100000),
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'warn',
        limit: 30,
        remedy: 'Add more points or treat the result as exploratory'
      },
      describeDeclaredCoordinates('points'),
      {
        input: 'points',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['events'],
        remedy: 'For measured values at places, use a hot spot analysis on the features instead'
      }
    ],
    outputs: [
      {
        tableId: 'recipe-hot-spots',
        kind: 'polygons',
        presentation: {
          column: 'bin',
          scale: 'class-scheme',
          classScheme: 'hot-spot-confidence-bins',
          classEncoding: 'contributor-integer'
        },
        interpretation: {summary: 'significant-hot-cold-count', cautions: []},
        next: ['which-groups-or-regions', 'how-related']
      }
    ],
    notes: [
      'The default cell size follows point density: at 1.5 times the mean spacing it holds about 2.25 points on average. Compare a cell half the size to check sensitivity to small clusters.',
      'Every cell of the extent is tested, including empty cells. Benjamini-Hochberg is the default correction. Count tests use conservative binomial tails with partial boundary-cell exposure; overlapping neighborhoods share counts, so assess sensitivity to the cell size and neighborhood.'
    ]
  },
  {
    bindingId: 'regression-gwr',
    questions: ['how-related'],
    action: 'Let relationships vary by place',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'response',
            label: 'Value to explain',
            accepts: ['number', 'integer'],
            measurement: RESPONSE_MEASUREMENT,
            required: true,
            parameter: 'responseProperty'
          },
          describePredictorsRole(10)
        ]
      }
    ],
    parameters: [
      describeConstant('kernel', 'basic'),
      // 0.75 of the standard distance of the points (the root mean squared distance from their
      // centre). For uniform points the bandwidth then covers about 29 neighbors of 100 at the
      // centre (n times pi times (0.75 * 0.41 W)^2 / W^2), ten times the 3 coefficients of a
      // two-predictor fit. Searched on random fixtures of three scales (the ratio is
      // scale-free) and two shapes (uniform and three clusters), 4 seeds each: factors 0.25, 0.35
      // and 0.5 left singular local fits (29, 8 and 2 of 100 on uniform points); 0.75 left none.
      // The constant 1,000 made 83 of 100 fits singular on a 20 km extent. It is not an optimised
      // bandwidth: with fewer than about 60 points the support holds fewer neighbors, and a
      // search over bandwidths (AICc) is what the binding lacks.
      {
        name: 'bandwidth',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'standard-distance',
          factor: 0.75
        },
        level: 'basic'
      },
      describeConstant('coordinateSystem', 'advanced')
    ],
    // The code needs more locations than predictors plus two, so one predictor needs four.
    requirements: [
      ...describeRegressionRequirements('points', 3000, 4),
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      {
        tableId: 'gwr-local',
        kind: 'points',
        kindFrom: 'points',
        // Coefficient columns are named after the predictors, so the fixed residual column is
        // the one a default map can use.
        presentation: {column: 'residual', scale: 'diverging', center: 0},
        interpretation: {summary: 'regression-fit', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'The bandwidth is a fixed distance, set from the spread of the points, not optimised; read the singular-fit count in the run summary and widen the bandwidth if it is not zero.',
      'No local t values or collinearity diagnostics are published, and a count as the response draws a warning: use a rate, or a count model.'
    ]
  },
  {
    bindingId: 'regression-probit',
    questions: ['how-related'],
    action: 'Predict a yes or no outcome',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'response',
            label: 'Outcome (0 or 1)',
            accepts: ['integer', 'number'],
            // Exactly the values 0 and 1 are accepted; a two-class code, not a quantity.
            measurement: ['nominal', 'ordinal'],
            required: true,
            parameter: 'responseProperty'
          },
          describePredictorsRole(10)
        ]
      }
    ],
    parameters: [describeConstant('coordinateSystem', 'advanced')],
    requirements: [
      ...describeRegressionRequirements('points', 20000, 3),
      {
        input: 'points',
        id: 'minimum-distinct-values',
        severity: 'block',
        limit: 2,
        role: 'response',
        remedy: 'The outcome needs both a 0 and a 1'
      }
    ],
    outputs: [
      {
        tableId: 'probit-coefficients',
        kind: 'summary',
        interpretation: {summary: 'regression-fit', cautions: []},
        next: ['where-significant']
      },
      {
        tableId: 'probit-fitted',
        kind: 'points',
        kindFrom: 'points',
        presentation: {column: 'probability', scale: 'sequential'},
        interpretation: {summary: 'regression-fit', cautions: []}
      }
    ],
    notes: [
      'The outcome must be exactly 0 and 1; samples that a predictor separates completely are refused.'
    ]
  },
  {
    bindingId: 'regression-sur',
    questions: ['how-related'],
    action: 'Fit related equations together',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'response',
            label: 'Values to explain, one per equation',
            accepts: ['number', 'integer'],
            measurement: RESPONSE_MEASUREMENT,
            required: true,
            multiple: {minimum: 2, maximum: 4},
            parameter: 'responseProperties'
          },
          {
            // The binding reads one comma-separated list per equation and separates the equations
            // with semicolons, every list as long as the others (one to seven columns, at most 28
            // names). The contract's `multiple` documents a plain comma list, so the host must
            // write this parameter in the semicolon form.
            role: 'predictors',
            label: 'Explanatory columns per equation',
            accepts: ['number', 'integer'],
            measurement: PREDICTOR_MEASUREMENT,
            required: true,
            multiple: {minimum: 2, maximum: 28},
            parameter: 'predictorProperties'
          }
        ]
      }
    ],
    parameters: [describeConstant('coordinateSystem', 'advanced')],
    // The code needs more complete rows than predictors per equation plus one: two rows for one
    // predictor each, with the number of equations adding no row.
    requirements: describeRegressionRequirements('points', 20000, 3),
    outputs: [
      {
        tableId: 'sur-coefficients',
        kind: 'summary',
        interpretation: {summary: 'regression-fit', cautions: []},
        next: ['where-significant']
      },
      {
        tableId: 'sur-residuals',
        kind: 'points',
        kindFrom: 'points',
        // One residual column per equation; the first is always present.
        presentation: {column: 'residual_0', scale: 'diverging', center: 0},
        interpretation: {summary: 'regression-fit', cautions: []}
      }
    ],
    notes: [
      'Predictors are written one list per equation, equations separated by semicolons, every list the same length.'
    ]
  }
];
