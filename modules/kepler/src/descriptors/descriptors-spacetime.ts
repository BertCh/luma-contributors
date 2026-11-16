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
import {
  SPACETIME_MAXIMUM_FEATURES,
  SPACETIME_MAXIMUM_LOCATIONS,
  SPACETIME_MAXIMUM_PERIODS
} from '../analysis/catalog-spacetime-common';

// The binding reads numeric epochs, not ISO strings or Date objects. A profile's generic
// `time` type does not prove numeric storage, so only numeric columns are offered here.
const TIME_FIELD: KeplerOperationFieldDescriptor = {
  role: 'time',
  label: 'Numeric epoch time',
  parameter: 'timeProperty',
  accepts: ['number', 'integer'],
  required: true
};
const LOCATION_FIELD: KeplerOperationFieldDescriptor = {
  role: 'location',
  label: 'Location identity (blank groups exact coordinates)',
  parameter: 'locationProperty',
  accepts: ['category', 'text', 'integer', 'number', 'boolean'],
  measurement: ['identifier', 'nominal'],
  required: false
};
const VALUE_FIELD: KeplerOperationFieldDescriptor = {
  role: 'value',
  label: 'Value at each location and period',
  parameter: 'valueProperty',
  accepts: ['number', 'integer'],
  measurement: ['count', 'amount', 'rate'],
  required: true
};
const TIME_NOTES = [
  'Time must be finite numeric epoch milliseconds or seconds, selected with timeUnit; ISO strings and Date objects must be converted first. The default unit is milliseconds and is not inferred.',
  'Epoch origins are subtracted in float64 before upload as relative float32 seconds. Temporal bandwidths, thresholds and maximum lags use seconds; series lags and windows use periods. The time-resolution guard rejects spans too coarse for the chosen interval.',
  'EPSG:4326 uses a local azimuthal equidistant frame in metres; planar data keep their coordinate units. Confirm the coordinate model before interpreting distances. GPU only, with at most 512 input rows; null results mean undefined or not tested.'
] as const;
const SERIES_NOTES = [
  ...TIME_NOTES,
  'A location series needs a complete location/time rectangle, unique location/time rows, fixed Point coordinates at each location, and equally spaced periods. Periods are sorted by time; rows need not arrive sorted. Blank locationProperty groups by exact original Point coordinates.',
  'At most 64 locations and 2 to 128 periods fit within the 512-row cap. Distinct-column checks cannot establish a complete rectangle or count coordinate-based locations; those checks remain in the binding.'
] as const;
const INFERENCE_NOTES =
  'P-values are uncorrected for multiple testing. These bindings offer no correction setting; interpret repeated tests as exploratory, and recompute inference on the full population before keeping it.';
const SCALE_NOTES =
  'Temporal scale defaults to one tenth of the observed time range in seconds; this is an exploratory starting value, not a fitted bandwidth. Missing or degenerate profile evidence falls back to the binding constant, with provenance. Spatial thresholds use an outlier-resistant neighbor distance, falling back to Scott bandwidth for repeated locations. Spatial scale rules use metres for geographic data and coordinate units for planar data.';

function describeConstant(
  name: string,
  level: 'basic' | 'advanced' = 'advanced',
  apply: 'live' | 'rebuild' = 'rebuild'
): KeplerOperationParameterDescriptor {
  return {name, level, apply, defaultPolicy: {from: 'constant'}};
}

function describeSpatialScale(
  name: string,
  rule:
    | 'normal-reference-bandwidth-scott'
    | 'outlier-resistant-nearest-neighbor-distance'
    | 'extent-diagonal',
  factor = 1
): KeplerOperationParameterDescriptor {
  return {
    name,
    level: 'basic',
    apply: 'rebuild',
    defaultPolicy: {
      from: 'data',
      input: 'points',
      rule,
      factor,
      // Repeated places can give zero event-to-event nearest distances. Spread still has a scale.
      ...(rule === 'outlier-resistant-nearest-neighbor-distance'
        ? {fallback: {rule: 'normal-reference-bandwidth-scott' as const}}
        : {})
    }
  };
}

function describeTemporalScale(name: string): KeplerOperationParameterDescriptor {
  return {
    name,
    level: 'basic',
    apply: 'rebuild',
    defaultPolicy: {
      from: 'data',
      input: 'points',
      role: 'time',
      rule: 'column-range',
      factor: 0.1,
      factorByParameter: {parameter: 'timeUnit', factors: {milliseconds: 0.001, seconds: 1}}
    }
  };
}

function describeNoMissing(role: string): KeplerOperationRequirement {
  return {
    input: 'points',
    id: 'no-missing-values',
    severity: 'block',
    role,
    remedy: `Supply a finite ${role} value on every input row`
  };
}

function describePoints(minimumRows: number, events = true): KeplerOperationRequirement[] {
  return [
    {
      input: 'points',
      id: 'minimum-rows',
      severity: 'block',
      limit: minimumRows,
      remedy: `Supply at least ${minimumRows} Point observations`
    },
    {
      input: 'points',
      id: 'maximum-rows',
      severity: 'block',
      limit: SPACETIME_MAXIMUM_FEATURES,
      remedy: 'Aggregate or restrict the input to at most 512 observations'
    },
    {
      input: 'points',
      id: 'maximum-vertices',
      severity: 'block',
      limit: SPACETIME_MAXIMUM_FEATURES,
      remedy: 'Supply one Point per observation, at most 512 in total'
    },
    {
      input: 'points',
      id: 'declared-coordinates',
      severity: 'warn',
      remedy: 'Declare whether the Point coordinates are geographic or planar'
    },
    describeNoMissing('time'),
    ...(events
      ? [
          {
            input: 'points',
            id: 'row-semantics',
            severity: 'warn',
            expects: ['events'],
            remedy:
              'Use event occurrences rather than repeated measurements of a continuous surface'
          } satisfies KeplerOperationRequirement
        ]
      : [])
  ];
}

function describeSeries(
  minimumPeriods = 2,
  minimumLocations?: number
): KeplerOperationRequirement[] {
  return [
    ...describePoints(minimumPeriods * (minimumLocations ?? 1), false),
    describeNoMissing('value'),
    describeNoMissing('location'),
    {
      input: 'points',
      id: 'minimum-distinct-values',
      severity: 'block',
      role: 'time',
      limit: minimumPeriods,
      remedy: `Supply at least ${minimumPeriods} equally spaced periods, and more if the selected lag requires them`
    },
    {
      input: 'points',
      id: 'maximum-distinct-values',
      severity: 'block',
      role: 'time',
      limit: SPACETIME_MAXIMUM_PERIODS,
      remedy: 'Aggregate the time axis to at most 128 equally spaced periods'
    },
    {
      input: 'points',
      id: 'maximum-distinct-values',
      severity: 'block',
      role: 'location',
      limit: SPACETIME_MAXIMUM_LOCATIONS,
      remedy: 'Restrict the input to at most 64 fixed locations'
    },
    ...(minimumLocations === undefined
      ? []
      : [
          {
            input: 'points',
            id: 'minimum-distinct-values',
            severity: 'block',
            role: 'location',
            limit: minimumLocations,
            remedy: `Supply at least ${minimumLocations} fixed locations`
          } satisfies KeplerOperationRequirement
        ])
  ];
}

function describeRequiredPeriods(
  parameter: string,
  offset = 0,
  absolute = false
): KeplerOperationRequirement {
  return {
    input: 'points',
    id: 'minimum-distinct-values',
    severity: 'block',
    role: 'time',
    limitFromParameter: {parameter, offset, absolute},
    remedy: `Supply enough periods for ${parameter}, or reduce ${parameter}`
  };
}

function describeOutput(
  tableId: 'grid' | 'points' | 'statistics' | 'bands',
  column?: string,
  scale: 'sequential' | 'diverging' | 'categorical' = 'sequential',
  cautions: NonNullable<KeplerOperationOutputDescriptor['interpretation']>['cautions'] = [],
  center?: number
): KeplerOperationOutputDescriptor {
  return {
    tableId,
    kind: tableId === 'grid' ? 'surface' : tableId === 'points' ? 'points' : 'summary',
    ...(column ? {presentation: {column, scale, ...(center === undefined ? {} : {center})}} : {}),
    // Raw statistics are not significance classes. Table-only outputs retain an honest row count.
    interpretation: {summary: column ? 'value-range' : 'row-count', cautions},
    next: ['what-changed', 'how-related']
  };
}

const TIME_SETTINGS = [describeConstant('coordinateSystem'), describeConstant('timeUnit', 'basic')];
const RANDOM_SETTINGS: readonly KeplerOperationParameterDescriptor[] = [
  {
    name: 'permutations',
    level: 'advanced',
    apply: 'rebuild',
    defaultPolicy: {
      from: 'preset',
      value: 999,
      reason:
        'Use 999 replicates for finer Monte Carlo p-value resolution than the binding default of 19; no multiple testing correction is applied'
    }
  },
  describeConstant('seed')
];
const DENSITY_SETTINGS = [
  describeSpatialScale('spatialBandwidth', 'normal-reference-bandwidth-scott'),
  describeConstant('gridSize', 'basic')
];
const TEMPORAL_DENSITY_SETTINGS = [
  describeTemporalScale('temporalBandwidth'),
  describeConstant('sliceCount', 'basic')
];
const THRESHOLD_SETTINGS = [
  describeSpatialScale('spatialThreshold', 'outlier-resistant-nearest-neighbor-distance'),
  describeTemporalScale('temporalThreshold')
];
const PAIR_SETTINGS = [
  describeSpatialScale('spatialBandwidth', 'normal-reference-bandwidth-scott'),
  describeSpatialScale('maximumDistance', 'extent-diagonal', 1 / 3),
  describeTemporalScale('maximumTimeLag'),
  describeConstant('classCount', 'basic')
];

/** Hand-written semantics for all fourteen bounded space-time catalog bindings. */
export const SPACETIME_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'space-time-kernel-density',
    questions: ['where-concentrated', 'what-changed'],
    action: 'Map density through time',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [...TIME_SETTINGS, ...DENSITY_SETTINGS, ...TEMPORAL_DENSITY_SETTINGS],
    requirements: describePoints(1),
    outputs: [describeOutput('grid', 'value')],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      'Product Gaussian intensity density, truncated at five bandwidths, without edge correction. The bandwidth-padded grid has at most 32 × 32 cells and 16 slices (16,384 voxels). Equal-time observations produce one slice; filter a time slice before interpreting coincident map points.'
    ]
  },
  {
    bindingId: 'space-time-relative-risk',
    questions: ['where-concentrated', 'what-changed'],
    action: 'Compare case and control density',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['events'],
        fields: [
          TIME_FIELD,
          {
            role: 'case',
            label: 'Case indicator (numeric 0 control, 1 case)',
            parameter: 'caseProperty',
            accepts: ['number', 'integer'],
            measurement: ['nominal'],
            required: true
          }
        ]
      }
    ],
    parameters: [...TIME_SETTINGS, ...DENSITY_SETTINGS, ...TEMPORAL_DENSITY_SETTINGS],
    requirements: [
      ...describePoints(2),
      describeNoMissing('case'),
      {
        input: 'points',
        id: 'numeric-range',
        severity: 'block',
        role: 'case',
        minimum: 0,
        maximum: 1,
        integer: true,
        remedy: 'Encode controls as 0 and cases as 1'
      },
      {
        input: 'points',
        id: 'minimum-distinct-values',
        severity: 'block',
        role: 'case',
        limit: 2,
        remedy: 'Include at least one case and one control'
      },
      {
        input: 'points',
        id: 'maximum-distinct-values',
        severity: 'block',
        role: 'case',
        limit: 2,
        remedy: 'Use exactly two indicator values: numeric 0 and 1'
      }
    ],
    outputs: [describeOutput('grid', 'value', 'diverging', [], 1)],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      'Requires literal numeric 0 and 1 with both cases and controls present; range and integer checks enforce the encoding alongside the two-population checks. Ratio of probability densities, not raw case counts or causal risk. Very small control density makes the ratio unstable; zero control density gives null. No edge correction; both groups share the grid and bandwidths. At most 16,384 voxels; filter a time slice before mapping.'
    ]
  },
  {
    bindingId: 'adaptive-bandwidth-kernel-density',
    questions: ['where-concentrated'],
    action: 'Map adaptive spatial density',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['events'],
        fields: [{...TIME_FIELD, label: 'Numeric epoch time (identity metadata only)'}]
      }
    ],
    parameters: [
      ...TIME_SETTINGS,
      ...DENSITY_SETTINGS,
      describeConstant('sensitivity', 'basic', 'live'),
      describeConstant('factorLimit')
    ],
    requirements: describePoints(1),
    outputs: [describeOutput('grid', 'value'), describeOutput('points', 'bandwidth')],
    notes: [
      ...TIME_NOTES,
      'Spatial Abramson adaptive Gaussian intensity density; the required time column is retained as metadata and is not used by the spatial contributor. No time-resolution guard is needed. The grid has at most 1,024 cells. Pilot bandwidth factors are clamped by factorLimit; bandwidth is in metres or planar coordinate units.'
    ]
  },
  {
    bindingId: 'space-time-hawkes',
    questions: ['how-related', 'where-concentrated'],
    action: 'Fit event triggering through time',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [
      ...TIME_SETTINGS,
      ...THRESHOLD_SETTINGS,
      describeConstant('maximumIterations'),
      describeConstant('maximumParents')
    ],
    requirements: describePoints(2),
    outputs: [describeOutput('points', 'background_probability'), describeOutput('statistics')],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      'Exponential-time Gaussian-space Hawkes fit by EM, not proof of causation. Spatial and temporal thresholds truncate candidate parents. The study area is inferred from padded event bounds and duration is at least one second. Inspect converged, iterationLimitReached, candidateOverflow and approximation before trusting estimates; at most 100 iterations and 512 parents per event.'
    ]
  },
  {
    bindingId: 'expectation-based-space-time-scan',
    questions: ['where-significant', 'what-changed'],
    action: 'Scan for excess counts over time',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          TIME_FIELD,
          LOCATION_FIELD,
          {...VALUE_FIELD, label: 'Observed integer count', measurement: ['count']},
          {
            role: 'baseline',
            label: 'Expected count (strictly positive)',
            parameter: 'baselineProperty',
            accepts: ['number', 'integer'],
            measurement: ['count'],
            required: true
          }
        ]
      }
    ],
    // Float64 storage remains eligible; the requirement checks integrality without replacing
    // a fractional observed-count column with an unrelated integer-valued column.
    parameters: [
      ...TIME_SETTINGS,
      ...RANDOM_SETTINGS,
      describeConstant('maximumWindowLength', 'basic'),
      describeConstant('neighborCount', 'basic')
    ],
    requirements: [
      ...describeSeries(),
      describeNoMissing('baseline'),
      {
        input: 'points',
        id: 'numeric-range',
        severity: 'block',
        role: 'value',
        minimum: 0,
        maximum: 16777215,
        integer: true,
        remedy: 'Supply observed integer counts between 0 and 16,777,215'
      },
      {
        input: 'points',
        id: 'positive-values',
        severity: 'block',
        role: 'baseline',
        remedy: 'Supply a strictly positive expected count for every location and period'
      }
    ],
    outputs: [
      describeOutput('points', 'cluster_member', 'categorical'),
      describeOutput('statistics')
    ],
    notes: [
      ...SERIES_NOTES,
      'Expectation-based Poisson scan: observed counts must be integers from 0 to 16,777,215; the supplied baseline is an expected count, not a population denominator, and is never fitted here. Windows end at the latest period, with at most 32 periods and 32 nearest locations, clamped to available periods and locations. Poisson Monte Carlo replicates give a minimum p-value of 1 / (permutations + 1); they are not permutations of observed counts. The Monte Carlo p-value is for the selected cluster under this scan, not a separate significance test at every member row.'
    ]
  },
  {
    bindingId: 'space-time-cross-correlation',
    questions: ['how-related', 'what-changed'],
    action: 'Correlate two series through time',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          TIME_FIELD,
          LOCATION_FIELD,
          VALUE_FIELD,
          {
            ...VALUE_FIELD,
            role: 'second-value',
            label: 'Second location series',
            parameter: 'secondValueProperty'
          }
        ]
      }
    ],
    parameters: [...TIME_SETTINGS, describeConstant('lag', 'basic')],
    requirements: [
      ...describeSeries(4),
      describeRequiredPeriods('lag', 4, true),
      describeNoMissing('second-value')
    ],
    outputs: [
      describeOutput('points', 'correlation', 'diverging', ['multiple-testing-uncorrected'], 0)
    ],
    notes: [
      ...SERIES_NOTES,
      INFERENCE_NOTES,
      'One signed lag in periods; periodCount - abs(lag) must be at least four (the default lag of 1 needs five periods). Pearson correlation and Fisher-z approximate two-sided p-values; constant series yield undefined results. Per-location results are repeated on every observation, not independent tests per row.'
    ]
  },
  {
    bindingId: 'space-time-local-moran',
    questions: ['where-significant', 'what-changed'],
    action: 'Test local association through time',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: [TIME_FIELD, LOCATION_FIELD, VALUE_FIELD]}],
    parameters: [
      ...TIME_SETTINGS,
      ...RANDOM_SETTINGS,
      describeConstant('neighborCount', 'basic'),
      describeConstant('lag', 'basic'),
      describeConstant('variant', 'basic')
    ],
    requirements: [...describeSeries(2, 4), describeRequiredPeriods('lag', 1)],
    outputs: [
      describeOutput(
        'points',
        'observed',
        'diverging',
        ['pseudo-p-values', 'multiple-testing-uncorrected', 'row-order-dependent-inference'],
        0
      )
    ],
    notes: [
      ...SERIES_NOTES,
      INFERENCE_NOTES,
      'Requires at least four locations and a positive lag strictly below periodCount. Lagged inward or differential Moran uses equal positive k-nearest weights, at most 32 neighbors, clamped below the location count. Only latest-period source rows are tested; other rows are null. Conditional permutation two-sided p-values, not the corrected spatial-only Moran binding. Spatial ties follow source order.'
    ]
  },
  {
    bindingId: 'jacquez-test',
    questions: ['how-related', 'where-significant'],
    action: 'Test space and time neighbor overlap',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [...TIME_SETTINGS, describeConstant('neighborCount', 'basic'), ...RANDOM_SETTINGS],
    requirements: [
      ...describePoints(2),
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'block',
        limitFromParameter: {parameter: 'neighborCount', offset: 1},
        remedy: 'Reduce neighborCount below the number of events, or supply more events'
      }
    ],
    outputs: [
      describeOutput('statistics', undefined, 'sequential', [
        'pseudo-p-values',
        'row-order-dependent-inference'
      ])
    ],
    notes: [
      ...TIME_NOTES,
      'Global directed overlap of spatial and temporal k-nearest event neighbors; no per-point output. neighborCount must be strictly below event count. Time-permutation greater-tail p-value. Spatial ties use source row order and temporal ties use stable time order.'
    ]
  },
  {
    bindingId: 'local-knox',
    questions: ['where-significant', 'how-related'],
    action: 'Test local space-time close pairs',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [...TIME_SETTINGS, ...THRESHOLD_SETTINGS, ...RANDOM_SETTINGS],
    requirements: describePoints(2),
    outputs: [
      describeOutput('points', 'observed', 'sequential', [
        'pseudo-p-values',
        'multiple-testing-uncorrected'
      ])
    ],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      INFERENCE_NOTES,
      'Each event counts incident space-time close pairs with a shuffled-time greater-tail test. Spatial lists include both directions; do not sum event counts as unique pairs. Closed radius filtering compares uploaded float32 distances and thresholds.'
    ]
  },
  {
    bindingId: 'near-repeat-knox',
    questions: ['how-related', 'where-significant'],
    action: 'Test near repeats by distance and time',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [
      ...TIME_SETTINGS,
      ...THRESHOLD_SETTINGS,
      ...RANDOM_SETTINGS,
      describeConstant('bandCount', 'basic')
    ],
    requirements: describePoints(2),
    outputs: [
      describeOutput('bands', undefined, 'sequential', [
        'pseudo-p-values',
        'multiple-testing-uncorrected'
      ])
    ],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      INFERENCE_NOTES,
      'At most 8 × 8 distance/time bands, with a greater-tail shuffled-time p-value per band. Each unordered pair is counted once. Bands are open on the left and closed on the right, except the first includes zero; comparisons use uploaded float32 distances and thresholds.'
    ]
  },
  {
    bindingId: 'space-time-diggle-test',
    questions: ['how-related', 'where-significant'],
    action: 'Test space-time interaction',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [...TIME_SETTINGS, ...PAIR_SETTINGS, ...RANDOM_SETTINGS],
    requirements: describePoints(2),
    outputs: [
      describeOutput('bands'),
      describeOutput('statistics', undefined, 'sequential', ['pseudo-p-values'])
    ],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      'Diggle interaction D and K surface with one global time-permutation test; band rows do not have individual p-values. Uses the splancs tie convention and no edge correction. The rectangular study window is inferred from event bounds plus spatialBandwidth padding; time is padded by one second. At most 8 × 8 classes. Default maximumDistance is one third of the extent diagonal, an exploratory scale that needs sensitivity checking.'
    ]
  },
  {
    bindingId: 'space-time-k-function',
    questions: ['how-related', 'where-concentrated'],
    action: 'Measure space-time interaction',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', rowSemantics: ['events'], fields: [TIME_FIELD]}],
    parameters: [...TIME_SETTINGS, ...PAIR_SETTINGS],
    requirements: describePoints(2),
    outputs: [describeOutput('bands')],
    notes: [
      ...TIME_NOTES,
      SCALE_NOTES,
      'Descriptive K and interaction D, with homogeneous intensity, the stpp tie convention and no edge correction; no p-value. The rectangular study window is inferred from event bounds plus spatialBandwidth padding; time is padded by one second. At most 8 × 8 classes. Default maximumDistance is one third of the extent diagonal, an exploratory scale that needs sensitivity checking.'
    ]
  },
  {
    bindingId: 'pettitt-test',
    questions: ['what-changed'],
    action: 'Find a change in each location series',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          TIME_FIELD,
          LOCATION_FIELD,
          {...VALUE_FIELD, measurement: ['count', 'amount', 'rate', 'ordinal']}
        ]
      }
    ],
    parameters: TIME_SETTINGS,
    requirements: describeSeries(),
    outputs: [
      describeOutput('points', 'change_time', 'sequential', ['multiple-testing-uncorrected'])
    ],
    notes: [
      ...SERIES_NOTES,
      INFERENCE_NOTES,
      'One rank-based Pettitt change per location with approximate analytic two-sided p-values. change_slice is a period index; change_time is in the input epoch unit. Per-location statistic, change and p-value are repeated on every observation, not independent discoveries per row.'
    ]
  },
  {
    bindingId: 'location-change-points',
    questions: ['what-changed'],
    action: 'Segment each location series',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: [TIME_FIELD, LOCATION_FIELD, VALUE_FIELD]}],
    parameters: [
      ...TIME_SETTINGS,
      describeConstant('penalty', 'basic', 'live'),
      describeConstant('minimumSegmentLength', 'basic', 'live'),
      describeConstant('cost', 'basic')
    ],
    requirements: [
      ...describeSeries(),
      describeRequiredPeriods('minimumSegmentLength'),
      {
        input: 'points',
        id: 'numeric-range',
        severity: 'block',
        parameter: 'minimumSegmentLength',
        minimum: 3,
        integer: true,
        activeWhen: {parameter: 'cost', equals: ['slope']},
        remedy: 'Set minimumSegmentLength to at least 3 for slope cost'
      },
      {
        input: 'points',
        id: 'numeric-range',
        severity: 'block',
        parameter: 'minimumSegmentLength',
        minimum: 2,
        integer: true,
        activeWhen: {parameter: 'cost', equals: ['variance', 'mean-variance']},
        remedy: 'Set minimumSegmentLength to at least 2 for variance costs'
      },
      {
        input: 'points',
        id: 'non-negative-values',
        severity: 'block',
        role: 'value',
        activeWhen: {parameter: 'cost', equals: ['poisson']},
        remedy: 'Use nonnegative values for Poisson segmentation, or choose another cost'
      }
    ],
    outputs: [describeOutput('points', 'change_count')],
    notes: [
      ...SERIES_NOTES,
      'Optimal partitioning with no p-value. minimumSegmentLength cannot exceed periodCount; variance and mean-variance require at least two periods per segment, slope at least three, other costs at least one. Poisson costs require nonnegative values. At most 32 change points per location; overflow fails the run. change_count is repeated per location, while segment and is_change_point describe individual observations. Penalty is cost-dependent, so inspect sensitivity rather than treating the constant 10 as universal.'
    ]
  }
];
