// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisCaution,
  GPUAnalysisParameterCondition,
  GPUAnalysisPresentationSpec,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {NETWORK_OPERATION_DESCRIPTORS} from './descriptors-network';
import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// These are semantic records for the existing bindings, not alternate execution definitions.
// All unmarked settings rebuild prepared inputs, allocations or compile-time contributor props.
// Live settings below are read from existing parameter views; scheduling stays with the host.
const PROPERTY_COSTS: GPUAnalysisParameterCondition = {parameter: 'costMode', equals: ['property']};
const VERTICAL_OPERATION: GPUAnalysisParameterCondition = {
  parameter: 'operation',
  equals: ['vertical', 'combined']
};
const INVERSE_DIRECTION: GPUAnalysisParameterCondition = {
  parameter: 'direction',
  equals: ['inverse']
};

// Reuse the existing graph-builder roles (including optional reverse costs and one-way flags).
const STREET_INPUT = NETWORK_OPERATION_DESCRIPTORS.find(
  descriptor => descriptor.bindingId === 'network-build-graph'
)!.inputs[0];

function describeParameters(
  names: readonly string[],
  basicNames: readonly string[],
  liveNames: readonly string[] = []
): KeplerOperationParameterDescriptor[] {
  return names.map(name => ({
    name,
    apply: liveNames.includes(name) ? 'live' : 'rebuild',
    level: basicNames.includes(name) ? 'basic' : 'advanced',
    defaultPolicy: {from: 'constant'}
  }));
}

function describeCapacity(
  input: string,
  maximumRows: number,
  maximumVertices = maximumRows
): KeplerOperationRequirement[] {
  return [
    {
      input,
      id: 'maximum-rows',
      severity: 'block',
      limit: maximumRows,
      remedy: 'Filter or aggregate the input to fit the source row limit'
    },
    {
      input,
      id: 'maximum-vertices',
      severity: 'block',
      limit: maximumVertices,
      remedy: 'Simplify the input or use a smaller area'
    }
  ];
}

function describeNumericField(
  parameter: string,
  role: string,
  label: string,
  required = true
): KeplerOperationFieldDescriptor {
  return {parameter, role, label, accepts: ['number', 'integer'], required};
}

function describeNumericRequirements(
  input: string,
  role: string,
  nonnegative = false
): KeplerOperationRequirement[] {
  return [
    {
      input,
      role,
      id: 'no-missing-values',
      severity: 'block',
      remedy: 'Supply a finite numeric value on every row'
    },
    ...(nonnegative
      ? [
          {
            input,
            role,
            id: 'non-negative-values' as const,
            severity: 'block' as const,
            remedy: 'Supply finite nonnegative weights on every row'
          }
        ]
      : [])
  ];
}

function describeCoordinates(input: string, geographic = false): KeplerOperationRequirement {
  return {
    input,
    id: geographic ? 'geographic-coordinates' : 'declared-coordinates',
    severity: geographic ? 'block' : 'warn',
    remedy: geographic
      ? 'Reproject the input points to longitude and latitude (EPSG:4326)'
      : 'Confirm EPSG:4326 for longitude/latitude or planar for projected coordinates'
  };
}

function describeOutput(
  tableId: string,
  kind: GPUSpatialAnalysisDataKind,
  presentation?: GPUAnalysisPresentationSpec,
  cautions: readonly GPUAnalysisCaution[] = []
): KeplerOperationOutputDescriptor {
  return {
    tableId,
    kind,
    ...(presentation ? {presentation} : {}),
    interpretation: {
      summary:
        presentation && ['sequential', 'diverging'].includes(presentation.scale)
          ? 'value-range'
          : 'row-count',
      cautions
    },
    next: ['how-related', 'what-changed']
  };
}

const FLOW_NOTE =
  'Zones need unique feature ids (row indices when ids are absent) and one centroid each. OD endpoint columns reference those identities, not coordinates. Missing pairs are zero; repeated OD pairs must be aggregated first. The profile cannot verify these cross-input identities.';
const FLOW_DEFAULT_NOTE =
  'Location weights, outflows and flow weights are nonnegative totals. Neighborhood size 2 and bandwidth ratio 0.5 are declared constants, not a distance bandwidth inferred from the map extent.';
const NETWORK_NOTE =
  'Use Network build to inspect node indices before choosing anchors. The constructed graph must have 1-256 nodes and at most 4096 directed arcs; raw row/vertex counts cannot establish those limits. Shared-vertex splitting, snapping and one-way restrictions change them. Crossings without a shared vertex remain disconnected.';
const COST_NOTE =
  'Length costs use metres for EPSG:4326 and native planar units otherwise; property costs use the column unit. Set budgets and departure times in that cost unit. Missing or nonnumeric property costs remove the affected direction; negative costs fail. The constant budget is not derived from geographic extent.';

const ZONE_POPULATION: KeplerOperationFieldDescriptor = {
  ...describeNumericField('populationProperty', 'population', 'Zone population or location weight'),
  measurement: ['count', 'amount']
};
const FLOW_ENDPOINTS: readonly KeplerOperationFieldDescriptor[] = [
  {
    parameter: 'originProperty',
    role: 'origin',
    label: 'Origin zone identity',
    accepts: ['text', 'category', 'integer', 'number'],
    measurement: ['identifier', 'nominal'],
    required: true
  },
  {
    parameter: 'destinationProperty',
    role: 'destination',
    label: 'Destination zone identity',
    accepts: ['text', 'category', 'integer', 'number'],
    measurement: ['identifier', 'nominal'],
    required: true
  },
  {
    ...describeNumericField('weightProperty', 'weight', 'Observed flow weight'),
    measurement: ['count', 'amount']
  }
];

const ROUTING_PARAMETERS = [
  'coordinateSystem',
  'snapTolerance',
  'splitAtSharedVertices',
  'costMode',
  'originNode'
];
function describeRoutingRequirements(): KeplerOperationRequirement[] {
  return [
    ...describeCapacity('lines', 1024, 8192),
    describeCoordinates('lines'),
    ...['forward-cost', 'reverse-cost'].flatMap(role => [
      {
        input: 'lines',
        role,
        id: 'non-negative-values' as const,
        severity: 'block' as const,
        activeWhen: PROPERTY_COSTS,
        remedy: 'Correct negative directional costs before routing'
      },
      {
        input: 'lines',
        role,
        id: 'no-missing-values' as const,
        severity: 'warn' as const,
        activeWhen: PROPERTY_COSTS,
        remedy: 'Missing costs exclude the affected direction; fill them to retain it'
      }
    ])
  ];
}

const CATEGORY_FIELD: KeplerOperationFieldDescriptor = {
  parameter: 'categoryProperty',
  role: 'category',
  label: 'Category or pre-stratified factor',
  accepts: ['category', 'text', 'integer', 'boolean', 'number', 'time'],
  measurement: ['nominal'],
  required: true
};
function describeCategoryRequirements(): KeplerOperationRequirement[] {
  return [
    ...describeCapacity('points', 1024),
    {
      input: 'points',
      role: 'category',
      id: 'minimum-distinct-values',
      severity: 'block',
      limit: 1,
      remedy: 'Choose a category column with at least one nonmissing category'
    },
    {
      input: 'points',
      role: 'category',
      id: 'maximum-distinct-values',
      severity: 'block',
      limit: 32,
      remedy: 'Recode or aggregate to at most 32 categories'
    },
    {
      input: 'points',
      role: 'category',
      id: 'no-missing-values',
      severity: 'warn',
      remedy: 'Missing categories are excluded from categorical statistics; recode them if needed'
    }
  ];
}

function describeRadius(): KeplerOperationParameterDescriptor {
  return {
    name: 'radius',
    apply: 'rebuild',
    level: 'basic',
    defaultPolicy: {from: 'data', input: 'points', rule: 'maximum-nearest-neighbor-distance'}
  };
}
const RADIUS_NOTE =
  'The distance-band default is the largest nearest-neighbor distance when available, in metres for EPSG:4326 or native planar units. It aims to give each point a neighbor; it does not guarantee connected weights. Verify boundary membership after Float32 projection. All-coincident or unprofiled points use the declared radius constant.';
const CATEGORY_NOTE =
  RADIUS_NOTE +
  ' Category labels retain their original scalar type; missing categories are excluded. No permutation inference is performed.';
const CATEGORY_ACTIONS = [
  ['colocation-quotient', 'Compare category co-location'],
  ['local-category-dependence', 'Find local category association'],
  ['spatial-entropy', 'Measure category mixing'],
  ['category-pair-counts', 'Count neighboring category pairs']
] as const;

/** Hand-written meaning, checks, defaults and result guidance for all 15 AX-80 D2 bindings. */
export const NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'kernel-flow-smoothing',
    questions: ['how-it-moves', 'where-concentrated'],
    action: 'Smooth observed zone flows',
    prominence: 3,
    inputs: [
      {name: 'zones', kind: 'points', rowSemantics: ['objects'], fields: [ZONE_POPULATION]},
      {name: 'flows', kind: 'lines', fields: FLOW_ENDPOINTS}
    ],
    parameters: describeParameters(
      ['coordinateSystem', 'neighborhoodSize', 'bandwidthRatio'],
      ['coordinateSystem', 'neighborhoodSize', 'bandwidthRatio'],
      ['neighborhoodSize', 'bandwidthRatio']
    ),
    requirements: [
      ...describeCapacity('zones', 128),
      ...describeCapacity('flows', 16384, 32768),
      {
        input: 'zones',
        id: 'minimum-rows',
        severity: 'block',
        limit: 2,
        remedy: 'Supply at least two zone centroids'
      },
      describeCoordinates('zones'),
      ...describeNumericRequirements('zones', 'population', true),
      ...describeNumericRequirements('flows', 'weight', true),
      ...['origin', 'destination'].map(role => ({
        input: 'flows',
        role,
        id: 'no-missing-values' as const,
        severity: 'block' as const,
        remedy: 'Every flow endpoint must name an existing zone identity'
      }))
    ],
    outputs: [
      describeOutput('kernel-flow-smoothing', 'lines', {column: 'flow', scale: 'sequential'})
    ],
    notes: [
      FLOW_NOTE,
      FLOW_DEFAULT_NOTE,
      'Smoothing is a weighted mean and does not conserve total flow. Dense OD output includes zero flows and the origin/destination neighborhood radii.'
    ]
  },
  {
    bindingId: 'radiation-flow-model',
    questions: ['how-it-moves'],
    action: 'Model flows between populated zones',
    prominence: 3,
    inputs: [
      {
        name: 'zones',
        kind: 'points',
        rowSemantics: ['objects'],
        fields: [
          ZONE_POPULATION,
          {
            ...describeNumericField('outflowProperty', 'outflow', 'Total departing flow'),
            measurement: ['count', 'amount']
          }
        ]
      }
    ],
    parameters: describeParameters(
      ['coordinateSystem', 'normalization'],
      ['coordinateSystem', 'normalization']
    ),
    requirements: [
      ...describeCapacity('zones', 128),
      {
        input: 'zones',
        id: 'minimum-rows',
        severity: 'block',
        limit: 2,
        remedy: 'Supply at least two zone centroids'
      },
      describeCoordinates('zones'),
      ...describeNumericRequirements('zones', 'population', true),
      ...describeNumericRequirements('zones', 'outflow', true)
    ],
    outputs: [
      describeOutput('radiation-flow-model', 'lines', {column: 'flow', scale: 'sequential'})
    ],
    notes: [
      FLOW_NOTE,
      'The radiation model uses strictly nearer intervening populations, a zero diagonal and the declared normalization (none by default). It predicts expected flows, not observed trips or random draws; zero populations can yield null statistics.'
    ]
  },
  {
    bindingId: 'map-matched-edge-volumes',
    questions: ['how-it-moves', 'where-concentrated'],
    action: 'Count matched edge traversals',
    prominence: 3,
    inputs: [
      {name: 'edges', kind: 'network', fields: []},
      {
        name: 'tracks',
        kind: 'trajectories',
        fields: [
          {
            parameter: 'edgesProperty',
            role: 'traversed-edges',
            label: 'Array of traversed edge row indices',
            accepts: ['text'],
            required: true
          }
        ]
      }
    ],
    parameters: describeParameters(['traversedEdgeCapacity'], ['traversedEdgeCapacity']),
    requirements: [
      ...describeCapacity('edges', 4096, 32768),
      ...describeCapacity('tracks', 1024, 32768)
    ],
    outputs: [
      describeOutput('map-matched-edge-volumes', 'network', {column: 'volume', scale: 'sequential'})
    ],
    notes: [
      'Consumes already matched edge-index arrays, not raw GPS fixes or routes inferred from track geometry. Each entry counts, including repeated traversals; null or 0xffffffff separates sub-tracks. Edges are identified by their source row index, not feature id. Every edge is published, including zero-volume edges.',
      'The scalar profile represents list-valued columns as text with null statistics; manually select the traversed edge list. Its array shape, index range and edges-per-track capacity cannot be validated by scalar profile statistics and are checked by the binding before dispatch. Text containing JSON is not an accepted substitute for an array.'
    ]
  },
  {
    bindingId: 'network-potential-path-area',
    questions: ['what-is-near-or-reachable', 'how-it-moves'],
    action: 'Find edges within a travel budget',
    prominence: 3,
    inputs: [STREET_INPUT],
    parameters: describeParameters(
      [...ROUTING_PARAMETERS, 'destinationNode', 'budget'],
      ['costMode', 'originNode', 'destinationNode', 'budget'],
      ['budget']
    ),
    requirements: describeRoutingRequirements(),
    outputs: [
      describeOutput('network-potential-path-area', 'network', {
        column: 'in_prism',
        scale: 'categorical'
      })
    ],
    notes: [
      NETWORK_NOTE,
      COST_NOTE,
      'Membership tests whole directed edges between the two anchors; this is a network prism, not an isochrone polygon. Inspect in_prism separately from slack: unreachable slack is null, and Float32 costs near the budget can change membership.'
    ]
  },
  {
    bindingId: 'network-travel-time-percentiles',
    questions: ['what-is-near-or-reachable', 'how-it-moves'],
    action: 'Compare departure travel times',
    prominence: 3,
    inputs: [
      {
        ...STREET_INPUT,
        fields: [
          ...STREET_INPUT.fields,
          {
            ...describeNumericField(
              'slopeProperty',
              'duration-slope',
              'Nonnegative affine duration slope',
              false
            ),
            autoAssign: false,
            measurement: ['rate']
          }
        ]
      }
    ],
    parameters: describeParameters(
      [
        ...ROUTING_PARAMETERS,
        'firstDeparture',
        'departureStep',
        'departureCount',
        'percentiles',
        'maximumRelaxations'
      ],
      ['costMode', 'originNode', 'firstDeparture', 'departureStep', 'percentiles']
    ),
    requirements: [
      ...describeRoutingRequirements(),
      ...describeNumericRequirements('lines', 'duration-slope', true)
    ],
    outputs: [
      describeOutput('network-travel-time-percentiles', 'points', {
        column: 'travel_time',
        scale: 'sequential'
      })
    ],
    notes: [
      NETWORK_NOTE,
      COST_NOTE,
      'Durations are base + slope * departure with nonnegative slopes (FIFO). Select the slope column explicitly to use duration slopes; otherwise slopes are zero. Nonzero slopes require per-edge lines or splitAtSharedVertices disabled; the profile cannot verify that relation.',
      'Percentiles must be strictly ascending and use nearest rank over the regular departure window. Unreachable departures count as infinity and publish null when selected; inspect reached_departures and each departure Valid/Converged flag before reading the travel_time map. Multiple percentile rows share each node; filter to one percentile before comparing nodes.'
    ]
  },
  ...CATEGORY_ACTIONS.map(
    ([bindingId, action]): KeplerOperationDescriptor => ({
      bindingId,
      action,
      questions: ['how-related', 'which-groups-or-regions'],
      prominence: 2,
      inputs: [{name: 'points', kind: 'points', fields: [CATEGORY_FIELD]}],
      parameters: [
        describeRadius(),
        ...describeParameters(['coordinateSystem'], ['coordinateSystem'])
      ],
      requirements: [...describeCategoryRequirements(), describeCoordinates('points')],
      outputs:
        bindingId === 'colocation-quotient'
          ? [
              describeOutput(bindingId, 'summary'),
              describeOutput('colocation-local', 'points', {
                column: 'quotient',
                scale: 'diverging',
                center: 1
              })
            ]
          : [
              describeOutput(
                bindingId,
                bindingId === 'local-category-dependence' ? 'points' : 'summary',
                bindingId === 'local-category-dependence'
                  ? {column: 'proportion', scale: 'sequential'}
                  : undefined,
                bindingId === 'local-category-dependence' ? ['multiple-testing-uncorrected'] : []
              )
            ],
      notes: [
        CATEGORY_NOTE,
        bindingId === 'colocation-quotient'
          ? 'Quotients compare distance-decay neighbor shares with the global category baseline: 1 is the reference. Local rows repeat each point for every destination_category; filter to one destination category before mapping.'
          : bindingId === 'local-category-dependence'
            ? 'LICD uses binary links. Proportion is the share of valid neighbors of the same category. upperTailProbability is an uncorrected local tail probability, not a permutation p-value; the default map shows proportions, not significance.'
            : bindingId === 'spatial-entropy'
              ? 'The report gives named entropy and mutual-information statistics in the statistic/value columns, including linkCount. A high entropy describes mixing, not statistical significance.'
              : 'The report counts directed neighboring category pairs using binary links. A symmetric pair contributes both directions; origin_rows counts source rows, not links.'
      ]
    })
  ),
  {
    bindingId: 'geographical-detector',
    questions: ['how-related'],
    action: 'Compare variation across strata',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [CATEGORY_FIELD, describeNumericField('valueProperty', 'value', 'Response value')]
      }
    ],
    parameters: [],
    requirements: [
      ...describeCategoryRequirements(),
      ...describeNumericRequirements('points', 'value'),
      {
        input: 'points',
        role: 'value',
        id: 'varying-values',
        severity: 'warn',
        remedy: 'Choose a varying response to obtain a defined variance ratio'
      }
    ],
    outputs: [
      describeOutput('geographical-detector', 'summary', undefined, [
        'multiple-testing-uncorrected'
      ])
    ],
    notes: [
      'One pre-stratified categorical factor and one response: q measures the fraction of response variance explained by strata, not causation. No distance band or interaction detector is used. The statistic/value report includes an analytical pValue; degenerate strata or zero variance can yield null statistics. Treat any significance after selecting strata or trying factors as exploratory.'
    ]
  },
  {
    bindingId: 'redcap-single-linkage',
    questions: ['which-groups-or-regions'],
    action: 'Group connected similar points',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [describeNumericField('valueProperty', 'value', 'Attribute to regionalize')]
      }
    ],
    parameters: [
      describeRadius(),
      ...describeParameters(
        ['coordinateSystem', 'regionCount', 'componentIterations'],
        ['regionCount'],
        ['regionCount']
      )
    ],
    requirements: [
      ...describeCapacity('points', 1024),
      describeCoordinates('points'),
      ...describeNumericRequirements('points', 'value'),
      {
        input: 'points',
        id: 'connected-weights',
        severity: 'warn',
        remedy: 'Inspect disconnected components; they may force more regions than requested'
      }
    ],
    outputs: [
      describeOutput('redcap-single-linkage', 'points', {column: 'region', scale: 'categorical'})
    ],
    notes: [
      RADIUS_NOTE,
      'The contributor standardizes one attribute and cuts the heaviest spanning-tree edges. This is constrained single linkage, not the REDCAP SSD partition. Region count 2 is a constant starting point; choose no more regions than source points. Disconnected components may exceed that target. Inspect regionCount, converged, iterationLimitReached and invalidCount; there is no minimum region size.'
    ]
  },
  {
    bindingId: 'healpix-point-to-cell',
    questions: ['prepare-data', 'how-much-per-area'],
    action: 'Assign spherical cell identities',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: describeParameters(['order'], ['order']),
    requirements: [...describeCapacity('points', 20000), describeCoordinates('points', true)],
    outputs: [
      describeOutput('healpix-point-to-cell', 'points', {column: 'cell_id', scale: 'categorical'})
    ],
    notes: [
      'HEALPix uses the nested spherical scheme, orders 0-14. Order 8 is a declared constant, not derived from dataset extent. Cell ids are exact decimal strings, including valid cell zero. Float32 coordinate rounding can change assignments at cell boundaries; inspect validity.'
    ]
  },
  {
    bindingId: 'healpix-cell-aggregation',
    questions: ['how-much-per-area'],
    action: 'Summarize points in spherical cells',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          describeNumericField('valueProperty', 'value', 'Optional value to summarize', false)
        ]
      }
    ],
    parameters: describeParameters(['order', 'cellCapacity'], ['order']),
    requirements: [
      ...describeCapacity('points', 20000),
      describeCoordinates('points', true),
      ...describeNumericRequirements('points', 'value')
    ],
    outputs: [
      describeOutput('healpix-cell-aggregation', 'cells', {column: 'count', scale: 'sequential'})
    ],
    notes: [
      'Nested HEALPix order 8 and cell capacity 20000 are declared constants. Choose order for the analysis scale; no extent-to-order rule is provided. With no value column, only counts are computed; when selected, values must be finite on every row. Sum, minimum and maximum are null without a value column.',
      'Capacity overflow prevents partial publication and reports requiredCount; increase cellCapacity or lower order. Equal-area cells are spherical, not ellipsoidal. Display polygons sample curved boundaries and unwrap antimeridian longitudes; Float32 input rounding can change cell-edge assignments.'
    ]
  },
  {
    bindingId: 'datum-grid-transform',
    questions: ['prepare-data'],
    action: 'Apply a decoded datum shift grid',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            ...describeNumericField(
              'heightProperty',
              'height',
              'Height (or point altitude)',
              false
            ),
            autoAssign: false
          }
        ]
      },
      {name: 'samples', kind: 'points', fields: []}
    ],
    parameters: describeParameters(
      [
        'width',
        'height',
        'originLongitude',
        'originLatitude',
        'spacingLongitude',
        'spacingLatitude',
        'operation',
        'direction',
        'sourceVerticalReference',
        'targetVerticalReference',
        'heightUnitMeters',
        'maximumIterations',
        'tolerance'
      ],
      ['width', 'height', 'operation', 'direction']
    ).map(parameter => ({
      ...parameter,
      ...(['sourceVerticalReference', 'targetVerticalReference', 'heightUnitMeters'].includes(
        parameter.name
      )
        ? {activeWhen: VERTICAL_OPERATION}
        : {}),
      ...(['maximumIterations', 'tolerance'].includes(parameter.name)
        ? {activeWhen: INVERSE_DIRECTION}
        : {})
    })),
    requirements: [
      ...describeCapacity('points', 20000),
      ...describeCapacity('samples', 65536),
      describeCoordinates('points', true)
    ],
    outputs: [
      describeOutput('datum-grid-transform', 'points', {column: 'status', scale: 'categorical'})
    ],
    notes: [
      'Supply a caller-decoded regular grid in row-major sample order, with width * height equal to sample rows. Sample geometry is ignored. Fixed columns longitudeShift/latitudeShift are degrees, heightShift is metres, and valid=false or 0 masks nodata. These fixed schema columns have no remapping parameters. The profile cannot check grid dimensions, sample order or four-corner validity.',
      'Grid dimensions, origin and positive spacings are declared constants that must be set from the decoded grid metadata; no NTv2/GeoTIFF parser or CRS reprojection is provided. Input longitude must share the grid longitude branch.',
      'Select the height column explicitly to override point altitude; horizontal mode preserves heights and defaults to zero if absent. Vertical/combined modes require a finite height (column or altitude) and explicit source/target vertical references; the placeholder reference names are not data-derived. The height role stays optional because altitude is a valid fallback.',
      'Status bits: 0 success, 1 invalid, 2 no coverage, 4 nodata, 8 inverse exhausted. Failed rows retain source identities with null output geometry. Inspect the status table as well as the map, which cannot display failed rows.'
    ]
  },
  {
    bindingId: 'sinkhorn-divergence',
    questions: ['what-changed', 'how-related'],
    action: 'Compare spatial mass distributions',
    prominence: 3,
    inputs: [
      {
        name: 'cells',
        kind: 'cells',
        fields: [
          {
            ...describeNumericField('sourceProperty', 'source', 'Source mass'),
            measurement: ['count', 'amount']
          },
          {
            ...describeNumericField('targetProperty', 'target', 'Target mass'),
            measurement: ['count', 'amount']
          }
        ]
      }
    ],
    parameters: describeParameters(
      ['width', 'height', 'cellWidth', 'cellHeight', 'epsilon', 'tolerance', 'maximumIterations'],
      ['width', 'height', 'cellWidth', 'cellHeight', 'epsilon'],
      ['cellWidth', 'cellHeight', 'epsilon', 'tolerance']
    ),
    requirements: [
      ...describeCapacity('cells', 16384),
      ...describeNumericRequirements('cells', 'source', true),
      ...describeNumericRequirements('cells', 'target', true)
    ],
    outputs: [describeOutput('sinkhorn-divergence', 'summary')],
    notes: [
      'Input rows must form one explicit row-major planar lattice: width * height equals row count. Geometry is ignored; cellWidth/cellHeight define squared planar transport cost. Dimensions, cell sizes and epsilon are declared constants and must come from lattice metadata; epsilon should be at least about 1e-3 of the squared grid diagonal.',
      'Each mass field needs positive total mass. Both are normalized to one, so the divergence compares spatial distributions and discards total mass differences. Nonnegative values alone do not establish positive totals; the binding checks totals before dispatch.',
      'The report includes divergence, cross_cost, source_self_cost, target_self_cost, marginal_error, converged, iterationLimitReached, iterationCount, invalidCount and approximation. Inspect convergence before comparing results; exhausted iterations publish the current estimate. Tiny negative divergence may reflect Float32 rounding. No transport plan or geographic cost is computed.'
    ]
  }
];
