// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationInputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: these settings change prepared geometry, row/vertex allocation, graph construction
// or compiled shader constants. Rebuild keeps those resources consistent with the chosen value.

// Descriptors of the network bindings (`catalog-bindings.ts`, `catalog-extra-bindings.ts`,
// `catalog-gpu-bindings.ts`, `catalog-snap-binding.ts`, `catalog-platform3-a-bindings.ts`). A
// descriptor says what a binding means to a user; the binding stays the source of how it runs.
//
// Shared choices:
// - Street lines are kind `network`; points that are facilities, demand or events are `points`.
// - Costs are in the unit of the cost column when `costMode` is `property` and in metres (or
//   coordinate units for planar data) when it is `length`. No distance or cost parameter is
//   derived from the data. A snap distance is how far a point sits from its street, and the
//   cost limits (`budgets`, `maximumCost`, `decayScale`) are in the cost unit: an extent-based
//   value is metres, which is wrong as soon as `costMode` is `property` with travel times, and a
//   descriptor cannot make a default depend on another parameter. The notes say so.
// - The network bindings read three optional line columns. The cost columns are read only while
//   `costMode` is `property`; the one-way column is read in every mode.
// - `activeWhen` is used where a parameter really is inert in some state of another parameter
//   (`decayScale`, `penalty`, `maximumOverlap`). No non-column parameter depends on `costMode`:
//   that dependence is on the cost columns, whose requirements apply only in property mode.

// Graph builders enforce these edge caps; input vertices cannot determine graph size.
const ROUTING_EDGE_CAP = 50000;
const NETWORK_EDGE_CAP = 50000;
// The catalog bounds raw input allocation separately from graph allocation.
const STREET_ROW_LIMIT = 20000;
const STREET_VERTEX_LIMIT = 200000;
// Centrality has smaller raw input limits and an all-sources search limited to 1,500 nodes.
const CENTRALITY_VERTEX_LIMIT = 50000;
// Map matching uses the same raw geometry budget as centrality before routing begins.
const MAP_MATCHING_VERTEX_LIMIT = 50000;
const CENTRALITY_NODE_CAP = 1500;

const STREET_REMEDY = 'Filter the streets to the area of interest or simplify the lines';

/**
 * Columns of the street lines. `forward-cost` is required only while `costMode` is `property`; the
 * reverse cost is optional and meaningful only in property mode.
 */
const STREET_COLUMN_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    parameter: 'forwardCostProperty',
    role: 'forward-cost',
    label: 'Travel cost along the line',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: {whenParameter: 'costMode', equals: ['property']}
  },
  {
    parameter: 'reverseCostProperty',
    role: 'reverse-cost',
    label: 'Travel cost against the line',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: false
  },
  {
    parameter: 'onewayProperty',
    role: 'one-way',
    label: 'One-way flag',
    // The order of `accepts` is the preference of the role matcher: a flag column (boolean) before
    // a category, an integer (1, -1) and free text. The run reads the value as text: yes, true, 1, forward, f and ft allow only the drawn
    // direction; -1, reverse, backward and tf allow only the opposite direction.
    accepts: ['boolean', 'category', 'integer', 'text'],
    measurement: ['nominal'],
    required: false
  }
];

/** Parameters every network binding has that are not column names. */
const NETWORK_PARAMETER_NAMES = [
  'coordinateSystem',
  'snapTolerance',
  'splitAtSharedVertices',
  'costMode',
  'crossings'
] as const;

type ParameterOverrides = Readonly<
  Record<string, Pick<KeplerOperationParameterDescriptor, 'activeWhen'>>
>;

function describeParameters(
  names: readonly string[],
  basicNames: readonly string[],
  overrides: ParameterOverrides = {}
): KeplerOperationParameterDescriptor[] {
  return names.map(name => ({
    name,
    apply: 'rebuild',
    defaultPolicy: {from: 'constant'},
    level: basicNames.includes(name) ? 'basic' : 'advanced',
    ...overrides[name]
  }));
}

/** Capacity of a point input: rows only, as vertices equal rows. */
function makePointCapacity(input: string, maximumRows: number, remedy: string) {
  const requirements: KeplerOperationRequirement[] = [
    {id: 'maximum-rows', severity: 'block', input, limit: maximumRows, remedy}
  ];
  return requirements;
}

/** Raw input allocation and exact physical-edge capacity are independent requirements. */
function makeStreetCapacity(
  edgeLimit: number,
  rowLimit = STREET_ROW_LIMIT,
  vertexLimit = STREET_VERTEX_LIMIT
): KeplerOperationRequirement[] {
  return [
    {id: 'maximum-rows', severity: 'block', input: 'lines', limit: rowLimit, remedy: STREET_REMEDY},
    {
      id: 'maximum-vertices',
      severity: 'block',
      input: 'lines',
      limit: vertexLimit,
      remedy: STREET_REMEDY
    },
    {
      id: 'maximum-network-edges',
      severity: 'block',
      input: 'lines',
      limit: edgeLimit,
      remedy:
        'Profile the selected network topology; reduce lines or shared vertices, or disable splitAtSharedVertices if the physical edge count exceeds the limit'
    }
  ];
}

const COORDINATE_REMEDY =
  'Set the coordinate system to EPSG:4326 for longitude and latitude, or planar for projected coordinates';

/** The coordinate system is a parameter the user must set; the binding never infers it. */
function makeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {id: 'declared-coordinates', severity: 'warn', input, remedy: COORDINATE_REMEDY};
}

/** Explain excluded costs and reject invalid negative costs before routing. */
function makeCostRequirements(): KeplerOperationRequirement[] {
  return ['forward-cost', 'reverse-cost'].flatMap(role => [
    {
      id: 'no-missing-values' as const,
      severity: 'warn' as const,
      input: 'lines',
      role,
      activeWhen: {parameter: 'costMode', equals: ['property']},
      remedy:
        'Fill or remove missing costs: the affected direction is excluded and counted in diagnostics'
    },
    {
      id: 'non-negative-values' as const,
      severity: 'warn' as const,
      input: 'lines',
      role,
      activeWhen: {parameter: 'costMode', equals: ['property']},
      remedy: 'Correct negative costs before routing; negative costs throw a named validation error'
    }
  ]);
}

const COST_NOTE =
  'In property cost mode numeric text is accepted; missing, blank, non-numeric and non-finite costs exclude the affected direction and are counted in diagnostics. Negative costs throw a named validation error. Length mode visibly warns when named cost columns are ignored.';
const COST_UNIT_NOTE =
  'Costs are metres (coordinate units for planar data) while the cost mode is length, and the unit of the cost column in property mode; every cost limit, budget and radius below is in that unit and must be set for it.';
const ROUTING_GRAPH_NOTE =
  'This binding builds its graph with at most 50,000 edges and honors splitAtSharedVertices. Capacity uses exact physical-edge counts for the selected topology; until those are computed the check remains unknown.';
const NETWORK_GRAPH_NOTE =
  'The graph holds at most 50,000 edges (catalog-binding-types.ts:204); input vertices do not count graph edges. Profile the selected topology to check the physical-edge capacity.';
const TOPOLOGY_NOTE =
  'This geometry-only binding does not expose cost columns or one-way restrictions; obsolete saved parameters are ignored.';
const NO_CROSSING_NOTE =
  'Lines that cross without sharing a vertex are not connected by the CPU engine; node them first.';

/** The street input: network lines with the three optional columns. */
function makeStreetInput(
  fields: readonly KeplerOperationFieldDescriptor[] = STREET_COLUMN_FIELDS
): KeplerOperationInputDescriptor {
  return {name: 'lines', kind: 'network', fields};
}

export const NETWORK_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'network-build-graph',
    questions: ['prepare-data'],
    action: 'Build a network from lines',
    prominence: 3,
    inputs: [makeStreetInput()],
    parameters: describeParameters(NETWORK_PARAMETER_NAMES, [
      'coordinateSystem',
      'costMode',
      'snapTolerance'
    ]),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-edges',
        kind: 'network',
        presentation: {column: 'forward_cost', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable', 'which-groups-or-regions']
      }
    ],
    notes: [COST_NOTE, NETWORK_GRAPH_NOTE, NO_CROSSING_NOTE]
  },
  {
    bindingId: 'network-shortest-path',
    questions: ['what-is-near-or-reachable', 'how-it-moves'],
    action: 'Find shortest routes',
    prominence: 2,
    inputs: [
      makeStreetInput(),
      {name: 'origin', kind: 'points', fields: []},
      {name: 'destinations', kind: 'points', fields: []}
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance'],
      ['coordinateSystem', 'costMode']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origin', 1, 'Use one origin; run the routes again for another'),
      ...makePointCapacity('destinations', 500, 'Use fewer destinations'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-paths',
        kind: 'lines',
        presentation: {column: 'total_cost', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-it-moves']
      }
    ],
    notes: [COST_NOTE, COST_UNIT_NOTE, NETWORK_GRAPH_NOTE, NO_CROSSING_NOTE]
  },
  {
    bindingId: 'network-service-area',
    questions: ['what-is-near-or-reachable'],
    action: 'Find the service area',
    prominence: 1,
    inputs: [makeStreetInput(), {name: 'origin', kind: 'points', fields: []}],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'budgets', 'includeUnreachable'],
      ['coordinateSystem', 'costMode', 'budgets']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origin', 1, 'Use one origin; use several origins for more'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-service-area',
        kind: 'lines',
        presentation: {column: 'first_reach_cost', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'The default budgets 500 and 1,000 are distances in metres, not travel times, unless the cost mode is property with a time column.',
      NETWORK_GRAPH_NOTE,
      NO_CROSSING_NOTE
    ]
  },
  {
    bindingId: 'network-structure',
    // Degree, component and node kind describe how the network is connected, so the primary
    // question is which parts hang together; the same table is how a user checks a new network.
    questions: ['which-groups-or-regions', 'prepare-data'],
    action: 'Check how streets connect',
    prominence: 2,
    // Structure exposes geometry and topology parameters only.
    inputs: [makeStreetInput([])],
    parameters: describeParameters(
      NETWORK_PARAMETER_NAMES.filter(name => name !== 'costMode'),
      ['coordinateSystem', 'snapTolerance']
    ),
    requirements: [...makeStreetCapacity(NETWORK_EDGE_CAP), makeDeclaredCoordinates('lines')],
    outputs: [
      {
        tableId: 'network-structure',
        kind: 'points',
        presentation: {column: 'degree', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable', 'how-it-moves']
      }
    ],
    notes: [TOPOLOGY_NOTE, NETWORK_GRAPH_NOTE, NO_CROSSING_NOTE]
  },
  {
    bindingId: 'network-centrality',
    // Closeness says how easily a junction reaches the rest of the network, which is potential
    // for movement; it does not say which junctions form a group.
    questions: ['how-it-moves'],
    action: 'Find the best connected junctions',
    prominence: 2,
    inputs: [makeStreetInput()],
    parameters: describeParameters(NETWORK_PARAMETER_NAMES, ['coordinateSystem', 'costMode']),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP, 5000, CENTRALITY_VERTEX_LIMIT),
      {
        id: 'maximum-network-nodes',
        severity: 'block',
        input: 'lines',
        limit: CENTRALITY_NODE_CAP,
        remedy:
          'Profile the selected network topology; filter the streets or increase snapping to keep at most 1,500 network nodes'
      },
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-centrality',
        kind: 'points',
        presentation: {column: 'closeness_centrality', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-concentrated', 'which-groups-or-regions']
      }
    ],
    notes: [
      COST_NOTE,
      'The run stops above 1,500 network nodes because it searches from every node (catalog-extra-bindings.ts:107, catalog-extra-network.ts:106). Exact topology-specific profile counts decide this capacity; without them the check remains unknown; raw input limits remain 5,000 rows and 50,000 vertices.',
      'Closeness is computed over the nodes a node can reach, so a node in a small component scores on that component alone.'
    ]
  },
  {
    bindingId: 'network-cost-matrix',
    questions: ['what-is-near-or-reachable'],
    action: 'Compute travel costs between places',
    prominence: 2,
    inputs: [
      makeStreetInput(),
      {name: 'origins', kind: 'points', fields: []},
      {name: 'destinations', kind: 'points', fields: []}
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance'],
      ['coordinateSystem', 'costMode']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origins', 100, 'Use fewer origins'),
      ...makePointCapacity('destinations', 100, 'Use fewer destinations'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        // A table of pairs has no geometry, so nothing is coloured.
        tableId: 'network-cost-matrix',
        kind: 'pairs',
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-it-moves', 'which-groups-or-regions']
      }
    ],
    notes: [COST_NOTE, COST_UNIT_NOTE, NETWORK_GRAPH_NOTE, NO_CROSSING_NOTE]
  },
  {
    bindingId: 'network-neighborhood',
    questions: ['what-is-near-or-reachable'],
    action: 'Find junctions within reach',
    prominence: 2,
    inputs: [makeStreetInput(), {name: 'origin', kind: 'points', fields: []}],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'maximumCost'],
      ['coordinateSystem', 'costMode', 'maximumCost']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origin', 1, 'Use one origin'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-neighborhood',
        kind: 'points',
        presentation: {column: 'cost', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'The default maximum cost of 1,000 is a distance in metres, not travel times, unless the cost mode is property with a time column.',
      NETWORK_GRAPH_NOTE,
      NO_CROSSING_NOTE
    ]
  },
  {
    bindingId: 'network-line-graph',
    questions: ['prepare-data'],
    action: 'Link streets that meet',
    prominence: 3,
    // Adjacency is undirected and ignores costs and one-way flags.
    inputs: [makeStreetInput([])],
    parameters: describeParameters(
      NETWORK_PARAMETER_NAMES.filter(name => name !== 'costMode'),
      ['coordinateSystem', 'snapTolerance']
    ),
    requirements: [...makeStreetCapacity(NETWORK_EDGE_CAP), makeDeclaredCoordinates('lines')],
    outputs: [
      {
        // A table of edge pairs has no geometry and no value to colour by.
        tableId: 'network-line-graph',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []},
        next: ['which-groups-or-regions']
      }
    ],
    notes: [
      TOPOLOGY_NOTE,
      'More than 100,000 links stop the run (catalog-extra-bindings.ts, network-line-graph).',
      NETWORK_GRAPH_NOTE,
      NO_CROSSING_NOTE
    ]
  },
  {
    bindingId: 'network-snap-points',
    questions: ['prepare-data', 'what-is-near-or-reachable'],
    action: 'Snap points to streets',
    prominence: 3,
    // Snapping picks the nearest street by geometry, so costs and one-way flags cannot change a
    // snapped position.
    inputs: [makeStreetInput([]), {name: 'points', kind: 'points', fields: []}],
    // The snap distance is the purpose of this operation, so it is basic here and only here: in
    // the bindings that route, it is a tolerance with a stated default and a diagnostic for a
    // point that does not snap, so it is advanced.
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES.filter(name => name !== 'costMode'), 'maximumSnapDistance'],
      ['coordinateSystem', 'maximumSnapDistance']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP).map(requirement =>
        requirement.id === 'maximum-network-edges'
          ? {
              ...requirement,
              remedy:
                'Profile the snapping topology; reduce lines or shared vertices if the physical edge count exceeds the limit. Snapping always splits at shared vertices'
            }
          : requirement
      ),
      ...makePointCapacity('points', 5000, 'Snap a sample or split the points'),
      makeDeclaredCoordinates('lines')
    ],
    outputs: [
      {
        tableId: 'snapped-points',
        kind: 'points',
        kindFrom: 'points',
        presentation: {column: 'distance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['what-is-near-or-reachable', 'how-it-moves']
      }
    ],
    notes: [
      TOPOLOGY_NOTE,
      NETWORK_GRAPH_NOTE,
      'Snapping always splits at shared vertices and does not node crossings; the splitAtSharedVertices and crossings settings do not change its topology.'
    ]
  },
  {
    // GPU only: verified by reading, not run.
    bindingId: 'network-crossing-noding',
    questions: ['prepare-data'],
    action: 'Cut lines where they cross',
    prominence: 3,
    // Noding is a street-network operation. It cuts geometry only; the binding has no cost or
    // one-way parameters.
    inputs: [{name: 'lines', kind: 'network', fields: []}],
    parameters: describeParameters(
      ['coordinateSystem', 'snapTolerance'],
      ['coordinateSystem', 'snapTolerance']
    ),
    requirements: [
      {
        id: 'maximum-rows',
        severity: 'block',
        input: 'lines',
        limit: STREET_ROW_LIMIT,
        remedy: STREET_REMEDY
      },
      {
        id: 'maximum-vertices',
        severity: 'block',
        input: 'lines',
        limit: 200000,
        remedy: STREET_REMEDY
      },
      makeDeclaredCoordinates('lines')
    ],
    outputs: [
      {
        tableId: 'noded-edges',
        kind: 'network',
        presentation: {column: 'length', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable']
      },
      {
        tableId: 'noded-nodes',
        kind: 'points',
        presentation: {column: 'degree', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'Every crossing becomes a node, so bridges and tunnels are joined; each crossing adds a piece, so the noded output (at most 50,000 pieces) can exceed the input vertex count.'
    ]
  },
  {
    // GPU only: verified by reading, not run.
    bindingId: 'network-multi-origin-service-area',
    questions: ['what-is-near-or-reachable'],
    action: 'Find service areas for several places',
    prominence: 2,
    inputs: [makeStreetInput(), {name: 'origins', kind: 'points', fields: []}],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'budgets'],
      ['coordinateSystem', 'costMode', 'budgets']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origins', 32, 'Use at most 32 facilities'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-multi-origin',
        kind: 'lines',
        presentation: {column: 'first_reach_cost', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'The default budgets 500 and 1,000 are distances in metres, not travel times, unless the cost mode is property with a time column.',
      NETWORK_GRAPH_NOTE
    ]
  },
  {
    // GPU only: verified by reading, not run.
    bindingId: 'network-isochrone-polygons',
    questions: ['what-is-near-or-reachable'],
    // With the default cost mode the bands are distances along streets, not travel times.
    action: 'Draw areas reachable by street',
    prominence: 1,
    inputs: [makeStreetInput(), {name: 'origins', kind: 'points', fields: []}],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'budgets', 'bufferRadius', 'rasterSize'],
      ['coordinateSystem', 'costMode', 'budgets']
    ),
    requirements: [
      ...makeStreetCapacity(NETWORK_EDGE_CAP),
      ...makePointCapacity('origins', 16, 'Use at most 16 facilities'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'network-isochrones',
        kind: 'polygons',
        // Bands are exclusive annuli ordered by cost, so the upper cost of a band orders them.
        presentation: {column: 'maximum_cost', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'By default the bands are distances along streets in metres, not travel times (the default budgets are 500 and 1,000); they are travel times only when the cost mode is property with a time column.',
      NETWORK_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-accessibility',
    questions: ['what-is-near-or-reachable'],
    action: 'Count what each place can reach',
    prominence: 2,
    inputs: [
      makeStreetInput(),
      {name: 'origins', kind: 'points', fields: []},
      {
        name: 'opportunities',
        kind: 'points',
        fields: [
          {
            parameter: 'weightProperty',
            role: 'weight',
            label: 'Opportunity weight',
            accepts: ['number', 'integer'],
            // A weight is summed over the opportunities reached, so it is a count or an amount.
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      }
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'maximumCost', 'decay', 'decayScale'],
      ['maximumCost', 'decay', 'coordinateSystem', 'costMode'],
      {
        // Only the exponential and gaussian decays read the scale; the others use the maximum cost.
        decayScale: {activeWhen: {parameter: 'decay', equals: ['exponential', 'gaussian']}}
      }
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      ...makePointCapacity('origins', 500, 'Use fewer origins'),
      ...makePointCapacity('opportunities', 5000, 'Use fewer opportunities'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'accessibility',
        kind: 'points',
        kindFrom: 'origins',
        presentation: {column: 'accessibility', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-concentrated', 'where-significant']
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'The default maximum cost of 1,000 and decay scale of 500 are distances in metres, not travel times, unless the cost mode is property with a time column.',
      'Missing or non-numeric opportunity weights are excluded from reachable counts and scores and counted in diagnostics; negative weights throw a named validation error.',
      ROUTING_GRAPH_NOTE,
      NO_CROSSING_NOTE
    ]
  },
  {
    bindingId: 'network-subgraph-filter',
    questions: ['prepare-data'],
    action: 'Keep part of the network',
    prominence: 3,
    inputs: [
      {
        name: 'lines',
        kind: 'network',
        fields: [
          ...STREET_COLUMN_FIELDS,
          {
            // The run compares the column's value, as text, with `propertyValue`. The value is a
            // plain parameter: the contract cannot say that it is needed once a column is named.
            parameter: 'propertyName',
            role: 'filter-property',
            label: 'Column to filter on',
            accepts: ['category', 'text', 'integer', 'boolean', 'number'],
            measurement: ['nominal'],
            required: false
          }
        ]
      }
    ],
    parameters: describeParameters(
      [
        ...NETWORK_PARAMETER_NAMES,
        'minimumCost',
        'maximumCost',
        'propertyValue',
        'largestComponentOnly'
      ],
      ['propertyValue', 'maximumCost', 'largestComponentOnly', 'coordinateSystem']
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'subgraph-edges',
        kind: 'network',
        presentation: {column: 'forward_cost', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable', 'which-groups-or-regions']
      }
    ],
    notes: [
      COST_NOTE,
      'A column name with an empty value keeps only the streets whose column is empty, which is usually none, so every street is removed (catalog-platform3-a-network.ts:307-309). Fill in the value whenever a column is named.',
      'The cost range filters on the forward cost (the reverse cost where only the reverse direction is allowed), in the cost unit.',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-coarsening',
    questions: ['prepare-data'],
    action: 'Merge street chains',
    prominence: 3,
    inputs: [makeStreetInput()],
    parameters: describeParameters(NETWORK_PARAMETER_NAMES, ['coordinateSystem', 'costMode']),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'coarse-edges',
        kind: 'network',
        presentation: {column: 'edge_count', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable', 'how-it-moves']
      }
    ],
    notes: [
      COST_NOTE,
      'Chains merge at every node with two streets, whatever their road class or name.',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-flow-aggregation',
    questions: ['how-it-moves'],
    action: 'Add up flow on streets',
    prominence: 2,
    inputs: [
      makeStreetInput(),
      {
        // Origins and destinations pair by feature order, and the weight is read from the origin.
        name: 'origins',
        kind: 'points',
        fields: [
          {
            parameter: 'weightProperty',
            role: 'weight',
            label: 'Flow carried by each origin',
            accepts: ['number', 'integer'],
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      },
      {name: 'destinations', kind: 'points', fields: []}
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance'],
      ['coordinateSystem', 'costMode']
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      ...makePointCapacity('origins', 2000, 'Use fewer origins'),
      ...makePointCapacity('destinations', 2000, 'Use fewer destinations'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements(),
      {
        // A pair whose origin has no numeric weight is marked `invalid-pair` and is not routed
        // (`catalog-platform3-a-network.ts:529-531`).
        id: 'no-missing-values',
        severity: 'warn',
        input: 'origins',
        role: 'weight',
        remedy: 'Fill the missing weights: those pairs are marked invalid-pair and carry no flow'
      },
      {
        // A negative weight makes the pair `invalid-pair` and it is not routed.
        id: 'non-negative-values',
        severity: 'warn',
        input: 'origins',
        role: 'weight',
        remedy: 'Remove negative weights: those pairs are marked invalid-pair and carry no flow'
      }
    ],
    outputs: [
      {
        tableId: 'flow-edges',
        kind: 'lines',
        presentation: {column: 'flow', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-concentrated']
      },
      {
        // The pairs table has no geometry, so nothing is coloured.
        tableId: 'flow-pairs',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      COST_NOTE,
      COST_UNIT_NOTE,
      'Origins and destinations are paired by row order, the first with the first; surplus rows of the longer layer are ignored (diagnostic unpaired-features, catalog-platform3-a-network.ts:497-507).',
      'A pair with a missing or negative weight is marked invalid-pair and carries no flow; a pair that does not snap or is not connected is listed with its status and carries none either.',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-alternative-routes',
    questions: ['what-is-near-or-reachable', 'how-it-moves'],
    action: 'Find alternative routes',
    prominence: 3,
    inputs: [
      makeStreetInput(),
      {name: 'origin', kind: 'points', fields: []},
      {name: 'destination', kind: 'points', fields: []}
    ],
    parameters: describeParameters(
      [
        ...NETWORK_PARAMETER_NAMES,
        'maximumSnapDistance',
        'routeCount',
        'penalty',
        'maximumOverlap'
      ],
      ['routeCount', 'penalty', 'coordinateSystem', 'costMode'],
      {
        // With one route there is nothing to penalise and nothing to compare for overlap.
        penalty: {activeWhen: {parameter: 'routeCount', equals: [2, 3, 4, 5, 6]}},
        maximumOverlap: {activeWhen: {parameter: 'routeCount', equals: [2, 3, 4, 5, 6]}}
      }
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      ...makePointCapacity('origin', 1, 'Use one origin'),
      ...makePointCapacity('destination', 1, 'Use one destination'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'alternative-routes',
        kind: 'lines',
        // How much longer than the best route each alternative is.
        presentation: {column: 'detour_ratio', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-it-moves']
      }
    ],
    notes: [
      COST_NOTE,
      'Fewer routes than requested can result when the alternatives overlap the best route too much.',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-map-matching',
    questions: ['how-it-moves'],
    action: 'Match a GPS trace to streets',
    prominence: 2,
    inputs: [
      makeStreetInput(),
      // The trace is the ordered vertices of its features, so it is a trajectory.
      {name: 'trace', kind: 'trajectories', fields: []}
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'searchRadius', 'candidatesPerPoint', 'sigma', 'beta'],
      ['searchRadius', 'sigma', 'coordinateSystem']
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP, 5000, MAP_MATCHING_VERTEX_LIMIT),
      ...makePointCapacity('trace', 500, 'Match the trace in pieces of 500 points'),
      makeDeclaredCoordinates('lines'),
      ...makeCostRequirements()
    ],
    outputs: [
      {
        tableId: 'matched-points',
        kind: 'points',
        // How far each GPS fix sits from the street it was matched to.
        presentation: {column: 'distance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-concentrated']
      },
      {
        // One row per matched piece of the route; nothing numeric to colour by.
        tableId: 'matched-route',
        kind: 'lines',
        presentationException: {
          type: 'uniform-style',
          reason:
            'Route pieces encode the matched path and have no comparable numeric measure; show the route uniformly and colour the matched points by their distance from the street.'
        },
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      COST_NOTE,
      'The trace is the vertices of its features in order. A fix with no street within the search radius is unmatched and breaks the route.',
      'The search radius, sigma and beta are in the coordinate unit of the data (metres for longitude and latitude).',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-k-function',
    questions: ['where-concentrated'],
    action: 'Describe clustering along streets',
    prominence: 3,
    // The binding ABI includes cost columns, although K rejects property-cost mode. Retain
    // conditional roles for those columns and document that they have no effect in length mode.
    inputs: [
      makeStreetInput(
        STREET_COLUMN_FIELDS.map(field =>
          field.role === 'one-way'
            ? field
            : {...field, required: {whenParameter: 'costMode', equals: ['property']}}
        )
      ),
      {name: 'points', kind: 'points', rowSemantics: ['events'], fields: []}
    ],
    parameters: describeParameters(
      [...NETWORK_PARAMETER_NAMES, 'maximumSnapDistance', 'radii'],
      ['radii', 'coordinateSystem']
    ),
    requirements: [
      ...makeStreetCapacity(ROUTING_EDGE_CAP),
      ...makePointCapacity('points', 600, 'Use a sample of the points'),
      // Fewer than two snapped points leave K empty, so the result would say nothing.
      {
        id: 'minimum-rows',
        severity: 'block',
        input: 'points',
        limit: 2,
        remedy: 'Provide at least two points, or raise the snap distance so more points snap'
      },
      makeDeclaredCoordinates('lines')
    ],
    outputs: [
      {
        // The table is one row per radius with no geometry, so nothing is coloured. No p-value is
        // published, so the value-range sentence applies, not the global-statistic one.
        tableId: 'network-k',
        kind: 'summary',
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'which-groups-or-regions']
      }
    ],
    notes: [
      'Network K requires length cost mode and rejects property costs: physical network length and radii must share units.',
      'The K function has no edge correction, envelope or significance test: it shows clustering, it does not test it.',
      ROUTING_GRAPH_NOTE
    ]
  },
  {
    bindingId: 'network-edge-bundling',
    questions: ['how-it-moves'],
    action: 'Bundle flow lines',
    // A display treatment for origin-destination lines, not an analysis of a street network.
    prominence: 3,
    // The input is origin-destination lines, not streets: only the two end vertices of each
    // line are read. The contract has no kind or row semantics for "flow lines between places"
    // (`lines` and `objects` also fit a street layer), so a plain street layer still matches.
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: describeParameters(
      ['cycles', 'iterations', 'stiffness', 'compatibilityThreshold', 'coordinateSystem'],
      ['compatibilityThreshold', 'cycles', 'coordinateSystem']
    ),
    requirements: [
      {
        id: 'maximum-rows',
        severity: 'block',
        input: 'lines',
        limit: 400,
        remedy: 'Bundle a sample or the busiest 400 lines'
      },
      {
        id: 'maximum-vertices',
        severity: 'block',
        input: 'lines',
        limit: 4000,
        remedy: 'Bundle a sample or the busiest 400 lines'
      },
      makeDeclaredCoordinates('lines')
    ],
    outputs: [
      {
        tableId: 'bundled-edges',
        kind: 'lines',
        presentation: {column: 'compatible_count', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-concentrated']
      }
    ],
    notes: [
      'Each line is reduced to its first and last vertex, so a street layer is bundled by its end points and means nothing.'
    ]
  }
];
