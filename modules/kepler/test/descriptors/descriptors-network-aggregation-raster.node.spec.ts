// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

import {
  computeGPUAnalysisSpacingStatistics,
  matchGPUAnalysisFieldRole,
  resolveGPUAnalysisDefault,
  tokenizeGPUAnalysisColumnName,
  validateGPUAnalysisResultEncoding
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {GPUAnalysisDataProfile} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperation
} from '../../src/analysis/catalog-bindings';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {MAXIMUM_NETWORK_EDGES} from '../../src/analysis/catalog-binding-types';
import {MAXIMUM_ROUTING_EDGES} from '../../src/analysis/catalog-platform3-a-network';
import {makeGridSpec} from '../../src/analysis/catalog-platform3-b-common';
import type {
  CatalogOperationBinding,
  CatalogParameterValue
} from '../../src/analysis/catalog-binding-types';
import {AGGREGATION_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-aggregation';
import {NETWORK_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-network';
import {RASTER_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-raster';
import type {KeplerOperationDescriptor} from '../../src/descriptors/operation-descriptor-types';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';

type Inputs = Record<string, CatalogFeatureCollection>;
type Values = Record<string, CatalogParameterValue>;

// ---------------------------------------------------------------------------------------------
// Synthetic inputs. Streets are a three by three grid in planar units, 100 apart, so every
// junction shares a vertex; the cost columns differ per street so that a wrong column changes
// the answer.

function makeCollection(features: CatalogFeature[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features};
}

function makeLine(
  id: string,
  coordinates: number[][],
  properties: Record<string, unknown> = {}
): CatalogFeature {
  return {id, geometry: {type: 'LineString', coordinates}, properties};
}

function makePoint(
  coordinates: number[],
  properties: Record<string, unknown> = {}
): CatalogFeature {
  return {geometry: {type: 'Point', coordinates}, properties};
}

function makeSquare(
  west: number,
  south: number,
  size: number,
  properties: Record<string, unknown> = {}
): CatalogFeature {
  return {
    id: `zone-${west}-${south}`,
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [west, south],
          [west + size, south],
          [west + size, south + size],
          [west, south + size],
          [west, south]
        ]
      ]
    },
    properties
  };
}

/** Streets with travel times, a reverse travel time, one-way flags and a road class. */
const STREETS = makeCollection([
  makeLine(
    'row-0',
    [
      [0, 0],
      [100, 0],
      [200, 0]
    ],
    {travel_time: 10, travel_time_back: 40, oneway: 'yes', road_class: 'primary'}
  ),
  makeLine(
    'row-1',
    [
      [0, 100],
      [100, 100],
      [200, 100]
    ],
    {travel_time: 20, travel_time_back: 20, oneway: 'no', road_class: 'primary'}
  ),
  makeLine(
    'row-2',
    [
      [0, 200],
      [100, 200],
      [200, 200]
    ],
    {travel_time: 30, travel_time_back: 90, oneway: '-1', road_class: 'primary'}
  ),
  makeLine(
    'column-0',
    [
      [0, 0],
      [0, 100],
      [0, 200]
    ],
    {travel_time: 15, travel_time_back: 15, oneway: 'no', road_class: 'residential'}
  ),
  makeLine(
    'column-1',
    [
      [100, 0],
      [100, 100],
      [100, 200]
    ],
    {travel_time: 25, travel_time_back: 25, oneway: 'no', road_class: 'residential'}
  ),
  makeLine(
    'column-2',
    [
      [200, 0],
      [200, 100],
      [200, 200]
    ],
    {travel_time: 35, travel_time_back: 35, oneway: 'no', road_class: 'residential'}
  ),
  // A street that does not touch the grid, so the network has two components.
  makeLine(
    'island',
    [
      [500, 500],
      [600, 500]
    ],
    {travel_time: 5, travel_time_back: 5, oneway: 'no', road_class: 'residential'}
  )
]);

const ORIGIN = makeCollection([makePoint([200, 0])]);
const DESTINATIONS = makeCollection([
  makePoint([0, 200]),
  makePoint([100, 100]),
  makePoint([200, 200])
]);
const FACILITIES_ON_STREETS = makeCollection([makePoint([200, 0]), makePoint([0, 200])]);
const OPPORTUNITIES = makeCollection([
  makePoint([50, 5], {jobs: 5}),
  makePoint([150, 95], {jobs: 10}),
  makePoint([100, 195], {jobs: 2}),
  makePoint([10, 100], {jobs: 8}),
  makePoint([190, 190], {jobs: 1})
]);
const FLOW_ORIGINS = makeCollection([
  makePoint([0, 0], {trips: 100}),
  makePoint([200, 0], {trips: 40})
]);
const FLOW_DESTINATIONS = makeCollection([makePoint([200, 200]), makePoint([0, 200])]);
const KFUNCTION_POINTS = makeCollection([
  makePoint([5, 0]),
  makePoint([60, 3]),
  makePoint([100, 90]),
  makePoint([190, 100]),
  makePoint([100, 190]),
  makePoint([2, 195])
]);
// A trace driven west along row 0, against its one-way direction.
const TRACE = makeCollection([
  makeLine('trace', [
    [190, 5],
    [150, -3],
    [90, 4],
    [30, -2]
  ])
]);
const ORIGIN_DESTINATION_LINES = makeCollection([
  makeLine('od-0', [
    [0, 0],
    [200, 200]
  ]),
  makeLine('od-1', [
    [0, 20],
    [200, 180]
  ]),
  makeLine('od-2', [
    [0, 40],
    [200, 160]
  ]),
  makeLine('od-3', [
    [0, 200],
    [200, 0]
  ])
]);
const PLANAR_ZONES = makeCollection([
  makeSquare(0, 0, 150, {name: 'south-west'}),
  makeSquare(150, 150, 50, {name: 'north-east'})
]);

/** Twenty events with a measured value, about a kilometre across near Chicago, EPSG:4326. */
function makeEvents(longitudeShift: number, latitudeShift: number, valueOffset: number) {
  const features: CatalogFeature[] = [];
  for (let index = 0; index < 20; index++) {
    features.push(
      makePoint(
        [
          -87.65 + longitudeShift + (index % 5) * 0.003 + (index % 3) * 0.0005,
          41.88 + latitudeShift + Math.floor(index / 5) * 0.003
        ],
        {value: index + 1 + valueOffset, rate: (index % 4) * 0.25}
      )
    );
  }
  return makeCollection(features);
}

const EVENTS = makeEvents(0, 0, 0);
const EVENTS_AFTER = makeEvents(0.0015, 0.001, 5);

/** Four zones that tile the extent of the events. */
const ZONES = makeCollection([
  makeSquare(-87.651, 41.879, 0.0075),
  makeSquare(-87.6435, 41.879, 0.0075),
  makeSquare(-87.651, 41.8865, 0.0075),
  makeSquare(-87.6435, 41.8865, 0.0075)
]);
const GEOGRAPHIC_FACILITIES = makeCollection([
  makePoint([-87.649, 41.881]),
  makePoint([-87.64, 41.881]),
  makePoint([-87.645, 41.89])
]);

type BindingCase = {
  id: string;
  inputs: Inputs;
  /** Parameters other than the column names. */
  parameters: Values;
  /** Parameter that names a column, with a column that exists in the inputs. */
  columns: Record<string, string>;
};

const NETWORK_COLUMNS = {
  forwardCostProperty: 'travel_time',
  reverseCostProperty: 'travel_time_back',
  onewayProperty: 'oneway'
};
const NETWORK_VALUES: Values = {coordinateSystem: 'planar', costMode: 'property'};

const CASES: BindingCase[] = [
  {
    id: 'network-build-graph',
    inputs: {lines: STREETS},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-shortest-path',
    inputs: {lines: STREETS, origin: ORIGIN, destinations: DESTINATIONS},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-service-area',
    inputs: {lines: STREETS, origin: ORIGIN},
    parameters: {...NETWORK_VALUES, budgets: [20, 60]},
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-structure',
    inputs: {lines: STREETS},
    parameters: {coordinateSystem: 'planar'},
    columns: {}
  },
  {
    id: 'network-centrality',
    inputs: {lines: STREETS},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-cost-matrix',
    inputs: {lines: STREETS, origins: FACILITIES_ON_STREETS, destinations: DESTINATIONS},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-neighborhood',
    inputs: {lines: STREETS, origin: ORIGIN},
    parameters: {...NETWORK_VALUES, maximumCost: 60},
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-line-graph',
    inputs: {lines: STREETS},
    parameters: {coordinateSystem: 'planar'},
    columns: {}
  },
  {
    id: 'network-snap-points',
    inputs: {lines: STREETS, points: OPPORTUNITIES},
    parameters: {coordinateSystem: 'planar'},
    columns: {}
  },
  {
    id: 'network-accessibility',
    inputs: {lines: STREETS, origins: FACILITIES_ON_STREETS, opportunities: OPPORTUNITIES},
    parameters: {...NETWORK_VALUES, maximumCost: 40},
    columns: {...NETWORK_COLUMNS, weightProperty: 'jobs'}
  },
  {
    id: 'network-subgraph-filter',
    inputs: {lines: STREETS},
    parameters: {...NETWORK_VALUES, propertyValue: 'primary'},
    columns: {...NETWORK_COLUMNS, propertyName: 'road_class'}
  },
  {
    id: 'network-coarsening',
    inputs: {lines: STREETS},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-flow-aggregation',
    inputs: {lines: STREETS, origins: FLOW_ORIGINS, destinations: FLOW_DESTINATIONS},
    parameters: NETWORK_VALUES,
    columns: {...NETWORK_COLUMNS, weightProperty: 'trips'}
  },
  {
    id: 'network-alternative-routes',
    inputs: {lines: STREETS, origin: ORIGIN, destination: makeCollection([makePoint([0, 200])])},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-map-matching',
    inputs: {lines: STREETS, trace: TRACE},
    parameters: NETWORK_VALUES,
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-k-function',
    inputs: {lines: STREETS, points: KFUNCTION_POINTS},
    parameters: {...NETWORK_VALUES, costMode: 'length', radii: [8, 15, 30, 60]},
    columns: NETWORK_COLUMNS
  },
  {
    id: 'network-edge-bundling',
    inputs: {lines: ORIGIN_DESTINATION_LINES},
    parameters: {coordinateSystem: 'planar'},
    columns: {}
  },
  {
    id: 'line-length-per-polygon',
    inputs: {polygons: PLANAR_ZONES, lines: STREETS},
    parameters: {coordinateSystem: 'planar'},
    columns: {}
  },
  {
    id: 'cell-aggregation',
    inputs: {points: EVENTS},
    parameters: {cellSize: 0.005},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'cell-rollup',
    inputs: {points: EVENTS},
    parameters: {cellSize: 0.005},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'cell-compare',
    inputs: {before: EVENTS, after: EVENTS_AFTER},
    parameters: {cellSize: 0.005, statistic: 'sum'},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'zonal-point-statistics',
    inputs: {polygons: ZONES, points: EVENTS},
    parameters: {},
    columns: {valueProperty: 'value'}
  },
  {id: 'dggs-point-index', inputs: {points: EVENTS}, parameters: {resolution: 12}, columns: {}},
  {
    id: 'dggs-cell-aggregation',
    inputs: {points: EVENTS},
    parameters: {resolution: 16},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'dggs-cell-rollup',
    inputs: {points: EVENTS},
    parameters: {resolution: 16},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'dggs-cell-compare',
    inputs: {before: EVENTS, after: EVENTS_AFTER},
    parameters: {resolution: 16, measure: 'sum'},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'recipe-points-in-polygons-choropleth',
    inputs: {points: EVENTS, polygons: ZONES},
    parameters: {statistic: 'mean'},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'recipe-zonal-distribution',
    inputs: {points: EVENTS, polygons: ZONES},
    parameters: {},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'raster-join-points-in-polygons',
    inputs: {points: EVENTS, polygons: ZONES},
    parameters: {},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'recipe-straight-line-catchments',
    inputs: {facilities: GEOGRAPHIC_FACILITIES, observations: EVENTS},
    parameters: {},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'raster-density-smoothing',
    inputs: {points: EVENTS},
    parameters: {},
    columns: {valueProperty: 'value'}
  },
  {
    id: 'raster-cell-clusters',
    inputs: {points: EVENTS},
    parameters: {minimumPoints: 1, cellSize: 400},
    columns: {valueProperty: 'value'}
  }
];

const NETWORK_IDS = [
  'network-build-graph',
  'network-shortest-path',
  'network-service-area',
  'network-structure',
  'network-centrality',
  'network-cost-matrix',
  'network-neighborhood',
  'network-line-graph',
  'network-snap-points',
  'network-crossing-noding',
  'network-multi-origin-service-area',
  'network-isochrone-polygons',
  'network-accessibility',
  'network-subgraph-filter',
  'network-coarsening',
  'network-flow-aggregation',
  'network-alternative-routes',
  'network-map-matching',
  'network-k-function',
  'network-edge-bundling'
];
const AGGREGATION_IDS = [
  'line-length-per-polygon',
  'cell-aggregation',
  'cell-rollup',
  'cell-compare',
  'zonal-point-statistics',
  'dggs-point-index',
  'dggs-cell-aggregation',
  'dggs-cell-rollup',
  'dggs-cell-compare',
  'recipe-points-in-polygons-choropleth',
  'recipe-zonal-distribution'
];
const RASTER_IDS = [
  'raster-join-points-in-polygons',
  'recipe-straight-line-catchments',
  'raster-density-smoothing',
  'raster-cell-clusters'
];

/** Bindings with no CPU engine: verified by reading, with the reason the run is skipped. */
const SKIPPED_WITHOUT_CPU_ENGINE: Record<string, string> = {
  'network-crossing-noding':
    'GPU only (GPUNetworkNoding); columns read from catalog-gpu-network.ts',
  'network-multi-origin-service-area':
    'GPU only (GPUNetworkReachability); columns read from catalog-gpu-network.ts',
  'network-isochrone-polygons':
    'GPU only (GPUNetworkIsochrones); columns read from catalog-gpu-network.ts'
};

const ALL_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  ...NETWORK_OPERATION_DESCRIPTORS,
  ...AGGREGATION_OPERATION_DESCRIPTORS,
  ...RASTER_OPERATION_DESCRIPTORS
];
const DESCRIPTOR_IDS = [...NETWORK_IDS, ...AGGREGATION_IDS, ...RASTER_IDS];

function getDescriptor(id: string): KeplerOperationDescriptor {
  const descriptor = ALL_DESCRIPTORS.find(candidate => candidate.bindingId === id);
  if (!descriptor) throw new Error(`No descriptor for ${id}`);
  return descriptor;
}

function getBinding(id: string): CatalogOperationBinding {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`No binding ${id}`);
  return binding;
}

function getCase(id: string): BindingCase {
  const found = CASES.find(candidate => candidate.id === id);
  if (!found) throw new Error(`No case for ${id}`);
  return found;
}

function runCase(testCase: BindingCase, columns: Record<string, string>, extra: Values = {}) {
  const binding = getBinding(testCase.id);
  const parameters = {
    ...getCatalogDefaultParameters(binding),
    ...testCase.parameters,
    ...columns,
    ...extra
  };
  return runCatalogOperation(binding, parameters, testCase.inputs);
}

/** The whole observable outcome of a run, or the error message when it throws. */
function observe(run: () => ReturnType<typeof runCatalogOperation>): string {
  try {
    const result = run();
    return JSON.stringify({
      tables: result.tables,
      diagnostics: result.diagnostics,
      summary: result.summary
    });
  } catch (error) {
    return `error: ${(error as Error).message}`;
  }
}

function getFieldParameters(descriptor: KeplerOperationDescriptor): string[] {
  // One parameter can be named by the same role on several inputs (the compare bindings).
  return [
    ...new Set(descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter)))
  ].sort();
}

const RUNNABLE_IDS = DESCRIPTOR_IDS.filter(id => !(id in SKIPPED_WITHOUT_CPU_ENGINE));

describe('network, aggregation and raster descriptors: coverage and structure', () => {
  it('describes the 35 catalog bindings exactly once, in the order of the task', () => {
    expect(NETWORK_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      NETWORK_IDS
    );
    expect(AGGREGATION_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      AGGREGATION_IDS
    );
    expect(RASTER_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      RASTER_IDS
    );
    expect(new Set(DESCRIPTOR_IDS).size).toBe(35);
  });

  it.each(DESCRIPTOR_IDS)('%s passes validateKeplerOperationDescriptor with no issues', id => {
    expect(validateKeplerOperationDescriptor(getDescriptor(id), getBinding(id))).toEqual([]);
  });

  it('marks at most four parameters basic and at least one', () => {
    for (const descriptor of ALL_DESCRIPTORS) {
      const basicCount = descriptor.parameters.filter(
        parameter => parameter.level === 'basic'
      ).length;
      // Bindings with only column parameters have nothing to mark.
      if (descriptor.parameters.length === 0) continue;
      expect(basicCount, descriptor.bindingId).toBeGreaterThanOrEqual(1);
      expect(basicCount, descriptor.bindingId).toBeLessThanOrEqual(4);
    }
  });

  it('keeps every action phrase imperative and short', () => {
    for (const descriptor of ALL_DESCRIPTORS) {
      expect(descriptor.action.length, descriptor.bindingId).toBeLessThanOrEqual(40);
      // An action starts with a verb, not an article or a noun phrase.
      expect(descriptor.action, descriptor.bindingId).toMatch(/^[A-Z][a-z]+ /);
    }
  });

  it('gives the leading suggestions the prominence the task names', () => {
    const prominence = Object.fromEntries(
      ALL_DESCRIPTORS.map(descriptor => [descriptor.bindingId, descriptor.prominence])
    );
    expect(prominence['network-service-area']).toBe(1);
    expect(prominence['network-isochrone-polygons']).toBe(1);
    expect(prominence['cell-aggregation']).toBe(1);
    expect(prominence['recipe-points-in-polygons-choropleth']).toBe(1);
    for (const id of [
      'network-build-graph',
      'network-crossing-noding',
      'network-line-graph',
      'cell-rollup',
      'dggs-cell-rollup',
      'dggs-point-index'
    ]) {
      expect(prominence[id], id).toBe(3);
    }
  });

  it('answers the questions the task assigns, primary first', () => {
    const primary = Object.fromEntries(
      ALL_DESCRIPTORS.map(descriptor => [descriptor.bindingId, descriptor.questions[0]])
    );
    for (const id of [
      'network-shortest-path',
      'network-service-area',
      'network-cost-matrix',
      'network-multi-origin-service-area',
      'network-isochrone-polygons',
      'network-accessibility'
    ]) {
      expect(primary[id], id).toBe('what-is-near-or-reachable');
    }
    expect(primary['network-map-matching']).toBe('how-it-moves');
    expect(primary['network-flow-aggregation']).toBe('how-it-moves');
    expect(primary['network-structure']).toBe('which-groups-or-regions');
    expect(primary['network-centrality']).toBe('how-it-moves');
    for (const id of [
      'cell-aggregation',
      'zonal-point-statistics',
      'recipe-points-in-polygons-choropleth'
    ]) {
      expect(primary[id], id).toBe('how-much-per-area');
    }
    expect(primary['cell-compare']).toBe('what-changed');
    expect(primary['dggs-cell-compare']).toBe('what-changed');
    expect(primary['raster-density-smoothing']).toBe('where-concentrated');
  });

  it('describes street lines as a network and point features as points', () => {
    for (const id of NETWORK_IDS) {
      if (id === 'network-edge-bundling') continue;
      const lines = getDescriptor(id).inputs.find(input => input.name === 'lines');
      expect(lines?.kind, id).toBe('network');
    }
    // Origin-destination lines are not a network.
    expect(getDescriptor('network-edge-bundling').inputs[0].kind).toBe('lines');
    expect(
      getDescriptor('network-accessibility').inputs.find(input => input.name === 'opportunities')
        ?.kind
    ).toBe('points');
    expect(
      getDescriptor('zonal-point-statistics').inputs.find(input => input.name === 'polygons')?.kind
    ).toBe('polygons');
  });
});

describe('network, aggregation and raster descriptors: requirements match the bindings', () => {
  /** Edge caps of the graph builders, from the code (not from the input specs). */
  const ROUTING_CAP_IDS = [
    'network-accessibility',
    'network-subgraph-filter',
    'network-coarsening',
    'network-flow-aggregation',
    'network-alternative-routes',
    'network-map-matching',
    'network-k-function'
  ];
  const NETWORK_CAP_IDS = [
    'network-build-graph',
    'network-shortest-path',
    'network-service-area',
    'network-structure',
    'network-centrality',
    'network-cost-matrix',
    'network-neighborhood',
    'network-line-graph',
    'network-snap-points',
    'network-multi-origin-service-area',
    'network-isochrone-polygons'
  ];

  function getEdgeCap(id: string): number | undefined {
    if (ROUTING_CAP_IDS.includes(id)) return MAXIMUM_ROUTING_EDGES;
    if (NETWORK_CAP_IDS.includes(id)) return MAXIMUM_NETWORK_EDGES;
    return undefined;
  }

  it('reads the edge caps from the code', () => {
    expect(MAXIMUM_ROUTING_EDGES).toBe(50000);
    expect(MAXIMUM_NETWORK_EDGES).toBe(50000);
  });

  it.each(
    DESCRIPTOR_IDS
  )('%s states raw input capacities separately from topology-specific graph edge caps', id => {
    const descriptor = getDescriptor(id);
    const cap = getEdgeCap(id);
    for (const spec of getBinding(id).inputs) {
      const rows = descriptor.requirements.find(
        requirement => requirement.id === 'maximum-rows' && requirement.input === spec.name
      );
      expect(rows?.limit, `${id} ${spec.name} rows`).toBe(spec.maximumFeatures);
      expect(rows?.severity).toBe('block');
      const vertices = descriptor.requirements.find(
        requirement =>
          requirement.id === 'maximum-vertices' &&
          requirement.input === spec.name &&
          requirement.severity === 'block'
      );
      if (spec.name === 'lines' && cap !== undefined) {
        expect(vertices?.limit, `${id} lines vertices`).toBe(spec.maximumVertices);
        const warning = descriptor.requirements.find(
          requirement =>
            requirement.id === 'maximum-network-edges' &&
            requirement.severity === 'block' &&
            requirement.input === 'lines'
        );
        // Physical edges consume this cap, independently of directed arcs.
        expect(warning?.limit, id).toBe(cap);
        expect(warning?.remedy, id).toContain('physical edge count');
      } else if (spec.maximumVertices === spec.maximumFeatures) {
        expect(vertices, `${id} ${spec.name} vertices`).toBeUndefined();
      } else {
        expect(vertices?.limit, `${id} ${spec.name} vertices`).toBe(spec.maximumVertices);
      }
    }
  });

  /** n by n street grid: 2n lines of n vertices, so 2 n^2 vertices and 2 n (n - 1) edges. */
  function makeStreetGrid(side: number): CatalogFeatureCollection {
    const features: CatalogFeature[] = [];
    for (let line = 0; line < side; line++) {
      const across: number[][] = [];
      const down: number[][] = [];
      for (let step = 0; step < side; step++) {
        across.push([step * 10, line * 10]);
        down.push([line * 10, step * 10]);
      }
      features.push(makeLine(`row-${line}`, across), makeLine(`column-${line}`, down));
    }
    return makeCollection(features);
  }

  it.each(
    ROUTING_CAP_IDS
  )('%s: states exact edge capacity without substituting vertices for edges', id => {
    const testCase = getCase(id);
    const limit = getDescriptor(id).requirements.find(
      requirement =>
        requirement.id === 'maximum-network-edges' &&
        requirement.input === 'lines' &&
        requirement.severity === 'block'
    )!.limit!;
    const run = (side: number) =>
      runCase(
        {...testCase, inputs: {...testCase.inputs, lines: makeStreetGrid(side)}},
        {},
        {costMode: 'length'}
      );
    // The evidence grid fits below the corrected 50,000-edge warning.
    expect(limit).toBe(50000);
    expect(2 * 105 * 105).toBe(22050);
    expect(() => run(105)).not.toThrow();
    // 198 lines, 19,602 vertices: inside the requirement, and the run succeeds.
    expect(2 * 99 * 99).toBeLessThanOrEqual(limit);
    expect(() => run(99)).not.toThrow();
  });

  it('network-centrality: declares a 1,500-node capacity and the run throws above it', () => {
    const testCase = getCase('network-centrality');
    const warning = getDescriptor('network-centrality').requirements.find(
      requirement =>
        requirement.id === 'maximum-network-nodes' &&
        requirement.severity === 'block' &&
        requirement.limit === 1500
    );
    expect(warning?.limit).toBe(1500);
    const run = (side: number) =>
      runCase({...testCase, inputs: {lines: makeStreetGrid(side)}}, {}, {costMode: 'length'});
    // 105 by 105 junctions are 11,025 nodes.
    expect(() => run(105)).toThrow('network-centrality: lines exceed the 1,500 node limit');
    // 38 by 38 junctions are 1,444 nodes: the run is fine although its 2,888 vertices pass the
    // node cap, demonstrating why the requirement must use nodes.
    expect(() => run(38)).not.toThrow();
    expect(2 * 38 * 38).toBeGreaterThan(warning!.limit!);
  });

  it('gives every requirement a remedy, and never uses connected-weights for a network', () => {
    for (const descriptor of ALL_DESCRIPTORS) {
      for (const requirement of descriptor.requirements) {
        expect(
          requirement.remedy?.length,
          `${descriptor.bindingId} ${requirement.id}`
        ).toBeGreaterThan(0);
        expect(requirement.id).not.toBe('connected-weights');
      }
    }
  });

  it('does reject a capacity overrun, as maximum-rows says', () => {
    // network-k-function takes at most 600 points: 601 is over the limit the descriptor states.
    const descriptor = getDescriptor('network-k-function');
    const limit = descriptor.requirements.find(
      requirement => requirement.id === 'maximum-rows' && requirement.input === 'points'
    )!.limit!;
    const points = makeCollection(
      Array.from({length: limit + 1}, (_, index) => makePoint([index % 200, 0]))
    );
    const binding = getBinding('network-k-function');
    expect(() =>
      runCatalogOperation(
        binding,
        {...getCatalogDefaultParameters(binding), coordinateSystem: 'planar'},
        {lines: STREETS, points}
      )
    ).toThrow();
    const accepted = makeCollection(
      Array.from({length: limit}, (_, index) => makePoint([index % 200, 0]))
    );
    expect(() =>
      runCatalogOperation(
        binding,
        {...getCatalogDefaultParameters(binding), coordinateSystem: 'planar'},
        {lines: STREETS, points: accepted}
      )
    ).not.toThrow();
  });
});

describe('network, aggregation and raster descriptors: the engines agree with the descriptors', () => {
  it('lists the skipped bindings and why', () => {
    for (const [id, reason] of Object.entries(SKIPPED_WITHOUT_CPU_ENGINE)) {
      const binding = getBinding(id);
      expect(binding.engine, id).toBe('gpu');
      expect(binding.run, id).toBeUndefined();
      expect(reason.length).toBeGreaterThan(0);
    }
    // Every other described binding has a CPU engine to run.
    for (const id of RUNNABLE_IDS) expect(getBinding(id).run, id).toBeDefined();
  });

  it('has one case per runnable binding, naming exactly the described column parameters', () => {
    expect(CASES.map(testCase => testCase.id)).toEqual(RUNNABLE_IDS);
    for (const id of RUNNABLE_IDS) {
      expect(Object.keys(getCase(id).columns).sort(), id).toEqual(
        getFieldParameters(getDescriptor(id))
      );
    }
  });

  it.each(RUNNABLE_IDS)('%s runs with default parameters plus its column parameters', id => {
    const testCase = getCase(id);
    const result = runCase(testCase, testCase.columns);
    expect(result.tables.map(table => table.id).sort()).toEqual(
      getBinding(id)
        .outputs.map(output => output.tableId)
        .sort()
    );
    expect(result.diagnostics.filter(diagnostic => diagnostic.severity === 'error')).toEqual([]);
    // The run must produce something to describe.
    expect(
      result.tables.some(table => table.rowCount > 0),
      id
    ).toBe(true);
  });

  it.each(RUNNABLE_IDS)('%s: each presentation column exists in the table it colours', id => {
    const testCase = getCase(id);
    const result = runCase(testCase, testCase.columns);
    for (const output of getDescriptor(id).outputs) {
      const table = result.tables.find(candidate => candidate.id === output.tableId);
      expect(table, `${id} ${output.tableId}`).toBeDefined();
      if (output.presentation) {
        expect(Object.keys(table!.columns), `${id} ${output.tableId}`).toContain(
          output.presentation.column
        );
        // A colour scale needs numbers; a category scale needs something to tell rows apart.
        const values = table!.columns[output.presentation.column];
        if (output.presentation.scale !== 'categorical') {
          expect(
            values.every(value => value === null || typeof value === 'number'),
            `${id} ${output.presentation.column} is numeric`
          ).toBe(true);
        }
        // Colouring a table of one row by a column cannot vary; every larger table must.
        if (table!.rowCount > 1) {
          expect(
            new Set(values).size,
            `${id} ${output.presentation.column} varies`
          ).toBeGreaterThan(1);
        }
      }
    }
  });

  it.each(
    RUNNABLE_IDS
  )('%s: a column parameter naming a missing column changes the outcome', id => {
    const testCase = getCase(id);
    const baseline = observe(() => runCase(testCase, testCase.columns));
    expect(baseline.startsWith('error')).toBe(false);
    for (const parameter of Object.keys(testCase.columns)) {
      const changed = observe(() =>
        runCase(testCase, {...testCase.columns, [parameter]: 'no_such_column'})
      );
      if (
        id === 'network-k-function' &&
        ['forwardCostProperty', 'reverseCostProperty'].includes(parameter)
      ) {
        // K accepts length costs only. These ABI column parameters are conditional roles but
        // deliberately have no effect in the supported mode.
        expect(changed, `${id}.${parameter} inactive in length mode`).toBe(baseline);
      } else expect(changed, `${id}.${parameter}`).not.toBe(baseline);
    }
  });

  it('exposes no obsolete cost or one-way columns on topology bindings', () => {
    for (const id of ['network-structure', 'network-line-graph', 'network-snap-points']) {
      expect(getFieldParameters(getDescriptor(id))).toEqual([]);
      expect(
        getBinding(id)
          .parameters.map(parameter => parameter.name)
          .sort()
      ).toEqual(
        (id === 'network-snap-points'
          ? [
              'coordinateSystem',
              'crossings',
              'maximumSnapDistance',
              'snapTolerance',
              'splitAtSharedVertices'
            ]
          : ['coordinateSystem', 'crossings', 'snapTolerance', 'splitAtSharedVertices']
        ).sort()
      );
    }
  });

  it.each(RUNNABLE_IDS)('%s: required matches what the run does without the column', id => {
    const testCase = getCase(id);
    const descriptor = getDescriptor(id);
    const baselineParameters = {
      ...getCatalogDefaultParameters(getBinding(id)),
      ...testCase.parameters
    };
    for (const input of descriptor.inputs) {
      for (const field of input.fields) {
        const baseline = observe(() => runCase(testCase, testCase.columns));
        const withoutColumn = observe(() =>
          runCase(testCase, {...testCase.columns, [field.parameter]: ''})
        );
        const required =
          typeof field.required === 'object'
            ? evaluateGPUAnalysisParameterCondition(field.required, baselineParameters)
            : field.required;
        if (required) {
          // Required: leaving the column out throws or silently changes the answer.
          expect(withoutColumn, `${id}.${field.role} required`).not.toBe(baseline);
        } else if (typeof field.required === 'object') {
          // The condition does not hold in this case, so the column must not be needed.
          expect(withoutColumn.startsWith('error'), `${id}.${field.role} not needed`).toBe(false);
        } else {
          expect(withoutColumn.startsWith('error'), `${id}.${field.role} optional`).toBe(false);
        }
      }
    }
  });

  it('requires a value column exactly when the statistic or measure needs one', () => {
    // Choropleth: count needs no column, every other statistic throws without one.
    const choropleth = getCase('recipe-points-in-polygons-choropleth');
    for (const statistic of ['sum', 'mean', 'minimum', 'maximum']) {
      expect(() => runCase(choropleth, {valueProperty: ''}, {statistic}), statistic).toThrow();
    }
    expect(() => runCase(choropleth, {valueProperty: ''}, {statistic: 'count'})).not.toThrow();
    const condition = getDescriptor('recipe-points-in-polygons-choropleth').inputs[0].fields[0]
      .required;
    expect(condition).toEqual({
      whenParameter: 'statistic',
      equals: ['sum', 'mean', 'minimum', 'maximum']
    });
    // DGGS compare: only the sum measure reads the column.
    const compare = getCase('dggs-cell-compare');
    expect(() => runCase(compare, {valueProperty: ''}, {measure: 'sum'})).toThrow();
    expect(() => runCase(compare, {valueProperty: ''}, {measure: 'count'})).not.toThrow();
    // Lattice compare: sum and mean silently read zero without a column; count does not need one.
    const lattice = getCase('cell-compare');
    const withColumn = observe(() =>
      runCase(lattice, {valueProperty: 'value'}, {statistic: 'mean'})
    );
    const without = observe(() => runCase(lattice, {valueProperty: ''}, {statistic: 'mean'}));
    expect(without).not.toBe(withColumn);
    const countWith = observe(() =>
      runCase(lattice, {valueProperty: 'value'}, {statistic: 'count'})
    );
    const countWithout = observe(() => runCase(lattice, {valueProperty: ''}, {statistic: 'count'}));
    expect(countWithout).toBe(countWith);
  });
});

describe('network, aggregation and raster descriptors: what each requirement rests on', () => {
  it('network cost requirements: negative costs fail ingestion with a named limit', () => {
    const testCase = getCase('network-shortest-path');
    const negative = makeCollection(
      STREETS.features.map((feature, index) =>
        index === 0 ? {...feature, properties: {...feature.properties, travel_time: -5}} : feature
      )
    );
    expect(() =>
      runCase({...testCase, inputs: {...testCase.inputs, lines: negative}}, testCase.columns)
    ).toThrow(/^network-shortest-path.*forwardCostProperty.*at least 0/);
    const requirements = getDescriptor('network-shortest-path').requirements;
    for (const id of ['no-missing-values', 'non-negative-values']) {
      expect(
        requirements.some(
          requirement => requirement.id === id && requirement.role === 'forward-cost'
        )
      ).toBe(true);
    }
  });

  it('network structure: costs cannot change it, so no cost requirement is declared', () => {
    const testCase = getCase('network-structure');
    const lengthMode = observe(() => runCase(testCase, {}, {costMode: 'length'}));
    const propertyMode = observe(() => runCase(testCase, testCase.columns));
    const tablesOf = (text: string) => JSON.parse(text).tables;
    expect(tablesOf(propertyMode)).toEqual(tablesOf(lengthMode));
    for (const id of ['network-structure', 'network-line-graph']) {
      expect(
        getDescriptor(id).requirements.some(requirement => requirement.role !== undefined),
        id
      ).toBe(false);
    }
  });

  it('accessibility: missing weights are excluded and diagnosed', () => {
    const testCase = getCase('network-accessibility');
    const result = runCase(testCase, {...testCase.columns, weightProperty: 'no_such_column'});
    const score = result.tables[0].columns['accessibility'] as number[];
    expect(score.every(value => value === 0)).toBe(true);
    expect(result.tables[0].columns['reachable_count'].every(value => value === 0)).toBe(true);
    expect(
      result.diagnostics.some(diagnostic => diagnostic.code === 'opportunity-weight-missing')
    ).toBe(true);
    const roles = getDescriptor('network-accessibility').requirements.filter(
      requirement => requirement.role === 'weight'
    );
    expect(roles).toEqual([]);
    expect(getDescriptor('network-accessibility').notes?.join(' ')).toContain(
      'excluded from reachable counts'
    );
  });

  it('flow aggregation: an origin without a weight carries no flow and is listed', () => {
    const testCase = getCase('network-flow-aggregation');
    const origins = makeCollection([makePoint([0, 0], {trips: 100}), makePoint([200, 0])]);
    const result = runCase({...testCase, inputs: {...testCase.inputs, origins}}, testCase.columns);
    const pairs = result.tables.find(table => table.id === 'flow-pairs')!;
    expect(pairs.columns['status'][1]).not.toBe('ok');
    expect(pairs.columns['status'][0]).toBe('ok');
  });

  it('k function: fewer than two points leave K empty, as minimum-rows says', () => {
    const testCase = getCase('network-k-function');
    const single = makeCollection([makePoint([5, 0])]);
    const result = runCase(
      {...testCase, inputs: {...testCase.inputs, points: single}},
      testCase.columns
    );
    const kValues = result.tables[0].columns['k'];
    expect(kValues.every(value => value === null)).toBe(true);
    const requirement = getDescriptor('network-k-function').requirements.find(
      candidate => candidate.id === 'minimum-rows'
    );
    expect(requirement?.limit).toBe(2);
    // Two points are enough to produce a value for the largest radius.
    const full = runCase(testCase, testCase.columns).tables[0].columns['k'];
    expect(full.some(value => typeof value === 'number')).toBe(true);
  });

  it('grid cell families: projected coordinates fall outside the grid, as geographic-coordinates says', () => {
    const projected = makeCollection([makePoint([500000, 4649776]), makePoint([500100, 4649800])]);
    for (const id of ['dggs-point-index', 'dggs-cell-aggregation', 'dggs-cell-rollup']) {
      const testCase = getCase(id);
      const result = runCase({...testCase, inputs: {points: projected}}, testCase.columns);
      expect(result.tables[0].rowCount, id).toBe(0);
      expect(
        getDescriptor(id).requirements.some(
          requirement =>
            requirement.id === 'geographic-coordinates' && requirement.severity === 'block'
        ),
        id
      ).toBe(true);
    }
  });

  it('cell-aggregation retains missing-value points in count and reports their missing values', () => {
    const withGap = makeCollection(
      EVENTS.features.map((feature, index) =>
        index === 0 ? {...feature, properties: {...feature.properties, value: null}} : feature
      )
    );
    const testCase = getCase('cell-aggregation');
    const full = runCase(testCase, testCase.columns);
    const gap = runCase({...testCase, inputs: {points: withGap}}, testCase.columns);
    const total = (result: typeof full) =>
      (result.tables[0].columns['count'] as number[]).reduce((sum, value) => sum + value, 0);
    expect(total(full)).toBe(20);
    expect(total(gap)).toBe(20);
    expect(gap.summary['missingValueCount']).toBe(1);
    expect(gap.diagnostics.map(diagnostic => diagnostic.code)).toContain('missing-value');
  });

  it('overlapping zones count a point once for each zone, as disjoint-coverage says', () => {
    const testCase = getCase('zonal-point-statistics');
    const overlapping = makeCollection([...ZONES.features, ZONES.features[0]]);
    const separate = runCase(testCase, testCase.columns);
    const doubled = runCase(
      {...testCase, inputs: {...testCase.inputs, polygons: overlapping}},
      testCase.columns
    );
    const total = (result: typeof separate) =>
      (result.tables[0].columns['count'] as number[]).reduce((sum, value) => sum + value, 0);
    expect(total(doubled)).toBeGreaterThan(total(separate));
    expect(
      getDescriptor('zonal-point-statistics').requirements.some(
        requirement => requirement.id === 'disjoint-coverage'
      )
    ).toBe(true);
  });
});

describe('network, aggregation and raster descriptors: derived defaults', () => {
  const derived = ALL_DESCRIPTORS.flatMap(descriptor =>
    descriptor.parameters
      .filter(parameter => parameter.defaultPolicy.from === 'data')
      .map(parameter => ({descriptor, parameter}))
  );

  it('derives the seven cell sizes, with the rule that matches the unit of the parameter', () => {
    expect(derived.map(({descriptor}) => descriptor.bindingId).sort()).toEqual(
      [
        'raster-join-points-in-polygons',
        'recipe-straight-line-catchments',
        'raster-density-smoothing',
        'raster-cell-clusters',
        'cell-aggregation',
        'cell-rollup',
        'cell-compare'
      ].sort()
    );
    for (const {descriptor, parameter} of derived) {
      expect(parameter.name).toBe('cellSize');
      const spec = getBinding(descriptor.bindingId).parameters.find(
        candidate => candidate.name === 'cellSize'
      );
      // The ground rule returns metres for geographic data and coordinate units for planar data,
      // which is the `crs` unit of the grid bindings. The lattice bindings measure cells in
      // coordinate units (degrees), which only the source-unit rule returns.
      const expectedRule =
        spec?.unit === 'crs'
          ? 'extent-shorter-side-over-250'
          : 'extent-shorter-side-over-250-in-source-units';
      expect(['crs', 'coordinate-units'], descriptor.bindingId).toContain(spec?.unit);
      expect(parameter.defaultPolicy).toMatchObject({from: 'data', rule: expectedRule});
    }
  });

  it('reads the extent of an input the binding has', () => {
    for (const {descriptor, parameter} of derived) {
      if (parameter.defaultPolicy.from !== 'data') continue;
      expect(
        descriptor.inputs.map(input => input.name),
        descriptor.bindingId
      ).toContain(parameter.defaultPolicy.input);
    }
  });

  /** Four hundred planar points spread over a rectangle of the given size. */
  function makeSpread(width: number, height: number) {
    return makeCollection(
      Array.from({length: 400}, (_, index) =>
        makePoint([(index % 20) * (width / 20) + 1, Math.floor(index / 20) * (height / 20) + 1], {
          value: index
        })
      )
    );
  }

  function resolveCellSize(descriptor: KeplerOperationDescriptor, shorterSide: number): number {
    const parameter = descriptor.parameters.find(candidate => candidate.name === 'cellSize')!;
    if (parameter.defaultPolicy.from !== 'data') throw new Error('cellSize is not derived');
    return (shorterSide / 250) * (parameter.defaultPolicy.factor ?? 1);
  }

  it.each([
    'raster-density-smoothing',
    'raster-cell-clusters'
  ])('%s: the derived cell size stays inside the 40,000 cell grid; the bare rule and the fixed default do not', id => {
    const binding = getBinding(id);
    const parameters = {
      ...getCatalogDefaultParameters(binding),
      coordinateSystem: 'planar',
      valueProperty: 'value'
    };
    // 5,000 by 3,000 units: the shorter side is 3,000 and the aspect ratio 1.7.
    const points = makeSpread(5000, 3000);
    const derivedSize = resolveCellSize(getDescriptor(id), 3000);
    expect(() =>
      runCatalogOperation(binding, {...parameters, cellSize: derivedSize}, {points})
    ).not.toThrow();
    // The extent rule without the factor makes 417 by 251 cells.
    expect(() =>
      runCatalogOperation(binding, {...parameters, cellSize: 3000 / 250}, {points})
    ).toThrow();
    // The fixed default of 10 units makes 501 by 301 cells.
    expect(() => runCatalogOperation(binding, {...parameters, cellSize: 10}, {points})).toThrow();
  });

  it('recipe-straight-line-catchments: the derived cell size fits the 250,000 cell grid', () => {
    const binding = getBinding('recipe-straight-line-catchments');
    const parameters = {
      ...getCatalogDefaultParameters(binding),
      coordinateSystem: 'planar',
      valueProperty: 'value'
    };
    const facilities = makeCollection([makePoint([1000, 1000]), makePoint([15000, 9000])]);
    const observations = makeSpread(20000, 12000);
    const derivedSize = resolveCellSize(getDescriptor('recipe-straight-line-catchments'), 12000);
    const result = runCatalogOperation(
      binding,
      {...parameters, cellSize: derivedSize},
      {facilities, observations}
    );
    expect(Number(result.summary['columns']) * Number(result.summary['rows'])).toBeLessThanOrEqual(
      250000
    );
    // The fixed default of 10 units would need two million cells.
    expect(() =>
      runCatalogOperation(binding, {...parameters, cellSize: 10}, {facilities, observations})
    ).toThrow();
  });

  it('raster-join-points-in-polygons: the derived cell size fits the 250,000 cell grid of the GPU engine', () => {
    // Only the GPU engine rasterizes the polygons (the CPU engine joins exactly and ignores the
    // cell size), and it builds its grid with makeGridSpec, so the grid is checked there.
    const binding = getBinding('raster-join-points-in-polygons');
    expect(binding.run).toBeDefined();
    // The polygons span 20,000 by 10,000 units: the shorter side is 10,000.
    const derivedSize = resolveCellSize(getDescriptor('raster-join-points-in-polygons'), 10000);
    const grid = makeGridSpec(20000, 10000, derivedSize);
    expect(grid.columns * grid.rows).toBeLessThanOrEqual(250000);
    expect(() => makeGridSpec(20000, 10000, 10)).toThrow();
    // The bare rule gives 501 by 251 cells here: still inside, so the factor is a margin for
    // elongated extents, shown by an aspect ratio of 12.
    expect(() => makeGridSpec(48000, 4000, 4000 / 250)).toThrow();
    expect(() =>
      makeGridSpec(
        48000,
        4000,
        resolveCellSize(getDescriptor('raster-join-points-in-polygons'), 4000)
      )
    ).not.toThrow();
  });
});

describe('network, aggregation and raster descriptors: derived defaults on geographic data', () => {
  // Four hundred events over about 8.3 km by 5 km (a city), EPSG:4326.
  const CITY_EVENTS = makeCollection(
    Array.from({length: 400}, (_, index) =>
      makePoint([-87.6993 + (index % 20) * 0.005, 41.8603 + Math.floor(index / 20) * 0.0024], {
        value: index + 1
      })
    )
  );
  // Two zones that cover the events: 0.1 degrees wide and 0.05 high.
  const CITY_ZONES = makeCollection([
    makeSquare(-87.7, 41.86, 0.05),
    makeSquare(-87.65, 41.86, 0.05)
  ]);
  const CITY_FACILITIES = makeCollection([makePoint([-87.68, 41.87]), makePoint([-87.62, 41.89])]);

  function collectCoordinates(collection: CatalogFeatureCollection): number[] {
    const coordinates: number[] = [];
    for (const feature of collection.features) {
      const geometry = feature.geometry as {type: string; coordinates: unknown};
      if (geometry.type === 'Point') coordinates.push(...(geometry.coordinates as number[]));
      else
        for (const position of (geometry.coordinates as number[][][])[0])
          coordinates.push(...position);
    }
    return coordinates;
  }

  /** The profile a host builds for one input before resolving a default. */
  function makeProfile(collection: CatalogFeatureCollection): GPUAnalysisDataProfile {
    const spacing = computeGPUAnalysisSpacingStatistics(collectCoordinates(collection), {
      coordinateModel: 'geographic'
    });
    if (!spacing) throw new Error('The fixture has no points');
    return {
      kind: 'points',
      rowCount: collection.features.length,
      coordinates: {model: 'geographic', source: 'declared'},
      extent: spacing.extent,
      columns: [],
      spacing
    };
  }

  function resolveDescriptorCellSize(id: string, collection: CatalogFeatureCollection) {
    const parameter = getDescriptor(id).parameters.find(
      candidate => candidate.name === 'cellSize'
    )!;
    if (parameter.defaultPolicy.from !== 'data') throw new Error('cellSize is not derived');
    return resolveGPUAnalysisDefault(
      parameter.defaultPolicy.rule,
      makeProfile(collection),
      parameter.defaultPolicy.factor
    );
  }

  it('resolves the extent rule in metres for geographic data (the rule must be the fixed one)', () => {
    // Expects: a ground distance in metres, from the spacing statistics' extent width and height.
    const resolution = resolveGPUAnalysisDefault(
      'extent-shorter-side-over-250',
      makeProfile(CITY_EVENTS)
    );
    expect(resolution.unit).toBe('metre');
    const spacing = makeProfile(CITY_EVENTS).spacing!;
    // About 0.0456 degrees of latitude is about 5 km; degrees would give a value below 0.001.
    expect(spacing.extentHeight).toBeGreaterThan(4000);
    expect(resolution.value).toBeCloseTo(
      Math.min(spacing.extentWidth, spacing.extentHeight) / 250,
      6
    );
    expect(resolution.value!).toBeGreaterThan(10);
  });

  it.each([
    ['raster-density-smoothing', 'points', 40000],
    ['raster-cell-clusters', 'points', 40000]
  ])('%s: the derived metre cell size runs on EPSG:4326 under the cell cap', (id, inputName, cap) => {
    const binding = getBinding(id);
    const resolution = resolveDescriptorCellSize(id, CITY_EVENTS);
    expect(resolution.unit).toBe('metre');
    // The cell size is in the `crs` unit, which the binding resolves in a local metre frame.
    expect(binding.parameters.find(parameter => parameter.name === 'cellSize')?.unit).toBe('crs');
    const parameters = {
      ...getCatalogDefaultParameters(binding),
      coordinateSystem: 'EPSG:4326',
      valueProperty: 'value',
      cellSize: resolution.value!
    };
    const result = runCatalogOperation(binding, parameters, {[inputName]: CITY_EVENTS});
    expect(Number(result.summary['columns']) * Number(result.summary['rows'])).toBeLessThanOrEqual(
      cap
    );
    // The fixed default of 10 metres makes about 830 by 500 cells on this extent.
    expect(() =>
      runCatalogOperation(binding, {...parameters, cellSize: 10}, {[inputName]: CITY_EVENTS})
    ).toThrow();
  });

  it('recipe-straight-line-catchments: the derived metre cell size runs on EPSG:4326 under the cell cap', () => {
    const id = 'recipe-straight-line-catchments';
    const binding = getBinding(id);
    const resolution = resolveDescriptorCellSize(id, CITY_EVENTS);
    expect(resolution.unit).toBe('metre');
    const parameters = {
      ...getCatalogDefaultParameters(binding),
      coordinateSystem: 'EPSG:4326',
      valueProperty: 'value',
      cellSize: resolution.value!
    };
    const result = runCatalogOperation(binding, parameters, {
      facilities: CITY_FACILITIES,
      observations: CITY_EVENTS
    });
    expect(Number(result.summary['columns']) * Number(result.summary['rows'])).toBeLessThanOrEqual(
      250000
    );
    expect(result.tables[0].rowCount).toBe(2);
    expect(() =>
      runCatalogOperation(
        binding,
        {...parameters, cellSize: 1},
        {facilities: CITY_FACILITIES, observations: CITY_EVENTS}
      )
    ).toThrow();
  });

  it('raster-join-points-in-polygons: the derived metre cell size gives a grid the GPU engine accepts', () => {
    const id = 'raster-join-points-in-polygons';
    const binding = getBinding(id);
    const resolution = resolveDescriptorCellSize(id, CITY_ZONES);
    expect(resolution.unit).toBe('metre');
    const spacing = makeProfile(CITY_ZONES).spacing!;
    const grid = makeGridSpec(spacing.extentWidth, spacing.extentHeight, resolution.value!);
    expect(grid.columns * grid.rows).toBeLessThanOrEqual(250000);
    // A cell size of 1 metre would need about 11 million cells.
    expect(() => makeGridSpec(spacing.extentWidth, spacing.extentHeight, 1)).toThrow();
    // The CPU engine runs with the derived value on geographic input and counts every event.
    const result = runCatalogOperation(
      binding,
      {
        ...getCatalogDefaultParameters(binding),
        coordinateSystem: 'EPSG:4326',
        valueProperty: 'value',
        cellSize: resolution.value!
      },
      {points: CITY_EVENTS, polygons: CITY_ZONES}
    );
    const counts = result.tables[0].columns['count'] as number[];
    expect(counts.reduce((sum, value) => sum + value, 0)).toBe(400);
  });

  it('gives no value for no points, so the constant stands', () => {
    expect(
      computeGPUAnalysisSpacingStatistics([], {coordinateModel: 'geographic'})
    ).toBeUndefined();
  });
});

describe('network, aggregation and raster descriptors: class schemes', () => {
  it('uses no class scheme: no output here has a column that is a class scheme with its own noise or flag values', () => {
    // Component and cluster ids of these bindings carry no noise label (the grid components are
    // row-major cell numbers plus one, the network components are plain ids), so none of them is
    // `cluster-membership`; none writes a flag. Each scale is therefore categorical, sequential
    // or diverging, and any descriptor that adds a scheme must pass the check below.
    for (const descriptor of ALL_DESCRIPTORS) {
      for (const output of descriptor.outputs) {
        expect(output.presentation?.scale, `${descriptor.bindingId} ${output.tableId}`).not.toBe(
          'class-scheme'
        );
      }
    }
  });

  it.each(RUNNABLE_IDS)('%s: any class scheme named scores 1 on the real output', id => {
    const testCase = getCase(id);
    const result = runCase(testCase, testCase.columns);
    for (const output of getDescriptor(id).outputs) {
      const presentation = output.presentation;
      if (presentation?.scale !== 'class-scheme') continue;
      const table = result.tables.find(candidate => candidate.id === output.tableId)!;
      expect(
        validateGPUAnalysisResultEncoding(
          presentation.classScheme!,
          presentation.classEncoding ?? 'contributor-integer',
          table.columns[presentation.column]
        )
      ).toBe(1);
    }
  });
});

describe('network, aggregation and raster descriptors: fix round', () => {
  const COST_ROLES = ['forward-cost', 'reverse-cost'];
  const COST_READING_IDS = NETWORK_IDS.filter(id => {
    // K retains column roles for its binding ABI, but rejects property-cost execution.
    if (id === 'network-k-function') return false;
    const descriptor = getDescriptor(id);
    return descriptor.inputs.some(input =>
      input.fields.some(
        field => field.role === 'forward-cost' && typeof field.required === 'object'
      )
    );
  });
  const INERT_IDS = ['network-structure', 'network-line-graph', 'network-snap-points'];

  it('separates supported cost bindings from inert bindings and length-only Network K', () => {
    for (const id of INERT_IDS) expect(COST_READING_IDS).not.toContain(id);
    // Noding and edge bundling have no cost parameters. Network K supports length mode only.
    expect(COST_READING_IDS).not.toContain('network-k-function');
    expect(COST_READING_IDS.length).toBe(NETWORK_IDS.length - INERT_IDS.length - 3);
  });

  it.each(
    COST_READING_IDS
  )('%s: missing costs exclude directions with a warning and are noted', id => {
    const descriptor = getDescriptor(id);
    for (const role of COST_ROLES) {
      const requirement = descriptor.requirements.find(
        candidate => candidate.id === 'no-missing-values' && candidate.role === role
      );
      expect(requirement?.severity, `${id} ${role}`).toBe('warn');
      expect(requirement?.remedy).toMatch(/Fill or remove missing costs/);
    }
    expect(
      descriptor.notes?.some(note => /missing, blank/.test(note)),
      id
    ).toBe(true);
  });

  it('null, empty and absent costs all remove arcs and report the count', () => {
    const testCase = getCase('network-shortest-path');
    const withCost = (value: unknown) =>
      makeCollection(
        STREETS.features.map(feature => ({
          ...feature,
          properties: {...feature.properties, travel_time: value}
        }))
      );
    // Only the forward cost is named, so the reverse cost equals it.
    const forwardOnly = {forwardCostProperty: 'travel_time'};
    const run = (lines: CatalogFeatureCollection, columns: Record<string, string> = forwardOnly) =>
      runCase({...testCase, inputs: {...testCase.inputs, lines}}, columns);
    for (const missing of [null, '']) {
      const result = run(withCost(missing));
      expect(result.diagnostics.map(diagnostic => diagnostic.code)).toContain(
        'invalid-cost-arc-removed'
      );
      const costs = result.tables[0].columns['total_cost'] as (number | null)[];
      // No route can traverse a street without its required cost.
      expect(costs.every(cost => cost === null)).toBe(true);
    }
    // An absent column is reported, so it is a warning and not a silent wrong answer.
    const absent = run(STREETS, {forwardCostProperty: 'no_such_column'});
    expect(absent.diagnostics.map(diagnostic => diagnostic.code)).toContain(
      'invalid-cost-arc-removed'
    );
  });

  it.each(
    INERT_IDS
  )('%s: topology exposes no cost roles and ignores obsolete saved parameters', id => {
    const descriptor = getDescriptor(id);
    const street = descriptor.inputs.find(input => input.name === 'lines')!;
    expect(street.fields).toEqual([]);
    expect(getBinding(id).parameters.map(parameter => parameter.name)).not.toContain('costMode');
    expect(descriptor.requirements.some(requirement => requirement.role !== undefined)).toBe(false);
    expect(descriptor.notes?.some(note => /obsolete saved parameters are ignored/.test(note))).toBe(
      true
    );
    // The claim holds in the binding: switching to property costs does not change the output.
    const testCase = getCase(id);
    const tables = (extra: Values, columns: Record<string, string>) =>
      JSON.stringify(runCase(testCase, columns, extra).tables);
    expect(tables({costMode: 'property'}, NETWORK_COLUMNS)).toBe(tables({costMode: 'length'}, {}));
  });

  it('the roles of structure and the others stay required in property mode where costs matter', () => {
    for (const id of COST_READING_IDS) {
      const field = getDescriptor(id)
        .inputs.flatMap(input => input.fields)
        .find(candidate => candidate.role === 'forward-cost')!;
      expect(field.required).toEqual({whenParameter: 'costMode', equals: ['property']});
    }
  });

  it('network-accessibility: missing weights are excluded and negative weights are rejected', () => {
    const testCase = getCase('network-accessibility');
    const result = runCase(testCase, {...testCase.columns, weightProperty: 'no_such_column'});
    const reachable = result.tables[0].columns['reachable_count'] as number[];
    expect(reachable.every(count => count === 0)).toBe(true);
    const descriptor = getDescriptor('network-accessibility');
    expect(
      descriptor.notes?.some(
        note => /excluded from reachable counts/.test(note) && /negative/.test(note)
      )
    ).toBe(true);
    const remedies = descriptor.requirements
      .filter(requirement => requirement.role === 'weight')
      .map(requirement => requirement.remedy);
    expect(remedies).toEqual([]);
    expect(descriptor.notes?.join(' ')).toContain(
      'negative weights throw a named validation error'
    );
    // Negative weights fail ingestion rather than reducing the score.
    const negative = makeCollection(
      OPPORTUNITIES.features.map(feature => ({...feature, properties: {jobs: -5}}))
    );
    expect(() =>
      runCase(
        {...testCase, inputs: {...testCase.inputs, opportunities: negative}},
        testCase.columns
      )
    ).toThrow(/^network-accessibility.*weightProperty.*at least 0/);
  });

  it('network-flow-aggregation: a negative weight is an invalid pair, and the comment says so', () => {
    const testCase = getCase('network-flow-aggregation');
    const origins = makeCollection([
      makePoint([0, 0], {trips: -3}),
      makePoint([200, 0], {trips: 40})
    ]);
    const result = runCase({...testCase, inputs: {...testCase.inputs, origins}}, testCase.columns);
    const pairs = result.tables.find(table => table.id === 'flow-pairs')!;
    expect(pairs.columns['status'][0]).toBe('invalid-pair');
    const descriptor = getDescriptor('network-flow-aggregation');
    const requirement = descriptor.requirements.find(
      candidate => candidate.id === 'non-negative-values' && candidate.role === 'weight'
    );
    expect(requirement?.remedy).toMatch(/invalid-pair/);
    expect(
      descriptor.notes?.some(note => /paired by row order/.test(note) && /surplus/.test(note))
    ).toBe(true);
    // Surplus rows are ignored with a diagnostic.
    const surplus = makeCollection([...FLOW_DESTINATIONS.features, makePoint([100, 100])]);
    const unpaired = runCase(
      {...testCase, inputs: {...testCase.inputs, destinations: surplus}},
      testCase.columns
    );
    expect(unpaired.diagnostics.map(diagnostic => diagnostic.code)).toContain('unpaired-features');
  });

  it('network-subgraph-filter: a column name with an empty value removes every edge, as the note says', () => {
    const testCase = getCase('network-subgraph-filter');
    const result = runCase(testCase, testCase.columns, {propertyValue: ''});
    expect(result.tables[0].rowCount).toBe(0);
    expect(
      getDescriptor('network-subgraph-filter').notes?.some(note => /empty value/.test(note))
    ).toBe(true);
  });

  it('activeWhen: decayScale only matters for the exponential and gaussian decay', () => {
    const testCase = getCase('network-accessibility');
    const parameter = getDescriptor('network-accessibility').parameters.find(
      candidate => candidate.name === 'decayScale'
    );
    expect(parameter?.activeWhen).toEqual({
      parameter: 'decay',
      equals: ['exponential', 'gaussian']
    });
    const observeWith = (decay: string, decayScale: number) =>
      observe(() => runCase(testCase, testCase.columns, {decay, decayScale}));
    for (const decay of ['cumulative', 'linear']) {
      expect(observeWith(decay, 5)).toBe(observeWith(decay, 500));
    }
    for (const decay of ['exponential', 'gaussian']) {
      expect(observeWith(decay, 5)).not.toBe(observeWith(decay, 500));
    }
  });

  it('activeWhen: penalty and maximumOverlap only matter when there is more than one route', () => {
    const testCase = getCase('network-alternative-routes');
    const descriptor = getDescriptor('network-alternative-routes');
    for (const name of ['penalty', 'maximumOverlap']) {
      expect(descriptor.parameters.find(parameter => parameter.name === name)?.activeWhen).toEqual({
        parameter: 'routeCount',
        equals: [2, 3, 4, 5, 6]
      });
    }
    const observeWith = (penalty: number, maximumOverlap: number) =>
      observe(() => runCase(testCase, testCase.columns, {routeCount: 1, penalty, maximumOverlap}));
    expect(observeWith(0.1, 0.1)).toBe(observeWith(9, 1));
  });

  it('describes outputs without repeating the primary question, without ids to colour by, and without a map for tables', () => {
    const highCardinality = [
      'component',
      'component_id',
      'source_index',
      'level',
      'route_rank',
      'cell_id'
    ];
    for (const descriptor of ALL_DESCRIPTORS) {
      const binding = getBinding(descriptor.bindingId);
      for (const output of descriptor.outputs) {
        const id = `${descriptor.bindingId} ${output.tableId}`;
        expect(output.next ?? [], id).not.toContain(descriptor.questions[0]);
        if (output.presentation) {
          expect(highCardinality, id).not.toContain(output.presentation.column);
        }
        const spec = binding.outputs.find(candidate => candidate.tableId === output.tableId)!;
        if (spec.geometry === 'none') expect(output.presentation, id).toBeUndefined();
      }
    }
  });

  it('marks maximumSnapDistance basic only where snapping is the point of the operation', () => {
    for (const descriptor of NETWORK_OPERATION_DESCRIPTORS) {
      const parameter = descriptor.parameters.find(
        candidate => candidate.name === 'maximumSnapDistance'
      );
      if (!parameter) continue;
      expect(parameter.level, descriptor.bindingId).toBe(
        descriptor.bindingId === 'network-snap-points' ? 'basic' : 'advanced'
      );
    }
  });

  it('states what is true by default for the bands and limits of cost', () => {
    expect(getDescriptor('network-isochrone-polygons').action).not.toMatch(/time/);
    for (const id of [
      'network-service-area',
      'network-multi-origin-service-area',
      'network-isochrone-polygons',
      'network-neighborhood',
      'network-accessibility'
    ]) {
      expect(
        getDescriptor(id).notes?.some(note => /metres, not travel times/.test(note)),
        id
      ).toBe(true);
    }
    // The defaults are lengths: with costMode length the budgets are metres.
    expect(getCatalogDefaultParameters(getBinding('network-service-area'))['costMode']).toBe(
      'length'
    );
    // No cost limit is derived from the data: they are in the cost unit, which can be time.
    for (const id of ['network-neighborhood', 'network-accessibility', 'network-service-area']) {
      for (const parameter of getDescriptor(id).parameters) {
        if (['maximumCost', 'decayScale', 'budgets'].includes(parameter.name)) {
          expect(parameter.defaultPolicy, `${id}.${parameter.name}`).toEqual({from: 'constant'});
        }
      }
    }
  });

  it('notes the edge bundling input and the 40,000 and 250,000 cell limits where they apply', () => {
    expect(getDescriptor('network-edge-bundling').prominence).toBe(3);
    for (const id of ['raster-density-smoothing', 'raster-cell-clusters']) {
      expect(
        getDescriptor(id).notes?.some(note => /40,000/.test(note)),
        id
      ).toBe(true);
    }
    for (const id of [
      'recipe-straight-line-catchments',
      'raster-density-smoothing',
      'raster-cell-clusters'
    ]) {
      expect(
        getDescriptor(id).notes?.some(note => /Isolated points/.test(note)),
        id
      ).toBe(true);
    }
  });

  it('network-k-function and the choropleth use templates the sentence builder can write', () => {
    expect(getDescriptor('network-k-function').outputs[0].interpretation?.summary).toBe(
      'value-range'
    );
    const choropleth = getDescriptor('recipe-points-in-polygons-choropleth');
    // The sentence counts joined points for every selected map statistic.
    expect(
      getBinding('recipe-points-in-polygons-choropleth').parameters.find(
        p => p.name === 'statistic'
      )?.defaultValue
    ).toBe('count');
    expect(choropleth.outputs[0].interpretation?.summary).toBe('count-per-area');
    expect(choropleth.notes).toContain(
      'The sentence counts joined points; the map shows the selected statistic.'
    );
    const testCase = getCase('cell-aggregation');
    const total = (columns: Record<string, string>) =>
      (runCase(testCase, columns).tables[0].columns['count'] as number[]).reduce(
        (sum, value) => sum + value,
        0
      );
    // Counts include every point even when the named value column is absent.
    expect(total({valueProperty: ''})).toBe(20);
    expect(total({valueProperty: 'no_such_column'})).toBe(20);
  });

  it('retains the distribution coverage warning while corrected joins count every match', () => {
    for (const id of ['recipe-zonal-distribution']) {
      expect(
        getDescriptor(id).requirements.some(
          requirement =>
            requirement.id === 'disjoint-coverage' &&
            requirement.severity === 'warn' &&
            requirement.input === 'polygons' &&
            (requirement.remedy ?? '').length > 0
        ),
        id
      ).toBe(true);
    }
    for (const id of ['recipe-points-in-polygons-choropleth', 'raster-join-points-in-polygons']) {
      const testCase = getCase(id);
      const overlapping = makeCollection([ZONES.features[0], ZONES.features[0]]);
      const result = runCase(
        {...testCase, inputs: {...testCase.inputs, polygons: overlapping}},
        testCase.columns,
        id.startsWith('raster') ? {} : {statistic: 'count'}
      );
      const counts = result.tables[0].columns[
        id.startsWith('raster') ? 'count' : 'point_count'
      ] as number[];
      // Both polygons receive the points, with multiple matches reported per input point.
      expect(counts).toEqual([7, 7]);
      expect(result.summary['multipleMatchCount']).toBe(7);
    }
  });

  it('raster-join-points-in-polygons: the action is true without a GPU, the cell size is advanced and the count exact', () => {
    const descriptor = getDescriptor('raster-join-points-in-polygons');
    expect(descriptor.action).not.toMatch(/fast/i);
    expect(descriptor.parameters.find(parameter => parameter.name === 'cellSize')?.level).toBe(
      'advanced'
    );
    expect(descriptor.notes?.some(note => /exact/.test(note))).toBe(true);
    const testCase = getCase('raster-join-points-in-polygons');
    // The CPU engine ignores the cell size.
    const tables = (cellSize: number) =>
      JSON.stringify(runCase(testCase, testCase.columns, {cellSize}).tables);
    expect(tables(10)).toBe(tables(5000));
  });

  it('cell-compare and dggs-cell-compare read the column from both layers and say so', () => {
    for (const id of ['cell-compare', 'dggs-cell-compare']) {
      const descriptor = getDescriptor(id);
      expect(
        descriptor.notes?.some(note => /both layers/.test(note)),
        id
      ).toBe(true);
      const testCase = getCase(id);
      const after = makeCollection(
        EVENTS_AFTER.features.map(feature => ({...feature, properties: {}}))
      );
      // The after layer has no such column: its rows are dropped (dggs) or read as 0 (lattice),
      // so the result differs from the one with the column in both layers.
      expect(
        observe(() => runCase({...testCase, inputs: {...testCase.inputs, after}}, testCase.columns))
      ).not.toBe(observe(() => runCase(testCase, testCase.columns)));
    }
    // Both layers declare the same role on the same parameter, with the same condition.
    for (const id of ['cell-compare', 'dggs-cell-compare']) {
      const descriptor = getDescriptor(id);
      const fields = descriptor.inputs.map(input => input.fields[0]);
      expect(fields.length).toBe(2);
      expect(fields[0]).toEqual(fields[1]);
      expect(fields[0].parameter).toBe('valueProperty');
      expect(
        descriptor.requirements
          .filter(requirement => requirement.id === 'no-missing-values')
          .map(requirement => requirement.input)
          .sort()
      ).toEqual(['after', 'before']);
      expect(
        descriptor.notes?.some(note =>
          /curated mapping offers only columns present in both/.test(note)
        )
      ).toBe(true);
      expect(validateKeplerOperationDescriptor(descriptor, getBinding(id))).toEqual([]);
      // The validator rejects two different roles on one parameter.
      const different = {
        ...descriptor,
        inputs: descriptor.inputs.map((input, index) => ({
          ...input,
          fields: input.fields.map(field => ({...field, role: index === 0 ? 'value' : 'other'}))
        }))
      };
      expect(
        validateKeplerOperationDescriptor(different, getBinding(id)).some(issue =>
          /receives the roles/.test(issue)
        )
      ).toBe(true);
    }
    // No other binding of these streams reads one column parameter from two inputs.
    for (const descriptor of ALL_DESCRIPTORS) {
      if (['cell-compare', 'dggs-cell-compare'].includes(descriptor.bindingId)) continue;
      const parameters = getFieldParameters(descriptor);
      expect(new Set(parameters).size, descriptor.bindingId).toBe(parameters.length);
    }
  });
});

describe('network, aggregation and raster descriptors: lattice defaults and outliers', () => {
  /** Deterministic uniform numbers (linear congruential generator). */
  function makeRandom(seed: number) {
    let state = seed;
    return () => {
      state = (state * 1664525 + 1013904223) % 4294967296;
      return state / 4294967296;
    };
  }

  function makeUniformPoints(
    count: number,
    west: number,
    south: number,
    width: number,
    height: number,
    seed: number
  ) {
    const next = makeRandom(seed);
    return makeCollection(
      Array.from({length: count}, (_, index) =>
        makePoint([west + next() * width, south + next() * height], {value: index % 7})
      )
    );
  }

  function makeExtentProfile(
    collection: CatalogFeatureCollection,
    model: 'geographic' | 'planar'
  ): GPUAnalysisDataProfile {
    const coordinates = collection.features.flatMap(
      feature => (feature.geometry as {coordinates: number[]}).coordinates
    );
    const spacing = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: model});
    if (!spacing) throw new Error('The fixture has no points');
    return {
      kind: 'points',
      rowCount: collection.features.length,
      coordinates: {model, source: 'declared'},
      extent: spacing.extent,
      columns: [],
      spacing
    };
  }

  function resolveLatticeCellSize(
    id: string,
    collection: CatalogFeatureCollection,
    model: 'geographic' | 'planar'
  ) {
    const parameter = getDescriptor(id).parameters.find(
      candidate => candidate.name === 'cellSize'
    )!;
    if (parameter.defaultPolicy.from !== 'data') throw new Error('cellSize is not derived');
    return resolveGPUAnalysisDefault(
      parameter.defaultPolicy.rule,
      makeExtentProfile(collection, model),
      parameter.defaultPolicy.factor
    );
  }

  // Three thousand points over 33 km by 33 km in metres (UTM-like) and over 2 km by 1.6 km in
  // degrees (a city).
  const PLANAR_POINTS = makeUniformPoints(3000, 376675, 5803711, 33526, 32902, 7);
  const GEOGRAPHIC_POINTS = makeUniformPoints(3000, -87.7, 41.85, 0.025, 0.015, 11);

  it.each([
    ['planar', PLANAR_POINTS],
    ['geographic', GEOGRAPHIC_POINTS]
  ] as const)('cell-aggregation, cell-rollup and cell-compare: the %s default gives neither one cell nor one cell per point', (model, points) => {
    for (const id of ['cell-aggregation', 'cell-rollup', 'cell-compare']) {
      const binding = getBinding(id);
      const resolution = resolveLatticeCellSize(id, points, model);
      expect(resolution.unit, id).toBe(model === 'planar' ? 'coordinate-unit' : 'degree');
      const parameters = {
        ...getCatalogDefaultParameters(binding),
        valueProperty: 'value',
        cellSize: resolution.value!
      };
      const inputs = id === 'cell-compare' ? {before: points, after: points} : {points};
      const result = runCatalogOperation(binding, parameters, inputs);
      const table = result.tables[0];
      const rows =
        id === 'cell-rollup'
          ? (table.columns['level'] as number[]).filter(level => level === 0).length
          : table.rowCount;
      // Between 100 and 1,500 occupied cells for 3,000 points: not one cell, and several
      // points per cell on average.
      expect(rows, `${model} ${id}`).toBeGreaterThan(100);
      expect(rows, `${model} ${id}`).toBeLessThan(1500);
    }
  });

  it('the old constant 0.01 gives one cell per point in metres and one cell for a city in degrees', () => {
    const binding = getBinding('cell-aggregation');
    const run = (points: CatalogFeatureCollection) =>
      runCatalogOperation(
        binding,
        {...getCatalogDefaultParameters(binding), valueProperty: 'value'},
        {points}
      );
    // 0.01 m cells: each point is alone in its cell (a few coincide at most).
    expect(run(PLANAR_POINTS).tables[0].rowCount).toBeGreaterThan(2900);
    // 0.01 degrees is about 1 km: the whole city is two or three cells.
    expect(run(GEOGRAPHIC_POINTS).tables[0].rowCount).toBeLessThanOrEqual(6);
  });

  it('prefers the lattice for a city and the grid cells for comparison across wide extents', () => {
    expect(getDescriptor('cell-aggregation').prominence).toBe(1);
    expect(getDescriptor('dggs-cell-aggregation').prominence).toBe(2);
    const dggs = getDescriptor('dggs-cell-aggregation');
    expect(dggs.notes?.some(note => /no rule that maps an extent to a level/.test(note))).toBe(
      true
    );
    expect(dggs.notes?.some(note => /156 km/.test(note) && /0\.46 km/.test(note))).toBe(true);
    // The constant resolution gives one row for a city, as the note says.
    const binding = getBinding('dggs-cell-aggregation');
    const result = runCatalogOperation(
      binding,
      {...getCatalogDefaultParameters(binding), valueProperty: 'value'},
      {points: GEOGRAPHIC_POINTS}
    );
    expect(result.tables[0].rowCount).toBeLessThanOrEqual(2);
  });

  it('isolated points inflate every extent-based grid size; the tests of both rules', () => {
    const clean = makeUniformPoints(3000, -87.7, 41.85, 0.1, 0.07, 3);
    const withOutliers = makeCollection([
      ...clean.features,
      makePoint([-87.2, 42.3], {value: 1}),
      makePoint([-88.2, 41.4], {value: 1})
    ]);
    const sizes = (collection: CatalogFeatureCollection) => {
      const profile = makeExtentProfile(collection, 'geographic');
      return {
        extent: resolveGPUAnalysisDefault('extent-shorter-side-over-250', profile, 4).value!,
        spacing: resolveGPUAnalysisDefault('mean-point-spacing', profile, 2).value!,
        fence: resolveGPUAnalysisDefault('outlier-resistant-nearest-neighbor-distance', profile, 2)
          .value!
      };
    };
    const before = sizes(clean);
    const after = sizes(withOutliers);
    // The extent rule and mean-point-spacing both follow the extent, so two outliers inflate them
    // about tenfold; mean-point-spacing is no remedy.
    expect(after.extent / before.extent).toBeGreaterThan(5);
    expect(after.spacing / before.spacing).toBeGreaterThan(5);
    // The outlier-resistant rule does not move, but it knows nothing of the extent: its grid on
    // the outlier extent has many times more cells than on the clean extent, and no cap applies.
    expect(after.fence / before.fence).toBeLessThan(1.1);
    const profile = makeExtentProfile(withOutliers, 'geographic');
    const cells = (size: number) =>
      (profile.spacing!.extentWidth / size + 1) * (profile.spacing!.extentHeight / size + 1);
    const cleanProfile = makeExtentProfile(clean, 'geographic');
    expect(cells(after.fence)).toBeGreaterThan(
      20 *
        ((cleanProfile.spacing!.extentWidth / before.fence + 1) *
          (cleanProfile.spacing!.extentHeight / before.fence + 1))
    );
    // The extent rule with the descriptor factor keeps the grid at about 4,900 cells whatever the
    // outliers do, which is why it stays the default: it cannot exceed the cap.
    expect(cells(after.extent)).toBeLessThan(5500);
    // On a real run the city collapses into a few cells, as the note says.
    const binding = getBinding('raster-density-smoothing');
    const run = (collection: CatalogFeatureCollection, cellSize: number) =>
      runCatalogOperation(
        binding,
        {...getCatalogDefaultParameters(binding), valueProperty: 'value', cellSize},
        {points: collection}
      );
    const occupied = (collection: CatalogFeatureCollection, cellSize: number) =>
      (run(collection, cellSize).tables[0].columns['point_count'] as number[]).filter(
        count => count > 0
      ).length;
    expect(occupied(withOutliers, after.extent)).toBeLessThan(
      0.15 * occupied(clean, before.extent)
    );
    // Rounding at the cap: the descriptor's size never throws.
    expect(() => run(withOutliers, after.extent)).not.toThrow();
  });
});

describe('network descriptors: the one-way role', () => {
  it('prefers a boolean flag, then a category, an integer and text', () => {
    for (const id of NETWORK_IDS) {
      const role = getDescriptor(id)
        .inputs.flatMap(input => input.fields)
        .find(field => field.role === 'one-way');
      if (role) expect(role.accepts, id).toEqual(['boolean', 'category', 'integer', 'text']);
    }
  });

  it('matches a column named oneway ahead of unrelated categories', () => {
    const role = getDescriptor('network-shortest-path')
      .inputs.flatMap(input => input.fields)
      .find(field => field.role === 'one-way')!;
    expect(tokenizeGPUAnalysisColumnName(role.role)).toEqual(['one', 'way']);
    const column = (name: string, type: 'boolean' | 'category' | 'integer') => ({
      name,
      type,
      rowCount: 10,
      nullCount: 0
    });
    const matches = matchGPUAnalysisFieldRole(role, [
      column('highway', 'category'),
      column('road_class', 'category'),
      column('oneway', 'category'),
      column('is_closed', 'boolean')
    ]);
    // A boolean flag comes before categories by type, and among categories the name wins.
    expect(matches.map(match => match.name)).toEqual([
      'is_closed',
      'oneway',
      'highway',
      'road_class'
    ]);
    const names = matchGPUAnalysisFieldRole(role, [
      column('highway', 'category'),
      column('OneWay', 'category')
    ]).map(match => match.name);
    expect(names[0]).toBe('OneWay');
  });
});

describe('review regressions: network applicability and descriptor contracts', () => {
  function assessNetwork(
    identifier: string,
    lines: CatalogFeatureCollection,
    overrides: Values = {}
  ) {
    const binding = getBinding(identifier);
    const descriptor = getDescriptor(identifier);
    const inputs = {...getCase(identifier).inputs, lines};
    const profiles = Object.entries(inputs).map(([name, features]) =>
      profileKeplerFeatureCollection(features, {
        datasetId: name,
        label: name,
        coordinateSystem: 'planar'
      })
    );
    return evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      overrides,
      executionEngine: 'cpu'
    });
  }

  function makeRectangularGrid(columns: number, rows: number) {
    const position = (column: number, row: number) => [420000 + column * 100, 5100000 + row * 100];
    return makeCollection([
      ...Array.from({length: rows}, (_, row) =>
        makeLine(
          `row-${row}`,
          Array.from({length: columns}, (_, column) => position(column, row))
        )
      ),
      ...Array.from({length: columns}, (_, column) =>
        makeLine(
          `column-${column}`,
          Array.from({length: rows}, (_, row) => position(column, row))
        )
      )
    ]);
  }

  it('all seven routing bindings accept 19,999 edges despite 20,200 vertices', () => {
    const lines = makeRectangularGrid(100, 101);
    for (const identifier of [
      'network-accessibility',
      'network-subgraph-filter',
      'network-coarsening',
      'network-flow-aggregation',
      'network-alternative-routes',
      'network-map-matching',
      'network-k-function'
    ]) {
      const assessment = assessNetwork(identifier, lines);
      expect(assessment.status, identifier).not.toBe('blocked');
      expect(
        assessment.requirements.some(
          requirement =>
            requirement.id === 'maximum-vertices' &&
            requirement.severity === 'warn' &&
            requirement.state === 'failed'
        ),
        identifier
      ).toBe(false);
    }
    expect(
      runCase({...getCase('network-subgraph-filter'), inputs: {lines}}, {}, {costMode: 'length'})
        .tables[0].rowCount
    ).toBe(19999);
  });

  it('centrality stays unknown without topology facts despite a large vertex count', () => {
    const lines = makeRectangularGrid(39, 39);
    const assessment = assessNetwork('network-centrality', lines);
    expect(assessment.status).not.toBe('blocked');
    expect(
      assessment.requirements.some(
        requirement =>
          requirement.id === 'maximum-network-nodes' &&
          requirement.severity === 'block' &&
          requirement.state === 'unknown' &&
          requirement.remedy?.includes('1,500 network nodes')
      )
    ).toBe(true);
    expect(() =>
      runCase({...getCase('network-centrality'), inputs: {lines}}, {}, {costMode: 'length'})
    ).toThrow('network-centrality: lines exceed the 1,500 node limit');
    expect(() =>
      runCase(
        {...getCase('network-centrality'), inputs: {lines: makeRectangularGrid(38, 38)}},
        {},
        {costMode: 'length'}
      )
    ).not.toThrow();
  });

  it('does not derive a graph lower bound from duplicate or multipart vertices', () => {
    const repeated = makeCollection([
      makeLine(
        'duplicate',
        Array.from({length: 20200}, () => [420000, 5100000])
      )
    ]);
    expect(assessNetwork('network-subgraph-filter', repeated).status).not.toBe('blocked');
    expect(
      runCase(
        {...getCase('network-subgraph-filter'), inputs: {lines: repeated}},
        {},
        {costMode: 'length'}
      ).tables[0].rowCount
    ).toBe(0);
    const multipart = makeCollection([
      {
        geometry: {
          type: 'MultiLineString',
          coordinates: Array.from({length: 10001}, () => [
            [420000, 5100000],
            [420100, 5100000]
          ])
        },
        properties: {}
      }
    ]);
    expect(assessNetwork('network-subgraph-filter', multipart).status).not.toBe('blocked');
    expect(
      runCase(
        {...getCase('network-subgraph-filter'), inputs: {lines: multipart}},
        {},
        {costMode: 'length'}
      ).tables[0].rowCount
    ).toBe(10001);
  });

  it('gates every cost-role requirement on property mode and leaves unused null costs inert', () => {
    for (const descriptor of NETWORK_OPERATION_DESCRIPTORS) {
      for (const requirement of descriptor.requirements.filter(candidate =>
        ['forward-cost', 'reverse-cost'].includes(candidate.role ?? '')
      )) {
        expect(requirement.activeWhen, descriptor.bindingId).toEqual({
          parameter: 'costMode',
          equals: ['property']
        });
      }
    }
    const lines = makeCollection([
      makeLine(
        'first',
        [
          [420000, 5100000],
          [420100, 5100000]
        ],
        {cost: null}
      ),
      makeLine(
        'second',
        [
          [420100, 5100000],
          [420200, 5100000]
        ],
        {cost: 10}
      )
    ]);
    expect(assessNetwork('network-build-graph', lines, {costMode: 'length'}).status).not.toBe(
      'blocked'
    );
    expect(assessNetwork('network-build-graph', lines, {costMode: 'property'}).status).toBe(
      'needs-confirmation'
    );
  });

  it('offers Network K with length costs and describes its unsupported property-cost mode', () => {
    const descriptor = getDescriptor('network-k-function');
    expect(descriptor.action).toBe('Describe clustering along streets');
    expect(
      descriptor.parameters.find(parameter => parameter.name === 'costMode')?.defaultPolicy
    ).toEqual({from: 'constant'});
    expect(descriptor.inputs[0].fields.map(field => field.role)).toEqual([
      'forward-cost',
      'reverse-cost',
      'one-way'
    ]);
    expect(descriptor.parameters.map(parameter => parameter.name)).not.toContain(
      'forwardCostProperty'
    );
    expect(descriptor.parameters.map(parameter => parameter.name)).not.toContain(
      'reverseCostProperty'
    );
    expect(
      descriptor.requirements.some(requirement =>
        ['forward-cost', 'reverse-cost'].includes(requirement.role ?? '')
      )
    ).toBe(false);
    expect(descriptor.notes?.[0]).toContain('rejects property costs');
    expect(descriptor.notes?.[0]).toContain('physical network length and radii must share units');
    const assessment = assessNetwork('network-k-function', STREETS);
    expect(assessment.parameters['costMode']).toBe('length');
    expect(assessment.fieldBindings).not.toHaveProperty('forwardCostProperty');
    expect(assessment.fieldBindings).not.toHaveProperty('reverseCostProperty');
  });

  it('describes numeric text conversion and direction removal only in property mode', () => {
    expect(getDescriptor('network-build-graph').notes?.[0]).toContain('In property cost mode');
    expect(getDescriptor('network-build-graph').notes?.[0]).toContain('numeric text is accepted');
    expect(getDescriptor('network-build-graph').notes?.[0]).toContain('affected direction');
    const lines = makeCollection([
      makeLine(
        'numeric-text',
        [
          [420000, 5100000],
          [420100, 5100000]
        ],
        {cost: '14', reverse: 'invalid'}
      )
    ]);
    const result = runCase(
      {...getCase('network-build-graph'), inputs: {lines}},
      {forwardCostProperty: 'cost', reverseCostProperty: 'reverse'},
      {costMode: 'property'}
    );
    expect(result.tables[0].columns['forward_cost']).toEqual([14]);
    expect(result.tables[0].columns['reverse_cost']).toEqual([null]);
  });

  it('declares polygon output inheritance and only the pairs that require area overlap', () => {
    const pairs: Record<string, readonly [string, string][]> = {
      'line-length-per-polygon': [['lines', 'polygons']],
      'zonal-point-statistics': [['points', 'polygons']],
      'recipe-points-in-polygons-choropleth': [['points', 'polygons']],
      'recipe-zonal-distribution': [['points', 'polygons']],
      'raster-join-points-in-polygons': [['points', 'polygons']]
    };
    for (const descriptor of ALL_DESCRIPTORS) {
      expect(descriptor.overlappingInputs ?? [], descriptor.bindingId).toEqual(
        pairs[descriptor.bindingId] ?? []
      );
    }
    for (const identifier of [
      'line-length-per-polygon',
      'recipe-points-in-polygons-choropleth',
      'recipe-zonal-distribution',
      'raster-join-points-in-polygons'
    ]) {
      expect(getDescriptor(identifier).outputs[0]).toMatchObject({
        kind: 'polygons',
        kindFrom: 'polygons'
      });
    }
  });
});

describe('aggregation disjoint input regressions', () => {
  it.each([
    'cell-compare',
    'dggs-cell-compare'
  ])('%s runs and reports appeared and disappeared cells', identifier => {
    const inputs = {
      before: makeCollection([makePoint([-75.2, 40.1]), makePoint([-75.19, 40.11])]),
      after: makeCollection([makePoint([-73.4, 41.2]), makePoint([-73.39, 41.21])])
    };
    const overrides: Values = identifier === 'cell-compare' ? {cellSize: 0.1} : {resolution: 8};
    const assessment = evaluateKeplerOperationApplicability(
      getBinding(identifier),
      getDescriptor(identifier),
      Object.entries(inputs).map(([name, features]) =>
        profileKeplerFeatureCollection(features, {
          datasetId: name,
          label: name,
          coordinateSystem: 'EPSG:4326'
        })
      ),
      {overrides, inputAssignments: {before: 'before', after: 'after'}, executionEngine: 'cpu'}
    );
    expect(assessment.status).toBe('ready');
    const result = runCatalogOperation(getBinding(identifier), assessment.parameters, inputs);
    expect(result.tables[0].columns['status'].slice().sort()).toEqual(['appeared', 'disappeared']);
    expect(
      result.tables[0].columns[identifier === 'cell-compare' ? 'change' : 'delta'].slice().sort()
    ).toEqual([-2, 2]);
  });

  it.each([
    ['line-length-per-polygon', 'line_count'],
    ['zonal-point-statistics', 'count'],
    ['recipe-points-in-polygons-choropleth', 'point_count'],
    ['recipe-zonal-distribution', 'row_count']
  ])('%s blocks disjoint inputs with zero memberships', (identifier, countColumn) => {
    const polygons = makeCollection([makeSquare(420000, 5100000, 10)]);
    const inputs =
      identifier === 'line-length-per-polygon'
        ? {
            polygons,
            lines: makeCollection([
              makeLine('outside', [
                [420040, 5100000],
                [420050, 5100000]
              ])
            ])
          }
        : {polygons, points: makeCollection([makePoint([420040, 5100005], {value: 7})])};
    const assessment = evaluateKeplerOperationApplicability(
      getBinding(identifier),
      getDescriptor(identifier),
      Object.entries(inputs).map(([name, features]) =>
        profileKeplerFeatureCollection(features, {
          datasetId: name,
          label: name,
          coordinateSystem: 'planar'
        })
      ),
      {
        fieldBindings: identifier === 'line-length-per-polygon' ? {} : {valueProperty: 'value'},
        inputAssignments: Object.fromEntries(Object.keys(inputs).map(name => [name, name])),
        executionEngine: 'cpu'
      }
    );
    expect(assessment.status).toBe('blocked');
    expect(assessment.missing.some(item => item.what.includes('do not overlap'))).toBe(true);
    const result = runCatalogOperation(getBinding(identifier), assessment.parameters, inputs);
    expect(result.tables[0].columns[countColumn]).toEqual([0]);
  });
});
