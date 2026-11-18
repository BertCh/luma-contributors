// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {readCatalogFeaturesFromNativeDataset} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {computeHealpixPointToCellOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {NETWORK_CATEGORICAL_BINDINGS} from '../src/analysis/catalog-network-categorical-bindings';
import {getKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {planKeplerResultPresentation} from '../src/presentation/result-presentation';
import {getNetworkCategoricalSampleInputs} from '../../../examples/deck/kepler-compute/src/analysis-catalog-network-categorical-samples';

type Inputs = Record<string, CatalogFeatureCollection>;

function makePoints(
  coordinates: number[][],
  properties: Record<string, unknown>[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      id: `source-${index}`,
      geometry: {type: 'Point', coordinates: position},
      properties: properties[index]
    }))
  };
}

async function run(id: string, values: Record<string, unknown>, inputs: Inputs) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runCatalogOperationAsync(getCatalogBinding(id)!, values, inputs, {device});
  expect(result.engine).toBe('gpu');
  expect(result.dispatchedContributors).toContain(getCatalogBinding(id)!.gpuContributorNames[0]);
  expect(result.fallbackReason).toBeUndefined();
  return result;
}

// Exercise defaults through exactly the fixtures loaded by the UI, without CPU fallback.
for (const binding of NETWORK_CATEGORICAL_BINDINGS) {
  test(`${binding.id} runs its UI sample on real GPU contributors`, async () => {
    const samples = getNetworkCategoricalSampleInputs(binding.id)!;
    const result = await run(
      binding.id,
      {},
      Object.fromEntries(Object.entries(samples).map(([name, text]) => [name, JSON.parse(text)]))
    );
    expect(result.tables[0].rowCount).toBeGreaterThan(0);
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    for (const output of descriptor.outputs) {
      const table = result.tables.find(candidate => candidate.id === output.tableId);
      if (!table) throw new Error(`${binding.id}: missing descriptor output ${output.tableId}`);
      if (output.presentation) {
        const column = output.presentation.column;
        if (!(column in table.columns)) {
          throw new Error(
            `${binding.id}.${table.id}: presentation column ${column} absent; columns=${Object.keys(table.columns).join(', ')}`
          );
        }
        const plan = planKeplerResultPresentation(table, output);
        if (!plan || plan.column !== column || plan.missingCount === table.rowCount) {
          throw new Error(
            `${binding.id}.${table.id}: unusable presentation ${JSON.stringify({output, plan, values: table.columns[column]})}`
          );
        }
      }
    }

    expect(
      result.tables.every(table =>
        Object.values(table.columns).every(column => column.length === table.rowCount)
      )
    ).toBe(true);
  });
}

const CATEGORY_POINTS = makePoints(
  [
    [0, 0],
    [1, 0],
    [3, 0],
    [4, 0]
  ],
  [
    {category: 'A', value: 0},
    {category: 'A', value: 2},
    {category: 'B', value: 8},
    {category: 'B', value: 10}
  ]
);
const CATEGORY_PARAMETERS = {coordinateSystem: 'planar', radius: 1.1};

test('category pair counts, entropy, CLQ and LICD match hand-computed disconnected category pairs', async () => {
  const pairs = await run('category-pair-counts', CATEGORY_PARAMETERS, {points: CATEGORY_POINTS});
  expect(pairs.tables[0].columns['value']).toEqual([2, 0, 0, 2]);
  expect(pairs.tables[0].columns['origin_rows']).toEqual([2, 2, 2, 2]);
  const entropy = await run('spatial-entropy', CATEGORY_PARAMETERS, {points: CATEGORY_POINTS});
  const statistics = Object.fromEntries(
    entropy.tables[0].columns['statistic'].map((name, index) => [
      String(name),
      entropy.tables[0].columns['value'][index]
    ])
  );
  expect(statistics['shannon']).toBeCloseTo(Math.log(2), 5);
  expect(statistics['cooccurrence']).toBeCloseTo(Math.log(2), 5);
  expect(statistics['mutualInformation']).toBeCloseTo(Math.log(2), 5);
  expect(statistics['relativeMutualInformation']).toBeCloseTo(1, 5);
  expect(statistics['linkCount']).toBe(4);
  const quotient = await run('colocation-quotient', CATEGORY_PARAMETERS, {points: CATEGORY_POINTS});
  expect(quotient.tables[0].columns['value'][0]).toBeCloseTo(3, 5);
  expect(quotient.tables[0].columns['value'][3]).toBeCloseTo(3, 5);
  expect(quotient.tables[0].columns['value'][1]).toBe(0);
  const local = await run('local-category-dependence', CATEGORY_PARAMETERS, {
    points: CATEGORY_POINTS
  });
  expect(local.tables[0].columns['source_id']).toEqual([
    'source-0',
    'source-1',
    'source-2',
    'source-3'
  ]);
  expect(local.tables[0].columns['likeCount']).toEqual([1, 1, 1, 1]);
  for (const probability of local.tables[0].columns['upperTailProbability'])
    expect(probability).toBeCloseTo(0.5, 5);
});

test('categorical missing rows and islands publish null statistics without shifting identities', async () => {
  const points = makePoints(
    [
      [0, 0],
      [1, 0],
      [100, 0]
    ],
    [{category: 'A'}, {category: null}, {category: 'B'}]
  );
  const result = await run('local-category-dependence', CATEGORY_PARAMETERS, {points});
  expect(result.tables[0].columns['source_index']).toEqual([0, 1, 2]);
  expect(result.tables[0].columns['neighborCount']).toEqual([0, 0, 0]);
  expect(result.tables[0].columns['proportion']).toEqual([null, null, null]);
});

test('factor detector retains q and F when a four-row result exceeds the p-value range', async () => {
  const points = makePoints(
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0]
    ],
    [
      {category: 'A', value: 1000},
      {category: 'B', value: 1001},
      {category: 'B', value: 1002},
      {category: 'B', value: 1003}
    ]
  );
  const detector = await run('geographical-detector', {}, {points});
  const table = detector.tables[0];
  const valueFor = (statistic: string) =>
    table.columns['value'][table.columns['statistic'].indexOf(statistic)];
  expect(valueFor('q')).toBeCloseTo(0.6, 5);
  expect(valueFor('fStatistic')).toBeCloseTo(3, 4);
  expect(valueFor('pValue')).toBeNull();
  expect(detector.summary['pValueStatus']).toBe('unavailable');
  expect(detector.diagnostics).toContainEqual({
    code: 'geographical-detector-p-value-unavailable',
    severity: 'warning',
    count: 1,
    message: expect.stringMatching(/nonCentrality.*10000/)
  });
});

test('factor detector q and single-linkage regions match explicit partitions', async () => {
  const detector = await run('geographical-detector', {}, {points: CATEGORY_POINTS});
  const qIndex = detector.tables[0].columns['statistic'].indexOf('q');
  expect(detector.tables[0].columns['value'][qIndex]).toBeCloseTo(16 / 17, 5);
  const regions = await run(
    'redcap-single-linkage',
    {...CATEGORY_PARAMETERS, radius: 10, regionCount: 2},
    {points: CATEGORY_POINTS}
  );
  const labels = regions.tables[0].columns['region'];
  expect(labels[0]).toBe(labels[1]);
  expect(labels[2]).toBe(labels[3]);
  expect(labels[0]).not.toBe(labels[2]);
  expect(regions.summary['converged']).toBe(1);
  expect(regions.summary['regionCount']).toBe(2);
});

test('zero-neighborhood smoothing reproduces the unique OD matrix and radiation matches the definition', async () => {
  const zones = makePoints(
    [
      [0, 0],
      [1, 0],
      [3, 0]
    ],
    [
      {population: 10, outflow: 6},
      {population: 20, outflow: 4},
      {population: 30, outflow: 2}
    ]
  );
  const flows: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [1, 0]
          ]
        },
        properties: {origin: 'source-0', destination: 'source-1', weight: 5}
      }
    ]
  };
  const smoothed = await run(
    'kernel-flow-smoothing',
    {coordinateSystem: 'planar', neighborhoodSize: 0},
    {zones, flows}
  );
  expect(smoothed.tables[0].columns['flow']).toEqual([0, 5, 0, 0, 0, 0, 0, 0, 0]);
  expect(smoothed.tables[0].columns['origin_id'][1]).toBe('source-0');
  const modeled = await run('radiation-flow-model', {coordinateSystem: 'planar'}, {zones});
  const populations = [10, 20, 30];
  const outflows = [6, 4, 2];
  const positions = [0, 1, 3];
  for (let origin = 0; origin < 3; origin++)
    for (let destination = 0; destination < 3; destination++) {
      const intervening = populations.reduce(
        (sum, population, other) =>
          sum +
          (other !== origin &&
          other !== destination &&
          Math.abs(positions[other] - positions[origin]) <
            Math.abs(positions[destination] - positions[origin])
            ? population
            : 0),
        0
      );
      const expected =
        origin === destination
          ? 0
          : (outflows[origin] * populations[origin] * populations[destination]) /
            ((populations[origin] + intervening) *
              (populations[origin] + populations[destination] + intervening));
      expect(modeled.tables[0].columns['flow'][origin * 3 + destination]).toBeCloseTo(expected, 5);
    }
});

test('edge volumes count repeated traversals across breaks and publish unused edges', async () => {
  const edges: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [0, 1, 2].map(index => ({
      id: `edge-${index}`,
      geometry: {
        type: 'LineString',
        coordinates: [
          [index, 0],
          [index + 1, 0]
        ]
      }
    }))
  };
  const nativeTracks = new arrow.Table({
    geometry: arrow.vectorFromArray(
      [
        JSON.stringify({
          type: 'LineString',
          coordinates: [
            [0, 0],
            [2, 0]
          ]
        })
      ],
      new arrow.Utf8()
    ),
    traversedEdges: arrow.vectorFromArray(
      [[0, 1, null, 0, 1]],
      new arrow.List(new arrow.Field('item', new arrow.Uint32(), true))
    )
  });
  const tracks = readCatalogFeaturesFromNativeDataset(
    {
      fields: nativeTracks.schema.fields,
      dataContainer: {
        numRows: () => nativeTracks.numRows,
        valueAt: (row, column) => nativeTracks.getChildAt(column)!.get(row)
      }
    },
    'geometry'
  );
  const result = await run('map-matched-edge-volumes', {}, {edges, tracks});
  expect(result.tables[0].columns['volume']).toEqual([2, 2, 0]);
  expect(result.tables[0].columns['source_id']).toEqual(['edge-0', 'edge-1', 'edge-2']);
  expect(result.summary['candidateOverflow']).toBe(0);
});

const NETWORK: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      id: 'first',
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [1, 0]
        ]
      },
      properties: {slope: 1}
    },
    {
      id: 'second',
      geometry: {
        type: 'LineString',
        coordinates: [
          [1, 0],
          [3, 0]
        ]
      },
      properties: {slope: 0}
    },
    {
      id: 'island',
      geometry: {
        type: 'LineString',
        coordinates: [
          [10, 0],
          [11, 0]
        ]
      },
      properties: {slope: 0}
    }
  ]
};

test('network prism preserves directed source edges and excludes unreachable components', async () => {
  const result = await run(
    'network-potential-path-area',
    {coordinateSystem: 'planar', destinationNode: 2, budget: 3},
    {lines: NETWORK}
  );
  expect(result.summary['prismNodeCount']).toBe(3);
  expect(result.summary['prismEdgeCount']).toBe(2);
  expect(result.tables[0].columns['in_prism'].filter(Boolean)).toHaveLength(2);
  const sourceIds = result.tables[0].columns['source_id'];
  sourceIds.forEach((identity, index) => {
    if (identity === 'island') {
      expect(result.tables[0].columns['slack'][index]).toBeNull();
      expect(result.tables[0].columns['in_prism'][index]).toBe(false);
    }
  });
});

test('time-dependent percentiles use nearest rank and retain unreachable nodes', async () => {
  const result = await run(
    'network-travel-time-percentiles',
    {
      coordinateSystem: 'planar',
      slopeProperty: 'slope',
      departureCount: 3,
      firstDeparture: 0,
      departureStep: 1,
      percentiles: [50, 100]
    },
    {lines: NETWORK}
  );
  expect(result.tables[0].columns['travel_time']).toEqual([
    0,
    0,
    2,
    3,
    4,
    5,
    null,
    null,
    null,
    null
  ]);
  expect(result.tables[0].columns['reached_departures']).toEqual([3, 3, 3, 3, 3, 3, 0, 0, 0, 0]);
  expect(result.summary['departure0Converged']).toBe(1);
  const exhausted = await run(
    'network-travel-time-percentiles',
    {coordinateSystem: 'planar', departureCount: 1, maximumRelaxations: 0},
    {lines: NETWORK}
  );
  expect(exhausted.summary['departure0Converged']).toBe(0);
});

test('HEALPix keys match the library CPU twin and aggregation publishes real cell geometry', async () => {
  const points = makePoints(
    [
      [-74.003, 40.701],
      [-74.003, 40.701],
      [15.123, -30.456]
    ],
    [{value: 2}, {value: 3}, {value: 7}]
  );
  const indexed = await run('healpix-point-to-cell', {order: 4}, {points});
  const expected = computeHealpixPointToCellOnCPU(
    new Float32Array([-74.003, 40.701, -74.003, 40.701, 15.123, -30.456]),
    {order: 4}
  );
  expect(indexed.tables[0].columns['cell_id']).toEqual(
    Array.from({length: 3}, (_, index) =>
      (
        BigInt(expected.cells[index * 2]) |
        (BigInt(expected.cells[index * 2 + 1]) << 32n)
      ).toString()
    )
  );
  const aggregated = await run(
    'healpix-cell-aggregation',
    {order: 4, valueProperty: 'value'},
    {points}
  );
  expect(aggregated.tables[0].columns['count'].slice().sort()).toEqual([1, 2]);
  expect(aggregated.tables[0].columns['sum'].slice().sort()).toEqual([5, 7]);
  for (const text of aggregated.tables[0].columns['geometry']) {
    const geometry = JSON.parse(String(text));
    expect(geometry.type).toBe('Polygon');
    expect(geometry.coordinates[0]).toHaveLength(33);
  }
  await expect(
    run('healpix-cell-aggregation', {order: 4, cellCapacity: 1}, {points})
  ).rejects.toThrow('requiredCount=2');
  // A failed run releases the session; a subsequent run still returns complete results.
  expect(
    (await run('healpix-cell-aggregation', {order: 4, cellCapacity: 2}, {points})).tables[0]
      .rowCount
  ).toBe(2);
});

test('datum grids retain sub-Float32 shifts, invert them, and expose no-coverage/nodata statuses', async () => {
  const samples = makePoints(
    [
      [0, 0],
      [2, 0],
      [0, 2],
      [2, 2]
    ],
    Array.from({length: 4}, () => ({longitudeShift: 1e-7, latitudeShift: 2e-7, heightShift: 3}))
  );
  const points = makePoints(
    [
      [1, 1],
      [3, 3]
    ],
    [{height: 10}, {height: 20}]
  );
  const parameters = {
    originLongitude: 0,
    originLatitude: 0,
    spacingLongitude: 2,
    spacingLatitude: 2,
    operation: 'combined'
  };
  const result = await run('datum-grid-transform', parameters, {points, samples});
  expect(result.tables[0].columns['status']).toEqual([0, 2]);
  expect(result.tables[0].columns['longitude'][0]).toBeCloseTo(1.0000001, 10);
  expect(result.tables[0].columns['height'][0]).toBeCloseTo(13, 5);
  expect(result.tables[0].columns['geometry'][1]).toBeNull();
  const inversePoints = makePoints([[1.0000001, 1.0000002]], [{height: 13}]);
  const inverse = await run(
    'datum-grid-transform',
    {...parameters, direction: 'inverse'},
    {points: inversePoints, samples}
  );
  expect(inverse.tables[0].columns['longitude'][0]).toBeCloseTo(1, 10);
  expect(inverse.tables[0].columns['height'][0]).toBeCloseTo(10, 5);
  samples.features[0].properties = {valid: false};
  const nodata = await run('datum-grid-transform', parameters, {points, samples});
  expect(nodata.tables[0].columns['status']).toEqual([4, 2]);
});

test('Sinkhorn identical masses vanish; displaced point mass has squared distance cost', async () => {
  const cells = makePoints(
    [
      [0, 0],
      [1, 0]
    ],
    [
      {source: 1, target: 1},
      {source: 0, target: 0}
    ]
  );
  const identical = await run('sinkhorn-divergence', {width: 2, height: 1, cellWidth: 2}, {cells});
  expect(identical.tables[0].columns['divergence'][0]).toBeCloseTo(0, 5);
  cells.features[0].properties!['target'] = 0;
  cells.features[1].properties!['target'] = 1;
  const shifted = await run('sinkhorn-divergence', {width: 2, height: 1, cellWidth: 2}, {cells});
  expect(shifted.tables[0].columns['divergence'][0]).toBeCloseTo(4, 4);
  expect(shifted.tables[0].columns['converged'][0]).toBe(1);
  expect(shifted.tables[0].columns['approximation'][0]).toBe(0);
});

test('HEALPix cell zero remains valid through aggregation', async () => {
  const points = makePoints(
    [
      [45, 45],
      [45, 45]
    ],
    [{}, {}]
  );
  const indexed = await run('healpix-point-to-cell', {order: 0}, {points});
  expect(indexed.tables[0].columns['cell_id']).toEqual(['0', '0']);
  expect(indexed.tables[0].columns['validity']).toEqual([true, true]);
  const aggregated = await run('healpix-cell-aggregation', {order: 0}, {points});
  expect(aggregated.tables[0].columns['cell_id']).toEqual(['0']);
  expect(aggregated.tables[0].columns['count']).toEqual([2]);
});

test('datum inverse iteration exhaustion is visible and withholds output geometry', async () => {
  const points = makePoints([[1.1, 1]], [{}]);
  const samples = makePoints(
    [
      [0, 0],
      [2, 0],
      [0, 2],
      [2, 2]
    ],
    [0, 0.2, 0, 0.2].map(longitudeShift => ({longitudeShift, latitudeShift: 0, heightShift: 0}))
  );
  const result = await run(
    'datum-grid-transform',
    {
      originLongitude: 0,
      originLatitude: 0,
      spacingLongitude: 2,
      spacingLatitude: 2,
      direction: 'inverse',
      maximumIterations: 1
    },
    {points, samples}
  );
  expect(result.tables[0].columns['status']).toEqual([8]);
  expect(result.tables[0].columns['geometry']).toEqual([null]);
  expect(result.tables[0].columns['source_id']).toEqual(['source-0']);
});

test('Sinkhorn singleton grid is valid with a full minimum blur window', async () => {
  const cells = makePoints([[0, 0]], [{source: 1, target: 3}]);
  const result = await run('sinkhorn-divergence', {width: 1, height: 1}, {cells});
  expect(result.tables[0].columns['divergence'][0]).toBeCloseTo(0, 5);
  expect(result.tables[0].columns['converged'][0]).toBe(1);
});

test('time-dependent source parts reject new shared-vertex splits; static splits preserve arrival', async () => {
  const values = {
    coordinateSystem: 'planar',
    costMode: 'property',
    forwardCostProperty: 'cost',
    slopeProperty: 'slope',
    onewayProperty: 'oneway',
    departureCount: 1,
    percentiles: [100]
  };
  const lines: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [1, 0],
            [2, 0]
          ]
        },
        properties: {cost: 2, slope: 1, oneway: 'forward'}
      }
    ]
  };
  const original = await run('network-travel-time-percentiles', values, {lines});
  expect(original.tables[0].columns['travel_time'][2]).toBe(2);
  lines.features.push({
    geometry: {
      type: 'LineString',
      coordinates: [
        [1, 0],
        [1, 1]
      ]
    },
    properties: {cost: 1, slope: 0, oneway: 'forward'}
  });
  await expect(run('network-travel-time-percentiles', values, {lines})).rejects.toThrow(
    /nonzero.*slope.*split/
  );
  expect(
    (
      await run(
        'network-travel-time-percentiles',
        {...values, splitAtSharedVertices: false},
        {lines}
      )
    ).tables[0].columns['travel_time'][2]
  ).toBe(2);
  lines.features[0].properties!['slope'] = 0;
  const split = await run('network-travel-time-percentiles', values, {lines});
  expect(split.tables[0].columns['travel_time'][2]).toBe(2);
});

for (const operation of ['kernel-flow-smoothing', 'radiation-flow-model']) {
  test(`${operation} publishes singleton MultiPoints as flat precise arc endpoints`, async () => {
    const endpoints = [
      [0.123456789, 0.987654321],
      [1.123456789, 2.987654321]
    ];
    const zones: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: endpoints.map((position, index) => ({
        id: index,
        geometry: {
          type: index === 0 ? 'MultiPoint' : 'Point',
          coordinates: index === 0 ? [position] : position
        },
        properties: {population: 1, outflow: 1}
      }))
    };
    const flows: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {type: 'LineString', coordinates: endpoints},
          properties: {origin: 0, destination: 1, weight: 1}
        }
      ]
    };
    const result = await run(operation, {coordinateSystem: 'planar'}, {zones, flows});
    expect(JSON.parse(result.tables[0].columns['geometry'][1] as string)).toEqual({
      type: 'LineString',
      coordinates: endpoints
    });
  });
}
