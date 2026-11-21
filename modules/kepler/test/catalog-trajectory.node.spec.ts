// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {
  KEPLER_CATALOG_BINDINGS,
  getCatalogDefaultParameters,
  getCatalogBinding,
  listCatalogEntries,
  normalizeCatalogParameters,
  runCatalogOperation,
  serializeCatalogInvocation,
  parseCatalogInvocation
} from '../src/analysis/catalog-bindings';
import {TRAJECTORY_BINDINGS} from '../src/analysis/catalog-trajectory-bindings';
import {
  prepareCatalogTrajectory,
  MAXIMUM_TRAJECTORY_FIXES,
  MAXIMUM_TRAJECTORY_TRACKS
} from '../src/analysis/catalog-trajectory-common';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {
  prepareTrajectoryHomeRange,
  makeTrajectoryHomeRange
} from '../src/analysis/catalog-trajectory-cpu';
import {resolveKeplerOperationParameters} from '../src/descriptors/default-parameters';
import {resolveKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {getWorkflowOperationRegistry} from '../src/workflows/workflow-operations';
import {normalizeWorkflowParameters} from '../src/workflows/workflow-parameters';
import {planWorkflow} from '../src/workflows/workflow-plan';

test('a singleton bridge keeps the requested zero raster and empty evidence through home-range publication', () => {
  const tracks = observations([[7, 9]]);
  const values = {
    ...getCatalogDefaultParameters(getCatalogBinding('trajectory-brownian-bridge')!),
    coordinateSystem: 'planar',
    gridWidth: 2,
    gridHeight: 3,
    padding: 1
  };
  const trajectory = prepareCatalogTrajectory(tracks, values);
  const grid = prepareTrajectoryHomeRange(trajectory, values);
  // With no consecutive pair there are no positive-duration bridges: all six cells are zero.
  expect(grid.gridSize).toEqual([2, 3]);
  expect(grid.extent).toEqual([-1, -1, 1, 1]);
  expect(Array.from(grid.density)).toEqual([0, 0, 0, 0, 0, 0]);
  const bridge = run('trajectory-brownian-bridge', tracks, values);
  expect(bridge.tables[0].columns['density']).toEqual([0, 0, 0, 0, 0, 0]);
  expect(bridge.summary['empty']).toBe(1);
  // Isopleths of a zero-mass raster have volume=1, inside=0 at every cell.
  const isopleth = makeTrajectoryHomeRange(
    trajectory,
    grid,
    new Float32Array(6).fill(1),
    new Uint32Array(6)
  );
  expect(isopleth.tables[0].rowCount).toBe(0);
  expect(isopleth.summary['empty']).toBe(1);
  expect(isopleth.summary['gridWidth']).toBe(2);
  expect(isopleth.summary['gridHeight']).toBe(3);
  expect(() => prepareTrajectoryHomeRange(trajectory, {...values, padding: 0})).toThrow(
    /extent.*max > min/
  );
});

test('home-range publication detects positive density before isopleth filtering without rescanning (S3)', () => {
  const values = {
    ...getCatalogDefaultParameters(getCatalogBinding('trajectory-brownian-bridge')!),
    coordinateSystem: 'planar',
    gridWidth: 2,
    gridHeight: 2,
    padding: 1
  };
  const trajectory = prepareCatalogTrajectory(observations([[0, 0]]), values);
  const grid = {
    ...prepareTrajectoryHomeRange(trajectory, values),
    density: Float32Array.of(0, 2, 0, 0)
  };
  const rescan = vi.spyOn(grid.density, 'some').mockImplementation(() => {
    throw new Error('second serial raster scan');
  });
  try {
    const result = makeTrajectoryHomeRange(
      trajectory,
      grid,
      new Float32Array(4).fill(1),
      new Uint32Array(4)
    );
    expect(result.tables[0].rowCount).toBe(0);
    expect(result.summary['empty']).toBe(0);
    expect(rescan).not.toHaveBeenCalled();
  } finally {
    rescan.mockRestore();
  }
});

const ORIGIN = Date.UTC(2026, 0, 1);
function observations(
  positions: number[][],
  times = positions.map((_, index) => index * 10),
  trackId = 'animal-a'
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: positions.map((position, index) => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {
        track_id: trackId,
        timestamp: ORIGIN + times[index] * 1000,
        value: index < 2 ? 'a' : 'b'
      }
    }))
  };
}
const MOVEMENT = observations([
  [0, 0],
  [0, 0],
  [0, 0],
  [10, 0],
  [20, 0],
  [30, 0],
  [30, 0],
  [30, 0]
]);
function run(id: string, tracks = MOVEMENT, parameters: Record<string, unknown> = {}) {
  return runCatalogOperation(
    getCatalogBinding(id)!,
    {coordinateSystem: 'planar', ...parameters},
    {tracks}
  );
}

const STAY_PARAMETERS = {distanceThreshold: 1, timeThreshold: 15, includeLast: true};

test('all thirteen trajectory contributors are registered once, discoverable and serializable', () => {
  expect(TRAJECTORY_BINDINGS).toHaveLength(13);
  expect(new Set(KEPLER_CATALOG_BINDINGS.map(binding => binding.id)).size).toBe(
    KEPLER_CATALOG_BINDINGS.length
  );
  const entries = listCatalogEntries();
  for (const binding of TRAJECTORY_BINDINGS) {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    const parameters = getCatalogDefaultParameters(binding);
    expect(normalizeCatalogParameters(binding, parameters)).toEqual(parameters);
    expect(
      parseCatalogInvocation(serializeCatalogInvocation(binding, parameters, {tracks: MOVEMENT}))
        .binding
    ).toBe(binding);
    for (const contributorName of binding.contributorNames)
      expect(
        entries.some(
          entry => entry.identity.exportName === contributorName && entry.runnable !== 'no'
        ),
        contributorName
      ).toBe(true);
  }
});

test('packing sorts interleaved tracks and preserves epoch milliseconds and source row identity', () => {
  const collection = observations(
    [
      [2, 0],
      [1, 0],
      [0, 0]
    ],
    [20, 10, 0]
  );
  collection.features.splice(
    1,
    0,
    ...observations(
      [
        [100, 0],
        [101, 0]
      ],
      [0, 1],
      'animal-b'
    ).features
  );
  const result = run('trajectory-thinning', collection, {mode: 'minimum-time', tolerance: 0});
  const fixes = result.tables.find(table => table.id === 'trajectory-fixes')!;
  expect(fixes.columns['source_index']).toEqual([4, 3, 0, 1, 2]);
  expect(fixes.columns['timestamp']).toEqual([
    ORIGIN,
    ORIGIN + 10000,
    ORIGIN + 20000,
    ORIGIN,
    ORIGIN + 1000
  ]);
  const path = JSON.parse(result.tables[0].columns['geometry'][0] as string);
  expect(path.coordinates).toEqual([
    [0, 0, 0, ORIGIN],
    [1, 0, 0, ORIGIN + 10000],
    [2, 0, 0, ORIGIN + 20000]
  ]);
});

test('timed LineStrings and Point observations yield identical paths, with separate vertex identities', () => {
  const points = observations([
    [0, 0],
    [10, 0],
    [20, 0]
  ]);
  const tracks: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'animal-a',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0, 0, ORIGIN],
            [10, 0, 0, ORIGIN + 10000],
            [20, 0, 0, ORIGIN + 20000]
          ]
        }
      }
    ]
  };
  const pointResult = run('trajectory-thinning', points, {tolerance: 0});
  const tripResult = run('trajectory-thinning', tracks, {tolerance: 0});
  expect(tripResult.tables[0].columns['geometry']).toEqual(
    pointResult.tables[0].columns['geometry']
  );
  expect(tripResult.tables[1].columns['source_index']).toEqual([0, 0, 0]);
  expect(tripResult.tables[1].columns['vertex_index']).toEqual([0, 1, 2]);
});

test('time encoding converts seconds and ISO input without epoch Float32 truncation', () => {
  for (const timestampEncoding of ['unix-second', 'iso8601']) {
    const tracks = observations(
      [
        [0, 0],
        [1, 0]
      ],
      [0, 0.001]
    );
    tracks.features.forEach(feature => {
      const milliseconds = feature.properties!['timestamp'] as number;
      feature.properties!['timestamp'] =
        timestampEncoding === 'iso8601'
          ? new Date(milliseconds).toISOString()
          : milliseconds / 1000;
    });
    expect(
      run('trajectory-thinning', tracks, {timestampEncoding, mode: 'minimum-time', tolerance: 0})
        .tables[1].columns['timestamp']
    ).toEqual([ORIGIN, ORIGIN + 1]);
  }
});

// Both accepted layouts share timestamp reading: Point properties and timed-line arrays.
function isoObservations(
  timestamps: readonly string[],
  representation: 'Point' | 'LineString'
): CatalogFeatureCollection {
  const coordinates = timestamps.map((_, index) => [index, 0]);
  if (representation === 'LineString')
    return {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {type: 'LineString', coordinates},
          properties: {track_id: 'one', timestamp: timestamps}
        }
      ]
    };
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {track_id: 'one', timestamp: timestamps[index]}
    }))
  };
}

for (const representation of ['Point', 'LineString'] as const) {
  test(`${representation} ISO timestamps use UTC for missing zones and retain explicit offsets`, () => {
    // Also run with TZ=America/New_York: Date.parse would move the unzoned fix by five hours.
    const tracks = isoObservations(
      ['2026-01-01T00:00:01Z', '2026-01-01T00:00:00', '2025-12-31T19:00:02-05:00'],
      representation
    );
    const prepared = prepareCatalogTrajectory(tracks, {
      ...getCatalogDefaultParameters(getCatalogBinding('trajectory-thinning')!),
      coordinateSystem: 'planar',
      timestampEncoding: 'iso8601'
    });
    expect(prepared.fixes.map(fix => fix.timestamp)).toEqual([
      ORIGIN,
      ORIGIN + 1000,
      ORIGIN + 2000
    ]);
    expect(
      prepared.fixes.map(fix => (representation === 'Point' ? fix.sourceIndex : fix.vertexIndex))
    ).toEqual([1, 0, 2]);
    const published = run('trajectory-thinning', tracks, {
      timestampEncoding: 'iso8601',
      mode: 'minimum-time',
      tolerance: 0
    }).tables[1];
    expect(published.columns['timestamp']).toEqual([ORIGIN, ORIGIN + 1000, ORIGIN + 2000]);
  });

  test(`${representation} ISO timestamps reject impossible dates and times before output`, () => {
    for (const timestamp of [
      '2026-02-30T00:00:00Z',
      '2026-02-29T00:00:00Z',
      '2026-01-01T24:00:00Z',
      '2026-02-30T00:00:00',
      '2026-02-30T00:00:00+05:30'
    ])
      expect(
        () =>
          run('trajectory-thinning', isoObservations([timestamp], representation), {
            timestampEncoding: 'iso8601'
          }),
        timestamp
      ).toThrow(/timestamps/);
    expect(
      run('trajectory-thinning', isoObservations(['2024-02-29T01:30:00+01:30'], representation), {
        timestampEncoding: 'iso8601'
      }).tables[1].columns['timestamp']
    ).toEqual([Date.UTC(2024, 1, 29)]);
  });

  test(`${representation} ISO timestamps reject nonzero sub-millisecond digits without losing exact milliseconds`, () => {
    for (const origin of ['1970-01-01T00:00:00', '2026-01-01T00:00:00']) {
      for (const fraction of ['0001', '1234', '1230000001', '000000000000000001'])
        expect(
          () =>
            run(
              'trajectory-thinning',
              isoObservations([`${origin}.${fraction}Z`], representation),
              {timestampEncoding: 'iso8601'}
            ),
          fraction
        ).toThrow(/sub-millisecond/);
      expect(
        run('trajectory-thinning', isoObservations([`${origin}.1230000000Z`], representation), {
          timestampEncoding: 'iso8601'
        }).tables[1].columns['timestamp']
      ).toEqual([origin.startsWith('1970') ? 123 : ORIGIN + 123]);
    }
  });
}

test('gap and value-change splitting obey strict gaps and repeat only value boundaries', () => {
  const tracks = observations(
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0]
    ],
    [0, 1, 10, 11]
  );
  const gaps = run('trajectory-split', tracks, {maximumGap: 2});
  expect(gaps.summary['segmentCount']).toBe(2);
  expect(gaps.tables[0].columns['source_indices']).toEqual(['[0,1]', '[2,3]']);
  const changes = run('trajectory-split', tracks, {mode: 'value-change', valueProperty: 'value'});
  expect(changes.tables[0].columns['source_indices']).toEqual(['[0,1,2]', '[2,3]']);
  expect(changes.tables[1].columns['fix_index']).toEqual([0, 1, 2, 2, 3]);
  expect(changes.tables[1].columns['output_index']).toEqual([0, 1, 2, 3, 4]);
  expect(run('trajectory-split', tracks, {maximumGap: 9}).summary['segmentCount']).toBe(1);
});

test('minimum distance/time thinning keeps endpoints independently for every track', () => {
  const tracks = observations([
    [0, 0],
    [1, 0],
    [2, 0],
    [10, 0]
  ]);
  expect(
    run('trajectory-thinning', tracks, {tolerance: 5}).tables[1].columns['source_index']
  ).toEqual([0, 3]);
  expect(
    run('trajectory-thinning', tracks, {mode: 'minimum-time', tolerance: 15}).tables[1].columns[
      'source_index'
    ]
  ).toEqual([0, 2, 3]);
});

test('staypoints and location clustering publish native points with correct times and centers', () => {
  const stays = run('trajectory-staypoints', MOVEMENT, STAY_PARAMETERS).tables[0];
  expect(stays.rowCount).toBe(2);
  expect(stays.columns['duration_seconds']).toEqual([30, 20]);
  expect(stays.columns['start_time']).toEqual([ORIGIN, ORIGIN + 50000]);
  const locations = run('trajectory-locations', MOVEMENT, {
    ...STAY_PARAMETERS,
    epsilon: 100
  }).tables;
  expect(locations[0].rowCount).toBe(1);
  expect(locations[0].columns['staypoint_count']).toEqual([2]);
  expect(JSON.parse(locations[0].columns['geometry'][0] as string).coordinates).toEqual([15, 0]);
  const separate = run('trajectory-locations', MOVEMENT, {...STAY_PARAMETERS, epsilon: 1})
    .tables[0];
  expect(separate.rowCount).toBe(2);
  expect(
    separate.columns['geometry'].map(value => JSON.parse(value as string).coordinates)
  ).toEqual([
    [0, 0],
    [30, 0]
  ]);
});

test('segments publish moving triplegs, trips and source-linked stays', () => {
  const result = run('trajectory-segments', MOVEMENT, {
    ...STAY_PARAMETERS,
    activityMinimumDuration: 15
  });
  expect(result.summary['triplegCount']).toBe(1);
  expect(result.tables.map(table => table.id)).toEqual([
    'trajectory-tracks',
    'trajectory-trips',
    'trajectory-staypoints'
  ]);
  // The staypoint mask owns row 5, so the moving interval ends at row 4.
  expect(result.tables[0].columns['source_indices']).toEqual(['[3,4]']);
  expect(result.tables[1].rowCount).toBe(1);
});

test('stop split uses the existing staypoint masks and excludes stopped fixes', () => {
  const result = run('trajectory-split', MOVEMENT, {...STAY_PARAMETERS, mode: 'stop'});
  expect(result.tables[0].columns['source_indices']).toEqual(['[3,4]']);
});

test('mobility and modal split expose the library metrics with second-based UI units', () => {
  const mobility = run('trajectory-mobility-indicators', MOVEMENT, {
    ...STAY_PARAMETERS,
    epsilon: 1
  });
  const columns = mobility.tables[0].columns;
  expect(columns['radius_of_gyration']).toEqual([15]);
  expect(columns['random_entropy']).toEqual([1]);
  expect(columns['home_location']).toEqual([0]);
  expect(columns['work_location']).toEqual([1]);
  expect(mobility.tables[1].columns['jump_length']).toEqual([0, 30]);
  const modal = run('trajectory-modal-split', MOVEMENT, {
    ...STAY_PARAMETERS,
    metric: 'duration',
    modeSpeedUpperBoundsFloat32: [0.5, 2, 50]
  });
  expect(modal.tables[0].columns['value']).toEqual([0, 10, 0]);
  expect(modal.tables[1].columns['mode']).toEqual([1]);
  const binding = getCatalogBinding('trajectory-modal-split')!;
  for (const candidate of [binding, {parameters: binding.parameters}]) {
    expect(() => normalizeCatalogParameters(candidate, {modeSpeedBreaks: [1, 8, 50]})).toThrow(
      /modeSpeedBreaks was removed.*modeSpeedUpperBoundsFloat32/
    );
  }
  expect(() =>
    run('trajectory-modal-split', MOVEMENT, {modeSpeedUpperBoundsFloat32: [2, 1]})
  ).toThrow(/ascending/);
});

test('modal split rejects retired speed bounds in defaults and workflow validation', () => {
  const operationId = 'trajectory-modal-split';
  const binding = getCatalogBinding(operationId)!;
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const registry = getWorkflowOperationRegistry();
  const operation = registry.get(operationId)!;
  const message = 'modeSpeedBreaks was removed (use modeSpeedUpperBoundsFloat32)';
  const overrides = {modeSpeedBreaks: [2, 4]};
  expect(() => resolveKeplerOperationParameters(binding, descriptor, {}, {overrides})).toThrow(
    message
  );
  expect(() =>
    normalizeWorkflowParameters(operation.parameters, overrides, {operationTitle: operation.title})
  ).toThrow(message);
  const plan = planWorkflow(
    {
      schemaVersion: 1,
      id: 'modal-split',
      title: 'Modal split',
      inputs: [],
      steps: [
        {
          id: 'modes',
          operationId,
          inputs: {},
          parameters: {modeSpeedBreaks: {from: 'static', value: [2, 4]}},
          position: {x: 0, y: 0}
        }
      ],
      settings: {engine: 'cpu', autoRun: false}
    },
    {registry}
  );
  expect(plan.steps['modes'].issues).toContainEqual(
    expect.objectContaining({
      code: 'unknown-parameter',
      parameter: 'modeSpeedBreaks',
      message: `${operation.title}: ${message}`
    })
  );
});

test('recursions retain radius/row identity and expose censored first-passage times as null', () => {
  const tracks = observations(
    [
      [0, 0],
      [10, 0],
      [0, 0]
    ],
    [0, 10, 20]
  );
  const table = run('trajectory-recursions', tracks, {radii: [1, 20]}).tables[0];
  expect(table.rowCount).toBe(6);
  expect(table.columns['radius']).toEqual([1, 20, 1, 20, 1, 20]);
  expect(table.columns['revisits'][0]).toBe(1);
  expect(table.columns['first_passage_seconds'][1]).toBeNull();
  expect(table.columns['first_passage_censored'][1]).toBe(true);
});

test('Brownian bridge density has correct Gaussian normalization and rectangular cell geometry', () => {
  const tracks = observations(
    [
      [0, 0],
      [0, 0]
    ],
    [0, 1]
  );
  const table = run('trajectory-brownian-bridge', tracks, {
    gridWidth: 2,
    gridHeight: 2,
    padding: 1,
    motionVariance: 0,
    locationError: 1,
    subStepCount: 1
  }).tables[0];
  const expected = Math.exp(-0.5) / Math.PI;
  expect(table.rowCount).toBe(4);
  for (const density of table.columns['density']) expect(density).toBeCloseTo(expected, 7);
  expect(JSON.parse(table.columns['geometry'][0] as string)).toEqual({
    type: 'Polygon',
    coordinates: [
      [
        [-1, -1],
        [0, -1],
        [0, 0],
        [-1, 0],
        [-1, -1]
      ]
    ]
  });
});

test('invalid, duplicate, sub-millisecond and oversized trajectories fail before partial output', () => {
  expect(() =>
    run(
      'trajectory-thinning',
      observations(
        [
          [0, 0],
          [1, 0]
        ],
        [0, 0]
      )
    )
  ).toThrow(/distinct/);
  expect(() => run('trajectory-thinning', observations([[NaN, 0]]))).toThrow(/coordinates/);
  expect(() => run('trajectory-thinning', observations([[0, 0]], [0.0005]))).toThrow(
    /sub-millisecond/
  );
  expect(() =>
    run('trajectory-thinning', observations(Array.from({length: 2049}, () => [0, 0])))
  ).toThrow(/capacity/i);
  const malformed = observations([[0, 0]]);
  delete malformed.features[0].properties!['timestamp'];
  expect(() => run('trajectory-thinning', malformed)).toThrow(/timestamps/);
  expect(() =>
    run('trajectory-thinning', observations([[0, 91]]), {coordinateSystem: 'EPSG:4326'})
  ).toThrow(
    'trajectory-thinning: coordinateSystem EPSG:4326 requires longitude in [-180, 180] and latitude in [-90, 90]'
  );
});

test('empty input and singleton paths have explicit behavior', () => {
  expect(() => run('trajectory-thinning', {type: 'FeatureCollection', features: []})).toThrow(
    /Input required/
  );
  const singleton = run('trajectory-thinning', observations([[0, 0]]));
  expect(singleton.tables[0].rowCount).toBe(0);
  expect(singleton.tables[1].rowCount).toBe(1);
});

test('packing preserves signed pre-epoch timestamps as Int64 words', () => {
  const tracks = observations(
    [
      [0, 0],
      [1, 0]
    ],
    [0, 1]
  );
  tracks.features.forEach((feature, index) => {
    feature.properties!['timestamp'] = -1000 + index;
  });
  const binding = getCatalogBinding('trajectory-thinning')!;
  const prepared = prepareCatalogTrajectory(tracks, {
    ...getCatalogDefaultParameters(binding),
    coordinateSystem: 'planar'
  });
  expect(Array.from(prepared.timestamps)).toEqual([4294966296, 4294967295, 4294966297, 4294967295]);
});

test('trajectory admission accepts the fix and track limits and rejects one beyond each before packing', () => {
  const values = {
    ...getCatalogDefaultParameters(getCatalogBinding('trajectory-thinning')!),
    coordinateSystem: 'planar'
  };
  const maximumFixes = observations(Array.from({length: MAXIMUM_TRAJECTORY_FIXES}, () => [0, 0]));
  expect(prepareCatalogTrajectory(maximumFixes, values).fixes).toHaveLength(2048);
  const tracks = observations(Array.from({length: MAXIMUM_TRAJECTORY_TRACKS}, () => [0, 0]));
  tracks.features.forEach((feature, index) => {
    feature.properties!['track_id'] = `track-${index}`;
  });
  expect(prepareCatalogTrajectory(tracks, values).trackIds).toHaveLength(128);
  tracks.features.push(...observations([[0, 0]], [0], 'track-overflow').features);
  // Coordinate access would throw if admission happened after per-fix packing.
  for (const feature of tracks.features)
    Object.defineProperty(feature.geometry, 'coordinates', {
      get() {
        throw new Error('packed before admission');
      }
    });
  expect(() => prepareCatalogTrajectory(tracks, values)).toThrow(
    'catalog-trajectory trackCount must be an integer in [1, 128]'
  );
  const tooManyFixes = {
    type: 'FeatureCollection' as const,
    features: Array.from({length: MAXIMUM_TRAJECTORY_FIXES + 1}, () => tracks.features[0])
  };
  expect(() => prepareCatalogTrajectory(tooManyFixes, values)).toThrow(
    'catalog-trajectory fixCount must be an integer in [1, 2048]'
  );
});
