// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {
  CatalogFeature,
  CatalogFeatureCollection,
  CatalogGeometry,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  computeGPUAnalysisSpacingStatistics,
  getGPUAnalysisExtentSizes
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {countCollectionVertices} from '../../src/analysis/catalog-geojson';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  applyKeplerDatasetProfileDeclarations,
  getKeplerDatasetProfileUndeclared,
  type KeplerDatasetProfileColumn,
  describeKeplerDatasetProfile,
  KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT,
  profileKeplerColumnTable,
  profileKeplerFeatureCollection,
  toKeplerRecommendationSubject
} from '../../src/profile/dataset-profile';

const BASE_OPTIONS = {datasetId: 'dataset-1', label: 'Dataset'};

function makePoint(longitude: number, latitude: number, properties: Record<string, unknown> = {}) {
  return {
    geometry: {type: 'Point', coordinates: [longitude, latitude]},
    properties
  } as CatalogFeature;
}

function collect(features: CatalogFeature[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features};
}

function getColumn(
  profile: {columns: readonly KeplerDatasetProfileColumn[]},
  name: string
): KeplerDatasetProfileColumn {
  const column = profile.columns.find(candidate => candidate.name === name);
  if (!column) throw new Error(`no column ${name}`);
  return column;
}

describe('profileKeplerFeatureCollection: curated samples', () => {
  it('profiles the city incidents as points with a time column', () => {
    const collection = createKeplerCuratedAnalysisSample('city-incidents')['points'];
    const profile = profileKeplerFeatureCollection(collection, BASE_OPTIONS);
    expect(profile.kind).toBe('points');
    expect(profile.catalogGeometry).toBe('point');
    expect(profile.rowCount).toBe(collection.features.length);
    expect(profile.vertexCount).toBe(collection.features.length);
    expect(profile.coordinates?.model).toBe('geographic');
    expect(profile.coordinates?.source).toBe('guess');
    expect(profile.spacing?.pointCount).toBe(collection.features.length);
    expect(profile.contiguity).toBeUndefined();
    // Severity is an ordinal level and the other column is a time: no measured quantity, so the
    // rows read as occurrences. The roles guess is always a low-confidence suggestion.
    expect(profile.rowSemantics).toMatchObject({
      value: 'events',
      source: 'guess',
      confidence: 'low'
    });
    expect(profile.rowSemantics?.reason).toBeTruthy();
    const severity = getColumn(profile, 'severity');
    expect(severity.type).toBe('integer');
    expect(severity.minimum).toBeGreaterThanOrEqual(1);
    expect(severity.maximum).toBeLessThanOrEqual(10);
    expect(severity.nullCount).toBe(0);
    expect(severity.measurement).toBe('ordinal');
    expect(severity.measurementSource).toBe('guess');
    const observed = getColumn(profile, 'observed');
    expect(observed.type).toBe('time');
    expect(observed.storage).toBe('string');
    // Timestamps run over 30 days from 2025-01-01.
    expect(observed.minimum).toBeGreaterThanOrEqual(Date.UTC(2025, 0, 1));
    expect(observed.maximum).toBeLessThanOrEqual(Date.UTC(2025, 0, 31, 1));
    expect(observed.measurement).toBeUndefined();
    expect(profile.geometryFeatureCounts).toEqual({
      point: collection.features.length,
      line: 0,
      polygon: 0,
      withoutGeometry: 0
    });
  });

  it('profiles the district rates as polygons with count, rate and identifier columns', () => {
    const collection = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const profile = profileKeplerFeatureCollection(collection, BASE_OPTIONS);
    expect(profile.kind).toBe('polygons');
    expect(profile.catalogGeometry).toBe('polygon');
    expect(profile.rowCount).toBe(144);
    expect(profile.vertexCount).toBe(countCollectionVertices(collection));
    expect(profile.vertexCount).toBe(144 * 5);
    expect(profile.coordinates?.model).toBe('geographic');
    // One centroid per polygon feeds the spacing.
    expect(profile.spacing?.pointCount).toBe(144);
    expect(getColumn(profile, 'cases').measurement).toBe('count');
    expect(getColumn(profile, 'population').measurement).toBe('count');
    expect(getColumn(profile, 'rate').measurement).toBe('rate');
    expect(getColumn(profile, 'district').measurement).toBe('identifier');
    expect(getColumn(profile, 'district').type).toBe('text');
    expect(getColumn(profile, 'district').distinctCount).toBe(144);
  });

  it('profiles the street grid as lines and the rain gauges as points', () => {
    const sample = createKeplerCuratedAnalysisSample('street-grid-facilities');
    const streets = sample['streets'] ?? Object.values(sample)[0];
    const profile = profileKeplerFeatureCollection(streets, BASE_OPTIONS);
    expect(profile.kind).toBe('lines');
    expect(profile.catalogGeometry).toBe('line');
    expect(profile.rowCount).toBe(2 * 11 * 10);
    expect(profile.vertexCount).toBe(2 * 11 * 10 * 2);
    const gauges = profileKeplerFeatureCollection(
      createKeplerCuratedAnalysisSample('rain-gauges')['points'],
      BASE_OPTIONS
    );
    expect(gauges.kind).toBe('points');
    expect(getColumn(gauges, 'rainfall').type).toBe('number');
  });

  it('gives the same profile twice', () => {
    const collection = createKeplerCuratedAnalysisSample('city-incidents')['points'];
    expect(profileKeplerFeatureCollection(collection, BASE_OPTIONS)).toEqual(
      profileKeplerFeatureCollection(collection, BASE_OPTIONS)
    );
  });
});

describe('profileKeplerFeatureCollection: statistics against NumPy', () => {
  // /usr/bin/python3: v = [x for x in fare if finite]; v.min(), v.max(), v.mean(), v.std(ddof=1)
  // fare 6 7.25 30.0 16.666666666666668 8.57855854247476 (ddof=0 would be 7.831116707653442)
  const fares = [12.5, 7.25, null, 30.0, 18.75, Number.NaN, 9.5, 22.0];
  // count 1.0 55.0 17.5 18.670067717376618
  const counts = [3, 5, 8, 1, 13, 21, 34, 55];

  it('matches mean, sample standard deviation, minimum and maximum', () => {
    const profile = profileKeplerFeatureCollection(
      collect(fares.map((fare, index) => makePoint(index, 0, {fare, count: counts[index]}))),
      BASE_OPTIONS
    );
    const fare = getColumn(profile, 'fare');
    expect(fare.type).toBe('number');
    expect(fare.rowCount).toBe(8);
    expect(fare.nullCount).toBe(2);
    expect(fare.minimum).toBe(7.25);
    expect(fare.maximum).toBe(30);
    expect(fare.mean).toBeCloseTo(16.666666666666668, 12);
    expect(fare.standardDeviation).toBeCloseTo(8.57855854247476, 12);
    expect(fare.integral).toBe(false);
    expect(fare.distinctCount).toBe(6);
    expect(fare.distinctCountIsEstimate).toBeUndefined();
    expect(fare.sampleSize).toBeUndefined();
    const count = getColumn(profile, 'count');
    expect(count.type).toBe('integer');
    expect(count.integral).toBe(true);
    expect(count.mean).toBeCloseTo(17.5, 12);
    expect(count.standardDeviation).toBeCloseTo(18.670067717376618, 12);
  });

  it('matches a 1000 row table of fractional values', () => {
    // /usr/bin/python3: i = arange(1000); v = (i*7919 % 1000)/10
    // 0.0 99.9 49.95 28.88194360957494 1000 distinct
    const values = Array.from({length: 1000}, (_, index) => ((index * 7919) % 1000) / 10);
    const profile = profileKeplerColumnTable(
      {rowCount: 1000, columns: [{name: 'value', values}]},
      BASE_OPTIONS
    );
    const column = getColumn(profile, 'value');
    expect(column.minimum).toBe(0);
    expect(column.maximum).toBe(99.9);
    expect(column.mean).toBeCloseTo(49.95, 10);
    expect(column.standardDeviation).toBeCloseTo(28.88194360957494, 10);
    expect(column.distinctCount).toBe(1000);
  });

  it('does not lose the spread of values on a large offset', () => {
    // /usr/bin/python3: w = 1e9 + (arange(1000) % 10); mean 1000000004.5, std(ddof=1) 2.8737185419345193
    const values = Array.from({length: 1000}, (_, index) => 1e9 + (index % 10));
    const profile = profileKeplerColumnTable(
      {rowCount: 1000, columns: [{name: 'w', values}]},
      BASE_OPTIONS
    );
    const column = getColumn(profile, 'w');
    // One unit in the last place at 1e9 is 1.2e-7, so the mean is right to about 1e-6.
    expect(column.mean).toBeCloseTo(1000000004.5, 5);
    expect(column.standardDeviation).toBeCloseTo(2.8737185419345193, 8);
  });
});

describe('profileKeplerFeatureCollection: awkward input', () => {
  it('profiles an empty collection', () => {
    const profile = profileKeplerFeatureCollection(collect([]), BASE_OPTIONS);
    expect(profile.rowCount).toBe(0);
    expect(profile.kind).toBe('attributes');
    expect(profile.columns).toEqual([]);
    expect(profile.extent).toBeUndefined();
    expect(profile.coordinates).toBeUndefined();
    expect(profile.spacing).toBeUndefined();
    expect(profile.catalogGeometry).toBeUndefined();
    expect(describeKeplerDatasetProfile(profile).headline).toBe('Table · 0 rows');
  });

  it('profiles a single row: no standard deviation, no spacing neighbours', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(10, 20, {amount: 4.5})]),
      BASE_OPTIONS
    );
    const amount = getColumn(profile, 'amount');
    expect(amount.mean).toBe(4.5);
    expect(amount.standardDeviation).toBeUndefined();
    expect(amount.distinctCount).toBe(1);
    expect(profile.spacing?.pointCount).toBe(1);
    expect(profile.spacing?.nearestNeighbor).toBeUndefined();
    expect(profile.extent).toEqual([10, 20, 10, 20]);
    expect(describeKeplerDatasetProfile(profile).headline).toContain('1 row ·');
  });

  it('counts null, undefined and non-finite numbers as null and keeps an all-null column', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(0, 0, {a: null, b: Number.NaN, c: 1}),
        makePoint(1, 1, {a: undefined, b: Number.POSITIVE_INFINITY, c: Number.NEGATIVE_INFINITY}),
        makePoint(2, 2, {a: null, b: Number.NaN})
      ]),
      BASE_OPTIONS
    );
    const allNull = getColumn(profile, 'a');
    expect(allNull.nullCount).toBe(3);
    expect(allNull.minimum).toBeUndefined();
    expect(allNull.distinctCount).toBeUndefined();
    expect(allNull.measurement).toBeUndefined();
    // A column of only non-finite numbers is still a number column.
    const nonFinite = getColumn(profile, 'b');
    expect(nonFinite.type).toBe('number');
    expect(nonFinite.nullCount).toBe(3);
    expect(nonFinite.mean).toBeUndefined();
    const partial = getColumn(profile, 'c');
    expect(partial.nullCount).toBe(2);
    expect(partial.minimum).toBe(1);
    expect(partial.maximum).toBe(1);
    expect(partial.standardDeviation).toBeUndefined();
  });

  it('describes a constant column without a spread', () => {
    const profile = profileKeplerFeatureCollection(
      collect(Array.from({length: 6}, (_, index) => makePoint(index, 0, {flat: 7, same: 'a'}))),
      BASE_OPTIONS
    );
    const flat = getColumn(profile, 'flat');
    expect(flat.standardDeviation).toBe(0);
    expect(flat.distinctCount).toBe(1);
    expect(flat.minimum).toBe(flat.maximum);
    const same = getColumn(profile, 'same');
    expect(same.type).toBe('category');
    expect(same.distinctCount).toBe(1);
  });

  it('types a column that mixes numbers and strings as text and counts every value', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(0, 0, {mixed: 1}),
        makePoint(1, 0, {mixed: '1'}),
        makePoint(2, 0, {mixed: 'x'}),
        makePoint(3, 0, {mixed: true}),
        makePoint(4, 0, {mixed: null})
      ]),
      BASE_OPTIONS
    );
    const mixed = getColumn(profile, 'mixed');
    expect(mixed.type).toBe('text');
    expect(mixed.nullCount).toBe(1);
    // 1 and "1" are different values; true is a fourth.
    expect(mixed.distinctCount).toBe(4);
    expect(mixed.minimum).toBeUndefined();
    expect(mixed.mean).toBeUndefined();
    expect(mixed.storage).toBe('string');
  });

  it('keeps numbers written as strings as text and booleans as booleans', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(0, 0, {digits: '12', flag: true}),
        makePoint(1, 0, {digits: '13', flag: false}),
        makePoint(2, 0, {digits: '14', flag: true})
      ]),
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'digits').type).toBe('text');
    const flag = getColumn(profile, 'flag');
    expect(flag.type).toBe('boolean');
    expect(flag.distinctCount).toBe(2);
    expect(flag.measurement).toBe('nominal');
    expect(flag.storage).toBe('boolean');
  });

  it('keeps features without geometry as rows and leaves them out of the locations', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(0, 0, {v: 1}),
        {geometry: null, properties: {v: 2}} as CatalogFeature,
        makePoint(1, 1, {v: 3}),
        {geometry: null} as CatalogFeature
      ]),
      BASE_OPTIONS
    );
    expect(profile.rowCount).toBe(4);
    expect(profile.kind).toBe('points');
    expect(profile.spacing?.pointCount).toBe(2);
    expect(getColumn(profile, 'v').nullCount).toBe(1);
    expect(profile.geometryFeatureCounts).toEqual({
      point: 2,
      line: 0,
      polygon: 0,
      withoutGeometry: 2
    });
  });

  it('profiles a collection with no geometry at all as a table', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        {geometry: null, properties: {v: 1}} as CatalogFeature,
        {geometry: null, properties: {v: 2}} as CatalogFeature
      ]),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('attributes');
    expect(profile.vertexCount).toBeUndefined();
    expect(toKeplerRecommendationSubject(profile).kind).toBe('table');
  });

  it('keeps integers beyond 2^53 as numbers, not integers, and never throws', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(0, 0, {big: 2 ** 53 + 2, mid: 5}),
        makePoint(1, 0, {big: 2 ** 53 + 4, mid: 6}),
        makePoint(2, 0, {big: BigInt(2) ** BigInt(60), mid: 7})
      ]),
      BASE_OPTIONS
    );
    const big = getColumn(profile, 'big');
    expect(big.type).toBe('number');
    expect(big.integral).toBe(true);
    expect(big.storage).toBe('float64');
    expect(big.maximum).toBe(2 ** 60);
    expect(big.nullCount).toBe(0);
    expect(getColumn(profile, 'mid').type).toBe('integer');
    expect(getColumn(profile, 'mid').storage).toBe('int32');
  });

  it('stores an integer column that exceeds 32 bits as float64', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(0, 0, {n: 3_000_000_000}), makePoint(1, 0, {n: 1})]),
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'n').type).toBe('integer');
    expect(getColumn(profile, 'n').storage).toBe('float64');
  });

  it('reads a property that only some features have', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(0, 0, {}), makePoint(1, 0, {late: 5}), makePoint(2, 0, {late: 6})]),
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'late').nullCount).toBe(1);
    expect(getColumn(profile, 'late').rowCount).toBe(3);
  });

  it('rejects bad options by name', () => {
    expect(() => profileKeplerFeatureCollection(collect([]), {datasetId: '', label: 'x'})).toThrow(
      /options\.datasetId/
    );
    expect(() =>
      profileKeplerFeatureCollection(collect([]), {...BASE_OPTIONS, maximumSampleSize: 1})
    ).toThrow(/options\.maximumSampleSize/);
    expect(() =>
      profileKeplerFeatureCollection(collect([]), {...BASE_OPTIONS, revision: 1.5})
    ).toThrow(/options\.revision/);
  });
});

describe('mixed geometry', () => {
  const polygon = (properties: Record<string, unknown>) =>
    ({
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
          ]
        ]
      },
      properties
    }) as CatalogFeature;

  it('takes the family with the most rows and says what is ignored', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(5, 5), makePoint(6, 6), makePoint(7, 7), polygon({}), polygon({})]),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('points');
    expect(profile.geometryFeatureCounts).toEqual({
      point: 3,
      line: 0,
      polygon: 2,
      withoutGeometry: 0
    });
    expect(profile.notes?.[0]).toMatch(
      /3 point rows are the dominant kind and 2 polygon rows are of other kinds.*must be split or filtered to one kind before the catalog can read it/
    );
    expect(profile.mixedGeometry).toEqual({point: 3, line: 0, polygon: 2});
    expect(profile.catalogGeometry).toBe('point');
    // The extent covers every vertex, the spacing only the dominant family.
    expect(profile.extent).toEqual([0, 0, 7, 7]);
    expect(profile.spacing?.pointCount).toBe(3);
    expect(toKeplerRecommendationSubject(profile).kind).toBe('features');
    expect(describeKeplerDatasetProfile(profile).headline).toContain('Points (mixed geometry)');
  });

  it('breaks a tie towards polygons', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(5, 5), polygon({})]),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('polygons');
    expect(profile.catalogGeometry).toBe('polygon');
  });

  it('counts a geometry collection as a row without geometry', () => {
    const profile = profileKeplerFeatureCollection(
      collect([{geometry: {type: 'GeometryCollection'}, properties: {}} as CatalogFeature]),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('attributes');
    expect(profile.geometryFeatureCounts?.withoutGeometry).toBe(1);
  });

  it('gives every position of a multi-point row to the spacing', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        {
          geometry: {
            type: 'MultiPoint',
            coordinates: [
              [0, 0],
              [0.001, 0.001]
            ]
          },
          properties: {}
        } as CatalogFeature
      ]),
      BASE_OPTIONS
    );
    expect(profile.rowCount).toBe(1);
    expect(profile.vertexCount).toBe(2);
    expect(profile.spacing?.pointCount).toBe(2);
  });
});

describe('coordinates', () => {
  const points = collect([makePoint(-87.6, 41.8), makePoint(-87.7, 41.9), makePoint(-87.5, 41.7)]);

  it('guesses from the extent and marks the guess', () => {
    const profile = profileKeplerFeatureCollection(points, BASE_OPTIONS);
    expect(profile.coordinates?.source).toBe('guess');
    expect(profile.coordinates?.reason).toBeTruthy();
    expect(profile.spacing?.unit).toBe('metre');
  });

  it('uses the declared coordinate system and its units', () => {
    const profile = profileKeplerFeatureCollection(points, {
      ...BASE_OPTIONS,
      coordinateSystem: 'planar'
    });
    expect(profile.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    expect(profile.spacing?.unit).toBe('coordinate-unit');
  });

  it('prefers the declaration of the user over the coordinate system option', () => {
    const profile = profileKeplerFeatureCollection(points, {
      ...BASE_OPTIONS,
      coordinateSystem: 'planar',
      declarations: {coordinateModel: 'geographic'}
    });
    expect(profile.coordinates).toMatchObject({model: 'geographic', source: 'declared'});
    expect(profile.spacing?.unit).toBe('metre');
  });

  it('reads coordinates outside degrees as planar', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(500000, 4000000), makePoint(500100, 4000100)]),
      BASE_OPTIONS
    );
    expect(profile.coordinates?.model).toBe('planar');
  });
});

describe('sampling', () => {
  it('computes column statistics on a stride sample and says so', () => {
    const features = Array.from({length: 1000}, (_, index) =>
      makePoint(index / 100, 0, {n: index})
    );
    const profile = profileKeplerFeatureCollection(collect(features), {
      ...BASE_OPTIONS,
      maximumSampleSize: 100
    });
    const column = getColumn(profile, 'n');
    expect(column.rowCount).toBe(1000);
    expect(column.sampleSize).toBe(100);
    expect(column.nullCount).toBe(0);
    // Every tenth row: 0, 10, ..., 990.
    expect(column.minimum).toBe(0);
    expect(column.maximum).toBe(990);
    expect(profile.rowCount).toBe(1000);
    expect(profile.spacing?.sampled).toBe(true);
    expect(profile.spacing?.sampleSize).toBe(100);
    expect(profile.notes?.some(note => /stride sample of 100 of 1000/.test(note))).toBe(true);
  });
});

describe('profileKeplerColumnTable: types and distinct counts', () => {
  it('maps catalog column types and infers time from string columns', () => {
    const table: CatalogOutputTable = {
      id: 't',
      label: 'T',
      rowCount: 4,
      columns: {
        score: [1.5, 2.5, null, 4],
        rank: [1, 2, 3, 4],
        kind: ['a', 'b', 'a', 'b'],
        when: ['2025-01-01T00:00:00Z', '2025-01-02', null, '2025-01-03T12:00:00+02:00'],
        ok: [true, false, true, null]
      },
      columnTypes: {score: 'float64', rank: 'int32', kind: 'string', when: 'string', ok: 'boolean'}
    };
    const profile = profileKeplerColumnTable(table, BASE_OPTIONS);
    expect(profile.kind).toBe('attributes');
    expect(profile.columns.map(column => [column.name, column.type, column.storage])).toEqual([
      ['score', 'number', 'float64'],
      ['rank', 'integer', 'int32'],
      ['kind', 'category', 'string'],
      ['when', 'time', 'string'],
      ['ok', 'boolean', 'boolean']
    ]);
    const when = getColumn(profile, 'when');
    expect(when.minimum).toBe(Date.UTC(2025, 0, 1));
    expect(when.maximum).toBe(Date.UTC(2025, 0, 3, 10));
    expect(when.nullCount).toBe(1);
    expect(getColumn(profile, 'kind').distinctCount).toBe(2);
  });

  it('counts a value of another class than the declared type as null', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 4,
        columns: [{name: 'amount', type: 'float64', values: [1, 'two', 3, null]}]
      },
      BASE_OPTIONS
    );
    const amount = getColumn(profile, 'amount');
    expect(amount.type).toBe('number');
    expect(amount.nullCount).toBe(2);
    expect(amount.maximum).toBe(3);
    expect(amount.distinctCount).toBe(2);
  });

  it('reads date-time text without a zone as UTC, whatever the machine zone', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 2,
        columns: [{name: 'at', values: ['2025-03-01T10:00:00', '2025-03-01T11:30:00']}]
      },
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'at').minimum).toBe(Date.UTC(2025, 2, 1, 10));
    expect(getColumn(profile, 'at').maximum).toBe(Date.UTC(2025, 2, 1, 11, 30));
  });

  it('maps Date values to a float64 time column', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 2,
        columns: [
          {name: 'at', values: [new Date(Date.UTC(2025, 0, 1)), new Date(Date.UTC(2025, 0, 2))]}
        ]
      },
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'at').type).toBe('time');
    expect(toKeplerRecommendationSubject(profile).columns).toEqual([{name: 'at', type: 'float64'}]);
  });

  it('counts distinct values exactly up to the limit and estimates beyond it', () => {
    const exactCount = KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT;
    const atLimit = Array.from({length: exactCount}, (_, index) => index);
    const exact = getColumn(
      profileKeplerColumnTable(
        {rowCount: exactCount, columns: [{name: 'v', values: atLimit}]},
        BASE_OPTIONS
      ),
      'v'
    );
    expect(exact.distinctCount).toBe(exactCount);
    expect(exact.distinctCountIsEstimate).toBeUndefined();

    const total = 200_000;
    const numbers = Array.from({length: total}, (_, index) => index * 3);
    const estimated = getColumn(
      profileKeplerColumnTable(
        {rowCount: total, columns: [{name: 'v', values: numbers}]},
        BASE_OPTIONS
      ),
      'v'
    );
    expect(estimated.distinctCountIsEstimate).toBe(true);
    expect(Math.abs((estimated.distinctCount as number) - total) / total).toBeLessThan(0.01);

    // Strings take the same route; repeats must not inflate the estimate.
    const strings = Array.from({length: total}, (_, index) => `key-${index % 50_000}`);
    const stringEstimate = getColumn(
      profileKeplerColumnTable(
        {rowCount: total, columns: [{name: 's', values: strings}]},
        BASE_OPTIONS
      ),
      's'
    );
    expect(stringEstimate.distinctCountIsEstimate).toBe(true);
    expect(Math.abs((stringEstimate.distinctCount as number) - 50_000) / 50_000).toBeLessThan(0.01);
    expect(stringEstimate.type).toBe('text');
  });

  it('rejects a table whose columns differ in length or repeat a name', () => {
    expect(() =>
      profileKeplerColumnTable({rowCount: 3, columns: [{name: 'a', values: [1, 2]}]}, BASE_OPTIONS)
    ).toThrow(/column "a" has 2 values/);
    expect(() =>
      profileKeplerColumnTable(
        {
          rowCount: 1,
          columns: [
            {name: 'a', values: [1]},
            {name: 'a', values: [2]}
          ]
        },
        BASE_OPTIONS
      )
    ).toThrow(/appears twice/);
  });
});

describe('profileKeplerColumnTable: coordinates', () => {
  function makeTable(columns: Record<string, number[]>) {
    const names = Object.keys(columns);
    return {
      rowCount: columns[names[0]].length,
      columns: names.map(name => ({name, values: columns[name]}))
    };
  }

  it('detects lat/lon columns by name when values are in range', () => {
    const profile = profileKeplerColumnTable(
      makeTable({fare: [3, 4, 5], lon: [-87.6, -87.7, -87.5], lat: [41.8, 41.9, 41.7]}),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('points');
    expect(profile.pointColumns).toEqual({longitude: 'lon', latitude: 'lat'});
    expect(profile.catalogGeometry).toBe('point');
    expect(profile.coordinates?.model).toBe('geographic');
    expect(profile.coordinates?.reason).toMatch(/Detected the coordinate columns lon\/lat by name/);
    expect(profile.extent).toEqual([-87.7, 41.7, -87.5, 41.9]);
    expect(profile.spacing?.pointCount).toBe(3);
    expect(describeKeplerDatasetProfile(profile).headline).toBe(
      'Points · 3 rows · lon/lat · rows probably objects (guessed, low confidence) · 1 numeric column'
    );
  });

  it('recognises the usual spellings and separators', () => {
    for (const [longitude, latitude] of [
      ['longitude', 'latitude'],
      ['Lng', 'Lat'],
      ['pickup_longitude', 'pickup_latitude'],
      ['dropoffLon', 'dropoffLat'],
      ['LONG', 'LAT'],
      ['lon-1', 'lat-1']
    ]) {
      const profile = profileKeplerColumnTable(
        makeTable({[latitude]: [41.8, 41.9], [longitude]: [-87.6, -87.7]}),
        BASE_OPTIONS
      );
      expect(profile.pointColumns, `${longitude}/${latitude}`).toEqual({longitude, latitude});
    }
  });

  it('does not take other names as coordinates', () => {
    const profile = profileKeplerColumnTable(
      makeTable({latency: [41.8, 41.9], longevity: [-87.6, -87.7], x: [1, 2], y: [3, 4]}),
      BASE_OPTIONS
    );
    expect(profile.pointColumns).toBeUndefined();
    expect(profile.kind).toBe('attributes');
  });

  it('does not take lon/lat-named columns with values out of range', () => {
    const profile = profileKeplerColumnTable(
      makeTable({lon: [500000, 500100], lat: [4000000, 4000100]}),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('attributes');
    expect(profile.pointColumns).toBeUndefined();
    expect(profile.coordinates).toBeUndefined();
    expect(profile.spacing).toBeUndefined();
    expect(profile.notes?.[0]).toMatch(
      /lon\/lat rejected as coordinates: lat has 2 values outside -90 to 90/
    );
  });

  it('rejects a pair when one value is out of range, and when the columns are swapped', () => {
    const oneBad = profileKeplerColumnTable(
      makeTable({lon: [-87.6, -87.7, 10], lat: [41.8, 41.9, 999]}),
      BASE_OPTIONS
    );
    expect(oneBad.pointColumns).toBeUndefined();
    const swapped = profileKeplerColumnTable(
      makeTable({lon: [41.8, 41.9], lat: [-120.5, -130.5]}),
      BASE_OPTIONS
    );
    expect(swapped.pointColumns).toBeUndefined();
    // Both are exact boundaries: 90 and 180 are valid.
    const boundary = profileKeplerColumnTable(
      makeTable({lon: [-180, 180], lat: [-90, 90]}),
      BASE_OPTIONS
    );
    expect(boundary.pointColumns).toEqual({longitude: 'lon', latitude: 'lat'});
  });

  it('does not take a text column or an all-null column named latitude', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 2,
        columns: [
          {name: 'lat', values: ['north', 'south']},
          {name: 'lon', values: [1, 2]}
        ]
      },
      BASE_OPTIONS
    );
    expect(profile.pointColumns).toBeUndefined();
    const allNull = profileKeplerColumnTable(
      {
        rowCount: 2,
        columns: [
          {name: 'lat', values: [null, null]},
          {name: 'lon', type: 'float64', values: [1, 2]}
        ]
      },
      BASE_OPTIONS
    );
    expect(allNull.pointColumns).toBeUndefined();
  });

  it('with two candidate pairs takes the plain pair and names the other', () => {
    const profile = profileKeplerColumnTable(
      makeTable({
        pickup_lat: [41.8, 41.9],
        pickup_lng: [-87.6, -87.7],
        lat: [40.1, 40.2],
        lng: [-88.1, -88.2]
      }),
      BASE_OPTIONS
    );
    expect(profile.pointColumns).toEqual({longitude: 'lng', latitude: 'lat'});
    expect(profile.coordinates?.reason).toMatch(
      /Other valid pairs were found and not used: pickup_lng\/pickup_lat/
    );
    expect(profile.extent).toEqual([-88.2, 40.1, -88.1, 40.2]);
  });

  it('skips an invalid first pair and takes the next valid one', () => {
    const profile = profileKeplerColumnTable(
      makeTable({
        lat: [4000000, 4000001],
        lng: [500000, 500001],
        end_lat: [41.8, 41.9],
        end_lng: [-87.6, -87.7]
      }),
      BASE_OPTIONS
    );
    expect(profile.pointColumns).toEqual({longitude: 'end_lng', latitude: 'end_lat'});
    expect(profile.notes?.[0]).toMatch(/lng\/lat rejected/);
  });

  it('uses given point columns without a range check and guesses planar from the extent', () => {
    const profile = profileKeplerColumnTable(
      makeTable({easting: [500000, 500100], northing: [4000000, 4000100]}),
      {...BASE_OPTIONS, pointColumns: {longitude: 'easting', latitude: 'northing'}}
    );
    expect(profile.kind).toBe('points');
    expect(profile.coordinates?.model).toBe('planar');
    expect(profile.coordinates?.reason).toMatch(/were given by the host/);
    expect(() =>
      profileKeplerColumnTable(makeTable({a: [1]}), {
        ...BASE_OPTIONS,
        pointColumns: {longitude: 'a', latitude: 'missing'}
      })
    ).toThrow(/point column "missing"/);
  });

  it('skips rows with a missing coordinate and counts the rest', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 4,
        columns: [
          {name: 'lat', values: [41.8, null, 41.9, Number.NaN]},
          {name: 'lng', values: [-87.6, -87.7, -87.5, -87.4]}
        ]
      },
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('points');
    expect(profile.rowCount).toBe(4);
    expect(profile.spacing?.pointCount).toBe(2);
  });
});

describe('profileKeplerColumnTable: geometry text columns', () => {
  it('reads the geometry column of a catalog table and leaves it out of the statistics', () => {
    const geometries = [0, 1, 2].map(index =>
      JSON.stringify({type: 'Point', coordinates: [-87.6 + index * 0.01, 41.8]})
    );
    const table: CatalogOutputTable = {
      id: 't',
      label: 'T',
      rowCount: 4,
      columns: {geometry: [...geometries, 'not json'], value: [1, 2, 3, 4]},
      columnTypes: {geometry: 'string', value: 'int32'},
      geometryColumn: 'geometry'
    };
    const profile = profileKeplerColumnTable(table, BASE_OPTIONS);
    expect(profile.kind).toBe('points');
    expect(profile.geometryColumns).toEqual(['geometry']);
    expect(profile.columns.map(column => column.name)).toEqual(['value']);
    expect(profile.geometryFeatureCounts?.withoutGeometry).toBe(1);
    expect(profile.spacing?.pointCount).toBe(3);
  });

  it('refuses geometry columns together with point columns', () => {
    expect(() =>
      profileKeplerColumnTable(
        {
          rowCount: 0,
          columns: [
            {name: 'g', values: []},
            {name: 'lat', values: []},
            {name: 'lon', values: []}
          ]
        },
        {
          ...BASE_OPTIONS,
          geometryColumns: ['g'],
          pointColumns: {longitude: 'lon', latitude: 'lat'}
        }
      )
    ).toThrow(/either options\.pointColumns or geometry columns/);
  });
});

describe('declarations', () => {
  const collection = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];

  it('replace a guess, mark it declared and leave the original profile alone', () => {
    const original = profileKeplerFeatureCollection(collection, BASE_OPTIONS);
    const before = JSON.stringify(original);
    expect(getColumn(original, 'cases').measurement).toBe('count');
    const corrected = applyKeplerDatasetProfileDeclarations(original, {
      measurement: {cases: 'amount', missing: 'rate'},
      rowSemantics: 'objects',
      coordinateModel: 'planar'
    });
    expect(JSON.stringify(original)).toBe(before);
    const cases = getColumn(corrected, 'cases');
    expect(cases.measurement).toBe('amount');
    expect(cases.measurementSource).toBe('declared');
    expect(cases.measurementReason).toBeUndefined();
    expect(getColumn(corrected, 'rate').measurementSource).toBe('guess');
    expect(corrected.rowSemantics).toMatchObject({value: 'objects', source: 'declared'});
    expect(corrected.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    // The spacing was in metres; a planar declaration recomputes it in coordinate units.
    expect(corrected.spacing?.unit).toBe('coordinate-unit');
    expect(original.spacing?.unit).toBe('metre');
    // The statistics are the same objects, not recomputed.
    expect(corrected.columns.find(column => column.name === 'population')).toBe(
      original.columns.find(column => column.name === 'population')
    );
  });

  it('keep the spacing when the declared model matches the one used', () => {
    const original = profileKeplerFeatureCollection(collection, BASE_OPTIONS);
    const corrected = applyKeplerDatasetProfileDeclarations(original, {
      coordinateModel: 'geographic'
    });
    expect(corrected.coordinates?.source).toBe('declared');
    expect(corrected.spacing).toBe(original.spacing);
  });

  it('given at profile time give the same result as applying them afterwards', () => {
    const declarations = {measurement: {cases: 'amount' as const}, rowSemantics: 'events' as const};
    const direct = profileKeplerFeatureCollection(collection, {...BASE_OPTIONS, declarations});
    const later = applyKeplerDatasetProfileDeclarations(
      profileKeplerFeatureCollection(collection, BASE_OPTIONS),
      declarations
    );
    expect(direct).toEqual(later);
  });

  it('can declare a measurement on a column with no guess, such as a time column', () => {
    const points = createKeplerCuratedAnalysisSample('city-incidents')['points'];
    const profile = profileKeplerFeatureCollection(points, {
      ...BASE_OPTIONS,
      declarations: {measurement: {observed: 'cyclic'}}
    });
    expect(getColumn(profile, 'observed').measurement).toBe('cyclic');
    expect(getColumn(profile, 'observed').measurementSource).toBe('declared');
  });

  it('ignore a coordinate model for a table without coordinates', () => {
    const profile = profileKeplerColumnTable(
      {rowCount: 1, columns: [{name: 'a', values: [1]}]},
      {...BASE_OPTIONS, declarations: {coordinateModel: 'planar'}}
    );
    expect(profile.coordinates).toBeUndefined();
  });
});

describe('describeKeplerDatasetProfile', () => {
  it('writes the headline and one chip per measured column', () => {
    const rows = 48_210;
    const profile = profileKeplerColumnTable(
      {
        rowCount: rows,
        columns: [
          {
            name: 'longitude',
            type: 'float64',
            values: Array.from({length: rows}, (_, index) => -87.9 + (index % 1000) / 5000)
          },
          {
            name: 'latitude',
            type: 'float64',
            values: Array.from({length: rows}, (_, index) => 41.6 + (index % 777) / 5000)
          },
          {
            name: 'fare',
            type: 'float64',
            values: Array.from({length: rows}, (_, index) => 3 + (index % 100) * 0.35)
          },
          {
            name: 'tip_amount',
            type: 'float64',
            values: Array.from({length: rows}, (_, index) => (index % 13) * 0.5)
          },
          {
            name: 'passengers',
            type: 'int32',
            values: Array.from({length: rows}, (_, index) => 1 + (index % 4))
          },
          {
            name: 'pickup_time',
            type: 'string',
            values: Array.from(
              {length: rows},
              (_, index) => `2025-01-${String(1 + (index % 28)).padStart(2, '0')}T08:00:00Z`
            )
          }
        ]
      },
      {...BASE_OPTIONS, declarations: {measurement: {fare: 'amount'}}}
    );
    const description = describeKeplerDatasetProfile(profile);
    expect(description.headline).toBe(
      'Points · 48,210 rows · longitude/latitude · 3 numeric columns · 1 time column'
    );
    const fare = description.chips.find(chip => chip.column === 'fare');
    expect(fare).toEqual({column: 'fare', label: 'fare: amount', guessed: false});
    const passengers = description.chips.find(chip => chip.column === 'passengers');
    expect(passengers?.guessed).toBe(true);
    expect(description.chips.some(chip => chip.column === 'pickup_time')).toBe(false);
  });
});

describe('toKeplerRecommendationSubject', () => {
  it('maps kinds, row counts and storage types', () => {
    const points = profileKeplerFeatureCollection(
      createKeplerCuratedAnalysisSample('city-incidents')['points'],
      BASE_OPTIONS
    );
    const subject = toKeplerRecommendationSubject(points);
    expect(subject.kind).toBe('point');
    expect(subject.rowCount).toBe(points.rowCount);
    expect(subject.columns).toEqual([
      {name: 'severity', type: 'int32'},
      {name: 'observed', type: 'string'}
    ]);
    const areas = profileKeplerFeatureCollection(
      createKeplerCuratedAnalysisSample('city-district-rates')['areas'],
      BASE_OPTIONS
    );
    expect(toKeplerRecommendationSubject(areas).kind).toBe('polygon');
  });

  it('maps a profile from elsewhere (no storage) with time as string', () => {
    const subject = toKeplerRecommendationSubject({
      datasetId: 'x',
      label: 'x',
      kind: 'attributes',
      rowCount: 2,
      geometryColumns: [],
      columns: [
        {name: 'a', type: 'number', rowCount: 2, nullCount: 0},
        {name: 'b', type: 'time', rowCount: 2, nullCount: 0},
        {name: 'c', type: 'category', rowCount: 2, nullCount: 0},
        {name: 'd', type: 'boolean', rowCount: 2, nullCount: 0},
        {name: 'e', type: 'integer', rowCount: 2, nullCount: 0}
      ]
    });
    expect(subject.columns.map(column => column.type)).toEqual([
      'float64',
      'string',
      'string',
      'boolean',
      'int32'
    ]);
    expect(subject.kind).toBe('table');
  });
});

describe('speed', () => {
  it('profiles 200,000 rows by 10 columns', () => {
    const rowCount = 200_000;
    const columns = Array.from({length: 10}, (_, columnIndex) => ({
      name: `column_${columnIndex}`,
      values: Array.from({length: rowCount}, (_, index) =>
        columnIndex % 3 === 0
          ? ((index * 7919 + columnIndex) % 100_003) / 7
          : columnIndex % 3 === 1
            ? (index * 31 + columnIndex) % 5000
            : `group-${(index + columnIndex) % 37}`
      )
    }));
    const startedAt = performance.now();
    const profile = profileKeplerColumnTable({rowCount, columns}, BASE_OPTIONS);
    const elapsedMilliseconds = performance.now() - startedAt;
    process.stderr.write(
      `profile 200000 rows x 10 columns: ${elapsedMilliseconds.toFixed(0)} ms\n`
    );
    expect(profile.columns).toHaveLength(10);
    expect(elapsedMilliseconds).toBeLessThan(5000);
  });
});

function makeGrid(
  columnCount: number,
  rowCount: number,
  originLongitude = 10,
  originLatitude = 20
) {
  const features: CatalogFeature[] = [];
  for (let row = 0; row < rowCount; row++)
    for (let column = 0; column < columnCount; column++)
      features.push(
        makePoint(originLongitude + column * 0.1, originLatitude + row * 0.1, {
          amount: row * columnCount + column
        })
      );
  return collect(features);
}

describe('declared coordinate model recomputes the location statistics', () => {
  const grid = makeGrid(8, 5);

  it('gives spacing in coordinate units that equals profiling with the planar system', () => {
    const guessed = profileKeplerFeatureCollection(grid, BASE_OPTIONS);
    expect(guessed.coordinates?.model).toBe('geographic');
    expect(guessed.spacing?.unit).toBe('metre');
    const corrected = applyKeplerDatasetProfileDeclarations(guessed, {coordinateModel: 'planar'});
    expect(corrected.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    expect(corrected.spacing?.unit).toBe('coordinate-unit');
    expect(corrected.spacing?.nearestNeighbor?.median).toBeCloseTo(0.1, 9);
    const direct = profileKeplerFeatureCollection(grid, {
      ...BASE_OPTIONS,
      coordinateSystem: 'planar'
    });
    expect(corrected.spacing).toEqual(direct.spacing);
    // The guess is untouched.
    expect(guessed.spacing?.unit).toBe('metre');
  });

  it('goes back to geographic spacing when the declaration changes again', () => {
    const guessed = profileKeplerFeatureCollection(grid, BASE_OPTIONS);
    const planar = applyKeplerDatasetProfileDeclarations(guessed, {coordinateModel: 'planar'});
    const geographic = applyKeplerDatasetProfileDeclarations(planar, {
      coordinateModel: 'geographic'
    });
    expect(geographic.spacing).toEqual(guessed.spacing);
  });

  it('is applied by the profile call itself, so the options declaration gives the same result', () => {
    const viaOptions = profileKeplerFeatureCollection(grid, {
      ...BASE_OPTIONS,
      declarations: {coordinateModel: 'planar'}
    });
    const viaApply = applyKeplerDatasetProfileDeclarations(
      profileKeplerFeatureCollection(grid, BASE_OPTIONS),
      {coordinateModel: 'planar'}
    );
    expect(viaOptions.spacing?.unit).toBe('coordinate-unit');
    expect(viaOptions).toEqual(viaApply);
  });

  it('drops spacing with a note for a profile that holds no locations', () => {
    const guessed = profileKeplerFeatureCollection(grid, BASE_OPTIONS);
    // A copy through JSON is a profile from elsewhere: no locations are kept with it.
    const foreign = JSON.parse(JSON.stringify(guessed));
    const corrected = applyKeplerDatasetProfileDeclarations(foreign, {coordinateModel: 'planar'});
    expect(corrected.spacing).toBeUndefined();
    expect(corrected.coordinates.model).toBe('planar');
    expect(corrected.notes.join(' ')).toMatch(
      /spacing was dropped because the coordinate model was declared planar .* profile the dataset again/
    );
    // Nothing to say when the foreign profile had no spacing to lose.
    const withoutSpacing = {...foreign, spacing: undefined, notes: undefined};
    expect(
      applyKeplerDatasetProfileDeclarations(withoutSpacing, {coordinateModel: 'planar'}).notes
    ).toBeUndefined();
  });

  it('keeps the spacing when the declared model is the one in use, with no note', () => {
    const guessed = profileKeplerFeatureCollection(grid, BASE_OPTIONS);
    const corrected = applyKeplerDatasetProfileDeclarations(guessed, {
      coordinateModel: 'geographic'
    });
    expect(corrected.spacing).toBe(guessed.spacing);
    expect(corrected.notes).toBeUndefined();
  });

  it('preserves full-data summaries and hashed sampling provenance when the model changes', () => {
    const big = makeGrid(10, 10);
    const guessed = profileKeplerFeatureCollection(big, {...BASE_OPTIONS, maximumSampleSize: 20});
    const corrected = applyKeplerDatasetProfileDeclarations(guessed, {coordinateModel: 'planar'});
    expect(corrected.spacing?.unit).toBe('coordinate-unit');
    expect(corrected.spacing?.pointCount).toBe(100);
    expect(corrected.spacing?.sampleSize).toBe(20);
    expect(corrected.spacing?.sampled).toBe(true);
    const locations = big.features.flatMap(
      feature => (feature.geometry as {coordinates: number[]}).coordinates
    );
    expect(corrected.spacing).toEqual(
      computeGPUAnalysisSpacingStatistics(locations, {
        coordinateModel: 'planar',
        maximumSampleSize: 20
      })
    );
  });

  it('survives a copy of the profile object made by the host with a spread', () => {
    const guessed = profileKeplerFeatureCollection(grid, BASE_OPTIONS);
    const spread = {...guessed};
    // A spread is a new object: the locations are not carried, which is the documented limit.
    expect(
      applyKeplerDatasetProfileDeclarations(spread, {coordinateModel: 'planar'}).spacing
    ).toBeUndefined();
  });
});

describe('notes', () => {
  function makeTable(columns: Record<string, unknown[]>, options: Record<string, unknown> = {}) {
    const names = Object.keys(columns);
    return profileKeplerColumnTable(
      {
        rowCount: columns[names[0]].length,
        columns: names.map(name => ({name, values: columns[name]}))
      },
      {...BASE_OPTIONS, ...options}
    );
  }

  it('says when point columns never both have a value', () => {
    const profile = makeTable({
      lat: [41, Number.NaN, 42, Number.NaN],
      lng: [Number.NaN, -87, Number.NaN, -88]
    });
    expect(profile.kind).toBe('points');
    expect(profile.extent).toBeUndefined();
    expect(profile.spacing).toBeUndefined();
    expect(profile.notes?.join(' ')).toMatch(/never both have a finite value in the same row/);
  });

  it('says how many locations a declared geographic model leaves out or finds projected', () => {
    const profile = makeTable(
      {e: [1, 2, 3, 4, 500, 600], n: [1, 2, 3, 4, 5, 600]},
      {pointColumns: {longitude: 'e', latitude: 'n'}, declarations: {coordinateModel: 'geographic'}}
    );
    expect(profile.coordinates).toMatchObject({model: 'geographic', source: 'declared'});
    const notes = profile.notes?.join(' ') ?? '';
    expect(notes).toMatch(
      /1 of 6 locations have a latitude outside -90 to 90 and are left out of the spacing statistics/
    );
    expect(notes).toMatch(
      /2 of 6 locations have a longitude outside -180 to 180; the values look projected/
    );
    expect(profile.spacing?.pointCount).toBe(5);
  });

  it('says nothing about the model when geographic values are in range', () => {
    const profile = makeTable({lat: [41, 42], lng: [-87, -88]}, {coordinateSystem: 'EPSG:4326'});
    expect(profile.notes).toBeUndefined();
  });

  it('names x/y and easting/northing pairs without taking them as coordinates', () => {
    const profile = makeTable({
      x: [1, 2, 3],
      y: [4, 5, 6],
      easting: [500000, 500010, 500020],
      northing: [4000000, 4000010, 4000020],
      start_x: [1, 2, 3],
      start_y: [1, 2, 3],
      label: ['a', 'b', 'c']
    });
    expect(profile.pointColumns).toBeUndefined();
    expect(profile.kind).toBe('attributes');
    const notes = profile.notes?.join('\n') ?? '';
    expect(notes).toMatch(
      /Columns x\/y look like planar point columns and were not taken as coordinates/
    );
    expect(notes).toMatch(/Columns easting\/northing look like planar point columns/);
    expect(notes).toMatch(/Columns start_x\/start_y look like planar point columns/);
    expect(notes).toMatch(/options\.pointColumns/);
  });

  it('names x/y even when a lat/lng pair was taken, and never names the chosen pair', () => {
    const profile = makeTable({lat: [41, 42], lng: [-87, -88], x: [1, 2], y: [3, 4]});
    expect(profile.pointColumns).toEqual({longitude: 'lng', latitude: 'lat'});
    expect(profile.notes?.join(' ')).toMatch(/Columns x\/y look like planar point columns/);
    expect(profile.notes?.join(' ')).not.toMatch(/lng\/lat look like/);
  });

  it('does not name a text x/y pair, and offers the pair as planar point columns that then work', () => {
    expect(makeTable({x: ['a', 'b'], y: ['c', 'd']}).notes).toBeUndefined();
    const planar = makeTable(
      {x: [1, 2, 3], y: [4, 5, 6]},
      {pointColumns: {longitude: 'x', latitude: 'y'}, coordinateSystem: 'planar'}
    );
    expect(planar.kind).toBe('points');
    expect(planar.spacing?.unit).toBe('coordinate-unit');
    expect(planar.notes).toBeUndefined();
  });

  it('says why a column is not a time when its dates are impossible', () => {
    const profile = makeTable({when: ['2020-01-01', '2021-02-29']});
    expect(profile.notes?.join(' ')).toMatch(
      /Column "when" has 1 values shaped like dates that name impossible dates/
    );
  });

  it('suggests declaring an epoch-like column with a time name as time', () => {
    const profile = makeTable({
      created_ms: [1600000000000, 1600000001000],
      created_s: [1600000000, 1600000001],
      amount: [1600000000000, 1600000001000],
      time: [5, 10],
      updated_at: [1600000000.5, 1600000001]
    });
    const notes = profile.notes?.join('\n') ?? '';
    expect(notes).toMatch(
      /Column "created_ms" holds whole numbers in the range of epoch seconds or milliseconds/
    );
    expect(notes).toMatch(/Column "created_s"/);
    // No time word in the name; small values; fractional values.
    expect(notes).not.toMatch(/Column "amount"/);
    expect(notes).not.toMatch(/Column "time"/);
    expect(notes).not.toMatch(/Column "updated_at"/);
    // An undeclared epoch column stays an integer.
    expect(profile.columns.find(column => column.name === 'created_ms')?.type).toBe('integer');
    // A time-like numeric column gets no measurement level, so it cannot fill a value role.
    expect(
      profile.columns.find(column => column.name === 'created_ms')?.measurement
    ).toBeUndefined();
  });
});

describe('epoch numbers in a column declared time', () => {
  it('counts finite numbers as time values and stores them as float64', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 4,
        columns: [
          {name: 'at', type: 'time', values: [1600000001000, 1600000000000, null, Number.NaN]}
        ]
      },
      BASE_OPTIONS
    );
    const column = getColumn(profile, 'at');
    expect(column.type).toBe('time');
    expect(column.nullCount).toBe(2);
    expect(column.minimum).toBe(1600000000000);
    expect(column.maximum).toBe(1600000001000);
    expect(column.distinctCount).toBe(2);
    expect(column.storage).toBe('float64');
    expect(profile.notes).toBeUndefined();
    expect(toKeplerRecommendationSubject(profile).columns).toEqual([{name: 'at', type: 'float64'}]);
  });

  it('can be declared through options.columnTypes for a feature collection', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(0, 0, {at: 1600000000}), makePoint(1, 1, {at: 1600000600})]),
      {...BASE_OPTIONS, columnTypes: {at: 'time'}}
    );
    expect(getColumn(profile, 'at')).toMatchObject({
      type: 'time',
      nullCount: 0,
      minimum: 1600000000,
      maximum: 1600000600
    });
    expect(profile.notes?.some(note => /epoch/.test(note))).toBe(false);
  });

  it('mixes epoch numbers with ISO text declared as time, stored as string', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 3,
        columns: [{name: 'at', type: 'time', values: [0, '1970-01-01T00:00:01Z', 'garbage']}]
      },
      BASE_OPTIONS
    );
    expect(getColumn(profile, 'at')).toMatchObject({
      nullCount: 1,
      minimum: 0,
      maximum: 1000,
      storage: 'string'
    });
  });
});

describe('impossible dates', () => {
  const impossible = [
    '2020-02-30',
    '2021-02-29',
    '2020-04-31',
    '2020-13-01',
    '2020-00-10',
    '2020-01-00',
    '2020-01-01T24:00:00Z',
    '2020-01-01T23:60:00Z',
    '2020-01-01T23:59:60Z',
    '2020-01-01T10:00:00+24:00',
    '1900-02-29'
  ];
  for (const text of impossible) {
    it(`makes a column holding ${text} text, with a note`, () => {
      const profile = profileKeplerColumnTable(
        {rowCount: 3, columns: [{name: 'd', values: ['2020-01-01', text, '2020-01-02']}]},
        BASE_OPTIONS
      );
      expect(getColumn(profile, 'd').type).toBe('text');
      expect(profile.notes?.join(' ')).toMatch(/impossible dates or times/);
    });
  }

  it('accepts real edge dates and reads them as the right instant', () => {
    const values = [
      '2020-02-29',
      '2000-02-29',
      '2020-12-31T23:59:59.999Z',
      '2020-01-01T00:00:00-05:00',
      '2020-06-15T12:30+0530',
      '0001-01-01',
      '9999-12-31'
    ];
    const profile = profileKeplerColumnTable(
      {rowCount: values.length, columns: [{name: 'd', values}]},
      BASE_OPTIONS
    );
    const column = getColumn(profile, 'd');
    expect(column.type).toBe('time');
    expect(profile.notes).toBeUndefined();
    // Python datetime, UTC: 0001-01-01 is -62135596800000 ms and 9999-12-31 is 253402214400000 ms.
    expect(column.minimum).toBe(-62135596800000);
    expect(column.maximum).toBe(253402214400000);
    for (const [text, expected] of [
      ['2020-02-29', Date.UTC(2020, 1, 29)],
      ['2020-12-31T23:59:59.999Z', Date.UTC(2020, 11, 31, 23, 59, 59, 999)],
      ['2020-01-01T00:00:00-05:00', Date.UTC(2020, 0, 1, 5)],
      ['2020-06-15T12:30+0530', Date.UTC(2020, 5, 15, 7)]
    ] as const) {
      const single = getColumn(
        profileKeplerColumnTable(
          {rowCount: 1, columns: [{name: 'd', values: [text]}]},
          BASE_OPTIONS
        ),
        'd'
      );
      expect(single.minimum, text).toBe(expected);
    }
  });
});

describe('statistics at the extremes', () => {
  // /usr/bin/python3, exact Fractions: values = scale * (1 + ((i * 7919 % 1000) / 1000 - 0.5)), i < 1000;
  // sample standard deviation from the exact variance. Results (mean, sd):
  //   1e160  (9.995e+159, 2.888194360957494e+159)
  //   1e-200 (9.995e-201, 2.888194360957494e-201)
  //   1e300  (9.995e+299, 2.888194360957494e+299)
  //   [1e200, -1e200, 1e200, -1e200]  (0.0, 1.1547005383792515e+200)
  const makeValues = (scale: number) =>
    Array.from({length: 1000}, (_, index) => scale * (1 + (((index * 7919) % 1000) / 1000 - 0.5)));
  const profileValues = (values: unknown[]) =>
    getColumn(
      profileKeplerColumnTable(
        {rowCount: values.length, columns: [{name: 'v', values}]},
        BASE_OPTIONS
      ),
      'v'
    );

  for (const [scale, mean, deviation] of [
    [1e160, 9.995e159, 2.888194360957494e159],
    [1e-200, 9.995e-201, 2.888194360957494e-201],
    [1e300, 9.995e299, 2.888194360957494e299]
  ] as const) {
    it(`keeps the spread of values around ${scale}`, () => {
      const column = profileValues(makeValues(scale));
      expect(Number.isFinite(column.standardDeviation)).toBe(true);
      expect((column.standardDeviation as number) / deviation).toBeCloseTo(1, 12);
      expect((column.mean as number) / mean).toBeCloseTo(1, 12);
      // Not a constant column: it varies, whatever the magnitude.
      expect(column.measurementReason ?? '').not.toMatch(/constant|same value/i);
    });
  }

  it('does not overflow when the squared deviations would', () => {
    const column = profileValues([1e200, -1e200, 1e200, -1e200]);
    expect(column.mean).toBe(0);
    expect((column.standardDeviation as number) / 1.1547005383792515e200).toBeCloseTo(1, 12);
  });

  it('rescales when larger values arrive late and when the first values are zero', () => {
    // /usr/bin/python3: stdev([0, 0, 1, 1e100]) with Fractions = 5.0e99; mean 2.5e99
    const column = profileValues([0, 0, 1, 1e100]);
    expect((column.mean as number) / 2.5e99).toBeCloseTo(1, 12);
    expect((column.standardDeviation as number) / 5e99).toBeCloseTo(1, 12);
    expect(profileValues([0, 0, 0]).standardDeviation).toBe(0);
  });

  it('counts bigints beyond 2^53 exactly and withholds the guess', () => {
    const profile = profileKeplerColumnTable(
      {rowCount: 3, columns: [{name: 'v', values: [2n ** 60n, 2n ** 60n + 1n, 5n]}]},
      BASE_OPTIONS
    );
    const column = getColumn(profile, 'v');
    expect(column.type).toBe('number');
    expect(column.distinctCount).toBe(3);
    expect(column.measurement).toBeUndefined();
    expect(profile.notes?.join(' ')).toMatch(/Column "v" holds integers beyond 2\^53/);
  });

  it('withholds the guess for doubles that are 64 bit identifiers, not for large floats', () => {
    const ids = profileKeplerColumnTable(
      {rowCount: 3, columns: [{name: 'v', values: [2 ** 60, 2 ** 60 + 2 ** 10, 2 ** 61]}]},
      BASE_OPTIONS
    );
    expect(getColumn(ids, 'v').measurement).toBeUndefined();
    expect(ids.notes?.length).toBe(1);
    const floats = profileKeplerColumnTable(
      {rowCount: 3, columns: [{name: 'v', values: [1e160, 2e160, 3e160]}]},
      BASE_OPTIONS
    );
    expect(floats.notes).toBeUndefined();
  });
});

describe('polygon anchors', () => {
  // /usr/bin/python3 with Shapely: Polygon(...).centroid for each case below.
  const anchorOf = (geometry: object) =>
    profileKeplerFeatureCollection(collect([{geometry, properties: {}} as CatalogFeature]), {
      ...BASE_OPTIONS,
      coordinateSystem: 'planar'
    }).spacing?.center as readonly [number, number];
  const ring = (...points: number[][]) => [...points, points[0]];

  it('subtracts holes (Shapely: 5.571428571428571, 5.0)', () => {
    const center = anchorOf({
      type: 'Polygon',
      coordinates: [ring([0, 0], [10, 0], [10, 10], [0, 10]), ring([1, 1], [1, 9], [3, 9], [3, 1])]
    });
    expect(center[0]).toBeCloseTo(5.571428571428571, 12);
    expect(center[1]).toBeCloseTo(5, 12);
  });

  it('adds multipolygon parts of opposite orientation (Shapely: 8.5, 4.5)', () => {
    const center = anchorOf({
      type: 'MultiPolygon',
      coordinates: [
        [ring([0, 0], [10, 0], [10, 10], [0, 10])],
        [ring([20, 0], [20, 5], [25, 5], [25, 0])]
      ]
    });
    expect(center[0]).toBeCloseTo(8.5, 12);
    expect(center[1]).toBeCloseTo(4.5, 12);
  });

  it('weights unequal parts by area (Shapely: 14.856435643564357, 4.955445544554456)', () => {
    const center = anchorOf({
      type: 'MultiPolygon',
      coordinates: [
        [ring([0, 0], [1, 0], [1, 1], [0, 1])],
        [ring([10, 0], [20, 0], [20, 10], [10, 10])]
      ]
    });
    expect(center[0]).toBeCloseTo(14.856435643564357, 12);
    expect(center[1]).toBeCloseTo(4.955445544554456, 12);
  });

  it('handles a concave shape (Shapely: 4.076923076923077, 5.0)', () => {
    const center = anchorOf({
      type: 'Polygon',
      coordinates: [ring([0, 0], [10, 0], [10, 2], [2, 2], [2, 8], [10, 8], [10, 10], [0, 10])]
    });
    expect(center[0]).toBeCloseTo(4.076923076923077, 12);
    expect(center[1]).toBeCloseTo(5, 12);
  });

  it('does not take a small polygon in degrees for a degenerate one (Shapely: 5e-08, 5e-08)', () => {
    const center = anchorOf({
      type: 'Polygon',
      coordinates: [ring([0, 0], [1e-7, 0], [1e-7, 1e-7], [0, 1e-7])]
    });
    expect(center[0]).toBeCloseTo(5e-8, 15);
    expect(center[1]).toBeCloseTo(5e-8, 15);
  });

  it('keeps precision far from the origin (Shapely: 1000005, 1000005)', () => {
    const center = anchorOf({
      type: 'Polygon',
      coordinates: [
        ring([1000000, 1000000], [1000010, 1000000], [1000010, 1000010], [1000000, 1000010])
      ]
    });
    expect(center[0]).toBeCloseTo(1000005, 9);
    expect(center[1]).toBeCloseTo(1000005, 9);
  });

  it('takes the line centroid of a zero-area ring (Shapely: 5.0, 0.0) and an unclosed ring like a closed one', () => {
    const line = anchorOf({
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [5, 0],
          [10, 0],
          [0, 0]
        ]
      ]
    });
    expect(line[0]).toBeCloseTo(5, 12);
    expect(line[1]).toBeCloseTo(0, 12);
    const unclosed = anchorOf({
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10]
        ]
      ]
    });
    expect(unclosed[0]).toBeCloseTo(5, 12);
    expect(unclosed[1]).toBeCloseTo(5, 12);
  });

  it('takes the point itself for a polygon whose vertices all coincide', () => {
    const center = anchorOf({
      type: 'Polygon',
      coordinates: [
        [
          [3, 4],
          [3, 4],
          [3, 4],
          [3, 4]
        ]
      ]
    });
    expect(center).toEqual([3, 4]);
  });
});

describe('geographic extent across the antimeridian', () => {
  const sizesOf = (profile: {
    extent?: readonly [number, number, number, number];
    coordinates?: {model: 'geographic' | 'planar'};
  }) =>
    getGPUAnalysisExtentSizes(
      profile.extent as readonly [number, number, number, number],
      profile.coordinates?.model as 'geographic' | 'planar'
    );
  const shiftLongitude = (position: number[], shift: number) => [position[0] + shift, position[1]];
  const wrap = (longitude: number) => ((((longitude + 180) % 360) + 360) % 360) - 180;
  const lineAt = (positions: number[][], shift: number) =>
    ({
      geometry: {
        type: 'LineString',
        coordinates: positions.map(position => [wrap(position[0] + shift), position[1]])
      },
      properties: {}
    }) as CatalogFeature;
  const polygonAt = (positions: number[][], shift: number) =>
    ({
      geometry: {
        type: 'Polygon',
        coordinates: [
          [...positions, positions[0]].map(position => [wrap(position[0] + shift), position[1]])
        ]
      },
      properties: {}
    }) as CatalogFeature;

  // Wrapped vertices without cutting their edges describe broad geometries; points have no edges.
  const pointPositions = [
    [178, 60],
    [179.5, 62],
    [182, 65],
    [180.5, 70]
  ];
  const linePositions = [
    [178.5, 10],
    [181, 12],
    [182.5, 15]
  ];
  const polygonPositions = [
    [179, -30],
    [181, -30],
    [181, -25],
    [179, -25]
  ];

  it('gives a point set the same extent sizes as the same points at longitude 0', () => {
    const across = profileKeplerFeatureCollection(
      collect(pointPositions.map(position => makePoint(wrap(position[0]), position[1]))),
      BASE_OPTIONS
    );
    const atZero = profileKeplerFeatureCollection(
      collect(
        pointPositions.map(position => makePoint(shiftLongitude(position, -180)[0], position[1]))
      ),
      BASE_OPTIONS
    );
    expect(across.coordinates?.model).toBe('geographic');
    // West is greater than east: the arc from 178 eastwards to -178.
    expect(across.extent).toEqual([178, 60, -178, 70]);
    expect(atZero.extent).toEqual([-2, 60, 2, 70]);
    const acrossSizes = sizesOf(across);
    const zeroSizes = sizesOf(atZero);
    expect(acrossSizes.width).toBeCloseTo(zeroSizes.width, 3);
    expect(acrossSizes.height).toBeCloseTo(zeroSizes.height, 3);
    expect(acrossSizes.diagonal).toBeCloseTo(zeroSizes.diagonal, 3);
    // The spacing agrees with the profile: it measures its own extent the same way.
    expect(across.spacing?.extent).toEqual(across.extent);
    expect(across.spacing?.extentWidth).toBeCloseTo(acrossSizes.width, 3);
  });

  it('preserves the interior longitudes of uncut broad line edges', () => {
    const across = profileKeplerFeatureCollection(
      collect([lineAt(linePositions, 0)]),
      BASE_OPTIONS
    );
    const atZero = profileKeplerFeatureCollection(
      collect([lineAt(linePositions, -180)]),
      BASE_OPTIONS
    );
    expect(across.kind).toBe('lines');
    expect(across.extent).toEqual([-179, 10, 178.5, 15]);
    expect(atZero.extent).toEqual([-1.5, 10, 2.5, 15]);
    expect(sizesOf(across).width).toBeGreaterThan(sizesOf(atZero).width);
  });

  it('preserves the interior of uncut broad polygons rather than wrapping their vertices', () => {
    const across = profileKeplerFeatureCollection(
      collect([polygonAt(polygonPositions, 0)]),
      BASE_OPTIONS
    );
    const atZero = profileKeplerFeatureCollection(
      collect([polygonAt(polygonPositions, -180)]),
      BASE_OPTIONS
    );
    expect(across.kind).toBe('polygons');
    expect(across.extent).toEqual([-179, -30, 179, -25]);
    expect(atZero.extent).toEqual([-1, -30, 1, -25]);
    expect(sizesOf(across).width).toBeGreaterThan(sizesOf(atZero).width);
    expect(sizesOf(across).height).toBeCloseTo(sizesOf(atZero).height, 3);
    // One polygon has one anchor; the extent still covers every vertex.
    expect(across.spacing?.pointCount).toBe(1);
    expect(sizesOf(across).width).toBeGreaterThan(0);
  });

  it('uses all geometry families of a mixed dataset for the extent', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        makePoint(179, 0),
        makePoint(-179, 1),
        lineAt(
          [
            [170, 5],
            [175, 6]
          ],
          0
        )
      ]),
      BASE_OPTIONS
    );
    expect(profile.kind).toBe('points');
    expect(profile.extent).toEqual([170, 0, -179, 6]);
  });

  it('works for points held as columns', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 3,
        columns: [
          {name: 'lng', values: [179, -179.5, 179.5]},
          {name: 'lat', values: [10, 11, 12]}
        ]
      },
      BASE_OPTIONS
    );
    expect(profile.extent).toEqual([179, 10, -179.5, 12]);
  });

  it('keeps minimum and maximum when the data spans more than half the globe', () => {
    // Gaps between sorted longitudes: 120 (50 to 170 is not a gap of data; sorted wrapped
    // longitudes are 50, 170, 190, 300), so the smallest covering arc is 240 degrees.
    const profile = profileKeplerFeatureCollection(
      collect([-170, -60, 50, 170].map(longitude => makePoint(longitude, longitude / 10))),
      BASE_OPTIONS
    );
    expect(profile.extent).toEqual([170, -17, 50, 17]);
    const width = sizesOf(profile).width;
    const expected = getGPUAnalysisExtentSizes([0, -17, 240, 17], 'geographic').width;
    expect(width).toBeCloseTo(expected, 3);
    // And a set whose plain range is already the smallest arc is left alone.
    const plain = profileKeplerFeatureCollection(
      collect([-100, 0, 100].map(longitude => makePoint(longitude, 10))),
      BASE_OPTIONS
    );
    expect(plain.extent).toEqual([-100, 10, 100, 10]);
  });

  it('keeps the plain extent for data around the whole circle', () => {
    // Every 5 degrees: the largest gap is 5, so the arc (355) is not narrower than the plain range.
    const profile = profileKeplerFeatureCollection(
      collect(Array.from({length: 72}, (_, index) => makePoint(-180 + index * 5, 5))),
      BASE_OPTIONS
    );
    expect(profile.extent).toEqual([-180, 5, 175, 5]);
    // With 90 degree gaps the smallest covering arc is 270 degrees, not the plain 359.95.
    const gapped = profileKeplerFeatureCollection(
      collect([-180, -90, 0, 90, 179.95].map(longitude => makePoint(longitude, 5))),
      BASE_OPTIONS
    );
    expect(gapped.extent).toEqual([90, 5, 0, 5]);
  });

  it('is planar min and max when the coordinate system is planar', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(179, 0), makePoint(-179, 1)]),
      {...BASE_OPTIONS, coordinateSystem: 'planar'}
    );
    expect(profile.extent).toEqual([-179, 0, 179, 1]);
  });

  it('is recomputed when a guessed geographic model is declared planar, and back', () => {
    const guessed = profileKeplerFeatureCollection(
      collect([makePoint(179, 0), makePoint(-179, 1), makePoint(178, 2)]),
      BASE_OPTIONS
    );
    expect(guessed.coordinates?.source).toBe('guess');
    expect(guessed.extent).toEqual([178, 0, -179, 2]);
    const planar = applyKeplerDatasetProfileDeclarations(guessed, {coordinateModel: 'planar'});
    expect(planar.extent).toEqual([-179, 0, 179, 2]);
    expect(planar.spacing?.extent).toEqual([-179, 0, 179, 2]);
    const geographic = applyKeplerDatasetProfileDeclarations(planar, {
      coordinateModel: 'geographic'
    });
    expect(geographic.extent).toEqual(guessed.extent);
    // Declared through options, the result is the same.
    const viaOptions = profileKeplerFeatureCollection(
      collect([makePoint(179, 0), makePoint(-179, 1), makePoint(178, 2)]),
      {...BASE_OPTIONS, declarations: {coordinateModel: 'planar'}}
    );
    expect(viaOptions.extent).toEqual([-179, 0, 179, 2]);
  });

  it('notes that a profile without locations keeps an extent that crosses 180 when declared planar', () => {
    const guessed = profileKeplerFeatureCollection(
      collect([makePoint(179, 0), makePoint(-179, 1)]),
      BASE_OPTIONS
    );
    const foreign = JSON.parse(JSON.stringify(guessed));
    const corrected = applyKeplerDatasetProfileDeclarations(foreign, {coordinateModel: 'planar'});
    expect(corrected.extent).toEqual([179, 0, -179, 1]);
    expect(corrected.notes.join(' ')).toMatch(/crosses the antimeridian .* not recomputed/);
  });

  it('survives spacing sampling: the extent comes from every vertex', () => {
    const features = Array.from({length: 100}, (_, index) =>
      makePoint(wrap(170 + index * 0.2), index * 0.1)
    );
    const profile = profileKeplerFeatureCollection(collect(features), {
      ...BASE_OPTIONS,
      maximumSampleSize: 10
    });
    expect(profile.extent?.[0]).toBe(170);
    expect(profile.extent?.[2]).toBeCloseTo(wrap(170 + 99 * 0.2), 9);
    const planar = applyKeplerDatasetProfileDeclarations(profile, {coordinateModel: 'planar'});
    // Planar: minimum and maximum of every vertex, so west is below east.
    expect(planar.extent?.[0]).toBeCloseTo(-180, 9);
    expect(planar.extent?.[2]).toBeCloseTo(179.8, 9);
  });
});

const makeRows = (count: number, make: (index: number) => number[]) =>
  collect(
    Array.from({length: count}, (_, index) => makePoint(make(index)[0], make(index)[1], {n: index}))
  );

describe('confidence of guesses', () => {
  it('stores the confidence of a measurement guess and removes it when a declaration replaces the guess', () => {
    const features = Array.from({length: 40}, (_, index) =>
      makePoint(-87 + index * 0.01, 41, {vessel_id: 200000 + index, count: index % 7})
    );
    const profile = profileKeplerFeatureCollection(collect(features), BASE_OPTIONS);
    const vessel = getColumn(profile, 'vessel_id');
    expect(vessel.measurement).toBe('identifier');
    expect(vessel.measurementConfidence).toBe('high');
    expect(getColumn(profile, 'count').measurementSource).toBe('guess');
    expect(['high', 'low']).toContain(getColumn(profile, 'count').measurementConfidence);
    const declared = applyKeplerDatasetProfileDeclarations(profile, {
      measurement: {vessel_id: 'count'}
    });
    const column = getColumn(declared, 'vessel_id');
    expect(column.measurementSource).toBe('declared');
    expect('measurementConfidence' in column).toBe(false);
    expect('measurementReason' in column).toBe(false);
    // Declared at profile time: the same.
    const viaOptions = profileKeplerFeatureCollection(collect(features), {
      ...BASE_OPTIONS,
      declarations: {measurement: {vessel_id: 'count'}}
    });
    expect('measurementConfidence' in getColumn(viaOptions, 'vessel_id')).toBe(false);
  });

  it('stores the confidence of the coordinate guess and removes it when declared', () => {
    const confident = profileKeplerFeatureCollection(
      makeRows(5, index => [-87.6 + index * 0.01, 41.8]),
      BASE_OPTIONS
    );
    expect(confident.coordinates).toMatchObject({source: 'guess', confidence: 'high'});
    const declared = applyKeplerDatasetProfileDeclarations(confident, {
      coordinateModel: 'geographic'
    });
    expect(declared.coordinates?.source).toBe('declared');
    expect('confidence' in (declared.coordinates ?? {})).toBe(false);
    const viaSystem = profileKeplerFeatureCollection(
      makeRows(5, index => [-87.6 + index, 41.8]),
      {
        ...BASE_OPTIONS,
        coordinateSystem: 'EPSG:4326'
      }
    );
    expect('confidence' in (viaSystem.coordinates ?? {})).toBe(false);
  });

  it('sets rowSemantics from the roles guess, as a low-confidence guess with its reason', () => {
    const gauges = profileKeplerFeatureCollection(
      createKeplerCuratedAnalysisSample('rain-gauges')['points'],
      BASE_OPTIONS
    );
    expect(gauges.rowSemantics).toMatchObject({
      value: 'surface-samples',
      source: 'guess',
      confidence: 'low'
    });
    expect(gauges.rowSemantics?.reason).toBeTruthy();
    const areas = profileKeplerFeatureCollection(
      createKeplerCuratedAnalysisSample('city-district-rates')['areas'],
      BASE_OPTIONS
    );
    expect(areas.rowSemantics).toMatchObject({
      value: 'objects',
      source: 'guess',
      confidence: 'low'
    });
  });

  it('leaves rowSemantics absent when the evidence conflicts or the kind has none', () => {
    // A measured quantity together with an object attribute conflict.
    const conflict = profileKeplerFeatureCollection(
      collect(
        Array.from({length: 20}, (_, index) =>
          makePoint(-87 + index * 0.01, 41, {rainfall: index * 1.5, revenue: index * 100.5})
        )
      ),
      BASE_OPTIONS
    );
    expect(conflict.rowSemantics).toBeUndefined();
    const table = profileKeplerColumnTable(
      {rowCount: 2, columns: [{name: 'a', values: [1, 2]}]},
      BASE_OPTIONS
    );
    expect(table.rowSemantics).toBeUndefined();
  });

  it('replaces the guess with a declaration, which carries no confidence, and the guess returns when it is removed', () => {
    const gauges = createKeplerCuratedAnalysisSample('rain-gauges')['points'];
    const guessed = profileKeplerFeatureCollection(gauges, BASE_OPTIONS);
    const declared = applyKeplerDatasetProfileDeclarations(guessed, {rowSemantics: 'events'});
    expect(declared.rowSemantics).toEqual({
      value: 'events',
      source: 'declared',
      reason: 'Declared by the user.'
    });
    expect('confidence' in (declared.rowSemantics ?? {})).toBe(false);
    // At profile time the same.
    const viaOptions = profileKeplerFeatureCollection(gauges, {
      ...BASE_OPTIONS,
      declarations: {rowSemantics: 'events'}
    });
    expect(viaOptions.rowSemantics).toEqual(declared.rowSemantics);
    // Applying to the profile without the declaration (what the cache keeps) restores the guess.
    expect(getKeplerDatasetProfileUndeclared(viaOptions).rowSemantics).toEqual(
      guessed.rowSemantics
    );
    expect(applyKeplerDatasetProfileDeclarations(guessed, {}).rowSemantics).toEqual(
      guessed.rowSemantics
    );
  });

  it('makes the guess again when a declared measurement changes its evidence, and keeps a declared value', () => {
    const features = Array.from({length: 30}, (_, index) =>
      makePoint(-87 + index * 0.01, 41, {weight: (index % 9) * 1.7 + 1})
    );
    const guessed = profileKeplerFeatureCollection(collect(features), BASE_OPTIONS);
    // `weight` reads as an attribute of a thing at each point.
    expect(getColumn(guessed, 'weight').measurement).toBe('amount');
    expect(guessed.rowSemantics?.value).toBe('objects');
    // Declared an identifier it is no value any more, and the rows read as occurrences.
    const changed = applyKeplerDatasetProfileDeclarations(guessed, {
      measurement: {weight: 'identifier'}
    });
    expect(changed.rowSemantics).toMatchObject({
      value: 'events',
      source: 'guess',
      confidence: 'low'
    });
    // A declared value is never replaced by a new guess.
    const keptDeclared = applyKeplerDatasetProfileDeclarations(
      applyKeplerDatasetProfileDeclarations(guessed, {rowSemantics: 'surface-samples'}),
      {measurement: {weight: 'identifier'}}
    );
    expect(keptDeclared.rowSemantics).toMatchObject({value: 'surface-samples', source: 'declared'});
  });

  it('describes guessed row semantics as a guess and declared ones plainly', () => {
    const gauges = createKeplerCuratedAnalysisSample('rain-gauges')['points'];
    const guessed = describeKeplerDatasetProfile(
      profileKeplerFeatureCollection(gauges, BASE_OPTIONS)
    );
    expect(guessed.headline).toContain('rows probably surface-samples (guessed, low confidence)');
    expect(guessed.rowSemantics).toEqual({
      value: 'surface-samples',
      guessed: true,
      confidence: 'low'
    });
    const declared = describeKeplerDatasetProfile(
      profileKeplerFeatureCollection(gauges, {
        ...BASE_OPTIONS,
        declarations: {rowSemantics: 'events'}
      })
    );
    expect(declared.headline).toContain('rows are events');
    expect(declared.rowSemantics).toMatchObject({value: 'events', guessed: false});
  });
});

describe('mixed geometry for the catalog', () => {
  it('sets mixedGeometry with the per-family row counts and keeps the dominant catalogGeometry', () => {
    const features = [
      ...Array.from({length: 300}, (_, index) => makePoint(-87 + index * 0.001, 41)),
      ...Array.from(
        {length: 50},
        (_, index) =>
          ({
            geometry: {
              type: 'LineString',
              coordinates: [
                [-87, 41 + index * 0.001],
                [-86.9, 41.1]
              ]
            },
            properties: {}
          }) as CatalogFeature
      ),
      ...Array.from(
        {length: 20},
        () =>
          ({
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 0]
                ]
              ]
            },
            properties: {}
          }) as CatalogFeature
      )
    ];
    const profile = profileKeplerFeatureCollection(collect(features), BASE_OPTIONS);
    expect(profile.mixedGeometry).toEqual({point: 300, line: 50, polygon: 20});
    expect(profile.catalogGeometry).toBe('point');
    expect(profile.notes?.[0]).toMatch(/must be split or filtered to one kind/);
    expect(profile.notes?.[0]).not.toMatch(/ignored by the geometry analyses/);
    expect(toKeplerRecommendationSubject(profile).kind).toBe('features');
  });

  it('leaves mixedGeometry absent for one family, with or without rows lacking geometry', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(1, 1), {geometry: null, properties: {}} as CatalogFeature]),
      BASE_OPTIONS
    );
    expect(profile.mixedGeometry).toBeUndefined();
  });
});

describe('longitudes from 0 to 360', () => {
  const buoys = collect(
    Array.from({length: 30}, (_, index) => makePoint(150 + index * (100 / 29), -10 + index * 0.5))
  );

  it('says so in the reason and in a note, and keeps the stored values', () => {
    const profile = profileKeplerFeatureCollection(buoys, BASE_OPTIONS);
    expect(profile.coordinates).toMatchObject({
      model: 'geographic',
      source: 'guess',
      confidence: 'low'
    });
    expect(profile.coordinates?.reason).toMatch(/longitudes look like 0 to 360 values/);
    expect(profile.notes?.join(' ')).toMatch(/the catalog expects -180 to 180/);
    expect(profile.extent).toEqual([150, -10, 250, 4.5]);
    const sizes = getGPUAnalysisExtentSizes(
      profile.extent as [number, number, number, number],
      'geographic'
    );
    // 100 degrees of longitude at about -3 degrees latitude: roughly 11,100 km.
    expect(sizes.width).toBeGreaterThan(1.05e7);
    expect(sizes.width).toBeLessThan(1.15e7);
  });

  it('wraps across 360 for the arc, not for the stored coordinates', () => {
    const profile = profileKeplerFeatureCollection(
      collect([makePoint(350, 0), makePoint(10, 5), makePoint(355, 2)]),
      BASE_OPTIONS
    );
    // Sorted modulo 360 (10, 350, 355): the largest gap is 10 to 350, so the arc is 350 to 10.
    expect(profile.extent).toEqual([-10, 0, 10, 5]);
  });

  it('says nothing about 0 to 360 for ordinary longitudes', () => {
    const profile = profileKeplerFeatureCollection(
      makeRows(5, index => [-87.6 + index * 0.01, 41.8]),
      BASE_OPTIONS
    );
    expect(profile.notes).toBeUndefined();
    expect(profile.coordinates?.reason).not.toMatch(/0 to 360 values/);
  });
});

describe('local metres guessed as degrees', () => {
  const site = collect(
    Array.from({length: 300}, (_, index) =>
      makePoint((index * 7) % 51, (index * 13) % 51, {tree: index})
    )
  );

  it('notes that small extents near the origin may be local metres', () => {
    const profile = profileKeplerFeatureCollection(site, BASE_OPTIONS);
    expect(profile.coordinates).toMatchObject({
      model: 'geographic',
      source: 'guess',
      confidence: 'low'
    });
    expect(profile.notes?.join(' ')).toMatch(/small extents near the origin may be local metres/);
  });

  it('shows the guess and its confidence in the description, and not when declared or confident', () => {
    const guessed = describeKeplerDatasetProfile(
      profileKeplerFeatureCollection(site, BASE_OPTIONS)
    );
    expect(guessed.headline).toContain('geographic coordinates (guessed, low confidence)');
    expect(guessed.coordinateModel).toEqual({
      model: 'geographic',
      guessed: true,
      confidence: 'low'
    });
    const declared = describeKeplerDatasetProfile(
      profileKeplerFeatureCollection(site, {...BASE_OPTIONS, coordinateSystem: 'planar'})
    );
    expect(declared.headline).toContain('planar coordinates ·');
    expect(declared.headline).not.toContain('coordinates (guessed');
    expect(declared.coordinateModel).toMatchObject({guessed: false});
    const confident = describeKeplerDatasetProfile(
      profileKeplerFeatureCollection(
        makeRows(5, index => [-87.6 + index * 0.01, 41.8]),
        BASE_OPTIONS
      )
    );
    expect(confident.headline).toContain('geographic coordinates');
    expect(confident.headline).not.toContain('coordinates (guessed');
  });

  it('shows a low-confidence guess for point columns too', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 3,
        columns: [
          {name: 'lng', values: [0, 10, 20]},
          {name: 'lat', values: [0, 10, 20]}
        ]
      },
      BASE_OPTIONS
    );
    expect(describeKeplerDatasetProfile(profile).headline).toContain(
      'lng/lat (geographic guessed, low confidence)'
    );
  });
});

describe('review regressions: model declarations and accepted distinct values', () => {
  it('uses the same hashed rows and full-data statistics after declaring a sampled ordered track planar', () => {
    // Exact seed and normal generator from the reviewer's 100,000-location track probe.
    let state = 2;
    const random = () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    let horizontalPosition = 0;
    let verticalPosition = 0;
    const locations: number[] = [];
    const longitudes: number[] = [];
    const latitudes: number[] = [];
    for (let index = 0; index < 100_000; index++) {
      horizontalPosition += 1 + normal() * 0.1;
      verticalPosition += normal() * 0.2;
      longitudes.push(horizontalPosition / 1e6);
      latitudes.push(verticalPosition / 1e6);
      locations.push(horizontalPosition / 1e6, verticalPosition / 1e6);
    }
    const profile = profileKeplerColumnTable(
      {
        rowCount: longitudes.length,
        columns: [
          {name: 'lon', values: longitudes},
          {name: 'lat', values: latitudes}
        ]
      },
      {...BASE_OPTIONS, maximumSampleSize: 5000}
    );
    const planar = applyKeplerDatasetProfileDeclarations(profile, {coordinateModel: 'planar'});
    const reference = computeGPUAnalysisSpacingStatistics(locations, {
      coordinateModel: 'planar',
      maximumSampleSize: 5000
    });
    expect(planar.spacing).toEqual(reference);
    expect(planar.spacing).toMatchObject({pointCount: 100_000, sampleSize: 5000, sampled: true});
    expect(
      applyKeplerDatasetProfileDeclarations(planar, {coordinateModel: 'geographic'}).spacing
    ).toEqual(profile.spacing);
  });

  it('counts only model-valid locations when a sampled planar profile becomes geographic', () => {
    const profile = profileKeplerColumnTable(
      {
        rowCount: 40,
        columns: [
          {name: 'lon', values: Array.from({length: 40}, (_, index) => index)},
          {name: 'lat', values: Array.from({length: 40}, (_, index) => (index < 30 ? index : 100))}
        ]
      },
      {
        ...BASE_OPTIONS,
        coordinateSystem: 'planar',
        maximumSampleSize: 8,
        pointColumns: {longitude: 'lon', latitude: 'lat'}
      }
    );
    const geographic = applyKeplerDatasetProfileDeclarations(profile, {
      coordinateModel: 'geographic'
    });
    expect(geographic.spacing).toMatchObject({pointCount: 30, sampled: true, sampleSize: 8});
    expect(geographic.notes?.join(' ')).toContain('10 of 40 locations');
    expect(
      applyKeplerDatasetProfileDeclarations(geographic, {coordinateModel: 'planar'}).spacing
    ).toEqual(profile.spacing);
  });

  it.each([
    {
      name: 'bigints and text',
      values: [2n ** 60n, 2n ** 60n + 1n, 'x'],
      type: undefined,
      distinctCount: 3,
      nullCount: 0
    },
    {
      name: 'repeated bigints and their text',
      values: [2n ** 60n, 2n ** 60n, '1152921504606846976'],
      type: undefined,
      distinctCount: 2,
      nullCount: 0
    },
    {
      name: 'bigints declared time',
      values: [2n ** 60n, 2n ** 60n + 1n, 'garbage'],
      type: 'time',
      distinctCount: 2,
      nullCount: 1
    },
    {
      name: 'unparseable declared times',
      values: ['2020-01-01', '2020-01-01', 'garbage', '2020-01-02'],
      type: 'time',
      distinctCount: 2,
      nullCount: 1
    },
    {
      name: 'impossible and unparseable times',
      values: ['2020-02-30', 'garbage', '2020-01-01', '2020-01-01', null],
      type: 'time',
      distinctCount: 1,
      nullCount: 3
    },
    {
      name: 'only rejected times',
      values: ['garbage', '2020-02-30'],
      type: 'time',
      distinctCount: undefined,
      nullCount: 2
    },
    {
      name: 'bigints rejected by text declaration',
      values: [2n ** 60n, 'x', 'x'],
      type: 'text',
      distinctCount: 1,
      nullCount: 1
    },
    {
      name: 'non-finite bigints',
      values: [10n ** 400n, 2n ** 60n, 'x'],
      type: undefined,
      distinctCount: 2,
      nullCount: 1
    }
  ] as const)('counts accepted values for $name', ({values, type, distinctCount, nullCount}) => {
    const profile = profileKeplerColumnTable(
      {rowCount: values.length, columns: [{name: 'value', values, type}]},
      BASE_OPTIONS
    );
    expect(profile.columns[0].distinctCount).toBe(distinctCount);
    expect(profile.columns[0].nullCount).toBe(nullCount);
  });
});

describe('review regressions: geographic geometry edges', () => {
  function profileGeometry(geometry: CatalogGeometry) {
    return profileKeplerFeatureCollection(collect([{geometry, properties: {}} as CatalogFeature]), {
      ...BASE_OPTIONS,
      coordinateSystem: 'EPSG:4326'
    });
  }

  it('covers the centre and all edges of the reviewer rectangle from -170 to 170', () => {
    const profile = profileGeometry({
      type: 'Polygon',
      coordinates: [
        [
          [-170, -10],
          [170, -10],
          [170, 10],
          [-170, 10],
          [-170, -10]
        ]
      ]
    });
    expect(profile.extent).toEqual([-170, -10, 170, 10]);
    expect(profile.spacing?.center).toEqual([0, 0]);
    const planar = applyKeplerDatasetProfileDeclarations(profile, {coordinateModel: 'planar'});
    expect(
      applyKeplerDatasetProfileDeclarations(planar, {coordinateModel: 'geographic'}).extent
    ).toEqual(profile.extent);
  });

  it('wraps a true antimeridian polygon cut into parts as RFC 7946 section 3.1.9 recommends', () => {
    const profile = profileGeometry({
      type: 'MultiPolygon',
      coordinates: [
        [
          [
            [170, -10],
            [180, -10],
            [180, 10],
            [170, 10],
            [170, -10]
          ]
        ],
        [
          [
            [-180, -10],
            [-170, -10],
            [-170, 10],
            [-180, 10],
            [-180, -10]
          ]
        ]
      ]
    });
    expect(profile.extent).toEqual([170, -10, -170, 10]);
  });

  it('distinguishes a broad uncut line from a crossing line cut at the antimeridian', () => {
    expect(
      profileGeometry({
        type: 'LineString',
        coordinates: [
          [-170, 0],
          [170, 10]
        ]
      }).extent
    ).toEqual([-170, 0, 170, 10]);
    expect(
      profileGeometry({
        type: 'MultiLineString',
        coordinates: [
          [
            [170, 0],
            [180, 5]
          ],
          [
            [-180, 5],
            [-170, 10]
          ]
        ]
      }).extent
    ).toEqual([170, 0, -170, 10]);
  });

  it('covers edge interiors even when no vertex is inside a gap chosen by the point-set algorithm', () => {
    expect(
      profileGeometry({
        type: 'LineString',
        coordinates: [
          [-120, 0],
          [0, 1],
          [120, 2]
        ]
      }).extent
    ).toEqual([-120, 0, 120, 2]);
    expect(
      profileGeometry({
        type: 'LineString',
        coordinates: [
          [-180, 0],
          [180, 1]
        ]
      }).extent
    ).toEqual([-180, 0, 180, 1]);
  });
});

describe('review regressions: ring degeneracy against Shapely', () => {
  // Reviewer fixtures (24) plus ten additional thin, translated, holed and multipart geometries.
  // Oracle command, with the geometries below as a JSON object on stdin:
  // /usr/bin/python3 -c 'import json,sys; from shapely.geometry import shape; print(json.dumps({name: [shape(geometry).centroid.x, shape(geometry).centroid.y] for name, geometry in json.load(sys.stdin).items()}))'
  const cases: {name: string; geometry: CatalogGeometry; expected: readonly [number, number]}[] = [
    {
      name: 'squareHole',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0]
          ],
          [
            [1, 1],
            [1, 9],
            [3, 9],
            [3, 1],
            [1, 1]
          ]
        ]
      },
      expected: [5.571428571428571, 5.0]
    },
    {
      name: 'oppositeParts',
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [10, 0],
              [10, 10],
              [0, 10],
              [0, 0]
            ]
          ],
          [
            [
              [20, 0],
              [20, 5],
              [25, 5],
              [25, 0],
              [20, 0]
            ]
          ]
        ]
      },
      expected: [8.5, 4.5]
    },
    {
      name: 'lShape',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0, 10],
            [2, 10],
            [2, 2],
            [10, 2],
            [10, 0],
            [0, 0]
          ]
        ]
      },
      expected: [3.2222222222222223, 3.2222222222222223]
    },
    {
      name: 'cShape',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 2],
            [2, 2],
            [2, 8],
            [10, 8],
            [10, 10],
            [0, 10],
            [0, 0]
          ]
        ]
      },
      expected: [4.076923076923077, 5.0]
    },
    {
      name: 'bowtie',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [4, 4],
            [0, 4],
            [4, 0],
            [0, 0]
          ]
        ]
      },
      expected: [2.0, 2.0]
    },
    {
      name: 'touching',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2],
            [0, 0],
            [-2, 0],
            [-2, -2],
            [0, -2],
            [0, 0]
          ]
        ]
      },
      expected: [-0.0, -0.0]
    },
    {
      name: 'tiny',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1e-7, 0],
            [1e-7, 1e-7],
            [0, 1e-7],
            [0, 0]
          ]
        ]
      },
      expected: [4.999999999999999e-8, 4.999999999999999e-8]
    },
    {
      name: 'tinyOffset',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-87.6, 41.8],
            [-87.5999999, 41.8],
            [-87.5999999, 41.8000002],
            [-87.6, 41.8000002],
            [-87.6, 41.8]
          ]
        ]
      },
      expected: [-87.59999995, 41.8000001]
    },
    {
      name: 'farOffset',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [1000000.0, 1000000.0],
            [1000010.0, 1000000.0],
            [1000010.0, 1000010.0],
            [1000000.0, 1000010.0],
            [1000000.0, 1000000.0]
          ]
        ]
      },
      expected: [1000005.0, 1000005.0]
    },
    {
      name: 'collapsed',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [5, 0],
            [10, 0],
            [0, 0]
          ]
        ]
      },
      expected: [5.0, 0.0]
    },
    {
      name: 'unclosed',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
          ]
        ]
      },
      expected: [5.0, 5.0]
    },
    {
      name: 'spacedSmallParts',
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [1, 0],
              [1, 1],
              [0, 1],
              [0, 0]
            ]
          ],
          [
            [
              [10000000.0, 0],
              [10000002.0, 0],
              [10000002.0, 2],
              [10000000.0, 2],
              [10000000.0, 0]
            ]
          ]
        ]
      },
      expected: [8000000.9, 0.9]
    },
    {
      name: 'narrowTriangle',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1e-13],
            [0, 0]
          ]
        ]
      },
      expected: [0.6666666666666666, 3.3333333333333334e-14]
    },
    {
      name: 'asymmetricHole',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0]
          ],
          [
            [1, 1],
            [3, 1],
            [3, 7],
            [1, 7],
            [1, 1]
          ]
        ]
      },
      expected: [5.409090909090909, 5.136363636363637]
    },
    {
      name: 'twoHoles',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [20, 0],
            [20, 20],
            [0, 20],
            [0, 0]
          ],
          [
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
            [1, 1]
          ],
          [
            [10, 10],
            [14, 10],
            [14, 14],
            [10, 14],
            [10, 10]
          ]
        ]
      },
      expected: [10.0, 10.0]
    },
    {
      name: 'pointDegenerate',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [3, 4],
            [3, 4],
            [3, 4],
            [3, 4]
          ]
        ]
      },
      expected: [3.0, 4.0]
    },
    {
      name: 'scaledTriangle0',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0.2, 0],
            [0, 0.1],
            [0, 0]
          ]
        ]
      },
      expected: [0.06666666666666668, 0.03333333333333334]
    },
    {
      name: 'scaledTriangle1',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0.02, 0],
            [0, 0.01],
            [0, 0]
          ]
        ]
      },
      expected: [0.006666666666666668, 0.003333333333333334]
    },
    {
      name: 'scaledTriangle2',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0.002, 0],
            [0, 0.001],
            [0, 0]
          ]
        ]
      },
      expected: [0.0006666666666666668, 0.0003333333333333334]
    },
    {
      name: 'scaledTriangle3',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0.0002, 0],
            [0, 0.0001],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666667e-5, 3.3333333333333335e-5]
    },
    {
      name: 'scaledTriangle4',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2e-5, 0],
            [0, 1e-5],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666667e-6, 3.3333333333333333e-6]
    },
    {
      name: 'scaledTriangle5',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2e-6, 0],
            [0, 1e-6],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666666e-7, 3.333333333333333e-7]
    },
    {
      name: 'scaledTriangle6',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2e-7, 0],
            [0, 1e-7],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666665e-8, 3.333333333333333e-8]
    },
    {
      name: 'scaledTriangle7',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2e-8, 0],
            [0, 1e-8],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666667e-9, 3.3333333333333334e-9]
    },
    {
      name: 'veryThinHorizontal',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [3, 0],
            [3, 1e-20],
            [0, 0]
          ]
        ]
      },
      expected: [2.0, 3.333333333333333e-21]
    },
    {
      name: 'veryThinVertical',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [0, 3],
            [1e-20, 3],
            [0, 0]
          ]
        ]
      },
      expected: [3.333333333333333e-21, 2.0]
    },
    {
      name: 'negativeThinTriangle',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-4, -2],
            [-1, -2],
            [-1, -1.999999999999],
            [-4, -2]
          ]
        ]
      },
      expected: [-2.0, -1.9999999999996665]
    },
    {
      name: 'reversedThinTriangle',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 1e-15],
            [1, 0],
            [0, 0]
          ]
        ]
      },
      expected: [0.6666666666666666, 3.3333333333333336e-16]
    },
    {
      name: 'distantUnequalParts',
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [-1000000000.0, -10],
              [-999999998.0, -10],
              [-999999998.0, -7],
              [-1000000000.0, -7],
              [-1000000000.0, -10]
            ]
          ],
          [
            [
              [1000000000.0, 10],
              [1000000004.0, 10],
              [1000000004.0, 15],
              [1000000000.0, 15],
              [1000000000.0, 10]
            ]
          ]
        ]
      },
      expected: [538461540.2307693, 7.653846153846154]
    },
    {
      name: 'distantWithHole',
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [10, 0],
              [10, 10],
              [0, 10],
              [0, 0]
            ],
            [
              [1, 1],
              [3, 1],
              [3, 3],
              [1, 3],
              [1, 1]
            ]
          ],
          [
            [
              [100000000.0, 0],
              [100000001.0, 0],
              [100000001.0, 1],
              [100000000.0, 1],
              [100000000.0, 0]
            ]
          ]
        ]
      },
      expected: [1030932.912371134, 5.077319587628866]
    },
    {
      name: 'rotatedSliver',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 10],
            [10, 10.0000000001],
            [0, 0]
          ]
        ]
      },
      expected: [6.666666666666667, 6.6666666667]
    },
    {
      name: 'horizontalThinHole',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [5, 0],
            [5, 1e-15],
            [0, 1e-15],
            [0, 0]
          ],
          [
            [1, 1e-16],
            [2, 1e-16],
            [2, 3e-16],
            [1, 3e-16],
            [1, 1e-16]
          ]
        ]
      },
      expected: [2.5416666666666665, 5.125000000000001e-16]
    },
    {
      name: 'largeTranslatedTriangle',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [1000000000000.0, 1000000000000.0],
            [1000000000002.0, 1000000000000.0],
            [1000000000000.0, 1000000000001.0],
            [1000000000000.0, 1000000000000.0]
          ]
        ]
      },
      expected: [1000000000000.6666, 1000000000000.3334]
    },
    {
      name: 'zeroPartAndSquare',
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [1, 0],
              [2, 0],
              [0, 0]
            ]
          ],
          [
            [
              [100, 100],
              [102, 100],
              [102, 102],
              [100, 102],
              [100, 100]
            ]
          ]
        ]
      },
      expected: [101.0, 101.0]
    }
  ];

  it.each(cases)('matches the Shapely centroid of $name', ({geometry, expected}) => {
    const profile = profileKeplerFeatureCollection(
      collect([{geometry, properties: {}} as CatalogFeature]),
      {
        ...BASE_OPTIONS,
        coordinateSystem: 'planar'
      }
    );
    const actual = profile.spacing?.center;
    expect(actual).toBeDefined();
    for (let axis = 0; axis < 2; axis++) {
      // A relative tolerance also checks the tiny axis of thin triangles, rather than masking it
      // behind an absolute tolerance of 1e-12 or the much larger coordinate on the other axis.
      const tolerance = Math.max(Number.MIN_VALUE, Math.abs(expected[axis]) * 1e-12);
      expect(
        Math.abs((actual as readonly [number, number])[axis] - expected[axis])
      ).toBeLessThanOrEqual(tolerance);
    }
  });
});

describe('verification regressions: anchors across antimeridian cuts', () => {
  // Exact rational midpoint: (178.25 + 183.5) / 2 = 180.875, wrapped to -179.125.
  const rectangle = (west: number, south: number, width: number, height: number) =>
    [
      [west, south],
      [west + width, south],
      [west + width, south + height],
      [west, south + height],
      [west, south]
    ] as [number, number][];
  const cases: {name: string; latitude: number; uncut: CatalogGeometry; cut: CatalogGeometry}[] = [
    {
      name: 'polygon',
      latitude: 7,
      uncut: {type: 'Polygon', coordinates: [rectangle(178.25, 5, 5.25, 4)]},
      cut: {
        type: 'MultiPolygon',
        coordinates: [[rectangle(178.25, 5, 1.75, 4)], [rectangle(-180, 5, 3.5, 4)]]
      }
    },
    {
      name: 'line with an existing seam vertex',
      latitude: 7,
      uncut: {
        type: 'LineString',
        coordinates: [
          [178.25, 7],
          [180, 7],
          [183.5, 7]
        ]
      },
      cut: {
        type: 'MultiLineString',
        coordinates: [
          [
            [178.25, 7],
            [180, 7]
          ],
          [
            [-180, 7],
            [-176.5, 7]
          ]
        ]
      }
    },
    {
      name: 'line with an added seam vertex',
      latitude: 8,
      uncut: {
        type: 'LineString',
        coordinates: [
          [178.25, 5],
          [183.5, 11]
        ]
      },
      cut: {
        type: 'MultiLineString',
        coordinates: [
          [
            [178.25, 5],
            [180, 7]
          ],
          [
            [-180, 7],
            [-176.5, 11]
          ]
        ]
      }
    }
  ];

  it.each(cases)('keeps the $name anchor when cutting or reversing parts', ({
    latitude,
    uncut,
    cut
  }) => {
    const profile = (geometry: CatalogGeometry) =>
      profileKeplerFeatureCollection(collect([{geometry, properties: {}} as CatalogFeature]), {
        ...BASE_OPTIONS,
        coordinateSystem: 'EPSG:4326'
      });
    const direct = profile(uncut);
    const split = profile(cut);
    const reversed = profile({
      ...cut,
      coordinates: [...(cut?.coordinates as unknown[])].reverse()
    } as CatalogGeometry);
    for (const result of [direct, split, reversed]) {
      expect(result.spacing?.center[0]).toBeCloseTo(-179.125, 12);
      expect(result.spacing?.center[1]).toBeCloseTo(latitude, 12);
    }
    // Changing declarations must select the anchors computed for that model.
    const planar = applyKeplerDatasetProfileDeclarations(split, {coordinateModel: 'planar'});
    expect(planar.spacing?.center[0]).toBeGreaterThan(-100);
    expect(
      applyKeplerDatasetProfileDeclarations(planar, {coordinateModel: 'geographic'}).spacing
    ).toEqual(split.spacing);
  });

  it('keeps the literal interior of a broad uncut polygon and line', () => {
    for (const geometry of [
      {type: 'Polygon', coordinates: [rectangle(-162, 5, 316, 4)]},
      {
        type: 'LineString',
        coordinates: [
          [-162, 5],
          [154, 9]
        ]
      }
    ] as CatalogGeometry[]) {
      const profile = profileKeplerFeatureCollection(
        collect([{geometry, properties: {}} as CatalogFeature]),
        {...BASE_OPTIONS, coordinateSystem: 'EPSG:4326'}
      );
      expect(profile.spacing?.center[0]).toBeCloseTo(-4, 12);
      expect(profile.spacing?.center[1]).toBeCloseTo(7, 12);
      expect(profile.extent).toEqual([-162, 5, 154, 9]);
    }
  });
});

describe('verification regressions: scaled polygon centroid moments', () => {
  it('keeps finite centroids when finite endpoint differences overflow', () => {
    // /usr/bin/python3: float(-fractions.Fraction.from_float(1e308) / 3)
    const expected = -3.333333333333333e307;
    const vertices: [number, number][] = [
      [-1e308, -1e308],
      [1e308, -1e308],
      [-1e308, 1e308]
    ];
    for (const ring of [vertices, [...vertices].reverse()]) {
      const profile = profileKeplerFeatureCollection(
        collect([
          {
            geometry: {type: 'Polygon', coordinates: [[...ring, ring[0]]]},
            properties: {}
          } as CatalogFeature
        ]),
        {...BASE_OPTIONS, coordinateSystem: 'planar'}
      );
      expect(profile.spacing).toBeDefined();
      expect(profile.spacing!.center[0] / expected).toBeCloseTo(1, 13);
      expect(profile.spacing!.center[1] / expected).toBeCloseTo(1, 13);
    }
  });

  // Exact oracle: /usr/bin/python3, fractions.Fraction.from_float on each scaled coordinate.
  // For each ring: cross = sum(x*v-u*y); centerX = sum((x+u)*(x*v-u*y))/(3*cross),
  // likewise centerY. Sum abs(cross)*center for shells, subtract it for holes, divide by
  // the signed sum of abs(cross), then float() once. No floating area/moment products.
  const cases = [
    {
      scale: 1e-150,
      triangle: [6.666666666666667e-151, 3.3333333333333334e-151],
      translated: [3.6666666666666666e-150, -4.6666666666666664e-150],
      multipart: [5.374999999999999e-150, 2.8750000000000005e-150]
    },
    {
      scale: 1e-120,
      triangle: [6.666666666666666e-121, 3.333333333333333e-121],
      translated: [3.666666666666666e-120, -4.6666666666666665e-120],
      multipart: [5.375e-120, 2.8749999999999996e-120]
    },
    {
      scale: 1,
      triangle: [0.6666666666666666, 0.3333333333333333],
      translated: [3.6666666666666665, -4.666666666666667],
      multipart: [5.375, 2.875]
    },
    {
      scale: 1e120,
      triangle: [6.666666666666667e119, 3.333333333333333e119],
      translated: [3.666666666666667e120, -4.6666666666666666e120],
      multipart: [5.375e120, 2.8750000000000004e120]
    },
    {
      scale: 1e150,
      triangle: [6.666666666666666e149, 3.333333333333333e149],
      translated: [3.666666666666666e150, -4.6666666666666665e150],
      multipart: [5.375e150, 2.8749999999999997e150]
    },
    // A representable centroid does not require a representable unscaled area either.
    {
      scale: 1e-200,
      triangle: [6.666666666666667e-201, 3.3333333333333335e-201],
      translated: [3.666666666666667e-200, -4.666666666666667e-200],
      multipart: [5.374999999999999e-200, 2.875e-200]
    },
    {
      scale: 1e200,
      triangle: [6.666666666666667e199, 3.3333333333333334e199],
      translated: [3.666666666666667e200, -4.666666666666666e200],
      multipart: [5.375e200, 2.8749999999999998e200]
    }
  ];
  it.each(cases)('matches exact rational centroids at scale $scale', ({
    scale,
    triangle,
    translated,
    multipart
  }) => {
    const ring = (vertices: number[][]): [number, number][] =>
      [...vertices, vertices[0]].map(([x, y]) => [x * scale, y * scale]);
    const geometries: {geometry: CatalogGeometry; expected: number[]}[] = [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            ring([
              [0, 0],
              [2, 0],
              [0, 1]
            ])
          ]
        },
        expected: triangle
      },
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            ring([
              [3, -5],
              [5, -5],
              [3, -4]
            ])
          ]
        },
        expected: translated
      },
      {
        geometry: {
          type: 'MultiPolygon',
          coordinates: [
            [
              ring([
                [0, 0],
                [8, 0],
                [8, 6],
                [0, 6]
              ]),
              ring([
                [1, 1],
                [3, 1],
                [3, 4],
                [1, 4]
              ])
            ],
            [
              ring([
                [12, 0],
                [12, 3],
                [14, 3],
                [14, 0]
              ])
            ]
          ]
        },
        expected: multipart
      }
    ];
    for (const {geometry, expected} of geometries) {
      const profile = profileKeplerFeatureCollection(
        collect([{geometry, properties: {}} as CatalogFeature]),
        {...BASE_OPTIONS, coordinateSystem: 'planar'}
      );
      expect(profile.spacing).toBeDefined();
      for (const axis of [0, 1]) {
        expect(profile.spacing!.center[axis] / expected[axis]).toBeCloseTo(1, 13);
      }
    }
  });
});

describe('shared exact covering-arc regressions', () => {
  it('retains a gap smaller than the former 0.1-degree buckets across model changes', () => {
    const profile = profileKeplerFeatureCollection(
      collect([
        {
          geometry: {
            type: 'LineString',
            coordinates: [
              [0.08, -5],
              [180, 5]
            ]
          },
          properties: {}
        },
        {
          geometry: {
            type: 'LineString',
            coordinates: [
              [-180, 0],
              [0.02, 1]
            ]
          },
          properties: {}
        }
      ] as CatalogFeature[]),
      {...BASE_OPTIONS, coordinateSystem: 'EPSG:4326'}
    );
    expect(profile.extent?.[0]).toBeCloseTo(0.08, 12);
    expect(profile.extent?.[2]).toBeCloseTo(0.02, 12);
    const planar = applyKeplerDatasetProfileDeclarations(profile, {coordinateModel: 'planar'});
    expect(planar.extent).toEqual([-180, -5, 180, 5]);
    expect(
      applyKeplerDatasetProfileDeclarations(planar, {coordinateModel: 'geographic'}).extent
    ).toEqual(profile.extent);
  });
});
