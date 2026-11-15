// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogFeature} from '../../src/analysis/catalog-geojson';
import {
  createKeplerDatasetProfileCache,
  parseKeplerDatasetProfileDeclarations,
  serializeKeplerDatasetProfileDeclarations
} from '../../src/profile/dataset-profile-cache';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

function makeProfile(datasetId: string, revision?: number) {
  const features = [0, 1, 2, 3].map(
    index =>
      ({
        geometry: {type: 'Point', coordinates: [-87.6 + index * 0.01, 41.8]},
        properties: {cases: index + 3, rate: index / 10}
      }) as CatalogFeature
  );
  return profileKeplerFeatureCollection(
    {type: 'FeatureCollection', features},
    {datasetId, label: datasetId, revision}
  );
}

function getMeasurement(
  profile:
    | {columns: readonly {name: string; measurement?: string; measurementSource?: string}[]}
    | undefined,
  name: string
) {
  return profile?.columns.find(column => column.name === name);
}

describe('createKeplerDatasetProfileCache', () => {
  it('returns the stored profile for its revision and nothing for another', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 4});
    const profile = makeProfile('a', 1);
    cache.set(profile);
    expect(cache.get('a', 1)).toBe(profile);
    expect(cache.get('a', 2)).toBeUndefined();
    expect(cache.get('a')).toBeUndefined();
    expect(cache.get('missing', 1)).toBeUndefined();
    const unversioned = makeProfile('b');
    cache.set(unversioned);
    expect(cache.get('b')).toBe(unversioned);
  });

  it('evicts the least recently used entry', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    cache.set(makeProfile('a', 1));
    cache.set(makeProfile('b', 1));
    // Reading `a` makes `b` the oldest.
    expect(cache.get('a', 1)).toBeDefined();
    cache.set(makeProfile('c', 1));
    expect(cache.size).toBe(2);
    expect(cache.get('b', 1)).toBeUndefined();
    expect(cache.get('a', 1)).toBeDefined();
    expect(cache.get('c', 1)).toBeDefined();
  });

  it('replaces the entry of a dataset on a revision bump instead of growing', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    cache.set(makeProfile('a', 1));
    cache.set(makeProfile('a', 2));
    expect(cache.size).toBe(1);
    expect(cache.get('a', 1)).toBeUndefined();
    expect(cache.get('a', 2)).toBeDefined();
  });

  it('invalidates one dataset only, and keeps its declarations', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    cache.setDeclarations('a', {measurement: {cases: 'amount'}});
    cache.set(makeProfile('a', 1));
    cache.set(makeProfile('b', 1));
    cache.invalidate('a');
    expect(cache.get('a', 1)).toBeUndefined();
    expect(cache.get('b', 1)).toBeDefined();
    expect(cache.getDeclarations('a')).toEqual({measurement: {cases: 'amount'}});
    cache.invalidate('never-seen');
    const again = cache.set(makeProfile('a', 2));
    expect(getMeasurement(again, 'cases')?.measurement).toBe('amount');
  });

  it('applies declarations stored before and after a profile, and they survive a revision bump', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    const first = cache.set(makeProfile('a', 1));
    expect(getMeasurement(first, 'cases')?.measurementSource).toBe('guess');
    const corrected = cache.setDeclarations('a', {
      measurement: {cases: 'amount'},
      rowSemantics: 'objects'
    });
    expect(getMeasurement(corrected, 'cases')).toMatchObject({
      measurement: 'amount',
      measurementSource: 'declared'
    });
    expect(cache.get('a', 1)).toBe(corrected);
    const second = cache.set(makeProfile('a', 2));
    expect(getMeasurement(second, 'cases')).toMatchObject({
      measurement: 'amount',
      measurementSource: 'declared'
    });
    expect(second.rowSemantics?.value).toBe('objects');
    expect(cache.get('a', 2)).toBe(second);
    // The other column is still a guess.
    expect(getMeasurement(second, 'rate')?.measurementSource).toBe('guess');
  });

  it('goes back to the guess when a declaration is removed', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    cache.set(makeProfile('a', 1));
    cache.setDeclarations('a', {measurement: {cases: 'amount'}});
    const reverted = cache.setDeclarations('a', undefined);
    expect(getMeasurement(reverted, 'cases')).toMatchObject({
      measurement: 'count',
      measurementSource: 'guess'
    });
    expect(reverted?.rowSemantics).toBeUndefined();
    expect(cache.getDeclarations('a')).toBeUndefined();
  });

  it('keeps declarations of one dataset away from another', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    cache.setDeclarations('a', {measurement: {cases: 'amount'}});
    const other = cache.set(makeProfile('b', 1));
    expect(getMeasurement(other, 'cases')?.measurementSource).toBe('guess');
  });

  it('keeps declarations of an evicted dataset', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.setDeclarations('a', {measurement: {cases: 'amount'}});
    cache.set(makeProfile('a', 1));
    cache.set(makeProfile('b', 1));
    expect(cache.get('a', 1)).toBeUndefined();
    expect(getMeasurement(cache.set(makeProfile('a', 1)), 'cases')?.measurement).toBe('amount');
  });

  it('returns undefined when declarations are set for a dataset that is not cached', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    expect(cache.setDeclarations('a', {rowSemantics: 'events'})).toBeUndefined();
  });

  it('rejects an entry limit that is not an integer from 1 to the limit', () => {
    for (const maximumEntries of [0, -1, 1.5, Number.NaN, 10_001]) {
      expect(() => createKeplerDatasetProfileCache({maximumEntries})).toThrow(
        /options\.maximumEntries must be an integer from 1 to 10000/
      );
    }
  });
});

describe('declarations text', () => {
  it('round trips and writes the same text for the same corrections', () => {
    const declarations = {
      measurement: {zeta: 'rate' as const, alpha: 'count' as const},
      rowSemantics: 'events' as const,
      coordinateModel: 'planar' as const
    };
    const text = serializeKeplerDatasetProfileDeclarations(declarations);
    expect(text).toBe(
      '{"version":1,"measurement":{"alpha":"count","zeta":"rate"},"rowSemantics":"events","coordinateModel":"planar"}'
    );
    expect(parseKeplerDatasetProfileDeclarations(text)).toEqual(declarations);
    expect(serializeKeplerDatasetProfileDeclarations({})).toBe('{"version":1}');
    expect(parseKeplerDatasetProfileDeclarations('{"version":1}')).toEqual({});
  });

  it('reads a __proto__ column name as a name, not a prototype', () => {
    const parsed = parseKeplerDatasetProfileDeclarations(
      '{"version":1,"measurement":{"__proto__":"count"}}'
    );
    expect(Object.keys(parsed.measurement ?? {})).toEqual(['__proto__']);
    expect(Object.getPrototypeOf(parsed.measurement)).toBe(Object.prototype);
    expect(({} as Record<string, unknown>)['count']).toBeUndefined();
  });

  const malformed: [string, string, RegExp][] = [
    ['text that is not JSON', '{nope', /\$: not valid JSON/],
    ['an array root', '[]', /\$: expected an object/],
    ['a null root', 'null', /\$: expected an object/],
    ['a missing version', '{}', /\$\.version: expected 1; got undefined/],
    ['another version', '{"version":2}', /\$\.version: expected 1; got 2/],
    ['an unknown key', '{"version":1,"colour":"red"}', /\$\.colour: unknown key/],
    [
      'measurement as an array',
      '{"version":1,"measurement":[]}',
      /\$\.measurement: expected an object/
    ],
    [
      'measurement as null',
      '{"version":1,"measurement":null}',
      /\$\.measurement: expected an object/
    ],
    [
      'a level of the wrong type',
      '{"version":1,"measurement":{"fare":3}}',
      /\$\.measurement\.fare: expected a string; got 3/
    ],
    [
      'an unknown level',
      '{"version":1,"measurement":{"fare":"money"}}',
      /\$\.measurement\.fare: unknown measurement level "money"/
    ],
    [
      'unknown row semantics',
      '{"version":1,"rowSemantics":"things"}',
      /\$\.rowSemantics: expected one of/
    ],
    [
      'row semantics of the wrong type',
      '{"version":1,"rowSemantics":1}',
      /\$\.rowSemantics: expected one of/
    ],
    [
      'an unknown coordinate model',
      '{"version":1,"coordinateModel":"polar"}',
      /\$\.coordinateModel: expected one of geographic, planar/
    ],
    [
      'a null coordinate model',
      '{"version":1,"coordinateModel":null}',
      /\$\.coordinateModel: expected one of/
    ]
  ];
  for (const [name, text, pattern] of malformed) {
    it(`rejects ${name}`, () => {
      expect(() => parseKeplerDatasetProfileDeclarations(text)).toThrow(pattern);
      expect(() => parseKeplerDatasetProfileDeclarations(text)).toThrow(
        /^parseKeplerDatasetProfileDeclarations /
      );
    });
  }

  it('refuses to write declarations the parser would reject', () => {
    expect(() =>
      serializeKeplerDatasetProfileDeclarations({
        measurement: {fare: 'money' as never}
      })
    ).toThrow(
      /serializeKeplerDatasetProfileDeclarations \$\.measurement\.fare: unknown measurement level/
    );
    expect(() =>
      serializeKeplerDatasetProfileDeclarations({rowSemantics: 'things' as never})
    ).toThrow(/\$\.rowSemantics/);
  });
});

function makeGridProfile(
  datasetId: string,
  revision: number,
  options: Record<string, unknown> = {}
) {
  const features: CatalogFeature[] = [];
  for (let index = 0; index < 40; index++)
    features.push({
      geometry: {
        type: 'Point',
        coordinates: [10 + (index % 8) * 0.1, 20 + Math.floor(index / 8) * 0.1]
      },
      properties: {cases: index + 3}
    } as CatalogFeature);
  return profileKeplerFeatureCollection(
    {type: 'FeatureCollection', features},
    {datasetId, label: datasetId, revision, ...options}
  );
}

describe('declared coordinate model through the cache', () => {
  it('keeps spacing in coordinate units when declared before the profile, across a revision', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    cache.setDeclarations('a', {coordinateModel: 'planar'});
    const first = cache.set(makeGridProfile('a', 1));
    expect(first.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    expect(first.spacing?.unit).toBe('coordinate-unit');
    expect(first.spacing?.nearestNeighbor?.median).toBeCloseTo(0.1, 9);
    const second = cache.set(makeGridProfile('a', 2));
    expect(second.coordinates?.model).toBe('planar');
    expect(second.spacing?.unit).toBe('coordinate-unit');
    expect(cache.get('a', 2)).toBe(second);
    expect(cache.get('a', 1)).toBeUndefined();
  });

  it('recomputes the cached profile when the declaration arrives later, and goes back when removed', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 3});
    const guessed = cache.set(makeGridProfile('a', 1));
    expect(guessed.spacing?.unit).toBe('metre');
    const planar = cache.setDeclarations('a', {coordinateModel: 'planar'});
    expect(planar?.spacing?.unit).toBe('coordinate-unit');
    const reverted = cache.setDeclarations('a', undefined);
    expect(reverted?.spacing).toEqual(guessed.spacing);
    expect(reverted?.coordinates?.source).toBe('guess');
  });
});

describe('declarations in the cache are validated and copied', () => {
  it('rejects values the parser would reject, naming the path, and stores nothing', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    cache.set(makeProfile('a', 1));
    const bad: [unknown, RegExp][] = [
      [{coordinateModel: 'moon'}, /\$\.coordinateModel/],
      [
        {measurement: {cases: 'bogus'}},
        /\$\.measurement\.cases: unknown measurement level "bogus"/
      ],
      [{rowSemantics: 5}, /\$\.rowSemantics/],
      [{colour: 'red'}, /\$\.colour: unknown key/],
      [null, /\$: expected an object/]
    ];
    for (const [declarations, pattern] of bad) {
      expect(() => cache.setDeclarations('a', declarations as never)).toThrow(pattern);
    }
    expect(cache.getDeclarations('a')).toBeUndefined();
    expect(getMeasurement(cache.get('a', 1), 'cases')?.measurementSource).toBe('guess');
  });

  it('copies what the host passes, and what it gets back cannot be changed', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    const declarations = {measurement: {cases: 'amount' as const}};
    cache.setDeclarations('a', declarations);
    declarations.measurement.cases = 'rate' as never;
    (declarations.measurement as Record<string, string>)['rate'] = 'count';
    const applied = cache.set(makeProfile('a', 1));
    expect(getMeasurement(applied, 'cases')?.measurement).toBe('amount');
    expect(getMeasurement(applied, 'rate')?.measurementSource).toBe('guess');
    const stored = cache.getDeclarations('a');
    expect(Object.isFrozen(stored)).toBe(true);
    expect(() => {
      (stored?.measurement as Record<string, string>)['cases'] = 'rate';
    }).toThrow();
  });
});

describe('removing a declaration made with options.declarations', () => {
  it('goes back to the guess', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    const made = makeProfile('a', 1);
    const features = [0, 1, 2, 3].map(
      index =>
        ({
          geometry: {type: 'Point', coordinates: [-87.6 + index * 0.01, 41.8]},
          properties: {cases: index + 3}
        }) as CatalogFeature
    );
    const withOptions = profileKeplerFeatureCollection(
      {type: 'FeatureCollection', features},
      {
        datasetId: 'a',
        label: 'a',
        revision: 1,
        declarations: {measurement: {cases: 'amount'}, coordinateModel: 'planar'}
      }
    );
    expect(made.columns.length).toBeGreaterThan(0);
    // With no stored declarations the profile is kept as given.
    expect(cache.set(withOptions)).toBe(withOptions);
    expect(getMeasurement(cache.get('a', 1), 'cases')?.measurement).toBe('amount');
    // Removing declarations that were never stored changes nothing.
    expect(cache.setDeclarations('a', undefined)).toBe(withOptions);
    // Storing others replaces the baked-in ones; removing those goes back to the guess.
    const stored = cache.setDeclarations('a', {rowSemantics: 'events'});
    expect(getMeasurement(stored, 'cases')).toMatchObject({
      measurement: 'count',
      measurementSource: 'guess'
    });
    expect(stored?.coordinates?.source).toBe('guess');
    expect(stored?.spacing?.unit).toBe('metre');
    const reverted = cache.setDeclarations('a', {measurement: {cases: 'amount'}});
    expect(getMeasurement(reverted, 'cases')?.measurementSource).toBe('declared');
    const back = cache.setDeclarations('a', undefined);
    expect(getMeasurement(back, 'cases')).toMatchObject({
      measurement: 'count',
      measurementSource: 'guess'
    });
    expect(back?.coordinates?.source).toBe('guess');
  });
});

describe('profile identity', () => {
  it('returns the same profile object for a dataset revision on every get', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    const returned = cache.set(makeProfile('a', 1));
    expect(cache.get('a', 1)).toBe(returned);
    expect(cache.get('a', 1)).toBe(cache.get('a', 1));
    // Reading another dataset does not change it, and a new revision gives a new object.
    cache.set(makeProfile('b', 1));
    expect(cache.get('a', 1)).toBe(returned);
    const next = cache.set(makeProfile('a', 2));
    expect(next).not.toBe(returned);
    // A declaration change makes a new object, which then stays the same until the next change.
    const corrected = cache.setDeclarations('a', {rowSemantics: 'events'});
    expect(corrected).not.toBe(next);
    expect(cache.get('a', 2)).toBe(corrected);
    expect(cache.get('a', 2)).toBe(cache.get('a', 2));
  });
});

describe('row semantics through the cache', () => {
  it('returns to the guess when the declaration is removed', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    const guessed = cache.set(makeProfile('a', 1));
    const declared = cache.setDeclarations('a', {rowSemantics: 'surface-samples'});
    expect(declared?.rowSemantics).toMatchObject({value: 'surface-samples', source: 'declared'});
    const back = cache.setDeclarations('a', undefined);
    expect(back?.rowSemantics).toEqual(guessed.rowSemantics);
    expect(back?.rowSemantics?.source ?? 'guess').toBe('guess');
  });
});
