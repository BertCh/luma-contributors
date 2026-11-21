// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  expectLandscapeSectionsBeforePublication,
  LANDSCAPE_PUBLICATION_MARKERS,
  LANDSCAPE_VALIDATION_MARKERS
} from './urban-landscape-generator-contract';

it('S2 writes the landscape fixture only after proximity and tile validation', () => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  expectLandscapeSectionsBeforePublication(source);
});

it.each(LANDSCAPE_VALIDATION_MARKERS)('B7 rejects deletion of validation %s', marker => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  expect(() =>
    expectLandscapeSectionsBeforePublication(source.replace(marker, 'removed()'))
  ).toThrow();
});

it('B7 rejects an indented extra write_json', () => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  expect(() =>
    expectLandscapeSectionsBeforePublication(`${source}\n  write_json(result, 'extra.json')`)
  ).toThrow();
});

it.each(LANDSCAPE_PUBLICATION_MARKERS)('URB-105 rejects deletion of %s', marker => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  expect(() =>
    expectLandscapeSectionsBeforePublication(source.replaceAll(marker, 'removed('))
  ).toThrow();
});

it('S3 documents the clustering launcher with the pinned environment command', () => {
  const source = readFileSync(new URL('./spatial-clustering-fixtures.py', import.meta.url), 'utf8');
  expect(source).toContain(
    'node scripts/run-oracle-fixture.mjs spatial-clustering-fixtures --python <clustering-venv>/bin/python3'
  );
});
