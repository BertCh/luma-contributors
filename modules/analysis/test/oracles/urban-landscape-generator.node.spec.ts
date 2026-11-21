// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  expectLandscapeSectionsBeforePublication,
  LANDSCAPE_PUBLICATION_MARKERS,
  LANDSCAPE_VALIDATION_MARKERS
} from './urban-landscape-generator-contract';

it('URB-105 requires each section to exist before its sole final publication', () => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  expectLandscapeSectionsBeforePublication(source);
  for (const section of [...LANDSCAPE_PUBLICATION_MARKERS, ...LANDSCAPE_VALIDATION_MARKERS]) {
    expect(() =>
      expectLandscapeSectionsBeforePublication(source.replaceAll(section, 'removed('))
    ).toThrow();
  }
});

it('URB-105 validates all two-patch proximity vectors before publication', () => {
  const source = readFileSync(new URL('./urban-landscape.R', import.meta.url), 'utf8');
  const validation = source.match(
    /for \(proximity_values in list\(([\s\S]*?)\)\) \{\s*stopifnot\(length\(proximity_values\) == 2,\s*all\(is\.finite\(proximity_values\)\)\)\s*\}/
  );
  expect(validation, 'two finite values required for each patch metric').not.toBeNull();
  for (const field of ['area', 'nearest', 'composedProximity', 'coreEdge', 'coreOpen']) {
    expect(validation![1]).toContain(`result$proximity$${field}`);
  }
  expect(source.indexOf(validation![0])).toBeLessThan(source.indexOf('write_json('));
});
