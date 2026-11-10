// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {
  classifyFailure,
  diffJson,
  flattenJson,
  parseExpectedFixtures,
  resolveGenerator
} from '../../../../scripts/run-oracle-fixture.mjs';

it('diffJson reports changed, added and removed leaves and the largest numeric difference', () => {
  const diff = diffJson(
    {versions: {numpy: '2.5.4'}, values: [1, 2, 3], gone: true},
    {versions: {numpy: '2.5.5'}, values: [1, 2.5, 3], fresh: 1}
  );
  expect(diff.changed.map(item => item.path)).toEqual(['versions.numpy', 'values[1]']);
  expect(diff.changedNumberCount).toBe(1);
  expect(diff.maximumAbsoluteDifference).toBe(0.5);
  expect(diff.added).toEqual(['fresh']);
  expect(diff.removed).toEqual(['gone']);
  expect(diff.metadataOnly).toBe(false);
});

it('diffJson recognises version-only drift and treats NaN as equal to NaN', () => {
  const diff = diffJson(
    {versions: {scipy: '1'}, x: [Number.NaN]},
    {versions: {scipy: '2'}, x: [Number.NaN]}
  );
  expect(diff.metadataOnly).toBe(true);
  expect(diffJson({a: [1, {b: 2}]}, {a: [1, {b: 2}]}).changed).toEqual([]);
  expect([...flattenJson({a: [], b: {}}).keys()]).toEqual(['a[]', 'b{}']);
});

it('classifyFailure names a missing Python or R package and ignores ordinary failures', () => {
  expect(classifyFailure('.py', "ModuleNotFoundError: No module named 'esda.moran'")).toContain(
    'missing Python package: esda'
  );
  expect(
    classifyFailure('.R', 'Error in library(gstat) : there is no package called ‘gstat’')
  ).toBe('missing R package: gstat');
  expect(classifyFailure('.py', 'AssertionError: wrong version')).toBeUndefined();
});

it('parseExpectedFixtures reads GENERATORS.md sections and resolveGenerator finds scripts', () => {
  const expected = parseExpectedFixtures(
    '## Literal mappings\n\n### `a-b.py`\n\n`a-b.json`, `a-c.json` (2)\n\n## Counts\n\n`ignored.json`\n'
  );
  expect(expected.get('a-b.py')).toEqual(['a-b.json', 'a-c.json']);
  const directory = mkdtempSync(join(tmpdir(), 'plat06-'));
  try {
    for (const file of ['one.py', 'two.R', 'two.py', 'three.cpp'])
      writeFileSync(join(directory, file), '');
    expect(resolveGenerator('one', directory)).toEqual({file: 'one.py'});
    expect(resolveGenerator('two', directory).error).toContain('ambiguous');
    expect(resolveGenerator('two.R', directory)).toEqual({file: 'two.R'});
    expect(resolveGenerator('three', directory).error).toContain('not runnable');
    expect(resolveGenerator('missing', directory).error).toContain('no generator');
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
});
