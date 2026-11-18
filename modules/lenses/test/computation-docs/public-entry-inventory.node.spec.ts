// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {expect, test} from 'vitest';
import {publicEntryMaps} from '../../src/computation-lenses/public-entry-maps.js';
import {createPublicEntryInventory} from './public-entry-inventory.mjs';

test('browser entry inventory matches concrete real export maps and conditions', () => {
  expect(publicEntryMaps).toEqual(createPublicEntryInventory());
});

test('entry loader works with analysis nested beneath lenses instead of a sibling', () => {
  const temporary = mkdtempSync(join(tmpdir(), 'lens-nested-'));
  try {
    const lenses = join(temporary, 'node_modules/@luma-contributors/lenses');
    const nestedAnalysis = join(lenses, 'node_modules/@luma-contributors/analysis');
    mkdirSync(join(lenses, 'dist/computation-lenses'), {recursive: true});
    mkdirSync(nestedAnalysis, {recursive: true});
    writeFileSync(join(lenses, 'package.json'), '{"type":"module"}');
    writeFileSync(
      join(nestedAnalysis, 'package.json'),
      '{"name":"@luma-contributors/analysis","type":"module"}'
    );
    const loader = join(lenses, 'dist/computation-lenses/public-entry-maps.js');
    cpSync(new URL('../../src/computation-lenses/public-entry-maps.js', import.meta.url), loader);
    const result = execFileSync(
      process.execPath,
      [
        '--input-type=module',
        '-e',
        `import {publicEntryMaps} from ${JSON.stringify(pathToFileURL(loader).href)}; console.log(publicEntryMaps.length);`
      ],
      {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}
    );
    expect(result.trim()).toBe('5');
  } finally {
    rmSync(temporary, {recursive: true, force: true});
  }
});
