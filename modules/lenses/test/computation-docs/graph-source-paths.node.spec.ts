// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync, readdirSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '../../../..');
const PACKAGE_SOURCE_ROOT = resolve(REPOSITORY_ROOT, 'modules/lenses/src');

it('every modules/graph/src path cited by lens packs and scenarios exists', () => {
  const missing: string[] = [];
  let checkedCount = 0;
  for (const file of readdirSync(PACKAGE_SOURCE_ROOT, {recursive: true, encoding: 'utf8'})) {
    if (!file.endsWith('.ts')) continue;
    const source = readFileSync(join(PACKAGE_SOURCE_ROOT, file), 'utf8');
    for (const [path] of source.matchAll(/modules\/graph\/src\/[A-Za-z0-9_./-]+\.[a-z]+/g)) {
      checkedCount++;
      if (!existsSync(join(REPOSITORY_ROOT, path))) missing.push(`${file}: ${path}`);
    }
  }
  expect(checkedCount).toBeGreaterThan(100);
  expect(missing).toEqual([]);
});
