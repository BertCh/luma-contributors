// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Load every non-wildcard export of every package through ESM and CommonJS, as a consumer would.
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
// Entry points imported only for their registration side effects.
const SIDE_EFFECT_ENTRIES = new Set(['@luma-contributors/lenses/computation-lenses/scenarios/index']);
const failures = [];
let checked = 0;
for (const moduleName of ['graph', 'lenses', 'kepler']) {
  const packageJson = JSON.parse(readFileSync(new URL(`../modules/${moduleName}/package.json`, import.meta.url), 'utf8'));
  for (const [key, target] of Object.entries(packageJson.exports)) {
    if (key.includes('*') || typeof target === 'string') continue;
    const specifier = key === '.' ? packageJson.name : `${packageJson.name}/${key.slice(2)}`;
    try {
      const esm = await import(specifier);
      if (!Object.keys(esm).length && !SIDE_EFFECT_ENTRIES.has(specifier)) {
        throw new Error('ESM entry has no exports');
      }
      if (target.require) {
        const cjs = require(specifier);
        if (Object.keys(cjs).length !== Object.keys(esm).length) {
          throw new Error(`CommonJS exports ${Object.keys(cjs).length} names, ESM ${Object.keys(esm).length}`);
        }
      }
      checked++;
    } catch (error) {
      failures.push(`${specifier}: ${String(error.message).split('\n')[0]}`);
    }
  }
}
process.stdout.write(`verified ${checked} entry points\n`);
if (failures.length) {
  process.stdout.write(failures.join('\n') + '\n');
  process.exitCode = 1;
}
