// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {listTestFiles} from './verification-policy.mjs';

const rootDirectory = fileURLToPath(new URL('../', import.meta.url));
const files = listTestFiles(rootDirectory, 'benchmarks');
if (!files.length) throw new Error('No benchmark files found');
const environment = {...process.env, LUMA_CONTRIBUTORS_BENCHMARKS: 'true'};
// Activate only boolean entry gates, never size, revision or alternate-route options.
for (const file of files) {
  const source = readFileSync(join(rootDirectory, file), 'utf8');
  const flags = [...source.matchAll(/\[\s*'(VITE_[A-Z_]+)'\s*\]\s*(?:===|!==)\s*'true'/g)]
    .map(match => match[1])
    .filter(flag => /(?:BENCHMARK|PROFILE)$/.test(flag));
  for (const flag of flags) environment[flag] = 'true';
  process.stdout.write(`benchmark: ${file}; activation: ${flags.join(', ') || 'always enabled'}\n`);
}
// Vitest's verbose reporter names skipped tests; the inventory above makes missing files visible.
const result = spawnSync(
  process.execPath,
  [
    join(rootDirectory, 'node_modules/vitest/vitest.mjs'),
    'run',
    '--project',
    'headless',
    '--maxWorkers=1',
    '--reporter=verbose',
    ...process.argv.slice(2)
  ],
  {cwd: rootDirectory, stdio: 'inherit', env: environment}
);
process.exit(result.status ?? 1);
