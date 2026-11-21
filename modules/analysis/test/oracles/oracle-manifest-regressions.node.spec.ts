// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

const manifest = JSON.parse(
  readFileSync(new URL('../../../../scripts/oracle-fixtures.json', import.meta.url), 'utf8')
);

it('TOOL-107 stages the NOAA source intact and requires a freshly recreated NOAA result', () => {
  const generator = 'noaa-staging-miniature.mjs';
  const declaration = manifest.generators['parity-wp5-solar-noaa.py'];
  for (const recreate of [true, false]) {
    const root = realpathSync(mkdtempSync(join(tmpdir(), 'noaa-staging-')));
    try {
      const scripts = join(root, 'scripts');
      const oracles = join(root, 'modules/analysis/test/oracles');
      mkdirSync(scripts, {recursive: true});
      mkdirSync(join(oracles, 'fixtures'), {recursive: true});
      writeFileSync(
        join(scripts, 'run-oracle-fixture.mjs'),
        readFileSync(new URL('../../../../scripts/run-oracle-fixture.mjs', import.meta.url))
      );
      writeFileSync(
        join(scripts, 'oracle-fixtures.json'),
        JSON.stringify({
          version: 1,
          profiles: {test: {runtime: 'node'}},
          generators: {[generator]: {...declaration, profile: 'test'}}
        })
      );
      writeFileSync(join(oracles, 'fixtures/parity-wp5-solar-position.json'), '{"source":17}');
      writeFileSync(join(oracles, 'fixtures/parity-wp5-solar-noaa-position.json'), '{"result":17}');
      // This miniature uses the real manifest declaration and staging implementation.
      // It proves that an old result cannot count as regeneration, even on exit zero.
      writeFileSync(
        join(oracles, generator),
        `import {readFileSync, existsSync, writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const directory = new URL('./fixtures/', import.meta.url);
assert.equal(readFileSync(new URL('parity-wp5-solar-position.json', directory), 'utf8'), '{"source":17}');
const result = new URL('parity-wp5-solar-noaa-position.json', directory);
assert.equal(existsSync(result), false);
${recreate ? `writeFileSync(result, '{"result":17}');` : ''}
`
      );
      const run = spawnSync(
        process.execPath,
        [join(scripts, 'run-oracle-fixture.mjs'), generator],
        {encoding: 'utf8'}
      );
      expect(run.status, `${run.stdout}\n${run.stderr}`).toBe(recreate ? 0 : 3);
      expect(recreate ? run.stdout : run.stderr).toContain(recreate ? 'IDENTICAL' : 'NOT WRITTEN');
      expect(readFileSync(join(oracles, 'fixtures/parity-wp5-solar-position.json'), 'utf8')).toBe(
        '{"source":17}'
      );
    } finally {
      rmSync(root, {recursive: true, force: true});
    }
  }
  expect(declaration.inputs).toEqual(['oracles/fixtures/parity-wp5-solar-position.json']);
  expect(declaration.outputs).toEqual(['oracles/fixtures/parity-wp5-solar-noaa-position.json']);
});

it('PTS-141 registers the single clustering producer and its separate compatible profile', () => {
  const declaration = manifest.generators['spatial-clustering-fixtures.py'];
  expect(declaration).toEqual({
    kind: 'fixture',
    profile: 'python-clustering',
    inputs: ['gpu-spatial-analysis/spatial-clustering/clustering-fixture-oracles.py'],
    outputs: ['gpu-spatial-analysis/spatial-clustering/clustering-fixture-oracles.json']
  });
  expect(manifest.profiles['python-clustering'].description).toContain('NumPy 2.0.2');
  expect(manifest.profiles['python-clustering'].description).toContain('scikit-learn 1.6.1');
});

it.skipIf(!process.env['ORACLE_CLUSTERING_PYTHON'])(
  'PTS-141 recreates the staged clustering family byte for byte with its pinned profile',
  () => {
    const run = spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL('../../../../scripts/run-oracle-fixture.mjs', import.meta.url)),
        'spatial-clustering-fixtures',
        '--python',
        process.env['ORACLE_CLUSTERING_PYTHON']!
      ],
      {encoding: 'utf8', timeout: 60_000}
    );
    expect(run.status, `${run.stdout}\n${run.stderr}`).toBe(0);
    expect(run.stdout).toContain(
      'IDENTICAL  modules/analysis/test/gpu-spatial-analysis/spatial-clustering/clustering-fixture-oracles.json'
    );
  },
  65_000
);
