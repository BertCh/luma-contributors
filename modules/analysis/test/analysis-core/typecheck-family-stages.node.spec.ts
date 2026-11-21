// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typeScript from 'typescript';
import {afterEach, expect, it} from 'vitest';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];

function write(directory: string, file: string, source: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), source);
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it('typechecks every ordinary test root once in sequential package and family projects with independent caches', () => {
  const directory = mkdtempSync(join(tmpdir(), 'typecheck-families-'));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  mkdirSync(join(directory, 'node_modules'));
  symlinkSync(
    join(ROOT, 'node_modules/typescript'),
    join(directory, 'node_modules/typescript'),
    'dir'
  );
  copyFileSync(
    join(ROOT, 'scripts/typecheck-tests.mjs'),
    join(directory, 'scripts/typecheck-tests.mjs')
  );
  copyFileSync(join(ROOT, 'tsconfig.json'), join(directory, 'tsconfig.json'));
  copyFileSync(
    join(ROOT, 'scripts/verification-policy.mjs'),
    join(directory, 'scripts/verification-policy.mjs')
  );
  const files = [
    'modules/analysis/test/analysis-core/core.node.spec.ts',
    'modules/analysis/test/gpu-network/network-analysis/routes.spec.ts',
    'modules/analysis/test/gpu-spatial-analysis/point-density/density.spec.ts',
    'modules/analysis/test/oracles/shared.ts',
    'modules/lenses/test/computation-lenses/lens.node.spec.ts',
    'modules/kepler/test/workflows/workflow.node.spec.ts',
    'test/shared.node.spec.ts'
  ];
  for (const file of files) write(directory, file, 'export {};');
  const runtimeOnly = 'modules/kepler/test/interpolation-geotiff.node.spec.ts';
  write(directory, runtimeOnly, 'export {};');
  write(
    directory,
    'modules/analysis/test/example.spec.ts',
    "import '../../../examples/deck/kepler-compute/helper';"
  );
  write(directory, 'modules/analysis/test/excluded.spec.ts', "import './example';");
  write(
    directory,
    'modules/analysis/test/website.spec.ts',
    "import '../../../website/src/helper';"
  );
  write(
    directory,
    'test/kepler-compute-example-specs.json',
    JSON.stringify({
      excludedFromRuns: {[runtimeOnly]: 'runtime-only exclusion'},
      excludedFromTypecheck: {'modules/analysis/test/excluded.spec.ts': 'fixture'}
    })
  );
  write(
    directory,
    'node_modules/.bin/tsc',
    `#!${process.execPath}\nconst fs = require('node:fs'); const path = require('node:path'); const configuration = process.argv[process.argv.indexOf('-p') + 1]; const contents = JSON.parse(fs.readFileSync(configuration, 'utf8')); fs.appendFileSync('runs.jsonl', JSON.stringify({configuration, contents}) + '\\n');`
  );
  chmodSync(join(directory, 'node_modules/.bin/tsc'), 0o755);
  const result = spawnSync(process.execPath, ['scripts/typecheck-tests.mjs'], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 10000,
    env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256'}
  });
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const runs = readFileSync(join(directory, 'runs.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  // A broad test/source program cannot distinguish family roots and fails this fixture.
  const ordinary = runs.filter(run => run.configuration.includes('/test-families/'));
  expect(ordinary).toHaveLength(7);
  for (const run of ordinary)
    expect(run.contents.compilerOptions.types).toEqual(
      expect.arrayContaining(['@webgpu/types', 'node', 'vite/client'])
    );
  const roots = ordinary.flatMap(run =>
    run.contents.files.map((file: string) => resolve(directory, dirname(run.configuration), file))
  );
  expect(roots.sort()).toEqual(files.map(file => join(directory, file)).sort());
  expect(new Set(ordinary.map(run => run.contents.compilerOptions.tsBuildInfoFile)).size).toBe(7);
  expect(ordinary.map(run => run.contents.include)).toEqual(Array.from({length: 7}, () => []));
  const configurations = ordinary.map(run => run.configuration);
  expect(configurations[0]).toContain('analysis-analysis-core');
  expect(configurations.findIndex(file => file.includes('lenses-'))).toBeGreaterThan(
    configurations.findIndex(file => file.includes('analysis-oracles'))
  );
  expect(configurations.findIndex(file => file.includes('kepler-'))).toBeGreaterThan(
    configurations.findIndex(file => file.includes('lenses-'))
  );
  const example = runs.find(run => run.configuration.endsWith('tsconfig.example-specs.json'));
  expect(example.contents.include).toContain(`../${runtimeOnly}`);
  expect(example.contents.include).toContain('../modules/analysis/test/example.spec.ts');
  expect(example.contents.include).not.toContain('../modules/analysis/test/excluded.spec.ts');
  const website = runs.find(run => run.configuration.endsWith('tsconfig.website-specs.json'));
  expect(website.contents.include).toContain('../modules/analysis/test/website.spec.ts');
});

it('generates configs without launching compilers and typechecks Vite glob, env and raw imports in a family project', () => {
  const directory = mkdtempSync(join(tmpdir(), 'typecheck-vite-'));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  for (const script of ['typecheck-tests.mjs', 'verification-policy.mjs'])
    copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
  copyFileSync(join(ROOT, 'tsconfig.json'), join(directory, 'tsconfig.json'));
  write(
    directory,
    'test/kepler-compute-example-specs.json',
    '{"excludedFromRuns":{},"excludedFromTypecheck":{}}'
  );
  write(directory, 'modules/analysis/test/vite-client/payload.ts', 'export const value = 1;');
  write(
    directory,
    'modules/analysis/test/vite-client/client.node.spec.ts',
    `import raw from './payload.ts?raw';\nconst mode: string = import.meta.env.MODE;\nconst modules = import.meta.glob<string>('./*.ts', {query: '?raw', import: 'default'});\nexport const texts: Promise<string[]> = Promise.all(Object.values(modules).map(async load => {const text: string = await load(); return text + raw + mode;}));`
  );
  for (const dependency of ['vite', '@types', '@webgpu', 'typescript']) {
    mkdirSync(join(directory, 'node_modules'), {recursive: true});
    symlinkSync(
      join(ROOT, 'node_modules', dependency),
      join(directory, 'node_modules', dependency),
      'dir'
    );
  }
  const generated = spawnSync(
    process.execPath,
    ['scripts/typecheck-tests.mjs', '--generate-only'],
    {cwd: directory, encoding: 'utf8', timeout: 10000}
  );
  expect(generated.status, `${generated.stdout}${generated.stderr}`).toBe(0);
  const stages = JSON.parse(
    readFileSync(join(directory, '.typecheck/test-family-stages.json'), 'utf8')
  );
  const stage = stages.find((candidate: {files: string[]}) =>
    candidate.files.includes('modules/analysis/test/vite-client/client.node.spec.ts')
  );
  expect(stage).toBeDefined();
  const checked = spawnSync(
    process.execPath,
    [join(ROOT, 'node_modules/typescript/lib/tsc.js'), '-p', stage.configuration, '--noEmit'],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 15000,
      env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256'}
    }
  );
  expect(checked.status, `${checked.stdout}${checked.stderr}`).toBe(0);
});

it.each([
  false,
  true
])('final3 H6 resolves inherited ambient types from a root config with comments=%s', comments => {
  const directory = mkdtempSync(join(tmpdir(), 'typecheck-config-'));
  directories.push(directory);
  for (const script of ['typecheck-tests.mjs', 'verification-policy.mjs'])
    write(directory, `scripts/${script}`, readFileSync(join(ROOT, 'scripts', script), 'utf8'));
  copyFileSync(join(ROOT, 'tsconfig.json'), join(directory, 'tsconfig.base.json'));
  write(
    directory,
    'tsconfig.json',
    `${comments ? '// Valid TypeScript JSONC configuration\n' : ''}{"extends":"./tsconfig.base.json"}`
  );
  write(directory, 'test/kepler-compute-example-specs.json', '{"excludedFromRuns":{}}');
  write(directory, 'modules/analysis/test/analysis-core/inherited.node.spec.ts', 'export {};');
  write(directory, 'modules/analysis/src/index.ts', 'export {};');
  mkdirSync(join(directory, 'node_modules'), {recursive: true});
  symlinkSync(
    join(ROOT, 'node_modules/typescript'),
    join(directory, 'node_modules/typescript'),
    'dir'
  );
  const parsed = typeScript.getParsedCommandLineOfConfigFile(
    join(directory, 'tsconfig.json'),
    {},
    {
      ...typeScript.sys,
      onUnRecoverableConfigFileDiagnostic: diagnostic => {
        throw new Error(typeScript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
      }
    }
  );
  expect(parsed?.errors).toEqual([]);
  const generated = spawnSync(
    process.execPath,
    ['scripts/typecheck-tests.mjs', '--generate-only'],
    {cwd: directory, encoding: 'utf8', timeout: 10000}
  );
  expect(generated.status, `${generated.stdout}${generated.stderr}`).toBe(0);
  const configuration = JSON.parse(
    readFileSync(join(directory, '.typecheck/test-families/analysis-analysis-core.json'), 'utf8')
  );
  expect(configuration.compilerOptions.types).toEqual(['@webgpu/types', 'node', 'vite/client']);
});
