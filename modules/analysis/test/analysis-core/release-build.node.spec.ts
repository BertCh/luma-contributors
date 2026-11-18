// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];
const MODULES = ['analysis', 'lenses', 'kepler'];

function write(directory: string, path: string, source: string) {
  mkdirSync(dirname(join(directory, path)), {recursive: true});
  writeFileSync(join(directory, path), source);
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function fixture() {
  const directory = mkdtempSync(join(tmpdir(), 'release-build-'));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  copyFileSync(
    join(ROOT, 'scripts/build-release.mjs'),
    join(directory, 'scripts/build-release.mjs')
  );
  copyFileSync(join(ROOT, 'scripts/build-record.mjs'), join(directory, 'scripts/build-record.mjs'));
  copyFileSync(
    join(ROOT, 'scripts/verification-policy.mjs'),
    join(directory, 'scripts/verification-policy.mjs')
  );
  write(directory, 'package.json', '{"type":"module"}');
  write(directory, 'tsconfig.json', '{}');
  write(directory, 'tsconfig.build.json', '{"files":[],"references":[]}');
  for (const moduleName of MODULES) {
    write(directory, `modules/${moduleName}/src/index.ts`, 'export const value = 2;');
    write(
      directory,
      `modules/${moduleName}/package.json`,
      JSON.stringify({
        name: `@luma-contributors/${moduleName}`,
        type: 'module',
        exports: './dist/index.js'
      })
    );
    write(directory, `modules/${moduleName}/tsconfig.json`, '{"include":["src/**/*"]}');
    write(directory, `modules/${moduleName}/README.md`, 'snapshot');
    write(directory, `modules/${moduleName}/dist/index.js`, 'old verified output');
    write(directory, `modules/${moduleName}/tsconfig.tsbuildinfo`, 'warm developer cache');
    write(directory, `modules/${moduleName}/test/large.json`, 'do not copy test fixtures');
  }
  write(
    directory,
    'scripts/build.mjs',
    `
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
export async function buildPackages(rootDirectory) {
  if (process.env.FAIL_BUILD) throw new Error('compiler failed');
  for (const moduleName of ${JSON.stringify(MODULES)}) {
    const moduleDirectory = join(rootDirectory, 'modules', moduleName);
    if (existsSync(join(moduleDirectory, 'dist')) || existsSync(join(moduleDirectory, 'tsconfig.tsbuildinfo')) || existsSync(join(moduleDirectory, 'test'))) throw new Error('not a clean snapshot');
    mkdirSync(join(moduleDirectory, 'dist'));
    writeFileSync(join(moduleDirectory, 'dist/index.js'), readFileSync(join(moduleDirectory, 'src/index.ts')));
  }
}
`
  );
  write(directory, 'scripts/typescript-project-inputs.mjs', 'export {};');
  const gate = `
import {readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const original = process.env.ORIGINAL_DIRECTORY;
if (process.env.CHECK_CLI_HEAP) {
  const {getHeapStatistics} = await import('node:v8');
  const heap = getHeapStatistics().heap_size_limit / 1024 ** 2;
  process.stdout.write('release-child-heap-mib=' + heap + '\\n');
  if (heap >= 384) throw new Error('release wrapper dropped CLI heap');
}
for (const moduleName of ${JSON.stringify(MODULES)}) {
  if (readFileSync(join(original, 'modules', moduleName, 'dist/index.js'), 'utf8') !== 'old verified output') throw new Error('promoted before all gates passed');
}
if (process.env.FAIL_GATE && import.meta.url.includes(process.env.FAIL_GATE)) throw new Error('gate failed');
if (process.env.MUTATE_INPUT) writeFileSync(join(original, 'modules/analysis/src/index.ts'), 'changed during verification');
if (process.argv.includes('--package-directory')) {
  const directory = process.argv[process.argv.indexOf('--package-directory') + 1];
  if (process.env.MUTATE_STAGE_INPUT) writeFileSync(join(directory, 'src/index.ts'), 'changed staged source');
  if (process.env.MUTATE_STAGE_OUTPUT) writeFileSync(join(directory, '../lenses/dist/index.js'), 'export const value = 99;');
  if ((await import(pathToFileURL(join(directory, 'dist/index.js')).href)).value !== 2) throw new Error('checked original package instead of snapshot');
  const tarball = join(process.env.TMPDIR, 'checked.tgz'); writeFileSync(tarball, 'checked bytes'); process.stdout.write('checked tarball: ' + tarball + '\\n');
}
`;
  write(directory, 'scripts/verify-package-exports.mjs', gate);
  write(directory, 'scripts/check-packed-consumer.mjs', gate);
  return directory;
}

function run(directory: string, arguments_: string[] = [], environment = {}) {
  const result = spawnSync(process.execPath, ['scripts/build-release.mjs', ...arguments_], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 10000,
    env: {...process.env, ORIGINAL_DIRECTORY: directory, ...environment}
  });
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

function assertOldOutputs(directory: string) {
  for (const moduleName of MODULES) {
    expect(readFileSync(join(directory, `modules/${moduleName}/dist/index.js`), 'utf8')).toBe(
      'old verified output'
    );
    expect(
      readFileSync(join(directory, `modules/${moduleName}/tsconfig.tsbuildinfo`), 'utf8')
    ).toBe('warm developer cache');
  }
}

it('stages a clean source snapshot and verifies it without replacing existing outputs or caches', () => {
  const directory = fixture();
  const result = run(directory);
  expect(result.status, result.output).toBe(0);
  assertOldOutputs(directory);
  const record = JSON.parse(
    readFileSync(join(directory, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(record.status).toBe('staged');
  expect(
    readFileSync(join(record.directory, 'workspace/modules/analysis/dist/index.js'), 'utf8')
  ).toContain('value = 2');
});

it('promotes all packages only after both artifact gates pass and retains prior outputs for recovery', () => {
  const directory = fixture();
  const result = run(directory, ['--promote']);
  expect(result.status, result.output).toBe(0);
  const record = JSON.parse(
    readFileSync(join(directory, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(record.status).toBe('promoted');
  for (const moduleName of MODULES) {
    expect(readFileSync(join(directory, `modules/${moduleName}/dist/index.js`), 'utf8')).toContain(
      'value = 2'
    );
    expect(readFileSync(join(record.directory, 'previous', moduleName, 'index.js'), 'utf8')).toBe(
      'old verified output'
    );
    expect(existsSync(join(directory, `modules/${moduleName}/tsconfig.tsbuildinfo`))).toBe(false);
    expect(
      readFileSync(join(record.directory, 'previous', `${moduleName}.tsbuildinfo`), 'utf8')
    ).toBe('warm developer cache');
  }
});

it('leaves prior outputs intact when compilation, an artifact gate, or source freshness fails', () => {
  for (const environment of [
    {FAIL_BUILD: 'true'},
    {FAIL_GATE: 'verify-package-exports'},
    {FAIL_GATE: 'check-packed-consumer'},
    {MUTATE_STAGE_INPUT: 'true'},
    {MUTATE_INPUT: 'true'}
  ]) {
    const directory = fixture();
    const result = run(directory, ['--promote'], environment);
    expect(result.status, result.output).toBe(1);
    assertOldOutputs(directory);
    expect(existsSync(join(directory, '.typecheck/release-build-latest.json'))).toBe(false);
  }
});

it('rolls back every previously installed package when a later output rename fails', async () => {
  const {promotePackages} = await import('../../../../scripts/build-release.mjs');
  const directory = fixture();
  const releaseDirectory = join(directory, 'release');
  for (const moduleName of MODULES)
    write(releaseDirectory, `workspace/modules/${moduleName}/dist/index.js`, 'new output');
  let failOnce = true;
  expect(() =>
    promotePackages(directory, releaseDirectory, (source, destination) => {
      if (String(source).endsWith('workspace/modules/lenses/dist') && failOnce) {
        failOnce = false;
        throw new Error('simulated rename failure');
      }
      renameSync(source, destination);
    })
  ).toThrow('rolled-back');
  assertOldOutputs(directory);
});

it('R7 rolls outputs and caches back if final metadata publication fails, preserving the prior latest record', () => {
  const directory = fixture();
  const latest = join(realpathSync(directory), '.typecheck/release-build-latest.json');
  write(directory, '.typecheck/release-build-latest.json', '{"status":"previous-success"}');
  write(
    directory,
    'fail-record.cjs',
    `const fileSystem = require('node:fs'); const originalWrite = fileSystem.writeFileSync; const originalRename = fileSystem.renameSync; const latest = ${JSON.stringify(latest)}; const fail = () => Object.assign(new Error('injected final record EISDIR'), {code:'EISDIR'}); fileSystem.writeFileSync = (file, ...values) => { if (String(file) === latest) { originalWrite(file, ''); throw fail(); } return originalWrite(file, ...values); }; fileSystem.renameSync = (source, destination) => { if (String(destination) === latest) throw fail(); return originalRename(source, destination); }; require('node:module').syncBuiltinESMExports();`
  );
  const result = run(directory, ['--promote'], {
    NODE_OPTIONS: `--require=${join(directory, 'fail-record.cjs')}`
  });
  expect(result.status, result.output).toBe(1);
  assertOldOutputs(directory);
  expect(readFileSync(latest, 'utf8')).toBe('{"status":"previous-success"}');
});

it('final1 F4 rolls promotion back if the packed gate changes a previously verified lenses output', () => {
  const directory = fixture();
  const latest = join(directory, '.typecheck/release-build-latest.json');
  write(directory, '.typecheck/release-build-latest.json', '{"status":"previous-success"}');
  const result = run(directory, ['--promote'], {MUTATE_STAGE_OUTPUT: 'true'});
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('verified output');
  assertOldOutputs(directory);
  expect(readFileSync(latest, 'utf8')).toBe('{"status":"previous-success"}');
});

it('final2 G8 carries the smaller CLI heap through release child invocations', () => {
  const directory = fixture();
  const result = spawnSync(
    process.execPath,
    ['--max-old-space-size=256', 'scripts/build-release.mjs'],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 10000,
      env: {
        ...process.env,
        ORIGINAL_DIRECTORY: directory,
        CHECK_CLI_HEAP: 'true',
        NODE_OPTIONS: '--max-old-space-size=512'
      }
    }
  );
  const output = `${result.stdout}${result.stderr}`;
  expect(result.status, output).toBe(0);
  const heaps = [...output.matchAll(/release-child-heap-mib=([\d.]+)/g)].map(match =>
    Number(match[1])
  );
  expect(heaps).toHaveLength(2);
  for (const heap of heaps) {
    expect(heap).toBeGreaterThan(256);
    expect(heap).toBeLessThan(384);
  }
});

it('compiles a tiny three-package snapshot with real project references and loads its native ESM/CommonJS entries', () => {
  const directory = fixture();
  for (const script of [
    'build.mjs',
    'build-commonjs.mjs',
    'typescript-project-inputs.mjs',
    'verification-policy.mjs',
    'build-record.mjs',
    'verify-package-exports.mjs'
  ]) {
    copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
  }
  for (const dependency of ['typescript', 'esbuild', '@vis.gl']) {
    mkdirSync(join(directory, 'node_modules'), {recursive: true});
    symlinkSync(
      join(ROOT, 'node_modules', dependency),
      join(directory, 'node_modules', dependency),
      'dir'
    );
  }
  write(
    directory,
    'node_modules/.bin/tspc',
    `#!${process.execPath}\nrequire('node:child_process').execFileSync(process.execPath, [${JSON.stringify(join(ROOT, 'node_modules/ts-patch/bin/tspc.js'))}, ...process.argv.slice(2)], {stdio:'inherit'});`
  );
  chmodSync(join(directory, 'node_modules/.bin/tspc'), 0o755);
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ES2022',
        moduleResolution: 'bundler',
        types: [],
        lib: ['ES2022'],
        strict: true,
        declaration: true,
        plugins: JSON.parse(readFileSync(join(ROOT, 'tsconfig.json'), 'utf8')).compilerOptions
          .plugins,
        paths: Object.fromEntries(
          MODULES.map(moduleName => [
            `@luma-contributors/${moduleName}`,
            [`./modules/${moduleName}/src/index.ts`]
          ])
        )
      }
    })
  );
  write(
    directory,
    'tsconfig.build.json',
    JSON.stringify({
      files: [],
      references: MODULES.map(moduleName => ({path: `./modules/${moduleName}`}))
    })
  );
  for (const [index, moduleName] of MODULES.entries()) {
    write(
      directory,
      `modules/${moduleName}/tsconfig.json`,
      JSON.stringify({
        extends: '../../tsconfig.json',
        include: ['src/**/*'],
        compilerOptions: {composite: true, rootDir: 'src', outDir: 'dist'},
        references: MODULES.slice(0, index).map(dependency => ({path: `../${dependency}`}))
      })
    );
    write(
      directory,
      `modules/${moduleName}/package.json`,
      JSON.stringify({
        name: `@luma-contributors/${moduleName}`,
        type: 'module',
        exports: {
          '.': {types: './dist/index.d.ts', import: './dist/index.js', require: './dist/index.cjs'}
        }
      })
    );
  }
  write(directory, 'modules/analysis/src/value.ts', 'export const value = 2;');
  write(directory, 'modules/analysis/src/index.ts', "export {value} from './value';");
  write(
    directory,
    'modules/lenses/src/index.ts',
    "import {value} from '@luma-contributors/analysis'; export const lensValue = value + 1;"
  );
  write(
    directory,
    'modules/kepler/src/index.ts',
    "import {lensValue} from '@luma-contributors/lenses'; export const adapterValue = lensValue + 1;"
  );
  const result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('verified 3 entry points');
  assertOldOutputs(directory);
});
