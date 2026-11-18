// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  statSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const MODULES = ['analysis', 'lenses', 'kepler'];
const directories: string[] = [];

function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function fixture(value = 1) {
  const directory = mkdtempSync(join(tmpdir(), 'build-review-'));
  directories.push(directory);
  for (const script of [
    'build.mjs',
    'build-commonjs.mjs',
    'typescript-project-inputs.mjs',
    'verification-policy.mjs',
    'verify-package-exports.mjs',
    'build-record.mjs'
  ]) {
    if (existsSync(join(ROOT, 'scripts', script))) {
      mkdirSync(join(directory, 'scripts'), {recursive: true});
      copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
    }
  }
  write(directory, 'package.json', '{"type":"module"}');
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
        declaration: true
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
  for (const dependency of ['typescript', 'esbuild']) {
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
    `#!${process.execPath}\nrequire('node:child_process').execFileSync(process.execPath, [${JSON.stringify(join(ROOT, 'node_modules/typescript/lib/tsc.js'))}, ...process.argv.slice(2)], {stdio:'inherit'});`
  );
  chmodSync(join(directory, 'node_modules/.bin/tspc'), 0o755);
  for (const moduleName of MODULES) {
    write(
      directory,
      `modules/${moduleName}/src/index.ts`,
      `export function readValue(): number { return ${value}; }`
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
    write(
      directory,
      `modules/${moduleName}/tsconfig.json`,
      JSON.stringify({
        extends: '../../tsconfig.json',
        include: ['src/**/*'],
        compilerOptions: {composite: true, rootDir: 'src', outDir: 'dist'}
      })
    );
    mkdirSync(join(directory, 'node_modules/@luma-contributors'), {recursive: true});
    symlinkSync(
      join(directory, 'modules', moduleName),
      join(directory, 'node_modules/@luma-contributors', moduleName),
      'dir'
    );
  }
  return directory;
}

function run(directory: string, script = 'build.mjs', environment = {}) {
  const result = spawnSync(process.execPath, [join(directory, 'scripts', script)], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 15000,
    env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256', ...environment}
  });
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

function readValue(
  directory: string,
  format: 'js' | 'cjs',
  name = 'index',
  exportedName = 'readValue'
) {
  const file = join(directory, `modules/analysis/dist/${name}.${format}`);
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `const values = await import(${JSON.stringify(file)}); process.stdout.write(String(values[${JSON.stringify(exportedName)}]()));`
    ],
    {encoding: 'utf8', timeout: 10000}
  );
  expect(result.status, result.stderr).toBe(0);
  return Number(result.stdout);
}

it('R3 accepts unchanged declarations and metadata-only rebuilds, but rejects stale source and corrupt output', () => {
  const directory = fixture();
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  write(
    directory,
    'modules/analysis/src/index.ts',
    'export function readValue(): number { return 2; }'
  );
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  let gate = run(directory, 'verify-package-exports.mjs');
  expect(gate.status, gate.output).toBe(0);
  const manifest = JSON.parse(
    readFileSync(join(directory, 'modules/analysis/package.json'), 'utf8')
  );
  write(
    directory,
    'modules/analysis/package.json',
    JSON.stringify({...manifest, description: 'metadata edit'})
  );
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  gate = run(directory, 'verify-package-exports.mjs');
  expect(gate.status, gate.output).toBe(0);
  write(
    directory,
    'modules/analysis/src/index.ts',
    'export function readValue(): number { return 3; }'
  );
  expect(run(directory, 'verify-package-exports.mjs').status).toBe(1);
  expect(run(directory).status).toBe(0);
  write(
    directory,
    'modules/analysis/dist/index.d.ts',
    'export declare function readValue(): string;'
  );
  expect(run(directory, 'verify-package-exports.mjs').status).toBe(1);
});

it('R4 invalidates a promoted developer cache before a source revert and rebuilds ESM and CommonJS', async () => {
  const {promotePackages} = await import('../../../../scripts/build-release.mjs');
  const directory = fixture(1);
  const staged = fixture(2);
  expect(run(directory).status).toBe(0);
  expect(run(staged).status).toBe(0);
  write(
    directory,
    'modules/analysis/src/index.ts',
    'export function readValue(): number { return 2; }'
  );
  const releaseDirectory = join(directory, 'release');
  for (const moduleName of MODULES)
    cpSync(
      join(staged, 'modules', moduleName, 'dist'),
      join(releaseDirectory, 'workspace/modules', moduleName, 'dist'),
      {recursive: true}
    );
  promotePackages(directory, releaseDirectory);
  write(
    directory,
    'modules/analysis/src/index.ts',
    'export function readValue(): number { return 1; }'
  );
  const result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readValue(directory, 'js')).toBe(1);
  expect(readValue(directory, 'cjs')).toBe(1);
});

it('R5 retries a failed CommonJS write without certifying a truncated target', () => {
  const directory = fixture(1);
  expect(run(directory).status).toBe(0);
  write(
    directory,
    'modules/analysis/src/index.ts',
    'export function readValue(): number { return 2; }'
  );
  write(
    directory,
    'fail-write.cjs',
    `const fileSystem = require('node:fs'); const promises = require('node:fs/promises'); const originalWrite = promises.writeFile; let failed = false; promises.writeFile = async (file, ...values) => { const path = String(file); if (!failed && path.includes('/analysis/dist/') && path.includes('index.cjs')) { failed = true; fileSystem.writeFileSync(file, ''); const future = new Date(Date.now() + 60000); fileSystem.utimesSync(file, future, future); throw new Error('injected partial write'); } return originalWrite(file, ...values); }; require('node:module').syncBuiltinESMExports();`
  );
  const failed = run(directory, 'build.mjs', {
    NODE_OPTIONS: `--max-old-space-size=256 --require=${join(directory, 'fail-write.cjs')}`
  });
  expect(failed.status, failed.output).toBe(1);
  const retry = run(directory);
  expect(retry.status, retry.output).toBe(0);
  expect(readValue(directory, 'cjs')).toBe(2);
});

it('R8 repairs missing compiler-native CommonJS outputs from cts sources', () => {
  const directory = fixture();
  write(
    directory,
    'modules/analysis/src/native.cts',
    'export function nativeValue(): number { return 7; }'
  );
  expect(run(directory).status).toBe(0);
  rmSync(join(directory, 'modules/analysis/dist/native.cjs'));
  const repaired = run(directory);
  expect(repaired.status, repaired.output).toBe(0);
  expect(existsSync(join(directory, 'modules/analysis/dist/native.cjs'))).toBe(true);
  expect(readValue(directory, 'cjs', 'native', 'nativeValue')).toBe(7);
});

it('final1 F2 rebuilds a backdated content change before certifying either native format', () => {
  const directory = fixture(1);
  expect(run(directory).status).toBe(0);
  // Restoring output metadata must not make an older CJS implementation appear reusable.
  const laterOutputTime = new Date(Date.now() + 60000);
  utimesSync(join(directory, 'modules/analysis/dist/index.cjs'), laterOutputTime, laterOutputTime);
  const source = join(directory, 'modules/analysis/src/index.ts');
  writeFileSync(source, 'export function readValue(): number { return 2; }');
  const oldTime = new Date('2001-01-01T00:00:00Z');
  utimesSync(source, oldTime, oldTime);
  const rebuilt = run(directory);
  expect(rebuilt.status, rebuilt.output).toBe(0);
  expect(readValue(directory, 'js')).toBe(2);
  expect(readValue(directory, 'cjs')).toBe(2);
  const verified = run(directory, 'verify-package-exports.mjs');
  expect(verified.status, verified.output).toBe(0);
});

it('final2 G2 rebuilds inputs that share the compiler cache millisecond', () => {
  const directory = fixture(1);
  expect(run(directory).status).toBe(0);
  const source = join(directory, 'modules/analysis/src/index.ts');
  const cache = join(directory, 'modules/analysis/tsconfig.tsbuildinfo');
  writeFileSync(source, 'export function readValue(): number { return 2; }');
  const future = Math.ceil(Date.now() / 1000) * 1000 + 10000;
  utimesSync(cache, (future + 0.1) / 1000, (future + 0.1) / 1000);
  utimesSync(source, (future + 0.3) / 1000, (future + 0.3) / 1000);
  expect(statSync(source).mtimeMs).toBeGreaterThan(statSync(cache).mtimeMs);
  expect(statSync(source).mtime.getTime()).toBe(statSync(cache).mtime.getTime());
  const rebuilt = run(directory);
  expect(rebuilt.status, rebuilt.output).toBe(0);
  expect(readValue(directory, 'js')).toBe(2);
  expect(readValue(directory, 'cjs')).toBe(2);
});

it('final2 G2 rebuilds legacy warm outputs before establishing the first build record', () => {
  const directory = fixture(1);
  expect(run(directory).status).toBe(0);
  rmSync(join(directory, 'modules/analysis/dist/.build-record.json'));
  const source = join(directory, 'modules/analysis/src/index.ts');
  writeFileSync(source, 'export function readValue(): number { return 2; }');
  const past = new Date('2001-01-01T00:00:00Z');
  utimesSync(source, past, past);
  const rebuilt = run(directory);
  expect(rebuilt.status, rebuilt.output).toBe(0);
  expect(readValue(directory, 'js')).toBe(2);
  expect(readValue(directory, 'cjs')).toBe(2);
});

it('final3 H2 preserves a standalone caller heap above the verify ceiling in the compiler', () => {
  const directory = fixture(1);
  write(
    directory,
    'node_modules/.bin/tspc',
    `#!${process.execPath}\nrequire('node:fs').writeFileSync('compiler-options.json', JSON.stringify(process.env.NODE_OPTIONS)); require('node:child_process').execFileSync(process.execPath, [${JSON.stringify(join(ROOT, 'node_modules/typescript/lib/tsc.js'))}, ...process.argv.slice(2)], {stdio:'inherit'});`
  );
  const options = '--max-old-space-size=12288 --trace-warnings';
  const result = run(directory, 'build.mjs', {NODE_OPTIONS: options});
  expect(result.status, result.output).toBe(0);
  expect(JSON.parse(readFileSync(join(directory, 'compiler-options.json'), 'utf8'))).toBe(options);
  expect(readValue(directory, 'js')).toBe(1);
  expect(readValue(directory, 'cjs')).toBe(1);
});

it.each([
  'analysis',
  'lenses'
])('final3 H4 refuses all build records when analysis changes during the %s CommonJS pass and allows retry', moduleName => {
  const directory = fixture(1);
  expect(run(directory).status).toBe(0);
  const record = join(directory, 'modules/analysis/dist/.build-record.json');
  const previous = readFileSync(record, 'utf8');
  const source = join(directory, 'modules/analysis/src/index.ts');
  // Force new output bytes, so publishing analysis early would replace its prior record.
  writeFileSync(source, 'export function readValue(): number { return 3; }');
  write(
    directory,
    'edit-during-build.cjs',
    `const fileSystem = require('node:fs'); const promises = require('node:fs/promises'); const originalRead = promises.readFile; let changed = false; promises.readFile = async (file, ...arguments_) => { if (!changed && String(file).endsWith('/${moduleName}/dist/index.js')) { changed = true; fileSystem.writeFileSync(${JSON.stringify(source)}, 'export function readValue(): number { return 2; }'); } return originalRead(file, ...arguments_); }; require('node:module').syncBuiltinESMExports();`
  );
  const changed = run(directory, 'build.mjs', {
    NODE_OPTIONS: `--max-old-space-size=256 --require=${join(directory, 'edit-during-build.cjs')}`
  });
  expect(changed.status, changed.output).toBe(1);
  expect(changed.output).toContain('inputs changed during build');
  expect(readFileSync(record, 'utf8')).toBe(previous);
  const retry = run(directory);
  expect(retry.status, retry.output).toBe(0);
  expect(readValue(directory, 'js')).toBe(2);
  expect(readValue(directory, 'cjs')).toBe(2);
});
