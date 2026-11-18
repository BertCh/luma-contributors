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
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];

function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
}

function createFixture() {
  const directory = mkdtempSync(join(tmpdir(), 'measure-typecheck-'));
  directories.push(directory);
  for (const file of ['measure-typecheck.mjs', 'verification-policy.mjs']) {
    expect(existsSync(join(ROOT, 'scripts', file)), `missing scripts/${file}`).toBe(true);
    mkdirSync(join(directory, 'scripts'), {recursive: true});
    copyFileSync(join(ROOT, 'scripts', file), join(directory, 'scripts', file));
  }
  write(directory, 'package.json', '{"type":"module"}');
  // Tiny fixture compilers do not consume GiB; make resource-policy tests independent of
  // shared-machine disk pressure. This preload exists only in the temporary fixture.
  write(
    directory,
    'fixture-disk.cjs',
    `const fs = require('node:fs'); fs.statfsSync = () => ({bavail: Number(process.env.FIXTURE_DISK_GIB ?? 64) * 1024 ** 3, bsize: 1}); require('node:module').syncBuiltinESMExports();`
  );
  mkdirSync(join(directory, 'node_modules'), {recursive: true});
  symlinkSync(
    join(ROOT, 'node_modules/typescript'),
    join(directory, 'node_modules/typescript'),
    'dir'
  );
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        incremental: true,
        noEmit: true,
        tsBuildInfoFile: './protected-cache.tsbuildinfo'
      },
      files: ['source.ts']
    })
  );
  write(directory, 'source.ts', 'export const value: number = 1;');
  write(directory, 'protected-cache.tsbuildinfo', 'original cache');
  // The compiler stand-in records cold/warm cache state and creates only the selected cache.
  write(
    directory,
    'compiler.mjs',
    `
import {readFileSync, writeFileSync, existsSync, appendFileSync, renameSync} from 'node:fs';
const projectArgument = process.argv.findIndex(argument => argument === '-p' || argument === '-b');
const configuration = process.argv[projectArgument + 1];
const project = JSON.parse(readFileSync(configuration, 'utf8'));
const cache = project.compilerOptions.tsBuildInfoFile;
const wasWarm = cache ? existsSync(cache) : false;
writeFileSync('compiler.pid', String(process.pid));
if (wasWarm && process.env.COMPILER_STATE_DELAY) await new Promise(resolve => setTimeout(resolve, Number(process.env.COMPILER_STATE_DELAY)));
writeFileSync('compiler-state.tmp', JSON.stringify({processId: process.pid, warm: wasWarm}));
renameSync('compiler-state.tmp', 'compiler-state.json');
appendFileSync('runs.jsonl', JSON.stringify({cache, warm: cache ? existsSync(cache) : false, configuration, options: process.env.NODE_OPTIONS}) + '\\n');
if (cache) writeFileSync(cache, 'isolated cache');
if (process.env.MUTATE_INPUT && (!process.env.MUTATE_ON_WARM || wasWarm)) appendFileSync(process.env.MUTATE_INPUT, '\\n// concurrent edit');
console.log('Files: 7');
await new Promise(resolve => setTimeout(resolve, Number(process.env.COMPILER_DELAY ?? 120)));
if (process.env.FAIL_COMPILER) process.exit(2);
`
  );
  write(
    directory,
    'bin/ps',
    `#!${process.execPath}\nprocess.stderr.write('fixture ps denied'); process.exit(1);`
  );
  chmodSync(join(directory, 'bin/ps'), 0o755);
  return directory;
}

function run(
  directory: string,
  arguments_: string[] = [],
  environment: Record<string, string> = {},
  compiler: string | null = 'compiler.mjs'
) {
  return spawnSync(
    process.execPath,
    [
      'scripts/measure-typecheck.mjs',
      '--project',
      'tsconfig.json',
      ...(compiler === null ? [] : ['--compiler', compiler]),
      ...arguments_
    ],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 15000,
      env: {
        ...process.env,
        NODE_OPTIONS: `--max-old-space-size=256 --trace-warnings --require=${JSON.stringify(join(directory, 'fixture-disk.cjs'))}`,
        PATH: `${join(directory, 'bin')}:${process.env['PATH']}`,
        ...environment
      }
    }
  );
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it('measures cold then warm in a private cache without touching the original cache or config', () => {
  const directory = createFixture();
  const original = readFileSync(join(directory, 'tsconfig.json'), 'utf8');
  const result = run(directory, ['--timing-only']);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const runs = readFileSync(join(directory, 'runs.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  expect(runs.map(run => run.warm)).toEqual([false, true]);
  expect(runs[0].cache).toBe(runs[1].cache);
  expect(runs[0].cache).toContain(join(directory, '.typecheck/measurements/'));
  expect(runs[0].options).toContain('--max-old-space-size=256');
  expect(readFileSync(join(directory, 'protected-cache.tsbuildinfo'), 'utf8')).toBe(
    'original cache'
  );
  expect(readFileSync(join(directory, 'tsconfig.json'), 'utf8')).toBe(original);
  expect(result.stdout).toContain('| tsconfig.json | 7 |');
  expect(result.stdout).toContain('unavailable (timing only)');
  const reportMatch = result.stdout.match(/Report: (.*)/);
  const report = JSON.parse(readFileSync(join(directory, reportMatch![1]), 'utf8'));
  expect(report.inputFingerprint).toMatch(/^[a-f0-9]{64}$/);
});

it('refuses unavailable RSS before starting a compiler and never reports zero as a measurement', () => {
  const directory = createFixture();
  const result = run(directory);
  expect(result.status).toBe(1);
  expect(`${result.stdout}${result.stderr}`).toContain('cannot monitor process-tree RSS');
  expect(existsSync(join(directory, 'runs.jsonl'))).toBe(false);
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it('fails on compiler errors without publishing a passing comparison or running the warm check', () => {
  const directory = createFixture();
  const result = run(directory, ['--timing-only'], {FAIL_COMPILER: '1'});
  expect(result.status).toBe(1);
  expect(`${result.stdout}${result.stderr}`).toContain('cold compiler exited with 2');
  expect(readFileSync(join(directory, 'runs.jsonl'), 'utf8').trim().split('\n')).toHaveLength(1);
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it('uses the verify sampler to sum descendants and exclude unrelated processes', () => {
  const directory = createFixture();
  write(
    directory,
    'bin/ps',
    `#!${process.execPath}\nconsole.log('10 1 100\\n11 10 200\\n12 11 300\\n20 1 900');`
  );
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      "import {readTreeRSS} from './scripts/verification-policy.mjs'; console.log(readTreeRSS(10));"
    ],
    {
      cwd: directory,
      encoding: 'utf8',
      env: {...process.env, PATH: `${join(directory, 'bin')}:${process.env['PATH']}`}
    }
  );
  expect(result.status, result.stderr).toBe(0);
  expect(Number(result.stdout.trim())).toBe(600 * 1024);
  expect(readFileSync(join(ROOT, 'scripts/verify.mjs'), 'utf8')).toContain('readTreeRSS');
});

it('preserves a nonincremental baseline instead of making its warm run artificially faster', () => {
  const directory = createFixture();
  write(directory, 'tsconfig.json', JSON.stringify({files: ['source.ts']}));
  const result = run(directory, ['--timing-only']);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const runs = readFileSync(join(directory, 'runs.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  expect(runs.map(run => run.warm)).toEqual([false, false]);
  expect(runs.every(run => run.cache === undefined)).toBe(true);
});

it('enforces the disk floor even when RSS measurement is explicitly disabled', () => {
  const directory = createFixture();
  const result = run(directory, ['--timing-only'], {FIXTURE_DISK_GIB: '7'});
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('8 GiB floor');
  expect(existsSync(join(directory, 'runs.jsonl'))).toBe(false);
});

it('isolates an actual composite reference build and reuses its private outputs on the warm run', () => {
  const directory = createFixture();
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({files: [], references: [{path: './analysis'}, {path: './lenses'}]})
  );
  for (const packageName of ['analysis', 'lenses']) {
    write(
      directory,
      `${packageName}/tsconfig.json`,
      JSON.stringify({
        compilerOptions: {
          composite: true,
          rootDir: 'src',
          outDir: 'dist',
          skipLibCheck: true,
          types: []
        },
        include: ['src'],
        references: packageName === 'lenses' ? [{path: '../analysis'}] : []
      })
    );
    write(directory, `${packageName}/dist/keep.txt`, 'original output');
    write(directory, `${packageName}/tsconfig.tsbuildinfo`, 'original compiler cache');
  }
  write(directory, 'analysis/src/index.ts', 'export const value: number = 3;');
  write(
    directory,
    'lenses/src/index.ts',
    "import {value} from '../../analysis/src/index'; export const doubled = value * 2;"
  );
  for (const mode of ['declarations', 'build']) {
    const result = run(directory, [
      '--compiler',
      join(ROOT, 'node_modules/typescript/bin/tsc'),
      '--mode',
      mode,
      '--timing-only'
    ]);
    expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
    const reportMatch = result.stdout.match(/Report: (.*)/);
    expect(reportMatch).not.toBeNull();
    const reportFile = join(directory, reportMatch![1]);
    const report = JSON.parse(readFileSync(reportFile, 'utf8'));
    expect(report.cold.files).toBeGreaterThan(0);
    expect(report.warm.files).toBe(0);
    expect(existsSync(join(dirname(reportFile), 'output-1/index.d.ts'))).toBe(true);
    expect(existsSync(join(dirname(reportFile), 'output-2/index.d.ts'))).toBe(true);
    expect(existsSync(join(dirname(reportFile), 'output-2/index.js'))).toBe(mode === 'build');
  }
  for (const packageName of ['analysis', 'lenses']) {
    expect(readFileSync(join(directory, `${packageName}/dist/keep.txt`), 'utf8')).toBe(
      'original output'
    );
    expect(readFileSync(join(directory, `${packageName}/tsconfig.tsbuildinfo`), 'utf8')).toBe(
      'original compiler cache'
    );
    expect(existsSync(join(directory, `${packageName}/dist/index.d.ts`))).toBe(false);
  }
});

it('rejects symlinked cache roots before any compiler can write outside .typecheck', () => {
  const directory = createFixture();
  mkdirSync(join(directory, 'protected'));
  symlinkSync(join(directory, 'protected'), join(directory, '.typecheck'), 'dir');
  const result = run(directory, ['--timing-only']);
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('.typecheck must not be a symlink');
  expect(existsSync(join(directory, 'runs.jsonl'))).toBe(false);
});

it('records the sampled compiler descendant peak separately from the measurement wrapper', () => {
  const directory = createFixture();
  write(
    directory,
    'bin/ps',
    `#!${process.execPath}
import fs from 'node:fs';
const compiler = fs.existsSync('compiler.pid') ? Number(fs.readFileSync('compiler.pid', 'utf8')) : 999999;
console.log(process.ppid + ' 1 9000');
console.log(compiler + ' ' + process.ppid + ' 100');
console.log('999997 ' + compiler + ' 200');
console.log('999998 999997 300');
console.log('999996 1 12000');
`
  );
  const result = run(directory, [], {COMPILER_DELAY: '1100'});
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const reportMatch = result.stdout.match(/Report: (.*)/);
  const report = JSON.parse(readFileSync(join(directory, reportMatch![1]), 'utf8'));
  expect(report.cold.peakRSSBytes).toBe(600 * 1024);
  expect(report.warm.peakRSSBytes).toBe(600 * 1024);
});

it.each([
  'check',
  'declarations',
  'build'
])('refuses a %s comparison if source bytes change during the cold run', mode => {
  const directory = createFixture();
  const result = run(directory, ['--mode', mode, '--timing-only'], {
    MUTATE_INPUT: join(directory, 'source.ts')
  });
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain('inputs changed during cold measurement');
  expect(readFileSync(join(directory, 'runs.jsonl'), 'utf8').trim().split('\n')).toHaveLength(1);
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it('includes imported source dependencies in the measurement input guard', () => {
  const directory = createFixture();
  write(directory, 'source.ts', "import {value} from './dependency'; export {value};");
  write(directory, 'dependency.ts', 'export const value = 1;');
  const result = run(directory, ['--timing-only'], {
    MUTATE_INPUT: join(directory, 'dependency.ts')
  });
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain('inputs changed during cold measurement');
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it.each([
  'check',
  'declarations',
  'build'
])('refuses a %s comparison if source bytes change during the warm run', mode => {
  const directory = createFixture();
  const result = run(directory, ['--mode', mode, '--timing-only'], {
    MUTATE_INPUT: join(directory, 'source.ts'),
    MUTATE_ON_WARM: '1'
  });
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain('inputs changed during warm measurement');
  expect(readFileSync(join(directory, 'runs.jsonl'), 'utf8').trim().split('\n')).toHaveLength(2);
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it('includes configuration edits and newly discovered roots in the input guard', () => {
  for (const target of ['tsconfig.json', 'added.ts']) {
    const directory = createFixture();
    const configuration = JSON.parse(readFileSync(join(directory, 'tsconfig.json'), 'utf8'));
    delete configuration.files;
    configuration.include = ['*.ts'];
    write(directory, 'tsconfig.json', JSON.stringify(configuration));
    const result = run(directory, ['--timing-only'], {MUTATE_INPUT: join(directory, target)});
    expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
    expect(result.stderr).toContain('inputs changed during cold measurement');
    expect(result.stdout).not.toContain('| tsconfig.json |');
  }
});

function createPackageFixture() {
  const directory = createFixture();
  symlinkSync(join(ROOT, 'scripts/build.mjs'), join(directory, 'scripts/build.mjs'));
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ES2022',
        moduleResolution: 'bundler',
        types: [],
        skipLibCheck: true
      }
    })
  );
  const packageNames = ['analysis', 'lenses', 'kepler'];
  write(
    directory,
    'tsconfig.build.json',
    JSON.stringify({files: [], references: packageNames.map(name => ({path: `./modules/${name}`}))})
  );
  for (const packageName of packageNames) {
    write(
      directory,
      `modules/${packageName}/package.json`,
      JSON.stringify({name: `fixture-${packageName}`, type: 'module'})
    );
    write(
      directory,
      `modules/${packageName}/tsconfig.json`,
      JSON.stringify({
        extends: '../../tsconfig.json',
        compilerOptions: {composite: true, rootDir: 'src', outDir: 'dist'},
        include: ['src']
      })
    );
    write(directory, `modules/${packageName}/src/index.ts`, 'export const value = 3;');
    write(directory, `modules/${packageName}/dist/keep.txt`, 'original output');
    write(directory, `modules/${packageName}/tsconfig.tsbuildinfo`, 'original cache');
  }
  return directory;
}

it('measures the real package build pipeline including cold and reused CommonJS outputs in a private source snapshot', () => {
  const directory = createPackageFixture();
  const packageNames = ['analysis', 'lenses', 'kepler'];
  const result = run(directory, [
    '--project',
    'tsconfig.build.json',
    '--mode',
    'packages',
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const reportMatch = result.stdout.match(/Report: (.*)/);
  const reportFile = join(directory, reportMatch![1]);
  const snapshot = join(dirname(reportFile), 'workspace');
  expect(readFileSync(join(dirname(reportFile), 'cold.log'), 'utf8')).toContain(
    'analysis: wrote 1 CommonJS modules'
  );
  expect(readFileSync(join(dirname(reportFile), 'warm.log'), 'utf8')).toContain(
    'analysis: wrote 0 CommonJS modules'
  );
  for (const packageName of packageNames) {
    expect(existsSync(join(snapshot, `modules/${packageName}/dist/index.cjs`))).toBe(true);
    const consumer = spawnSync(
      process.execPath,
      [
        '-e',
        'process.stdout.write(String(require(process.argv[1]).value));',
        join(snapshot, `modules/${packageName}/dist/index.cjs`)
      ],
      {encoding: 'utf8'}
    );
    expect(consumer.status, consumer.stderr).toBe(0);
    expect(consumer.stdout).toBe('3');
    expect(existsSync(join(snapshot, `modules/${packageName}/dist/.build-record.json`))).toBe(true);
    expect(readFileSync(join(directory, `modules/${packageName}/dist/keep.txt`), 'utf8')).toBe(
      'original output'
    );
    expect(
      readFileSync(join(directory, `modules/${packageName}/tsconfig.tsbuildinfo`), 'utf8')
    ).toBe('original cache');
    expect(existsSync(join(directory, `modules/${packageName}/dist/index.cjs`))).toBe(false);
  }
});

it.each([
  'tsBuildInfoFile',
  'outDir'
])('R1 refuses an absolute package %s outside the private workspace before changing sentinel bytes', option => {
  const directory = createPackageFixture();
  const sentinel =
    option === 'tsBuildInfoFile' ? 'developer.tsbuildinfo' : 'developer-output/index.js';
  write(directory, sentinel, 'protected developer bytes');
  const configurationPath = join(directory, 'modules/analysis/tsconfig.json');
  const configuration = JSON.parse(readFileSync(configurationPath, 'utf8'));
  configuration.compilerOptions[option] = join(
    directory,
    option === 'tsBuildInfoFile' ? sentinel : 'developer-output'
  );
  writeFileSync(configurationPath, JSON.stringify(configuration));
  const result = run(directory, [
    '--project',
    'tsconfig.build.json',
    '--mode',
    'packages',
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(readFileSync(join(directory, sentinel), 'utf8')).toBe('protected developer bytes');
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain('outside private package workspace');
});

it.each([
  'cold',
  'warm'
])('R2 refuses a JSDoc-only dependency change during %s while both real compilations remain valid', interval => {
  const directory = createFixture();
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        rootDir: '.',
        allowJs: true,
        checkJs: true,
        noEmit: true,
        types: [],
        incremental: true,
        tsBuildInfoFile: './protected-cache.tsbuildinfo'
      },
      files: ['index.js']
    })
  );
  write(
    directory,
    'index.js',
    '/** @import {Answer} from "./model" */\n/** @type {Answer} */\nexport const answer = 1;'
  );
  write(directory, 'model.d.ts', 'export type Answer = number;');
  write(
    directory,
    'jsdoc-compiler.mjs',
    `
import {spawnSync} from 'node:child_process';
import {existsSync, writeFileSync} from 'node:fs';
const warm = existsSync('compiled-once');
const result = spawnSync(${JSON.stringify(process.execPath)}, [${JSON.stringify(join(ROOT, 'node_modules/typescript/bin/tsc'))}, ...process.argv.slice(2)], {stdio: 'inherit'});
if (result.status !== 0) process.exit(result.status ?? 1);
writeFileSync('compiled-once', 'yes');
if (${JSON.stringify(interval)} === 'cold' || warm) writeFileSync('model.d.ts', 'export type Answer = number | string;');
`
  );
  const result = run(directory, ['--compiler', 'jsdoc-compiler.mjs', '--timing-only']);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain(`inputs changed during ${interval} measurement`);
  expect(result.stdout).not.toContain('| tsconfig.json |');
});

it('R3 preserves implicit roots and surfaces the original source diagnostic with the real compiler', () => {
  const directory = createFixture();
  write(directory, 'tsconfig.json', JSON.stringify({compilerOptions: {noEmit: true, types: []}}));
  write(directory, 'index.ts', 'export const value: string = 1;');
  const result = run(directory, [
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(result.status).toBe(1);
  expect(result.stderr).toContain('TS2322');
  expect(result.stderr).toContain('index.ts');
});

it('R4 defaults packages to the patched compiler and loads sibling re-exports in native ESM and CommonJS', () => {
  const directory = createPackageFixture();
  for (const dependency of ['ts-patch', '@vis.gl'])
    symlinkSync(
      join(ROOT, 'node_modules', dependency),
      join(directory, 'node_modules', dependency),
      'dir'
    );
  const configuration = JSON.parse(readFileSync(join(directory, 'tsconfig.json'), 'utf8'));
  configuration.compilerOptions.plugins = JSON.parse(
    readFileSync(join(ROOT, 'tsconfig.json'), 'utf8')
  ).compilerOptions.plugins;
  write(directory, 'tsconfig.json', JSON.stringify(configuration));
  for (const packageName of ['analysis', 'lenses', 'kepler']) {
    write(directory, `modules/${packageName}/src/index.ts`, "export {value} from './value';");
    write(directory, `modules/${packageName}/src/value.ts`, 'export const value = 3;');
  }
  const result = run(
    directory,
    ['--project', 'tsconfig.build.json', '--mode', 'packages', '--timing-only'],
    {},
    null
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const reportMatch = result.stdout.match(/Report: (.*)/);
  const snapshot = join(directory, dirname(reportMatch![1]), 'workspace');
  for (const packageName of ['analysis', 'lenses', 'kepler']) {
    const entry = join(snapshot, `modules/${packageName}/dist/index`);
    const consumer = spawnSync(
      process.execPath,
      [
        '--input-type=module',
        '-e',
        "const esm = await import(process.argv[1]); const {createRequire} = await import('node:module'); const commonjs = createRequire(import.meta.url)(process.argv[2]); console.log(esm.value, commonjs.value);",
        `${entry}.js`,
        `${entry}.cjs`
      ],
      {encoding: 'utf8'}
    );
    expect(consumer.status, consumer.stderr).toBe(0);
    expect(consumer.stdout.trim()).toBe('3 3');
  }
});

it.each([
  'cold',
  'warm'
])('R5 refuses a measured comparison if successful ps listings omit the %s compiler', interval => {
  const directory = createFixture();
  write(
    directory,
    'bin/ps',
    `#!${process.execPath}
import fs from 'node:fs';
console.log(process.ppid + ' 1 9000');
if (${JSON.stringify(interval)} === 'warm') await new Promise(resolve => setTimeout(resolve, 100));
const state = fs.existsSync('compiler-state.json') ? JSON.parse(fs.readFileSync('compiler-state.json', 'utf8')) : undefined;
const reportsCompiler = ${JSON.stringify(interval)} === 'warm' && state?.warm === false;
fs.appendFileSync('ps-observations.jsonl', JSON.stringify({warm: state?.warm ?? null, reportsCompiler}) + '\\n');
if (reportsCompiler) console.log(state.processId + ' ' + process.ppid + ' 100');
`
  );
  const result = run(directory, [], {COMPILER_DELAY: '1100', COMPILER_STATE_DELAY: '250'});
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(result.stderr).toContain(`no positive RSS sample observed for ${interval}`);
  expect(result.stdout).not.toContain('| tsconfig.json |');
  const runs = readFileSync(join(directory, 'runs.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  expect(runs.map(run => run.warm)).toEqual(interval === 'warm' ? [false, true] : [false]);
  if (interval === 'warm') {
    const observations = readFileSync(join(directory, 'ps-observations.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .map(line => JSON.parse(line));
    expect(
      observations.some(observation => observation.warm === false && observation.reportsCompiler)
    ).toBe(true);
    const warmObservations = observations.filter(observation => observation.warm === true);
    expect(warmObservations.length).toBeGreaterThan(0);
    expect(warmObservations.every(observation => !observation.reportsCompiler)).toBe(true);
  }
});

it.each([
  'check',
  'build',
  'declarations'
])('X4 preserves app-local ambient declarations instead of checkout-root types in %s mode', mode => {
  const directory = createFixture();
  write(
    directory,
    'app/tsconfig.json',
    JSON.stringify({
      compilerOptions: {rootDir: '.', types: ['fixture-only'], noEmit: true},
      files: ['index.ts']
    })
  );
  write(directory, 'app/index.ts', "export const value: 'app' = ambientSource;");
  write(
    directory,
    'app/node_modules/@types/fixture-only/index.d.ts',
    "declare const ambientSource: 'app';"
  );
  write(
    directory,
    'node_modules/@types/fixture-only/index.d.ts',
    "declare const ambientSource: 'checkout';"
  );
  const original = spawnSync(
    process.execPath,
    [
      join(ROOT, 'node_modules/typescript/bin/tsc'),
      '-p',
      join(directory, 'app/tsconfig.json'),
      '--pretty',
      'false'
    ],
    {encoding: 'utf8'}
  );
  expect(original.status, `${original.stdout}${original.stderr}`).toBe(0);
  const result = run(directory, [
    '--project',
    'app/tsconfig.json',
    '--mode',
    mode,
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  const runMatch = result.stdout.match(/measure-typecheck: (\S+);/);
  expect(runMatch, result.stdout).not.toBeNull();
  const resolveDeclaration = (configuration: string) => {
    const project = typescript.getParsedCommandLineOfConfigFile(
      configuration,
      {},
      {
        ...typescript.sys,
        onUnRecoverableConfigFileDiagnostic: diagnostic => {
          throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
        }
      }
    );
    if (!project) throw new Error('missing fixture project');
    return typescript.resolveTypeReferenceDirective(
      'fixture-only',
      undefined,
      project.options,
      typescript.sys
    ).resolvedTypeReferenceDirective?.resolvedFileName;
  };
  const expected = resolveDeclaration(join(directory, 'app/tsconfig.json'));
  expect(expected).toContain('/app/node_modules/@types/fixture-only/index.d.ts');
  expect(resolveDeclaration(join(directory, runMatch![1], 'project-0.json'))).toBe(expected);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});

it.each([
  'build',
  'declarations'
])('X6 preserves a nonincremental developer cache sentinel in %s mode', mode => {
  const directory = createFixture();
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        rootDir: '.',
        types: [],
        incremental: false,
        tsBuildInfoFile: './developer.tsbuildinfo'
      },
      files: ['source.ts']
    })
  );
  write(directory, 'developer.tsbuildinfo', 'protected nonincremental cache');
  const result = run(directory, [
    '--mode',
    mode,
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(readFileSync(join(directory, 'developer.tsbuildinfo'), 'utf8')).toBe(
    'protected nonincremental cache'
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});

for (const selection of [
  {
    name: 'default production',
    jsx: 'react-jsx',
    packageName: 'react',
    runtime: 'jsx-runtime',
    pragma: ''
  },
  {
    name: 'configured development',
    jsx: 'react-jsxdev',
    packageName: 'configured-runtime',
    runtime: 'jsx-dev-runtime',
    pragma: ''
  },
  {
    name: 'per-file source pragma',
    jsx: 'react-jsx',
    packageName: 'file-runtime',
    runtime: 'jsx-runtime',
    pragma: '/** @jsxImportSource file-runtime */\n'
  }
]) {
  it.each([
    'cold',
    'warm'
  ])(`X7 refuses a ${selection.name} implicit JSX runtime declaration change during %s`, interval => {
    const directory = createFixture();
    write(
      directory,
      'tsconfig.json',
      JSON.stringify({
        compilerOptions: {
          rootDir: '.',
          types: [],
          jsx: selection.jsx,
          jsxImportSource:
            selection.name === 'default production' ? undefined : 'configured-runtime',
          module: 'ES2022',
          moduleResolution: 'bundler',
          noEmit: true,
          incremental: true
        },
        files: ['entry.tsx']
      })
    );
    write(directory, 'entry.tsx', `${selection.pragma}export const element = <div/>;`);
    const runtime = `node_modules/${selection.packageName}/${selection.runtime}.d.ts`;
    write(
      directory,
      `node_modules/${selection.packageName}/package.json`,
      JSON.stringify({
        name: selection.packageName,
        types: `${selection.runtime}.d.ts`,
        exports: {[`./${selection.runtime}`]: `./${selection.runtime}.d.ts`}
      })
    );
    write(
      directory,
      runtime,
      'export namespace JSX { interface Element {} interface IntrinsicElements { div: {}; } } export function jsx(...values: unknown[]): JSX.Element; export function jsxDEV(...values: unknown[]): JSX.Element;'
    );
    write(
      directory,
      'jsx-compiler.mjs',
      `
import {spawnSync} from 'node:child_process';
import {existsSync, writeFileSync, appendFileSync} from 'node:fs';
const warm = existsSync('compiled-once');
const result = spawnSync(${JSON.stringify(process.execPath)}, [${JSON.stringify(join(ROOT, 'node_modules/typescript/bin/tsc'))}, ...process.argv.slice(2)], {stdio: 'inherit'});
if (result.status !== 0) process.exit(result.status ?? 1);
writeFileSync('compiled-once', 'yes');
if (${JSON.stringify(interval)} === 'cold' || warm) appendFileSync(${JSON.stringify(join(directory, runtime))}, '\\nexport type ChangedRuntime = string;');
`
    );
    const result = run(directory, ['--compiler', 'jsx-compiler.mjs', '--timing-only']);
    expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
    expect(result.stderr).toContain(`inputs changed during ${interval} measurement`);
    expect(result.stdout).not.toContain('| tsconfig.json |');
  });
}

it('X4 preserves directly typed ambient packages outside @types', () => {
  const directory = createFixture();
  write(
    directory,
    'app/tsconfig.json',
    JSON.stringify({
      compilerOptions: {rootDir: '.', types: ['direct-fixture'], noEmit: true},
      files: ['index.ts']
    })
  );
  write(directory, 'app/index.ts', "export const value: 'app' = ambientSource;");
  for (const [root, value] of [
    ['app/', 'app'],
    ['', 'checkout']
  ]) {
    write(
      directory,
      `${root}node_modules/direct-fixture/package.json`,
      JSON.stringify({name: 'direct-fixture', types: 'index.d.ts'})
    );
    write(
      directory,
      `${root}node_modules/direct-fixture/index.d.ts`,
      `declare const ambientSource: '${value}';`
    );
  }
  const result = run(directory, [
    '--project',
    'app/tsconfig.json',
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});

it('X4 preserves package-local ambient types in the package source snapshot', () => {
  const directory = createPackageFixture();
  const configurationPath = join(directory, 'modules/analysis/tsconfig.json');
  const configuration = JSON.parse(readFileSync(configurationPath, 'utf8'));
  configuration.compilerOptions.types = ['fixture-only'];
  writeFileSync(configurationPath, JSON.stringify(configuration));
  write(
    directory,
    'modules/analysis/src/index.ts',
    "export const value: typeof ambientSource = 'package';"
  );
  write(
    directory,
    'modules/analysis/node_modules/@types/fixture-only/index.d.ts',
    "declare const ambientSource: 'package';"
  );
  write(
    directory,
    'node_modules/@types/fixture-only/index.d.ts',
    "declare const ambientSource: 'checkout';"
  );
  const result = run(directory, [
    '--project',
    'tsconfig.build.json',
    '--mode',
    'packages',
    '--compiler',
    join(ROOT, 'node_modules/typescript/bin/tsc'),
    '--timing-only'
  ]);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});
