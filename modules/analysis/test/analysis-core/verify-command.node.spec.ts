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
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import {transformSync} from 'esbuild';
import {afterEach, expect, it} from 'vitest';
import {
  getChildNodeOptions,
  getIsolatedNodeHeapArguments,
  getOptInBrowserDefines,
  getOptInTestExclusions,
  listTestFiles,
  mapWithConcurrency
} from '../../../../scripts/verification-policy.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];

function write(directory: string, path: string, contents: string) {
  mkdirSync(dirname(join(directory, path)), {recursive: true});
  writeFileSync(join(directory, path), contents);
}

function fixture(script: string) {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'verify-command-')));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
  if (script === 'typecheck-tests.mjs') {
    copyFileSync(join(ROOT, 'tsconfig.json'), join(directory, 'tsconfig.json'));
    mkdirSync(join(directory, 'node_modules'));
    symlinkSync(
      join(ROOT, 'node_modules/typescript'),
      join(directory, 'node_modules/typescript'),
      'dir'
    );
  }
  for (const helper of [
    'verification-policy.mjs',
    'typescript-project-inputs.mjs',
    'build-record.mjs'
  ]) {
    if (existsSync(join(ROOT, 'scripts', helper))) {
      copyFileSync(join(ROOT, 'scripts', helper), join(directory, 'scripts', helper));
    }
  }
  return directory;
}

function run(directory: string, script: string, arguments_: string[] = [], environment = {}) {
  const diskFixture = join(directory, 'fixture-disk.cjs');
  const nodeArguments = existsSync(diskFixture) ? ['--require', diskFixture] : [];
  const result = spawnSync(
    process.execPath,
    [...nodeArguments, join(directory, 'scripts', script), ...arguments_],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 15000,
      env: {...process.env, ...environment}
    }
  );
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it('preserves caller heap and other NODE_OPTIONS in every test typecheck child', () => {
  const directory = fixture('typecheck-tests.mjs');
  write(directory, 'test/kepler-compute-example-specs.json', '{"excludedFromRuns":{}}');
  mkdirSync(join(directory, 'modules'));
  write(directory, 'test/ordinary.node.spec.ts', 'export {};');
  write(
    directory,
    'node_modules/.bin/tsc',
    `#!${process.execPath}\nprocess.stdout.write(process.env.NODE_OPTIONS + '\\n');`
  );
  chmodSync(join(directory, 'node_modules/.bin/tsc'), 0o755);
  const result = run(directory, 'typecheck-tests.mjs', [], {
    NODE_OPTIONS: '--max-old-space-size=256 --trace-warnings'
  });
  expect(result.status, result.output).toBe(0);
  expect(result.output.match(/--max-old-space-size=256 --trace-warnings/g)).toHaveLength(6);
  expect(result.output).not.toContain('16384');
  const defaults = run(directory, 'typecheck-tests.mjs', [], {NODE_OPTIONS: '--trace-warnings'});
  expect(defaults.status, defaults.output).toBe(0);
  expect(defaults.output.match(/--trace-warnings --max-old-space-size=8192/g)).toHaveLength(6);
  const signed = run(directory, 'typecheck-tests.mjs', [], {
    NODE_OPTIONS: '--max_old_space_size=+256 --trace-warnings'
  });
  expect(signed.status, signed.output).toBe(0);
  expect(signed.output.match(/--max_old_space_size=\+256 --trace-warnings/g)).toHaveLength(6);
  write(directory, 'node_modules/.bin/tsc', `#!${process.execPath}\nprocess.exit(4);`);
  const failed = run(directory, 'typecheck-tests.mjs', [], {NODE_OPTIONS: ''});
  expect(failed.status, failed.output).toBe(4);
  expect(failed.output).not.toContain('example-backed specs');
  expect(failed.output).not.toContain('website-backed specs');
  for (const name of ['example-specs', 'website-specs']) {
    const configuration = JSON.parse(
      readFileSync(join(directory, `.typecheck/tsconfig.${name}.json`), 'utf8')
    );
    expect(configuration.compilerOptions.incremental).toBe(true);
    expect(configuration.compilerOptions.tsBuildInfoFile).toBe(`./${name}.tsbuildinfo`);
  }
});

it('excludes every benchmark suffix from correctness runs and activates the full benchmark inventory', () => {
  const config = readFileSync(join(ROOT, 'vitest.config.ts'), 'utf8');
  expect(config).toContain('BENCHMARK_PATTERNS');
  expect(config).toContain('maxWorkers: 1');
  const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
  expect(manifest.scripts['test-benchmarks']).toBe('node scripts/test-benchmarks.mjs');
});

it('keeps warm build state and bounds concurrent CommonJS work', () => {
  const source = readFileSync(join(ROOT, 'scripts/build.mjs'), 'utf8');
  expect(source).not.toContain('await rm(new URL(`modules/${moduleName}/tsconfig.tsbuildinfo`');
  expect(source).not.toContain('files.map(async file');
  expect(source).toContain('mapWithConcurrency');
});

it('provides a resumable sequential verification command', () => {
  expect(existsSync(join(ROOT, 'scripts/verify.mjs'))).toBe(true);
  const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
  expect(manifest.scripts.verify).toBe('node scripts/verify.mjs');
});

it('final2 G8 forwards a CLI heap override through verify instead of the larger environment heap', () => {
  const directory = verificationFixture();
  const result = spawnSync(
    process.execPath,
    ['--max-old-space-size=256', join(directory, 'scripts/verify.mjs'), '--only', 'lint'],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 10000,
      env: {
        ...process.env,
        NODE_OPTIONS: '--max-old-space-size=512',
        PATH: `${join(directory, 'node_modules/.bin')}:${process.env['PATH']}`
      }
    }
  );
  const output = `${result.stdout}${result.stderr}`;
  expect(result.status, output).toBe(0);
  const heap = Number(output.match(/heap-mib=([\d.]+)/)?.[1]);
  expect(heap).toBeGreaterThan(256);
  expect(heap).toBeLessThan(384);
});

it('TOOL-103 accepts a lower CLI heap that overrides an environment heap above the ceiling', () => {
  // Node applies the last heap option and command-line options follow NODE_OPTIONS (AGENTS.md).
  expect(
    getIsolatedNodeHeapArguments('--max-old-space-size=16384 "--max-old-space-size=256"')
  ).toEqual(['--max-old-space-size=256']);
  expect(getChildNodeOptions('--max-old-space-size=16384', ['--max-old-space-size=256'])).toBe(
    '--max-old-space-size=16384 --max-old-space-size=256'
  );
  // The effective value still meets the ceiling, and percentages are refused anywhere.
  expect(() =>
    getChildNodeOptions('--max-old-space-size=256', ['--max-old-space-size=16384'])
  ).toThrow('at most 8192 MiB');
  expect(() =>
    getChildNodeOptions('--max-old-space-size-percentage=50', ['--max-old-space-size=256'])
  ).toThrow('at most 8192 MiB');
  expect(() =>
    getChildNodeOptions('--max-old-space-size=abc', ['--max-old-space-size=256'])
  ).toThrow('at most 8192 MiB');

  const directory = verificationFixture();
  const result = spawnSync(
    process.execPath,
    ['--max-old-space-size=256', join(directory, 'scripts/verify.mjs'), '--only', 'lint'],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 10000,
      env: {
        ...process.env,
        NODE_OPTIONS: '--max-old-space-size=16384',
        PATH: `${join(directory, 'node_modules/.bin')}:${process.env['PATH']}`
      }
    }
  );
  const output = `${result.stdout}${result.stderr}`;
  expect(result.status, output).toBe(0);
  const heap = Number(output.match(/heap-mib=([\d.]+)/)?.[1]);
  expect(heap).toBeGreaterThan(256);
  expect(heap).toBeLessThan(384);
});

function verificationFixture() {
  const directory = fixture('verify.mjs');
  // These commands are tiny stand-ins; test disk policy independently of host free space.
  write(
    directory,
    'fixture-disk.cjs',
    `const fs = require('node:fs');
fs.statfsSync = () => ({bsize: 1024, bavail: Number(process.env.FIXTURE_FREE_DISK_GIB ?? 16) * 1024 ** 2});
require('node:module').syncBuiltinESMExports();`
  );
  // The shared sandbox forbids ps. A deterministic stand-in exercises enforcement without
  // inspecting unrelated processes or starting any real verification command.
  write(
    directory,
    'node_modules/.bin/ps',
    `#!${process.execPath}\nprocess.stdout.write(process.ppid + ' 1 ' + (process.env.RSS_KIB ?? '10240') + '\\n');`
  );
  chmodSync(join(directory, 'node_modules/.bin/ps'), 0o755);
  write(
    directory,
    'node_modules/.bin/yarn',
    `#!/bin/sh\nexec "${process.execPath}" "${join(directory, 'fake-yarn.cjs')}" "$@"\n`
  );
  chmodSync(join(directory, 'node_modules/.bin/yarn'), 0o755);
  write(
    directory,
    'fake-yarn.cjs',
    `
const fs = require('node:fs');
const path = require('node:path');
const args = process.argv.slice(2);
if (args[0] === '--version') { process.stdout.write('1.22.22\\n'); process.exit(0); }
fs.appendFileSync('.typecheck/commands.jsonl', JSON.stringify(args) + '\\n');
process.stdout.write('heap-mib=' + require('node:v8').getHeapStatistics().heap_size_limit / 1024 ** 2 + '\\n');
if (args[0] === process.env.FAIL_STEP) process.exit(7);
if (args[0] === 'test-headless' || args[0] === 'test-node') {
  const files = args[0] === 'test-node'
    ? ['modules/analysis/test/ordinary.node.spec.ts']
    : args.slice(1).filter(argument => argument.endsWith('.spec.ts'));
  if (process.env.OMIT_FILE) files.pop();
  const report = {testResults: files.map(file => ({name: path.resolve(file), assertionResults: [{status: process.env.ALL_SKIPPED ? 'skipped' : 'passed'}]}))};
  fs.writeFileSync(args.find(argument => argument.startsWith('--outputFile.json=')).slice('--outputFile.json='.length), JSON.stringify(report));
}
`
  );
  write(directory, 'modules/analysis/test/ordinary.node.spec.ts', 'export {};');
  write(directory, 'modules/analysis/test/first/one.spec.ts', '');
  write(directory, 'modules/analysis/test/second/two.spec.ts', '');
  write(directory, 'modules/analysis/test/second/timing-bench.spec.ts', '');
  return directory;
}

function runVerification(directory: string, arguments_: string[], environment = {}) {
  return run(directory, 'verify.mjs', arguments_, {
    npm_execpath: join(directory, 'fake-yarn.cjs'),
    PATH: `${join(directory, 'node_modules/.bin')}:${process.env['PATH']}`,
    ...environment
  });
}

function readCommands(directory: string): string[][] {
  const file = join(directory, '.typecheck/commands.jsonl');
  return existsSync(file)
    ? readFileSync(file, 'utf8')
        .trim()
        .split('\n')
        .map(line => JSON.parse(line))
    : [];
}

it('runs steps in order, stops on the first failure, and resumes only the requested steps', () => {
  const directory = verificationFixture();
  const failed = runVerification(directory, [], {FAIL_STEP: 'check-contributor-contract'});
  expect(failed.status, failed.output).toBe(1);
  expect(readCommands(directory).map(command => command[0])).toEqual([
    'lint',
    'check-upstream-first',
    'check-contributor-contract'
  ]);
  rmSync(join(directory, '.typecheck/commands.jsonl'));
  const resumed = runVerification(directory, ['--from', 'test-node']);
  expect(resumed.status, resumed.output).toBe(0);
  const commands = readCommands(directory);
  expect(commands.map(command => command[0])).toEqual([
    'test-node',
    'test-headless',
    'test-headless'
  ]);
  const receipt = JSON.parse(
    readFileSync(join(directory, '.typecheck/verify/latest.json'), 'utf8')
  );
  expect(commands[0]).toEqual([
    'test-node',
    '--maxWorkers=1',
    '--reporter=default',
    '--reporter=json',
    `--outputFile.json=${join(directory, '.typecheck/verify', `${receipt.id}-node.json`)}`
  ]);
  expect(receipt.inventories.node.selected).toEqual([
    'modules/analysis/test/ordinary.node.spec.ts'
  ]);
  expect(receipt.inputChanges).toEqual([]);
  expect(resumed.output).toContain('matched 2 files in 2 directory chunks');
  expect(resumed.output).toContain('(partial)');
  rmSync(join(directory, '.typecheck/commands.jsonl'));
  expect(runVerification(directory, ['--only', 'lint']).status).toBe(0);
  expect(readCommands(directory)).toEqual([['lint']]);
});

it('R1 launches the Yarn shell shim through PATH and runs lint exactly once', () => {
  const directory = verificationFixture();
  const result = runVerification(directory, ['--only', 'lint'], {
    npm_execpath: join(directory, 'node_modules/.bin/yarn')
  });
  expect(result.status, result.output).toBe(0);
  expect(readCommands(directory)).toEqual([['lint']]);
});

it('R2 excludes four opt-in jobs by default and keeps explicitly enabled jobs eligible', () => {
  const directory = verificationFixture();
  const jobs = [
    [
      'modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/ab/delaunay-ab.spec.ts',
      'A5_AB',
      '1'
    ],
    [
      'modules/analysis/test/gpu-spatial-analysis/polygon-overlay/ab/overlay-ab.spec.ts',
      'A5_AB',
      '1'
    ],
    [
      'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts',
      'VITE_VECCHIA_FLOAT32_STUDY',
      '1'
    ],
    [
      'modules/lenses/test/computation-lenses/capture-records.spec.ts',
      'VITE_CAPTURE_LENS_RECORDS',
      '1'
    ]
  ];
  for (const [file] of jobs) write(directory, file, '');
  expect(listTestFiles(directory, 'headless', {})).toEqual([
    'modules/analysis/test/first/one.spec.ts',
    'modules/analysis/test/second/two.spec.ts'
  ]);
  for (const [file, flag, value] of jobs)
    expect(listTestFiles(directory, 'headless', {[flag]: value})).toContain(file);
  for (const [file, flag, value] of jobs) {
    expect(getOptInTestExclusions({})).toContain(file);
    expect(getOptInTestExclusions({[flag]: value})).not.toContain(file);
  }
  const skipped = runVerification(directory, ['--only', 'test-headless'], {ALL_SKIPPED: 'true'});
  expect(skipped.status).toBe(1);
  expect(skipped.output).toContain('No passing test executed');
});

it('R2 forwards only A5_AB to both direct and reflective browser gates', () => {
  const source = readFileSync(join(ROOT, 'test/utils/verification-opt-in-env.mjs'), 'utf8');
  for (const value of ['', '1']) {
    const define = getOptInBrowserDefines({A5_AB: value, SECRET: 'must not be forwarded'});
    expect(JSON.stringify(define)).not.toContain('must not be forwarded');
    const compiled = transformSync(source, {define}).code;
    const context = {process: {env: {A5_AB: undefined as string | undefined}}};
    runInNewContext(compiled, context);
    expect(context.process.env.A5_AB).toBe(value);
    expect(Reflect.get(context.process.env, 'A5_AB')).toBe(value);
  }
});

it('R6 rejects underscore percentage heaps and preserves signed lower heaps in child Node', () => {
  const directory = verificationFixture();
  const percentage = runVerification(directory, ['--only', 'lint'], {
    NODE_OPTIONS: '--max_old_space_size_percentage=90'
  });
  expect(percentage.status, percentage.output).toBe(1);
  expect(percentage.output).toContain('at most 8192 MiB');
  expect(readCommands(directory)).toEqual([]);
  const signed = runVerification(directory, ['--only', 'lint'], {
    NODE_OPTIONS: '--max-old-space-size=+256'
  });
  expect(signed.status, signed.output).toBe(0);
  const heap = Number(signed.output.match(/heap-mib=([\d.]+)/)?.[1]);
  expect(heap).toBeGreaterThan(256);
  expect(heap).toBeLessThan(384);
  expect(signed.output).not.toContain('--max-old-space-size=8192');
});

it('fails and terminates the command when sampled RSS exceeds the documented ceiling', () => {
  const directory = verificationFixture();
  const result = runVerification(directory, ['--only', 'lint'], {
    RSS_KIB: String(13 * 1024 * 1024)
  });
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('RSS exceeded 12 GiB');
});

it('refuses low-disk heavy steps before spawning them and rejects unknown resume steps', () => {
  const directory = verificationFixture();
  const result = runVerification(directory, ['--only', 'build'], {FIXTURE_FREE_DISK_GIB: '4'});
  expect(result.status).toBe(1);
  expect(result.output).toContain('Free disk 4.00 GiB is below the 8 GiB floor');
  expect(result.output).toContain('no heavy command started');
  expect(readCommands(directory)).toEqual([]);
  expect(runVerification(directory, ['--only', 'typo']).status).toBe(1);
  expect(runVerification(directory, ['--from', 'lint', '--only', 'build']).status).toBe(1);
});

it('rejects a successful browser process that omitted a file or executed no tests', () => {
  const directory = verificationFixture();
  for (const environment of [{OMIT_FILE: 'true'}, {ALL_SKIPPED: 'true'}]) {
    const result = runVerification(directory, ['--only', 'test-headless'], environment);
    expect(result.status, result.output).toBe(1);
    expect(result.output).toMatch(/Headless inventory mismatch|No passing test executed/);
  }
});

it('selects new benchmark files by suffix and activates their boolean gates without changing size options', () => {
  const directory = fixture('test-benchmarks.mjs');
  write(
    directory,
    'modules/analysis/test/new/new-bench.spec.ts',
    `const run = environment?.['VITE_NEW_BENCHMARK'] === 'true';`
  );
  write(
    directory,
    'modules/analysis/test/new/profile-benchmark.spec.ts',
    `const run = environment?.[\n'VITE_NEW_PROFILE'\n] !== 'true';`
  );
  write(directory, 'modules/analysis/test/new/correctness.spec.ts', '');
  write(
    directory,
    'node_modules/vitest/vitest.mjs',
    `process.stdout.write(JSON.stringify({args: process.argv.slice(2), enabled: process.env.VITE_NEW_BENCHMARK, profile: process.env.VITE_NEW_PROFILE, rows: process.env.VITE_POINTCLOUD_MAX_ROWS}));`
  );
  expect(listTestFiles(directory, 'headless')).toEqual([
    'modules/analysis/test/new/correctness.spec.ts'
  ]);
  expect(listTestFiles(directory, 'benchmarks')).toHaveLength(2);
  const result = run(directory, 'test-benchmarks.mjs', [], {VITE_POINTCLOUD_MAX_ROWS: '12'});
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('"enabled":"true"');
  expect(result.output).toContain('"profile":"true"');
  expect(result.output).toContain('"rows":"12"');
});

it('limits the number of simultaneous transforms even when all inputs are ready', async () => {
  let activeCount = 0;
  let peakCount = 0;
  const completed: number[] = [];
  await mapWithConcurrency([1, 2, 3, 4, 5, 6], 2, async (value: number) => {
    activeCount++;
    peakCount = Math.max(activeCount, peakCount);
    await new Promise(resolveDelay => setTimeout(resolveDelay, 5));
    completed.push(value);
    activeCount--;
  });
  expect(peakCount).toBe(2);
  expect(completed.sort()).toEqual([1, 2, 3, 4, 5, 6]);
});

it('preserves warm compiler caches, prunes removed-source outputs, and skips unchanged CommonJS transforms', () => {
  const directory = fixture('build.mjs');
  write(
    directory,
    'scripts/build-commonjs.mjs',
    `export async function transformCommonJSModule(source) { return {code: source}; }`
  );
  // Use the production planner/freshness API; only compilation and transformation are fake.
  mkdirSync(join(directory, 'node_modules'), {recursive: true});
  for (const dependency of ['typescript', 'esbuild']) {
    symlinkSync(
      join(ROOT, 'node_modules', dependency),
      join(directory, 'node_modules', dependency),
      'dir'
    );
  }
  write(
    directory,
    'node_modules/.bin/tspc',
    `#!${process.execPath}\nconst fs = require('node:fs'); fs.appendFileSync('compiler-args.jsonl', JSON.stringify(process.argv.slice(2)) + '\\n'); for (const moduleName of ['analysis', 'lenses', 'kepler']) { const cache = 'modules/' + moduleName + '/tsconfig.tsbuildinfo'; const cached = fs.existsSync(cache); fs.appendFileSync('cache-state.jsonl', JSON.stringify({moduleName, cached}) + '\\n'); if (cached && fs.readFileSync(cache, 'utf8') !== 'warm') process.exit(9); fs.writeFileSync(cache, 'warm'); const target = 'modules/' + moduleName + '/dist/index.js'; if (!fs.existsSync(target) && process.argv.includes('--force')) fs.writeFileSync(target, 'export const value = 1;'); }`
  );
  chmodSync(join(directory, 'node_modules/.bin/tspc'), 0o755);
  for (const moduleName of ['analysis', 'lenses', 'kepler']) {
    write(
      directory,
      `modules/${moduleName}/tsconfig.json`,
      JSON.stringify({
        compilerOptions: {rootDir: 'src', outDir: 'dist', declaration: true, types: []},
        include: ['src/**/*']
      })
    );
    write(directory, `modules/${moduleName}/package.json`, '{"type":"module"}');
    write(directory, `modules/${moduleName}/tsconfig.tsbuildinfo`, 'warm');
    write(directory, `modules/${moduleName}/src/index.ts`, 'export const value = 1;');
    write(directory, `modules/${moduleName}/dist/index.js`, 'export const value = 1;');
    write(directory, `modules/${moduleName}/dist/index.d.ts`, 'export declare const value = 1;');
    write(directory, `modules/${moduleName}/dist/removed.js`, 'old');
    write(directory, `modules/${moduleName}/dist/removed.cjs`, 'old');
    write(directory, `modules/${moduleName}/dist/removed.d.ts`, 'old');
  }
  const cold = run(directory, 'build.mjs');
  expect(cold.status, cold.output).toBe(0);
  expect(cold.output.match(/wrote 1 CommonJS/g)).toHaveLength(3);
  for (const moduleName of ['analysis', 'lenses', 'kepler']) {
    for (const extension of ['js', 'd.ts', 'cjs']) {
      expect(existsSync(join(directory, `modules/${moduleName}/dist/index.${extension}`))).toBe(
        true
      );
      expect(existsSync(join(directory, `modules/${moduleName}/dist/removed.${extension}`))).toBe(
        false
      );
    }
  }
  const warm = run(directory, 'build.mjs');
  expect(warm.status, warm.output).toBe(0);
  expect(warm.output.match(/wrote 0 CommonJS/g)).toHaveLength(3);
  const cacheStates = readFileSync(join(directory, 'cache-state.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  expect(cacheStates.slice(0, 3).map(state => state.cached)).toEqual([false, false, false]);
  expect(cacheStates.slice(3, 6).map(state => state.cached)).toEqual([true, true, true]);
  expect(readFileSync(join(directory, 'compiler-args.jsonl'), 'utf8')).not.toContain('--force');
  rmSync(join(directory, 'modules/analysis/dist/index.js'));
  const repaired = run(directory, 'build.mjs');
  expect(repaired.status, repaired.output).toBe(0);
  expect(existsSync(join(directory, 'modules/analysis/dist/index.js'))).toBe(true);
  expect(readFileSync(join(directory, 'compiler-args.jsonl'), 'utf8')).toContain('--force');
});

it.each([
  'test-node',
  'test-headless'
])('review B5 preserves equals signs in the %s report path', step => {
  const directory = verificationFixture();
  const reportFile = join(directory, 'r19=a', '.typecheck/verify/run=node.json');
  const writes: string[] = [];
  // Run the actual fake-runner text with captured filesystem writes: the old split would
  // otherwise write outside the requested report directory. No probe writes reach disk.
  runInNewContext(readFileSync(join(directory, 'fake-yarn.cjs'), 'utf8'), {
    require(name: string) {
      if (name === 'node:fs')
        return {
          appendFileSync() {},
          writeFileSync(destination: string) {
            writes.push(destination);
          }
        };
      if (name === 'node:path') return {resolve};
      if (name === 'node:v8')
        return {getHeapStatistics: () => ({heap_size_limit: 256 * 1024 ** 2})};
      throw new Error(`unexpected fake-runner import ${name}`);
    },
    process: {
      argv: [
        'node',
        'fake-yarn.cjs',
        step,
        'modules/analysis/test/first/one.spec.ts',
        `--outputFile.json=${reportFile}`
      ],
      env: {},
      stdout: {write() {}}
    }
  });
  expect(writes).toEqual([reportFile]);
});
