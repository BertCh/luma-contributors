// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawn, spawnSync, type ChildProcess} from 'node:child_process';
import {
  copyFileSync,
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync
} from 'node:fs';
import {setTimeout as delay} from 'node:timers/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';
import typeScript from 'typescript';
import {validateOracleManifest} from '../../../../scripts/run-oracle-fixture.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];
const processes: ChildProcess[] = [];
const processGroups: number[] = [];

function write(directory: string, file: string, text: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), text);
}

function declareOracle(directory: string, outputs: string[], kind = 'fixture') {
  write(
    directory,
    'scripts/oracle-fixtures.json',
    JSON.stringify({
      version: 1,
      profiles: {node: {runtime: 'node'}},
      generators: {
        'fixture.mjs': {kind, profile: 'node', outputs, reason: 'Test declaration'},
        'inventory.mjs': {kind: 'inventory', profile: 'node', reason: 'Source audit only'}
      }
    })
  );
}

function fixture(script: string): string {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'gate-scripts-')));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
  if (script === 'verify-package-exports.mjs' || script === 'typecheck-status.mjs') {
    copyFileSync(
      join(ROOT, 'scripts/build-record.mjs'),
      join(directory, 'scripts/build-record.mjs')
    );
    copyFileSync(
      join(ROOT, 'scripts/typescript-project-inputs.mjs'),
      join(directory, 'scripts/typescript-project-inputs.mjs')
    );
    // Keep the fake tsc executable local; only its compiler-library import delegates to the
    // installed TypeScript. A directory symlink here would overwrite the real tsc in tests.
    write(
      directory,
      'node_modules/typescript/index.js',
      `module.exports = require(${JSON.stringify(join(ROOT, 'node_modules/typescript/lib/typescript.js'))});`
    );
  }
  return directory;
}

function run(directory: string, script: string, ...arguments_: string[]) {
  const result = spawnSync(process.execPath, [join(directory, 'scripts', script), ...arguments_], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 10_000
  });
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

function packageFixture(): string {
  const directory = fixture('verify-package-exports.mjs');
  for (const name of ['analysis', 'lenses', 'kepler']) {
    write(
      directory,
      `modules/${name}/package.json`,
      JSON.stringify({
        name: `@luma-contributors/${name}`,
        type: 'module',
        exports: {
          '.': {types: './dist/index.d.ts', import: './dist/index.js', require: './dist/index.cjs'}
        }
      })
    );
    write(directory, `modules/${name}/src/index.ts`, 'export const value = 1;');
    write(directory, `modules/${name}/dist/index.js`, 'export const value = 1;');
    write(directory, `modules/${name}/dist/index.cjs`, 'exports.value = 1;');
    write(directory, `modules/${name}/dist/index.d.ts`, 'export declare const value = 1;');
    mkdirSync(join(directory, 'node_modules/@luma-contributors'), {recursive: true});
    symlinkSync(
      join(directory, 'modules', name),
      join(directory, 'node_modules/@luma-contributors', name)
    );
  }
  return directory;
}

afterEach(async () => {
  for (const processId of processGroups.splice(0)) {
    try {
      process.kill(-processId, 'SIGTERM');
    } catch {}
  }
  for (const child of processes.splice(0)) {
    if (child.exitCode !== null || child.signalCode !== null) continue;
    const exited = new Promise<void>(resolve => child.once('exit', () => resolve()));
    child.kill('SIGTERM');
    await exited;
  }
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it('rejects a source file missing from an otherwise loadable build', () => {
  const directory = packageFixture();
  write(directory, 'modules/analysis/src/new.ts', 'export const added = 2;');
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('new');
});

it('rejects a build older than the current source', () => {
  const directory = packageFixture();
  utimesSync(
    join(directory, 'modules/analysis/src/index.ts'),
    new Date(),
    new Date(Date.now() + 10_000)
  );
  expect(run(directory, 'verify-package-exports.mjs').status).toBe(1);
});

it('compares export names when ESM and CommonJS have equal counts', () => {
  const directory = packageFixture();
  write(directory, 'modules/analysis/dist/index.cjs', 'exports.wrong = 1;');
  expect(run(directory, 'verify-package-exports.mjs').status).toBe(1);
});

it('checks a wildcard export instead of silently skipping it', () => {
  const directory = packageFixture();
  write(
    directory,
    'modules/analysis/package.json',
    JSON.stringify({
      name: '@luma-contributors/analysis',
      type: 'module',
      exports: {
        './*': {import: './dist/*.js', require: './dist/*.cjs'}
      }
    })
  );
  write(directory, 'modules/analysis/dist/index.cjs', 'exports.wrong = 1;');
  utimesSync(join(directory, 'modules/analysis/package.json'), new Date(0), new Date(0));
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('CommonJS names [wrong] differ from ESM [value]');
});

it('accepts a fresh build with matching names and declared types', () => {
  const directory = packageFixture();
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('verified 3 entry points');
});

it('checks string wildcard asset targets and rejects empty wildcard patterns', () => {
  const directory = packageFixture();
  const manifest = {
    name: '@luma-contributors/analysis',
    type: 'module',
    exports: {'./data/*': './dist/data/*'}
  };
  write(directory, 'modules/analysis/package.json', JSON.stringify(manifest));
  utimesSync(join(directory, 'modules/analysis/package.json'), new Date(0), new Date(0));
  expect(run(directory, 'verify-package-exports.mjs').status).toBe(1);
  write(directory, 'modules/analysis/dist/data/fixture.json', '{}');
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('verified 3 entry points');
});

it('rejects a missing promised declaration target', () => {
  const directory = packageFixture();
  rmSync(join(directory, 'modules/analysis/dist/index.d.ts'));
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('index.d.ts');
});

it.each([
  false,
  true
])('rejects an oracle that writes none or only some expected fixtures (partial=%s)', partial => {
  const directory = fixture('run-oracle-fixture.mjs');
  declareOracle(directory, ['oracles/fixtures/one.json', 'oracles/fixtures/two.json']);
  write(
    directory,
    'modules/analysis/test/oracles/GENERATORS.md',
    '### `fixture.mjs`\n`one.json`, `two.json`\n'
  );
  write(
    directory,
    'modules/analysis/test/oracles/fixture.mjs',
    partial
      ? "import fs from 'node:fs'; fs.writeFileSync('modules/analysis/test/oracles/fixtures/one.json', '{\"changed\":true}');"
      : ''
  );
  for (const name of ['one', 'two']) {
    write(directory, `modules/analysis/test/oracles/fixtures/${name}.json`, '{}');
    utimesSync(
      join(directory, `modules/analysis/test/oracles/fixtures/${name}.json`),
      new Date(0),
      new Date(0)
    );
  }
  const result = run(directory, 'run-oracle-fixture.mjs', 'fixture', '--write');
  expect(result.status, result.output).toBe(3);
  expect(
    readFileSync(join(directory, 'modules/analysis/test/oracles/fixtures/one.json'), 'utf8')
  ).toBe('{}');
});

it('rejects an old clean watcher log with no live owner', () => {
  const directory = fixture('typecheck-status.mjs');
  write(
    directory,
    '.typecheck/watch.log',
    'Starting compilation in watch mode\nFound 0 errors. Watching for file changes.\n'
  );
  utimesSync(join(directory, '.typecheck/watch.log'), new Date(0), new Date(0));
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(2);
});

async function watchFixture(
  errors = '',
  files: Record<string, string> = {},
  packageCommand = false,
  beforeStart?: (directory: string) => void
) {
  const directory = fixture('typecheck-status.mjs');
  for (const [file, contents] of Object.entries(files)) write(directory, file, contents);
  beforeStart?.(directory);
  write(
    directory,
    'node_modules/typescript/bin/tsc',
    `#!/usr/bin/env node
    console.log('Starting compilation in watch mode');
    ${errors ? `console.log(${JSON.stringify(errors)});` : ''}
    console.log('Found ${errors ? 1 : 0} errors. Watching for file changes.');
    setInterval(() => {}, 1000);
  `
  );
  if (packageCommand) {
    const scripts = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).scripts;
    write(
      directory,
      'package.json',
      JSON.stringify({
        private: true,
        scripts: {
          'typecheck-watch': scripts['typecheck-watch'],
          'typecheck-status': scripts['typecheck-status']
        }
      })
    );
    mkdirSync(join(directory, 'node_modules/.bin'), {recursive: true});
    chmodSync(join(directory, 'node_modules/typescript/bin/tsc'), 0o755);
    symlinkSync('../typescript/bin/tsc', join(directory, 'node_modules/.bin/tsc'));
  }
  const child = spawn(
    packageCommand ? 'npm' : process.execPath,
    packageCommand
      ? ['run', '--silent', 'typecheck-watch']
      : [join(directory, 'scripts/typecheck-status.mjs'), '--watch'],
    {cwd: directory, detached: packageCommand, stdio: ['ignore', 'pipe', 'pipe']}
  );
  if (packageCommand && child.pid) processGroups.push(child.pid);
  let output = '';
  child.stdout?.on('data', chunk => {
    output += chunk;
  });
  child.stderr?.on('data', chunk => {
    output += chunk;
  });
  processes.push(child);
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    try {
      if (
        readFileSync(join(directory, '.typecheck/watch.log'), 'utf8').includes(
          'Watching for file changes.'
        )
      )
        return directory;
    } catch {}
    await delay(20);
  }
  throw new Error(`Fake watcher did not start: ${output}`);
}

it('accepts a live completed cycle and prevents a second watcher', async () => {
  const directory = await watchFixture();
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('live cycle 1');
  expect(run(directory, 'typecheck-status.mjs', '--watch').status).toBe(2);
  expect(run(directory, 'typecheck-status.mjs').status).toBe(0);
});

it.each([
  'missing',
  'stale'
])('rejects one %s wildcard asset while another output still exists', condition => {
  const directory = packageFixture();
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({
      compilerOptions: {allowJs: true, resolveJsonModule: true},
      include: ['src/**/*', 'src/**/*.json']
    })
  );
  for (const name of ['a', 'b']) {
    write(directory, `modules/analysis/src/data/${name}.json`, '{}');
    write(directory, `modules/analysis/dist/data/${name}.json`, '{}');
  }
  const manifest = JSON.parse(
    readFileSync(join(directory, 'modules/analysis/package.json'), 'utf8')
  );
  manifest.exports['./data/*'] = './dist/data/*';
  write(directory, 'modules/analysis/package.json', JSON.stringify(manifest));
  for (const file of ['package.json', 'tsconfig.json'])
    utimesSync(join(directory, 'modules/analysis', file), new Date(0), new Date(0));
  if (condition === 'missing') rmSync(join(directory, 'modules/analysis/dist/data/a.json'));
  else utimesSync(join(directory, 'modules/analysis/dist/data/a.json'), new Date(0), new Date(0));
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('dist/data/a.json');
});

it('rejects a stale JavaScript loader from an allowJs build input', () => {
  const directory = packageFixture();
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({compilerOptions: {allowJs: true}, include: ['src/**/*']})
  );
  utimesSync(join(directory, 'modules/analysis/tsconfig.json'), new Date(0), new Date(0));
  write(directory, 'modules/analysis/src/loader.js', 'export const loader = 1;');
  for (const extension of ['js', 'cjs', 'd.ts']) {
    write(directory, `modules/analysis/dist/loader.${extension}`, '');
    utimesSync(
      join(directory, `modules/analysis/dist/loader.${extension}`),
      new Date(0),
      new Date(0)
    );
  }
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
  expect(result.output).toContain('dist/loader.js');
});

it.each([
  'missing prefix',
  'directory'
])('rejects a fresh target that Node considers an invalid public export (%s)', invalid => {
  const directory = packageFixture();
  const manifest = JSON.parse(
    readFileSync(join(directory, 'modules/analysis/package.json'), 'utf8')
  );
  manifest.exports['.'].import = invalid === 'directory' ? './dist' : 'dist/index.js';
  write(directory, 'modules/analysis/package.json', JSON.stringify(manifest));
  utimesSync(join(directory, 'modules/analysis/package.json'), new Date(0), new Date(0));
  const result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(1);
});

it('rejects an edit arriving while status waits for the requested heartbeat', async () => {
  const directory = await watchFixture('', {
    'modules/analysis/src/input.ts': 'export const value = 1;'
  });
  let heartbeat = JSON.parse(
    readFileSync(join(directory, '.typecheck/watch.owner.json'), 'utf8')
  ).heartbeatAt;
  for (;;) {
    await delay(10);
    const next = JSON.parse(
      readFileSync(join(directory, '.typecheck/watch.owner.json'), 'utf8')
    ).heartbeatAt;
    if (next !== heartbeat) break;
  }
  const child = spawn(process.execPath, [join(directory, 'scripts/typecheck-status.mjs')]);
  let output = '';
  child.stdout.on('data', chunk => {
    output += chunk;
  });
  child.stderr.on('data', chunk => {
    output += chunk;
  });
  const exited = new Promise<number | null>(resolve => child.once('exit', resolve));
  await delay(150);
  write(directory, 'modules/analysis/src/input.ts', 'export const value = 2;');
  expect(await exited, output).toBe(2);
});

it('rejects a completed cycle when only an imported lens JSON changes under the real config shape', async () => {
  const loader = 'modules/lenses/src/computation-lenses/scenarios/gpu-network/construction-data.js';
  const files: Record<string, string> = {
    'tsconfig.json': readFileSync(join(ROOT, 'tsconfig.json'), 'utf8'),
    'tsconfig.check-src.json': readFileSync(join(ROOT, 'tsconfig.check-src.json'), 'utf8'),
    [loader]: readFileSync(join(ROOT, loader), 'utf8')
  };
  for (const name of [
    'n-chicago-loop-crashes',
    'n-chicago-od-flows',
    'n-chicago-loop-polylines',
    'n-chicago-loop-traces'
  ])
    files[`modules/lenses/src/computation-lenses/data/${name}.json`] = '{}';
  const directory = await watchFixture('', files);
  expect(run(directory, 'typecheck-status.mjs').status).toBe(0);
  write(
    directory,
    'modules/lenses/src/computation-lenses/data/n-chicago-loop-crashes.json',
    '{"changed":true}'
  );
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(2);
  expect(result.output).toContain('stale');
});

it.each([
  'JSON',
  'JavaScript',
  'TypeScript'
] as const)('rejects an invalid imported %s edit with its size and original mtime restored', async kind => {
  const loader = 'modules/lenses/src/computation-lenses/scenarios/gpu-network/construction-data.js';
  const typescript = 'modules/lenses/src/index.ts';
  const input =
    kind === 'JSON'
      ? 'modules/lenses/src/computation-lenses/data/n-chicago-loop-crashes.json'
      : kind === 'JavaScript'
        ? loader
        : typescript;
  const files: Record<string, string> = {
    'tsconfig.json': readFileSync(join(ROOT, 'tsconfig.json'), 'utf8'),
    'tsconfig.check-src.json': readFileSync(join(ROOT, 'tsconfig.check-src.json'), 'utf8'),
    [loader]: readFileSync(join(ROOT, loader), 'utf8'),
    [typescript]: "import './computation-lenses/scenarios/gpu-network/construction-data.js';\n"
  };
  for (const name of [
    'n-chicago-loop-crashes',
    'n-chicago-od-flows',
    'n-chicago-loop-polylines',
    'n-chicago-loop-traces'
  ])
    files[`modules/lenses/src/computation-lenses/data/${name}.json`] = '{}';
  const directory = await watchFixture('', files, false, directory => {
    utimesSync(join(directory, input), 1_700_000_000, 1_700_000_000);
  });
  const file = join(directory, input);
  const original = statSync(file);
  const syntaxErrors = (text: string) =>
    kind === 'JSON'
      ? typeScript.parseConfigFileTextToJson(file, text).error
        ? 1
        : 0
      : (typeScript
          .transpileModule(text, {
            fileName: file,
            reportDiagnostics: true,
            compilerOptions: {
              module: typeScript.ModuleKind.ESNext,
              target: typeScript.ScriptTarget.ES2022
            }
          })
          .diagnostics?.filter(
            diagnostic => diagnostic.category === typeScript.DiagnosticCategory.Error
          ).length ?? 0);
  const contents = readFileSync(file, 'utf8');
  expect(syntaxErrors(contents)).toBe(0);
  expect(run(directory, 'typecheck-status.mjs').status).toBe(0);
  writeFileSync(file, kind === 'JSON' ? '!}' : contents.slice(0, -1) + ')');
  utimesSync(file, original.atimeMs / 1000, original.mtimeMs / 1000);
  expect(statSync(file).size).toBe(original.size);
  expect(statSync(file).mtimeMs).toBe(original.mtimeMs);
  expect(syntaxErrors(readFileSync(file, 'utf8'))).toBeGreaterThan(0);
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(2);
  expect(result.output).toContain('stale');
});

it.each([
  'delete imported input',
  'edit JavaScript input'
])('rejects a stale cycle after %s', async change => {
  const directory = await watchFixture('', {
    'modules/analysis/src/index.ts': "import './dependency.js';",
    'modules/analysis/src/dependency.js': 'export const value = 1;'
  });
  if (change === 'delete imported input')
    rmSync(join(directory, 'modules/analysis/src/dependency.js'));
  else {
    write(directory, 'modules/analysis/src/dependency.js', 'export const value = 2;');
    utimesSync(
      join(directory, 'modules/analysis/src/dependency.js'),
      new Date(),
      new Date(Date.now() + 1000)
    );
  }
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(2);
});

it('runs the paired package watcher and status commands with the stand-in compiler', async () => {
  const directory = await watchFixture('', {}, true);
  const result = spawnSync('npm', ['run', '--silent', 'typecheck-status'], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 10000
  });
  expect(result.status, result.stdout + result.stderr).toBe(0);
  expect(result.stdout).toContain('live cycle 1');
});

it('rejects a live completed cycle predating a source edit', async () => {
  const directory = await watchFixture();
  write(directory, 'modules/analysis/src/changed.ts', 'export const changed = true;');
  utimesSync(
    join(directory, 'modules/analysis/src/changed.ts'),
    new Date(),
    new Date(Date.now() + 1000)
  );
  const result = run(directory, 'typecheck-status.mjs');
  expect(result.status, result.output).toBe(2);
  expect(result.output).toContain('stale');
});

it('labels filtered success while retaining the total error count', async () => {
  const directory = await watchFixture('other.ts(1,1): error TS1234: other error');
  expect(run(directory, 'typecheck-status.mjs').status).toBe(1);
  const result = run(directory, 'typecheck-status.mjs', 'my-file.ts');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('0 filtered matching errors (1 total)');
});

it('compares freshly regenerated oracle bytes even when they are identical', () => {
  const directory = fixture('run-oracle-fixture.mjs');
  declareOracle(directory, ['oracles/fixtures/one.json']);
  write(
    directory,
    'modules/analysis/test/oracles/GENERATORS.md',
    '### `fixture.mjs`\n`one.json`\n'
  );
  write(directory, 'modules/analysis/test/oracles/fixtures/one.json', '{}');
  write(
    directory,
    'modules/analysis/test/oracles/fixture.mjs',
    "import fs from 'node:fs'; fs.writeFileSync('modules/analysis/test/oracles/fixtures/one.json', '{}');"
  );
  const result = run(directory, 'run-oracle-fixture.mjs', 'fixture');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('IDENTICAL');
});

it('refuses to run or advertise an inventory script with no declared fixtures', () => {
  const directory = fixture('run-oracle-fixture.mjs');
  declareOracle(directory, [], 'unconfigured');
  write(directory, 'modules/analysis/test/oracles/GENERATORS.md', '');
  write(
    directory,
    'modules/analysis/test/oracles/inventory.mjs',
    "throw new Error('must not run');"
  );
  const result = run(directory, 'run-oracle-fixture.mjs', 'inventory');
  expect(result.status, result.output).toBe(64);
  expect(result.output).toContain('no declared fixture outputs');
  expect(run(directory, 'run-oracle-fixture.mjs', '--list').output).not.toContain('inventory');
});

it('rejects missing committed oracle fixtures before executing the generator', () => {
  const directory = fixture('run-oracle-fixture.mjs');
  declareOracle(directory, ['oracles/fixtures/missing.json']);
  write(
    directory,
    'modules/analysis/test/oracles/GENERATORS.md',
    '### `fixture.mjs`\n`missing.json`\n'
  );
  write(directory, 'modules/analysis/test/oracles/fixture.mjs', "throw new Error('must not run');");
  const result = run(directory, 'run-oracle-fixture.mjs', 'fixture');
  expect(result.status, result.output).toBe(3);
  expect(result.output).toContain('expected fixture missing.json');
});

it('requires manifest outputs even when the documentation lists fewer outputs', () => {
  const directory = fixture('run-oracle-fixture.mjs');
  write(
    directory,
    'modules/analysis/test/oracles/GENERATORS.md',
    '### `fixture.mjs`\n`one.json`\n'
  );
  for (const name of ['one', 'two'])
    write(directory, `modules/analysis/test/oracles/fixtures/${name}.json`, '{}');
  write(
    directory,
    'scripts/oracle-fixtures.json',
    JSON.stringify({
      version: 1,
      profiles: {node: {runtime: 'node'}},
      generators: {
        'fixture.mjs': {
          kind: 'fixture',
          profile: 'node',
          outputs: ['oracles/fixtures/one.json', 'oracles/fixtures/two.json']
        }
      }
    })
  );
  write(
    directory,
    'modules/analysis/test/oracles/fixture.mjs',
    "import fs from 'node:fs'; fs.writeFileSync('modules/analysis/test/oracles/fixtures/one.json', '{}');"
  );
  const result = run(directory, 'run-oracle-fixture.mjs', 'fixture');
  expect(result.status, result.output).toBe(3);
  expect(result.output).toContain('two.json');
});

it('runs the source inventory only with explicit output paths and keeps visibility tooling configurable', () => {
  const inventory = join(ROOT, 'modules/analysis/test/oracles/parity-wp5-round8-showcase.py');
  const visibility = join(ROOT, 'modules/analysis/test/oracles/parity-wp5-deep-visibility.py');
  // Check before execution so the old inventory cannot attempt a historical external write.
  expect(readFileSync(inventory, 'utf8')).not.toContain('/private/tmp/claude');
  expect(readFileSync(visibility, 'utf8')).not.toContain('/private/tmp/claude');
  const directory = fixture('run-oracle-fixture.mjs');
  write(
    directory,
    'examples/deck/spatial-analysis-showcase/src/horizon.scene.ts',
    'new GPUTerrainHorizon({maximumDistance: 10});'
  );
  symlinkSync(join(ROOT, 'node_modules'), join(directory, 'node_modules'), 'dir');
  const output = join(directory, 'inventory-output');
  const result = spawnSync(
    '/usr/bin/python3',
    [inventory, '--repository-root', directory, '--output-directory', output],
    {encoding: 'utf8'}
  );
  expect(result.status, result.stderr).toBe(0);
  const generated = JSON.parse(
    readFileSync(join(output, 'parity-wp5-round8-showcase.json'), 'utf8')
  );
  expect(generated.calls).toHaveLength(1);
  expect(generated.calls[0].class).toBe('GPUTerrainHorizon');
  expect(readFileSync(join(output, 'parity-wp5-round8-showcase.md'), 'utf8')).toContain(
    'maximumDistance: 10'
  );
  const help = spawnSync('/usr/bin/python3', [visibility, '--help'], {encoding: 'utf8'});
  expect(help.status, help.stderr).toBe(0);
  for (const option of [
    '--reference-python',
    '--grass',
    '--input-directory',
    '--output-directory',
    '--audit-log',
    '--audit-output'
  ])
    expect(help.stdout).toContain(option);
  const dispatch = spawnSync(
    '/usr/bin/python3',
    [
      '-I',
      '-c',
      `
import json, runpy, sys
from pathlib import Path
main = runpy.run_path(sys.argv[1], run_name='oracle_configuration_test')['main']
namespace = main.__globals__
calls = []
namespace['generate'] = lambda: calls.append(['generate', str(namespace['INPUT']), str(namespace['OUTPUT']), str(namespace['REFERENCE_PYTHON']), str(namespace['GRASS'])])
namespace['write_audit'] = lambda source, target: calls.append(['audit', str(source), str(target)])
namespace['run_gdal_batch'] = lambda source, target: calls.append(['batch', source, target])
namespace['run_gdal_grid'] = lambda source, target: calls.append(['grid', source, target])
root = Path(sys.argv[2])
arguments = ['--reference-python', str(root/'python'), '--grass', str(root/'grass'), '--input-directory', str(root/'input'), '--output-directory', str(root/'output')]
main(arguments)
main(['--gdal-batch', 'batch-input', 'batch-output'])
main(['--gdal-grid', 'grid-input', 'grid-output'])
main(arguments + ['--audit-log', str(root/'audit.log'), '--audit-output', str(root/'audit.json')])
print(json.dumps(calls))
`,
      visibility,
      directory
    ],
    {encoding: 'utf8'}
  );
  expect(dispatch.status, dispatch.stderr).toBe(0);
  expect(JSON.parse(dispatch.stdout)).toEqual([
    [
      'generate',
      join(directory, 'input'),
      join(directory, 'output'),
      join(directory, 'python'),
      join(directory, 'grass')
    ],
    ['batch', 'batch-input', 'batch-output'],
    ['grid', 'grid-input', 'grid-output'],
    [
      'generate',
      join(directory, 'input'),
      join(directory, 'output'),
      join(directory, 'python'),
      join(directory, 'grass')
    ],
    ['audit', join(directory, 'audit.log'), join(directory, 'audit.json')]
  ]);
});

it('keeps the real oracle manifest complete and all configured outputs inside the test tree', () => {
  const manifest = JSON.parse(readFileSync(join(ROOT, 'scripts/oracle-fixtures.json'), 'utf8'));
  expect(() => validateOracleManifest(manifest)).not.toThrow();
  const oracleRoot = join(ROOT, 'modules/analysis/test/oracles');
  const scripts = readdirSync(oracleRoot)
    .filter(file => /\.(py|R|mjs|cpp)$/.test(file))
    .sort();
  expect(Object.keys(manifest.generators).sort()).toEqual(scripts);
  for (const declaration of Object.values(manifest.generators) as {
    kind: string;
    outputs?: string[];
    inputs?: string[];
  }[]) {
    if (declaration.kind !== 'fixture') continue;
    for (const output of [...(declaration.outputs ?? []), ...(declaration.inputs ?? [])]) {
      expect(existsSync(join(ROOT, 'modules/analysis/test', output)), output).toBe(true);
    }
  }
  expect(manifest.generators['parity-wp5-round8-showcase.py'].kind).toBe('inventory');
  expect(manifest.generators['rts04-cold-pyxccd.py'].profile).toBe('python-pyxccd');
  expect(manifest.generators['rts04-cold-pyxccd.py'].outputs).toHaveLength(2);
});
